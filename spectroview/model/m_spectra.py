#spectroview/model/spectra.py
from copy import deepcopy
from threading import Thread
from multiprocessing import Queue
import dill

from fitspy.core.spectra import Spectra as FitspySpectra

class MSpectra(FitspySpectra):
    """Model: container for SpectrumM"""

    def add(self, spectrum):
        self.append(spectrum)

    def remove(self, indices):
        for i in sorted(indices, reverse=True):
            del self[i]

    def reorder(self, new_order):
        self[:] = [self[i] for i in new_order]

    def names(self):
        return [s.fname for s in self]

    def get(self, indices):
        if not indices:
            return []
        n = len(self)
        return [self[i] for i in indices if 0 <= i < n]


    def __len__(self):
        return super().__len__()

    
    
    def apply_model(self, model_dict, fnames=None, ncpus=1, show_progressbar=True, queue_incr=None):
        """ Apply 'model' to all or part of the spectra."""
        if fnames is None:
            fnames = self.fnames

        spectra = []
        for fname in fnames:
            spectrum, _ = self.get_objects(fname)
            
            # Customize the model_dict for this spectrum
            custom_model = deepcopy(model_dict)
            if hasattr(spectrum, "xcorrection_value"):  # reassign current xcorrection_value
                custom_model["xcorrection_value"] = spectrum.xcorrection_value
            if hasattr(spectrum, "label"):  
                custom_model["label"] = spectrum.label
            if hasattr(spectrum, "color"):  
                custom_model["color"] = spectrum.color

            spectrum.set_attributes(custom_model)
            spectrum.fname = fname  # reassign the correct fname
            spectra.append(spectrum)

        self.pbar_index = 0

        # Use provided queue or create new one
        if queue_incr is None:
            queue_incr = Queue()
        
        # Only start progressbar thread if show_progressbar is True
        if show_progressbar:
            args = (queue_incr, len(fnames), ncpus, show_progressbar)
            thread = Thread(target=self.progressbar, args=args)
            thread.start()
        else:
            thread = None

        if ncpus == 1:
            # Sequential processing
            for spectrum in spectra:
                spectrum.preprocess()
                spectrum.fit()
                queue_incr.put(1)
        else:
            # Parallel processing using joblib
            from joblib import Parallel, delayed
            # Worker function for joblib (no Queue - it's not picklable)
            def _fit_worker(spectrum_data):
                """Worker function that fits a single spectrum."""
                import dill
                # Deserialize spectrum
                spectrum = dill.loads(spectrum_data)
                
                # Fit the spectrum
                spectrum.preprocess()
                spectrum.fit()
                
                # Return serialized results
                return (
                    spectrum.x, 
                    spectrum.y, 
                    spectrum.weights,
                    spectrum.baseline.y_eval, 
                    spectrum.baseline.is_subtracted,
                    dill.dumps(spectrum.result_fit)
                )
            
            # Serialize spectra for parallel processing
            import dill
            serialized_spectra = [dill.dumps(s) for s in spectra]
            
            # Run parallel fitting with joblib
            results = Parallel(n_jobs=ncpus, backend='loky', verbose=0)(
                delayed(_fit_worker)(spec_data) 
                for spec_data in serialized_spectra
            )
            
            # Update spectra with results
            for i, spectrum in enumerate(spectra):
                res = results[i]
                spectrum.x = res[0]
                spectrum.y = res[1]
                spectrum.weights = res[2]
                spectrum.baseline.y_eval = res[3]
                spectrum.baseline.is_subtracted = res[4]
                spectrum.result_fit = dill.loads(res[5])
                spectrum.reassign_params()
                
                # Report progress after each result is processed
                queue_incr.put(1)
        
        # Only join thread if it was started
        if thread is not None:
            thread.join()     