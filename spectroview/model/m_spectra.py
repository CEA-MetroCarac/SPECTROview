#spectroview/model/spectra.py
from copy import deepcopy
from threading import Thread
from multiprocessing import Queue

from fitspy.core.spectra import Spectra as FitspySpectra

class SpectraM(FitspySpectra):
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

    
    def apply_model(self, model_dict, fnames=None, ncpus=1,show_progressbar=True):
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

        queue_incr = Queue()
        args = (queue_incr, len(fnames), ncpus, show_progressbar)
        thread = Thread(target=self.progressbar, args=args)
        thread.start()

        if ncpus == 1:
            for spectrum in spectra:
                spectrum.preprocess()
                spectrum.fit()
                queue_incr.put(1)
        else:
            fit_mp(spectra, ncpus, queue_incr)
        thread.join()     