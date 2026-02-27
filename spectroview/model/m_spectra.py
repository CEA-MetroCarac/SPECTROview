#spectroview/model/spectra.py
from threading import Thread
from multiprocessing import Queue
import numpy as np
import zlib
import base64

from fitspy.core.spectra import Spectra as FitspySpectra
from fitspy.core.utils_mp import fit_mp
from spectroview.viewmodel.utils import apply_fit_model_to_spectrum

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

    def save(self, is_map=False):
        """Override Fitspy's save() to handle custom attributes and fix O(N^2) bottleneck.
        
        Args:
            is_map: If True, metadata is NOT saved per-spectrum (saved per-map instead).
                   x0/y0 are also not saved for maps (retrieved from map DataFrame).
        
        Returns:
            dict: Serialized spectra data with custom attributes handled properly.
        """
        spectrums_data = {}
        
        # Process each spectrum directly to avoid Fitspy core O(N^2) get_objects loop
        for i, spectrum in enumerate(self):
            # spectrum.save() creates the base dict using vars(self) -> capturing all custom attrs natively
            spectrum_dict = spectrum.save(save_data=False)
            
            # Remove redundant baseline fields to reduce payload
            if 'baseline' in spectrum_dict:
                spectrum_dict['baseline'].pop('y_eval', None)
            
            # Save x0, y0 only if it's not a map
            if not is_map:
                spectrum_dict.update({
                    "x0": self._compress(spectrum.x0),
                    "y0": self._compress(spectrum.y0)
                })
            else:
                # For Maps : Remove metadata added by Fitspy's .save()
                # (metadata is saved once per map in maps_metadata, not per-spectrum)
                spectrum_dict.pop('metadata', None)
            
            spectrums_data[i] = spectrum_dict
        
        return spectrums_data
    
    def apply_model(self, model_dict, fnames=None, ncpus=1, show_progressbar=True, queue_incr=None):
        """ Apply 'model' to all or part of the spectra."""
        if fnames is None:
            fnames = self.fnames

        spectra = []
        for fname in fnames:
            spectrum, _ = self.get_objects(fname)

            #apply only CUSTOM MODEL keep some information 
            apply_custom_fit_model(spectrum, model_dict, fname)
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
            for spectrum in spectra:
                spectrum.preprocess()
                spectrum.fit()
                queue_incr.put(1)
        else:
            fit_mp(spectra, ncpus, queue_incr)
        
        # Only join thread if it was started
        if thread is not None:
            thread.join()     


    @staticmethod
    def _compress(array):
        """Compress and encode a numpy array to a base64 string."""
        if array is None:
            return None
        compressed = zlib.compress(array.tobytes())
        encoded = base64.b64encode(compressed).decode('utf-8')
        return encoded
    
    @staticmethod
    def _decompress(data, dtype=np.float64):
        """Decode and decompress a base64 string to a numpy array."""
        if data is None:
            return None
        decoded = base64.b64decode(data.encode('utf-8'))
        decompressed = zlib.decompress(decoded)
        return np.frombuffer(decompressed, dtype=dtype) 