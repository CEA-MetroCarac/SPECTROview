#spectroview/model/spectra.py
from copy import deepcopy
from threading import Thread
from multiprocessing import Queue
import numpy as np
import zlib
import base64

from fitspy.core.spectra import Spectra as FitspySpectra
from fitspy.core.utils_mp import fit_mp

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
        """Override Fitspy's save() to handle custom attributes.
        
        Args:
            is_map: If True, metadata is NOT saved per-spectrum (saved per-map instead).
                   x0/y0 are also not saved for maps (retrieved from map DataFrame).
        
        Returns:
            dict: Serialized spectra data with custom attributes handled properly.
        """
        # Call parent's save() to get base Fitspy attributes
        spectrums_data = super().save()
        
        # Process each spectrum to handle custom attributes
        for i, spectrum in enumerate(self):
            spectrum_dict = {}
            
            # Save x0, y0 only if it's not a map
            if not is_map:
                spectrum_dict.update({
                    "x0": self._compress(spectrum.x0),
                    "y0": self._compress(spectrum.y0)
                })
            else:
                # For Maps : Remove metadata added by Fitspy's .save()
                # (metadata is saved once per map in maps_metadata, not per-spectrum)
                spectrums_data[i].pop('metadata', None)
            
            # Update the spectrums_data with custom attributes
            spectrums_data[i].update(spectrum_dict)
        
        return spectrums_data
    
    @staticmethod
    def load_from_dict(spectrum_class, spectrum_data, is_map=True, maps=None):
        """Load a spectrum from dictionary data.
        
        Args:
            spectrum_class: The MSpectrum class to instantiate
            spectrum_data: Dictionary containing spectrum attributes
            is_map: If True, x0/y0 are retrieved from maps DataFrame
            maps: Dictionary of map DataFrames (required if is_map=True)
        
        Returns:
            MSpectrum: Reconstructed spectrum object
        """
        from spectroview.model.m_spectrum import MSpectrum
        
        # Pop custom attributes before set_attributes() to prevent Fitspy crash
        # (Fitspy's set_attributes tries to process all keys as model attributes
        #  and crashes when calling .keys() on string values in metadata dict)
        saved_metadata = spectrum_data.pop('metadata', None)
        
        # Create spectrum and set Fitspy attributes
        spectrum = MSpectrum()
        spectrum.set_attributes(spectrum_data)
        
        # Restore metadata
        if saved_metadata:
            spectrum.metadata = saved_metadata
        
        if is_map:
            # Retrieve x0 and y0 from map DataFrame
            if maps is None:
                raise ValueError("maps must be provided when is_map=True.")
            
            # Parse map_name and coordinates from fname
            fname = spectrum.fname
            map_name, coord_str = fname.rsplit('_', 1)
            coord_str = coord_str.strip('()')
            coord = tuple(map(float, coord_str.split(',')))
            
            # Retrieve x0 and y0 from the corresponding map_df
            if map_name in maps:
                map_df = maps[map_name]
                map_df = map_df.iloc[:, :-1]  # Drop the last column from map_df (NaN)
                coord_x, coord_y = coord
                
                # Use nearest neighbor matching to handle floating point precision differences
                # between coordinates in filename (string) and saved CSV data
                dist = (map_df['X'] - coord_x)**2 + (map_df['Y'] - coord_y)**2
                min_dist_idx = dist.values.argmin()
                
                # Check if the closest point is within a reasonable tolerance (e.g., < 1e-4 units)
                # This ensures we don't match random points if the map changed
                if dist.iloc[min_dist_idx] < 1e-4:
                    row = map_df.iloc[[min_dist_idx]]
                    
                    x0 = map_df.columns[2:].astype(float).values
                    spectrum.x0 = x0 + spectrum.xcorrection_value
                    spectrum.y0 = row.iloc[0, 2:].values
                else:
                    # If no match found, initialize as None (will likely cause issues downstream, 
                    # but better than matching wrong point)
                    spectrum.x0 = None
                    spectrum.y0 = None
            else:
                spectrum.x0 = None
                spectrum.y0 = None
        else:
            # Decompress x0 and y0 for non-map spectra
            if 'x0' in spectrum_data:
                spectrum.x0 = MSpectra._decompress(spectrum_data['x0'])
            if 'y0' in spectrum_data:
                spectrum.y0 = MSpectra._decompress(spectrum_data['y0'])
        
        return spectrum

    
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