import numpy as np

from fitspy.core.spectrum import Spectrum as FitspySpectrum


class MSpectrum(FitspySpectrum):
    """Customized of Spectrum class."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = None   # user-defined legend label
        self.color = None   # user-defined color
        
        self.xcorrection_value = 0   # peak position correction using reference value
        self.source_path = None
        
                    
    def reinit(self):
        """ Reinitialize the main attributes """
        self.range_min = None
        self.range_max = None
        self.x = self.x0.copy()
        self.y = self.y0.copy()
        self.weights = self.weights0.copy() if self.weights0 is not None else None
        self.outliers_limit = None
        self.normalize = False
        self.normalize_range_min = None
        self.normalize_range_max = None
        self.remove_models()
        self.result_fit = lambda: None
        self.baseline.reinit()
        self.baseline.mode = "Linear"
        
        self.color = None
        self.label = None

    def preprocess(self):
        """ Preprocess the spectrum """
        self.load_profile(self.fname)
        self.apply_range()
        self.eval_baseline()
        self.subtract_baseline()
        self.normalization()

    def apply_xcorrection(self, new_xcorr_value=None):
        """ Apply peak position correction """
        # Undo existing correction if needed
        if self.xcorrection_value != 0:
            self.undo_xcorrection()

        # If user provides a new correction, update the value
        if new_xcorr_value is not None:
            self.xcorrection_value = new_xcorr_value

        # Apply correction
        if self.xcorrection_value != 0:
            self.x0 = self.x0 + self.xcorrection_value
            self.x = self.x + self.xcorrection_value

    def undo_xcorrection(self):
        """Undo peak position correction (restore original x and x0)."""
        if self.xcorrection_value != 0:
            self.x0 = self.x0 - self.xcorrection_value
            self.x = self.x - self.xcorrection_value
            self.xcorrection_value = 0