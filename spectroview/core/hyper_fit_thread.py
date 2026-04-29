# spectroview/core/hyper_fit_thread.py
"""
QThread wrapper for the batch fitting engine.

Drop-in replacement for ApplyFitModelThread that uses the high-performance
BatchFittingEngine for significantly faster fitting of hyperspectral maps.
"""

import os
import time
import warnings
import numpy as np
from copy import deepcopy

from PySide6.QtCore import QThread, Signal

from spectroview.core.batch_engine import BatchFittingEngine


class HyperFitThread(QThread):
    """High-performance fitting thread for hyperspectral map data.

    Signal-compatible replacement for ApplyFitModelThread. Uses the batch
    fitting engine with spatial neighbor propagation and optimized
    scipy.optimize.least_squares.

    Signals:
        progress_changed(current, total, percentage, elapsed_time)
    """

    progress_changed = Signal(int, int, int, float)

    def __init__(self, spectrums, fit_model, fnames, ncpus=1, coords=None,
                 apply_model_to_spectra=True):
        """
        Args:
            spectrums: MSpectra collection (or list-like of MSpectrum)
            fit_model: Fit model dict (from spectrum.save() or loaded JSON)
            fnames: List of spectrum fnames to fit
            ncpus: Number of CPU threads for parallel fitting
            coords: Optional (N, 2) array of spatial coordinates for maps.
                    Enables neighbor parameter propagation.
            apply_model_to_spectra: If True, apply fit_model to spectra before
                                    fitting (for first-time model application).
                                    Set False for re-fitting (models already set).
        """
        super().__init__()
        self.spectrums = spectrums
        self.fit_model = deepcopy(fit_model)
        self.fnames = fnames
        self.ncpus = ncpus
        self.coords = coords
        self.apply_model_to_spectra = apply_model_to_spectra
        self._is_cancelled = False

    def stop(self):
        """Request the thread to stop gracefully."""
        self._is_cancelled = True

    def run(self):
        """Execute batch fitting with progress tracking."""
        warnings.filterwarnings(
            "ignore",
            message=".*Using UFloat objects with std_dev==0.*",
            category=UserWarning,
        )

        total = len(self.fnames)
        start_time = time.time()
        print(f"[HyperFitThread] Starting: {total} spectra, ncpus={self.ncpus}, "
              f"coords={'Yes' if self.coords is not None else 'No'}, "
              f"apply_model={self.apply_model_to_spectra}")

        # Emit initial progress
        self.progress_changed.emit(0, total, 0, 0.0)

        # Build O(1) lookup for spectrum objects
        t0 = time.perf_counter()
        spectra_dict = {os.path.normpath(s.fname): s for s in self.spectrums}

        # Collect spectra in fname order
        spectra_to_fit = []
        for fname in self.fnames:
            norm_fname = os.path.normpath(fname)
            spectrum = spectra_dict.get(norm_fname)
            if spectrum is not None:
                spectra_to_fit.append(spectrum)

        if not spectra_to_fit:
            self.progress_changed.emit(total, total, 100, time.time() - start_time)
            return
        print(f"  [HyperFitThread] Lookup: {time.perf_counter()-t0:.3f}s "
              f"({len(spectra_to_fit)} spectra matched)")

        # Build coords array for the valid spectra (if map data)
        coords = None
        if self.coords is not None:
            coords = self.coords
            if len(coords) != len(spectra_to_fit):
                coords = None  # Mismatch — disable propagation

        # Load fit settings from spectrum (if available)
        fit_params = {}
        for spectrum in spectra_to_fit:
            if hasattr(spectrum, "fit_params") and spectrum.fit_params:
                fit_params = spectrum.fit_params.copy()
                break

        # Progress callback
        def on_progress(current, n_total):
            if self._is_cancelled:
                return
            percentage = int((current / max(n_total, 1)) * 100)
            elapsed = time.time() - start_time
            self.progress_changed.emit(current, n_total, percentage, elapsed)

        # Cancel check
        def check_cancel():
            return self._is_cancelled

        # Run the batch engine
        engine = BatchFittingEngine()

        try:
            engine.fit_spectra(
                spectra=spectra_to_fit,
                fit_model=self.fit_model,
                coords=coords,
                fit_params=fit_params,
                ncpus=self.ncpus,
                progress_callback=on_progress,
                cancel_check=check_cancel,
                apply_model_to_spectra=self.apply_model_to_spectra,
            )
        except Exception as e:
            print(f"[HyperFitThread] Batch engine error: {e}")
            import traceback
            traceback.print_exc()

        # Final progress
        elapsed = time.time() - start_time
        if not self._is_cancelled:
            self.progress_changed.emit(total, total, 100, elapsed)
        print(f"[HyperFitThread] DONE in {elapsed:.2f}s")
