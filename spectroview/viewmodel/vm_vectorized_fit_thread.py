"""
spectroview/viewmodel/vm_vectorized_fit_thread.py
==================================================
QThread wrapper for FastFitEngine — the vectorized 2D map fitting engine.

Emits the same ``progress_changed`` signal format as the existing
``ApplyFitModelThread`` so the view layer needs no changes.
"""

import time
import numpy as np
from copy import deepcopy
from PySide6.QtCore import QThread, Signal

from spectroview.model.fast_fit_engine import FastFitEngine


class VectorizedMapFitThread(QThread):
    """
    Fits all spectra in a 2D map using the vectorized scipy engine.

    Signals
    -------
    progress_changed(done, total, percentage, elapsed_s)
    finished()
    """

    progress_changed = Signal(int, int, int, float)

    def __init__(self, spectra_list, fit_model_dict, coords_2d=None, parent=None):
        """
        Parameters
        ----------
        spectra_list : list of MSpectrum
            Pre-processed spectrum objects (x, y already set; baseline may not
            be subtracted — the engine will use spectrum.y which has been
            preprocessed by the normal single-CPU path only if available,
            otherwise uses y0).
        fit_model_dict : dict
            fitspy model dict (from ``Spectra.load_model()``).
        coords_2d : list of (float, float), optional
            Physical (x, y) coordinates matching spectra_list order.
            Required for neighbour-guess propagation.
        """
        super().__init__(parent)
        self.spectra_list    = spectra_list
        self.fit_model_dict  = fit_model_dict
        self.coords_2d       = coords_2d
        self._cancelled      = [False]   # mutable list so closure can see updates

        # Public result handle (set after successful fit)
        self.engine: FastFitEngine | None = None

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def stop(self):
        """Request cancellation (non-blocking)."""
        self._cancelled[0] = True

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------

    def run(self):
        start_time = time.time()
        spectra    = self.spectra_list
        N          = len(spectra)

        if N == 0:
            return

        # ---- Step 1: Apply fit model structure to all spectra ----------
        #
        # Two issues to solve:
        #   a) range_min/range_max: set_attributes copies these from the fit
        #      model dict; if we skip it, reinit'd spectra have range=None and
        #      preprocess() produces full-length y, while the template (which
        #      did get set_attributes) gets clipped → inhomogeneous Y-matrix.
        #   b) peak_labels: set_attributes with peak_models={} resets
        #      peak_labels to [] → IndexError in v_peak_table.
        #
        # FAST two-pass approach:
        #   Pass 1: set_attributes with peak_models=[] on each spectrum.
        #           This is fast (no create_model() calls) but copies
        #           all scalar settings: range_min, range_max, baseline, etc.
        #   Pass 2: replace peak_models with deepcopies from template, fix labels.
        try:
            # Create the full model dict ONCE; build peak_models from it ONCE
            template = spectra[0]
            template.set_attributes(deepcopy(self.fit_model_dict))
            base_peak_models = [deepcopy(pm) for pm in template.peak_models]
            base_bkg_model   = deepcopy(template.bkg_model)
            n_peaks          = len(base_peak_models)

            # ---- Build the safe scalar model dict ---------------------------
            # CRITICAL: fit_model_dict is saved FROM A SPECTRUM — it includes
            # fname, x0, y0, color, label, and other identity/data attributes.
            # If we pass those to set_attributes() for every spectrum we
            # overwrite their unique (X, Y) coordinates, destroying the map.
            # Only forward the "model configuration" scalars: range and baseline.
            # Everything else (data, identity, display) must stay untouched.
            _IDENTITY_KEYS = {
                # Spectrum identity / raw data — NEVER overwrite across spectra
                'fname', 'x0', 'y0', 'x', 'y',
                'weights', 'weights0',
                # Display attributes
                'color', 'label',
                # Fit state — will be written by the engine after fitting
                'result_fit', 'peak_models', 'bkg_model',
                'peak_labels', 'peak_index',
                # Active/checkbox state managed by the list widget
                'is_active',
            }
            scalar_model_dict = {
                k: v for k, v in self.fit_model_dict.items()
                if k not in _IDENTITY_KEYS
            }
            # Always clear peak_models to avoid create_model() being called
            scalar_model_dict['peak_models'] = {}

            for s in spectra:
                # Fast: applies range_min, range_max, baseline, xcorrection only.
                # fname, x0/y0, color, label are NOT in scalar_model_dict.
                s.set_attributes(deepcopy(scalar_model_dict))
                # Restore peak_models from template (already constructed lmfit objects)
                s.peak_models = [deepcopy(pm) for pm in base_peak_models]
                s.bkg_model   = deepcopy(base_bkg_model)
                # peak_labels is reset to [] by set_attributes({peak_models: {}})
                s.peak_labels = list(map(str, range(1, n_peaks + 1)))

        except Exception as e:
            import traceback
            print(f"[VectorizedMapFitThread] set_attributes failed: {e}")
            traceback.print_exc()
            return

        # ---- Step 2: Preprocess all spectra (baseline, range-clip etc.) -
        # All spectra now have the same range_min/range_max from the fit model
        # so preprocess() produces identically-shaped x/y arrays.
        try:
            for s in spectra:
                s.preprocess()
        except Exception as e:
            print(f"[VectorizedMapFitThread] preprocess failed: {e}")
            return

        # ---- Step 3: Build Y matrix from preprocessed spectra -----------
        # All spectra have the same x shape after uniform range-clipping above.
        try:
            x   = spectra[0].x
            M   = len(x)
            Y   = np.zeros((N, M), dtype=np.float64)
            for i, s in enumerate(spectra):
                if len(s.y) == M:
                    Y[i] = s.y
                else:
                    # Fallback: interpolate to common grid if lengths diverge
                    Y[i] = np.interp(x, s.x, s.y)
        except Exception as e:
            print(f"[VectorizedMapFitThread] Y-matrix build failed: {e}")
            return

        self.progress_changed.emit(0, N, 0, 0.0)

        # ---- Step 4: Progress callback -----------------------------------
        def _progress(done, total):
            if self._cancelled[0]:
                return
            elapsed = time.time() - start_time
            pct = int(done / total * 100)
            self.progress_changed.emit(done, total, pct, elapsed)

        # ---- Step 5: Run the vectorized engine ---------------------------
        try:
            engine = FastFitEngine(
                x              = x,
                Y              = Y,
                fit_model_dict = self.fit_model_dict,
                coords_2d      = self.coords_2d,
                progress_callback = _progress,
            )
            engine.fit(cancelled_flag=self._cancelled)
        except Exception as e:
            import traceback
            print(f"[VectorizedMapFitThread] Engine error: {e}")
            traceback.print_exc()
            return

        if self._cancelled[0]:
            return

        self.engine = engine

        # ---- Step 6: Write results back into spectrum objects -----------
        # build_fitspy_result_for_spectrum sets result_fit + updates peak_models
        # reassign_params() copies result_fit.params → peak_model.param_hints
        # so the spectrum viewer and peak table show the correct fitted values.
        try:
            order = engine._scan_order()
            for idx in order:
                if self._cancelled[0]:
                    break
                engine.build_fitspy_result_for_spectrum(spectra[idx], idx)
                spectra[idx].reassign_params()

            elapsed = time.time() - start_time
            self.progress_changed.emit(N, N, 100, elapsed)
        except Exception as e:
            import traceback
            print(f"[VectorizedMapFitThread] Result write-back failed: {e}")
            traceback.print_exc()
