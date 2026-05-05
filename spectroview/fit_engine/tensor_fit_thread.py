"""QThread wrapper for the Tensor Fitting Engine.

Drop-in replacement for HyperFitThread from core/.
"""

import time
import numpy as np
import sys
from PySide6.QtCore import QThread, Signal
from spectroview.fit_engine.tensor_engine import TensorFittingEngine


class TensorFitThread(QThread):
    """Run the TensorFittingEngine on a background thread.

    Supports two modes:
        1. **Single-model mode** (legacy): all spectra share one fit_model.
        2. **Batched mode**: spectra are pre-grouped by model structure.
           Each batch is a (fit_model, spectra_list) tuple and is processed
           sequentially through the tensor engine while spectra within a
           batch are optimised in parallel.

    signals:
        progress_changed(current, total, percent, elapsed)
        timings_ready(str)
        finished()
    """
    progress_changed = Signal(int, int, int, float)
    timings_ready = Signal(str)

    def __init__(self, spectrums, fit_model, fnames,
                 coords=None, apply_model_to_spectra=True,
                 batches=None):
        super().__init__()
        
        # Increase stack size to 8MB (macOS defaults to 512KB for QThread)
        # Prevents segmentation faults when LAPACK (np.linalg.solve) allocates 
        # large arrays on the stack during batched tensor operations.
        if sys.platform == "darwin":
            self.setStackSize(8 * 1024 * 1024)
            
        self._spectrums = spectrums      # MSpectra collection
        self._fit_model = fit_model
        self._fnames = fnames
        self._coords = coords
        self._apply_model_to_spectra = apply_model_to_spectra
        self._batches = batches          # list of (fit_model, spectra_list)
        self._is_cancelled = False

    def stop(self):
        self._is_cancelled = True

    def run(self):
        if self._batches:
            self._run_batched()
        else:
            self._run_single()

    # ── Batched mode (fit() with per-spectrum models) ────────────────────

    def _run_batched(self):
        """Process multiple groups of spectra, each with its own fit model.

        Spectra within each group share the same peak-model structure and
        are optimised in parallel by the tensor engine.  Groups are
        processed sequentially.
        """
        t_start = time.perf_counter()

        total = sum(len(grp) for _, grp in self._batches)
        processed = 0

        engine = TensorFittingEngine()

        for fit_model, spectra in self._batches:
            if self._is_cancelled:
                break

            fit_params = fit_model.get("fit_params", {})
            batch_size = len(spectra)

            # Progress callback that accounts for previously processed groups
            def on_progress(current, batch_total, _offset=processed):
                elapsed = time.perf_counter() - t_start
                done = _offset + current
                pct = int(100 * done / max(total, 1))
                self.progress_changed.emit(done, total, pct, elapsed)

            try:
                engine.fit_spectra(
                    spectra=spectra,
                    fit_model=fit_model,
                    fit_params=fit_params,
                    progress_callback=on_progress,
                    cancel_check=lambda: self._is_cancelled,
                    apply_model_to_spectra=False,
                )
            except Exception as e:
                print(f"[TensorFitThread] batch failed: {e}")

            processed += batch_size

        if not self._is_cancelled:
            elapsed = time.perf_counter() - t_start
            self.progress_changed.emit(total, total, 100, elapsed)

            timings_str = f"Fit time: {elapsed:.2f}s\n"
            timings_str += "\n".join(
                [f"  {k}: {v}" for k, v in engine.timings.items()]
            )
            n_groups = len(self._batches)
            if n_groups > 1:
                timings_str += f"\n  Groups: {n_groups}"
            self.timings_ready.emit(timings_str)

    # ── Single-model mode (apply_fit_model / Maps) ───────────────────────

    def _run_single(self):
        """Original behaviour: one fit_model applied to all spectra."""
        t_start = time.perf_counter()

        # Build fname → spectrum lookup
        fname_set = set(self._fnames)
        spectra = [s for s in self._spectrums if s.fname in fname_set]

        n = len(spectra)

        if not spectra:
            return

        # Extract fit_params from the fit model or from the first spectrum
        fit_params = None
        if isinstance(self._fit_model, dict):
            fit_params = self._fit_model.get("fit_params", None)
        if fit_params is None and spectra:
            fit_params = getattr(spectra[0], "fit_params", None)
        if fit_params is None:
            fit_params = {"method": "leastsq", "xtol": 1e-4, "max_ite": 500}

        # Progress callback
        def on_progress(current, total):
            elapsed = time.perf_counter() - t_start
            pct = int(100 * current / max(total, 1))
            self.progress_changed.emit(current, total, pct, elapsed)

        # Run tensor engine

        engine = TensorFittingEngine()
        fit_results = engine.fit_spectra(
            spectra=spectra,
            fit_model=self._fit_model,
            fit_params=fit_params,
            progress_callback=on_progress,
            cancel_check=lambda: self._is_cancelled,
            apply_model_to_spectra=self._apply_model_to_spectra,
        )

        if not self._is_cancelled:
            elapsed = time.perf_counter() - t_start
            self.progress_changed.emit(n, n, 100, elapsed)
            
            # Emit timings string
            timings_str = f"Fit time: {elapsed:.2f}s\n"
            timings_str += "\n".join([f"  {k}: {v}" for k, v in engine.timings.items()])
            self.timings_ready.emit(timings_str)
