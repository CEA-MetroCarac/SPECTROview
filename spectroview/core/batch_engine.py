# spectroview/core/batch_engine.py
"""
Main orchestrator for the high-performance batch fitting engine.

Provides the public BatchFittingEngine class that coordinates preprocessing,
model evaluation, spatial traversal, and optimization for fitting large
collections of spectra efficiently.
"""

import os
import time
import warnings
import numpy as np
from copy import deepcopy

from spectroview.core.models import PeakModelEvaluator, FitResult
from spectroview.core.baseline import batch_preprocess
from spectroview.core.spatial import build_traversal_order, NeighborPropagator
from spectroview.core.optimizer import (
    fit_batch_sequential,
    fit_batch_threaded,
    fit_single_spectrum,
)


# Fit method mapping (SPECTROview names → scipy names)
_METHOD_MAP = {
    "leastsq": "lm",          # Levenberg-Marquardt
    "least_squares": "trf",   # Trust Region Reflective
    "nelder": "trf",          # No direct equiv in scipy.least_squares; use trf
    "slsqp": "trf",           # No direct equiv; use trf
}


class BatchFittingEngine:
    """High-performance fitting engine for batch spectrum processing.

    Designed for hyperspectral 2D maps but works for any collection of spectra.
    Key optimizations:
    - Direct scipy.optimize.least_squares (no lmfit per-spectrum overhead)
    - Spatial neighbor parameter propagation (for 2D maps)
    - Single-pass fitting with full tolerance (propagation provides good p0)

    Usage:
        engine = BatchFittingEngine()
        engine.fit_spectra(
            spectra=list_of_MSpectrum,
            fit_model=model_dict,
            coords=np.array([[x1,y1], [x2,y2], ...]),
            fit_params={'method': 'leastsq', 'xtol': 1e-4, ...},
            ncpus=4,
            progress_callback=lambda current, total: ...,
        )
    """

    def __init__(self):
        self._evaluator = None

    def fit_spectra(
        self,
        spectra,
        fit_model,
        coords=None,
        fit_params=None,
        ncpus=1,
        progress_callback=None,
        cancel_check=None,
        apply_model_to_spectra=True,
    ):
        """Fit all spectra using the batch engine.

        Args:
            spectra: List of MSpectrum objects with x0/y0 data loaded
            fit_model: Fit model dict (from spectrum.save() or loaded JSON)
            coords: (N, 2) spatial coordinates for neighbor propagation.
                    None for non-map (sequential fitting without propagation).
            fit_params: Dict with fitting parameters:
                - method: str ("leastsq", "least_squares", etc.)
                - xtol: float (convergence tolerance)
                - max_ite: int (max iterations)
                - fit_negative: bool
                - coef_noise: float
            ncpus: Number of CPU threads (used only for non-spatial parallel path)
            progress_callback: callable(current, total) for progress updates
            cancel_check: callable() -> bool to check for cancellation
            apply_model_to_spectra: If True, apply fit_model to spectra first
                                    (baseline, peaks, preprocessing)

        Returns:
            list of FitResult objects, one per spectrum
        """
        warnings.filterwarnings(
            "ignore",
            message=".*Using UFloat objects with std_dev==0.*",
            category=UserWarning,
        )

        if not spectra:
            return []

        n_spectra = len(spectra)
        t_total = time.perf_counter()

        # ─── 1. Apply fit model to spectra (set peak_models, baseline, etc.) ───
        if apply_model_to_spectra:
            t0 = time.perf_counter()
            self._apply_model_to_all(spectra, fit_model)
            print(f"  [BatchEngine] Step 1 - apply_model: {time.perf_counter()-t0:.3f}s")

        # ─── 2. Build the PeakModelEvaluator ───
        self._evaluator = PeakModelEvaluator.from_fit_model(fit_model)

        if self._evaluator.n_params_free == 0:
            # No free parameters — nothing to optimize
            if progress_callback:
                progress_callback(n_spectra, n_spectra)
            return [FitResult(True, {}, np.array([])) for _ in spectra]

        # ─── 3. Preprocess all spectra (range, baseline, normalization) ───
        t0 = time.perf_counter()
        for spectrum in spectra:
            spectrum.preprocess()
        print(f"  [BatchEngine] Step 2 - preprocess: {time.perf_counter()-t0:.3f}s")

        # Detect shared x-axis and extract data matrix
        x_shared = self._detect_shared_x(spectra)
        x_array, Y_matrix = self._extract_data_matrix(spectra, x_shared)

        if x_array is None or Y_matrix is None:
            return [FitResult(False, {}, np.array([])) for _ in spectra]

        # ─── 4. Parse fit parameters ───
        if fit_params is None:
            fit_params = {}

        method_name = fit_params.get("method", "leastsq")
        scipy_method = _METHOD_MAP.get(method_name.lower(), "trf")
        xtol = float(fit_params.get("xtol", 1e-4))
        max_ite = int(fit_params.get("max_ite", 200))
        fit_negative = bool(fit_params.get("fit_negative", False))
        coef_noise = float(fit_params.get("coef_noise", 0))

        n_free = self._evaluator.n_params_free
        max_nfev = max(2, max_ite) * n_free

        # Get initial params and bounds
        p0 = self._evaluator.initial_params
        bounds = self._evaluator.bounds

        # ─── 5. Reinitialize amplitudes from actual data ───
        p0 = self._reinit_amplitudes(x_array, Y_matrix, p0)

        # ─── 6. Choose fitting strategy ───
        use_spatial = coords is not None and len(coords) == n_spectra

        t0 = time.perf_counter()
        if use_spatial:
            # Map fitting: single pass with spatial traversal + neighbor propagation.
            # Propagation provides excellent initial guesses, so full tolerance
            # converges quickly. This is inherently sequential (each pixel depends
            # on its neighbor's result), but each fit is very fast (~2-5ms).
            results = self._fit_with_propagation(
                x_array, Y_matrix, p0, bounds,
                coords, scipy_method, xtol, max_nfev,
                fit_negative, coef_noise,
                progress_callback, cancel_check,
            )
        elif ncpus > 1 and n_spectra > ncpus * 2:
            # Parallel fitting without spatial awareness
            results = fit_batch_threaded(
                x_array, Y_matrix, self._evaluator, p0, bounds,
                method=scipy_method, xtol=xtol, max_nfev=max_nfev,
                ncpus=ncpus,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                fit_negative=fit_negative,
                coef_noise=coef_noise,
            )
        else:
            # Sequential fitting (small batch or single CPU)
            results = fit_batch_sequential(
                x_array, Y_matrix, self._evaluator, p0, bounds,
                method=scipy_method, xtol=xtol, max_nfev=max_nfev,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                fit_negative=fit_negative,
                coef_noise=coef_noise,
            )
        print(f"  [BatchEngine] Step 3 - fitting ({n_spectra} spectra): "
              f"{time.perf_counter()-t0:.3f}s")

        # ─── 7. Build FitResult objects and write back to spectra ───
        t0 = time.perf_counter()
        fit_results = []
        for i, spectrum in enumerate(spectra):
            if results[i] is None:
                fr = FitResult(False, {}, np.zeros_like(x_array))
            else:
                p_opt, success, cost = results[i]
                fr = self._evaluator.build_result(p_opt, spectrum.x, spectrum.y, success)

            # Write result back to the spectrum object
            self._evaluator.write_back_to_spectrum(spectrum, fr)
            fit_results.append(fr)
        print(f"  [BatchEngine] Step 4 - write_back: {time.perf_counter()-t0:.3f}s")
        print(f"  [BatchEngine] TOTAL: {time.perf_counter()-t_total:.3f}s")

        return fit_results

    def _apply_model_to_all(self, spectra, fit_model):
        """Apply fit model dict to all spectra (set peak_models, baseline, etc.)."""
        from spectroview.viewmodel.utils import apply_custom_fit_model

        for spectrum in spectra:
            fname = spectrum.fname
            apply_custom_fit_model(spectrum, fit_model, fname)

    def _detect_shared_x(self, spectra):
        """Detect if all spectra share the same x-axis (common for maps).

        Returns the shared x-array if all spectra have the same x, else None.
        """
        if not spectra:
            return None

        ref_x = spectra[0].x
        if ref_x is None:
            return None

        for spectrum in spectra[1:]:
            if spectrum.x is None or len(spectrum.x) != len(ref_x):
                return None
            if not np.array_equal(spectrum.x, ref_x):
                return None

        return ref_x

    def _extract_data_matrix(self, spectra, x_shared=None):
        """Build the data matrix from already-preprocessed spectra."""
        if not spectra:
            return None, None

        n_spectra = len(spectra)

        if x_shared is not None:
            x_array = x_shared
        else:
            x_array = spectra[0].x

        if x_array is None:
            return None, None

        n_wavelengths = len(x_array)
        Y_matrix = np.empty((n_spectra, n_wavelengths), dtype=np.float64)

        for i, spectrum in enumerate(spectra):
            if spectrum.y is not None and len(spectrum.y) == n_wavelengths:
                Y_matrix[i] = spectrum.y
            else:
                Y_matrix[i] = np.zeros(n_wavelengths)

        return x_array, Y_matrix

    def _reinit_amplitudes(self, x, Y_matrix, p0):
        """Reinitialize amplitude parameters from actual spectrum data.

        Only adjusts amplitudes when the model's initial guess is wildly
        different from the actual data (>10x off). For small batches where
        the fit model was manually designed, we trust the model's values.
        """
        n_spectra = Y_matrix.shape[0]
        if n_spectra == 0:
            return p0

        # Only reinit for large maps where the model may not match individual spectra
        if n_spectra < 10:
            return p0

        # Use median spectrum for robust initial guess
        y_median = np.median(Y_matrix, axis=0)
        p0 = p0.copy()

        evaluator = self._evaluator
        for i in range(evaluator._n_peaks):
            start, end = evaluator._model_slices[i]
            _, param_names, prefix = evaluator._peak_specs[i]

            for j, pname in enumerate(param_names):
                global_idx = start + j

                if pname == "ampli" and evaluator._param_vary[global_idx]:
                    # Find the x0 parameter for this same peak
                    x0_idx = None
                    for k, pn in enumerate(param_names):
                        if pn == "x0":
                            x0_idx = start + k
                            break

                    if x0_idx is not None:
                        x0_val = evaluator._param_values[x0_idx]
                        closest_idx = np.argmin(np.abs(x - x0_val))
                        ampli_from_data = max(y_median[closest_idx], 1e-6)

                        # Find position in free vector
                        free_pos = np.searchsorted(
                            evaluator._free_mask, global_idx
                        )
                        if (free_pos < len(evaluator._free_mask) and
                                evaluator._free_mask[free_pos] == global_idx):
                            current_val = abs(p0[free_pos])
                            # Only update if wildly off (>10x ratio)
                            if current_val > 0 and ampli_from_data > 0:
                                ratio = max(current_val, ampli_from_data) / min(current_val, ampli_from_data)
                                if ratio > 10:
                                    p0[free_pos] = ampli_from_data

        return p0

    def _fit_with_propagation(self, x, Y_matrix, p0, bounds, coords,
                              method, xtol, max_nfev,
                              fit_negative, coef_noise,
                              progress_callback, cancel_check):
        """Fit map spectra with spatial traversal and neighbor propagation.

        Uses a single pass with full tolerance. Neighbor propagation provides
        excellent initial guesses for each pixel, so convergence is fast
        (typically 2-3ms per spectrum vs 5-6ms without propagation).

        This is inherently sequential because each pixel's initial guess
        depends on its fitted neighbor's result. Multi-threading does not help
        here (tested: thread overhead > gain for ~5ms fits).
        """
        n_spectra = Y_matrix.shape[0]
        evaluator = self._evaluator

        # Build spatial traversal order (spiral from center)
        traversal_order = build_traversal_order(coords, strategy="spiral")

        # Create neighbor propagator
        propagator = NeighborPropagator(coords, k_neighbors=4)

        # Single pass: full tolerance, with propagation for good initial guesses
        results = fit_batch_sequential(
            x, Y_matrix, evaluator, p0, bounds,
            method=method, xtol=xtol, max_nfev=max_nfev,
            traversal_order=traversal_order,
            propagator=propagator,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            fit_negative=fit_negative,
            coef_noise=coef_noise,
        )

        return results
