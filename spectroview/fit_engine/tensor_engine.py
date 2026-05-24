"""Tensor Fitting Engine — public API.

Fits all spectra in a hyperspectral map simultaneously using a custom
batched Levenberg-Marquardt optimizer.

Usage:
    engine = TensorFittingEngine()
    results = engine.fit_spectra(x, Y, fit_model, ...)
"""

import time
import numpy as np

from spectroview.fit_engine.evaluator import TensorEvaluator
from spectroview.fit_engine.optimizer import batched_levenberg_marquardt


class TensorFittingEngine:
    """High-performance tensor fitting engine for hyperspectral data."""
    
    def __init__(self):
        self.timings = {}

    def fit_spectra(
        self,
        x: np.ndarray,
        Y: np.ndarray,
        fit_model: dict,
        weights: np.ndarray = None,
        fit_params: dict = None,
        progress_callback=None,
        cancel_check=None,
        print_benchmark: bool = True,
    ):
        """Fit all spectra simultaneously using the tensor engine.

        Args:
            x: (M,) or (N, M) wavenumber axis
            Y: (N, M) intensity matrix (already preprocessed)
            fit_model: fit model dict
            weights: (N, M) fit weights (optional)
            fit_params: dict with 'method', 'xtol', 'max_ite', etc.
            progress_callback: callable(current, total)
            cancel_check: callable() → bool
            print_benchmark: if True, print the benchmark fitting times.

        Returns:
            p_full: (N, K) fitted parameters
            success: (N,) bool array
            rsquared: (N,) array
            best_fits: (N, M) array
            Y_peaks: list of (N, M) arrays for individual peaks
            param_names: list of str
        """
        N = Y.shape[0]
        t_total = time.perf_counter()

        # ─── 1. Build evaluator ───
        evaluator = TensorEvaluator.from_fit_model(fit_model)
        param_names = evaluator._param_names

        if evaluator.n_params_free == 0:
            if progress_callback:
                progress_callback(N, N)
            return (
                np.zeros((N, evaluator._n_total)), 
                np.ones(N, dtype=bool), 
                np.zeros(N), 
                np.zeros_like(Y), 
                [],
                param_names
            )

        # Detect shared x-axis
        shared_x = x.ndim == 1

        # ─── 2. Build initial parameter matrix ───
        t0 = time.perf_counter()
        p0 = evaluator.build_p0_matrix(x, Y)
            
        evaluator.apply_noise_threshold(x, Y, p0, fit_params)
        t_step3 = time.perf_counter() - t0
        self.timings["Step 3 - build p0"] = f"{t_step3:.3f}s"
        if print_benchmark:
            print(f"[TensorEngine] Step 3 - build p0: {t_step3:.3f}s")

        # ─── 3. Parse fit parameters ───
        if fit_params is None:
            fit_params = {}
        xtol = float(fit_params.get("xtol", 1e-4))
        ftol = float(fit_params.get("ftol", 1e-4))
        max_ite = int(fit_params.get("max_ite", 200))

        # ─── 4. TENSOR FIT ───
        t0 = time.perf_counter()
        p_opt, success, cost = batched_levenberg_marquardt(
            x=x,
            Y_data=Y,
            evaluate_fn=evaluator.evaluate,
            jacobian_fn=evaluator.jacobian,
            p0=p0,
            lower_bounds=evaluator.lower_bounds,
            upper_bounds=evaluator.upper_bounds,
            weights=weights,
            max_iter=max_ite,
            xtol=xtol,
            ftol=ftol,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
        fit_time = time.perf_counter() - t0
        self.timings["Step 4 - tensor fit"] = f"{fit_time:.3f}s ({fit_time/N*1000:.3f} ms/spectrum, {success.sum()}/{N} converged)"
        if print_benchmark:
            print(f"[TensorEngine] Step 4 - tensor fit: {fit_time:.3f}s ({fit_time/N*1000:.1f} ms/spectrum, {success.sum()}/{N} converged)")

        evaluator.apply_noise_threshold(x, Y, p_opt, fit_params, p0_matrix=p0)
        
        # ─── 5. Write back results (batch-optimized) ───
        t0 = time.perf_counter()
        p_full, success, rsquared, best_fits, Y_peaks = evaluator.build_results_batch(
            p_opt, x, Y, success, weights, shared_x
        )
        t_step5 = time.perf_counter() - t0
        self.timings["Step 5 - write_back"] = f"{t_step5:.3f}s"
        if print_benchmark:
            print(f"[TensorEngine] Step 5 - write_back: {t_step5:.3f}s")
            print(f"[TensorEngine] TOTAL: {time.perf_counter()-t_total:.3f}s")
        
        return p_full, success, rsquared, best_fits, Y_peaks, param_names
