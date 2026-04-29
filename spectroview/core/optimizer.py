# spectroview/core/optimizer.py
"""
Core fitting optimizer for the batch engine.

Wraps scipy.optimize.least_squares with support for:
- Single spectrum fitting
- Multi-threaded batch fitting (ThreadPoolExecutor)
- Two-stage coarse → refined fitting
- Thread-safe progress reporting
"""

import warnings
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import least_squares

from spectroview.core.models import PeakModelEvaluator


def fit_single_spectrum(x, y, evaluator, p0, bounds, method="trf",
                        xtol=1e-4, max_nfev=None, weights=None):
    """Fit a single spectrum using scipy.optimize.least_squares.

    Args:
        x: 1D array of x-values
        y: 1D array of y-data (baseline-subtracted)
        evaluator: PeakModelEvaluator instance
        p0: Initial free parameter vector
        bounds: (lower, upper) bound arrays for free params
        method: Optimization method ("trf", "dogbox", or "lm")
        xtol: Relative tolerance for convergence
        max_nfev: Maximum number of function evaluations
        weights: Optional 1D array of weights

    Returns:
        tuple: (p_opt, success, cost)
            - p_opt: Optimized free parameter vector
            - success: Whether optimization converged
            - cost: Final residual cost (sum of squares / 2)
    """
    if max_nfev is None:
        max_nfev = 200 * len(p0)

    # Ensure p0 is within bounds (handle infinite bounds safely)
    lb, ub = bounds
    lb_safe = np.where(np.isfinite(lb), lb + 1e-10, lb)
    ub_safe = np.where(np.isfinite(ub), ub - 1e-10, ub)
    p0 = np.clip(p0, lb_safe, ub_safe)

    # Handle infinite bounds for the "lm" method (which doesn't support bounds)
    if method == "lm":
        # Levenberg-Marquardt doesn't support bounds — use trf fallback if bounds exist
        has_finite_bounds = np.any(np.isfinite(lb)) or np.any(np.isfinite(ub))
        if has_finite_bounds:
            method = "trf"

    def residual_func(p):
        r = evaluator.residual(p, x, y)
        if weights is not None:
            r = r * weights
        return r

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if method == "lm":
                result = least_squares(
                    residual_func, p0,
                    method="lm",
                    xtol=xtol,
                    max_nfev=max_nfev,
                )
            else:
                result = least_squares(
                    residual_func, p0,
                    bounds=bounds,
                    method=method,
                    xtol=xtol,
                    max_nfev=max_nfev,
                )

        # Check convergence (status > 0 means converged)
        success = result.status > 0
        return result.x, success, result.cost

    except Exception:
        # If optimization fails, return initial guess as unsuccessful
        return p0, False, np.inf


def _build_fit_mask(x, y, n_free, fit_negative=False):
    """Build a boolean mask for valid data points to fit.

    Only removes truly invalid data (negative values when fit_negative=False).
    Does NOT remove low-signal points — the optimizer handles those via
    the residual function.

    Args:
        x: x-values array
        y: y-values array
        n_free: Number of free parameters (minimum data points needed)
        fit_negative: Whether to include negative y-values

    Returns:
        tuple: (x_fit, y_fit) arrays to use for fitting, or (None, None) if
               insufficient data points
    """
    mask = np.ones(len(x), dtype=bool)

    if not fit_negative:
        mask[y < 0] = False

    n_valid = np.sum(mask)
    if n_valid <= n_free:
        return None, None

    if not np.all(mask):
        return x[mask], y[mask]
    return x, y


def fit_batch_sequential(x, Y_matrix, evaluator, p0_array, bounds,
                         method="trf", xtol=1e-4, max_nfev=None,
                         traversal_order=None, propagator=None,
                         progress_callback=None, cancel_check=None,
                         fit_negative=False, coef_noise=0):
    """Fit all spectra sequentially with optional neighbor propagation.

    Args:
        x: 1D shared x-axis array
        Y_matrix: 2D array (N_spectra, N_wavelengths) of y-data
        evaluator: PeakModelEvaluator instance
        p0_array: 2D array (N_spectra, N_free_params) of initial guesses,
                  or 1D array (N_free_params,) to use same guess for all
        bounds: (lower, upper) bound arrays for free params
        method: Optimization method
        xtol: Relative tolerance
        max_nfev: Maximum function evaluations per spectrum
        traversal_order: Array of indices defining fit order (default: sequential)
        propagator: NeighborPropagator instance for parameter propagation
        progress_callback: callable(current, total) for progress updates
        cancel_check: callable() -> bool to check for cancellation
        fit_negative: Whether to fit negative y-values
        coef_noise: Noise coefficient (reserved for future use, currently ignored
                    to match fitspy behavior)

    Returns:
        results: list of (p_opt, success, cost) tuples, one per spectrum
                 (in original order, not traversal order)
    """
    n_spectra = Y_matrix.shape[0]
    n_free = evaluator.n_params_free

    if traversal_order is None:
        traversal_order = np.arange(n_spectra)

    # Handle p0: either per-spectrum or broadcast
    if p0_array.ndim == 1:
        # Same initial guess for all spectra
        default_p0 = p0_array.copy()
    else:
        default_p0 = None

    results = [None] * n_spectra

    for step, idx in enumerate(traversal_order):
        # Check cancellation
        if cancel_check and cancel_check():
            # Fill remaining with failures
            for remaining_idx in traversal_order[step:]:
                if results[remaining_idx] is None:
                    p_dummy = default_p0 if default_p0 is not None else p0_array[remaining_idx]
                    results[remaining_idx] = (p_dummy, False, np.inf)
            break

        y = Y_matrix[idx]

        # Get initial guess
        if propagator is not None:
            # Use neighbor's fitted params as initial guess
            p0_default = default_p0 if default_p0 is not None else p0_array[idx]
            p0 = propagator.get_initial_guess(idx, p0_default)
        elif default_p0 is not None:
            p0 = default_p0.copy()
        else:
            p0 = p0_array[idx].copy()

        # Build mask for valid data points (only removes negatives if needed)
        x_fit, y_fit = _build_fit_mask(x, y, n_free, fit_negative=fit_negative)

        if x_fit is None:
            # Not enough data points — skip
            results[idx] = (p0, False, np.inf)
            if progress_callback:
                progress_callback(step + 1, n_spectra)
            continue

        p_opt, success, cost = fit_single_spectrum(
            x_fit, y_fit, evaluator, p0, bounds,
            method=method, xtol=xtol, max_nfev=max_nfev
        )

        results[idx] = (p_opt, success, cost)

        # Store result for neighbor propagation
        if propagator is not None and success:
            propagator.store_result(idx, p_opt)

        # Report progress
        if progress_callback:
            progress_callback(step + 1, n_spectra)

    return results


def fit_batch_threaded(x, Y_matrix, evaluator, p0_array, bounds,
                       method="trf", xtol=1e-4, max_nfev=None,
                       ncpus=4, progress_callback=None, cancel_check=None,
                       fit_negative=False, coef_noise=0):
    """Fit spectra in parallel using ThreadPoolExecutor.

    Uses threads (not processes) since scipy.optimize.least_squares releases
    the GIL during the C-level computation. This avoids all serialization
    overhead compared to ProcessPoolExecutor + dill.

    Note: This does NOT use neighbor propagation (since spectra are fitted
    in parallel without a defined order). For maps, prefer sequential fitting
    with propagation when the number of spectra is moderate, or use threaded
    fitting when the dataset is very large and initial guesses are good.

    Args:
        x: 1D shared x-axis array
        Y_matrix: 2D array (N_spectra, N_wavelengths)
        evaluator: PeakModelEvaluator instance
        p0_array: Initial guesses (1D broadcast or 2D per-spectrum)
        bounds: (lower, upper) bound arrays
        method: Optimization method
        xtol: Tolerance
        max_nfev: Max function evaluations
        ncpus: Number of threads
        progress_callback: callable(current, total)
        cancel_check: callable() -> bool
        fit_negative: Whether to fit negative values
        coef_noise: Noise coefficient (reserved, currently ignored)

    Returns:
        results: list of (p_opt, success, cost) tuples
    """
    n_spectra = Y_matrix.shape[0]
    n_free = evaluator.n_params_free

    if p0_array.ndim == 1:
        default_p0 = p0_array
    else:
        default_p0 = None

    results = [None] * n_spectra
    completed = [0]  # mutable counter for thread-safe progress

    def _fit_one(idx):
        """Fit a single spectrum (called from thread pool)."""
        y = Y_matrix[idx]
        p0 = default_p0.copy() if default_p0 is not None else p0_array[idx].copy()

        x_fit, y_fit = _build_fit_mask(x, y, n_free, fit_negative=fit_negative)
        if x_fit is None:
            return idx, (p0, False, np.inf)

        p_opt, success, cost = fit_single_spectrum(
            x_fit, y_fit, evaluator, p0, bounds,
            method=method, xtol=xtol, max_nfev=max_nfev
        )
        return idx, (p_opt, success, cost)

    with ThreadPoolExecutor(max_workers=ncpus) as executor:
        futures = {}
        for idx in range(n_spectra):
            if cancel_check and cancel_check():
                break
            future = executor.submit(_fit_one, idx)
            futures[future] = idx

        for future in as_completed(futures):
            if cancel_check and cancel_check():
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break

            try:
                idx, result = future.result()
                results[idx] = result
            except Exception:
                idx = futures[future]
                p0 = default_p0.copy() if default_p0 is not None else p0_array[idx].copy()
                results[idx] = (p0, False, np.inf)

            completed[0] += 1
            if progress_callback:
                progress_callback(completed[0], n_spectra)

    # Fill any None results (from cancellation)
    for i in range(n_spectra):
        if results[i] is None:
            p0 = default_p0.copy() if default_p0 is not None else p0_array[i].copy()
            results[i] = (p0, False, np.inf)

    return results
