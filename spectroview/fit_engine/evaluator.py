"""Tensor Evaluator — maps a fit_model dict to the batched tensor API.

Handles:
  - Parsing peak_models dict → flat (K,) parameter layout
  - Routing to the correct batched model + Jacobian functions
  - Mixed model types (e.g. peak 1 = Lorentzian, peak 2 = Gaussian)
  - Fixed vs free parameters
  - Building batched fit results (parameters, R², best fits, per-peak curves)
"""

import numpy as np
from spectroview.fit_engine.scalar_models import PEAK_MODEL_REGISTRY
from spectroview.fit_engine.models import BATCHED_MODELS, numerical_jacobian
from spectroview.fit_engine.noise import mad_noise, moving_average_5


class VBFevaluator:
    """Manages the mapping between a fit_model dict and tensor parameter matrices.

    After construction, provides:
        evaluate(x, p)   → (N, M)  predicted spectra
        jacobian(x, p)   → (N, M, K_free)  Jacobian
        initial_params   → (K_free,)  initial values
        lower / upper    → (K_free,)  bounds
    """

    def __init__(self):
        self._peaks = []            # list of (model_name, param_slice, eval_fn, jac_fn)
        self._param_names = []      # full prefixed names e.g. "m01_ampli"
        self._param_values = []     # initial values (all params)
        self._param_lower = []      # lower bounds
        self._param_upper = []      # upper bounds
        self._param_vary = []       # bool: free or fixed
        self._exprs = []            # list of (idx, expr_str) for parameters with expressions

        self._n_total = 0           # total number of params (free + fixed)
        self._free_idx = None       # indices of free params in the full vector
        self._fixed_idx = None
        self._fixed_values = None

    @classmethod
    def from_fit_model(cls, fit_model_dict):
        """Build evaluator from a fit model dictionary (spectrum.save() or JSON)."""
        ev = cls()

        peak_models_dict = fit_model_dict.get("peak_models", {})
        sorted_keys = sorted(peak_models_dict.keys(), key=lambda k: int(k))

        for peak_key in sorted_keys:
            peak_def = peak_models_dict[peak_key]
            for model_name, param_hints in peak_def.items():
                peak_num = int(peak_key) + 1
                prefix = f"P{peak_num}_"

                # Look up batched model
                if model_name in BATCHED_MODELS:
                    eval_fn, jac_fn, canonical_params = BATCHED_MODELS[model_name]
                    has_analytical_jac = True
                else:
                    # Fallback: use scalar model from core, wrap it
                    if model_name not in PEAK_MODEL_REGISTRY:
                        raise ValueError(f"Unknown peak model: {model_name}")
                    scalar_fn, canonical_params = PEAK_MODEL_REGISTRY[model_name]
                    eval_fn = _make_batched_scalar(scalar_fn, len(canonical_params))
                    jac_fn = None
                    has_analytical_jac = False

                start = len(ev._param_names)

                for pname in canonical_params:
                    hints = param_hints.get(pname, {})
                    value = hints.get("value", 1.0)
                    pmin = hints.get("min", -np.inf)
                    pmax = hints.get("max", np.inf)
                    vary = hints.get("vary", True)
                    
                    expr = hints.get("expr", "")
                    if expr is None:
                        expr = ""
                    expr = str(expr).strip()

                    if pmin is None:
                        pmin = -np.inf
                    if pmax is None:
                        pmax = np.inf
                        
                    if expr and expr.lower() != "none":
                        vary = False
                        ev._exprs.append((len(ev._param_names), expr))

                    value = float(max(pmin, min(pmax, value)))

                    ev._param_names.append(prefix + pname)
                    ev._param_values.append(value)
                    ev._param_lower.append(float(pmin))
                    ev._param_upper.append(float(pmax))
                    ev._param_vary.append(bool(vary))

                end = len(ev._param_names)
                ev._peaks.append((model_name, slice(start, end), eval_fn, jac_fn, has_analytical_jac))

        # Finalise indexing
        ev._n_total = len(ev._param_names)
        vary_arr = np.array(ev._param_vary, dtype=bool)
        ev._free_idx = np.where(vary_arr)[0]
        ev._fixed_idx = np.where(~vary_arr)[0]
        ev._fixed_values = np.array(ev._param_values)[ev._fixed_idx]

        return ev

    # ── Public properties ────────────────────────────────────────────────

    @property
    def n_params_free(self):
        return len(self._free_idx)

    @property
    def n_params_total(self):
        return self._n_total

    @property
    def initial_params(self):
        """(K_free,) initial values for free parameters only."""
        return np.array(self._param_values)[self._free_idx]

    @property
    def lower_bounds(self):
        return np.array(self._param_lower)[self._free_idx]

    @property
    def upper_bounds(self):
        return np.array(self._param_upper)[self._free_idx]

    # ── Reconstruction: free → full ──────────────────────────────────────

    def _to_full(self, p_free):
        """Expand (N, K_free) or (K_free,) to (..., K_total)."""
        if p_free.ndim == 1:
            p_full = np.empty(self._n_total)
            p_full[self._free_idx] = p_free
            p_full[self._fixed_idx] = self._fixed_values
        else:
            N = p_free.shape[0]
            p_full = np.empty((N, self._n_total))
            p_full[:, self._free_idx] = p_free
            p_full[:, self._fixed_idx] = self._fixed_values
            
        if self._exprs:
            loc = {name: p_full[..., i] for i, name in enumerate(self._param_names)}
            loc["np"] = np
            loc["pi"] = np.pi
            loc["sqrt"] = np.sqrt
            loc["exp"] = np.exp
            loc["log"] = np.log
            loc["sin"] = np.sin
            loc["cos"] = np.cos
            
            pending = list(self._exprs)
            for _ in range(len(pending)):
                if not pending:
                    break
                next_pending = []
                for idx, expr in pending:
                    try:
                        val = eval(expr, {"__builtins__": None}, loc)
                        if p_full.ndim == 1:
                            p_full[idx] = val
                        else:
                            p_full[:, idx] = val
                        loc[self._param_names[idx]] = p_full[..., idx]
                    except NameError:
                        next_pending.append((idx, expr))
                    except Exception:
                        next_pending.append((idx, expr))
                if len(next_pending) == len(pending):
                    break
                pending = next_pending
                
        return p_full

    # ── Model evaluation ─────────────────────────────────────────────────

    def evaluate(self, x, p_free):
        """Evaluate the composite model for all spectra.

        Args:
            x: (M,) or (N, M)  — shared or per-spectrum x-axis
            p_free: (N, K_free)  free parameters

        Returns: (N, M)
        """
        p_full = self._to_full(p_free)
        N = p_full.shape[0] if p_full.ndim == 2 else 1
        M = x.shape[-1] if hasattr(x, 'shape') else len(x)
        Y = np.zeros((N, M))

        for model_name, slc, eval_fn, jac_fn, has_jac in self._peaks:
            Y += eval_fn(x, p_full[:, slc] if p_full.ndim == 2 else p_full[slc][None, :])

        return Y

    def jacobian(self, x, p_free):
        """Compute Jacobian w.r.t. free parameters.

        Returns: (N, M, K_free)
        """
        p_full = self._to_full(p_free)
        N = p_full.shape[0]
        M = x.shape[-1] if hasattr(x, 'shape') else len(x)
        J_full = np.zeros((N, M, self._n_total))

        for model_name, slc, eval_fn, jac_fn, has_jac in self._peaks:
            p_peak = p_full[:, slc]
            if has_jac and jac_fn is not None:
                J_full[:, :, slc] = jac_fn(x, p_peak)
            else:
                J_full[:, :, slc] = numerical_jacobian(eval_fn, x, p_peak)

        if not getattr(self, '_exprs', []):
            # Return only columns for free parameters
            return J_full[:, :, self._free_idx]

        # Chain rule via J_expr: J_true = J_full @ J_expr
        K_free = p_free.shape[1] if p_free.ndim == 2 else len(p_free)
        is_1d = p_free.ndim == 1
        p_free_2d = p_free[None, :] if is_1d else p_free
        N_batch = p_free_2d.shape[0]
        
        J_expr = np.zeros((N_batch, self._n_total, K_free))
        eps = 1e-8
        
        p_full_2d = p_full[None, :] if is_1d else p_full
        
        for f in range(K_free):
            p_free_plus = p_free_2d.copy()
            step = np.maximum(np.abs(p_free_2d[:, f]) * 1e-6, eps)
            p_free_plus[:, f] += step
            p_full_plus = self._to_full(p_free_plus)
            J_expr[:, :, f] = (p_full_plus - p_full_2d) / step[:, None]
            
        if is_1d:
            J_true = np.einsum('nmk,nkf->nmf', J_full[None, :, :], J_expr)[0]
        else:
            J_true = np.einsum('nmk,nkf->nmf', J_full, J_expr)
            
        return J_true

    # ── Result construction ──────────────────────────────────────────────

    def build_results_batch(self, p_opt, x, Y_data, success, weights, shared_x):
        """Build FitResults for all spectra in one vectorized pass.

        Args:
            p_opt: (N, K_free) optimised parameters
            x: (M,) or (N, M) x-axis
            Y_data: (N, M) measured data
            success: (N,) bool
            weights: (N, M) or None
            shared_x: bool — whether all spectra share the same x-axis
            
        Returns:
            p_full: (N, K_total) full parameters
            success: (N,) bool array
            rsquared: (N,) r2 array
            best_fits: (N, M) best fit array
            Y_peaks: list of (N, M) for each peak model
        """
        N = p_opt.shape[0]

        # ── Build full parameter matrix once ──
        p_full = self._to_full(p_opt)  # (N, K_total)

        # ── Evaluate each peak once; best_fit is their sum (avoids a second
        #    pass over every peak that a separate evaluate() call would cost) ──
        M = x.shape[-1] if hasattr(x, "shape") else len(x)
        Y_peaks = [eval_fn(x, p_full[:, slc]) for _, slc, eval_fn, _, _ in self._peaks]
        best_fits = np.zeros((N, M))
        for Y_p in Y_peaks:
            best_fits += Y_p

        # ── Vectorized R² ──
        if weights is not None:
            residuals = weights * (Y_data - best_fits)
            ss_res = np.sum(residuals * residuals, axis=1)  # (N,)
            w_sum = weights.sum(axis=1)
            # Weighted mean
            y_mean = np.where(
                w_sum > 0,
                np.sum(weights * Y_data, axis=1) / np.maximum(w_sum, 1e-30),
                0.0
            )
            ss_tot = np.sum(weights * (Y_data - y_mean[:, None])**2, axis=1)
        else:
            ss_res = np.sum((Y_data - best_fits)**2, axis=1)
            y_mean = Y_data.mean(axis=1)
            ss_tot = np.sum((Y_data - y_mean[:, None])**2, axis=1)

        rsquared = np.where(ss_tot > 0, 1.0 - ss_res / np.maximum(ss_tot, 1e-30), 0.0)

        return p_full, success, rsquared, best_fits, Y_peaks

    def build_p0_matrix(self, x: np.ndarray, Y: np.ndarray):
        """Build (N, K_free) initial guess matrix with per-spectrum amplitude scaling."""
        N = Y.shape[0]
        p0_base = self.initial_params.copy()
        p0_matrix = np.tile(p0_base, (N, 1))  # (N, K)

        for model_name, slc, eval_fn, jac_fn, has_jac in self._peaks:
            peak_pnames = self._param_names[slc]
            for local_i, pname in enumerate(peak_pnames):
                global_i = slc.start + local_i
                if "ampli" in pname and global_i in self._free_idx:
                    free_i = np.searchsorted(self._free_idx, global_i)
                    x0_val = None
                    for local_j, pname_j in enumerate(peak_pnames):
                        if "x0" in pname_j:
                            x0_val = self._param_values[slc.start + local_j]
                            break

                    if x.ndim == 1:
                        # [NORMAL CASE - 99% of datasets] 
                        # All spectra share the exact same X-axis. This means the index `closest` 
                        # for `x0` is identical for all spectra. We can compute the window bounds 
                        # once and slice the entire 2D Y matrix in a single, highly vectorized 
                        # Numpy operation, making this practically instantaneous.
                        if x0_val is None or x0_val < x[0] or x0_val > x[-1]:
                            continue
                        
                        closest = np.argmin(np.abs(x - x0_val))
                        low = max(0, closest - 1)
                        high = min(len(x), closest + 2)
                        
                        if Y.shape[1] > closest:
                            window = np.maximum(np.abs(Y[:, low:high]), 1e-6)
                            data_amp = np.max(window, axis=1) # shape: (N,)
                            model_amp = max(abs(p0_base[free_i]), 1e-6)
                            ratio = data_amp / model_amp
                            
                            mask = (ratio > 0.01) & (ratio < 100)
                            p0_matrix[mask, free_i] = data_amp[mask]
                    else:
                        # [EDGE CASE - Uncalibrated/Raw datasets]
                        # Every single spectrum has its own unique X-axis calibration (x is a 2D matrix).
                        # Because the X-axis varies per row, `x0` might land at index 150 for Spectrum 1, 
                        # but index 154 for Spectrum 2. Since the window slice boundaries (`low:high`) vary 
                        # per row, we cannot easily vector-slice the 2D matrix. We fall back to a safe 
                        # sequential loop to guarantee correct initial guesses without crashing.
                        for n in range(N):
                            x_n = x[n]
                            y = Y[n]
                            if x0_val is None or x0_val < x_n[0] or x0_val > x_n[-1]:
                                continue
                                
                            closest = np.argmin(np.abs(x_n - x0_val))
                            low = max(0, closest - 1)
                            high = min(len(x_n), closest + 2)
                            
                            if len(y) > closest:
                                window = np.maximum(np.abs(y[low:high]), 1e-6)
                                data_amp = float(np.max(window))
                                model_amp = max(abs(p0_base[free_i]), 1e-6)
                                ratio = data_amp / model_amp
                                if 0.01 < ratio < 100:
                                    p0_matrix[n, free_i] = data_amp

        return p0_matrix

    @staticmethod
    def compute_noise_stats(Y: np.ndarray, coef_noise: float):
        """Return (ymean, noise_level) used to flag noisy x-positions.

        This depends only on the raw data and coef_noise (not on the current
        parameter matrix), so callers that need to invoke
        apply_noise_threshold() more than once on the same Y (e.g. once
        before fitting, once after) can compute it a single time and pass
        it in via the `noise_stats` argument instead of paying for the
        median/smoothing pass twice.
        """
        ymean = moving_average_5(Y)
        return ymean, coef_noise * mad_noise(Y)  # noise_level: (N,) or scalar

    def apply_noise_threshold(self, x: np.ndarray, Y: np.ndarray, p_matrix: np.ndarray, fit_params: dict, p0_matrix=None, noise_stats=None):
        """Force ampli=0 and fwhm=0 for peaks located in noisy areas (highly vectorized)."""
        if fit_params is None:
            return
        coef_noise = float(fit_params.get("coef_noise", 0))
        if coef_noise <= 0:
            return

        N = Y.shape[0]

        if noise_stats is not None:
            ymean, noise_level = noise_stats
        else:
            ymean, noise_level = self.compute_noise_stats(Y, coef_noise)

        for model_name, slc, eval_fn, jac_fn, has_jac in self._peaks:
            peak_pnames = self._param_names[slc]
            x0_global_idx = None
            for local_j, pname_j in enumerate(peak_pnames):
                if "x0" in pname_j:
                    x0_global_idx = slc.start + local_j
                    break
            
            if x0_global_idx is None:
                continue
                
            if x0_global_idx in self._free_idx:
                free_idx = np.searchsorted(self._free_idx, x0_global_idx)
                x0_vals = (p0_matrix if p0_matrix is not None else p_matrix)[:, free_idx] # (N,)
            else:
                x0_vals = np.full(N, self._param_values[x0_global_idx]) # (N,)

            # Find closest index in x
            if x.ndim == 1:
                dists = np.abs(x[None, :] - x0_vals[:, None]) # (N, M)
                inds = np.argmin(dists, axis=1) # (N,)
            else:
                dists = np.abs(x - x0_vals[:, None]) # (N, M)
                inds = np.argmin(dists, axis=1) # (N,)
                
            ymean_at_x0 = ymean[np.arange(N), inds] if Y.ndim == 2 else ymean[inds]
            below_noise = ymean_at_x0 < noise_level # (N,) bool array

            for local_j, pname_j in enumerate(peak_pnames):
                global_idx = slc.start + local_j
                if global_idx in self._free_idx:
                    free_idx = np.searchsorted(self._free_idx, global_idx)
                    if "ampli" in pname_j or "fwhm" in pname_j:
                        p_matrix[below_noise, free_idx] = 0.0
                    elif p0_matrix is not None:
                        p_matrix[below_noise, free_idx] = p0_matrix[below_noise, free_idx]


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_batched_scalar(scalar_fn, n_params):
    """Wrap a scalar model function as a batched function.

    Handles both 1D x (shared axis) and 2D x (per-spectrum axis).
    """
    def batched_fn(x, params):
        N = params.shape[0]
        M = x.shape[-1] if hasattr(x, 'shape') else len(x)
        Y = np.empty((N, M))
        for i in range(N):
            args = [params[i, j] for j in range(n_params)]
            xi = x[i] if (hasattr(x, 'ndim') and x.ndim == 2) else x
            Y[i] = scalar_fn(xi, *args)
        return Y
    return batched_fn


def eval_peak_initial(x, p_model):
    """Evaluate a peak model dictionary (from fit_model['peak_models']) on x.
    
    p_model is of format: {shape_name: {param_name: {'value': float, ...}}}
    """
    shape_name = list(p_model.keys())[0]
    param_hints = p_model[shape_name]
    
    if shape_name not in PEAK_MODEL_REGISTRY:
        return np.zeros_like(x)
        
    func, param_names = PEAK_MODEL_REGISTRY[shape_name]
    args = []
    for pname in param_names:
        hints = param_hints.get(pname, {})
        val = hints.get("value", 1.0)
        args.append(val)
        
    return func(x, *args)

