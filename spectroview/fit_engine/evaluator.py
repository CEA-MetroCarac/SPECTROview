# spectroview/core2/evaluator.py
"""Tensor Evaluator — maps a fit_model dict to the batched tensor API.

Handles:
  - Parsing peak_models dict → flat (K,) parameter layout
  - Routing to the correct batched model + Jacobian functions
  - Mixed model types (e.g. peak 1 = Lorentzian, peak 2 = Gaussian)
  - Fixed vs free parameters
  - Building FitResult objects and writing back to MSpectrum
"""

import numpy as np
from spectroview.fit_engine.scalar_models import FitResult, ParamValue, PEAK_MODEL_REGISTRY
from spectroview.fit_engine.models import BATCHED_MODELS, numerical_jacobian
from fitspy.core.utils import eval_noise_amplitude


class TensorEvaluator:
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
                peak_num = len(ev._peaks) + 1
                prefix = f"m{peak_num:02d}_"

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
        vary_arr = np.array(ev._param_vary)
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

    def build_result(self, p_free, x, y, success, weights=None):
        """Build a FitResult for a single spectrum.

        Args:
            p_free: (K_free,) optimised free parameters
            x: (M,) x-axis
            y: (M,) measured data
            success: bool
            weights: (M,) weights mask (optional)
        """
        p_full = self._to_full(p_free)
        best_fit = self.evaluate(x, p_free[None, :])[0]

        params_dict = {}
        for i, name in enumerate(self._param_names):
            params_dict[name] = p_full[i]

        rsquared = 0.0
        if y is not None:
            if weights is not None:
                residuals = weights * (y - best_fit)
                ss_res = np.sum(residuals**2)
                if weights.sum() > 0:
                    y_mean = np.average(y, weights=weights)
                    ss_tot = np.sum(weights * (y - y_mean)**2)
                else:
                    ss_tot = 0.0
            else:
                ss_res = np.sum((y - best_fit)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                
            if ss_tot > 0:
                rsquared = 1.0 - (ss_res / ss_tot)

        return FitResult(success=success, params_dict=params_dict, best_fit=best_fit, rsquared=rsquared)

    def write_back_to_spectrum(self, spectrum, fit_result):
        """Write fit result back to MSpectrum (same interface as core)."""
        spectrum.result_fit = fit_result

        for i, peak_model in enumerate(spectrum.peak_models):
            if i >= len(self._peaks):
                break
            for key in peak_model.param_names:
                name = key[4:]  # remove actual prefix 'mXX_' from fitspy
                tensor_key = f"m{i+1:02d}_{name}" # use the sequentially generated prefix from the evaluator
                if tensor_key in fit_result.params:
                    peak_model.set_param_hint(name, value=fit_result.params[tensor_key].value)

        if spectrum.bkg_model is not None:
            for key in spectrum.bkg_model.param_names:
                if key in fit_result.params:
                    spectrum.bkg_model.set_param_hint(
                        key, value=fit_result.params[key].value
                    )

    def extract_p0_from_spectrum(self, spectrum):
        """Extract free parameter array from an already-fitted MSpectrum."""
        p0_full = np.array(self._param_values, dtype=np.float64)

        for i, peak_model in enumerate(spectrum.peak_models):
            if i >= len(self._peaks):
                break
            for key, hint in peak_model.param_hints.items():
                full_name = f"m{i+1:02d}_{key}"
                try:
                    idx = self._param_names.index(full_name)
                    if 'value' in hint and hint['value'] is not None:
                        p0_full[idx] = hint['value']
                except ValueError:
                    pass

        return p0_full[self._free_idx]

    def build_p0_matrix(self, spectra):
        """Build (N, K_free) initial guess matrix with per-spectrum amplitude scaling.

        For each spectrum, scales the amplitude parameters proportionally
        to the spectrum's actual intensity at the peak position.
        """
        N = len(spectra)
        K = self.n_params_free
        p0_base = self.initial_params.copy()
        p0_matrix = np.tile(p0_base, (N, 1))  # (N, K)

        # Find amplitude indices and their associated x0 values
        for model_name, slc, eval_fn, jac_fn, has_jac in self._peaks:
            # Get parameter names for this peak
            peak_pnames = self._param_names[slc]

            # Find ampli index in the FREE param space
            for local_i, pname in enumerate(peak_pnames):
                global_i = slc.start + local_i
                if "ampli" in pname and global_i in self._free_idx:
                    free_i = np.searchsorted(self._free_idx, global_i)

                    # Find corresponding x0
                    x0_val = None
                    for local_j, pname_j in enumerate(peak_pnames):
                        if "x0" in pname_j:
                            x0_val = self._param_values[slc.start + local_j]
                            break

                    for n in range(N):
                        spectrum = spectra[n]
                        x_n = spectrum.x
                        if x_n is None or x0_val is None or x0_val < x_n[0] or x0_val > x_n[-1]:
                            continue
                            
                        closest = np.argmin(np.abs(x_n - x0_val))
                        low = max(0, closest - 1)
                        high = min(len(x_n), closest + 2)
                        
                        y = getattr(spectrum, 'y_no_outliers', spectrum.y)
                        if y is not None and len(y) > closest:
                            window = np.maximum(np.abs(y[low:high]), 1e-6)
                            data_amp = float(np.max(window))
                            model_amp = max(abs(p0_base[free_i]), 1e-6)
                            ratio = data_amp / model_amp
                            if 0.01 < ratio < 100:
                                p0_matrix[n, free_i] = data_amp

        return p0_matrix

    def apply_noise_threshold(self, spectra, p_matrix, fit_params):
        """Force ampli=0 and fwhm=0 for peaks located in noisy areas."""
        if fit_params is None:
            return
        coef_noise = float(fit_params.get("coef_noise", 0))
        if coef_noise <= 0:
            return

        for i, spectrum in enumerate(spectra):
            y = spectrum.y
            if y is None:
                continue

            if not hasattr(spectrum, '_fit_ampli_noise'):
                spectrum._fit_ampli_noise = eval_noise_amplitude(y)
                spectrum._fit_ymean = np.convolve(y, np.ones(5, dtype=np.float64) / 5.0, mode='same')
                
            ampli_noise = spectrum._fit_ampli_noise
            ymean = spectrum._fit_ymean
            noise_level = coef_noise * ampli_noise
            x_array = spectrum.x
            if x_array is None:
                continue

            for model_name, slc, eval_fn, jac_fn, has_jac in self._peaks:
                x0_val = None
                peak_pnames = self._param_names[slc]
                for local_j, pname_j in enumerate(peak_pnames):
                    if "x0" in pname_j:
                        global_idx = slc.start + local_j
                        x0_val = self._param_values[global_idx]
                        if global_idx in self._free_idx:
                            free_idx = np.searchsorted(self._free_idx, global_idx)
                            x0_val = p_matrix[i, free_idx]
                        break

                if x0_val is None or x0_val < x_array[0] or x0_val > x_array[-1]:
                    continue

                ind = np.argmin(np.abs(x_array - x0_val))
                if ymean[ind] < noise_level:
                    for local_j, pname_j in enumerate(peak_pnames):
                        if "ampli" in pname_j or "fwhm" in pname_j:
                            global_idx = slc.start + local_j
                            if global_idx in self._free_idx:
                                free_idx = np.searchsorted(self._free_idx, global_idx)
                                p_matrix[i, free_idx] = 0.0


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
