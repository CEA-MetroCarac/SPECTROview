# spectroview/core/models.py
"""
Vectorized peak model evaluation for the batch fitting engine.

Provides pure-NumPy implementations of all supported peak shapes that match
the exact functional forms used by fitspy. Handles the mapping between
structured param_hints dicts and flat parameter vectors for scipy.optimize.
"""

import numpy as np
from copy import deepcopy

# ═══════════════════════════════════════════════════════════════════════════
# Pure-NumPy peak model functions (matching fitspy definitions exactly)
# ═══════════════════════════════════════════════════════════════════════════

_LOG2 = np.log(2.0)
_SQRT_2LOG2 = np.sqrt(2.0 * _LOG2)


def gaussian(x, ampli, fwhm, x0):
    """Gaussian: ampli * exp(-(x-x0)^2 / (2*sigma^2)), sigma = fwhm/(2*sqrt(2*ln2))"""
    sigma = fwhm / (2.0 * _SQRT_2LOG2)
    coef = 1.0 / (2.0 * sigma ** 2 + 1e-30)
    return ampli * np.exp(-coef * (x - x0) ** 2)


def lorentzian(x, ampli, fwhm, x0):
    """Lorentzian: ampli * fwhm^2 / (4*((x-x0)^2 + fwhm^2/4))"""
    return ampli * fwhm ** 2 / (4.0 * ((x - x0) ** 2 + fwhm ** 2 / 4.0) + 1e-6)


def pseudovoigt(x, ampli, fwhm, x0, alpha=0.5):
    """PseudoVoigt: alpha * Gaussian + (1 - alpha) * Lorentzian"""
    return (alpha * gaussian(x, ampli, fwhm, x0) +
            (1.0 - alpha) * lorentzian(x, ampli, fwhm, x0))


def gaussian_asym(x, ampli, fwhm_l, fwhm_r, x0):
    """Asymmetric Gaussian: piecewise Gaussian with different left/right FWHM"""
    left = (x < x0) * gaussian(x, ampli, fwhm_l, x0)
    right = (x >= x0) * gaussian(x, ampli, fwhm_r, x0)
    return left + right


def lorentzian_asym(x, ampli, fwhm_l, fwhm_r, x0):
    """Asymmetric Lorentzian: piecewise Lorentzian with different left/right FWHM"""
    left = (x < x0) * lorentzian(x, ampli, fwhm_l, x0)
    right = (x >= x0) * lorentzian(x, ampli, fwhm_r, x0)
    return left + right


def fano(x, ampli, fwhm, x0, q=1.0):
    """Fano lineshape: ampli * (q + eps)^2 / (1 + eps^2)"""
    gamma_half = fwhm / 2.0
    epsilon = (x - x0) / (gamma_half + 1e-10)
    return ampli * (q + epsilon) ** 2 / (1.0 + epsilon ** 2)


def decay_single_exp(x, A, tau, B):
    """Single exponential decay: A * exp(-x/tau) + B"""
    return A * np.exp(-x / (tau + 1e-30)) + B


def decay_bi_exp(x, A1, tau1, A2, tau2, B):
    """Bi-exponential decay: A1*exp(-x/tau1) + A2*exp(-x/tau2) + B"""
    return A1 * np.exp(-x / (tau1 + 1e-30)) + A2 * np.exp(-x / (tau2 + 1e-30)) + B


# Background model functions
def bkg_constant(x, c):
    """Constant background"""
    return np.full_like(x, c, dtype=np.float64)


def bkg_linear(x, intercept, slope):
    """Linear background: intercept + slope * x"""
    return intercept + slope * x


def bkg_parabolic(x, a, b, c):
    """Parabolic background: a*x^2 + b*x + c"""
    return a * x ** 2 + b * x + c


# ═══════════════════════════════════════════════════════════════════════════
# Model registry — maps model names to (function, ordered_param_names)
# ═══════════════════════════════════════════════════════════════════════════

PEAK_MODEL_REGISTRY = {
    "Gaussian":       (gaussian,       ["ampli", "fwhm", "x0"]),
    "Lorentzian":     (lorentzian,     ["ampli", "fwhm", "x0"]),
    "PseudoVoigt":    (pseudovoigt,    ["ampli", "fwhm", "x0", "alpha"]),
    "GaussianAsym":   (gaussian_asym,  ["ampli", "fwhm_l", "fwhm_r", "x0"]),
    "LorentzianAsym": (lorentzian_asym, ["ampli", "fwhm_l", "fwhm_r", "x0"]),
    "Fano":           (fano,           ["ampli", "fwhm", "x0", "q"]),
    "DecaySingleExp": (decay_single_exp, ["A", "tau", "B"]),
    "DecayBiExp":     (decay_bi_exp,    ["A1", "tau1", "A2", "tau2", "B"]),
}

BKG_MODEL_REGISTRY = {
    "Constant":  (bkg_constant,  ["c"]),
    "Linear":    (bkg_linear,    ["intercept", "slope"]),
    "Parabolic": (bkg_parabolic, ["a", "b", "c"]),
}


# ═══════════════════════════════════════════════════════════════════════════
# Lightweight fit result (compatible with existing code expectations)
# ═══════════════════════════════════════════════════════════════════════════

class ParamValue:
    """Mimics lmfit Parameter with a .value attribute."""
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


class FitResult:
    """Lightweight fit result compatible with the existing codebase.

    Supports:
        result.success          -> bool
        result.params[key]      -> ParamValue with .value
        result.best_fit         -> np.ndarray
        result.best_values      -> dict of {param_name: float} (lmfit compat)
        key in result.params    -> bool
    """

    def __init__(self, success, params_dict, best_fit):
        self.success = success
        self.params = {k: ParamValue(v) for k, v in params_dict.items()}
        self.best_fit = best_fit
        # lmfit ModelResult compatibility: best_values is a plain dict
        self.best_values = dict(params_dict)


# ═══════════════════════════════════════════════════════════════════════════
# PeakModelEvaluator — builds and evaluates composite models
# ═══════════════════════════════════════════════════════════════════════════

class PeakModelEvaluator:
    """Manages the mapping between structured peak model definitions and
    a flat parameter vector for scipy.optimize.least_squares.

    Usage:
        evaluator = PeakModelEvaluator.from_fit_model(fit_model_dict)
        p0 = evaluator.initial_params
        bounds = evaluator.bounds
        residual = evaluator.residual(p, x, y)
    """

    def __init__(self):
        # List of (func, param_names, prefix) tuples for each peak
        self._peak_specs = []
        # Background model spec: (func, param_names) or None
        self._bkg_spec = None

        # Flat parameter layout
        self._param_names = []      # Full prefixed names (e.g., "m01_x0")
        self._param_values = []     # Initial values
        self._param_mins = []       # Lower bounds
        self._param_maxs = []       # Upper bounds
        self._param_vary = []       # Whether each param is free
        self._param_exprs = []      # Expression strings (for info only)

        # Index mapping: _param_slices[i] = (start, end) in the flat vector
        # for the i-th model (peak or bkg)
        self._model_slices = []

        # After build: indices of free vs fixed parameters
        self._free_mask = None
        self._fixed_mask = None
        self._fixed_values = None

        # Number of peak models (excluding bkg)
        self._n_peaks = 0

    @classmethod
    def from_fit_model(cls, fit_model_dict):
        """Build evaluator from a fit model dictionary (as returned by spectrum.save()).

        Expected format:
            fit_model_dict = {
                "peak_models": {
                    "0": {"Lorentzian": {"x0": {"value":..., "min":..., "max":..., "vary":..., "expr":...}, ...}},
                    "1": {"Gaussian": {...}},
                    ...
                },
                "bkg_model": {"Linear": {"intercept": {...}, "slope": {...}}} or None,
                "peak_labels": [...],
                ...
            }
        """
        evaluator = cls()

        # Parse peak models
        peak_models_dict = fit_model_dict.get("peak_models", {})
        # Handle both integer and string keys (load_model returns int, JSON uses str)
        sorted_keys = sorted(peak_models_dict.keys(), key=lambda k: int(k))
        for peak_key in sorted_keys:
            peak_def = peak_models_dict[peak_key]
            for model_name, param_hints in peak_def.items():
                if model_name not in PEAK_MODEL_REGISTRY:
                    raise ValueError(f"Unknown peak model: {model_name}")

                func, canonical_params = PEAK_MODEL_REGISTRY[model_name]
                peak_num = len(evaluator._peak_specs) + 1
                prefix = f"m{peak_num:02d}_"

                start_idx = len(evaluator._param_names)

                for pname in canonical_params:
                    full_name = prefix + pname
                    hints = param_hints.get(pname, {})

                    value = hints.get("value", 1.0)
                    pmin = hints.get("min", -np.inf)
                    pmax = hints.get("max", np.inf)
                    vary = hints.get("vary", True)
                    expr = hints.get("expr", None)

                    # Sanitize bounds
                    if pmin is None:
                        pmin = -np.inf
                    if pmax is None:
                        pmax = np.inf

                    # Clamp initial value to bounds
                    value = max(pmin, min(pmax, value))

                    evaluator._param_names.append(full_name)
                    evaluator._param_values.append(value)
                    evaluator._param_mins.append(pmin)
                    evaluator._param_maxs.append(pmax)
                    evaluator._param_vary.append(bool(vary))
                    evaluator._param_exprs.append(expr or "")

                end_idx = len(evaluator._param_names)
                evaluator._model_slices.append((start_idx, end_idx))
                evaluator._peak_specs.append((func, canonical_params, prefix))

        evaluator._n_peaks = len(evaluator._peak_specs)

        # Parse background model
        bkg_dict = fit_model_dict.get("bkg_model", None)
        if bkg_dict and bkg_dict is not None:
            for model_name, param_hints in bkg_dict.items():
                if model_name in ("None", "none") or model_name not in BKG_MODEL_REGISTRY:
                    break

                func, canonical_params = BKG_MODEL_REGISTRY[model_name]
                start_idx = len(evaluator._param_names)

                for pname in canonical_params:
                    hints = param_hints.get(pname, {})
                    value = hints.get("value", 0.0)
                    pmin = hints.get("min", -np.inf)
                    pmax = hints.get("max", np.inf)
                    vary = hints.get("vary", True)

                    if pmin is None:
                        pmin = -np.inf
                    if pmax is None:
                        pmax = np.inf
                    value = max(pmin, min(pmax, value))

                    evaluator._param_names.append(pname)
                    evaluator._param_values.append(value)
                    evaluator._param_mins.append(pmin)
                    evaluator._param_maxs.append(pmax)
                    evaluator._param_vary.append(bool(vary))
                    evaluator._param_exprs.append("")

                end_idx = len(evaluator._param_names)
                evaluator._model_slices.append((start_idx, end_idx))
                evaluator._bkg_spec = (func, canonical_params)

        # Build index arrays
        evaluator._finalize()
        return evaluator

    def _finalize(self):
        """Pre-compute index arrays for free/fixed parameter separation."""
        vary = np.array(self._param_vary, dtype=bool)
        self._free_mask = np.where(vary)[0]
        self._fixed_mask = np.where(~vary)[0]
        self._fixed_values = np.array(self._param_values, dtype=np.float64)

    @property
    def n_params_total(self):
        return len(self._param_names)

    @property
    def n_params_free(self):
        return len(self._free_mask)

    @property
    def param_names(self):
        return self._param_names

    @property
    def initial_params(self):
        """Return initial values for FREE parameters only."""
        all_vals = np.array(self._param_values, dtype=np.float64)
        return all_vals[self._free_mask]

    @property
    def bounds(self):
        """Return (lower, upper) bound arrays for FREE parameters only."""
        mins = np.array(self._param_mins, dtype=np.float64)
        maxs = np.array(self._param_maxs, dtype=np.float64)
        return (mins[self._free_mask], maxs[self._free_mask])

    def get_all_initial_params(self):
        """Return initial values for ALL parameters (free + fixed)."""
        return np.array(self._param_values, dtype=np.float64)

    def free_to_full(self, p_free):
        """Expand free parameter vector to full parameter vector."""
        full = self._fixed_values.copy()
        full[self._free_mask] = p_free
        return full

    def full_to_free(self, p_full):
        """Extract free parameters from full vector."""
        return p_full[self._free_mask]

    def evaluate(self, x, p_full):
        """Evaluate the composite model (sum of all peaks + bkg) at x.

        Args:
            x: 1D numpy array of x-values
            p_full: Full parameter vector (all params, free + fixed)

        Returns:
            1D numpy array of model values
        """
        y = np.zeros_like(x, dtype=np.float64)

        n_models = self._n_peaks + (1 if self._bkg_spec else 0)

        for i in range(n_models):
            start, end = self._model_slices[i]
            params = p_full[start:end]

            if i < self._n_peaks:
                func = self._peak_specs[i][0]
                y += func(x, *params)
            elif self._bkg_spec:
                func = self._bkg_spec[0]
                y += func(x, *params)

        return y

    def evaluate_from_free(self, x, p_free):
        """Evaluate composite model from free parameter vector."""
        return self.evaluate(x, self.free_to_full(p_free))

    def residual(self, p_free, x, y):
        """Compute residual vector: y_model - y_data (for least_squares)."""
        return self.evaluate_from_free(x, p_free) - y

    def build_result(self, p_free, x, y, success):
        """Build a FitResult from optimized free parameters.

        Args:
            p_free: Optimized free parameter vector
            x: x-values
            y: y-data (baseline-subtracted)
            success: Whether optimization converged

        Returns:
            FitResult compatible with existing codebase
        """
        p_full = self.free_to_full(p_free)
        best_fit = self.evaluate(x, p_full)

        params_dict = {}
        for i, name in enumerate(self._param_names):
            params_dict[name] = p_full[i]

        return FitResult(
            success=success,
            params_dict=params_dict,
            best_fit=best_fit,
        )

    def update_initial_params(self, p_free):
        """Update internal initial values from optimized free parameters.

        Used for neighbor propagation: set the result from one spectrum
        as the initial guess for the next.
        """
        for i, idx in enumerate(self._free_mask):
            self._param_values[idx] = p_free[i]

    def clone_with_new_p0(self, p_free):
        """Create a shallow copy with updated initial parameters.

        Returns a new evaluator sharing the same model structure but with
        different initial values. Used for thread-safe parallel fitting.
        """
        new = PeakModelEvaluator.__new__(PeakModelEvaluator)
        new._peak_specs = self._peak_specs
        new._bkg_spec = self._bkg_spec
        new._param_names = self._param_names
        new._param_values = list(self._param_values)  # copy values
        new._param_mins = self._param_mins
        new._param_maxs = self._param_maxs
        new._param_vary = self._param_vary
        new._param_exprs = self._param_exprs
        new._model_slices = self._model_slices
        new._free_mask = self._free_mask
        new._fixed_mask = self._fixed_mask
        new._fixed_values = self._fixed_values.copy()
        new._n_peaks = self._n_peaks

        # Apply new free params
        for i, idx in enumerate(new._free_mask):
            new._param_values[idx] = p_free[i]
            new._fixed_values[idx] = p_free[i]

        return new

    def write_back_to_spectrum(self, spectrum, fit_result):
        """Write batch engine fit result back to an MSpectrum object.

        Sets result_fit and calls reassign_params() so existing Views
        and collect_fit_results() work unchanged.

        Args:
            spectrum: MSpectrum object
            fit_result: FitResult from build_result()
        """
        spectrum.result_fit = fit_result

        # Reassign fitted values to peak_models param_hints
        for i, peak_model in enumerate(spectrum.peak_models):
            if i >= self._n_peaks:
                break
            for key in peak_model.param_names:
                if key in fit_result.params:
                    name = key[4:]  # remove prefix 'mXX_'
                    peak_model.set_param_hint(name, value=fit_result.params[key].value)

        # Reassign bkg_model if present
        if spectrum.bkg_model is not None and self._bkg_spec is not None:
            for key in spectrum.bkg_model.param_names:
                if key in fit_result.params:
                    spectrum.bkg_model.set_param_hint(
                        key, value=fit_result.params[key].value
                    )
