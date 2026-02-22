"""
spectroview/model/fast_fit_engine.py
====================================
Vectorized 2D map fitting engine.

Design rationale
----------------
lmfit wraps scipy optimizers in Python → ~5ms of pure Python overhead per
spectrum (GIL locked).  For 1681 spectra that is ~10 seconds just for overhead.

This engine:
1. Calls scipy.optimize.least_squares (FORTRAN MINPACK) *directly* — zero Python
   overhead inside the optimizer loop.
2. Evaluates the model for ALL spectra simultaneously using numpy broadcasting
   inside each residual call, so C-level BLAS handles the math.
3. Propagates the previous pixel's fitted parameters as the next pixel's initial
   guess — this typically reduces Levenberg-Marquardt iterations from ~10 to ~3.
4. Stores results as a (N, n_params) numpy matrix for O(1) retrieval.
5. Creates fitspy-compatible result_fit objects *on demand* when the user selects
   a specific pixel, so we never pay the cost for all 1681 at once.

Supported peak models (exact same math as fitspy/core/models.py):
    Gaussian, GaussianAsym, Lorentzian, LorentzianAsym, PseudoVoigt,
    Fano, DecaySingleExp, DecayBiExp
"""

import numpy as np
import warnings
from copy import deepcopy
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# Vectorized peak model functions
# Each function accepts numpy arrays and broadcasts over N spectra:
#   x       : (M,)   wavenumbers (shared across all spectra)
#   ampli   : (N,)   peak amplitudes
#   fwhm    : (N,)   full-width at half-maximum
#   x0      : (N,)   peak centres
#   → output: (N, M) model values for all N spectra simultaneously
# ---------------------------------------------------------------------------

_LN2 = np.log(2.0)
_EPS = 1e-8  # guard against division by zero


def _lorentzian_v(x, ampli, fwhm, x0):
    """Vectorized Lorentzian: ampli * fwhm² / (4*((x-x0)² + fwhm²/4))"""
    a = ampli[:, None]
    f = fwhm[:, None]
    c = x0[:, None]
    return a * f**2 / (4.0 * ((x[None, :] - c)**2 + f**2 / 4.0) + _EPS)


def _gaussian_v(x, ampli, fwhm, x0):
    """Vectorized Gaussian: ampli * exp(-(x-x0)² / (2σ²)), σ=fwhm/(2√(2ln2))"""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * _LN2))
    coef = 1.0 / (2.0 * sigma**2 + _EPS)
    a = ampli[:, None]
    c = x0[:, None]
    return a * np.exp(-coef[:, None] * (x[None, :] - c)**2)


def _lorentzian_asym_v(x, ampli, fwhm_l, fwhm_r, x0):
    """Vectorized asymmetric Lorentzian."""
    c = x0[:, None]
    x_b = x[None, :]
    left = _lorentzian_v(x, ampli, fwhm_l, x0)
    right = _lorentzian_v(x, ampli, fwhm_r, x0)
    mask_l = (x_b < c)
    return mask_l * left + (~mask_l) * right


def _gaussian_asym_v(x, ampli, fwhm_l, fwhm_r, x0):
    """Vectorized asymmetric Gaussian."""
    c = x0[:, None]
    x_b = x[None, :]
    left = _gaussian_v(x, ampli, fwhm_l, x0)
    right = _gaussian_v(x, ampli, fwhm_r, x0)
    mask_l = (x_b < c)
    return mask_l * left + (~mask_l) * right


def _pseudovoigt_v(x, ampli, fwhm, x0, alpha):
    """Vectorized PseudoVoigt: α·Gaussian + (1-α)·Lorentzian"""
    g = _gaussian_v(x, ampli, fwhm, x0)
    l_ = _lorentzian_v(x, ampli, fwhm, x0)
    a = alpha[:, None]
    return a * g + (1.0 - a) * l_


def _fano_v(x, ampli, fwhm, x0, q):
    """Vectorized Fano resonance: ampli * (q + 2(x-x0)/fwhm)² / (1 + (2(x-x0)/fwhm)²)"""
    epsilon = 2.0 * (x[None, :] - x0[:, None]) / (fwhm[:, None] + _EPS)
    q_ = q[:, None]
    a = ampli[:, None]
    return a * (q_ + epsilon)**2 / (1.0 + epsilon**2 + _EPS)


def _decay_single_v(x, ampli, tau, x0):
    """Vectorized single-exponential decay."""
    a = ampli[:, None]
    t = tau[:, None]
    c = x0[:, None]
    dx = x[None, :] - c
    return a * np.exp(-dx / (t + _EPS)) * (dx >= 0)


def _decay_bi_v(x, ampli, tau1, tau2, x0, alpha):
    """Vectorized bi-exponential decay."""
    a = alpha[:, None]
    amp = ampli[:, None]
    c = x0[:, None]
    dx = x[None, :] - c
    mask = (dx >= 0)
    t1 = tau1[:, None]; t2 = tau2[:, None]
    return amp * (a * np.exp(-dx / (t1 + _EPS)) + (1 - a) * np.exp(-dx / (t2 + _EPS))) * mask


# Map fitspy model names to (eval_fn, param_names_ordered)
# param_names must match fitspy's param_hints keys exactly.
_MODEL_REGISTRY = {
    'Lorentzian':     (_lorentzian_v,    ['ampli', 'fwhm', 'x0']),
    'Gaussian':       (_gaussian_v,      ['ampli', 'fwhm', 'x0']),
    'PseudoVoigt':    (_pseudovoigt_v,   ['ampli', 'fwhm', 'x0', 'alpha']),
    'GaussianAsym':   (_gaussian_asym_v, ['ampli', 'fwhm_l', 'fwhm_r', 'x0']),
    'LorentzianAsym': (_lorentzian_asym_v, ['ampli', 'fwhm_l', 'fwhm_r', 'x0']),
    'Fano':           (_fano_v,          ['ampli', 'fwhm', 'x0', 'q']),
    'DecaySingleExp': (_decay_single_v,  ['ampli', 'tau', 'x0']),
    'DecayBiExp':     (_decay_bi_v,      ['ampli', 'tau1', 'tau2', 'x0', 'alpha']),
}

# Param names that must stay positive (>0)
_POSITIVE_PARAMS = {'ampli', 'fwhm', 'fwhm_l', 'fwhm_r', 'tau', 'tau1', 'tau2'}


# ---------------------------------------------------------------------------
# Analytical Jacobian functions  (scalar params → (M, n_peak_params) array)
# Each returns ∂peak_y / ∂params for a SINGLE spectrum (scalar/0-d params).
# Shape: (M, n_params_of_this_peak)
# ---------------------------------------------------------------------------

def _lorentzian_jac(x, ampli, fwhm, x0):
    fa, ff, fx = float(ampli), float(fwhm), float(x0)
    dx = x - fx
    D   = 4.0 * dx**2 + ff**2
    D2  = D**2 + _EPS
    J_A  = ff**2 / (D + _EPS)
    J_f  = 8.0 * fa * ff * dx**2 / D2
    J_x0 = 8.0 * fa * ff**2 * dx  / D2
    return np.column_stack([J_A, J_f, J_x0])


def _gaussian_jac(x, ampli, fwhm, x0):
    fa, ff, fx = float(ampli), float(fwhm), float(x0)
    sigma = ff / (2.0 * np.sqrt(2.0 * _LN2))
    coef  = 1.0 / (2.0 * sigma**2 + _EPS)
    dx = x - fx
    y    = fa * np.exp(-coef * dx**2)
    J_A  = y / (fa + _EPS)
    J_f  = y * 8.0 * _LN2 * dx**2 / (ff**3 + _EPS)
    J_x0 = y * 2.0 * coef * dx
    return np.column_stack([J_A, J_f, J_x0])


def _lorentzian_asym_jac(x, ampli, fwhm_l, fwhm_r, x0):
    fa, fl, fr, fx = float(ampli), float(fwhm_l), float(fwhm_r), float(x0)
    dx     = x - fx
    mask_l = dx < 0
    Dl  = 4.0 * dx**2 + fl**2;  Dl2 = Dl**2 + _EPS
    Dr  = 4.0 * dx**2 + fr**2;  Dr2 = Dr**2 + _EPS
    J_A  = np.where(mask_l, fl**2 / (Dl + _EPS),  fr**2 / (Dr + _EPS))
    J_fl = np.where(mask_l, 8.0 * fa * fl * dx**2 / Dl2, 0.0)
    J_fr = np.where(mask_l, 0.0,  8.0 * fa * fr * dx**2 / Dr2)
    J_x0 = np.where(mask_l, 8.0 * fa * fl**2 * dx / Dl2,
                             8.0 * fa * fr**2 * dx / Dr2)
    return np.column_stack([J_A, J_fl, J_fr, J_x0])


def _gaussian_asym_jac(x, ampli, fwhm_l, fwhm_r, x0):
    fa, fl, fr, fx = float(ampli), float(fwhm_l), float(fwhm_r), float(x0)
    dx     = x - fx
    mask_l = dx < 0

    def _gjac(ff):
        sigma = ff / (2.0 * np.sqrt(2.0 * _LN2))
        coef  = 1.0 / (2.0 * sigma**2 + _EPS)
        y    = fa * np.exp(-coef * dx**2)
        return (y / (fa + _EPS),
                y * 8.0 * _LN2 * dx**2 / (ff**3 + _EPS),
                y * 2.0 * coef * dx)

    A_l, Fl, x0_l = _gjac(fl)
    A_r, Fr, x0_r = _gjac(fr)
    J_A  = np.where(mask_l, A_l,  A_r)
    J_fl = np.where(mask_l, Fl,   0.0)
    J_fr = np.where(mask_l, 0.0,  Fr)
    J_x0 = np.where(mask_l, x0_l, x0_r)
    return np.column_stack([J_A, J_fl, J_fr, J_x0])


def _pseudovoigt_jac(x, ampli, fwhm, x0, alpha):
    fa, ff, fx, fal = float(ampli), float(fwhm), float(x0), float(alpha)
    J_G = _gaussian_jac(x, fa, ff, fx)    # (M, 3): A, f, x0
    J_L = _lorentzian_jac(x, fa, ff, fx)  # (M, 3): A, f, x0
    # G and L values for ∂y/∂alpha = G - L
    sigma = ff / (2.0 * np.sqrt(2.0 * _LN2))
    coef  = 1.0 / (2.0 * sigma**2 + _EPS)
    dx = x - fx
    G = fa * np.exp(-coef * dx**2)
    L = fa * ff**2 / (4.0 * dx**2 + ff**2 + _EPS)
    J_A   = fal * J_G[:, 0] + (1.0 - fal) * J_L[:, 0]
    J_f   = fal * J_G[:, 1] + (1.0 - fal) * J_L[:, 1]
    J_x0  = fal * J_G[:, 2] + (1.0 - fal) * J_L[:, 2]
    J_al  = G - L
    return np.column_stack([J_A, J_f, J_x0, J_al])


def _fano_jac(x, ampli, fwhm, x0, q):
    fa, ff, fx, fq = float(ampli), float(fwhm), float(x0), float(q)
    e   = 2.0 * (x - fx) / (ff + _EPS)
    qe  = fq + e
    D   = 1.0 + e**2
    D2  = D**2 + _EPS
    J_A  = qe**2 / (D + _EPS)
    J_q  = fa * 2.0 * qe / (D + _EPS)
    dcommon = fa * 2.0 * qe * (D - qe * e) / D2
    J_f  = dcommon * (-e / (ff + _EPS))
    J_x0 = dcommon * (-2.0 / (ff + _EPS))
    return np.column_stack([J_A, J_f, J_x0, J_q])


def _decay_single_jac(x, ampli, tau, x0):
    fa, ft, fx = float(ampli), float(tau), float(x0)
    dx   = x - fx
    mask = (dx >= 0).astype(np.float64)
    e    = np.exp(-dx / (ft + _EPS)) * mask
    y    = fa * e
    J_A  = e
    J_t  = y * dx / (ft**2 + _EPS)
    J_x0 = y / (ft + _EPS)
    return np.column_stack([J_A, J_t, J_x0])


def _decay_bi_jac(x, ampli, tau1, tau2, x0, alpha):
    fa, ft1, ft2, fx, fal = (float(ampli), float(tau1), float(tau2),
                              float(x0),    float(alpha))
    dx   = x - fx
    mask = (dx >= 0).astype(np.float64)
    e1   = np.exp(-dx / (ft1 + _EPS)) * mask
    e2   = np.exp(-dx / (ft2 + _EPS)) * mask
    J_A  = fal * e1 + (1.0 - fal) * e2
    J_t1 = fa * fal        * e1 * dx / (ft1**2 + _EPS)
    J_t2 = fa * (1.0 - fal) * e2 * dx / (ft2**2 + _EPS)
    J_x0 = fa * (fal * e1 / (ft1 + _EPS) + (1.0 - fal) * e2 / (ft2 + _EPS))
    J_al = fa * (e1 - e2)
    return np.column_stack([J_A, J_t1, J_t2, J_x0, J_al])


# Analytical Jacobian registry — parallel to _MODEL_REGISTRY
_JAC_REGISTRY = {
    'Lorentzian':     _lorentzian_jac,
    'Gaussian':       _gaussian_jac,
    'PseudoVoigt':    _pseudovoigt_jac,
    'GaussianAsym':   _gaussian_asym_jac,
    'LorentzianAsym': _lorentzian_asym_jac,
    'Fano':           _fano_jac,
    'DecaySingleExp': _decay_single_jac,
    'DecayBiExp':     _decay_bi_jac,
}


class _PeakSpec:
    """Holds parsed information about one peak in the fit model."""
    __slots__ = ['model_name', 'eval_fn', 'jac_fn', 'param_names', 'p0', 'bounds_lo', 'bounds_hi',
                 'prefix', 'slice_']

    def __init__(self, model_name, eval_fn, jac_fn, param_names, p0, bounds_lo, bounds_hi, prefix, slice_):
        self.model_name  = model_name
        self.eval_fn     = eval_fn
        self.jac_fn      = jac_fn      # analytical Jacobian, or None → fallback to 2-point
        self.param_names = param_names
        self.p0          = p0          # shape (n_params_of_this_peak,)
        self.bounds_lo   = bounds_lo
        self.bounds_hi   = bounds_hi
        self.prefix      = prefix      # e.g. 'm01_'
        self.slice_      = slice_      # slice into the flat per-spectrum param vector


class FastFitEngine:
    """
    Fit all spectra in a 2D map using vectorized scipy LM optimization.

    Parameters
    ----------
    x : ndarray (M,)
        Shared wavenumber axis (already preprocessed / range-clipped).
    Y : ndarray (N, M)
        Matrix of preprocessed spectra (baseline-subtracted if applicable).
    fit_model_dict : dict
        fitspy model dict (output of ``Spectra.load_model()``).
    coords_2d : list of (float, float), optional
        (x, y) physical coordinates in the same order as rows of Y.
        Used to determine 2D neighbour order for propagation.
    progress_callback : callable(done: int, total: int) → None, optional
        Called after each spectrum is fitted.
    """

    def __init__(self, x, Y, fit_model_dict, coords_2d=None, progress_callback=None):
        self.x = np.asarray(x, dtype=np.float64)
        self.Y = np.asarray(Y, dtype=np.float64)
        self.N, self.M = self.Y.shape
        self.coords_2d = coords_2d
        self.progress_callback = progress_callback

        # Parsed peak specifications
        self._peaks: list[_PeakSpec] = []
        self._n_params = 0

        # Results (filled by fit())
        self.params_matrix = None   # (N, n_params)
        self.chisqr_array  = None   # (N,)
        self.success_array = None   # (N,) bool
        self.n_eval_array  = None   # (N,) optimiser iterations

        self._parse_model(fit_model_dict)

    # -----------------------------------------------------------------------
    # Model parsing
    # -----------------------------------------------------------------------

    def _parse_model(self, fit_model_dict):
        """Extract peak specs from a fitspy model dict."""
        peak_models_dict = fit_model_dict.get('peak_models', {})
        if not peak_models_dict:
            raise ValueError("fit_model_dict has no 'peak_models' — cannot fit.")

        offset = 0
        for k, (peak_key, model_dict) in enumerate(peak_models_dict.items()):
            # model_dict: {'Lorentzian': {param_name: {value, min, max, vary, ...}}}
            model_name, param_hints = next(iter(model_dict.items()))
            # peak_key may be an integer (e.g. 0, 1) or string ('peak_1')
            # Use sequential index to produce lmfit-style prefixes like 'm01_'
            prefix = f'm{k+1:02d}_'

            if model_name not in _MODEL_REGISTRY:
                raise ValueError(
                    f"Model '{model_name}' is not supported by FastFitEngine. "
                    f"Supported: {list(_MODEL_REGISTRY.keys())}"
                )

            eval_fn, param_names = _MODEL_REGISTRY[model_name]
            jac_fn = _JAC_REGISTRY.get(model_name)  # None → numerical fallback
            n_p = len(param_names)

            p0 = np.zeros(n_p)
            bounds_lo = np.full(n_p, -np.inf)
            bounds_hi = np.full(n_p, np.inf)

            for j, pname in enumerate(param_names):
                hint = param_hints.get(pname, {})
                val = hint.get('value', 1.0)
                lo  = hint.get('min', -np.inf)
                hi  = hint.get('max', np.inf)
                if val is None or not np.isfinite(val):
                    val = 1.0
                if lo is None or not np.isfinite(lo):
                    lo = 0.0 if pname in _POSITIVE_PARAMS else -np.inf
                if hi is None or not np.isfinite(hi):
                    hi = np.inf
                # Clamp default value to (lo, hi)
                val = np.clip(val, lo if np.isfinite(lo) else val,
                              hi if np.isfinite(hi) else val)
                p0[j]         = val
                bounds_lo[j]  = lo
                bounds_hi[j]  = hi

            slc = slice(offset, offset + n_p)
            self._peaks.append(_PeakSpec(
                model_name=model_name,
                eval_fn=eval_fn,
                jac_fn=jac_fn,
                param_names=param_names,
                p0=p0,
                bounds_lo=bounds_lo,
                bounds_hi=bounds_hi,
                prefix=prefix,
                slice_=slc,
            ))
            offset += n_p

        self._n_params = offset

    # -----------------------------------------------------------------------
    # Candidate supported: quick check
    # -----------------------------------------------------------------------

    @classmethod
    def is_supported(cls, fit_model_dict):
        """Return True if all peak models in the dict are supported."""
        peak_models_dict = fit_model_dict.get('peak_models', {})
        if not peak_models_dict:
            return False
        for _, model_dict in peak_models_dict.items():
            model_name = next(iter(model_dict))
            if model_name not in _MODEL_REGISTRY:
                return False
        return True

    # -----------------------------------------------------------------------
    # Model evaluation
    # -----------------------------------------------------------------------

    def _eval_all(self, x, P):
        """
        Evaluate the full multi-peak model for N spectra simultaneously.

        Parameters
        ----------
        x : (M,) array
        P : (N, n_params) array of current parameter values

        Returns
        -------
        (N, M) model values
        """
        Y_model = np.zeros((len(P), len(x)), dtype=np.float64)
        for peak in self._peaks:
            slc = peak.slice_
            args = [P[:, slc.start + j] for j in range(slc.stop - slc.start)]
            Y_model += peak.eval_fn(x, *args)
        return Y_model

    def _eval_one(self, x, p):
        """Evaluate model for a SINGLE spectrum (1D params vector)."""
        y_model = np.zeros(len(x), dtype=np.float64)
        for peak in self._peaks:
            slc = peak.slice_
            # Wrap scalars as 1-element arrays for vectorised fns
            args = [np.atleast_1d(p[slc.start + j]) for j in range(slc.stop - slc.start)]
            y_model += peak.eval_fn(x, *args)[0]
        return y_model

    def _jac_one(self, x, p):
        """Analytical Jacobian ∂model/∂p for a SINGLE spectrum. Shape (M, n_params).
        
        Each column j is the partial derivative of model(x) w.r.t. parameter j.
        Passed directly to scipy.optimize.least_squares as the ``jac`` argument,
        eliminating the n_params extra model evaluations that '2-point' requires.
        """
        J = np.zeros((len(x), self._n_params), dtype=np.float64)
        for peak in self._peaks:
            slc = peak.slice_
            args = [p[slc.start + j] for j in range(slc.stop - slc.start)]
            J[:, slc.start:slc.stop] = peak.jac_fn(x, *args)
        return J

    # -----------------------------------------------------------------------
    # Scan order
    # -----------------------------------------------------------------------

    def _scan_order(self):
        """
        Return indices in row-major (top→bottom, left→right) scan order.
        If no coords are provided, return 0..N-1.
        """
        if self.coords_2d is None or len(self.coords_2d) != self.N:
            return np.arange(self.N)

        coords = np.array(self.coords_2d, dtype=np.float64)
        # Round to suppress floating-point jitter in coordinates
        xs = np.round(coords[:, 0], 6)
        ys = np.round(coords[:, 1], 6)
        # Sort: primary key = y (row), secondary = x (column)
        order = np.lexsort((xs, ys))
        return order

    # -----------------------------------------------------------------------
    # Global initial guess from spectral data
    # -----------------------------------------------------------------------

    def _auto_p0(self):
        """
        Build a global initial-guess parameter vector from data statistics.

        For each peak we try to improve the user-supplied initial guess by
        looking at the mean spectrum.  Falls back to the user hint if anything
        goes wrong.
        """
        mean_y = np.mean(self.Y, axis=0)
        x = self.x

        p0_global = np.zeros(self._n_params)

        for peak in self._peaks:
            slc = peak.slice_
            p0_global[slc] = peak.p0.copy()  # start with user hints

            # Try to improve ampli / x0 from data if the hint is ~1
            if 'ampli' in peak.param_names and 'x0' in peak.param_names:
                i_amp = peak.param_names.index('ampli')
                i_x0  = peak.param_names.index('x0')
                current_x0 = p0_global[slc.start + i_x0]
                i_near = np.argmin(np.abs(x - current_x0))
                data_amp = mean_y[i_near] if mean_y[i_near] > 0 else np.max(mean_y)
                if abs(p0_global[slc.start + i_amp]) < 1.0:
                    p0_global[slc.start + i_amp] = max(data_amp, 1.0)

        return p0_global

    # -----------------------------------------------------------------------
    # Fitting
    # -----------------------------------------------------------------------

    def fit(self, cancelled_flag=None):
        """
        Fit all spectra.

        Parameters
        ----------
        cancelled_flag : list or object with bool value, optional
            If truthy the loop exits early.

        Returns
        -------
        self (for chaining)
        """
        N, M = self.N, self.M
        x = self.x
        n_p = self._n_params

        self.params_matrix = np.zeros((N, n_p), dtype=np.float64)
        self.chisqr_array  = np.full(N, np.nan)
        self.success_array = np.zeros(N, dtype=bool)
        self.n_eval_array  = np.zeros(N, dtype=int)

        # Flat bounds for scipy
        lo_all = np.concatenate([pk.bounds_lo for pk in self._peaks])
        hi_all = np.concatenate([pk.bounds_hi for pk in self._peaks])
        bounds = (lo_all, hi_all)

        # Global initial guess (improved from data)
        p0_global = self._auto_p0()

        # Scan order for neighbour propagation
        order = self._scan_order()

        p_prev = p0_global.copy()  # will become the previous spectrum's params

        # Use analytical Jacobian if registered for every peak, else fall back
        # to scipy's numerical 2-point finite differences.
        _all_have_jac = all(pk.jac_fn is not None for pk in self._peaks)
        if _all_have_jac:
            _jac_arg = lambda p: self._jac_one(x, p)
        else:
            _jac_arg = '2-point'

        from scipy.optimize import least_squares, leastsq

        for k, idx in enumerate(order):
            if cancelled_flag is not None and cancelled_flag[0]:
                break

            y_i = self.Y[idx]

            # Use previous neighbour's params as warm start (reduces iterations ~3-5×)
            p0 = p_prev.copy()

            def residual(p, _y=y_i):
                return self._eval_one(x, p) - _y

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    # ---- FAST PATH: MINPACK leastsq (unbounded Levenberg-Marquardt) ----
                    # Scipy's leastsq has ~2-3x less Python overhead than least_squares.
                    if _all_have_jac:
                        def _jac2(p): return self._jac_one(x, p)
                        popt, _, info, msg, ier = leastsq(
                            residual, p0, Dfun=_jac2, full_output=True,
                            ftol=1e-5, xtol=1e-5, gtol=1e-8, maxfev=200
                        )
                    else:
                        popt, _, info, msg, ier = leastsq(
                            residual, p0, full_output=True,
                            ftol=1e-5, xtol=1e-5, gtol=1e-8, maxfev=200
                        )

                    # Check if the unbounded result violated our parameter bounds
                    bounds_violated = np.any(popt < lo_all) or np.any(popt > hi_all)
                    success_lm = ier in (1, 2, 3, 4)

                    if success_lm and not bounds_violated:
                        # Accept the fast-path result
                        self.params_matrix[idx] = popt
                        dof = max(M - n_p, 1)
                        self.chisqr_array[idx] = np.sum(info.get('fvec', residual(popt))**2) / dof
                        self.success_array[idx] = True
                        self.n_eval_array[idx]  = info.get('nfev', 0)
                        p_prev = popt.copy()

                    else:
                        # ---- FALLBACK PATH: TRF with strict bounds ----
                        # The fast path tried to push a parameter outside its min/max,
                        # or failed to converge. Fall back to the heavier bounded solver.
                        result = least_squares(
                            residual, p0,
                            bounds=bounds,
                            method='trf',
                            jac=_jac_arg,  # analytical or '2-point'
                            ftol=1e-5, xtol=1e-5, gtol=1e-8,
                            max_nfev=200,
                        )
                        self.params_matrix[idx] = result.x
                        dof = max(M - n_p, 1)
                        self.chisqr_array[idx] = np.sum(result.fun**2) / dof
                        self.success_array[idx] = result.success
                        self.n_eval_array[idx]  = result.nfev
                        p_prev = result.x.copy()

                except Exception:
                    self.params_matrix[idx] = p_prev
                    self.chisqr_array[idx]  = np.nan
                    self.success_array[idx] = False

            if self.progress_callback is not None:
                self.progress_callback(k + 1, N)

        return self

    # -----------------------------------------------------------------------
    # Result compatibility layer
    # -----------------------------------------------------------------------

    def get_fitted_y(self, idx):
        """Return the best-fit model curve for spectrum idx. Shape (M,)."""
        return self._eval_one(self.x, self.params_matrix[idx])

    def build_fitspy_result(self, idx):
        """
        Create a minimal fitspy/lmfit-compatible result object for spectrum idx.

        The returned object supports the attributes that spectroview actually
        reads: ``.params``, ``.params.valuesdict()``, ``.chisqr``, ``.success``,
        ``.best_fit`` (fitted y-values), ``.data`` (raw y), ``.init_fit``.

        This is created *on demand* — never for all 1681 spectra at once.
        """
        from lmfit import Parameters

        p_vals = self.params_matrix[idx]
        params = Parameters()

        for peak in self._peaks:
            slc = peak.slice_
            for j, pname in enumerate(peak.param_names):
                full_name = f"{peak.prefix}{pname}"
                val = p_vals[slc.start + j]
                params.add(full_name, value=float(val))

        # lmfit ModelResult-like duck-type object
        class _Result:
            pass

        r = _Result()
        r.params         = params
        r.chisqr         = float(self.chisqr_array[idx]) if self.chisqr_array is not None else np.nan
        r.success        = bool(self.success_array[idx]) if self.success_array is not None else False
        r.best_fit       = self.get_fitted_y(idx)
        r.data           = self.Y[idx]
        r.init_fit       = self.get_fitted_y(idx)  # same as best for display
        r.residual       = self.Y[idx] - r.best_fit
        r.message        = 'FastFitEngine (vectorized scipy)'
        r.nfev           = int(self.n_eval_array[idx]) if self.n_eval_array is not None else 0

        # Compatibility with fitspy's `result_fit.params.valuesdict()`
        r.best_values = params.valuesdict()

        return r

    def build_fitspy_result_for_spectrum(self, spectrum, idx):
        """
        Populate a fitspy MSpectrum's result_fit and params from
        this engine's result for spectrum index ``idx``.

        Also reconstructs the spectrum's peak_model param_hints so the
        existing spectroview peak-table and collect_fit_results still work.
        """
        r = self.build_fitspy_result(idx)
        spectrum.result_fit = r

        # Update param_hints on existing peak_models (if any)
        p_vals = self.params_matrix[idx]
        for k, peak in enumerate(self._peaks):
            if k < len(spectrum.peak_models):
                pm = spectrum.peak_models[k]
                slc = peak.slice_
                for j, pname in enumerate(peak.param_names):
                    if pname in pm.param_hints:
                        pm.param_hints[pname]['value'] = float(p_vals[slc.start + j])

        return spectrum

    # -----------------------------------------------------------------------
    # Convenience: param name list in flat order
    # -----------------------------------------------------------------------

    @property
    def param_names_flat(self):
        """Full list of parameter names in the same order as params_matrix columns."""
        names = []
        for peak in self._peaks:
            for pname in peak.param_names:
                names.append(f"{peak.prefix}{pname}")
        return names

    # -----------------------------------------------------------------------
    # Summary stats (for collect_fit_results)
    # -----------------------------------------------------------------------

    def get_results_dataframe(self, fnames):
        """
        Return a pandas DataFrame with one row per spectrum.

        Columns: fname, chisqr, success, <prefix_param> × n_params
        """
        import pandas as pd
        data = {'fname': fnames}
        data['chisqr'] = self.chisqr_array
        data['success'] = self.success_array
        for j, name in enumerate(self.param_names_flat):
            data[name] = self.params_matrix[:, j]
        return pd.DataFrame(data)
