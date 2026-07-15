"""
Vectorized peak model evaluation for the batch fitting engine.

Provides pure-NumPy implementations of all supported peak shapes that match
the exact functional forms used by the lmfit engine. Handles the mapping between
structured param_hints dicts and flat parameter vectors for scipy.optimize.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Pure-NumPy peak model functions
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

