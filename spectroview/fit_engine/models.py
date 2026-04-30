"""Batched (tensor) peak model functions with analytical Jacobians.

All functions operate on tensors:
    x:      (M,)        — shared wavelength axis
    params: (N, n_p)    — parameter matrix for N spectra

Returns:
    Y: (N, M)           — model values
    J: (N, M, n_p)      — Jacobian matrix  (for *_jac functions)

The functional forms match lmfit exactly so results are interchangeable.
"""

import numpy as np

_LOG2 = np.log(2.0)
_4LOG2 = 4.0 * _LOG2

# ═══════════════════════════════════════════════════════════════════════════
# Lorentzian:  L = ampli / (1 + 4·((x-x0)/fwhm)²)
#   params[:, 0] = ampli,  params[:, 1] = fwhm,  params[:, 2] = x0
# ═══════════════════════════════════════════════════════════════════════════

def batched_lorentzian(x, params):
    a = params[:, 0:1]          # (N,1)
    w = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    c = params[:, 2:3]
    dx = x[None, :] - c         # (N,M)
    w2 = w * w
    D = 1.0 + 4.0 * dx * dx / w2
    return a / D                # (N,M)


def batched_lorentzian_jac(x, params):
    a = params[:, 0:1]
    w = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    c = params[:, 2:3]
    dx = x[None, :] - c
    w2 = w * w
    dx2 = dx * dx
    D = 1.0 + 4.0 * dx2 / w2
    D2 = D * D
    invD = 1.0 / D

    N, M = dx.shape
    J = np.empty((N, M, 3))
    J[:, :, 0] = invD                               # dL/da
    J[:, :, 1] = 8.0 * a * dx2 / (w2 * w * D2)      # dL/dw
    J[:, :, 2] = 8.0 * a * dx / (w2 * D2)           # dL/dx0
    return J


# ═══════════════════════════════════════════════════════════════════════════
# Gaussian:  G = ampli · exp(-4·ln2·((x-x0)/fwhm)²)
#   params[:, 0] = ampli,  params[:, 1] = fwhm,  params[:, 2] = x0
# ═══════════════════════════════════════════════════════════════════════════

def batched_gaussian(x, params):
    a = params[:, 0:1]
    w = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    c = params[:, 2:3]
    dx = x[None, :] - c
    t2 = dx * dx / (w * w)
    return a * np.exp(-_4LOG2 * t2)


def batched_gaussian_jac(x, params):
    a = params[:, 0:1]
    w = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    c = params[:, 2:3]
    dx = x[None, :] - c
    w2 = w * w
    w3 = w2 * w
    t2 = dx * dx / w2
    G = a * np.exp(-_4LOG2 * t2)       # (N,M)

    N, M = dx.shape
    J = np.empty((N, M, 3))
    J[:, :, 0] = G / (a + 1e-30)                    # dG/da  = G/a
    J[:, :, 1] = G * (2.0 * _4LOG2 * dx * dx / w3)  # dG/dw
    J[:, :, 2] = G * (2.0 * _4LOG2 * dx / w2)       # dG/dx0
    return J


# ═══════════════════════════════════════════════════════════════════════════
# PseudoVoigt:  PV = alpha·G + (1-alpha)·L   (fitspy convention)
#   params[:, 0] = ampli,  params[:, 1] = fwhm,  params[:, 2] = x0,
#   params[:, 3] = alpha
# ═══════════════════════════════════════════════════════════════════════════

def batched_pseudovoigt(x, params):
    p3 = params[:, :3]                # (N,3) — [ampli, fwhm, x0]
    alpha = params[:, 3:4]            # (N,1)
    G = batched_gaussian(x, p3)
    L = batched_lorentzian(x, p3)
    return alpha * G + (1.0 - alpha) * L


def batched_pseudovoigt_jac(x, params):
    p3 = params[:, :3]
    alpha = params[:, 3:4]
    G = batched_gaussian(x, p3)        # (N,M)
    L = batched_lorentzian(x, p3)
    JG = batched_gaussian_jac(x, p3)   # (N,M,3)
    JL = batched_lorentzian_jac(x, p3)

    N, M = G.shape
    J = np.empty((N, M, 4))
    # dPV/d(ampli,fwhm,x0)
    J[:, :, :3] = alpha[:, :, None] * JG + (1.0 - alpha[:, :, None]) * JL
    # dPV/dalpha = G - L
    J[:, :, 3] = G - L
    return J


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════

BATCHED_MODELS = {
    "Gaussian":    (batched_gaussian,    batched_gaussian_jac,    ["ampli", "fwhm", "x0"]),
    "Lorentzian":  (batched_lorentzian,  batched_lorentzian_jac,  ["ampli", "fwhm", "x0"]),
    "PseudoVoigt": (batched_pseudovoigt, batched_pseudovoigt_jac, ["ampli", "fwhm", "x0", "alpha"]),
}


def numerical_jacobian(model_func, x, params, eps=1e-7):
    """Finite-difference Jacobian fallback for models without analytical J.

    Uses relative perturbation: h = max(abs(param) * eps, eps) for each
    parameter, ensuring accurate gradients regardless of parameter scale.
    """
    N, K = params.shape
    M = len(x)
    J = np.empty((N, M, K))
    for k in range(K):
        # Relative perturbation: scale eps by the parameter magnitude
        h = np.maximum(np.abs(params[:, k]) * eps, eps)  # (N,)
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[:, k] += h
        p_minus[:, k] -= h
        J[:, :, k] = (model_func(x, p_plus) - model_func(x, p_minus)) / (2.0 * h[:, None])
    return J
