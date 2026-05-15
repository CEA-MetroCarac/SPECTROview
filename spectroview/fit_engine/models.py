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
    if x.ndim == 1:
        dx = x[None, :] - c         # (N,M)
    else:
        dx = x - c
    w2 = w * w
    D = 1.0 + 4.0 * dx * dx / w2
    return a / D                # (N,M)


def batched_lorentzian_jac(x, params):
    a = params[:, 0:1]
    w = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    c = params[:, 2:3]
    if x.ndim == 1:
        dx = x[None, :] - c
    else:
        dx = x - c
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
    if x.ndim == 1:
        dx = x[None, :] - c
    else:
        dx = x - c
    t2 = dx * dx / (w * w)
    return a * np.exp(-_4LOG2 * t2)


def batched_gaussian_jac(x, params):
    a = params[:, 0:1]
    w = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    c = params[:, 2:3]
    if x.ndim == 1:
        dx = x[None, :] - c
    else:
        dx = x - c
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
# GaussianAsym:  Piecewise Gaussian with different left/right FWHM
#   params[:, 0] = ampli,  params[:, 1] = fwhm_l,
#   params[:, 2] = fwhm_r, params[:, 3] = x0
# ═══════════════════════════════════════════════════════════════════════════

def batched_gaussian_asym(x, params):
    a  = params[:, 0:1]
    wl = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    wr = np.maximum(np.abs(params[:, 2:3]), 1e-15)
    c  = params[:, 3:4]
    if x.ndim == 1:
        dx = x[None, :] - c
    else:
        dx = x - c
    left  = (dx < 0).astype(np.float64)
    right = 1.0 - left
    w = left * wl + right * wr          # effective FWHM per point
    t2 = dx * dx / (w * w)
    return a * np.exp(-_4LOG2 * t2)


def batched_gaussian_asym_jac(x, params):
    a  = params[:, 0:1]
    wl = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    wr = np.maximum(np.abs(params[:, 2:3]), 1e-15)
    c  = params[:, 3:4]
    if x.ndim == 1:
        dx = x[None, :] - c
    else:
        dx = x - c
    left  = (dx < 0).astype(np.float64)
    right = 1.0 - left
    w = left * wl + right * wr
    w2 = w * w
    w3 = w2 * w
    dx2 = dx * dx
    t2 = dx2 / w2
    G = a * np.exp(-_4LOG2 * t2)

    N, M = dx.shape
    J = np.empty((N, M, 4))
    J[:, :, 0] = G / (a + 1e-30)                     # dG/d_ampli
    J[:, :, 1] = G * (2.0 * _4LOG2 * dx2 / w3) * left   # dG/d_fwhm_l
    J[:, :, 2] = G * (2.0 * _4LOG2 * dx2 / w3) * right  # dG/d_fwhm_r
    J[:, :, 3] = G * (2.0 * _4LOG2 * dx / w2)        # dG/d_x0
    return J


# ═══════════════════════════════════════════════════════════════════════════
# LorentzianAsym:  Piecewise Lorentzian with different left/right FWHM
#   params[:, 0] = ampli,  params[:, 1] = fwhm_l,
#   params[:, 2] = fwhm_r, params[:, 3] = x0
# ═══════════════════════════════════════════════════════════════════════════

def batched_lorentzian_asym(x, params):
    a  = params[:, 0:1]
    wl = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    wr = np.maximum(np.abs(params[:, 2:3]), 1e-15)
    c  = params[:, 3:4]
    if x.ndim == 1:
        dx = x[None, :] - c
    else:
        dx = x - c
    left  = (dx < 0).astype(np.float64)
    right = 1.0 - left
    w = left * wl + right * wr
    w2 = w * w
    D = 1.0 + 4.0 * dx * dx / w2
    return a / D


def batched_lorentzian_asym_jac(x, params):
    a  = params[:, 0:1]
    wl = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    wr = np.maximum(np.abs(params[:, 2:3]), 1e-15)
    c  = params[:, 3:4]
    if x.ndim == 1:
        dx = x[None, :] - c
    else:
        dx = x - c
    left  = (dx < 0).astype(np.float64)
    right = 1.0 - left
    w = left * wl + right * wr
    w2 = w * w
    w3 = w2 * w
    dx2 = dx * dx
    D = 1.0 + 4.0 * dx2 / w2
    D2 = D * D
    invD = 1.0 / D

    N, M = dx.shape
    J = np.empty((N, M, 4))
    J[:, :, 0] = invD                                          # dL/d_ampli
    J[:, :, 1] = (8.0 * a * dx2 / (w3 * D2)) * left           # dL/d_fwhm_l
    J[:, :, 2] = (8.0 * a * dx2 / (w3 * D2)) * right          # dL/d_fwhm_r
    J[:, :, 3] = 8.0 * a * dx / (w2 * D2)                     # dL/d_x0
    return J


# ═══════════════════════════════════════════════════════════════════════════
# Fano:  f = ampli · (q + ε)² / (1 + ε²),  ε = 2·(x - x0) / fwhm
#   params[:, 0] = ampli,  params[:, 1] = fwhm,
#   params[:, 2] = x0,     params[:, 3] = q
# ═══════════════════════════════════════════════════════════════════════════

def batched_fano(x, params):
    a = params[:, 0:1]
    w = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    c = params[:, 2:3]
    q = params[:, 3:4]
    if x.ndim == 1:
        dx = x[None, :] - c
    else:
        dx = x - c
    eps = 2.0 * dx / w                   # ε = (x - x0) / (Γ/2)
    num = (q + eps) ** 2
    den = 1.0 + eps * eps
    return a * num / den


def batched_fano_jac(x, params):
    a = params[:, 0:1]
    w = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    c = params[:, 2:3]
    q = params[:, 3:4]
    if x.ndim == 1:
        dx = x[None, :] - c
    else:
        dx = x - c
    eps = 2.0 * dx / w
    qe = q + eps
    num = qe * qe
    den = 1.0 + eps * eps
    den2 = den * den

    # df/dε = a · 2(q+ε)(1 - εq) / (1+ε²)²
    df_deps = a * 2.0 * qe * (1.0 - eps * q) / den2

    N, M = dx.shape
    J = np.empty((N, M, 4))
    J[:, :, 0] = num / den                          # df/d_ampli
    J[:, :, 1] = df_deps * (-eps / w)               # df/d_fwhm  (dε/dw = -ε/w)
    J[:, :, 2] = df_deps * (-2.0 / w)               # df/d_x0    (dε/dx0 = -2/w)
    J[:, :, 3] = a * 2.0 * qe / den                 # df/d_q
    return J


# ═══════════════════════════════════════════════════════════════════════════
# DecaySingleExp:  f = A · exp(-x / τ) + B
#   params[:, 0] = A,  params[:, 1] = tau,  params[:, 2] = B
# ═══════════════════════════════════════════════════════════════════════════

def batched_decay_single_exp(x, params):
    A   = params[:, 0:1]
    tau = np.maximum(np.abs(params[:, 1:2]), 1e-30)
    B   = params[:, 2:3]
    if x.ndim == 1:
        xv = x[None, :]
    else:
        xv = x
    E = np.exp(-xv / tau)
    return A * E + B


def batched_decay_single_exp_jac(x, params):
    A   = params[:, 0:1]
    tau = np.maximum(np.abs(params[:, 1:2]), 1e-30)
    if x.ndim == 1:
        xv = x[None, :]
    else:
        xv = x
    E = np.exp(-xv / tau)
    tau2 = tau * tau

    N, M = E.shape
    J = np.empty((N, M, 3))
    J[:, :, 0] = E                          # df/dA
    J[:, :, 1] = A * xv * E / tau2          # df/dτ = A·x·exp(-x/τ)/τ²
    J[:, :, 2] = 1.0                        # df/dB
    return J


# ═══════════════════════════════════════════════════════════════════════════
# DecayBiExp:  f = A1·exp(-x/τ1) + A2·exp(-x/τ2) + B
#   params[:, 0] = A1,  params[:, 1] = tau1,  params[:, 2] = A2,
#   params[:, 3] = tau2, params[:, 4] = B
# ═══════════════════════════════════════════════════════════════════════════

def batched_decay_bi_exp(x, params):
    A1   = params[:, 0:1]
    tau1 = np.maximum(np.abs(params[:, 1:2]), 1e-30)
    A2   = params[:, 2:3]
    tau2 = np.maximum(np.abs(params[:, 3:4]), 1e-30)
    B    = params[:, 4:5]
    if x.ndim == 1:
        xv = x[None, :]
    else:
        xv = x
    E1 = np.exp(-xv / tau1)
    E2 = np.exp(-xv / tau2)
    return A1 * E1 + A2 * E2 + B


def batched_decay_bi_exp_jac(x, params):
    A1   = params[:, 0:1]
    tau1 = np.maximum(np.abs(params[:, 1:2]), 1e-30)
    A2   = params[:, 2:3]
    tau2 = np.maximum(np.abs(params[:, 3:4]), 1e-30)
    if x.ndim == 1:
        xv = x[None, :]
    else:
        xv = x
    E1 = np.exp(-xv / tau1)
    E2 = np.exp(-xv / tau2)

    N, M = E1.shape
    J = np.empty((N, M, 5))
    J[:, :, 0] = E1                              # df/dA1
    J[:, :, 1] = A1 * xv * E1 / (tau1 * tau1)   # df/dτ1
    J[:, :, 2] = E2                              # df/dA2
    J[:, :, 3] = A2 * xv * E2 / (tau2 * tau2)   # df/dτ2
    J[:, :, 4] = 1.0                             # df/dB
    return J


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════

BATCHED_MODELS = {
    "Gaussian":       (batched_gaussian,          batched_gaussian_jac,          ["ampli", "fwhm", "x0"]),
    "Lorentzian":     (batched_lorentzian,        batched_lorentzian_jac,        ["ampli", "fwhm", "x0"]),
    "PseudoVoigt":    (batched_pseudovoigt,       batched_pseudovoigt_jac,       ["ampli", "fwhm", "x0", "alpha"]),
    "GaussianAsym":   (batched_gaussian_asym,     batched_gaussian_asym_jac,     ["ampli", "fwhm_l", "fwhm_r", "x0"]),
    "LorentzianAsym": (batched_lorentzian_asym,   batched_lorentzian_asym_jac,   ["ampli", "fwhm_l", "fwhm_r", "x0"]),
    "Fano":           (batched_fano,              batched_fano_jac,              ["ampli", "fwhm", "x0", "q"]),
    "DecaySingleExp": (batched_decay_single_exp,  batched_decay_single_exp_jac,  ["A", "tau", "B"]),
    "DecayBiExp":     (batched_decay_bi_exp,      batched_decay_bi_exp_jac,      ["A1", "tau1", "A2", "tau2", "B"]),
}


def numerical_jacobian(model_func, x, params, eps=1e-7):
    """Finite-difference Jacobian fallback for models without analytical J.

    Uses relative perturbation: h = max(abs(param) * eps, eps) for each
    parameter, ensuring accurate gradients regardless of parameter scale.
    """
    N, K = params.shape
    M = x.shape[-1] if hasattr(x, 'shape') else len(x)
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
