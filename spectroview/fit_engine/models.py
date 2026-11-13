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

def batched_lorentzian(x, params, out=None):
    a = params[:, 0:1]          # (N,1)
    w = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    c = params[:, 2:3]
    D = x[None, :] - c if x.ndim == 1 else x - c  # (N,M), squared in-place below
    inv_w2 = 1.0 / (w * w)      # (N,1) — one reciprocal instead of dividing (N,M) by w2 below
    D *= D
    D *= (4.0 * inv_w2)
    D += 1.0
    if out is not None:
        np.divide(a, D, out=out)
        return out
    return a / D                # (N,M)


def batched_lorentzian_jac(x, params, out=None):
    a = params[:, 0:1]
    w = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    c = params[:, 2:3]
    dx = x[None, :] - c if x.ndim == 1 else x - c
    inv_w2 = 1.0 / (w * w)
    dx2 = dx * dx
    invD = dx2 * (4.0 * inv_w2)
    invD += 1.0
    np.reciprocal(invD, out=invD)
    invD2 = invD * invD

    N, M = dx.shape
    J = np.empty((N, M, 3)) if out is None else out
    J[:, :, 0] = invD                               # dL/da
    coef_w = (8.0 * a) * inv_w2 * np.sqrt(inv_w2)   # (N,1) == 8a/w^3, avoids an (N,M) divide
    dx2 *= invD2
    dx2 *= coef_w
    J[:, :, 1] = dx2                                # dL/dw
    coef_x0 = (8.0 * a) * inv_w2
    dx *= invD2
    dx *= coef_x0
    J[:, :, 2] = dx                                 # dL/dx0
    return J


# ═══════════════════════════════════════════════════════════════════════════
# Gaussian:  G = ampli · exp(-4·ln2·((x-x0)/fwhm)²)
#   params[:, 0] = ampli,  params[:, 1] = fwhm,  params[:, 2] = x0
# ═══════════════════════════════════════════════════════════════════════════

def batched_gaussian(x, params, out=None):
    a = params[:, 0:1]
    w = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    c = params[:, 2:3]
    t2 = x[None, :] - c if x.ndim == 1 else x - c
    inv_w2 = 1.0 / (w * w)
    t2 *= t2
    t2 *= (-_4LOG2 * inv_w2)
    np.exp(t2, out=t2)
    if out is not None:
        np.multiply(t2, a, out=out)
        return out
    t2 *= a
    return t2


def batched_gaussian_jac(x, params, out=None):
    a = params[:, 0:1]
    w = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    c = params[:, 2:3]
    dx = x[None, :] - c if x.ndim == 1 else x - c
    inv_w = 1.0 / w
    inv_w2 = inv_w * inv_w
    dx2 = dx * dx
    g_unit = dx2 * (-_4LOG2 * inv_w2)
    np.exp(g_unit, out=g_unit)

    N, M = dx.shape
    J = np.empty((N, M, 3)) if out is None else out
    J[:, :, 0] = g_unit                               # dG/da  = exp(-4·ln2·t²)
    G = g_unit * a
    coef_w = (2.0 * _4LOG2) * inv_w2 * inv_w          # (N,1) == 2·4log2/w^3
    dx2 *= G
    dx2 *= coef_w
    J[:, :, 1] = dx2                                  # dG/dw
    coef_x0 = (2.0 * _4LOG2) * inv_w2                 # (N,1) == 2·4log2/w^2
    dx *= G
    dx *= coef_x0
    J[:, :, 2] = dx                                   # dG/dx0
    return J


# ═══════════════════════════════════════════════════════════════════════════
# PseudoVoigt:  PV = alpha·G + (1-alpha)·L   (legacy convention)
#   params[:, 0] = ampli,  params[:, 1] = fwhm,  params[:, 2] = x0,
#   params[:, 3] = alpha
# ═══════════════════════════════════════════════════════════════════════════

def batched_pseudovoigt(x, params, out=None):
    a = params[:, 0:1]
    w = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    c = params[:, 2:3]
    alpha = params[:, 3:4]
    t2 = x[None, :] - c if x.ndim == 1 else x - c
    t2 *= t2
    t2 *= (1.0 / (w * w))

    g_unit = t2 * (-_4LOG2)
    np.exp(g_unit, out=g_unit)

    # Reuse t2 buffer in-place for l_unit = 1 / (4·t2 + 1).
    t2 *= 4.0
    t2 += 1.0
    np.reciprocal(t2, out=t2)

    # alpha*G_unit + (1-alpha)*L_unit == L_unit + alpha*(G_unit - L_unit)
    g_unit -= t2
    g_unit *= alpha
    g_unit += t2
    if out is not None:
        np.multiply(a, g_unit, out=out)
        return out
    g_unit *= a
    return g_unit


def batched_pseudovoigt_jac(x, params, out=None):
    a = params[:, 0:1]
    w = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    c = params[:, 2:3]
    alpha = params[:, 3:4]
    dx = x[None, :] - c if x.ndim == 1 else x - c
    inv_w = 1.0 / w
    inv_w2 = inv_w * inv_w
    dx2 = dx * dx
    t2 = dx2 * inv_w2

    # Compute g_unit and invD once cleanly from shared t2 instead of calling
    # Gaussian/Lorentzian eval+jac four times.
    g_unit = t2 * (-_4LOG2)
    np.exp(g_unit, out=g_unit)

    invD = t2
    invD *= 4.0
    invD += 1.0
    np.reciprocal(invD, out=invD)
    invD2 = invD * invD

    N, M = dx.shape
    J = np.empty((N, M, 4)) if out is None else out

    # dPV/dampli = invD + alpha*(g_unit - invD); dPV/dalpha = a*(g_unit - invD)
    diff_unit = g_unit - invD
    d_ampli = diff_unit * alpha
    d_ampli += invD
    J[:, :, 0] = d_ampli
    J[:, :, 3] = a * diff_unit

    # dPV/dx0 = dx * inv_w2 * a * (alpha*(2*4*ln2)*g_unit + (1-alpha)*8*invD^2)
    # dPV/dw  = (dx / w) * dPV/dx0  (both G and L depend strictly on u = dx/w)
    g_unit *= ((2.0 * _4LOG2) * alpha)
    invD2 *= (8.0 * (1.0 - alpha))
    g_unit += invD2
    g_unit *= (a * inv_w2)
    g_unit *= dx
    J[:, :, 2] = g_unit

    g_unit *= dx
    g_unit *= inv_w
    J[:, :, 1] = g_unit
    return J


# ═══════════════════════════════════════════════════════════════════════════
# GaussianAsym:  Piecewise Gaussian with different left/right FWHM
#   params[:, 0] = ampli,  params[:, 1] = fwhm_l,
#   params[:, 2] = fwhm_r, params[:, 3] = x0
# ═══════════════════════════════════════════════════════════════════════════

def batched_gaussian_asym(x, params, out=None):
    a  = params[:, 0:1]
    wl = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    wr = np.maximum(np.abs(params[:, 2:3]), 1e-15)
    c  = params[:, 3:4]
    dx = x[None, :] - c if x.ndim == 1 else x - c
    w = np.where(dx < 0, wl, wr)        # effective FWHM per point (exact select, no float-cast blend)
    inv_w2 = 1.0 / (w * w)
    dx *= dx
    dx *= (-_4LOG2 * inv_w2)
    np.exp(dx, out=dx)
    if out is not None:
        np.multiply(a, dx, out=out)
        return out
    dx *= a
    return dx


def batched_gaussian_asym_jac(x, params, out=None):
    a  = params[:, 0:1]
    wl = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    wr = np.maximum(np.abs(params[:, 2:3]), 1e-15)
    c  = params[:, 3:4]
    dx = x[None, :] - c if x.ndim == 1 else x - c
    neg = dx < 0
    w = np.where(neg, wl, wr)
    inv_w2 = 1.0 / (w * w)
    dx2 = dx * dx
    g_unit = dx2 * (-_4LOG2 * inv_w2)
    np.exp(g_unit, out=g_unit)

    N, M = dx.shape
    J = np.empty((N, M, 4)) if out is None else out
    J[:, :, 0] = g_unit                                 # dG/d_ampli
    G = g_unit * a
    coef = G * dx2 * (2.0 * _4LOG2) * inv_w2 / w        # shared magnitude, |w|-side selected below
    J[:, :, 1] = np.where(neg, coef, 0.0)               # dG/d_fwhm_l
    J[:, :, 2] = np.where(neg, 0.0, coef)                # dG/d_fwhm_r
    J[:, :, 3] = G * dx * (2.0 * _4LOG2) * inv_w2        # dG/d_x0
    return J


# ═══════════════════════════════════════════════════════════════════════════
# LorentzianAsym:  Piecewise Lorentzian with different left/right FWHM
#   params[:, 0] = ampli,  params[:, 1] = fwhm_l,
#   params[:, 2] = fwhm_r, params[:, 3] = x0
# ═══════════════════════════════════════════════════════════════════════════

def batched_lorentzian_asym(x, params, out=None):
    a  = params[:, 0:1]
    wl = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    wr = np.maximum(np.abs(params[:, 2:3]), 1e-15)
    c  = params[:, 3:4]
    dx = x[None, :] - c if x.ndim == 1 else x - c
    w = np.where(dx < 0, wl, wr)
    inv_w2 = 1.0 / (w * w)
    dx *= dx
    dx *= (4.0 * inv_w2)
    dx += 1.0
    if out is not None:
        np.divide(a, dx, out=out)
        return out
    return a / dx


def batched_lorentzian_asym_jac(x, params, out=None):
    a  = params[:, 0:1]
    wl = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    wr = np.maximum(np.abs(params[:, 2:3]), 1e-15)
    c  = params[:, 3:4]
    dx = x[None, :] - c if x.ndim == 1 else x - c
    neg = dx < 0
    w = np.where(neg, wl, wr)
    inv_w2 = 1.0 / (w * w)
    dx2 = dx * dx
    invD = dx2 * (4.0 * inv_w2)
    invD += 1.0
    np.reciprocal(invD, out=invD)
    invD2 = invD * invD

    N, M = dx.shape
    J = np.empty((N, M, 4)) if out is None else out
    J[:, :, 0] = invD                                    # dL/d_ampli
    coef = (8.0 * a) * dx2 * invD2 * inv_w2 / w          # shared magnitude, |w|-side selected below
    J[:, :, 1] = np.where(neg, coef, 0.0)                # dL/d_fwhm_l
    J[:, :, 2] = np.where(neg, 0.0, coef)                 # dL/d_fwhm_r
    J[:, :, 3] = (8.0 * a) * dx * invD2 * inv_w2          # dL/d_x0
    return J


# ═══════════════════════════════════════════════════════════════════════════
# Fano:  f = ampli · (q + ε)² / (1 + ε²),  ε = 2·(x - x0) / fwhm
#   params[:, 0] = ampli,  params[:, 1] = fwhm,
#   params[:, 2] = x0,     params[:, 3] = q
# ═══════════════════════════════════════════════════════════════════════════

def batched_fano(x, params, out=None):
    a = params[:, 0:1]
    w = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    c = params[:, 2:3]
    q = params[:, 3:4]
    dx = x[None, :] - c if x.ndim == 1 else x - c
    inv_w = 1.0 / w
    eps = dx * (2.0 * inv_w)             # ε = (x - x0) / (Γ/2)
    qe = q + eps
    num = qe * qe
    den = eps * eps
    den += 1.0
    num *= a
    if out is not None:
        np.divide(num, den, out=out)
        return out
    return num / den


def batched_fano_jac(x, params, out=None):
    a = params[:, 0:1]
    w = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    c = params[:, 2:3]
    q = params[:, 3:4]
    dx = x[None, :] - c if x.ndim == 1 else x - c
    inv_w = 1.0 / w
    eps = dx * (2.0 * inv_w)
    qe = q + eps
    num = qe * qe
    den = eps * eps
    den += 1.0
    inv_den = np.reciprocal(den)
    inv_den2 = inv_den * inv_den

    # df/dε = a · 2(q+ε)(1 - εq) / (1+ε²)²
    df_deps = qe * (1.0 - eps * q)
    df_deps *= (2.0 * a)
    df_deps *= inv_den2

    N, M = dx.shape
    J = np.empty((N, M, 4)) if out is None else out
    J[:, :, 0] = num * inv_den                      # df/d_ampli
    J[:, :, 1] = df_deps * (-eps * inv_w)           # df/d_fwhm  (dε/dw = -ε/w)
    J[:, :, 2] = df_deps * (-2.0 * inv_w)           # df/d_x0    (dε/dx0 = -2/w)
    J[:, :, 3] = (2.0 * a) * qe * inv_den           # df/d_q
    return J


# ═══════════════════════════════════════════════════════════════════════════
# DecaySingleExp:  f = A · exp(-x / τ) + B
#   params[:, 0] = A,  params[:, 1] = tau,  params[:, 2] = B
# ═══════════════════════════════════════════════════════════════════════════

def batched_decay_single_exp(x, params, out=None):
    A   = params[:, 0:1]
    tau = np.maximum(np.abs(params[:, 1:2]), 1e-30)
    B   = params[:, 2:3]
    xv = x[None, :] if x.ndim == 1 else x
    inv_tau = 1.0 / tau
    E = xv * (-inv_tau)
    np.exp(E, out=E)
    if out is not None:
        np.multiply(E, A, out=out)
        out += B
        return out
    E *= A
    E += B
    return E


def batched_decay_single_exp_jac(x, params, out=None):
    A   = params[:, 0:1]
    tau = np.maximum(np.abs(params[:, 1:2]), 1e-30)
    xv = x[None, :] if x.ndim == 1 else x
    inv_tau = 1.0 / tau
    E = xv * (-inv_tau)
    np.exp(E, out=E)

    N, M = E.shape
    J = np.empty((N, M, 3)) if out is None else out
    J[:, :, 0] = E                                    # df/dA
    J[:, :, 1] = (A * inv_tau) * xv * E * inv_tau     # df/dτ = A·x·exp(-x/τ)/τ²
    J[:, :, 2] = 1.0                                  # df/dB
    return J


# ═══════════════════════════════════════════════════════════════════════════
# DecayBiExp:  f = A1·exp(-x/τ1) + A2·exp(-x/τ2) + B
#   params[:, 0] = A1,  params[:, 1] = tau1,  params[:, 2] = A2,
#   params[:, 3] = tau2, params[:, 4] = B
# ═══════════════════════════════════════════════════════════════════════════

def batched_decay_bi_exp(x, params, out=None):
    A1   = params[:, 0:1]
    tau1 = np.maximum(np.abs(params[:, 1:2]), 1e-30)
    A2   = params[:, 2:3]
    tau2 = np.maximum(np.abs(params[:, 3:4]), 1e-30)
    B    = params[:, 4:5]
    xv = x[None, :] if x.ndim == 1 else x
    Y = xv * (-1.0 / tau1)
    np.exp(Y, out=Y)
    Y *= A1
    E2 = xv * (-1.0 / tau2)
    np.exp(E2, out=E2)
    E2 *= A2
    Y += E2
    if out is not None:
        np.add(Y, B, out=out)
        return out
    Y += B
    return Y


def batched_decay_bi_exp_jac(x, params, out=None):
    A1   = params[:, 0:1]
    tau1 = np.maximum(np.abs(params[:, 1:2]), 1e-30)
    A2   = params[:, 2:3]
    tau2 = np.maximum(np.abs(params[:, 3:4]), 1e-30)
    xv = x[None, :] if x.ndim == 1 else x
    inv_tau1 = 1.0 / tau1
    E1 = xv * (-inv_tau1)
    np.exp(E1, out=E1)
    inv_tau2 = 1.0 / tau2
    E2 = xv * (-inv_tau2)
    np.exp(E2, out=E2)

    N, M = E1.shape
    J = np.empty((N, M, 5)) if out is None else out
    J[:, :, 0] = E1                                     # df/dA1
    J[:, :, 1] = (A1 * inv_tau1) * xv * E1 * inv_tau1   # df/dτ1
    J[:, :, 2] = E2                                     # df/dA2
    J[:, :, 3] = (A2 * inv_tau2) * xv * E2 * inv_tau2   # df/dτ2
    J[:, :, 4] = 1.0                                    # df/dB
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
