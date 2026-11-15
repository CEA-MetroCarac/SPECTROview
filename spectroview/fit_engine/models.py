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
    inv_w2 = 1.0 / (w * w)      # (N,1) — one reciprocal instead of dividing (N,M) by w2 below
    if out is None:
        D = x[None, :] - c if x.ndim == 1 else x - c
    else:
        D = out
        np.subtract(x[None, :] if x.ndim == 1 else x, c, out=D)
    D *= D
    D *= (4.0 * inv_w2)
    D += 1.0
    np.divide(a, D, out=D)
    return D


def batched_lorentzian_jac(x, params, out=None):
    a = params[:, 0:1]
    w = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    c = params[:, 2:3]
    inv_w = 1.0 / w
    inv_w2 = inv_w * inv_w
    N = params.shape[0]
    M = x.shape[-1]
    J = np.empty((N, M, 3), dtype=np.float64) if out is None else out
    J0 = J[:, :, 0]
    J1 = J[:, :, 1]
    J2 = J[:, :, 2]

    np.subtract(x[None, :] if x.ndim == 1 else x, c, out=J1)  # J1 = dx
    np.square(J1, out=J0)
    J0 *= (4.0 * inv_w2)
    J0 += 1.0
    np.reciprocal(J0, out=J0)                                  # dL/da = invD
    np.square(J0, out=J2)
    J2 *= ((8.0 * a) * inv_w2)
    J2 *= J1                                                   # dL/dx0 = dx * 8a/w^2 * invD^2
    J1 *= J2
    J1 *= inv_w                                                # dL/dw = (dx/w) * dL/dx0
    return J


# ═══════════════════════════════════════════════════════════════════════════
# Gaussian:  G = ampli · exp(-4·ln2·((x-x0)/fwhm)²)
#   params[:, 0] = ampli,  params[:, 1] = fwhm,  params[:, 2] = x0
# ═══════════════════════════════════════════════════════════════════════════

def batched_gaussian(x, params, out=None):
    a = params[:, 0:1]
    w = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    c = params[:, 2:3]
    inv_w2 = 1.0 / (w * w)
    if out is None:
        t2 = x[None, :] - c if x.ndim == 1 else x - c
    else:
        t2 = out
        np.subtract(x[None, :] if x.ndim == 1 else x, c, out=t2)
    t2 *= t2
    t2 *= (-_4LOG2 * inv_w2)
    np.exp(t2, out=t2)
    t2 *= a
    return t2


def batched_gaussian_jac(x, params, out=None):
    a = params[:, 0:1]
    w = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    c = params[:, 2:3]
    inv_w = 1.0 / w
    inv_w2 = inv_w * inv_w
    N = params.shape[0]
    M = x.shape[-1]
    J = np.empty((N, M, 3), dtype=np.float64) if out is None else out
    J0 = J[:, :, 0]
    J1 = J[:, :, 1]
    J2 = J[:, :, 2]

    np.subtract(x[None, :] if x.ndim == 1 else x, c, out=J1)  # J1 = dx
    np.square(J1, out=J0)
    J0 *= (-_4LOG2 * inv_w2)
    np.exp(J0, out=J0)                                         # dG/da  = exp(-4·ln2·t²)
    np.multiply(J0, ((2.0 * _4LOG2) * a * inv_w2), out=J2)
    J2 *= J1                                                   # dG/dx0
    J1 *= J2
    J1 *= inv_w                                                # dG/dw = (dx/w) * dG/dx0
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

    if out is None:
        g_unit = t2 * (-_4LOG2)
    else:
        g_unit = out
        np.multiply(t2, -_4LOG2, out=g_unit)
    np.exp(g_unit, out=g_unit)

    # Reuse t2 buffer in-place for l_unit = 1 / (4·t2 + 1).
    t2 *= 4.0
    t2 += 1.0
    np.reciprocal(t2, out=t2)

    # alpha*G_unit + (1-alpha)*L_unit == L_unit + alpha*(G_unit - L_unit)
    g_unit -= t2
    g_unit *= alpha
    g_unit += t2
    g_unit *= a
    return g_unit


def batched_pseudovoigt_jac(x, params, out=None):
    a = params[:, 0:1]
    w = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    c = params[:, 2:3]
    alpha = params[:, 3:4]
    inv_w = 1.0 / w
    inv_w2 = inv_w * inv_w
    N = params.shape[0]
    M = x.shape[-1]
    J = np.empty((N, M, 4), dtype=np.float64) if out is None else out
    J0 = J[:, :, 0]
    J1 = J[:, :, 1]
    J2 = J[:, :, 2]
    J3 = J[:, :, 3]

    np.subtract(x[None, :] if x.ndim == 1 else x, c, out=J1)  # J1 = dx
    np.square(J1, out=J0)
    J0 *= inv_w2                                               # J0 = t2 = (dx/w)^2

    # Compute g_unit and invD once cleanly from shared t2.
    np.multiply(J0, -_4LOG2, out=J2)
    np.exp(J2, out=J2)                                         # J2 = g_unit

    J0 *= 4.0
    J0 += 1.0
    np.reciprocal(J0, out=J0)                                  # J0 = invD = l_unit

    np.subtract(J2, J0, out=J3)                                # J3 = g_unit - invD

    # dPV/dx0 = dx * inv_w2 * a * (alpha*(2*4*ln2)*g_unit + (1-alpha)*8*invD^2)
    J2 *= ((2.0 * _4LOG2) * alpha)
    scratch = J0 * J0
    scratch *= (8.0 * (1.0 - alpha))
    J2 += scratch
    J2 *= (a * inv_w2)
    J2 *= J1                                                   # dPV/dx0

    # dPV/dampli = invD + alpha*(g_unit - invD); dPV/dalpha = a*(g_unit - invD)
    np.multiply(J3, alpha, out=scratch)
    J0 += scratch                                              # dPV/dampli
    J3 *= a                                                    # dPV/dalpha

    # dPV/dw = (dx / w) * dPV/dx0  (both G and L depend strictly on u = dx/w)
    J1 *= J2
    J1 *= inv_w                                                # dPV/dw
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
    if out is None:
        dx = x[None, :] - c if x.ndim == 1 else x - c
    else:
        dx = out
        np.subtract(x[None, :] if x.ndim == 1 else x, c, out=dx)
    coef_l = (-_4LOG2) / (wl * wl)
    coef_r = (-_4LOG2) / (wr * wr)
    coef = np.where(dx < 0, coef_l, coef_r)
    dx *= dx
    dx *= coef
    np.exp(dx, out=dx)
    dx *= a
    return dx


def batched_gaussian_asym_jac(x, params, out=None):
    a  = params[:, 0:1]
    wl = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    wr = np.maximum(np.abs(params[:, 2:3]), 1e-15)
    c  = params[:, 3:4]
    inv_wl = 1.0 / wl
    inv_wr = 1.0 / wr
    N = params.shape[0]
    M = x.shape[-1]
    J = np.empty((N, M, 4), dtype=np.float64) if out is None else out
    J0 = J[:, :, 0]
    J1 = J[:, :, 1]
    J2 = J[:, :, 2]
    J3 = J[:, :, 3]

    np.subtract(x[None, :] if x.ndim == 1 else x, c, out=J1)  # J1 = dx
    neg = J1 < 0
    inv_w = np.where(neg, inv_wl, inv_wr)
    inv_w2 = inv_w * inv_w
    np.square(J1, out=J0)
    J0 *= (-_4LOG2 * inv_w2)
    np.exp(J0, out=J0)                                         # dG/d_ampli = g_unit

    np.multiply(J0, inv_w2, out=J3)
    J3 *= ((2.0 * _4LOG2) * a)
    J3 *= J1                                                   # dG/d_x0
    J1 *= J3
    J1 *= inv_w                                                # shared width derivative
    J2[:] = np.where(neg, 0.0, J1)                             # dG/d_fwhm_r
    J1[:] = np.where(neg, J1, 0.0)                             # dG/d_fwhm_l
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
    if out is None:
        dx = x[None, :] - c if x.ndim == 1 else x - c
    else:
        dx = out
        np.subtract(x[None, :] if x.ndim == 1 else x, c, out=dx)
    coef_l = 4.0 / (wl * wl)
    coef_r = 4.0 / (wr * wr)
    coef = np.where(dx < 0, coef_l, coef_r)
    dx *= dx
    dx *= coef
    dx += 1.0
    np.divide(a, dx, out=dx)
    return dx


def batched_lorentzian_asym_jac(x, params, out=None):
    a  = params[:, 0:1]
    wl = np.maximum(np.abs(params[:, 1:2]), 1e-15)
    wr = np.maximum(np.abs(params[:, 2:3]), 1e-15)
    c  = params[:, 3:4]
    inv_wl = 1.0 / wl
    inv_wr = 1.0 / wr
    N = params.shape[0]
    M = x.shape[-1]
    J = np.empty((N, M, 4), dtype=np.float64) if out is None else out
    J0 = J[:, :, 0]
    J1 = J[:, :, 1]
    J2 = J[:, :, 2]
    J3 = J[:, :, 3]

    np.subtract(x[None, :] if x.ndim == 1 else x, c, out=J1)  # J1 = dx
    neg = J1 < 0
    inv_w = np.where(neg, inv_wl, inv_wr)
    inv_w2 = inv_w * inv_w
    np.square(J1, out=J0)
    J0 *= (4.0 * inv_w2)
    J0 += 1.0
    np.reciprocal(J0, out=J0)                                  # dL/d_ampli = invD

    np.square(J0, out=J3)
    J3 *= inv_w2
    J3 *= (8.0 * a)
    J3 *= J1                                                   # dL/d_x0
    J1 *= J3
    J1 *= inv_w                                                # shared width derivative
    J2[:] = np.where(neg, 0.0, J1)                             # dL/d_fwhm_r
    J1[:] = np.where(neg, J1, 0.0)                             # dL/d_fwhm_l
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
    inv_w = 1.0 / w
    N = params.shape[0]
    M = x.shape[-1]
    J = np.empty((N, M, 4), dtype=np.float64) if out is None else out
    J0 = J[:, :, 0]
    J1 = J[:, :, 1]
    J2 = J[:, :, 2]
    J3 = J[:, :, 3]

    # J1 = eps = 2*(x - c)/w
    np.subtract(x[None, :] if x.ndim == 1 else x, c, out=J1)
    J1 *= (2.0 * inv_w)
    # J0 = qe = q + eps
    np.add(J1, q, out=J0)
    # J3 = inv_den = 1 / (1 + eps^2)
    np.square(J1, out=J3)
    J3 += 1.0
    np.reciprocal(J3, out=J3)

    # J2 = df_deps = a · 2(q+ε)(1 - εq) / (1+ε²)²
    np.multiply(J1, -q, out=J2)
    J2 += 1.0
    J2 *= J0
    J2 *= (2.0 * a)
    J2 *= J3
    J2 *= J3

    # J3 = qe * inv_den -> J0 = qe^2 * inv_den, J3 = 2*a * qe * inv_den
    J3 *= J0
    J0 *= J3                                        # df/d_ampli
    J3 *= (2.0 * a)                                 # df/d_q
    J1 *= J2
    J1 *= (-inv_w)                                  # df/d_fwhm  (dε/dw = -ε/w)
    J2 *= (-2.0 * inv_w)                            # df/d_x0    (dε/dx0 = -2/w)
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
    if out is None:
        E = xv * (-inv_tau)
    else:
        E = out
        np.multiply(xv, -inv_tau, out=E)
    np.exp(E, out=E)
    E *= A
    E += B
    return E


def batched_decay_single_exp_jac(x, params, out=None):
    A   = params[:, 0:1]
    tau = np.maximum(np.abs(params[:, 1:2]), 1e-30)
    xv = x[None, :] if x.ndim == 1 else x
    inv_tau = 1.0 / tau
    N = params.shape[0]
    M = x.shape[-1]
    J = np.empty((N, M, 3), dtype=np.float64) if out is None else out
    J0 = J[:, :, 0]
    J1 = J[:, :, 1]
    J2 = J[:, :, 2]

    np.multiply(xv, -inv_tau, out=J0)
    np.exp(J0, out=J0)                              # df/dA = exp(-x/τ)
    np.multiply(xv, J0, out=J1)
    J1 *= (A * (inv_tau * inv_tau))                 # df/dτ = A·x·exp(-x/τ)/τ²
    J2.fill(1.0)                                    # df/dB
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
    inv_tau2 = 1.0 / tau2
    N = params.shape[0]
    M = x.shape[-1]
    J = np.empty((N, M, 5), dtype=np.float64) if out is None else out
    J0 = J[:, :, 0]
    J1 = J[:, :, 1]
    J2 = J[:, :, 2]
    J3 = J[:, :, 3]
    J4 = J[:, :, 4]

    np.multiply(xv, -inv_tau1, out=J0)
    np.exp(J0, out=J0)                              # df/dA1
    np.multiply(xv, J0, out=J1)
    J1 *= (A1 * (inv_tau1 * inv_tau1))              # df/dτ1
    np.multiply(xv, -inv_tau2, out=J2)
    np.exp(J2, out=J2)                              # df/dA2
    np.multiply(xv, J2, out=J3)
    J3 *= (A2 * (inv_tau2 * inv_tau2))              # df/dτ2
    J4.fill(1.0)                                    # df/dB
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
