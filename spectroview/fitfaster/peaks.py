import numpy as np
import numba as nb

@nb.njit(fastmath=True)
def gaussian(x, amp, fwhm, x0):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return amp * np.exp(-(x - x0)**2 / (2*sigma**2))

@nb.njit(fastmath=True)
def gaussian_asym(x, amp, fwhm_l, fwhm_r, x0):
    y = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] < x0:
            y[i] = gaussian(x[i:i+1], amp, fwhm_l, x0)[0]
        else:
            y[i] = gaussian(x[i:i+1], amp, fwhm_r, x0)[0]
    return y

@nb.njit(fastmath=True)
def lorentzian(x, amp, fwhm, x0):
    return amp * fwhm**2 / (4*(x-x0)**2 + fwhm**2 + 1e-12)

@nb.njit(fastmath=True)
def lorentzian_asym(x, amp, fwhm_l, fwhm_r, x0):
    y = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] < x0:
            y[i] = lorentzian(x[i:i+1], amp, fwhm_l, x0)[0]
        else:
            y[i] = lorentzian(x[i:i+1], amp, fwhm_r, x0)[0]
    return y

@nb.njit(fastmath=True)
def pseudovoigt(x, amp, fwhm, x0, alpha):
    return alpha * gaussian(x, amp, fwhm, x0) + \
           (1-alpha) * lorentzian(x, amp, fwhm, x0)
