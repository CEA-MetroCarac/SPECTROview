# spectroview/model/m_fit_models.py
"""
Custom fitting models for SPECTROview.
Extends fitspy's built-in models with additional lineshapes.
"""
import numpy as np

np.seterr(divide='ignore', invalid='ignore')


def fano(x, ampli, fwhm, x0, q=1.0):
    r"""
    Fano lineshape function.

    Describes asymmetric spectral profiles arising from quantum mechanical
    interference between a discrete resonant state and a continuum of states.

    The function is defined as:
    :math:`ampli * (q + \epsilon)^2 / (1 + \epsilon^2)`
    where :math:`\epsilon = (x - x0) / (\Gamma / 2)` is the reduced energy.

    Parameters
    ----------
    x : array-like
        Independent variable (e.g., wavenumber, energy, frequency)
    ampli : float
        Amplitude of the peak
    fwhm : float
        Full Width at Half Maximum (Γ)
    x0 : float
        Center position of the resonance
    q : float, optional
        Fano asymmetry parameter. Default is 1.0.
        - q = 0: anti-Lorentzian dip (symmetric minimum)
        - q → ±∞: recovers a Lorentzian peak
        - |q| ~ 1: strongly asymmetric profile

    Returns
    -------
    array-like
        Intensity values at each point x
    """
    gamma_half = fwhm / 2.0
    epsilon = (x - x0) / (gamma_half + 1e-10)  # Avoid division by zero if fwhm → 0

    # Pure Fano formula: ampli * (q + ε)² / (1 + ε²)
    return ampli * (q + epsilon) ** 2 / (1 + epsilon ** 2)
