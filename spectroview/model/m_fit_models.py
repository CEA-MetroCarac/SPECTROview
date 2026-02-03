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


def decay_single_exp(x, A, tau, B):
    r"""
    Single exponential decay function for TRPL (Time-Resolved Photoluminescence).
    
    The function is defined as:
    :math:`I(t) = A \cdot e^{-t/\tau} + B`
    
    Parameters
    ----------
    x : array-like
        Time values (typically in nanoseconds)
    A : float
        Amplitude of the exponential decay
    tau : float
        Decay time constant (lifetime)
    B : float
        Baseline offset (background intensity)
    
    Returns
    -------
    array-like
        Intensity values at each time point
    
    Example
    -------
    >>> import numpy as np
    >>> t = np.linspace(0, 100, 1001)
    >>> y = decay_single_exp(t, A=1000, tau=20, B=10)
    """
    return A * np.exp(-x / tau) + B


def decay_bi_exp(x, A1, tau1, A2, tau2, B):
    r"""
    Bi-exponential decay function for TRPL (Time-Resolved Photoluminescence).
    
    Models decay with two distinct lifetime components, commonly observed in
    systems with multiple decay pathways or heterogeneous populations.
    
    The function is defined as:
    :math:`I(t) = A_1 \cdot e^{-t/\tau_1} + A_2 \cdot e^{-t/\tau_2} + B`
    
    Parameters
    ----------
    x : array-like
        Time values (typically in nanoseconds)
    A1 : float
        Amplitude of the first exponential component
    tau1 : float
        First decay time constant (fast lifetime)
    A2 : float
        Amplitude of the second exponential component
    tau2 : float
        Second decay time constant (slow lifetime)
    B : float
        Baseline offset (background intensity)
    
    Returns
    -------
    array-like
        Intensity values at each time point
    
    Example
    -------
    >>> import numpy as np
    >>> t = np.linspace(0, 100, 1001)
    >>> y = decay_bi_exp(t, A1=800, tau1=10, A2=200, tau2=40, B=5)
    """
    return A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2) + B
