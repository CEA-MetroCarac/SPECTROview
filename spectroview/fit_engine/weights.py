"""Fit-weight computation shared by the GUI's VBFthread and the headless API."""
import numpy as np


def compute_fit_weights(Y: np.ndarray, fit_params: dict) -> np.ndarray:
    """Compute per-point fit weights the same way the GUI does before every fit.

    Two exclusions are applied (weight set to 0.0):
      - Negative-intensity points, unless fit_params['fit_negative'] is True.
      - Points below a noise floor derived from fit_params['coef_noise']
        (median absolute deviation of the first difference, smoothed with a
        5-point moving average).

    Args:
        Y: Intensity matrix, shape (N, M).
        fit_params: dict, may contain 'fit_negative' (bool) and 'coef_noise' (float).

    Returns:
        weights: float64[N, M], all 1.0 where no exclusion applies.
    """
    N, M = Y.shape
    weights = np.ones((N, M), dtype=np.float64)

    fit_negative = bool(fit_params.get("fit_negative", False))
    if not fit_negative:
        weights[Y < 0] = 0.0

    coef_noise = float(fit_params.get("coef_noise", 0))
    if coef_noise > 0:
        dy = np.diff(Y, axis=1)
        ampli_noise = np.median(np.abs(dy), axis=1) / 0.6745 * np.sqrt(2)
        Y_padded = np.pad(Y, ((0, 0), (2, 2)), mode='edge')
        ymean = (Y_padded[:, 0:-4] + Y_padded[:, 1:-3] + Y_padded[:, 2:-2] + Y_padded[:, 3:-1] + Y_padded[:, 4:]) / 5.0
        noise_level = coef_noise * ampli_noise
        weights[ymean < noise_level[:, None]] = 0.0

    return weights
