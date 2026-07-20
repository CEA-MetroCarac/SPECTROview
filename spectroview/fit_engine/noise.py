"""Robust noise estimation shared across the fit engine.

Central home for the two noise primitives reused by `weights.py` (per-fit
weighting) and `evaluator.compute_noise_stats` (noise thresholding), plus the
single-spectrum estimate the Spectra viewer displays. Keeping the exact
formulas here avoids the drift that comes from re-deriving them per call site.
"""
import numpy as np


def mad_noise(y: np.ndarray, axis: int = -1) -> np.ndarray:
    """Estimate noise amplitude from the median absolute first difference.

    For i.i.d. samples the first differences have std sigma·sqrt(2), and
    median(|dy|)/0.6745 is a robust estimator of that std, so the result
    approximates the noise amplitude. Returns a scalar for 1-D input, else (N,).
    """
    dy = np.diff(y, axis=axis)
    return np.median(np.abs(dy), axis=axis) / 0.6745 * np.sqrt(2)


def moving_average_5(y: np.ndarray) -> np.ndarray:
    """5-point moving average along the last axis (edge-padded for 2-D input)."""
    if y.ndim == 2:
        yp = np.pad(y, ((0, 0), (2, 2)), mode='edge')
        return (yp[:, 0:-4] + yp[:, 1:-3] + yp[:, 2:-2] + yp[:, 3:-1] + yp[:, 4:]) / 5.0
    return np.convolve(y, np.ones(5, dtype=np.float64) / 5.0, mode='same')


def detect_noise_level(y: np.ndarray) -> float:
    """Scalar noise amplitude for a single 1-D spectrum (see `mad_noise`)."""
    return float(mad_noise(y))
