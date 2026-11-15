"""Robust noise estimation shared across the fit engine.

Central home for the two noise primitives reused by `weights.py` (per-fit
weighting) and `evaluator.compute_noise_stats` (noise thresholding), plus the
single-spectrum estimate the Spectra viewer displays. Keeping the exact
formulas here avoids the drift that comes from re-deriving them per call site.
"""
import numpy as np

_LAST_2D_MAD_CACHE = (None, None)
_LAST_2D_MA5_CACHE = (None, None)


def _array_2d_fingerprint(y: np.ndarray):
    """Return a lightweight mutation-sensitive key for a non-empty 2-D array."""
    n, m = y.shape
    return (
        id(y),
        y.shape,
        y.strides,
        y.dtype.str,
        float(y[0, 0]),
        float(y[-1, -1]),
        float(y[n >> 1, m >> 1]),
        float(np.sum(y)),
    )


def mad_noise(y: np.ndarray, axis: int = -1) -> np.ndarray:
    """Estimate noise amplitude from the median absolute first difference.

    For i.i.d. samples the first differences have std sigma·sqrt(2), and
    median(|dy|)/0.6745 is a robust estimator of that std, so the result
    approximates the noise amplitude. Returns a scalar for 1-D input, else (N,).
    """
    global _LAST_2D_MAD_CACHE
    if y is None or y.shape[axis] < 2:
        if y is None or y.ndim == 1:
            return 0.0
        out_shape = list(y.shape)
        out_shape.pop(axis)
        return np.zeros(out_shape, dtype=np.float64)
    is_row_2d = y.ndim == 2 and (axis == -1 or axis == 1) and y.size >= 4096
    if is_row_2d:
        fp = _array_2d_fingerprint(y)
        cached_fp, cached_val = _LAST_2D_MAD_CACHE
        if cached_fp == fp:
            return cached_val
    dy = np.diff(y, axis=axis)
    np.abs(dy, out=dy)
    res = np.median(dy, axis=axis, overwrite_input=True) / 0.6745 * np.sqrt(2)
    if is_row_2d:
        _LAST_2D_MAD_CACHE = (fp, res)
    return res


def moving_average_5(y: np.ndarray) -> np.ndarray:
    """5-point moving average along the last axis (edge-padded for 2-D input)."""
    global _LAST_2D_MA5_CACHE
    if y.ndim == 2:
        N, M = y.shape
        use_cache = y.size >= 4096 and M > 0 and N > 0
        if use_cache:
            fp = _array_2d_fingerprint(y)
            cached_fp, cached_val = _LAST_2D_MA5_CACHE
            if cached_fp == fp:
                return cached_val
        if M >= 5:
            out = np.empty((N, M), dtype=np.float64)
            np.add(y[:, 0:-4], y[:, 1:-3], out=out[:, 2:-2])
            out[:, 2:-2] += y[:, 2:-2]
            out[:, 2:-2] += y[:, 3:-1]
            out[:, 2:-2] += y[:, 4:]
            out[:, 0] = y[:, 0] + y[:, 0] + y[:, 0] + y[:, 1] + y[:, 2]
            out[:, 1] = y[:, 0] + y[:, 0] + y[:, 1] + y[:, 2] + y[:, 3]
            out[:, -2] = y[:, -4] + y[:, -3] + y[:, -2] + y[:, -1] + y[:, -1]
            out[:, -1] = y[:, -3] + y[:, -2] + y[:, -1] + y[:, -1] + y[:, -1]
            out /= 5.0
        else:
            yp = np.pad(y, ((0, 0), (2, 2)), mode='edge')
            out = (yp[:, 0:-4] + yp[:, 1:-3] + yp[:, 2:-2] + yp[:, 3:-1] + yp[:, 4:]) / 5.0
        if use_cache:
            _LAST_2D_MA5_CACHE = (fp, out)
        return out
    return np.convolve(y, np.ones(5, dtype=np.float64) / 5.0, mode='same')


def detect_noise_level(y: np.ndarray) -> float:
    """Scalar noise amplitude for a single 1-D spectrum (see `mad_noise`)."""
    if y is None or len(y) < 2:
        return 0.0
    return float(mad_noise(y))
