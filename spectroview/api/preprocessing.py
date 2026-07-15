"""Array-level spectral preprocessing: baseline, cropping, normalization.

These are stateless functions operating directly on (x, Y) arrays. For a
stateful session that tracks preprocessing history and fit models across
many spectra, see `spectroview.api.workspace.SpectraWorkspace`.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np

from spectroview.fit_engine.baseline import eval_baseline_batch


def subtract_baseline(x: np.ndarray, Y: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract a baseline from multiple spectra simultaneously.

    Args:
        x: Wavenumber axis, shape (M,).
        Y: Intensity matrix, shape (N, M).
        config: Baseline configuration dict, e.g. {'mode': 'arpls', 'coef': 5},
            or any pybaselines method name as 'mode' (e.g. 'asls', 'airpls').

    Returns:
        (Y_corrected, Y_baseline): both shape (N, M).
    """
    Y_baseline = eval_baseline_batch(x, Y, config)
    Y_corrected = Y - Y_baseline
    return Y_corrected, Y_baseline


def crop_spectra(
    x: np.ndarray,
    Y: np.ndarray,
    range_min: Optional[float] = None,
    range_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop spectra to a wavenumber range.

    Args:
        x: Wavenumber axis, shape (M,).
        Y: Intensity matrix, shape (N, M).
        range_min: Minimum wavenumber (cm-1). None means no lower bound.
        range_max: Maximum wavenumber (cm-1). None means no upper bound.

    Returns:
        (x_cropped, Y_cropped).
    """
    if range_min is None and range_max is None:
        return x.copy(), Y.copy()

    mask = np.ones_like(x, dtype=bool)
    if range_min is not None:
        mask &= x >= range_min
    if range_max is not None:
        mask &= x <= range_max

    return x[mask], Y[:, mask]


def normalize_spectra(Y: np.ndarray) -> np.ndarray:
    """Normalize each spectrum to its own maximum intensity.

    Args:
        Y: Intensity matrix, shape (N, M).

    Returns:
        Y_normalized: shape (N, M), max of each row is 1.0.
    """
    max_vals = np.max(Y, axis=1, keepdims=True)
    max_vals[max_vals == 0] = 1.0
    return Y / max_vals
