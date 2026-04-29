# spectroview/core/baseline.py
"""
Batch baseline processing for the high-performance fitting engine.

Handles baseline evaluation and subtraction for multiple spectra sharing
the same x-axis, avoiding redundant computations.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


def batch_preprocess(spectra, x_shared=None):
    """Preprocess a batch of spectra: apply range, evaluate baseline, subtract.

    Uses fitspy's own spectrum.preprocess() to ensure full compatibility
    with the existing engine's behavior.

    Args:
        spectra: List of MSpectrum objects (must have x0, y0 already set)
        x_shared: If provided, the shared x-axis for all spectra. If None,
                  each spectrum uses its own x-axis.

    Returns:
        tuple: (x_array, Y_matrix, baseline_matrix)
            - x_array: 1D array of x-values (shared or from first spectrum)
            - Y_matrix: 2D array (N_spectra, N_wavelengths) baseline-subtracted
            - baseline_matrix: None (reserved for future use)
    """
    if not spectra:
        return None, None, None

    n_spectra = len(spectra)

    # Preprocess each spectrum using fitspy's own method
    for spectrum in spectra:
        spectrum.preprocess()

    # Extract the data matrix
    if x_shared is not None:
        x_array = x_shared
    else:
        x_array = spectra[0].x

    n_wavelengths = len(x_array)
    Y_matrix = np.empty((n_spectra, n_wavelengths), dtype=np.float64)

    for i, spectrum in enumerate(spectra):
        if spectrum.y is not None and len(spectrum.y) == n_wavelengths:
            Y_matrix[i] = spectrum.y
        else:
            # Handle size mismatch (shouldn't happen for maps, but be safe)
            Y_matrix[i] = np.zeros(n_wavelengths)

    return x_array, Y_matrix, None


def _preprocess_single(spectrum):
    """Preprocess a single spectrum without file I/O.

    Unlike fitspy's spectrum.preprocess(), this skips load_profile()
    since map spectra already have their data loaded in memory.
    It applies: range → baseline evaluation → baseline subtraction → normalization.
    """
    # Skip load_profile() — data already in memory for map spectra

    # Ensure base arrays exist
    if spectrum.x is None and spectrum.x0 is not None:
        spectrum.x = spectrum.x0.copy()
    if spectrum.y is None and spectrum.y0 is not None:
        spectrum.y = spectrum.y0.copy()

    if spectrum.x is None or spectrum.y is None:
        return

    # Apply range
    _apply_range(spectrum)

    # Evaluate and subtract baseline
    _eval_and_subtract_baseline(spectrum)

    # Normalization
    _normalization(spectrum)


def _apply_range(spectrum):
    """Apply spectral range to spectrum data."""
    range_min = spectrum.range_min
    range_max = spectrum.range_max

    if range_min is None and range_max is None:
        # No range restriction — use full data
        spectrum.x = spectrum.x0.copy()
        spectrum.y = spectrum.y0.copy()
        if spectrum.weights0 is not None:
            spectrum.weights = spectrum.weights0.copy()
        return

    mask = np.ones(len(spectrum.x0), dtype=bool)
    if range_min is not None:
        mask &= spectrum.x0 >= range_min
    if range_max is not None:
        mask &= spectrum.x0 <= range_max

    spectrum.x = spectrum.x0[mask].copy()
    spectrum.y = spectrum.y0[mask].copy()
    if spectrum.weights0 is not None:
        spectrum.weights = spectrum.weights0[mask].copy()


def _eval_and_subtract_baseline(spectrum):
    """Evaluate baseline and subtract from spectrum if not already done."""
    baseline = spectrum.baseline

    if baseline.is_subtracted:
        # Already subtracted — nothing to do
        return

    if baseline.mode is None:
        # No baseline mode set — nothing to do
        return

    # Evaluate baseline (uses fitspy's baseline.eval)
    try:
        baseline.eval(spectrum.x, spectrum.y, attached=baseline.attached)
    except Exception:
        # If baseline evaluation fails, skip subtraction
        return

    # Subtract baseline
    if baseline.y_eval is not None:
        spectrum.y = spectrum.y - baseline.y_eval
        baseline.is_subtracted = True


def _normalization(spectrum):
    """Apply normalization if enabled."""
    if not spectrum.normalize:
        return

    xmin = spectrum.normalize_range_min or -np.inf
    xmax = spectrum.normalize_range_max or np.inf
    mask = np.logical_and(spectrum.x >= xmin, spectrum.x <= xmax)
    max_value = spectrum.y[mask].max()
    if max_value > 0:
        spectrum.y *= 100.0 / max_value
