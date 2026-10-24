"""Spectral Parameters calculation module.

Provides vectorized computation of spectral parameters over selected spectral
intervals for hyperspectral datasets and single spectra, as well as 2D map grid
generation for visualization.
"""

from __future__ import annotations
from typing import Tuple, Optional, Any, Dict
import numpy as np


PARAM_INTEGRATED_INTENSITY = "Integrated intensity"
PARAM_MEAN_INTENSITY = "Mean intensity"
PARAM_MAX_INTENSITY = "Maximum intensity"
PARAM_PEAK_POSITION = "Peak position"
PARAM_PEAK_HEIGHT = "Peak height"
PARAM_PEAK_POSITION_FIT = "Peak position (fit)"
PARAM_PEAK_HEIGHT_FIT = "Peak height (fit)"
PARAM_PEAK_AREA_FIT = "Peak area (fit)"

# All supported parameters
SPECTRAL_PARAMETERS = [
    PARAM_INTEGRATED_INTENSITY,
    PARAM_MEAN_INTENSITY,
    PARAM_MAX_INTENSITY,
    PARAM_PEAK_POSITION,
    PARAM_PEAK_HEIGHT,
    PARAM_PEAK_POSITION_FIT,
    PARAM_PEAK_HEIGHT_FIT,
    PARAM_PEAK_AREA_FIT,
]

RAW_PARAMETERS = {
    PARAM_INTEGRATED_INTENSITY,
    PARAM_MEAN_INTENSITY,
    PARAM_MAX_INTENSITY,
    PARAM_PEAK_POSITION,
    PARAM_PEAK_HEIGHT,
}

FIT_PARAMETERS = {
    PARAM_PEAK_POSITION_FIT,
    PARAM_PEAK_HEIGHT_FIT,
    PARAM_PEAK_AREA_FIT,
}

SELECTOR_RED = "red"
SELECTOR_BLUE = "blue"
SELECTOR_GREEN = "green"

SELECTOR_KEYS = [SELECTOR_RED, SELECTOR_BLUE, SELECTOR_GREEN]

SELECTOR_COLORS = {
    SELECTOR_RED: "#e74c3c",    # Vibrant red
    SELECTOR_BLUE: "#3498db",   # Vibrant blue
    SELECTOR_GREEN: "#2ecc71",  # Vibrant green
}

SELECTOR_NAMES = {
    SELECTOR_RED: "Red",
    SELECTOR_BLUE: "Blue",
    SELECTOR_GREEN: "Green",
}


def is_fit_parameter(parameter: str) -> bool:
    """Return True if parameter requires curve fitting results."""
    return parameter in FIT_PARAMETERS or "(fit)" in parameter.lower()


def compute_spectral_parameter(
    x_axis: np.ndarray,
    Y: np.ndarray,
    x_range: Tuple[float, float],
    parameter: str = PARAM_INTEGRATED_INTENSITY,
    baseline_subtracted: bool = False,
    peak_params: Optional[np.ndarray] = None,
    param_names: Optional[list[str]] = None,
    fit_model: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, bool, str]:
    """Vectorized calculation of spectral parameter across N spectra.

    Parameters
    ----------
    x_axis : np.ndarray
        Spectral wavenumber axis of shape (M,).
    Y : np.ndarray
        Intensity matrix of shape (N, M) or 1D array of shape (M,).
    x_range : tuple of (float, float)
        Spectral interval (x_start, x_end).
    parameter : str
        The parameter to calculate.
    baseline_subtracted : bool
        Whether Y is already baseline-subtracted.
    peak_params : np.ndarray, optional
        Fitted parameters matrix of shape (N, K).
    param_names : list of str, optional
        List of K fitted parameter names (e.g. ['m01_x0', 'm01_ampli', ...]).
    fit_model : dict, optional
        Fit model specification.

    Returns
    -------
    values : np.ndarray
        Array of shape (N,) containing calculated parameter values.
    is_valid : bool
        True if calculation succeeded, False if parameters could not be calculated.
    message : str
        Status message or reason for fallback.
    """
    if x_axis is None or len(x_axis) == 0:
        return np.array([np.nan]), False, "Empty spectral axis"

    x_arr = np.asarray(x_axis, dtype=np.float64)
    y_arr = np.asarray(Y, dtype=np.float64)

    # Ensure 2D (N, M)
    is_1d = (y_arr.ndim == 1)
    if is_1d:
        y_arr = y_arr[np.newaxis, :]
    n_spectra, n_points = y_arr.shape

    if n_points != len(x_arr):
        return np.full(n_spectra, np.nan), False, f"Axis length mismatch: {len(x_arr)} vs {n_points}"

    # Normalize range
    if x_range is None or len(x_range) < 2:
        return np.full(n_spectra, np.nan), False, "Invalid spectral interval"

    x_min = min(float(x_range[0]), float(x_range[1]))
    x_max = max(float(x_range[0]), float(x_range[1]))

    # Find points inside interval
    mask = (x_arr >= x_min) & (x_arr <= x_max)
    n_sub = int(np.sum(mask))

    if n_sub == 0:
        return np.full(n_spectra, np.nan), False, f"No spectral points in interval [{x_min:.1f}, {x_max:.1f}]"

    x_sub = x_arr[mask]
    Y_sub = y_arr[:, mask]

    # Clean NaNs and Infs
    if np.isnan(Y_sub).any() or np.isinf(Y_sub).any():
        Y_sub = np.nan_to_num(Y_sub, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Fit-dependent parameters ──
    if is_fit_parameter(parameter):
        return _compute_fit_parameter(
            x_min, x_max, n_spectra, parameter,
            peak_params, param_names, fit_model
        )

    # ── Direct / Raw parameters ──
    # 1. Integrated intensity
    if parameter == PARAM_INTEGRATED_INTENSITY:
        if n_sub >= 2:
            trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
            # Sort x_sub if descending so integration is positive
            if x_sub[0] > x_sub[-1]:
                val = np.abs(trapz_fn(Y_sub[:, ::-1], x_sub[::-1], axis=1))
            else:
                val = np.abs(trapz_fn(Y_sub, x_sub, axis=1))
        else:
            val = Y_sub[:, 0].copy()
        return val, True, "OK"

    # 2. Mean intensity
    elif parameter == PARAM_MEAN_INTENSITY:
        val = np.mean(Y_sub, axis=1)
        return val, True, "OK"

    # 3. Maximum intensity
    elif parameter == PARAM_MAX_INTENSITY:
        val = np.max(Y_sub, axis=1)
        return val, True, "OK"

    # 4. Peak position (raw/spectrum)
    elif parameter == PARAM_PEAK_POSITION:
        idx_max = np.argmax(Y_sub, axis=1)
        val = x_sub[idx_max]
        return val, True, "OK"

    # 5. Peak height (raw/spectrum, relative to local baseline)
    elif parameter == PARAM_PEAK_HEIGHT:
        if baseline_subtracted:
            # Already baseline corrected; height is simply max intensity
            val = np.max(Y_sub, axis=1)
        else:
            if n_sub >= 2:
                # Local linear baseline connecting interval endpoints
                dx = x_sub[-1] - x_sub[0]
                if abs(dx) > 1e-12:
                    frac = (x_sub - x_sub[0]) / dx
                    baseline = Y_sub[:, :1] + frac[None, :] * (Y_sub[:, -1:] - Y_sub[:, :1])
                    val = np.max(Y_sub - baseline, axis=1)
                else:
                    val = np.max(Y_sub, axis=1) - np.min(Y_sub, axis=1)
            else:
                val = Y_sub[:, 0].copy()
        return val, True, "OK"

    # Unknown parameter fallback
    else:
        val = np.mean(Y_sub, axis=1)
        return val, True, f"Unknown parameter '{parameter}', defaulted to mean intensity"


def _compute_fit_parameter(
    x_min: float,
    x_max: float,
    n_spectra: int,
    parameter: str,
    peak_params: Optional[np.ndarray],
    param_names: Optional[list[str]],
    fit_model: Optional[Dict[str, Any]],
) -> Tuple[np.ndarray, bool, str]:
    """Extract fitted peak parameter for peaks located within [x_min, x_max]."""
    if peak_params is None or param_names is None or len(param_names) == 0:
        return np.full(n_spectra, np.nan), False, "No fitting results available. Run Fit Model first."

    p_arr = np.asarray(peak_params)
    if p_arr.shape[0] != n_spectra:
        return np.full(n_spectra, np.nan), False, "Fit parameters shape mismatch"

    # Identify all peaks from param_names (e.g. m01_x0, m02_x0 or x0_Peak 1)
    # Map prefix/peak to its indices
    centers = {}  # peak_id -> (col_idx, name)
    amplis = {}   # peak_id -> (col_idx, name)
    fwhms = {}    # peak_id -> (col_idx, name)

    for i, name in enumerate(param_names):
        # Format 1: m01_x0, m01_ampli, m01_fwhm
        if "_" in name:
            parts = name.split("_")
            if len(parts) == 2:
                p_prefix, p_prop = parts[0], parts[1]
                if p_prop == "x0":
                    centers[p_prefix] = i
                elif p_prop in ("ampli", "amplitude", "height", "A"):
                    amplis[p_prefix] = i
                elif p_prop in ("fwhm", "w", "sigma"):
                    fwhms[p_prefix] = i
                # Format 2: x0_Peak 1, ampli_Peak 1
                elif p_prefix == "x0":
                    centers[p_prop] = i
                elif p_prefix in ("ampli", "amplitude", "height", "A"):
                    amplis[p_prop] = i
                elif p_prefix in ("fwhm", "w", "sigma"):
                    fwhms[p_prop] = i

    if not centers:
        return np.full(n_spectra, np.nan), False, "No peak centers found in fit results"

    # For each peak, evaluate whether its median center is inside [x_min, x_max]
    candidates = []
    for peak_id, c_idx in centers.items():
        peak_centers = p_arr[:, c_idx]
        median_center = np.nanmedian(peak_centers)
        if x_min <= median_center <= x_max:
            # Measure amplitude if available
            a_idx = amplis.get(peak_id, None)
            mean_amp = np.nanmean(p_arr[:, a_idx]) if a_idx is not None else 1.0
            candidates.append((peak_id, c_idx, a_idx, fwhms.get(peak_id, None), mean_amp))

    if not candidates:
        return (
            np.full(n_spectra, np.nan),
            False,
            f"No fitted peak centers in interval [{x_min:.1f}, {x_max:.1f}]",
        )

    # Pick peak with highest mean amplitude
    candidates.sort(key=lambda item: item[4], reverse=True)
    best_peak_id, best_c_idx, best_a_idx, best_f_idx, _ = candidates[0]

    if parameter == PARAM_PEAK_POSITION_FIT:
        val = p_arr[:, best_c_idx].copy()
        return val, True, f"Fitted peak center ({best_peak_id})"

    elif parameter == PARAM_PEAK_HEIGHT_FIT:
        if best_a_idx is not None:
            val = p_arr[:, best_a_idx].copy()
            return val, True, f"Fitted peak height ({best_peak_id})"
        else:
            return np.full(n_spectra, np.nan), False, f"Amplitude parameter missing for {best_peak_id}"

    elif parameter == PARAM_PEAK_AREA_FIT:
        if best_a_idx is not None and best_f_idx is not None:
            # Approximate area ~ ampli * fwhm * shape_factor
            # Gaussian: 1.064 * A * w; Lorentzian: 1.571 * A * w
            # Standard SPECTROview area calculation from evaluator
            amp = p_arr[:, best_a_idx]
            fwhm = p_arr[:, best_f_idx]
            val = amp * fwhm * 1.0644  # Normalized pseudo-area
            return val, True, f"Fitted peak area ({best_peak_id})"
        elif best_a_idx is not None:
            val = p_arr[:, best_a_idx].copy()
            return val, True, f"Fitted peak amplitude used as area ({best_peak_id})"
        else:
            return np.full(n_spectra, np.nan), False, f"Area parameters missing for {best_peak_id}"

    return np.full(n_spectra, np.nan), False, f"Unknown fit parameter '{parameter}'"


def compute_map_grid(
    coords: np.ndarray,
    values: np.ndarray,
) -> Tuple[Optional[np.ndarray], list[float], float, float]:
    """Construct 2D interpolated or pixelated grid from coordinates and values.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (N, 2) containing spatial (X, Y) positions.
    values : np.ndarray
        Array of shape (N,) containing parameter values to map.

    Returns
    -------
    grid : np.ndarray or None
        2D array of shape (n_y, n_x) representing the map grid.
    extent : list of 4 floats
        Bounding box [xmin, xmax, ymin, ymax] for imshow.
    vmin : float
        Suggested minimum display value.
    vmax : float
        Suggested maximum display value.
    """
    if coords is None or len(coords) == 0 or values is None or len(values) == 0:
        return None, [0.0, 1.0, 0.0, 1.0], 0.0, 100.0

    c_arr = np.asarray(coords, dtype=np.float64)
    v_arr = np.asarray(values, dtype=np.float64)

    if c_arr.ndim != 2 or c_arr.shape[1] != 2 or len(c_arr) != len(v_arr):
        return None, [0.0, 1.0, 0.0, 1.0], 0.0, 100.0

    X = c_arr[:, 0]
    Y = c_arr[:, 1]

    # Robust vmin and vmax (ignoring NaNs)
    valid_vals = v_arr[np.isfinite(v_arr)]
    if len(valid_vals) > 0:
        vmin = float(np.percentile(valid_vals, 1))
        vmax = float(np.percentile(valid_vals, 99))
        if vmin == vmax:
            vmin = float(valid_vals.min())
            vmax = float(valid_vals.max())
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0
    else:
        vmin, vmax = 0.0, 1.0

    u_x = np.sort(np.unique(X))
    u_y = np.sort(np.unique(Y))

    # Calculate pixel extents
    if len(u_x) > 1:
        dx = float(u_x[1] - u_x[0])
    else:
        dx = 1.0

    if len(u_y) > 1:
        dy = float(u_y[1] - u_y[0])
    else:
        dy = 1.0

    extent = [
        float(u_x[0] - 0.5 * dx),
        float(u_x[-1] + 0.5 * dx),
        float(u_y[0] - 0.5 * dy),
        float(u_y[-1] + 0.5 * dy),
    ]

    # Fast 2D regular grid via searchsorted
    # Check if grid is reasonably compact
    n_expected = len(u_x) * len(u_y)
    if n_expected <= len(X) * 4 and len(u_x) >= 1 and len(u_y) >= 1:
        grid = np.full((len(u_y), len(u_x)), np.nan, dtype=np.float64)
        x_idx = np.searchsorted(u_x, X)
        y_idx = np.searchsorted(u_y, Y)
        # Safe indexing
        valid_idx = (x_idx < len(u_x)) & (y_idx < len(u_y))
        grid[y_idx[valid_idx], x_idx[valid_idx]] = v_arr[valid_idx]
        return grid, extent, vmin, vmax

    # Non-regular / scattered points fallback
    if len(valid_vals) >= 4:
        try:
            from scipy.interpolate import griddata
            grid_x, grid_y = np.meshgrid(
                np.linspace(u_x[0], u_x[-1], 80),
                np.linspace(u_y[0], u_y[-1], 80),
            )
            valid_mask = np.isfinite(v_arr)
            grid_z = griddata(
                (X[valid_mask], Y[valid_mask]),
                v_arr[valid_mask],
                (grid_x, grid_y),
                method="linear",
            )
            return grid_z, extent, vmin, vmax
        except Exception:
            pass

    # Simple fallback: 1xN or Nx1 grid
    grid = np.array([v_arr])
    return grid, extent, vmin, vmax
