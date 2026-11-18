"""Spectral Parameters calculation module.

Provides vectorized computation of spectral parameters over selected spectral
intervals for hyperspectral datasets and single spectra, as well as 2D map grid
generation for visualization.
"""

from __future__ import annotations
from dataclasses import dataclass
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

# Derived display option (Green channel only): pixel-wise ratio of the
# integrated intensities of the Red and Blue spectral ranges.
PARAM_RB_RATIO = "R/B ratio"

# Denominator values below RATIO_REL_EPS * max(|denominator|) are treated as
# undefined (NaN) to avoid division-by-zero and noise-dominated ratios.
RATIO_REL_EPS = 1e-3

# Collinearity tolerance (ratio of the 2nd to the 1st singular value of the
# centered coordinates) under which a map is treated as a 1D profile.
PROFILE_COLLINEAR_TOL = 1e-3

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
    active_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool, str]:
    """Vectorized calculation of spectral parameter across N spectra.

    Slices only the requested wavenumber band (and active rows, when provided)
    before converting to ``float64`` to minimize peak memory overhead on large
    hyperspectral maps.
    """
    if x_axis is None or len(x_axis) == 0:
        return np.array([np.nan]), False, "Empty spectral axis"

    x_arr = np.asarray(x_axis, dtype=np.float64)
    y_raw = np.asarray(Y)

    is_1d = (y_raw.ndim == 1)
    if y_raw.ndim not in (1, 2):
        return np.array([np.nan]), False, "Invalid intensity array dimensions"
    n_spectra = 1 if is_1d else int(y_raw.shape[0])
    n_points = int(y_raw.shape[0] if is_1d else y_raw.shape[1])

    if n_points != len(x_arr):
        return np.full(n_spectra, np.nan), False, f"Axis length mismatch: {len(x_arr)} vs {n_points}"

    if x_range is None or len(x_range) < 2:
        return np.full(n_spectra, np.nan), False, "Invalid spectral interval"

    x_min = min(float(x_range[0]), float(x_range[1]))
    x_max = max(float(x_range[0]), float(x_range[1]))

    act_mask: Optional[np.ndarray] = None
    if active_mask is not None and not is_1d:
        act_arr = np.asarray(active_mask, dtype=bool)
        if len(act_arr) == n_spectra:
            if not np.any(act_arr):
                return np.full(n_spectra, np.nan, dtype=np.float64), False, "No active spectra in map"
            if not np.all(act_arr):
                act_mask = act_arr

    # ── Fit-dependent parameters (no Y slicing or upcasting needed) ──
    if is_fit_parameter(parameter):
        val, ok, msg = _compute_fit_parameter(
            x_min, x_max, n_spectra, parameter,
            peak_params, param_names, fit_model,
        )
        if ok and act_mask is not None:
            val = val.copy()
            val[~act_mask] = np.nan
        return val, ok, msg

    # Slice columns BEFORE float64 conversion to avoid full-matrix allocation
    mask = (x_arr >= x_min) & (x_arr <= x_max)
    n_sub = int(np.count_nonzero(mask))
    if n_sub == 0:
        return np.full(n_spectra, np.nan), False, f"No spectral points in interval [{x_min:.1f}, {x_max:.1f}]"

    x_sub = x_arr[mask]
    if is_1d:
        Y_sub = np.asarray(y_raw[mask], dtype=np.float64)[np.newaxis, :]
    elif act_mask is not None:
        if int(np.count_nonzero(act_mask)) < (n_spectra // 2):
            Y_sub = np.asarray(y_raw[act_mask][:, mask], dtype=np.float64)
        else:
            Y_sub = np.asarray(y_raw[:, mask][act_mask], dtype=np.float64)
    else:
        Y_sub = np.asarray(y_raw[:, mask], dtype=np.float64)

    if np.isnan(Y_sub).any() or np.isinf(Y_sub).any():
        Y_sub = np.nan_to_num(Y_sub, nan=0.0, posinf=0.0, neginf=0.0)

    val, ok, msg = _compute_raw_parameter(x_sub, Y_sub, parameter, baseline_subtracted)
    if act_mask is not None:
        out = np.full(n_spectra, np.nan, dtype=np.float64)
        out[act_mask] = val
        return out, ok, msg
    return val, ok, msg


def _compute_raw_parameter(
    x_sub: np.ndarray,
    Y_sub: np.ndarray,
    parameter: str,
    baseline_subtracted: bool,
) -> Tuple[np.ndarray, bool, str]:
    """Compute non-fit spectral parameter on a pre-sliced ``float64`` sub-band."""
    n_sub = len(x_sub)

    if parameter == PARAM_INTEGRATED_INTENSITY:
        if n_sub >= 2:
            trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
            if x_sub[0] > x_sub[-1]:
                val = np.abs(trapz_fn(Y_sub[:, ::-1], x_sub[::-1], axis=1))
            else:
                val = np.abs(trapz_fn(Y_sub, x_sub, axis=1))
        else:
            val = Y_sub[:, 0].copy()
        return val, True, "OK"

    if parameter == PARAM_MEAN_INTENSITY:
        return np.mean(Y_sub, axis=1), True, "OK"

    if parameter == PARAM_MAX_INTENSITY:
        return np.max(Y_sub, axis=1), True, "OK"

    if parameter == PARAM_PEAK_POSITION:
        return x_sub[np.argmax(Y_sub, axis=1)], True, "OK"

    if parameter == PARAM_PEAK_HEIGHT:
        if baseline_subtracted:
            return np.max(Y_sub, axis=1), True, "OK"
        if n_sub >= 2:
            dx = x_sub[-1] - x_sub[0]
            if abs(dx) > 1e-12:
                frac = (x_sub - x_sub[0]) / dx
                baseline = Y_sub[:, :1] + frac[None, :] * (Y_sub[:, -1:] - Y_sub[:, :1])
                return np.max(Y_sub - baseline, axis=1), True, "OK"
            return np.max(Y_sub, axis=1) - np.min(Y_sub, axis=1), True, "OK"
        return Y_sub[:, 0].copy(), True, "OK"

    return np.mean(Y_sub, axis=1), True, f"Unknown parameter '{parameter}', defaulted to mean intensity"


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


def compute_series_spectral_parameter(
    mds: list[Any],
    x_range: Tuple[float, float],
    parameter: str = PARAM_INTEGRATED_INTENSITY,
) -> Tuple[np.ndarray, list[str], bool, str]:
    """Compute a spectral parameter across a selected series of MapData objects.

    Supports both homogeneous series (batched in a single vectorized NumPy call)
    and heterogeneous series (differing x-axes or per-spectrum fit models).

    Parameters
    ----------
    mds : list of MapData
        Currently selected MapData blocks (typically N=1 per block in Spectra workspace).
    x_range : tuple of (float, float)
        Spectral interval (x_start, x_end).
    parameter : str
        The spectral parameter to calculate.

    Returns
    -------
    values : np.ndarray
        1D array of calculated parameter values across all selected spectra.
    labels : list of str
        Display labels / filenames for each spectrum in the series.
    is_valid : bool
        True if at least one valid value was calculated, False otherwise.
    message : str
        Status message or reason for failure.
    """
    valid_mds = [
        md for md in (mds or [])
        if md is not None and getattr(md, "x_axis", None) is not None and getattr(md, "y_matrix", None) is not None
    ]
    if not valid_mds:
        return np.array([], dtype=np.float64), [], False, "No spectra selected"

    labels: list[str] = []
    for md in valid_mds:
        n_rows = md.y_matrix.shape[0] if md.y_matrix.ndim == 2 else 1
        for i in range(n_rows):
            lbl = None
            if getattr(md, "labels", None) and i < len(md.labels):
                lbl = md.labels[i]
            if not lbl and getattr(md, "fnames", None) and i < len(md.fnames):
                lbl = md.fnames[i]
            if not lbl:
                lbl = getattr(md, "name", f"Spectrum {len(labels) + 1}")
            labels.append(str(lbl))

    # Fast vectorized batch path when all blocks share the same x_axis and baseline status
    if not is_fit_parameter(parameter) and len(valid_mds) >= 1:
        ref_x = valid_mds[0].x_axis
        ref_bl = bool(getattr(valid_mds[0], "is_baseline_subtracted", False))
        same_axis = all(
            len(md.x_axis) == len(ref_x)
            and bool(getattr(md, "is_baseline_subtracted", False)) == ref_bl
            and np.array_equal(md.x_axis, ref_x)
            for md in valid_mds[1:]
        )
        if same_axis:
            Y_stacked = np.vstack([
                md.y_matrix if md.y_matrix.ndim == 2 else md.y_matrix[np.newaxis, :]
                for md in valid_mds
            ])
            vals, is_valid, msg = compute_spectral_parameter(
                x_axis=ref_x,
                Y=Y_stacked,
                x_range=x_range,
                parameter=parameter,
                baseline_subtracted=ref_bl,
            )
            return vals, labels, is_valid, msg

    # Per-block evaluation (handles differing x-axes or per-spectrum fit parameters)
    all_vals = []
    last_msg = ""
    any_valid = False
    for md in valid_mds:
        has_fit = md.has_fit_results() if hasattr(md, "has_fit_results") else False
        v, ok, m = compute_spectral_parameter(
            x_axis=md.x_axis,
            Y=md.y_matrix,
            x_range=x_range,
            parameter=parameter,
            baseline_subtracted=getattr(md, "is_baseline_subtracted", False),
            peak_params=md.peak_params if has_fit else None,
            param_names=md.param_names if has_fit else None,
            fit_model=getattr(md, "fit_model", None),
        )
        all_vals.append(v)
        if ok:
            any_valid = True
            last_msg = m
        elif not last_msg:
            last_msg = m

    vals = np.concatenate(all_vals) if all_vals else np.array([], dtype=np.float64)
    if not any_valid or not np.any(np.isfinite(vals)):
        return vals, labels, False, last_msg or "Unable to calculate parameter"
    return vals, labels, True, last_msg or "OK"


@dataclass(frozen=True)
class MapGridder:
    """Precomputed spatial lookup/weights for repeated 2D heatmap rendering.

    Caches either regular-lattice integer indices or scattered Delaunay
    barycentric weights (falling back to cKDTree nearest-neighbor on Qhull errors)
    so interactive Live Maps bracket dragging avoids re-triangulating coordinates.
    """

    n_points: int
    is_regular: bool
    extent: list[float]
    grid_res: int = 80
    n_x: int = 0
    n_y: int = 0
    x_idx: Optional[np.ndarray] = None
    y_idx: Optional[np.ndarray] = None
    valid_idx: Optional[np.ndarray] = None
    vertices: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    valid_q: Optional[np.ndarray] = None
    nn_indices: Optional[np.ndarray] = None

    def evaluate(self, values: np.ndarray) -> np.ndarray:
        """Project a length-``n_points`` value vector onto a 2D heatmap grid."""
        v_arr = np.asarray(values, dtype=np.float64)
        if self.is_regular:
            grid = np.full((self.n_y, self.n_x), np.nan, dtype=np.float64)
            v_idx = self.valid_idx
            grid[self.y_idx[v_idx], self.x_idx[v_idx]] = v_arr[v_idx]
            return grid

        if self.vertices is not None and self.weights is not None and self.valid_q is not None:
            out_flat = np.full(self.grid_res * self.grid_res, np.nan, dtype=np.float64)
            gathered = v_arr[self.vertices]
            out_flat[self.valid_q] = np.sum(self.weights * gathered, axis=1)
            return out_flat.reshape(self.grid_res, self.grid_res)

        if self.nn_indices is not None:
            return v_arr[self.nn_indices].reshape(self.grid_res, self.grid_res)

        return np.array([v_arr])


def build_map_gridder(
    coords: np.ndarray,
    grid_res: int = 80,
    active_mask: Optional[np.ndarray] = None,
) -> Optional[MapGridder]:
    """Precompute regular-grid indices or scattered Delaunay weights for ``coords``."""
    if coords is None or len(coords) == 0:
        return None
    c_arr = np.asarray(coords, dtype=np.float64)
    if c_arr.ndim != 2 or c_arr.shape[1] != 2:
        return None

    X = c_arr[:, 0]
    Y = c_arr[:, 1]
    u_x = np.sort(np.unique(X))
    u_y = np.sort(np.unique(Y))
    dx = float(u_x[1] - u_x[0]) if len(u_x) > 1 else 1.0
    dy = float(u_y[1] - u_y[0]) if len(u_y) > 1 else 1.0
    extent = [
        float(u_x[0] - 0.5 * dx),
        float(u_x[-1] + 0.5 * dx),
        float(u_y[0] - 0.5 * dy),
        float(u_y[-1] + 0.5 * dy),
    ]

    n_expected = len(u_x) * len(u_y)
    if n_expected <= len(X) * 4 and len(u_x) >= 1 and len(u_y) >= 1:
        x_idx = np.searchsorted(u_x, X)
        y_idx = np.searchsorted(u_y, Y)
        valid_idx = (x_idx < len(u_x)) & (y_idx < len(u_y))
        return MapGridder(
            n_points=len(c_arr),
            is_regular=True,
            extent=extent,
            grid_res=grid_res,
            n_x=len(u_x),
            n_y=len(u_y),
            x_idx=x_idx,
            y_idx=y_idx,
            valid_idx=valid_idx,
        )

    src_idx = np.flatnonzero(np.asarray(active_mask, dtype=bool)) if active_mask is not None else np.arange(len(c_arr))
    if len(src_idx) < 4:
        return None

    pts = c_arr[src_idx]
    gx, gy = np.meshgrid(
        np.linspace(u_x[0], u_x[-1], grid_res),
        np.linspace(u_y[0], u_y[-1], grid_res),
    )
    q_pts = np.column_stack((gx.ravel(), gy.ravel()))

    try:
        from scipy.spatial import Delaunay, QhullError  # deferred: startup import rule
        tri = Delaunay(pts)
        simplices = tri.find_simplex(q_pts)
        valid_q = simplices >= 0
        s_valid = simplices[valid_q]
        T = tri.transform[s_valid, :2, :]
        r = q_pts[valid_q] - tri.transform[s_valid, 2, :]
        b01 = np.einsum("qij,qj->qi", T, r)
        b2 = 1.0 - b01.sum(axis=1, keepdims=True)
        weights = np.hstack((b01, b2))
        vertices = src_idx[tri.simplices[s_valid]]
        return MapGridder(
            n_points=len(c_arr),
            is_regular=False,
            extent=extent,
            grid_res=grid_res,
            vertices=vertices,
            weights=weights,
            valid_q=valid_q,
        )
    except QhullError:
        try:
            from scipy.spatial import cKDTree  # deferred: startup import rule
            _, nn_local = cKDTree(pts).query(q_pts)
            return MapGridder(
                n_points=len(c_arr),
                is_regular=False,
                extent=extent,
                grid_res=grid_res,
                nn_indices=src_idx[nn_local],
            )
        except Exception:
            return None
    except Exception:
        return None


def compute_map_grid(
    coords: np.ndarray,
    values: np.ndarray,
    gridder: Optional[MapGridder] = None,
) -> Tuple[Optional[np.ndarray], list[float], float, float]:
    """Construct 2D interpolated or pixelated grid from coordinates and values.

    When ``gridder`` is supplied and matches ``len(coords)``, reuses its
    precomputed regular indices or scattered Delaunay barycentric weights.
    """
    if coords is None or len(coords) == 0 or values is None or len(values) == 0:
        return None, [0.0, 1.0, 0.0, 1.0], 0.0, 100.0

    c_arr = np.asarray(coords, dtype=np.float64)
    v_arr = np.asarray(values, dtype=np.float64)

    if c_arr.ndim != 2 or c_arr.shape[1] != 2 or len(c_arr) != len(v_arr):
        return None, [0.0, 1.0, 0.0, 1.0], 0.0, 100.0

    # Robust vmin and vmax (ignoring NaNs)
    valid_vals = v_arr[np.isfinite(v_arr)]
    if len(valid_vals) == 0:
        return None, [0.0, 1.0, 0.0, 1.0], 0.0, 1.0
    vmin = float(np.percentile(valid_vals, 1))
    vmax = float(np.percentile(valid_vals, 99))
    if vmin == vmax:
        vmin = float(valid_vals.min())
        vmax = float(valid_vals.max())
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0

    if gridder is not None and gridder.n_points == len(c_arr):
        return gridder.evaluate(v_arr), list(gridder.extent), vmin, vmax

    resolved = build_map_gridder(c_arr, active_mask=np.isfinite(v_arr))
    if resolved is not None:
        return resolved.evaluate(v_arr), list(resolved.extent), vmin, vmax

    X = c_arr[:, 0]
    Y = c_arr[:, 1]
    u_x = np.sort(np.unique(X))
    u_y = np.sort(np.unique(Y))
    dx = float(u_x[1] - u_x[0]) if len(u_x) > 1 else 1.0
    dy = float(u_y[1] - u_y[0]) if len(u_y) > 1 else 1.0
    extent = [
        float(u_x[0] - 0.5 * dx),
        float(u_x[-1] + 0.5 * dx),
        float(u_y[0] - 0.5 * dy),
        float(u_y[-1] + 0.5 * dy),
    ]
    return np.array([v_arr]), extent, vmin, vmax


@dataclass(frozen=True)
class MapGeometry:
    """Spatial dimensionality of a map, derived from its (X, Y) coordinates.

    Attributes
    ----------
    ndim : int
        ``2`` for spatially distributed 2D maps, ``1`` for 1D profiles
        (line scans), i.e. maps whose measurement sites are collinear.
    positions : np.ndarray or None
        For 1D profiles: position of every spectrum along the profile, shape (N,).
        ``None`` for 2D maps.
    order : np.ndarray or None
        For 1D profiles: indices sorting ``positions`` ascending (stable).
    axis_label : str
        Name of the profile axis (``"X"``, ``"Y"`` or ``"Distance"``); empty for 2D maps.
    """

    ndim: int
    positions: Optional[np.ndarray] = None
    order: Optional[np.ndarray] = None
    axis_label: str = ""

    @property
    def is_profile(self) -> bool:
        return self.ndim == 1


_GEOMETRY_2D = MapGeometry(ndim=2)


def analyze_map_geometry(
    coords: Optional[np.ndarray],
    collinear_tol: float = PROFILE_COLLINEAR_TOL,
) -> MapGeometry:
    """Detect whether a map is a 2D map or a 1D profile from its coordinates.

    A map is a 1D profile when it has at least two distinct measurement sites
    and all sites lie on a single straight line: either one coordinate is
    constant (axis-aligned line scan) or the centered coordinates have a
    negligible second singular value (diagonal line scan). Single-site maps
    and anything else are treated as 2D maps (existing heatmap behavior).

    Parameters
    ----------
    coords : np.ndarray
        Spatial positions of shape (N, 2).
    collinear_tol : float
        Maximum ratio ``s[1] / s[0]`` of the singular values of the centered
        coordinates for the sites to be considered collinear.
    """
    if coords is None:
        return _GEOMETRY_2D
    c = np.asarray(coords, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 2 or len(c) < 2 or not np.all(np.isfinite(c)):
        return _GEOMETRY_2D

    x, y = c[:, 0], c[:, 1]
    x_const = bool(np.ptp(x) == 0.0)
    y_const = bool(np.ptp(y) == 0.0)
    if x_const and y_const:
        return _GEOMETRY_2D  # single measurement site

    if y_const:
        positions, label = x.copy(), "X"
    elif x_const:
        positions, label = y.copy(), "Y"
    else:
        centered = c - c.mean(axis=0)
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        if s[0] <= 0.0 or s[1] > collinear_tol * s[0]:
            return _GEOMETRY_2D
        direction = vt[0]
        # Nearly axis-aligned lines (e.g. rounding jitter) keep the native axis
        if abs(direction[0]) >= 0.9999:
            positions, label = x.copy(), "X"
        elif abs(direction[1]) >= 0.9999:
            positions, label = y.copy(), "Y"
        else:
            proj = centered @ direction
            positions, label = proj - proj.min(), "Distance"

    order = np.argsort(positions, kind="stable")
    return MapGeometry(ndim=1, positions=positions, order=order, axis_label=label)


def compute_intensity_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    rel_eps: float = RATIO_REL_EPS,
) -> Tuple[np.ndarray, int]:
    """Element-wise ratio ``numerator / denominator`` with NaN masking.

    Points where either operand is non-finite, or where the denominator
    magnitude is at or below ``rel_eps * max(|denominator|)`` (zero or
    near-zero intensity), are set to NaN instead of producing ``inf`` or
    noise-dominated values.

    Returns
    -------
    ratio : np.ndarray
        Ratio array (float64) of the broadcast shape; undefined points are NaN.
    n_masked : int
        Number of points set to NaN.
    """
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    num, den = np.broadcast_arrays(num, den)

    abs_den = np.abs(den)
    finite = np.isfinite(num) & np.isfinite(den)
    max_den = float(abs_den[finite].max()) if np.any(finite) else 0.0
    threshold = rel_eps * max_den
    valid = finite & (abs_den > threshold) & (abs_den > 0.0)

    ratio = np.full(num.shape, np.nan, dtype=np.float64)
    np.divide(num, den, out=ratio, where=valid)
    return ratio, int(ratio.size - np.count_nonzero(valid))
