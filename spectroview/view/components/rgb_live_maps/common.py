"""Shared constants, theme helpers, and normalization utilities for RGB Live Maps."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import matplotlib as mpl
import numpy as np

from spectroview import PLOT_POLICY_DARK, PLOT_POLICY_LIGHT
from spectroview.model.spectral_parameters import (
    SELECTOR_BLUE,
    SELECTOR_GREEN,
    SELECTOR_RED,
)
from spectroview.settings.m_settings import LIVE_MAPS_VIEW_OPTION_DEFAULTS


DEFAULT_PALETTES = {
    SELECTOR_RED: "Reds",
    SELECTOR_BLUE: "Blues",
    SELECTOR_GREEN: "Greens",
}

AVAILABLE_PALETTES = [
    "viridis", "plasma", "inferno", "magma", "cividis", "turbo",
    "Reds", "Blues", "Greens", "coolwarm", "bwr", "jet", "rainbow",
]

_DEFAULT_VIEW_OPTIONS = dict(LIVE_MAPS_VIEW_OPTION_DEFAULTS)


def _normalize_plot_theme(theme: Optional[str]) -> str:
    """Normalize any theme identifier to 'Light Mode' or 'Dark Mode'."""
    return "Dark Mode" if "dark" in str(theme or "").lower() else "Light Mode"


def _resolve_style_path(theme: Optional[str]) -> Tuple[bool, str]:
    """Return ``(is_dark, mplstyle_path)`` for a plot/app theme string."""
    is_dark = _normalize_plot_theme(theme) == "Dark Mode"
    return is_dark, (PLOT_POLICY_DARK if is_dark else PLOT_POLICY_LIGHT)


@lru_cache(maxsize=8)
def _get_cached_style_dict(style_path: str) -> dict:
    """Cache parsed mplstyle dictionaries to avoid repeated disk I/O and global rcParams mutations."""
    return dict(mpl.rc_params_from_file(style_path))


def _copy_view_options(src: Optional[dict] = None) -> dict:
    """Return an isolated copy of Live Maps view options with cloned dicts."""
    base = dict(_DEFAULT_VIEW_OPTIONS)
    if src:
        base.update(src)
    base["plot_theme"] = _normalize_plot_theme(base.get("plot_theme"))
    base["copy_fig_theme"] = _normalize_plot_theme(base.get("copy_fig_theme"))
    base["copy_fig_width"] = str(base.get("copy_fig_width") or "5.5").strip() or "5.5"
    base["copy_fig_height"] = str(base.get("copy_fig_height") or "4.0").strip() or "4.0"
    base["show_figure_label"] = bool(base.get("show_figure_label", False))
    base["custom_labels"] = dict(base.get("custom_labels") or {})
    base["custom_colors"] = dict(base.get("custom_colors") or {})
    return base


def _parse_norm_bound(raw: str) -> Optional[float]:
    """Parse an optional numeric bound from Xmin/Xmax input, ignoring invalid text."""
    text = str(raw or "").strip().replace(",", ".")
    if not text:
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    return val if np.isfinite(val) else None


def _normalize_1d_values(
    x: np.ndarray,
    y: np.ndarray,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
) -> np.ndarray:
    """Normalize a 1D curve to max height overall or within [xmin, xmax] (view-only)."""
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.size == 0:
        return y_arr
    x_arr = np.asarray(x, dtype=np.float64)

    sub_y = y_arr
    if (xmin is not None or xmax is not None) and x_arr.size == y_arr.size:
        valid_xy = np.isfinite(x_arr) & np.isfinite(y_arr)
        lo, hi = xmin, xmax
        if lo is not None and hi is not None and lo > hi:
            lo, hi = hi, lo
        mask = valid_xy.copy()
        if lo is not None:
            mask &= x_arr >= lo
        if hi is not None:
            mask &= x_arr <= hi
        if np.any(mask):
            sub_y = y_arr[mask]
        elif lo is not None and hi is not None and np.any(valid_xy):
            valid_x = x_arr[valid_xy]
            x_min_val, x_max_val = float(np.min(valid_x)), float(np.max(valid_x))
            if hi >= x_min_val and lo <= x_max_val:
                # Narrow [xmin, xmax] inside the curve span without direct samples: use nearest point
                mid = 0.5 * (lo + hi)
                valid_idx = np.flatnonzero(valid_xy)
                nearest = valid_idx[int(np.argmin(np.abs(x_arr[valid_idx] - mid)))]
                sub_y = y_arr[nearest : nearest + 1]

    finite_sub = sub_y[np.isfinite(sub_y)]
    if finite_sub.size == 0:
        return y_arr
    norm_val = float(np.max(finite_sub))
    if norm_val == 0.0 or not np.isfinite(norm_val):
        return y_arr
    return y_arr / norm_val

