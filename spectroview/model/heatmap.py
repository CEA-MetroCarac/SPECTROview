"""Pure-function 2D heatmap grid construction.

Extracted so both the GUI's map viewer (spectroview/view/components/
v_map_viewer.py) and the headless API (spectroview.api.workspace.MapsWorkspace)
can build the exact same grid from scattered (x, y, z) samples without
depending on Qt.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def get_wafer_radius(map_type: str) -> float:
    """Wafer radius in mm implied by a map_type string (e.g. 'wafer_300mm')."""
    if "300" in map_type:
        return 150.0
    if "200" in map_type:
        return 100.0
    if "100" in map_type:
        return 50.0
    return 150.0


def build_heatmap_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    map_type: str = "2Dmap",
    method: str = "linear",
    grid_size: int = 80,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 2D grid from scattered (x, y, z) samples.

    Args:
        x, y, z: 1D arrays of equal length — scattered sample coordinates and values.
        map_type: '2Dmap' uses a fast pivot on the regular grid implied by the
            unique x/y values (no interpolation, matches how raster/2D-map
            acquisitions are naturally laid out). Any other value ('wafer_300mm',
            'wafer_200mm', 'wafer_100mm', ...) is treated as a wafer map: scattered
            points are interpolated onto a `grid_size` x `grid_size` mesh spanning
            the wafer radius implied by `map_type` (see `get_wafer_radius`).
        method: interpolation method passed to `scipy.interpolate.griddata`
            for wafer maps (ignored for '2Dmap').
        grid_size: mesh resolution per axis for wafer maps (ignored for '2Dmap').

    Returns:
        (xi, yi, zi):
          - '2Dmap': xi, yi are 1D arrays of sorted unique coordinates; zi is a
            2D array of shape (len(yi), len(xi)) (NaN where no sample exists).
          - wafer: xi, yi are 1D arrays of length grid_size spanning [-r, r];
            zi is a 2D array of shape (grid_size, grid_size) (NaN outside the
            convex hull of the samples).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    if map_type == "2Dmap":
        df = pd.DataFrame({"X": x, "Y": y, "Z": z})
        pivot = df.pivot_table(index="Y", columns="X", values="Z", aggfunc="mean")
        xi = pivot.columns.to_numpy(dtype=np.float64)
        yi = pivot.index.to_numpy(dtype=np.float64)
        zi = pivot.to_numpy(dtype=np.float64)
        return xi, yi, zi

    r = get_wafer_radius(map_type)
    xi = np.linspace(-r, r, grid_size)
    yi = np.linspace(-r, r, grid_size)
    grid_x, grid_y = np.meshgrid(xi, yi)

    valid = ~np.isnan(z)
    if not valid.any():
        zi = np.full_like(grid_x, np.nan)
    else:
        zi = griddata((x[valid], y[valid]), z[valid], (grid_x, grid_y), method=method)

    return xi, yi, zi
