# spectroview/core/spatial.py
"""
Spatial traversal order and neighbor parameter propagation for 2D maps.

In a hyperspectral 2D map, adjacent pixels typically have very similar
spectral characteristics. By fitting pixels in a spatial order and using
the optimized parameters from a previously fitted neighbor as the initial
guess for the next pixel, we can dramatically reduce optimizer iterations.
"""

import numpy as np
from scipy.spatial import KDTree


def build_traversal_order(coords, strategy="spiral"):
    """Build an optimized traversal order for fitting 2D map pixels.

    Args:
        coords: (N, 2) array of (X, Y) spatial coordinates
        strategy: Traversal strategy:
            - "spiral": Start from map center, spiral outward (best for maps)
            - "raster": Row-by-row scan (simple but still benefits from propagation)
            - "sequential": Original order (no spatial optimization)

    Returns:
        order: 1D array of indices in the optimized traversal order
    """
    n = len(coords)
    if n <= 1:
        return np.arange(n)

    coords = np.asarray(coords, dtype=np.float64)

    if strategy == "sequential":
        return np.arange(n)

    elif strategy == "raster":
        return _build_raster_order(coords)

    elif strategy == "spiral":
        return _build_spiral_order(coords)

    else:
        return np.arange(n)


def _build_raster_order(coords):
    """Row-by-row raster scan order (sorted by Y then X)."""
    # Sort by Y coordinate first, then X within each row
    order = np.lexsort((coords[:, 0], coords[:, 1]))
    return order


def _build_spiral_order(coords):
    """Spiral outward from the map center.

    Uses a greedy nearest-neighbor approach starting from the center pixel:
    at each step, visit the nearest unvisited pixel. This naturally produces
    a spiral-like traversal pattern for regular grids.
    """
    n = len(coords)

    # Find center of the map
    center = coords.mean(axis=0)

    # Find the pixel closest to the center
    distances_to_center = np.linalg.norm(coords - center, axis=1)
    start_idx = np.argmin(distances_to_center)

    # Build KDTree for fast neighbor queries
    tree = KDTree(coords)

    # Greedy nearest-neighbor traversal
    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=np.intp)
    order[0] = start_idx
    visited[start_idx] = True

    current = start_idx
    for step in range(1, n):
        # Query increasing number of neighbors until we find an unvisited one
        k = min(10, n)
        while True:
            dists, indices = tree.query(coords[current], k=k)
            # Handle both scalar and array returns
            if np.ndim(indices) == 0:
                indices = np.array([indices])

            for idx in indices:
                if not visited[idx]:
                    order[step] = idx
                    visited[idx] = True
                    current = idx
                    break
            else:
                # All k neighbors are visited — increase search radius
                k = min(k * 2, n)
                if k >= n:
                    # Fallback: find any unvisited pixel
                    remaining = np.where(~visited)[0]
                    if len(remaining) > 0:
                        order[step] = remaining[0]
                        visited[remaining[0]] = True
                        current = remaining[0]
                    break
                continue
            break

    return order


class NeighborPropagator:
    """Manages parameter propagation between spatially adjacent pixels.

    After fitting a pixel, its optimized parameters are cached. When fitting
    the next pixel, the best initial guess is retrieved from the nearest
    already-fitted neighbor.

    This typically reduces optimizer iterations by 3-5x for map data.
    """

    def __init__(self, coords, k_neighbors=4):
        """
        Args:
            coords: (N, 2) array of spatial coordinates
            k_neighbors: Number of neighbors to consider for initial guess
        """
        self.coords = np.asarray(coords, dtype=np.float64)
        self.k_neighbors = min(k_neighbors, len(coords))
        self.tree = KDTree(self.coords) if len(coords) > 1 else None

        # Cache: index → fitted free parameter vector
        self._cache = {}

    def store_result(self, index, p_free):
        """Store fitted parameters for a pixel.

        Args:
            index: Pixel index in the original array
            p_free: Fitted free parameter vector (numpy array)
        """
        self._cache[index] = p_free.copy()

    def get_initial_guess(self, index, default_p0):
        """Get the best initial guess for a pixel from its fitted neighbors.

        Args:
            index: Pixel index to fit
            default_p0: Fallback initial guess (from model definition)

        Returns:
            p0: Initial parameter vector (from nearest fitted neighbor, or default)
        """
        if not self._cache or self.tree is None:
            return default_p0

        # Find k nearest neighbors
        k = min(self.k_neighbors, len(self.coords))
        dists, indices = self.tree.query(self.coords[index], k=k)

        if np.ndim(indices) == 0:
            indices = [indices]
            dists = [dists]

        # Return params from the nearest fitted neighbor
        for idx in indices:
            if idx in self._cache:
                return self._cache[idx].copy()

        return default_p0

    def clear(self):
        """Clear the parameter cache."""
        self._cache.clear()
