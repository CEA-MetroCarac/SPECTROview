"""Shared spatial geometry, selection resolution, and Matplotlib overlay helpers.

Provides unified click, Ctrl+Click, and rectangular drag selection plus red
highlight overlays for both ``VMapViewer`` (2D/wafer heatmaps) and
``LiveMapSubpanel`` (RGB Live Maps 2D heatmaps and 1D spatial profiles).
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import numpy as np
from matplotlib import patches
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication


Point2D = Tuple[float, float]


def is_ctrl_event(event) -> bool:
    """Return True if Control modifier is active via Qt state or Matplotlib event."""
    if bool(QApplication.keyboardModifiers() & Qt.ControlModifier):
        return True
    if event is not None:
        key = getattr(event, "key", None)
        mods = getattr(event, "modifiers", None) or ()
        if key in ("control", "ctrl") or "ctrl" in mods or "control" in mods:
            return True
    return False


def get_drag_box_edgecolor(is_dark: bool) -> str:
    """Return high-contrast dashed drag-box border color for the plot theme."""
    return "#FFFFFF" if is_dark else "#111111"


def compute_2d_spatial_geometry(x_col, y_col) -> dict:
    """Compute 2D spatial grid regularity, pixel size (dx, dy), and extent."""
    x_arr = np.asarray(x_col, dtype=float)
    y_arr = np.asarray(y_col, dtype=float)
    if x_arr.size > 0 and y_arr.size == x_arr.size:
        finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_arr = x_arr[finite_mask]
        y_arr = y_arr[finite_mask]

    if len(x_arr) == 0 or len(x_arr) != len(y_arr):
        return {
            "is_regular": False,
            "dx": 1.0,
            "dy": 1.0,
            "extent": [0.0, 0.0, 0.0, 0.0],
            "has_duplicates": False,
        }

    u_x = np.unique(x_arr)
    u_y = np.unique(y_arr)
    pts = np.column_stack([x_arr, y_arr])
    unique_pts = np.unique(pts, axis=0)
    has_duplicates = len(unique_pts) < len(x_arr)

    is_regular = False
    step_x = 1.0
    step_y = 1.0
    if len(u_x) > 1 and len(u_y) > 1:
        diffs_x = np.diff(u_x)
        diffs_y = np.diff(u_y)
        step_x = float((u_x[-1] - u_x[0]) / (len(u_x) - 1))
        step_y = float((u_y[-1] - u_y[0]) / (len(u_y) - 1))
        if step_x > 0 and step_y > 0:
            is_uniform_x = np.allclose(diffs_x, step_x, rtol=0.01, atol=1e-5)
            is_uniform_y = np.allclose(diffs_y, step_y, rtol=0.01, atol=1e-5)
            if is_uniform_x and is_uniform_y and len(unique_pts) >= 0.8 * (len(u_x) * len(u_y)):
                is_regular = True

    if is_regular:
        dx = step_x
        dy = step_y
        extent = [
            float(u_x[0] - 0.5 * dx),
            float(u_x[-1] + 0.5 * dx),
            float(u_y[0] - 0.5 * dy),
            float(u_y[-1] + 0.5 * dy),
        ]
    else:
        x_min, x_max = float(u_x[0]), float(u_x[-1])
        y_min, y_max = float(u_y[0]), float(u_y[-1])
        x_span = x_max - x_min
        y_span = y_max - y_min

        if len(unique_pts) > 1:
            if len(unique_pts) <= 500:
                diff = unique_pts[:, np.newaxis, :] - unique_pts[np.newaxis, :, :]
                dists_sq = np.sum(diff ** 2, axis=-1)
                np.fill_diagonal(dists_sq, np.inf)
                min_dist = float(np.sqrt(dists_sq.min()))
            else:
                from scipy.spatial import cKDTree  # deferred off startup import graph

                tree = cKDTree(unique_pts)
                dists, _ = tree.query(unique_pts, k=2)
                min_dist = float(dists[:, 1].min())
            span_max = max(x_span, y_span)
            pixel_size = min(min_dist * 0.35, span_max / 20.0) if span_max > 0 else min_dist * 0.35
            dx = dy = float(max(pixel_size, 1e-4))
        else:
            dx = dy = 1.0

        margin_x = max(dx * 1.5, x_span * 0.05 if x_span > 0 else 1.0)
        margin_y = max(dy * 1.5, y_span * 0.05 if y_span > 0 else 1.0)
        extent = [
            float(x_min - margin_x),
            float(x_max + margin_x),
            float(y_min - margin_y),
            float(y_max + margin_y),
        ]

    return {
        "is_regular": is_regular,
        "dx": float(dx),
        "dy": float(dy),
        "extent": extent,
        "has_duplicates": has_duplicates,
    }


def _apply_selection_update(
    current: Iterable[Point2D],
    picked: list[Point2D],
    *,
    is_rect: bool,
    ctrl_held: bool,
) -> list[Point2D]:
    """Apply Ctrl/non-Ctrl merge or toggle semantics to selected coordinates."""
    if is_rect:
        result = list(current) if ctrl_held else []
        seen = set(result)
        for pt in picked:
            if pt not in seen:
                seen.add(pt)
                result.append(pt)
        return result

    if not picked:
        return list(current)
    pt = picked[0]
    if ctrl_held:
        result = list(current)
        if pt in result:
            result.remove(pt)
        else:
            result.append(pt)
        return result
    return [pt]


def resolve_xy_selection(
    all_x,
    all_y,
    selected_points: Iterable[Point2D],
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    is_rect: bool,
    ctrl_held: bool,
) -> list[Point2D]:
    """Resolve click or drag-box selection against 2D spatial site arrays."""
    x_arr = np.asarray(all_x, dtype=float)
    y_arr = np.asarray(all_y, dtype=float)
    if x_arr.size == 0 or y_arr.size != x_arr.size:
        return list(selected_points)

    if is_rect:
        xmin, xmax = sorted((float(x0), float(x1)))
        ymin, ymax = sorted((float(y0), float(y1)))
        inside = (x_arr >= xmin) & (x_arr <= xmax) & (y_arr >= ymin) & (y_arr <= ymax)
        picked = [(float(x), float(y)) for x, y in zip(x_arr[inside], y_arr[inside])]
        return _apply_selection_update(selected_points, picked, is_rect=True, ctrl_held=ctrl_held)

    distances_sq = (x_arr - float(x1)) ** 2 + (y_arr - float(y1)) ** 2
    idx = int(np.argmin(distances_sq))
    pt = (float(x_arr[idx]), float(y_arr[idx]))
    return _apply_selection_update(selected_points, [pt], is_rect=False, ctrl_held=ctrl_held)


def resolve_1d_profile_selection(
    site_positions,
    site_values,
    site_coords,
    selected_points: Iterable[Point2D],
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    is_rect: bool,
    ctrl_held: bool,
) -> list[Point2D]:
    """Resolve click or drag-box selection on a 1D spatial profile back to (X, Y)."""
    pos = np.asarray(site_positions, dtype=float)
    coords = np.asarray(site_coords, dtype=float)
    if pos.size == 0 or coords.ndim != 2 or len(coords) != len(pos) or coords.shape[1] < 2:
        return list(selected_points)

    valid = np.isfinite(pos) & np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
    vals = None
    if site_values is not None and len(site_values) == len(pos):
        candidate_vals = np.asarray(site_values, dtype=float)
        val_finite = np.isfinite(candidate_vals)
        if np.any(valid & val_finite):
            valid = valid & val_finite
            vals = candidate_vals

    if not np.any(valid):
        return list(selected_points)

    pos_v = pos[valid]
    coords_v = coords[valid]
    vals_v = vals[valid] if vals is not None else None

    if is_rect:
        xmin, xmax = sorted((float(x0), float(x1)))
        ymin, ymax = sorted((float(y0), float(y1)))
        inside = (pos_v >= xmin) & (pos_v <= xmax)
        if vals_v is not None:
            inside = inside & (vals_v >= ymin) & (vals_v <= ymax)
        picked = [(float(coords_v[i, 0]), float(coords_v[i, 1])) for i in np.flatnonzero(inside)]
        return _apply_selection_update(selected_points, picked, is_rect=True, ctrl_held=ctrl_held)

    dx_arr = np.abs(pos_v - float(x1))
    min_dx = float(np.min(dx_arr))
    tol = 1e-9 * max(1.0, float(np.ptp(pos_v)))
    cand_idx = np.flatnonzero(dx_arr <= min_dx + tol)
    if len(cand_idx) > 1 and vals_v is not None:
        best_sub = int(np.argmin(np.abs(vals_v[cand_idx] - float(y1))))
        idx = int(cand_idx[best_sub])
    else:
        idx = int(cand_idx[0])
    pt = (float(coords_v[idx, 0]), float(coords_v[idx, 1]))
    return _apply_selection_update(selected_points, [pt], is_rect=False, ctrl_held=ctrl_held)


class MapCanvasSelector:
    """Reusable drag-rectangle interaction and red selection overlay helper."""

    def __init__(self, ax, canvas, drag_threshold_px: float = 4.0):
        self.ax = ax
        self.canvas = canvas
        self.drag_threshold_px = float(drag_threshold_px)
        self.rect_start: Optional[Point2D] = None
        self.start_px: Optional[Point2D] = None
        self.rect_patch: Optional[patches.Rectangle] = None
        self.selection_collection: Optional[PatchCollection] = None
        self.selection_markers: Optional[Line2D] = None

    def clear_drag_box(self) -> None:
        """Remove temporary drag selection rectangle patch if present."""
        if self.rect_patch is not None:
            try:
                self.rect_patch.remove()
            except Exception:
                pass
            self.rect_patch = None

    def clear_overlay(self) -> None:
        """Remove red 2D box collection and 1D selection markers."""
        if self.selection_collection is not None:
            try:
                self.selection_collection.remove()
            except Exception:
                pass
            self.selection_collection = None
        if self.selection_markers is not None:
            try:
                self.selection_markers.remove()
            except Exception:
                pass
            self.selection_markers = None

    def reset(self) -> None:
        """Reset all interaction state and overlay artists."""
        self.rect_start = None
        self.start_px = None
        self.clear_drag_box()
        self.clear_overlay()

    def _event_to_data_clamped(self, event) -> Optional[Point2D]:
        """Resolve an event position inside axes bounds, clamping edge drags."""
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            return (float(event.xdata), float(event.ydata))
        if event.x is None or event.y is None:
            return None
        try:
            x_d, y_d = self.ax.transData.inverted().transform((float(event.x), float(event.y)))
            x_lo, x_hi = sorted(self.ax.get_xlim())
            y_lo, y_hi = sorted(self.ax.get_ylim())
            x_c = max(x_lo, min(x_hi, float(x_d)))
            y_c = max(y_lo, min(y_hi, float(y_d)))
            if math.isfinite(x_c) and math.isfinite(y_c):
                return (x_c, y_c)
        except Exception:
            pass
        return None

    def begin_press(self, event) -> bool:
        """Record press start if event is left-click inside the host axes."""
        if event.inaxes != self.ax or event.button != 1:
            return False
        if event.xdata is None or event.ydata is None:
            return False
        self.rect_start = (float(event.xdata), float(event.ydata))
        if event.x is not None and event.y is not None:
            self.start_px = (float(event.x), float(event.y))
        else:
            self.start_px = None
        self.clear_drag_box()
        return True

    def update_move(self, event, edgecolor: str = "#111111") -> bool:
        """Create or resize dashed drag box when drag threshold is exceeded."""
        if self.rect_start is None:
            return False
        coords = self._event_to_data_clamped(event)
        if coords is None:
            return False
        if event.inaxes != self.ax and self.rect_patch is None:
            return False

        if (
            self.rect_patch is None
            and self.start_px is not None
            and event.x is not None
            and event.y is not None
            and self.drag_threshold_px > 0
        ):
            if math.hypot(event.x - self.start_px[0], event.y - self.start_px[1]) < self.drag_threshold_px:
                return False

        x0, y0 = self.rect_start
        x1, y1 = coords
        if self.rect_patch is None:
            self.rect_patch = patches.Rectangle(
                (min(x0, x1), min(y0, y1)),
                abs(x1 - x0),
                abs(y1 - y0),
                linewidth=1.2,
                edgecolor=edgecolor,
                facecolor="none",
                linestyle="--",
                zorder=11,
            )
            self.ax.add_patch(self.rect_patch)
        else:
            self.rect_patch.set_x(min(x0, x1))
            self.rect_patch.set_y(min(y0, y1))
            self.rect_patch.set_width(abs(x1 - x0))
            self.rect_patch.set_height(abs(y1 - y0))

        if self.canvas is not None:
            self.canvas.draw_idle()
        return True

    def finish_release(self, event) -> Optional[tuple[float, float, float, float, bool]]:
        """Complete a click or drag gesture and return ``(x0, y0, x1, y1, is_rect)``."""
        if self.rect_start is None:
            return None
        x0, y0 = self.rect_start
        was_rect = self.rect_patch is not None

        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            x1, y1 = float(event.xdata), float(event.ydata)
        elif was_rect:
            clamped = self._event_to_data_clamped(event)
            if clamped is None:
                self.clear_drag_box()
                self.rect_start = None
                self.start_px = None
                if self.canvas is not None:
                    self.canvas.draw_idle()
                return None
            x1, y1 = clamped
        else:
            self.clear_drag_box()
            self.rect_start = None
            self.start_px = None
            return None

        self.clear_drag_box()
        self.rect_start = None
        self.start_px = None
        return (x0, y0, x1, y1, was_rect)

    def draw_2d_overlay(
        self,
        selected_points: Iterable[Point2D],
        dx: float,
        dy: float,
        draw: bool = True,
    ) -> Optional[PatchCollection]:
        """Render red rectangular outlines around selected 2D measurement sites."""
        self.clear_overlay()
        pts = list(selected_points)
        if pts:
            safe_dx = float(dx) if (dx is not None and dx > 0) else 1.0
            safe_dy = float(dy) if (dy is not None and dy > 0) else 1.0
            rects = [
                patches.Rectangle((x - safe_dx / 2.0, y - safe_dy / 2.0), safe_dx, safe_dy)
                for x, y in pts
            ]
            collection = PatchCollection(
                rects,
                linewidth=1.2,
                edgecolor="red",
                facecolor="none",
                zorder=10,
            )
            self.ax.add_collection(collection)
            self.selection_collection = collection
        if draw and self.canvas is not None:
            self.canvas.draw_idle()
        return self.selection_collection

    def draw_1d_overlay(
        self,
        x_points: Iterable[float],
        y_points: Iterable[float],
        markersize: float = 6.5,
        draw: bool = True,
    ) -> Optional[Line2D]:
        """Render red circle markers at selected points on a 1D profile curve."""
        self.clear_overlay()
        xs = list(x_points)
        ys = list(y_points)
        if xs and len(xs) == len(ys):
            (line,) = self.ax.plot(
                xs,
                ys,
                linestyle="None",
                marker="o",
                markersize=float(markersize),
                markerfacecolor="none",
                markeredgecolor="red",
                markeredgewidth=1.6,
                zorder=10,
                label="_selection_overlay_",
            )
            self.selection_markers = line
        if draw and self.canvas is not None:
            self.canvas.draw_idle()
        return self.selection_markers
