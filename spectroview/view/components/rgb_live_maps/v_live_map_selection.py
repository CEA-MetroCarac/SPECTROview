"""Interactive spatial site selection overlay & canvas mouse interaction mixin for LiveMapSubpanel."""

from __future__ import annotations

from typing import Optional

import numpy as np

from spectroview.view.components.map_viewer.map_selection import (
    compute_2d_spatial_geometry,
    get_drag_box_edgecolor,
    is_ctrl_event,
    resolve_1d_profile_selection,
    resolve_xy_selection,
)


class LiveMapSelectionMixin:
    """Spatial site overlay rendering and interactive click/drag selection on Live Maps."""

    def set_spatial_sites(self, sites: Optional[dict]):
        """Store current map spatial site metadata and refresh selection overlay."""
        if not sites:
            self._spatial_sites = None
            if self._selector is not None:
                self._selector.clear_drag_box()
                self._selector.clear_overlay()
                if self.canvas is not None:
                    self.canvas.draw_idle()
            return

        coords = np.asarray(sites.get("coords", []), dtype=np.float64)
        if coords.ndim != 2 or len(coords) == 0 or coords.shape[1] < 2:
            self._spatial_sites = None
            return

        is_profile = bool(sites.get("is_profile", False))
        raw_active = sites.get("active_mask")
        if raw_active is not None and len(raw_active) == len(coords):
            active_mask = np.asarray(raw_active, dtype=bool)
        else:
            active_mask = np.ones(len(coords), dtype=bool)

        positions = sites.get("positions")
        pos_arr = (
            np.asarray(positions, dtype=np.float64)
            if (is_profile and positions is not None and len(positions) == len(coords))
            else None
        )

        active_site_keys: set[tuple[float, float]] = set()
        site_index: dict[tuple[float, float], int] = {}
        for idx in range(len(coords)):
            key = (round(float(coords[idx, 0]), 4), round(float(coords[idx, 1]), 4))
            if bool(active_mask[idx]):
                active_site_keys.add(key)
                site_index.setdefault(key, idx)

        cell_dx, cell_dy = (1.0, 1.0)
        if not is_profile:
            geom_2d = compute_2d_spatial_geometry(coords[:, 0], coords[:, 1])
            cell_dx, cell_dy = float(geom_2d["dx"]), float(geom_2d["dy"])

        self._spatial_sites = {
            "map_name": sites.get("map_name"),
            "is_profile": is_profile,
            "coords": coords[:, :2],
            "positions": pos_arr,
            "active_mask": active_mask,
            "active_site_keys": active_site_keys,
            "site_index": site_index,
            "cell_dx": cell_dx,
            "cell_dy": cell_dy,
        }
        if self._display_kind is not None:
            self._update_selection_overlay(draw=True)

    def set_selected_points(self, points: list):
        """Update highlighted (X, Y) spectrum selection without emitting feedback signal."""
        self._selected_points = list(
            dict.fromkeys((float(pt[0]), float(pt[1])) for pt in (points or []))
        )
        self._update_selection_overlay(draw=True)

    def _is_event_on_legend(self, event) -> bool:
        """Return True if mouse event falls inside the visible legend box."""
        if self._legend is None or not self._legend.get_visible():
            return False
        if event.x is None or event.y is None or self.canvas is None:
            return False
        try:
            renderer = self.canvas.get_renderer()
            if renderer is None:
                return False
            return bool(self._legend.get_window_extent(renderer=renderer).contains(event.x, event.y))
        except Exception:
            return False

    def _is_selection_interactive(self) -> bool:
        """Allow interactive XY click/drag selection only in Map mode when nav tools are idle."""
        if self.mode != "map" or self._spatial_sites is None or self._selector is None:
            return False
        if bool(getattr(self.toolbar, "mode", "")) or self.btn_pan.isChecked() or self.btn_zoom.isChecked():
            return False
        if self._display_kind == "map":
            return self.img is not None and not self._spatial_sites.get("is_profile", False)
        if self._display_kind in ("series", "multi_profile"):
            return bool(self.lines) and bool(self._spatial_sites.get("is_profile", False))
        return False

    def _get_primary_profile_values(self) -> Optional[np.ndarray]:
        """Return plotted Y values corresponding to the primary 1D spatial curve."""
        if self._display_kind == "series":
            vals = self._series_values
            if vals is None:
                return None
            x_arr = (
                self._series_x
                if self._series_x is not None
                else np.arange(1, len(vals) + 1, dtype=np.float64)
            )
            return self._apply_normalization(x_arr, vals)
        if self._display_kind == "multi_profile" and self._multi_entries:
            target = next(
                (entry for entry in self._multi_entries if entry.get("is_primary")),
                self._multi_entries[0],
            )
            px = np.asarray(target.get("positions", []), dtype=np.float64)
            vy = np.asarray(target.get("values", []), dtype=np.float64)
            return self._apply_normalization(px, vy)
        return None

    def _filter_active_selected_points(self) -> list[tuple[float, float]]:
        """Filter stored selected points to active sites on the current spatial map."""
        if not self._spatial_sites:
            return list(self._selected_points)
        active_keys = self._spatial_sites.get("active_site_keys")
        if active_keys is None:
            return list(self._selected_points)
        return [
            pt for pt in self._selected_points
            if (round(float(pt[0]), 4), round(float(pt[1]), 4)) in active_keys
        ]

    def _update_selection_overlay(self, draw: bool = False):
        """Draw red selection boxes (2D) or markers (1D profile) matching active map sites."""
        if self._selector is None:
            return
        self._selector.ax = self.ax
        if (
            self.mode != "map"
            or not self._selected_points
            or self._spatial_sites is None
            or self._display_kind not in ("map", "series", "multi_profile")
        ):
            self._selector.clear_overlay()
            if draw and self.canvas is not None:
                self.canvas.draw_idle()
            return

        if self._display_kind == "map" and self.img is not None and not self._spatial_sites.get("is_profile", False):
            pts = self._filter_active_selected_points()
            dx = self._spatial_sites.get("cell_dx", 1.0)
            dy = self._spatial_sites.get("cell_dy", 1.0)
            self._selector.draw_2d_overlay(pts, dx, dy, draw=draw)
            return

        if self._display_kind in ("series", "multi_profile") and self.lines and self._spatial_sites.get("is_profile", False):
            positions = self._spatial_sites.get("positions")
            site_index = self._spatial_sites.get("site_index")
            active_mask = self._spatial_sites.get("active_mask")
            vals = self._get_primary_profile_values()
            if positions is None or site_index is None or vals is None or len(vals) != len(positions):
                self._selector.clear_overlay()
                if draw and self.canvas is not None:
                    self.canvas.draw_idle()
                return
            xs, ys = [], []
            for x, y in self._selected_points:
                idx = site_index.get((round(float(x), 4), round(float(y), 4)))
                if idx is None:
                    continue
                if active_mask is not None and not bool(active_mask[idx]):
                    continue
                px = float(positions[idx])
                py = float(vals[idx])
                if np.isfinite(px) and np.isfinite(py):
                    xs.append(px)
                    ys.append(py)
            markersize = max(5.5, float(self._view_options.get("dot_size", 1.5)) + 4.5)
            self._selector.draw_1d_overlay(xs, ys, markersize=markersize, draw=draw)
            return

        self._selector.clear_overlay()
        if draw and self.canvas is not None:
            self.canvas.draw_idle()

    def _on_canvas_button_press(self, event):
        """Handle legend double-click editing and interactive map spectrum click start."""
        if getattr(event, "dblclick", False) and self._is_event_on_legend(event):
            self.legend_edit_requested.emit()
            return
        if not self._is_selection_interactive():
            return
        if self._is_event_on_legend(event):
            return
        self._selector.ax = self.ax
        self._selector.begin_press(event)

    def _on_canvas_motion(self, event):
        """Update rectangular selection marquee while dragging on the plot."""
        if not self._is_selection_interactive():
            return
        edgecolor = get_drag_box_edgecolor(self._current_theme == "Dark Mode")
        self._selector.ax = self.ax
        self._selector.update_move(event, edgecolor=edgecolor)

    def _on_canvas_release(self, event):
        """Finish click/drag selection and emit chosen (X, Y) spectrum coordinates."""
        if self._selector is None:
            return
        self._selector.ax = self.ax
        gesture = self._selector.finish_release(event)
        if gesture is None or not self._is_selection_interactive():
            return

        x0, y0, x1, y1, is_rect = gesture
        ctrl_held = is_ctrl_event(event)
        sites = self._spatial_sites
        coords = sites["coords"]
        active = sites["active_mask"]

        if self._display_kind == "map":
            act_coords = coords[active]
            new_points = resolve_xy_selection(
                act_coords[:, 0],
                act_coords[:, 1],
                self._selected_points,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                is_rect=is_rect,
                ctrl_held=ctrl_held,
            )
        else:
            positions = sites.get("positions")
            if positions is None or len(positions) != len(coords):
                return
            primary_vals = self._get_primary_profile_values()
            act_pos = positions[active]
            act_coords = coords[active]
            act_vals = (
                primary_vals[active]
                if (primary_vals is not None and len(primary_vals) == len(coords))
                else None
            )
            new_points = resolve_1d_profile_selection(
                act_pos,
                act_vals,
                act_coords,
                self._selected_points,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                is_rect=is_rect,
                ctrl_held=ctrl_held,
            )

        self._selected_points = new_points
        self._update_selection_overlay(draw=True)
        self.spectra_selected.emit(list(self._selected_points))
