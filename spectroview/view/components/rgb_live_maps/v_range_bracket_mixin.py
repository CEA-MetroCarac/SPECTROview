"""Range-bracket UI wiring and blitted dragging mixin for VSpectraViewer."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from spectroview.model.spectra.spectral_parameters import (
    SELECTOR_BLUE,
    SELECTOR_GREEN,
    SELECTOR_RED,
)
from spectroview.view.components.rgb_live_maps.v_range_selector import (
    RangeSelectorManager,
    VRangeSelectorPanel,
)


class SpectraRangeBracketMixin:
    """Mixin encapsulating RGB range-bracket UI wiring and blitted dragging for VSpectraViewer."""

    def _init_range_brackets(self) -> None:
        """Initialize the RangeSelectorManager and drag/blitting state on the host viewer."""
        self.range_manager = RangeSelectorManager(self)
        self.range_manager.range_toggled.connect(self.range_toggled.emit)
        self.range_manager.range_changed.connect(self.range_changed.emit)
        self._range_drag_cids: List[int] = []
        self._range_hover_active: bool = False
        self._ranges_initialized: bool = False
        self._range_drag_bg = None
        self._range_drag_bg_size = None
        self._suspended_range_layout = None

    def _create_range_selector_panel(self, parent_panel: QWidget) -> VRangeSelectorPanel:
        """Create the vertical R/B/G toggle panel and wire shortcut button references."""
        self.range_panel = VRangeSelectorPanel(parent_panel)
        self.range_panel.selector_toggled.connect(self._toggle_range_selector)
        self.btn_range_red = self.range_panel.buttons[SELECTOR_RED]
        self.btn_range_blue = self.range_panel.buttons[SELECTOR_BLUE]
        self.btn_range_green = self.range_panel.buttons[SELECTOR_GREEN]
        return self.range_panel

    def _sync_ranges_with_plot_data(self, x_arr) -> None:
        """Initialize or clamp range selector bounds when spectral plot data updates."""
        if x_arr is None:
            return
        first_x = x_arr[0] if isinstance(x_arr, list) and len(x_arr) > 0 else x_arr
        if hasattr(first_x, "__len__") and len(first_x) > 1:
            if not getattr(self, "_ranges_initialized", False):
                self.range_manager.initialize_ranges_from_axis(first_x)
                self._ranges_initialized = True
            else:
                self.range_manager.refresh_or_clamp_ranges(
                    float(np.min(first_x)), float(np.max(first_x))
                )

    def _handle_range_mouse_press(self, event) -> bool:
        """Start dragging a visible range bracket if clicked; return True when handled."""
        if not hasattr(self, "range_manager") or not self.range_manager.has_visible_brackets():
            return False
        hit = self.range_manager.hit_test(self.ax, event, zoom_active=self.zoom_pan_active)
        if not hit or event.button != 1:
            return False

        # Suppress concurrent Matplotlib zoom interaction if active
        if hasattr(self.toolbar, "_zoom_info") and self.toolbar._zoom_info is not None:
            try:
                if hasattr(self.toolbar._zoom_info, "cid"):
                    self.canvas.mpl_disconnect(self.toolbar._zoom_info.cid)
                self.toolbar.remove_rubberband()
                self.toolbar._zoom_info = None
            except Exception:
                pass

        color_id, target = hit
        self.range_manager.start_drag(color_id, target, event.xdata)
        self._prepare_range_drag_blit()
        self.canvas.setCursor(Qt.ClosedHandCursor if target == "body" else Qt.SizeHorCursor)
        cid_m = self.canvas.mpl_connect("motion_notify_event", self._on_range_drag_motion)
        cid_r = self.canvas.mpl_connect("button_release_event", self._on_range_drag_release)
        self._range_drag_cids = [cid_m, cid_r]
        return True

    def _handle_range_hover(self, event) -> bool:
        """Update cursor feedback when hovering over a visible range bracket."""
        if not hasattr(self, "range_manager"):
            return False
        if self.range_manager.has_visible_brackets():
            hit = self.range_manager.hit_test(self.ax, event, zoom_active=self.zoom_pan_active)
            if hit:
                if hit[1] in ("min", "max"):
                    self.canvas.setCursor(Qt.SizeHorCursor)
                    self._range_hover_active = True
                    return True
                if hit[1] == "body":
                    self.canvas.setCursor(Qt.OpenHandCursor)
                    self._range_hover_active = True
                    return True
        if getattr(self, "_range_hover_active", False):
            self.canvas.unsetCursor()
            self._range_hover_active = False
        return False

    def set_range_selectors_visible(self, visible: bool):
        """Show or hide the spectral range selector panel on the right toolbar.

        When hidden, any active selectors are cleanly deactivated.
        """
        self.range_panel.setVisible(visible)
        if not visible:
            self.deactivate_all_range_selectors()

    def deactivate_all_range_selectors(self):
        """Deactivate all range selectors and clear their visual markers."""
        if self._range_drag_bg is not None or self._suspended_range_layout is not None:
            self._end_range_drag_blit()
        for color_id, btn in self.range_panel.buttons.items():
            if btn.isChecked():
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
                self.range_manager.set_active(color_id, False)
        if self.ax is not None:
            self.range_manager.render(self.ax)
            self.canvas.draw_idle()

    def set_range_selector_parameter(self, color_id: str, parameter: str) -> None:
        """Update the Live Maps parameter for a selector and refresh bracket visibility."""
        if not hasattr(self, "range_manager"):
            return
        if self._range_drag_bg is not None or self._suspended_range_layout is not None:
            self._end_range_drag_blit()
        was_dragging = self.range_manager.is_dragging()
        self.range_manager.set_parameter(color_id, parameter)
        if was_dragging and not self.range_manager.is_dragging():
            for cid in getattr(self, "_range_drag_cids", []):
                try:
                    self.canvas.mpl_disconnect(cid)
                except Exception:
                    pass
            if hasattr(self, "_range_drag_cids"):
                self._range_drag_cids.clear()
            if self.canvas is not None:
                self.canvas.unsetCursor()
            self._range_hover_active = False
        elif getattr(self, "_range_hover_active", False) and not self.range_manager.has_visible_brackets():
            if self.canvas is not None:
                self.canvas.unsetCursor()
            self._range_hover_active = False
        if self.ax is not None:
            self.range_manager.render(self.ax)
            if self.canvas is not None:
                self.canvas.draw_idle()

    def _get_current_data_xrange(self) -> Tuple[Optional[float], Optional[float]]:
        """Return (min_x, max_x) of currently loaded spectra data."""
        if self._tensor_data:
            x_arr = self._tensor_data.get("x")
            if x_arr is not None:
                if isinstance(x_arr, list) and len(x_arr) > 0:
                    mins = [float(np.min(xa)) for xa in x_arr if hasattr(xa, "__len__") and len(xa) > 1]
                    maxs = [float(np.max(xa)) for xa in x_arr if hasattr(xa, "__len__") and len(xa) > 1]
                    if mins and maxs:
                        return min(mins), max(maxs)
                elif hasattr(x_arr, "__len__") and len(x_arr) > 1:
                    return float(np.min(x_arr)), float(np.max(x_arr))
        if self._current_spectra:
            x_vals = []
            for s in self._current_spectra:
                if hasattr(s, "x") and s.x is not None and len(s.x) > 0:
                    x_vals.extend([float(np.min(s.x)), float(np.max(s.x))])
            if x_vals:
                return min(x_vals), max(x_vals)
        if self.ax is not None:
            xlim = self.ax.get_xlim()
            if xlim[0] < xlim[1]:
                return float(xlim[0]), float(xlim[1])
        return None, None

    def _toggle_range_selector(self, color_id: str, checked: bool):
        """Activate or deactivate one of the spectral range selectors."""
        if self._range_drag_bg is not None or self._suspended_range_layout is not None:
            self._end_range_drag_blit()
        if checked and self.btn_zoom.isChecked():
            self.btn_zoom.blockSignals(True)
            self.btn_zoom.setChecked(False)
            self.btn_zoom.blockSignals(False)
            self._set_zoom_tool_active(False)

        if checked:
            curr_xmin, curr_xmax = self._get_current_data_xrange()
            if curr_xmin is not None and curr_xmax is not None:
                self.range_manager.refresh_or_clamp_ranges(curr_xmin, curr_xmax, target_color_id=color_id)

        self.range_manager.set_active(color_id, checked)
        if self.ax is not None:
            self.range_manager.render(self.ax)
            self.canvas.draw_idle()

    def _prepare_range_drag_blit(self) -> bool:
        """Capture clean static canvas background once before blitted bracket dragging."""
        self._end_range_drag_blit()
        if self.ax is None or self.canvas is None:
            return False
        if not (
            hasattr(self.canvas, "copy_from_bbox")
            and hasattr(self.canvas, "restore_region")
            and hasattr(self.canvas, "blit")
        ):
            return False
        try:
            w = int(self.canvas.width())
            h = int(self.canvas.height())
            if w <= 0 or h <= 0:
                return False

            # Ensure selector artists exist, then mark them animated before capturing
            # the background so the initial bracket position isn't baked into the buffer.
            if not self.range_manager.active_artists():
                self.range_manager.render(self.ax)
            self.range_manager.set_animated(True)
            self.canvas.draw()

            # Suspend constrained/compressed layout during drag in case any fallback redraw runs.
            layout_engine = self.figure.get_layout_engine()
            if layout_engine is not None:
                self._suspended_range_layout = layout_engine
                self.figure.set_layout_engine(None)

            self._range_drag_bg = self.canvas.copy_from_bbox(self.figure.bbox)
            self._range_drag_bg_size = (w, h)
            # Re-paint selector artists over the freshly captured clean buffer.
            self._blit_range_selectors()
            return True
        except Exception:
            self._end_range_drag_blit()
            return False

    def _blit_range_selectors(self) -> bool:
        """Redraw only range-selector artists on top of the cached background."""
        if self.ax is None or self.canvas is None or self._range_drag_bg is None:
            return False
        try:
            cur_size = (int(self.canvas.width()), int(self.canvas.height()))
            if cur_size != self._range_drag_bg_size:
                if not self._prepare_range_drag_blit():
                    return False
            self.canvas.restore_region(self._range_drag_bg)
            for art in self.range_manager.active_artists():
                self.ax.draw_artist(art)
            self.canvas.blit(self.figure.bbox)
            return True
        except Exception:
            self._end_range_drag_blit()
            return False

    def _end_range_drag_blit(self) -> None:
        """Restore layout engine and normal non-animated drawing after range drag."""
        if getattr(self, "_suspended_range_layout", None) is not None:
            try:
                if self.figure.get_layout_engine() is None:
                    self.figure.set_layout_engine(self._suspended_range_layout)
            except Exception:
                pass
            self._suspended_range_layout = None
        self._range_drag_bg = None
        self._range_drag_bg_size = None
        if hasattr(self, "range_manager"):
            self.range_manager.set_animated(False)

    def _event_to_clamped_xdata(self, event) -> Optional[float]:
        """Extract or invert event X coordinate clamped to current axes X limits."""
        if self.ax is None:
            return None
        xdata = event.xdata if event.inaxes == self.ax and event.xdata is not None else None
        if xdata is None:
            if event.x is None or event.y is None:
                return None
            try:
                xdata = float(self.ax.transData.inverted().transform((event.x, event.y))[0])
            except Exception:
                return None
        xlim = self.ax.get_xlim()
        lo, hi = min(float(xlim[0]), float(xlim[1])), max(float(xlim[0]), float(xlim[1]))
        return float(np.clip(xdata, lo, hi))

    def _on_range_drag_motion(self, event):
        """Update range selector boundaries while user drags."""
        if not hasattr(self, "range_manager") or not self.range_manager.is_dragging():
            return
        xdata = self._event_to_clamped_xdata(event)
        if xdata is None:
            return
        color_id = self.range_manager.update_drag(xdata)
        if color_id and self.ax is not None:
            self.range_manager.render(self.ax)
            if not self._blit_range_selectors():
                self.canvas.draw_idle()

    def _on_range_drag_release(self, event):
        """Finalize range selector dragging on mouse button release."""
        if hasattr(self, "_range_drag_cids"):
            for cid in self._range_drag_cids:
                try:
                    self.canvas.mpl_disconnect(cid)
                except Exception:
                    pass
            self._range_drag_cids.clear()

        # Suppress any lingering Matplotlib zoom interaction
        if hasattr(self.toolbar, "_zoom_info") and self.toolbar._zoom_info is not None:
            try:
                if hasattr(self.toolbar._zoom_info, "cid"):
                    self.canvas.mpl_disconnect(self.toolbar._zoom_info.cid)
                self.toolbar.remove_rubberband()
                self.toolbar._zoom_info = None
            except Exception:
                pass

        if hasattr(self, "range_manager") and self.range_manager.is_dragging():
            color_id = self.range_manager.finish_drag()
            self._end_range_drag_blit()
            if self.ax is not None:
                self.range_manager.render(self.ax)
                self.canvas.draw_idle()
            self.canvas.unsetCursor()
            self._range_hover_active = False
            if color_id:
                self.range_drag_finished.emit(color_id)
        elif self._range_drag_bg is not None or self._suspended_range_layout is not None:
            self._end_range_drag_blit()
