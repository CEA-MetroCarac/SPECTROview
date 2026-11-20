"""Canvas mouse interactions, live coordinates, and figure copy helpers for MapViewer."""

from __future__ import annotations

from PySide6.QtCore import QTimer

from spectroview.i18n import combo_data_or_text
from spectroview.view.components.map_viewer.map_selection import (
    get_drag_box_edgecolor,
    is_ctrl_event,
    resolve_xy_selection,
)
from spectroview.view.theme import Icons
from spectroview.viewmodel.common.export_utils import copy_fig_to_clb


class MapInteractionMixin:
    """Mouse selection handlers, live coordinate tracking, and clipboard copy feedback."""

    def _on_mouse_click(self, event):
        """Start selection on mouse click."""
        if self.toolbar.mode or self._selector is None:
            return
        self._selector.ax = self.ax
        self._selector.begin_press(event)

    def _update_coordinates(self, event):
        """Update built-in live coordinate display label under the map canvas.

        Uses toolbar._mouse_event_to_message when available so that artist cursor
        data (such as the Z value on 2D heatmaps and wafer maps) is formatted and
        preserved along with the spatial (X, Y) coordinates.
        """
        if hasattr(self, "toolbar") and hasattr(self.toolbar, "_mouse_event_to_message"):
            try:
                msg = self.toolbar._mouse_event_to_message(event)
                self.toolbar.set_message(msg)
                return
            except Exception:
                pass
        if hasattr(self, "lbl_coordinates"):
            if event.inaxes and event.xdata is not None and event.ydata is not None:
                try:
                    coord_str = event.inaxes.format_coord(event.xdata, event.ydata)
                    clean_str = " ".join(coord_str.split())
                    self.lbl_coordinates.setText(clean_str)
                    if hasattr(self.toolbar, "locLabel"):
                        self.toolbar.locLabel.setText(clean_str)
                except Exception:
                    self.lbl_coordinates.setText("")
            else:
                self.lbl_coordinates.setText("")

    def _on_figure_leave(self, event):
        """Clear live coordinate display when cursor leaves the map canvas."""
        if hasattr(self, "lbl_coordinates"):
            self.lbl_coordinates.setText("")
        if hasattr(self.toolbar, "locLabel"):
            self.toolbar.locLabel.setText("")

    def _on_mouse_move(self, event):
        """Update rectangle during drag."""
        self._update_coordinates(event)
        if self.toolbar.mode or self._selector is None:
            return
        edgecolor = get_drag_box_edgecolor(self.current_style == 'Dark Mode')
        self._selector.ax = self.ax
        self._selector.update_move(event, edgecolor=edgecolor)

    def _on_mouse_release(self, event):
        """Finish selection on mouse release."""
        if self.toolbar.mode or self._selector is None:
            return

        self._selector.ax = self.ax
        gesture = self._selector.finish_release(event)
        if gesture is None:
            return

        x0, y0, x1, y1, is_rect = gesture
        all_x, all_y = self._get_measurement_sites()
        if len(all_x) == 0:
            return

        ctrl_held = is_ctrl_event(event)
        self.selected_points = resolve_xy_selection(
            all_x,
            all_y,
            self.selected_points,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            is_rect=is_rect,
            ctrl_held=ctrl_held,
        )

        # Emit signal with selected points
        self.spectra_selected.emit(self.selected_points)

        # Update selection overlay without full redraw
        self._update_selection_overlay()

    def _copy_figure_to_clipboard(self):
        """Copy the matplotlib figure to clipboard."""
        target_theme = "Light Mode"

        if hasattr(self, 'v_spectra_viewer_ref'):
            sv = self.v_spectra_viewer_ref
            if hasattr(sv, 'cbb_copy_theme'):
                target_theme = combo_data_or_text(sv.cbb_copy_theme)

        original_theme = getattr(self, 'current_style', "Light Mode")

        if target_theme != original_theme:
            try:
                self.current_style = target_theme
                self._do_plot_heatmap()
                self.canvas.draw()
                copy_fig_to_clb(self.canvas)
            finally:
                self.current_style = original_theme
                self._do_plot_heatmap()
        else:
            copy_fig_to_clb(self.canvas)
        self._flash_copy_feedback()

    def _flash_copy_feedback(self) -> None:
        """Provide 1-second visual checkmark feedback on copy button."""
        self.btn_copy.setIcon(Icons.check())
        QTimer.singleShot(1000, self, self._reset_copy_feedback)

    def _reset_copy_feedback(self) -> None:
        try:
            from spectroview.view.theme import ThemeManager
            icon_color = ThemeManager.instance().icon_color if ThemeManager.instance() else None
            self.btn_copy.setIcon(Icons.copy(color=icon_color))
        except (RuntimeError, AttributeError, TypeError):
            pass
