"""Canvas interactions, peak hover/drag, zoom sync, and cosmic-ray eraser for SpectraViewer."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QLabel

from spectroview.view.components.common.v_notification import show_notification
from spectroview.viewmodel.common.spectral_utils import fano_display_amplitude


class SpectraInteractionMixin:
    """Mouse/scroll interactions, live coordinates, peak dragging, and cosmic-ray eraser."""

    def _on_tool_btn_clicked(self, clicked_btn):
        """Handle mutual exclusivity across toolbar tools while allowing toggling off."""
        if clicked_btn.isChecked():
            for b in (self.btn_zoom, self.btn_baseline, self.btn_peak):
                if b is not clicked_btn and b.isChecked():
                    b.blockSignals(True)
                    b.setChecked(False)
                    b.blockSignals(False)
        if clicked_btn is self.btn_zoom:
            self._set_zoom_tool_active(clicked_btn.isChecked())
        else:
            if self.btn_zoom.isChecked():
                self.btn_zoom.blockSignals(True)
                self.btn_zoom.setChecked(False)
                self.btn_zoom.blockSignals(False)
            self._set_zoom_tool_active(False)

    def _rescale(self):
        """Rescale the spectra plot to fit within the axes."""
        if getattr(self, "_3d_mode", False):
            self._reset_3d_view = True
            self._plot()
            return
        self.ax.autoscale()
        self.canvas.draw_idle()

    def _mpl_zoom_active(self) -> bool:
        """Whether matplotlib's own internal zoom-rectangle mode is currently on."""
        return bool(getattr(self.toolbar.mode, "value", self.toolbar.mode))

    def _set_zoom_tool_active(self, active: bool):
        """Synchronize zoom_pan_active with matplotlib's internal zoom mode.

        `toolbar.zoom()` is a raw toggle, not a setter — calling it
        unconditionally (as every call site used to) silently flips
        matplotlib's zoom-rectangle tool out of sync with our own
        zoom_pan_active flag whenever it's called from more than one place
        (e.g. the eraser tool disabling/restoring zoom around its own
        activation). Routing every on/off request through here makes it
        idempotent: the toolbar is only actually toggled when its current
        state disagrees with what we're asking for.
        """
        if self._mpl_zoom_active() != active:
            self.toolbar.zoom()
        self.zoom_pan_active = active

    def _toggle_zoom_pan(self):
        """Zoom feature for button Zoom."""
        self._set_zoom_tool_active(self.btn_zoom.isChecked())

    def _on_mouse_click(self, event):
        if event.inaxes != self.ax:
            return

        # 3D mode: mouse interactions are handled natively by Axes3D (rotate/pan/zoom)
        if getattr(self, "_3d_mode", False):
            return

        # ── Check if clicking on a visible range selector marker or span ──
        if self._handle_range_mouse_press(event):
            return

        if self.zoom_pan_active:
            return

        # Heatmap mode: no peak/baseline clicking
        if self._heatmap_mode:
            return

        # The cosmic-ray eraser has its own press/motion/release handlers
        # (see _erase_on_press etc.) — without this guard, peak/baseline
        # picking could also fire for the same click if btn_peak/btn_baseline
        # happened to still be checked from before erase mode was entered.
        if self._erase_mode:
            return

        # Skip legend clicks
        legend_obj = getattr(self, "_legend_obj", None)
        if legend_obj is not None:
            try:
                self._legend_bbox = legend_obj.get_window_extent(
                    self.canvas.get_renderer()
                )
            except Exception:
                pass

        legend_bbox = getattr(self, "_legend_bbox", None)
        if legend_bbox is not None and event.x is not None and event.y is not None:
            if legend_bbox.contains(event.x, event.y):
                return

        x = event.xdata
        y = event.ydata

        # ─── Peak tool ─────────────────────────
        if self.btn_peak.isChecked():
            if event.button == 1:  # left click
                if self.btn_norm.isChecked():
                    show_notification(
                        "Deactivate Normalization before adding a peak to use the real intensity scale.",
                        notification_type="warning",
                        title="Normalization Active",
                    )
                    return
                # Check if clicking on an existing peak to drag it
                clicked_peak = self._find_peak_at_position(x, y)
                if clicked_peak:
                    # Start dragging
                    self._dragging_peak = clicked_peak
                    # Connect drag events
                    self._drag_cid = self.canvas.mpl_connect(
                        "motion_notify_event", self._on_drag_peak
                    )
                    self._release_cid = self.canvas.mpl_connect(
                        "button_release_event", self._on_release_drag
                    )
                else:
                    # Add new peak
                    self.peak_add_requested.emit(x)
            elif event.button == 3:  # right click
                self.peak_remove_requested.emit(x)

        # ─── Baseline tool ─────────────────────
        elif self.btn_baseline.isChecked():
            if event.button == 1:
                if self.btn_norm.isChecked():
                    show_notification(
                        "Deactivate Normalization before adding a baseline point to use the real intensity scale.",
                        notification_type="warning",
                        title="Normalization Active",
                    )
                    return
                self.baseline_add_requested.emit(x, y)
            elif event.button == 3:
                self.baseline_remove_requested.emit(x)

    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return

        if getattr(self, "_3d_mode", False) and hasattr(self.ax, "get_zlim"):
            z_min, z_max = self.ax.get_zlim()
            dz = (z_max - z_min) * 0.1
            if event.step < 0:  # scroll up
                z_max += dz
            else:  # scroll down
                z_max = max(z_min + 1e-9, z_max - dz)
            self.ax.set_zlim(z_min, z_max)
            self.canvas.draw_idle()
            return

        self._range_drag_bg = None
        self._range_drag_bg_size = None
        y_min, y_max = self.ax.get_ylim()
        dy = (y_max - y_min) * 0.1

        if event.step < 0:  # scroll up
            y_max += dy
        else:  # scroll down
            y_max = max(y_min + 1e-9, y_max - dy)

        self.ax.set_ylim(y_min, y_max)
        self.canvas.draw_idle()

    def _update_coordinates(self, event):
        """Update built-in live coordinate display label under the plot canvas.

        Uses toolbar._mouse_event_to_message when available so that any artist cursor
        data is formatted and preserved along with the spatial (X, Y) coordinates.
        """
        if hasattr(self, "toolbar") and hasattr(
            self.toolbar, "_mouse_event_to_message"
        ):
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
        """Clear live coordinate display when cursor leaves the plot canvas."""
        if hasattr(self, "lbl_coordinates"):
            self.lbl_coordinates.setText("")
        if hasattr(self.toolbar, "locLabel"):
            self.toolbar.locLabel.setText("")

    def _on_hover(self, event):
        self._update_coordinates(event)

        if getattr(self, "_3d_mode", False):
            self._hide_tooltip()
            self._reset_highlight()
            return

        # Skip heavy hover/peak hit-testing while dragging a range bracket
        if hasattr(self, "range_manager") and self.range_manager.is_dragging():
            return

        if event.inaxes != self.ax:
            self._hide_tooltip()
            self._reset_highlight()
            return

        # ── Range selector hover feedback ──
        if self._handle_range_hover(event):
            return

        if not self._fitted_lines:
            self._hide_tooltip()
            self._reset_highlight()
            return

        for line, info in self._fitted_lines:
            # Check if line still has a valid figure before calling contains()
            if line.figure is None:
                continue

            # Fast bounding check along x before calling expensive line.contains()
            if "x0" in info and event.xdata is not None:
                fwhm = info.get("fwhm", 10.0) or 10.0
                if abs(event.xdata - info["x0"]) > max(3.0 * float(fwhm), 20.0):
                    continue

            hit, _ = line.contains(event)
            if not hit:
                continue

            peak_model_obj = info.get("peak_model")
            model_name = getattr(peak_model_obj, "name2", "") if peak_model_obj else ""

            raw_intensity = info.get("amplitude") or info.get("ampli")
            display_intensity = raw_intensity

            # Fano model correction: show actual peak height for 'ampli'
            if model_name == "Fano" and display_intensity is not None:
                q = info.get("q", 50.0)
                display_intensity = fano_display_amplitude(display_intensity, q)

            fields = [
                ("label", info.get("peak_label")),
                ("center", info.get("x0")),
                ("intensity", display_intensity),
                ("fwhm", info.get("fwhm")),
                ("fwhm_l", info.get("fwhm_l")),
                ("fwhm_r", info.get("fwhm_r")),
                ("alpha", info.get("alpha")),
            ]

            lines = []
            for label, val in fields:
                if val is None:
                    continue
                try:
                    val_str = f"{val:.3f}" if isinstance(val, (float, int)) else str(val)
                except Exception:
                    val_str = str(val)
                lines.append(f"{label}: {val_str}")

            # Do NOT show tooltip if nothing to show
            if not lines:
                self._hide_tooltip()
                self._reset_highlight()
                return

            text = "\n".join(lines)
            self._show_tooltip(event, text)
            self._highlight_line(line)
            return

        # Nothing hit
        self._hide_tooltip()
        self._reset_highlight()

    def _show_tooltip(self, event, text):
        if not hasattr(self, "_tooltip"):
            self._tooltip = QLabel(self.canvas)
            self._tooltip.setStyleSheet("""
                background-color: rgba(255, 255, 255, 0.5);
                color: black;
                border: 0.1px solid gray;
                padding: 2px;
            """)
            self._tooltip.setWindowFlags(Qt.ToolTip)

        self._tooltip.setText(text)
        self._tooltip.move(QCursor.pos() + QPoint(10, -40))
        self._tooltip.show()

    def _hide_tooltip(self, *args):
        if hasattr(self, "_tooltip"):
            self._tooltip.hide()

    def _highlight_line(self, line):
        if self._highlighted_line == line:
            return

        self._reset_highlight()
        line._orig_lw = line.get_linewidth()
        line.set_linewidth(3)
        self._highlighted_line = line
        self.canvas.draw_idle()

    def _reset_highlight(self):
        if self._highlighted_line is not None:
            lw = getattr(self._highlighted_line, "_orig_lw", 1.5)
            self._highlighted_line.set_linewidth(lw)
            self._highlighted_line = None
            self.canvas.draw_idle()

    def _find_peak_at_position(self, x, y):
        """Find if a peak line is near the clicked position."""
        if not self._fitted_lines:
            return None

        # Use pixel-based tolerance for consistent visual click area
        TOLERANCE_PX = 10

        try:
            click_pix = self.ax.transData.transform([(x, 0)])[0][0]
        except Exception:
            return None

        closest_peak = None
        min_distance = float("inf")

        for line, info in self._fitted_lines:
            if "x0" in info:
                peak_x = info["x0"]
                try:
                    peak_pix = self.ax.transData.transform([(peak_x, 0)])[0][0]
                    distance = abs(click_pix - peak_pix)
                    if distance < TOLERANCE_PX and distance < min_distance:
                        min_distance = distance
                        closest_peak = (line, info)
                except Exception:
                    continue

        return closest_peak

    def _on_drag_peak(self, event):
        """Handle peak dragging - update peak position in real-time."""
        if not hasattr(self, "_dragging_peak") or self._dragging_peak is None:
            return

        if event.xdata is None or event.ydata is None:
            return

        line, info = self._dragging_peak
        new_x = event.xdata
        new_y = event.ydata

        # Emit signal to ViewModel to update model
        self.peak_dragged.emit(new_x, new_y)

        # Update visual feedback immediately
        spec_idx = info.get("spec_idx", 0)
        peak_idx_str = str(info.get("index", 0))

        if self._tensor_data:
            fit_models = self._tensor_data.get("fit_models", [])
            if fit_models and spec_idx < len(fit_models):
                fit_model = fit_models[spec_idx]
                if (
                    fit_model
                    and "peak_models" in fit_model
                    and peak_idx_str in fit_model["peak_models"]
                ):
                    pdict = fit_model["peak_models"][peak_idx_str]
                    shape = list(pdict.keys())[0] if pdict else ""
                    if shape:
                        if "x0" in pdict[shape]:
                            pdict[shape]["x0"]["value"] = new_x
                        if "ampli" in pdict[shape]:
                            pdict[shape]["ampli"]["value"] = new_y
                        elif "amplitude" in pdict[shape]:
                            pdict[shape]["amplitude"]["value"] = new_y

        self._plot()

    def _on_release_drag(self, event):
        """Handle peak drag release - finalize the change."""
        if hasattr(self, "_dragging_peak") and self._dragging_peak is not None:
            if hasattr(self, "_drag_cid"):
                self.canvas.mpl_disconnect(self._drag_cid)
                delattr(self, "_drag_cid")

            if hasattr(self, "_release_cid"):
                self.canvas.mpl_disconnect(self._release_cid)
                delattr(self, "_release_cid")

            self.peak_drag_finished.emit()
            self._dragging_peak = None

    # ═════════════════════════════════════════════════════════════════════
    # Cosmic-Ray Eraser Tool
    # ═════════════════════════════════════════════════════════════════════

    def _toggle_erase_mode(self, checked: bool):
        """Activate or deactivate the cosmic-ray erase mode."""
        self._erase_mode = checked

        # Show/hide companion buttons
        self.btn_erase_undo.setVisible(checked)
        self.btn_erase_validate.setVisible(checked)

        if checked:
            self._pre_erase_tool_btn = next(
                (
                    b
                    for b in (self.btn_zoom, self.btn_baseline, self.btn_peak)
                    if b.isChecked()
                ),
                None,
            )
            if self.btn_zoom.isChecked():
                self.btn_zoom.blockSignals(True)
                self.btn_zoom.setChecked(False)
                self.btn_zoom.blockSignals(False)
            self._set_zoom_tool_active(False)

            self.canvas.setCursor(Qt.CursorShape.SizeHorCursor)

            cid_press = self.canvas.mpl_connect(
                "button_press_event", self._erase_on_press
            )
            cid_motion = self.canvas.mpl_connect(
                "motion_notify_event", self._erase_on_motion
            )
            cid_release = self.canvas.mpl_connect(
                "button_release_event", self._erase_on_release
            )
            self._erase_cids = [cid_press, cid_motion, cid_release]

            self._erase_drag_start_x = None
            self._erase_highlight = None
            self._erase_history.clear()
        else:
            for cid in self._erase_cids:
                self.canvas.mpl_disconnect(cid)
            self._erase_cids.clear()

            self._erase_clear_highlight()
            self.canvas.unsetCursor()

            restore_btn = getattr(self, "_pre_erase_tool_btn", None) or self.btn_zoom
            self._pre_erase_tool_btn = None
            restore_btn.blockSignals(True)
            restore_btn.setChecked(True)
            restore_btn.blockSignals(False)
            self._set_zoom_tool_active(restore_btn is self.btn_zoom)

    def _erase_on_press(self, event):
        """Record the start position of an erase drag."""
        if not self._erase_mode:
            return
        if event.inaxes != self.ax or event.button != 1:
            return
        if event.xdata is None:
            return

        self._erase_drag_start_x = event.xdata
        self._erase_clear_highlight()

    def _erase_on_motion(self, event):
        """Update the highlight span while the user drags."""
        if not self._erase_mode:
            return
        if self._erase_drag_start_x is None:
            return
        if event.inaxes != self.ax or event.xdata is None:
            return

        xmin = min(self._erase_drag_start_x, event.xdata)
        xmax = max(self._erase_drag_start_x, event.xdata)

        if self._erase_highlight is not None:
            try:
                self._erase_highlight.remove()
            except Exception:
                pass
        self._erase_highlight = self.ax.axvspan(
            xmin, xmax, color="red", alpha=0.25, zorder=5
        )
        self.canvas.draw_idle()

    def _erase_on_release(self, event):
        """Perform the erasure when the user releases the mouse button."""
        if not self._erase_mode:
            return
        if self._erase_drag_start_x is None:
            return
        if event.button != 1 or event.xdata is None:
            self._erase_drag_start_x = None
            self._erase_clear_highlight()
            return

        x0 = min(self._erase_drag_start_x, event.xdata)
        x1 = max(self._erase_drag_start_x, event.xdata)
        self._erase_drag_start_x = None
        self._erase_clear_highlight()

        if not self._tensor_data:
            return

        viewer_slot = 0  # we always edit the first displayed spectrum
        tensor_indices = self._tensor_data.get("indices")
        md_row = int(tensor_indices[viewer_slot]) if tensor_indices is not None else viewer_slot

        x_data = self._tensor_data.get("x")
        Y_data = self._tensor_data.get("y")

        if x_data is None or Y_data is None:
            return

        if isinstance(x_data, list):
            if viewer_slot >= len(x_data):
                return
            x_arr = np.asarray(x_data[viewer_slot], dtype=float)
            y_arr = np.asarray(Y_data[viewer_slot], dtype=float).copy()
        else:
            x_arr = np.asarray(x_data, dtype=float)
            y_arr = np.asarray(Y_data[viewer_slot], dtype=float).copy()

        mask = (x_arr >= x0) & (x_arr <= x1)
        if not mask.any():
            return

        erase_indices = np.where(mask)[0]
        i_left = max(erase_indices[0] - 1, 0)
        i_right = min(erase_indices[-1] + 1, len(x_arr) - 1)

        self._erase_history.append((viewer_slot, y_arr.copy()))

        # Estimate noise sigma via first-difference MAD
        _n = len(y_arr)
        if _n >= 10:
            diffs = np.diff(y_arr)
            med_diff = np.median(diffs)
            mad = np.median(np.abs(diffs - med_diff))
            noise_sigma = float(mad / (0.6745 * np.sqrt(2)))
        else:
            noise_sigma = 0.0

        # Linear interpolation across the erased region
        x_left, y_left = x_arr[i_left], y_arr[i_left]
        x_right, y_right = x_arr[i_right], y_arr[i_right]

        if x_right != x_left:
            slope = (y_right - y_left) / (x_right - x_left)
            y_interp = y_left + slope * (x_arr[mask] - x_left)
        else:
            y_interp = np.full(mask.sum(), (y_left + y_right) / 2.0)

        if noise_sigma > 0:
            rng = np.random.default_rng()
            y_interp = y_interp + rng.normal(0.0, noise_sigma, size=y_interp.shape)

        y_arr[mask] = y_interp

        if isinstance(Y_data, list):
            self._tensor_data["y"][viewer_slot] = y_arr
        else:
            Y_data[viewer_slot] = y_arr

        self.cosmicRayErased.emit(md_row, y_arr)
        self._plot()

    def _erase_clear_highlight(self):
        """Remove the selection highlight span from the axes."""
        if self._erase_highlight is not None:
            try:
                self._erase_highlight.remove()
            except Exception:
                pass
            self._erase_highlight = None
            self.canvas.draw_idle()

    def _erase_undo(self):
        """Undo the last erase operation."""
        if not self._erase_history:
            return

        spec_idx, y_snapshot = self._erase_history.pop()

        if not self._tensor_data:
            return

        Y_data = self._tensor_data.get("y")
        if Y_data is None:
            return

        if isinstance(Y_data, list):
            if spec_idx < len(Y_data):
                self._tensor_data["y"][spec_idx] = y_snapshot
        else:
            Y_data[spec_idx] = y_snapshot

        self._plot()

    def _erase_validate(self):
        """Accept all erasures and persist them to the model."""
        self._erase_history.clear()
        self.cosmicRayValidated.emit()

        self.btn_eraser.blockSignals(True)
        self.btn_eraser.setChecked(False)
        self.btn_eraser.blockSignals(False)
        self._toggle_erase_mode(False)
