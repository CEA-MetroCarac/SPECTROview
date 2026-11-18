"""2D heatmap rendering and filename-token Y-axis helpers for SpectraViewer."""

from __future__ import annotations

import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QComboBox

from spectroview import PLOT_POLICY_DARK, PLOT_POLICY_LIGHT
from spectroview.view.components.spectra_viewer.common import (
    SPECTRA_DISCRETE_PALETTES,
    extract_filename_tokens,
    extract_series_parameter_values,
    record_token_selection,
    refresh_token_combobox,
    token_pattern,
)


class SpectraHeatmapMixin:
    """2D pcolormesh heatmap rendering, colorbar cleanup, and token Y-axis selectors."""

    def _remove_heatmap_colorbar(self):
        """Remove the colorbar created by a previous heatmap draw, if any."""
        if self._heatmap_colorbar is not None:
            try:
                self._heatmap_colorbar.remove()
            except Exception:
                pass
            self._heatmap_colorbar = None

    @staticmethod
    def _extract_filename_tokens(fname):
        """Split *fname* on underscores and return tokens that contain digits."""
        return extract_filename_tokens(fname)

    def _collect_heatmap_yaxis_tokens(self):
        """Return filename tokens suitable as heatmap Y-axis or colorbar choices.

        Parse the *first* selected spectrum's filename, find every
        underscore-delimited token that contains a numeric value, and
        return them as ``(position, token_text)`` pairs.
        """
        fnames = (
            self._tensor_data.get("fnames", []) if self._tensor_data else []
        )
        if not fnames:
            return []
        return [
            (pos, tok)
            for pos, tok, _val in self._extract_filename_tokens(fnames[0])
        ]

    @staticmethod
    def _token_pattern(token_text: str) -> str:
        """Normalize numeric parts of a filename token to '#' (e.g. '25°C' -> '#°C')."""
        return token_pattern(token_text)

    def _record_token_selection(self, combo: QComboBox, state_dict: dict):
        """Record the active parameter selection (text, token slot position, unit pattern)."""
        record_token_selection(combo, state_dict)

    def _on_heatmap_yaxis_changed(self):
        """Track user-chosen heatmap Y-axis parameter for the session and redraw."""
        self._record_token_selection(
            self.cbb_heatmap_yaxis, self._heatmap_yaxis_selection
        )
        self._emit_view_options()

    def _on_heatmap_ylabel_changed(self):
        """Debounce custom Y-axis / colorbar label edits before redrawing."""
        if not hasattr(self, "_ylabel_timer"):
            self._ylabel_timer = QTimer(self)
            self._ylabel_timer.setSingleShot(True)
            self._ylabel_timer.setInterval(400)
            self._ylabel_timer.timeout.connect(self._emit_view_options)
        self._ylabel_timer.start()

    def _refresh_token_combobox(self, combo: QComboBox, state_dict: dict):
        """Rebuild a parameter combobox from filenames while preserving session choice."""
        tokens = self._collect_heatmap_yaxis_tokens()
        refresh_token_combobox(combo, state_dict, tokens)

    def _refresh_heatmap_yaxis_choices(self):
        """Rebuild the heatmap Y-axis combobox items from the selected filenames."""
        self._refresh_token_combobox(
            self.cbb_heatmap_yaxis, self._heatmap_yaxis_selection
        )

    def _extract_series_parameter_values(
        self,
        selected_token: str,
        custom_label: str,
        token_pos: int | None = None,
    ) -> tuple[np.ndarray, str]:
        """Return ``(values, label)`` parsed from filenames or spectrum order."""
        fnames = (
            self._tensor_data.get("fnames", []) if self._tensor_data else []
        )
        n_spectra = len(self._tensor_data.get("y", [])) if self._tensor_data else 0
        return extract_series_parameter_values(
            fnames,
            n_spectra,
            selected_token,
            custom_label,
            token_pos=token_pos,
        )

    def _get_heatmap_y_values(self):
        """Return ``(y_values, y_label)`` for the heatmap Y-axis."""
        return self._extract_series_parameter_values(
            self.cbb_heatmap_yaxis.currentText(),
            self.le_heatmap_ylabel.text(),
            token_pos=self.cbb_heatmap_yaxis.currentData(),
        )

    def _plot_heatmap(self):
        """Render the selected spectra as a 2D pcolormesh heatmap."""
        from scipy.interpolate import interp1d

        style_name = self.cbb_theme.currentText()
        if style_name == "Dark Mode":
            style_path = PLOT_POLICY_DARK
        else:
            style_path = PLOT_POLICY_LIGHT

        fg_color = plt.rcParams.get("axes.labelcolor", "black")

        # Save current zoom/pan limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        is_default = xlim == (0.0, 1.0) and ylim == (0.0, 1.0)

        self.ax.clear()
        self._remove_heatmap_colorbar()
        self._fitted_lines.clear()

        Y_data = self._tensor_data.get("y", [])
        x_data = self._tensor_data.get("x")
        N = len(Y_data)

        if N < 2 or x_data is None:
            self.ax.set_title("Heatmap requires ≥ 2 spectra", color=fg_color)
            self.canvas.draw_idle()
            return

        # ── Build intensity matrix ──
        # For tensor_list (ragged), interpolate all spectra onto a common x grid
        is_list = isinstance(Y_data, list)
        if is_list:
            x_arrays = [
                np.asarray(x_data[i] if isinstance(x_data, list) else x_data)
                for i in range(N)
            ]
            x_min = max(np.nanmin(xa) for xa in x_arrays)
            x_max = min(np.nanmax(xa) for xa in x_arrays)
            n_pts = max(len(xa) for xa in x_arrays)
            raman_shift = np.linspace(x_min, x_max, n_pts)

            intensity_rows = []
            for i in range(N):
                yi = np.asarray(Y_data[i])
                xi = x_arrays[i]
                f = interp1d(
                    xi, yi, kind="linear", bounds_error=False, fill_value=np.nan
                )
                intensity_rows.append(f(raman_shift))
            intensity_matrix = np.array(intensity_rows)
        else:
            raman_shift = np.asarray(x_data)
            intensity_matrix = np.asarray(Y_data)

        # ── Apply normalization if enabled ──
        intensity_matrix = np.asarray(
            self._get_normalized_y_tensor(raman_shift, intensity_matrix)
        )

        # ── Y-axis values ──
        y_values, y_label = self._get_heatmap_y_values()

        valid_y = ~np.isnan(y_values)
        if not np.all(valid_y):
            y_values = y_values[valid_y]
            intensity_matrix = intensity_matrix[valid_y]
            N = len(y_values)
            if N == 0:
                return

        # Sort by Y values for a coherent image
        sort_idx = np.argsort(y_values)
        y_values = y_values[sort_idx]
        intensity_matrix = intensity_matrix[sort_idx]

        # ── Handle duplicate Y-values (average them) ──
        unique_y, inverse_idx = np.unique(y_values, return_inverse=True)
        if len(unique_y) < len(y_values):
            new_intensity = np.zeros((len(unique_y), intensity_matrix.shape[1]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for i in range(len(unique_y)):
                    new_intensity[i] = np.nanmean(
                        intensity_matrix[inverse_idx == i], axis=0
                    )
            y_values = unique_y
            intensity_matrix = new_intensity
            N = len(y_values)

        # ── Optional interpolation along Y ──
        n_interp = self.spin_heatmap_interp.value()
        if n_interp > 0 and N >= 3 and len(np.unique(y_values)) >= 3:
            y_fine = np.linspace(y_values.min(), y_values.max(), n_interp)
            kind = "cubic" if len(y_values) >= 4 else "linear"

            # Fast vectorized path: when all entries are valid (typical case),
            # interpolate the whole matrix along axis=0 at once!
            if not np.isnan(intensity_matrix).any():
                f = interp1d(
                    y_values,
                    intensity_matrix,
                    kind=kind,
                    axis=0,
                    bounds_error=False,
                    fill_value=np.nan,
                )
                intensity_plot = f(y_fine)
            else:
                interp_cols = []
                for col_idx in range(intensity_matrix.shape[1]):
                    col = intensity_matrix[:, col_idx]
                    valid = ~np.isnan(col)
                    if valid.sum() >= 2:
                        col_kind = "cubic" if valid.sum() >= 4 else "linear"
                        f = interp1d(
                            y_values[valid],
                            col[valid],
                            kind=col_kind,
                            bounds_error=False,
                            fill_value=np.nan,
                        )
                        interp_cols.append(f(y_fine))
                    else:
                        interp_cols.append(np.full_like(y_fine, np.nan))
                intensity_plot = np.array(interp_cols).T
            y_plot = y_fine
        else:
            intensity_plot = intensity_matrix
            y_plot = y_values

        # ── Colormap from current palette ──
        palette_name = self.cbb_color_palette.currentText()
        if palette_name in SPECTRA_DISCRETE_PALETTES:
            cmap_name = "jet"  # Discrete palettes aren't suitable for heatmaps
        else:
            cmap_name = palette_name
        try:
            cmap = mpl.colormaps[cmap_name]
        except (KeyError, AttributeError):
            cmap = mpl.colormaps["jet"]

        # ── Draw pcolormesh ──
        with plt.style.context(style_path):
            mesh = self.ax.pcolormesh(
                raman_shift,
                y_plot,
                intensity_plot,
                shading="auto",
                cmap=cmap,
            )

            cbar = self.figure.colorbar(mesh, ax=self.ax)
            cbar.set_label(self.cbb_yaxis.currentText(), color=fg_color)
            cbar.ax.tick_params(colors=fg_color)
            self._heatmap_colorbar = cbar

        self.ax.set_xlabel(self.cbb_xaxis.currentText())
        self.ax.set_ylabel(y_label)

        if self.act_grid.isChecked():
            self.ax.grid(True, linestyle="--", linewidth=0.5, color="gray")

        if not is_default:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

        self.lbl_r2.setText("")
        self.lbl_noise.setText("")

        # ── Render active spectral range selectors ──
        if hasattr(self, "range_manager") and self.ax is not None:
            self.range_manager.render(self.ax)

        self.canvas.draw_idle()

    def _toggle_heatmap_mode(self, checked: bool):
        """Activate or deactivate heatmap mode, disabling incompatible tools."""
        self._heatmap_mode = bool(checked)
        if checked:
            if getattr(self, "_3d_mode", False) or (
                hasattr(self, "btn_3d") and self.btn_3d.isChecked()
            ):
                self._3d_mode = False
                if hasattr(self, "btn_3d"):
                    self.btn_3d.blockSignals(True)
                    self.btn_3d.setChecked(False)
                    self.btn_3d.blockSignals(False)

        self._sync_mode_toolbar_state()
        self._emit_view_options()
