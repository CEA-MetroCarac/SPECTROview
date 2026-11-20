"""1D spectrum plotting, normalization, waterfall shifts, and fit overlays for SpectraViewer."""

from __future__ import annotations

import warnings

import matplotlib as mpl
import matplotlib.collections as mcoll
import matplotlib.lines as mlines
import numpy as np
from PySide6.QtCore import QSettings

from spectroview.fit_engine.evaluator import eval_peak_initial
from spectroview.fit_engine.noise import detect_noise_level
from spectroview.i18n import combo_data_or_text, tr
from spectroview.view.components.spectra_viewer.common import (
    _MockPeakModelObj,
    compute_normalized_y,
    compute_normalized_y_tensor,
    get_colors_for_palette,
    get_palette_color_choices,
)


class SpectraPlot1DMixin:
    """1D multi-spectrum segment building, overlays, normalization, and finalization."""

    def _compute_shift_steps(self, Y_norm=None):
        """Compute per-spectrum X and Y shift steps based on slider values and data range."""
        y_slider = self.slider_y_shift.value()  # 0–100
        x_slider = self.slider_x_shift.value()  # 0–100

        if y_slider == 0 and x_slider == 0:
            return 0.0, 0.0

        if not self._tensor_data or len(self._tensor_data["y"]) == 0:
            return 0.0, 0.0

        x = self._tensor_data["x"]
        if Y_norm is None:
            Y = self._tensor_data["y"]
            Y_norm = self._get_normalized_y_tensor(x, Y)

        if isinstance(Y_norm, list):
            y_max = max(np.max(y_arr) for y_arr in Y_norm)
            y_min = min(np.min(y_arr) for y_arr in Y_norm)
            y_range = float(y_max - y_min)
        else:
            y_range = float(np.max(Y_norm) - np.min(Y_norm))

        if isinstance(x, list):
            x_max = max(np.max(x_arr) for x_arr in x)
            x_min = min(np.min(x_arr) for x_arr in x)
            x_range = float(x_max - x_min)
        else:
            x_range = float(np.max(x) - np.min(x))

        # Map slider percentage to a fraction of the data range
        y_shift_step = (y_slider / 100.0) * y_range if y_range > 0 else 0.0
        x_shift_step = (x_slider / 100.0) * x_range if x_range > 0 else 0.0

        return x_shift_step, y_shift_step

    def _get_normalized_y_tensor(self, x, Y):
        """Apply normalization if enabled for a batch of spectra (VIEW-ONLY)."""
        if not self.btn_norm.isChecked():
            return Y

        try:
            xmin = float(self.norm_xmin.text()) if self.norm_xmin.text() else None
            xmax = float(self.norm_xmax.text()) if self.norm_xmax.text() else None
        except ValueError:
            xmin = xmax = None

        return compute_normalized_y_tensor(x, Y, xmin=xmin, xmax=xmax)

    def _get_normalized_y(self, x, y, xmin=None, xmax=None):
        """Apply normalization if enabled (VIEW-ONLY)."""
        if not self.btn_norm.isChecked():
            return y

        if xmin is None and xmax is None:
            try:
                xmin = float(self.norm_xmin.text()) if self.norm_xmin.text() else None
                xmax = float(self.norm_xmax.text()) if self.norm_xmax.text() else None
            except ValueError:
                xmin = xmax = None

        return compute_normalized_y(x, y, xmin=xmin, xmax=xmax)

    def _get_colors_for_palette(self, palette_name, count=None):
        """Return ``count`` colors sampled from a named spectrum palette."""
        return get_colors_for_palette(palette_name, count=count)

    def _get_colors_cycle(self, count=None):
        """Return colors sampled from the currently selected palette."""
        palette_name = (
            self.cbb_color_palette.currentText()
            if hasattr(self, "cbb_color_palette")
            else "DEFAULT_COLORS"
        )
        return self._get_colors_for_palette(palette_name, count)

    def _get_palette_color_choices(self, palette_name, spectrum_count=0):
        """Return the selectable swatches for a palette in the legend editor."""
        return get_palette_color_choices(palette_name, spectrum_count=spectrum_count)

    def _resolve_series_colors(self, count: int) -> list:
        """Resolve per-spectrum colors honoring explicit overrides, colorbar, and palette."""
        if self._is_legend_colorbar_active():
            param_values, _ = self._get_colorbar_values()
            _, _, colors_cycle = self._resolve_colorbar_cmap_and_norm(count, param_values)
        else:
            colors_cycle = self._get_colors_cycle(count)
        t_colors = self._tensor_data.get("colors") if self._tensor_data else None
        if t_colors is None:
            t_colors = []
        return [
            t_colors[i] if i < len(t_colors) and t_colors[i] else colors_cycle[i]
            for i in range(count)
        ]

    def _build_tensor_segments(
        self,
        x_shift_step,
        y_shift_step,
        plot_style,
        lw,
        fg_color,
        colors_cycle,
        Y_norm=None,
    ):
        """Build bulk LineCollection / scatter data for tensor-mode rendering."""
        # These lists are filled below and passed to _finalize_plot, which
        # converts them into a single LineCollection for efficient rendering.
        main_segments, main_colors = [], []
        raw_segments, raw_colors = [], []
        all_x_dots, all_y_dots, all_c_dots = [], [], []

        N = len(self._tensor_data["y"])
        x_offsets = np.arange(N) * x_shift_step
        y_offsets = np.arange(N) * y_shift_step

        x = self._tensor_data["x"]
        Y = self._tensor_data["y"]
        if Y_norm is None:
            Y_norm = self._get_normalized_y_tensor(x, Y)

        # Resolve each spectrum's color: use stored color if available, else cycle
        t_colors = self._tensor_data.get("colors")
        if t_colors is None:
            t_colors = []
        main_colors = [
            t_colors[i]
            if i < len(t_colors) and t_colors[i]
            else colors_cycle[i]
            for i in range(N)
        ]

        # tensor_list stores ragged arrays as a Python list; tensor stores a uniform 2-D ndarray
        is_list = isinstance(Y, list)

        if plot_style == "line":
            # Build one (M×2) coordinate array per spectrum for LineCollection
            if is_list:  # Ragged case: each spectrum may have a different length
                main_segments = [
                    np.column_stack([
                        (x[i] if isinstance(x, list) else x) + x_offsets[i],
                        Y_norm[i] + y_offsets[i],
                    ])
                    for i in range(N)
                ]
            else:  # Uniform case: broadcast offset addition on the whole 2-D array at once
                Y_plot = Y_norm + y_offsets[:, None]
                X_plot = x + x_offsets[:, None]
                main_segments = [
                    np.column_stack([X_plot[i], Y_plot[i]]) for i in range(N)
                ]
        else:  # "dot" — collect all points into flat arrays for a single scatter call
            if is_list:
                for i in range(N):
                    xi = x[i] if isinstance(x, list) else x
                    all_x_dots.append(xi + x_offsets[i])
                    all_y_dots.append(Y_norm[i] + y_offsets[i])
                    all_c_dots.extend([main_colors[i]] * len(xi))
            else:
                Y_plot = Y_norm + y_offsets[:, None]
                X_plot = x + x_offsets[:, None]
                all_x_dots.append(X_plot.flatten())
                all_y_dots.append(Y_plot.flatten())
                for i in range(N):
                    all_c_dots.extend([main_colors[i]] * len(x))

        # ── Raw (unprocessed) overlay segments ──
        # x0/y0 hold the original data before baseline subtraction / normalisation.
        if self.act_raw.isChecked():
            x0 = self._tensor_data.get("x0")
            Y0 = self._tensor_data.get("y0")
            if x0 is not None and Y0 is not None:
                if isinstance(Y0, list):
                    raw_segments = [
                        np.column_stack([
                            (x0[i] if isinstance(x0, list) else x0) + x_offsets[i],
                            Y0[i] + y_offsets[i],
                        ])
                        for i in range(N)
                    ]
                else:
                    Y0_plot = Y0 + y_offsets[:, None]
                    X0_plot = x0 + x_offsets[:, None]
                    raw_segments = [
                        np.column_stack([X0_plot[i], Y0_plot[i]]) for i in range(N)
                    ]
                raw_colors = [fg_color] * N  # Raw data drawn in a neutral foreground color

        # ── Proxy legend lines ──
        # Capped by spin_max_overlays to avoid an unmanageable legend for large maps.
        # Skipped when colorbar mode replaces the discrete legend box.
        if not self._is_legend_colorbar_active():
            t_labels = self._tensor_data.get("labels", [])
            t_fnames = self._tensor_data.get("fnames", [])
            proxies = self._tensor_data.get("proxies", [])  # Spectrum objects for interactive tooltip
            for spec_idx in range(min(N, self.spin_max_legend_items.value())):
                spec_color = main_colors[spec_idx]
                # Priority: custom label > filename > generic fallback
                label_str = (
                    t_labels[spec_idx]
                    if spec_idx < len(t_labels) and t_labels[spec_idx]
                    else (
                        t_fnames[spec_idx]
                        if spec_idx < len(t_fnames)
                        else tr("Spectrum {index}", index=spec_idx + 1)
                    )
                )
                proxy_line = mlines.Line2D(
                    [], [], color=spec_color, label=label_str, lw=lw
                )  # Zero-length: invisible on plot
                if spec_idx < len(proxies):
                    proxy_line._spectrum_ref = proxies[spec_idx]  # Used by hover tooltip handler
                self.ax.add_line(proxy_line)

        # Return all collected data; _finalize_plot will render it
        return {
            "main_segments": main_segments,
            "main_colors": main_colors,
            "raw_segments": raw_segments,
            "raw_colors": raw_colors,
            "all_x_dots": all_x_dots,
            "all_y_dots": all_y_dots,
            "all_c_dots": all_c_dots,
        }

    def _get_tensor_item(self, key, spec_idx, is_tensor_list):
        """Safely retrieve a per-spectrum item from ``_tensor_data``."""
        items = self._tensor_data.get(key, [])
        if is_tensor_list:
            return items[spec_idx] if items and spec_idx < len(items) else None
        return items[spec_idx] if items is not None and len(items) > spec_idx else None

    def _draw_tensor_overlays(self, x_shift_step, y_shift_step, lw, fg_color):
        """Draw baseline, bestfit peaks, residual, and noise overlays (tensor mode)."""
        n_specs = len(self._tensor_data.get("fnames", []))
        is_tensor_list = self._tensor_data.get("type") == "tensor_list"
        proxies = self._tensor_data.get("proxies", [])

        for spec_idx in range(min(n_specs, self.spin_max_legend_items.value())):
            x_val = self._tensor_data.get("x")
            x = x_val[spec_idx] if isinstance(x_val, list) else x_val

            y_val = self._tensor_data.get("y")
            y_raw = y_val[spec_idx] if isinstance(y_val, list) else y_val[spec_idx]

            x_offset = spec_idx * x_shift_step
            y_offset = spec_idx * y_shift_step

            # Check if baseline is subtracted
            is_sub = False
            if spec_idx < len(proxies):
                is_sub = getattr(proxies[spec_idx].baseline, "is_subtracted", False)
                if isinstance(is_sub, np.ndarray):
                    is_sub = bool(is_sub.any())

            # ── Baseline curve ──
            y_base = self._get_tensor_item("y_baseline", spec_idx, is_tensor_list)
            if y_base is not None and not is_sub:
                self.ax.plot(
                    x + x_offset, y_base + y_offset, "g--", lw=1.5, label=tr("baseline")
                )

            # ── Baseline anchor points ──
            bl_config = self._get_tensor_item(
                "baseline_config", spec_idx, is_tensor_list
            )
            if bl_config and bl_config.get("points") and not is_sub:
                self._draw_tensor_baseline_points(
                    bl_config, x, y_raw, x_offset, y_offset, fg_color
                )

            # ── Peaks and Bestfit ──
            if self.btn_bestfit.isChecked():
                self._draw_tensor_bestfit(
                    spec_idx,
                    is_tensor_list,
                    x,
                    y_raw,
                    y_base,
                    x_offset,
                    y_offset,
                    lw,
                    fg_color,
                )

            # ── Residual ──
            if self.act_residual.isChecked():
                y_bestfit = self._get_tensor_item(
                    "y_bestfit", spec_idx, is_tensor_list
                )
                if y_bestfit is not None:
                    residual = y_raw - y_bestfit
                    self.ax.plot(
                        x + x_offset,
                        residual + y_offset,
                        "blue",
                        lw=1.0,
                        label=tr("residual"),
                    )

            # ── Noise level ──
            if self.act_noise_level.isChecked():
                self._draw_noise_level(y_raw, spec_idx, y_offset)

    def _draw_tensor_baseline_points(
        self, bl_config, x, y_raw, x_offset, y_offset, fg_color
    ):
        """Draw anchor-point markers for a tensor-mode baseline."""
        mode = bl_config.get("mode", "")
        if mode not in ("Linear", "Polynomial"):
            return
        pts = bl_config["points"]
        if len(pts) != 2 or len(pts[0]) == 0:
            return
        xs_arr = np.asarray(pts[0], dtype=float)
        if bl_config.get("attached", False):
            inds = [np.argmin(np.abs(x - xp)) for xp in xs_arr]
            sigma = bl_config.get("sigma", 4)
            if sigma > 0:
                from scipy.ndimage import gaussian_filter1d

                y_curve = gaussian_filter1d(y_raw, sigma=sigma)
            else:
                y_curve = y_raw
            ys_arr = np.asarray([y_curve[ind] for ind in inds], dtype=float)
            xs_arr = np.asarray([x[ind] for ind in inds], dtype=float)
        else:
            ys_arr = np.asarray(pts[1], dtype=float)
        self.ax.plot(
            xs_arr + x_offset,
            ys_arr + y_offset,
            marker="o",
            color=fg_color,
            mfc="none",
            ms=5,
            ls="none",
        )

    def _draw_tensor_bestfit(
        self,
        spec_idx,
        is_tensor_list,
        x,
        y_raw,
        y_base,
        x_offset,
        y_offset,
        lw,
        fg_color,
    ):
        """Draw bestfit peak curves and composite for one spectrum (tensor mode)."""
        fit_model = self._get_tensor_item("fit_models", spec_idx, is_tensor_list)

        # Retrieve individual peak curves
        y_peaks_list = self._tensor_data.get("y_peaks", [])
        y_peaks = None
        if y_peaks_list:
            if is_tensor_list:
                y_peaks = (
                    y_peaks_list[spec_idx] if spec_idx < len(y_peaks_list) else None
                )
            else:
                y_peaks = [p[spec_idx] for p in y_peaks_list]

        color_kwargs = {}
        if (
            hasattr(self, "act_bestfit_colorful")
            and not self.act_bestfit_colorful.isChecked()
        ):
            color_kwargs["color"] = fg_color

        if fit_model and fit_model.get("peak_models") and len(x) > 1:
            # ── High-resolution smooth curves via eval_peak_initial ──
            # Ensure we don't downsample high-res spectra by using at least 3x the original points
            n_fine_pts = max(3000, len(x) * 5)
            x_fine = np.linspace(x.min(), x.max(), n_fine_pts)
            sorted_keys = sorted(
                fit_model["peak_models"].keys(), key=lambda k: int(k)
            )

            y_fit_fine = np.zeros_like(x_fine)
            for i, k in enumerate(sorted_keys):
                p_model = fit_model["peak_models"][k]
                y_peak_fine = eval_peak_initial(x_fine, p_model)
                y_fit_fine += y_peak_fine

                curr_kwargs = dict(color_kwargs)
                if "color" not in curr_kwargs:
                    curr_kwargs["color"] = f"C{(i + 1) % 10}"

                (peak_line,) = self.ax.plot(
                    x_fine + x_offset,
                    y_peak_fine + y_offset,
                    lw=(lw * 0.6),
                    **curr_kwargs,
                )

                peak_label = f"Peak{i + 1}"
                if "peak_labels" in fit_model and i < len(fit_model["peak_labels"]):
                    peak_label = fit_model["peak_labels"][i]

                peak_info = self._build_tensor_peak_info(
                    p_model, peak_label, int(k), spec_idx
                )
                self._fitted_lines.append((peak_line, peak_info))

                self._annotate_peak_label(
                    x_fine,
                    y_peak_fine,
                    peak_label,
                    peak_line,
                    x_offset,
                    y_offset,
                    fg_color,
                )

            # Add interpolated baseline for smooth composite curve
            if y_base is not None:
                y_base_fine = np.interp(x_fine, x, y_base)
                y_fit_fine += y_base_fine

            # Plot smooth composite bestfit
            self.ax.plot(
                x_fine + x_offset, y_fit_fine + y_offset, lw=(lw * 0.6), color=fg_color
            )

        elif y_peaks is not None:
            # ── Fallback: discrete peak curves ──
            for i, y_peak in enumerate(y_peaks):
                curr_kwargs = dict(color_kwargs)
                if "color" not in curr_kwargs:
                    curr_kwargs["color"] = f"C{(i + 1) % 10}"

                (peak_line,) = self.ax.plot(
                    x + x_offset,
                    y_peak + y_offset,
                    lw=(lw * 0.6),
                    **curr_kwargs,
                )

                peak_label = f"Peak{i + 1}"
                if (
                    fit_model
                    and "peak_labels" in fit_model
                    and i < len(fit_model["peak_labels"])
                ):
                    peak_label = fit_model["peak_labels"][i]

                if (
                    fit_model
                    and "peak_models" in fit_model
                    and str(i) in fit_model["peak_models"]
                ):
                    peak_info = self._build_tensor_peak_info(
                        fit_model["peak_models"][str(i)], peak_label, i, spec_idx
                    )
                else:
                    peak_info = {
                        "peak_label": peak_label,
                        "peak_model": None,
                        "index": i,
                        "spec_idx": spec_idx,
                    }

                self._fitted_lines.append((peak_line, peak_info))

                self._annotate_peak_label(
                    x, y_peak, peak_label, peak_line, x_offset, y_offset, fg_color
                )

        # Composite bestfit curve (fallback from y_bestfit data)
        y_bestfit = self._get_tensor_item("y_bestfit", spec_idx, is_tensor_list)
        if y_bestfit is not None:
            if not (fit_model and fit_model.get("peak_models")):
                self.ax.plot(
                    x + x_offset,
                    y_bestfit + y_offset,
                    lw=(lw * 0.6),
                    color=fg_color,
                )

    def _build_tensor_peak_info(self, p_model, peak_label, index, spec_idx):
        """Build a peak-info dict from a tensor-mode peak model dictionary."""
        shape_name = list(p_model.keys())[0] if p_model else ""
        peak_info = {
            "peak_label": peak_label,
            "peak_model": (_MockPeakModelObj(shape_name) if shape_name else None),
            "index": index,
            "shape": shape_name,
            "spec_idx": spec_idx,
        }
        if shape_name:
            params = p_model[shape_name]
            for param_name, param_dict in params.items():
                if isinstance(param_dict, dict) and "value" in param_dict:
                    peak_info[param_name] = param_dict["value"]
                    if param_name == "amplitude":
                        peak_info["ampli"] = param_dict["value"]
        return peak_info

    def _annotate_peak_label(
        self, x_arr, y_arr, peak_label, peak_line, x_offset, y_offset, fg_color
    ):
        """Draw a text label at the peak maximum if the option is enabled."""
        if not (
            hasattr(self, "act_show_peak_label")
            and self.act_show_peak_label.isChecked()
        ):
            return
        idx = np.argmax(np.abs(y_arr))
        px, py = x_arr[idx], y_arr[idx]
        txt_color = (
            fg_color
            if not self.act_bestfit_colorful.isChecked()
            else peak_line.get_color()
        )
        self.ax.text(
            px + x_offset,
            py + y_offset,
            peak_label,
            color=txt_color,
            fontsize=9,
            ha="center",
            va="bottom" if py >= 0 else "top",
        )

    def _draw_noise_level(self, y_data, spec_idx, y_offset):
        """Draw a horizontal noise-level line."""
        try:
            coef_noise = QSettings("CEA-Leti", "SPECTROview").value(
                "fit_settings/coef_noise", 1.0, float
            )
            ampli_noise = detect_noise_level(y_data)
            noise_level = coef_noise * ampli_noise
            noise_label = (
                tr("noise ({coef}×{ampli})", coef=coef_noise, ampli=f"{ampli_noise:.1f}")
                if spec_idx == 0
                else None
            )
            self.ax.axhline(
                y=noise_level + y_offset,
                color="orange",
                ls="--",
                lw=1.0,
                alpha=0.8,
                label=noise_label,
            )
        except Exception:
            pass

    def _finalize_plot(
        self, segments, plot_style, lw, dot_size, fg_color, xlim, ylim, is_default
    ):
        """Render bulk collections, then apply legend, grid, axes labels, and limits."""
        main_segments = segments["main_segments"]
        main_colors = segments["main_colors"]
        raw_segments = segments["raw_segments"]
        raw_colors = segments["raw_colors"]
        all_x_dots = segments["all_x_dots"]
        all_y_dots = segments["all_y_dots"]
        all_c_dots = segments["all_c_dots"]

        # ── Render Bulk Collections ──
        if plot_style == "dot" and all_x_dots:
            self.ax.scatter(
                np.concatenate(all_x_dots),
                np.concatenate(all_y_dots),
                s=dot_size,
                c=all_c_dots,
                marker="o",
                edgecolors="none",
            )
            all_points = np.column_stack(
                [np.concatenate(all_x_dots), np.concatenate(all_y_dots)]
            )
            self.ax.update_datalim(all_points)
        elif main_segments:
            lc = mcoll.LineCollection(
                main_segments, colors=main_colors, linewidths=lw
            )
            self.ax.add_collection(lc)
            all_points = np.vstack(main_segments)
            self.ax.update_datalim(all_points)

        if raw_segments:
            lc_raw = mcoll.LineCollection(
                raw_segments, colors=raw_colors, linewidths=0.8, alpha=0.8
            )
            self.ax.add_collection(lc_raw)
            raw_points = np.vstack(raw_segments)
            self.ax.update_datalim(raw_points)

        self.ax.autoscale_view()

        # ── Legend / axes / grid ──
        self._legend_obj = None
        self._legend_bbox = None
        if self.btn_legend.isChecked():
            if self._is_legend_colorbar_active():
                param_values, cbar_label = self._get_colorbar_values()
                cmap, norm, _ = self._resolve_colorbar_cmap_and_norm(
                    len(self._tensor_data.get("y", [])), param_values
                )
                sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                cbar = self.figure.colorbar(sm, ax=self.ax)
                cbar.set_label(cbar_label, color=fg_color)
                cbar.ax.tick_params(colors=fg_color)
                self._heatmap_colorbar = cbar
            else:
                handles, labels = self.ax.get_legend_handles_labels()
                max_items = self.spin_max_legend_items.value()
                if (
                    getattr(self, "act_legend_outside", None)
                    and self.act_legend_outside.isChecked()
                ):
                    legend = self.ax.legend(
                        handles[:max_items],
                        labels[:max_items],
                        loc="center left",
                        bbox_to_anchor=(1.02, 0.5),
                    )
                else:
                    legend = self.ax.legend(
                        handles[:max_items], labels[:max_items], loc="upper right"
                    )
                self._make_legend_pickable(legend)

        if self.act_grid.isChecked():
            self.ax.grid(True, linestyle="--", linewidth=0.5, color="gray")

        self.ax.set_xlabel(self.cbb_xaxis.currentText())
        self.ax.set_ylabel(self.cbb_yaxis.currentText())
        is_log = combo_data_or_text(self.cbb_yscale) == "Log"
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Attempt to set non-positive .*lim on a log-scaled axis.*",
                category=UserWarning,
            )
            if is_log:
                cur_lo, cur_hi = self.ax.get_ylim()
                if cur_hi > 0 and cur_lo <= 0:
                    self.ax.set_ylim(max(cur_lo, min(cur_hi * 1e-4, 1e-9)), cur_hi)
            self.ax.set_yscale("log" if is_log else "linear")

            if not is_default:
                self.ax.set_xlim(xlim)
                if is_log:
                    if ylim[1] > 0:
                        lo = ylim[0] if ylim[0] > 0 else min(ylim[1] * 1e-4, 1e-9)
                        self.ax.set_ylim(lo, ylim[1])
                else:
                    self.ax.set_ylim(ylim)

        # ── Render active spectral range selectors ──
        if hasattr(self, "range_manager") and self.ax is not None:
            self.range_manager.render(self.ax)

        self.canvas.draw_idle()

    def _update_r2_display(self):
        """Display R² value and noise level from the first selected spectrum."""
        if not self._tensor_data:
            self.lbl_r2.setText("R²=0")
            self.lbl_noise.setText(tr("Noise={value}", value="0"))
            return

        r2_vals = self._tensor_data.get("fit_r2")
        y_vals = self._tensor_data.get("y")
        if r2_vals is not None and len(r2_vals) > 0 and r2_vals[0] is not None:
            r2 = r2_vals[0]
            self.lbl_r2.setText(f"R²={r2:.3f}" if r2 > 0 else "R²=0")
        else:
            self.lbl_r2.setText("R²=0")

        if y_vals is not None and len(y_vals) > 0:
            y0 = y_vals[0]
            try:
                coef_noise = QSettings("CEA-Leti", "SPECTROview").value(
                    "fit_settings/coef_noise", 1.0, float
                )
                noise = detect_noise_level(y0)
                noise_val = coef_noise * noise
                self.lbl_noise.setText(tr("Noise={value}", value=f"{noise_val:.1f}"))
            except Exception:
                self.lbl_noise.setText(tr("Noise={value}", value="0"))
