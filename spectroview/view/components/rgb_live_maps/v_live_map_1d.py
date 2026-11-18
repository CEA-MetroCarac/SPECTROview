"""1D plotting, scaling, legend, and styling mixin for LiveMapSubpanel."""

from __future__ import annotations

from typing import List, Optional, Tuple

from matplotlib.ticker import AutoLocator, MaxNLocator
import numpy as np

from spectroview import DEFAULT_COLORS
from spectroview.view.components.rgb_live_maps.common import (
    _get_cached_style_dict,
    _normalize_plot_theme,
    _resolve_style_path,
)
from spectroview.view.components.rgb_live_maps.v_live_maps_legend_editor import (
    LEGEND_LOCATIONS,
)


class LiveMap1DMixin:
    """1D line and multi-profile rendering, axis scaling, legend, and theme styling."""

    @staticmethod
    def _padded_ylim(val_arr: np.ndarray) -> Tuple[float, float]:
        ymin = float(np.nanmin(val_arr))
        ymax = float(np.nanmax(val_arr))
        span = ymax - ymin
        pad = span * 0.12 if span > 1e-12 else (abs(ymin) * 0.1 if abs(ymin) > 1e-12 else 1.0)
        return ymin - pad, ymax + pad

    def _curve_style_kwargs(self, is_primary: bool = True, multi: bool = False) -> dict:
        """Build uniform Matplotlib linestyle, linewidth, marker, and markersize from view options."""
        opts = self._view_options
        style = opts.get("plot_style", "line")
        lw = float(opts.get("line_width", 1.5))
        ds = float(opts.get("dot_size", 1.5))
        if style == "dot":
            return {"linestyle": "none", "linewidth": 0.0, "marker": "o", "markersize": ds}
        if style == "line":
            return {"linestyle": "-", "linewidth": lw, "marker": "", "markersize": 0.0}
        return {"linestyle": "-", "linewidth": lw, "marker": "o", "markersize": ds}

    def _apply_x_limits(self, x: np.ndarray, is_index_axis: bool):
        """Set x-limits/ticks for an index axis (Spectrum #) or a spatial profile axis."""
        if len(x) == 0:
            return
        n = len(x)
        if is_index_axis:
            if n == 1:
                self.ax.set_xlim(0.5, 1.5)
                self.ax.set_xticks([1])
            else:
                margin = max(0.25, 0.04 * (n - 1))
                self.ax.set_xlim(1.0 - margin, float(n) + margin)
                self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            return
        x_lo = float(np.nanmin(x))
        x_hi = float(np.nanmax(x))
        span = x_hi - x_lo
        margin = span * 0.03 if span > 1e-12 else 0.5
        self.ax.set_xlim(x_lo - margin, x_hi + margin)
        self.ax.xaxis.set_major_locator(AutoLocator())

    def _apply_1d_scales_and_limits(
        self,
        all_x: np.ndarray,
        all_y: np.ndarray,
        is_index_axis: bool,
    ):
        """Apply linear/log X and Y axis scaling with guards for non-positive domains."""
        opts = self._view_options
        req_xscale = opts.get("xscale", "linear")
        req_yscale = opts.get("yscale", "linear")

        finite_y = all_y[np.isfinite(all_y)]
        if finite_y.size > 0:
            self._valid_y_bounds = self._padded_ylim(finite_y)
            pos_y = finite_y[finite_y > 0]
            if pos_y.size > 0:
                p_min = float(np.min(pos_y))
                p_max = float(np.max(pos_y))
                if p_max / max(p_min, 1e-30) < 1.05:
                    self._valid_pos_y_bounds = (max(p_min * 0.7, 1e-300), p_max * 1.4)
                else:
                    self._valid_pos_y_bounds = (max(p_min * 0.85, 1e-300), p_max * 1.15)
            else:
                self._valid_pos_y_bounds = None
        else:
            self._valid_y_bounds = None
            self._valid_pos_y_bounds = None

        if req_yscale == "log" and self._valid_pos_y_bounds is not None:
            if self.ax.get_yscale() != "log":
                self.ax.set_yscale("log")
            self.ax.set_ylim(*self._valid_pos_y_bounds)
        else:
            if self.ax.get_yscale() != "linear":
                self.ax.set_yscale("linear")
            if self._valid_y_bounds is not None:
                self.ax.set_ylim(*self._valid_y_bounds)

        finite_x = all_x[np.isfinite(all_x)]
        if req_xscale == "log":
            pos_x = finite_x[finite_x > 0]
            if pos_x.size > 0:
                x_min = float(np.min(pos_x))
                x_max = float(np.max(pos_x))
                if x_max / max(x_min, 1e-30) < 1.05:
                    x_bounds = (max(x_min * 0.8, 1e-300), x_max * 1.25)
                else:
                    x_bounds = (max(x_min * 0.92, 1e-300), x_max * 1.08)
                x_sig = ("log", x_bounds[0], x_bounds[1])
                if self.ax.get_xscale() != "log" or self._styled_x_domain_sig != x_sig:
                    if self.ax.get_xscale() != "log":
                        self.ax.set_xscale("log")
                    self.ax.set_xlim(*x_bounds)
                    self._styled_x_domain_sig = x_sig
                return
        if self.ax.get_xscale() != "linear":
            self.ax.set_xscale("linear")
            self._styled_x_domain_sig = None
        x_target = finite_x if not is_index_axis else all_x
        if is_index_axis:
            x_sig = ("index", len(x_target))
        elif finite_x.size > 0:
            x_sig = ("linear", len(finite_x), float(np.min(finite_x)), float(np.max(finite_x)))
        else:
            x_sig = ("empty", 0)
        if self._styled_x_domain_sig != x_sig:
            self._apply_x_limits(x_target, is_index_axis)
            self._styled_x_domain_sig = x_sig

    def _apply_grid_state(self, show_grid: bool, theme: Optional[str] = None, force: bool = True):
        """Apply Matplotlib grid state cleanly without raising false+kwargs warnings."""
        active_theme = _normalize_plot_theme(theme if theme is not None else self._current_theme)
        sig = (bool(show_grid), active_theme if show_grid else None)
        if not force and self._styled_grid_sig == sig:
            return
        self._styled_grid_sig = sig
        if show_grid:
            _, style_path = _resolve_style_path(active_theme)
            grid_col = _get_cached_style_dict(style_path).get("grid.color", "#888888")
            self.ax.grid(True, alpha=0.25, linestyle="--", color=grid_col)
        else:
            self.ax.grid(False)

    def _apply_legend_theme_colors(self):
        """Ensure explicit high-contrast legend frame and label colors for the active plot theme."""
        if self._legend is None:
            return
        is_dark, _ = _resolve_style_path(self._current_theme)
        leg_fg = "#FFFFFF" if is_dark else "#111111"
        leg_bg = "#2A2A2A" if is_dark else "#FFFFFF"
        leg_edge = "#555555" if is_dark else "#CCCCCC"
        frame = self._legend.get_frame()
        if frame is not None:
            frame.set_facecolor(leg_bg)
            frame.set_edgecolor(leg_edge)
            frame.set_alpha(0.88)
        for text in self._legend.get_texts():
            text.set_color(leg_fg)

    def _current_legend_signature(self):
        opts = self._view_options
        labels_tuple = tuple(
            str(ln.get_label()) for ln in self.lines
            if ln.get_label() and not str(ln.get_label()).startswith("_")
        )
        colors_tuple = tuple(
            ln.get_color() for ln in self.lines
            if ln.get_label() and not str(ln.get_label()).startswith("_")
        )
        return (
            bool(opts.get("show_legend", False)),
            str(opts.get("legend_loc", "best")),
            max(1, int(opts.get("legend_ncol", 1))),
            max(6, int(opts.get("legend_fontsize", 8))),
            labels_tuple,
            colors_tuple,
        )

    def _sync_legend(self, force_rebuild: bool = False):
        """Create, update, or remove the Matplotlib legend box per current view options."""
        want_show = bool(self._view_options.get("show_legend", False))
        if not want_show or self._display_kind not in ("series", "multi_profile"):
            if self._legend is not None:
                try:
                    self._legend.remove()
                except Exception:
                    pass
                self._legend = None
            self._legend_signature = None
            return

        handles = []
        labels = []
        for ln in self.lines:
            lbl = ln.get_label()
            if lbl and not str(lbl).startswith("_"):
                handles.append(ln)
                labels.append(str(lbl))
        if not handles:
            if self._legend is not None:
                try:
                    self._legend.remove()
                except Exception:
                    pass
                self._legend = None
            self._legend_signature = None
            return

        sig = self._current_legend_signature()
        if not force_rebuild and self._legend is not None and sig == self._legend_signature:
            # Reuse existing legend and only refresh theme/handle styling in place
            self._apply_legend_theme_colors()
            return

        if self._legend is not None:
            try:
                self._legend.remove()
            except Exception:
                pass
            self._legend = None

        loc = self._view_options.get("legend_loc", "best")
        ncol = max(1, int(self._view_options.get("legend_ncol", 1)))
        fontsize = max(6, int(self._view_options.get("legend_fontsize", 8)))
        if loc == "outside right":
            self._legend = self.ax.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                ncol=ncol,
                fontsize=fontsize,
                frameon=True,
            )
        elif loc == "outside top":
            self._legend = self.ax.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=ncol,
                fontsize=fontsize,
                frameon=True,
            )
        else:
            if loc not in LEGEND_LOCATIONS:
                loc = "best"
            self._legend = self.ax.legend(
                handles,
                labels,
                loc=loc,
                ncol=ncol,
                fontsize=fontsize,
                frameon=True,
            )
            self._legend.set_draggable(True)

        self._legend_signature = sig
        self._apply_legend_theme_colors()

    def _apply_figure_theme(self, theme: Optional[str] = None, force: bool = True):
        """Explicitly paint Figure, Axes, ticks, spines, grid, legend, and colorbar for the active theme."""
        active_theme = _normalize_plot_theme(theme if theme is not None else self._current_theme)
        self._current_theme = active_theme
        grid_on = bool(self._display_kind is not None and self._view_options.get("show_grid", False))
        sig = (active_theme, self.cbar is not None, grid_on)
        if not force and self._styled_theme_sig == sig:
            self._apply_legend_theme_colors()
            return
        self._styled_theme_sig = sig
        _is_dark, style_path = _resolve_style_path(active_theme)
        s = _get_cached_style_dict(style_path)
        self.figure.patch.set_facecolor(s.get("figure.facecolor", "#FFFFFF"))
        self.figure.patch.set_edgecolor(s.get("figure.edgecolor", "white"))
        self.ax.set_facecolor(s.get("axes.facecolor", "#FFFFFF"))
        label_col = s.get("axes.labelcolor", "#000000")
        self.ax.xaxis.label.set_color(label_col)
        self.ax.yaxis.label.set_color(label_col)
        self.ax.title.set_color(s.get("text.color", "black"))
        tick_col = s.get("xtick.color", "#000000")
        self.ax.tick_params(axis="both", colors=tick_col)
        edge_col = s.get("axes.edgecolor", "#000000")
        for spine in self.ax.spines.values():
            spine.set_edgecolor(edge_col)
        if self.cbar is not None:
            try:
                self.cbar.ax.tick_params(colors=s.get("ytick.color", tick_col))
                self.cbar.outline.set_edgecolor(edge_col)
            except Exception:
                pass
        if grid_on:
            self._apply_grid_state(True, active_theme, force=force)
        self._apply_legend_theme_colors()

    def _apply_text_sizes(self, force: bool = True):
        """Apply user-configured text size to title, axis labels, ticks, and colorbar ticks."""
        text_fs = max(6, int(self._view_options.get("text_fontsize", 10)))
        sig = (text_fs, self.cbar is not None)
        if not force and self._styled_text_fs_sig == sig:
            return
        self._styled_text_fs_sig = sig
        tick_fs = max(6, text_fs - 1)
        self.ax.title.set_fontsize(text_fs)
        self.ax.xaxis.label.set_fontsize(text_fs)
        self.ax.yaxis.label.set_fontsize(text_fs)
        self.ax.tick_params(axis="both", labelsize=tick_fs)
        if self.cbar is not None:
            try:
                self.cbar.ax.tick_params(labelsize=tick_fs)
            except Exception:
                pass

    def _sync_title(self, has_content: Optional[bool] = None):
        """Show or clear the Matplotlib axes title label per ``show_figure_label``."""
        if has_content is None:
            has_content = bool(self.img is not None or self.lines)
        want_label = bool(self._view_options.get("show_figure_label", False)) and bool(has_content)
        target = f"{self.color_name}: {self.cbb_param.currentText()}" if want_label else ""
        if self.ax.get_title() != target:
            text_fs = max(6, int(self._view_options.get("text_fontsize", 10)))
            self.ax.set_title(target, fontsize=text_fs, pad=4)

    def _sync_1d_line_styles(self):
        """Update curve linestyle, linewidth, marker, markersize, and legend handles in place."""
        if not self.lines:
            return
        multi = self._display_kind == "multi_profile"
        entries = self._multi_entries if multi else []
        for idx, ln in enumerate(self.lines):
            is_primary = bool(entries[idx].get("is_primary", False)) if idx < len(entries) else True
            kw = self._curve_style_kwargs(is_primary=is_primary, multi=multi)
            ln.set_linestyle(kw["linestyle"])
            ln.set_linewidth(kw["linewidth"])
            ln.set_marker(kw["marker"])
            ln.set_markersize(kw["markersize"])
        if self._legend is not None:
            leg_lines = self._legend.get_lines()
            for idx, leg_ln in enumerate(leg_lines):
                if idx < len(self.lines):
                    src = self.lines[idx]
                    leg_ln.set_linestyle(src.get_linestyle())
                    leg_ln.set_linewidth(src.get_linewidth())
                    leg_ln.set_marker(src.get_marker())
                    leg_ln.set_markersize(src.get_markersize())
                    leg_ln.set_color(src.get_color())

    def update_series(
        self,
        values: Optional[np.ndarray],
        labels: Optional[list[str]] = None,
        status_msg: str = "",
        theme: Optional[str] = None,
        x_values: Optional[np.ndarray] = None,
        x_label: str = "Spectrum #",
    ):
        """Render or update a 1D line plot.

        Without ``x_values`` the values are plotted against the spectrum index
        (Spectra workspace). With ``x_values`` they are plotted against those
        positions, e.g. the spatial coordinate of a 1D profile map.
        """
        val_arr = np.asarray(values, dtype=np.float64) if values is not None else np.array([], dtype=np.float64)
        is_index_axis = x_values is None
        if is_index_axis:
            x_arr = np.arange(1, len(val_arr) + 1, dtype=np.float64)
        else:
            x_arr = np.asarray(x_values, dtype=np.float64)
            if len(x_arr) != len(val_arr):
                val_arr = np.array([], dtype=np.float64)
        self._series_values = val_arr
        self._series_labels = list(labels) if labels else []
        self._series_x = None if is_index_axis else x_arr
        self._series_xlabel = x_label
        self._multi_entries = []
        self._status_msg = status_msg
        active_theme = _normalize_plot_theme(
            theme if theme is not None else self._view_options.get("plot_theme", self._current_theme)
        )
        self._current_theme = active_theme

        param_name = self.cbb_param.currentText()
        title_text = f"{self.color_name}: {param_name}"
        custom_labels = self._view_options.get("custom_labels", {})
        custom_colors = self._view_options.get("custom_colors", {})
        curve_label = custom_labels.get(self.color_id) or title_text
        curve_color = custom_colors.get(self.color_id) or self.color_hex
        style_kw = self._curve_style_kwargs(is_primary=True, multi=False)
        show_grid = bool(self._view_options.get("show_grid", False))

        has_valid = len(val_arr) > 0 and bool(np.any(np.isfinite(val_arr)))
        plot_vals = self._apply_normalization(x_arr, val_arr) if has_valid else val_arr
        text_fs = max(6, int(self._view_options.get("text_fontsize", 10)))

        # Fast path: in-place single-line update during continuous bracket dragging
        if (
            self._display_kind == "series"
            and len(self.lines) == 1
            and self.line is not None
            and has_valid
            and len(self.line.get_xdata()) == len(plot_vals)
        ):
            self._sync_1d_controls_visibility()
            self.line.set_data(x_arr, plot_vals)
            self.line.set_color(curve_color)
            self.line.set_linestyle(style_kw["linestyle"])
            self.line.set_linewidth(style_kw["linewidth"])
            self.line.set_marker(style_kw["marker"])
            self.line.set_markersize(style_kw["markersize"])
            self.line.set_label(curve_label)
            finite_mask = np.isfinite(x_arr) & np.isfinite(plot_vals)
            self._apply_1d_scales_and_limits(
                x_arr[finite_mask] if np.any(finite_mask) else x_arr,
                plot_vals[finite_mask] if np.any(finite_mask) else plot_vals,
                is_index_axis=is_index_axis,
            )
            self._apply_grid_state(show_grid, active_theme, force=False)
            if self.ax.get_xlabel() != x_label:
                self.ax.set_xlabel(x_label)
            if self.ax.get_ylabel() != param_name:
                self.ax.set_ylabel(param_name)
            self._sync_title(has_content=True)
            self._apply_figure_theme(active_theme, force=False)
            self._apply_text_sizes(force=False)
            self._sync_legend()
            self._update_selection_overlay(draw=False)
            self.canvas.draw_idle()
            return

        self._display_kind = "series"
        self._sync_palette_visibility()
        self._reset_axes()
        self._apply_figure_theme(active_theme)

        if not has_valid:
            self._valid_y_bounds = None
            self._valid_pos_y_bounds = None
            msg = status_msg or ("No spectra selected" if is_index_axis else "No profile data available")
            self.ax.text(
                0.5, 0.5, msg,
                ha="center", va="center", transform=self.ax.transAxes,
                fontsize=text_fs, color="gray", wrap=True
            )
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw_idle()
            return

        self.lines = list(
            self.ax.plot(
                x_arr,
                plot_vals,
                color=curve_color,
                alpha=0.95,
                label=curve_label,
                **style_kw,
            )
        )

        finite_mask = np.isfinite(x_arr) & np.isfinite(plot_vals)
        self._apply_1d_scales_and_limits(
            x_arr[finite_mask] if np.any(finite_mask) else x_arr,
            plot_vals[finite_mask] if np.any(finite_mask) else plot_vals,
            is_index_axis=is_index_axis,
        )
        self._apply_grid_state(show_grid, active_theme)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(param_name)
        self._sync_title(has_content=True)
        self._apply_figure_theme(active_theme)
        self._apply_text_sizes()
        self._sync_legend()
        self._update_selection_overlay(draw=False)
        try:
            self.toolbar.update()
        except Exception:
            pass
        self.canvas.draw_idle()

    def update_multi_profile(
        self,
        entries: List[dict],
        shared_x_label: str = "X",
        status_msg: str = "",
        theme: Optional[str] = None,
    ):
        """Render or update multiple stacked 1D spatial profiles in one subpanel."""
        self._multi_entries = list(entries) if entries else []
        self._multi_x_label = shared_x_label
        self._series_values = None
        self._series_labels = []
        self._series_x = None
        self._status_msg = status_msg
        active_theme = _normalize_plot_theme(
            theme if theme is not None else self._view_options.get("plot_theme", self._current_theme)
        )
        self._current_theme = active_theme

        param_name = self.cbb_param.currentText()
        x_label = (
            shared_x_label
            if shared_x_label.endswith("(mm)")
            else f"{shared_x_label or 'Position'} (mm)"
        )

        # Compute finite bounds across normalized curves without mutating cached raw entries
        plotted_curves = []
        finite_x_parts = []
        finite_y_parts = []
        for entry in self._multi_entries:
            px = np.asarray(entry.get("positions", []), dtype=np.float64)
            raw_vy = np.asarray(entry.get("values", []), dtype=np.float64)
            vy = self._apply_normalization(px, raw_vy) if len(px) == len(raw_vy) else raw_vy
            plotted_curves.append((entry, px, vy))
            if len(px) == 0 or len(px) != len(vy):
                continue
            mask = np.isfinite(px) & np.isfinite(vy)
            if np.any(mask):
                finite_x_parts.append(px[mask])
                finite_y_parts.append(vy[mask])

        has_valid = len(finite_y_parts) > 0
        if has_valid:
            all_x = np.concatenate(finite_x_parts)
            all_y = np.concatenate(finite_y_parts)
        else:
            all_x = np.array([], dtype=np.float64)
            all_y = np.array([], dtype=np.float64)

        custom_labels = self._view_options.get("custom_labels", {})
        custom_colors = self._view_options.get("custom_colors", {})
        show_grid = bool(self._view_options.get("show_grid", False))
        text_fs = max(6, int(self._view_options.get("text_fontsize", 10)))

        # Fast path: reuse existing Line2D pool when entry count matches
        if (
            self._display_kind == "multi_profile"
            and has_valid
            and self.img is None
            and len(self.lines) == len(plotted_curves)
            and len(self.lines) > 0
        ):
            self._sync_1d_controls_visibility()
            for idx, (ln, (entry, px, vy)) in enumerate(zip(self.lines, plotted_curves)):
                key = entry.get("name", f"Map {idx + 1}")
                auto_color = (
                    self.color_hex
                    if len(plotted_curves) == 1
                    else DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
                )
                color = custom_colors.get(key) or auto_color
                is_primary = bool(entry.get("is_primary", False))
                style_kw = self._curve_style_kwargs(is_primary=is_primary, multi=True)
                ln.set_data(px, vy)
                ln.set_color(color)
                ln.set_linestyle(style_kw["linestyle"])
                ln.set_linewidth(style_kw["linewidth"])
                ln.set_marker(style_kw["marker"])
                ln.set_markersize(style_kw["markersize"])
                ln.set_alpha(1.0 if is_primary else 0.78)
                ln.set_zorder(5 if is_primary else 3)
                ln.set_label(custom_labels.get(key) or key)
            self._apply_1d_scales_and_limits(all_x, all_y, is_index_axis=False)
            self._apply_grid_state(show_grid, active_theme, force=False)
            if self.ax.get_xlabel() != x_label:
                self.ax.set_xlabel(x_label)
            if self.ax.get_ylabel() != param_name:
                self.ax.set_ylabel(param_name)
            self._sync_title(has_content=True)
            self._apply_figure_theme(active_theme, force=False)
            self._apply_text_sizes(force=False)
            self._sync_legend()
            self._update_selection_overlay(draw=False)
            self.canvas.draw_idle()
            return

        self._display_kind = "multi_profile"
        self._sync_palette_visibility()
        self._reset_axes()
        self._apply_figure_theme(active_theme)

        if not has_valid:
            self._valid_y_bounds = None
            self._valid_pos_y_bounds = None
            msg = status_msg or "No valid profile data within selected range"
            self.ax.text(
                0.5, 0.5, msg,
                ha="center", va="center", transform=self.ax.transAxes,
                fontsize=text_fs, color="gray", wrap=True,
            )
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw_idle()
            return

        new_lines = []
        for idx, (entry, px, vy) in enumerate(plotted_curves):
            key = entry.get("name", f"Map {idx + 1}")
            auto_color = (
                self.color_hex
                if len(plotted_curves) == 1
                else DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
            )
            color = custom_colors.get(key) or auto_color
            is_primary = bool(entry.get("is_primary", False))
            style_kw = self._curve_style_kwargs(is_primary=is_primary, multi=True)
            created = self.ax.plot(
                px,
                vy,
                color=color,
                alpha=1.0 if is_primary else 0.78,
                zorder=5 if is_primary else 3,
                label=custom_labels.get(key) or key,
                **style_kw,
            )
            new_lines.extend(created)
        self.lines = new_lines

        self._apply_1d_scales_and_limits(all_x, all_y, is_index_axis=False)
        self._apply_grid_state(show_grid, active_theme)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(param_name)
        self._sync_title(has_content=True)
        self._apply_figure_theme(active_theme)
        self._apply_text_sizes()
        self._sync_legend()
        self._update_selection_overlay(draw=False)
        try:
            self.toolbar.update()
        except Exception:
            pass
        self.canvas.draw_idle()
