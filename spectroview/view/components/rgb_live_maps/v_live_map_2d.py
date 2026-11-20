"""2D heatmap rendering mixin for LiveMapSubpanel."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np

from spectroview.i18n import tr
from spectroview.view.components.rgb_live_maps.common import _normalize_plot_theme


class LiveMap2DMixin:
    """2D spatial heatmap rendering, color normalization, and colormap updates."""

    def _on_palette_changed(self, palette_name: str):
        """Update colormap without recomputing grid data."""
        if self.img is not None and self.canvas is not None:
            try:
                cmap = plt.get_cmap(palette_name).copy()
                cmap.set_bad(color="none")
                self.img.set_cmap(cmap)
                self.canvas.draw_idle()
            except Exception:
                pass

    def _resolve_2d_render_params(self, grid: np.ndarray, extent: list[float], vmin: float, vmax: float):
        """Compute normalized grid, norm, interpolation, axis scales, and aspect for 2D heatmaps."""
        opts = self._view_options
        interp = "bilinear" if opts.get("heatmap_smoothing", False) else "none"
        use_log_z = opts.get("zscale", "linear") == "log"
        norm = None
        display_grid = grid

        if use_log_z and vmax > 0:
            pos_vals = grid[np.isfinite(grid) & (grid > 0)]
            if pos_vals.size > 0:
                pos_min = float(np.min(pos_vals))
                safe_vmin = max(vmin, pos_min) if vmin > 0 else pos_min
                if safe_vmin >= vmax:
                    safe_vmin = pos_min * 0.9 if pos_min > 0 else 1e-12
                display_grid = np.where(np.isfinite(grid) & (grid > 0), grid, np.nan)
                norm = LogNorm(vmin=max(safe_vmin, 1e-300), vmax=max(vmax, safe_vmin * 1.01))

        if norm is None:
            norm = Normalize(vmin=vmin, vmax=vmax)

        req_xscale = "log" if (opts.get("xscale", "linear") == "log" and len(extent) >= 2 and extent[1] > 0) else "linear"
        req_yscale = "log" if (opts.get("yscale", "linear") == "log" and len(extent) >= 4 and extent[3] > 0) else "linear"
        aspect = "equal" if (req_xscale == "linear" and req_yscale == "linear") else "auto"
        return display_grid, norm, interp, req_xscale, req_yscale, aspect

    def _apply_2d_axis_scales(self, extent: list[float], xscale: str, yscale: str, show_grid: bool):
        """Set linear/log scales and valid domain bounds on 2D heatmap axes."""
        if xscale == "log":
            self.ax.set_xscale("log")
            x_hi = float(extent[1])
            x_lo = float(extent[0]) if float(extent[0]) > 0 else max(x_hi * 1e-3, 1e-6)
            self.ax.set_xlim(x_lo, x_hi)
        else:
            self.ax.set_xscale("linear")
            self.ax.set_xlim(extent[0], extent[1])

        if yscale == "log":
            self.ax.set_yscale("log")
            y_hi = float(extent[3])
            y_lo = float(extent[2]) if float(extent[2]) > 0 else max(y_hi * 1e-3, 1e-6)
            self.ax.set_ylim(y_lo, y_hi)
        else:
            self.ax.set_yscale("linear")
            self.ax.set_ylim(extent[2], extent[3])

        self._apply_grid_state(show_grid)

    def update_map(
        self,
        grid: Optional[np.ndarray],
        extent: list[float],
        vmin: float,
        vmax: float,
        status_msg: str = "",
        theme: Optional[str] = None,
    ):
        """Render or update the 2D heatmap on the canvas."""
        self._grid = grid
        self._extent = extent
        self._vmin = vmin
        self._vmax = vmax
        self._series_values = None
        self._multi_entries = []
        self._valid_y_bounds = None
        self._valid_pos_y_bounds = None
        self._status_msg = status_msg
        active_theme = _normalize_plot_theme(
            theme if theme is not None else self._view_options.get("plot_theme", self._current_theme)
        )
        self._current_theme = active_theme

        want_colorbar = bool(self._view_options.get("show_colorbar", True))
        show_grid = bool(self._view_options.get("show_grid", False))
        has_valid_grid = grid is not None and not np.all(np.isnan(grid))
        text_fs = max(6, int(self._view_options.get("text_fontsize", 10)))

        # Fast path: in-place update when image and colorbar presence match expectations
        if (
            self._display_kind == "map"
            and self.img is not None
            and (self.cbar is not None) == want_colorbar
            and has_valid_grid
        ):
            display_grid, norm, interp, xscale, yscale, aspect = self._resolve_2d_render_params(
                grid, extent, vmin, vmax
            )
            self.img.set_data(display_grid)
            self.img.set_extent(extent)
            self.img.set_norm(norm)
            self.img.set_interpolation(interp)
            self.ax.set_aspect(aspect, adjustable="box")
            self._apply_2d_axis_scales(extent, xscale, yscale, show_grid)
            self._sync_title(has_content=True)
            if self.cbar is not None:
                self.cbar.update_normal(self.img)
            self._apply_figure_theme(active_theme)
            self._apply_text_sizes()
            self._update_selection_overlay(draw=False)
            self.canvas.draw_idle()
            return

        # Re-render path for status messages, colorbar toggle, or first draw
        self._display_kind = "map"
        self._sync_palette_visibility()
        self._reset_axes()
        self._apply_figure_theme(active_theme)

        if not has_valid_grid:
            msg = status_msg or tr("No map data available")
            self.ax.text(
                0.5, 0.5, msg,
                ha="center", va="center", transform=self.ax.transAxes,
                fontsize=text_fs, color="gray", wrap=True
            )
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw_idle()
            return

        cmap_name = self.cbb_palette.currentText()
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color="none")

        display_grid, norm, interp, xscale, yscale, aspect = self._resolve_2d_render_params(
            grid, extent, vmin, vmax
        )
        self.img = self.ax.imshow(
            display_grid,
            extent=extent,
            origin="lower",
            aspect=aspect,
            cmap=cmap,
            norm=norm,
            interpolation=interp,
        )

        if want_colorbar:
            self.cbar = self.figure.colorbar(self.img, ax=self.ax, fraction=0.046, pad=0.04)

        self._apply_2d_axis_scales(extent, xscale, yscale, show_grid)
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self._sync_title(has_content=True)
        self._apply_figure_theme(active_theme)
        self._apply_text_sizes()
        self._update_selection_overlay(draw=False)
        try:
            self.toolbar.update()
        except Exception:
            pass
        self.canvas.draw_idle()
