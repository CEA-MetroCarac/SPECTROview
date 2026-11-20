"""Heatmap, wafer map, stats box, selection overlay, and height-profile rendering."""

from __future__ import annotations

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
import numpy as np

from spectroview import PLOT_POLICY_DARK, PLOT_POLICY_LIGHT
from spectroview.i18n import combo_data_or_text, tr
from spectroview.view.components.common.v_notification import show_notification
from spectroview.view.theme import current_tokens


class MapPlotMixin:
    """Matplotlib rendering, fast-path updates, stats overlay, and profile line drawing."""

    def apply_plot_style(self, style_name=None):
        """Update figure and axes colors using local style context."""
        if not hasattr(self, 'figure') or not hasattr(self, 'ax'):
            return

        if style_name is not None:
            self.current_style = style_name

        # Manually replot heatmap to update colorbars and contents with new style
        if self.map_df is None or self.map_df.empty:
            self._do_plot_heatmap()
        else:
            self.plot_heatmap()

    def _update_selection_overlay(self):
        """Update selection overlay without full plot redraw (fast).

        Uses PatchCollection for O(1) rendering instead of individual add_patch() calls.
        """
        if self._selector is not None:
            self._selector.clear_overlay()

        # Remove old profile lines
        if self._profile_line is not None:
            try:
                self._profile_line.remove()
            except Exception:
                pass
            self._profile_line = None

        if self._profile_height_line is not None:
            try:
                self._profile_height_line.remove()
            except Exception:
                pass
            self._profile_height_line = None

        # Add new selection overlay
        if self.selected_points:
            try:
                dx = getattr(self, '_current_dx', None)
                dy = getattr(self, '_current_dy', None)
                if self._selector is not None:
                    self._selector.ax = self.ax
                    self._selector.draw_2d_overlay(self.selected_points, dx, dy, draw=False)

                # Plot profile if exactly 2 points selected and in 2D map mode
                map_type = combo_data_or_text(self.cbb_map_type)
                if len(self.selected_points) == 2 and map_type == '2Dmap':
                    self._plot_height_profile_on_map()

            except Exception as e:
                print(f"Error drawing selection overlay: {e}")

        self.canvas.draw_idle()

    def plot_heatmap(self):
        """Schedule a debounced heatmap plot update."""
        self.plot_timer.start()

    def _do_plot_heatmap(self):
        """Wrapper to plot using local style context, isolating from global rcParams."""
        if self.current_style == 'Dark Mode':
            style_path = PLOT_POLICY_DARK
        else:
            style_path = PLOT_POLICY_LIGHT
        with plt.style.context(style_path):
            self._do_plot_heatmap_internal()

    def _do_plot_heatmap_internal(self):
        """Plot 2D heatmap or wafer map based on current data and settings."""
        if self.map_df is None or self.map_df.empty:
            self.figure.clf()
            fig_bg = plt.rcParams.get('figure.facecolor', 'white')
            self.figure.patch.set_facecolor(fig_bg)
            if hasattr(self, "canvas_card"):
                tok = current_tokens()
                fig_hex = fig_bg if str(fig_bg).startswith("#") or str(fig_bg).isalpha() else f"#{fig_bg}"
                self.canvas_card.setStyleSheet(
                    f"QFrame#canvasCard {{ background-color: {fig_hex}; border: 1px solid {tok.border}; border-radius: 8px; }}"
                )
            self.ax = self.figure.add_subplot(111)
            self.img = None
            self.cbar = None
            if self._selector is not None:
                self._selector.ax = self.ax
                self._selector.selection_collection = None
                self._selector.rect_patch = None
                self._selector.rect_start = None
            self._profile_line = None
            self._profile_height_line = None
            self._last_stats_text_artist = None
            self.canvas.draw_idle()
            return

        map_type = combo_data_or_text(self.cbb_map_type)
        show_stats = self.action_show_stats.isChecked() if map_type != '2Dmap' else False
        show_grid = self.action_grid.isChecked()

        if getattr(self, '_need_z_range_update', False):
            self._need_z_range_update = False
            self._update_z_range()

        cache_key = self._build_cache_key()
        last_rendered_key = getattr(self, '_last_rendered_cache_key', None)

        # Fast path: update existing Artists in-place without rebuilding Figure / tight_layout
        can_fast_update = (
            getattr(self, 'ax', None) is not None
            and getattr(self, 'img', None) is not None
            and getattr(self, 'cbar', None) is not None
            and getattr(self, '_last_rendered_map_name', None) == self.map_df_name
            and getattr(self, '_last_map_type', None) == map_type
            and getattr(self, '_last_show_stats', None) == show_stats
            and getattr(self, '_last_grid', None) == show_grid
            and getattr(self, '_last_style', None) == self.current_style
            and not show_stats  # full redraw if stats box is enabled
        )

        if can_fast_update:
            try:
                vmin_plot, vmax_plot = self.z_range_slider.value()

                # ULTRA-FAST PATH: if heatmap data parameters did not change,
                # only update clim/norm and colorbar without touching any data or collections
                if last_rendered_key == cache_key:
                    if combo_data_or_text(self.cbb_color_scale) == "Log" and vmin_plot is not None and vmin_plot > 0:
                        self.img.set_norm(LogNorm(vmin=vmin_plot, vmax=vmax_plot))
                    else:
                        self.img.set_norm(None)
                        self.img.set_clim(vmin_plot, vmax_plot)
                    self.cbar.update_normal(self.img)
                    self.canvas.draw_idle()
                    return

                heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col = self._get_data_for_heatmap()
                self._last_final_z_col = final_z_col
                self._last_rendered_cache_key = cache_key

                cmap = plt.get_cmap(self.cbb_palette.currentText()).copy()
                cmap.set_bad(color='none')
                interpolation = 'bilinear' if self.action_smoothing.isChecked() else 'none'

                if map_type != '2Dmap' and grid_z is not None:
                    img_data = grid_z
                    if hasattr(self, '_scatter_sites') and self._scatter_sites is not None:
                        all_x, all_y = self._get_measurement_sites()
                        if len(all_x) > 0:
                            self._scatter_sites.set_offsets(np.column_stack([all_x, all_y]))
                elif not heatmap_pivot.empty:
                    img_data = heatmap_pivot.values if hasattr(heatmap_pivot, 'values') else np.asarray(heatmap_pivot)
                elif isinstance(self.img, PatchCollection):
                    if len(self.img.get_paths()) != len(final_z_col):
                        raise ValueError("PatchCollection count mismatch")
                    img_data = None
                    self.img.set_cmap(cmap)
                    if combo_data_or_text(self.cbb_color_scale) == "Log" and vmin_plot is not None and vmin_plot > 0:
                        self.img.set_norm(LogNorm(vmin=vmin_plot, vmax=vmax_plot))
                    else:
                        self.img.set_norm(None)
                        self.img.set_clim(vmin_plot, vmax_plot)
                    self.img.set_array(final_z_col)
                    self.ax.set_xlim(extent[0], extent[1])
                    self.ax.set_ylim(extent[2], extent[3])
                    self.cbar.update_normal(self.img)
                    title = self.cbb_zparameter.currentText()
                    self.ax.set_title(title, fontsize=13)
                    self._update_selection_overlay()
                    self.canvas.draw_idle()
                    return
                else:
                    img_data = None

                if img_data is not None and hasattr(self.img, 'set_data'):
                    self.img.set_cmap(cmap)
                    self.img.set_interpolation(interpolation)
                    if combo_data_or_text(self.cbb_color_scale) == "Log" and vmin_plot is not None and vmin_plot > 0:
                        self.img.set_norm(LogNorm(vmin=vmin_plot, vmax=vmax_plot))
                    else:
                        self.img.set_norm(None)
                        self.img.set_clim(vmin_plot, vmax_plot)
                    self.img.set_data(img_data)
                    self.img.set_extent(extent)
                    self.ax.set_xlim(extent[0], extent[1])
                    self.ax.set_ylim(extent[2], extent[3])
                    self.cbar.update_normal(self.img)
                    title = self.cbb_zparameter.currentText()
                    self.ax.set_title(title, fontsize=13)
                    self._update_selection_overlay()
                    self.canvas.draw_idle()
                    return
            except Exception:
                # If anything fails during fast path, seamlessly fall through to full redraw
                pass

        # Clear figure and recreate axes to ensure no leftover colorbars and reset sizes
        self.figure.clf()
        fig_bg = plt.rcParams.get('figure.facecolor', 'white')
        self.figure.patch.set_facecolor(fig_bg)
        if hasattr(self, "canvas_card"):
            tok = current_tokens()
            fig_hex = fig_bg if str(fig_bg).startswith("#") or str(fig_bg).isalpha() else f"#{fig_bg}"
            self.canvas_card.setStyleSheet(
                f"QFrame#canvasCard {{ background-color: {fig_hex}; border: 1px solid {tok.border}; border-radius: 8px; }}"
            )
        self.ax = self.figure.add_subplot(111)
        self.img = None
        self.cbar = None
        if self._selector is not None:
            self._selector.ax = self.ax
            self._selector.selection_collection = None
            self._selector.rect_patch = None
            self._selector.rect_start = None
        self._profile_line = None
        self._profile_height_line = None
        self._last_stats_text_artist = None
        self._last_rendered_map_name = self.map_df_name
        self._last_map_type = map_type
        self._last_show_stats = show_stats
        self._last_grid = show_grid
        self._last_style = self.current_style
        self._last_rendered_cache_key = cache_key
        self._wafer_circle = None
        self._scatter_sites = None

        # Plot wafer circle for wafer maps
        if map_type != '2Dmap':
            r = self._get_wafer_radius(map_type)
            if r:
                edge_color = plt.rcParams.get('axes.edgecolor', 'black')
                self._wafer_circle = patches.Circle((0, 0), radius=r, fill=False,
                                             color=edge_color, linewidth=1)
                self.ax.add_patch(self._wafer_circle)

                # Show all measurement sites
                all_x, all_y = self._get_measurement_sites()
                self._scatter_sites = self.ax.scatter(all_x, all_y, marker='x', color='gray', s=15)

                # Remove X labels for wafer, keep Y
                self.ax.tick_params(axis='x', which='both', bottom=False,
                                   top=False, labelbottom=False)
                self.ax.tick_params(axis='y', which='both', left=True,
                                   right=False, labelleft=True)
        else:
            # 2D map: show both axes
            self.ax.tick_params(axis='x', which='both', bottom=True,
                               top=False, labelbottom=True)
            self.ax.tick_params(axis='y', which='both', left=True,
                               right=False, labelleft=True)

        # Get heatmap data and plot (consolidated error handling)
        try:
            # Get heatmap data (also updates self._current_dx and self._current_dy)
            heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col = self._get_data_for_heatmap()
            self._last_final_z_col = final_z_col

            # Plot settings
            cmap = plt.get_cmap(self.cbb_palette.currentText()).copy()
            # Set NaN values to transparent so they don't hide the background
            cmap.set_bad(color='none')

            interpolation = 'bilinear' if self.action_smoothing.isChecked() else 'none'
            vmin_plot, vmax_plot = self.z_range_slider.value()
            color_kwargs = self._build_color_kwargs(vmin_plot, vmax_plot)

            # Plot heatmap - different approaches for wafer vs 2D maps
            if map_type != '2Dmap' and grid_z is not None:
                # Wafer maps: Use griddata result with user-selected interpolation (default 'none' / nearest-neighbor)
                self.img = self.ax.imshow(grid_z, extent=extent,
                                         origin='lower', aspect='equal', cmap=cmap,
                                         interpolation=interpolation,
                                         **color_kwargs)
            elif not heatmap_pivot.empty:
                # 2D maps (regular grid): Use pivot table with optional smoothing
                self.img = self.ax.imshow(heatmap_pivot, extent=extent,
                                         origin='lower', aspect='equal', cmap=cmap,
                                         interpolation=interpolation, **color_kwargs)
            else:
                # 2D maps (irregular / scattered grid, e.g. wafer files or non-uniform coordinates in 2Dmap mode):
                # Render square color pixels centered at actual (x, y) coordinates to eliminate coordinate drift.
                x_col, y_col = self._get_measurement_sites()
                dx = getattr(self, '_current_dx', None)
                dy = getattr(self, '_current_dy', None)
                if dx is None or dy is None or dx <= 0 or dy <= 0:
                    geom = self._get_2d_spatial_geometry()
                    dx, dy = geom['dx'], geom['dy']
                    self._current_dx = float(dx)
                    self._current_dy = float(dy)
                rects = [
                    patches.Rectangle((x - dx / 2, y - dy / 2), dx, dy)
                    for x, y in zip(x_col, y_col)
                ]
                norm = color_kwargs.get('norm', None)
                vmin = color_kwargs.get('vmin', None)
                vmax = color_kwargs.get('vmax', None)

                collection = PatchCollection(
                    rects, cmap=cmap, norm=norm, edgecolors='none'
                )
                if norm is None and vmin is not None and vmax is not None:
                    collection.set_clim(vmin, vmax)
                collection.set_array(final_z_col)

                # Enable mouseover for live coordinate and cursor data display
                collection.set_mouseover(True)

                def _get_cursor_data(event, _xs=x_col, _ys=y_col, _dx=dx, _dy=dy):
                    zs = getattr(self, '_last_final_z_col', None)
                    if event.xdata is None or event.ydata is None or len(_xs) == 0 or zs is None or len(zs) != len(_xs):
                        return None
                    dists_sq = (_xs - event.xdata) ** 2 + (_ys - event.ydata) ** 2
                    idx = int(np.argmin(dists_sq))
                    if abs(_xs[idx] - event.xdata) <= _dx / 2 and abs(_ys[idx] - event.ydata) <= _dy / 2:
                        same_site = np.where((_xs == _xs[idx]) & (_ys == _ys[idx]))[0]
                        if len(same_site) > 1:
                            valid_site = same_site[~np.isnan(zs[same_site])]
                            if len(valid_site) > 0:
                                return zs[valid_site[-1]]
                        return zs[idx]
                    return None

                collection.get_cursor_data = _get_cursor_data

                self.ax.add_collection(collection)
                self.ax.set_xlim(extent[0], extent[1])
                self.ax.set_ylim(extent[2], extent[3])
                self.ax.set_aspect('equal')
                self.img = collection

        except (ValueError, TypeError) as e:
            # Handle non-numeric data errors (from both scipy griddata and matplotlib imshow)
            error_msg = str(e)
            if "could not convert string to float" in error_msg or "dtype object cannot be converted to float" in error_msg:
                show_notification(
                    tr("Selected parameter contains non-numeric values and cannot be plotted."),
                    notification_type="warning",
                    title=tr("Non-Numeric Data"),
                )
            else:
                show_notification(
                    tr("Error plotting heatmap: {error}", error=error_msg),
                    notification_type="error",
                    title=tr("Plot Error"),
                )
            self.ax.clear()
            self.canvas.draw_idle()
            return
        except Exception as e:
            # Handle any other errors
            show_notification(
                tr("Error plotting heatmap: {error}", error=str(e)),
                notification_type="error",
                title=tr("Plot Error"),
            )
            self.ax.clear()
            self.canvas.draw_idle()
            return

        # Colorbar
        if self.img:
            if hasattr(self, 'cbar') and self.cbar is not None:
                try:
                    self.cbar.remove()
                except Exception:
                    pass
            self.cbar = self.ax.figure.colorbar(self.img, ax=self.ax)

        # Grid
        if self.action_grid.isChecked():
            self.ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

        # Show stats (wafer only)
        if map_type != '2Dmap' and self.action_show_stats.isChecked():
            self._draw_stats_box()

        # Title
        title = self.cbb_zparameter.currentText()
        self.ax.set_title(title, fontsize=13)

        self.ax.get_figure().tight_layout()
        self.canvas.draw_idle()

        # Update selection overlay after plot is complete
        # (separate method handles adding/removing selection highlights)
        self._update_selection_overlay()

    def _build_color_kwargs(self, vmin, vmax):
        """imshow color kwargs for the selected color scale.

        Returns a LogNorm when 'Log' is chosen and the range is strictly
        positive; otherwise plain vmin/vmax (linear). Log requires positive
        values -- intensity/area are clipped >=0 and fit parameters can be
        <=0 -- so fall back to linear rather than raising when vmin <= 0.
        """
        if combo_data_or_text(self.cbb_color_scale) == "Log" and vmin is not None and vmin > 0:
            return dict(norm=LogNorm(vmin=vmin, vmax=vmax))
        return dict(vmin=vmin, vmax=vmax)

    def _draw_stats_box(self):
        """Draw statistics text box on the plot (wafer mode)."""
        # Remove old stats
        if self._last_stats_text_artist is not None:
            try:
                self._last_stats_text_artist.remove()
            except Exception:
                pass
            self._last_stats_text_artist = None

        if self._last_final_z_col is None:
            return

        arr = np.asarray(self._last_final_z_col, dtype=float)
        arr_clean = arr[~np.isnan(arr)]
        if len(arr_clean) == 0:
            return

        mean = float(np.mean(arr_clean))
        mn = float(np.min(arr_clean))
        mx = float(np.max(arr_clean))
        std = float(np.std(arr_clean, ddof=1)) if len(arr_clean) > 1 else 0.0
        sigma3 = 3 * std

        txt = (
            f"{tr('mean: {val}', val=f'{mean:.2f}')}\n"
            f"{tr('min: {val}', val=f'{mn:.2f}')}\n"
            f"{tr('max: {val}', val=f'{mx:.2f}')}\n"
            f"3σ: {sigma3:.2f}"
        )

        # Place text box on axes
        self._last_stats_text_artist = self.ax.text(
            0.02, 0.98, txt,
            transform=self.ax.transAxes,
            fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="black")
        )

    def _plot_height_profile_on_map(self):
        """Plot height profile directly on heatmap when exactly 2 points selected."""
        if len(self.selected_points) != 2:
            return

        x, y = zip(*self.selected_points)

        # Draw dotted line connecting the two points
        self._profile_line, = self.ax.plot(
            x, y, color='black', linestyle='dotted', linewidth=2, zorder=11
        )

        # Extract profile values and plot height visualization
        profile_df = self._extract_profile()
        if profile_df is not None and not profile_df.empty:
            extent = self.ax.get_xlim() + self.ax.get_ylim()
            diagonal_length = np.sqrt((extent[1] - extent[0])**2 + (extent[3] - extent[2])**2)
            max_normalized_value = 0.3 * diagonal_length

            # Calculate height values (normalized)
            values = np.asarray(profile_df['values'].values, dtype=float)
            valid_vals = values[~np.isnan(values)]
            val_max = float(valid_vals.max()) if len(valid_vals) > 0 else 0.0
            if val_max > 0:
                height = (values / val_max) * max_normalized_value
                height -= float(np.nanmin(height))

                dx = x[1] - x[0]
                dy = y[1] - y[0]
                length = np.sqrt(dx**2 + dy**2)

                if length > 0:
                    perp_dx = -dy / length
                    perp_dy = dx / length
                    x_vals = profile_df['X'].values + perp_dx * height
                    y_vals = profile_df['Y'].values + perp_dy * height
                    self._profile_height_line, = self.ax.plot(
                        x_vals, y_vals, color='black', linestyle='-', linewidth=2, zorder=11
                    )
