"""Spatial heatmap and wafer-map rendering helpers for PlotRenderer."""

from __future__ import annotations

import matplotlib.patches as patches
from matplotlib.colors import CenteredNorm, LogNorm
import numpy as np
import pandas as pd

# scipy is imported inside WaferPlot.plot() so `scipy.interpolate` stays off startup imports.


def _clear_degenerate_zlim(zmin, zmax):
    """Treat an explicit zmin == zmax (both set, equal) as unset (None, None).

    Old saved .graphs files can carry zmin == zmax == 0.0; honoring that
    collapses the wafer/2Dmap color scale to a single flat color
    (Normalize(vmin=0, vmax=0)). A degenerate range never carries useful
    information, so fall back to the data-derived range like an unset limit.
    """
    if zmin is not None and zmax is not None and zmin == zmax:
        return None, None
    return zmin, zmax


def _build_color_norm(vmin, vmax, norm_kind, center=0.0):
    """Build a color-scale Normalize object for a wafer/2Dmap imshow call.

    Returns None for 'linear' (caller should pass plain vmin=/vmax=) or when
    the requested normalization isn't valid for this data range (e.g. log
    norm on data that isn't strictly positive, or a degenerate zero-width
    range) -- falls back to linear rather than raising, matching this
    codebase's existing graceful-degradation style around wafer/2Dmap
    rendering (see WaferPlot / _plot_2dmap error handling).
    """
    if vmin is None or vmax is None:
        return None
    if norm_kind == 'log':
        if vmin <= 0:
            return None
        return LogNorm(vmin=vmin, vmax=vmax)
    if norm_kind == 'centered':
        halfrange = max(abs(vmax - center), abs(vmin - center))
        if halfrange <= 0:
            return None
        return CenteredNorm(vcenter=center, halfrange=halfrange)
    return None


class SpatialPlotMixin:
    """Mixin providing 2D heatmap and wafer-map plotting for PlotRenderer."""

    def _apply_colorbar_fontsize(self, cbar):
        """Apply the configured tick-label size to a wafer/2Dmap colorbar."""
        fs = getattr(self.vg, 'colorbar_fontsize', None)
        if cbar is not None and fs:
            cbar.ax.tick_params(labelsize=fs)

    def _plot_2dmap(self, df, y):
        """Plot 2D heatmap."""
        x_col = self.vg.x
        y_col = y if isinstance(self.vg.y, list) else self.vg.y
        z_col = self.vg.z

        xmin = df[x_col].min()
        xmax = df[x_col].max()
        ymin = df[y_col].min()
        ymax = df[y_col].max()

        heatmap_data = df.pivot(index=y_col, columns=x_col, values=z_col)
        zmin, zmax = _clear_degenerate_zlim(self.vg.zmin, self.vg.zmax)
        vmin = zmin if zmin is not None else heatmap_data.min().min()
        vmax = zmax if zmax is not None else heatmap_data.max().max()
        vmin, vmax = _clear_degenerate_zlim(vmin, vmax)

        norm = _build_color_norm(
            float(vmin) if pd.notna(vmin) else None,
            float(vmax) if pd.notna(vmax) else None,
            getattr(self.vg, 'colormap_norm', 'linear'),
            getattr(self.vg, 'colormap_center', 0.0),
        )
        imshow_kwargs = dict(vmin=vmin, vmax=vmax) if norm is None else dict(norm=norm)

        heatmap = self.vg.ax.imshow(
            heatmap_data,
            aspect='equal',
            extent=[xmin, xmax, ymin, ymax],
            cmap=self.vg.color_palette,
            origin='lower',
            **imshow_kwargs,
        )

        # Remove existing colorbar if present to prevent accumulation
        if hasattr(self.vg.ax, '_2dmap_colorbar') and self.vg.ax._2dmap_colorbar is not None:
            try:
                self.vg.ax._2dmap_colorbar.remove()
            except Exception:
                pass

        # Add new colorbar and store reference
        self.vg.ax._2dmap_colorbar = self.vg.ax.figure.colorbar(
            heatmap, ax=self.vg.ax, orientation='vertical'
        )
        self._apply_colorbar_fontsize(self.vg.ax._2dmap_colorbar)

    def _plot_wafer(self, df):
        """Plot wafer plot by creating an object of WaferPlot Class."""
        vmin, vmax = _clear_degenerate_zlim(self.vg.zmin, self.vg.zmax)

        wdf = WaferPlot()
        wdf.plot(
            self.vg.ax,
            x=df[self.vg.x],
            y=df[self.vg.y[0]],
            z=df[self.vg.z],
            cmap=self.vg.color_palette,
            vmin=vmin,
            vmax=vmax,
            stats=self.vg.wafer_stats,
            r=(self.vg.wafer_size / 2),
            norm_kind=getattr(self.vg, 'colormap_norm', 'linear'),
            norm_center=getattr(self.vg, 'colormap_center', 0.0),
        )
        self._apply_colorbar_fontsize(getattr(self.vg.ax, '_wafer_colorbar', None))

        # Annotate slot number if active filter
        if hasattr(self.vg, "filters") and isinstance(self.vg.filters, (list, dict)):
            filters_list = (
                self.vg.filters
                if isinstance(self.vg.filters, list)
                else self.vg.filters.get("filters", [])
            )
            for f in filters_list:
                expr = f.get("expression", "")
                state = f.get("state", False)
                if state and "Slot ==" in expr:
                    try:
                        slot_num = expr.split("==")[1].strip()
                        self.vg.ax.text(
                            0.02, 0.98, f"Slot {slot_num}",
                            transform=self.vg.ax.transAxes,
                            fontsize=12, color='black',
                            fontweight='bold',
                            verticalalignment='top',
                            horizontalalignment='left',
                        )
                    except Exception:
                        pass
                    break


class WaferPlot:
    """Class to plot wafer map."""

    def __init__(self, inter_method='linear'):
        self.inter_method = inter_method

    def plot(
        self,
        ax,
        x,
        y,
        z,
        cmap="jet",
        r=100,
        vmax=None,
        vmin=None,
        stats=True,
        norm_kind='linear',
        norm_center=0.0,
    ):
        """Plot wafer map with interpolated data."""
        from scipy.interpolate import griddata

        xi, yi = np.meshgrid(np.linspace(-r, r, 600), np.linspace(-r, r, 600))
        zi = griddata((x, y), z, (xi, yi), method=self.inter_method)

        zi_valid = zi[~np.isnan(zi)]
        resolved_vmin = vmin if vmin is not None else (float(zi_valid.min()) if zi_valid.size else None)
        resolved_vmax = vmax if vmax is not None else (float(zi_valid.max()) if zi_valid.size else None)
        resolved_vmin, resolved_vmax = _clear_degenerate_zlim(resolved_vmin, resolved_vmax)
        norm = None
        if norm_kind != 'linear':
            norm = _build_color_norm(resolved_vmin, resolved_vmax, norm_kind, norm_center)
        imshow_kwargs = dict(vmin=resolved_vmin, vmax=resolved_vmax) if norm is None else dict(norm=norm)

        im = ax.imshow(
            zi,
            extent=[-r - 1, r + 1, -r - 0.5, r + 0.5],
            origin='lower',
            cmap=cmap,
            interpolation='nearest',
            zorder=2,
            **imshow_kwargs,
        )

        ax.scatter(x, y, facecolors='none', edgecolors='black', s=20, zorder=3)

        wafer_circle = patches.Circle((0, 0), radius=r, fill=False, color='black', linewidth=1)
        ax.add_patch(wafer_circle)

        ax.set_ylabel("Wafer size (mm)")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x', which='both', bottom=False, top=False)
        ax.tick_params(axis='y', which='both', right=False, left=True)
        ax.set_xticklabels([])

        # Remove existing colorbar if present to prevent accumulation
        if hasattr(ax, '_wafer_colorbar') and ax._wafer_colorbar is not None:
            try:
                ax._wafer_colorbar.remove()
            except Exception:
                pass

        # Add new colorbar and store reference
        ax._wafer_colorbar = ax.figure.colorbar(im, ax=ax)

        if stats:
            self.stats(z, ax)

    def stats(self, z, ax):
        """Calculate and display statistical values in the wafer plot."""
        mean_value = z.mean()
        max_value = z.max()
        min_value = z.min()
        sigma_value = z.std()
        three_sigma_value = 3 * sigma_value

        ax.text(
            0.05, -0.1, f"3\u03C3: {three_sigma_value:.2f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
        )
        ax.text(
            0.99, -0.1, f"Max: {max_value:.2f}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
        )
        ax.text(
            0.99, -0.05, f"Min: {min_value:.2f}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
        )
        ax.text(
            0.05, -0.05, f"Mean: {mean_value:.2f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
        )
