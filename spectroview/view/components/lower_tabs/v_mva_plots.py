"""MVA plot tab widget and Matplotlib plotting mixin (Summary, Scree, Loadings, Heatmap, Scores, Residuals)."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtWidgets import QVBoxLayout, QWidget

from spectroview import PLOT_POLICY_LIGHT
from spectroview.view.theme import current_tokens
from spectroview.viewmodel.common.spectral_utils import parse_coords_from_fname


class _PlotTab(QWidget):
    """A widget containing a matplotlib Figure, Canvas, and NavigationToolbar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        with plt.style.context(PLOT_POLICY_LIGHT):
            self.figure = Figure(layout="compressed", dpi=80)
            self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        class ToolbarEventFilter(QObject):
            def __init__(self, toolbar):
                # Parented to `toolbar` so Qt cancels the pending single-shot
                # timer on teardown -- unparented, it could fire 100ms later
                # against an already-deleted toolbar.
                super().__init__(toolbar)
                self.toolbar = toolbar
                self._timer = QTimer(self)
                self._timer.setSingleShot(True)
                self._timer.setInterval(100)
                self._timer.timeout.connect(self._update_icons)

            def eventFilter(self, obj, event):
                if event.type() == QEvent.PaletteChange:
                    self._timer.start()
                return False

            def _update_icons(self):
                try:
                    action_dict = {action.text(): action for action in self.toolbar.actions() if action.text()}
                except RuntimeError:
                    return  # toolbar's C++ object already deleted
                for text, tooltip_text, image_file, name_of_method in self.toolbar.toolitems:
                    if text in action_dict and image_file is not None:
                        try:
                            icon = self.toolbar._icon(image_file + '.png')
                            action_dict[text].setIcon(icon)
                        except Exception:
                            pass

        self.toolbar_filter = ToolbarEventFilter(self.toolbar)
        self.toolbar.installEventFilter(self.toolbar_filter)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def clear_and_placeholder(self, message: str = "Run an analysis to see results here"):
        """Clear the axes and show a placeholder message."""
        tok = current_tokens()
        self.ax.clear()
        self.ax.text(
            0.5, 0.5, message,
            transform=self.ax.transAxes,
            ha="center", va="center",
            fontsize=13, color=tok.text_muted, style="italic",
        )
        self.ax.set_axis_off()
        self.canvas.draw_idle()

    def redraw(self):
        """Convenience method to redraw the canvas."""
        self.canvas.draw_idle()


class MVAPlotMixin:
    """Plot rendering helpers for Summary, Scree, Loadings, Heatmap, Scores, and Residuals tabs."""

    def _show_all_placeholders(self):
        """Show placeholder messages on all plot tabs."""
        self.tab_summary.clear_and_placeholder()
        self.tab_scree.clear_and_placeholder()
        self.tab_loadings.clear_and_placeholder()
        self.tab_heatmap.clear_and_placeholder()
        self.tab_scores.clear_and_placeholder()
        self.tab_residuals.clear_and_placeholder()

    def _populate_score_combos(self, n_components: int, prefix: str = "PC"):
        """Fill the score and heatmap axis comboboxes."""
        self.cbb_score_x.blockSignals(True)
        self.cbb_score_y.blockSignals(True)
        self.cbb_heatmap_comp.blockSignals(True)
        self.cbb_score_x.clear()
        self.cbb_score_y.clear()
        self.cbb_heatmap_comp.clear()
        labels = [f"{prefix}{i+1}" for i in range(n_components)]
        self.cbb_score_x.addItems(labels)
        self.cbb_score_y.addItems(labels)
        self.cbb_heatmap_comp.addItems(labels)
        if n_components >= 2:
            self.cbb_score_x.setCurrentIndex(0)
            self.cbb_score_y.setCurrentIndex(1)
        if n_components >= 1:
            self.cbb_heatmap_comp.setCurrentIndex(0)
        self.cbb_score_x.blockSignals(False)
        self.cbb_score_y.blockSignals(False)
        self.cbb_heatmap_comp.blockSignals(False)

    def _replot_heatmap(self):
        """Replot the heatmap based on current component selection."""
        if self._pca_payload is not None:
            result = self._pca_payload["result"]
            fnames = self._pca_payload["fnames"]
            self._plot_heatmap(result.scores, fnames, prefix="PC")
        elif self._nmf_payload is not None:
            result = self._nmf_payload["result"]
            fnames = self._nmf_payload["fnames"]
            self._plot_heatmap(result.W, fnames, prefix="NMF")
        else:
            self.tab_heatmap.clear_and_placeholder()

    def _plot_heatmap(self, scores, fnames, prefix="PC"):
        """2D spatial map of selected component score."""
        ax = self.tab_heatmap.ax
        self.tab_heatmap.figure.clear()
        ax = self.tab_heatmap.figure.add_subplot(111)
        self.tab_heatmap.ax = ax

        comp_idx = self.cbb_heatmap_comp.currentIndex()
        if comp_idx < 0 or comp_idx >= scores.shape[1]:
            return

        # Extract coordinates
        coords_x = []
        coords_y = []
        valid_indices = []

        for i, fname in enumerate(fnames):
            coords = parse_coords_from_fname(fname)
            if coords is not None:
                coords_x.append(coords[0])
                coords_y.append(coords[1])
                valid_indices.append(i)

        if not coords_x:
            self.tab_heatmap.clear_and_placeholder("Cannot render heatmap:\nNo spatial (x, y) coordinates found in spectra names.")
            return

        coords_x = np.array(coords_x)
        coords_y = np.array(coords_y)
        z_data = scores[valid_indices, comp_idx]

        sc = ax.scatter(coords_x, coords_y, c=z_data, cmap="viridis", s=60, marker='s')

        x_range = coords_x.max() - coords_x.min()
        y_range = coords_y.max() - coords_y.min()
        if x_range > 0 and y_range > 0:
            ax.set_aspect('equal', adjustable='datalim')

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Spatial Map: {prefix}{comp_idx+1} Scores")

        self.tab_heatmap.figure.colorbar(sc, ax=ax, label="Score Value")

        self.tab_heatmap.redraw()

    def _plot_summary(self):
        """Plot a summary grid containing Scree, Loadings, and Scores."""
        self.tab_summary.figure.clear()

        is_pca = self._pca_payload is not None
        payload = self._pca_payload if is_pca else self._nmf_payload

        if payload is None:
            self.tab_summary.clear_and_placeholder()
            return

        result = payload["result"]
        x_axis = payload["x_axis"]

        prefix = "PC" if is_pca else "NMF"
        explained_var_ratio = result.explained_variance_ratio if is_pca else None
        cumulative = result.cumulative_variance if is_pca else None

        loadings = result.loadings if is_pca else result.H
        scores = result.scores if is_pca else result.W

        # 2x2 grid layout
        axs = self.tab_summary.figure.subplots(2, 2)
        ax_scree, ax_loadings = axs[0, 0], axs[0, 1]
        ax_sc1, ax_sc2 = axs[1, 0], axs[1, 1]

        if not is_pca:
            ax_scree.set_visible(False)

        # 1. Plot Scree (if PCA)
        if is_pca:
            n = len(explained_var_ratio)
            indices = np.arange(1, min(n + 1, 11))  # Top 10 max
            ax_scree.bar(indices, explained_var_ratio[:10] * 100, color="#5B9BD5", alpha=0.85)
            ax_scree.set_title("Scree Plot (Top 10)", fontsize=9)
            ax_scree.set_ylabel("Variance (%)", fontsize=8)
            ax_scree.tick_params(labelsize=8)
            ax2 = ax_scree.twinx()
            ax2.plot(indices, cumulative[:10] * 100, "o-", color="#E87722", ms=4, lw=1.5)
            ax2.tick_params(labelsize=8)
            ax_scree.set_xticks(indices)

        # 2. Plot Loadings (First 3)
        n_comp_plot = min(3, loadings.shape[0])
        cmap = plt.cm.get_cmap("tab10")
        for i in range(n_comp_plot):
            color = cmap(i % 10)
            label = f"{prefix}{i+1}"
            if explained_var_ratio is not None:
                label += f" ({explained_var_ratio[i]*100:.1f}%)"
            ax_loadings.plot(x_axis, loadings[i], lw=1.0, color=color, label=label)
        ax_loadings.set_title(f"Loadings (First {n_comp_plot})", fontsize=9)
        ax_loadings.legend(fontsize=6)
        ax_loadings.axhline(0, color="gray", lw=0.5, ls="--")
        ax_loadings.tick_params(labelsize=8)

        # 3. Plot Scores (1 vs 2)
        if scores.shape[1] >= 2:
            ax_sc1.scatter(scores[:, 0], scores[:, 1], s=15, c="#5B9BD5", alpha=0.7)
            x_label = f"{prefix}1"
            y_label = f"{prefix}2"
            if explained_var_ratio is not None:
                x_label += f" ({explained_var_ratio[0]*100:.1f}%)"
                y_label += f" ({explained_var_ratio[1]*100:.1f}%)"
            ax_sc1.set_xlabel(x_label, fontsize=8)
            ax_sc1.set_ylabel(y_label, fontsize=8)
            ax_sc1.set_title(f"Scores: {prefix}1 vs {prefix}2", fontsize=9)
            ax_sc1.axhline(0, color="gray", lw=0.5, ls="--")
            ax_sc1.axvline(0, color="gray", lw=0.5, ls="--")
            ax_sc1.tick_params(labelsize=8)
        else:
            ax_sc1.set_visible(False)

        # 4. Plot Scores (2 vs 3 if >=3 comps, else 1 vs n)
        if scores.shape[1] >= 3:
            ax_sc2.scatter(scores[:, 1], scores[:, 2], s=15, c="#70AD47", alpha=0.7)
            x_label = f"{prefix}2"
            y_label = f"{prefix}3"
            if explained_var_ratio is not None:
                x_label += f" ({explained_var_ratio[1]*100:.1f}%)"
                y_label += f" ({explained_var_ratio[2]*100:.1f}%)"
            ax_sc2.set_xlabel(x_label, fontsize=8)
            ax_sc2.set_ylabel(y_label, fontsize=8)
            ax_sc2.set_title(f"Scores: {prefix}2 vs {prefix}3", fontsize=9)
            ax_sc2.axhline(0, color="gray", lw=0.5, ls="--")
            ax_sc2.axvline(0, color="gray", lw=0.5, ls="--")
            ax_sc2.tick_params(labelsize=8)
        else:
            ax_sc2.set_visible(False)

        self.tab_summary.redraw()

    def _plot_scree(self, explained_var_ratio, cumulative):
        """Bar chart of explained variance + cumulative line."""
        ax = self.tab_scree.ax
        self.tab_scree.figure.clear()
        ax = self.tab_scree.figure.add_subplot(111)
        self.tab_scree.ax = ax

        n = len(explained_var_ratio)
        indices = np.arange(1, n + 1)

        # Bars
        bars = ax.bar(
            indices, explained_var_ratio * 100,
            color="#5B9BD5", edgecolor="#2E6EA6", alpha=0.85, zorder=2,
        )
        # Annotate bars
        for bar, val in zip(bars, explained_var_ratio * 100):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=8,
            )

        # Cumulative line
        ax2 = ax.twinx()
        ax2.plot(indices, cumulative * 100, "o-", color="#E87722", lw=2, ms=5, zorder=3)
        ax2.set_ylabel("Cumulative (%)")
        ax2.set_ylim(0, 105)

        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance (%)")
        ax.set_title("Scree Plot")
        ax.set_xticks(indices)
        ax.set_xlim(0.4, n + 0.6)

        self.tab_scree.redraw()

    def _plot_loadings(self, x_axis, loadings, prefix="PC",
                       explained_var_ratio=None):
        """Overlay spectral loadings on wavenumber axis with fill-between."""
        ax = self.tab_loadings.ax
        self.tab_loadings.figure.clear()
        ax = self.tab_loadings.figure.add_subplot(111)
        self.tab_loadings.ax = ax

        n_components = loadings.shape[0]
        cmap = plt.cm.get_cmap("tab10")

        for i in range(n_components):
            color = cmap(i % 10)
            label = f"{prefix}{i+1}"
            if explained_var_ratio is not None and i < len(explained_var_ratio):
                label += f" ({explained_var_ratio[i]*100:.1f}%)"
            ax.plot(
                x_axis, loadings[i],
                lw=1.2, color=color,
                label=label,
            )
            # Subtle fill-between for visual clarity (inspired by PyFASMA)
            ax.fill_between(x_axis, loadings[i], alpha=0.08, color=color)

        ax.set_xlabel("Wavenumber")
        ax.set_ylabel("Loading")
        ax.set_title(f"{prefix} Loadings")
        ax.legend(loc="best", fontsize=8)
        ax.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.5)

        self.tab_loadings.redraw()

    def _replot_scores(self):
        """Replot the scores scatter based on current axis selection."""
        if self._pca_payload is not None:
            result = self._pca_payload["result"]
            fnames = self._pca_payload["fnames"]
            self._plot_scores(result.scores, fnames, prefix="PC",
                              explained_var_ratio=result.explained_variance_ratio)
        elif self._nmf_payload is not None:
            result = self._nmf_payload["result"]
            fnames = self._nmf_payload["fnames"]
            self._plot_scores(result.W, fnames, prefix="NMF")
        else:
            self.tab_scores.clear_and_placeholder()

    def _plot_scores(self, scores, fnames, prefix="PC",
                     explained_var_ratio=None):
        """2D scatter of scores with spectrum labels."""
        ax = self.tab_scores.ax
        self.tab_scores.figure.clear()
        ax = self.tab_scores.figure.add_subplot(111)
        self.tab_scores.ax = ax

        ix = self.cbb_score_x.currentIndex()
        iy = self.cbb_score_y.currentIndex()

        if ix < 0 or iy < 0 or ix >= scores.shape[1] or iy >= scores.shape[1]:
            return

        x_data = scores[:, ix]
        y_data = scores[:, iy]

        ax.scatter(x_data, y_data, s=50, c="#5B9BD5", edgecolors="#2E6EA6", zorder=3)

        # Build axis labels with variance info if available
        x_label = f"{prefix}{ix+1}"
        y_label = f"{prefix}{iy+1}"
        if explained_var_ratio is not None:
            if ix < len(explained_var_ratio):
                x_label += f" ({explained_var_ratio[ix]*100:.1f}%)"
            if iy < len(explained_var_ratio):
                y_label += f" ({explained_var_ratio[iy]*100:.1f}%)"

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"Scores: {prefix}{ix+1} vs {prefix}{iy+1}")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axvline(0, color="gray", lw=0.5, ls="--")

        self.tab_scores.redraw()

    def _plot_residuals(self, fnames, reconstruction_errors):
        """Bar chart of per-spectrum reconstruction errors."""
        ax = self.tab_residuals.ax
        self.tab_residuals.figure.clear()
        ax = self.tab_residuals.figure.add_subplot(111)
        self.tab_residuals.ax = ax

        n = len(fnames)
        indices = np.arange(n)

        ax.bar(
            indices, reconstruction_errors,
            color="#70AD47", edgecolor="#548235", alpha=0.85, zorder=2,
        )

        ax.set_xlabel("Spectrum")
        ax.set_ylabel("Reconstruction Error (L2)")
        ax.set_title("Per-Spectrum Reconstruction Error")

        # Show spectrum names on x-axis if not too many
        if n <= 30:
            # Truncate names for readability
            short_names = [f[:15] + "…" if len(f) > 15 else f for f in fnames]
            ax.set_xticks(indices)
            ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
        else:
            ax.set_xticks([0, n // 4, n // 2, 3 * n // 4, n - 1])

        self.tab_residuals.redraw()
