"""View for Multivariate Analysis (MVA) tab — UI controls and result plots.

Provides a tabbed interface for visualizing MVA results with dedicated tabs
for Scree Plot, Loadings, Scores, and Residuals. Each tab includes a
matplotlib navigation toolbar for interactive plot exploration.

References
----------
.. [1] S. Wold, K. Esbensen, and P. Geladi, "Principal component analysis,"
       *Chemometrics and intelligent laboratory systems*, vol. 2, no. 1-3,
       pp. 37-52, 1987.
.. [2] D. D. Lee and H. S. Seung, "Learning the parts of objects by
       non-negative matrix factorization," *Nature*, vol. 401, no. 6755,
       pp. 788-791, 1999.
"""
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QComboBox, QRadioButton, QButtonGroup,
    QGroupBox, QSplitter, QLineEdit, QScrollArea, QTabWidget,
    QCheckBox,
)
from PySide6.QtCore import Qt, Signal

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from spectroview import PLOT_POLICY_LIGHT


# ═══════════════════════════════════════════════════════════════════════
# Helper: a single plot tab with Figure + Canvas + Toolbar
# ═══════════════════════════════════════════════════════════════════════

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

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def clear_and_placeholder(self, message: str = "Run an analysis to see results here"):
        """Clear the axes and show a placeholder message."""
        self.ax.clear()
        self.ax.text(
            0.5, 0.5, message,
            transform=self.ax.transAxes,
            ha="center", va="center",
            fontsize=13, color="#999", style="italic",
        )
        self.ax.set_axis_off()
        self.canvas.draw_idle()

    def redraw(self):
        """Convenience method to redraw the canvas."""
        self.canvas.draw_idle()


# ═══════════════════════════════════════════════════════════════════════
# Main MVA View
# ═══════════════════════════════════════════════════════════════════════

class VMVA(QWidget):
    """View for Multivariate Analysis (MVA) feature.

    The right panel uses a QTabWidget with dedicated tabs for each plot
    type (Scree Plot, Loadings, Scores, Residuals), each equipped with a
    matplotlib NavigationToolbar.

    References
    ----------
    Visualization approach inspired by PyFASMA [1]_.
    """

    # ───── View → ViewModel signals ─────
    run_pca_requested = Signal(int, bool)            # n_components, center
    run_nmf_requested = Signal(int, int, float, int) # n_components, max_iter, tol, seed
    send_to_graphs_requested = Signal(str)           # df_name

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pca_payload = None   # cached PCA results for re-plotting
        self._nmf_payload = None   # cached NMF results for re-plotting
        self.init_ui()

    # ══════════════════════════════════════════════════════════════════
    # UI Construction
    # ══════════════════════════════════════════════════════════════════

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ── LEFT PANEL: Controls ──────────────────────────────────────
        left_scroll_area = QScrollArea()
        left_scroll_area.setWidgetResizable(True)
        left_scroll_area.setMaximumWidth(340)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(2, 2, 2, 2)
        left_layout.setSpacing(4)

        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.rb_nmf = QRadioButton("NMF")
        self.rb_pca = QRadioButton("PCA")
        self.rb_pca.setChecked(True)
        self.method_group = QButtonGroup(self)
        self.method_group.addButton(self.rb_pca)
        self.method_group.addButton(self.rb_nmf)
        
        method_layout.addWidget(self.rb_pca)
        method_layout.addWidget(self.rb_nmf)
        left_layout.addLayout(method_layout)

        # ── PCA Parameters ────────────────────────────────────────────
        self.grp_pca_params = QGroupBox("PCA Parameters")
        pca_params_layout = QVBoxLayout(self.grp_pca_params)

        # Number of components
        row_nc_pca = QHBoxLayout()
        row_nc_pca.addWidget(QLabel("Components:"))
        self.spin_n_components_pca = QSpinBox()
        self.spin_n_components_pca.setRange(2, 50)
        self.spin_n_components_pca.setValue(5)
        row_nc_pca.addWidget(self.spin_n_components_pca)
        pca_params_layout.addLayout(row_nc_pca)

        # Mean centering checkbox
        self.cb_center = QCheckBox("Mean centering")
        self.cb_center.setChecked(True)
        self.cb_center.setToolTip(
            "Subtract the mean spectrum before SVD.\n"
            "Standard practice for PCA on spectral data."
        )
        pca_params_layout.addWidget(self.cb_center)

        left_layout.addWidget(self.grp_pca_params)

        # ── NMF Parameters ────────────────────────────────────────────
        self.grp_nmf_params = QGroupBox("NMF Parameters")
        nmf_params_layout = QVBoxLayout(self.grp_nmf_params)

        # Number of components
        row_nc_nmf = QHBoxLayout()
        row_nc_nmf.addWidget(QLabel("Components:"))
        self.spin_n_components_nmf = QSpinBox()
        self.spin_n_components_nmf.setRange(2, 50)
        self.spin_n_components_nmf.setValue(5)
        row_nc_nmf.addWidget(self.spin_n_components_nmf)
        nmf_params_layout.addLayout(row_nc_nmf)

        # Max iterations
        row_iter = QHBoxLayout()
        row_iter.addWidget(QLabel("Max iterations:"))
        self.spin_max_iter = QSpinBox()
        self.spin_max_iter.setRange(50, 5000)
        self.spin_max_iter.setValue(500)
        self.spin_max_iter.setSingleStep(50)
        row_iter.addWidget(self.spin_max_iter)
        nmf_params_layout.addLayout(row_iter)

        # Tolerance
        row_tol = QHBoxLayout()
        row_tol.addWidget(QLabel("Tolerance:"))
        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setDecimals(6)
        self.spin_tol.setRange(1e-8, 1.0)
        self.spin_tol.setValue(1e-4)
        self.spin_tol.setSingleStep(1e-4)
        row_tol.addWidget(self.spin_tol)
        nmf_params_layout.addLayout(row_tol)

        # Random seed
        row_seed = QHBoxLayout()
        row_seed.addWidget(QLabel("Random seed:"))
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 99999)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip("Random seed for reproducible NMF results")
        row_seed.addWidget(self.spin_seed)
        nmf_params_layout.addLayout(row_seed)

        left_layout.addWidget(self.grp_nmf_params)

        # Run button
        self.btn_run = QPushButton("▶  Run Analysis")
        self.btn_run.setMinimumHeight(30)
        left_layout.addWidget(self.btn_run)

        # Export section
        grp_export = QGroupBox("Export to Graphs")
        export_layout = QVBoxLayout(grp_export)
        self.ent_df_name = QLineEdit()
        self.ent_df_name.setPlaceholderText("DataFrame name")
        self.ent_df_name.setText("MVA_scores")
        export_layout.addWidget(self.ent_df_name)
        self.btn_send = QPushButton("Send to Graphs")
        export_layout.addWidget(self.btn_send)
        left_layout.addWidget(grp_export)

        # Status
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #888; font-style: italic;")
        left_layout.addWidget(self.lbl_status)

        left_layout.addStretch()

        # ── RIGHT PANEL: Tabbed plot area ─────────────────────────────
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.plot_tabs = QTabWidget()

        # Create plot tabs
        self.tab_summary = _PlotTab()
        self.tab_scree = _PlotTab()
        self.tab_loadings = _PlotTab()
        self.tab_heatmap = _PlotTab()
        self.tab_scores = _PlotTab()
        self.tab_residuals = _PlotTab()

        self.plot_tabs.addTab(self.tab_summary, "Summary")
        self.plot_tabs.addTab(self.tab_scree, "Scree Plot")
        self.plot_tabs.addTab(self.tab_loadings, "Loadings")
        self.plot_tabs.addTab(self.tab_heatmap, "Heatmap")
        self.plot_tabs.addTab(self.tab_scores, "Scores")
        self.plot_tabs.addTab(self.tab_residuals, "Residuals")

        # Score axis selectors — placed above the tab widget
        score_controls = QWidget()
        score_controls_layout = QHBoxLayout(score_controls)
        score_controls_layout.setContentsMargins(4, 2, 4, 2)
        score_controls_layout.addWidget(QLabel("Score axes:"))
        score_controls_layout.addWidget(QLabel("X:"))
        self.cbb_score_x = QComboBox()
        self.cbb_score_x.setMinimumWidth(60)
        score_controls_layout.addWidget(self.cbb_score_x)
        score_controls_layout.addWidget(QLabel("Y:"))
        self.cbb_score_y = QComboBox()
        self.cbb_score_y.setMinimumWidth(60)
        score_controls_layout.addWidget(self.cbb_score_y)
        score_controls_layout.addStretch()
        self.score_controls = score_controls

        # Heatmap controls
        heatmap_controls = QWidget()
        heatmap_controls_layout = QHBoxLayout(heatmap_controls)
        heatmap_controls_layout.setContentsMargins(4, 2, 4, 2)
        heatmap_controls_layout.addWidget(QLabel("Component to map:"))
        self.cbb_heatmap_comp = QComboBox()
        self.cbb_heatmap_comp.setMinimumWidth(80)
        heatmap_controls_layout.addWidget(self.cbb_heatmap_comp)
        heatmap_controls_layout.addStretch()
        self.heatmap_controls = heatmap_controls

        right_layout.addWidget(self.plot_tabs)
        right_layout.addWidget(self.score_controls)
        right_layout.addWidget(self.heatmap_controls)

        # Set scroll area widget
        left_scroll_area.setWidget(left_panel)

        # Assemble splitter
        splitter.addWidget(left_scroll_area)
        splitter.addWidget(right_panel)
        splitter.setSizes([280, 720])

        # ── Internal connections ──────────────────────────────────────
        self.rb_pca.toggled.connect(self._on_method_changed)
        self.rb_nmf.toggled.connect(self._on_method_changed)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_send.clicked.connect(self._on_send)
        self.cbb_score_x.currentIndexChanged.connect(self._replot_scores)
        self.cbb_score_y.currentIndexChanged.connect(self._replot_scores)
        self.cbb_heatmap_comp.currentIndexChanged.connect(self._replot_heatmap)
        self.plot_tabs.currentChanged.connect(self._on_tab_changed)

        # Initial state
        self._on_method_changed()
        self._show_all_placeholders()

    # ══════════════════════════════════════════════════════════════════
    # Internal UI logic
    # ══════════════════════════════════════════════════════════════════

    def _on_method_changed(self):
        """Show/hide method-specific parameter groups and tabs."""
        is_pca = self.rb_pca.isChecked()
        self.grp_pca_params.setVisible(is_pca)
        self.grp_nmf_params.setVisible(not is_pca)

        # Show/hide Scree Plot tab (only for PCA)
        scree_idx = self.plot_tabs.indexOf(self.tab_scree)
        if is_pca:
            if scree_idx < 0:
                self.plot_tabs.insertTab(1, self.tab_scree, "Scree Plot")
        else:
            if scree_idx >= 0:
                self.plot_tabs.removeTab(scree_idx)

        # Update score controls visibility
        self._update_score_controls_visibility()

    def _on_tab_changed(self):
        """Update score controls visibility when tab changes."""
        self._update_score_controls_visibility()

    def _update_score_controls_visibility(self):
        """Show score axis selectors only when Scores tab is active."""
        current_tab = self.plot_tabs.currentWidget()
        self.score_controls.setVisible(current_tab is self.tab_scores)
        self.heatmap_controls.setVisible(current_tab is self.tab_heatmap)

    def _on_run(self):
        """Emit the appropriate signal based on selected method."""
        if self.rb_pca.isChecked():
            nc = self.spin_n_components_pca.value()
            center = self.cb_center.isChecked()
            self.run_pca_requested.emit(nc, center)
        else:
            nc = self.spin_n_components_nmf.value()
            self.run_nmf_requested.emit(
                nc,
                self.spin_max_iter.value(),
                self.spin_tol.value(),
                self.spin_seed.value(),
            )

    def _on_send(self):
        """Emit export signal."""
        name = self.ent_df_name.text().strip()
        if name:
            self.send_to_graphs_requested.emit(name)

    def _show_all_placeholders(self):
        """Show placeholder messages on all plot tabs."""
        self.tab_summary.clear_and_placeholder()
        self.tab_scree.clear_and_placeholder()
        self.tab_loadings.clear_and_placeholder()
        self.tab_heatmap.clear_and_placeholder()
        self.tab_scores.clear_and_placeholder()
        self.tab_residuals.clear_and_placeholder()

    # ══════════════════════════════════════════════════════════════════
    # Public slots — called by ViewModel via signals
    # ══════════════════════════════════════════════════════════════════

    def display_pca_results(self, payload: dict):
        """Receive PCA results from ViewModel and plot all tabs."""
        self._pca_payload = payload
        self._nmf_payload = None

        result = payload["result"]
        self._populate_score_combos(result.scores.shape[1], prefix="PC")

        # Ensure method selector reflects PCA
        self.rb_pca.setChecked(True)

        # Plot all tabs
        self._plot_summary()
        self._plot_scree(result.explained_variance_ratio, result.cumulative_variance)
        self._plot_loadings(
            payload["x_axis"], result.loadings,
            prefix="PC",
            explained_var_ratio=result.explained_variance_ratio,
        )
        self._replot_heatmap()
        self._replot_scores()
        self._plot_residuals(payload["fnames"], payload["reconstruction_errors"])

    def display_nmf_results(self, payload: dict):
        """Receive NMF results from ViewModel and plot all tabs."""
        self._nmf_payload = payload
        self._pca_payload = None

        result = payload["result"]
        self._populate_score_combos(result.W.shape[1], prefix="NMF")

        # Ensure method selector reflects NMF
        self.rb_nmf.setChecked(True)

        # Plot all tabs (Scree is hidden for NMF)
        self._plot_summary()
        self._plot_loadings(payload["x_axis"], result.H, prefix="NMF")
        self._replot_heatmap()
        self._replot_scores()
        self._plot_residuals(payload["fnames"], payload["reconstruction_errors"])

    def set_status(self, msg: str):
        """Update the status label."""
        self.lbl_status.setText(msg)

    # ══════════════════════════════════════════════════════════════════
    # Plotting
    # ══════════════════════════════════════════════════════════════════

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
            if '(' in fname and ')' in fname:
                coords_str = fname[fname.rfind('(')+1:fname.rfind(')')]
                try:
                    x_str, y_str = coords_str.split(',')
                    coords_x.append(float(x_str.strip()))
                    coords_y.append(float(y_str.strip()))
                    valid_indices.append(i)
                except:
                    pass
                    
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
            indices = np.arange(1, min(n + 1, 11)) # Top 10 max
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

        bars = ax.bar(
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
