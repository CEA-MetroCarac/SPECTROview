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

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from spectroview.view.components.lower_tabs.v_mva_plots import MVAPlotMixin, _PlotTab
from spectroview.view.theme import Icons, current_tokens


class VMVA(QWidget, MVAPlotMixin):
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
        left_scroll_area.setFrameShape(QScrollArea.NoFrame)
        left_scroll_area.setWidgetResizable(True)
        left_scroll_area.setMaximumWidth(340)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)
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
        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.setIcon(Icons.play())
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
        self.btn_send.setIcon(Icons.send())
        export_layout.addWidget(self.btn_send)
        left_layout.addWidget(grp_export)

        # Status
        tok = current_tokens()
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(f"color: {tok.text_muted}; font-style: italic;")
        left_layout.addWidget(self.lbl_status)

        left_layout.addStretch()

        # ── RIGHT PANEL: Tabbed plot area ─────────────────────────────
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(3, 0, 0, 0)

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

    def clear(self):
        """Clear all results, dropdown choices, and reset plots to placeholders."""
        self._pca_payload = None
        self._nmf_payload = None
        self.cbb_score_x.clear()
        self.cbb_score_y.clear()
        self.cbb_heatmap_comp.clear()
        self._show_all_placeholders()
        self.set_status("Ready")

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

    def apply_theme(self, theme: str = ""):
        """Update icons and colors on theme change."""
        tok = current_tokens()
        self.btn_run.setIcon(Icons.play(color=tok.text_primary))
        self.btn_send.setIcon(Icons.send(color=tok.text_primary))
        self.lbl_status.setStyleSheet(f"color: {tok.text_muted}; font-style: italic;")
