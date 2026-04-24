"""View for Multivariate Analysis (MVA) tab — UI controls and result plots."""
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QComboBox, QRadioButton, QButtonGroup,
    QGroupBox, QSplitter, QLineEdit, QScrollArea,
)
from PySide6.QtCore import Qt, Signal

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from spectroview import PLOT_POLICY_LIGHT


class VMVA(QWidget):
    """View for Multivariate Analysis (MVA) feature."""

    # ───── View → ViewModel signals ─────
    run_pca_requested = Signal(int)               # n_components
    run_nmf_requested = Signal(int, int, float)    # n_components, max_iter, tol
    send_to_graphs_requested = Signal(str)         # df_name

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

        # Warning label
        lbl_warning = QLabel("(MVA features are under developement. Do not use !)")
        lbl_warning.setStyleSheet("color: red;")
        lbl_warning.setAlignment(Qt.AlignCenter)
        lbl_warning.setWordWrap(True)
        left_layout.addWidget(lbl_warning)

        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Methods:"))
        self.rb_nmf = QRadioButton("NMF")
        self.rb_pca = QRadioButton("PCA")
        self.rb_nmf.setChecked(True)
        self.method_group = QButtonGroup(self)
        self.method_group.addButton(self.rb_pca)
        self.method_group.addButton(self.rb_nmf)
        
        method_layout.addWidget(self.rb_nmf)
        method_layout.addWidget(self.rb_pca)
        left_layout.addLayout(method_layout)

        # Parameters
        grp_params = QGroupBox("Parameters")
        params_layout = QVBoxLayout(grp_params)

        # Number of components
        row_nc = QHBoxLayout()
        row_nc.addWidget(QLabel("Components:"))
        self.spin_n_components = QSpinBox()
        self.spin_n_components.setRange(2, 50)
        self.spin_n_components.setValue(5)
        row_nc.addWidget(self.spin_n_components)
        params_layout.addLayout(row_nc)

        # NMF-specific: max iterations
        self.lbl_max_iter = QLabel("Max iterations:")
        self.spin_max_iter = QSpinBox()
        self.spin_max_iter.setRange(50, 5000)
        self.spin_max_iter.setValue(500)
        self.spin_max_iter.setSingleStep(50)
        row_iter = QHBoxLayout()
        row_iter.addWidget(self.lbl_max_iter)
        row_iter.addWidget(self.spin_max_iter)
        params_layout.addLayout(row_iter)

        # NMF-specific: tolerance
        self.lbl_tol = QLabel("Tolerance:")
        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setDecimals(6)
        self.spin_tol.setRange(1e-8, 1.0)
        self.spin_tol.setValue(1e-4)
        self.spin_tol.setSingleStep(1e-4)
        row_tol = QHBoxLayout()
        row_tol.addWidget(self.lbl_tol)
        row_tol.addWidget(self.spin_tol)
        params_layout.addLayout(row_tol)

        left_layout.addWidget(grp_params)

        # Run button
        self.btn_run = QPushButton("▶  Run Analysis")
        self.btn_run.setMinimumHeight(30)
        left_layout.addWidget(self.btn_run)

        # Plot type selector
        grp_plot = QGroupBox("Plot Type")
        plot_layout = QVBoxLayout(grp_plot)
        self.cbb_plot_type = QComboBox()
        self.cbb_plot_type.addItems(["Scree Plot", "Loadings", "Scores"])
        plot_layout.addWidget(self.cbb_plot_type)

        # Score axis selectors
        score_row = QHBoxLayout()
        score_row.addWidget(QLabel("X:"))
        self.cbb_score_x = QComboBox()
        self.cbb_score_x.setMinimumWidth(60)
        score_row.addWidget(self.cbb_score_x)
        score_row.addWidget(QLabel("Y:"))
        self.cbb_score_y = QComboBox()
        self.cbb_score_y.setMinimumWidth(60)
        score_row.addWidget(self.cbb_score_y)
        plot_layout.addLayout(score_row)

        left_layout.addWidget(grp_plot)

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

        # ── RIGHT PANEL: Matplotlib canvas ────────────────────────────
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        with plt.style.context(PLOT_POLICY_LIGHT):
            self.figure = Figure(layout="compressed", dpi=80)
            self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        # Set scroll area widget
        left_scroll_area.setWidget(left_panel)

        # Assemble splitter
        splitter.addWidget(left_scroll_area)
        splitter.addWidget(right_panel)
        splitter.setSizes([340,660])

        # ── Internal connections ──────────────────────────────────────
        self.rb_pca.toggled.connect(self._on_method_changed)
        self.rb_nmf.toggled.connect(self._on_method_changed)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_send.clicked.connect(self._on_send)
        self.cbb_plot_type.currentIndexChanged.connect(self._on_plot_type_changed)
        self.cbb_score_x.currentIndexChanged.connect(self._replot)
        self.cbb_score_y.currentIndexChanged.connect(self._replot)

        # Initial state
        self._on_method_changed()
        self._show_placeholder()

    # ══════════════════════════════════════════════════════════════════
    # Internal UI logic
    # ══════════════════════════════════════════════════════════════════

    def _on_method_changed(self):
        """Show/hide NMF-specific controls."""
        is_nmf = self.rb_nmf.isChecked()
        for w in (self.lbl_max_iter, self.spin_max_iter, self.lbl_tol, self.spin_tol):
            w.setVisible(is_nmf)

        # Update plot type options
        current = self.cbb_plot_type.currentText()
        self.cbb_plot_type.blockSignals(True)
        self.cbb_plot_type.clear()
        if self.rb_pca.isChecked():
            self.cbb_plot_type.addItems(["Scree Plot", "Loadings", "Scores"])
        else:
            self.cbb_plot_type.addItems(["Loadings", "Scores"])
        # Restore selection if valid
        idx = self.cbb_plot_type.findText(current)
        if idx >= 0:
            self.cbb_plot_type.setCurrentIndex(idx)
        self.cbb_plot_type.blockSignals(False)

    def _on_run(self):
        """Emit the appropriate signal based on selected method."""
        nc = self.spin_n_components.value()
        if self.rb_pca.isChecked():
            self.run_pca_requested.emit(nc)
        else:
            self.run_nmf_requested.emit(nc, self.spin_max_iter.value(), self.spin_tol.value())

    def _on_send(self):
        """Emit export signal."""
        name = self.ent_df_name.text().strip()
        if name:
            self.send_to_graphs_requested.emit(name)

    def _on_plot_type_changed(self):
        """Update score axis visibility and replot."""
        is_scores = self.cbb_plot_type.currentText() == "Scores"
        self.cbb_score_x.setVisible(is_scores)
        self.cbb_score_y.setVisible(is_scores)
        self._replot()

    def _show_placeholder(self):
        """Show a placeholder message on the canvas."""
        self.ax.clear()
        self.ax.text(
            0.5, 0.5,
            "Run an analysis to see results here",
            transform=self.ax.transAxes,
            ha="center", va="center",
            fontsize=13, color="#999", style="italic",
        )
        self.ax.set_axis_off()
        self.canvas.draw_idle()

    # ══════════════════════════════════════════════════════════════════
    # Public slots — called by ViewModel via signals
    # ══════════════════════════════════════════════════════════════════

    def display_pca_results(self, payload: dict):
        """Receive PCA results from ViewModel and plot."""
        self._pca_payload = payload
        self._nmf_payload = None
        self._populate_score_combos(payload["result"].scores.shape[1], prefix="PC")
        self._replot()

    def display_nmf_results(self, payload: dict):
        """Receive NMF results from ViewModel and plot."""
        self._nmf_payload = payload
        self._pca_payload = None
        self._populate_score_combos(payload["result"].W.shape[1], prefix="NMF")
        self._replot()

    def set_status(self, msg: str):
        """Update the status label."""
        self.lbl_status.setText(msg)

    # ══════════════════════════════════════════════════════════════════
    # Plotting
    # ══════════════════════════════════════════════════════════════════

    def _populate_score_combos(self, n_components: int, prefix: str = "PC"):
        """Fill the score axis comboboxes."""
        self.cbb_score_x.blockSignals(True)
        self.cbb_score_y.blockSignals(True)
        self.cbb_score_x.clear()
        self.cbb_score_y.clear()
        labels = [f"{prefix}{i+1}" for i in range(n_components)]
        self.cbb_score_x.addItems(labels)
        self.cbb_score_y.addItems(labels)
        if n_components >= 2:
            self.cbb_score_x.setCurrentIndex(0)
            self.cbb_score_y.setCurrentIndex(1)
        self.cbb_score_x.blockSignals(False)
        self.cbb_score_y.blockSignals(False)

    def _replot(self):
        """Dispatch to the correct plot method based on cached results and plot type."""
        plot_type = self.cbb_plot_type.currentText()

        if self._pca_payload is not None:
            result = self._pca_payload["result"]
            x_axis = self._pca_payload["x_axis"]
            fnames = self._pca_payload["fnames"]

            if plot_type == "Scree Plot":
                self._plot_scree(result.explained_variance_ratio, result.cumulative_variance)
            elif plot_type == "Loadings":
                self._plot_loadings(x_axis, result.loadings, prefix="PC")
            elif plot_type == "Scores":
                self._plot_scores(result.scores, fnames, prefix="PC")

        elif self._nmf_payload is not None:
            result = self._nmf_payload["result"]
            x_axis = self._nmf_payload["x_axis"]
            fnames = self._nmf_payload["fnames"]

            if plot_type == "Loadings":
                self._plot_loadings(x_axis, result.H, prefix="NMF")
            elif plot_type == "Scores":
                self._plot_scores(result.W, fnames, prefix="NMF")
        else:
            self._show_placeholder()

    def _plot_scree(self, explained_var_ratio, cumulative):
        """Bar chart of explained variance + cumulative line."""
        self.ax.clear()
        n = len(explained_var_ratio)
        indices = np.arange(1, n + 1)

        # Bars
        bars = self.ax.bar(
            indices, explained_var_ratio * 100,
            color="#5B9BD5", edgecolor="#2E6EA6", alpha=0.85, zorder=2,
        )
        # Annotate bars
        for bar, val in zip(bars, explained_var_ratio * 100):
            self.ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=8,
            )

        # Cumulative line
        ax2 = self.ax.twinx()
        ax2.plot(indices, cumulative * 100, "o-", color="#E87722", lw=2, ms=5, zorder=3)
        ax2.set_ylabel("Cumulative (%)")
        ax2.set_ylim(0, 105)

        self.ax.set_xlabel("Principal Component")
        self.ax.set_ylabel("Explained Variance (%)")
        self.ax.set_title("Scree Plot")
        self.ax.set_xticks(indices)
        self.ax.set_xlim(0.4, n + 0.6)

        self.canvas.draw_idle()

    def _plot_loadings(self, x_axis, loadings, prefix="PC"):
        """Overlay spectral loadings on wavenumber axis."""
        self.ax.clear()
        n_components = loadings.shape[0]
        cmap = plt.cm.get_cmap("tab10")

        for i in range(n_components):
            color = cmap(i % 10)
            self.ax.plot(
                x_axis, loadings[i],
                lw=1.2, color=color,
                label=f"{prefix}{i+1}",
            )

        self.ax.set_xlabel("Wavenumber")
        self.ax.set_ylabel("Loading")
        self.ax.set_title(f"{prefix} Loadings")
        self.ax.legend(loc="best", fontsize=8)
        self.canvas.draw_idle()

    def _plot_scores(self, scores, fnames, prefix="PC"):
        """2D scatter of scores with spectrum labels."""
        self.ax.clear()
        ix = self.cbb_score_x.currentIndex()
        iy = self.cbb_score_y.currentIndex()

        if ix < 0 or iy < 0 or ix >= scores.shape[1] or iy >= scores.shape[1]:
            return

        x_data = scores[:, ix]
        y_data = scores[:, iy]

        self.ax.scatter(x_data, y_data, s=50, c="#5B9BD5", edgecolors="#2E6EA6", zorder=3)

        # Label each point
        for i, fname in enumerate(fnames):
            self.ax.annotate(
                fname, (x_data[i], y_data[i]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=7, alpha=0.8,
            )

        self.ax.set_xlabel(f"{prefix}{ix+1}")
        self.ax.set_ylabel(f"{prefix}{iy+1}")
        self.ax.set_title(f"Scores: {prefix}{ix+1} vs {prefix}{iy+1}")
        self.ax.axhline(0, color="gray", lw=0.5, ls="--")
        self.ax.axvline(0, color="gray", lw=0.5, ls="--")
        self.canvas.draw_idle()
