# spectroview/view/components/customize_graph/tab_plot_styles.py
"""Plot Style and Fitting tab: style options, unified error bars, trendline equations, and histogram controls."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from spectroview.view.components.customize_graph.common import (
    create_scrollable_container,
    highlight_group_style,
)
from spectroview.view.theme import Icons

_ERROR_BAR_STYLES = ("point", "line", "bar")
_ERROR_BAR_LABELS_POINT_LINE = [
    ("None", "none"), ("Std. Dev.", "sd"), ("Std. Error", "sem"), ("95% CI", "ci95"),
]
_ERROR_BAR_LABELS_BAR = [
    ("Std. Dev.", "sd"), ("Std. Error", "sem"), ("95% CI", "ci95"),
]


class TabPlotStyles(QWidget):
    """Category 6: Plot Style & Fitting."""

    KEYWORDS = [
        "plot style", "style", "options", "point", "scatter", "dodge",
        "join", "wafer", "statistics", "wafer stats", "error bar",
        "error bars", "capsize", "sd", "sem", "ci95", "std dev",
        "trendline", "fit", "fitting", "polynomial", "order", "anchor",
        "equation", "r2", "r-squared", "copy", "histogram", "bins",
        "kde", "step", "filled",
    ]

    def __init__(self, graph_widget: QWidget, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.graph_widget = graph_widget
        self._setup_ui()
        self.load_settings()

    def switch_graph(self, graph_widget: QWidget) -> None:
        """Switch to a new graph widget and reload settings."""
        self.graph_widget = graph_widget
        self.load_settings()

    def _setup_ui(self) -> None:
        self._scroll, _container, layout = create_scrollable_container(self)

        # ───── Plot Style Options ─────
        self.grp_style_options = QGroupBox("Plot Style Options:")
        opt_layout = QVBoxLayout(self.grp_style_options)
        opt_layout.setContentsMargins(8, 8, 8, 8)
        opt_layout.setSpacing(6)

        self._cb_join = QCheckBox("Join data points (point plot)")
        self._cb_join.setToolTip("Connect consecutive data points with line segments in point plot mode.")
        opt_layout.addWidget(self._cb_join)

        self._cb_dodge = QCheckBox("Dodge overlapping points (point plot)")
        self._cb_dodge.setToolTip("Separate overlapping data points along category intervals in point plot mode.")
        opt_layout.addWidget(self._cb_dodge)

        self._cb_dodge_scatter = QCheckBox("Dodge overlapping points (scatter plot)")
        self._cb_dodge_scatter.setToolTip("Slightly shift overlapping points horizontally to reduce visual overplotting.")
        opt_layout.addWidget(self._cb_dodge_scatter)

        self._cb_wafer_stats = QCheckBox("Show statistics summary (wafer plot)")
        self._cb_wafer_stats.setToolTip("Display statistical summary overlay (Mean, Std Dev, Min, Max, Uniformity) for wafer maps.")
        opt_layout.addWidget(self._cb_wafer_stats)

        layout.addWidget(self.grp_style_options)

        # ───── Error Bars (Unified) ─────
        self.grp_error_bars = QGroupBox("Error Bars:")
        eb_layout = QVBoxLayout(self.grp_error_bars)
        eb_layout.setContentsMargins(8, 8, 8, 8)
        eb_layout.setSpacing(6)

        self._cb_error_bar = QCheckBox("Show error bars (bar plot)")
        self._cb_error_bar.setToolTip("Toggle error bar visibility on bar plots.")
        eb_layout.addWidget(self._cb_error_bar)

        eb_ctrl_row = QHBoxLayout()
        lbl_eb_type = QLabel("Type:")
        lbl_eb_type.setToolTip("Dispersion or confidence calculation for error bars (Std Dev, Std Error, or 95% Confidence Interval).")
        eb_ctrl_row.addWidget(lbl_eb_type)
        self.cbb_error_bar_type = QComboBox()
        self.cbb_error_bar_type.setMinimumWidth(130)
        self.cbb_error_bar_type.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        if self.cbb_error_bar_type.view():
            self.cbb_error_bar_type.view().setMinimumWidth(130)
        self.cbb_error_bar_type.setToolTip("Dispersion or confidence calculation for error bars.")
        eb_ctrl_row.addWidget(self.cbb_error_bar_type)

        eb_ctrl_row.addSpacing(16)
        lbl_cap = QLabel("Cap size:")
        lbl_cap.setToolTip("Width of horizontal crossbars at error bar endpoints in points (pt).")
        eb_ctrl_row.addWidget(lbl_cap)
        self.spin_error_bar_capsize = QDoubleSpinBox()
        self.spin_error_bar_capsize.setRange(0.0, 20.0)
        self.spin_error_bar_capsize.setDecimals(1)
        self.spin_error_bar_capsize.setSingleStep(0.5)
        self.spin_error_bar_capsize.setValue(3.0)
        self.spin_error_bar_capsize.setMaximumWidth(70)
        self.spin_error_bar_capsize.setToolTip("Width of horizontal crossbars at error bar endpoints in points (pt).")
        eb_ctrl_row.addWidget(self.spin_error_bar_capsize)

        eb_ctrl_row.addStretch()
        eb_layout.addLayout(eb_ctrl_row)
        layout.addWidget(self.grp_error_bars)

        # ───── Trendline Fitting (trendline style only) ─────
        self._trendline_group = QGroupBox("Trendline Fitting:")
        tl_layout = QVBoxLayout(self._trendline_group)
        tl_layout.setContentsMargins(8, 8, 8, 8)
        tl_layout.setSpacing(6)

        order_row = QHBoxLayout()
        lbl_ord = QLabel("Polynomial order:")
        lbl_ord.setToolTip("Degree of polynomial fit (1 = Linear, 2 = Quadratic, etc.).")
        order_row.addWidget(lbl_ord)
        self._spin_order = QSpinBox()
        self._spin_order.setRange(1, 10)
        self._spin_order.setValue(1)
        self._spin_order.setMaximumWidth(60)
        self._spin_order.setToolTip("Degree of polynomial fit (1 = Linear, 2 = Quadratic, etc.).")
        order_row.addWidget(self._spin_order)
        order_row.addWidget(QLabel("<i>(Apply to recompute equations)</i>"))
        order_row.addStretch()
        tl_layout.addLayout(order_row)

        self._anchor_grp = QGroupBox("Anchor point")
        self._anchor_grp.setCheckable(True)
        self._anchor_grp.setChecked(False)
        self._anchor_grp.setToolTip("Force polynomial fitted curve to pass through a specific fixed coordinate.")
        anchor_layout = QVBoxLayout(self._anchor_grp)
        anchor_layout.setContentsMargins(6, 6, 6, 6)
        anchor_layout.setSpacing(6)

        self._rb_origin = QRadioButton("Through origin (0, 0)")
        self._rb_origin.setChecked(True)
        self._rb_origin.setToolTip("Force trendline curve through (0, 0).")
        self._rb_custom = QRadioButton("Custom point:")
        self._rb_custom.setToolTip("Force trendline curve through custom (X₀, Y₀).")
        anchor_layout.addWidget(self._rb_origin)

        custom_row = QHBoxLayout()
        custom_row.addWidget(self._rb_custom)
        custom_row.addWidget(QLabel("X₀:"))
        self._spin_ax = QDoubleSpinBox()
        self._spin_ax.setRange(-999999, 999999)
        self._spin_ax.setValue(0.0)
        self._spin_ax.setMaximumWidth(90)
        self._spin_ax.setToolTip("Fixed X anchor coordinate.")
        custom_row.addWidget(self._spin_ax)
        custom_row.addWidget(QLabel("Y₀:"))
        self._spin_ay = QDoubleSpinBox()
        self._spin_ay.setRange(-999999, 999999)
        self._spin_ay.setValue(0.0)
        self._spin_ay.setMaximumWidth(90)
        self._spin_ay.setToolTip("Fixed Y anchor coordinate.")
        custom_row.addWidget(self._spin_ay)
        custom_row.addStretch()
        anchor_layout.addLayout(custom_row)

        self._rb_origin.toggled.connect(lambda on: self._spin_ax.setEnabled(not on))
        self._rb_origin.toggled.connect(lambda on: self._spin_ay.setEnabled(not on))
        self._spin_ax.setEnabled(False)
        self._spin_ay.setEnabled(False)
        tl_layout.addWidget(self._anchor_grp)

        eq_header_row = QHBoxLayout()
        eq_header_row.addWidget(QLabel("Fit Equation(s):"))
        self._btn_copy_eq = QPushButton("Copy")
        self._btn_copy_eq.setIcon(Icons.copy())
        self._btn_copy_eq.setMaximumWidth(80)
        self._btn_copy_eq.setToolTip("Copy fit equations and R² coefficients to clipboard.")
        self._btn_copy_eq.clicked.connect(self._copy_equations)
        eq_header_row.addStretch()
        eq_header_row.addWidget(self._btn_copy_eq)
        tl_layout.addLayout(eq_header_row)

        self._eq_table = QTableWidget(0, 3)
        self._eq_table.setHorizontalHeaderLabels(["Group", "Equation", "R²"])
        self._eq_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._eq_table.setMinimumHeight(90)
        tl_layout.addWidget(self._eq_table)

        layout.addWidget(self._trendline_group)

        # ───── Histogram Settings (histogram style only) ─────
        self._histogram_group = QGroupBox("Histogram Settings:")
        hist_layout = QVBoxLayout(self._histogram_group)
        hist_layout.setContentsMargins(8, 8, 8, 8)
        hist_layout.setSpacing(6)

        bins_row = QHBoxLayout()
        lbl_bins = QLabel("Bins count:")
        lbl_bins.setToolTip("Number of equal-width bins used to partition the data range.")
        bins_row.addWidget(lbl_bins)
        self._spin_bins = QSpinBox()
        self._spin_bins.setRange(2, 500)
        self._spin_bins.setSingleStep(10)
        self._spin_bins.setValue(20)
        self._spin_bins.setMaximumWidth(80)
        self._spin_bins.setToolTip("Number of equal-width bins used to partition the data range.")
        bins_row.addWidget(self._spin_bins)
        bins_row.addStretch()
        hist_layout.addLayout(bins_row)

        self._cb_kde = QCheckBox("Overlay KDE density curve")
        self._cb_kde.setToolTip("Plot a smooth Gaussian Kernel Density Estimate curve alongside bars.")
        hist_layout.addWidget(self._cb_kde)

        step_row = QHBoxLayout()
        lbl_fill = QLabel("Fill style:")
        lbl_fill.setToolTip("Bar appearance: solid filled bars or outline step style.")
        step_row.addWidget(lbl_fill)
        self._rb_filled = QRadioButton("Filled")
        self._rb_filled.setToolTip("Render histogram with solid filled bars.")
        self._rb_step = QRadioButton("Step (outline)")
        self._rb_step.setToolTip("Render histogram with outline step boundary.")
        self._rb_filled.setChecked(True)
        step_row.addWidget(self._rb_filled)
        step_row.addWidget(self._rb_step)
        step_row.addStretch()
        hist_layout.addLayout(step_row)

        layout.addWidget(self._histogram_group)
        layout.addStretch()

    def _refresh_equation_table(self) -> None:
        equations = getattr(self.graph_widget, "trendline_equations", [])
        self._eq_table.setRowCount(len(equations))
        for row, entry in enumerate(equations):
            self._eq_table.setItem(row, 0, QTableWidgetItem(str(entry.get("label", ""))))
            self._eq_table.setItem(row, 1, QTableWidgetItem(str(entry.get("equation", ""))))
            self._eq_table.setItem(row, 2, QTableWidgetItem(str(entry.get("r2", ""))))
        self._eq_table.resizeColumnsToContents()

    def _copy_equations(self) -> None:
        equations = getattr(self.graph_widget, "trendline_equations", [])
        if not equations:
            return
        lines = ["Group\tEquation\tR²"]
        for entry in equations:
            lines.append(f"{entry.get('label', '')}\t{entry.get('equation', '')}\t{entry.get('r2', '')}")
        QApplication.clipboard().setText("\n".join(lines))

    def _load_error_bar_settings(self) -> None:
        style = getattr(self.graph_widget, "plot_style", "")
        self.grp_error_bars.setVisible(style in _ERROR_BAR_STYLES)
        if style not in _ERROR_BAR_STYLES:
            return

        is_bar = (style == "bar")
        self._cb_error_bar.setVisible(is_bar)
        self._cb_error_bar.setChecked(getattr(self.graph_widget, "show_bar_plot_error_bar", True))

        labels = _ERROR_BAR_LABELS_BAR if is_bar else _ERROR_BAR_LABELS_POINT_LINE
        current_value = (
            getattr(self.graph_widget, "bar_error_bar_type", "sd") if is_bar
            else getattr(self.graph_widget, "error_bar_type", "ci95")
        )

        self.cbb_error_bar_type.blockSignals(True)
        self.cbb_error_bar_type.clear()
        for text, value in labels:
            self.cbb_error_bar_type.addItem(text, value)
        idx = self.cbb_error_bar_type.findData(current_value)
        self.cbb_error_bar_type.setCurrentIndex(idx if idx >= 0 else 0)
        self.cbb_error_bar_type.blockSignals(False)

        self.spin_error_bar_capsize.setValue(getattr(self.graph_widget, "error_bar_capsize", 3.0))

    def load_settings(self) -> None:
        """Load plot style options into controls from graph widget."""
        gw = self.graph_widget
        style = getattr(gw, "plot_style", "")

        # Style options
        self._cb_join.setChecked(getattr(gw, "join_for_point_plot", False))
        self._cb_dodge.setChecked(getattr(gw, "dodge_point_plot", True))
        self._cb_dodge_scatter.setChecked(getattr(gw, "dodge_scatter_plot", False))
        self._cb_wafer_stats.setChecked(getattr(gw, "wafer_stats", True))

        self._cb_join.setEnabled(style == "point")
        self._cb_dodge.setEnabled(style == "point")
        self._cb_dodge_scatter.setEnabled(style == "scatter")
        self._cb_wafer_stats.setEnabled(style == "wafer")

        # Error bars
        self._load_error_bar_settings()

        # Trendline
        is_trendline = (style == "trendline")
        self._trendline_group.setVisible(is_trendline)
        if is_trendline:
            self._spin_order.setValue(getattr(gw, "trendline_order", 1))
            anchor_on = getattr(gw, "trendline_anchor_enabled", False)
            self._anchor_grp.setChecked(anchor_on)
            origin = getattr(gw, "trendline_anchor_origin", True)
            self._rb_origin.setChecked(origin)
            self._rb_custom.setChecked(not origin)
            self._spin_ax.setValue(getattr(gw, "trendline_anchor_x", 0.0))
            self._spin_ay.setValue(getattr(gw, "trendline_anchor_y", 0.0))
            self._refresh_equation_table()

        # Histogram
        is_hist = (style == "histogram")
        self._histogram_group.setVisible(is_hist)
        if is_hist:
            self._spin_bins.setValue(getattr(gw, "hist_bins", 20))
            self._cb_kde.setChecked(getattr(gw, "hist_kde", False))
            step = getattr(gw, "hist_step", False)
            self._rb_step.setChecked(step)
            self._rb_filled.setChecked(not step)

    def apply_settings(self, replot: bool = False) -> dict:
        """Write controls back to graph widget, returning changed property dict."""
        gw = self.graph_widget
        style = getattr(gw, "plot_style", "")

        gw.join_for_point_plot = self._cb_join.isChecked()
        gw.dodge_point_plot = self._cb_dodge.isChecked()
        gw.dodge_scatter_plot = self._cb_dodge_scatter.isChecked()
        gw.wafer_stats = self._cb_wafer_stats.isChecked()

        gw.show_bar_plot_error_bar = self._cb_error_bar.isChecked()
        if style == "bar":
            gw.bar_error_bar_type = self.cbb_error_bar_type.currentData()
        elif style in ("point", "line"):
            gw.error_bar_type = self.cbb_error_bar_type.currentData()
        gw.error_bar_capsize = self.spin_error_bar_capsize.value()

        props = {
            "join_for_point_plot": gw.join_for_point_plot,
            "dodge_point_plot": gw.dodge_point_plot,
            "dodge_scatter_plot": gw.dodge_scatter_plot,
            "wafer_stats": gw.wafer_stats,
            "show_bar_plot_error_bar": gw.show_bar_plot_error_bar,
            "error_bar_capsize": gw.error_bar_capsize,
        }
        if style == "bar":
            props["bar_error_bar_type"] = gw.bar_error_bar_type
        elif style in ("point", "line"):
            props["error_bar_type"] = gw.error_bar_type

        if style == "trendline":
            gw.trendline_order = self._spin_order.value()
            gw.trendline_anchor_enabled = self._anchor_grp.isChecked()
            gw.trendline_anchor_origin = self._rb_origin.isChecked()
            gw.trendline_anchor_x = self._spin_ax.value()
            gw.trendline_anchor_y = self._spin_ay.value()
            props.update({
                "trendline_order": gw.trendline_order,
                "trendline_anchor_enabled": gw.trendline_anchor_enabled,
                "trendline_anchor_origin": gw.trendline_anchor_origin,
                "trendline_anchor_x": gw.trendline_anchor_x,
                "trendline_anchor_y": gw.trendline_anchor_y,
            })

        if style == "histogram":
            gw.hist_bins = self._spin_bins.value()
            gw.hist_kde = self._cb_kde.isChecked()
            gw.hist_step = self._rb_step.isChecked()
            props.update({
                "hist_bins": gw.hist_bins,
                "hist_kde": gw.hist_kde,
                "hist_step": gw.hist_step,
            })

        if replot:
            if getattr(gw, "df", None) is not None:
                gw.plot(gw.df)
            if style == "trendline":
                self._refresh_equation_table()

        return props

    # Alias for existing test calls
    _apply = apply_settings

    def highlight_search(self, query: str) -> None:
        """Highlight matching groupboxes based on search query."""
        q = query.strip().lower()
        self.grp_style_options.setStyleSheet("")
        self.grp_error_bars.setStyleSheet("")
        self._trendline_group.setStyleSheet("")
        self._histogram_group.setStyleSheet("")

        if not q:
            return

        opt_terms = ["join", "dodge", "wafer", "stats", "point", "scatter"]
        eb_terms = ["error", "error bar", "error bars", "capsize", "sd", "sem", "ci95", "std dev"]
        tl_terms = ["trendline", "fit", "fitting", "polynomial", "order", "anchor", "origin", "equation", "r2", "r-squared"]
        hist_terms = ["histogram", "bins", "kde", "density", "step", "filled"]

        first_match: Optional[QWidget] = None
        if any(t in q or q in t for t in opt_terms):
            self.grp_style_options.setStyleSheet(highlight_group_style())
            if first_match is None:
                first_match = self.grp_style_options

        if any(t in q or q in t for t in eb_terms):
            self.grp_error_bars.setStyleSheet(highlight_group_style())
            if first_match is None:
                first_match = self.grp_error_bars

        if any(t in q or q in t for t in tl_terms):
            self._trendline_group.setStyleSheet(highlight_group_style())
            if first_match is None:
                first_match = self._trendline_group

        if any(t in q or q in t for t in hist_terms):
            self._histogram_group.setStyleSheet(highlight_group_style())
            if first_match is None:
                first_match = self._histogram_group

        if first_match:
            self._scroll.ensureWidgetVisible(first_match, 0, 30)

