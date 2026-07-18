"""'More options' tab of the Customize Graph dialog: adaptive controls
(join/dodge/error-bar/wafer-stats, data sorting, trendline, histogram) shown
based on the current plot_style.

Split out of customize_graph_dialog.py; no behavior changes.
"""
from PySide6.QtGui import QIcon, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QRadioButton, QLineEdit,
    QApplication, QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QHeaderView, QColorDialog,
)

from spectroview import ICON_DIR

# Text <-> MGraph.figure_theme value.
_THEME_TEXT_MAP = {"Light": "light", "Dark": "dark", "Soft Dark": "soft_dark"}
_THEME_VALUE_TEXT = {v: k for k, v in _THEME_TEXT_MAP.items()}


class CustomizeMoreOptions(QWidget):
    """Adaptive 'More Options' tab — shows controls relevant to the current plot_style."""

    def __init__(self, graph_widget, parent=None):
        super().__init__(parent)
        self.graph_widget = graph_widget
        self._setup_ui()
        self.load_settings()

    def switch_graph(self, graph_widget):
        """Switch to a new graph and reload settings."""
        self.graph_widget = graph_widget
        self.load_settings()

    # ------------------------------------------------------------------ #
    #  UI Construction
    # ------------------------------------------------------------------ #

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Scrollable inner area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        inner = QWidget()
        self._inner_layout = QVBoxLayout(inner)
        self._inner_layout.setContentsMargins(4, 4, 4, 4)
        self._inner_layout.setSpacing(8)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # ---- General (always visible) ----
        self._build_general_section()

        # ---- Figure style section (always visible) ----
        self._build_figure_style_section()

        # ---- Data sorting section ----
        self._build_sorting_section()

        # ---- Trendline section ----
        self._build_trendline_section()

        # ---- Histogram section ----
        self._build_histogram_section()

        self._inner_layout.addStretch()

    # ---- General section ------------------------------------------------

    def _build_general_section(self):
        grp = QGroupBox("Plot options:")
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Join points (point plot)
        self._cb_join = QCheckBox("Join data points (point plot)")
        layout.addWidget(self._cb_join)

        # Dodge points (point plot)
        self._cb_dodge = QCheckBox("Dodge overlapping points (point plot)")
        layout.addWidget(self._cb_dodge)

        # Dodge points (scatter plot)
        self._cb_dodge_scatter = QCheckBox("Dodge overlapping points (scatter plot)")
        layout.addWidget(self._cb_dodge_scatter)

        # Error bar (bar plot)
        self._cb_error_bar = QCheckBox("Show error bar (bar plot)")
        layout.addWidget(self._cb_error_bar)

        # Wafer stats (wafer plot)
        self._cb_wafer_stats = QCheckBox("Show statistics (wafer plot)")
        layout.addWidget(self._cb_wafer_stats)

        self._general_group = grp
        self._inner_layout.addWidget(grp)

    # ---- Figure style section --------------------------------------------

    def _build_figure_style_section(self):
        grp = QGroupBox("Figure style:")
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Theme (selects the underlying mplstyle -- background/text/grid
        # colors as a set; figure_facecolor below is a separate, more
        # specific override layered on top of whichever theme is chosen)
        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel("Theme:"))
        self._combo_figure_theme = QComboBox()
        self._combo_figure_theme.addItems(["Light", "Dark", "Soft Dark"])
        theme_row.addWidget(self._combo_figure_theme)
        theme_row.addStretch()
        layout.addLayout(theme_row)

        # Background color
        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("Background color:"))
        self._btn_figure_facecolor = QPushButton("(default)")
        self._btn_figure_facecolor.setFixedWidth(90)
        self._btn_figure_facecolor.clicked.connect(self._pick_figure_facecolor)
        bg_row.addWidget(self._btn_figure_facecolor)
        bg_row.addStretch()
        layout.addLayout(bg_row)

        # Subtitle
        sub_row = QHBoxLayout()
        sub_row.addWidget(QLabel("Subtitle:"))
        self._edit_subtitle = QLineEdit()
        self._edit_subtitle.setPlaceholderText("Optional text shown under the title")
        sub_row.addWidget(self._edit_subtitle)
        sub_row.addWidget(QLabel("Font size:"))
        self._spin_subtitle_fontsize = QSpinBox()
        self._spin_subtitle_fontsize.setRange(4, 72)
        self._spin_subtitle_fontsize.setSingleStep(1)
        self._spin_subtitle_fontsize.setValue(10)  # matches the pre-existing hardcoded fallback
        self._spin_subtitle_fontsize.setMaximumWidth(60)
        sub_row.addWidget(self._spin_subtitle_fontsize)
        layout.addLayout(sub_row)

        # Spine visibility
        spine_row = QHBoxLayout()
        spine_row.addWidget(QLabel("Show spines:"))
        self._cb_spine_top = QCheckBox("Top")
        self._cb_spine_right = QCheckBox("Right")
        self._cb_spine_bottom = QCheckBox("Bottom")
        self._cb_spine_left = QCheckBox("Left")
        for cb in (self._cb_spine_top, self._cb_spine_right, self._cb_spine_bottom, self._cb_spine_left):
            spine_row.addWidget(cb)
        spine_row.addStretch()
        layout.addLayout(spine_row)

        # Margins
        margin_row = QHBoxLayout()
        margin_row.addWidget(QLabel("Margins:"))
        margin_row.addWidget(QLabel("X"))
        self._spin_x_margin = QDoubleSpinBox()
        self._spin_x_margin.setRange(0.0, 1.0)
        self._spin_x_margin.setDecimals(2)
        self._spin_x_margin.setSingleStep(0.05)
        self._spin_x_margin.setValue(0.05)  # matches matplotlib's own default axes.xmargin
        margin_row.addWidget(self._spin_x_margin)
        margin_row.addWidget(QLabel("Y"))
        self._spin_y_margin = QDoubleSpinBox()
        self._spin_y_margin.setRange(0.0, 1.0)
        self._spin_y_margin.setDecimals(2)
        self._spin_y_margin.setSingleStep(0.05)
        self._spin_y_margin.setValue(0.05)  # matches matplotlib's own default axes.ymargin
        margin_row.addWidget(self._spin_y_margin)
        margin_row.addStretch()
        layout.addLayout(margin_row)

        self._figure_style_group = grp
        self._inner_layout.addWidget(grp)

    def _pick_figure_facecolor(self):
        """Open color picker for the figure/axes background override."""
        current = QColor(self._btn_figure_facecolor.text()) if self._btn_figure_facecolor.text() != "(default)" else QColor('white')
        color = QColorDialog.getColor(current, self, "Select Background Color")
        if color.isValid():
            self._btn_figure_facecolor.setText(color.name())
            self._btn_figure_facecolor.setStyleSheet(f"background-color: {color.name()};")

    # ---- Data sorting section -------------------------------------------

    def _build_sorting_section(self):
        grp = QGroupBox("Data sorting:")
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        sort_row = QHBoxLayout()

        # Checkbox to enable/disable intelligent sorting
        self._cb_sort_enabled = QCheckBox("Enable intelligent data sorting")
        self._cb_sort_enabled.setChecked(True)
        self._cb_sort_enabled.toggled.connect(self._on_sort_enabled_toggled)
        sort_row.addWidget(self._cb_sort_enabled)

        self._lbl_sort_by = QLabel("Sort by:")
        sort_row.addWidget(self._lbl_sort_by)

        self._cbb_sort_by = QComboBox()
        self._cbb_sort_by.addItems(["Z (hue values)", "X values", "Y values"])
        self._cbb_sort_by.setCurrentIndex(0)  # Default: Z
        self._cbb_sort_by.setMaximumWidth(160)
        sort_row.addWidget(self._cbb_sort_by)

        sort_row.addStretch()
        layout.addLayout(sort_row)

        # Info label
        info = QLabel("Sorts legend and data order for consistent, deterministic plots.")
        info.setStyleSheet("color: gray; font-style: italic; font-size: 10px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        self._sorting_group = grp
        self._inner_layout.addWidget(grp)

    # ---- Trendline section ----------------------------------------------

    def _build_trendline_section(self):
        grp = QGroupBox("Trendline settings")
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Polynomial order
        order_row = QHBoxLayout()
        order_row.addWidget(QLabel("Polynomial order:"))
        self._spin_order = QSpinBox()
        self._spin_order.setRange(1, 10)
        self._spin_order.setValue(1)
        self._spin_order.setMaximumWidth(60)
        order_row.addWidget(self._spin_order)
        order_row.addWidget(QLabel("<i>(Click 'Apply' to refresh equation)</i>"))
        order_row.addStretch()
        layout.addLayout(order_row)

        # Anchor group
        anchor_grp = QGroupBox("Anchor point")
        anchor_grp.setCheckable(True)
        anchor_grp.setChecked(False)
        anchor_layout = QVBoxLayout(anchor_grp)
        anchor_layout.setContentsMargins(4, 4, 4, 4)
        anchor_layout.setSpacing(8)
        self._anchor_grp = anchor_grp

        self._rb_origin = QRadioButton("Through origin (0, 0)")
        self._rb_origin.setChecked(True)
        self._rb_custom = QRadioButton("Custom point:")
        anchor_layout.addWidget(self._rb_origin)

        custom_row = QHBoxLayout()
        custom_row.addWidget(self._rb_custom)
        custom_row.addWidget(QLabel("X₀"))
        self._spin_ax = QDoubleSpinBox()
        self._spin_ax.setRange(-999999, 999999)
        self._spin_ax.setValue(0.0)
        self._spin_ax.setMaximumWidth(90)
        custom_row.addWidget(self._spin_ax)
        custom_row.addWidget(QLabel("Y₀"))
        self._spin_ay = QDoubleSpinBox()
        self._spin_ay.setRange(-999999, 999999)
        self._spin_ay.setValue(0.0)
        self._spin_ay.setMaximumWidth(90)
        custom_row.addWidget(self._spin_ay)
        custom_row.addStretch()
        anchor_layout.addLayout(custom_row)

        # Enable/disable spinboxes based on radio selection
        self._rb_origin.toggled.connect(lambda on: self._spin_ax.setEnabled(not on))
        self._rb_origin.toggled.connect(lambda on: self._spin_ay.setEnabled(not on))
        self._spin_ax.setEnabled(False)
        self._spin_ay.setEnabled(False)

        layout.addWidget(anchor_grp)

        # Equation table
        eq_label_row = QHBoxLayout()
        eq_label_row.addWidget(QLabel("Fit equation(s):"))
        self._btn_copy_eq = QPushButton("Copy")
        self._btn_copy_eq.setIcon(QIcon(f"{ICON_DIR}/copy.png"))
        self._btn_copy_eq.setMaximumWidth(80)
        self._btn_copy_eq.clicked.connect(self._copy_equations)
        eq_label_row.addStretch()
        eq_label_row.addWidget(self._btn_copy_eq)
        layout.addLayout(eq_label_row)

        self._eq_table = QTableWidget(0, 3)
        self._eq_table.setHorizontalHeaderLabels(["Group", "Equation", "R²"])
        self._eq_table.horizontalHeader().setStretchLastSection(False)
        self._eq_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._eq_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._eq_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._eq_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._eq_table.setMinimumHeight(90)
        layout.addWidget(self._eq_table)

        self._trendline_group = grp
        self._inner_layout.addWidget(grp)

    # ---- Histogram section ----------------------------------------------

    def _build_histogram_section(self):
        grp = QGroupBox("Histogram settings")
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        bins_row = QHBoxLayout()
        bins_row.addWidget(QLabel("Bins:"))
        self._spin_bins = QSpinBox()
        self._spin_bins.setRange(2, 500)
        self._spin_bins.setSingleStep(10)
        self._spin_bins.setValue(20)
        self._spin_bins.setMaximumWidth(80)
        bins_row.addWidget(self._spin_bins)
        bins_row.addStretch()
        layout.addLayout(bins_row)

        self._cb_kde = QCheckBox("Overlay KDE curve")
        layout.addWidget(self._cb_kde)

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Fill style:"))
        self._rb_filled = QRadioButton("Filled")
        self._rb_step = QRadioButton("Step (outline)")
        self._rb_filled.setChecked(True)
        step_row.addWidget(self._rb_filled)
        step_row.addWidget(self._rb_step)
        step_row.addStretch()
        layout.addLayout(step_row)

        self._histogram_group = grp
        self._inner_layout.addWidget(grp)

    # ------------------------------------------------------------------ #
    #  Load / Apply
    # ------------------------------------------------------------------ #

    def load_settings(self):
        """Load all settings from the graph widget and update visibility."""
        gw = self.graph_widget
        style = gw.plot_style

        # --- General section ---
        self._cb_join.setChecked(getattr(gw, 'join_for_point_plot', False))
        self._cb_dodge.setChecked(getattr(gw, 'dodge_point_plot', True))
        self._cb_dodge_scatter.setChecked(getattr(gw, 'dodge_scatter_plot', False))
        self._cb_error_bar.setChecked(getattr(gw, 'show_bar_plot_error_bar', True))
        self._cb_wafer_stats.setChecked(getattr(gw, 'wafer_stats', True))

        # Highlight relevant checkboxes based on style
        self._cb_join.setEnabled(style == 'point')
        self._cb_dodge.setEnabled(style == 'point')
        self._cb_dodge_scatter.setEnabled(style == 'scatter')
        self._cb_error_bar.setEnabled(style == 'bar')
        self._cb_wafer_stats.setEnabled(style == 'wafer')

        # --- Figure style section ---
        self._combo_figure_theme.setCurrentText(
            _THEME_VALUE_TEXT.get(getattr(gw, 'figure_theme', 'light'), "Light")
        )

        facecolor = getattr(gw, 'figure_facecolor', None)
        if facecolor:
            self._btn_figure_facecolor.setText(facecolor)
            self._btn_figure_facecolor.setStyleSheet(f"background-color: {facecolor};")
        else:
            self._btn_figure_facecolor.setText("(default)")
            self._btn_figure_facecolor.setStyleSheet("")

        self._edit_subtitle.setText(getattr(gw, 'plot_subtitle', None) or "")
        self._spin_subtitle_fontsize.setValue(getattr(gw, 'subtitle_fontsize', None) or 10)

        spines = getattr(gw, 'spines_visible', None) or {'top': True, 'right': True, 'bottom': True, 'left': True}
        self._cb_spine_top.setChecked(spines.get('top', True))
        self._cb_spine_right.setChecked(spines.get('right', True))
        self._cb_spine_bottom.setChecked(spines.get('bottom', True))
        self._cb_spine_left.setChecked(spines.get('left', True))

        margins = getattr(gw, 'figure_margins', None) or [0.05, 0.05]
        self._spin_x_margin.setValue(margins[0])
        self._spin_y_margin.setValue(margins[1])

        # --- Data sorting section ---
        sort_enabled = getattr(gw, 'sort_data_enabled', True)
        self._cb_sort_enabled.setChecked(sort_enabled)
        self._cbb_sort_by.setVisible(sort_enabled)
        if hasattr(self, '_lbl_sort_by'):
            self._lbl_sort_by.setVisible(sort_enabled)

        sort_by = getattr(gw, 'sort_data_by', 'Z')
        sort_map = {'Z': 0, 'X': 1, 'Y': 2}
        self._cbb_sort_by.setCurrentIndex(sort_map.get(sort_by, 0))

        # --- Trendline section ---
        is_trendline = (style == 'trendline')
        self._trendline_group.setVisible(is_trendline)
        if is_trendline:
            self._spin_order.setValue(getattr(gw, 'trendline_order', 1))
            anchor_on = getattr(gw, 'trendline_anchor_enabled', False)
            self._anchor_grp.setChecked(anchor_on)
            origin = getattr(gw, 'trendline_anchor_origin', True)
            self._rb_origin.setChecked(origin)
            self._rb_custom.setChecked(not origin)
            self._spin_ax.setValue(getattr(gw, 'trendline_anchor_x', 0.0))
            self._spin_ay.setValue(getattr(gw, 'trendline_anchor_y', 0.0))
            self._refresh_equation_table()

        # --- Histogram section ---
        is_hist = (style == 'histogram')
        self._histogram_group.setVisible(is_hist)
        if is_hist:
            self._spin_bins.setValue(getattr(gw, 'hist_bins', 20))
            self._cb_kde.setChecked(getattr(gw, 'hist_kde', False))
            step = getattr(gw, 'hist_step', False)
            self._rb_step.setChecked(step)
            self._rb_filled.setChecked(not step)

    def _refresh_equation_table(self):
        """Populate the equation table from trendline_equations stored on the graph widget."""
        equations = getattr(self.graph_widget, 'trendline_equations', [])
        self._eq_table.setRowCount(len(equations))
        for row, entry in enumerate(equations):
            self._eq_table.setItem(row, 0, QTableWidgetItem(str(entry.get('label', ''))))
            self._eq_table.setItem(row, 1, QTableWidgetItem(str(entry.get('equation', ''))))
            self._eq_table.setItem(row, 2, QTableWidgetItem(str(entry.get('r2', ''))))
        self._eq_table.resizeColumnsToContents()

    def _copy_equations(self):
        """Copy the equation table to the clipboard as tab-separated text."""
        equations = getattr(self.graph_widget, 'trendline_equations', [])
        if not equations:
            return
        lines = ["Group\tEquation\tR²"]
        for entry in equations:
            lines.append(f"{entry.get('label','')}\t{entry.get('equation','')}\t{entry.get('r2','')}")
        QApplication.clipboard().setText("\n".join(lines))

    def _apply(self):
        """Write widget values back to the graph widget and replot."""
        gw = self.graph_widget
        style = gw.plot_style

        # General
        gw.join_for_point_plot = self._cb_join.isChecked()
        gw.dodge_point_plot = self._cb_dodge.isChecked()
        gw.dodge_scatter_plot = self._cb_dodge_scatter.isChecked()
        gw.show_bar_plot_error_bar = self._cb_error_bar.isChecked()
        gw.wafer_stats = self._cb_wafer_stats.isChecked()

        # Figure style
        gw.figure_theme = _THEME_TEXT_MAP[self._combo_figure_theme.currentText()]
        facecolor_text = self._btn_figure_facecolor.text()
        gw.figure_facecolor = facecolor_text if facecolor_text != "(default)" else None
        gw.plot_subtitle = self._edit_subtitle.text().strip() or None
        gw.subtitle_fontsize = self._spin_subtitle_fontsize.value()
        gw.spines_visible = {
            'top': self._cb_spine_top.isChecked(),
            'right': self._cb_spine_right.isChecked(),
            'bottom': self._cb_spine_bottom.isChecked(),
            'left': self._cb_spine_left.isChecked(),
        }
        gw.figure_margins = [self._spin_x_margin.value(), self._spin_y_margin.value()]

        # Data sorting — capture old values first to detect changes
        old_sort_enabled = getattr(gw, 'sort_data_enabled', True)
        old_sort_by = getattr(gw, 'sort_data_by', 'Z')

        gw.sort_data_enabled = self._cb_sort_enabled.isChecked()
        sort_index = self._cbb_sort_by.currentIndex()
        gw.sort_data_by = ['Z', 'X', 'Y'][sort_index]

        # If sort settings changed, reset legend_properties so the legend is
        # rebuilt from scratch in the new sorted order (labels + colors in sync).
        sort_settings_changed = (
            gw.sort_data_enabled != old_sort_enabled or
            gw.sort_data_by != old_sort_by
        )
        if sort_settings_changed:
            gw.legend_properties = []

        # Trendline
        if style == 'trendline':
            gw.trendline_order = self._spin_order.value()
            gw.trendline_anchor_enabled = self._anchor_grp.isChecked()
            gw.trendline_anchor_origin = self._rb_origin.isChecked()
            gw.trendline_anchor_x = self._spin_ax.value()
            gw.trendline_anchor_y = self._spin_ay.value()

        # Histogram
        if style == 'histogram':
            gw.hist_bins = self._spin_bins.value()
            gw.hist_kde = self._cb_kde.isChecked()
            gw.hist_step = self._rb_step.isChecked()


        # Replot
        if gw.df is not None:
            gw.plot(gw.df)

        # Refresh equation table after replot (new equations computed)
        if style == 'trendline':
            self._refresh_equation_table()

        # Emit properties_changed so ViewModel persists the new settings
        props = {
            'join_for_point_plot': gw.join_for_point_plot,
            'dodge_point_plot': gw.dodge_point_plot,
            'dodge_scatter_plot': gw.dodge_scatter_plot,
            'show_bar_plot_error_bar': gw.show_bar_plot_error_bar,
            'wafer_stats': gw.wafer_stats,
            'sort_data_enabled': gw.sort_data_enabled,
            'sort_data_by': gw.sort_data_by,
            'figure_theme': gw.figure_theme,
            'figure_facecolor': gw.figure_facecolor,
            'plot_subtitle': gw.plot_subtitle,
            'subtitle_fontsize': gw.subtitle_fontsize,
            'spines_visible': gw.spines_visible,
            'figure_margins': gw.figure_margins,
        }
        if style == 'trendline':
            props.update({
                'trendline_order': gw.trendline_order,
                'trendline_anchor_enabled': gw.trendline_anchor_enabled,
                'trendline_anchor_origin': gw.trendline_anchor_origin,
                'trendline_anchor_x': gw.trendline_anchor_x,
                'trendline_anchor_y': gw.trendline_anchor_y,
            })
        if style == 'histogram':
            props.update({
                'hist_bins': gw.hist_bins,
                'hist_kde': gw.hist_kde,
                'hist_step': gw.hist_step,
            })
        gw.properties_changed.emit(gw.graph_id, props)

    def _on_sort_enabled_toggled(self, checked):
        """Show/hide the sort-by combobox based on the sort checkbox."""
        self._cbb_sort_by.setVisible(checked)
        if hasattr(self, '_lbl_sort_by'):
            self._lbl_sort_by.setVisible(checked)
