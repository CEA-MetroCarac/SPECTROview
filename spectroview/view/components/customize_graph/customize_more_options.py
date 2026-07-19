"""'More options' tab of the Customize Graph dialog: general plot options
(incl. figure theme), font sizes, data sorting, and adaptive controls
(trendline, histogram, colormap) shown based on the current plot_style.

Split out of customize_graph_dialog.py; no behavior changes.
"""
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QRadioButton,
    QApplication, QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QHeaderView,
)

from spectroview import ICON_DIR

# Text <-> MGraph.figure_theme value.
_THEME_TEXT_MAP = {"Light": "light", "Dark": "dark", "Soft Dark": "soft_dark"}
_THEME_VALUE_TEXT = {v: k for k, v in _THEME_TEXT_MAP.items()}

# Real mplstyle defaults (same across all three theme files -- they only
# differ in color, not typography) so the font-size spinboxes always show a
# concrete, meaningful number instead of a blank/sentinel value.
_DEFAULT_TITLE_FONTSIZE = 12
_DEFAULT_SUBTITLE_FONTSIZE = 10
_DEFAULT_AXIS_LABEL_FONTSIZE = 12
_DEFAULT_TICK_FONTSIZE = 9


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

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        inner = QWidget()
        self._inner_layout = QVBoxLayout(inner)
        self._inner_layout.setContentsMargins(4, 4, 4, 4)
        self._inner_layout.setSpacing(8)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        self._build_general_section()
        self._build_font_size_section()
        self._build_sorting_section()
        self._build_trendline_section()
        self._build_histogram_section()
        self._build_colormap_section()

        self._inner_layout.addStretch()

    # ---- General section (plot options + figure theme) ------------------

    def _build_general_section(self):
        grp = QGroupBox("Plot options:")
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        self._cb_join = QCheckBox("Join data points (point plot)")
        layout.addWidget(self._cb_join)

        self._cb_dodge = QCheckBox("Dodge overlapping points (point plot)")
        layout.addWidget(self._cb_dodge)

        self._cb_dodge_scatter = QCheckBox("Dodge overlapping points (scatter plot)")
        layout.addWidget(self._cb_dodge_scatter)

        self._cb_error_bar = QCheckBox("Show error bar (bar plot)")
        layout.addWidget(self._cb_error_bar)

        self._cb_wafer_stats = QCheckBox("Show statistics (wafer plot)")
        layout.addWidget(self._cb_wafer_stats)

        # Theme selects the underlying mplstyle (background/text/grid
        # colors as a set). X-label rotation and Grid live in the same row
        # -- migrated here from the workspace's bottom toolbar, which only
        # ever applied them as one-off new-plot defaults.
        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel("Theme:"))
        self._combo_figure_theme = QComboBox()
        self._combo_figure_theme.addItems(["Light", "Dark", "Soft Dark"])
        theme_row.addWidget(self._combo_figure_theme)

        theme_row.addSpacing(12)
        theme_row.addWidget(QLabel("X label rotation:"))
        self._spin_xlabel_rotation = QSpinBox()
        self._spin_xlabel_rotation.setRange(0, 90)
        self._spin_xlabel_rotation.setSingleStep(10)
        self._spin_xlabel_rotation.setMaximumWidth(60)
        theme_row.addWidget(self._spin_xlabel_rotation)

        theme_row.addSpacing(12)
        self._cb_grid = QCheckBox("Grid")
        theme_row.addWidget(self._cb_grid)

        theme_row.addStretch()
        layout.addLayout(theme_row)

        self._general_group = grp
        self._inner_layout.addWidget(grp)

    # ---- Font sizes section -----------------------------------------------

    def _build_font_size_section(self):
        """Every font-size control on a graph, in one row: Title, Subtitle
        (the subtitle text itself is edited in the workspace's side panel,
        not duplicated here), Axis label, Tick label."""
        grp = QGroupBox("Font sizes (pt):")
        layout = QHBoxLayout(grp)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        for label_text, attr, default in [
            ("Title:", "_spin_title_fontsize", _DEFAULT_TITLE_FONTSIZE),
            ("Subtitle:", "_spin_subtitle_fontsize", _DEFAULT_SUBTITLE_FONTSIZE),
            ("Axis label:", "_spin_axis_label_fontsize", _DEFAULT_AXIS_LABEL_FONTSIZE),
            ("Tick label:", "_spin_tick_fontsize", _DEFAULT_TICK_FONTSIZE),
        ]:
            layout.addWidget(QLabel(label_text))
            spin = QSpinBox()
            spin.setRange(4, 72)
            spin.setSingleStep(1)
            spin.setValue(default)
            spin.setMaximumWidth(60)
            setattr(self, attr, spin)
            layout.addWidget(spin)

        layout.addStretch()
        self._inner_layout.addWidget(grp)

    # ---- Data sorting section -------------------------------------------

    def _build_sorting_section(self):
        grp = QGroupBox("Data sorting:")
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        sort_row = QHBoxLayout()

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

        self._rb_origin.toggled.connect(lambda on: self._spin_ax.setEnabled(not on))
        self._rb_origin.toggled.connect(lambda on: self._spin_ay.setEnabled(not on))
        self._spin_ax.setEnabled(False)
        self._spin_ay.setEnabled(False)

        layout.addWidget(anchor_grp)

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

    # ---- Colormap normalization section (wafer/2Dmap) --------------------

    def _build_colormap_section(self):
        grp = QGroupBox("Colormap scale (wafer / 2Dmap)")
        layout = QHBoxLayout(grp)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Normalization:"))
        self._combo_colormap_norm = QComboBox()
        self._combo_colormap_norm.addItem("Linear", "linear")
        self._combo_colormap_norm.addItem("Log", "log")
        self._combo_colormap_norm.addItem("Centered", "centered")
        self._combo_colormap_norm.currentIndexChanged.connect(self._on_colormap_norm_changed)
        layout.addWidget(self._combo_colormap_norm)

        layout.addSpacing(10)
        self._lbl_colormap_center = QLabel("Center value:")
        layout.addWidget(self._lbl_colormap_center)
        self._spin_colormap_center = QDoubleSpinBox()
        self._spin_colormap_center.setRange(-1e6, 1e6)
        self._spin_colormap_center.setDecimals(2)
        self._spin_colormap_center.setSingleStep(1.0)
        self._spin_colormap_center.setMaximumWidth(90)
        layout.addWidget(self._spin_colormap_center)
        layout.addStretch()

        self._colormap_group = grp
        self._inner_layout.addWidget(grp)

    def _on_colormap_norm_changed(self):
        """Center value only means anything for the 'centered' norm."""
        is_centered = self._combo_colormap_norm.currentData() == "centered"
        self._lbl_colormap_center.setEnabled(is_centered)
        self._spin_colormap_center.setEnabled(is_centered)

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

        self._cb_join.setEnabled(style == 'point')
        self._cb_dodge.setEnabled(style == 'point')
        self._cb_dodge_scatter.setEnabled(style == 'scatter')
        self._cb_error_bar.setEnabled(style == 'bar')
        self._cb_wafer_stats.setEnabled(style == 'wafer')

        self._combo_figure_theme.setCurrentText(
            _THEME_VALUE_TEXT.get(getattr(gw, 'figure_theme', 'light'), "Light")
        )
        self._spin_xlabel_rotation.setValue(getattr(gw, 'x_rot', 0))
        self._cb_grid.setChecked(getattr(gw, 'grid', False))

        # --- Font sizes section ---
        self._spin_title_fontsize.setValue(getattr(gw, 'title_fontsize', None) or _DEFAULT_TITLE_FONTSIZE)
        self._spin_subtitle_fontsize.setValue(
            getattr(gw, 'subtitle_fontsize', None) or _DEFAULT_SUBTITLE_FONTSIZE
        )
        self._spin_axis_label_fontsize.setValue(
            getattr(gw, 'axis_label_fontsize', None) or _DEFAULT_AXIS_LABEL_FONTSIZE
        )
        self._spin_tick_fontsize.setValue(getattr(gw, 'tick_label_fontsize', None) or _DEFAULT_TICK_FONTSIZE)

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

        # --- Colormap normalization section (wafer/2Dmap only) ---
        is_map = style in ('wafer', '2Dmap')
        self._colormap_group.setVisible(is_map)
        if is_map:
            norm_kind = getattr(gw, 'colormap_norm', 'linear')
            idx = self._combo_colormap_norm.findData(norm_kind)
            self._combo_colormap_norm.setCurrentIndex(idx if idx >= 0 else 0)
            self._spin_colormap_center.setValue(getattr(gw, 'colormap_center', 0.0))
            self._on_colormap_norm_changed()

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

    def _apply(self, replot: bool = True):
        """Write widget values back to the graph widget and replot.

        `replot=False` (the dialog's debounced live preview) skips both the
        replot and the trendline-equation-table refresh that depends on
        its freshly computed results, leaving the combined
        restyle()-or-replot decision to the dialog after every tab has
        applied its fields."""
        gw = self.graph_widget
        style = gw.plot_style

        gw.join_for_point_plot = self._cb_join.isChecked()
        gw.dodge_point_plot = self._cb_dodge.isChecked()
        gw.dodge_scatter_plot = self._cb_dodge_scatter.isChecked()
        gw.show_bar_plot_error_bar = self._cb_error_bar.isChecked()
        gw.wafer_stats = self._cb_wafer_stats.isChecked()
        gw.figure_theme = _THEME_TEXT_MAP[self._combo_figure_theme.currentText()]
        gw.x_rot = self._spin_xlabel_rotation.value()
        gw.grid = self._cb_grid.isChecked()

        gw.title_fontsize = self._spin_title_fontsize.value()
        gw.subtitle_fontsize = self._spin_subtitle_fontsize.value()
        gw.axis_label_fontsize = self._spin_axis_label_fontsize.value()
        gw.tick_label_fontsize = self._spin_tick_fontsize.value()

        # Data sorting — capture old values first to detect changes
        old_sort_enabled = getattr(gw, 'sort_data_enabled', True)
        old_sort_by = getattr(gw, 'sort_data_by', 'Z')
        gw.sort_data_enabled = self._cb_sort_enabled.isChecked()
        sort_index = self._cbb_sort_by.currentIndex()
        gw.sort_data_by = ['Z', 'X', 'Y'][sort_index]

        # Sort settings changed -> reset legend_properties so it rebuilds
        # from scratch in the new order (labels + colors stay in sync).
        if gw.sort_data_enabled != old_sort_enabled or gw.sort_data_by != old_sort_by:
            gw.legend_properties = []

        if style == 'trendline':
            gw.trendline_order = self._spin_order.value()
            gw.trendline_anchor_enabled = self._anchor_grp.isChecked()
            gw.trendline_anchor_origin = self._rb_origin.isChecked()
            gw.trendline_anchor_x = self._spin_ax.value()
            gw.trendline_anchor_y = self._spin_ay.value()

        if style == 'histogram':
            gw.hist_bins = self._spin_bins.value()
            gw.hist_kde = self._cb_kde.isChecked()
            gw.hist_step = self._rb_step.isChecked()

        if style in ('wafer', '2Dmap'):
            gw.colormap_norm = self._combo_colormap_norm.currentData()
            gw.colormap_center = self._spin_colormap_center.value()

        if replot:
            if gw.df is not None:
                gw.plot(gw.df)
            if style == 'trendline':
                self._refresh_equation_table()

        props = {
            'join_for_point_plot': gw.join_for_point_plot,
            'dodge_point_plot': gw.dodge_point_plot,
            'dodge_scatter_plot': gw.dodge_scatter_plot,
            'show_bar_plot_error_bar': gw.show_bar_plot_error_bar,
            'wafer_stats': gw.wafer_stats,
            'figure_theme': gw.figure_theme,
            'x_rot': gw.x_rot,
            'grid': gw.grid,
            'title_fontsize': gw.title_fontsize,
            'subtitle_fontsize': gw.subtitle_fontsize,
            'axis_label_fontsize': gw.axis_label_fontsize,
            'tick_label_fontsize': gw.tick_label_fontsize,
            'sort_data_enabled': gw.sort_data_enabled,
            'sort_data_by': gw.sort_data_by,
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
        if style in ('wafer', '2Dmap'):
            props.update({
                'colormap_norm': gw.colormap_norm,
                'colormap_center': gw.colormap_center,
            })
        gw.properties_changed.emit(gw.graph_id, props)

    def _on_sort_enabled_toggled(self, checked):
        """Show/hide the sort-by combobox based on the sort checkbox."""
        self._cbb_sort_by.setVisible(checked)
        if hasattr(self, '_lbl_sort_by'):
            self._lbl_sort_by.setVisible(checked)
