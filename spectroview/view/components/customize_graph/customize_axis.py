"""Axis tab of the Customize Graph dialog: scale/data-type, limits, tick
direction/format, minor ticks/spines, broken axis, inset (zoom) axes, and
secondary (Y2/Y3/X2) axes -- everything "axis-shaped" lives in one tab.

Split out of customize_graph_dialog.py; no behavior changes.
"""
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QCheckBox, QMessageBox, QLineEdit, QColorDialog, QScrollArea,
)

from spectroview import MARKERS
from spectroview.view.components.customize_graph.spin_widgets import PlaceholderDoubleSpinBox

try:
    from superqt import QLabeledDoubleRangeSlider
except ImportError:  # pragma: no cover - superqt is a hard dependency in practice
    QLabeledDoubleRangeSlider = None

# Text <-> MGraph.tick_direction ("Default" = None, matplotlib's own default).
_TICK_DIRECTION_MAP = {"Default": None, "In": "in", "Out": "out", "In & Out": "inout"}
_TICK_DIRECTION_TEXT = {v: k for k, v in _TICK_DIRECTION_MAP.items()}

# Tick-label-format presets (replaces a freeform printf-style text field).
# None = "Auto" -- matplotlib's own default ScalarFormatter.
_TICK_FORMAT_PRESETS = [
    ("Auto (default)", None),
    ("Integer  (e.g. 1234)", "%.0f"),
    ("1 decimal  (e.g. 12.3)", "%.1f"),
    ("2 decimals  (e.g. 12.34)", "%.2f"),
    ("Scientific  (e.g. 1.2e+03)", "%.1e"),
]

# Width that fits a PlaceholderDoubleSpinBox's "default" placeholder without
# clipping; shared by every optional-limit spinbox in this tab.
_PLACEHOLDER_SPIN_WIDTH = 105


class CustomizeAxis(QWidget):
    """Widget for customizing axis settings (scale, limits, style, breaks,
    inset axes, secondary axes)."""

    def __init__(self, graph_widget, parent=None):
        super().__init__(parent)
        self.graph_widget = graph_widget
        self._setup_ui()
        self.load_axis_settings()

    def switch_graph(self, graph_widget):
        """Switch to a different graph widget and reload axis settings."""
        self.graph_widget = graph_widget
        self.load_axis_settings()

    # Sentinel for "no limit set" -- the spinbox's own range minimum, shown
    # as a grayed placeholder instead of the raw number. Shared by primary,
    # inset, and secondary-axis limit spinboxes alike.
    _UNSET_LIMIT = -999999

    def _make_limit_spinbox(self, start_value=0.0) -> PlaceholderDoubleSpinBox:
        """One limit spinbox that shows a grayed placeholder when unset."""
        spin = PlaceholderDoubleSpinBox(self._UNSET_LIMIT, start_value, "default")
        spin.setRange(self._UNSET_LIMIT, 999999)
        return spin

    def _add_limit_row_with_slider(self, parent_layout, axis_label, prefix):
        """One 'label: [min] [====slider====] [max]' row for a primary axis
        (X/Y/Z), bidirectionally synced. Falls back to spinbox-only if
        superqt is missing."""
        row = QHBoxLayout()
        row.addWidget(QLabel(f"{axis_label} axis limits:"))

        spin_min = self._make_limit_spinbox()
        spin_min.setMaximumWidth(_PLACEHOLDER_SPIN_WIDTH)
        setattr(self, f'spin_{prefix}min', spin_min)
        row.addWidget(spin_min)

        slider = None
        if QLabeledDoubleRangeSlider is not None:
            slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
            slider.setEdgeLabelMode(QLabeledDoubleRangeSlider.EdgeLabelMode.NoLabel)
            slider.setHandleLabelPosition(QLabeledDoubleRangeSlider.LabelPosition.NoLabel)
            slider.setRange(0, 100)
            slider.setValue((0, 100))
            setattr(self, f'{prefix}_range_slider', slider)
            row.addWidget(slider, stretch=1)

        spin_max = self._make_limit_spinbox()
        spin_max.setMaximumWidth(_PLACEHOLDER_SPIN_WIDTH)
        setattr(self, f'spin_{prefix}max', spin_max)
        row.addWidget(spin_max)

        if slider is not None:
            slider.valueChanged.connect(lambda values, p=prefix: self._on_range_slider_changed(p, values))
            # *_a absorbs Qt's variable arg count for this signal across
            # PySide6 builds -- don't depend on how many args it passes.
            spin_min.valueChanged.connect(lambda *_a, p=prefix: self._update_range_slider_from_spins(p))
            spin_max.valueChanged.connect(lambda *_a, p=prefix: self._update_range_slider_from_spins(p))

        parent_layout.addLayout(row)

    def _on_range_slider_changed(self, prefix, values):
        """Slider dragged -- mirror (min, max) into the spinboxes (applies
        on Apply, like every other field here, not live)."""
        spin_min = getattr(self, f'spin_{prefix}min')
        spin_max = getattr(self, f'spin_{prefix}max')
        spin_min.blockSignals(True)
        spin_max.blockSignals(True)
        spin_min.setValue(values[0])
        spin_max.setValue(values[1])
        spin_min._update_placeholder_style()
        spin_max._update_placeholder_style()
        spin_min.blockSignals(False)
        spin_max.blockSignals(False)

    def _update_range_slider_from_spins(self, prefix):
        """Spinbox edited -- mirror it into the slider. An unset spinbox
        shows the slider spanning its full range (matches "no limit set")."""
        slider = getattr(self, f'{prefix}_range_slider', None)
        if slider is None:
            return
        spin_min = getattr(self, f'spin_{prefix}min')
        spin_max = getattr(self, f'spin_{prefix}max')
        u = self._UNSET_LIMIT
        lo, hi = slider.minimum(), slider.maximum()
        vmin = lo if spin_min.value() == u else spin_min.value()
        vmax = hi if spin_max.value() == u else spin_max.value()
        vmin = max(lo, min(vmin, hi))
        vmax = max(lo, min(vmax, hi))
        if vmin <= vmax:
            slider.blockSignals(True)
            slider.setValue((vmin, vmax))
            slider.blockSignals(False)

    def _update_range_slider_bounds(self):
        """Derive each slider's drag range from the graph's current data
        (padded 10%, mirroring matplotlib's own auto-margin) and refresh
        each spinbox's arrow-click start value to match."""
        gw = self.graph_widget
        df = getattr(gw, 'df', None)
        columns = {'x': gw.x, 'y': (gw.y[0] if gw.y else None), 'z': gw.z}
        for prefix, col in columns.items():
            slider = getattr(self, f'{prefix}_range_slider', None)
            if slider is None:
                continue
            lo, hi = 0.0, 100.0
            if df is not None and col is not None and col in getattr(df, 'columns', []):
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                if not series.empty:
                    data_min, data_max = float(series.min()), float(series.max())
                    pad = (data_max - data_min) * 0.1 or 1.0
                    lo, hi = data_min - pad, data_max + pad
            # Blocked: setRange() can clamp a stale slider value and fire
            # valueChanged, stomping an unset spinbox via _on_range_slider_changed.
            slider.blockSignals(True)
            slider.setRange(lo, hi)
            slider.blockSignals(False)
            getattr(self, f'spin_{prefix}min').set_start_value(lo)
            getattr(self, f'spin_{prefix}max').set_start_value(hi)
            self._update_range_slider_from_spins(prefix)

    def _setup_ui(self):
        """Setup the UI components for the axis customization widget.

        Scrollable: this tab now also hosts inset and secondary-axes
        content (merged in from their old standalone tabs), tall enough
        that a fixed-height layout could clip on smaller screens.
        """
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # ===== Axis Properties Section =====
        props_group = QGroupBox("Axis properties:")
        props_layout = QVBoxLayout(props_group)
        props_layout.setContentsMargins(4, 4, 4, 4)
        props_layout.setSpacing(8)

        x_prop_layout = QHBoxLayout()
        x_prop_layout.addWidget(QLabel("X axis:  "))
        x_prop_layout.addWidget(QLabel("Scale:"))
        self.combo_x_scale = QComboBox()
        self.combo_x_scale.addItems(["Linear", "Logarithmic", "Symlog"])
        x_prop_layout.addWidget(self.combo_x_scale)
        x_prop_layout.addSpacing(10)
        x_prop_layout.addWidget(QLabel("Data type:"))
        self.combo_x_type = QComboBox()
        self.combo_x_type.addItems(["Auto", "Category", "Numerical"])
        x_prop_layout.addWidget(self.combo_x_type)
        x_prop_layout.addSpacing(10)
        self.cb_invert_x = QCheckBox("Inverted")
        x_prop_layout.addWidget(self.cb_invert_x)
        x_prop_layout.addStretch()

        y_prop_layout = QHBoxLayout()
        y_prop_layout.addWidget(QLabel("Y axis:  "))
        y_prop_layout.addWidget(QLabel("Scale:"))
        self.combo_y_scale = QComboBox()
        self.combo_y_scale.addItems(["Linear", "Logarithmic", "Symlog"])
        y_prop_layout.addWidget(self.combo_y_scale)
        y_prop_layout.addSpacing(10)
        y_prop_layout.addWidget(QLabel("Data type:"))
        self.combo_y_type = QComboBox()
        self.combo_y_type.addItems(["Auto", "Category", "Numerical"])
        y_prop_layout.addWidget(self.combo_y_type)
        y_prop_layout.addSpacing(10)
        self.cb_invert_y = QCheckBox("Inverted")
        y_prop_layout.addWidget(self.cb_invert_y)
        y_prop_layout.addStretch()

        props_layout.addLayout(x_prop_layout)
        props_layout.addLayout(y_prop_layout)

        # ===== Axis Appearance Section (minor ticks/spines, tick
        # direction/format -- "how the axis itself looks") =====
        style_group = QGroupBox("Axis Appearance:")
        style_layout = QVBoxLayout(style_group)
        style_layout.setContentsMargins(4, 4, 4, 4)
        style_layout.setSpacing(8)

        grid_layout = QGridLayout()
        lbl_minor = QLabel("Show minor ticks:")
        self.cb_minor_top = QCheckBox("Top")
        self.cb_minor_right = QCheckBox("Right")
        self.cb_minor_bottom = QCheckBox("Bottom")
        self.cb_minor_left = QCheckBox("Left")
        grid_layout.addWidget(lbl_minor, 0, 0)
        grid_layout.addWidget(self.cb_minor_top, 0, 1)
        grid_layout.addWidget(self.cb_minor_right, 0, 2)
        grid_layout.addWidget(self.cb_minor_bottom, 0, 3)
        grid_layout.addWidget(self.cb_minor_left, 0, 4)

        lbl_spine = QLabel("Show spines:")
        self.cb_spine_top = QCheckBox("Top")
        self.cb_spine_right = QCheckBox("Right")
        self.cb_spine_bottom = QCheckBox("Bottom")
        self.cb_spine_left = QCheckBox("Left")
        grid_layout.addWidget(lbl_spine, 1, 0)
        grid_layout.addWidget(self.cb_spine_top, 1, 1)
        grid_layout.addWidget(self.cb_spine_right, 1, 2)
        grid_layout.addWidget(self.cb_spine_bottom, 1, 3)
        grid_layout.addWidget(self.cb_spine_left, 1, 4)
        grid_layout.setColumnStretch(5, 1)
        style_layout.addLayout(grid_layout)

        tick_row = QHBoxLayout()
        tick_row.addWidget(QLabel("Tick direction:"))
        self.combo_tick_direction = QComboBox()
        self.combo_tick_direction.addItems(list(_TICK_DIRECTION_MAP.keys()))
        tick_row.addWidget(self.combo_tick_direction)
        tick_row.addSpacing(10)
        tick_row.addWidget(QLabel("Tick label format:"))
        self.combo_tick_format = QComboBox()
        for label, value in _TICK_FORMAT_PRESETS:
            self.combo_tick_format.addItem(label, value)
        tick_row.addWidget(self.combo_tick_format)
        tick_row.addStretch()
        style_layout.addLayout(tick_row)

        # ===== Axis Limits Section =====
        limits_group = QGroupBox("Set Axis Limits:")
        limits_layout = QVBoxLayout(limits_group)
        limits_layout.setContentsMargins(4, 4, 4, 4)
        limits_layout.setSpacing(8)

        # X/Y hidden for wafer/2Dmap (see load_axis_settings()): those are
        # spatial axes governed by wafer_size, not a user-set min/max.
        self.xy_limits_widget = QWidget()
        xy_limits_layout = QVBoxLayout(self.xy_limits_widget)
        xy_limits_layout.setContentsMargins(0, 0, 0, 0)
        xy_limits_layout.setSpacing(8)
        self._add_limit_row_with_slider(xy_limits_layout, "X", "x")
        self._add_limit_row_with_slider(xy_limits_layout, "Y", "y")
        limits_layout.addWidget(self.xy_limits_widget)
        self._add_limit_row_with_slider(limits_layout, "Z", "z")

        limits_btn_layout = QHBoxLayout()
        self.btn_set_limits = QPushButton("Get current limits from plot")
        self.btn_set_limits.clicked.connect(self._on_get_current_limits)
        self.btn_clear_limits = QPushButton("Reinitialize limits")
        self.btn_clear_limits.clicked.connect(self._on_clear_limits)
        limits_btn_layout.addWidget(self.btn_set_limits)
        limits_btn_layout.addWidget(self.btn_clear_limits)
        limits_layout.addLayout(limits_btn_layout)

        # ===== Axis Break Section =====
        break_group = QGroupBox("Broken axis:")
        break_layout = QVBoxLayout()
        break_layout.setContentsMargins(4, 4, 4, 4)
        break_layout.setSpacing(8)

        x_break_layout = QHBoxLayout()
        x_break_layout.setContentsMargins(0, 0, 0, 0)
        x_break_layout.setSpacing(8)
        # X and Y breaks are mutually exclusive -- see _on_x/y_break_toggled.
        self.x_break_enabled = QCheckBox("X-axis break")
        self.x_break_enabled.setToolTip("X and Y breaks cannot both be active on the same graph.")
        self.x_break_start = QDoubleSpinBox()
        self.x_break_start.setRange(-999999, 999999)
        self.x_break_start.setDecimals(2)
        self.x_break_end = QDoubleSpinBox()
        self.x_break_end.setRange(-999999, 999999)
        self.x_break_end.setDecimals(2)
        x_break_layout.addWidget(self.x_break_enabled)
        x_break_layout.addWidget(QLabel("from:"))
        x_break_layout.addWidget(self.x_break_start)
        x_break_layout.addWidget(QLabel("to:"))
        x_break_layout.addWidget(self.x_break_end)
        x_break_layout.addStretch()

        y_break_layout = QHBoxLayout()
        y_break_layout.setContentsMargins(0, 0, 0, 0)
        y_break_layout.setSpacing(8)
        self.y_break_enabled = QCheckBox("Y-axis break")
        self.y_break_enabled.setToolTip("X and Y breaks cannot both be active on the same graph.")
        self.y_break_start = QDoubleSpinBox()
        self.y_break_start.setRange(-999999, 999999)
        self.y_break_start.setDecimals(2)
        self.y_break_end = QDoubleSpinBox()
        self.y_break_end.setRange(-999999, 999999)
        self.y_break_end.setDecimals(2)
        y_break_layout.addWidget(self.y_break_enabled)
        y_break_layout.addWidget(QLabel("from:"))
        y_break_layout.addWidget(self.y_break_start)
        y_break_layout.addWidget(QLabel("to:"))
        y_break_layout.addWidget(self.y_break_end)
        y_break_layout.addStretch()

        break_layout.addLayout(x_break_layout)
        break_layout.addLayout(y_break_layout)
        break_group.setLayout(break_layout)
        self.x_break_enabled.toggled.connect(self._on_x_break_toggled)
        self.y_break_enabled.toggled.connect(self._on_y_break_toggled)

        layout.addWidget(limits_group)
        layout.addWidget(props_group)
        layout.addWidget(style_group)
        layout.addWidget(break_group)
        self._setup_inset_section(layout)
        self._setup_secondary_axes_section(layout)
        layout.addStretch()

    # ------------------------------------------------------------------ #
    #  Inset (zoom) axes
    # ------------------------------------------------------------------ #

    def _setup_inset_section(self, layout):
        """One optional inset (zoom) Axes per graph, showing the same
        series as the main plot at its own x/y limits."""
        inset_group = QGroupBox("Inset (zoom) axes:")
        inset_group.setCheckable(True)  # unchecked = disabled + greys out contents for free
        inset_layout = QVBoxLayout(inset_group)
        inset_layout.setContentsMargins(4, 4, 4, 4)
        inset_layout.setSpacing(8)
        self.inset_group = inset_group

        bounds_row = QHBoxLayout()
        bounds_row.addWidget(QLabel("Position (x0, y0):"))
        self.spin_inset_x0 = QDoubleSpinBox()
        self.spin_inset_x0.setRange(0.0, 1.0)
        self.spin_inset_x0.setDecimals(2)
        self.spin_inset_x0.setSingleStep(0.05)
        bounds_row.addWidget(self.spin_inset_x0)
        self.spin_inset_y0 = QDoubleSpinBox()
        self.spin_inset_y0.setRange(0.0, 1.0)
        self.spin_inset_y0.setDecimals(2)
        self.spin_inset_y0.setSingleStep(0.05)
        bounds_row.addWidget(self.spin_inset_y0)
        bounds_row.addSpacing(10)
        bounds_row.addWidget(QLabel("Size (w, h):"))
        self.spin_inset_w = QDoubleSpinBox()
        self.spin_inset_w.setRange(0.05, 1.0)
        self.spin_inset_w.setDecimals(2)
        self.spin_inset_w.setSingleStep(0.05)
        bounds_row.addWidget(self.spin_inset_w)
        self.spin_inset_h = QDoubleSpinBox()
        self.spin_inset_h.setRange(0.05, 1.0)
        self.spin_inset_h.setDecimals(2)
        self.spin_inset_h.setSingleStep(0.05)
        bounds_row.addWidget(self.spin_inset_h)
        bounds_row.addStretch()
        inset_layout.addLayout(bounds_row)

        inset_xlim_row = QHBoxLayout()
        inset_xlim_row.addWidget(QLabel("X limits:"))
        self.spin_inset_xmin = self._make_limit_spinbox()
        self.spin_inset_xmin.setMaximumWidth(_PLACEHOLDER_SPIN_WIDTH)
        inset_xlim_row.addWidget(self.spin_inset_xmin)
        self.spin_inset_xmax = self._make_limit_spinbox()
        self.spin_inset_xmax.setMaximumWidth(_PLACEHOLDER_SPIN_WIDTH)
        inset_xlim_row.addWidget(self.spin_inset_xmax)
        inset_xlim_row.addStretch()
        inset_layout.addLayout(inset_xlim_row)

        inset_ylim_row = QHBoxLayout()
        inset_ylim_row.addWidget(QLabel("Y limits:"))
        self.spin_inset_ymin = self._make_limit_spinbox()
        self.spin_inset_ymin.setMaximumWidth(_PLACEHOLDER_SPIN_WIDTH)
        inset_ylim_row.addWidget(self.spin_inset_ymin)
        self.spin_inset_ymax = self._make_limit_spinbox()
        self.spin_inset_ymax.setMaximumWidth(_PLACEHOLDER_SPIN_WIDTH)
        inset_ylim_row.addWidget(self.spin_inset_ymax)
        inset_ylim_row.addStretch()
        inset_layout.addLayout(inset_ylim_row)

        self.cb_inset_zoom_indicator = QCheckBox("Show zoom indicator (connector lines on main plot)")
        self.cb_inset_zoom_indicator.setChecked(True)
        inset_layout.addWidget(self.cb_inset_zoom_indicator)

        layout.addWidget(inset_group)

    def _load_inset_settings(self, gw, x_min, x_max, y_min, y_max):
        u = self._UNSET_LIMIT
        self.inset_group.setChecked(getattr(gw, 'inset_enabled', False))
        bounds = getattr(gw, 'inset_bounds', None) or [0.55, 0.55, 0.35, 0.35]
        self.spin_inset_x0.setValue(bounds[0])
        self.spin_inset_y0.setValue(bounds[1])
        self.spin_inset_w.setValue(bounds[2])
        self.spin_inset_h.setValue(bounds[3])
        self.spin_inset_xmin.setValue(gw.inset_xmin if gw.inset_xmin is not None else u)
        self.spin_inset_xmax.setValue(gw.inset_xmax if gw.inset_xmax is not None else u)
        self.spin_inset_ymin.setValue(gw.inset_ymin if gw.inset_ymin is not None else u)
        self.spin_inset_ymax.setValue(gw.inset_ymax if gw.inset_ymax is not None else u)
        self.cb_inset_zoom_indicator.setChecked(getattr(gw, 'inset_show_zoom_indicator', True))

        # Placeholder: the inset's own rendered limits when it exists, else
        # the main axes' limits as the closest available estimate.
        inset_ax = getattr(gw, 'inset_ax', None)
        if inset_ax is not None:
            try:
                inset_x_min, inset_x_max = inset_ax.get_xlim()
                inset_y_min, inset_y_max = inset_ax.get_ylim()
            except Exception:
                inset_x_min, inset_x_max, inset_y_min, inset_y_max = x_min, x_max, y_min, y_max
        else:
            inset_x_min, inset_x_max, inset_y_min, inset_y_max = x_min, x_max, y_min, y_max
        self.spin_inset_xmin.set_placeholder_value(inset_x_min)
        self.spin_inset_xmax.set_placeholder_value(inset_x_max)
        self.spin_inset_ymin.set_placeholder_value(inset_y_min)
        self.spin_inset_ymax.set_placeholder_value(inset_y_max)

    def _apply_inset_settings(self, gw, props):
        # Inset axes always force a full replot (not in
        # graph_style.RESTYLE_SAFE_FIELDS) since enabling/disabling one adds
        # or removes an Axes.
        u = self._UNSET_LIMIT
        gw.inset_enabled = self.inset_group.isChecked()
        gw.inset_bounds = [
            self.spin_inset_x0.value(), self.spin_inset_y0.value(),
            self.spin_inset_w.value(), self.spin_inset_h.value(),
        ]
        gw.inset_xmin = self.spin_inset_xmin.value() if self.spin_inset_xmin.value() != u else None
        gw.inset_xmax = self.spin_inset_xmax.value() if self.spin_inset_xmax.value() != u else None
        gw.inset_ymin = self.spin_inset_ymin.value() if self.spin_inset_ymin.value() != u else None
        gw.inset_ymax = self.spin_inset_ymax.value() if self.spin_inset_ymax.value() != u else None
        gw.inset_show_zoom_indicator = self.cb_inset_zoom_indicator.isChecked()
        props.update({
            'inset_enabled': gw.inset_enabled,
            'inset_bounds': gw.inset_bounds,
            'inset_xmin': gw.inset_xmin, 'inset_xmax': gw.inset_xmax,
            'inset_ymin': gw.inset_ymin, 'inset_ymax': gw.inset_ymax,
            'inset_show_zoom_indicator': gw.inset_show_zoom_indicator,
        })

    # ------------------------------------------------------------------ #
    #  Secondary axes (Y2/Y3/X2)
    # ------------------------------------------------------------------ #

    def _setup_secondary_axes_section(self, layout):
        """Per-axis label/limits/scale/style for the Y2/Y3/X2 twin axes.
        A row is disabled until that axis has a column assigned (column
        assignment itself is in the side panel, not duplicated here)."""
        secondary_group = QGroupBox("Secondary axes:")
        secondary_layout = QGridLayout(secondary_group)
        secondary_layout.setContentsMargins(4, 4, 4, 4)
        secondary_layout.setHorizontalSpacing(8)
        secondary_layout.setVerticalSpacing(6)

        headers = ["Axis", "Label", "Min", "Max", "Log", "Color", "Marker"]
        for col, text in enumerate(headers):
            secondary_layout.addWidget(QLabel(text), 0, col)

        self._secondary_axis_rows = {}
        for row, (axis_key, axis_display) in enumerate(
            [('y2', 'Y2'), ('y3', 'Y3'), ('x2', 'X2')], start=1
        ):
            secondary_layout.addWidget(QLabel(axis_display), row, 0)

            edit_label = QLineEdit()
            edit_label.setMaximumWidth(90)
            secondary_layout.addWidget(edit_label, row, 1)

            spin_min = self._make_limit_spinbox()
            spin_min.setMaximumWidth(_PLACEHOLDER_SPIN_WIDTH)
            secondary_layout.addWidget(spin_min, row, 2)

            spin_max = self._make_limit_spinbox()
            spin_max.setMaximumWidth(_PLACEHOLDER_SPIN_WIDTH)
            secondary_layout.addWidget(spin_max, row, 3)

            cb_log = QCheckBox()
            secondary_layout.addWidget(cb_log, row, 4)

            btn_color = QPushButton()
            btn_color.setFixedWidth(70)
            btn_color.clicked.connect(lambda *_a, k=axis_key: self._pick_secondary_axis_color(k))
            secondary_layout.addWidget(btn_color, row, 5)

            combo_marker = QComboBox()
            combo_marker.addItems(MARKERS)
            combo_marker.setMaximumWidth(60)
            secondary_layout.addWidget(combo_marker, row, 6)

            self._secondary_axis_rows[axis_key] = {
                'label': edit_label, 'min': spin_min, 'max': spin_max,
                'log': cb_log, 'color': btn_color, 'marker': combo_marker,
            }

        layout.addWidget(secondary_group)

    def _load_secondary_axis_settings(self, gw):
        u = self._UNSET_LIMIT
        for axis_key, row in self._secondary_axis_rows.items():
            active = bool(getattr(gw, axis_key, None))
            for widget in row.values():
                widget.setEnabled(active)

            row['label'].setText(getattr(gw, f'{axis_key}label', None) or "")
            row['min'].setValue(getattr(gw, f'{axis_key}min', None) if getattr(gw, f'{axis_key}min', None) is not None else u)
            row['max'].setValue(getattr(gw, f'{axis_key}max', None) if getattr(gw, f'{axis_key}max', None) is not None else u)
            row['log'].setChecked(getattr(gw, f'{axis_key}logscale', False))

            color = getattr(gw, f'{axis_key}color', 'red')
            row['color'].setText(color)
            row['color'].setStyleSheet(f"background-color: {QColor(color).name()};")
            row['marker'].setCurrentText(getattr(gw, f'{axis_key}marker', 'o'))

    def _pick_secondary_axis_color(self, axis_key):
        row = self._secondary_axis_rows[axis_key]
        current = QColor(row['color'].text())
        color = QColorDialog.getColor(current, self, "Select Axis Color")
        if color.isValid():
            row['color'].setText(color.name())
            row['color'].setStyleSheet(f"background-color: {color.name()};")

    def _apply_secondary_axis_settings(self, gw, props):
        for axis_key, row in self._secondary_axis_rows.items():
            label = row['label'].text().strip() or None
            u = self._UNSET_LIMIT
            vmin = row['min'].value() if row['min'].value() != u else None
            vmax = row['max'].value() if row['max'].value() != u else None
            log = row['log'].isChecked()
            color = row['color'].text()
            marker = row['marker'].currentText()

            setattr(gw, f'{axis_key}label', label)
            setattr(gw, f'{axis_key}min', vmin)
            setattr(gw, f'{axis_key}max', vmax)
            setattr(gw, f'{axis_key}logscale', log)
            setattr(gw, f'{axis_key}color', color)
            setattr(gw, f'{axis_key}marker', marker)

            props.update({
                f'{axis_key}label': label, f'{axis_key}min': vmin, f'{axis_key}max': vmax,
                f'{axis_key}logscale': log, f'{axis_key}color': color, f'{axis_key}marker': marker,
            })

    # ------------------------------------------------------------------ #
    #  Load / Apply
    # ------------------------------------------------------------------ #

    def load_axis_settings(self):
        """Load current axis settings (limits and breaks) from graph widget."""
        gw = self.graph_widget

        # X/Y limits don't apply to wafer/2Dmap (spatial axes governed by
        # wafer_size); Z stays visible as their color-scale control.
        self.xy_limits_widget.setVisible(gw.plot_style not in ('wafer', '2Dmap'))

        u = self._UNSET_LIMIT
        self.spin_xmin.setValue(gw.xmin if gw.xmin is not None else u)
        self.spin_xmax.setValue(gw.xmax if gw.xmax is not None else u)
        self.spin_ymin.setValue(gw.ymin if gw.ymin is not None else u)
        self.spin_ymax.setValue(gw.ymax if gw.ymax is not None else u)
        self.spin_zmin.setValue(gw.zmin if gw.zmin is not None else u)
        self.spin_zmax.setValue(gw.zmax if gw.zmax is not None else u)
        self._update_range_slider_bounds()

        # Placeholder shows the real current/effective limit (grayed), not
        # a generic word -- X/Y use the actually-rendered Axes limits (Z has
        # no live get_zlim() concept, so keeps the data-range estimate).
        x_min, x_max, y_min, y_max = self._get_current_ax_limits()
        self.spin_xmin.set_placeholder_value(x_min)
        self.spin_xmax.set_placeholder_value(x_max)
        self.spin_ymin.set_placeholder_value(y_min)
        self.spin_ymax.set_placeholder_value(y_max)
        self.spin_zmin.set_placeholder_value(self.spin_zmin._start_value)
        self.spin_zmax.set_placeholder_value(self.spin_zmax._start_value)

        # Log/symlog share the xlogscale/ylogscale on/off gate;
        # xscale_mode/yscale_mode pick which of the two.
        if getattr(gw, 'xlogscale', False):
            self.combo_x_scale.setCurrentText(
                "Symlog" if getattr(gw, 'xscale_mode', 'log') == 'symlog' else "Logarithmic"
            )
        else:
            self.combo_x_scale.setCurrentText("Linear")

        if getattr(gw, 'ylogscale', False):
            self.combo_y_scale.setCurrentText(
                "Symlog" if getattr(gw, 'yscale_mode', 'log') == 'symlog' else "Logarithmic"
            )
        else:
            self.combo_y_scale.setCurrentText("Linear")

        x_num = getattr(gw, 'x_as_numeric', None)
        if x_num is None:
            self.combo_x_type.setCurrentText("Auto")
        elif x_num:
            self.combo_x_type.setCurrentText("Numerical")
        else:
            self.combo_x_type.setCurrentText("Category")

        y_num = getattr(gw, 'y_as_numeric', None)
        if y_num is None:
            self.combo_y_type.setCurrentText("Auto")
        elif y_num:
            self.combo_y_type.setCurrentText("Numerical")
        else:
            self.combo_y_type.setCurrentText("Category")

        self.cb_invert_x.setChecked(getattr(gw, 'x_inverted', False))
        self.cb_invert_y.setChecked(getattr(gw, 'y_inverted', False))

        self.combo_tick_direction.setCurrentText(
            _TICK_DIRECTION_TEXT.get(getattr(gw, 'tick_direction', None), "Default")
        )
        self._load_tick_format(getattr(gw, 'tick_label_format', None))

        if not hasattr(self.graph_widget, 'axis_breaks'):
            self.graph_widget.axis_breaks = {'x': None, 'y': None}
        breaks = self.graph_widget.axis_breaks

        # Reused for the break-range suggestion below and the limit
        # spinboxes' placeholder text above.
        x_min, x_max, y_min, y_max = self._get_current_ax_limits()

        if breaks.get('x'):
            self.x_break_enabled.setChecked(True)
            self.x_break_start.setValue(breaks['x']['start'])
            self.x_break_end.setValue(breaks['x']['end'])
        else:
            self.x_break_enabled.setChecked(False)
            mid_x = (x_min + x_max) / 2
            range_x = (x_max - x_min) / 4
            self.x_break_start.setValue(mid_x - range_x / 2)
            self.x_break_end.setValue(mid_x + range_x / 2)

        y_breaks = breaks.get('y', {})
        if y_breaks:
            self.y_break_enabled.setChecked(True)
            self.y_break_start.setValue(y_breaks.get('start', 0.0))
            self.y_break_end.setValue(y_breaks.get('end', 0.0))
        else:
            self.y_break_enabled.setChecked(False)
            mid_y = (y_min + y_max) / 2
            range_y = (y_max - y_min) / 4
            self.y_break_start.setValue(mid_y - range_y / 2)
            self.y_break_end.setValue(mid_y + range_y / 2)

        self.cb_minor_bottom.setChecked(getattr(gw, 'minor_ticks_bottom', True))
        self.cb_minor_left.setChecked(getattr(gw, 'minor_ticks_left', True))
        self.cb_minor_top.setChecked(getattr(gw, 'minor_ticks_top', False))
        self.cb_minor_right.setChecked(getattr(gw, 'minor_ticks_right', False))

        spines = getattr(gw, 'spines_visible', None) or {'top': True, 'right': True, 'bottom': True, 'left': True}
        self.cb_spine_top.setChecked(spines.get('top', True))
        self.cb_spine_right.setChecked(spines.get('right', True))
        self.cb_spine_bottom.setChecked(spines.get('bottom', True))
        self.cb_spine_left.setChecked(spines.get('left', True))

        self._load_inset_settings(gw, x_min, x_max, y_min, y_max)
        self._load_secondary_axis_settings(gw)

    def _load_tick_format(self, value):
        """Select the preset matching `value`. A value set outside the GUI
        (scripting/AI-agent) that doesn't match any preset gets a one-off
        "Custom" entry so it isn't silently discarded by opening this tab."""
        combo = self.combo_tick_format
        idx = combo.findData(value)
        if combo.itemData(combo.count() - 1) not in dict(_TICK_FORMAT_PRESETS).values():
            combo.removeItem(combo.count() - 1)
            idx = combo.findData(value)
        if idx < 0:
            combo.addItem(f"Custom  ({value})", value)
            idx = combo.count() - 1
        combo.setCurrentIndex(idx)

    def _apply_axis_settings(self, silent: bool = False, replot: bool = True):
        """Apply axis settings to the graph.

        `silent=True` (the dialog's debounced live-preview timer) leaves an
        invalid mid-typing axis break unchanged instead of popping a
        blocking warning; the explicit Apply button (`silent=False`) blocks
        and aborts the whole apply so an invalid range is never saved.

        `replot=False` skips `_refresh_plot()`: the dialog applies every
        tab's fields first and decides on one combined restyle()-or-replot
        afterward, rather than each tab replotting on its own.
        """
        gw = self.graph_widget

        u = self._UNSET_LIMIT
        gw.xmin = self.spin_xmin.value() if self.spin_xmin.value() != u else None
        gw.xmax = self.spin_xmax.value() if self.spin_xmax.value() != u else None
        gw.ymin = self.spin_ymin.value() if self.spin_ymin.value() != u else None
        gw.ymax = self.spin_ymax.value() if self.spin_ymax.value() != u else None
        gw.zmin = self.spin_zmin.value() if self.spin_zmin.value() != u else None
        gw.zmax = self.spin_zmax.value() if self.spin_zmax.value() != u else None

        if self.x_break_enabled.isChecked():
            start, end = self.x_break_start.value(), self.x_break_end.value()
            if start >= end:
                if not silent:
                    QMessageBox.warning(self, "Invalid Range", "X-axis break start must be less than end.")
                    return
            else:
                self.graph_widget.axis_breaks['x'] = {'start': start, 'end': end}
        else:
            self.graph_widget.axis_breaks['x'] = None

        if self.y_break_enabled.isChecked():
            start, end = self.y_break_start.value(), self.y_break_end.value()
            if start >= end:
                if not silent:
                    QMessageBox.warning(self, "Invalid Range", "Y-axis break start must be less than end.")
                    return
            else:
                self.graph_widget.axis_breaks['y'] = {'start': start, 'end': end}
        else:
            self.graph_widget.axis_breaks['y'] = None

        gw.minor_ticks_bottom = self.cb_minor_bottom.isChecked()
        gw.minor_ticks_left = self.cb_minor_left.isChecked()
        gw.minor_ticks_top = self.cb_minor_top.isChecked()
        gw.minor_ticks_right = self.cb_minor_right.isChecked()

        gw.spines_visible = {
            'top': self.cb_spine_top.isChecked(),
            'right': self.cb_spine_right.isChecked(),
            'bottom': self.cb_spine_bottom.isChecked(),
            'left': self.cb_spine_left.isChecked(),
        }

        x_scale_text = self.combo_x_scale.currentText()
        gw.xlogscale = x_scale_text in ("Logarithmic", "Symlog")
        gw.xscale_mode = "symlog" if x_scale_text == "Symlog" else "log"

        y_scale_text = self.combo_y_scale.currentText()
        gw.ylogscale = y_scale_text in ("Logarithmic", "Symlog")
        gw.yscale_mode = "symlog" if y_scale_text == "Symlog" else "log"

        x_text = self.combo_x_type.currentText()
        gw.x_as_numeric = None if x_text == "Auto" else (x_text == "Numerical")

        y_text = self.combo_y_type.currentText()
        gw.y_as_numeric = None if y_text == "Auto" else (y_text == "Numerical")

        gw.x_inverted = self.cb_invert_x.isChecked()
        gw.y_inverted = self.cb_invert_y.isChecked()
        gw.tick_direction = _TICK_DIRECTION_MAP[self.combo_tick_direction.currentText()]
        gw.tick_label_format = self.combo_tick_format.currentData()

        props = {
            'xmin': gw.xmin, 'xmax': gw.xmax,
            'ymin': gw.ymin, 'ymax': gw.ymax,
            'zmin': gw.zmin, 'zmax': gw.zmax,
            'axis_breaks': gw.axis_breaks,
            'minor_ticks_bottom': gw.minor_ticks_bottom,
            'minor_ticks_left': gw.minor_ticks_left,
            'minor_ticks_top': gw.minor_ticks_top,
            'minor_ticks_right': gw.minor_ticks_right,
            'spines_visible': gw.spines_visible,
            'xlogscale': gw.xlogscale,
            'ylogscale': gw.ylogscale,
            'xscale_mode': gw.xscale_mode,
            'yscale_mode': gw.yscale_mode,
            'x_as_numeric': gw.x_as_numeric,
            'y_as_numeric': gw.y_as_numeric,
            'x_inverted': gw.x_inverted,
            'y_inverted': gw.y_inverted,
            'tick_direction': gw.tick_direction,
            'tick_label_format': gw.tick_label_format,
        }
        self._apply_inset_settings(gw, props)
        self._apply_secondary_axis_settings(gw, props)
        gw.properties_changed.emit(gw.graph_id, props)

        if replot:
            self._refresh_plot()

    def _get_current_ax_limits(self):
        """Best-effort (x_min, x_max, y_min, y_max) of the rendered Axes,
        falling back to (0, 100, 0, 100) if nothing's plotted yet. Shared
        by the break-range suggestion and the limit spinboxes' placeholder."""
        x_min, x_max, y_min, y_max = 0, 100, 0, 100
        if hasattr(self.graph_widget, 'ax') and self.graph_widget.ax is not None:
            try:
                x_min, x_max = self.graph_widget.ax.get_xlim()
                y_min, y_max = self.graph_widget.ax.get_ylim()
            except Exception:
                pass
        return x_min, x_max, y_min, y_max

    def _on_get_current_limits(self):
        """Get current axis limits from plot."""
        try:
            if hasattr(self.graph_widget, 'ax') and self.graph_widget.ax is not None:
                xmin, xmax = self.graph_widget.ax.get_xlim()
                ymin, ymax = self.graph_widget.ax.get_ylim()
                self.spin_xmin.setValue(round(xmin, 3))
                self.spin_xmax.setValue(round(xmax, 3))
                self.spin_ymin.setValue(round(ymin, 3))
                self.spin_ymax.setValue(round(ymax, 3))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error getting limits: {str(e)}")

    def _on_clear_limits(self):
        """Clear axis limits (spinboxes display blank at this sentinel)."""
        u = self._UNSET_LIMIT
        self.spin_xmin.setValue(u)
        self.spin_xmax.setValue(u)
        self.spin_ymin.setValue(u)
        self.spin_ymax.setValue(u)
        self.spin_zmin.setValue(u)
        self.spin_zmax.setValue(u)

    def _on_x_break_toggled(self, checked):
        """X/Y breaks are mutually exclusive -- the renderer only supports
        one at a time. Enabling one turns the other off outright, so Apply
        can't save a graph with both set."""
        if checked and self.y_break_enabled.isChecked():
            self.y_break_enabled.setChecked(False)

    def _on_y_break_toggled(self, checked):
        if checked and self.x_break_enabled.isChecked():
            self.x_break_enabled.setChecked(False)

    def _refresh_plot(self):
        """Refresh the plot with updated axis settings."""
        self.graph_widget.ax.clear()
        if self.graph_widget.df is not None:
            self.graph_widget.plot(self.graph_widget.df)

        # Sync the matplotlib toolbar so "Home" restores to these new limits.
        if hasattr(self.graph_widget, 'toolbar') and self.graph_widget.toolbar is not None:
            self.graph_widget.toolbar.update()
            self.graph_widget.toolbar.push_current()
