"""Axis tab of the Customize Graph dialog: axis scale/data-type, limits,
tick direction/format/font-size, axis inversion, minor ticks, and the
"Broken axis (beta)" feature.

Split out of customize_graph_dialog.py; no behavior changes.
"""
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QMessageBox,
)

from spectroview import ICON_DIR
from spectroview.view.components.customize_graph.spin_widgets import PlaceholderDoubleSpinBox

try:
    from superqt import QLabeledDoubleRangeSlider
except ImportError:  # pragma: no cover - superqt is a hard dependency in practice
    QLabeledDoubleRangeSlider = None

# Text <-> MGraph.tick_direction value ("Default" means None -- matplotlib's
# own default, not overridden).
_TICK_DIRECTION_MAP = {"Default": None, "In": "in", "Out": "out", "In & Out": "inout"}
_TICK_DIRECTION_TEXT = {v: k for k, v in _TICK_DIRECTION_MAP.items()}

# Tick-label-format presets, replacing a freeform printf-style text field
# (confusing for non-technical users) with a small set of common choices.
# Value None means "Auto" -- matplotlib's own default ScalarFormatter.
_TICK_FORMAT_PRESETS = [
    ("Auto (default)", None),
    ("Integer  (e.g. 1234)", "%.0f"),
    ("1 decimal  (e.g. 12.3)", "%.1f"),
    ("2 decimals  (e.g. 12.34)", "%.2f"),
    ("Scientific  (e.g. 1.2e+03)", "%.1e"),
]

# Real mplstyle defaults (identical across all three theme files -- they
# only differ in color, not typography) so the font-size spinboxes always
# show a concrete, meaningful number instead of a blank/sentinel value.
_DEFAULT_TITLE_FONTSIZE = 12
_DEFAULT_AXIS_LABEL_FONTSIZE = 12
_DEFAULT_TICK_FONTSIZE = 9

# QDoubleSpinBox.sizeHint() for a PlaceholderDoubleSpinBox showing "default"
# measures ~102px (text + up/down arrow buttons); a narrower explicit
# setMaximumWidth() clips the word to e.g. "defau" instead of shrinking the
# arrows. Kept as one named width so every optional-override spinbox in this
# tab clips consistently (i.e. never).
_PLACEHOLDER_SPIN_WIDTH = 105


class CustomizeAxis(QWidget):
    """Widget for customizing axis settings (scale, limits, style, breaks)."""

    def __init__(self, graph_widget, parent=None):
        super().__init__(parent)
        self.graph_widget = graph_widget

        self._setup_ui()

        # Load current settings
        self.load_axis_settings()

    def switch_graph(self, graph_widget):
        """Switch to a different graph widget and reload axis settings."""
        self.graph_widget = graph_widget
        self.load_axis_settings()

    # Sentinel used throughout this class for "no limit set": the spinbox's
    # own range minimum doubles as the unset marker, so no separate flag is
    # needed. PlaceholderDoubleSpinBox renders that sentinel state as a
    # grayed "default" placeholder instead of the confusing literal
    # "-999999", and fixes the up/down-arrow click to jump to a sensible
    # starting value (updated per-graph via set_start_value()) instead of
    # stepping from the sentinel itself.
    _UNSET_LIMIT = -999999

    def _make_limit_spinbox(self, start_value=0.0) -> PlaceholderDoubleSpinBox:
        """Create one axis-limit spinbox that shows a grayed "default"
        placeholder when unset."""
        spin = PlaceholderDoubleSpinBox(self._UNSET_LIMIT, start_value, "default")
        spin.setRange(self._UNSET_LIMIT, 999999)
        return spin

    def _add_limit_row_with_slider(self, parent_layout, axis_label, prefix):
        """Build one 'label: [min spin] [====slider====] [max spin]' row for
        a primary axis (X/Y/Z), bidirectionally synced, following the same
        QLabeledDoubleRangeSlider pattern used by v_map_viewer.py's
        range sliders. Falls back to spinbox-only if superqt is missing."""
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
            # *_a (not a bare `_`) absorbs however many positional args Qt's
            # signal/slot arg-count auto-detection decides to pass for a
            # lambda whose only other parameter has a default (observed to
            # vary: 6.10.2 here passes the emitted double, but a user report
            # against a different PySide6 build passed zero args to the
            # equivalent QPushButton.clicked case below and raised
            # "missing 1 required positional argument") -- don't depend on it.
            spin_min.valueChanged.connect(lambda *_a, p=prefix: self._update_range_slider_from_spins(p))
            spin_max.valueChanged.connect(lambda *_a, p=prefix: self._update_range_slider_from_spins(p))

        parent_layout.addLayout(row)

    def _on_range_slider_changed(self, prefix, values):
        """Double-slider dragged -- mirror the new (min, max) into the
        spinboxes (no live replot; takes effect on Apply like every other
        field in this dialog)."""
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
        """Spinbox edited -- mirror it into the slider. An unset ("default")
        spinbox shows the slider spanning its full range, matching what
        "no limit set" actually means (matplotlib auto-scale)."""
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
        """Derive each slider's drag range from the graph's current data,
        padded past the raw extent (mirroring matplotlib's own auto-margin)
        so dragging isn't clipped to the exact data min/max. Also updates
        each spinbox's arrow-click start value to the same padded bound."""
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
            # Block signals: setRange() alone can clamp the slider's current
            # value into the new bounds (e.g. a stale (0, 100) default sitting
            # outside real data range), which fires valueChanged and — via
            # _on_range_slider_changed — overwrites the spinboxes with that
            # clamped number, stomping an unset ("default") limit with a real
            # one. _update_range_slider_from_spins() below is the single
            # source of truth for repositioning the slider from the spinboxes.
            slider.blockSignals(True)
            slider.setRange(lo, hi)
            slider.blockSignals(False)
            getattr(self, f'spin_{prefix}min').set_start_value(lo)
            getattr(self, f'spin_{prefix}max').set_start_value(hi)
            self._update_range_slider_from_spins(prefix)

    def _setup_ui(self):
        """Setup the UI components for the axis customization widget."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # ===== Axis Properties Section =====
        props_group = QGroupBox("Axis properties:")
        props_layout = QVBoxLayout(props_group)
        props_layout.setContentsMargins(4, 4, 4, 4)
        props_layout.setSpacing(8)

        # X axis properties
        x_prop_layout = QHBoxLayout()
        x_prop_layout.addWidget(QLabel("X axis:  "))

        x_prop_layout.addWidget(QLabel("Scale:"))
        self.combo_x_scale = QComboBox()
        self.combo_x_scale.addItems(["Linear", "Logarithmic", "Symlog"])
        x_prop_layout.addWidget(self.combo_x_scale)

        x_prop_layout.addSpacing(20)

        x_prop_layout.addWidget(QLabel("Data type:"))
        self.combo_x_type = QComboBox()
        self.combo_x_type.addItems(["Auto", "Category", "Numerical"])
        x_prop_layout.addWidget(self.combo_x_type)

        x_prop_layout.addSpacing(20)
        self.cb_invert_x = QCheckBox("Inverted")
        x_prop_layout.addWidget(self.cb_invert_x)
        x_prop_layout.addStretch()

        # Y axis properties
        y_prop_layout = QHBoxLayout()
        y_prop_layout.addWidget(QLabel("Y axis:  "))

        y_prop_layout.addWidget(QLabel("Scale:"))
        self.combo_y_scale = QComboBox()
        self.combo_y_scale.addItems(["Linear", "Logarithmic", "Symlog"])
        y_prop_layout.addWidget(self.combo_y_scale)

        y_prop_layout.addSpacing(20)

        y_prop_layout.addWidget(QLabel("Data type:"))
        self.combo_y_type = QComboBox()
        self.combo_y_type.addItems(["Auto", "Category", "Numerical"])
        y_prop_layout.addWidget(self.combo_y_type)

        y_prop_layout.addSpacing(20)
        self.cb_invert_y = QCheckBox("Inverted")
        y_prop_layout.addWidget(self.cb_invert_y)
        y_prop_layout.addStretch()

        props_layout.addLayout(x_prop_layout)
        props_layout.addLayout(y_prop_layout)

        layout.addWidget(props_group)

        # ===== Axis Appearance Section (tick direction/format/font sizes,
        # minor ticks -- merged into one group since they're all "how the
        # axis itself looks" decisions) =====
        style_group = QGroupBox("Axis Appearance:")
        style_layout = QVBoxLayout(style_group)
        style_layout.setContentsMargins(4, 4, 4, 4)
        style_layout.setSpacing(8)

        minor_row = QHBoxLayout()
        minor_row.addWidget(QLabel("Minor ticks:"))
        self.cb_minor_bottom = QCheckBox("X (Bottom)")
        self.cb_minor_top = QCheckBox("X (Top)")
        self.cb_minor_left = QCheckBox("Y (Left)")
        self.cb_minor_right = QCheckBox("Y (Right)")
        minor_row.addWidget(self.cb_minor_bottom)
        minor_row.addWidget(self.cb_minor_top)
        minor_row.addWidget(self.cb_minor_left)
        minor_row.addWidget(self.cb_minor_right)
        minor_row.addStretch()
        style_layout.addLayout(minor_row)

        tick_row = QHBoxLayout()
        tick_row.addWidget(QLabel("Tick direction:"))
        self.combo_tick_direction = QComboBox()
        self.combo_tick_direction.addItems(list(_TICK_DIRECTION_MAP.keys()))
        tick_row.addWidget(self.combo_tick_direction)

        tick_row.addSpacing(20)
        tick_row.addWidget(QLabel("Tick label format:"))
        self.combo_tick_format = QComboBox()
        for label, value in _TICK_FORMAT_PRESETS:
            self.combo_tick_format.addItem(label, value)
        tick_row.addWidget(self.combo_tick_format)
        tick_row.addStretch()
        style_layout.addLayout(tick_row)

        font_row = QHBoxLayout()
        font_row.addWidget(QLabel("Font size (pt):"))
        for label_text, attr, default in [
            ("Title:", "spin_title_fontsize", _DEFAULT_TITLE_FONTSIZE),
            ("Axis label:", "spin_axis_label_fontsize", _DEFAULT_AXIS_LABEL_FONTSIZE),
            ("Tick label:", "spin_tick_fontsize", _DEFAULT_TICK_FONTSIZE),
        ]:
            font_row.addWidget(QLabel(label_text))
            spin = QSpinBox()
            spin.setRange(4, 72)
            spin.setSingleStep(1)
            spin.setValue(default)
            setattr(self, attr, spin)
            font_row.addWidget(spin)
        font_row.addStretch()
        style_layout.addLayout(font_row)

        layout.addWidget(style_group)

        # ===== Axis Limits Section =====
        limits_group = QGroupBox("Set Axis Limits:")
        limits_layout = QVBoxLayout(limits_group)
        limits_layout.setContentsMargins(4, 4, 4, 4)
        limits_layout.setSpacing(8)

        # X, Y limits -- hidden for wafer/2Dmap (see load_axis_settings()),
        # where X/Y are spatial axes governed by wafer_size, not a
        # user-set min/max; Z is the relevant limit control there.
        self.xy_limits_widget = QWidget()
        xy_limits_layout = QVBoxLayout(self.xy_limits_widget)
        xy_limits_layout.setContentsMargins(0, 0, 0, 0)
        xy_limits_layout.setSpacing(8)
        self._add_limit_row_with_slider(xy_limits_layout, "X", "x")
        self._add_limit_row_with_slider(xy_limits_layout, "Y", "y")
        limits_layout.addWidget(self.xy_limits_widget)

        # Z limits
        self._add_limit_row_with_slider(limits_layout, "Z", "z")

        # Limit buttons
        limits_btn_layout = QHBoxLayout()
        self.btn_set_limits = QPushButton("Get current limits from plot")
        self.btn_set_limits.clicked.connect(self._on_get_current_limits)

        self.btn_clear_limits = QPushButton("Clear limits")
        self.btn_clear_limits.setIcon(QIcon(f"{ICON_DIR}/clear.png"))
        self.btn_clear_limits.clicked.connect(self._on_clear_limits)

        limits_btn_layout.addWidget(self.btn_set_limits)
        limits_btn_layout.addWidget(self.btn_clear_limits)
        limits_layout.addLayout(limits_btn_layout)

        # ===== Axis Break Section =====
        break_group = QGroupBox("Broken axis (beta):")
        break_layout = QVBoxLayout()
        break_layout.setContentsMargins(4, 4, 4, 4)
        break_layout.setSpacing(8)

        x_break_layout = QHBoxLayout()
        x_break_layout.setContentsMargins(0, 0, 0, 0)
        x_break_layout.setSpacing(8)

        # Enable checkbox
        self.x_break_enabled = QCheckBox("X-axis break")

        # Input fields
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

        # Enable checkbox
        self.y_break_enabled = QCheckBox("Y-axis break")

        # Input fields
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

        layout.addWidget(limits_group)
        layout.addWidget(break_group)
        layout.addStretch()

    def load_axis_settings(self):
        """Load current axis settings (limits and breaks) from graph widget."""
        gw = self.graph_widget

        # X/Y limits don't apply to wafer/2Dmap plots (spatial axes governed
        # by wafer_size, not a user-set min/max) -- hide them there; Z stays
        # visible since it's the color-scale control for those styles.
        self.xy_limits_widget.setVisible(gw.plot_style not in ('wafer', '2Dmap'))

        # Load limits
        u = self._UNSET_LIMIT
        self.spin_xmin.setValue(gw.xmin if gw.xmin is not None else u)
        self.spin_xmax.setValue(gw.xmax if gw.xmax is not None else u)
        self.spin_ymin.setValue(gw.ymin if gw.ymin is not None else u)
        self.spin_ymax.setValue(gw.ymax if gw.ymax is not None else u)
        self.spin_zmin.setValue(gw.zmin if gw.zmin is not None else u)
        self.spin_zmax.setValue(gw.zmax if gw.zmax is not None else u)
        self._update_range_slider_bounds()

        # Load Axis Scales (log/symlog share the xlogscale/ylogscale on/off
        # gate; xscale_mode/yscale_mode pick which of the two)
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

        # Load Axis Types
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

        # Load axis inversion
        self.cb_invert_x.setChecked(getattr(gw, 'x_inverted', False))
        self.cb_invert_y.setChecked(getattr(gw, 'y_inverted', False))

        # Load tick direction / label format / font sizes
        self.combo_tick_direction.setCurrentText(
            _TICK_DIRECTION_TEXT.get(getattr(gw, 'tick_direction', None), "Default")
        )
        self._load_tick_format(getattr(gw, 'tick_label_format', None))
        self.spin_title_fontsize.setValue(getattr(gw, 'title_fontsize', None) or _DEFAULT_TITLE_FONTSIZE)
        self.spin_axis_label_fontsize.setValue(
            getattr(gw, 'axis_label_fontsize', None) or _DEFAULT_AXIS_LABEL_FONTSIZE
        )
        self.spin_tick_fontsize.setValue(getattr(gw, 'tick_label_fontsize', None) or _DEFAULT_TICK_FONTSIZE)

        if not hasattr(self.graph_widget, 'axis_breaks'):
            self.graph_widget.axis_breaks = {'x': None, 'y': None}

        breaks = self.graph_widget.axis_breaks

        # Get current axis limits from the graph
        x_min, x_max, y_min, y_max = 0, 100, 0, 100  # defaults
        if hasattr(self.graph_widget, 'ax') and self.graph_widget.ax is not None:
            try:
                x_min, x_max = self.graph_widget.ax.get_xlim()
                y_min, y_max = self.graph_widget.ax.get_ylim()
            except:
                pass  # Use defaults if axis not available

        # Load X-axis break
        if breaks.get('x'):
            self.x_break_enabled.setChecked(True)
            self.x_break_start.setValue(breaks['x']['start'])
            self.x_break_end.setValue(breaks['x']['end'])
        else:
            self.x_break_enabled.setChecked(False)
            # Set to middle range as suggestion
            mid_x = (x_min + x_max) / 2
            range_x = (x_max - x_min) / 4
            self.x_break_start.setValue(mid_x - range_x/2)
            self.x_break_end.setValue(mid_x + range_x/2)

        # Load Y-axis break
        y_breaks = breaks.get('y', {})
        if y_breaks:
            self.y_break_enabled.setChecked(True)
            self.y_break_start.setValue(y_breaks.get('start', 0.0))
            self.y_break_end.setValue(y_breaks.get('end', 0.0))

        self.cb_minor_bottom.setChecked(getattr(gw, 'minor_ticks_bottom', True))
        self.cb_minor_left.setChecked(getattr(gw, 'minor_ticks_left', True))
        self.cb_minor_top.setChecked(getattr(gw, 'minor_ticks_top', False))
        self.cb_minor_right.setChecked(getattr(gw, 'minor_ticks_right', False))
        if not y_breaks:
            self.y_break_enabled.setChecked(False)
            # Set to middle range as suggestion
            mid_y = (y_min + y_max) / 2
            range_y = (y_max - y_min) / 4
            self.y_break_start.setValue(mid_y - range_y/2)
            self.y_break_end.setValue(mid_y + range_y/2)

    def _load_tick_format(self, value):
        """Select the preset matching `value` in the tick-format combo. A
        value set outside the GUI (scripting/AI-agent tool) that doesn't
        match any preset gets a one-off "Custom" entry inserted so it isn't
        silently discarded/overwritten by opening this dialog."""
        combo = self.combo_tick_format
        idx = combo.findData(value)
        # Drop any previously-inserted custom entry before re-checking.
        if combo.itemData(combo.count() - 1) not in dict(_TICK_FORMAT_PRESETS).values():
            combo.removeItem(combo.count() - 1)
            idx = combo.findData(value)
        if idx < 0:
            combo.addItem(f"Custom  ({value})", value)
            idx = combo.count() - 1
        combo.setCurrentIndex(idx)

    def _apply_axis_settings(self):
        """Apply axis settings to the graph."""
        gw = self.graph_widget

        # Save limits
        u = self._UNSET_LIMIT
        gw.xmin = self.spin_xmin.value() if self.spin_xmin.value() != u else None
        gw.xmax = self.spin_xmax.value() if self.spin_xmax.value() != u else None
        gw.ymin = self.spin_ymin.value() if self.spin_ymin.value() != u else None
        gw.ymax = self.spin_ymax.value() if self.spin_ymax.value() != u else None
        gw.zmin = self.spin_zmin.value() if self.spin_zmin.value() != u else None
        gw.zmax = self.spin_zmax.value() if self.spin_zmax.value() != u else None

        # Validate and save X-axis break
        if self.x_break_enabled.isChecked():
            start = self.x_break_start.value()
            end = self.x_break_end.value()

            if start >= end:
                QMessageBox.warning(self, "Invalid Range",
                                  "X-axis break start must be less than end.")
                return

            self.graph_widget.axis_breaks['x'] = {'start': start, 'end': end}
        else:
            self.graph_widget.axis_breaks['x'] = None

        # Validate and save Y-axis break
        if self.y_break_enabled.isChecked():
            start = self.y_break_start.value()
            end = self.y_break_end.value()

            if start >= end:
                QMessageBox.warning(self, "Invalid Range",
                                  "Y-axis break start must be less than end.")
                return

            self.graph_widget.axis_breaks['y'] = {'start': start, 'end': end}
        else:
            self.graph_widget.axis_breaks['y'] = None

        # Connect to properties_changed signal to update ViewModel when graph properties change
        gw.minor_ticks_bottom = self.cb_minor_bottom.isChecked()
        gw.minor_ticks_left = self.cb_minor_left.isChecked()
        gw.minor_ticks_top = self.cb_minor_top.isChecked()
        gw.minor_ticks_right = self.cb_minor_right.isChecked()

        x_scale_text = self.combo_x_scale.currentText()
        gw.xlogscale = x_scale_text in ("Logarithmic", "Symlog")
        gw.xscale_mode = "symlog" if x_scale_text == "Symlog" else "log"

        y_scale_text = self.combo_y_scale.currentText()
        gw.ylogscale = y_scale_text in ("Logarithmic", "Symlog")
        gw.yscale_mode = "symlog" if y_scale_text == "Symlog" else "log"

        x_text = self.combo_x_type.currentText()
        if x_text == "Auto":
            gw.x_as_numeric = None
        elif x_text == "Numerical":
            gw.x_as_numeric = True
        else:
            gw.x_as_numeric = False

        y_text = self.combo_y_type.currentText()
        if y_text == "Auto":
            gw.y_as_numeric = None
        elif y_text == "Numerical":
            gw.y_as_numeric = True
        else:
            gw.y_as_numeric = False

        gw.x_inverted = self.cb_invert_x.isChecked()
        gw.y_inverted = self.cb_invert_y.isChecked()
        gw.tick_direction = _TICK_DIRECTION_MAP[self.combo_tick_direction.currentText()]
        gw.tick_label_format = self.combo_tick_format.currentData()
        gw.title_fontsize = self.spin_title_fontsize.value()
        gw.axis_label_fontsize = self.spin_axis_label_fontsize.value()
        gw.tick_label_fontsize = self.spin_tick_fontsize.value()

        # Emit signal to ViewModel
        gw.properties_changed.emit(gw.graph_id, {
            'xmin': gw.xmin, 'xmax': gw.xmax,
            'ymin': gw.ymin, 'ymax': gw.ymax,
            'zmin': gw.zmin, 'zmax': gw.zmax,
            'axis_breaks': gw.axis_breaks,
            'minor_ticks_bottom': gw.minor_ticks_bottom,
            'minor_ticks_left': gw.minor_ticks_left,
            'minor_ticks_top': gw.minor_ticks_top,
            'minor_ticks_right': gw.minor_ticks_right,
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
            'title_fontsize': gw.title_fontsize,
            'axis_label_fontsize': gw.axis_label_fontsize,
            'tick_label_fontsize': gw.tick_label_fontsize,
        })

        # Refresh the plot with settings applied
        self._refresh_plot()

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

    def _refresh_plot(self):
        """Refresh the plot with updated axis settings."""
        self.graph_widget.ax.clear()
        if self.graph_widget.df is not None:
            self.graph_widget.plot(self.graph_widget.df)

        # Update the matplotlib navigation toolbar so "Home" restores to these new limits
        if hasattr(self.graph_widget, 'toolbar') and self.graph_widget.toolbar is not None:
            self.graph_widget.toolbar.update()
            self.graph_widget.toolbar.push_current()
