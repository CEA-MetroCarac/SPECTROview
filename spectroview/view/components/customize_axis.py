"""Axis tab of the Customize Graph dialog: axis scale/data-type, limits,
minor ticks, and the "Broken axis (beta)" feature.

Split out of customize_graph_dialog.py; no behavior changes.
"""
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QCheckBox, QMessageBox,
)

from spectroview import ICON_DIR


class CustomizeAxis(QWidget):
    """Widget for customizing axis settings (breaks)."""

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
    # needed. setSpecialValueText() makes that sentinel state render as a
    # blank box instead of the confusing literal "-999999" -- note the text
    # must be a single space, not "": Qt treats an *empty* special-value
    # string as "no special text configured" and falls back to showing the
    # raw number, so "" alone doesn't produce a blank display.
    _UNSET_LIMIT = -999999
    _BLANK_TEXT = " "

    def _make_limit_spinbox(self) -> QDoubleSpinBox:
        """Create one axis-limit spinbox that displays blank when unset."""
        spin = QDoubleSpinBox()
        spin.setRange(self._UNSET_LIMIT, 999999)
        spin.setSpecialValueText(self._BLANK_TEXT)
        return spin

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
        self.combo_x_scale.addItems(["Linear", "Logarithmic"])
        x_prop_layout.addWidget(self.combo_x_scale)

        x_prop_layout.addSpacing(20)

        x_prop_layout.addWidget(QLabel("Data type:"))
        self.combo_x_type = QComboBox()
        self.combo_x_type.addItems(["Auto", "Category", "Numerical"])
        x_prop_layout.addWidget(self.combo_x_type)
        x_prop_layout.addStretch()

        # Y axis properties
        y_prop_layout = QHBoxLayout()
        y_prop_layout.addWidget(QLabel("Y axis:  "))

        y_prop_layout.addWidget(QLabel("Scale:"))
        self.combo_y_scale = QComboBox()
        self.combo_y_scale.addItems(["Linear", "Logarithmic"])
        y_prop_layout.addWidget(self.combo_y_scale)

        y_prop_layout.addSpacing(20)

        y_prop_layout.addWidget(QLabel("Data type:"))
        self.combo_y_type = QComboBox()
        self.combo_y_type.addItems(["Auto", "Category", "Numerical"])
        y_prop_layout.addWidget(self.combo_y_type)
        y_prop_layout.addStretch()

        props_layout.addLayout(x_prop_layout)
        props_layout.addLayout(y_prop_layout)

        layout.addWidget(props_group)


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
        for axis in ['X', 'Y']:
            h_layout = QHBoxLayout()
            h_layout.addWidget(QLabel(f"{axis} axis limits:"))
            for limit_type in ['min', 'max']:
                spin = self._make_limit_spinbox()
                setattr(self, f'spin_{axis.lower()}{limit_type}', spin)
                h_layout.addWidget(QLabel(limit_type))
                h_layout.addWidget(spin)
            xy_limits_layout.addLayout(h_layout)
        limits_layout.addWidget(self.xy_limits_widget)

        # Z limits
        z_limits_layout = QHBoxLayout()
        z_limits_layout.addWidget(QLabel("Z axis limits:"))
        for limit_type in ['min', 'max']:
            spin = self._make_limit_spinbox()
            setattr(self, f'spin_z{limit_type}', spin)
            z_limits_layout.addWidget(QLabel(limit_type))
            z_limits_layout.addWidget(spin)
        limits_layout.addLayout(z_limits_layout)

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

        # ===== Minor Ticks Section =====
        minor_ticks_group = QGroupBox("Add minor tick:")
        minor_ticks_layout = QHBoxLayout()
        minor_ticks_layout.setContentsMargins(4, 4, 4, 4)
        minor_ticks_layout.setSpacing(8)

        self.cb_minor_bottom = QCheckBox("X (Bottom)")
        self.cb_minor_top = QCheckBox("X (Top)")
        self.cb_minor_left = QCheckBox("Y (Left)")
        self.cb_minor_right = QCheckBox("Y (Right)")

        minor_ticks_layout.addWidget(self.cb_minor_bottom)
        minor_ticks_layout.addWidget(self.cb_minor_top)
        minor_ticks_layout.addWidget(self.cb_minor_left)
        minor_ticks_layout.addWidget(self.cb_minor_right)
        minor_ticks_layout.addStretch()

        minor_ticks_group.setLayout(minor_ticks_layout)

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
        layout.addWidget(minor_ticks_group)
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

        # Load Axis Scales
        if getattr(gw, 'xlogscale', False):
            self.combo_x_scale.setCurrentText("Logarithmic")
        else:
            self.combo_x_scale.setCurrentText("Linear")

        if getattr(gw, 'ylogscale', False):
            self.combo_y_scale.setCurrentText("Logarithmic")
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

        gw.xlogscale = (self.combo_x_scale.currentText() == "Logarithmic")
        gw.ylogscale = (self.combo_y_scale.currentText() == "Logarithmic")

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
            'x_as_numeric': gw.x_as_numeric,
            'y_as_numeric': gw.y_as_numeric
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
