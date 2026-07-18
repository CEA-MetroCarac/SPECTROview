"""Secondary axes tab of the Customize Graph dialog: per-axis label,
limits, log-scale, color, and marker overrides for the Y2/Y3/X2 twin axes.

Split out of customize_axis.py into its own dedicated tab so these controls
are more discoverable -- they only take effect when a secondary axis is
actually assigned to the graph (column assignment itself stays in the side
panel's "Plot multiple axes" tab, not duplicated here).
"""
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QGroupBox, QLabel, QPushButton,
    QComboBox, QLineEdit, QCheckBox, QColorDialog,
)

from spectroview import MARKERS
from spectroview.view.components.customize_graph.spin_widgets import PlaceholderDoubleSpinBox

# QDoubleSpinBox.sizeHint() for a PlaceholderDoubleSpinBox showing "default"
# measures ~102px (text + up/down arrow buttons); a narrower explicit
# setMaximumWidth() clips the word to e.g. "defau" instead of shrinking the
# arrows.
_PLACEHOLDER_SPIN_WIDTH = 105


class CustomizeSecondaryAxes(QWidget):
    """Widget for customizing Y2/Y3/X2 secondary-axis label/limits/scale/style.

    A row is disabled when that secondary axis isn't currently assigned a
    column on this graph.
    """

    # Sentinel for "no limit set" -- mirrors the pattern in customize_axis.py.
    _UNSET_LIMIT = -999999

    def __init__(self, graph_widget, parent=None):
        super().__init__(parent)
        self.graph_widget = graph_widget
        self._setup_ui()
        self.load_settings()

    def switch_graph(self, graph_widget):
        """Switch to a different graph widget and reload secondary-axis settings."""
        self.graph_widget = graph_widget
        self.load_settings()

    def _make_limit_spinbox(self) -> PlaceholderDoubleSpinBox:
        """Create one axis-limit spinbox that shows a grayed "default"
        placeholder when unset."""
        spin = PlaceholderDoubleSpinBox(self._UNSET_LIMIT, 0.0, "default")
        spin.setRange(self._UNSET_LIMIT, 999999)
        return spin

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Closes two gaps at once: y2min/y2max/y3min/y3max/x2min/x2max were
        # previously only settable via scripting (no GUI at all), and
        # color/marker for these axes were hardcoded literals in v_graph.py.
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
            # *_a (not a bare `_`) absorbs however many positional args Qt's
            # signal/slot arg-count auto-detection decides to pass -- that
            # count isn't stable across PySide6 builds (see
            # graph_workspace_review.md / prior fix in customize_axis.py).
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
        layout.addStretch()

    def load_settings(self):
        """Populate the Y2/Y3/X2 rows from the graph widget."""
        gw = self.graph_widget
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
        """Open color picker for a secondary axis's color override."""
        row = self._secondary_axis_rows[axis_key]
        current = QColor(row['color'].text())
        color = QColorDialog.getColor(current, self, "Select Axis Color")
        if color.isValid():
            row['color'].setText(color.name())
            row['color'].setStyleSheet(f"background-color: {color.name()};")

    def apply_changes(self):
        """Write the Y2/Y3/X2 rows back to the graph widget, replot, and
        notify the ViewModel."""
        gw = self.graph_widget
        u = self._UNSET_LIMIT
        props = {}
        for axis_key, row in self._secondary_axis_rows.items():
            label = row['label'].text().strip() or None
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

        if gw.df is not None:
            gw.plot(gw.df)
        else:
            gw.canvas.draw_idle()

        gw.properties_changed.emit(gw.graph_id, props)
