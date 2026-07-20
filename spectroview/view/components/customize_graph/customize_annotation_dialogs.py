"""Small standalone dialogs/delegates used by the Customize Graph dialog's
Annotations and Legend tabs -- modal editors for a single annotation, plus
the color-swatch item delegate the Legend tab's color comboboxes use.

Split out of customize_graph_dialog.py (which had grown to ~1700 lines
across 8 largely independent classes) purely to shrink that file.

Each editor previews live: given the annotation's graph widget, every
control change repaints the plot (debounced), and Cancel restores the
original state. Coordinate spinboxes take a data-relative range/step from
the graph's current axis limits so stepping is meaningful, and span
start/end also get a double-range slider (like the Axis tab's limits).
"""
import copy

from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QCheckBox, QFormLayout, QDialogButtonBox, QStyledItemDelegate, QColorDialog,
)

try:
    from superqt import QLabeledDoubleRangeSlider
except ImportError:  # pragma: no cover - superqt is a hard dependency in practice
    QLabeledDoubleRangeSlider = None


class LivePreviewAnnotationDialog(QDialog):
    """Base for annotation edit dialogs.

    When constructed with the annotation's `graph_widget`, control changes
    preview live on the plot (debounced) and Cancel restores the original
    values. Subclasses build their own controls, call `_connect_live(...)`
    on them, and implement `get_properties()`; the base handles the
    debounce, the replot, and the restore-on-cancel.
    """

    def __init__(self, annotation, graph_widget=None, parent=None):
        super().__init__(parent)
        self.annotation = annotation
        self.graph_widget = graph_widget
        self._original = copy.deepcopy(annotation)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(100)
        self._preview_timer.timeout.connect(self._apply_preview)

    # ----- axis-derived spinbox ranges -----------------------------------

    def _axis_ranges(self):
        """(xlim, ylim) of the graph's current axes, or None if unavailable."""
        ax = getattr(self.graph_widget, 'ax', None) if self.graph_widget else None
        if ax is None:
            return None
        return ax.get_xlim(), ax.get_ylim()

    @staticmethod
    def _configure_coord_spinbox(spin, lo, hi, positive_only=False):
        """Give a coordinate spinbox a data-relative range/step -- the raw
        -999999..999999 range with a step of 1 is unusable at most data
        scales. Padded generously so a value can still be dragged past the
        current view."""
        span = abs(hi - lo) or 1.0
        pad = span * 5
        spin.setDecimals(3)
        spin.setSingleStep(span / 100.0)
        spin.setRange(0.001 if positive_only else min(lo, hi) - pad, max(lo, hi) + pad)

    # ----- live preview --------------------------------------------------

    def _connect_live(self, *widgets):
        """Wire each control's change signal to the debounced preview."""
        for w in widgets:
            if hasattr(w, 'valueChanged'):
                w.valueChanged.connect(self._schedule_preview)
            elif hasattr(w, 'currentIndexChanged'):
                w.currentIndexChanged.connect(self._schedule_preview)
            elif hasattr(w, 'textChanged'):
                w.textChanged.connect(self._schedule_preview)
            elif hasattr(w, 'toggled'):
                w.toggled.connect(self._schedule_preview)

    def _schedule_preview(self, *_):
        if self.graph_widget is not None:
            self._preview_timer.start()

    def _apply_preview(self):
        self.annotation.update(self.get_properties())
        self._replot()

    def _replot(self):
        gw = self.graph_widget
        if gw is None or getattr(gw, 'df', None) is None:
            return
        gw.ax.clear()
        gw.plot(gw.df)

    def reject(self):
        """Cancel: drop any pending preview, restore the original annotation
        values, and repaint before closing."""
        self._preview_timer.stop()
        if self.graph_widget is not None:
            self.annotation.clear()
            self.annotation.update(self._original)
            self._replot()
        super().reject()

    def get_properties(self):  # pragma: no cover - overridden by subclasses
        raise NotImplementedError


class EditLineDialog(LivePreviewAnnotationDialog):
    """Dialog for editing line annotations (vline/hline)."""

    def __init__(self, annotation, graph_widget=None, parent=None):
        super().__init__(annotation, graph_widget, parent)
        self.setWindowTitle("Edit Line Annotation")
        self.resize(350, 200)

        layout = QFormLayout(self)

        self.color_button = QPushButton()
        current_color = QColor(annotation.get('color', 'red'))
        self.color_button.setStyleSheet(f"background-color: {current_color.name()};")
        self.color_button.setText(current_color.name())
        self.color_button.clicked.connect(self._pick_color)

        self.linestyle_combo = QComboBox()
        self.linestyle_combo.addItem("Solid", "-")
        self.linestyle_combo.addItem("Dashed", "--")
        self.linestyle_combo.addItem("Dotted", ":")
        self.linestyle_combo.addItem("Dash-Dot", "-.")
        index = self.linestyle_combo.findData(annotation.get('linestyle', '--'))
        if index >= 0:
            self.linestyle_combo.setCurrentIndex(index)

        self.linewidth_spin = QDoubleSpinBox()
        self.linewidth_spin.setRange(0.5, 5.0)
        self.linewidth_spin.setSingleStep(0.5)
        self.linewidth_spin.setValue(annotation.get('linewidth', 1.5))

        layout.addRow("Color:", self.color_button)
        layout.addRow("Line Style:", self.linestyle_combo)
        layout.addRow("Line Width:", self.linewidth_spin)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

        self._connect_live(self.linestyle_combo, self.linewidth_spin)

    def _pick_color(self):
        current_color = QColor(self.color_button.text())
        color = QColorDialog.getColor(current_color, self, "Select Line Color")
        if color.isValid():
            self.color_button.setStyleSheet(f"background-color: {color.name()};")
            self.color_button.setText(color.name())
            self._schedule_preview()

    def get_properties(self):
        return {
            'color': self.color_button.text(),
            'linestyle': self.linestyle_combo.currentData(),
            'linewidth': self.linewidth_spin.value(),
        }


class EditTextDialog(LivePreviewAnnotationDialog):
    """Dialog for editing text annotations."""

    def __init__(self, annotation, graph_widget=None, parent=None):
        super().__init__(annotation, graph_widget, parent)
        self.setWindowTitle("Edit Text Annotation")
        self.resize(400, 350)

        layout = QFormLayout(self)

        self.text_edit = QLineEdit()
        self.text_edit.setText(annotation.get('text', 'Text'))

        self.fontsize_spin = QSpinBox()
        self.fontsize_spin.setRange(6, 72)
        self.fontsize_spin.setValue(annotation.get('fontsize', 12))

        self.text_color_button = QPushButton()
        current_color = QColor(annotation.get('color', 'black'))
        self.text_color_button.setStyleSheet(f"background-color: {current_color.name()};")
        self.text_color_button.setText(current_color.name())
        self.text_color_button.clicked.connect(self._pick_text_color)

        self.frame_checkbox = QCheckBox("Show frame/box")
        bbox = annotation.get('bbox')
        self.frame_checkbox.setChecked(bbox is not None)

        self.bg_color_button = QPushButton()
        if isinstance(bbox, dict) and bbox.get('facecolor'):
            bg_color = QColor(bbox.get('facecolor'))
        else:
            bg_color = QColor('yellow')
        self.bg_color_button.setStyleSheet(f"background-color: {bg_color.name()};")
        self.bg_color_button.setText(bg_color.name())
        self.bg_color_button.clicked.connect(self._pick_bg_color)

        self.transparency_slider = QSpinBox()
        self.transparency_slider.setRange(0, 100)
        self.transparency_slider.setSingleStep(10)
        self.transparency_slider.setSuffix("%")
        current_alpha = 0.7
        if isinstance(bbox, dict) and 'alpha' in bbox:
            current_alpha = bbox['alpha']
        self.transparency_slider.setValue(int(current_alpha * 100))

        layout.addRow("Text:", self.text_edit)
        layout.addRow("Font Size:", self.fontsize_spin)
        layout.addRow("Text Color:", self.text_color_button)
        layout.addRow("", self.frame_checkbox)
        layout.addRow("BG Color:", self.bg_color_button)
        layout.addRow("Transparency:", self.transparency_slider)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

        self._connect_live(self.text_edit, self.fontsize_spin,
                           self.transparency_slider, self.frame_checkbox)

    def _pick_text_color(self):
        current_color = QColor(self.text_color_button.text())
        color = QColorDialog.getColor(current_color, self, "Select Text Color")
        if color.isValid():
            self.text_color_button.setStyleSheet(f"background-color: {color.name()};")
            self.text_color_button.setText(color.name())
            self._schedule_preview()

    def _pick_bg_color(self):
        current_color = QColor(self.bg_color_button.text())
        color = QColorDialog.getColor(current_color, self, "Select Background Color")
        if color.isValid():
            self.bg_color_button.setStyleSheet(f"background-color: {color.name()};")
            self.bg_color_button.setText(color.name())
            self._schedule_preview()

    def get_properties(self):
        props = {
            'text': self.text_edit.text(),
            'fontsize': self.fontsize_spin.value(),
            'color': self.text_color_button.text(),
            'ha': 'center',
            'va': 'center',
        }
        if self.frame_checkbox.isChecked():
            props['bbox'] = {
                'facecolor': self.bg_color_button.text(),
                'edgecolor': 'black',
                'boxstyle': 'round,pad=0.3',
                'alpha': self.transparency_slider.value() / 100.0,
            }
        else:
            props['bbox'] = None
        return props


class EditArrowDialog(LivePreviewAnnotationDialog):
    """Dialog for editing arrow annotations."""

    def __init__(self, annotation, graph_widget=None, parent=None):
        super().__init__(annotation, graph_widget, parent)
        self.setWindowTitle("Edit Arrow Annotation")
        self.resize(350, 300)

        layout = QFormLayout(self)

        self.x1_spin = QDoubleSpinBox()
        self.y1_spin = QDoubleSpinBox()
        self.x2_spin = QDoubleSpinBox()
        self.y2_spin = QDoubleSpinBox()
        self._init_coord_spinboxes(
            (self.x1_spin, 'x1', 'x'), (self.y1_spin, 'y1', 'y'),
            (self.x2_spin, 'x2', 'x'), (self.y2_spin, 'y2', 'y'),
        )

        self.color_button = QPushButton()
        current_color = QColor(annotation.get('color', 'black'))
        self.color_button.setStyleSheet(f"background-color: {current_color.name()};")
        self.color_button.setText(current_color.name())
        self.color_button.clicked.connect(self._pick_color)

        self.linestyle_combo = QComboBox()
        self.linestyle_combo.addItem("Solid", "-")
        self.linestyle_combo.addItem("Dashed", "--")
        self.linestyle_combo.addItem("Dotted", ":")
        self.linestyle_combo.addItem("Dash-Dot", "-.")
        index = self.linestyle_combo.findData(annotation.get('linestyle', '-'))
        if index >= 0:
            self.linestyle_combo.setCurrentIndex(index)

        self.linewidth_spin = QDoubleSpinBox()
        self.linewidth_spin.setRange(0.5, 5.0)
        self.linewidth_spin.setSingleStep(0.5)
        self.linewidth_spin.setValue(annotation.get('linewidth', 1.5))

        layout.addRow("Start X:", self.x1_spin)
        layout.addRow("Start Y:", self.y1_spin)
        layout.addRow("End X:", self.x2_spin)
        layout.addRow("End Y:", self.y2_spin)
        layout.addRow("Color:", self.color_button)
        layout.addRow("Line Style:", self.linestyle_combo)
        layout.addRow("Line Width:", self.linewidth_spin)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

        self._connect_live(self.x1_spin, self.y1_spin, self.x2_spin, self.y2_spin,
                           self.linestyle_combo, self.linewidth_spin)

    def _init_coord_spinboxes(self, *specs):
        """Build each (spin, key, axis) coordinate spinbox with an
        axis-derived range/step, falling back to the old wide range when no
        graph is attached."""
        ranges = self._axis_ranges()
        for spin, key, axis in specs:
            if ranges is not None:
                lo, hi = ranges[0] if axis == 'x' else ranges[1]
                self._configure_coord_spinbox(spin, lo, hi)
            else:
                spin.setRange(-999999, 999999)
                spin.setDecimals(3)
            spin.setValue(self.annotation.get(key, 0.0))

    def _pick_color(self):
        current_color = QColor(self.color_button.text())
        color = QColorDialog.getColor(current_color, self, "Select Arrow Color")
        if color.isValid():
            self.color_button.setStyleSheet(f"background-color: {color.name()};")
            self.color_button.setText(color.name())
            self._schedule_preview()

    def get_properties(self):
        return {
            'x1': self.x1_spin.value(), 'y1': self.y1_spin.value(),
            'x2': self.x2_spin.value(), 'y2': self.y2_spin.value(),
            'color': self.color_button.text(),
            'linestyle': self.linestyle_combo.currentData(),
            'linewidth': self.linewidth_spin.value(),
        }


class EditSpanDialog(LivePreviewAnnotationDialog):
    """Dialog for editing span annotations (vspan/hspan) -- start/end via a
    double-range slider synced with the two spinboxes."""

    def __init__(self, annotation, graph_widget=None, parent=None):
        super().__init__(annotation, graph_widget, parent)
        self.is_vertical = annotation.get('type') == 'vspan'
        self.setWindowTitle("Edit Span Annotation")
        self.resize(380, 240)

        layout = QFormLayout(self)

        axis_label = "X" if self.is_vertical else "Y"
        key1, key2 = ('x1', 'x2') if self.is_vertical else ('y1', 'y2')

        self.start_spin = QDoubleSpinBox()
        self.end_spin = QDoubleSpinBox()
        ranges = self._axis_ranges()
        axis_lim = None
        if ranges is not None:
            axis_lim = ranges[0] if self.is_vertical else ranges[1]
            for spin in (self.start_spin, self.end_spin):
                self._configure_coord_spinbox(spin, axis_lim[0], axis_lim[1])
        else:
            for spin in (self.start_spin, self.end_spin):
                spin.setRange(-999999, 999999)
                spin.setDecimals(3)
        self.start_spin.setValue(annotation.get(key1, 0.0))
        self.end_spin.setValue(annotation.get(key2, 1.0))

        # Double-range slider spanning the axis (mirrors the Axis tab).
        self.range_slider = None
        if QLabeledDoubleRangeSlider is not None and axis_lim is not None:
            self.range_slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
            self.range_slider.setEdgeLabelMode(QLabeledDoubleRangeSlider.EdgeLabelMode.NoLabel)
            self.range_slider.setHandleLabelPosition(QLabeledDoubleRangeSlider.LabelPosition.NoLabel)
            span = abs(axis_lim[1] - axis_lim[0]) or 1.0
            pad = span * 0.1
            self.range_slider.setRange(min(axis_lim) - pad, max(axis_lim) + pad)
            self.range_slider.setValue((self.start_spin.value(), self.end_spin.value()))
            self.range_slider.valueChanged.connect(self._on_slider_changed)

        self.color_button = QPushButton()
        current_color = QColor(annotation.get('color', 'orange'))
        self.color_button.setStyleSheet(f"background-color: {current_color.name()};")
        self.color_button.setText(current_color.name())
        self.color_button.clicked.connect(self._pick_color)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setValue(annotation.get('alpha', 0.3))

        layout.addRow(f"{axis_label} start:", self.start_spin)
        layout.addRow(f"{axis_label} end:", self.end_spin)
        if self.range_slider is not None:
            layout.addRow("Range:", self.range_slider)
        layout.addRow("Color:", self.color_button)
        layout.addRow("Transparency:", self.alpha_spin)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

        self.start_spin.valueChanged.connect(self._on_spin_changed)
        self.end_spin.valueChanged.connect(self._on_spin_changed)
        self._connect_live(self.alpha_spin)

    def _on_slider_changed(self, values):
        """Slider dragged -- mirror into the spinboxes (which trigger the
        live preview via _on_spin_changed)."""
        self.start_spin.blockSignals(True)
        self.end_spin.blockSignals(True)
        self.start_spin.setValue(values[0])
        self.end_spin.setValue(values[1])
        self.start_spin.blockSignals(False)
        self.end_spin.blockSignals(False)
        self._schedule_preview()

    def _on_spin_changed(self, *_):
        """Spinbox edited -- mirror into the slider and preview."""
        if self.range_slider is not None:
            lo, hi = self.range_slider.minimum(), self.range_slider.maximum()
            vmin = max(lo, min(self.start_spin.value(), hi))
            vmax = max(lo, min(self.end_spin.value(), hi))
            if vmin <= vmax:
                self.range_slider.blockSignals(True)
                self.range_slider.setValue((vmin, vmax))
                self.range_slider.blockSignals(False)
        self._schedule_preview()

    def _pick_color(self):
        current_color = QColor(self.color_button.text())
        color = QColorDialog.getColor(current_color, self, "Select Span Color")
        if color.isValid():
            self.color_button.setStyleSheet(f"background-color: {color.name()};")
            self.color_button.setText(color.name())
            self._schedule_preview()

    def get_properties(self):
        key1, key2 = ('x1', 'x2') if self.is_vertical else ('y1', 'y2')
        return {
            key1: self.start_spin.value(),
            key2: self.end_spin.value(),
            'color': self.color_button.text(),
            'alpha': self.alpha_spin.value(),
        }


class EditBoxDialog(LivePreviewAnnotationDialog):
    """Dialog for editing box (rectangle) annotations."""

    def __init__(self, annotation, graph_widget=None, parent=None):
        super().__init__(annotation, graph_widget, parent)
        self.setWindowTitle("Edit Box Annotation")
        self.resize(350, 340)

        layout = QFormLayout(self)

        self.x_spin = QDoubleSpinBox()
        self.y_spin = QDoubleSpinBox()
        self.width_spin = QDoubleSpinBox()
        self.height_spin = QDoubleSpinBox()
        ranges = self._axis_ranges()
        if ranges is not None:
            (xlo, xhi), (ylo, yhi) = ranges
            self._configure_coord_spinbox(self.x_spin, xlo, xhi)
            self._configure_coord_spinbox(self.y_spin, ylo, yhi)
            self._configure_coord_spinbox(self.width_spin, xlo, xhi, positive_only=True)
            self._configure_coord_spinbox(self.height_spin, ylo, yhi, positive_only=True)
        else:
            for spin in (self.x_spin, self.y_spin):
                spin.setRange(-999999, 999999)
                spin.setDecimals(3)
            for spin in (self.width_spin, self.height_spin):
                spin.setRange(0.001, 999999)
                spin.setDecimals(3)
        self.x_spin.setValue(annotation.get('x', 0.0))
        self.y_spin.setValue(annotation.get('y', 0.0))
        self.width_spin.setValue(annotation.get('width', 1.0))
        self.height_spin.setValue(annotation.get('height', 1.0))

        self.facecolor_button = QPushButton()
        current_face = QColor(annotation.get('facecolor', 'yellow'))
        self.facecolor_button.setStyleSheet(f"background-color: {current_face.name()};")
        self.facecolor_button.setText(current_face.name())
        self.facecolor_button.clicked.connect(self._pick_facecolor)

        self.edgecolor_button = QPushButton()
        current_edge = QColor(annotation.get('edgecolor', 'black'))
        self.edgecolor_button.setStyleSheet(f"background-color: {current_edge.name()};")
        self.edgecolor_button.setText(current_edge.name())
        self.edgecolor_button.clicked.connect(self._pick_edgecolor)

        self.linewidth_spin = QDoubleSpinBox()
        self.linewidth_spin.setRange(0.0, 10.0)
        self.linewidth_spin.setSingleStep(0.5)
        self.linewidth_spin.setValue(annotation.get('linewidth', 1.5))

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setValue(annotation.get('alpha', 0.3))

        layout.addRow("X (anchor):", self.x_spin)
        layout.addRow("Y (anchor):", self.y_spin)
        layout.addRow("Width:", self.width_spin)
        layout.addRow("Height:", self.height_spin)
        layout.addRow("Face Color:", self.facecolor_button)
        layout.addRow("Edge Color:", self.edgecolor_button)
        layout.addRow("Line Width:", self.linewidth_spin)
        layout.addRow("Transparency:", self.alpha_spin)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

        self._connect_live(self.x_spin, self.y_spin, self.width_spin, self.height_spin,
                           self.linewidth_spin, self.alpha_spin)

    def _pick_facecolor(self):
        current_color = QColor(self.facecolor_button.text())
        color = QColorDialog.getColor(current_color, self, "Select Face Color")
        if color.isValid():
            self.facecolor_button.setStyleSheet(f"background-color: {color.name()};")
            self.facecolor_button.setText(color.name())
            self._schedule_preview()

    def _pick_edgecolor(self):
        current_color = QColor(self.edgecolor_button.text())
        color = QColorDialog.getColor(current_color, self, "Select Edge Color")
        if color.isValid():
            self.edgecolor_button.setStyleSheet(f"background-color: {color.name()};")
            self.edgecolor_button.setText(color.name())
            self._schedule_preview()

    def get_properties(self):
        return {
            'x': self.x_spin.value(), 'y': self.y_spin.value(),
            'width': self.width_spin.value(), 'height': self.height_spin.value(),
            'facecolor': self.facecolor_button.text(),
            'edgecolor': self.edgecolor_button.text(),
            'linewidth': self.linewidth_spin.value(),
            'alpha': self.alpha_spin.value(),
        }


class EditCalloutDialog(LivePreviewAnnotationDialog):
    """Dialog for editing callout annotations (text + arrow to a point)."""

    def __init__(self, annotation, graph_widget=None, parent=None):
        super().__init__(annotation, graph_widget, parent)
        self.setWindowTitle("Edit Callout Annotation")
        self.resize(380, 380)

        layout = QFormLayout(self)

        self.text_edit = QLineEdit()
        self.text_edit.setText(annotation.get('text', 'Text'))

        self.x_spin = QDoubleSpinBox()
        self.y_spin = QDoubleSpinBox()
        self.tx_spin = QDoubleSpinBox()
        self.ty_spin = QDoubleSpinBox()
        ranges = self._axis_ranges()
        for spin, key, axis in ((self.x_spin, 'x', 'x'), (self.y_spin, 'y', 'y'),
                                (self.tx_spin, 'tx', 'x'), (self.ty_spin, 'ty', 'y')):
            if ranges is not None:
                lo, hi = ranges[0] if axis == 'x' else ranges[1]
                self._configure_coord_spinbox(spin, lo, hi)
            else:
                spin.setRange(-999999, 999999)
                spin.setDecimals(3)
            spin.setValue(annotation.get(key, 0.0))

        self.fontsize_spin = QSpinBox()
        self.fontsize_spin.setRange(6, 72)
        self.fontsize_spin.setValue(annotation.get('fontsize', 11))

        self.text_color_button = QPushButton()
        current_color = QColor(annotation.get('color', 'black'))
        self.text_color_button.setStyleSheet(f"background-color: {current_color.name()};")
        self.text_color_button.setText(current_color.name())
        self.text_color_button.clicked.connect(self._pick_text_color)

        self.arrow_color_button = QPushButton()
        current_arrow = QColor(annotation.get('arrowcolor', 'black'))
        self.arrow_color_button.setStyleSheet(f"background-color: {current_arrow.name()};")
        self.arrow_color_button.setText(current_arrow.name())
        self.arrow_color_button.clicked.connect(self._pick_arrow_color)

        layout.addRow("Text:", self.text_edit)
        layout.addRow("Point X:", self.x_spin)
        layout.addRow("Point Y:", self.y_spin)
        layout.addRow("Text X:", self.tx_spin)
        layout.addRow("Text Y:", self.ty_spin)
        layout.addRow("Font Size:", self.fontsize_spin)
        layout.addRow("Text Color:", self.text_color_button)
        layout.addRow("Arrow Color:", self.arrow_color_button)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

        self._connect_live(self.text_edit, self.x_spin, self.y_spin,
                           self.tx_spin, self.ty_spin, self.fontsize_spin)

    def _pick_text_color(self):
        current_color = QColor(self.text_color_button.text())
        color = QColorDialog.getColor(current_color, self, "Select Text Color")
        if color.isValid():
            self.text_color_button.setStyleSheet(f"background-color: {color.name()};")
            self.text_color_button.setText(color.name())
            self._schedule_preview()

    def _pick_arrow_color(self):
        current_color = QColor(self.arrow_color_button.text())
        color = QColorDialog.getColor(current_color, self, "Select Arrow Color")
        if color.isValid():
            self.arrow_color_button.setStyleSheet(f"background-color: {color.name()};")
            self.arrow_color_button.setText(color.name())
            self._schedule_preview()

    def get_properties(self):
        return {
            'text': self.text_edit.text(),
            'x': self.x_spin.value(), 'y': self.y_spin.value(),
            'tx': self.tx_spin.value(), 'ty': self.ty_spin.value(),
            'fontsize': self.fontsize_spin.value(),
            'color': self.text_color_button.text(),
            'arrowcolor': self.arrow_color_button.text(),
        }


class ColorDelegate(QStyledItemDelegate):
    """Show color in background of color selector comboboxes."""

    def paint(self, painter, option, index):
        painter.save()
        color = index.data(Qt.BackgroundRole)
        if color:
            painter.fillRect(option.rect, color)
        painter.drawText(option.rect, Qt.AlignCenter, index.data(Qt.DisplayRole))
        painter.restore()

    def sizeHint(self, option, index):
        return QSize(70, 20)
