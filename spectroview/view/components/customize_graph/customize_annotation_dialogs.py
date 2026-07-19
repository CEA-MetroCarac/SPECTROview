"""Small standalone dialogs/delegates used by the Customize Graph dialog's
Annotations and Legend tabs -- modal editors for a single vline/hline/text
annotation, plus the color-swatch item delegate the Legend tab's color
comboboxes use.

Split out of customize_graph_dialog.py (which had grown to ~1700 lines
across 8 largely independent classes) purely to shrink that file; no
behavior changes.
"""
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QCheckBox, QFormLayout, QDialogButtonBox, QStyledItemDelegate, QColorDialog,
)


class EditLineDialog(QDialog):
    """Dialog for editing line annotations (vline/hline)."""

    def __init__(self, annotation, parent=None):
        super().__init__(parent)
        self.annotation = annotation
        self.setWindowTitle("Edit Line Annotation")
        self.resize(350, 200)

        layout = QFormLayout(self)

        # Color picker
        self.color_button = QPushButton()
        current_color = QColor(annotation.get('color', 'red'))
        self.color_button.setStyleSheet(f"background-color: {current_color.name()};")
        self.color_button.setText(current_color.name())
        self.color_button.clicked.connect(self._pick_color)

        # Line style
        self.linestyle_combo = QComboBox()
        self.linestyle_combo.addItem("Solid", "-")
        self.linestyle_combo.addItem("Dashed", "--")
        self.linestyle_combo.addItem("Dotted", ":")
        self.linestyle_combo.addItem("Dash-Dot", "-.")

        current_style = annotation.get('linestyle', '--')
        index = self.linestyle_combo.findData(current_style)
        if index >= 0:
            self.linestyle_combo.setCurrentIndex(index)

        # Line width
        self.linewidth_spin = QDoubleSpinBox()
        self.linewidth_spin.setRange(0.5, 5.0)
        self.linewidth_spin.setSingleStep(0.5)
        self.linewidth_spin.setValue(annotation.get('linewidth', 1.5))

        layout.addRow("Color:", self.color_button)
        layout.addRow("Line Style:", self.linestyle_combo)
        layout.addRow("Line Width:", self.linewidth_spin)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

    def _pick_color(self):
        """Open color picker dialog."""
        current_color = QColor(self.color_button.text())
        color = QColorDialog.getColor(current_color, self, "Select Line Color")
        if color.isValid():
            self.color_button.setStyleSheet(f"background-color: {color.name()};")
            self.color_button.setText(color.name())

    def get_properties(self):
        """Return updated properties."""
        return {
            'color': self.color_button.text(),
            'linestyle': self.linestyle_combo.currentData(),
            'linewidth': self.linewidth_spin.value()
        }


class EditTextDialog(QDialog):
    """Dialog for editing text annotations."""

    def __init__(self, annotation, parent=None):
        super().__init__(parent)
        self.annotation = annotation
        self.setWindowTitle("Edit Text Annotation")
        self.resize(400, 350)

        layout = QFormLayout(self)

        # Text content
        self.text_edit = QLineEdit()
        self.text_edit.setText(annotation.get('text', 'Text'))

        # Font size
        self.fontsize_spin = QSpinBox()
        self.fontsize_spin.setRange(6, 72)
        self.fontsize_spin.setValue(annotation.get('fontsize', 12))

        # Text color
        self.text_color_button = QPushButton()
        current_color = QColor(annotation.get('color', 'black'))
        self.text_color_button.setStyleSheet(f"background-color: {current_color.name()};")
        self.text_color_button.setText(current_color.name())
        self.text_color_button.clicked.connect(self._pick_text_color)

        # Background/frame options
        self.frame_checkbox = QCheckBox("Show frame/box")
        bbox = annotation.get('bbox')
        self.frame_checkbox.setChecked(bbox is not None)

        # Background color picker button (similar to text color)
        self.bg_color_button = QPushButton()
        if isinstance(bbox, dict) and bbox.get('facecolor'):
            bg_color = QColor(bbox.get('facecolor'))
        else:
            bg_color = QColor('yellow')  # Default background color
        self.bg_color_button.setStyleSheet(f"background-color: {bg_color.name()};")
        self.bg_color_button.setText(bg_color.name())
        self.bg_color_button.clicked.connect(self._pick_bg_color)

        # Transparency slider (0-100%)
        self.transparency_slider = QSpinBox()
        self.transparency_slider.setRange(0, 100)
        self.transparency_slider.setSingleStep(10)
        self.transparency_slider.setSuffix("%")

        # Get current alpha from bbox, default to 70%
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

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

    def _pick_text_color(self):
        """Open color picker for text color."""
        current_color = QColor(self.text_color_button.text())
        color = QColorDialog.getColor(current_color, self, "Select Text Color")
        if color.isValid():
            self.text_color_button.setStyleSheet(f"background-color: {color.name()};")
            self.text_color_button.setText(color.name())

    def _pick_bg_color(self):
        """Open color picker for background color."""
        current_color = QColor(self.bg_color_button.text())
        color = QColorDialog.getColor(current_color, self, "Select Background Color")
        if color.isValid():
            self.bg_color_button.setStyleSheet(f"background-color: {color.name()};")
            self.bg_color_button.setText(color.name())

    def get_properties(self):
        """Return updated properties."""
        props = {
            'text': self.text_edit.text(),
            'fontsize': self.fontsize_spin.value(),
            'color': self.text_color_button.text(),
            'ha': 'center',  # Always use center alignment
            'va': 'center'   # Always use center alignment
        }

        # Handle bbox
        if self.frame_checkbox.isChecked():
            facecolor = self.bg_color_button.text()

            # Get transparency from slider (convert percentage to 0-1 range)
            alpha = self.transparency_slider.value() / 100.0

            props['bbox'] = {
                'facecolor': facecolor,
                'edgecolor': 'black',
                'boxstyle': 'round,pad=0.3',
                'alpha': alpha
            }
        else:
            props['bbox'] = None

        return props


class EditArrowDialog(QDialog):
    """Dialog for editing arrow annotations."""

    def __init__(self, annotation, parent=None):
        super().__init__(parent)
        self.annotation = annotation
        self.setWindowTitle("Edit Arrow Annotation")
        self.resize(350, 300)

        layout = QFormLayout(self)

        self.x1_spin = QDoubleSpinBox()
        self.x1_spin.setRange(-999999, 999999)
        self.x1_spin.setDecimals(3)
        self.x1_spin.setValue(annotation.get('x1', 0.0))

        self.y1_spin = QDoubleSpinBox()
        self.y1_spin.setRange(-999999, 999999)
        self.y1_spin.setDecimals(3)
        self.y1_spin.setValue(annotation.get('y1', 0.0))

        self.x2_spin = QDoubleSpinBox()
        self.x2_spin.setRange(-999999, 999999)
        self.x2_spin.setDecimals(3)
        self.x2_spin.setValue(annotation.get('x2', 0.0))

        self.y2_spin = QDoubleSpinBox()
        self.y2_spin.setRange(-999999, 999999)
        self.y2_spin.setDecimals(3)
        self.y2_spin.setValue(annotation.get('y2', 0.0))

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

    def _pick_color(self):
        current_color = QColor(self.color_button.text())
        color = QColorDialog.getColor(current_color, self, "Select Arrow Color")
        if color.isValid():
            self.color_button.setStyleSheet(f"background-color: {color.name()};")
            self.color_button.setText(color.name())

    def get_properties(self):
        """Return updated properties."""
        return {
            'x1': self.x1_spin.value(), 'y1': self.y1_spin.value(),
            'x2': self.x2_spin.value(), 'y2': self.y2_spin.value(),
            'color': self.color_button.text(),
            'linestyle': self.linestyle_combo.currentData(),
            'linewidth': self.linewidth_spin.value(),
        }


class EditSpanDialog(QDialog):
    """Dialog for editing span annotations (vspan/hspan)."""

    def __init__(self, annotation, parent=None):
        super().__init__(parent)
        self.annotation = annotation
        self.is_vertical = annotation.get('type') == 'vspan'
        self.setWindowTitle("Edit Span Annotation")
        self.resize(350, 220)

        layout = QFormLayout(self)

        axis_label = "X" if self.is_vertical else "Y"
        key1, key2 = ('x1', 'x2') if self.is_vertical else ('y1', 'y2')

        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(-999999, 999999)
        self.start_spin.setDecimals(3)
        self.start_spin.setValue(annotation.get(key1, 0.0))

        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(-999999, 999999)
        self.end_spin.setDecimals(3)
        self.end_spin.setValue(annotation.get(key2, 1.0))

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
        layout.addRow("Color:", self.color_button)
        layout.addRow("Transparency:", self.alpha_spin)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

    def _pick_color(self):
        current_color = QColor(self.color_button.text())
        color = QColorDialog.getColor(current_color, self, "Select Span Color")
        if color.isValid():
            self.color_button.setStyleSheet(f"background-color: {color.name()};")
            self.color_button.setText(color.name())

    def get_properties(self):
        """Return updated properties (keyed x1/x2 for vspan, y1/y2 for hspan)."""
        key1, key2 = ('x1', 'x2') if self.is_vertical else ('y1', 'y2')
        return {
            key1: self.start_spin.value(),
            key2: self.end_spin.value(),
            'color': self.color_button.text(),
            'alpha': self.alpha_spin.value(),
        }


class EditBoxDialog(QDialog):
    """Dialog for editing box (rectangle) annotations."""

    def __init__(self, annotation, parent=None):
        super().__init__(parent)
        self.annotation = annotation
        self.setWindowTitle("Edit Box Annotation")
        self.resize(350, 340)

        layout = QFormLayout(self)

        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(-999999, 999999)
        self.x_spin.setDecimals(3)
        self.x_spin.setValue(annotation.get('x', 0.0))

        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(-999999, 999999)
        self.y_spin.setDecimals(3)
        self.y_spin.setValue(annotation.get('y', 0.0))

        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.001, 999999)
        self.width_spin.setDecimals(3)
        self.width_spin.setValue(annotation.get('width', 1.0))

        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.001, 999999)
        self.height_spin.setDecimals(3)
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

    def _pick_facecolor(self):
        current_color = QColor(self.facecolor_button.text())
        color = QColorDialog.getColor(current_color, self, "Select Face Color")
        if color.isValid():
            self.facecolor_button.setStyleSheet(f"background-color: {color.name()};")
            self.facecolor_button.setText(color.name())

    def _pick_edgecolor(self):
        current_color = QColor(self.edgecolor_button.text())
        color = QColorDialog.getColor(current_color, self, "Select Edge Color")
        if color.isValid():
            self.edgecolor_button.setStyleSheet(f"background-color: {color.name()};")
            self.edgecolor_button.setText(color.name())

    def get_properties(self):
        """Return updated properties."""
        return {
            'x': self.x_spin.value(), 'y': self.y_spin.value(),
            'width': self.width_spin.value(), 'height': self.height_spin.value(),
            'facecolor': self.facecolor_button.text(),
            'edgecolor': self.edgecolor_button.text(),
            'linewidth': self.linewidth_spin.value(),
            'alpha': self.alpha_spin.value(),
        }


class EditCalloutDialog(QDialog):
    """Dialog for editing callout annotations (text + arrow to a point)."""

    def __init__(self, annotation, parent=None):
        super().__init__(parent)
        self.annotation = annotation
        self.setWindowTitle("Edit Callout Annotation")
        self.resize(380, 380)

        layout = QFormLayout(self)

        self.text_edit = QLineEdit()
        self.text_edit.setText(annotation.get('text', 'Text'))

        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(-999999, 999999)
        self.x_spin.setDecimals(3)
        self.x_spin.setValue(annotation.get('x', 0.0))

        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(-999999, 999999)
        self.y_spin.setDecimals(3)
        self.y_spin.setValue(annotation.get('y', 0.0))

        self.tx_spin = QDoubleSpinBox()
        self.tx_spin.setRange(-999999, 999999)
        self.tx_spin.setDecimals(3)
        self.tx_spin.setValue(annotation.get('tx', 0.0))

        self.ty_spin = QDoubleSpinBox()
        self.ty_spin.setRange(-999999, 999999)
        self.ty_spin.setDecimals(3)
        self.ty_spin.setValue(annotation.get('ty', 0.0))

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

    def _pick_text_color(self):
        current_color = QColor(self.text_color_button.text())
        color = QColorDialog.getColor(current_color, self, "Select Text Color")
        if color.isValid():
            self.text_color_button.setStyleSheet(f"background-color: {color.name()};")
            self.text_color_button.setText(color.name())

    def _pick_arrow_color(self):
        current_color = QColor(self.arrow_color_button.text())
        color = QColorDialog.getColor(current_color, self, "Select Arrow Color")
        if color.isValid():
            self.arrow_color_button.setStyleSheet(f"background-color: {color.name()};")
            self.arrow_color_button.setText(color.name())

    def get_properties(self):
        """Return updated properties."""
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
