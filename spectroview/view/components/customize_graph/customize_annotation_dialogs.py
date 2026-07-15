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
