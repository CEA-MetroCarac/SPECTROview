"""Annotations tab of the Customize Graph dialog: add/edit/delete vline,
hline, and text annotations on a plot.

Split out of customize_graph_dialog.py; no behavior changes.
"""
import time
import os

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QLabel, QMessageBox, QDialog,
)

from spectroview import ICON_DIR
from spectroview.view.components.customize_annotation_dialogs import EditLineDialog, EditTextDialog


class CustomizeAnnotations(QWidget):
    """Widget for customizing graph annotations (vline, hline, text)."""

    def __init__(self, graph_widget, parent=None):
        super().__init__(parent)
        self.graph_widget = graph_widget

        self._setup_ui()

        # Connect to annotation position changed signal to update list when dragging
        self.graph_widget.annotation_position_changed.connect(self._on_annotation_dragged)

        # Load initial annotations
        self.load_annotations()

    def switch_graph(self, graph_widget):
        """Switch to a different graph widget and reload annotations."""
        # Disconnect from old graph's signal
        try:
            self.graph_widget.annotation_position_changed.disconnect(self._on_annotation_dragged)
        except (RuntimeError, TypeError):
            pass  # Already disconnected or never connected

        self.graph_widget = graph_widget

        # Connect to new graph's signal
        self.graph_widget.annotation_position_changed.connect(self._on_annotation_dragged)

        # Reload annotations for the new graph
        self.load_annotations()

    def _setup_ui(self):
        """Setup the UI components for the annotations widget."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Add buttons
        btn_layout = QHBoxLayout()

        self.btn_add_vline = QPushButton("V-Line")
        self.btn_add_vline.setIcon(QIcon(os.path.join(ICON_DIR, "add_color.png")))
        self.btn_add_vline.setIconSize(QSize(16, 16))

        self.btn_add_hline = QPushButton("H-Line")
        self.btn_add_hline.setIcon(QIcon(os.path.join(ICON_DIR, "add_color.png")))
        self.btn_add_hline.setIconSize(QSize(16, 16))

        self.btn_add_text = QPushButton("Text")
        self.btn_add_text.setIcon(QIcon(os.path.join(ICON_DIR, "add_color.png")))
        self.btn_add_text.setIconSize(QSize(16, 16))

        btn_layout.addWidget(self.btn_add_vline)
        btn_layout.addWidget(self.btn_add_hline)
        btn_layout.addWidget(self.btn_add_text)

        # Annotation list
        self.annotation_list = QListWidget()

        # Edit and Delete buttons
        mgmt_layout = QHBoxLayout()
        self.btn_edit = QPushButton("Edit")
        self.btn_edit.setIcon(QIcon(os.path.join(ICON_DIR, "edit.png")))
        self.btn_edit.setIconSize(QSize(16, 16))

        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setIcon(QIcon(os.path.join(ICON_DIR, "trash.png")))
        self.btn_delete.setIconSize(QSize(16, 16))

        mgmt_layout.addWidget(self.btn_edit)
        mgmt_layout.addWidget(self.btn_delete)

        layout.addLayout(btn_layout)
        layout.addWidget(QLabel("Current Annotations:"))
        layout.addWidget(self.annotation_list)
        layout.addLayout(mgmt_layout)

        # Connect signals
        self.btn_add_vline.clicked.connect(self._add_vline)
        self.btn_add_hline.clicked.connect(self._add_hline)
        self.btn_add_text.clicked.connect(self._add_text)
        self.btn_edit.clicked.connect(self._edit_annotation)
        self.btn_delete.clicked.connect(self._delete_annotation)

    def _on_annotation_dragged(self, graph_id, ann_id, new_x, new_y):
        """Handle annotation position change from dragging - refresh the list widget."""
        if graph_id == self.graph_widget.graph_id:
            self.load_annotations()

    def _get_plot_center(self):
        """Get center coordinates of the plot."""
        ax = self.graph_widget.ax
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        center_x = (xlim[0] + xlim[1]) / 2
        center_y = (ylim[0] + ylim[1]) / 2
        return center_x, center_y

    def _notify_annotations_changed(self):
        """Push the current annotation list to the ViewModel."""
        self.graph_widget.properties_changed.emit(
            self.graph_widget.graph_id, {'annotations': self.graph_widget.annotations}
        )

    def _add_vline(self):
        """Add vertical line at plot center."""
        center_x, _ = self._get_plot_center()

        ann_id = f"vline_{int(time.time() * 1000000)}"
        annotation = {
            'id': ann_id,
            'type': 'vline',
            'x': center_x,
            'color': 'red',
            'linestyle': '--',
            'linewidth': 1.5,
            'label': f'V-Line at x={center_x:.2f}'
        }

        self.graph_widget.annotations.append(annotation)
        self._notify_annotations_changed()
        self._refresh_plot()
        self.load_annotations()

    def _add_hline(self):
        """Add horizontal line at plot center."""
        _, center_y = self._get_plot_center()

        ann_id = f"hline_{int(time.time() * 1000000)}"
        annotation = {
            'id': ann_id,
            'type': 'hline',
            'y': center_y,
            'color': 'blue',
            'linestyle': '--',
            'linewidth': 1.5,
            'label': f'H-Line at y={center_y:.2f}'
        }

        self.graph_widget.annotations.append(annotation)
        self._notify_annotations_changed()
        self._refresh_plot()
        self.load_annotations()

    def _add_text(self):
        """Add text annotation at plot center."""
        center_x, center_y = self._get_plot_center()

        ann_id = f"text_{int(time.time() * 1000000)}"
        annotation = {
            'id': ann_id,
            'type': 'text',
            'x': center_x,
            'y': center_y,
            'text': 'Text',
            'fontsize': 11,
            'color': 'black',
            'ha': 'center',
            'va': 'center',
            'bbox': {
                'facecolor': 'yellow',
                'edgecolor': 'black',
                'boxstyle': 'round,pad=0.3',
                'alpha': 0.7
            }
        }

        self.graph_widget.annotations.append(annotation)

        self._notify_annotations_changed()
        self._refresh_plot()
        self.load_annotations()

    def _edit_annotation(self):
        """Edit selected annotation."""
        selected = self.annotation_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select an annotation to edit.")
            return

        ann_id = selected.data(Qt.UserRole)

        # Find the annotation
        annotation = None
        for ann in self.graph_widget.annotations:
            if ann.get('id') == ann_id:
                annotation = ann
                break

        if not annotation:
            return

        # Open appropriate edit dialog based on type
        if annotation['type'] in ['vline', 'hline']:
            dialog = EditLineDialog(annotation, self)
            if dialog.exec() == QDialog.Accepted:
                # Update annotation properties
                props = dialog.get_properties()
                annotation.update(props)

                # Update label
                if annotation['type'] == 'vline':
                    annotation['label'] = f"V-Line at x={annotation['x']:.2f}"
                else:
                    annotation['label'] = f"H-Line at y={annotation['y']:.2f}"

                self._notify_annotations_changed()
                self._refresh_plot()
                self.load_annotations()

        elif annotation['type'] == 'text':
            dialog = EditTextDialog(annotation, self)
            if dialog.exec() == QDialog.Accepted:
                # Update annotation properties
                props = dialog.get_properties()
                annotation.update(props)

                self._notify_annotations_changed()
                self._refresh_plot()
                self.load_annotations()

    def _delete_annotation(self):
        """Delete selected annotation."""
        selected = self.annotation_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select an annotation to delete.")
            return

        ann_id = selected.data(Qt.UserRole)
        self.graph_widget.annotations = [
            ann for ann in self.graph_widget.annotations
            if ann.get('id') != ann_id
        ]
        self._notify_annotations_changed()
        self._refresh_plot()
        self.load_annotations()

    def _refresh_plot(self):
        """Refresh the plot with updated annotations."""
        self.graph_widget.ax.clear()
        if self.graph_widget.df is not None:
            self.graph_widget.plot(self.graph_widget.df)

    def load_annotations(self):
        """Load annotations into the list widget."""
        self.annotation_list.clear()

        for ann in self.graph_widget.annotations:
            if ann['type'] == 'vline':
                text = f"├ VLine @ x={ann['x']:.2f} ({ann.get('color', 'red')})"
            elif ann['type'] == 'hline':
                text = f"├ HLine @ y={ann['y']:.2f} ({ann.get('color', 'blue')})"
            elif ann['type'] == 'text':
                text = f"└ Text \"{ann['text'][:20]}...\" @ ({ann['x']:.1f},{ann['y']:.1f})"
            else:
                text = f"Unknown type: {ann['type']}"

            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, ann['id'])
            self.annotation_list.addItem(item)
