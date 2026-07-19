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
from spectroview.view.components.customize_graph.customize_annotation_dialogs import (
    EditLineDialog, EditTextDialog, EditArrowDialog, EditSpanDialog,
    EditBoxDialog, EditCalloutDialog,
)


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

        # Add buttons -- two rows so the panel doesn't get too cramped
        btn_layout = QHBoxLayout()
        btn_layout2 = QHBoxLayout()

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

        self.btn_add_arrow = QPushButton("Arrow")
        self.btn_add_arrow.setIcon(QIcon(os.path.join(ICON_DIR, "add_color.png")))
        self.btn_add_arrow.setIconSize(QSize(16, 16))

        self.btn_add_vspan = QPushButton("V-Span")
        self.btn_add_vspan.setIcon(QIcon(os.path.join(ICON_DIR, "add_color.png")))
        self.btn_add_vspan.setIconSize(QSize(16, 16))

        self.btn_add_hspan = QPushButton("H-Span")
        self.btn_add_hspan.setIcon(QIcon(os.path.join(ICON_DIR, "add_color.png")))
        self.btn_add_hspan.setIconSize(QSize(16, 16))

        self.btn_add_box = QPushButton("Box")
        self.btn_add_box.setIcon(QIcon(os.path.join(ICON_DIR, "add_color.png")))
        self.btn_add_box.setIconSize(QSize(16, 16))

        self.btn_add_callout = QPushButton("Callout")
        self.btn_add_callout.setIcon(QIcon(os.path.join(ICON_DIR, "add_color.png")))
        self.btn_add_callout.setIconSize(QSize(16, 16))

        btn_layout2.addWidget(self.btn_add_arrow)
        btn_layout2.addWidget(self.btn_add_vspan)
        btn_layout2.addWidget(self.btn_add_hspan)
        btn_layout2.addWidget(self.btn_add_box)
        btn_layout2.addWidget(self.btn_add_callout)

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
        layout.addLayout(btn_layout2)
        layout.addWidget(QLabel("Current Annotations:"))
        layout.addWidget(self.annotation_list)
        layout.addLayout(mgmt_layout)

        # Connect signals
        self.btn_add_vline.clicked.connect(self._add_vline)
        self.btn_add_hline.clicked.connect(self._add_hline)
        self.btn_add_text.clicked.connect(self._add_text)
        self.btn_add_arrow.clicked.connect(self._add_arrow)
        self.btn_add_vspan.clicked.connect(self._add_vspan)
        self.btn_add_hspan.clicked.connect(self._add_hspan)
        self.btn_add_box.clicked.connect(self._add_box)
        self.btn_add_callout.clicked.connect(self._add_callout)
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

    def _get_plot_extent(self):
        """Get (x_range, y_range) of the current plot view -- used to size
        new span/box/arrow/callout defaults proportionally to the data
        instead of a fixed data-unit constant that could be huge or tiny
        depending on what's plotted."""
        ax = self.graph_widget.ax
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        return xlim[1] - xlim[0], ylim[1] - ylim[0]

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

    def _add_arrow(self):
        """Add a short diagonal arrow centered on the plot view."""
        center_x, center_y = self._get_plot_center()
        x_range, y_range = self._get_plot_extent()

        ann_id = f"arrow_{int(time.time() * 1000000)}"
        annotation = {
            'id': ann_id,
            'type': 'arrow',
            'x1': center_x - x_range * 0.1, 'y1': center_y - y_range * 0.1,
            'x2': center_x + x_range * 0.1, 'y2': center_y + y_range * 0.1,
            'color': 'black', 'linewidth': 1.5, 'linestyle': '-',
        }

        self.graph_widget.annotations.append(annotation)
        self._notify_annotations_changed()
        self._refresh_plot()
        self.load_annotations()

    def _add_vspan(self):
        """Add a vertical shaded span, 20% of the X range, centered."""
        center_x, _ = self._get_plot_center()
        x_range, _ = self._get_plot_extent()

        ann_id = f"vspan_{int(time.time() * 1000000)}"
        annotation = {
            'id': ann_id,
            'type': 'vspan',
            'x1': center_x - x_range * 0.1, 'x2': center_x + x_range * 0.1,
            'color': 'orange', 'alpha': 0.3,
        }

        self.graph_widget.annotations.append(annotation)
        self._notify_annotations_changed()
        self._refresh_plot()
        self.load_annotations()

    def _add_hspan(self):
        """Add a horizontal shaded span, 20% of the Y range, centered."""
        _, center_y = self._get_plot_center()
        _, y_range = self._get_plot_extent()

        ann_id = f"hspan_{int(time.time() * 1000000)}"
        annotation = {
            'id': ann_id,
            'type': 'hspan',
            'y1': center_y - y_range * 0.1, 'y2': center_y + y_range * 0.1,
            'color': 'orange', 'alpha': 0.3,
        }

        self.graph_widget.annotations.append(annotation)
        self._notify_annotations_changed()
        self._refresh_plot()
        self.load_annotations()

    def _add_box(self):
        """Add a rectangle box, 20%x20% of the plot view, centered."""
        center_x, center_y = self._get_plot_center()
        x_range, y_range = self._get_plot_extent()
        width, height = x_range * 0.2, y_range * 0.2

        ann_id = f"box_{int(time.time() * 1000000)}"
        annotation = {
            'id': ann_id,
            'type': 'box',
            'x': center_x - width / 2, 'y': center_y - height / 2,
            'width': width, 'height': height,
            'facecolor': 'yellow', 'edgecolor': 'black', 'linewidth': 1.5, 'alpha': 0.3,
        }

        self.graph_widget.annotations.append(annotation)
        self._notify_annotations_changed()
        self._refresh_plot()
        self.load_annotations()

    def _add_callout(self):
        """Add a callout: text offset up-right from a point at plot center,
        connected by an arrow."""
        center_x, center_y = self._get_plot_center()
        x_range, y_range = self._get_plot_extent()

        ann_id = f"callout_{int(time.time() * 1000000)}"
        annotation = {
            'id': ann_id,
            'type': 'callout',
            'x': center_x, 'y': center_y,
            'tx': center_x + x_range * 0.15, 'ty': center_y + y_range * 0.15,
            'text': 'Callout', 'fontsize': 11, 'color': 'black', 'arrowcolor': 'black',
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

        elif annotation['type'] == 'arrow':
            dialog = EditArrowDialog(annotation, self)
            if dialog.exec() == QDialog.Accepted:
                annotation.update(dialog.get_properties())
                self._notify_annotations_changed()
                self._refresh_plot()
                self.load_annotations()

        elif annotation['type'] in ('vspan', 'hspan'):
            dialog = EditSpanDialog(annotation, self)
            if dialog.exec() == QDialog.Accepted:
                annotation.update(dialog.get_properties())
                self._notify_annotations_changed()
                self._refresh_plot()
                self.load_annotations()

        elif annotation['type'] == 'box':
            dialog = EditBoxDialog(annotation, self)
            if dialog.exec() == QDialog.Accepted:
                annotation.update(dialog.get_properties())
                self._notify_annotations_changed()
                self._refresh_plot()
                self.load_annotations()

        elif annotation['type'] == 'callout':
            dialog = EditCalloutDialog(annotation, self)
            if dialog.exec() == QDialog.Accepted:
                annotation.update(dialog.get_properties())
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
            elif ann['type'] == 'arrow':
                text = (f"├ Arrow ({ann['x1']:.1f},{ann['y1']:.1f}) → "
                        f"({ann['x2']:.1f},{ann['y2']:.1f}) ({ann.get('color', 'black')})")
            elif ann['type'] == 'vspan':
                text = f"├ V-Span x=[{ann['x1']:.2f}, {ann['x2']:.2f}] ({ann.get('color', 'orange')})"
            elif ann['type'] == 'hspan':
                text = f"├ H-Span y=[{ann['y1']:.2f}, {ann['y2']:.2f}] ({ann.get('color', 'orange')})"
            elif ann['type'] == 'box':
                text = (f"├ Box @ ({ann['x']:.1f},{ann['y']:.1f}) "
                        f"{ann['width']:.2f}×{ann['height']:.2f} ({ann.get('facecolor', 'yellow')})")
            elif ann['type'] == 'callout':
                text = f"└ Callout \"{ann['text'][:20]}\" @ ({ann['x']:.1f},{ann['y']:.1f})"
            else:
                text = f"Unknown type: {ann['type']}"

            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, ann['id'])
            self.annotation_list.addItem(item)
