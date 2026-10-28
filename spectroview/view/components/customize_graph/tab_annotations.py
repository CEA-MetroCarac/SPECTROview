# spectroview/view/components/customize_graph/tab_annotations.py
"""Annotations tab: add, edit, and delete plot annotations (lines, text, arrows, spans, boxes, callouts)."""

from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from spectroview.view.components.customize_graph.common import (
    create_scrollable_container,
    highlight_group_style,
)
from spectroview.view.components.customize_graph.customize_annotation_dialogs import (
    EditArrowDialog,
    EditBoxDialog,
    EditCalloutDialog,
    EditLineDialog,
    EditSpanDialog,
    EditTextDialog,
)
from spectroview.view.theme import Icons


class TabAnnotations(QWidget):
    """Category 7: Annotations."""

    KEYWORDS = [
        "annotation", "annotations", "vline", "hline", "vertical line",
        "horizontal line", "text", "label", "arrow", "span", "vspan",
        "hspan", "box", "rectangle", "callout", "shapes", "draw",
        "edit", "delete", "drag",
    ]

    _EDIT_DIALOGS = {
        "vline": EditLineDialog, "hline": EditLineDialog,
        "text": EditTextDialog, "arrow": EditArrowDialog,
        "vspan": EditSpanDialog, "hspan": EditSpanDialog,
        "box": EditBoxDialog, "callout": EditCalloutDialog,
    }

    def __init__(self, graph_widget: QWidget, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.graph_widget = graph_widget
        self._setup_ui()

        # Connect position changed signal for interactive dragging
        if hasattr(self.graph_widget, "annotation_position_changed"):
            self.graph_widget.annotation_position_changed.connect(self._on_annotation_dragged)

        self.load_settings()

    def switch_graph(self, graph_widget: QWidget) -> None:
        """Switch to a new graph widget and reload annotations."""
        if hasattr(self.graph_widget, "annotation_position_changed"):
            try:
                self.graph_widget.annotation_position_changed.disconnect(self._on_annotation_dragged)
            except (RuntimeError, TypeError):
                pass

        self.graph_widget = graph_widget
        if hasattr(self.graph_widget, "annotation_position_changed"):
            self.graph_widget.annotation_position_changed.connect(self._on_annotation_dragged)

        self.load_settings()

    def _setup_ui(self) -> None:
        self._scroll, _container, layout = create_scrollable_container(self)

        # ───── Add Annotation Tools ─────
        self.grp_add = QGroupBox("Add New Annotation:")
        add_layout = QVBoxLayout(self.grp_add)
        add_layout.setContentsMargins(8, 8, 8, 8)
        add_layout.setSpacing(8)

        row1 = QHBoxLayout()
        self.btn_add_vline = QPushButton("V-Line")
        self.btn_add_vline.setIcon(Icons.ann_vline())
        self.btn_add_vline.setIconSize(QSize(16, 16))

        self.btn_add_hline = QPushButton("H-Line")
        self.btn_add_hline.setIcon(Icons.ann_hline())
        self.btn_add_hline.setIconSize(QSize(16, 16))

        self.btn_add_text = QPushButton("Text")
        self.btn_add_text.setIcon(Icons.ann_text())
        self.btn_add_text.setIconSize(QSize(16, 16))

        self.btn_add_arrow = QPushButton("Arrow")
        self.btn_add_arrow.setIcon(Icons.ann_arrow())
        self.btn_add_arrow.setIconSize(QSize(16, 16))

        row1.addWidget(self.btn_add_vline)
        row1.addWidget(self.btn_add_hline)
        row1.addWidget(self.btn_add_text)
        row1.addWidget(self.btn_add_arrow)
        add_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_add_vspan = QPushButton("V-Span")
        self.btn_add_vspan.setIcon(Icons.ann_vspan())
        self.btn_add_vspan.setIconSize(QSize(16, 16))

        self.btn_add_hspan = QPushButton("H-Span")
        self.btn_add_hspan.setIcon(Icons.ann_hspan())
        self.btn_add_hspan.setIconSize(QSize(16, 16))

        self.btn_add_box = QPushButton("Box")
        self.btn_add_box.setIcon(Icons.ann_box())
        self.btn_add_box.setIconSize(QSize(16, 16))

        self.btn_add_callout = QPushButton("Callout")
        self.btn_add_callout.setIcon(Icons.ann_callout())
        self.btn_add_callout.setIconSize(QSize(16, 16))

        row2.addWidget(self.btn_add_vspan)
        row2.addWidget(self.btn_add_hspan)
        row2.addWidget(self.btn_add_box)
        row2.addWidget(self.btn_add_callout)
        add_layout.addLayout(row2)

        layout.addWidget(self.grp_add)

        # ───── Current Annotations List & Management ─────
        self.grp_list = QGroupBox("Active Annotations:")
        list_layout = QVBoxLayout(self.grp_list)
        list_layout.setContentsMargins(8, 8, 8, 8)
        list_layout.setSpacing(8)

        self.annotation_list = QListWidget()
        self.annotation_list.setMinimumHeight(180)
        self.annotation_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        list_layout.addWidget(self.annotation_list)

        mgmt_row = QHBoxLayout()
        self.btn_edit = QPushButton("Edit Annotation")
        self.btn_edit.setIcon(Icons.edit())
        self.btn_edit.setIconSize(QSize(16, 16))
        self.btn_edit.clicked.connect(self._edit_annotation)

        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setIcon(Icons.trash())
        self.btn_delete.setIconSize(QSize(16, 16))
        self.btn_delete.clicked.connect(self._delete_annotation)

        mgmt_row.addWidget(self.btn_edit)
        mgmt_row.addWidget(self.btn_delete)
        mgmt_row.addStretch()
        list_layout.addLayout(mgmt_row)

        layout.addWidget(self.grp_list)
        layout.addStretch()

        # Connect add button actions
        self.btn_add_vline.clicked.connect(self._add_vline)
        self.btn_add_hline.clicked.connect(self._add_hline)
        self.btn_add_text.clicked.connect(self._add_text)
        self.btn_add_arrow.clicked.connect(self._add_arrow)
        self.btn_add_vspan.clicked.connect(self._add_vspan)
        self.btn_add_hspan.clicked.connect(self._add_hspan)
        self.btn_add_box.clicked.connect(self._add_box)
        self.btn_add_callout.clicked.connect(self._add_callout)

    def _on_annotation_dragged(self, graph_id: int, _ann_id: str, _new_x: float, _new_y: float) -> None:
        if hasattr(self.graph_widget, "graph_id") and graph_id == self.graph_widget.graph_id:
            self.load_settings()

    def _get_plot_center(self) -> tuple[float, float]:
        ax = getattr(self.graph_widget, "ax", None)
        if ax is not None:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            return (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2
        return 50.0, 50.0

    def _get_plot_extent(self) -> tuple[float, float]:
        ax = getattr(self.graph_widget, "ax", None)
        if ax is not None:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            return xlim[1] - xlim[0], ylim[1] - ylim[0]
        return 100.0, 100.0

    def _notify_annotations_changed(self) -> None:
        if hasattr(self.graph_widget, "properties_changed") and hasattr(self.graph_widget, "graph_id"):
            self.graph_widget.properties_changed.emit(
                self.graph_widget.graph_id, {"annotations": getattr(self.graph_widget, "annotations", [])}
            )

    def _refresh_plot(self) -> None:
        if hasattr(self.graph_widget, "ax") and self.graph_widget.ax is not None:
            self.graph_widget.ax.clear()
        if getattr(self.graph_widget, "df", None) is not None:
            self.graph_widget.plot(self.graph_widget.df)

    def _commit_new_annotation(self, annotation: dict) -> None:
        if not hasattr(self.graph_widget, "annotations") or self.graph_widget.annotations is None:
            self.graph_widget.annotations = []
        annotation["id"] = f"{annotation['type']}_{int(time.time() * 1000000)}"
        self.graph_widget.annotations.append(annotation)
        self._notify_annotations_changed()
        self._refresh_plot()
        self.load_settings()

    def _add_vline(self) -> None:
        center_x, _ = self._get_plot_center()
        self._commit_new_annotation({
            "type": "vline", "x": center_x,
            "color": "red", "linestyle": "--", "linewidth": 1.5,
            "label": f"V-Line at x={center_x:.2f}",
        })

    def _add_hline(self) -> None:
        _, center_y = self._get_plot_center()
        self._commit_new_annotation({
            "type": "hline", "y": center_y,
            "color": "blue", "linestyle": "--", "linewidth": 1.5,
            "label": f"H-Line at y={center_y:.2f}",
        })

    def _add_text(self) -> None:
        center_x, center_y = self._get_plot_center()
        self._commit_new_annotation({
            "type": "text", "x": center_x, "y": center_y,
            "text": "Text", "fontsize": 11, "color": "black",
            "ha": "center", "va": "center",
            "bbox": {"facecolor": "yellow", "edgecolor": "black", "boxstyle": "round,pad=0.3", "alpha": 0.7},
        })

    def _add_arrow(self) -> None:
        center_x, center_y = self._get_plot_center()
        x_range, y_range = self._get_plot_extent()
        self._commit_new_annotation({
            "type": "arrow",
            "x1": center_x - x_range * 0.1, "y1": center_y - y_range * 0.1,
            "x2": center_x + x_range * 0.1, "y2": center_y + y_range * 0.1,
            "color": "black", "linewidth": 1.5, "linestyle": "-",
        })

    def _add_vspan(self) -> None:
        center_x, _ = self._get_plot_center()
        x_range, _ = self._get_plot_extent()
        self._commit_new_annotation({
            "type": "vspan",
            "x1": center_x - x_range * 0.1, "x2": center_x + x_range * 0.1,
            "color": "orange", "alpha": 0.3,
        })

    def _add_hspan(self) -> None:
        _, center_y = self._get_plot_center()
        _, y_range = self._get_plot_extent()
        self._commit_new_annotation({
            "type": "hspan",
            "y1": center_y - y_range * 0.1, "y2": center_y + y_range * 0.1,
            "color": "orange", "alpha": 0.3,
        })

    def _add_box(self) -> None:
        center_x, center_y = self._get_plot_center()
        x_range, y_range = self._get_plot_extent()
        w, h = x_range * 0.2, y_range * 0.2
        self._commit_new_annotation({
            "type": "box", "x": center_x - w / 2, "y": center_y - h / 2,
            "width": w, "height": h,
            "facecolor": "yellow", "edgecolor": "black", "linewidth": 1.5, "alpha": 0.3,
        })

    def _add_callout(self) -> None:
        center_x, center_y = self._get_plot_center()
        x_range, y_range = self._get_plot_extent()
        self._commit_new_annotation({
            "type": "callout", "x": center_x, "y": center_y,
            "tx": center_x + x_range * 0.15, "ty": center_y + y_range * 0.15,
            "text": "Callout", "fontsize": 11, "color": "black", "arrowcolor": "black",
        })

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        self.annotation_list.setCurrentItem(item)
        self._edit_annotation()

    def _edit_annotation(self) -> None:
        selected = self.annotation_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select an annotation to edit.")
            return

        ann_id = selected.data(Qt.UserRole)
        annotations = getattr(self.graph_widget, "annotations", [])
        annotation = next((a for a in annotations if a.get("id") == ann_id), None)
        if not annotation:
            return

        dialog_cls = self._EDIT_DIALOGS.get(annotation.get("type"))
        if dialog_cls is None:
            return

        dialog = dialog_cls(annotation, self.graph_widget, self)
        if dialog.exec() == QDialog.Accepted:
            annotation.update(dialog.get_properties())
            if annotation["type"] == "vline":
                annotation["label"] = f"V-Line at x={annotation['x']:.2f}"
            elif annotation["type"] == "hline":
                annotation["label"] = f"H-Line at y={annotation['y']:.2f}"
            self._notify_annotations_changed()
            self._refresh_plot()
            self.load_settings()

    def _delete_annotation(self) -> None:
        selected = self.annotation_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select an annotation to delete.")
            return

        ann_id = selected.data(Qt.UserRole)
        annotations = getattr(self.graph_widget, "annotations", [])
        self.graph_widget.annotations = [a for a in annotations if a.get("id") != ann_id]
        self._notify_annotations_changed()
        self._refresh_plot()
        self.load_settings()

    def load_settings(self) -> None:
        """Load annotations into the list widget."""
        self.annotation_list.clear()
        annotations = getattr(self.graph_widget, "annotations", []) or []

        for ann in annotations:
            ann_type = ann.get("type")
            if ann_type == "vline":
                icon = Icons.ann_vline()
                text = f"VLine @ x={ann['x']:.2f} ({ann.get('color', 'red')})"
            elif ann_type == "hline":
                icon = Icons.ann_hline()
                text = f"HLine @ y={ann['y']:.2f} ({ann.get('color', 'blue')})"
            elif ann_type == "text":
                icon = Icons.ann_text()
                text = f"Text \"{ann.get('text', '')[:20]}\" @ ({ann.get('x', 0):.1f}, {ann.get('y', 0):.1f})"
            elif ann_type == "arrow":
                icon = Icons.ann_arrow()
                text = f"Arrow ({ann.get('x1', 0):.1f}, {ann.get('y1', 0):.1f}) → ({ann.get('x2', 0):.1f}, {ann.get('y2', 0):.1f})"
            elif ann_type == "vspan":
                icon = Icons.ann_vspan()
                text = f"V-Span x=[{ann.get('x1', 0):.2f}, {ann.get('x2', 0):.2f}] ({ann.get('color', 'orange')})"
            elif ann_type == "hspan":
                icon = Icons.ann_hspan()
                text = f"H-Span y=[{ann.get('y1', 0):.2f}, {ann.get('y2', 0):.2f}] ({ann.get('color', 'orange')})"
            elif ann_type == "box":
                icon = Icons.ann_box()
                text = f"Box @ ({ann.get('x', 0):.1f}, {ann.get('y', 0):.1f}) {ann.get('width', 0):.2f}×{ann.get('height', 0):.2f}"
            elif ann_type == "callout":
                icon = Icons.ann_callout()
                text = f"Callout \"{ann.get('text', '')[:20]}\" @ ({ann.get('x', 0):.1f}, {ann.get('y', 0):.1f})"
            else:
                icon = QIcon()
                text = f"Unknown annotation: {ann_type}"

            item = QListWidgetItem(icon, text)
            item.setData(Qt.UserRole, ann.get("id"))
            self.annotation_list.addItem(item)

    # Alias for existing test calls
    load_annotations = load_settings

    def apply_settings(self, replot: bool = False) -> dict:
        """Annotations are applied live to the graph widget; returns current annotations."""
        if replot:
            self._refresh_plot()
        return {"annotations": getattr(self.graph_widget, "annotations", [])}

    def highlight_search(self, query: str) -> None:
        """Highlight matching groupboxes based on search query."""
        q = query.strip().lower()
        self.grp_add.setStyleSheet("")
        self.grp_list.setStyleSheet("")

        if not q:
            return

        add_terms = ["add", "vline", "hline", "text", "arrow", "vspan", "hspan", "box", "callout", "shapes", "draw"]
        list_terms = ["list", "annotation", "annotations", "edit", "delete", "manage"]

        first_match: Optional[QWidget] = None
        if any(t in q or q in t for t in add_terms):
            self.grp_add.setStyleSheet(highlight_group_style())
            if first_match is None:
                first_match = self.grp_add

        if any(t in q or q in t for t in list_terms):
            self.grp_list.setStyleSheet(highlight_group_style())
            if first_match is None:
                first_match = self.grp_list

        if first_match:
            self._scroll.ensureWidgetVisible(first_match, 0, 30)
