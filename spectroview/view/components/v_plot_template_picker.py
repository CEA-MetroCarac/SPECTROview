#spectroview/view/components/v_plot_template_picker.py
"""Lightweight, apply-only template picker for the Graphs workspace —
browse and apply saved plot templates without the AI chat panel open.
Full management (rename/duplicate/delete) lives in VPlotTemplateDialog,
reachable from the AI chat panel, so this stays a thin "just apply" UI
rather than a second management surface for the same data.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QPushButton, QVBoxLayout
)

from spectroview.model.m_plot_template_store import MPlotTemplateStore


class VPlotTemplatePicker(QDialog):
    """Minimal browse-and-apply dialog for saved plot templates."""

    template_applied = Signal(list)  # list of plot-config dicts

    def __init__(self, store: MPlotTemplateStore, parent=None):
        super().__init__(parent)
        self.store = store
        self.setWindowTitle("📊 Apply Plot Template")
        self.setMinimumSize(380, 320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search templates...")
        self.search_input.textChanged.connect(self._render_list)
        layout.addWidget(self.search_input)

        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._on_apply_current)
        layout.addWidget(self.list_widget)

        self.lbl_hint = QLabel()
        self.lbl_hint.setStyleSheet("font-size: 11px;")
        self.lbl_hint.setWordWrap(True)
        layout.addWidget(self.lbl_hint)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setObjectName("btnRowPrimary")
        self.btn_apply.clicked.connect(self._on_apply_current)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_apply)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        self._summaries = []
        self._reload_and_render()

    def _reload_and_render(self, filter_text: str = "") -> None:
        self.store.scan_folder()
        self._summaries = self.store.list_templates()
        self._render_list(filter_text)

    def _render_list(self, filter_text: str = "") -> None:
        self.list_widget.clear()
        filter_text = filter_text.lower()
        for summary in self._summaries:
            if filter_text and filter_text not in summary.name.lower():
                continue
            plural = "s" if summary.graph_count != 1 else ""
            item = QListWidgetItem(f"{summary.name}  ({summary.graph_count} graph{plural})")
            item.setData(Qt.UserRole, summary.id)
            self.list_widget.addItem(item)

        if not self._summaries:
            self.lbl_hint.setText("No saved templates yet — save one from the AI Chat panel.")
        else:
            self.lbl_hint.setText("Double-click a template, or select it and click Apply.")

    def _on_apply_current(self, *_args) -> None:
        item = self.list_widget.currentItem()
        if not item:
            return
        template_id = item.data(Qt.UserRole)
        tpl = self.store.load_template(template_id)
        if tpl and tpl.configs:
            self.template_applied.emit(tpl.configs)
        self.accept()
