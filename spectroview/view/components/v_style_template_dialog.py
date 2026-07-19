#spectroview/view/components/v_style_template_dialog.py
"""Browse/apply/rename/delete saved per-graph style templates -- a leaner
sibling of v_plot_recipe_dialog.py's VPlotRecipeDialog (no "graph
count"/Duplicate: a style template is always exactly one style dict, see
model/graph_style.py and model/m_style_template_store.py)."""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QListWidget, QListWidgetItem, QWidget,
    QSizePolicy, QMessageBox, QInputDialog,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from spectroview.model.m_style_template_store import MStyleTemplateStore, StyleTemplateSummary


class _StyleTemplateItemWidget(QWidget):
    """Row widget for a single saved style template."""

    on_apply = Signal(str)
    on_rename = Signal(str)
    on_delete = Signal(str)

    def __init__(self, template_id: str, name: str, parent=None):
        super().__init__(parent)
        self.template_id = template_id

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        self.lbl_name = QLabel(name)
        self.lbl_name.setToolTip(name)
        font = QFont()
        font.setBold(True)
        self.lbl_name.setFont(font)
        self.lbl_name.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(self.lbl_name, stretch=1)

        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setStyleSheet("font-weight: bold;")
        self.btn_rename = QPushButton("Rename")
        self.btn_delete = QPushButton("Delete")

        for btn in (self.btn_apply, self.btn_rename, self.btn_delete):
            btn.setCursor(Qt.PointingHandCursor)
            layout.addWidget(btn)

        self.btn_apply.clicked.connect(lambda: self.on_apply.emit(self.template_id))
        self.btn_rename.clicked.connect(lambda: self.on_rename.emit(self.template_id))
        self.btn_delete.clicked.connect(lambda: self.on_delete.emit(self.template_id))


class VStyleTemplateDialog(QDialog):
    """Browse, search, apply, rename, and delete saved style templates."""

    style_applied = Signal(dict)  # emits the loaded style dict

    def __init__(self, store: MStyleTemplateStore, parent=None):
        super().__init__(parent)
        self.store = store
        self._cached_summaries: list[StyleTemplateSummary] = []
        self.setWindowTitle("🎨 Style Templates")
        self.setMinimumSize(480, 360)
        self.resize(520, 420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search style templates...")
        self.search_input.textChanged.connect(self._on_search)
        layout.addWidget(self.search_input)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("QListWidget::item { margin: 2px; }")
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

        self._reload_and_render()

    def _reload_and_render(self, filter_text: str = "") -> None:
        self.store.scan_folder()
        self._cached_summaries = self.store.list_templates()
        self._render_list(filter_text)

    def _render_list(self, filter_text: str = "") -> None:
        self.list_widget.clear()
        filter_text = filter_text.lower()

        if not self._cached_summaries:
            self.list_widget.addItem(QListWidgetItem("No style templates saved yet."))
            return

        for summary in self._cached_summaries:
            if filter_text and filter_text not in summary.name.lower():
                continue

            item = QListWidgetItem(self.list_widget)
            widget = _StyleTemplateItemWidget(
                template_id=summary.id, name=summary.name, parent=self.list_widget,
            )
            widget.on_apply.connect(self._on_apply)
            widget.on_rename.connect(self._on_rename)
            widget.on_delete.connect(self._on_delete)

            size = widget.sizeHint()
            size.setHeight(size.height() + 6)
            item.setSizeHint(size)

            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)

    def _on_search(self, text: str) -> None:
        self._render_list(text)

    def _on_apply(self, template_id: str) -> None:
        style = self.store.load_style(template_id)
        if style:
            self.style_applied.emit(style)
        self.accept()

    def _on_rename(self, template_id: str) -> None:
        summary = self.store.get_summary(template_id)
        current_name = summary.name if summary else ""
        new_name, ok = QInputDialog.getText(
            self, "Rename Style Template", "New name:", text=current_name
        )
        if ok and new_name:
            style = self.store.load_style(template_id)
            if style is not None:
                self.store.delete_template(template_id)
                self.store.save_template(new_name, style)
                self._reload_and_render(self.search_input.text())

    def _on_delete(self, template_id: str) -> None:
        name = next((s.name for s in self._cached_summaries if s.id == template_id), "this style")
        reply = QMessageBox.question(
            self, "Delete Style Template", f"Permanently delete '{name}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.store.delete_template(template_id)
        self._reload_and_render(self.search_input.text())
