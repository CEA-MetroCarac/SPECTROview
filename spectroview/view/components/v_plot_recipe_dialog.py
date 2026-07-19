#spectroview/view/components/v_plot_recipe_dialog.py
"""Full management dialog for saved plot recipes — modeled closely on
spectroview/ai_agent/v_history_dialog.py (search, Apply/Rename/Duplicate/
Delete), reused from both the AI chat panel and (in a lighter, apply-only
form) the Graphs workspace.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QListWidget, QListWidgetItem, QWidget,
    QSizePolicy, QMessageBox, QGridLayout, QInputDialog
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from spectroview.model.m_plot_recipe_store import MPlotRecipeStore, PlotRecipeSummary


class _PlotRecipeItemWidget(QWidget):
    """Row widget for a single saved recipe."""

    on_apply = Signal(str)
    on_rename = Signal(str)
    on_duplicate = Signal(str)
    on_delete = Signal(str)

    def __init__(self, recipe_id: str, name: str, graph_count: int, parent=None):
        super().__init__(parent)
        self.recipe_id = recipe_id

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        self.lbl_name = QLabel(name)
        self.lbl_name.setToolTip(name)
        font = QFont()
        font.setBold(True)
        self.lbl_name.setFont(font)
        self.lbl_name.setWordWrap(True)
        self.lbl_name.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.lbl_name.setMinimumHeight(34)
        self.lbl_name.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(self.lbl_name, stretch=1)

        right_widget = QWidget()
        right_layout = QGridLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        lbl_meta = QLabel(f"{graph_count} graph{'s' if graph_count != 1 else ''}")
        lbl_meta.setStyleSheet("font-size: 11px;")

        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setObjectName("btnRowPrimary")
        self.btn_rename = QPushButton("Rename")
        self.btn_rename.setObjectName("btnRowLink")
        self.btn_duplicate = QPushButton("Duplicate")
        self.btn_duplicate.setObjectName("btnRowLink")
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setObjectName("btnRowDanger")

        self.btn_apply.setCursor(Qt.PointingHandCursor)
        self.btn_apply.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.btn_apply.setStyleSheet("font-weight: bold; font-size: 13px;")

        for btn in (self.btn_rename, self.btn_duplicate, self.btn_delete):
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("font-size: 11px;")

        self.btn_apply.clicked.connect(lambda: self.on_apply.emit(self.recipe_id))
        self.btn_rename.clicked.connect(lambda: self.on_rename.emit(self.recipe_id))
        self.btn_duplicate.clicked.connect(lambda: self.on_duplicate.emit(self.recipe_id))
        self.btn_delete.clicked.connect(lambda: self.on_delete.emit(self.recipe_id))

        right_layout.addWidget(self.btn_apply, 0, 0, 2, 2)
        right_layout.addWidget(lbl_meta, 0, 2, 1, 3, Qt.AlignRight | Qt.AlignBottom)
        right_layout.addWidget(self.btn_rename, 1, 2, 1, 1, Qt.AlignRight)
        right_layout.addWidget(self.btn_duplicate, 1, 3, 1, 1, Qt.AlignRight)
        right_layout.addWidget(self.btn_delete, 1, 4, 1, 1, Qt.AlignRight)
        right_layout.setRowMinimumHeight(1, 20)
        right_layout.setColumnStretch(0, 1)
        right_layout.setColumnStretch(1, 1)

        layout.addWidget(right_widget, stretch=0)


class VPlotRecipeDialog(QDialog):
    """Browse, search, apply, rename, duplicate, and delete saved plot recipes."""

    recipe_applied = Signal(list)  # emits the recipe's list of plot-config dicts

    def __init__(self, store: MPlotRecipeStore, parent=None):
        super().__init__(parent)
        self.store = store
        self._cached_summaries: list[PlotRecipeSummary] = []
        self.setWindowTitle("📊 Plot Recipes")
        self.setMinimumSize(600, 400)
        self.resize(700, 500)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search recipes...")
        self.search_input.textChanged.connect(self._on_search)
        layout.addWidget(self.search_input)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("QListWidget::item { margin: 2px; }")
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

        self._reload_and_render()

    def _reload_and_render(self, filter_text: str = "") -> None:
        self.store.scan_folder()
        self._cached_summaries = self.store.list_recipes()
        self._render_list(filter_text)

    def _render_list(self, filter_text: str = "") -> None:
        self.list_widget.clear()
        filter_text = filter_text.lower()

        for summary in self._cached_summaries:
            if filter_text and filter_text not in summary.name.lower():
                continue

            item = QListWidgetItem(self.list_widget)
            widget = _PlotRecipeItemWidget(
                recipe_id=summary.id,
                name=summary.name,
                graph_count=summary.graph_count,
                parent=self.list_widget,
            )
            widget.on_apply.connect(self._on_apply)
            widget.on_rename.connect(self._on_rename)
            widget.on_duplicate.connect(self._on_duplicate)
            widget.on_delete.connect(self._on_delete)

            size = widget.sizeHint()
            size.setHeight(size.height() + 6)
            item.setSizeHint(size)

            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)

    def _on_search(self, text: str) -> None:
        self._render_list(text)

    def _on_apply(self, recipe_id: str) -> None:
        recipe = self.store.load_recipe(recipe_id)
        if recipe and recipe.configs:
            self.recipe_applied.emit(recipe.configs)
        self.accept()

    def _on_rename(self, recipe_id: str) -> None:
        summary = self.store.get_summary(recipe_id)
        if not summary:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename Recipe", "New name:", text=summary.name
        )
        if ok and new_name:
            recipe = self.store.load_recipe(recipe_id)
            if recipe:
                recipe.rename(new_name)
                recipe.save()
                self._reload_and_render(self.search_input.text())

    def _on_duplicate(self, recipe_id: str) -> None:
        recipe = self.store.load_recipe(recipe_id)
        if recipe:
            new_recipe = recipe.duplicate()
            new_recipe.save(self.store.folder_path)
            self._reload_and_render(self.search_input.text())

    def _on_delete(self, recipe_id: str) -> None:
        summary = self.store.get_summary(recipe_id)
        name = summary.name if summary else "this recipe"
        reply = QMessageBox.question(
            self, "Delete Recipe", f"Permanently delete '{name}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.store.delete_recipe(recipe_id)
        self._reload_and_render(self.search_input.text())
