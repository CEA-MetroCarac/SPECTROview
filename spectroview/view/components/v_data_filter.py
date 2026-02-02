# view/components/v_data_filter.py
"""DataFrame filtering widget with checkboxes."""

import os
from spectroview import ICON_DIR

from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QCheckBox, QMenu, QApplication, QCompleter
)
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt, Signal, QStringListModel

from spectroview.view.components.v_expression_lineedit import ExpressionLineEdit


class VDataFilter(QGroupBox):
    """DataFrame filter widget with add/remove/apply functionality."""
    
    # Signals
    filters_changed = Signal(list)  # List of filter dicts
    apply_requested = Signal()
    
    def __init__(self, parent=None):
        """Initialize the data filter widget."""
        super().__init__(parent)
        self.setTitle("Data filtering:")
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        layout_main = QVBoxLayout(self)
        
        # Horizontal layout for input and buttons
        layout_buttons = QHBoxLayout()
        layout_buttons.setSpacing(2)
        
        # Filter query input with autocomplete
        self.filter_query = ExpressionLineEdit()
        self.filter_query.setPlaceholderText("Enter your filter expression...")
        self.filter_query.returnPressed.connect(self._on_add_filter)
        layout_buttons.addWidget(self.filter_query)
        
        # Setup autocomplete (will be populated when DataFrame is set)
        self._completer_model = QStringListModel()
        self._completer = QCompleter()
        self._completer.setModel(self._completer_model)
        self._completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.filter_query.setCompleter(self._completer)
        
        # Add filter button
        self.btn_add = QPushButton()
        self.btn_add.setIcon(QIcon(os.path.join(ICON_DIR, "add.png")))
        self.btn_add.setToolTip("Add filter")
        self.btn_add.clicked.connect(self._on_add_filter)
        layout_buttons.addWidget(self.btn_add)
        
        # Remove filter button
        self.btn_remove = QPushButton()
        self.btn_remove.setIcon(QIcon(os.path.join(ICON_DIR, "close.png")))
        self.btn_remove.setToolTip("Remove selected filter")
        self.btn_remove.clicked.connect(self._on_remove_filter)
        layout_buttons.addWidget(self.btn_remove)
        
        # Apply filters button
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setIcon(QIcon(os.path.join(ICON_DIR, "done.png")))
        self.btn_apply.setToolTip("Click to apply checked filters to the selected dataframe")
        self.btn_apply.clicked.connect(self._on_apply)
        layout_buttons.addWidget(self.btn_apply)
        
        layout_main.addLayout(layout_buttons)
        
        # Filter list widget
        self.filter_listbox = QListWidget()
        self.filter_listbox.setContextMenuPolicy(Qt.CustomContextMenu)
        self.filter_listbox.customContextMenuRequested.connect(self._show_context_menu)
        self.filter_listbox.itemSelectionChanged.connect(self._on_filter_selected)
        layout_main.addWidget(self.filter_listbox)
    
    def _on_add_filter(self):
        """Add new filter expression to list."""
        filter_expression = self.filter_query.text().strip()
        if filter_expression:
            # Add checkbox item to listbox
            item = QListWidgetItem()
            checkbox = QCheckBox(filter_expression)
            item.setSizeHint(checkbox.sizeHint())
            self.filter_listbox.addItem(item)
            self.filter_listbox.setItemWidget(item, checkbox)
            
            # Clear input
            self.filter_query.clear()
            
            # Emit signal with all filters
            self.filters_changed.emit(self.get_filters())
    
    def _on_remove_filter(self):
        """Remove selected filter(s) from list."""
        selected_items = self.filter_listbox.selectedItems()
        for item in selected_items:
            row = self.filter_listbox.row(item)
            self.filter_listbox.takeItem(row)
        
        # Emit signal with updated filters
        self.filters_changed.emit(self.get_filters())
    
    def _on_apply(self):
        """Emit signal to apply filters."""
        self.apply_requested.emit()
    
    def _on_filter_selected(self):
        """Show selected filter text in input field."""
        selected_items = self.filter_listbox.selectedItems()
        if selected_items:
            item = selected_items[0]
            checkbox = self.filter_listbox.itemWidget(item)
            if checkbox:
                self.filter_query.setText(checkbox.text())
    
    def _show_context_menu(self, pos):
        """Show right-click menu to copy filter text."""
        item = self.filter_listbox.itemAt(pos)
        if not item:
            return
        
        checkbox = self.filter_listbox.itemWidget(item)
        if not checkbox:
            return
        
        menu = QMenu(self)
        copy_action = menu.addAction("Copy filter text")
        
        action = menu.exec(self.filter_listbox.mapToGlobal(pos))
        if action == copy_action:
            QApplication.clipboard().setText(checkbox.text())
    
    def get_filters(self) -> list:
        """Get all filter expressions and states."""
        filters = []
        for i in range(self.filter_listbox.count()):
            item = self.filter_listbox.item(i)
            checkbox = self.filter_listbox.itemWidget(item)
            if checkbox:
                filters.append({
                    "expression": checkbox.text(),
                    "state": checkbox.isChecked()
                })
        return filters
    
    def set_filters(self, filters: list):
        """Set filter expressions and states."""
        self.filter_listbox.clear()
        for filter_data in filters:
            item = QListWidgetItem()
            checkbox = QCheckBox(filter_data["expression"])
            checkbox.setChecked(filter_data.get("state", False))
            item.setSizeHint(checkbox.sizeHint())
            self.filter_listbox.addItem(item)
            self.filter_listbox.setItemWidget(item, checkbox)
    
    def clear_filters(self):
        """Clear all filters."""
        self.filter_listbox.clear()
        self.filter_query.clear()
        self.filters_changed.emit([])
    
    def update_column_list(self, columns: list):
        """Update autocomplete suggestions with DataFrame column names."""
        if columns:
            self._completer_model.setStringList(columns)

