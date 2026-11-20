# view/components/v_data_filter.py
"""DataFrame filtering widget with checkboxes."""

from spectroview.i18n import bind_placeholder, bind_text, bind_title, bind_tooltip, tr
from spectroview.view.theme import Icons

from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QMenu, QApplication, QCompleter
)
from PySide6.QtCore import Qt, Signal, QStringListModel, QEvent, QItemSelectionModel

from spectroview.view.components.common.customized_widgets import ExpressionLineEdit


class VDataFilter(QGroupBox):
    """DataFrame filter widget with add/remove/apply functionality."""
    
    # Signals
    filters_changed = Signal(list)  # List of filter dicts
    apply_requested = Signal()
    
    def __init__(self, parent=None):
        """Initialize the data filter widget."""
        super().__init__(parent)
        bind_title(self, "Dataframe filters:")
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        layout_main = QVBoxLayout(self)
        layout_main.setContentsMargins(2, 2, 2, 2)
        
        # Horizontal layout for input and buttons
        layout_buttons = QHBoxLayout()
        layout_buttons.setSpacing(2)
        
        # Filter query input with autocomplete
        self.filter_query = ExpressionLineEdit()
        bind_placeholder(self.filter_query, "Enter your filter expression...")
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
        self.btn_add.setIcon(Icons.plus())
        bind_tooltip(self.btn_add, "Add filter")
        self.btn_add.clicked.connect(self._on_add_filter)
        layout_buttons.addWidget(self.btn_add)
        
        # Remove filter button
        self.btn_remove = QPushButton()
        self.btn_remove.setIcon(Icons.close())
        bind_tooltip(self.btn_remove, "Remove selected filter")
        self.btn_remove.clicked.connect(self._on_remove_filter)
        layout_buttons.addWidget(self.btn_remove)
        
        # Apply filters button
        self.btn_apply = QPushButton()
        bind_text(self.btn_apply, "Apply")
        self.btn_apply.setIcon(Icons.check())
        bind_tooltip(self.btn_apply, "Click to apply checked filters to the selected dataframe")
        self.btn_apply.clicked.connect(self._on_apply)
        layout_buttons.addWidget(self.btn_apply)
        
        layout_main.addLayout(layout_buttons)
        
        # Filter list widget
        self.filter_listbox = QListWidget()
        self.filter_listbox.setContextMenuPolicy(Qt.CustomContextMenu)
        self.filter_listbox.customContextMenuRequested.connect(self._show_context_menu)
        self.filter_listbox.itemPressed.connect(self._on_filter_item_clicked)
        self.filter_listbox.itemClicked.connect(self._on_filter_item_clicked)
        self.filter_listbox.itemChanged.connect(self._on_filter_item_changed)
        self.filter_listbox.installEventFilter(self)
        layout_main.addWidget(self.filter_listbox)

    def _select_filter_item(self, item: QListWidgetItem, update_query: bool = True):
        """Select a filter item exclusively and optionally load its expression into the query field."""
        if item is None:
            return
        self.filter_listbox.setCurrentItem(item, QItemSelectionModel.ClearAndSelect)
        item.setSelected(True)
        if update_query:
            self.filter_query.setText(item.text())

    def _on_filter_item_changed(self, item: QListWidgetItem):
        """Sync selection/query and emit filters_changed when a checkbox state changes."""
        self._select_filter_item(item, update_query=True)
        self.filters_changed.emit(self.get_filters())

    def eventFilter(self, watched, event):
        """Intercept keyboard navigation in listbox to sync selection and query text."""
        if watched == self.filter_listbox and event.type() == QEvent.KeyRelease:
            if event.key() in (Qt.Key_Up, Qt.Key_Down, Qt.Key_PageUp, Qt.Key_PageDown, Qt.Key_Home, Qt.Key_End):
                current_item = self.filter_listbox.currentItem()
                if current_item:
                    self._select_filter_item(current_item, update_query=True)
        return super().eventFilter(watched, event)
    
    def _add_filter_item(self, expression: str, state: bool = True) -> QListWidgetItem:
        """Create and add a checkable filter item to listbox."""
        item = QListWidgetItem(expression)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if state else Qt.Unchecked)
        was_blocked = self.filter_listbox.blockSignals(True)
        try:
            self.filter_listbox.addItem(item)
        finally:
            self.filter_listbox.blockSignals(was_blocked)
        return item
    
    def has_filter(self, expression: str) -> bool:
        """Check if a filter expression already exists in the list."""
        expression = expression.strip()
        for i in range(self.filter_listbox.count()):
            if self.filter_listbox.item(i).text() == expression:
                return True
        return False

    def add_filter_definition(self, expression: str):
        """Add a filter expression to the available list as inactive if not already present."""
        expression = expression.strip()
        if not expression or self.has_filter(expression):
            return
        self._add_filter_item(expression, state=False)

    def add_filter(self, expression: str, state: bool = True):
        """Add or update a filter expression with the given active state."""
        expression = expression.strip()
        if not expression:
            return
        for i in range(self.filter_listbox.count()):
            item = self.filter_listbox.item(i)
            if item.text() == expression:
                was_blocked = self.filter_listbox.blockSignals(True)
                try:
                    item.setCheckState(Qt.Checked if state else Qt.Unchecked)
                finally:
                    self.filter_listbox.blockSignals(was_blocked)
                self._select_filter_item(item, update_query=False)
                return
        item = self._add_filter_item(expression, state=state)
        self._select_filter_item(item, update_query=False)

    def _on_add_filter(self):
        """Add new filter expression to list, keeping the entered expression in QLineEdit."""
        filter_expression = self.filter_query.text().strip()
        if filter_expression:
            self.add_filter(filter_expression, state=True)
            # Retain entered expression in self.filter_query (do not clear)
            # Emit signal with all filters
            self.filters_changed.emit(self.get_filters())
    
    def _on_remove_filter(self):
        """Remove selected filter(s) from list."""
        selected_items = self.filter_listbox.selectedItems()
        if not selected_items and self.filter_listbox.currentItem():
            selected_items = [self.filter_listbox.currentItem()]

        for item in selected_items:
            row = self.filter_listbox.row(item)
            self.filter_listbox.takeItem(row)
        
        # Emit signal with updated filters
        self.filters_changed.emit(self.get_filters())
    
    def _on_apply(self):
        """Emit signal to apply filters."""
        self.apply_requested.emit()
    
    def _on_filter_item_clicked(self, item):
        """Select clicked item and show filter text in input field."""
        self._select_filter_item(item, update_query=True)
    
    def _show_context_menu(self, pos):
        """Show right-click menu to copy filter text."""
        item = self.filter_listbox.itemAt(pos)
        if not item:
            return
        
        menu = QMenu(self)
        copy_action = menu.addAction(Icons.copy(), tr("Copy filter text"))
        
        action = menu.exec(self.filter_listbox.mapToGlobal(pos))
        if action == copy_action:
            QApplication.clipboard().setText(item.text())
    
    def get_filters(self) -> list:
        """Get all filter expressions and states."""
        filters = []
        for i in range(self.filter_listbox.count()):
            item = self.filter_listbox.item(i)
            filters.append({
                "expression": item.text(),
                "state": item.checkState() == Qt.Checked,
            })
        return filters

    def get_active_filters(self) -> list:
        """Get only active (checked) filter expressions."""
        return [f for f in self.get_filters() if f.get("state", False)]
    
    def update_filter_states(self, filters: list):
        """Update active/inactive states of filters according to a graph's filter list.
        
        Preserves all available filter definitions in the workspace/session.
        Expressions in `filters` with state=True are checked.
        Expressions not in `filters` or with state=False are unchecked.
        Any expressions in `filters` not previously known are added to the list.
        """
        active_map = {}
        for f in (filters or []):
            if isinstance(f, dict):
                expr = f.get("expression", "").strip()
                if expr:
                    state = f.get("state", True)
                    active_map[expr] = bool(state)

        existing_expressions = set()
        was_blocked = self.filter_listbox.blockSignals(True)
        try:
            for i in range(self.filter_listbox.count()):
                item = self.filter_listbox.item(i)
                expr = item.text()
                existing_expressions.add(expr)
                is_active = active_map.get(expr, False)
                item.setCheckState(Qt.Checked if is_active else Qt.Unchecked)

            # Add any new filter expressions from the graph that were not already in the list
            for expr, is_active in active_map.items():
                if expr not in existing_expressions:
                    self._add_filter_item(expr, state=is_active)
                    existing_expressions.add(expr)
        finally:
            self.filter_listbox.blockSignals(was_blocked)

    def set_filters(self, filters: list):
        """Set filter states for current graph, preserving all available session filters."""
        self.update_filter_states(filters)

    def replace_filters(self, filters: list):
        """Completely replace the list of filters (clears and re-adds)."""
        self.clear_filters()
        self.update_filter_states(filters)
    
    def clear_filters(self):
        """Clear all filters."""
        self.filter_listbox.clear()
        self.filter_query.clear()
        self.filters_changed.emit([])
    
    def update_column_list(self, columns: list):
        """Update autocomplete suggestions with DataFrame column names."""
        if columns:
            self._completer_model.setStringList(columns)

