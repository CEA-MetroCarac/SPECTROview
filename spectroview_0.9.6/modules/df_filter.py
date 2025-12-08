import os 
from spectroview import ICON_DIR

from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QLineEdit, QHBoxLayout, QPushButton,  QListWidget, QListWidgetItem, QCheckBox, QMenu, QApplication
from PySide6.QtGui import QIcon
from PySide6.QtCore import QCoreApplication, Qt


class DataframeFilter:
    """Class for Handling "Filter Features" in Querying Pandas DataFrames"""
    def __init__(self, df):
        self.df = df
        self.filters = []
        self.initUI()

    def initUI(self):
        """Initialize the UI components."""
        self.create_filter_widget()

    def create_filter_widget(self):
        """Create filter UI components and organize them directly within the QGroupBox."""
        # Create Group Box to hold all filter widgets
        self.gb_filter_widget = QGroupBox()
        self.gb_filter_widget.setTitle(QCoreApplication.translate("mainWindow", u"Data filtering:", None))

        # Set the main layout for the group box
        self.layout_main = QVBoxLayout(self.gb_filter_widget)

        # Horizontal layout to hold the filter entry and buttons
        self.layout_buttons = QHBoxLayout()
        self.layout_buttons.setSpacing(2)

        # Entry box for filter queries
        self.filter_query = QLineEdit(self.gb_filter_widget)
        self.filter_query.setPlaceholderText("Enter your filter expression...") 
        self.filter_query.returnPressed.connect(self.add_filter)
        self.layout_buttons.addWidget(self.filter_query)
        


        # Button to add a filter
        self.btn_add_filter = QPushButton(self.gb_filter_widget)
        icon_add = QIcon()
        icon_add.addFile(os.path.join(ICON_DIR, "add.png"))
        
        self.btn_add_filter.setIcon(icon_add)
        self.btn_add_filter.clicked.connect(self.add_filter) 
        self.layout_buttons.addWidget(self.btn_add_filter)

        # Button to remove selected filters
        self.btn_remove = QPushButton(self.gb_filter_widget)
        icon_remove = QIcon()
        icon_remove.addFile(os.path.join(ICON_DIR, "close.png"))
        self.btn_remove.setIcon(icon_remove)
        self.btn_remove.clicked.connect(self.remove_filter) 
        self.layout_buttons.addWidget(self.btn_remove)

        # Button to apply filters
        self.btn_apply = QPushButton(self.gb_filter_widget)
        icon_apply = QIcon()
        icon_apply.addFile(os.path.join(ICON_DIR, "done.png"))
        self.btn_apply.setIcon(icon_apply)
        self.btn_apply.setText("Apply")  
        self.btn_apply.setToolTip("Click to apply checked filters to the selected dataframe") 
        self.btn_apply.clicked.connect(self.apply_filters)  
        self.layout_buttons.addWidget(self.btn_apply)

        # Add the horizontal layout to the main layout of the group box
        self.layout_main.addLayout(self.layout_buttons)

        # Create QListWidget to display filter expressions as checkboxes
        self.filter_listbox = QListWidget(self.gb_filter_widget)
        self.layout_main.addWidget(self.filter_listbox)
        # Enable custom right-click menu
        self.filter_listbox.setContextMenuPolicy(Qt.CustomContextMenu)
        self.filter_listbox.customContextMenuRequested.connect(self.show_context_menu)
        self.filter_listbox.itemSelectionChanged.connect(self.on_filter_selected)
    
    def on_filter_selected(self):
        """
        When an item in the filter list is selected,
        display its text in the QLineEdit.
        """
        selected_items = self.filter_listbox.selectedItems()
        if not selected_items:
            return

        # Only handle the first selected item (since multi-select is possible)
        item = selected_items[0]
        checkbox = self.filter_listbox.itemWidget(item)
        if checkbox:
            self.filter_query.setText(checkbox.text())

    
    def show_context_menu(self, pos):
        """Show right-click menu to copy filter text."""
        item = self.filter_listbox.itemAt(pos)
        if not item:
            return
        checkbox = self.filter_listbox.itemWidget(item)
        if not checkbox:
            return

        menu = QMenu(self.filter_listbox)
        copy_action = menu.addAction("Copy filter text")
        
        action = menu.exec_(self.filter_listbox.mapToGlobal(pos))
        if action == copy_action:
            QApplication.clipboard().setText(checkbox.text())


    def set_dataframe(self, df):
        """Set the DataFrame to be filtered."""
        self.df = df

    def add_filter(self):
        """Add a filter expression to the filters list and update the UI."""
        filter_expression = self.filter_query.text().strip()
        if filter_expression:
            filter = {"expression": filter_expression, "state": False}
            self.filters.append(filter)
        # Add the filter expression to QListWidget as a checkbox item
        item = QListWidgetItem()
        checkbox = QCheckBox(filter_expression)
        item.setSizeHint(checkbox.sizeHint())
        self.filter_listbox.addItem(item)
        self.filter_listbox.setItemWidget(item, checkbox)

    def remove_filter(self):
        """Remove selected filter(s) from the filters list and UI."""
        selected_items = [item for item in self.filter_listbox.selectedItems()]
        for item in selected_items:
            checkbox = self.filter_listbox.itemWidget(item)
            filter_expression = checkbox.text()
            for filter in self.filters[:]:
                if filter.get("expression") == filter_expression:
                    self.filters.remove(filter)
            self.filter_listbox.takeItem(self.filter_listbox.row(item))

    def get_current_filters(self):
        """
        Retrieve the current state of filters as displayed in the UI.
        """
        checked_filters = []
        for i in range(self.filter_listbox.count()):
            item = self.filter_listbox.item(i)
            checkbox = self.filter_listbox.itemWidget(item)
            expression = checkbox.text()
            state = checkbox.isChecked()
            checked_filters.append({"expression": expression, "state": state})
        return checked_filters

    def apply_filters(self, filters=None):
        """
        Apply filters to the DataFrame based on the current or provided filters.
        """
        if filters:
            self.filters = filters
        else:
            checked_filters = self.get_current_filters()
            self.filters = checked_filters

        # Apply all filters at once
        self.filtered_df = self.df.copy() if self.df is not None else None

        if self.filtered_df is not None:  # Check if filtered_df is not None
            for filter_data in self.filters:
                filter_expr = filter_data["expression"]
                is_checked = filter_data["state"]
                if is_checked:
                    try:
                        filter_expr = str(filter_expr)
                        print(f"Applying filter expression: {filter_expr}")
                        # Apply the filter
                        self.filtered_df = self.filtered_df.query(filter_expr)
                    except Exception as e:
                        print(f"Error applying filter: {str(e)}")

        return self.filtered_df

    def upd_filter_listbox(self):
        """
        Update the listbox UI to reflect changes in filters.
        """
        self.filter_listbox.clear()
        for filter_data in self.filters:
            filter_expression = filter_data["expression"]
            item = QListWidgetItem()
            checkbox = QCheckBox(filter_expression)
            item.setSizeHint(checkbox.sizeHint())
            self.filter_listbox.addItem(item)
            self.filter_listbox.setItemWidget(item, checkbox)
            checkbox.setChecked(filter_data["state"])