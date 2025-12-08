

from PySide6.QtWidgets import QVBoxLayout, QTableWidget, QWidget, QTableWidgetItem, QApplication
from PySide6.QtGui import QAction, QColor
from PySide6.QtCore import Qt

class DataframeTable(QWidget):
    """Class to display a given dataframe in GUI via QTableWidget.
    """

    def __init__(self, layout):
        super().__init__()
        self.external_layout = layout
        self.initUI()

    def initUI(self):
        # Clear existing widgets from external layout
        while self.external_layout.count():
            item = self.external_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Set the internal layout
        layout = QVBoxLayout(self)
        self.setLayout(layout)

        # Create QTableWidget
        self.table_widget = QTableWidget()
        self.table_widget.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
        layout.addWidget(self.table_widget)

        # Enable copy action via context menu
        self.table_widget.setContextMenuPolicy(Qt.ActionsContextMenu)
        copy_action = QAction("Copy", self)
        copy_action.triggered.connect(self.copy_data)
        self.table_widget.addAction(copy_action)

        # Add this widget to the external layout
        self.external_layout.addWidget(self)

    
    def show(self, df, fill_colors=True):
        """Populates the QTableWidget with data from the given DataFrame, with colored columns."""
        if df is not None and not df.empty:
            self.table_widget.setRowCount(df.shape[0])
            self.table_widget.setColumnCount(df.shape[1])
            self.table_widget.setHorizontalHeaderLabels(df.columns)

            # Assign colors based on the column prefix
            column_colors = self.get_column_colors(df.columns)

            for row in range(df.shape[0]):
                for col in range(df.shape[1]):
                    item = QTableWidgetItem(str(df.iat[row, col]))
                    if fill_colors:
                        item.setBackground(column_colors[col])  # Set background color
                    self.table_widget.setItem(row, col, item)

            self.table_widget.resizeColumnsToContents()
        else:
            self.clear()
            
    def get_column_colors(self, columns):
        """Generates a color for each column """
        palette = ['#bda16d', '#a27ba0', '#cb5b12', '#23993b', '#008281', '#147ce4']
        prefix_colors = {}
        column_colors = []

        for col_name in columns:
            prefix = col_name.split("_")[0] if "_" in col_name else col_name
            # Assign the next available color from the palette
            if prefix not in prefix_colors:
                color_index = len(prefix_colors) % len(palette)
                prefix_colors[prefix] = QColor(palette[color_index])
            column_colors.append(prefix_colors[prefix])
        return column_colors

    
    def clear(self):
        """Clears all data from the QTableWidget."""
        self.table_widget.clearContents()
        self.table_widget.setRowCount(0)
        self.table_widget.setColumnCount(0)
        self.table_widget.setHorizontalHeaderLabels([])

    def copy_data(self):
        """Copies selected data from the QTableWidget to the clipboard."""
        selected_indexes = self.table_widget.selectedIndexes()
        if not selected_indexes:
            return

        # Collect unique rows and columns
        rows = set(index.row() for index in selected_indexes)
        cols = set(index.column() for index in selected_indexes)

        data = []
        for row in sorted(rows):
            row_data = []
            for col in sorted(cols):
                item = self.table_widget.item(row, col)
                if item is not None:
                    row_data.append(item.text())
                else:
                    row_data.append('')
            data.append('\t'.join(row_data))

        # Join all rows with newline character and copy to clipboard
        clipboard = QApplication.clipboard()
        clipboard.setText('\n'.join(data))

    def keyPressEvent(self, event):
        """Handles key press events to enable copying with Ctrl+C."""
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_C:
            self.copy_data()
        else:
            super().keyPressEvent(event)