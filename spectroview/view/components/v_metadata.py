# view/components/v_metadata.py
"""Widget for displaying spectrum acquisition metadata."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QLabel
)
from PySide6.QtCore import Qt


class VMetadata(QWidget):
    """View for Metadata tab - displays acquisition metadata for selected spectrum."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Table widget for metadata display
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Parameter", "Value"])
        
        # Configure table appearance
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)  # Read-only
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        
        main_layout.addWidget(self.table)
        
        # Placeholder label for no data
        self.lbl_placeholder = QLabel("No metadata available\n\nSelect a spectrum to view its metadata")
        self.lbl_placeholder.setAlignment(Qt.AlignCenter)
        self.lbl_placeholder.setStyleSheet("color: #999; font-size: 14px; padding: 40px;")
        main_layout.addWidget(self.lbl_placeholder)
        
        # Initially show placeholder
        self.table.setVisible(False)
        self.lbl_placeholder.setVisible(True)
    
    def show_metadata(self, metadata: dict):
        """Display metadata dictionary in the table.
        
        Args:
            metadata: Dictionary of metadata key-value pairs
        """
        if not metadata or len(metadata) == 0:
            self.clear_metadata()
            return
        
        # Hide placeholder, show table
        self.lbl_placeholder.setVisible(False)
        self.table.setVisible(True)
        
        # Clear existing rows
        self.table.setRowCount(0)
        
        # Populate table with metadata
        for row_idx, (key, value) in enumerate(metadata.items()):
            self.table.insertRow(row_idx)
            
            # Key column
            key_item = QTableWidgetItem(str(key))
            key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row_idx, 0, key_item)
            
            # Value column
            value_item = QTableWidgetItem(str(value))
            value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row_idx, 1, value_item)
    
    def clear_metadata(self):
        """Clear the metadata display and show placeholder."""
        self.table.setVisible(False)
        self.lbl_placeholder.setVisible(True)
        self.table.setRowCount(0)
