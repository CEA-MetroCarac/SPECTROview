# view/components/v_fit_results.py
import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QLabel, QComboBox, QSplitter, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon

from spectroview import ICON_DIR
from spectroview.view.components.v_dataframe_table import VDataframeTable


class VFitResults(QWidget):
    """View for Fit Results tab - displays collected fit results in a table."""
    
    # ───── View → ViewModel signals ─────
    collect_results_requested = Signal()
    split_fname_requested = Signal()
    add_column_requested = Signal(str, int)  # (column_name, part_index)
    save_results_requested = Signal()
    send_to_viz_requested = Signal(str)  # (dataframe_name)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        
        # Row 1: Collect Results button
        self.btn_collect = QPushButton("Collect Fit Results")
        self.btn_collect.setIcon(QIcon(os.path.join(ICON_DIR, "collect.png")))
        self.btn_collect.setMinimumHeight(50)
        self.btn_collect.clicked.connect(self.collect_results_requested.emit)
        left_layout.addWidget(self.btn_collect)
        
        # Row 2: Split filename controls
        lbl_split = QLabel("Split the filename of spectra:")
        left_layout.addWidget(lbl_split)
        
        split_layout = QHBoxLayout()
        
        self.btn_split = QPushButton("Split")
        self.btn_split.setFixedWidth(60)
        self.btn_split.clicked.connect(self.split_fname_requested.emit)
        
        self.cbb_split_fname = QComboBox()
        self.cbb_split_fname.setMinimumWidth(100)
        
        self.ent_col_name = QLineEdit()
        self.ent_col_name.setPlaceholderText("Column name")
        self.ent_col_name.setFixedWidth(100)
        
        self.btn_add = QPushButton("Add")
        self.btn_add.setFixedWidth(60)
        self.btn_add.clicked.connect(self._on_add_column)
        
        split_layout.addWidget(self.btn_split)
        split_layout.addWidget(self.cbb_split_fname)
        split_layout.addWidget(self.ent_col_name)
        split_layout.addWidget(self.btn_add)
        split_layout.addStretch()
        
        left_layout.addLayout(split_layout)
        
        # Row 3: Send to visualization
        lbl_send = QLabel("Send fit results to Graphs Workspace for visualization:")
        left_layout.addWidget(lbl_send)
        
        send_layout = QHBoxLayout()
        
        self.ent_send_df_to_viz = QLineEdit()
        self.ent_send_df_to_viz.setPlaceholderText("DataFrame name")
        self.ent_send_df_to_viz.setText("SPECTRUMS_best_fit")
        
        self.btn_send = QPushButton("Send to Graphs")
        self.btn_send.setFixedWidth(100)
        self.btn_send.clicked.connect(self._on_send_to_viz)
        
        send_layout.addWidget(self.ent_send_df_to_viz)
        send_layout.addWidget(self.btn_send)
        
        left_layout.addLayout(send_layout)
        
        # Row 4: Save button at bottom
        left_layout.addStretch()
        
        self.btn_save = QPushButton("Save Fit Results")
        self.btn_save.setToolTip("Save fit results to Excel file")
        self.btn_save.setIcon(QIcon(os.path.join(ICON_DIR, "save.png")))
        self.btn_save.setMinimumHeight(40)
        self.btn_save.clicked.connect(self.save_results_requested.emit)
        left_layout.addWidget(self.btn_save)
        
        # Right panel - Results table
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create scroll area for table
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create table widget container
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        
        # VDataframeTable
        self.df_table = VDataframeTable(table_layout)
        
        scroll_area.setWidget(table_container)
        right_layout.addWidget(scroll_area)
        
        # Add panels to splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([250, 750])
        
        main_layout.addWidget(splitter)
    
    def _on_add_column(self):
        """Emit signal with column name and selected part index."""
        col_name = self.ent_col_name.text().strip()
        part_index = self.cbb_split_fname.currentIndex()
        
        if col_name and part_index >= 0:
            self.add_column_requested.emit(col_name, part_index)
    
    def _on_send_to_viz(self):
        """Emit signal with dataframe name."""
        df_name = self.ent_send_df_to_viz.text().strip()
        if df_name:
            self.send_to_viz_requested.emit(df_name)
    
    def populate_split_combobox(self, parts: list):
        """Populate the split filename combobox with parts."""
        self.cbb_split_fname.clear()
        self.cbb_split_fname.addItems(parts)
    
    def show_results(self, df):
        """Display results dataframe in table."""
        self.df_table.show(df, fill_colors=True)
    
    def clear_results(self):
        """Clear the results table."""
        self.df_table.clear()
