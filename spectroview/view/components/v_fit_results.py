# view/components/v_fit_results.py
import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QLabel, QComboBox, QSplitter, QScrollArea
)
from PySide6.QtCore import Qt, Signal, QSize, QStringListModel
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QLabel, QComboBox, QSplitter, QScrollArea, QCompleter
)

from spectroview import ICON_DIR
from spectroview.view.components.v_dataframe_table import VDataframeTable


class ExpressionLineEdit(QLineEdit):
    """Custom QLineEdit with smart autocomplete for mathematical expressions.
    
    Supports autocomplete for column names anywhere in the expression,
    not just at the beginning.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._completer = None
    
    def setCompleter(self, completer):
        """Set the completer for this line edit."""
        if self._completer:
            self._completer.activated.disconnect()
        
        self._completer = completer
        
        if not self._completer:
            return
        
        self._completer.setWidget(self)
        self._completer.activated.connect(self._insert_completion)
    
    def completer(self):
        """Return the current completer."""
        return self._completer
    
    def _insert_completion(self, completion):
        """Insert the selected completion at cursor position."""
        if not self._completer:
            return
        
        # Get the text being completed
        text = self.text()
        cursor_pos = self.cursorPosition()
        
        # Find word boundaries around cursor
        start = cursor_pos
        while start > 0 and self._is_word_char(text[start - 1]):
            start -= 1
        
        # Replace the partial word with the completion
        new_text = text[:start] + completion + text[cursor_pos:]
        self.setText(new_text)
        self.setCursorPosition(start + len(completion))
    
    def _is_word_char(self, char):
        """Check if character is part of a word (column name)."""
        # Column names can contain letters, numbers, underscores, and backticks
        return char.isalnum() or char in ('_', '`')
    
    def _get_text_under_cursor(self):
        """Get the partial word being typed at cursor position."""
        text = self.text()
        cursor_pos = self.cursorPosition()
        
        # Find start of current word
        start = cursor_pos
        while start > 0 and self._is_word_char(text[start - 1]):
            start -= 1
        
        # Extract the partial word
        return text[start:cursor_pos]
    
    def keyPressEvent(self, event):
        """Handle key press events to show completer popup."""
        if self._completer and self._completer.popup().isVisible():
            # Let completer handle these keys when popup is visible
            if event.key() in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Escape, 
                              Qt.Key_Tab, Qt.Key_Backtab):
                event.ignore()
                return
        
        # Process the key normally
        super().keyPressEvent(event)
        
        # Update completer with current word
        if self._completer:
            completion_prefix = self._get_text_under_cursor()
            
            # Only show completer if we have at least 1 character
            if len(completion_prefix) >= 1:
                if completion_prefix != self._completer.completionPrefix():
                    self._completer.setCompletionPrefix(completion_prefix)
                    popup = self._completer.popup()
                    popup.setCurrentIndex(self._completer.completionModel().index(0, 0))
                
                # Position popup at cursor
                cursor_rect = self.cursorRect()
                cursor_rect.setWidth(
                    self._completer.popup().sizeHintForColumn(0) +
                    self._completer.popup().verticalScrollBar().sizeHint().width()
                )
                self._completer.complete(cursor_rect)
            else:
                self._completer.popup().hide()


class VFitResults(QWidget):
    """View for Fit Results tab - displays collected fit results in a table."""
    
    # ───── View → ViewModel signals ─────
    collect_results_requested = Signal()
    split_fname_requested = Signal()
    add_column_requested = Signal(str, int)  # (column_name, part_index)
    compute_column_requested = Signal(str, str)  # (column_name, expression)
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
        left_panel.setMaximumWidth(400)
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
        
        # Row 3: Computed column from expression
        lbl_compute = QLabel("Add computed column from expression:")
        left_layout.addWidget(lbl_compute)
        
        compute_layout = QHBoxLayout()
        
        self.ent_compute_col_name = QLineEdit()
        self.ent_compute_col_name.setPlaceholderText("New Column Name")
        self.ent_compute_col_name.setMinimumWidth(60)
        
        # Use custom ExpressionLineEdit for smart autocomplete
        self.ent_compute_expression = ExpressionLineEdit()
        self.ent_compute_expression.setPlaceholderText("e.g., column1 - column2 * 2")
        self.ent_compute_expression.setMinimumWidth(150)
        
        # Add autocomplete for column names
        self.completer = QCompleter()
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setCompletionMode(QCompleter.PopupCompletion)
        self.ent_compute_expression.setCompleter(self.completer)
        
        self.btn_compute = QPushButton("Compute && Add")
        self.btn_compute.setFixedWidth(100)
        self.btn_compute.clicked.connect(self._on_compute_column)
        
        compute_layout.addWidget(self.ent_compute_col_name)
        compute_layout.addWidget(self.ent_compute_expression)
        compute_layout.addWidget(self.btn_compute)
        compute_layout.addStretch()
        
        left_layout.addLayout(compute_layout)
        
        # Row 4: Send to visualization
        lbl_send = QLabel("Send fit results to Graphs Workspace for visualization:")
        left_layout.addWidget(lbl_send)
        
        send_layout = QHBoxLayout()
        
        self.ent_send_df_to_viz = QLineEdit()
        self.ent_send_df_to_viz.setPlaceholderText("DataFrame name")
        self.ent_send_df_to_viz.setText("SPECTRA_best_fit")
        
        self.btn_send = QPushButton("Send to Graphs")
        self.btn_send.setFixedWidth(100)
        self.btn_send.clicked.connect(self._on_send_to_viz)
        
        send_layout.addWidget(self.ent_send_df_to_viz)
        send_layout.addWidget(self.btn_send)
        
        left_layout.addLayout(send_layout)
        
        # Row 5: Save button at bottom
        left_layout.addStretch()
        
        self.btn_save = QPushButton("Save Fit Results")
        self.btn_save.setToolTip("Save fit results to Excel file")
        self.btn_save.setIcon(QIcon(os.path.join(ICON_DIR, "save.png")))
        self.btn_save.setIconSize(QSize(24, 24))
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
    
    def _on_compute_column(self):
        """Emit signal with column name and expression."""
        col_name = self.ent_compute_col_name.text().strip()
        expression = self.ent_compute_expression.text().strip()
        
        if col_name and expression:
            self.compute_column_requested.emit(col_name, expression)
    
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
        
        # Update autocomplete with column names
        if df is not None and not df.empty:
            column_names = list(df.columns)
            model = QStringListModel(column_names)
            self.completer.setModel(model)
    
    def clear_results(self):
        """Clear the results table."""
        self.df_table.clear()
