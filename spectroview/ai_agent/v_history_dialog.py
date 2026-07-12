import os
from typing import Optional, Callable
from datetime import datetime
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QListWidget, QListWidgetItem, QWidget, QMessageBox,
    QSizePolicy, QApplication
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QIcon

from spectroview.ai_agent.m_conversation_store import MConversationStore

class _ConversationItemWidget(QWidget):
    """Custom widget for a row in the history list."""

    on_open = Signal(str)
    on_rename = Signal(str)
    on_duplicate = Signal(str)
    on_delete = Signal(str)

    def __init__(self, conv_id: str, title: str, created_at: str, message_count: int, parent=None):
        super().__init__(parent)
        self.conv_id = conv_id
        self.title = title
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6) # Reduced padding
        layout.setSpacing(8)
        
        # Left side: Title (word wrap to allow ~2 rows)
        self.lbl_title = QLabel(title)
        self.lbl_title.setToolTip(title)
        font = QFont()
        font.setBold(True)
        self.lbl_title.setFont(font)
        self.lbl_title.setStyleSheet("color: white;")
        self.lbl_title.setWordWrap(True)
        self.lbl_title.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.lbl_title.setMinimumHeight(34) # Ensure it visually reserves space for ~2 rows
        self.lbl_title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(self.lbl_title, stretch=1)
        
        # Right side: Grid for timestamp and buttons
        right_widget = QWidget()
        from PySide6.QtWidgets import QGridLayout
        right_layout = QGridLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        
        # Format dates
        try:
            dt = datetime.fromisoformat(created_at)
            date_str = dt.strftime("%y-%m-%d %H:%M")
        except:
            date_str = "Unknown"
            
        lbl_meta = QLabel(f"Created: {date_str}  ·  {message_count} msgs")
        lbl_meta.setStyleSheet("color: gray; font-size: 11px;")
        
        # Buttons
        self.btn_open = QPushButton("Open")
        self.btn_rename = QPushButton("Rename")
        self.btn_duplicate = QPushButton("Duplicate")
        self.btn_delete = QPushButton("Delete")
        
        # Styling
        self.btn_open.setCursor(Qt.PointingHandCursor)
        self.btn_open.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.btn_open.setStyleSheet("""
            QPushButton {
                background-color: #3b429f; color: white; border: none; border-radius: 4px; padding: 4px 12px; font-weight: bold; font-size: 13px;
            }
            QPushButton:hover { background-color: #4b52c0; }
        """)
        
        for btn in (self.btn_rename, self.btn_duplicate):
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton { background: transparent; color: #64B5F6; text-decoration: underline; border: none; font-size: 11px; }
                QPushButton:hover { color: #90CAF9; }
            """)
            
        self.btn_delete.setCursor(Qt.PointingHandCursor)
        self.btn_delete.setStyleSheet("""
            QPushButton { background: transparent; color: #E57373; border: none; font-size: 11px; }
            QPushButton:hover { color: #EF9A9A; }
        """)
        
        self.btn_open.clicked.connect(lambda: self.on_open.emit(self.conv_id))
        self.btn_rename.clicked.connect(lambda: self.on_rename.emit(self.conv_id))
        self.btn_duplicate.clicked.connect(lambda: self.on_duplicate.emit(self.conv_id))
        self.btn_delete.clicked.connect(lambda: self.on_delete.emit(self.conv_id))
        
        # Layout buttons
        # Open button spans 2 rows, 2 columns
        right_layout.addWidget(self.btn_open, 0, 0, 2, 2)
        # Timestamp spans cols 2-4 on row 0
        right_layout.addWidget(lbl_meta, 0, 2, 1, 3, Qt.AlignRight | Qt.AlignBottom)
        # Actions on row 1
        right_layout.addWidget(self.btn_rename, 1, 2, 1, 1, Qt.AlignRight)
        right_layout.addWidget(self.btn_duplicate, 1, 3, 1, 1, Qt.AlignRight)
        right_layout.addWidget(self.btn_delete, 1, 4, 1, 1, Qt.AlignRight)
        
        # Ensure row 1 has enough height so the text isn't cut off
        right_layout.setRowMinimumHeight(1, 20)
        
        right_layout.setColumnStretch(0, 1)
        right_layout.setColumnStretch(1, 1)
        
        layout.addWidget(right_widget, stretch=0)



class VHistoryDialog(QDialog):
    """Dialog showing conversation history."""

    conversation_opened = Signal(str) # Emits conv_id
    
    def __init__(self, store: MConversationStore, parent=None):
        super().__init__(parent)
        self.store = store
        self.setWindowTitle("📜 Conversation History")
        self.setMinimumSize(600, 400)
        self.resize(700, 500)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6) # Reduced dialog padding
        layout.setSpacing(6)
        
        # Search box
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search conversations...")
        self.search_input.textChanged.connect(self._on_search)
        layout.addWidget(self.search_input)
        
        # List
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget { background: #1e1e1e; border: 1px solid #333; outline: 0; }
            QListWidget::item { border-bottom: 1px solid #333; margin: 2px; border-radius: 4px; }
            QListWidget::item:selected { background: #2d2d2d; border: 1px solid #64B5F6; }
        """)
        layout.addWidget(self.list_widget)
        
        # Bottom controls
        btn_layout = QHBoxLayout()
        
        self.btn_delete_all = QPushButton("Delete All")
        self.btn_delete_all.setCursor(Qt.PointingHandCursor)
        self.btn_delete_all.setStyleSheet("""
            QPushButton { background: transparent; color: #E57373; border: 1px solid #E57373; padding: 4px 12px; border-radius: 4px; }
            QPushButton:hover { background: #E57373; color: white; }
        """)
        self.btn_delete_all.clicked.connect(self._on_delete_all)
        btn_layout.addWidget(self.btn_delete_all)
        
        btn_layout.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)
        
        self._populate_list()
        
    def _populate_list(self, filter_text: str = ""):
        self.list_widget.clear()
        self.store.scan_folder()
        summaries = self.store.list_conversations()
        summaries.sort(key=lambda s: s.modified_at or s.created_at, reverse=True)
        
        filter_text = filter_text.lower()
        
        for summary in summaries:
            if filter_text and filter_text not in summary.title.lower():
                continue
                
            item = QListWidgetItem(self.list_widget)
            widget = _ConversationItemWidget(
                conv_id=summary.id,
                title=summary.title,
                created_at=summary.created_at,
                message_count=summary.message_count,
                parent=self.list_widget
            )
            
            # Connect signals
            widget.on_open.connect(self._on_open)
            widget.on_rename.connect(self._on_rename)
            widget.on_duplicate.connect(self._on_duplicate)
            widget.on_delete.connect(self._on_delete)
            
            # Size hint with extra padding to prevent selection frame cut off
            size = widget.sizeHint()
            size.setHeight(size.height() + 6)
            item.setSizeHint(size)
            
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)

    def _on_search(self, text: str):
        self._populate_list(text)
        
    def _on_open(self, conv_id: str):
        self.conversation_opened.emit(conv_id)
        self.accept()
        
    def _on_rename(self, conv_id: str):
        # Find current title
        summary = self.store._index.get(conv_id)
        if not summary:
            return
            
        import PySide6.QtWidgets as QtWidgets
        new_title, ok = QtWidgets.QInputDialog.getText(
            self, "Rename Conversation", "New title:", text=summary.title
        )
        if ok and new_title:
            conv = self.store.load_conversation(conv_id)
            if conv:
                conv.rename(new_title)
                conv.save()
                summary.title = new_title
                self._populate_list(self.search_input.text())
                
    def _on_duplicate(self, conv_id: str):
        conv = self.store.load_conversation(conv_id)
        if conv:
            new_conv = conv.duplicate()
            new_conv.save(self.store.folder_path)
            self.store.scan_folder()
            self._populate_list(self.search_input.text())
            
    def _on_delete(self, conv_id: str):
        self.store.delete_conversation(conv_id)
        self._populate_list(self.search_input.text())

    def _on_delete_all(self):
        for summary in list(self.store.list_conversations()):
            self.store.delete_conversation(summary.id)
        self._populate_list(self.search_input.text())
