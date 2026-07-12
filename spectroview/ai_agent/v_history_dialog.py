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
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # Top row: Title and dates
        top_layout = QHBoxLayout()
        title_disp = title if len(title) <= 50 else title[:47] + "..."
        self.lbl_title = QLabel(f"● {title_disp}")
        self.lbl_title.setToolTip(title)
        font = QFont()
        font.setBold(True)
        self.lbl_title.setFont(font)
        self.lbl_title.setStyleSheet("color: white;")
        top_layout.addWidget(self.lbl_title)
        
        top_layout.addStretch()
        
        # Format dates
        try:
            dt = datetime.fromisoformat(created_at)
            date_str = dt.strftime("%y-%m-%d %H:%M")
        except:
            date_str = "Unknown"
            
        lbl_meta = QLabel(f"Created: {date_str}  ·  {message_count} msgs")
        lbl_meta.setStyleSheet("color: gray; font-size: 11px;")
        top_layout.addWidget(lbl_meta)
        
        layout.addLayout(top_layout)
        
        # Bottom row: Actions
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(10, 0, 0, 0)
        
        self.btn_open = QPushButton("Open")
        self.btn_rename = QPushButton("Rename")
        self.btn_duplicate = QPushButton("Duplicate")
        self.btn_delete = QPushButton("🗑 Delete")
        
        for btn in (self.btn_open, self.btn_rename, self.btn_duplicate, self.btn_delete):
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton { background: transparent; color: #64B5F6; text-decoration: underline; border: none; font-size: 11px; }
                QPushButton:hover { color: #90CAF9; }
            """)
            
        self.btn_delete.setStyleSheet("""
            QPushButton { background: transparent; color: #E57373; border: none; font-size: 11px; }
            QPushButton:hover { color: #EF9A9A; }
        """)
        
        self.btn_open.clicked.connect(lambda: self.on_open.emit(self.conv_id))
        self.btn_rename.clicked.connect(lambda: self.on_rename.emit(self.conv_id))
        self.btn_duplicate.clicked.connect(lambda: self.on_duplicate.emit(self.conv_id))
        self.btn_delete.clicked.connect(lambda: self.on_delete.emit(self.conv_id))
        
        btn_layout.addWidget(self.btn_open)
        btn_layout.addWidget(self.btn_rename)
        btn_layout.addWidget(self.btn_duplicate)
        btn_layout.addWidget(self.btn_delete)
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)


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
        
        # Search box
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search conversations...")
        self.search_input.textChanged.connect(self._on_search)
        layout.addWidget(self.search_input)
        
        # List
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget { background: #1e1e1e; border: 1px solid #333; outline: 0; }
            QListWidget::item { border-bottom: 1px solid #333; }
            QListWidget::item:selected { background: #2d2d2d; }
        """)
        layout.addWidget(self.list_widget)
        
        # Bottom controls
        btn_layout = QHBoxLayout()
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
            
            # Size hint
            item.setSizeHint(widget.sizeHint())
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
        import PySide6.QtWidgets as QtWidgets
        reply = QtWidgets.QMessageBox.question(
            self, "Delete Conversation",
            "Are you sure you want to delete this conversation?\nThis action cannot be undone.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.store.delete_conversation(conv_id)
            self._populate_list(self.search_input.text())
