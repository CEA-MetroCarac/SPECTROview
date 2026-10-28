from datetime import date, datetime, timedelta
from typing import List, Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction, QFont, QMouseEvent
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from spectroview.ai_agent.m_conversation_store import ConversationSummary, MConversationStore
from spectroview.view.theme import Icons


class _ConversationItemWidget(QWidget):
    """Custom row widget in conversation history list inspired by PLUME."""

    on_open = Signal(str)
    on_pin = Signal(str)
    on_rename = Signal(str)
    on_duplicate = Signal(str)
    on_delete = Signal(str)

    def __init__(
        self,
        conv_id: str,
        title: str,
        timestamp_str: str,
        message_count: int,
        pinned: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.conv_id = conv_id
        self.title = title
        self.pinned = pinned
        self.setCursor(Qt.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(8)

        # Left: Pin icon (if pinned) + Title and Timestamp/Details
        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)

        # Title row (pin icon if pinned + title)
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(6)

        if self.pinned:
            lbl_pin = QLabel()
            lbl_pin.setPixmap(Icons.pin(size=14).pixmap(14, 14))
            lbl_pin.setToolTip("Pinned conversation")
            title_row.addWidget(lbl_pin)

        self.lbl_title = QLabel(title)
        self.lbl_title.setToolTip(title)
        title_font = QFont(self.font())
        title_font.setBold(True)
        title_font.setPointSize(max(10, title_font.pointSize()))
        self.lbl_title.setFont(title_font)
        self.lbl_title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        title_row.addWidget(self.lbl_title, 1)

        text_layout.addLayout(title_row)

        # Timestamp & message count details row
        formatted_dt = "Unknown"
        if timestamp_str:
            try:
                dt = datetime.fromisoformat(timestamp_str)
                formatted_dt = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                formatted_dt = timestamp_str

        msg_label = f"{message_count} message{'s' if message_count != 1 else ''}"
        self.lbl_details = QLabel(f"{formatted_dt}  ·  {msg_label}")
        self.lbl_details.setStyleSheet("color: #8E8E93; font-size: 11px;")
        text_layout.addWidget(self.lbl_details)

        layout.addLayout(text_layout, 1)

        # Right: Action buttons (Pin, Rename, Duplicate, Delete)
        actions_layout = QHBoxLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(4)

        btn_base_style = """
            QPushButton {
                background: transparent;
                border: 1px solid rgba(128, 128, 128, 0.25);
                border-radius: 4px;
                padding: 2px;
            }
            QPushButton:hover {
                background: rgba(128, 128, 128, 0.25);
                border-color: rgba(128, 128, 128, 0.45);
            }
        """

        # Pin / Unpin button
        self.btn_pin = QPushButton()
        self.btn_pin.setIcon(Icons.pin(size=14))
        self.btn_pin.setToolTip("Unpin conversation" if self.pinned else "Pin conversation")
        self.btn_pin.setFixedSize(28, 28)
        self.btn_pin.setCursor(Qt.PointingHandCursor)
        if self.pinned:
            self.btn_pin.setStyleSheet(
                btn_base_style + "QPushButton { background: rgba(91, 140, 255, 0.18); border-color: #5B8CFF; }"
            )
        else:
            self.btn_pin.setStyleSheet(btn_base_style)
        self.btn_pin.clicked.connect(lambda: self.on_pin.emit(self.conv_id))
        actions_layout.addWidget(self.btn_pin)

        # Rename button
        self.btn_rename = QPushButton()
        self.btn_rename.setIcon(Icons.edit(size=14))
        self.btn_rename.setToolTip("Rename conversation")
        self.btn_rename.setFixedSize(28, 28)
        self.btn_rename.setCursor(Qt.PointingHandCursor)
        self.btn_rename.setStyleSheet(btn_base_style)
        self.btn_rename.clicked.connect(lambda: self.on_rename.emit(self.conv_id))
        actions_layout.addWidget(self.btn_rename)

        # Duplicate button
        self.btn_duplicate = QPushButton()
        self.btn_duplicate.setIcon(Icons.copy(size=14))
        self.btn_duplicate.setToolTip("Duplicate conversation")
        self.btn_duplicate.setFixedSize(28, 28)
        self.btn_duplicate.setCursor(Qt.PointingHandCursor)
        self.btn_duplicate.setStyleSheet(btn_base_style)
        self.btn_duplicate.clicked.connect(lambda: self.on_duplicate.emit(self.conv_id))
        actions_layout.addWidget(self.btn_duplicate)

        # Delete button
        self.btn_delete = QPushButton()
        self.btn_delete.setIcon(Icons.trash(size=14))
        self.btn_delete.setToolTip("Delete conversation")
        self.btn_delete.setFixedSize(28, 28)
        self.btn_delete.setCursor(Qt.PointingHandCursor)
        self.btn_delete.setObjectName("btnRowDanger")
        self.btn_delete.setStyleSheet(
            btn_base_style + "QPushButton#btnRowDanger:hover { background: rgba(239, 68, 68, 0.22); border-color: #EF4444; }"
        )
        self.btn_delete.clicked.connect(lambda: self.on_delete.emit(self.conv_id))
        actions_layout.addWidget(self.btn_delete)

        layout.addLayout(actions_layout)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.on_open.emit(self.conv_id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu(self)

        act_open = QAction(Icons.open(size=14), "Open", menu)
        act_open.triggered.connect(lambda: self.on_open.emit(self.conv_id))
        menu.addAction(act_open)

        pin_label = "Unpin Conversation" if self.pinned else "Pin Conversation"
        act_pin = QAction(Icons.pin(size=14), pin_label, menu)
        act_pin.triggered.connect(lambda: self.on_pin.emit(self.conv_id))
        menu.addAction(act_pin)

        act_rename = QAction(Icons.edit(size=14), "Rename", menu)
        act_rename.triggered.connect(lambda: self.on_rename.emit(self.conv_id))
        menu.addAction(act_rename)

        act_dup = QAction(Icons.copy(size=14), "Duplicate", menu)
        act_dup.triggered.connect(lambda: self.on_duplicate.emit(self.conv_id))
        menu.addAction(act_dup)

        menu.addSeparator()

        act_del = QAction(Icons.trash(size=14), "Delete", menu)
        act_del.triggered.connect(lambda: self.on_delete.emit(self.conv_id))
        menu.addAction(act_del)

        menu.exec(event.globalPos())
        event.accept()


class VHistoryDialog(QDialog):
    """Dialog showing conversation history with timestamps, layout, rename, pin, delete, and delete all."""

    conversation_opened = Signal(str)  # Emits conv_id

    def __init__(self, store: MConversationStore, parent=None):
        super().__init__(parent)
        self.store = store
        self._cached_summaries: List[ConversationSummary] = []
        self.setWindowTitle("Conversation History")
        self.setMinimumSize(620, 440)
        self.resize(700, 530)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Search box with clear button
        self.search_input = QLineEdit()
        self.search_input.setFixedHeight(30)
        self.search_input.setPlaceholderText("Search conversations...")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self._on_search)
        layout.addWidget(self.search_input)

        # List
        self.list_widget = QListWidget()
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_widget.setSpacing(2)
        self.list_widget.setStyleSheet(
            "QListWidget { background: transparent; border: 1px solid rgba(128, 128, 128, 0.25); border-radius: 6px; padding: 4px; } "
            "QListWidget::item { border-radius: 5px; margin: 1px 0px; } "
            "QListWidget::item:selected { background-color: rgba(91, 140, 255, 0.25); }"
        )
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.list_widget, stretch=1)

        # Bottom controls
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(8)

        self.btn_delete_all = QPushButton("Delete All")
        self.btn_delete_all.setObjectName("btnDeleteAllOutline")
        self.btn_delete_all.setIcon(Icons.trash(size=14))
        self.btn_delete_all.setCursor(Qt.PointingHandCursor)
        self.btn_delete_all.setToolTip("Delete all unpinned conversations (keeps pinned)")
        self.btn_delete_all.clicked.connect(self._on_delete_all)
        btn_layout.addWidget(self.btn_delete_all)

        btn_layout.addStretch()
        btn_close = QPushButton("Close")
        btn_close.setIcon(Icons.close(size=14))
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

        self._reload_and_render()

    def _reload_and_render(self, filter_text: str = "") -> None:
        """Re-scan the history folder from disk, then render the result."""
        self.store.scan_folder()
        self._cached_summaries = self.store.list_conversations()
        self._render_list(filter_text)

    @staticmethod
    def _date_bucket(summary: ConversationSummary) -> str:
        raw = summary.modified_at or summary.created_at
        try:
            dt = datetime.fromisoformat(raw).date()
        except (TypeError, ValueError):
            return "Older"
        today = date.today()
        if dt == today:
            return "Today"
        if dt == today - timedelta(days=1):
            return "Yesterday"
        if dt >= today - timedelta(days=7):
            return "This Week"
        return "Older"

    def _render_list(self, filter_text: str = "") -> None:
        """Render self._cached_summaries into the list widget with Pinned and Date sections."""
        self.list_widget.clear()
        query = filter_text.strip().lower()

        filtered = [
            s for s in self._cached_summaries
            if not query or query in s.title.lower()
        ]

        has_unpinned = any(not s.pinned for s in self._cached_summaries)
        self.btn_delete_all.setEnabled(has_unpinned)

        if not filtered:
            item = QListWidgetItem("No conversations found.")
            item.setFlags(Qt.NoItemFlags)
            self.list_widget.addItem(item)
            return

        # Separate pinned from unpinned
        pinned_list = [s for s in filtered if s.pinned]
        unpinned_list = [s for s in filtered if not s.pinned]

        if pinned_list:
            self._add_section_header("📌 Pinned")
            for summary in pinned_list:
                self._add_row_item(summary)

        last_bucket = None
        for summary in unpinned_list:
            bucket = self._date_bucket(summary)
            if bucket != last_bucket:
                self._add_section_header(bucket)
                last_bucket = bucket
            self._add_row_item(summary)

    def _add_section_header(self, label_text: str) -> None:
        item = QListWidgetItem(self.list_widget)
        item.setFlags(Qt.NoItemFlags)
        lbl = QLabel(label_text)
        lbl.setObjectName("historyDateHeader")
        lbl.setStyleSheet(
            "font-weight: bold; font-size: 11px; color: #5B8CFF; "
            "padding: 8px 6px 3px 6px; background: transparent;"
        )
        # Give generous explicit height so the header is fully visible and never clipped
        item.setSizeHint(QSize(0, 32))
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, lbl)

    def _add_row_item(self, summary: ConversationSummary) -> None:
        item = QListWidgetItem(self.list_widget)
        item.setData(Qt.UserRole, summary.id)
        widget = _ConversationItemWidget(
            conv_id=summary.id,
            title=summary.title,
            timestamp_str=summary.modified_at or summary.created_at,
            message_count=summary.message_count,
            pinned=summary.pinned,
            parent=self.list_widget,
        )

        widget.on_open.connect(self._on_open)
        widget.on_pin.connect(self._on_pin)
        widget.on_rename.connect(self._on_rename)
        widget.on_duplicate.connect(self._on_duplicate)
        widget.on_delete.connect(self._on_delete)

        # Set comfortable row height for two lines of text + action buttons
        item.setSizeHint(QSize(0, 52))

        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, widget)

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        conv_id = item.data(Qt.UserRole)
        if isinstance(conv_id, str) and conv_id:
            self._on_open(conv_id)

    def _on_search(self, text: str) -> None:
        self._render_list(text)

    def _on_open(self, conv_id: str) -> None:
        self.conversation_opened.emit(conv_id)
        self.accept()

    def _on_pin(self, conv_id: str) -> None:
        summary = self.store.get_summary(conv_id)
        if not summary:
            return
        self.store.set_pinned(conv_id, not summary.pinned)
        self._reload_and_render(self.search_input.text())

    def _on_rename(self, conv_id: str) -> None:
        summary = self.store.get_summary(conv_id)
        if not summary:
            return

        new_title, ok = QInputDialog.getText(
            self, "Rename Conversation", "New title:", text=summary.title
        )
        if ok and new_title.strip():
            conv = self.store.load_conversation(conv_id)
            if conv:
                conv.rename(new_title.strip())
                conv.save(self.store.folder_path)
                self._reload_and_render(self.search_input.text())

    def _on_duplicate(self, conv_id: str) -> None:
        conv = self.store.load_conversation(conv_id)
        if conv:
            new_conv = conv.duplicate()
            new_conv.save(self.store.folder_path)
            self._reload_and_render(self.search_input.text())

    def _on_delete(self, conv_id: str) -> None:
        reply = QMessageBox.question(
            self,
            "Delete Conversation",
            "Permanently delete this conversation?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.store.delete_conversation(conv_id)
            self._reload_and_render(self.search_input.text())

    def _on_delete_all(self) -> None:
        unpinned = [s for s in self.store.list_conversations() if not s.pinned]
        if not unpinned:
            return
        reply = QMessageBox.question(
            self,
            "Delete All Conversations",
            f"Permanently delete all {len(unpinned)} unpinned conversation(s)?\n"
            "Pinned conversations will be kept.\n\n"
            "This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.store.delete_all(keep_pinned=True)
        self._reload_and_render(self.search_input.text())
