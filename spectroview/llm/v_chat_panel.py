"""
spectroview/llm/v_chat_panel.py
--------------------------------
View layer: AI Chat floating dialog.

Opens as a non-modal, always-on-top-optional window from the toolbar
button so users can query their data from any workspace tab.

UI layout

MVVM contract
-------------
This file imports VMChat but NEVER imports any other ViewModel.
All data comes in through signals; all actions go out through method calls.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Dict

import pandas as pd
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QScrollArea, QFrame, QComboBox, QSizePolicy,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit, QApplication,
    QSplitter, QTextBrowser
)
import markdown
from PySide6.QtCore import Qt, Signal, QTimer, QSize, QSettings
from PySide6.QtGui import QFont, QIcon, QKeyEvent, QColor, QPalette, QTextCursor

from spectroview import ICON_DIR
from spectroview.llm.vm_chat import VMChat, ChatResult
from spectroview.llm.m_llm_client import LLMClient, API_PROVIDERS, OPENAI_AVAILABLE, OLLAMA_AVAILABLE
from spectroview.llm.v_history_dialog import VHistoryDialog


# ═══════════════════════════════════════════════════════════════════════════
# Helper widgets
# ═══════════════════════════════════════════════════════════════════════════

class _MessageCard(QFrame):
    """A single chat message card with Markdown rendering and actions."""

    reply_clicked = Signal(str) # emits the raw text when reply is clicked

    def __init__(self, text: str, role: str, timestamp: str = "", parent=None) -> None:
        super().__init__(parent)
        self._role = role
        self._raw_text = text
        self.setObjectName(f"card_{role}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        # Header: Role, Timestamp, Actions
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        role_label = QLabel("You" if role == "user" else ("⚠ Error" if role == "error" else "🤖 SPECTROview AI Agent"))
        role_font = QFont()
        role_font.setBold(True)
        role_font.setPointSize(10)
        role_label.setFont(role_font)
        header_layout.addWidget(role_label)

        if timestamp:
            ts_label = QLabel(timestamp)
            ts_label.setStyleSheet("color: gray; font-size: 10px;")
            header_layout.addWidget(ts_label)
            
        header_layout.addStretch()

        if role == "assistant":
            btn_reply = QPushButton("↩ Reply")
            btn_reply.setToolTip("Reply to this message")
            btn_reply.setStyleSheet("color: #1976D2; border: none; font-size: 10px; font-weight: bold; background: transparent;")
            btn_reply.setCursor(Qt.PointingHandCursor)
            btn_reply.clicked.connect(lambda: self.reply_clicked.emit(self._raw_text))
            header_layout.addWidget(btn_reply)

        btn_copy = QPushButton("📋 Copy")
        btn_copy.setToolTip("Copy message text")
        btn_copy.setStyleSheet("color: gray; border: none; font-size: 10px; background: transparent;")
        btn_copy.setCursor(Qt.PointingHandCursor)
        btn_copy.clicked.connect(lambda: QApplication.clipboard().setText(self._raw_text))
        header_layout.addWidget(btn_copy)

        layout.addLayout(header_layout)

        # Content
        if role == "user":
            self.content_view = QLabel(text)
            self.content_view.setWordWrap(True)
            self.content_view.setTextInteractionFlags(Qt.TextSelectableByMouse)
        else:
            self.content_view = QTextBrowser()
            self.content_view.setOpenExternalLinks(False)
            self.content_view.setFrameShape(QFrame.NoFrame)
            self.content_view.setStyleSheet("background: transparent; border: none;")
            # Disable scrollbars to let the card expand
            self.content_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.content_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.content_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            self._update_markdown(text)

        self.content_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.content_view.setMinimumWidth(10)
        self.setMinimumWidth(10)
        layout.addWidget(self.content_view)

        self._apply_style()

    def _update_markdown(self, text: str):
        self._raw_text = text
        if self._role == "user":
            self.content_view.setText(text)
        else:
            # Convert Markdown to HTML
            html = markdown.markdown(text, extensions=['tables', 'fenced_code', 'codehilite'])
            # Add some basic CSS for tables and code blocks
            styled_html = f"""
            <style>
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 10px; }}
                th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }}
                th {{ background-color: rgba(128, 128, 128, 0.2); }}
                code {{ background-color: rgba(128, 128, 128, 0.15); padding: 2px 4px; border-radius: 3px; font-family: monospace; }}
                pre {{ background-color: rgba(0, 0, 0, 0.05); padding: 10px; border-radius: 5px; overflow-x: auto; }}
            </style>
            {html}
            """
            self.content_view.setUpdatesEnabled(False)
            self.content_view.setHtml(styled_html)
            self.content_view.document().setTextWidth(self.content_view.viewport().width())
            self.content_view.setFixedHeight(int(self.content_view.document().size().height()) + 5)
            self.content_view.setUpdatesEnabled(True)

    def set_text(self, text: str) -> None:
        if self._role == "user":
            self._raw_text = text
            self.content_view.setText(text)
        else:
            self._update_markdown(text)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if isinstance(self.content_view, QTextBrowser):
            self.content_view.setUpdatesEnabled(False)
            self.content_view.document().setTextWidth(self.content_view.viewport().width())
            self.content_view.setFixedHeight(int(self.content_view.document().size().height()) + 5)
            self.content_view.setUpdatesEnabled(True)

    def _apply_style(self):
        if self._role == "user":
            self.setStyleSheet("""
                QFrame#card_user {
                    background: rgba(30, 100, 200, 0.15);
                    border: 1px solid rgba(30, 100, 200, 0.3);
                    border-radius: 8px;
                }
            """)
        elif self._role == "error":
            self.setStyleSheet("""
                QFrame#card_error {
                    background: rgba(200, 50, 50, 0.15);
                    border: 1px solid rgba(200, 50, 50, 0.4);
                    border-radius: 8px;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame#card_assistant {
                    background: rgba(60, 180, 100, 0.10);
                    border: 1px solid rgba(60, 180, 100, 0.25);
                    border-radius: 8px;
                }
            """)


class _ThinkingDots(QLabel):
    """Animated '...' label shown while the LLM is thinking."""

    def __init__(self, parent=None) -> None:
        super().__init__("🤖  Thinking .", parent)
        self._dots = 1
        self._timer = QTimer(self)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self._tick)
        self.setStyleSheet("color: gray; font-style: italic; padding: 4px 12px;")

    def start(self):
        self._dots = 1
        self._timer.start()
        self.show()

    def stop(self):
        self._timer.stop()
        self.hide()

    def _tick(self):
        self._dots = (self._dots % 3) + 1
        self.setText(f"🤖  Thinking {'.' * self._dots}")


class _DataFramePreview(QWidget):
    """Compact table widget to display a filtered DataFrame inline."""

    MAX_ROWS = 50      # cap displayed rows for performance
    MAX_COLS = 15      # cap columns for readability

    def __init__(self, df: pd.DataFrame, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(2)

        display_df = df.iloc[: self.MAX_ROWS, : self.MAX_COLS]

        info = QLabel(
            f"Showing {len(display_df)} of {len(df)} row(s), "
            f"{len(df.columns)} column(s)"
        )
        info.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(info)

        table = QTableWidget(len(display_df), len(display_df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in display_df.columns])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setMaximumHeight(220)

        for r, (_, row) in enumerate(display_df.iterrows()):
            for c, val in enumerate(row):
                cell_text = f"{val:.4g}" if isinstance(val, float) else str(val)
                item = QTableWidgetItem(cell_text)
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(r, c, item)

        # Copy button
        btn_copy = QPushButton("📋 Copy table")
        btn_copy.setMaximumWidth(110)
        btn_copy.setStyleSheet("font-size: 10px;")
        btn_copy.clicked.connect(lambda: self._copy_to_clipboard(df))

        layout.addWidget(table)
        layout.addWidget(btn_copy)

    @staticmethod
    def _copy_to_clipboard(df: pd.DataFrame):
        QApplication.clipboard().setText(df.to_csv(sep="\t", index=False))


# ═══════════════════════════════════════════════════════════════════════════
# Main floating dialog
# ═══════════════════════════════════════════════════════════════════════════

# Provider entries shown in the combobox (Ollama first)
_PROVIDER_ENTRIES = ["Ollama (local)"] + list(API_PROVIDERS.keys())
# Map display name → internal provider key
_DISPLAY_TO_PROVIDER = {
    "Ollama (local)": "Ollama",
    **{k: k for k in API_PROVIDERS.keys()},
}
_PROVIDER_TO_DISPLAY = {v: k for k, v in _DISPLAY_TO_PROVIDER.items()}


class VChatPanel(QDialog):
    """Floating AI Chat panel — opened from the toolbar button.

    Signals
    -------
    plot_requested(dict)
        Emitted when the AI suggests a plot.  The dict contains keys
        compatible with ``VWorkspaceGraphs`` plot configuration.
    """

    plot_requested = Signal(dict)

    _SETTINGS_GROUP = "ai_chat"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("🤖  SPECTROview AI Agent")
        self.setWindowFlags(
            Qt.Dialog | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint
        )
        self.setMinimumSize(400, 500)
        self.resize(450, 740)

        self.vm = VMChat(self)
        self._active_card: Optional[_MessageCard] = None  # streaming target
        self._reply_to_index: Optional[int] = None

        self._build_ui()
        self._connect_signals()
        self._load_settings()          # restore persisted provider/key
        self._refresh_status()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header bar ──────────────────────────────────────────────
        header = self._make_header()
        root.addWidget(header)

        # ── Status bar ──────────────────────────────────────────────
        self.status_bar = self._make_status_bar()
        root.addWidget(self.status_bar)

        # ── Chat scroll area ────────────────────────────────────────
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.messages_container = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_container)
        self.messages_layout.setContentsMargins(8, 8, 8, 8)
        self.messages_layout.setSpacing(6)
        self.messages_layout.addStretch()   # pushes messages to top initially

        self.scroll_area.setWidget(self.messages_container)
        root.addWidget(self.scroll_area, stretch=1)

        # ── Thinking dots ───────────────────────────────────────────
        self.thinking_dots = _ThinkingDots(self)
        root.addWidget(self.thinking_dots)
        self.thinking_dots.hide()

        # ── Input Area ──────────────────────────────────────────────
        self.input_container = QWidget()
        input_vbox = QVBoxLayout(self.input_container)
        input_vbox.setContentsMargins(0, 0, 0, 0)
        input_vbox.setSpacing(0)
        
        self.reply_preview = self._make_reply_preview()
        self.reply_preview.hide()
        input_vbox.addWidget(self.reply_preview)
        
        input_bar = self._make_input_bar()
        input_vbox.addWidget(input_bar)
        
        root.addWidget(self.input_container)

    def _make_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("chatHeader")
        header.setStyleSheet("""
            QFrame#chatHeader {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a237e, stop:1 #283593
                );
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }
        """)
        header.setFixedHeight(48)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(6)


        # ── Provider selector ────────────────────────────────────────
        lbl_provider = QLabel("Provider:")
        lbl_provider.setStyleSheet("color: rgba(255,255,255,0.7); font-size: 11px;")
        layout.addWidget(lbl_provider)

        self.cbb_provider = QComboBox()
        self.cbb_provider.addItems(_PROVIDER_ENTRIES)
        self.cbb_provider.setMinimumWidth(120)
        self.cbb_provider.setToolTip("Select the LLM provider to use")
        self.cbb_provider.setStyleSheet("""
            QComboBox {
                background: rgba(255,255,255,0.15);
                color: white;
                border: 1px solid rgba(255,255,255,0.3);
                border-radius: 4px;
                padding: 2px 6px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { color: black; }
        """)
        layout.addWidget(self.cbb_provider)

        # ── Model selector ───────────────────────────────────────────
        self.cbb_model = QComboBox()
        self.cbb_model.setMinimumWidth(150)
        self.cbb_model.setToolTip("Select the model to use")
        self.cbb_model.setStyleSheet("""
            QComboBox {
                background: rgba(255,255,255,0.15);
                color: white;
                border: 1px solid rgba(255,255,255,0.3);
                border-radius: 4px;
                padding: 2px 6px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { color: black; }
        """)
        layout.addWidget(self.cbb_model)

        # ── Refresh button ───────────────────────────────────────────
        self.btn_refresh_models = QPushButton("")
        self.btn_refresh_models.setIcon(QIcon(os.path.join(ICON_DIR, "refresh.png")))
        self.btn_refresh_models.setIconSize(QSize(20, 20))
        self.btn_refresh_models.setFixedSize(28, 28)
        self.btn_refresh_models.setToolTip("Refresh model list / re-check connection")
        self.btn_refresh_models.setStyleSheet(
            "QPushButton { background: transparent; border: none; border-radius: 4px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.2); }"
        )
        layout.addWidget(self.btn_refresh_models)
        
        layout.addStretch()

        self.btn_history = QPushButton("")
        self.btn_history.setIcon(QIcon(os.path.join(ICON_DIR, "view-details.png")))
        self.btn_history.setIconSize(QSize(20, 20))
        self.btn_history.setFixedSize(28, 28)
        self.btn_history.setToolTip("Conversation History")

        self.btn_new_chat = QPushButton("")
        self.btn_new_chat.setIcon(QIcon(os.path.join(ICON_DIR, "ai_chat.png")))
        self.btn_new_chat.setIconSize(QSize(20, 20))
        self.btn_new_chat.setFixedSize(28, 28)
        self.btn_new_chat.setToolTip("New Chat")
        
        for btn in (self.btn_history, self.btn_new_chat):
            btn.setStyleSheet("""
                QPushButton { background: transparent; border: none; border-radius: 4px; }
                QPushButton:hover { background: rgba(255,255,255,0.2); }
            """)
            btn.setCursor(Qt.PointingHandCursor)
            layout.addWidget(btn)

        return header



    def _make_status_bar(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(28)
        bar.setObjectName("chatStatusBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 0, 10, 0)

        self.lbl_status = QLabel("Checking…")
        self.lbl_status.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.lbl_status)
        layout.addStretch()

        # No-data notice (shown when no df is loaded)
        self.lbl_no_data = QLabel("ⓘ No DataFrame selected")
        self.lbl_no_data.setStyleSheet("color: #FFA726; font-size: 11px;")
        layout.addWidget(self.lbl_no_data)

        return bar

    def _make_reply_preview(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("replyPreview")
        frame.setStyleSheet("""
            QFrame#replyPreview {
                background: #2a2a2a;
                border-top: 1px solid rgba(128,128,128,0.3);
                border-left: 3px solid #1976D2;
                margin: 0px;
            }
        """)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(10, 6, 10, 6)
        
        self.lbl_reply_text = QLabel()
        self.lbl_reply_text.setStyleSheet("color: #aaa; font-size: 11px;")
        self.lbl_reply_text.setWordWrap(True)
        layout.addWidget(self.lbl_reply_text, stretch=1)
        
        btn_close = QPushButton("✕")
        btn_close.setFixedSize(20, 20)
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.setStyleSheet("QPushButton { border: none; color: gray; font-weight: bold; } QPushButton:hover { color: white; }")
        btn_close.clicked.connect(self._clear_reply_state)
        layout.addWidget(btn_close)
        
        return frame

    def _make_input_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("chatInputBar")
        bar.setStyleSheet("""
            QFrame#chatInputBar {
                border-top: 1px solid rgba(128,128,128,0.3);
                padding: 2px;
            }
        """)
        bar.setFixedHeight(68)

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        # Clear button
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setFixedWidth(55)
        self.btn_clear.setFixedHeight(34)
        self.btn_clear.setToolTip("Clear conversation history")
        layout.addWidget(self.btn_clear)

        # Text input
        self.edit_input = _ChatLineEdit()
        self.edit_input.setPlaceholderText("Ask a question about your data…")
        self.edit_input.setFixedHeight(50)
        layout.addWidget(self.edit_input, stretch=1)

        # Send button
        self.btn_send = QPushButton("Send ▶")
        self.btn_send.setFixedHeight(50)
        self.btn_send.setFixedWidth(70)
        self.btn_send.setStyleSheet("""
            QPushButton {
                background: #1565C0;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover   { background: #1976D2; }
            QPushButton:pressed { background: #0D47A1; }
            QPushButton:disabled { background: #555; color: #888; }
        """)
        layout.addWidget(self.btn_send)

        return bar

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        # Input
        self.btn_send.clicked.connect(self._on_send)
        self.edit_input.send_requested.connect(self._on_send)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_history.clicked.connect(self._on_history_clicked)
        self.btn_new_chat.clicked.connect(self._on_new_chat_clicked)
        self.btn_refresh_models.clicked.connect(self._refresh_status)
        self.cbb_model.currentTextChanged.connect(self.vm.set_model)
        self.cbb_provider.currentTextChanged.connect(self._on_provider_changed)

        self.cbb_model.currentTextChanged.connect(lambda _: self._save_settings())

        # ViewModel → View
        self.vm.thinking_changed.connect(self._on_thinking_changed)
        self.vm.chunk_received.connect(self._on_chunk)
        self.vm.result_ready.connect(self._on_result_ready)
        self.vm.error_occurred.connect(self._on_error)
        self.vm.conversation_changed.connect(self._on_conversation_changed)

    # ------------------------------------------------------------------
    # Public API — called from main.py / VWorkspaceGraphs
    # ------------------------------------------------------------------

    def set_dataframes(self, dfs: Dict[str, pd.DataFrame], active_name: str = "") -> None:
        """Update the available DataFrames the chat can query."""
        self.vm.set_dataframes(dfs, active_name)
        if dfs:
            self.lbl_no_data.setText(
                f"📊 {len(dfs)} DataFrame(s) loaded  (Active: {active_name})" if active_name else f"📊 {len(dfs)} DataFrame(s) loaded"
            )
            self.lbl_no_data.setStyleSheet("color: #66BB6A; font-size: 11px;")
        else:
            self.lbl_no_data.setText("ⓘ No DataFrames available")
            self.lbl_no_data.setStyleSheet("color: #FFA726; font-size: 11px;")

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _load_settings(self) -> None:
        """Restore provider, API keys, and model from QSettings."""
        s = QSettings("SPECTROview", "AIChat")
        s.beginGroup(self._SETTINGS_GROUP)

        saved_provider = s.value("provider", "Ollama (local)")
        idx = self.cbb_provider.findText(saved_provider)
        if idx >= 0:
            self.cbb_provider.blockSignals(True)
            self.cbb_provider.setCurrentIndex(idx)
            self.cbb_provider.blockSignals(False)

        s.endGroup()

        # Apply the loaded provider
        self._on_provider_changed(saved_provider, restore=True)

    def _save_settings(self) -> None:
        """Persist current provider, API key, and model to QSettings."""
        s = QSettings("SPECTROview", "AIChat")
        s.beginGroup(self._SETTINGS_GROUP)

        provider_display = self.cbb_provider.currentText()
        provider_key = _DISPLAY_TO_PROVIDER.get(provider_display, "Ollama")

        s.setValue("provider", provider_display)

        current_model = self.cbb_model.currentText()
        if current_model:
            s.setValue(f"model_{provider_key}", current_model)

        s.endGroup()

    # ------------------------------------------------------------------
    # Provider change handler
    # ------------------------------------------------------------------

    def _on_provider_changed(self, display_name: str, restore: bool = False) -> None:
        """Update the ViewModel provider."""
        provider_key = _DISPLAY_TO_PROVIDER.get(display_name, "Ollama")
        is_ollama = (provider_key == "Ollama")

        if is_ollama:
            self.vm.set_provider("Ollama")
            self._refresh_status()
        else:
            self._apply_cloud_provider(provider_key)

    def _apply_cloud_provider(self, provider_key: str) -> None:
        """Push provider config to ViewModel and refresh the model list."""
        s = QSettings("SPECTROview", "AIChat")
        s.beginGroup(self._SETTINGS_GROUP)
        api_key = s.value(f"api_key_{provider_key}", "")
        base_url = s.value("custom_base_url", "") if provider_key == "Custom" else ""
        s.endGroup()
        
        model = self.cbb_model.currentText()

        self.vm.set_provider(provider_key, api_key=api_key, base_url=base_url, model=model)
        self._refresh_status()

    # ------------------------------------------------------------------
    # Refresh helpers
    # ------------------------------------------------------------------

    def _refresh_status(self) -> None:
        """Check availability, update status bar and model list."""
        provider_display = self.cbb_provider.currentText()
        provider_key = _DISPLAY_TO_PROVIDER.get(provider_display, "Ollama")
        is_ollama = (provider_key == "Ollama")

        # ── Package availability guard ──────────────────────────────────
        if is_ollama and not OLLAMA_AVAILABLE:
            self.lbl_status.setText("🔴  ollama package missing — run: pip install ollama")
            self.lbl_status.setStyleSheet("color: #EF5350; font-size: 11px;")
            self.edit_input.setEnabled(False)
            self.btn_send.setEnabled(False)
            return

        if not is_ollama and not OPENAI_AVAILABLE:
            self.lbl_status.setText("🔴  openai package missing — run: pip install openai")
            self.lbl_status.setStyleSheet("color: #EF5350; font-size: 11px;")
            self.edit_input.setEnabled(False)
            self.btn_send.setEnabled(False)
            return

        # ── API key missing ─────────────────────────────────────
        s = QSettings("SPECTROview", "AIChat")
        s.beginGroup(self._SETTINGS_GROUP)
        has_key = bool(s.value(f"api_key_{provider_key}", ""))
        s.endGroup()
        if not is_ollama and not has_key:
            self.lbl_status.setText(f"⚪  {provider_key} — configure API key in Settings (AI Tab)")
            self.lbl_status.setStyleSheet("color: #FFA726; font-size: 11px;")
            self.edit_input.setEnabled(False)
            self.btn_send.setEnabled(False)
            return

        available = self.vm.is_available()

        if available:
            models = self.vm.get_models()
            self.cbb_model.blockSignals(True)
            current = self.cbb_model.currentText()
            self.cbb_model.clear()
            if models:
                self.cbb_model.addItems(models)
            else:
                fallback = (
                    LLMClient.DEFAULT_MODEL
                    if is_ollama
                    else API_PROVIDERS.get(provider_key, {}).get("default_model", "")
                )
                if fallback:
                    self.cbb_model.addItem(fallback)

            # Restore previous selection if still present
            s = QSettings("SPECTROview", "AIChat")
            s.beginGroup(self._SETTINGS_GROUP)
            saved_model = str(s.value(f"model_{provider_key}", ""))
            s.endGroup()
            
            target_model = saved_model if saved_model else current
            idx = self.cbb_model.findText(target_model)
            if idx >= 0:
                self.cbb_model.setCurrentIndex(idx)
            elif self.cbb_model.findText(current) >= 0:
                self.cbb_model.setCurrentIndex(self.cbb_model.findText(current))
            if self.cbb_model.currentText():
                self.vm.set_model(self.cbb_model.currentText())
            self.cbb_model.blockSignals(False)

            if is_ollama:
                self.lbl_status.setText("🟢  Ollama connected")
            else:
                self.lbl_status.setText(f"🟢  {provider_key} API connected")
            self.lbl_status.setStyleSheet("color: #66BB6A; font-size: 11px;")
            self.edit_input.setEnabled(True)
            self.btn_send.setEnabled(True)

        else:
            if is_ollama:
                self.lbl_status.setText("🔴  Ollama not running — run: ollama serve")
            else:
                self.lbl_status.setText(f"🔴  {provider_key} API — invalid or expired API key")
            self.lbl_status.setStyleSheet("color: #EF5350; font-size: 11px;")
            self.edit_input.setEnabled(False)
            self.btn_send.setEnabled(False)

    # ------------------------------------------------------------------
    # Slot handlers
    # ------------------------------------------------------------------

    def _on_send(self) -> None:
        text = self.edit_input.toPlainText().strip()
        if not text or self.vm.is_busy():
            return

        self.edit_input.clear()
        
        reply_idx = self._reply_to_index
        self._clear_reply_state()
        
        self._add_user_card(text, reply_idx, timestamp=datetime.now().isoformat())
        self._active_card = self._add_ai_card("", timestamp=datetime.now().isoformat())  # placeholder for streaming
        
        # The AI message will be added to the conversation at this index once done
        ai_msg_idx = self.vm._conversation.message_count + 1 
        self._active_card.reply_clicked.connect(lambda text, idx=ai_msg_idx: self._on_reply_clicked(idx, text))
        
        self.vm.process_query(text, reply_to_index=reply_idx)

    def _on_clear(self) -> None:
        self.vm.clear_history()

    def _on_history_clicked(self) -> None:
        dialog = VHistoryDialog(self.vm.conversation_store, self)
        dialog.conversation_opened.connect(self._on_conversation_opened)
        dialog.exec()

    def _on_new_chat_clicked(self) -> None:
        self.vm.new_conversation()

    def _on_conversation_opened(self, conv_id: str) -> None:
        conv = self.vm.conversation_store.load_conversation(conv_id)
        if conv:
            self.vm.load_conversation(conv)
            
    def _on_conversation_changed(self, title: str) -> None:
        # Clear UI
        while self.messages_layout.count() > 1: # preserve the stretch
            item = self.messages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        # Rebuild from vm conversation
        for i, msg in enumerate(self.vm._conversation.messages):
            if msg["role"] == "user":
                self._add_user_card(msg["content"], msg.get("reply_to_index"), timestamp=msg.get("timestamp"))
            elif msg["role"] == "assistant":
                card = self._add_ai_card(msg["content"], timestamp=msg.get("timestamp"))
                card.reply_clicked.connect(lambda text, idx=i: self._on_reply_clicked(idx, text))
            elif msg["role"] == "error":
                card = _MessageCard(msg["content"], "error", timestamp=msg.get("timestamp"))
                self._insert_before_stretch(card)
                
        self._scroll_to_bottom()

    def _on_reply_clicked(self, idx: int, raw_text: str) -> None:
        self._reply_to_index = idx
        preview = raw_text[:80].replace("\n", " ") + "..." if len(raw_text) > 80 else raw_text.replace("\n", " ")
        self.lbl_reply_text.setText(f"Replying to: \"{preview}\"")
        self.reply_preview.show()
        self.edit_input.setFocus()
        
    def _clear_reply_state(self) -> None:
        self._reply_to_index = None
        self.reply_preview.hide()

    def _on_thinking_changed(self, is_thinking: bool) -> None:
        if is_thinking:
            self.thinking_dots.start()
            self.btn_send.setEnabled(False)
        else:
            self.thinking_dots.stop()
            self.btn_send.setEnabled(True)

    def _on_chunk(self, fragment: str) -> None:
        """Append a streaming fragment to the active AI bubble."""
        if self._active_card:
            current = self._active_card.content_view.toPlainText() if hasattr(self._active_card.content_view, 'toPlainText') else self._active_card.content_view.text()
            self._active_card.set_text(current + fragment)
            self._scroll_to_bottom()

    def _on_result_ready(self, result: ChatResult) -> None:
        """Replace the streaming bubble with the full structured result."""
        if self._active_card:
            # Set the explanation as the main text
            self._active_card.set_text(result.explanation or result.text_summary)

        # If a DataFrame was returned, attach a preview table below the bubble
        if result.dataframe is not None and not result.dataframe.empty:
            preview = _DataFramePreview(result.dataframe)
            self._insert_widget_after_card(preview)

        # If statistics / answer text, show in a text box
        elif result.text_summary and result.action in ("statistics", "answer"):
            if self._active_card:
                self._active_card.set_text(
                    (result.explanation or "") + "\n\n" + result.text_summary
                )

        # If a plot was suggested, update requested, or delete requested, automatically apply it
        if result.plot_config and result.action in ("plot", "update", "delete"):
            cfgs = result.plot_config if isinstance(result.plot_config, list) else [result.plot_config]

            for cfg in cfgs:
                self.plot_requested.emit(cfg)

            if result.action == "plot":
                btn = QPushButton(f"📊 Re-apply plot suggestion{'s' if len(cfgs) > 1 else ''}")
                btn.setMaximumWidth(220)
                btn.setStyleSheet(
                    "QPushButton { background: #283593; color: white; border-radius: 4px; padding: 4px 8px; }"
                    "QPushButton:hover { background: #3949AB; }"
                )
                btn.clicked.connect(lambda: [self.plot_requested.emit(c) for c in cfgs])
                self._insert_widget_after_card(btn)

        self._active_card = None
        self._scroll_to_bottom()

    def _on_error(self, message: str) -> None:
        # Replace placeholder bubble with an error bubble
        if self._active_card:
            self._active_card.deleteLater()
            self._active_card = None
        card = _MessageCard(message, "error")
        self._insert_before_stretch(card)
        self._scroll_to_bottom()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_user_card(self, text: str, reply_to_index: Optional[int] = None, timestamp: Optional[str] = None) -> _MessageCard:
        if reply_to_index is not None and 0 <= reply_to_index < len(self.vm._conversation.messages):
            replied_msg = self.vm._conversation.messages[reply_to_index]
            replied_content = replied_msg.get("content", "")
            preview = replied_content[:80].replace("\n", " ") + "..." if len(replied_content) > 80 else replied_content.replace("\n", " ")
            text = f"Replying to:\n\"{preview}\"\n\n{text}"
            
        formatted_ts = ""
        if timestamp:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp)
                formatted_ts = dt.strftime("%y-%m-%d %H:%M")
            except Exception:
                formatted_ts = timestamp
                
        card = _MessageCard(text, "user", timestamp=formatted_ts)
        wrapper = QWidget()
        l = QHBoxLayout(wrapper)
        l.setContentsMargins(40, 0, 4, 0)
        l.addWidget(card)
        self._insert_before_stretch(wrapper)
        self._scroll_to_bottom()
        return card

    def _add_ai_card(self, text: str, timestamp: Optional[str] = None) -> _MessageCard:
        formatted_ts = ""
        if timestamp:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp)
                formatted_ts = dt.strftime("%y-%m-%d %H:%M")
            except Exception:
                formatted_ts = timestamp
                
        card = _MessageCard(text, "assistant", timestamp=formatted_ts)
        wrapper = QWidget()
        l = QHBoxLayout(wrapper)
        l.setContentsMargins(4, 0, 40, 0)
        l.addWidget(card)
        self._insert_before_stretch(wrapper)
        self._scroll_to_bottom()
        return card

    def _insert_before_stretch(self, widget: QWidget) -> None:
        """Insert a widget just before the trailing stretch."""
        count = self.messages_layout.count()
        self.messages_layout.insertWidget(count - 1, widget)

    def _insert_widget_after_card(self, widget: QWidget) -> None:
        """Insert a widget just after the last _MessageCard."""
        count = self.messages_layout.count()
        self.messages_layout.insertWidget(count - 1, widget)

    def _scroll_to_bottom(self) -> None:
        QTimer.singleShot(
            30,
            lambda: self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            ),
        )

    # ------------------------------------------------------------------
    # Close event — hide instead of destroy so state is preserved
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._save_settings()
        self.vm.cancel()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────
# Subclassed QLineEdit to emit Enter-key as a signal
# ─────────────────────────────────────────────────────────────────────────

class _ChatLineEdit(QTextEdit):
    """A QTextEdit that emits ``send_requested`` on Enter / Return (but Shift+Enter adds newline)."""

    send_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptRichText(False)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
            else:
                self.send_requested.emit()
        else:
            super().keyPressEvent(event)



# ─────────────────────────────────────────────────────────────────────────
# Re-export LLMClient so main.py can reference it easily
# ─────────────────────────────────────────────────────────────────────────
from spectroview.llm.m_llm_client import LLMClient  # noqa: E402
