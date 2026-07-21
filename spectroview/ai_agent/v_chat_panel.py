"""
spectroview/ai_agent/v_chat_panel.py
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
    QTextBrowser, QInputDialog, QMessageBox
)
import markdown
from PySide6.QtCore import Qt, Signal, QTimer, QSize, QSettings, QThread, QObject
from PySide6.QtGui import QFont, QIcon, QKeyEvent, QTextCursor

from spectroview import ICON_DIR
from spectroview.ai_agent.vm_chat import VMChat, ChatResult
from spectroview.ai_agent.m_llm_client import API_PROVIDERS, OPENAI_AVAILABLE, OLLAMA_AVAILABLE
from spectroview.ai_agent.v_history_dialog import VHistoryDialog


# ═══════════════════════════════════════════════════════════════════════════
# Helper widgets
# ═══════════════════════════════════════════════════════════════════════════

# Rendered once per message update rather than rebuilt as a literal string
# each time — colors are translucent grays so they read correctly against
# both the card's dark-theme and light-theme background.
_MARKDOWN_CSS = """
<style>
    table { border-collapse: collapse; width: 100%; margin-bottom: 10px; }
    th, td { border: 1px solid rgba(128, 128, 128, 0.3); padding: 6px; text-align: left; }
    th { background-color: rgba(128, 128, 128, 0.2); }
    code { background-color: rgba(128, 128, 128, 0.15); padding: 2px 4px; border-radius: 3px; font-family: monospace; }
    pre { background-color: rgba(128, 128, 128, 0.10); padding: 10px; border-radius: 5px; overflow-x: auto; }
</style>
"""


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

    def _update_markdown(self, text: str):
        self._raw_text = text
        if self._role == "user":
            self.content_view.setText(text)
        else:
            # Convert Markdown to HTML
            html = markdown.markdown(text, extensions=['tables', 'fenced_code', 'codehilite'])
            self.content_view.setUpdatesEnabled(False)
            self.content_view.setHtml(_MARKDOWN_CSS + html)
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


class _ThinkingDots(QLabel):
    """Animated '...' label shown while the LLM is thinking."""

    def __init__(self, parent=None) -> None:
        super().__init__("🤖  Thinking .", parent)
        self.setObjectName("chatThinkingDots")
        self._dots = 1
        self._text_prefix = "Thinking"
        self._timer = QTimer(self)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self._tick)
        self.setStyleSheet("font-style: italic; padding: 4px 12px;")

    def start(self, text_prefix: str = "Thinking"):
        self._text_prefix = text_prefix
        self._dots = 1
        self._timer.start()
        self.show()

    def stop(self):
        self._timer.stop()
        self.hide()

    def _tick(self):
        self._dots = (self._dots % 3) + 1
        self.setText(f"🤖  {self._text_prefix} {'.' * self._dots}")


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
# Voice dictation — optional, graceful degradation if deps not installed
# ═══════════════════════════════════════════════════════════════════════════

try:
    import speech_recognition as _sr  # type: ignore[import-untyped]
    VOICE_AVAILABLE = True
except ImportError:
    _sr = None  # type: ignore[assignment]
    VOICE_AVAILABLE = False


class VoiceWorker(QThread):
    """Records from the default microphone and transcribes via Google Speech.

    Runs entirely in a background thread so the UI stays responsive.
    Emits the transcribed text when done, or an error message on failure.
    """

    text_ready = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        if not VOICE_AVAILABLE:
            self.error_occurred.emit(
                "Voice dictation unavailable.\n"
                "Install:  pip install SpeechRecognition pyaudio"
            )
            return

        recognizer = _sr.Recognizer()
        try:
            with _sr.Microphone() as source:
                if self._is_cancelled:
                    return
                # Brief ambient noise calibration
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                if self._is_cancelled:
                    return
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
        except OSError as exc:
            self.error_occurred.emit(f"No microphone found: {exc}")
            return
        except Exception as exc:
            self.error_occurred.emit(f"Recording failed: {exc}")
            return

        if self._is_cancelled:
            return

        try:
            text = recognizer.recognize_google(audio)
            self.text_ready.emit(text)
        except _sr.UnknownValueError:
            self.error_occurred.emit("Could not understand audio. Please try again.")
        except _sr.RequestError as exc:
            self.error_occurred.emit(f"Speech recognition service error: {exc}")
        except Exception as exc:
            self.error_occurred.emit(f"Transcription failed: {exc}")


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
        self.setWindowTitle("SPECTROview AI Agent")
        self.setWindowFlags(
            Qt.Dialog | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint
        )
        self.setMinimumSize(400, 500)
        self.resize(450, 740)

        self.vm = VMChat(self)
        self._active_card: Optional[_MessageCard] = None  # streaming target
        self._reply_to_index: Optional[int] = None
        self._voice_worker: Optional[VoiceWorker] = None

        # Streaming re-render throttle: markdown parsing + HTML re-layout
        # is too expensive to redo on every single streamed token, so
        # fragments are buffered and flushed at a capped rate instead.
        self._stream_buffer: str = ""
        self._chunk_flush_timer = QTimer(self)
        self._chunk_flush_timer.setSingleShot(True)
        self._chunk_flush_timer.timeout.connect(self._flush_chunk_buffer)

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

        # ── Scroll-to-bottom floating button ──────────────────────────
        # Shown only once the user has scrolled away from the bottom
        # while a response streams in below the visible area.
        self.btn_scroll_to_bottom = QPushButton("↓", self.scroll_area)
        self.btn_scroll_to_bottom.setObjectName("scrollToBottomBtn")
        self.btn_scroll_to_bottom.setFixedSize(32, 32)
        self.btn_scroll_to_bottom.setCursor(Qt.PointingHandCursor)
        self.btn_scroll_to_bottom.setToolTip("Scroll to bottom")
        self.btn_scroll_to_bottom.clicked.connect(self._scroll_to_bottom)
        self.btn_scroll_to_bottom.hide()
        self.scroll_area.verticalScrollBar().valueChanged.connect(
            self._update_scroll_to_bottom_visibility
        )
        self.scroll_area.verticalScrollBar().rangeChanged.connect(
            self._update_scroll_to_bottom_visibility
        )

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

    def _make_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("chatHeader")
        header.setFixedHeight(112)

        # Main layout holds three rows: connection controls, action
        # buttons, and the conversation title — split across rows rather
        # than crammed into one so nothing gets clipped at the panel's
        # default width.
        main_layout = QVBoxLayout(header)
        main_layout.setContentsMargins(10, 6, 10, 6)
        main_layout.setSpacing(4)

        # ── Row 1: Connection (provider / model / prompt tier) ─────────
        row1_layout = QHBoxLayout()
        row1_layout.setContentsMargins(0, 0, 0, 0)
        row1_layout.setSpacing(6)

        self.lbl_provider = QLabel("Provider:")
        self.lbl_provider.setStyleSheet("color: gray; font-weight: bold; font-size: 11px;")
        row1_layout.addWidget(self.lbl_provider)

        self.cbb_provider = QComboBox()
        for display in _DISPLAY_TO_PROVIDER.keys():
            self.cbb_provider.addItem(display)
        self.cbb_provider.setMinimumWidth(90)
        row1_layout.addWidget(self.cbb_provider, stretch=1)

        self.cbb_model = QComboBox()
        self.cbb_model.setMinimumWidth(110)
        # Editable so a model name can be typed directly — some endpoints
        # (custom OpenAI-compatible ones especially) don't expose a
        # model-listing API, so the dropdown would otherwise be empty.
        self.cbb_model.setEditable(True)
        self.cbb_model.setInsertPolicy(QComboBox.NoInsert)
        self.cbb_model.setToolTip("Select or type a model name")
        row1_layout.addWidget(self.cbb_model, stretch=1)

        self.cbb_prompt_tier = QComboBox()
        self.cbb_prompt_tier.addItems(["Auto", "Full prompt", "Simplified prompt"])
        self.cbb_prompt_tier.setMinimumWidth(80)
        self.cbb_prompt_tier.setToolTip(
            "Prompt tier for local Ollama models.\n"
            "Auto: detected automatically from the model's size.\n"
            "Full / Simplified: force a tier regardless of detection."
        )
        row1_layout.addWidget(self.cbb_prompt_tier, stretch=1)

        self.btn_refresh_models = QPushButton("")
        self.btn_refresh_models.setObjectName("btn_refresh_models")
        self.btn_refresh_models.setIcon(QIcon(os.path.join(ICON_DIR, "refresh.png")))
        self.btn_refresh_models.setIconSize(QSize(20, 20))
        self.btn_refresh_models.setFixedSize(28, 28)
        self.btn_refresh_models.setToolTip("Refresh model list / re-check connection")
        row1_layout.addWidget(self.btn_refresh_models)

        main_layout.addLayout(row1_layout)

        # ── Row 2: Status (left) + Actions (history, new chat; right) ───
        row2_layout = QHBoxLayout()
        row2_layout.setContentsMargins(0, 0, 0, 0)
        row2_layout.setSpacing(6)

        self.lbl_status = QLabel("Checking…")
        self.lbl_status.setStyleSheet("font-size: 11px;")
        row2_layout.addWidget(self.lbl_status)

        # No-data notice (shown when no df is loaded)
        self.lbl_no_data = QLabel("ⓘ No DataFrame selected")
        self.lbl_no_data.setStyleSheet("color: #FFA726; font-size: 11px;")
        row2_layout.addWidget(self.lbl_no_data)

        row2_layout.addStretch()

        self.btn_history = QPushButton("")
        self.btn_history.setObjectName("btn_history")
        self.btn_history.setIcon(QIcon(os.path.join(ICON_DIR, "view-details.png")))
        self.btn_history.setIconSize(QSize(26, 26))
        self.btn_history.setFixedSize(36, 36)
        self.btn_history.setToolTip("Conversation History")

        self.btn_new_chat = QPushButton("")
        self.btn_new_chat.setObjectName("btn_new_chat")
        self.btn_new_chat.setIcon(QIcon(os.path.join(ICON_DIR, "ai_chat.png")))
        self.btn_new_chat.setIconSize(QSize(26, 26))
        self.btn_new_chat.setFixedSize(36, 36)
        self.btn_new_chat.setToolTip("New Chat")

        for btn in (self.btn_history, self.btn_new_chat):
            btn.setCursor(Qt.PointingHandCursor)
            row2_layout.addWidget(btn)

        main_layout.addLayout(row2_layout)

        # ── Row 3: Title ─────────────────────────────────────────────
        self.edit_title = QLineEdit()
        self.edit_title.setObjectName("chatTitleEdit")
        self.edit_title.setPlaceholderText("New Conversation")
        self.edit_title.setStyleSheet("font-size: 14px; padding: 2px 4px;")
        self.edit_title.editingFinished.connect(self._on_title_edited)
        main_layout.addWidget(self.edit_title)

        return header

    def _make_reply_preview(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("replyPreview")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(10, 6, 10, 6)

        self.lbl_reply_text = QLabel()
        self.lbl_reply_text.setStyleSheet("font-size: 11px;")
        self.lbl_reply_text.setWordWrap(True)
        layout.addWidget(self.lbl_reply_text, stretch=1)
        
        btn_close = QPushButton("✕")
        btn_close.setFixedSize(20, 20)
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.setStyleSheet("QPushButton { background: transparent; border: none; font-weight: bold; }")
        btn_close.clicked.connect(self._clear_reply_state)
        layout.addWidget(btn_close)
        
        return frame

    def _make_input_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("chatInputBar")

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        # Clear button
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setFixedWidth(55)
        self.btn_clear.setFixedHeight(34)
        self.btn_clear.setToolTip("Clear conversation history")
        layout.addWidget(self.btn_clear, alignment=Qt.AlignBottom)

        # Text input
        self.edit_input = _ChatLineEdit()
        self.edit_input.setPlaceholderText("Ask a question about your data…")
        layout.addWidget(self.edit_input, stretch=1)

        # Microphone button (voice dictation)
        self.btn_mic = QPushButton("🎤")
        self.btn_mic.setObjectName("btn_mic")
        self.btn_mic.setFixedHeight(50)
        self.btn_mic.setFixedWidth(44)
        self.btn_mic.setCheckable(True)
        self.btn_mic.setCursor(Qt.PointingHandCursor)
        self.btn_mic.clicked.connect(self._on_mic_toggle)

        self.btn_mic.setEnabled(True)
        if not VOICE_AVAILABLE:
            self.btn_mic.setToolTip("Voice dictation unavailable — pip install SpeechRecognition pyaudio")
        else:
            self.btn_mic.setToolTip("Click to start voice recording")

        # Base look inherits the themed QPushButton rule; only the
        # "recording" (checked) state needs a distinct, semantic color.
        self.btn_mic.setStyleSheet("""
            QPushButton#btn_mic { font-size: 18px; }
            QPushButton#btn_mic:checked { background: #C62828; color: white; border-color: #E53935; }
        """)
        layout.addWidget(self.btn_mic)

        # Send / Stop button (toggles appearance based on AI state)
        self.btn_send = QPushButton("Send ▶")
        self.btn_send.setFixedHeight(50)
        self.btn_send.setFixedWidth(70)
        self._apply_send_button_style()
        layout.addWidget(self.btn_send)

        return bar

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        # Input — use a dispatcher so the button can send OR stop
        self.btn_send.clicked.connect(self._on_send_or_stop)
        self.edit_input.send_requested.connect(self._on_send_or_stop)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_history.clicked.connect(self._on_history_clicked)
        self.btn_new_chat.clicked.connect(self._on_new_chat_clicked)
        self.btn_refresh_models.clicked.connect(self._refresh_status)
        self.cbb_model.currentTextChanged.connect(self.vm.set_model)
        self.cbb_provider.currentTextChanged.connect(self._on_provider_changed)
        self.cbb_prompt_tier.currentIndexChanged.connect(self._on_prompt_tier_changed)

        self.cbb_model.currentTextChanged.connect(lambda _: self._save_settings())

        # ViewModel → View
        self.vm.thinking_changed.connect(self._on_thinking_changed)
        self.vm.chunk_received.connect(self._on_chunk)
        self.vm.result_ready.connect(self._on_result_ready)
        self.vm.error_occurred.connect(self._on_error)
        self.vm.conversation_changed.connect(self._on_conversation_changed)
        self.vm.tool_execution_received.connect(self._on_tool_execution)

    # ------------------------------------------------------------------
    # Send / Stop button — toggles between blue "Send ▶" and red "Stop ■"
    # ------------------------------------------------------------------

    _SEND_STYLE = """
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
    """

    _STOP_STYLE = """
        QPushButton {
            background: #C62828;
            color: white;
            border: none;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover   { background: #D32F2F; }
        QPushButton:pressed { background: #B71C1C; }
        QPushButton:disabled { background: #555; color: #888; }
    """

    def _apply_send_button_style(self) -> None:
        """Switch the action button to 'Send' mode (blue)."""
        self.btn_send.setText("Send ▶")
        self.btn_send.setStyleSheet(self._SEND_STYLE)
        self.btn_send.setEnabled(True)

    def _apply_stop_button_style(self) -> None:
        """Switch the action button to 'Stop' mode (red)."""
        self.btn_send.setText("Stop ■")
        self.btn_send.setStyleSheet(self._STOP_STYLE)
        self.btn_send.setEnabled(True)

    def _on_send_or_stop(self) -> None:
        """Dispatch to send or stop depending on whether the AI is busy."""
        if self.vm.is_busy():
            self.vm.cancel()
            # _on_thinking_changed(False) will be emitted by the VM
            # and will restore the send button appearance.
        else:
            self._on_send()

    # ------------------------------------------------------------------
    # Public API — called from main.py / VWorkspaceGraphs
    # ------------------------------------------------------------------

    def set_dataframes(self, dfs: Dict[str, pd.DataFrame], active_name: str = "") -> None:
        """Update the available DataFrames the chat can query."""
        self.vm.set_dataframes(dfs, active_name)
        if dfs:
            self.lbl_no_data.clear()
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

        saved_tier_idx = int(s.value("prompt_tier_index", 0))
        if 0 <= saved_tier_idx < self.cbb_prompt_tier.count():
            self.cbb_prompt_tier.blockSignals(True)
            self.cbb_prompt_tier.setCurrentIndex(saved_tier_idx)
            self.cbb_prompt_tier.blockSignals(False)
        self.vm.set_small_model_mode({0: None, 1: False, 2: True}.get(saved_tier_idx))

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
        s.setValue("prompt_tier_index", self.cbb_prompt_tier.currentIndex())

        current_model = self.cbb_model.currentText()
        if current_model:
            s.setValue(f"model_{provider_key}", current_model)

        s.endGroup()

    def _custom_model_names(self, provider_key: str) -> list[str]:
        """User-defined model names for the Custom provider.

        Stored (Settings ▸ AI ▸ Custom Models) as a comma-separated string;
        only applied for the Custom provider, whose endpoint often lacks a
        model-listing API. Returns an empty list for every other provider.
        """
        if provider_key != "Custom":
            return []
        s = QSettings("SPECTROview", "AIChat")
        s.beginGroup(self._SETTINGS_GROUP)
        raw = str(s.value("custom_models", ""))
        s.endGroup()
        return [m.strip() for m in raw.split(",") if m.strip()]

    def _on_prompt_tier_changed(self, index: int) -> None:
        """Apply the manual prompt-tier override (or resume auto-detection)."""
        self.vm.set_small_model_mode({0: None, 1: False, 2: True}.get(index))
        self._save_settings()
        self._refresh_status()

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
            self._apply_send_button_style()
            self.btn_send.setEnabled(False)
            return

        if not is_ollama and not OPENAI_AVAILABLE:
            self.lbl_status.setText("🔴  openai package missing — run: pip install openai")
            self.lbl_status.setStyleSheet("color: #EF5350; font-size: 11px;")
            self.edit_input.setEnabled(False)
            self._apply_send_button_style()
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
            self._apply_send_button_style()
            self.btn_send.setEnabled(False)
            return

        available = self.vm.is_available()

        if available:
            models = list(self.vm.get_models())
            # Merge in user-defined Custom Models (Settings ▸ AI) so they are
            # always selectable even when the endpoint has no listing API.
            for name in self._custom_model_names(provider_key):
                if name not in models:
                    models.append(name)
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
            elif target_model:
                # Editable combobox: keep a typed/saved model the provider
                # didn't list rather than silently dropping it.
                self.cbb_model.setCurrentText(target_model)
            elif self.cbb_model.findText(current) >= 0:
                self.cbb_model.setCurrentIndex(self.cbb_model.findText(current))
            if self.cbb_model.currentText():
                self.vm.set_model(self.cbb_model.currentText())
            self.cbb_model.blockSignals(False)

            if is_ollama:
                status_text = "🟢  Ollama connected"
            else:
                status_text = f"🟢  {provider_key} API connected"
            if self.vm.is_small_model_mode():
                status_text += "  ·  Simplified prompts"
            self.lbl_status.setText(status_text)
            self.lbl_status.setStyleSheet("color: #66BB6A; font-size: 11px;")
            self.edit_input.setEnabled(True)
            self._apply_send_button_style()

        else:
            if is_ollama:
                self.lbl_status.setText("🔴  Ollama not running — run: ollama serve")
            else:
                self.lbl_status.setText(f"🔴  {provider_key} API — invalid or expired API key")
            self.lbl_status.setStyleSheet("color: #EF5350; font-size: 11px;")
            self.edit_input.setEnabled(False)
            self._apply_send_button_style()
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

    # ------------------------------------------------------------------
    # Voice dictation handlers
    # ------------------------------------------------------------------

    def _on_mic_toggle(self, checked: bool) -> None:
        """Start or stop voice recording."""
        if checked:
            self._start_voice_recording()
        else:
            self._stop_voice_recording()

    def _start_voice_recording(self) -> None:
        """Launch the VoiceWorker background thread."""
        if not VOICE_AVAILABLE:
            self.btn_mic.setChecked(False)
            self._on_voice_error("Voice dictation unavailable.\nInstall:  pip install SpeechRecognition pyaudio")
            return

        # Cancel any previous worker
        self._stop_voice_recording()

        self._voice_worker = VoiceWorker(self)
        self._voice_worker.text_ready.connect(self._on_voice_text)
        self._voice_worker.error_occurred.connect(self._on_voice_error)
        self._voice_worker.finished.connect(lambda: self.btn_mic.setChecked(False))
        self._voice_worker.start()

    def _stop_voice_recording(self) -> None:
        """Cancel the in-progress voice worker if any."""
        if self._voice_worker and self._voice_worker.isRunning():
            self._voice_worker.cancel()
            self._voice_worker.wait(1000)  # give it a second to clean up
        self._voice_worker = None

    def _on_voice_text(self, text: str) -> None:
        """Append transcribed text to the input field."""
        current = self.edit_input.toPlainText().strip()
        if current:
            self.edit_input.setPlainText(current + " " + text)
        else:
            self.edit_input.setPlainText(text)
        # Move cursor to end
        cursor = self.edit_input.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.edit_input.setTextCursor(cursor)
        self.edit_input.setFocus()

    def _on_voice_error(self, message: str) -> None:
        """Show voice error as a temporary inline card."""
        card = _MessageCard(message, "error")
        self._insert_before_stretch(card)
        self._scroll_to_bottom()
        # Auto-remove after 5 seconds
        QTimer.singleShot(5000, card.deleteLater)

    def _on_clear(self) -> None:
        self.vm.cancel()
        self.vm.clear_history()
        while self.messages_layout.count() > 1: # preserve the stretch
            item = self.messages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._active_card = None

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
            
    def _on_title_edited(self):
        new_title = self.edit_title.text().strip()
        if new_title and new_title != self.vm._conversation.title:
            self.vm._conversation.rename(new_title)
            self.vm._conversation.save()
            
    def _on_conversation_changed(self, title: str) -> None:
        self.edit_title.setText(title)
        # Clear UI
        while self.messages_layout.count() > 1: # preserve the stretch
            item = self.messages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._active_card = None
                
        # Rebuild from vm conversation
        current_ai_card = None
        for i, msg in enumerate(self.vm._conversation.messages):
            if msg.get("is_hidden"):
                continue

            if msg["role"] == "user":
                self._add_user_card(msg["content"], msg.get("reply_to_index"), timestamp=msg.get("timestamp"))
                current_ai_card = None
            elif msg["role"] == "assistant":
                friendly_text = msg["content"]

                if not current_ai_card:
                    current_ai_card = self._add_ai_card(friendly_text, timestamp=msg.get("timestamp"))
                    current_ai_card.reply_clicked.connect(lambda text, idx=i: self._on_reply_clicked(idx, text))
                else:
                    # Append text to existing card
                    current = current_ai_card.content_view.toPlainText() if hasattr(current_ai_card.content_view, 'toPlainText') else current_ai_card.content_view.text()
                    if friendly_text:
                        current_ai_card.set_text(current + "\n\n" + friendly_text)

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

    def _on_thinking_changed(self, is_thinking: bool, status_text: str = "Thinking") -> None:
        if is_thinking:
            self.thinking_dots.start(status_text)
            self._apply_stop_button_style()
        else:
            self.thinking_dots.stop()
            self._apply_send_button_style()

    def _on_chunk(self, fragment: str) -> None:
        """Buffer a streaming fragment; the actual (expensive) markdown
        re-render is throttled by ``_flush_chunk_buffer`` rather than run
        on every single token."""
        if not self._active_card:
            return
        self._stream_buffer += fragment
        if not self._chunk_flush_timer.isActive():
            self._chunk_flush_timer.start(70)  # ~14 re-renders/sec, capped

    def _flush_chunk_buffer(self) -> None:
        if self._active_card and self._stream_buffer:
            # `_raw_text` is the true markdown source; unlike
            # content_view.toPlainText() (the rendered/stripped output),
            # it doesn't lose formatting syntax on repeated appends.
            current = getattr(self._active_card, "_raw_text", "")
            self._active_card.set_text(current + self._stream_buffer)
            self._stream_buffer = ""
            self._scroll_to_bottom()

    def _on_result_ready(self, result: ChatResult) -> None:
        """Replace the streaming bubble with the full structured result."""
        # Discard any not-yet-flushed streaming fragments — the final
        # text below is authoritative and a stale flush firing afterward
        # would otherwise briefly stomp on it.
        self._chunk_flush_timer.stop()
        self._stream_buffer = ""

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

            # Only genuine new-plot configs are meaningful to save as a
            # reusable recipe — not graph update/delete instructions.
            plot_only_cfgs = [c for c in cfgs if "_graph_update" not in c and "_graph_delete" not in c]
            if plot_only_cfgs:
                btn = QPushButton(
                    f"💾 Save {len(plot_only_cfgs)} plot{'s' if len(plot_only_cfgs) > 1 else ''} as Recipe"
                )
                btn.setObjectName("btnRowPrimary")
                btn.setMaximumWidth(220)
                btn.clicked.connect(lambda: self.prompt_and_save_recipe(plot_only_cfgs))
                self._insert_widget_after_card(btn)

        self._active_card = None
        self._scroll_to_bottom()

    def prompt_and_save_recipe(self, configs: list) -> None:
        """Prompt for a name and persist *configs* as a new plot recipe.

        Used by the inline "Save N plot(s) as Recipe" button offered
        after the AI creates plots (see _on_result_ready) — general
        recipe browse/apply/save-all management now lives in the Graphs
        workspace itself (VWorkspaceGraphs), not here.
        """
        if not configs:
            return
        self.vm.refresh_recipe_store()
        name, ok = QInputDialog.getText(self, "Save Plot Recipe", "Recipe name:")
        if ok and name:
            self.vm.recipe_store.save_recipe(name, configs)
            QMessageBox.information(
                self, "Recipe Saved",
                f"Saved '{name}' with {len(configs)} plot{'s' if len(configs) > 1 else ''}."
            )

    def _on_error(self, message: str) -> None:
        self._chunk_flush_timer.stop()
        self._stream_buffer = ""
        # Replace placeholder bubble with an error bubble
        if self._active_card:
            self._active_card.deleteLater()
            self._active_card = None
        card = _MessageCard(message, "error")
        self._insert_before_stretch(card)
        self._scroll_to_bottom()

    def _on_tool_execution(self, name: str, result_text: str) -> None:
        # Surface progress only through the transient "thinking" status.
        # Intermediate tool steps are intentionally NOT rendered as chips in
        # the conversation — they added noise without helping the user.
        self._on_thinking_changed(True, f"Executed tool '{name}'...")

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

    def _update_scroll_to_bottom_visibility(self) -> None:
        bar = self.scroll_area.verticalScrollBar()
        near_bottom = bar.maximum() - bar.value() < 40
        self.btn_scroll_to_bottom.setVisible(not near_bottom and bar.maximum() > 0)
        self._reposition_scroll_to_bottom_button()

    def _reposition_scroll_to_bottom_button(self) -> None:
        margin = 12
        x = self.scroll_area.width() - self.btn_scroll_to_bottom.width() - margin
        y = self.scroll_area.height() - self.btn_scroll_to_bottom.height() - margin
        self.btn_scroll_to_bottom.move(max(0, x), max(0, y))
        self.btn_scroll_to_bottom.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_scroll_to_bottom_button()

    # ------------------------------------------------------------------
    # Close event — hide instead of destroy so state is preserved
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._save_settings()
        self._stop_voice_recording()
        self.vm.cancel()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────
# Subclassed QLineEdit to emit Enter-key as a signal
# ─────────────────────────────────────────────────────────────────────────

class _ChatLineEdit(QTextEdit):
    """A QTextEdit that emits ``send_requested`` on Enter / Return (but Shift+Enter adds newline)
    and dynamically resizes its height based on its content."""

    send_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptRichText(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._min_height = 50
        self._max_height = 150
        self.setFixedHeight(self._min_height)
        self.textChanged.connect(self._adjust_height)

    def _adjust_height(self) -> None:
        doc_height = int(self.document().size().height())
        margins = self.contentsMargins()
        total_height = doc_height + margins.top() + margins.bottom() + 10
        new_height = max(self._min_height, min(total_height, self._max_height))
        if self.height() != new_height:
            self.setFixedHeight(new_height)

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
from spectroview.ai_agent.m_llm_client import LLMClient  # noqa: E402
