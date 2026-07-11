"""
spectroview/llm/v_chat_panel.py
--------------------------------
View layer: AI Chat floating dialog.

Opens as a non-modal, always-on-top-optional window from the toolbar
button so users can query their data from any workspace tab.

UI layout
---------
┌─────────────────────────────────────────────────┐
│  🤖  AI Data Chat              [model combo] [x]│
│─────────────────────────────────────────────────│
│  ● Ollama connected · gemma3:4b                 │
│─────────────────────────────────────────────────│
│                                                 │
│   ┌─[User]──────────────────────────────────┐  │
│   │ Show rows where fwhm_Si > 5             │  │
│   └─────────────────────────────────────────┘  │
│                                                 │
│   ┌─[AI]────────────────────────────────────┐  │
│   │ Found 42 rows matching `fwhm_Si > 5`    │  │
│   │ ┌────────┬──────┬──────┐                │  │
│   │ │ name   │ fwhm │ ...  │                │  │
│   │ └────────┴──────┴──────┘                │  │
│   └─────────────────────────────────────────┘  │
│                                                 │
│─────────────────────────────────────────────────│
│  [Clear]  [Type your question...]   [Send ▶]   │
└─────────────────────────────────────────────────┘

MVVM contract
-------------
This file imports VMChat but NEVER imports any other ViewModel.
All data comes in through signals; all actions go out through method calls.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QScrollArea, QFrame, QComboBox, QSizePolicy,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit, QApplication,
    QSplitter,
)
from PySide6.QtCore import Qt, Signal, QTimer, QSize, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont, QIcon, QKeyEvent, QColor, QPalette, QTextCursor

from spectroview import ICON_DIR
from spectroview.llm.vm_chat import VMChat, ChatResult


# ═══════════════════════════════════════════════════════════════════════════
# Helper widgets
# ═══════════════════════════════════════════════════════════════════════════

class _MessageBubble(QFrame):
    """A single chat message with role styling."""

    def __init__(self, text: str, role: str, parent=None) -> None:
        """
        Parameters
        ----------
        text : str
            The message content.
        role : str
            "user" | "assistant" | "error"
        """
        super().__init__(parent)
        self._role = role
        self.setObjectName(f"bubble_{role}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(2)

        # Role label
        role_label = QLabel("You" if role == "user" else ("⚠ Error" if role == "error" else "🤖 AI"))
        role_font = QFont()
        role_font.setBold(True)
        role_font.setPointSize(9)
        role_label.setFont(role_font)

        self.content_label = QLabel(text)
        self.content_label.setWordWrap(True)
        self.content_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.content_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        layout.addWidget(role_label)
        layout.addWidget(self.content_label)

        self._apply_style()

    def _apply_style(self):
        if self._role == "user":
            self.setStyleSheet("""
                QFrame {
                    background: rgba(30, 100, 200, 0.15);
                    border: 1px solid rgba(30, 100, 200, 0.3);
                    border-radius: 8px;
                    margin-left: 40px;
                    margin-right: 4px;
                }
            """)
        elif self._role == "error":
            self.setStyleSheet("""
                QFrame {
                    background: rgba(200, 50, 50, 0.15);
                    border: 1px solid rgba(200, 50, 50, 0.4);
                    border-radius: 8px;
                    margin-left: 4px;
                    margin-right: 40px;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    background: rgba(60, 180, 100, 0.10);
                    border: 1px solid rgba(60, 180, 100, 0.25);
                    border-radius: 8px;
                    margin-left: 4px;
                    margin-right: 40px;
                }
            """)

    def set_text(self, text: str) -> None:
        self.content_label.setText(text)


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

class VChatPanel(QDialog):
    """Floating AI Chat panel — opened from the toolbar button.

    Signals
    -------
    plot_requested(dict)
        Emitted when the AI suggests a plot.  The dict contains keys
        compatible with ``VWorkspaceGraphs`` plot configuration.
    """

    plot_requested = Signal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("🤖  AI Data Chat")
        self.setWindowFlags(
            Qt.Dialog | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint
        )
        self.setMinimumSize(520, 620)
        self.resize(560, 700)

        self.vm = VMChat(self)
        self._active_bubble: Optional[_MessageBubble] = None  # streaming target

        self._build_ui()
        self._connect_signals()
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

        # ── Input bar ───────────────────────────────────────────────
        input_bar = self._make_input_bar()
        root.addWidget(input_bar)

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

        title = QLabel("🤖  AI Data Chat")
        title.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        layout.addStretch()

        # Model selector
        self.cbb_model = QComboBox()
        self.cbb_model.setMinimumWidth(160)
        self.cbb_model.setToolTip("Select the local Ollama model to use")
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

        # Refresh button
        self.btn_refresh_models = QPushButton("⟳")
        self.btn_refresh_models.setFixedSize(26, 26)
        self.btn_refresh_models.setToolTip("Refresh model list")
        self.btn_refresh_models.setStyleSheet(
            "QPushButton { background: transparent; color: white; "
            "border: 1px solid rgba(255,255,255,0.3); border-radius: 4px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.2); }"
        )
        layout.addWidget(self.btn_refresh_models)

        return header

    def _make_status_bar(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(28)
        bar.setObjectName("chatStatusBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 0, 10, 0)

        self.lbl_status = QLabel("Checking Ollama…")
        self.lbl_status.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.lbl_status)
        layout.addStretch()

        # No-data notice (shown when no df is loaded)
        self.lbl_no_data = QLabel("ⓘ No DataFrame selected")
        self.lbl_no_data.setStyleSheet("color: #FFA726; font-size: 11px;")
        layout.addWidget(self.lbl_no_data)

        return bar

    def _make_input_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("chatInputBar")
        bar.setStyleSheet("""
            QFrame#chatInputBar {
                border-top: 1px solid rgba(128,128,128,0.3);
                padding: 2px;
            }
        """)
        bar.setFixedHeight(52)

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
        self.edit_input.setFixedHeight(34)
        layout.addWidget(self.edit_input, stretch=1)

        # Send button
        self.btn_send = QPushButton("Send ▶")
        self.btn_send.setFixedHeight(34)
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
        self.btn_refresh_models.clicked.connect(self._refresh_status)
        self.cbb_model.currentTextChanged.connect(self.vm.set_model)

        # ViewModel → View
        self.vm.thinking_changed.connect(self._on_thinking_changed)
        self.vm.chunk_received.connect(self._on_chunk)
        self.vm.result_ready.connect(self._on_result_ready)
        self.vm.error_occurred.connect(self._on_error)

    # ------------------------------------------------------------------
    # Public API — called from main.py / VWorkspaceGraphs
    # ------------------------------------------------------------------

    def set_dataframes(self, dfs: Dict[str, pd.DataFrame], active_name: str = "") -> None:
        """Update the available DataFrames the chat can query."""
        self.vm.set_dataframes(dfs, active_name)
        if dfs:
            total_rows = sum(len(df) for df in dfs.values())
            self.lbl_no_data.setText(
                f"📊 {len(dfs)} DataFrame(s) loaded  (Active: {active_name})" if active_name else f"📊 {len(dfs)} DataFrame(s) loaded"
            )
            self.lbl_no_data.setStyleSheet("color: #66BB6A; font-size: 11px;")
        else:
            self.lbl_no_data.setText("ⓘ No DataFrames available")
            self.lbl_no_data.setStyleSheet("color: #FFA726; font-size: 11px;")

    # ------------------------------------------------------------------
    # Refresh helpers
    # ------------------------------------------------------------------

    def _refresh_status(self) -> None:
        """Check Ollama availability and update status bar + model list."""
        available = self.vm.is_available()

        if available:
            models = self.vm.get_models()
            self.cbb_model.blockSignals(True)
            current = self.cbb_model.currentText()
            self.cbb_model.clear()
            self.cbb_model.addItems(models if models else [LLMClient.DEFAULT_MODEL])
            # Restore previous selection if still available
            idx = self.cbb_model.findText(current)
            if idx >= 0:
                self.cbb_model.setCurrentIndex(idx)
            # Set default model
            if self.cbb_model.currentText():
                self.vm.set_model(self.cbb_model.currentText())
            self.cbb_model.blockSignals(False)

            model_name = self.cbb_model.currentText()
            self.lbl_status.setText(f"🟢  Ollama connected")
            self.lbl_status.setStyleSheet("color: #66BB6A; font-size: 11px;")
            self.edit_input.setEnabled(True)
            self.btn_send.setEnabled(True)
        else:
            self.lbl_status.setText("🔴  Ollama not running — run: ollama serve")
            self.lbl_status.setStyleSheet("color: #EF5350; font-size: 11px;")
            self.edit_input.setEnabled(False)
            self.btn_send.setEnabled(False)

    # ------------------------------------------------------------------
    # Slot handlers
    # ------------------------------------------------------------------

    def _on_send(self) -> None:
        text = self.edit_input.text().strip()
        if not text or self.vm.is_busy():
            return

        self.edit_input.clear()
        self._add_user_bubble(text)
        self._active_bubble = self._add_ai_bubble("")  # placeholder for streaming
        self.vm.process_query(text)

    def _on_clear(self) -> None:
        self.vm.clear_history()
        # Remove all message widgets (keep the stretch at the end)
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._active_bubble = None

    def _on_thinking_changed(self, thinking: bool) -> None:
        if thinking:
            self.thinking_dots.start()
            self.btn_send.setEnabled(False)
        else:
            self.thinking_dots.stop()
            self.btn_send.setEnabled(True)

    def _on_chunk(self, fragment: str) -> None:
        """Append a streaming fragment to the active AI bubble."""
        if self._active_bubble:
            current = self._active_bubble.content_label.text()
            self._active_bubble.set_text(current + fragment)
            self._scroll_to_bottom()

    def _on_result_ready(self, result: ChatResult) -> None:
        """Replace the streaming bubble with the full structured result."""
        if self._active_bubble:
            # Set the explanation as the main text
            self._active_bubble.set_text(result.explanation or result.text_summary)

        # If a DataFrame was returned, attach a preview table below the bubble
        if result.dataframe is not None and not result.dataframe.empty:
            preview = _DataFramePreview(result.dataframe)
            self._insert_widget_after_bubble(preview)

        # If statistics / answer text, show in a text box
        elif result.text_summary and result.action in ("statistics", "answer"):
            if self._active_bubble:
                self._active_bubble.set_text(
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
                self._insert_widget_after_bubble(btn)

        self._active_bubble = None
        self._scroll_to_bottom()

    def _on_error(self, message: str) -> None:
        # Replace placeholder bubble with an error bubble
        if self._active_bubble:
            self._active_bubble.deleteLater()
            self._active_bubble = None
        bubble = _MessageBubble(message, "error")
        self._insert_before_stretch(bubble)
        self._scroll_to_bottom()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_user_bubble(self, text: str) -> _MessageBubble:
        bubble = _MessageBubble(text, "user")
        self._insert_before_stretch(bubble)
        self._scroll_to_bottom()
        return bubble

    def _add_ai_bubble(self, text: str) -> _MessageBubble:
        bubble = _MessageBubble(text, "assistant")
        self._insert_before_stretch(bubble)
        self._scroll_to_bottom()
        return bubble

    def _insert_before_stretch(self, widget: QWidget) -> None:
        """Insert a widget just before the trailing stretch."""
        count = self.messages_layout.count()
        self.messages_layout.insertWidget(count - 1, widget)

    def _insert_widget_after_bubble(self, widget: QWidget) -> None:
        """Insert a widget just after the last _MessageBubble."""
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
        self.vm.cancel()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────
# Subclassed QLineEdit to emit Enter-key as a signal
# ─────────────────────────────────────────────────────────────────────────

class _ChatLineEdit(QLineEdit):
    """A QLineEdit that emits ``send_requested`` on Enter / Return."""

    send_requested = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.send_requested.emit()
        else:
            super().keyPressEvent(event)


# ─────────────────────────────────────────────────────────────────────────
# Re-export LLMClient.DEFAULT_MODEL so main.py can reference it easily
# ─────────────────────────────────────────────────────────────────────────
from spectroview.llm.m_llm_client import LLMClient  # noqa: E402
