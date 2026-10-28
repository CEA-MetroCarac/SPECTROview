"""Tests for spectroview.ai_agent.v_chat_panel - VChatPanel UI layout and interactions."""

import pytest
from PySide6.QtCore import Qt, QPointF, QEvent
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QFrame

from spectroview.ai_agent.v_chat_panel import VChatPanel, _ChatLineEdit, _ComposerFrame


@pytest.fixture
def chat_panel(qapp, monkeypatch):
    """Instantiate a VChatPanel with network/worker calls mocked out."""
    # Prevent background status worker from launching real network probes
    monkeypatch.setattr(VChatPanel, "_refresh_status", lambda self: None)
    panel = VChatPanel()
    yield panel
    panel.close()


class TestVChatPanelComposerLayout:
    """Verify modern composer architecture embedding provider, model, refresh, mic, and send."""

    def test_composer_frame_exists_and_contains_input_and_toolbar(self, chat_panel):
        assert hasattr(chat_panel, "composer_frame")
        assert chat_panel.composer_frame.objectName() == "chatComposerFrame"

        # Text input is embedded inside composer
        assert chat_panel.edit_input.parent() == chat_panel.composer_frame
        assert chat_panel.edit_input.objectName() == "chatComposerInput"

    def test_controls_embedded_in_composer(self, chat_panel):
        # Provider and model combos are child widgets in the composer
        assert chat_panel.cbb_provider.objectName() == "cbb_provider"
        assert chat_panel.cbb_model.objectName() == "cbb_model"
        assert chat_panel.btn_refresh_models.objectName() == "btn_refresh_models"
        assert chat_panel.btn_mic.objectName() == "btn_mic"
        assert chat_panel.btn_send.objectName() == "btn_send"

        # The prompt tier combobox ("Auto, Full, Small") has been removed
        assert not hasattr(chat_panel, "cbb_prompt_tier")

    def test_header_is_streamlined(self, chat_panel):
        assert hasattr(chat_panel, "lbl_status")
        assert hasattr(chat_panel, "lbl_no_data")
        assert hasattr(chat_panel, "btn_history")
        assert hasattr(chat_panel, "btn_new_chat")
        assert hasattr(chat_panel, "btn_clear")
        assert hasattr(chat_panel, "edit_title")

        # Header height is compact (42px single-row bar)
        header = chat_panel.findChild(QFrame, "chatHeader")
        assert header is not None
        assert header.height() == 42

        # Title and action buttons are in the header; status label is not displayed in header
        assert chat_panel.edit_title.parent() == header
        assert chat_panel.btn_history.parent() == header
        assert chat_panel.btn_new_chat.parent() == header
        assert chat_panel.btn_clear.parent() == header
        assert chat_panel.lbl_status.parent() != header or chat_panel.lbl_status.isHidden()

    def test_send_button_appearance_reacts_to_text_input(self, chat_panel):
        # Empty input -> standard send style
        chat_panel.edit_input.setPlainText("")
        assert "Send" in chat_panel.btn_send.text()

        # Input with text -> active send style
        chat_panel.edit_input.setPlainText("Plot sample data")
        assert "Send" in chat_panel.btn_send.text()
        assert "#1565C0" in chat_panel.btn_send.styleSheet()

        # Cleared input -> returns to standard style
        chat_panel.edit_input.setPlainText("")
        assert "#1976D2" in chat_panel.btn_send.styleSheet()

    def test_send_button_stop_appearance_when_busy(self, chat_panel):
        chat_panel._apply_stop_button_style()
        assert "Stop" in chat_panel.btn_send.text()
        assert "#D32F2F" in chat_panel.btn_send.styleSheet()

        chat_panel._apply_send_button_style()
        assert "Send" in chat_panel.btn_send.text()

    def test_clear_button_clears_history(self, chat_panel):
        chat_panel.vm._conversation.add_message("user", "Hello")
        assert chat_panel.vm.message_count > 0

        chat_panel.btn_clear.click()
        assert chat_panel.vm.message_count == 0

    def test_custom_provider_label_display(self, chat_panel):
        chat_panel._settings.set_ai_value("custom_label", "Lab-LLM")
        chat_panel._populate_providers()

        items = [chat_panel.cbb_provider.itemText(i) for i in range(chat_panel.cbb_provider.count())]
        assert "Lab-LLM" in items
        assert "Custom" not in items
        assert chat_panel._display_to_provider("Lab-LLM") == "Custom"
        assert chat_panel._provider_to_display("Custom") == "Lab-LLM"

    def test_reload_settings_updates_custom_label_in_composer(self, chat_panel):
        # Initial state: default "Custom" provider selected
        chat_panel._settings.set_ai_value("custom_label", "")
        chat_panel._settings.set_ai_value("provider", "Custom")
        chat_panel._load_settings()
        assert chat_panel.cbb_provider.currentText() == "Custom"

        # User updates label in settings to "ABC"
        chat_panel._settings.set_ai_value("custom_label", "ABC")
        chat_panel.reload_settings()

        # Chat panel composer must now show "ABC"
        assert chat_panel.cbb_provider.currentText() == "ABC"
        items = [chat_panel.cbb_provider.itemText(i) for i in range(chat_panel.cbb_provider.count())]
        assert "ABC" in items
        assert "Custom" not in items
        assert chat_panel.cbb_provider.currentData() == "Custom"

    def test_composer_enabled_and_accepts_typing(self, chat_panel, qapp):
        # Composer must always be enabled
        assert chat_panel.edit_input.isEnabled() is True
        assert isinstance(chat_panel.edit_input, _ChatLineEdit)
        assert isinstance(chat_panel.composer_frame, _ComposerFrame)

        # Typing into edit_input
        ev = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_H, Qt.NoModifier, "H")
        qapp.sendEvent(chat_panel.edit_input, ev)
        ev2 = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_I, Qt.NoModifier, "i")
        qapp.sendEvent(chat_panel.edit_input, ev2)

        assert chat_panel.edit_input.toPlainText() == "Hi"

    def test_composer_frame_click_routes_focus(self, chat_panel, qapp):
        chat_panel.show()
        chat_panel.activateWindow()
        chat_panel.edit_title.setFocus()
        qapp.processEvents()
        assert chat_panel.edit_input.hasFocus() is False

        mouse_ev = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(5, 5),
            QPointF(5, 5),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        qapp.sendEvent(chat_panel.composer_frame, mouse_ev)
        qapp.processEvents()
        assert chat_panel.edit_input.hasFocus() is True

    def test_shift_enter_inserts_newline(self, chat_panel, qapp):
        chat_panel.edit_input.setPlainText("Line 1")
        shift_enter = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Return, Qt.ShiftModifier, "\n")
        qapp.sendEvent(chat_panel.edit_input, shift_enter)
        assert "\n" in chat_panel.edit_input.toPlainText()

    def test_send_unconfigured_preserves_prompt_text(self, chat_panel, monkeypatch):
        monkeypatch.setattr(chat_panel.vm, "is_available", lambda: False)
        chat_panel.edit_input.setPlainText("Important question")
        chat_panel._on_send()
        # Text must NOT be wiped out
        assert chat_panel.edit_input.toPlainText() == "Important question"

    def test_send_when_available_clears_and_dispatches(self, chat_panel, monkeypatch):
        monkeypatch.setattr(chat_panel.vm, "is_available", lambda: True)
        dispatched = []
        monkeypatch.setattr(chat_panel.vm, "process_query", lambda text, reply_to_index=None: dispatched.append(text))

        chat_panel.edit_input.setPlainText("Valid query")
        chat_panel._on_send()

        assert dispatched == ["Valid query"]
        assert chat_panel.edit_input.toPlainText() == ""

    def test_focus_input_method(self, chat_panel, qapp):
        chat_panel.show()
        chat_panel.activateWindow()
        chat_panel.edit_title.setFocus()
        qapp.processEvents()
        chat_panel.focus_input()
        qapp.processEvents()
        assert chat_panel.edit_input.hasFocus() is True




