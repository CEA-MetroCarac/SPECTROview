"""Tests for spectroview.ai_agent.v_chat_panel - VChatPanel UI layout and interactions."""

import pytest
from PySide6.QtWidgets import QFrame

from spectroview.ai_agent.v_chat_panel import VChatPanel


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



