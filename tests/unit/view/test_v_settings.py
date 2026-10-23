"""Tests for view/components/v_settings.py - VSettingsDialog.

Covers the AI tab's "Provider Presets" collapsible section: collapsed by
default, toggled by its button, and functionality (load/save round-trip)
unaffected by being nested inside the collapsible frame.
"""
from spectroview.view.components.v_settings import VSettingsDialog
from spectroview.viewmodel.vm_settings import VMSettings


def _dialog(qapp):
    vm = VMSettings()
    return VSettingsDialog(vm)


class TestProviderPresetsCollapsible:
    def test_collapsed_by_default(self, qapp):
        dialog = _dialog(qapp)
        assert dialog.btn_toggle_providers.isChecked() is False
        assert dialog.frame_provider_presets.isHidden() is True

    def test_toggle_expands_and_collapses(self, qapp):
        dialog = _dialog(qapp)
        dialog.btn_toggle_providers.setChecked(True)
        assert dialog.frame_provider_presets.isHidden() is False
        assert "▾" in dialog.btn_toggle_providers.text()

        dialog.btn_toggle_providers.setChecked(False)
        assert dialog.frame_provider_presets.isHidden() is True
        assert "▸" in dialog.btn_toggle_providers.text()


class TestAISettingsRoundTrip:
    def test_provider_and_top_level_fields_save_and_reload(self, qapp):
        dialog = _dialog(qapp)
        dialog.edit_custom.setText("custom-key")
        dialog.edit_custom_url.setText("https://example.com/v1")
        dialog.edit_custom_models.setText("model-a, model-b")
        dialog.edit_custom_label.setText("Lab-LLM")
        dialog.edit_openai.setText("openai-key")
        dialog.edit_anthropic.setText("anthropic-key")
        dialog.chk_mcp_enabled.setChecked(True)
        dialog.spin_mcp_port.setValue(9012)

        dialog._on_accept()

        reloaded = _dialog(qapp)
        assert reloaded.edit_custom.text() == "custom-key"
        assert reloaded.edit_custom_url.text() == "https://example.com/v1"
        assert reloaded.edit_custom_models.text() == "model-a, model-b"
        assert reloaded.edit_custom_label.text() == "Lab-LLM"
        assert reloaded.edit_openai.text() == "openai-key"
        assert reloaded.edit_anthropic.text() == "anthropic-key"
        assert reloaded.chk_mcp_enabled.isChecked() is True
        assert reloaded.spin_mcp_port.value() == 9012
        assert "9012" in reloaded.lbl_mcp_endpoint.text()


class TestAIConnectionTest:
    def test_test_connection_controls_exist(self, qapp):
        dialog = _dialog(qapp)
        assert hasattr(dialog, "cbb_test_provider")
        assert hasattr(dialog, "btn_test_connection")
        assert hasattr(dialog, "lbl_test_status")
        assert dialog.cbb_test_provider.currentText() == "Custom"
        assert "Test Connection" in dialog.btn_test_connection.text()

    def test_test_connection_success_flow(self, qapp):
        dialog = _dialog(qapp)
        dialog._on_connection_test_finished(True, "Connected to Ollama (3 models available)")
        assert dialog.btn_test_connection.isEnabled() is True
        assert "✓" in dialog.lbl_test_status.text()
        assert "#1E9E5A" in dialog.lbl_test_status.styleSheet()

    def test_test_connection_failure_flow(self, qapp):
        dialog = _dialog(qapp)
        dialog._on_connection_test_finished(False, "Connection refused: endpoint not reachable")
        assert dialog.btn_test_connection.isEnabled() is True
        assert "✕" in dialog.lbl_test_status.text()
        assert "#D14343" in dialog.lbl_test_status.styleSheet()

    def test_vm_test_connection_missing_key(self, qapp, qtbot):
        vm = VMSettings()
        with qtbot.waitSignal(vm.connection_test_finished, timeout=5000) as blocker:
            vm.test_connection("OpenAI", api_key="")
        success, msg = blocker.args
        assert success is False
        assert "API key is required" in msg
        vm.cancel_test_connection()


