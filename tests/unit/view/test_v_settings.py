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
        dialog.edit_openai.setText("openai-key")
        dialog.edit_anthropic.setText("anthropic-key")

        dialog._on_accept()

        reloaded = _dialog(qapp)
        assert reloaded.edit_custom.text() == "custom-key"
        assert reloaded.edit_custom_url.text() == "https://example.com/v1"
        assert reloaded.edit_custom_models.text() == "model-a, model-b"
        assert reloaded.edit_openai.text() == "openai-key"
        assert reloaded.edit_anthropic.text() == "anthropic-key"
