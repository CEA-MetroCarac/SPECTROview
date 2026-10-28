"""Tests for view/components/v_settings.py - VSettingsDialog.

Covers:
- AI tab's "Provider Presets" collapsible section
- Top-level AI fields round-trip
- Provider connection test flow
- Unified Working Folder controls (Browse, Open folder, Export/Restore Config buttons)
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


class TestUnifiedWorkingFolderUI:
    def test_chat_history_groupbox_removed_from_ai_tab(self, qapp):
        dialog = _dialog(qapp)
        assert not hasattr(dialog, "le_history_folder")
        assert not hasattr(dialog, "btn_history_folder")

    def test_open_working_folder_button_exists(self, qapp):
        dialog = _dialog(qapp)
        assert hasattr(dialog, "btn_open_working_folder")
        assert "Open" in dialog.btn_open_working_folder.text()

    def test_open_working_folder_triggers_desktop_services(self, qapp, tmp_path, monkeypatch):
        import os
        from PySide6.QtGui import QDesktopServices
        opened_urls = []
        monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened_urls.append(url.toLocalFile()))
        dialog = _dialog(qapp)
        target_dir = str(tmp_path / "work_dir")
        dialog.le_working_folder.setText(target_dir)
        dialog._on_open_working_folder()
        assert len(opened_urls) == 1
        assert os.path.normpath(opened_urls[0]) == os.path.normpath(target_dir)

    def test_config_backup_restore_buttons_exist(self, qapp):
        dialog = _dialog(qapp)
        assert hasattr(dialog, "btn_export_config")
        assert hasattr(dialog, "btn_import_config")
        assert "Export Config" in dialog.btn_export_config.text()
        assert "Restore Config" in dialog.btn_import_config.text()

    def test_export_config_flow(self, qapp, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        dialog = _dialog(qapp)
        monkeypatch.setattr(dialog.vm, "export_config_dialog", lambda parent: "/path/to/backup.json")
        info_called = []
        monkeypatch.setattr(QMessageBox, "information", lambda p, t, m: info_called.append(m))
        dialog._on_export_config()
        assert len(info_called) == 1
        assert "/path/to/backup.json" in info_called[0]

    def test_import_config_flow(self, qapp, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        dialog = _dialog(qapp)
        monkeypatch.setattr(dialog.vm, "import_config_dialog", lambda parent: True)
        info_called = []
        monkeypatch.setattr(QMessageBox, "information", lambda p, t, m: info_called.append(m))
        dialog._on_import_config()
        assert len(info_called) == 1
        assert "restored successfully" in info_called[0]


class TestTwoPanelNavigationAndSearch:
    def test_sidebar_contains_all_four_sections(self, qapp):
        dialog = _dialog(qapp)
        assert dialog._nav_list.count() == 4
        items = [dialog._nav_list.item(i).text() for i in range(4)]
        assert items == ["General", "Fit Settings", "AI Settings", "About"]

    def test_search_filters_navigation_items(self, qapp):
        dialog = _dialog(qapp)
        # Search for "tolerance" -> only "Fit Settings" matches
        dialog._search_edit.setText("tolerance")
        assert dialog._nav_list.item(0).isHidden() is True
        assert dialog._nav_list.item(1).isHidden() is False
        assert dialog._nav_list.item(2).isHidden() is True
        assert dialog._nav_list.item(3).isHidden() is True
        assert dialog._stack.currentIndex() == 1

        # Search for "mcp" -> only "AI Settings" matches
        dialog._search_edit.setText("mcp")
        assert dialog._nav_list.item(1).isHidden() is True
        assert dialog._nav_list.item(2).isHidden() is False
        assert dialog._stack.currentIndex() == 2

        # Search for "citation" -> only "About" matches
        dialog._search_edit.setText("citation")
        assert dialog._nav_list.item(2).isHidden() is True
        assert dialog._nav_list.item(3).isHidden() is False
        assert dialog._stack.currentIndex() == 3

        # Clear search -> all visible
        dialog._search_edit.setText("")
        for i in range(4):
            assert dialog._nav_list.item(i).isHidden() is False

    def test_select_section_and_open_about_tab(self, qapp):
        dialog = _dialog(qapp)
        dialog.select_section("fit")
        assert dialog._stack.currentIndex() == 1
        assert dialog._content_title.text() == "Fit Settings"

        dialog.select_section("ai")
        assert dialog._stack.currentIndex() == 2
        assert dialog._content_title.text() == "AI Settings"

        dialog.open_about_tab()
        assert dialog._stack.currentIndex() == 3
        assert dialog._content_title.text() == "About"

        dialog.select_section("general")
        assert dialog._stack.currentIndex() == 0
        assert dialog._content_title.text() == "General"


class TestAboutTabPage:
    def test_about_tab_displays_python_and_dependencies(self, qapp):
        from PySide6.QtWidgets import QLabel
        import platform
        from spectroview import VERSION, BUILD_NUMBER
        dialog = _dialog(qapp)
        about_page = dialog._tab_about
        assert about_page is not None

        labels = [lbl.text() for lbl in about_page.findChildren(QLabel)]
        combined_text = " ".join(labels)

        assert f"Version {VERSION}" in combined_text
        assert f"Build: {BUILD_NUMBER}" in combined_text
        assert f"Python:</b> {platform.python_version()}" in combined_text
        assert "PySide6:" in combined_text
        assert "numpy:" in combined_text
        assert "scipy:" in combined_text

    def test_runtime_package_versions_helper(self):
        from spectroview.settings.ui import get_runtime_package_versions
        versions = get_runtime_package_versions()
        assert "PySide6" in versions
        assert "numpy" in versions
        assert "scipy" in versions
        assert "matplotlib" in versions
        assert "pandas" in versions
        for pkg, ver in versions.items():
            assert isinstance(ver, str)
            assert len(ver) > 0


def test_main_about_opens_settings_about_section(qapp, monkeypatch):
    from spectroview.main import Main
    from spectroview.settings import VSettingsDialog
    main_win = Main()
    section_opened = []

    def mock_select(self, sec):
        section_opened.append(sec)

    monkeypatch.setattr(VSettingsDialog, "select_section", mock_select)
    monkeypatch.setattr(VSettingsDialog, "exec", lambda self: 0)

    main_win.about()
    assert len(section_opened) == 1
    assert section_opened[0] == "about"
