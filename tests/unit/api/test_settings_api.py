"""Tests for spectroview.api.settings. Relies on the autouse
`_isolate_qsettings` fixture (tests/conftest.py) to redirect QSettings to a
throwaway per-test store. Deliberately does NOT request the `qapp` fixture,
to prove this module works without a running QApplication."""
from spectroview.api import settings


class TestFitDefaults:
    def test_round_trip(self):
        settings.set_fit_defaults(fit_negative=True, max_ite=42, coef_noise=1.5)
        defaults = settings.get_fit_defaults()
        assert defaults["fit_negative"] is True
        assert defaults["max_ite"] == 42
        assert defaults["coef_noise"] == 1.5

    def test_has_documented_default_keys(self):
        defaults = settings.get_fit_defaults()
        expected = {"fit_negative", "max_ite", "xtol", "ftol", "coef_noise", "maxshift", "maxfwhm", "minfwhm"}
        assert expected.issubset(defaults.keys())


class TestWorkingFolder:
    def test_round_trip(self, tmp_path):
        settings.set_working_folder(tmp_path)
        assert settings.get_working_folder() == str(tmp_path)

    def test_fit_model_folder_is_a_subfolder_of_the_working_folder(self, tmp_path):
        settings.set_working_folder(tmp_path)
        assert settings.get_fit_model_folder() == str(tmp_path / "fit_model")

    def test_all_subfolders_of_working_folder(self, tmp_path):
        settings.set_working_folder(tmp_path)
        assert settings.get_plot_recipe_folder() == str(tmp_path / "plot_recipe")
        assert settings.get_plot_style_folder() == str(tmp_path / "plot_style")
        assert settings.get_template_folder() == str(tmp_path / "template")
        assert settings.get_history_folder() == str(tmp_path / "ai_chat_history")
        assert settings.get_ai_history_folder() == str(tmp_path / "ai_chat_history")

    def test_export_and_import_config_api(self, tmp_path):
        settings.set_working_folder(tmp_path)
        settings.set_fit_defaults(max_ite=333)
        export_file = tmp_path / "api_backup.json"
        res = settings.export_config(export_file)
        assert res == str(export_file)
        assert export_file.is_file()

        # Modify setting
        settings.set_fit_defaults(max_ite=100)
        assert settings.get_fit_defaults()["max_ite"] == 100

        # Import config
        success = settings.import_config(export_file)
        assert success is True
        assert settings.get_fit_defaults()["max_ite"] == 333


class TestLastDirectory:
    def test_round_trip(self, tmp_path):
        settings.set_last_directory(tmp_path)
        assert settings.get_last_directory() == str(tmp_path)


class TestMCPSettings:
    def test_defaults_are_local_and_disabled(self):
        assert settings.get_mcp_settings() == {
            "mcp_enabled": False,
            "mcp_port": 8765,
            "mcp_host": "127.0.0.1",
        }

    def test_round_trip(self):
        settings.set_mcp_settings(True, 9012)
        assert settings.get_mcp_settings()["mcp_enabled"] is True
        assert settings.get_mcp_settings()["mcp_port"] == 9012
