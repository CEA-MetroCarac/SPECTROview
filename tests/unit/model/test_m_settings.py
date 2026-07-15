"""Unit tests for model/m_settings.py - MSettings.

Relies on the session-wide _isolate_qsettings autouse fixture (tests/conftest.py)
so these round-trips never touch the real user registry/plist.
"""
from spectroview.model.m_settings import MSettings


class TestFitSettings:
    def test_defaults_when_unset(self, settings):
        data = settings.load_fit_settings()
        assert data["fit_negative"] is False
        assert data["max_ite"] == 200
        assert data["xtol"] == 1e-4
        assert data["coef_noise"] == 0.0

    def test_save_and_reload_round_trip(self, settings):
        settings.save_fit_settings({
            "fit_negative": True, "max_ite": 500, "xtol": 1e-6,
            "coef_noise": 2.5, "maxshift": 30.0,
        })
        reloaded = MSettings().load_fit_settings()
        assert reloaded["fit_negative"] is True
        assert reloaded["max_ite"] == 500
        assert reloaded["xtol"] == 1e-6
        assert reloaded["coef_noise"] == 2.5
        assert reloaded["maxshift"] == 30.0

    def test_save_is_visible_to_a_second_independent_instance(self, settings):
        settings.save_fit_settings({"max_ite": 777})
        other = MSettings()
        assert other.load_fit_settings()["max_ite"] == 777


class TestLastDirectory:
    def test_default_is_root(self, settings):
        assert settings.get_last_directory() == "/"

    def test_round_trip(self, settings, tmp_path):
        settings.set_last_directory(str(tmp_path))
        assert MSettings().get_last_directory() == str(tmp_path)


class TestModelFolder:
    def test_round_trip(self, settings, tmp_path):
        settings.set_model_folder(str(tmp_path))
        assert MSettings().get_model_folder() == str(tmp_path)


class TestUpdateChecker:
    def test_defaults(self, settings):
        assert settings.get_check_for_updates() is True
        assert settings.get_skipped_version() == ""
        assert settings.get_last_check_date() == ""

    def test_round_trip(self, settings):
        settings.set_check_for_updates(False)
        settings.set_skipped_version("26.0.0")
        settings.set_last_check_date("2026-01-01")
        other = MSettings()
        assert other.get_check_for_updates() is False
        assert other.get_skipped_version() == "26.0.0"
        assert other.get_last_check_date() == "2026-01-01"


class TestViewOptions:
    def test_defaults(self, settings):
        data = settings.load_view_options()
        assert data["theme"] == "Dark Mode"
        assert data["legend"] is False

    def test_round_trip(self, settings):
        settings.save_view_options({"theme": "Light Mode", "grid": True, "lw": 2.5})
        reloaded = MSettings().load_view_options()
        assert reloaded["theme"] == "Light Mode"
        assert reloaded["grid"] is True
        assert reloaded["lw"] == 2.5


class TestAiSettings:
    def test_round_trip_api_keys(self, settings):
        settings.save_ai_settings({"api_key_Anthropic": "sk-test-123", "history_folder": "/tmp/hist"})
        reloaded = MSettings().load_ai_settings()
        assert reloaded["api_key_Anthropic"] == "sk-test-123"
        assert reloaded["history_folder"] == "/tmp/hist"

    def test_unknown_keys_are_ignored(self, settings):
        settings.save_ai_settings({"not_a_real_key": "value"})
        reloaded = MSettings().load_ai_settings()
        assert "not_a_real_key" not in reloaded
