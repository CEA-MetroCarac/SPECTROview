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


class TestWorkingFolder:
    """One user-configured root folder with 3 auto-created subfolders
    (fit_model/plot_recipe/plot_style), replacing the old separate
    "Fit model folder"/"Plot template folder" settings."""

    def test_defaults_to_empty_when_unset(self, settings):
        assert settings.get_working_folder() == ""

    def test_round_trip(self, settings, tmp_path):
        settings.set_working_folder(str(tmp_path))
        assert MSettings().get_working_folder() == str(tmp_path)

    def test_set_creates_all_three_subfolders(self, settings, tmp_path):
        working = tmp_path / "spectroview_data"
        settings.set_working_folder(str(working))
        assert (working / "fit_model").is_dir()
        assert (working / "plot_recipe").is_dir()
        assert (working / "plot_style").is_dir()

    def test_derived_subfolder_getters(self, settings, tmp_path):
        settings.set_working_folder(str(tmp_path))
        assert settings.get_fit_model_folder() == str(tmp_path / "fit_model")
        assert settings.get_plot_recipe_folder() == str(tmp_path / "plot_recipe")
        assert settings.get_plot_style_folder() == str(tmp_path / "plot_style")

    def test_derived_subfolder_getters_empty_when_unconfigured(self, settings):
        assert settings.get_fit_model_folder() == ""
        assert settings.get_plot_recipe_folder() == ""
        assert settings.get_plot_style_folder() == ""

    def test_migrates_from_legacy_template_folder(self, settings, tmp_path):
        """One-time migration: an existing "template_folder" setting (from
        before this feature) seeds the new working_folder so it isn't
        orphaned -- this is also the regression case for the reported bug
        (a configured template_folder that appeared to do nothing)."""
        settings.settings.setValue("template_folder", str(tmp_path))

        assert MSettings().get_working_folder() == str(tmp_path)
        # The migration is persisted, not just computed on the fly.
        assert settings.settings.value("working_folder", "", str) == str(tmp_path)

    def test_migrates_from_legacy_model_folder_when_no_template_folder(self, settings, tmp_path):
        settings.settings.setValue("model_folder", str(tmp_path))
        assert MSettings().get_working_folder() == str(tmp_path)

    def test_legacy_template_folder_takes_priority_over_model_folder(self, settings, tmp_path):
        settings.settings.setValue("template_folder", str(tmp_path / "recipes"))
        settings.settings.setValue("model_folder", str(tmp_path / "models"))
        assert MSettings().get_working_folder() == str(tmp_path / "recipes")

    def test_explicit_working_folder_wins_over_legacy_settings(self, settings, tmp_path):
        settings.settings.setValue("template_folder", str(tmp_path / "old"))
        settings.set_working_folder(str(tmp_path / "new"))
        assert MSettings().get_working_folder() == str(tmp_path / "new")


class TestDefaultGraphStyle:
    """The user-chosen "Set as Default Style" baseline that new graphs
    start with (see VWorkspaceGraphs._apply_default_style_to_config()) --
    deliberately separate from "Reset to Default"
    (graph_style.default_style()), which always stays hardcoded/factory."""

    def test_defaults_to_empty_when_unset(self, settings):
        assert settings.get_default_graph_style() == {}

    def test_round_trip(self, settings):
        style = {"grid": True, "title_fontsize": 22, "color_palette": "viridis"}
        settings.set_default_graph_style(style)
        assert MSettings().get_default_graph_style() == style

    def test_round_trip_preserves_nested_values(self, settings):
        """Style dicts can contain nested containers (spines_visible dict,
        figure_margins list) -- must survive the JSON round trip intact."""
        style = {
            "spines_visible": {"top": False, "right": False, "bottom": True, "left": True},
            "figure_margins": [0.1, 0.15],
        }
        settings.set_default_graph_style(style)
        assert MSettings().get_default_graph_style() == style

    def test_clear_removes_it(self, settings):
        settings.set_default_graph_style({"grid": True})
        settings.clear_default_graph_style()
        assert MSettings().get_default_graph_style() == {}

    def test_corrupted_value_falls_back_to_empty(self, settings):
        """Defensive: a hand-edited or version-mismatched QSettings value
        that isn't valid JSON must not crash the app on load."""
        settings.settings.setValue("default_graph_style", "{not valid json")
        assert MSettings().get_default_graph_style() == {}


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


class TestExportOptions:
    def test_defaults(self, settings):
        data = settings.load_export_options()
        assert data["format"] == "png"
        assert data["dpi"] == 300
        assert data["transparent"] is False
        assert data["theme"] == "Light Mode"

    def test_round_trip(self, settings):
        settings.save_export_options({
            "format": "svg", "dpi": 600, "transparent": True, "theme": "Dark Mode",
        })
        reloaded = MSettings().load_export_options()
        assert reloaded["format"] == "svg"
        assert reloaded["dpi"] == 600
        assert reloaded["transparent"] is True
        assert reloaded["theme"] == "Dark Mode"


class TestAiSettings:
    def test_round_trip_api_keys(self, settings):
        settings.save_ai_settings({"api_key_Anthropic": "sk-test-123", "history_folder": "/tmp/hist"})
        reloaded = MSettings().load_ai_settings()
        assert reloaded["api_key_Anthropic"] == "sk-test-123"
        assert reloaded["history_folder"] == "/tmp/hist"

    def test_round_trip_custom_provider(self, settings):
        settings.save_ai_settings({
            "custom_base_url": "https://llm.example.com/v1",
            "custom_models": "model-a, model-b",
        })
        reloaded = MSettings().load_ai_settings()
        assert reloaded["custom_base_url"] == "https://llm.example.com/v1"
        assert reloaded["custom_models"] == "model-a, model-b"

    def test_unknown_keys_are_ignored(self, settings):
        settings.save_ai_settings({"not_a_real_key": "value"})
        reloaded = MSettings().load_ai_settings()
        assert "not_a_real_key" not in reloaded
