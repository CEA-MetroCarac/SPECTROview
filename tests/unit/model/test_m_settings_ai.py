"""
Tests for the AI agent's settings living in the single application store.

They used to sit in a second QSettings("SPECTROview", "AIChat") pair. Two
problems with that: the agent's configuration was invisible to the rest of the
app, and it slipped past the ``_isolate_qsettings`` fixture — so any test that
built a VMChat read the developer's real settings and wrote chat history into
their real history folder.

Everything now goes through ``MSettings`` under the ``ai_chat/`` prefix, with a
one-time migration so nobody loses an API key on upgrade.
"""
import tomllib
import uuid

from PySide6.QtCore import QSettings

from spectroview.model.m_settings import MSettings


def _legacy_store() -> QSettings:
    """The pre-unification store, resolved the same way the fixture does so it
    lands in the per-test INI rather than the real registry."""
    return QSettings(QSettings.IniFormat, QSettings.UserScope, *MSettings._LEGACY_AI_STORE)


def _seed_legacy(**values) -> None:
    s = _legacy_store()
    s.beginGroup(MSettings.AI_GROUP)
    for key, value in values.items():
        s.setValue(key, value)
    s.endGroup()
    s.sync()


class TestUnifiedStore:
    def test_values_are_written_under_the_app_store(self, qapp):
        settings = MSettings()
        settings.set_ai_value("custom_models", "model-a, model-b")
        assert settings.settings.value("ai_chat/custom_models") == "model-a, model-b"

    def test_round_trip(self, qapp):
        settings = MSettings()
        settings.set_ai_value("provider", "Gemini")
        assert MSettings().get_ai_value("provider", "", str) == "Gemini"

    def test_typed_read(self, qapp):
        settings = MSettings()
        settings.set_ai_value("prompt_tier_index", 2)
        assert settings.get_ai_value("prompt_tier_index", 0, int) == 2

    def test_missing_key_returns_the_default(self, qapp):
        assert MSettings().get_ai_value("never_set", "fallback", str) == "fallback"

    def test_dynamic_per_provider_keys(self, qapp):
        """api_key_<provider> / model_<provider> are built at runtime, which is
        why the accessor is generic rather than one getter per provider."""
        settings = MSettings()
        settings.set_ai_value("api_key_Gemini", "AIza-test")
        settings.set_ai_value("model_Gemini", "models/gemma-4-26b-a4b-it")
        assert settings.get_ai_value("api_key_Gemini", "", str) == "AIza-test"
        assert settings.get_ai_value("model_Gemini", "", str) == "models/gemma-4-26b-a4b-it"

    def test_load_save_dialog_round_trip(self, qapp):
        settings = MSettings()
        settings.save_ai_settings({"api_key_OpenAI": "sk-test", "custom_models": "m1"})
        loaded = settings.load_ai_settings()
        assert loaded["api_key_OpenAI"] == "sk-test"
        assert loaded["custom_models"] == "m1"
        assert set(loaded) == set(MSettings.AI_SETTING_KEYS)

    def test_save_ignores_unknown_keys(self, qapp):
        settings = MSettings()
        settings.save_ai_settings({"not_a_setting": "x"})
        assert settings.get_ai_value("not_a_setting", "", str) == ""


class TestSecretsStayOutOfTheProject:
    """API keys are the most sensitive thing this app stores. They belong in
    the per-user OS settings store and must never land in a file inside the
    project, where a `git add` could publish them."""

    def test_settings_file_is_outside_the_project_tree(self, qapp, project_root):
        # Under the test fixture this is a tmp INI; in production it is the
        # Windows registry / macOS defaults. Neither is in the repo.
        location = MSettings().settings.fileName()
        assert str(project_root).lower() not in location.lower()

    def test_saving_a_key_writes_nothing_into_the_project(self, qapp, project_root):
        # Generated at runtime so no source file can legitimately contain it —
        # a literal here would match this test file and mask a real leak.
        secret = f"sk-test-{uuid.uuid4().hex}"
        MSettings().set_ai_value("api_key_OpenAI", secret)

        skip = {".venv", ".venv_clean", "site", "build", "dist", "__pycache__", ".git"}
        hits = [
            path for path in project_root.rglob("*")
            if path.is_file()
            and path.suffix in {".json", ".yaml", ".yml", ".ini", ".md", ".py", ".toml"}
            and not skip & set(path.parts)
            and secret in path.read_text(encoding="utf-8", errors="ignore")
        ]
        assert hits == [], f"API key leaked into project files: {hits}"

    def test_runtime_user_data_is_not_part_of_the_packaged_data(self, qapp, project_root):
        """Chat history and saved templates are gitignored runtime data, so a
        package-data glob over them could only bundle whoever built the wheel's
        own files into a published release."""
        pyproject = tomllib.loads(
            (project_root / "pyproject.toml").read_bytes().decode("utf-8"))
        globs = pyproject["tool"]["setuptools"]["package-data"]["spectroview"]
        assert not [g for g in globs if "TEMPLATE" in g or "CHATLOG" in g]


class TestLegacyMigration:
    def test_legacy_values_are_carried_over(self, qapp):
        _seed_legacy(api_key_Gemini="AIza-legacy", history_folder=r"C:\chatlogs")
        settings = MSettings()
        assert settings.get_ai_value("api_key_Gemini", "", str) == "AIza-legacy"
        assert settings.get_ai_value("history_folder", "", str) == r"C:\chatlogs"

    def test_dynamic_legacy_keys_are_carried_over(self, qapp):
        """The migration copies whatever it finds — it cannot enumerate
        model_<provider> keys up front."""
        _seed_legacy(model_DeepSeek="deepseek-chat", provider="DeepSeek")
        settings = MSettings()
        assert settings.get_ai_value("model_DeepSeek", "", str) == "deepseek-chat"
        assert settings.get_ai_value("provider", "", str) == "DeepSeek"

    def test_migration_never_overwrites_a_newer_value(self, qapp):
        settings = MSettings()
        settings.set_ai_value("provider", "Ollama (local)")
        settings.settings.setValue(MSettings._AI_MIGRATED_KEY, False)   # force a re-run
        _seed_legacy(provider="Gemini")

        assert MSettings().get_ai_value("provider", "", str) == "Ollama (local)"

    def test_migration_runs_only_once(self, qapp):
        _seed_legacy(provider="Gemini")
        MSettings()                                   # migrates
        settings = MSettings()
        settings.set_ai_value("provider", "Mistral")  # user changes it afterwards
        assert MSettings().get_ai_value("provider", "", str) == "Mistral"

    def test_marker_is_recorded(self, qapp):
        settings = MSettings()
        assert settings.settings.value(MSettings._AI_MIGRATED_KEY, False, bool) is True

    def test_empty_legacy_store_is_harmless(self, qapp):
        assert MSettings().get_ai_value("provider", "none", str) == "none"

    def test_legacy_store_is_left_intact(self, qapp):
        """Not deleted: harmless to keep, and an older build still reads it."""
        _seed_legacy(api_key_Gemini="AIza-legacy")
        MSettings()
        s = _legacy_store()
        s.beginGroup(MSettings.AI_GROUP)
        remaining = s.value("api_key_Gemini", "", str)
        s.endGroup()
        assert remaining == "AIza-legacy"
