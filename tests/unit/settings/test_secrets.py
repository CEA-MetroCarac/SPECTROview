"""Unit tests for spectroview.settings.secrets.SecretStore.

Verifies:
- Unified cross-platform QSettings storage under _credentials/
- Obfuscation / deobfuscation roundtrip
- Automatic migration and cleanup from legacy ai_chat group
- In-memory caching and fresh-instance persistence
- API keys stay strictly out of exported config.json
"""

from spectroview.settings.secrets import SecretStore, _obfuscate, _deobfuscate
from spectroview.model.m_settings import MSettings


class TestSecretStoreObfuscation:
    def test_empty_string(self):
        assert _obfuscate("") == ""
        assert _deobfuscate("") == ""

    def test_roundtrip(self):
        raw = "sk-proj-1234567890abcdef"
        obfuscated = _obfuscate(raw)
        assert obfuscated != raw
        assert _deobfuscate(obfuscated) == raw

    def test_legacy_plaintext_passthrough(self):
        # If legacy plain text was stored, _deobfuscate returns it gracefully
        legacy_key = "plain-text-key-123!"
        assert _deobfuscate(legacy_key) == legacy_key


class TestUnifiedSecretStore:
    """Verifies that SecretStore stores and retrieves keys uniformly via QSettings."""

    def test_set_and_get_api_key_persists_across_instances(self, qapp):
        store = SecretStore()
        store.set_api_key("api_key_OpenAI", "sk-test-unified-value")

        # Verify in-memory cache read
        assert store.get_api_key("api_key_OpenAI") == "sk-test-unified-value"

        # Verify persistence across a new SecretStore instance
        store2 = SecretStore()
        assert store2.get_api_key("api_key_OpenAI") == "sk-test-unified-value"

        # Verify underlying QSettings value is obfuscated, not raw cleartext
        raw_stored = store2._settings.value("_credentials/api_key_OpenAI", "")
        assert raw_stored != "sk-test-unified-value"
        assert raw_stored == _obfuscate("sk-test-unified-value")

    def test_has_api_key_and_delete(self, qapp):
        store = SecretStore()
        store.set_api_key("api_key_Anthropic", "sk-ant-test")
        assert store.has_api_key("api_key_Anthropic") is True

        store.delete_api_key("api_key_Anthropic")
        assert store.has_api_key("api_key_Anthropic") is False
        assert store.get_api_key("api_key_Anthropic") == ""

        # Setting empty string also deletes the key
        store.set_api_key("api_key_Gemini", "AIza-123")
        assert store.has_api_key("api_key_Gemini") is True
        store.set_api_key("api_key_Gemini", "   ")
        assert store.has_api_key("api_key_Gemini") is False

    def test_auto_migration_from_legacy_ai_chat_group(self, qapp):
        store = SecretStore()
        settings = store._settings
        settings.setValue("ai_chat/api_key_Legacy", "sk-legacy-raw-123")
        settings.sync()

        # First read should auto-migrate from ai_chat/ to _credentials/ and delete legacy
        assert store.get_api_key("api_key_Legacy") == "sk-legacy-raw-123"
        assert settings.contains("ai_chat/api_key_Legacy") is False
        assert settings.contains("_credentials/api_key_Legacy") is True

        # Subsequent fresh instance reads from _credentials/
        store2 = SecretStore()
        assert store2.get_api_key("api_key_Legacy") == "sk-legacy-raw-123"


class TestExportExcludesSecrets:
    def test_config_dict_never_contains_credentials(self, qapp):
        m_settings = MSettings()
        m_settings.set_ai_value("api_key_OpenAI", "secret-test-key")
        m_settings.set_ai_value("provider", "OpenAI")

        cfg = m_settings.get_config_dict()
        cfg_str = str(cfg)

        assert "secret-test-key" not in cfg_str
        assert "api_key_OpenAI" not in cfg_str
        assert "_credentials" not in cfg_str
