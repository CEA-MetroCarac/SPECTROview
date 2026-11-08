"""Unified cross-platform credential storage backed by user-private QSettings.

API keys are stored per-user in the operating system's native user configuration store
via QSettings("CEA-Leti", "SPECTROview"):
- Windows: HKEY_CURRENT_USER\\Software\\CEA-Leti\\SPECTROview
- macOS:   ~/Library/Preferences/fr.cea.SPECTROview.plist
- Linux:   ~/.config/CEA-Leti/SPECTROview.conf

Credentials are obfuscated under the internal '_credentials/' namespace and are strictly
excluded from exported or portable config.json files to prevent credential leakage while
avoiding OS Keychain / KWallet permission prompts.
"""

from __future__ import annotations

import base64
from typing import Optional
from PySide6.QtCore import QSettings

SERVICE_NAME = "SPECTROview"


def _obfuscate(text: str) -> str:
    """Lightweight obfuscation for user QSettings storage."""
    if not text:
        return ""
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _deobfuscate(data: str) -> str:
    """Decode lightweight obfuscated value."""
    if not data:
        return ""
    try:
        return base64.b64decode(data.encode("ascii")).decode("utf-8")
    except Exception:
        # If legacy plain text was stored, return as-is
        return data


class SecretStore:
    """Cross-platform credential store using user-private QSettings with legacy migration."""

    def __init__(self, service: str = SERVICE_NAME) -> None:
        self._service = service
        self._settings = QSettings("CEA-Leti", "SPECTROview")
        self._cache: dict[str, str] = {}

    def get_api_key(self, key_name: str) -> str:
        """Retrieve an API key for key_name (e.g. 'api_key_OpenAI')."""
        if key_name in self._cache:
            return self._cache[key_name]

        val: Optional[str] = None

        # Check dedicated credentials storage in QSettings
        stored_val = self._settings.value(f"_credentials/{key_name}", "")
        if stored_val:
            val = _deobfuscate(str(stored_val))

        if not val:
            # Fallback / migration check from legacy ai_chat group in QSettings
            legacy_val = self._settings.value(f"ai_chat/{key_name}", "")
            if legacy_val:
                val = str(legacy_val)
                # Auto-migrate to _credentials and remove from ai_chat
                self._settings.setValue(f"_credentials/{key_name}", _obfuscate(val))
                self._settings.remove(f"ai_chat/{key_name}")
                self._settings.sync()

        result = val or ""
        self._cache[key_name] = result
        return result

    def set_api_key(self, key_name: str, value: str) -> None:
        """Store or delete an API key for key_name."""
        val = str(value).strip() if value else ""
        if val:
            self._cache[key_name] = val
            self._settings.setValue(f"_credentials/{key_name}", _obfuscate(val))
            # Clean any legacy cleartext value from QSettings ai_chat
            if self._settings.contains(f"ai_chat/{key_name}"):
                self._settings.remove(f"ai_chat/{key_name}")
            self._settings.sync()
        else:
            self.delete_api_key(key_name)

    def delete_api_key(self, key_name: str) -> None:
        """Remove an API key from storage and cache."""
        self._cache.pop(key_name, None)
        self._settings.remove(f"_credentials/{key_name}")
        if self._settings.contains(f"ai_chat/{key_name}"):
            self._settings.remove(f"ai_chat/{key_name}")
        self._settings.sync()

    def has_api_key(self, key_name: str) -> bool:
        return bool(self.get_api_key(key_name))
