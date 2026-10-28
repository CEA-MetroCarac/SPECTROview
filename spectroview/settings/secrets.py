"""Secure credential storage backed by Windows Credential Manager / OS keyring with fallback.

API keys are stored per-user in the operating system's native secure credential vault
(Windows Credential Manager via keyring) and are strictly excluded from exported or
portable config.json files to prevent credential leakage.
"""

from __future__ import annotations

import logging
from typing import Optional
from PySide6.QtCore import QSettings

log = logging.getLogger(__name__)

SERVICE_NAME = "SPECTROview"


class SecretStore:
    """Credential store using OS keyring with transparent QSettings migration fallback."""

    def __init__(self, service: str = SERVICE_NAME) -> None:
        self._service = service
        self._settings = QSettings("CEA-Leti", "SPECTROview")

    def get_api_key(self, key_name: str) -> str:
        """Retrieve an API key for key_name (e.g. 'api_key_OpenAI')."""
        val: Optional[str] = None
        try:
            import keyring
            val = keyring.get_password(self._service, key_name)
        except Exception as e:
            log.warning("Failed to read %s from keyring: %s", key_name, e)

        if not val:
            # Fallback / migration check from QSettings
            legacy_val = self._settings.value(f"ai_chat/{key_name}", "")
            if legacy_val:
                val = str(legacy_val)
                # Auto-migrate to keyring and remove from QSettings
                try:
                    import keyring
                    keyring.set_password(self._service, key_name, val)
                    self._settings.remove(f"ai_chat/{key_name}")
                    self._settings.sync()
                except Exception:
                    pass

        return val or ""

    def set_api_key(self, key_name: str, value: str) -> None:
        """Store or delete an API key for key_name."""
        val = str(value).strip() if value else ""
        if val:
            try:
                import keyring
                keyring.set_password(self._service, key_name, val)
            except Exception as e:
                log.warning("Failed to store %s in keyring, using fallback: %s", key_name, e)
                # Fallback to QSettings if keyring backend is unavailable
                self._settings.setValue(f"ai_chat/{key_name}", val)
                self._settings.sync()
                return

            # Clean any legacy cleartext value from QSettings
            if self._settings.contains(f"ai_chat/{key_name}"):
                self._settings.remove(f"ai_chat/{key_name}")
                self._settings.sync()
        else:
            self.delete_api_key(key_name)

    def delete_api_key(self, key_name: str) -> None:
        """Remove an API key from both keyring and QSettings."""
        try:
            import keyring
            keyring.delete_password(self._service, key_name)
        except Exception:
            pass
        if self._settings.contains(f"ai_chat/{key_name}"):
            self._settings.remove(f"ai_chat/{key_name}")
            self._settings.sync()

    def has_api_key(self, key_name: str) -> bool:
        return bool(self.get_api_key(key_name))

