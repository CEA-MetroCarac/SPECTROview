# spectroview/settings/vm_settings.py
"""ViewModel for application settings."""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QFileDialog

from spectroview.i18n import tr
from spectroview.settings.m_settings import MSettings


class _ConnectionTestWorker(QThread):
    """Background worker to probe AI provider connectivity without blocking Qt main thread."""
    result_ready = Signal(bool, str)

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def run(self) -> None:
        try:
            from spectroview.ai_agent.m_llm_client import test_provider_connection
            success, msg = test_provider_connection(
                self.provider, self.api_key, self.base_url, self.model
            )
        except Exception as exc:  # noqa: BLE001
            success, msg = False, str(exc)
        self.result_ready.emit(success, msg)


class VMSettings(QObject):
    """ViewModel for application settings."""
    # ───── ViewModel → View signals ─────
    settings_loaded = Signal(dict)
    settings_saved = Signal()
    working_folder_changed = Signal(str)
    connection_test_started = Signal()
    connection_test_finished = Signal(bool, str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.settings = MSettings()
        self._test_worker: Optional[_ConnectionTestWorker] = None

    # ---------- Load ----------
    def load(self):
        data = self.settings.load_fit_settings()
        working_folder = self.settings.get_working_folder()
        data["working_folder"] = working_folder
        data["peak_profiles"] = self.settings.get_all_peak_profiles(working_folder)
        data["user_peak_profiles"] = sorted(self.settings.load_user_peak_profiles(working_folder).keys())
        ai_data = self.settings.load_ai_settings()
        data.update(ai_data)
        data.update(self.settings.load_mcp_settings())
        data["check_updates_startup"] = self.settings.get_check_for_updates()
        data["include_prereleases"] = self.settings.get_update_include_prereleases()
        data["theme"] = self.settings.get_theme()
        data["language"] = self.settings.get_language()
        self.settings_loaded.emit(data)

    # ---------- Save ----------
    def save(self, data: dict):
        working_folder = data.pop("working_folder", "")
        data.pop("history_folder", None)
        data.pop("peak_profiles", None)
        data.pop("user_peak_profiles", None)

        if "check_updates_startup" in data:
            self.settings.set_check_for_updates(bool(data.pop("check_updates_startup")))

        if "include_prereleases" in data:
            self.settings.set_update_include_prereleases(bool(data.pop("include_prereleases")))

        if "theme" in data:
            self.settings.set_theme(str(data.pop("theme")))

        if "language" in data:
            self.settings.set_language(str(data.pop("language")))

        current_mcp = self.settings.load_mcp_settings()
        mcp_enabled = bool(data.pop("mcp_enabled", current_mcp["mcp_enabled"]))
        mcp_port = int(data.pop("mcp_port", current_mcp["mcp_port"]))

        ai_keys = ["api_key_OpenAI", "api_key_Anthropic", "api_key_Gemini",
                   "api_key_DeepSeek", "api_key_Mistral", "api_key_Custom",
                   "custom_base_url", "custom_models", "custom_label"]
        ai_data = {k: data.pop(k, "") for k in ai_keys if k in data}

        self.settings.save_fit_settings(data)
        self.settings.save_ai_settings(ai_data)
        self.settings.save_mcp_settings(mcp_enabled, mcp_port)

        if working_folder:
            self.settings.set_working_folder(working_folder)
        else:
            self.settings.export_config()

        self.settings_saved.emit()

    # ---------- Peak Profiles ----------
    def get_peak_profiles(
        self, working_folder: Optional[str] = None
    ) -> tuple[dict[str, dict[str, float]], list[str]]:
        """Return `(all_profiles, user_profile_names)` for the configured working folder."""
        all_profiles = self.settings.get_all_peak_profiles(working_folder)
        user_profiles = sorted(self.settings.load_user_peak_profiles(working_folder).keys())
        return all_profiles, user_profiles

    def save_peak_profile(
        self,
        name: str,
        maxshift: float,
        minfwhm: float,
        maxfwhm: float,
        working_folder: Optional[str] = None,
    ) -> str:
        """Save a custom peak profile under `<working_folder>/peak_profile/`."""
        return self.settings.save_peak_profile(
            name=name,
            maxshift=maxshift,
            minfwhm=minfwhm,
            maxfwhm=maxfwhm,
            working_folder=working_folder,
        )

    def delete_peak_profile(self, name: str, working_folder: Optional[str] = None) -> bool:
        """Delete a custom peak profile from `<working_folder>/peak_profile/`."""
        return self.settings.delete_peak_profile(name=name, working_folder=working_folder)

    # ---------- Folder picker ----------
    def pick_working_folder(self):
        """Persists immediately and auto-imports config.json if present in the picked folder."""
        folder = QFileDialog.getExistingDirectory(
            None, tr("Select SPECTROview Working Folder"), ""
        )
        if folder:
            self.settings.set_working_folder(folder)
            self.working_folder_changed.emit(folder)
            self.load()

    # ---------- Config Import / Export dialogs ----------
    def export_config_dialog(self, parent=None) -> Optional[str]:
        default_dir = self.settings.get_working_folder() or self.settings.get_last_directory()
        default_file = os.path.join(default_dir, "spectroview_config.json")
        filepath, _ = QFileDialog.getSaveFileName(
            parent, tr("Export SPECTROview Settings"), default_file, tr("JSON Files (*.json)")
        )
        if filepath:
            return self.settings.export_config(filepath)
        return None

    def import_config_dialog(self, parent=None) -> bool:
        default_dir = self.settings.get_working_folder() or self.settings.get_last_directory()
        filepath, _ = QFileDialog.getOpenFileName(
            parent, tr("Restore SPECTROview Settings"), default_dir, tr("JSON Files (*.json)")
        )
        if filepath:
            success = self.settings.import_config(filepath)
            if success:
                self.load()
            return success
        return False

    # ---------- Test Connection ----------
    def test_connection(
        self,
        provider: str,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
    ) -> None:
        """Trigger an asynchronous connectivity test for the selected provider."""
        self.cancel_test_connection()

        # Fast client-side validation
        if provider != "Ollama" and not api_key:
            self.connection_test_finished.emit(
                False,
                tr("API key is required to test {provider}.", provider=provider),
            )
            return

        self.connection_test_started.emit()

        worker = _ConnectionTestWorker(provider, api_key, base_url, model, parent=self)
        self._test_worker = worker

        def _on_finished(success: bool, message: str) -> None:
            self.connection_test_finished.emit(success, message)
            self._test_worker = None

        worker.result_ready.connect(_on_finished)
        worker.start()

    def cancel_test_connection(self) -> None:
        """Terminate any running connection probe worker."""
        if self._test_worker is not None:
            try:
                from shiboken6 import isValid

                if isValid(self._test_worker) and self._test_worker.isRunning():
                    self._test_worker.terminate()
                    self._test_worker.wait(500)
            except (RuntimeError, ReferenceError):
                pass
            self._test_worker = None

