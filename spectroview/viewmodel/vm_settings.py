from typing import Optional

from PySide6.QtCore import QObject, Signal, QThread
from PySide6.QtWidgets import QFileDialog

from spectroview.model.m_settings import MSettings


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
    history_folder_changed = Signal(str)
    connection_test_started = Signal()
    connection_test_finished = Signal(bool, str)

    def __init__(self):
        super().__init__()
        self.settings = MSettings()
        self._test_worker: Optional[_ConnectionTestWorker] = None

    # ---------- Load ----------
    def load(self):
        data = self.settings.load_fit_settings()
        data["working_folder"] = self.settings.get_working_folder()
        ai_data = self.settings.load_ai_settings()
        data.update(ai_data)
        data.update(self.settings.load_mcp_settings())
        self.settings_loaded.emit(data)

    # ---------- Save ----------
    def save(self, data: dict):
        working_folder = data.pop("working_folder", "")

        current_mcp = self.settings.load_mcp_settings()
        mcp_enabled = bool(data.pop("mcp_enabled", current_mcp["mcp_enabled"]))
        mcp_port = int(data.pop("mcp_port", current_mcp["mcp_port"]))

        ai_keys = ["api_key_OpenAI", "api_key_Anthropic", "api_key_Gemini",
                   "api_key_DeepSeek", "api_key_Mistral", "api_key_Custom",
                   "custom_base_url", "custom_models", "custom_label", "history_folder"]
        ai_data = {k: data.pop(k, "") for k in ai_keys if k in data}

        self.settings.save_fit_settings(data)
        self.settings.save_ai_settings(ai_data)
        self.settings.save_mcp_settings(mcp_enabled, mcp_port)

        if working_folder:
            self.settings.set_working_folder(working_folder)

        self.settings_saved.emit()

    # ---------- Folder picker ----------
    def pick_working_folder(self):
        """Persists immediately (unlike the old pick_template_folder(),
        which only emitted the signal to update the line edit and waited
        for the Settings dialog's OK button) -- part of the fix for
        recipe/style dialogs never seeing a folder configured after the
        Graph Workspace tab was already built."""
        folder = QFileDialog.getExistingDirectory(
            None, "Select SPECTROview Working Folder", ""
        )
        if folder:
            self.settings.set_working_folder(folder)
            self.working_folder_changed.emit(folder)

    def pick_history_folder(self):
        folder = QFileDialog.getExistingDirectory(
            None, "Select Chat History Folder", ""
        )
        if folder:
            self.history_folder_changed.emit(folder)

    # ---------- Connection testing ----------
    def test_connection(
        self,
        provider: str,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
    ) -> None:
        """Test connectivity to an AI provider without blocking the GUI."""
        self.cancel_test_connection()
        self.connection_test_started.emit()

        self._test_worker = _ConnectionTestWorker(
            provider, api_key, base_url, model, parent=self
        )
        self._test_worker.result_ready.connect(self._on_test_result)
        self._test_worker.finished.connect(self._test_worker.deleteLater)
        self._test_worker.start()

    def _on_test_result(self, success: bool, message: str) -> None:
        self.connection_test_finished.emit(success, message)

    def cancel_test_connection(self) -> None:
        """Abort any ongoing connection test."""
        if self._test_worker is not None and self._test_worker.isRunning():
            try:
                self._test_worker.result_ready.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._test_worker.quit()
            self._test_worker = None

