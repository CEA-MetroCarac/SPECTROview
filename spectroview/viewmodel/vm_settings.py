#spectroview/viewmodel/vm_settings.py

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog

from spectroview.model.m_settings import MSettings


class VMSettings(QObject):
    """ViewModel for application settings."""
    # ───── ViewModel → View signals ─────
    settings_loaded = Signal(dict)
    settings_saved = Signal()
    working_folder_changed = Signal(str)
    history_folder_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.settings = MSettings()

    # ---------- Load ----------
    def load(self):
        data = self.settings.load_fit_settings()
        data["working_folder"] = self.settings.get_working_folder()
        ai_data = self.settings.load_ai_settings()
        data.update(ai_data)
        self.settings_loaded.emit(data)

    # ---------- Save ----------
    def save(self, data: dict):
        working_folder = data.pop("working_folder", "")

        ai_keys = ["api_key_OpenAI", "api_key_Anthropic", "api_key_Gemini",
                   "api_key_DeepSeek", "api_key_Mistral", "api_key_Custom",
                   "custom_base_url", "custom_models", "history_folder"]
        ai_data = {k: data.pop(k, "") for k in ai_keys if k in data}

        self.settings.save_fit_settings(data)
        self.settings.save_ai_settings(ai_data)

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
