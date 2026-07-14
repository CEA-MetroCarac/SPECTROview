#spectroview/viewmodel/vm_settings.py

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog

from spectroview.model.m_settings import MSettings


class VMSettings(QObject):
    """ViewModel for application settings."""
    # ───── ViewModel → View signals ─────
    settings_loaded = Signal(dict)
    settings_saved = Signal()
    model_folder_changed = Signal(str)
    history_folder_changed = Signal(str)
    template_folder_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.settings = MSettings()

    # ---------- Load ----------
    def load(self):
        data = self.settings.load_fit_settings()
        data["model_folder"] = self.settings.get_model_folder()
        ai_data = self.settings.load_ai_settings()
        data.update(ai_data)
        self.settings_loaded.emit(data)

    # ---------- Save ----------
    def save(self, data: dict):
        model_folder = data.pop("model_folder", "")
        
        ai_keys = ["api_key_OpenAI", "api_key_Anthropic", "api_key_Gemini",
                   "api_key_DeepSeek", "api_key_Custom", "custom_base_url", "history_folder", "template_folder"]
        ai_data = {k: data.pop(k, "") for k in ai_keys if k in data}
        
        self.settings.save_fit_settings(data)
        self.settings.save_ai_settings(ai_data)

        if model_folder:
            self.settings.set_model_folder(model_folder)

        self.settings_saved.emit()

    # ---------- Folder picker ----------
    def pick_model_folder(self):
        folder = QFileDialog.getExistingDirectory(
            None, "Select Fit Model Folder", ""
        )
        if folder:
            self.settings.set_model_folder(folder)
            self.model_folder_changed.emit(folder)

    def pick_history_folder(self):
        folder = QFileDialog.getExistingDirectory(
            None, "Select Chat History Folder", ""
        )
        if folder:
            self.history_folder_changed.emit(folder)

    def pick_template_folder(self):
        folder = QFileDialog.getExistingDirectory(
            None, "Select Plot Template Folder", ""
        )
        if folder:
            self.template_folder_changed.emit(folder)
