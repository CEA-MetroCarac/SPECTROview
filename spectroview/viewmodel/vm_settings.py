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
    ai_agent_visibility_changed = Signal(bool)  # emitted when the unlock code is entered

    # Secret codes gating the (unreleased) AI Agent feature. Not case-sensitive.
    _AI_AGENT_UNLOCK_CODE = "AIAGENT"
    _AI_AGENT_LOCK_CODE = "AIAGENTOFF"

    def __init__(self):
        super().__init__()
        self.settings = MSettings()

    # ---------- Load ----------
    def load(self):
        data = self.settings.load_fit_settings()
        data["model_folder"] = self.settings.get_model_folder()
        data["template_folder"] = self.settings.get_template_folder()
        ai_data = self.settings.load_ai_settings()
        data.update(ai_data)
        self.settings_loaded.emit(data)

    # ---------- Save ----------
    def save(self, data: dict):
        model_folder = data.pop("model_folder", "")
        template_folder = data.pop("template_folder", "")
        
        ai_keys = ["api_key_OpenAI", "api_key_Anthropic", "api_key_Gemini",
                   "api_key_DeepSeek", "api_key_Mistral", "api_key_Custom", "custom_base_url", "history_folder"]
        ai_data = {k: data.pop(k, "") for k in ai_keys if k in data}
        
        self.settings.save_fit_settings(data)
        self.settings.save_ai_settings(ai_data)

        if model_folder:
            self.settings.set_model_folder(model_folder)
        self.settings.set_template_folder(template_folder)

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

    # ---------- AI Agent unlock ----------
    def try_ai_agent_code(self, code: str):
        """Check a secret code typed by the user against the AI Agent unlock/lock
        codes. Persists the new state and emits `ai_agent_visibility_changed`
        when the code matches. Returns True/False on match, None otherwise.
        """
        code = code.strip().upper()
        if code == self._AI_AGENT_UNLOCK_CODE:
            self.settings.set_ai_agent_enabled(True)
            self.ai_agent_visibility_changed.emit(True)
            return True
        if code == self._AI_AGENT_LOCK_CODE:
            self.settings.set_ai_agent_enabled(False)
            self.ai_agent_visibility_changed.emit(False)
            return False
        return None
