#spectroview/viewmodel/vm_settings.py

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog

from spectroview.model.m_settings import MSettings


class VMSettings(QObject):
    settings_loaded = Signal(dict)
    settings_saved = Signal()
    model_folder_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.model = MSettings()

    # ---------- Load ----------
    def load(self):
        data = self.model.load_fit_settings()
        data["model_folder"] = self.model.get_model_folder()
        self.settings_loaded.emit(data)

    # ---------- Save ----------
    def save(self, data: dict):
        model_folder = data.pop("model_folder", "")
        self.model.save_fit_settings(data)

        if model_folder:
            self.model.set_model_folder(model_folder)

        self.settings_saved.emit()

    # ---------- Folder picker ----------
    def pick_model_folder(self):
        folder = QFileDialog.getExistingDirectory(
            None, "Select Fit Model Folder", ""
        )
        if folder:
            self.model.set_model_folder(folder)
            self.model_folder_changed.emit(folder)
