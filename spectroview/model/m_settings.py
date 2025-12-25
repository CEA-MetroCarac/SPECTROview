#spectroview/model/m_settings.py

from PySide6.QtCore import QSettings
import os

class MSettings:
    """Model: persistent application settings"""

    def __init__(self):
        self.settings = QSettings()

    # ---------- Fit settings ----------
    def load_fit_settings(self) -> dict:
        return {
            "fit_negative": self.settings.value("fit_settings/fit_negative", False, bool),
            "method": self.settings.value("fit_settings/method", "Leastsq"),
            "max_ite": self.settings.value("fit_settings/max_ite", 200, int),
            "xtol": self.settings.value("fit_settings/xtol", 1e-4, float),
            "ncpu": self.settings.value("fit_settings/ncpu", 1, int),
            "maxshift": self.settings.value("fit_settings/maxshift", 20.0, float),
            "maxfwhm": self.settings.value("fit_settings/maxfwhm", 200.0, float),
        }

    def save_fit_settings(self, data: dict):
        for key, value in data.items():
            self.settings.setValue(f"fit_settings/{key}", value)
        self.settings.sync()

    # ---------- Model folder ----------
    def get_model_folder(self) -> str:
        return self.settings.value("model_folder", "", str)

    def set_model_folder(self, path: str):
        self.settings.setValue("model_folder", path)
        self.settings.sync()
