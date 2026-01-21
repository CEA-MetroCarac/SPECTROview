#spectroview/model/m_settings.py

from PySide6.QtCore import QSettings
import os

class MSettings:
    """Model: persistent application settings"""

    def __init__(self):
        self.settings = QSettings("CEA-Leti", "SPECTROview")

    # ---------- Fit settings ----------
    def load_fit_settings(self) -> dict:
        return {
            "fit_negative": self.settings.value("fit_settings/fit_negative", False, bool),
            "method": self.settings.value("fit_settings/method", "Leastsq"),
            "max_ite": self.settings.value("fit_settings/max_ite", 200, int),
            "xtol": self.settings.value("fit_settings/xtol", 1e-5, float),
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
    
    # ---------- Last directory ----------
    def get_last_directory(self) -> str:
        """Get the last working directory used for file operations."""
        return self.settings.value("last_directory", "/", str)
    
    def set_last_directory(self, path: str):
        """Set the last working directory for file operations."""
        self.settings.setValue("last_directory", path)
        self.settings.sync()
    
    # ---------- Theme ----------
    def get_theme(self) -> str:
        """Get current theme (light or dark)."""
        return self.settings.value("theme", "light", str)
    
    def set_theme(self, theme: str):
        """Set theme (light or dark)."""
        self.settings.setValue("theme", theme)
        self.settings.sync()
