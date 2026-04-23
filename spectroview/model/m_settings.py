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

    # ---------- View Options ----------
    def load_view_options(self) -> dict:
        """Load view options of spectra viewer from settings"""
        return {
            "theme": self.settings.value("view_options/theme", "Light Mode", str),
            "xaxis": self.settings.value("view_options/xaxis", "Wavenumber (cm⁻¹)", str),
            "yaxis": self.settings.value("view_options/yaxis", "Intensity (a.u.)", str),
            "yscale": self.settings.value("view_options/yscale", "Linear", str),
            "plotstyle": self.settings.value("view_options/plotstyle", "line", str),
            "lw": self.settings.value("view_options/lw", 1.5, float),
            "dotsize": self.settings.value("view_options/dotsize", 3.0, float),
            "raw": self.settings.value("view_options/raw", False, bool),
            "bestfit_colorful": self.settings.value("view_options/bestfit_colorful", True, bool),
            "show_peak_label": self.settings.value("view_options/show_peak_label", False, bool),
            "residual": self.settings.value("view_options/residual", False, bool),
            "grid": self.settings.value("view_options/grid", False, bool),
            "width": self.settings.value("view_options/width", "5.5", str),
            "height": self.settings.value("view_options/height", "4.0", str),
            "legend": self.settings.value("view_options/legend", False, bool),
            "bestfit": self.settings.value("view_options/bestfit", True, bool),
            "copy_fig_theme": self.settings.value("view_options/copy_fig_theme", "Light Mode", str),
        }

    def save_view_options(self, data: dict):
        for key, value in data.items():
            self.settings.setValue(f"view_options/{key}", value)
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
