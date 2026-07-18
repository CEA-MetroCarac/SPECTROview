#spectroview/model/m_settings.py

from PySide6.QtCore import QSettings

class MSettings:
    """Model: persistent application settings"""

    def __init__(self):
        self.settings = QSettings("CEA-Leti", "SPECTROview")

    # ---------- Fit settings ----------
    def load_fit_settings(self) -> dict:
        return {
            "fit_negative": self.settings.value("fit_settings/fit_negative", False, bool),
            "max_ite": self.settings.value("fit_settings/max_ite", 200, int),
            "xtol": self.settings.value("fit_settings/xtol", 1e-4, float),
            "ftol": self.settings.value("fit_settings/ftol", 1e-4, float),
            "coef_noise": self.settings.value("fit_settings/coef_noise", 0.0, float),
            "maxshift": self.settings.value("fit_settings/maxshift", 20.0, float),
            "maxfwhm": self.settings.value("fit_settings/maxfwhm", 200.0, float),
            "minfwhm": self.settings.value("fit_settings/minfwhm", 0.1, float),
        }

    def save_fit_settings(self, data: dict):
        for key, value in data.items():
            self.settings.setValue(f"fit_settings/{key}", value)
        self.settings.sync()

    # ---------- View Options ----------
    def load_view_options(self) -> dict:
        """Load view options of spectra viewer from settings"""
        return {
            "theme": self.settings.value("view_options/theme", "Dark Mode", str),
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
            "noise_level": self.settings.value("view_options/noise_level", False, bool),
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

    # ---------- Export Options (Graph Workspace figure export) ----------
    def load_export_options(self) -> dict:
        """Last-used figure export settings, remembered across dialog opens."""
        return {
            "format": self.settings.value("export_options/format", "png", str),
            "dpi": self.settings.value("export_options/dpi", 300, int),
            "transparent": self.settings.value("export_options/transparent", False, bool),
            "theme": self.settings.value("export_options/theme", "Light Mode", str),
        }

    def save_export_options(self, data: dict):
        for key, value in data.items():
            self.settings.setValue(f"export_options/{key}", value)
        self.settings.sync()

    # ---------- Model folder ----------
    def get_model_folder(self) -> str:
        return self.settings.value("model_folder", "", str)

    def set_model_folder(self, path: str):
        self.settings.setValue("model_folder", path)
        self.settings.sync()

    # ---------- Template folder ----------
    def get_template_folder(self) -> str:
        return self.settings.value("template_folder", "", str)

    def set_template_folder(self, path: str):
        self.settings.setValue("template_folder", path)
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
        return self.settings.value("theme", "dark", str)
    
    def set_theme(self, theme: str):
        """Set theme (light or dark)."""
        self.settings.setValue("theme", theme)
        self.settings.sync()

    # ---------- Update checker ----------
    def get_check_for_updates(self) -> bool:
        """Whether automatic update checks are enabled (default: True)."""
        return self.settings.value("update_checker/enabled", True, bool)

    def set_check_for_updates(self, enabled: bool):
        self.settings.setValue("update_checker/enabled", enabled)
        self.settings.sync()

    def get_skipped_version(self) -> str:
        """Return the version tag the user chose to skip, or empty string."""
        return self.settings.value("update_checker/skipped_version", "", str)

    def set_skipped_version(self, tag: str):
        self.settings.setValue("update_checker/skipped_version", tag)
        self.settings.sync()

    def get_last_check_date(self) -> str:
        """ISO date string (YYYY-MM-DD) of the last successful update check."""
        return self.settings.value("update_checker/last_check_date", "", str)

    def set_last_check_date(self, date_str: str):
        self.settings.setValue("update_checker/last_check_date", date_str)
        self.settings.sync()

    # ---------- AI Settings ----------
    def load_ai_settings(self) -> dict:
        s = QSettings("SPECTROview", "AIChat")
        s.beginGroup("ai_chat")
        data = {
            "api_key_OpenAI": s.value("api_key_OpenAI", "", str),
            "api_key_Anthropic": s.value("api_key_Anthropic", "", str),
            "api_key_Gemini": s.value("api_key_Gemini", "", str),
            "api_key_DeepSeek": s.value("api_key_DeepSeek", "", str),
            "api_key_Mistral": s.value("api_key_Mistral", "", str),
            "api_key_Custom": s.value("api_key_Custom", "", str),
            "custom_base_url": s.value("custom_base_url", "", str),
            "history_folder": s.value("history_folder", "", str),
        }
        s.endGroup()
        return data

    def save_ai_settings(self, data: dict):
        s = QSettings("SPECTROview", "AIChat")
        s.beginGroup("ai_chat")
        for key, value in data.items():
            if key in ["api_key_OpenAI", "api_key_Anthropic", "api_key_Gemini", "api_key_DeepSeek", "api_key_Mistral", "api_key_Custom", "custom_base_url", "history_folder"]:
                s.setValue(key, value)
        s.endGroup()

    # ---------- AI Agent unlock (feature not yet publicly released) ----------
    def get_ai_agent_enabled(self) -> bool:
        """Whether the AI Agent button/feature is unlocked (default: hidden)."""
        return self.settings.value("ai_agent/enabled", False, bool)

    def set_ai_agent_enabled(self, enabled: bool):
        self.settings.setValue("ai_agent/enabled", enabled)
        self.settings.sync()
