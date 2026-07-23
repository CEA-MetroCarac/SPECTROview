#spectroview/model/m_settings.py

import json
import os

from PySide6.QtCore import QSettings

class MSettings:
    """Model: persistent application settings"""

    def __init__(self):
        self.settings = QSettings("CEA-Leti", "SPECTROview")
        self._migrate_legacy_ai_settings()

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

    # ---------- Working folder ----------
    # One user-configured root folder with 3 auto-created subfolders,
    # replacing the old separate "Fit model folder"/"Plot template folder"
    # settings so the user only configures one path.
    _WORKING_SUBFOLDERS = ("fit_model", "plot_recipe", "plot_style")

    def get_working_folder(self) -> str:
        working_folder = self.settings.value("working_folder", "", str)
        if working_folder:
            return working_folder

        # One-time migration: adopt a legacy folder setting (if any) as the
        # seed, so an existing configured path isn't orphaned.
        legacy = (
            self.settings.value("template_folder", "", str)
            or self.settings.value("model_folder", "", str)
        )
        if legacy:
            self.set_working_folder(legacy)
            return legacy
        return ""

    def set_working_folder(self, path: str):
        self.settings.setValue("working_folder", path)
        self.settings.sync()
        if path:
            for name in self._WORKING_SUBFOLDERS:
                subfolder = os.path.join(path, name)
                if not os.path.exists(subfolder):
                    try:
                        os.makedirs(subfolder)
                    except OSError as e:
                        print(f"Error creating {subfolder}: {e}")

    def get_fit_model_folder(self) -> str:
        working_folder = self.get_working_folder()
        return os.path.join(working_folder, "fit_model") if working_folder else ""

    def get_plot_recipe_folder(self) -> str:
        working_folder = self.get_working_folder()
        return os.path.join(working_folder, "plot_recipe") if working_folder else ""

    def get_plot_style_folder(self) -> str:
        working_folder = self.get_working_folder()
        return os.path.join(working_folder, "plot_style") if working_folder else ""

    # ---------- Default graph style ----------
    # User-chosen baseline style new graphs start with -- separate from
    # "Reset to Default" (graph_style.default_style()), which always means
    # the hardcoded factory values, unaffected by this.
    def get_default_graph_style(self) -> dict:
        raw = self.settings.value("default_graph_style", "", str)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}

    def set_default_graph_style(self, style: dict):
        self.settings.setValue("default_graph_style", json.dumps(style))
        self.settings.sync()

    def clear_default_graph_style(self):
        self.settings.remove("default_graph_style")
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
    #
    # Everything the AI agent persists lives under the "ai_chat/" prefix of the
    # single application store. It used to sit in a second, separate
    # QSettings("SPECTROview", "AIChat") pair, which meant the agent's settings
    # were invisible to the rest of the app and, in tests, escaped the QSettings
    # isolation fixture and read/wrote the developer's real configuration.
    # _migrate_legacy_ai_settings() carries the old values over once.

    AI_GROUP = "ai_chat"

    #: The pre-unification store, kept readable for the one-time migration.
    _LEGACY_AI_STORE = ("SPECTROview", "AIChat")
    _AI_MIGRATED_KEY = "ai_chat/_migrated_from_legacy_store"

    #: Keys the Settings dialog (AI tab) owns.
    AI_SETTING_KEYS = (
        "api_key_OpenAI", "api_key_Anthropic", "api_key_Gemini",
        "api_key_DeepSeek", "api_key_Mistral", "api_key_Custom",
        "custom_base_url", "custom_models", "history_folder",
    )

    def _migrate_legacy_ai_settings(self):
        """Copy the old QSettings("SPECTROview", "AIChat") values over once.

        Runs at most once per installation (guarded by a marker key) and never
        overwrites a value already present in the unified store, so it cannot
        undo a change made after the upgrade. The legacy store is left in place
        rather than deleted — harmless, and it keeps an older build working.
        """
        if self.settings.value(self._AI_MIGRATED_KEY, False, bool):
            return

        legacy = QSettings(*self._LEGACY_AI_STORE)
        legacy.beginGroup(self.AI_GROUP)
        for key in legacy.allKeys():
            target = f"{self.AI_GROUP}/{key}"
            if not self.settings.contains(target):
                self.settings.setValue(target, legacy.value(key))
        legacy.endGroup()

        self.settings.setValue(self._AI_MIGRATED_KEY, True)
        self.settings.sync()

    def get_ai_value(self, key: str, default=None, value_type=None):
        """Read one ``ai_chat/<key>`` setting.

        Keys are partly dynamic (``api_key_<provider>``, ``model_<provider>``),
        so this stays generic rather than growing a getter per provider.
        """
        path = f"{self.AI_GROUP}/{key}"
        if value_type is not None:
            return self.settings.value(path, default, value_type)
        return self.settings.value(path, default)

    def set_ai_value(self, key: str, value):
        """Write one ``ai_chat/<key>`` setting."""
        self.settings.setValue(f"{self.AI_GROUP}/{key}", value)
        self.settings.sync()

    def load_ai_settings(self) -> dict:
        return {key: self.get_ai_value(key, "", str) for key in self.AI_SETTING_KEYS}

    def save_ai_settings(self, data: dict):
        for key, value in data.items():
            if key in self.AI_SETTING_KEYS:
                self.set_ai_value(key, value)
