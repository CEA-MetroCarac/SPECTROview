# spectroview/settings/m_settings.py
"""Model: persistent application settings stored in QSettings and portable JSON."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Union

from PySide6.QtCore import QSettings


class MSettings:
    """Model: persistent application settings"""

    def __init__(self):
        self.settings = QSettings("CEA-Leti", "SPECTROview")
        self._is_importing = False
        self._batch_saving = False
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
            "loss": self.settings.value("fit_settings/loss", "linear", str),
            "f_scale": self.settings.value("fit_settings/f_scale", 1.0, float),
        }

    def save_fit_settings(self, data: dict):
        for key, value in data.items():
            self.settings.setValue(f"fit_settings/{key}", value)
        self.settings.sync()
        self._auto_export_config()

    # ---------- View Options ----------
    def load_view_options(self) -> dict:
        """Load view options of spectra viewer from settings"""
        theme = self.settings.value("view_options/theme", "Dark Mode", str)
        if theme == "Soft Dark Mode":
            theme = "Dark Mode"
        copy_theme = self.settings.value("view_options/copy_fig_theme", "Light Mode", str)
        if copy_theme == "Soft Dark Mode":
            copy_theme = "Light Mode"
        return {
            "theme": theme,
            "xaxis": self.settings.value("view_options/xaxis", "Wavenumber (cm⁻¹)", str),
            "yaxis": self.settings.value("view_options/yaxis", "Intensity (a.u.)", str),
            "yscale": self.settings.value("view_options/yscale", "Linear", str),
            "plotstyle": self.settings.value("view_options/plotstyle", "line", str),
            "color_palette": self.settings.value(
                "view_options/color_palette", "DEFAULT_COLORS", str),
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
            "legend_outside": self.settings.value(
                "view_options/legend_outside", False, bool),
            "bestfit": self.settings.value("view_options/bestfit", True, bool),
            "max_legend_items": self.settings.value(
                "view_options/max_legend_items", 15, int),
            "copy_fig_theme": copy_theme,
        }

    def save_view_options(self, data: dict):
        for key, value in data.items():
            self.settings.setValue(f"view_options/{key}", value)
        self.settings.sync()
        self._auto_export_config()

    # ---------- Export Options (Graph Workspace figure export) ----------
    def load_export_options(self) -> dict:
        """Last-used figure export settings, remembered across dialog opens."""
        theme = self.settings.value("export_options/theme", "Light Mode", str)
        if theme == "Soft Dark Mode":
            theme = "Dark Mode"
        return {
            "format": self.settings.value("export_options/format", "png", str),
            "dpi": self.settings.value("export_options/dpi", 300, int),
            "transparent": self.settings.value("export_options/transparent", False, bool),
            "theme": theme,
        }

    def save_export_options(self, data: dict):
        for key, value in data.items():
            self.settings.setValue(f"export_options/{key}", value)
        self.settings.sync()
        self._auto_export_config()

    # ---------- Working folder ----------
    # Universal user-configured root folder with auto-created subfolders:
    # fit_model/ (fit models), plot_recipe/ (recipes), plot_style/ (styles),
    # ai_chat_history/ (chat logs), template/ (templates), plus config.json.
    _WORKING_SUBFOLDERS = ("fit_model", "plot_recipe", "plot_style", "ai_chat_history", "template")
    CONFIG_FILENAME = "config.json"

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
                        os.makedirs(subfolder, exist_ok=True)
                    except OSError as e:
                        print(f"Error creating {subfolder}: {e}")

            # Restore existing config or save initial config into working folder
            config_file = os.path.join(path, self.CONFIG_FILENAME)
            alt_config_file = os.path.join(path, "spectroview_config.json")
            if os.path.isfile(config_file):
                self.import_config(config_file)
            elif os.path.isfile(alt_config_file):
                self.import_config(alt_config_file)
            else:
                self.export_config(config_file)

    def get_fit_model_folder(self) -> str:
        working_folder = self.get_working_folder()
        return os.path.join(working_folder, "fit_model") if working_folder else ""

    def get_plot_recipe_folder(self) -> str:
        working_folder = self.get_working_folder()
        return os.path.join(working_folder, "plot_recipe") if working_folder else ""

    def get_plot_style_folder(self) -> str:
        working_folder = self.get_working_folder()
        return os.path.join(working_folder, "plot_style") if working_folder else ""

    def get_template_folder(self) -> str:
        """Return the folder for templates (the 'template' subfolder of the Working Folder)."""
        working_folder = self.get_working_folder()
        return os.path.join(working_folder, "template") if working_folder else ""

    def get_history_folder(self) -> str:
        """Return the folder where AI chat history JSON files are stored.

        Derives from the universal working folder's 'ai_chat_history' subfolder.
        Falls back to legacy 'ai_chat/history_folder' if working folder is unset.
        """
        working_folder = self.get_working_folder()
        if working_folder:
            return os.path.join(working_folder, "ai_chat_history")
        return self.get_ai_value("history_folder", "", str)

    def get_ai_history_folder(self) -> str:
        """Alias for get_history_folder()."""
        return self.get_history_folder()

    # ---------- Portable JSON Configuration (Backup & Restore) ----------
    def get_config_dict(self) -> dict:
        """Assemble persistent application configuration into a dictionary."""
        ai_data = {}
        self.settings.beginGroup(self.AI_GROUP)
        for k in self.settings.allKeys():
            if not k.startswith("_") and k != "history_folder":
                ai_data[k] = self.settings.value(k)
        self.settings.endGroup()

        return {
            "version": "1.0",
            "fit_settings": self.load_fit_settings(),
            "view_options": self.load_view_options(),
            "export_options": self.load_export_options(),
            "theme": self.get_theme(),
            "default_graph_style": self.get_default_graph_style(),
            "mcp": self.load_mcp_settings(),
            "ai_settings": ai_data,
        }

    def apply_config_dict(self, data: dict) -> None:
        """Apply a configuration dictionary into QSettings."""
        if not isinstance(data, dict):
            return

        self._is_importing = True
        try:
            if "fit_settings" in data and isinstance(data["fit_settings"], dict):
                self.save_fit_settings(data["fit_settings"])

            if "view_options" in data and isinstance(data["view_options"], dict):
                self.save_view_options(data["view_options"])

            if "export_options" in data and isinstance(data["export_options"], dict):
                self.save_export_options(data["export_options"])

            if "theme" in data and isinstance(data["theme"], str):
                self.set_theme(data["theme"])

            if "default_graph_style" in data and isinstance(data["default_graph_style"], dict):
                self.set_default_graph_style(data["default_graph_style"])

            if "mcp" in data and isinstance(data["mcp"], dict):
                mcp = data["mcp"]
                if "mcp_enabled" in mcp and "mcp_port" in mcp:
                    try:
                        self.save_mcp_settings(bool(mcp["mcp_enabled"]), int(mcp["mcp_port"]))
                    except Exception:
                        pass

            if "ai_settings" in data and isinstance(data["ai_settings"], dict):
                for k, v in data["ai_settings"].items():
                    if not k.startswith("_") and k != "history_folder":
                        self.set_ai_value(k, v)
        finally:
            self._is_importing = False

    def export_config(self, filepath: Optional[str] = None) -> Optional[str]:
        """Export settings to a JSON file. Defaults to config.json in working folder."""
        if getattr(self, "_is_importing", False):
            return None
        if not filepath:
            working = self.get_working_folder()
            if not working or not os.path.isdir(working):
                return None
            filepath = os.path.join(working, self.CONFIG_FILENAME)

        try:
            config_dict = self.get_config_dict()
            tmp_path = f"{filepath}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            if os.path.exists(filepath):
                os.replace(tmp_path, filepath)
            else:
                os.rename(tmp_path, filepath)
            return filepath
        except Exception as e:
            print(f"Error exporting config to {filepath}: {e}")
            if os.path.exists(f"{filepath}.tmp"):
                try:
                    os.remove(f"{filepath}.tmp")
                except OSError:
                    pass
            return None

    def import_config(self, filepath: Optional[str] = None) -> bool:
        """Import settings from a JSON file. Defaults to config.json in working folder."""
        if not filepath:
            working = self.get_working_folder()
            if not working or not os.path.isdir(working):
                return False
            filepath = os.path.join(working, self.CONFIG_FILENAME)
            if not os.path.exists(filepath):
                alt = os.path.join(working, "spectroview_config.json")
                if os.path.exists(alt):
                    filepath = alt

        if not os.path.isfile(filepath):
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.apply_config_dict(data)
            return True
        except Exception as e:
            print(f"Error importing config from {filepath}: {e}")
            return False

    def _auto_export_config(self) -> None:
        """Silently sync settings to config.json if a working folder exists."""
        if getattr(self, "_is_importing", False):
            return
        working = self.get_working_folder()
        if working and os.path.isdir(working):
            self.export_config()

    # ---------- Default graph style ----------
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
        self._auto_export_config()

    def clear_default_graph_style(self):
        self.settings.remove("default_graph_style")
        self.settings.sync()
        self._auto_export_config()

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
        """Get current theme (dark, light, classic_dark, classic_light)."""
        val = self.settings.value("theme", "dark", str)
        if val == "soft_dark":
            self.set_theme("dark")
            return "dark"
        if val in ("modern_warm", "warm"):
            self.set_theme("light")
            return "light"
        if val not in ("dark", "light", "classic_dark", "classic_light"):
            self.set_theme("dark")
            return "dark"
        return val

    def set_theme(self, theme: str):
        """Set theme (dark, light, classic_dark, classic_light)."""
        if theme == "soft_dark":
            theme = "dark"
        elif theme in ("modern_warm", "warm"):
            theme = "light"
        elif theme not in ("dark", "light", "classic_dark", "classic_light"):
            theme = "dark"
        self.settings.setValue("theme", theme)
        self.settings.sync()
        self._auto_export_config()

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
    AI_GROUP = "ai_chat"

    #: The pre-unification store, kept readable for the one-time migration.
    _LEGACY_AI_STORE = ("SPECTROview", "AIChat")
    _AI_MIGRATED_KEY = "ai_chat/_migrated_from_legacy_store"

    #: Keys the Settings dialog owns.
    AI_SETTING_KEYS = (
        "api_key_OpenAI", "api_key_Anthropic", "api_key_Gemini",
        "api_key_DeepSeek", "api_key_Mistral", "api_key_Custom",
        "custom_base_url", "custom_models", "custom_label", "history_folder",
    )

    def _migrate_legacy_ai_settings(self):
        """Copy the old QSettings("SPECTROview", "AIChat") values over once."""
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
        """Read one ``ai_chat/<key>`` setting."""
        path = f"{self.AI_GROUP}/{key}"
        if value_type is not None:
            return self.settings.value(path, default, value_type)
        return self.settings.value(path, default)

    def set_ai_value(self, key: str, value):
        """Write one ``ai_chat/<key>`` setting."""
        self.settings.setValue(f"{self.AI_GROUP}/{key}", value)
        self.settings.sync()
        if not getattr(self, "_batch_saving", False):
            self._auto_export_config()

    def load_ai_settings(self) -> dict:
        res = {key: self.get_ai_value(key, "", str) for key in self.AI_SETTING_KEYS}
        working = self.get_working_folder()
        if working:
            res["history_folder"] = self.get_history_folder()
        return res

    def save_ai_settings(self, data: dict):
        self._batch_saving = True
        try:
            for key, value in data.items():
                if key in self.AI_SETTING_KEYS:
                    self.set_ai_value(key, value)
        finally:
            self._batch_saving = False
        self._auto_export_config()

    # ---------- Local MCP server ----------
    def load_mcp_settings(self) -> dict:
        """Return local MCP settings (disabled and loopback-only by default)."""
        return {
            "mcp_enabled": self.settings.value("mcp/enabled", False, bool),
            "mcp_port": self.settings.value("mcp/port", 8765, int),
            "mcp_host": "127.0.0.1",
        }

    def save_mcp_settings(self, enabled: bool, port: int) -> None:
        port = int(port)
        if not 1024 <= port <= 65535:
            raise ValueError("MCP port must be between 1024 and 65535.")
        self.settings.setValue("mcp/enabled", bool(enabled))
        self.settings.setValue("mcp/port", port)
        self.settings.sync()
        self._auto_export_config()


# Module-level aliases for backward compatibility
_WORKING_SUBFOLDERS = MSettings._WORKING_SUBFOLDERS
CONFIG_FILENAME = MSettings.CONFIG_FILENAME
AI_SETTING_KEYS = MSettings.AI_SETTING_KEYS
