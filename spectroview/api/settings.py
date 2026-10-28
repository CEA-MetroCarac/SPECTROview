"""Headless access to SPECTROview's persistent configuration.

Backed by `spectroview.settings.MSettings`, which stores values in
Qt's `QSettings` (native OS-backed store: plist on macOS, registry on
Windows, INI on Linux). Constructing `MSettings()` never requires a running
`QApplication` event loop — only `PySide6.QtCore` importable, which is
already a hard dependency of the `spectroview` package.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Union

from spectroview.settings import MSettings


def get_fit_defaults() -> Dict[str, Any]:
    """Return the fit_params defaults the GUI's Fit Settings dialog uses.

    Keys: fit_negative, max_ite, xtol, ftol, coef_noise, maxshift, maxfwhm, minfwhm, loss, f_scale.
    """
    return MSettings().load_fit_settings()


def set_fit_defaults(**kwargs: Any) -> None:
    """Persist one or more fit defaults (see `get_fit_defaults` for keys)."""
    MSettings().save_fit_settings(kwargs)


def get_working_folder() -> str:
    """Return the configured universal SPECTROview Working Folder.

    Four subfolders are auto-created under it by `set_working_folder()`:
    fit_model/ (fit-model JSON templates), plot_recipe/ (Graph Workspace
    plot recipes), plot_style/ (Graph Workspace style templates),
    ai_chat_history/ (AI chat logs), and config.json (portable settings backup).
    """
    return MSettings().get_working_folder()


def set_working_folder(path: Union[str, Path]) -> None:
    """Set the universal Working Folder, creating its fit_model/, plot_recipe/,
    plot_style/, and ai_chat_history/ subfolders and config.json if they don't already exist.
    If an existing config.json is present in the specified folder, settings are automatically restored.
    """
    MSettings().set_working_folder(str(path))


def get_fit_model_folder() -> str:
    """Return the folder SPECTROview scans for fit-model JSON templates
    (the `fit_model` subfolder of the Working Folder)."""
    return MSettings().get_fit_model_folder()


def get_plot_recipe_folder() -> str:
    """Return the folder for Graph Workspace plot recipes
    (the `plot_recipe` subfolder of the Working Folder)."""
    return MSettings().get_plot_recipe_folder()


def get_plot_style_folder() -> str:
    """Return the folder for Graph Workspace style templates
    (the `plot_style` subfolder of the Working Folder)."""
    return MSettings().get_plot_style_folder()


def get_template_folder() -> str:
    """Return the folder for templates
    (the `template` subfolder of the Working Folder)."""
    return MSettings().get_template_folder()


def get_history_folder() -> str:
    """Return the folder where AI chat history JSON files are stored
    (the `ai_chat_history` subfolder of the Working Folder)."""
    return MSettings().get_history_folder()


def get_ai_history_folder() -> str:
    """Alias for get_history_folder()."""
    return MSettings().get_history_folder()


def export_config(filepath: Optional[Union[str, Path]] = None) -> Optional[str]:
    """Export configuration to a JSON file (defaults to config.json in Working Folder)."""
    return MSettings().export_config(str(filepath) if filepath else None)


def import_config(filepath: Optional[Union[str, Path]] = None) -> bool:
    """Import configuration from a JSON file (defaults to config.json in Working Folder)."""
    return MSettings().import_config(str(filepath) if filepath else None)


def get_last_directory() -> str:
    """Return the last directory used for file dialogs (GUI and API share this)."""
    return MSettings().get_last_directory()


def set_last_directory(path: Union[str, Path]) -> None:
    MSettings().set_last_directory(str(path))


def get_mcp_settings() -> Dict[str, Any]:
    """Return the opt-in local MCP endpoint configuration."""
    return MSettings().load_mcp_settings()


def set_mcp_settings(enabled: bool, port: int = 8765) -> None:
    """Enable/disable the loopback MCP endpoint for the desktop application.

    A running application applies changes made through its Settings dialog
    immediately. Programmatic changes take effect the next time SPECTROview
    starts or its settings are re-applied.
    """
    MSettings().save_mcp_settings(enabled, port)
