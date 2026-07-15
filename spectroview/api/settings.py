"""Headless access to SPECTROview's persistent configuration.

Backed by `spectroview.model.m_settings.MSettings`, which stores values in
Qt's `QSettings` (native OS-backed store: plist on macOS, registry on
Windows, INI on Linux). Constructing `MSettings()` never requires a running
`QApplication` event loop — only `PySide6.QtCore` importable, which is
already a hard dependency of the `spectroview` package.
"""
from pathlib import Path
from typing import Any, Dict, Union

from spectroview.model.m_settings import MSettings


def get_fit_defaults() -> Dict[str, Any]:
    """Return the fit_params defaults the GUI's Fit Settings dialog uses.

    Keys: fit_negative, max_ite, xtol, ftol, coef_noise, maxshift, maxfwhm, minfwhm.
    """
    return MSettings().load_fit_settings()


def set_fit_defaults(**kwargs: Any) -> None:
    """Persist one or more fit defaults (see `get_fit_defaults` for keys)."""
    MSettings().save_fit_settings(kwargs)


def get_model_folder() -> str:
    """Return the folder SPECTROview scans for fit-model JSON templates."""
    return MSettings().get_model_folder()


def set_model_folder(path: Union[str, Path]) -> None:
    """Set the folder SPECTROview scans for fit-model JSON templates."""
    MSettings().set_model_folder(str(path))


def get_last_directory() -> str:
    """Return the last directory used for file dialogs (GUI and API share this)."""
    return MSettings().get_last_directory()


def set_last_directory(path: Union[str, Path]) -> None:
    MSettings().set_last_directory(str(path))
