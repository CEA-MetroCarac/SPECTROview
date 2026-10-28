# spectroview/model/m_settings.py
"""Backward-compatible re-export. The settings model has moved to spectroview.settings.m_settings."""

from PySide6.QtCore import QSettings  # noqa: F401

from spectroview.settings.m_settings import (  # noqa: F401
    MSettings,
    _WORKING_SUBFOLDERS,
    CONFIG_FILENAME,
    AI_SETTING_KEYS,
)
