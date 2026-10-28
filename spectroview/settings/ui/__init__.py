# spectroview/settings/ui/__init__.py
"""UI layer for application settings."""

from __future__ import annotations

from spectroview.settings.ui.settings_dialog import SettingsDialog, VSettingsDialog
from spectroview.settings.ui.tab_general import GeneralTabPage
from spectroview.settings.ui.tab_fit import FitSettingsTabPage
from spectroview.settings.ui.tab_ai import AISettingsTabPage
from spectroview.settings.ui.tab_about import AboutTabPage, get_runtime_package_versions

__all__ = [
    "SettingsDialog",
    "VSettingsDialog",
    "GeneralTabPage",
    "FitSettingsTabPage",
    "AISettingsTabPage",
    "AboutTabPage",
    "get_runtime_package_versions",
]
