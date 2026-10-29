# spectroview/settings/__init__.py
"""Unified settings package for SPECTROview: model, viewmodel, and UI."""

from __future__ import annotations

from spectroview.settings.m_settings import MSettings
from spectroview.settings.vm_settings import VMSettings
from spectroview.settings.build_info import (
    TRACKED_PACKAGES,
    generate_build_snapshot,
    get_build_number,
    get_environment_summary,
    get_runtime_package_versions,
    is_frozen_app,
    load_build_snapshot,
)
from spectroview.settings.ui import (
    SettingsDialog,
    VSettingsDialog,
    GeneralTabPage,
    FitSettingsTabPage,
    AISettingsTabPage,
    AboutTabPage,
)

__all__ = [
    "MSettings",
    "VMSettings",
    "SettingsDialog",
    "VSettingsDialog",
    "GeneralTabPage",
    "FitSettingsTabPage",
    "AISettingsTabPage",
    "AboutTabPage",
    "TRACKED_PACKAGES",
    "generate_build_snapshot",
    "get_build_number",
    "get_environment_summary",
    "get_runtime_package_versions",
    "is_frozen_app",
    "load_build_snapshot",
]
