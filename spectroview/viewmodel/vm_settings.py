# spectroview/viewmodel/vm_settings.py
"""Backward-compatible re-export. The settings viewmodel has moved to spectroview.settings.vm_settings."""

from spectroview.settings.vm_settings import (  # noqa: F401
    VMSettings,
    _ConnectionTestWorker,
)
