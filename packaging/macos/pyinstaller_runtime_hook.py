"""PyInstaller runtime hook for macOS Apple Silicon.

Executes before any application modules are loaded to ensure matplotlib
does not crash on macOS during first-launch font cache generation.
"""

from __future__ import annotations

import sys

if sys.platform == "darwin":
    try:
        from spectroview.model.m_macos_font_patch import apply_macos_font_cache_fix
        apply_macos_font_cache_fix()
    except Exception:
        pass
