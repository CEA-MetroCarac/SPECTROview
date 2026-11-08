"""macOS font discovery fix for matplotlib on Apple Silicon / macOS Sonoma & Sequoia.

Work around an upstream bug in ``matplotlib.font_manager._get_macos_fonts``
(KeyError: '_items').

When generating its font cache on macOS, matplotlib queries system fonts by
executing ``system_profiler -xml SPFontsDataType`` via ``subprocess.check_output``
and unconditionally reads ``d['_items']``. On modern macOS systems (Sonoma,
Sequoia, Apple Silicon), ``system_profiler`` takes 10+ seconds and returns a
dictionary where the ``_items`` key is missing, causing an unhandled
``KeyError: '_items'`` at import time on any machine without a pre-existing
font cache.

By intercepting ``subprocess.check_output`` calls targeting ``system_profiler ...
SPFontsDataType`` and raising ``subprocess.CalledProcessError`` (which
matplotlib's existing ``try...except`` block explicitly catches), matplotlib
cleanly and rapidly falls back to scanning standard macOS font directories
(``/System/Library/Fonts``, ``/Library/Fonts``, ``~/Library/Fonts``), discovering
all 600+ system fonts in < 0.1s without freezing or crashing.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any, Sequence, Union


_ORIG_CHECK_OUTPUT = subprocess.check_output


def _is_system_profiler_font_query(cmd: Union[str, Sequence[Any]]) -> bool:
    """Return True if *cmd* executes macOS system_profiler for fonts."""
    if isinstance(cmd, (list, tuple)):
        if not cmd:
            return False
        first = str(cmd[0]).lower()
        if "system_profiler" in first and any("spfontsdatatype" in str(arg).lower() for arg in cmd[1:]):
            return True
    elif isinstance(cmd, str):
        lowered = cmd.lower()
        if "system_profiler" in lowered and "spfontsdatatype" in lowered:
            return True
    return False


def apply_macos_font_cache_fix() -> bool:
    """Apply early guard against matplotlib macOS font discovery crash and hang.

    Returns True if the patch was applied or already active, False if not on macOS.
    """
    if sys.platform != "darwin":
        return False

    current_check_output = subprocess.check_output
    if getattr(current_check_output, "_spectroview_macos_font_patched", False):
        return True

    def _safe_check_output(cmd: Union[str, Sequence[Any]], *args: Any, **kwargs: Any) -> Any:
        if _is_system_profiler_font_query(cmd):
            # Matplotlib's _get_macos_fonts catches CalledProcessError and returns []
            raise subprocess.CalledProcessError(1, cmd)
        return _ORIG_CHECK_OUTPUT(cmd, *args, **kwargs)

    _safe_check_output._spectroview_macos_font_patched = True
    subprocess.check_output = _safe_check_output

    # If matplotlib.font_manager has already been imported, also patch _get_macos_fonts directly
    if "matplotlib.font_manager" in sys.modules:
        try:
            fm_mod = sys.modules["matplotlib.font_manager"]
            if hasattr(fm_mod, "_get_macos_fonts"):
                fm_mod._get_macos_fonts = lambda: []
        except Exception:
            pass

    return True


def is_macos_font_cache_fix_applied() -> bool:
    """Return True if the macOS font cache interceptor is active."""
    return bool(getattr(subprocess.check_output, "_spectroview_macos_font_patched", False))
