# spectroview/view/theme/manager.py
"""
Centralised theme manager for SPECTROview.

``ThemeManager`` owns the canonical *current theme* state, applies
palette + QSS + style reset, synchronizes Windows 11 DWM native title bar colors,
and exposes helper properties used by workspaces for icon tinting and plot styling.
"""
from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QApplication, QWidget

from .tokens import ThemeTokens, THEME_TOKENS, DARK_TOKENS
from .palettes import build_palette
from .qss_template import build_stylesheet

if TYPE_CHECKING:
    from spectroview.model.m_settings import MSettings


# Classic themes use palette-only (no QSS).
_CLASSIC_THEMES = {"classic_dark", "classic_light"}

# Mapping from theme key → spectra-viewer combobox display name.
_VIEWER_THEME_NAMES = {
    "dark":          "Dark Mode",
    "light":         "Light Mode",
    "classic_dark":  "Dark Mode",
    "classic_light": "Light Mode",
}


# ── Native Windows 11 DWM Title Bar Integration ───────────────
_in_title_bar_theme: bool = False


def apply_window_title_bar_theme(
    window: QWidget | int,
    tokens: ThemeTokens | None = None,
) -> None:
    """Set the Windows native title bar caption, text, and border colors to match the theme."""
    if sys.platform != "win32" or os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return

    global _in_title_bar_theme
    if _in_title_bar_theme:
        return

    _in_title_bar_theme = True
    try:
        if isinstance(window, int):
            hwnd = window
        else:
            if not hasattr(window, "winId") or not window.isWindow():
                return
            if not window.isVisible() and getattr(window, "internalWinId", lambda: 0)() == 0:
                return
            hwnd = int(window.winId())

        if not hwnd:
            return

        import ctypes
        from ctypes import wintypes

        tok = tokens if tokens is not None else current_tokens()
        is_dark = tok.is_dark

        dwm = ctypes.WinDLL("dwmapi", use_last_error=True)

        # 20: DWMWA_USE_IMMERSIVE_DARK_MODE (Windows 10 18985+ / Windows 11)
        # 34: DWMWA_BORDER_COLOR (Windows 11)
        # 35: DWMWA_CAPTION_COLOR (Windows 11)
        # 36: DWMWA_TEXT_COLOR (Windows 11)
        dark_flag = wintypes.BOOL(1 if is_dark else 0)
        dwm.DwmSetWindowAttribute(
            wintypes.HWND(hwnd),
            20,
            ctypes.byref(dark_flag),
            ctypes.sizeof(dark_flag),
        )

        def _hex_to_colorref(hex_str: str) -> int:
            h = hex_str.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return r | (g << 8) | (b << 16)

        caption_val = _hex_to_colorref(tok.sidebar_bg)
        text_val = _hex_to_colorref(tok.text_primary)
        border_val = _hex_to_colorref(tok.border)

        caption_color = wintypes.DWORD(caption_val)
        text_color = wintypes.DWORD(text_val)
        border_color = wintypes.DWORD(border_val)

        dwm.DwmSetWindowAttribute(
            wintypes.HWND(hwnd),
            35,
            ctypes.byref(caption_color),
            ctypes.sizeof(caption_color),
        )
        dwm.DwmSetWindowAttribute(
            wintypes.HWND(hwnd),
            36,
            ctypes.byref(text_color),
            ctypes.sizeof(text_color),
        )
        dwm.DwmSetWindowAttribute(
            wintypes.HWND(hwnd),
            34,
            ctypes.byref(border_color),
            ctypes.sizeof(border_color),
        )
    except (AttributeError, OSError, Exception):
        return
    finally:
        _in_title_bar_theme = False


class _TitleBarThemeFilter(QObject):
    """Event filter that ensures any top-level window gets the themed title bar on show."""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Show and isinstance(watched, QWidget) and watched.isWindow():
            apply_window_title_bar_theme(watched, current_tokens())
        return super().eventFilter(watched, event)


# ── ThemeManager Class ─────────────────────────────────────────

class ThemeManager:
    """Single owner of application-wide theme state."""

    _instance: ThemeManager | None = None

    @classmethod
    def instance(cls) -> ThemeManager | None:
        """Return the active singleton instance of ThemeManager."""
        return cls._instance

    def __init__(self, settings: "MSettings") -> None:
        ThemeManager._instance = self
        self._settings = settings
        raw = settings.get_theme() or "dark"
        if raw == "soft_dark":
            raw = "dark"
            settings.set_theme("dark")
        elif raw in ("modern_warm", "warm"):
            raw = "light"
            settings.set_theme("light")
        self._current: str = raw
        self._title_bar_filter: _TitleBarThemeFilter | None = None

    # ── Properties ────────────────────────────────────────────

    @property
    def current_theme(self) -> str:
        """Return the current theme key (e.g. ``"dark"``)."""
        return self._current

    @property
    def is_dark(self) -> bool:
        """``True`` when the active theme has a dark background."""
        return self._current in ("dark", "classic_dark")

    @property
    def tokens(self) -> ThemeTokens:
        """Return the active ``ThemeTokens``."""
        return THEME_TOKENS.get(self._current, THEME_TOKENS["dark"])

    @property
    def icon_color(self) -> str:
        """Return the hex colour for tinted GUI icons (e.g. ``"#EDEDEF"``)."""
        return self.tokens.icon_color

    @property
    def workspace_theme(self) -> str:
        """Return ``"dark"`` or ``"light"`` for downstream icon switching."""
        return "dark" if self.is_dark else "light"

    @property
    def viewer_theme_name(self) -> str:
        """Return the display name for the spectra-viewer theme combo."""
        return _VIEWER_THEME_NAMES.get(self._current, "Light Mode" if not self.is_dark else "Dark Mode")

    # ── Core application ──────────────────────────────────────

    def apply(self, theme_key: str | None = None) -> None:
        """Apply *theme_key* (palette + QSS + title bar) to the running application."""
        app = QApplication.instance()
        if app is None:
            return

        if theme_key is None:
            theme_key = self._settings.get_theme() or "dark"
        if theme_key == "soft_dark":
            theme_key = "dark"
        elif theme_key in ("modern_warm", "warm"):
            theme_key = "light"

        self._current = theme_key
        tokens = THEME_TOKENS.get(theme_key, THEME_TOKENS["dark"])

        # 1) Qt ColorScheme hint
        try:
            if hasattr(app.styleHints(), "setColorScheme"):
                app.styleHints().setColorScheme(
                    Qt.ColorScheme.Dark if tokens.is_dark else Qt.ColorScheme.Light
                )
        except Exception:
            pass

        # 2) Palette
        app.setPalette(build_palette(theme_key))

        # 3) Stylesheet (empty for classic themes)
        if theme_key in _CLASSIC_THEMES:
            app.setStyleSheet("")
        else:
            app.setStyleSheet(build_stylesheet(tokens))

        # 4) Native Windows title bar theme
        if self._title_bar_filter is None and sys.platform == "win32" and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            self._title_bar_filter = _TitleBarThemeFilter(app)
            app.installEventFilter(self._title_bar_filter)

        if sys.platform == "win32" and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            for widget in app.topLevelWidgets():
                if widget.isWindow() and widget.isVisible():
                    apply_window_title_bar_theme(widget, tokens)

        # 5) Flush icon caches so theme-dependent icons repaint instantly
        try:
            from .icons_svg import clear_icon_caches
            clear_icon_caches()
        except Exception:
            pass

        # 6) Persist
        self._settings.set_theme(theme_key)


def current_tokens() -> ThemeTokens:
    """Return the active theme tokens, falling back to DARK_TOKENS."""
    mgr = ThemeManager.instance()
    if mgr is not None:
        return mgr.tokens
    return DARK_TOKENS
