# spectroview/view/theme/manager.py
"""
Centralised theme manager for SPECTROview.

``ThemeManager`` replaces the sprawling ``toggle_theme()`` method that lived
in ``main.py``.  It owns the canonical *current theme* state, applies
palette + QSS + Fusion-style reset, and exposes helper properties used by
workspaces for icon tinting and plot-style selection.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QApplication

from .tokens import ThemeTokens, THEME_TOKENS
from .palettes import build_palette
from .qss_template import build_stylesheet

if TYPE_CHECKING:
    from spectroview.model.m_settings import MSettings


# Classic themes use palette-only (no QSS).
_CLASSIC_THEMES = {"classic_dark", "classic_light"}

# Mapping from theme key → spectra-viewer combobox display name.
_VIEWER_THEME_NAMES = {
    "dark":          "Dark Mode",
    "soft_dark":     "Soft Dark Mode",
    "light":         "Light Mode",
    "classic_dark":  "Dark Mode",
    "classic_light": "Light Mode",
}


class ThemeManager:
    """Single owner of application-wide theme state.

    Typical usage inside ``Main.__init__``::

        self.theme_mgr = ThemeManager(self.settings)
        self.theme_mgr.apply(self.settings.get_theme())
    """

    def __init__(self, settings: "MSettings") -> None:
        self._settings = settings
        self._current: str = settings.get_theme() or "dark"

    # ── Properties ────────────────────────────────────────────

    @property
    def current_theme(self) -> str:
        """Return the current theme key (e.g. ``"dark"``)."""
        return self._current

    @property
    def is_dark(self) -> bool:
        """``True`` when the active theme has a dark background."""
        return self._current in ("dark", "soft_dark", "classic_dark")

    @property
    def tokens(self) -> ThemeTokens:
        """Return the active ``ThemeTokens``."""
        return THEME_TOKENS.get(self._current, THEME_TOKENS["dark"])

    @property
    def workspace_theme(self) -> str:
        """Return ``"dark"`` or ``"light"`` for downstream icon switching."""
        return "dark" if self.is_dark else "light"

    @property
    def viewer_theme_name(self) -> str:
        """Return the display name for the spectra-viewer theme combo."""
        return _VIEWER_THEME_NAMES.get(self._current, "Dark Mode")

    # ── Core application ──────────────────────────────────────

    def apply(self, theme_key: str | None = None) -> None:
        """Apply *theme_key* (palette + QSS) to the running application.

        If *theme_key* is ``None`` the stored setting is used.
        """
        app = QApplication.instance()
        if app is None:
            return

        if theme_key is None:
            theme_key = self._settings.get_theme() or "dark"
        self._current = theme_key

        # 1) Palette
        app.setPalette(build_palette(theme_key))

        # 2) Stylesheet (empty for classic themes)
        if theme_key in _CLASSIC_THEMES:
            app.setStyleSheet("")
        else:
            tokens = THEME_TOKENS.get(theme_key, THEME_TOKENS["dark"])
            app.setStyleSheet(build_stylesheet(tokens))

        # 3) Force Fusion style refresh to clear cached QSS
        app.setStyle("Windows")
        app.setStyle("Fusion")

        # 4) Deep unpolish / polish to eradicate stuck QSS
        for widget in app.allWidgets():
            widget.style().unpolish(widget)
            widget.style().polish(widget)

        # 5) Persist
        self._settings.set_theme(theme_key)
