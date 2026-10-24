# spectroview/view/theme/__init__.py
"""
Theme package for SPECTROview.

Public API
----------
- ``ThemeManager``      — apply / switch themes at runtime.
- ``ThemeTokens``       — frozen colour-token dataclass.
- ``DARK_TOKENS``, ``LIGHT_TOKENS`` — preset instances.
- ``THEME_TOKENS``      — dict mapping theme key → tokens.
- ``build_palette()``   — create a QPalette for a given theme key.
- ``build_stylesheet()``— render the QSS template for a ThemeTokens instance.
- ``get_application_font()`` — platform-appropriate application font.
- ``setup_application_font()`` — configure and apply default application font.

Backward compatibility
~~~~~~~~~~~~~~~~~~~~~~
Legacy imports ``from spectroview.view.style import dark_palette, …`` should
be updated to use this package.  The old ``style.py`` is removed.
"""

from .manager import ThemeManager
from .tokens import (
    ThemeTokens,
    DARK_TOKENS,
    LIGHT_TOKENS,
    MODERN_WARM_TOKENS,
    THEME_TOKENS,
)
from .palettes import build_palette, dark_palette, light_palette, modern_warm_palette
from .qss_template import build_stylesheet
from .fonts import get_application_font, setup_application_font

__all__ = [
    "ThemeManager",
    "ThemeTokens",
    "DARK_TOKENS",
    "LIGHT_TOKENS",
    "MODERN_WARM_TOKENS",
    "THEME_TOKENS",
    "build_palette",
    "build_stylesheet",
    "get_application_font",
    "setup_application_font",
    "dark_palette",
    "light_palette",
    "modern_warm_palette",
]
