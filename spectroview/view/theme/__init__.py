# spectroview/view/theme/__init__.py
"""
Theme package for SPECTROview.

Public API
----------
- ``ThemeManager``      — apply / switch themes at runtime.
- ``ThemeTokens``       — frozen colour-token dataclass.
- ``DARK_TOKENS``, ``SOFT_DARK_TOKENS``, ``LIGHT_TOKENS`` — preset instances.
- ``THEME_TOKENS``      — dict mapping theme key → tokens.
- ``build_palette()``   — create a QPalette for a given theme key.
- ``build_stylesheet()``— render the QSS template for a ThemeTokens instance.

Backward compatibility
~~~~~~~~~~~~~~~~~~~~~~
Legacy imports ``from spectroview.view.style import dark_palette, …`` should
be updated to use this package.  The old ``style.py`` is removed.
"""

from .manager import ThemeManager
from .tokens import (
    ThemeTokens,
    DARK_TOKENS,
    SOFT_DARK_TOKENS,
    LIGHT_TOKENS,
    THEME_TOKENS,
)
from .palettes import build_palette, dark_palette, soft_dark_palette, light_palette
from .qss_template import build_stylesheet

__all__ = [
    "ThemeManager",
    "ThemeTokens",
    "DARK_TOKENS",
    "SOFT_DARK_TOKENS",
    "LIGHT_TOKENS",
    "THEME_TOKENS",
    "build_palette",
    "build_stylesheet",
    "dark_palette",
    "soft_dark_palette",
    "light_palette",
]
