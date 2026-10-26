# spectroview/view/theme/__init__.py
"""
Theme package for SPECTROview.

Public API
----------
- ``ThemeManager``               — apply / switch themes at runtime.
- ``current_tokens``             — returns active ThemeTokens.
- ``apply_window_title_bar_theme`` — apply native Windows 11 DWM theme to top-level window.
- ``ThemeTokens``                — frozen colour-token dataclass.
- ``DARK_TOKENS``, ``LIGHT_TOKENS`` — preset token instances.
- ``CLASSIC_DARK_TOKENS``, ``CLASSIC_LIGHT_TOKENS`` — classic preset instances.
- ``THEME_TOKENS``               — dict mapping theme key → tokens.
- ``build_palette()``            — create a QPalette for a given theme key.
- ``modern_dark_palette()``, ``modern_light_palette()`` — token-synchronized palettes.
- ``classic_dark_palette()``, ``classic_light_palette()`` — legacy Fusion palettes.
- ``build_stylesheet()``         — render the QSS template for a ThemeTokens instance.
- ``get_application_font()``     — platform-appropriate application font.
- ``setup_application_font()``   — configure and apply default application font.
"""

from .manager import ThemeManager, current_tokens, apply_window_title_bar_theme
from .tokens import (
    ThemeTokens,
    DARK_TOKENS,
    LIGHT_TOKENS,
    CLASSIC_DARK_TOKENS,
    CLASSIC_LIGHT_TOKENS,
    THEME_TOKENS,
)
from .palettes import (
    build_palette,
    dark_palette,
    light_palette,
    modern_dark_palette,
    modern_light_palette,
    classic_dark_palette,
    classic_light_palette,
)
from .qss_template import build_stylesheet
from .fonts import (
    get_application_font,
    setup_application_font,
)

__all__ = [
    "ThemeManager",
    "current_tokens",
    "apply_window_title_bar_theme",
    "ThemeTokens",
    "DARK_TOKENS",
    "LIGHT_TOKENS",
    "CLASSIC_DARK_TOKENS",
    "CLASSIC_LIGHT_TOKENS",
    "THEME_TOKENS",
    "build_palette",
    "build_stylesheet",
    "get_application_font",
    "setup_application_font",
    "dark_palette",
    "light_palette",
    "modern_dark_palette",
    "modern_light_palette",
    "classic_dark_palette",
    "classic_light_palette",
]
