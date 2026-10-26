# spectroview/view/theme/palettes.py
"""
QPalette builders for SPECTROview themes.

Each builder produces a fully configured ``QPalette`` for the Fusion style.
Modern themes use ``_palette_for(ThemeTokens)`` synchronized with PLUME's
design tokens; classic themes preserve the legacy hand-tuned Fusion grays.
"""
from __future__ import annotations

from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt

from .tokens import ThemeTokens, DARK_TOKENS, LIGHT_TOKENS


# ═══════════════════════════════════════════════════════════════
#  PLUME-synchronized Modern Palette Builder
# ═══════════════════════════════════════════════════════════════

def _palette_for(t: ThemeTokens) -> QPalette:
    """Build a QPalette strictly synchronized with a ThemeTokens instance."""
    p = QPalette()
    p.setColor(QPalette.Window, QColor(t.background))
    p.setColor(QPalette.WindowText, QColor(t.text_primary))
    p.setColor(QPalette.Base, QColor(t.surface))
    p.setColor(QPalette.AlternateBase, QColor(t.surface_elevated))
    p.setColor(QPalette.Text, QColor(t.text_primary))
    p.setColor(QPalette.BrightText, QColor("#FFFFFF"))

    p.setColor(QPalette.Button, QColor(t.surface_elevated))
    p.setColor(QPalette.ButtonText, QColor(t.text_primary))

    p.setColor(QPalette.ToolTipBase, QColor(t.surface_elevated))
    p.setColor(QPalette.ToolTipText, QColor(t.text_primary))

    p.setColor(QPalette.Highlight, QColor(t.accent))
    p.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))

    p.setColor(QPalette.Link, QColor(t.accent))
    p.setColor(QPalette.LinkVisited, QColor(t.accent_hover))
    p.setColor(QPalette.PlaceholderText, QColor(t.text_muted))

    # Disabled state
    p.setColor(QPalette.Disabled, QPalette.WindowText, QColor(t.text_muted))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(t.text_muted))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(t.text_muted))
    p.setColor(QPalette.Disabled, QPalette.Highlight, QColor(t.border))
    p.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor(t.text_muted))

    # Shadows and bevels
    p.setColor(QPalette.Light, QColor(t.surface_elevated))
    p.setColor(QPalette.Midlight, QColor(t.surface))
    p.setColor(QPalette.Mid, QColor(t.border))
    p.setColor(QPalette.Dark, QColor(t.border_strong))
    p.setColor(QPalette.Shadow, QColor(t.background))

    return p


def modern_dark_palette() -> QPalette:
    """Modern Dark palette synchronized with DARK_TOKENS."""
    return _palette_for(DARK_TOKENS)


def modern_light_palette() -> QPalette:
    """Modern Light palette synchronized with LIGHT_TOKENS."""
    return _palette_for(LIGHT_TOKENS)


# ═══════════════════════════════════════════════════════════════
#  Classic / Legacy Palette Builders (Unchanged)
# ═══════════════════════════════════════════════════════════════

def classic_dark_palette() -> QPalette:
    """Classic dark palette — deep charcoal surfaces, bright white text."""
    p = QPalette()

    # Base surfaces
    p.setColor(QPalette.Window,        QColor(33, 33, 33))
    p.setColor(QPalette.Base,          QColor(60, 60, 60))
    p.setColor(QPalette.AlternateBase, QColor(68, 68, 68))

    # Text
    p.setColor(QPalette.WindowText,      QColor(240, 240, 240))
    p.setColor(QPalette.Text,            QColor(240, 240, 240))
    p.setColor(QPalette.ButtonText,      QColor(235, 235, 235))
    p.setColor(QPalette.PlaceholderText, QColor(135, 135, 135))

    # Buttons / controls
    p.setColor(QPalette.Button, QColor(58, 58, 58))
    p.setColor(QPalette.Light,  QColor(85, 85, 85))
    p.setColor(QPalette.Mid,    QColor(68, 68, 68))
    p.setColor(QPalette.Dark,   QColor(38, 38, 38))
    p.setColor(QPalette.Shadow, QColor(18, 18, 18))

    # Tooltips
    p.setColor(QPalette.ToolTipBase, QColor(55, 55, 55))
    p.setColor(QPalette.ToolTipText, QColor(240, 240, 240))

    # Accent
    accent = QColor(64, 156, 255)
    p.setColor(QPalette.Highlight,      accent)
    p.setColor(QPalette.HighlightedText, Qt.white)
    p.setColor(QPalette.Link,           QColor(100, 180, 255))

    # Disabled
    p.setColor(QPalette.Disabled, QPalette.Text,       QColor(110, 110, 110))
    p.setColor(QPalette.Disabled, QPalette.ButtonText,  QColor(110, 110, 110))
    p.setColor(QPalette.Disabled, QPalette.WindowText,  QColor(110, 110, 110))

    return p


def classic_light_palette() -> QPalette:
    """Classic light palette — soft off-white surfaces, dark text."""
    p = QPalette()

    p.setColor(QPalette.Window,        QColor(232, 234, 237))
    p.setColor(QPalette.Base,          QColor(255, 255, 255))
    p.setColor(QPalette.AlternateBase, QColor(240, 240, 240))

    p.setColor(QPalette.WindowText,      QColor(28, 28, 28))
    p.setColor(QPalette.Text,            QColor(28, 28, 28))
    p.setColor(QPalette.ButtonText,      QColor(32, 32, 32))
    p.setColor(QPalette.PlaceholderText, QColor(148, 148, 148))

    p.setColor(QPalette.Button,   QColor(235, 235, 235))
    p.setColor(QPalette.Light,    QColor(255, 255, 255))
    p.setColor(QPalette.Midlight, QColor(218, 218, 218))
    p.setColor(QPalette.Mid,      QColor(198, 198, 198))
    p.setColor(QPalette.Dark,     QColor(158, 158, 158))

    accent = QColor(42, 130, 218)
    p.setColor(QPalette.Highlight,      accent)
    p.setColor(QPalette.HighlightedText, Qt.white)
    p.setColor(QPalette.Link,           QColor(42, 130, 218))

    p.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    p.setColor(QPalette.ToolTipText, QColor(28, 28, 28))

    p.setColor(QPalette.Disabled, QPalette.Text,       QColor(158, 158, 158))
    p.setColor(QPalette.Disabled, QPalette.WindowText,  QColor(158, 158, 158))
    p.setColor(QPalette.Disabled, QPalette.ButtonText,  QColor(158, 158, 158))

    return p


# Backward compatibility aliases
dark_palette = classic_dark_palette
light_palette = classic_light_palette


# ── Lookup table ─────────────────────────────────────────────
_PALETTE_BUILDERS = {
    "dark":          modern_dark_palette,
    "light":         modern_light_palette,
    "classic_dark":  classic_dark_palette,
    "classic_light": classic_light_palette,
}


def build_palette(theme_key: str) -> QPalette:
    """Return a QPalette for the given *theme_key*."""
    if theme_key in ("modern_warm", "warm"):
        return modern_light_palette()
    builder = _PALETTE_BUILDERS.get(theme_key, modern_dark_palette)
    return builder()
