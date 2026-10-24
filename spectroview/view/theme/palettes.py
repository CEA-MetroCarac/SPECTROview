# spectroview/view/theme/palettes.py
"""
QPalette builders for SPECTROview themes.

Each builder produces a fully configured ``QPalette`` for the Fusion style.
Colours are kept in sync with the ``ThemeTokens`` accent values; the
palette grays are hand-tuned for best Fusion rendering.
"""
from __future__ import annotations

from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt


# ═══════════════════════════════════════════════════════════════
#  Individual palette builders
# ═══════════════════════════════════════════════════════════════

def dark_palette() -> QPalette:
    """Dark palette — deep charcoal surfaces, bright white text."""
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


def light_palette() -> QPalette:
    """Light palette — soft off-white surfaces, dark text."""
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


def modern_warm_palette() -> QPalette:
    """Modern Warm palette — soft linen/sand surfaces, crisp charcoal text."""
    p = QPalette()

    p.setColor(QPalette.Window,        QColor(239, 236, 230))  # #EFECE6
    p.setColor(QPalette.Base,          QColor(250, 247, 242))  # #FAF7F2
    p.setColor(QPalette.AlternateBase, QColor(244, 240, 232))  # #F4F0E8

    p.setColor(QPalette.WindowText,      QColor(28, 28, 30))   # #1C1C1E
    p.setColor(QPalette.Text,            QColor(28, 28, 30))
    p.setColor(QPalette.ButtonText,      QColor(28, 28, 30))
    p.setColor(QPalette.PlaceholderText, QColor(156, 151, 142)) # #9C978E

    p.setColor(QPalette.Button,   QColor(232, 227, 219))       # #E8E3DB
    p.setColor(QPalette.Light,    QColor(255, 255, 255))
    p.setColor(QPalette.Midlight, QColor(240, 235, 227))
    p.setColor(QPalette.Mid,      QColor(220, 215, 206))
    p.setColor(QPalette.Dark,     QColor(191, 185, 173))

    accent = QColor(24, 24, 26)                                # #18181A
    p.setColor(QPalette.Highlight,      accent)
    p.setColor(QPalette.HighlightedText, Qt.white)
    p.setColor(QPalette.Link,           accent)

    p.setColor(QPalette.ToolTipBase, QColor(24, 24, 26))
    p.setColor(QPalette.ToolTipText, QColor(250, 247, 242))

    p.setColor(QPalette.Disabled, QPalette.Text,       QColor(168, 163, 154))
    p.setColor(QPalette.Disabled, QPalette.WindowText,  QColor(168, 163, 154))
    p.setColor(QPalette.Disabled, QPalette.ButtonText,  QColor(168, 163, 154))

    return p


# ── Lookup table ─────────────────────────────────────────────
_PALETTE_BUILDERS = {
    "dark":          dark_palette,
    "light":         light_palette,
    "modern_warm":   modern_warm_palette,
    "classic_dark":  dark_palette,
    "classic_light": light_palette,
}


def build_palette(theme_key: str) -> QPalette:
    """Return a QPalette for the given *theme_key*."""
    builder = _PALETTE_BUILDERS.get(theme_key, dark_palette)
    return builder()
