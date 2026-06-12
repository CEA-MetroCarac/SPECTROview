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
    p.setColor(QPalette.Window,        QColor(30, 30, 32))
    p.setColor(QPalette.Base,          QColor(55, 55, 58))
    p.setColor(QPalette.AlternateBase, QColor(62, 62, 65))

    # Text
    p.setColor(QPalette.WindowText,      QColor(240, 240, 240))
    p.setColor(QPalette.Text,            QColor(240, 240, 240))
    p.setColor(QPalette.ButtonText,      QColor(235, 235, 235))
    p.setColor(QPalette.PlaceholderText, QColor(130, 130, 130))

    # Buttons / controls
    p.setColor(QPalette.Button, QColor(52, 52, 55))
    p.setColor(QPalette.Light,  QColor(80, 80, 84))
    p.setColor(QPalette.Mid,    QColor(62, 62, 66))
    p.setColor(QPalette.Dark,   QColor(35, 35, 38))
    p.setColor(QPalette.Shadow, QColor(16, 16, 18))

    # Tooltips
    p.setColor(QPalette.ToolTipBase, QColor(48, 48, 50))
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


def soft_dark_palette() -> QPalette:
    """Soft-dark palette — neutral warm grays, slightly lighter than dark."""
    p = QPalette()

    p.setColor(QPalette.Window,        QColor(50, 50, 52))
    p.setColor(QPalette.Base,          QColor(72, 72, 75))
    p.setColor(QPalette.AlternateBase, QColor(78, 78, 82))

    p.setColor(QPalette.WindowText,      QColor(235, 235, 235))
    p.setColor(QPalette.Text,            QColor(235, 235, 235))
    p.setColor(QPalette.ButtonText,      QColor(230, 230, 230))
    p.setColor(QPalette.PlaceholderText, QColor(135, 135, 135))

    p.setColor(QPalette.Button, QColor(66, 66, 68))
    p.setColor(QPalette.Light,  QColor(92, 92, 96))
    p.setColor(QPalette.Mid,    QColor(76, 76, 80))
    p.setColor(QPalette.Dark,   QColor(44, 44, 46))
    p.setColor(QPalette.Shadow, QColor(26, 26, 28))

    p.setColor(QPalette.ToolTipBase, QColor(64, 64, 66))
    p.setColor(QPalette.ToolTipText, QColor(235, 235, 235))

    accent = QColor(64, 156, 255)
    p.setColor(QPalette.Highlight,      accent)
    p.setColor(QPalette.HighlightedText, Qt.white)
    p.setColor(QPalette.Link,           QColor(100, 180, 255))

    p.setColor(QPalette.Disabled, QPalette.Text,       QColor(115, 115, 115))
    p.setColor(QPalette.Disabled, QPalette.ButtonText,  QColor(115, 115, 115))
    p.setColor(QPalette.Disabled, QPalette.WindowText,  QColor(115, 115, 115))

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


# ── Lookup table ─────────────────────────────────────────────
_PALETTE_BUILDERS = {
    "dark":          dark_palette,
    "soft_dark":     soft_dark_palette,
    "light":         light_palette,
    "classic_dark":  dark_palette,
    "classic_light": light_palette,
}


def build_palette(theme_key: str) -> QPalette:
    """Return a QPalette for the given *theme_key*."""
    builder = _PALETTE_BUILDERS.get(theme_key, dark_palette)
    return builder()
