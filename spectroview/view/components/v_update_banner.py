# spectroview/view/components/v_update_banner.py
"""
A slim, dismissable banner that appears at the top of the main window
whenever a new SPECTROview release is detected on GitHub.

------------
- Non-intrusive: slides in from the top and can be dismissed with one click.
- Theming: adapts to dark / light application theme.
- Respects user choice: 'Skip this version' stores the tag in QSettings so
  the banner never reappears for that release.
"""
import re

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QSizePolicy
)
from PySide6.QtCore import Qt, QUrl, QPropertyAnimation, QEasingCurve, Property, QByteArray
from PySide6.QtGui import QDesktopServices, QColor


# ── Palette (dark / light) ────────────────────────────────────────────────────
_DARK_STYLE = """
    QWidget#UpdateBanner {
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:0,
            stop:0 #1a3a5c,
            stop:0.5 #1e4976,
            stop:1 #1a3a5c
        );
        border-bottom: 1px solid #2d6aad;
    }
    QLabel#banner_icon { color: #64b5f6; font-size: 15px; }
    QLabel#banner_text { color: #e3f2fd; font-size: 12px; }
    QLabel#banner_version { color: #64b5f6; font-weight: bold; font-size: 12px; }
    QLabel#banner_hint { color: #90a4ae; font-size: 11px; font-style: italic; }
    QPushButton#btn_download {
        background: #1565c0;
        color: #e3f2fd;
        border: 1px solid #42a5f5;
        border-radius: 4px;
        padding: 3px 10px;
        font-size: 11px;
        font-weight: bold;
    }
    QPushButton#btn_download:hover { background: #1976d2; }
    QPushButton#btn_skip {
        background: transparent;
        color: #90a4ae;
        border: 1px solid #546e7a;
        border-radius: 4px;
        padding: 3px 8px;
        font-size: 11px;
    }
    QPushButton#btn_skip:hover { color: #cfd8dc; border-color: #78909c; }
    QPushButton#btn_dismiss {
        background: transparent;
        color: #90a4ae;
        border: none;
        font-size: 16px;
        padding: 0px 6px;
    }
    QPushButton#btn_dismiss:hover { color: #ef9a9a; }
"""

_LIGHT_STYLE = """
    QWidget#UpdateBanner {
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:0,
            stop:0 #dbeeff,
            stop:0.5 #c8e3fb,
            stop:1 #dbeeff
        );
        border-bottom: 1px solid #90caf9;
    }
    QLabel#banner_icon { color: #1565c0; font-size: 15px; }
    QLabel#banner_text { color: #1a2e46; font-size: 12px; }
    QLabel#banner_version { color: #1565c0; font-weight: bold; font-size: 12px; }
    QLabel#banner_hint { color: #607d8b; font-size: 11px; font-style: italic; }
    QPushButton#btn_download {
        background: #1976d2;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 3px 10px;
        font-size: 11px;
        font-weight: bold;
    }
    QPushButton#btn_download:hover { background: #1565c0; }
    QPushButton#btn_skip {
        background: transparent;
        color: #546e7a;
        border: 1px solid #b0bec5;
        border-radius: 4px;
        padding: 3px 8px;
        font-size: 11px;
    }
    QPushButton#btn_skip:hover { color: #37474f; border-color: #78909c; }
    QPushButton#btn_dismiss {
        background: transparent;
        color: #90a4ae;
        border: none;
        font-size: 16px;
        padding: 0px 6px;
    }
    QPushButton#btn_dismiss:hover { color: #e53935; }
"""


class VUpdateBanner(QWidget):
    """
    Animated top-of-window notification strip shown when an update is available.

    Parameters
    ----------
    tag : str
        Remote version tag, e.g. ``'v26.29.0'``.
    html_url : str
        URL of the GitHub release page.
    on_skip : callable
        Called with ``tag`` when the user clicks "Skip this version".
    on_dismiss : callable
        Called (no args) when the user clicks ✕.
    parent : QWidget, optional
    """

    def __init__(self, tag: str, html_url: str, on_skip, on_dismiss, parent=None):
        super().__init__(parent)
        self._tag = tag
        self._html_url = html_url
        self._on_skip = on_skip
        self._on_dismiss = on_dismiss

        self.setObjectName("UpdateBanner")
        self.setFixedHeight(36)
        self._build_ui()
        self.apply_theme("dark")   # default; caller can change via apply_theme()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 6, 0)
        layout.setSpacing(8)

        # 🔔 Icon
        icon_lbl = QLabel("🔔")
        icon_lbl.setObjectName("banner_icon")
        layout.addWidget(icon_lbl)

        # Text
        text_lbl = QLabel("A new version of SPECTROview is available:")
        text_lbl.setObjectName("banner_text")
        layout.addWidget(text_lbl)

        # Version tag (bold, colored)
        ver_lbl = QLabel(self._tag.lstrip("v"))
        ver_lbl.setObjectName("banner_version")
        layout.addWidget(ver_lbl)

        # Pip update hint
        hint_lbl = QLabel("— update via: pip install --upgrade spectroview")
        hint_lbl.setObjectName("banner_hint")
        layout.addWidget(hint_lbl)

        # Spacer
        layout.addStretch(1)

        # Download / changelog button
        self.btn_download = QPushButton("⬇ Download / Changelog")
        self.btn_download.setObjectName("btn_download")
        self.btn_download.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_download.clicked.connect(self._open_release_page)
        layout.addWidget(self.btn_download)

        # Skip this version
        self.btn_skip = QPushButton("Skip this version")
        self.btn_skip.setObjectName("btn_skip")
        self.btn_skip.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_skip.clicked.connect(self._skip)
        layout.addWidget(self.btn_skip)

        # Dismiss (×)
        self.btn_dismiss = QPushButton("✕")
        self.btn_dismiss.setObjectName("btn_dismiss")
        self.btn_dismiss.setFixedWidth(28)
        self.btn_dismiss.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_dismiss.clicked.connect(self._dismiss)
        layout.addWidget(self.btn_dismiss)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply_theme(self, theme_key: str):
        """Switch banner colours to match the application theme."""
        if theme_key in ("dark", "soft_dark", "classic_dark"):
            self.setStyleSheet(_DARK_STYLE)
        else:
            self.setStyleSheet(_LIGHT_STYLE)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _open_release_page(self):
        QDesktopServices.openUrl(QUrl(self._html_url))

    def _skip(self):
        self._on_skip(self._tag)
        self._dismiss()

    def _dismiss(self):
        self._on_dismiss()
        self.hide()
