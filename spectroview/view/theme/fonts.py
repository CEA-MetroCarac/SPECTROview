# spectroview/view/theme/fonts.py
"""
Cross-platform font configuration for SPECTROview.

Ensures consistent text rendering, typography, and readability across
Windows, macOS, and Linux desktop environments:
- Windows: 'Segoe UI' at 10pt (~13.3px at standard 96 DPI).
- macOS: '.AppleSystemUIFont' (San Francisco) at 13pt (13px at 72 DPI, matching Apple HIG).
- Linux / Other: System sans-serif at standard point size (>= 10pt).
"""
from __future__ import annotations

import sys
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication


def get_application_font(
    app: QApplication | None = None,
    platform: str | None = None,
) -> QFont:
    """Return the platform-appropriate default application font.

    Parameters
    ----------
    app : QApplication | None
        Running application instance. If ``None``, ``QApplication.instance()`` is checked.
    platform : str | None
        Target platform string (e.g. ``"darwin"``, ``"win32"``, ``"linux"``).
        Defaults to ``sys.platform`` if not specified.

    Returns
    -------
    QFont
        Configured QFont instance for the target platform.
    """
    plat = platform if platform is not None else sys.platform
    if app is None:
        app = QApplication.instance()

    if plat == "darwin":
        # macOS uses 72 DPI points where 1 pt = 1 px.
        # Apple Human Interface Guidelines set standard control/body text to 13pt.
        # .AppleSystemUIFont is the native San Francisco system font.
        font = QFont(".AppleSystemUIFont")
        point_size = app.font().pointSize() if app and app.font().pointSize() >= 12 else 13
        font.setPointSize(point_size)
        font.setStyleHint(QFont.StyleHint.SansSerif)
        return font
    elif plat.startswith("win"):
        # Windows uses 96 DPI standard scaling where 10pt = 13.33px.
        # 'Segoe UI' at 10pt provides the canonical Windows appearance.
        font = QFont("Segoe UI", 10)
        font.setStyleHint(QFont.StyleHint.SansSerif)
        return font
    else:
        # Linux / X11 / Wayland: inherit desktop font or default to sans-serif >= 10pt.
        font = QFont()
        point_size = app.font().pointSize() if app and app.font().pointSize() >= 10 else 10
        font.setPointSize(point_size)
        font.setStyleHint(QFont.StyleHint.SansSerif)
        return font


def setup_application_font(
    app: QApplication | None = None,
    platform: str | None = None,
) -> QFont:
    """Configure and apply the platform-appropriate default font to the application.

    Parameters
    ----------
    app : QApplication | None
        Target application instance. Defaults to ``QApplication.instance()``.
    platform : str | None
        Target platform identifier. Defaults to ``sys.platform``.

    Returns
    -------
    QFont
        The applied QFont instance.
    """
    if app is None:
        app = QApplication.instance()
    font = get_application_font(app=app, platform=platform)
    if app is not None:
        app.setFont(font)
    return font
