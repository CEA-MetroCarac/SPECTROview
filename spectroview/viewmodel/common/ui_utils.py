"""UI notification, alert, icon, and list-item status helpers."""

from __future__ import annotations

from functools import lru_cache
import os

from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QListWidgetItem, QMessageBox

from spectroview.i18n import tr


@lru_cache(maxsize=128)
def get_tinted_icon(path: str, color_str: str) -> QIcon:
    """Return a QIcon created from the image at ``path`` tinted to ``color_str``."""
    base = os.path.basename(path).lower()
    from spectroview.view.theme import LEGACY_ICON_MAP, svg_icon

    vec_key = LEGACY_ICON_MAP.get(base)
    if not vec_key and "." in base:
        vec_key = LEGACY_ICON_MAP.get(base.rsplit(".", 1)[0])

    if vec_key:
        is_standard_fg = (
            not color_str
            or color_str.upper()
            in (
                "#EDEDEF",
                "#F0F0F0",
                "#FFFFFF",
                "#141519",
                "#404040",
                "#1C1C1E",
                "#000000",
                "#333333",
                "#E0E0E0",
                "#EEEEEE",
                "#222222",
                "#D8D8D8",
            )
        )
        resolved_color = None if is_standard_fg else color_str
        return svg_icon(vec_key, color=resolved_color)

    pixmap = QPixmap(path)
    if pixmap.isNull():
        return QIcon()
    painter = QPainter(pixmap)
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), QColor(color_str))
    painter.end()
    return QIcon(pixmap)


def set_spectrum_item_color(item: QListWidgetItem, spectrum_info: dict):
    """Set list item background color based on spectrum processing status.

    Status hierarchy:
    1. Fitted (has_fit): converged -> success tint, unconverged -> error tint
    2. Baselined (has_baseline): accent tint
    3. Cropped (is_cropped): muted tint
    4. Original/Reinit: transparent
    """
    from spectroview.view.theme import current_tokens

    tok = current_tokens()

    is_cropped = spectrum_info.get("is_cropped", False)
    has_baseline = spectrum_info.get("has_baseline", False)
    has_fit = spectrum_info.get("has_fit", False)
    fit_success = spectrum_info.get("fit_success", False)

    if has_fit:
        c = QColor(tok.success if fit_success else tok.error)
        c.setAlpha(120)
        item.setBackground(c)
    elif has_baseline:
        c = QColor(tok.accent)
        c.setAlpha(120)
        item.setBackground(c)
    elif is_cropped:
        c = QColor(tok.text_muted)
        c.setAlpha(100)
        item.setBackground(c)
    else:
        item.setBackground(QColor(0, 0, 0, 0))


def show_alert(message):
    """Show a modal warning message box."""
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setWindowTitle(tr("Alert"))
    msg_box.setText(message)
    msg_box.exec_()


def show_toast_notification(
    parent=None, message="", title=None, duration=3000, preset=None
):
    """Show an auto-dismissing notification (backward-compatible adapter).

    Routes directly into SPECTROview's native floating notification system.
    """
    from spectroview.view.components.common.v_notification import (
        NotificationManager,
        NotificationType,
    )

    ntype = (
        NotificationType.from_value(preset)
        if preset is not None
        else NotificationType.infer_from_message(message)
    )
    return NotificationManager.instance().show(
        message=message,
        notification_type=ntype,
        duration=duration,
        title=title,
    )


def show_notification(
    message: str,
    notification_type=None,
    duration: int = 3500,
    title: str | None = None,
    parent=None,
):
    """Show a native floating notification in the SPECTROview application toolbar."""
    from spectroview.view.components.common.v_notification import NotificationManager

    return NotificationManager.instance().show(
        message=message,
        notification_type=notification_type,
        duration=duration,
        title=title,
    )
