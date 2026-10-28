# spectroview/view/components/customize_graph/common.py
"""Shared navigation delegate and styling helpers for CustomizeGraphDialog."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QModelIndex, QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QScrollArea,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from spectroview.view.theme import ThemeManager


def highlight_group_style() -> str:
    """Style applied to matching QGroupBoxes during search filtering."""
    return (
        "QGroupBox { border: 1.5px solid #3B82F6; border-radius: 6px; margin-top: 10px; font-weight: bold; } "
        "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 4px; color: #3B82F6; }"
    )


class NavItemDelegate(QStyledItemDelegate):
    """Renders sidebar items with rounded hover/selection styling and accent pill indicator."""

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        painter.save()
        try:
            painter.setRenderHint(QPainter.Antialiasing)

            rect = option.rect
            is_selected = bool(option.state & QStyle.State_Selected)
            is_hover = bool(option.state & QStyle.State_MouseOver)

            mgr = ThemeManager.instance()
            is_dark = mgr.is_dark if mgr is not None else True
            t = mgr.tokens if mgr is not None else None

            # Color palette
            accent_col = QColor(t.accent) if t else QColor("#3B82F6")
            text_primary = QColor(t.text_primary) if t else (QColor("#FFFFFF") if is_dark else QColor("#111827"))
            text_secondary = QColor(t.text_secondary) if t else (QColor("#9CA3AF") if is_dark else QColor("#6B7280"))
            border_col = QColor(t.border) if t else (QColor("#374151") if is_dark else QColor("#E5E7EB"))
            elevated_bg = QColor(t.surface_elevated) if t else (QColor("#25282F") if is_dark else QColor("#FFFFFF"))

            # Background pill
            bg_rect = rect.adjusted(6, 2, -6, -2)
            if is_selected:
                painter.setPen(QPen(border_col, 1))
                painter.setBrush(elevated_bg)
                painter.drawRoundedRect(bg_rect, 6, 6)

                # Left accent indicator pill
                pill_rect = QRectF(rect.left() + 8, rect.center().y() - 9, 3.5, 18)
                painter.setPen(Qt.NoPen)
                painter.setBrush(accent_col)
                painter.drawRoundedRect(pill_rect, 1.75, 1.75)
            elif is_hover:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(255, 255, 255, 18) if is_dark else QColor(0, 0, 0, 12))
                painter.drawRoundedRect(bg_rect, 6, 6)

            # Draw icon
            icon: Optional[QIcon] = index.data(Qt.DecorationRole)
            icon_x = rect.left() + 20
            if icon and not icon.isNull():
                icon_rect = QRect(icon_x, int(rect.center().y() - 10), 20, 20)
                icon.paint(painter, icon_rect, Qt.AlignCenter)
                text_x = icon_x + 28
            else:
                text_x = icon_x + 6

            # Draw text
            text = index.data(Qt.DisplayRole)
            text_rect = QRect(text_x, rect.top(), rect.right() - text_x - 8, rect.height())
            font = option.font
            font.setPointSize(10)
            if is_selected:
                font.setBold(True)
                painter.setFont(font)
                painter.setPen(text_primary)
            else:
                painter.setFont(font)
                painter.setPen(text_secondary)

            painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, text)
        finally:
            painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        return QSize(210, 40)


def create_scrollable_container(parent: QWidget) -> tuple[QScrollArea, QWidget, QVBoxLayout]:
    """Helper to create a standard frameless QScrollArea with a compact inner container."""
    outer = QVBoxLayout(parent)
    outer.setContentsMargins(0, 0, 0, 0)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)

    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(8)

    scroll.setWidget(container)
    outer.addWidget(scroll)

    return scroll, container, layout
