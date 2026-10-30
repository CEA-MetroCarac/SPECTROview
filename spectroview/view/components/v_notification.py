# spectroview/view/components/v_notification.py
"""Native floating notification system for SPECTROview.

Provides a unified, lightweight, floating pill-badge notification system
integrated into the application toolbar (VMenuBar). Replaces third-party toast
notifications with a native PySide6 implementation that supports:
- Automatic placement in the available blank space of the top toolbar/header.
- Semantic thin frames: yellow (warning), green (success), blue (info/notification), red (error).
- Smooth horizontal sliding animations when multiple notifications arrive.
- Auto-dismiss countdown with hover-pause and manual close buttons.
- Light and Dark theme compatibility.
- Thread-safe triggering from any thread via NotificationManager.
"""

from __future__ import annotations

from enum import Enum
import os
import time
from typing import Any, Optional
import weakref

from shiboken6 import isValid

from PySide6.QtCore import (
    QEasingCurve,
    QObject,
    QParallelAnimationGroup,
    QPoint,
    QPropertyAnimation,
    Qt,
    QThread,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QFont,
    QFontMetrics,
    QIcon,
    QMouseEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QWidget,
)

from spectroview.view.theme.icons_svg import svg_pixmap


# ---------------------------------------------------------------------------
# Notification Types & Presets
# ---------------------------------------------------------------------------
class NotificationType(str, Enum):
    """Semantic category for notification styling and iconography."""

    SUCCESS = "success"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @classmethod
    def from_value(cls, val: Any) -> NotificationType:
        """Convert a string, enum, or legacy preset to NotificationType."""
        if isinstance(val, NotificationType):
            return val
        if val is None:
            return cls.INFO
        s = str(val).strip().lower()
        if "succ" in s:
            return cls.SUCCESS
        if "warn" in s:
            return cls.WARNING
        if "err" in s or "fail" in s or "danger" in s:
            return cls.ERROR
        return cls.INFO

    @classmethod
    def infer_from_message(cls, message: str) -> NotificationType:
        """Infer appropriate notification type from message text heuristics."""
        if not message:
            return cls.INFO
        msg_lower = message.lower()

        # Error patterns
        if any(w in msg_lower for w in ("error", "failed", "failure", "rejected", "invalid", "cannot", "exception")):
            return cls.ERROR

        # Negative / warning patterns (must precede success so e.g. "No fit model copied" is a warning)
        if (
            msg_lower.startswith("no ")
            or msg_lower.startswith("not ")
            or msg_lower.startswith("nothing")
            or msg_lower.startswith("missing")
            or msg_lower.startswith("please ")
            or any(w in msg_lower for w in (
                "warning", "skipping", "skip", "no spectrum", "no map", "no data",
                "no peak", "no baseline", "no model", "no fit", "already",
                "please wait", "not found", "nothing to", "missing"
            ))
        ):
            return cls.WARNING

        # Success patterns
        if any(w in msg_lower for w in (
            "saved", "loaded successfully", "refreshed successfully",
            "created", "sent", "validated and saved", "undid", "redid",
            "copied", "done", "complete", "finished", "success", "successfully"
        )):
            return cls.SUCCESS

        return cls.INFO


# Drop-in compatibility alias for legacy pyqttoast.ToastPreset
class ToastPreset:
    SUCCESS = NotificationType.SUCCESS
    INFO = NotificationType.INFO
    WARNING = NotificationType.WARNING
    ERROR = NotificationType.ERROR


# Color schemes for notification badges
_TYPE_COLORS: dict[NotificationType, dict[str, str]] = {
    NotificationType.SUCCESS: {
        "icon": "check_circle",
        "accent_dark": "#10b981",        # Emerald green
        "accent_light": "#059669",
        "frame_dark": "#10b981",         # Green thin frame
        "frame_light": "#059669",
        "frame_hover_dark": "#34d399",
        "frame_hover_light": "#047857",
    },
    NotificationType.INFO: {
        "icon": "info",
        "accent_dark": "#38bdf8",        # Sky blue
        "accent_light": "#0284c7",
        "frame_dark": "#38bdf8",         # Blue thin frame
        "frame_light": "#0284c7",
        "frame_hover_dark": "#7dd3fc",
        "frame_hover_light": "#0369a1",
    },
    NotificationType.WARNING: {
        "icon": "alert_triangle",
        "accent_dark": "#f59e0b",        # Amber yellow
        "accent_light": "#d97706",
        "frame_dark": "#f59e0b",         # Yellow thin frame
        "frame_light": "#d97706",
        "frame_hover_dark": "#fbbf24",
        "frame_hover_light": "#b45309",
    },
    NotificationType.ERROR: {
        "icon": "alert_circle",
        "accent_dark": "#ef4444",        # Rose red
        "accent_light": "#dc2626",
        "frame_dark": "#ef4444",         # Red thin frame
        "frame_light": "#dc2626",
        "frame_hover_dark": "#f87171",
        "frame_hover_light": "#b91c1c",
    },
}


# ---------------------------------------------------------------------------
# Individual Floating Notification Badge Widget
# ---------------------------------------------------------------------------
class VNotificationBadge(QFrame):
    """Floating pill-badge representing a single notification item."""

    dismissed = Signal(object)  # Emits self when dismissed

    BADGE_HEIGHT = 26
    BORDER_RADIUS = 13
    MAX_TEXT_WIDTH = 280

    def __init__(
        self,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
        duration: int = 3500,
        title: str | None = None,
        is_dark_theme: bool = True,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._message = message
        self._title = title
        self._type = notification_type
        self._duration = duration
        self._is_dark_theme = is_dark_theme
        self._is_dismissing = False
        self._is_hovered = False

        self.setAttribute(Qt.WA_StyledBackground, True)

        # Countdown tracking for hover pause/resume
        self._remaining_ms = max(0, duration)
        self._timer_start_time = 0.0
        self._dismiss_timer = QTimer(self)
        self._dismiss_timer.setSingleShot(True)
        self._dismiss_timer.timeout.connect(self.dismiss)

        # Smooth opacity transition effect
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)

        self._anim_opacity = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        self._anim_opacity.finished.connect(self._on_opacity_anim_finished)

        self.init_ui()
        self.apply_theme(self._is_dark_theme)

    def init_ui(self) -> None:
        """Construct badge UI layout."""
        self.setFixedHeight(self.BADGE_HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(6)

        # Icon
        self._icon_label = QLabel(self)
        self._icon_label.setFixedSize(14, 14)
        self._icon_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        layout.addWidget(self._icon_label)

        # Text Label
        self._text_label = QLabel(self)
        font = self._text_label.font()
        font.setPointSize(10)
        font.setWeight(QFont.Weight.Medium)
        self._text_label.setFont(font)
        self._text_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        full_display_text = f"{self._title}: {self._message}" if self._title else self._message
        self.setToolTip(full_display_text)

        # Elide text if too long
        fm = QFontMetrics(font)
        elided = fm.elidedText(full_display_text, Qt.TextElideMode.ElideRight, self.MAX_TEXT_WIDTH)
        self._text_label.setText(elided)
        layout.addWidget(self._text_label)

        # Close button ('×')
        self._close_btn = QToolButton(self)
        self._close_btn.setFixedSize(14, 14)
        self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close_btn.setAutoRaise(True)
        self._close_btn.clicked.connect(self.dismiss)
        layout.addWidget(self._close_btn)

        self.adjustSize()

    @property
    def frame_color(self) -> str:
        """Return the active thin frame border color for this notification badge."""
        cfg = _TYPE_COLORS.get(self._type, _TYPE_COLORS[NotificationType.INFO])
        if self._is_hovered:
            return cfg.get(
                "frame_hover_dark" if self._is_dark_theme else "frame_hover_light",
                cfg["frame_dark" if self._is_dark_theme else "frame_light"],
            )
        return cfg["frame_dark"] if self._is_dark_theme else cfg["frame_light"]

    def apply_theme(self, is_dark: bool) -> None:
        """Apply theme-specific styling and icon colors."""
        self._is_dark_theme = is_dark
        cfg = _TYPE_COLORS.get(self._type, _TYPE_COLORS[NotificationType.INFO])
        accent = cfg["accent_dark"] if is_dark else cfg["accent_light"]
        icon_name = cfg["icon"]

        # Render semantic status icon
        self._icon_label.setPixmap(svg_pixmap(icon_name, size=14, color=accent))

        # Close icon
        close_color = "#94a3b8" if is_dark else "#64748b"
        self._close_btn.setIcon(QIcon(svg_pixmap("close", size=10, color=close_color)))

        # Frame/border color: yellow for WARNING, green for SUCCESS, blue for INFO, red for ERROR
        frame_color = cfg.get("frame_dark" if is_dark else "frame_light", accent)
        hover_frame = cfg.get("frame_hover_dark" if is_dark else "frame_hover_light", frame_color)
        active_border = hover_frame if self._is_hovered else frame_color

        if is_dark:
            bg = "rgba(22, 26, 34, 0.95)"
            text_color = "#f1f5f9"
        else:
            bg = "rgba(255, 255, 255, 0.98)"
            text_color = "#0f172a"

        self.setStyleSheet(f"""
            VNotificationBadge {{
                background-color: {bg};
                border: 1px solid {active_border};
                border-radius: {self.BORDER_RADIUS}px;
            }}
            QLabel {{
                color: {text_color};
                background: transparent;
                border: none;
            }}
            QToolButton {{
                background: transparent;
                border: none;
                border-radius: 7px;
                padding: 0px;
            }}
            QToolButton:hover {{
                background-color: rgba(148, 163, 184, 0.25);
            }}
        """)

    # ── Lifecycle & Animations ──────────────────────────────────────
    def start_entrance(self, duration_ms: int = 180) -> None:
        """Fade in the notification badge and start the auto-dismiss timer."""
        # Check test/offscreen mode
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen" or "PYTEST_CURRENT_TEST" in os.environ:
            self._opacity_effect.setOpacity(1.0)
            self._start_countdown()
            return

        self._anim_opacity.stop()
        self._anim_opacity.setDuration(duration_ms)
        self._anim_opacity.setStartValue(self._opacity_effect.opacity())
        self._anim_opacity.setEndValue(1.0)
        self._anim_opacity.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim_opacity.start()
        self._start_countdown()

    def _start_countdown(self) -> None:
        """Start auto-dismiss countdown if duration > 0."""
        if self._duration > 0 and self._remaining_ms > 0:
            self._timer_start_time = time.monotonic()
            self._dismiss_timer.start(self._remaining_ms)

    def dismiss(self, immediate: bool = False) -> None:
        """Dismiss the badge with a smooth fade-out (or immediately)."""
        if self._is_dismissing:
            return
        self._is_dismissing = True
        self._dismiss_timer.stop()

        if immediate or os.environ.get("QT_QPA_PLATFORM") == "offscreen" or "PYTEST_CURRENT_TEST" in os.environ:
            self._opacity_effect.setOpacity(0.0)
            self.dismissed.emit(self)
            return

        self._anim_opacity.stop()
        self._anim_opacity.setDuration(160)
        self._anim_opacity.setStartValue(self._opacity_effect.opacity())
        self._anim_opacity.setEndValue(0.0)
        self._anim_opacity.setEasingCurve(QEasingCurve.Type.InCubic)
        self._anim_opacity.start()

    def _on_opacity_anim_finished(self) -> None:
        """Handle animation completion."""
        if self._is_dismissing and self._opacity_effect.opacity() <= 0.05:
            self.dismissed.emit(self)

    # ── Hover & Mouse Events ────────────────────────────────────────
    def enterEvent(self, event: Any) -> None:
        """Pause auto-dismiss countdown and highlight badge on mouse hover."""
        try:
            super().enterEvent(event)
        except Exception:
            pass
        self._is_hovered = True
        if self._dismiss_timer.isActive():
            elapsed_ms = int((time.monotonic() - self._timer_start_time) * 1000)
            self._remaining_ms = max(0, self._remaining_ms - elapsed_ms)
            self._dismiss_timer.stop()
        self.apply_theme(self._is_dark_theme)

    def leaveEvent(self, event: Any) -> None:
        """Resume auto-dismiss countdown on mouse leave."""
        try:
            super().leaveEvent(event)
        except Exception:
            pass
        self._is_hovered = False
        if not self._is_dismissing and self._duration > 0:
            # Resume countdown with at least 1 second left if hovered
            resume_ms = max(1000, self._remaining_ms)
            self._remaining_ms = resume_ms
            self._timer_start_time = time.monotonic()
            self._dismiss_timer.start(resume_ms)
        self.apply_theme(self._is_dark_theme)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Dismiss badge when clicked anywhere."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dismiss()
            event.accept()
        else:
            super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Notification Area Container (Inside VMenuBar)
# ---------------------------------------------------------------------------
class VNotificationArea(QWidget):
    """Container widget hosted inside VMenuBar to display and animate notification badges.

    Badges are right-aligned in the available blank space. When a new notification
    arrives, older notifications smoothly slide to the left, and the newest notification
    fades in on the right.
    """

    SPACING = 8
    RIGHT_MARGIN = 8
    LEFT_MIN_MARGIN = 12
    MAX_VISIBLE_BADGES = 3

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._badges: list[VNotificationBadge] = []
        self._is_dark_theme = True
        self._active_anim_group: QParallelAnimationGroup | None = None

        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedHeight(32)

    def set_theme(self, theme_key: str) -> None:
        """Update active theme across container and all live badges."""
        is_dark = "dark" in str(theme_key).lower()
        self._is_dark_theme = is_dark
        for badge in self._badges:
            badge.apply_theme(is_dark)

    def show_notification(
        self,
        message: str,
        notification_type: NotificationType | str = NotificationType.INFO,
        duration: int = 3500,
        title: str | None = None,
    ) -> VNotificationBadge:
        """Create and animate a new floating notification badge into the sequence."""
        if not isinstance(notification_type, NotificationType):
            notification_type = NotificationType.from_value(notification_type)

        badge = VNotificationBadge(
            message=message,
            notification_type=notification_type,
            duration=duration,
            title=title,
            is_dark_theme=self._is_dark_theme,
            parent=self,
        )
        badge.dismissed.connect(self._on_badge_dismissed)

        self._badges.append(badge)
        badge.show()

        # Enforce maximum visible count by retiring the oldest
        self._prune_overflow()

        # Animate layout: slide existing badges left, fade new badge in
        self._relayout_badges(animate=True, new_badge=badge)
        return badge

    def _prune_overflow(self) -> None:
        """Ensure badge count and total width do not exceed available space."""
        while len(self._badges) > self.MAX_VISIBLE_BADGES:
            oldest = self._badges[0]
            oldest.dismiss(immediate=True)

        # Check physical width constraints
        avail_w = max(100, self.width() - self.RIGHT_MARGIN - self.LEFT_MIN_MARGIN)
        while len(self._badges) > 1:
            total_w = sum(b.width() for b in self._badges) + (len(self._badges) - 1) * self.SPACING
            if total_w > avail_w:
                oldest = self._badges[0]
                oldest.dismiss(immediate=True)
            else:
                break

    def _on_badge_dismissed(self, badge: VNotificationBadge) -> None:
        """Clean up dismissed badge and smoothly rearrange remaining badges."""
        if badge in self._badges:
            self._badges.remove(badge)
        badge.deleteLater()
        self._relayout_badges(animate=True)

    def _relayout_badges(self, animate: bool = True, new_badge: VNotificationBadge | None = None) -> None:
        """Calculate target positions and animate badges into position."""
        if not self._badges:
            if self._active_anim_group:
                self._active_anim_group.stop()
                self._active_anim_group = None
            return

        is_test = os.environ.get("QT_QPA_PLATFORM") == "offscreen" or "PYTEST_CURRENT_TEST" in os.environ
        if is_test:
            animate = False

        if self._active_anim_group:
            self._active_anim_group.stop()
            self._active_anim_group = None

        # Position badges from right to left
        current_right = self.width() - self.RIGHT_MARGIN

        anim_group = QParallelAnimationGroup(self) if animate else None

        for badge in reversed(self._badges):
            b_width = badge.width()
            b_height = badge.height()
            target_x = current_right - b_width
            target_y = (self.height() - b_height) // 2
            target_pos = QPoint(target_x, target_y)

            current_right = target_x - self.SPACING

            if badge is new_badge:
                # New badge starts slightly shifted to the right and fades in
                start_x = target_x + 16 if animate else target_x
                badge.move(start_x, target_y)
                if animate and anim_group is not None:
                    anim_pos = QPropertyAnimation(badge, b"pos", self)
                    anim_pos.setDuration(220)
                    anim_pos.setStartValue(QPoint(start_x, target_y))
                    anim_pos.setEndValue(target_pos)
                    anim_pos.setEasingCurve(QEasingCurve.Type.OutCubic)
                    anim_group.addAnimation(anim_pos)
                else:
                    badge.move(target_pos)
                badge.start_entrance()
            else:
                # Existing badge slides smoothly to the left
                if animate and anim_group is not None and badge.pos() != target_pos:
                    anim_pos = QPropertyAnimation(badge, b"pos", self)
                    anim_pos.setDuration(220)
                    anim_pos.setStartValue(badge.pos())
                    anim_pos.setEndValue(target_pos)
                    anim_pos.setEasingCurve(QEasingCurve.Type.OutCubic)
                    anim_group.addAnimation(anim_pos)
                else:
                    badge.move(target_pos)

        if animate and anim_group is not None and anim_group.animationCount() > 0:
            self._active_anim_group = anim_group
            anim_group.start()

    def clear_all(self) -> None:
        """Immediately clear and dismiss all active notification badges."""
        if self._active_anim_group:
            self._active_anim_group.stop()
            self._active_anim_group = None

        for badge in list(self._badges):
            badge.dismiss(immediate=True)
        self._badges.clear()

    def resizeEvent(self, event: Any) -> None:
        """Recalculate badge positions on container resize."""
        super().resizeEvent(event)
        self._prune_overflow()
        self._relayout_badges(animate=False)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Ignore clicks in empty space so they pass through to toolbar."""
        event.ignore()


# ---------------------------------------------------------------------------
# Notification Manager Service (Singleton)
# ---------------------------------------------------------------------------
class NotificationManager(QObject):
    """Central singleton manager orchestrating floating notifications throughout SPECTROview."""

    _instance: Optional[NotificationManager] = None
    notification_requested = Signal(str, str, int, str)  # (message, type, duration, title)

    def __init__(self):
        super().__init__()
        self._area_ref: Optional[weakref.ref[VNotificationArea]] = None
        self._history: list[dict[str, Any]] = []
        self.notification_requested.connect(self._on_notification_requested)

    @classmethod
    def instance(cls) -> NotificationManager:
        """Return the global NotificationManager singleton."""
        if cls._instance is None:
            cls._instance = NotificationManager()
        return cls._instance

    def set_area(self, area: VNotificationArea | None) -> None:
        """Register the top toolbar notification container."""
        if area is not None:
            self._area_ref = weakref.ref(area)
            try:
                area.destroyed.connect(self._on_area_destroyed)
            except Exception:
                pass
        else:
            self._area_ref = None

    def _on_area_destroyed(self, *args) -> None:
        """Clear reference when the notification area widget is destroyed."""
        self._area_ref = None

    def get_area(self) -> Optional[VNotificationArea]:
        """Return registered notification area if still alive."""
        if self._area_ref is not None:
            area = self._area_ref()
            if area is not None:
                try:
                    if not isValid(area):
                        self._area_ref = None
                        return None
                    return area
                except Exception:
                    self._area_ref = None
                    return None
        return None

    def get_history(self) -> list[dict[str, Any]]:
        """Return list of recently triggered notifications (useful for unit tests)."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear notification history."""
        self._history.clear()

    def show(
        self,
        message: str,
        notification_type: NotificationType | str | None = None,
        duration: int = 3500,
        title: str | None = None,
        type: NotificationType | str | None = None,
    ) -> Optional[VNotificationBadge]:
        """Display a floating notification in the application toolbar."""
        if not message:
            return None

        if notification_type is None and type is not None:
            notification_type = type

        # Auto-infer type if omitted
        if notification_type is None:
            resolved_type = NotificationType.infer_from_message(message)
        else:
            resolved_type = NotificationType.from_value(notification_type)

        record = {
            "message": message,
            "type": resolved_type.value,
            "duration": duration,
            "title": title,
            "timestamp": time.time(),
        }
        self._history.append(record)
        if len(self._history) > 100:
            self._history.pop(0)

        # Thread safety: ensure GUI execution on main thread
        app = QApplication.instance()
        if app is not None and QThread.currentThread() != app.thread():
            self.notification_requested.emit(message, resolved_type.value, duration, title or "")
            return None

        return self._deliver(message, resolved_type, duration, title)

    def _deliver(
        self,
        message: str,
        notification_type: NotificationType,
        duration: int,
        title: str | None,
    ) -> Optional[VNotificationBadge]:
        """Direct delivery to active VNotificationArea."""
        area = self.get_area()
        if area is not None:
            try:
                return area.show_notification(
                    message=message,
                    notification_type=notification_type,
                    duration=duration,
                    title=title,
                )
            except (RuntimeError, Exception):
                self._area_ref = None
                return None
        return None

    def _on_notification_requested(self, message: str, type_str: str, duration: int, title: str) -> None:
        """Slot for thread-safe cross-thread notification requests."""
        self._deliver(
            message=message,
            notification_type=NotificationType.from_value(type_str),
            duration=duration,
            title=title if title else None,
        )

    def success(self, message: str, duration: int = 3500, title: str | None = None) -> Optional[VNotificationBadge]:
        return self.show(message, NotificationType.SUCCESS, duration, title)

    def info(self, message: str, duration: int = 3500, title: str | None = None) -> Optional[VNotificationBadge]:
        return self.show(message, NotificationType.INFO, duration, title)

    def warning(self, message: str, duration: int = 3500, title: str | None = None) -> Optional[VNotificationBadge]:
        return self.show(message, NotificationType.WARNING, duration, title)

    def error(self, message: str, duration: int = 3500, title: str | None = None) -> Optional[VNotificationBadge]:
        return self.show(message, NotificationType.ERROR, duration, title)

    def clear(self) -> None:
        """Clear all active badges."""
        area = self.get_area()
        if area is not None:
            area.clear_all()


# ---------------------------------------------------------------------------
# Convenience Free Functions
# ---------------------------------------------------------------------------
def show_notification(
    message: str,
    notification_type: NotificationType | str | None = None,
    duration: int = 3500,
    title: str | None = None,
    parent: QWidget | None = None,
    type: NotificationType | str | None = None,
) -> Optional[VNotificationBadge]:
    """Universal function to show a floating notification in SPECTROview."""
    try:
        return NotificationManager.instance().show(
            message=message,
            notification_type=notification_type,
            duration=duration,
            title=title,
            type=type,
        )
    except Exception:
        return None
