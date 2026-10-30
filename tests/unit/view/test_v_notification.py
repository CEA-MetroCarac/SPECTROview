# tests/unit/view/test_v_notification.py
"""Unit tests for the SPECTROview native floating notification system."""

from PySide6.QtCore import Qt, QEvent, QSize
from PySide6.QtGui import QResizeEvent

from spectroview.view.components.v_notification import (
    NotificationType,
    ToastPreset,
    VNotificationBadge,
    VNotificationArea,
    NotificationManager,
    show_notification,
)
from spectroview.viewmodel.utils import show_toast_notification


# ---------------------------------------------------------------------------
# NotificationType & Classification Tests
# ---------------------------------------------------------------------------
def test_notification_type_from_value():
    """Verify conversion of various inputs to NotificationType."""
    assert NotificationType.from_value("success") == NotificationType.SUCCESS
    assert NotificationType.from_value("SUCCESS") == NotificationType.SUCCESS
    assert NotificationType.from_value("successful") == NotificationType.SUCCESS
    assert NotificationType.from_value("warning") == NotificationType.WARNING
    assert NotificationType.from_value("warn") == NotificationType.WARNING
    assert NotificationType.from_value("error") == NotificationType.ERROR
    assert NotificationType.from_value("failed") == NotificationType.ERROR
    assert NotificationType.from_value("danger") == NotificationType.ERROR
    assert NotificationType.from_value("info") == NotificationType.INFO
    assert NotificationType.from_value(None) == NotificationType.INFO
    assert NotificationType.from_value(NotificationType.WARNING) == NotificationType.WARNING

    # Test ToastPreset drop-in aliases
    assert ToastPreset.SUCCESS == NotificationType.SUCCESS
    assert ToastPreset.ERROR == NotificationType.ERROR
    assert ToastPreset.WARNING == NotificationType.WARNING
    assert ToastPreset.INFO == NotificationType.INFO


def test_notification_type_infer_from_message():
    """Verify semantic classification heuristics from message strings."""
    # Success patterns
    assert NotificationType.infer_from_message("Map 'test' saved to CSV.") == NotificationType.SUCCESS
    assert NotificationType.infer_from_message("Sent 5 spectra to Spectra workspace.") == NotificationType.SUCCESS
    assert NotificationType.infer_from_message("DataFrame 'df' refreshed successfully") == NotificationType.SUCCESS
    assert NotificationType.infer_from_message("DataFrame 'best_fit_results' sent successfully to Graphs workspace.") == NotificationType.SUCCESS
    assert NotificationType.infer_from_message("Undid last action.") == NotificationType.SUCCESS
    assert NotificationType.infer_from_message("Redid last action.") == NotificationType.SUCCESS
    assert NotificationType.infer_from_message("Created 2 wafer plot(s)") == NotificationType.SUCCESS
    assert NotificationType.infer_from_message("Cosmic ray erasure validated and saved.") == NotificationType.SUCCESS

    # Warning patterns
    assert NotificationType.infer_from_message("No fit model copied.") == NotificationType.WARNING
    assert NotificationType.infer_from_message("No peaks to copy.") == NotificationType.WARNING
    assert NotificationType.infer_from_message("No baseline copied.") == NotificationType.WARNING
    assert NotificationType.infer_from_message("Nothing to paste") == NotificationType.WARNING
    assert NotificationType.infer_from_message("Please enter a column name.") == NotificationType.WARNING
    assert NotificationType.infer_from_message("No spectrum selected.") == NotificationType.WARNING
    assert NotificationType.infer_from_message("No map selected.") == NotificationType.WARNING
    assert NotificationType.infer_from_message("Skipping x-logscale: data contains zeros.") == NotificationType.WARNING
    assert NotificationType.infer_from_message("Fit already in progress. Please wait...") == NotificationType.WARNING
    assert NotificationType.infer_from_message("Map data not found.") == NotificationType.WARNING
    assert NotificationType.infer_from_message("Map 'sample' already loaded, skipping.") == NotificationType.WARNING

    # Error patterns
    assert NotificationType.infer_from_message("Graph update rejected: syntax error") == NotificationType.ERROR
    assert NotificationType.infer_from_message("AI graph creation failed: timeout") == NotificationType.ERROR
    assert NotificationType.infer_from_message("Invalid fit model file format.") == NotificationType.ERROR
    assert NotificationType.infer_from_message("Failed to read fit model file.") == NotificationType.ERROR

    # General Info
    assert NotificationType.infer_from_message("Opening file: test.csv") == NotificationType.INFO
    assert NotificationType.infer_from_message("") == NotificationType.INFO


# ---------------------------------------------------------------------------
# VNotificationBadge Widget Tests
# ---------------------------------------------------------------------------
def test_notification_badge_creation_and_ui(qtbot):
    """Verify badge initialization, layout, labels, and icons."""
    badge = VNotificationBadge(
        message="Test notification message",
        notification_type=NotificationType.SUCCESS,
        duration=3000,
        title="Success Title",
        is_dark_theme=True,
    )
    qtbot.addWidget(badge)

    assert badge._message == "Test notification message"
    assert badge._title == "Success Title"
    assert badge._type == NotificationType.SUCCESS
    assert badge.height() == VNotificationBadge.BADGE_HEIGHT
    assert "Success Title: Test notification message" in badge.toolTip()


def test_notification_badge_doubled_width(qtbot):
    """Verify that MAX_TEXT_WIDTH is doubled to 560 to accommodate long messages."""
    assert VNotificationBadge.MAX_TEXT_WIDTH == 560
    long_msg = "Invalid column name in expression: name 'qsq_custom_param' is not defined in dataset"
    badge = VNotificationBadge(long_msg, notification_type=NotificationType.ERROR)
    qtbot.addWidget(badge)
    assert "Invalid column name in expression" in badge._text_label.text()


def test_notification_badge_theme_adaptation(qtbot):
    """Verify light/dark theme adaptation for badges."""
    badge = VNotificationBadge(
        message="Theme check",
        notification_type=NotificationType.WARNING,
        is_dark_theme=True,
    )
    qtbot.addWidget(badge)

    # Initial dark theme
    assert badge._is_dark_theme is True
    assert not badge._icon_label.pixmap().isNull()

    # Switch to light theme
    badge.apply_theme(is_dark=False)
    assert badge._is_dark_theme is False
    assert not badge._icon_label.pixmap().isNull()

    # Switch back to dark theme
    badge.apply_theme(is_dark=True)
    assert badge._is_dark_theme is True


def test_notification_badge_frame_colors(qtbot):
    """Verify distinct thin frame colors for warning (yellow), success (green), info (blue), and error (red)."""
    from PySide6.QtWidgets import QFrame

    # Warning -> yellow frame
    b_warn = VNotificationBadge("Warning msg", notification_type=NotificationType.WARNING, is_dark_theme=True)
    qtbot.addWidget(b_warn)
    assert isinstance(b_warn, QFrame)
    assert b_warn.frame_color == "#f59e0b"
    assert "border: 1px solid #f59e0b" in b_warn.styleSheet()

    # Success -> green frame
    b_succ = VNotificationBadge("Success msg", notification_type=NotificationType.SUCCESS, is_dark_theme=True)
    qtbot.addWidget(b_succ)
    assert isinstance(b_succ, QFrame)
    assert b_succ.frame_color == "#10b981"
    assert "border: 1px solid #10b981" in b_succ.styleSheet()

    # Info / Notification -> blue frame
    b_info = VNotificationBadge("Info msg", notification_type=NotificationType.INFO, is_dark_theme=True)
    qtbot.addWidget(b_info)
    assert isinstance(b_info, QFrame)
    assert b_info.frame_color == "#38bdf8"
    assert "border: 1px solid #38bdf8" in b_info.styleSheet()

    # Error -> red frame
    b_err = VNotificationBadge("Error msg", notification_type=NotificationType.ERROR, is_dark_theme=True)
    qtbot.addWidget(b_err)
    assert isinstance(b_err, QFrame)
    assert b_err.frame_color == "#ef4444"
    assert "border: 1px solid #ef4444" in b_err.styleSheet()

    # Light theme adaptation
    b_warn.apply_theme(is_dark=False)
    assert b_warn.frame_color == "#d97706"
    assert "border: 1px solid #d97706" in b_warn.styleSheet()

    b_succ.apply_theme(is_dark=False)
    assert b_succ.frame_color == "#059669"
    assert "border: 1px solid #059669" in b_succ.styleSheet()

    b_info.apply_theme(is_dark=False)
    assert b_info.frame_color == "#0284c7"
    assert "border: 1px solid #0284c7" in b_info.styleSheet()

    b_err.apply_theme(is_dark=False)
    assert b_err.frame_color == "#dc2626"
    assert "border: 1px solid #dc2626" in b_err.styleSheet()

    # Hover highlight
    b_warn.apply_theme(is_dark=True)
    enter_evt = QEvent(QEvent.Type.Enter)
    b_warn.enterEvent(enter_evt)
    assert b_warn.frame_color == "#fbbf24"
    assert "border: 1px solid #fbbf24" in b_warn.styleSheet()


def test_notification_badge_hover_pause_and_resume(qtbot):
    """Verify that mouse hover pauses the auto-dismiss timer and leaving resumes it."""
    badge = VNotificationBadge(
        message="Hover test",
        duration=4000,
    )
    qtbot.addWidget(badge)
    badge.start_entrance()

    assert badge._dismiss_timer.isActive()

    # Simulate enter event (hover)
    enter_evt = QEvent(QEvent.Type.Enter)
    badge.enterEvent(enter_evt)

    assert badge._is_hovered is True
    assert not badge._dismiss_timer.isActive()  # Timer paused

    # Simulate leave event
    leave_evt = QEvent(QEvent.Type.Leave)
    badge.leaveEvent(leave_evt)

    assert badge._is_hovered is False
    assert badge._dismiss_timer.isActive()  # Timer resumed


def test_notification_badge_dismiss_via_close_button(qtbot):
    """Verify that clicking the close button emits dismissed signal."""
    badge = VNotificationBadge(
        message="Close button test",
        duration=5000,
    )
    qtbot.addWidget(badge)

    dismissed_signals = []
    badge.dismissed.connect(dismissed_signals.append)

    # Click close button
    badge._close_btn.click()

    assert len(dismissed_signals) == 1
    assert dismissed_signals[0] is badge


def test_notification_badge_dismiss_via_badge_click(qtbot):
    """Verify that clicking directly on the badge pill dismisses it."""
    badge = VNotificationBadge(
        message="Badge click test",
        duration=5000,
    )
    qtbot.addWidget(badge)

    dismissed_signals = []
    badge.dismissed.connect(dismissed_signals.append)

    qtbot.mouseClick(badge, Qt.MouseButton.LeftButton)

    assert len(dismissed_signals) == 1
    assert dismissed_signals[0] is badge


# ---------------------------------------------------------------------------
# VNotificationArea Container Tests
# ---------------------------------------------------------------------------
def test_notification_area_single_notification(qtbot):
    """Verify displaying a single notification in the notification area."""
    area = VNotificationArea()
    area.resize(800, 36)
    qtbot.addWidget(area)

    badge = area.show_notification(
        message="Single notification",
        notification_type=NotificationType.INFO,
        duration=3000,
    )

    assert len(area._badges) == 1
    assert area._badges[0] is badge
    # Verify badge is placed near the right edge
    assert badge.x() + badge.width() <= area.width()


def test_notification_area_multiple_notifications_slide(qtbot):
    """Verify that multiple notifications are positioned in a horizontal row from right to left."""
    area = VNotificationArea()
    area.resize(1000, 36)
    qtbot.addWidget(area)

    b1 = area.show_notification("First notification", NotificationType.INFO)
    b2 = area.show_notification("Second notification", NotificationType.SUCCESS)

    assert len(area._badges) == 2
    assert area._badges[0] is b1
    assert area._badges[1] is b2

    # b2 (newer) must be to the right of b1 (older)
    assert b1.x() < b2.x()
    # b2 must be near the right edge
    assert b2.x() + b2.width() <= area.width()

    # Add a third notification
    b3 = area.show_notification("Third notification", NotificationType.WARNING)
    assert len(area._badges) == 3
    assert b1.x() < b2.x() < b3.x()
    assert b3.x() + b3.width() <= area.width()


def test_notification_area_max_visible_pruning(qtbot):
    """Verify that adding more than MAX_VISIBLE_BADGES dismisses the oldest."""
    area = VNotificationArea()
    area.resize(1200, 36)
    qtbot.addWidget(area)

    area.MAX_VISIBLE_BADGES = 3

    b1 = area.show_notification("Msg 1")
    b2 = area.show_notification("Msg 2")
    b3 = area.show_notification("Msg 3")
    assert len(area._badges) == 3

    # Add 4th notification -> b1 should be retired
    b4 = area.show_notification("Msg 4")
    assert len(area._badges) == 3
    assert b1 not in area._badges
    assert area._badges == [b2, b3, b4]


def test_notification_area_clear_all(qtbot):
    """Verify clearing all notifications."""
    area = VNotificationArea()
    area.resize(800, 36)
    qtbot.addWidget(area)

    area.show_notification("Msg 1")
    area.show_notification("Msg 2")
    assert len(area._badges) == 2

    area.clear_all()
    assert len(area._badges) == 0


def test_notification_area_theme_propagation(qtbot):
    """Verify set_theme propagates to all active badges."""
    area = VNotificationArea()
    area.resize(800, 36)
    qtbot.addWidget(area)

    b1 = area.show_notification("Msg 1")
    b2 = area.show_notification("Msg 2")

    area.set_theme("light")
    assert area._is_dark_theme is False
    assert b1._is_dark_theme is False
    assert b2._is_dark_theme is False

    area.set_theme("dark")
    assert area._is_dark_theme is True
    assert b1._is_dark_theme is True
    assert b2._is_dark_theme is True


def test_notification_area_resize_relayout(qtbot):
    """Verify badges reposition correctly when notification area is resized."""
    area = VNotificationArea()
    area.resize(800, 36)
    qtbot.addWidget(area)

    b1 = area.show_notification("Msg 1")
    b2 = area.show_notification("Msg 2")

    right_before = b2.x() + b2.width()

    # Resize window wider and trigger resizeEvent
    area.resize(1000, 36)
    area.resizeEvent(QResizeEvent(QSize(1000, 36), QSize(800, 36)))
    right_after = b2.x() + b2.width()

    # New right edge should shift to accommodate the wider container
    assert right_after > right_before
    assert b1.x() < b2.x()


# ---------------------------------------------------------------------------
# NotificationManager Service Tests
# ---------------------------------------------------------------------------
def test_notification_manager_singleton():
    """Verify NotificationManager is a singleton."""
    mgr1 = NotificationManager.instance()
    mgr2 = NotificationManager.instance()
    assert mgr1 is mgr2


def test_notification_manager_area_registration(qtbot):
    """Verify registering and weak referencing VNotificationArea."""
    mgr = NotificationManager.instance()
    area = VNotificationArea()
    qtbot.addWidget(area)

    mgr.set_area(area)
    assert mgr.get_area() is area

    mgr.clear_history()
    badge = mgr.show("Test manager message", type="success")
    assert badge is not None
    assert len(area._badges) == 1
    assert area._badges[0] is badge

    history = mgr.get_history()
    assert len(history) == 1
    assert history[0]["message"] == "Test manager message"
    assert history[0]["type"] == "success"

    # Clean up
    mgr.set_area(None)
    assert mgr.get_area() is None


def test_notification_manager_helper_methods(qtbot):
    """Verify success, info, warning, error helper methods on NotificationManager."""
    mgr = NotificationManager.instance()
    area = VNotificationArea()
    area.resize(800, 36)
    qtbot.addWidget(area)
    mgr.set_area(area)

    mgr.clear_history()
    mgr.success("Success msg")
    mgr.info("Info msg")
    mgr.warning("Warning msg")
    mgr.error("Error msg")

    history = mgr.get_history()
    assert len(history) == 4
    types = [h["type"] for h in history]
    assert types == ["success", "info", "warning", "error"]

    mgr.set_area(None)


def test_show_toast_notification_backward_compatibility(qtbot):
    """Verify backward-compatible show_toast_notification function routes to NotificationManager."""
    mgr = NotificationManager.instance()
    area = VNotificationArea()
    qtbot.addWidget(area)
    mgr.set_area(area)

    mgr.clear_history()

    # Call legacy signature
    show_toast_notification(
        parent=None,
        message="Legacy toast message",
        title="Legacy Title",
        duration=3000,
        preset=ToastPreset.SUCCESS,
    )

    history = mgr.get_history()
    assert len(history) == 1
    assert history[0]["message"] == "Legacy toast message"
    assert history[0]["title"] == "Legacy Title"
    assert history[0]["type"] == "success"

    mgr.set_area(None)


def test_show_notification_free_function(qtbot):
    """Verify top-level show_notification free function."""
    mgr = NotificationManager.instance()
    area = VNotificationArea()
    qtbot.addWidget(area)
    mgr.set_area(area)

    mgr.clear_history()
    show_notification("Free function test", notification_type="info")

    assert any(h["message"] == "Free function test" for h in mgr.get_history())

    mgr.set_area(None)
