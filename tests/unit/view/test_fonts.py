# tests/unit/view/test_fonts.py
"""Unit tests for cross-platform font configuration."""

from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabBar,
)
from spectroview.view.theme.fonts import get_application_font, setup_application_font
from spectroview.view.components.v_user_manual import VUserManualDialog
from spectroview import USER_MANUAL_DIR


def test_get_application_font_macos():
    """Verify that macOS resolves to .AppleSystemUIFont at 13pt (Apple HIG standard)."""
    font = get_application_font(platform="darwin")
    assert font.family() == ".AppleSystemUIFont"
    assert font.pointSize() == 13


def test_get_application_font_windows():
    """Verify that Windows resolves to Segoe UI at 10pt (standard 96-DPI Windows desktop font)."""
    font = get_application_font(platform="win32")
    assert font.family() == "Segoe UI"
    assert font.pointSize() == 10


def test_get_application_font_linux():
    """Verify that Linux resolves to a standard sans-serif font at >= 10pt."""
    font = get_application_font(platform="linux")
    assert font.pointSize() >= 10


def test_setup_application_font(qapp):
    """Verify that setup_application_font applies the font to QApplication."""
    applied_font = setup_application_font(qapp)
    assert qapp.font().pointSize() == applied_font.pointSize()
    assert qapp.font().family() == applied_font.family()


def test_widget_font_inheritance(qapp, qtbot):
    """Verify all major GUI controls inherit the application font consistently."""
    setup_application_font(qapp)

    tab_bar = QTabBar()
    line_edit = QLineEdit("123.456")
    combo_box = QComboBox()
    combo_box.addItems(["Item 1", "Item 2"])
    spin_box = QSpinBox()
    label = QLabel("Test Label")
    button = QPushButton("Click Me")

    for w in [tab_bar, line_edit, combo_box, spin_box, label, button]:
        qtbot.addWidget(w)
        # Point size must not be smaller than the application font (e.g. 10pt on Win, 13pt on Mac)
        assert w.font().pointSize() >= 10
        # Line height must be at least 14px for comfortable readability
        assert w.fontMetrics().height() >= 12


def test_user_manual_dialog_font_scaling(qapp, qtbot):
    """Verify VUserManualDialog initializes with platform-appropriate text size."""
    setup_application_font(qapp)
    expected_size = get_application_font().pointSize()

    dlg = VUserManualDialog(USER_MANUAL_DIR)
    qtbot.addWidget(dlg)

    assert dlg._text_size == expected_size
    assert dlg.section_list.font().pointSize() == expected_size
    assert dlg.content_browser.font().pointSize() == expected_size

    # Verify zoom operations adjust content text size correctly
    dlg._zoom_in()
    assert dlg._text_size == expected_size + 1
    assert dlg.content_browser.font().pointSize() == expected_size + 1

    dlg._zoom_out()
    assert dlg._text_size == expected_size
    assert dlg.content_browser.font().pointSize() == expected_size

    # Verify QTextBrowser frame and background match theme surface
    from spectroview.view.theme.qss_template import build_stylesheet
    from spectroview.view.theme.tokens import THEME_TOKENS
    dark_qss = build_stylesheet(THEME_TOKENS["dark"])
    light_qss = build_stylesheet(THEME_TOKENS["light"])
    assert "QTextBrowser" in dark_qss
    assert THEME_TOKENS["dark"].surface in dark_qss
    assert THEME_TOKENS["light"].surface in light_qss

