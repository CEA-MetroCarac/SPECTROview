from PySide6.QtCore import QSize
from spectroview.view.theme import Icons
from spectroview.view.components.v_menubar import VMenuBar, THEME_OPTIONS


def test_v_menubar_actions_and_icons(qtbot):
    """Verify that VMenuBar initializes with all actions, shortcuts, and valid icons."""
    menubar = VMenuBar()
    qtbot.addWidget(menubar)

    # Verify AI Chat action
    ai_action = menubar.actionAIChat
    assert ai_action is not None
    assert ai_action.text() == "SPECTROview AI Agent"
    assert ai_action.shortcut().toString() == "Ctrl+Shift+A"
    assert not ai_action.icon().isNull()

    # Pixmap rendering check
    pixmap = ai_action.icon().pixmap(QSize(30, 30))
    assert not pixmap.isNull()
    assert pixmap.width() == 30
    assert pixmap.height() == 30

    # Ensure vector icon exists and renders
    assert not Icons.get("llm_ai").isNull()

    # Ensure theme options contains the 4 canonical themes
    theme_keys = {key for key, _ in THEME_OPTIONS if key != "---"}
    assert theme_keys == {"dark", "light", "classic_dark", "classic_light"}
    assert len(theme_keys) == 4


def test_v_menubar_theme_icon_switching(qtbot):
    """Verify that theme button shows moon for dark themes and sun for light themes."""
    menubar = VMenuBar()
    qtbot.addWidget(menubar)

    # Initial default is dark -> moon icon
    assert menubar.actionTheme.icon().name() == "moon"

    # Switching to light -> sun icon
    menubar.set_current_theme("light")
    assert menubar.actionTheme.icon().name() == "sun"

    # Switching to classic_light -> sun icon
    menubar.set_current_theme("classic_light")
    assert menubar.actionTheme.icon().name() == "sun"

    # Switching to dark -> moon icon
    menubar.set_current_theme("dark")
    assert menubar.actionTheme.icon().name() == "moon"

    # Switching to classic_dark -> moon icon
    menubar.set_current_theme("classic_dark")
    assert menubar.actionTheme.icon().name() == "moon"

    # apply_theme propagates and maintains proper moon/sun icon
    menubar.apply_theme("light")
    assert menubar.actionTheme.icon().name() == "sun"

    menubar.apply_theme("dark")
    assert menubar.actionTheme.icon().name() == "moon"


def test_v_menubar_aichat_position(qtbot):
    """Verify that actionAIChat is positioned on the right side just before actionCheckUpdate."""
    menubar = VMenuBar()
    qtbot.addWidget(menubar)
    actions = menubar.actions()
    idx_ai = actions.index(menubar.actionAIChat)
    idx_update = actions.index(menubar.actionCheckUpdate)
    assert idx_ai == idx_update - 1

