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
