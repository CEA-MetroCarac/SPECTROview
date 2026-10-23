import os
from PySide6.QtCore import QSize
from spectroview import ICON_DIR
from spectroview.view.components.v_menubar import VMenuBar


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

    # Ensure icon files exist on disk
    assert os.path.exists(os.path.join(ICON_DIR, "llm_ai.png"))
