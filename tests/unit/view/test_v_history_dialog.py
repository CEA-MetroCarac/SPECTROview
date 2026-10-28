import pytest
from PySide6.QtCore import Qt
from spectroview.ai_agent.m_conversation_store import MConversationStore
from spectroview.ai_agent.v_history_dialog import VHistoryDialog, _ConversationItemWidget


def test_v_history_dialog_render_and_filter(tmp_path, qapp):
    store = MConversationStore(str(tmp_path))
    c1 = store.create_conversation()
    c1.rename("First conversation")
    c1.add_message("user", "Hello there")
    c1.save(str(tmp_path))

    c2 = store.create_conversation()
    c2.rename("Second conversation")
    c2.set_pinned(True)
    c2.add_message("user", "Pinned topic")
    c2.save(str(tmp_path))

    dlg = VHistoryDialog(store)
    dlg.show()

    # List should contain section headers ("PINNED", "TODAY") plus 2 items
    assert dlg.list_widget.count() >= 2
    assert dlg.btn_delete_all.isEnabled()

    # Verify item and header size hints to prevent vertical clipping regressions
    for i in range(dlg.list_widget.count()):
        item = dlg.list_widget.item(i)
        w = dlg.list_widget.itemWidget(item)
        if isinstance(w, _ConversationItemWidget):
            assert item.sizeHint().height() >= 48
        else:
            # Section header
            assert item.sizeHint().height() >= 28

    # Filter by search
    dlg.search_input.setText("Second")
    # Only Second should be displayed
    item_widgets = [
        dlg.list_widget.itemWidget(dlg.list_widget.item(i))
        for i in range(dlg.list_widget.count())
    ]
    conv_widgets = [w for w in item_widgets if isinstance(w, _ConversationItemWidget)]
    assert len(conv_widgets) == 1
    assert conv_widgets[0].title == "Second conversation"
    assert conv_widgets[0].pinned is True

    # Clear filter
    dlg.search_input.setText("")
    conv_widgets = [
        dlg.list_widget.itemWidget(dlg.list_widget.item(i))
        for i in range(dlg.list_widget.count())
        if isinstance(dlg.list_widget.itemWidget(dlg.list_widget.item(i)), _ConversationItemWidget)
    ]
    assert len(conv_widgets) == 2
    dlg.close()

