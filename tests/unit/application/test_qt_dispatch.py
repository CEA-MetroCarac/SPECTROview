"""The external server must never execute QObject work on its own thread."""

import threading

from PySide6.QtCore import QThread

from spectroview.application.dispatch import QtMainThreadDispatcher


def test_worker_call_executes_on_dispatcher_qt_thread(qapp, qtbot):
    dispatcher = QtMainThreadDispatcher(timeout=2.0)
    observed = {}

    def callback():
        observed["thread"] = QThread.currentThread()
        return "ok"

    def worker():
        observed["result"] = dispatcher.call(callback)

    thread = threading.Thread(target=worker)
    thread.start()
    qtbot.waitUntil(lambda: not thread.is_alive(), timeout=3000)
    thread.join()

    assert observed["result"] == "ok"
    assert observed["thread"] == dispatcher.thread()
