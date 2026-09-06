"""Synchronous command dispatch onto the Qt GUI thread.

MCP's HTTP handler runs on its own asyncio/uvicorn thread.  Every access to a
QObject-owned ViewModel or widget must therefore cross this boundary first.
The application API remains synchronous for callers; a queued Qt signal does
the actual hand-off and a small event object carries the result back.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from PySide6.QtCore import QCoreApplication, QObject, QThread, Qt, Signal, Slot

from spectroview.application.errors import ApplicationAPIError


class Dispatcher(Protocol):
    """Minimal dispatcher surface used by :class:`SpectroviewApplicationAPI`."""

    def call(self, callback: Callable[[], Any]) -> Any:
        ...


class DirectDispatcher:
    """Qt-free dispatcher for tests and already-thread-safe controllers."""

    def call(self, callback: Callable[[], Any]) -> Any:
        return callback()


@dataclass
class _Invocation:
    callback: Callable[[], Any]
    finished: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[BaseException] = None


class QtMainThreadDispatcher(QObject):
    """Run callbacks on the thread that owns this QObject.

    Construct this object on the QApplication thread. Calls originating on
    that same thread execute directly; calls from the embedded MCP server are
    delivered with ``Qt.QueuedConnection`` and block only the server worker.
    """

    _invoke = Signal(object)

    def __init__(self, parent: Optional[QObject] = None, timeout: float = 30.0) -> None:
        super().__init__(parent)
        self._timeout = float(timeout)
        self._accepting = True
        self._invoke.connect(self._execute, Qt.QueuedConnection)

    def call(self, callback: Callable[[], Any]) -> Any:
        if not self._accepting:
            raise ApplicationAPIError(
                "APPLICATION_NOT_READY", "SPECTROview is shutting down."
            )

        app = QCoreApplication.instance()
        if app is None:
            raise ApplicationAPIError(
                "APPLICATION_NOT_READY", "The Qt application event loop is not available."
            )

        if QThread.currentThread() == self.thread():
            return callback()

        invocation = _Invocation(callback)
        self._invoke.emit(invocation)
        if not invocation.finished.wait(self._timeout):
            raise ApplicationAPIError(
                "MAIN_THREAD_TIMEOUT",
                "The SPECTROview GUI thread did not answer the request in time.",
                {"timeout_seconds": self._timeout},
            )
        if invocation.error is not None:
            raise invocation.error
        return invocation.result

    @Slot(object)
    def _execute(self, invocation: _Invocation) -> None:
        try:
            invocation.result = invocation.callback()
        except BaseException as exc:  # propagated to the waiting server thread
            invocation.error = exc
        finally:
            invocation.finished.set()

    def close(self) -> None:
        """Reject new work during application shutdown."""
        self._accepting = False
