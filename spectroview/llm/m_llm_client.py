"""
spectroview/llm/m_llm_client.py
--------------------------------
Model layer: Ollama connection manager and background QThread worker.

This module is completely self-contained and has a single external
dependency: the optional `ollama` Python package.  If that package
is absent every public helper returns a safe sentinel value so the
rest of the application can disable the AI Chat feature gracefully.

Typical usage
-------------
>>> from spectroview.llm.m_llm_client import LLMClient
>>> client = LLMClient()
>>> if client.is_available():
...     client.chat("Explain my data", df, history, on_chunk, on_done, on_error)
"""

from __future__ import annotations

import json
from typing import Callable, List, Dict, Optional

from PySide6.QtCore import QThread, Signal, QObject

# ---------------------------------------------------------------------------
# Optional import guard — the rest of the app works fine if ollama is absent
# ---------------------------------------------------------------------------
try:
    import ollama as _ollama  # type: ignore
    OLLAMA_AVAILABLE = True
except ImportError:
    _ollama = None          # type: ignore[assignment]
    OLLAMA_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# Background worker — runs inside a QThread so the UI stays responsive
# ═══════════════════════════════════════════════════════════════════════════

class LLMWorker(QThread):
    """Sends a chat request to Ollama and streams the response back.

    Signals
    -------
    chunk_received(str)
        Emitted for every streamed token fragment so the UI can show
        a live typing effect.
    response_ready(str)
        Emitted once with the full assembled response when streaming ends.
    error_occurred(str)
        Emitted if Ollama is unreachable or returns an error.
    """

    chunk_received  = Signal(str)
    response_ready  = Signal(str)
    error_occurred  = Signal(str)

    def __init__(
        self,
        model: str,
        messages: List[Dict[str, str]],
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._model    = model
        self._messages = messages
        self._full_response = ""

    # ------------------------------------------------------------------
    def run(self) -> None:                      # executed in worker thread
        if not OLLAMA_AVAILABLE:
            self.error_occurred.emit(
                "The `ollama` Python package is not installed.\n"
                "Run:  pip install ollama"
            )
            return

        try:
            stream = _ollama.chat(
                model=self._model,
                messages=self._messages,
                stream=True,
            )
            for chunk in stream:
                fragment = chunk["message"]["content"]
                self._full_response += fragment
                self.chunk_received.emit(fragment)

            self.response_ready.emit(self._full_response)

        except Exception as exc:               # noqa: BLE001
            self.error_occurred.emit(str(exc))


# ═══════════════════════════════════════════════════════════════════════════
# Client — thin facade used by VMChat
# ═══════════════════════════════════════════════════════════════════════════

class LLMClient:
    """Manages the connection to a local Ollama service.

    This is a pure-Python class (no Qt inheritance) so it can be
    instantiated cheaply without any Qt objects being created.
    """

    DEFAULT_MODEL = "qwen2.5-coder:7b"

    def __init__(self) -> None:
        self._worker: Optional[LLMWorker] = None

    # ------------------------------------------------------------------
    # Availability helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """Return *True* if both the `ollama` package and the local
        Ollama daemon are reachable."""
        if not OLLAMA_AVAILABLE:
            return False
        try:
            # list() makes a lightweight HTTP call to /api/tags
            _ollama.list()
            return True
        except Exception:           # noqa: BLE001
            return False

    @staticmethod
    def get_models() -> List[str]:
        """Return names of locally downloaded models, or ``[]``."""
        if not OLLAMA_AVAILABLE:
            return []
        try:
            response = _ollama.list()
            # ollama ≥ 0.4 returns an object with a .models attribute
            models = getattr(response, "models", None) or response.get("models", [])
            names = []
            for m in models:
                name = getattr(m, "model", None) or m.get("model") or m.get("name", "")
                if name:
                    names.append(name)
            return sorted(names)
        except Exception:           # noqa: BLE001
            return []

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        on_chunk: Callable[[str], None],
        on_done:  Callable[[str], None],
        on_error: Callable[[str], None],
        parent:   Optional[QObject] = None,
    ) -> None:
        """Spawn a background worker to call Ollama.

        Parameters
        ----------
        model:
            Name of the local Ollama model, e.g. ``"gemma3:4b"``.
        messages:
            Full conversation history in ``[{"role": ..., "content": ...}]``
            format, including the system prompt.
        on_chunk:
            Slot called with each streamed token fragment.
        on_done:
            Slot called once with the complete assembled response.
        on_error:
            Slot called if an error occurs.
        parent:
            Optional Qt parent object for the worker thread.
        """
        # Cancel any previous request that is still running
        self.cancel()

        self._worker = LLMWorker(model, messages, parent)
        self._worker.chunk_received.connect(on_chunk)
        self._worker.response_ready.connect(on_done)
        self._worker.error_occurred.connect(on_error)
        self._worker.start()

    def cancel(self) -> None:
        """Abort the current worker thread if one is running."""
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(2000)
        self._worker = None

    def is_busy(self) -> bool:
        """Return *True* while a worker thread is running."""
        return bool(self._worker and self._worker.isRunning())
