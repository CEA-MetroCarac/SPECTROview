"""
spectroview/llm/m_llm_client.py
--------------------------------
Model layer: LLM connection manager and background QThread workers.

Supports two backends:
  1. **Ollama** — local models via the `ollama` Python package.
  2. **Cloud API** — any OpenAI-compatible REST endpoint (Google Gemini,
     DeepSeek, OpenAI, or a custom URL) via the `openai` Python package.

Both backends are completely optional; if their respective packages are
absent, the rest of the application runs unaffected.

Typical usage
-------------
>>> from spectroview.llm.m_llm_client import LLMClient
>>> client = LLMClient()
>>> client.set_provider("Gemini", api_key="...", model="gemini-2.5-flash")
>>> if client.is_available():
...     client.chat(model, messages, on_chunk, on_done, on_error)
"""

from __future__ import annotations

from typing import Callable, List, Dict, Optional

from PySide6.QtCore import QThread, Signal, QObject

# ---------------------------------------------------------------------------
# Optional import guards — the rest of the app works fine if either is absent
# ---------------------------------------------------------------------------
try:
    import ollama as _ollama  # type: ignore
    OLLAMA_AVAILABLE = True
except ImportError:
    _ollama = None          # type: ignore[assignment]
    OLLAMA_AVAILABLE = False

try:
    import openai as _openai  # type: ignore
    OPENAI_AVAILABLE = True
except ImportError:
    _openai = None          # type: ignore[assignment]
    OPENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Known cloud API providers — all use OpenAI-compatible endpoints
# ---------------------------------------------------------------------------

API_PROVIDERS: Dict[str, Dict[str, str]] = {
    "Gemini": {
        "base_url":      "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.5-flash",
    },
    "DeepSeek": {
        "base_url":      "https://api.deepseek.com",
        "default_model": "deepseek-chat",
    },
    "OpenAI": {
        "base_url":      "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
    "Custom": {
        "base_url":      "",
        "default_model": "",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Worker 1 — Ollama local backend
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
# Worker 2 — Cloud / OpenAI-compatible API backend
# ═══════════════════════════════════════════════════════════════════════════

class APIWorker(QThread):
    """Sends a chat request to an OpenAI-compatible cloud API and streams
    the response back.

    Works with Google Gemini, DeepSeek, OpenAI, and any compatible endpoint.

    Signals
    -------
    chunk_received(str)
        Emitted for every streamed token fragment.
    response_ready(str)
        Emitted once with the full assembled response.
    error_occurred(str)
        Emitted on network or auth errors.
    """

    chunk_received  = Signal(str)
    response_ready  = Signal(str)
    error_occurred  = Signal(str)

    def __init__(
        self,
        api_key:  str,
        base_url: str,
        model:    str,
        messages: List[Dict[str, str]],
        parent:   Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._api_key  = api_key
        self._base_url = base_url
        self._model    = model
        self._messages = messages
        self._full_response = ""

    # ------------------------------------------------------------------
    def run(self) -> None:                      # executed in worker thread
        if not OPENAI_AVAILABLE:
            self.error_occurred.emit(
                "The `openai` Python package is not installed.\n"
                "Run:  pip install openai"
            )
            return

        if not self._api_key:
            self.error_occurred.emit(
                "No API key set. Please enter your API key in the chat panel."
            )
            return

        try:
            client = _openai.OpenAI(
                api_key=self._api_key,
                base_url=self._base_url or None,
            )
            stream = client.chat.completions.create(
                model=self._model,
                messages=self._messages,  # type: ignore[arg-type]
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                fragment = getattr(delta, "content", None) or ""
                if fragment:
                    self._full_response += fragment
                    self.chunk_received.emit(fragment)

            self.response_ready.emit(self._full_response)

        except Exception as exc:               # noqa: BLE001
            self.error_occurred.emit(str(exc))


# ═══════════════════════════════════════════════════════════════════════════
# Client — unified facade used by VMChat
# ═══════════════════════════════════════════════════════════════════════════

class LLMClient:
    """Manages LLM connectivity for both local Ollama and cloud API providers.

    This is a pure-Python class (no Qt inheritance) so it can be
    instantiated cheaply without any Qt objects being created.

    Provider selection
    ------------------
    * ``"Ollama"``  — local Ollama daemon (default, backward-compatible)
    * ``"Gemini"``  — Google Gemini via OpenAI-compatible API
    * ``"DeepSeek"``— DeepSeek via OpenAI-compatible API
    * ``"OpenAI"``  — OpenAI API
    * ``"Custom"``  — user-specified base URL
    """

    DEFAULT_MODEL   = "qwen2.5-coder:7b"
    DEFAULT_PROVIDER = "Ollama"

    def __init__(self) -> None:
        self._worker: Optional[QThread] = None          # LLMWorker or APIWorker

        # Provider state
        self._provider:  str = self.DEFAULT_PROVIDER
        self._api_key:   str = ""
        self._base_url:  str = ""
        self._api_model: str = ""

    # ------------------------------------------------------------------
    # Provider configuration
    # ------------------------------------------------------------------

    def set_provider(
        self,
        provider: str,
        api_key:  str = "",
        base_url: str = "",
        model:    str = "",
    ) -> None:
        """Switch the active LLM backend.

        Parameters
        ----------
        provider:
            One of ``"Ollama"``, ``"Gemini"``, ``"DeepSeek"``, ``"OpenAI"``,
            or ``"Custom"``.
        api_key:
            API key for cloud providers (ignored for Ollama).
        base_url:
            Override the base URL. If empty, the preset for the chosen
            provider is used.
        model:
            Override the model name. If empty, the provider preset default
            is used.
        """
        self._provider = provider
        self._api_key  = api_key

        if provider == "Ollama":
            self._base_url  = ""
            self._api_model = ""
        else:
            preset = API_PROVIDERS.get(provider, {})
            self._base_url  = base_url or preset.get("base_url", "")
            self._api_model = model or preset.get("default_model", "")

    def get_provider(self) -> str:
        """Return the active provider name."""
        return self._provider

    def get_api_key(self) -> str:
        return self._api_key

    def get_base_url(self) -> str:
        return self._base_url

    # ------------------------------------------------------------------
    # Availability helpers
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return ``True`` if the active provider is reachable / configured."""
        if self._provider == "Ollama":
            return self._is_ollama_available()
        else:
            # Cloud API: available if openai package present and key provided
            return OPENAI_AVAILABLE and bool(self._api_key)

    @staticmethod
    def _is_ollama_available() -> bool:
        if not OLLAMA_AVAILABLE:
            return False
        try:
            _ollama.list()
            return True
        except Exception:           # noqa: BLE001
            return False

    def get_models(self) -> List[str]:
        """Return a list of available model names for the active provider."""
        if self._provider == "Ollama":
            return self._get_ollama_models()
        else:
            return self._get_api_models()

    @staticmethod
    def _get_ollama_models() -> List[str]:
        if not OLLAMA_AVAILABLE:
            return []
        try:
            response = _ollama.list()
            models = getattr(response, "models", None) or response.get("models", [])
            names = []
            for m in models:
                name = getattr(m, "model", None) or m.get("model") or m.get("name", "")
                if name:
                    names.append(name)
            return sorted(names)
        except Exception:           # noqa: BLE001
            return []

    def _get_api_models(self) -> List[str]:
        """Fetch model list from the cloud API (best-effort)."""
        if not OPENAI_AVAILABLE or not self._api_key:
            preset = API_PROVIDERS.get(self._provider, {})
            default = preset.get("default_model", "")
            return [default] if default else []
        try:
            client = _openai.OpenAI(
                api_key=self._api_key,
                base_url=self._base_url or None,
            )
            models_resp = client.models.list()
            names = sorted(m.id for m in models_resp.data)
            return names if names else self._fallback_models()
        except Exception:           # noqa: BLE001
            return self._fallback_models()

    def _fallback_models(self) -> List[str]:
        """Return provider preset default when API model listing fails."""
        preset = API_PROVIDERS.get(self._provider, {})
        default = preset.get("default_model", "")
        return [default] if default else []

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        model:    str,
        messages: List[Dict[str, str]],
        on_chunk: Callable[[str], None],
        on_done:  Callable[[str], None],
        on_error: Callable[[str], None],
        parent:   Optional[QObject] = None,
    ) -> None:
        """Spawn a background worker to call the active LLM backend.

        Parameters
        ----------
        model:
            Model name.  For Ollama this is the local model tag; for cloud
            providers this is the API model identifier.
        messages:
            Full conversation history in ``[{"role": ..., "content": ...}]``
            format, including the system prompt.
        on_chunk, on_done, on_error:
            Callbacks connected to the worker's signals.
        parent:
            Optional Qt parent object for the worker thread.
        """
        self.cancel()

        if self._provider == "Ollama":
            self._worker = LLMWorker(model, messages, parent)
        else:
            # Use stored base_url & api_key; model argument takes precedence
            # over the stored api_model to allow per-request overrides.
            self._worker = APIWorker(
                api_key=self._api_key,
                base_url=self._base_url,
                model=model,
                messages=messages,
                parent=parent,
            )

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
        """Return ``True`` while a worker thread is running."""
        return bool(self._worker and self._worker.isRunning())
