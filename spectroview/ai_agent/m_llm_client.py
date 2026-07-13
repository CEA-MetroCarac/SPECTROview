"""
spectroview/ai_agent/m_llm_client.py
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
>>> from spectroview.ai_agent.m_llm_client import LLMClient
>>> client = LLMClient()
>>> client.set_provider("Gemini", api_key="...", model="gemini-2.5-flash")
>>> if client.is_available():
...     client.chat(model, messages, on_chunk, on_done, on_error)
"""

from __future__ import annotations

from typing import Callable, List, Dict, Optional, Any

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

try:
    import anthropic as _anthropic # type: ignore
    ANTHROPIC_AVAILABLE = True
except ImportError:
    _anthropic = None       # type: ignore[assignment]
    ANTHROPIC_AVAILABLE = False


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
    "Anthropic": {
        "base_url":      "https://api.anthropic.com",
        "default_model": "claude-3-5-sonnet-20241022",
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
    response_ready  = Signal(str, list)
    error_occurred  = Signal(str)

    def __init__(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._model    = model
        self._messages = messages
        self._tools    = tools
        self._full_response = ""
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    # ------------------------------------------------------------------
    def run(self) -> None:                      # executed in worker thread
        if not OLLAMA_AVAILABLE:
            self.error_occurred.emit(
                "The `ollama` Python package is not installed.\n"
                "Run:  pip install ollama"
            )
            return

        try:
            kwargs = {}
            if self._tools:
                kwargs["tools"] = self._tools
                
            stream = _ollama.chat(
                model=self._model,
                messages=self._messages,
                stream=True,
                **kwargs
            )
            
            tool_calls = []
            for chunk in stream:
                if self._is_cancelled:
                    break
                    
                message = chunk.get("message", {})
                
                # Handle tool calls
                if "tool_calls" in message:
                    for tc in message["tool_calls"]:
                        # Append or accumulate tool calls
                        tool_calls.append(tc)
                        
                fragment = message.get("content") or ""
                if fragment:
                    self._full_response += fragment
                    self.chunk_received.emit(fragment)

            self.response_ready.emit(self._full_response, tool_calls)

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
    response_ready  = Signal(str, list)
    error_occurred  = Signal(str)

    def __init__(
        self,
        api_key:  str,
        base_url: str,
        model:    str,
        messages: List[Dict[str, Any]],
        tools:    Optional[List[Dict[str, Any]]] = None,
        parent:   Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._api_key  = api_key
        self._base_url = base_url
        self._model    = model
        self._messages = messages
        self._tools    = tools
        self._full_response = ""
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

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
            kwargs = {}
            if self._tools:
                kwargs["tools"] = self._tools
                
            stream = client.chat.completions.create(
                model=self._model,
                messages=self._messages,  # type: ignore[arg-type]
                stream=True,
                **kwargs
            )
            
            tool_calls_dict = {}
            for chunk in stream:
                if self._is_cancelled:
                    break
                delta = chunk.choices[0].delta
                
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_dict:
                            tool_calls_dict[idx] = {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": ""}}
                        if tc.function.arguments:
                            tool_calls_dict[idx]["function"]["arguments"] += tc.function.arguments
                            
                fragment = getattr(delta, "content", None) or ""
                if fragment:
                    self._full_response += fragment
                    self.chunk_received.emit(fragment)

            tool_calls = []
            for tc in tool_calls_dict.values():
                tool_calls.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]
                    }
                })

            self.response_ready.emit(self._full_response, tool_calls)

        except Exception as exc:               # noqa: BLE001
            self.error_occurred.emit(str(exc))


# ═══════════════════════════════════════════════════════════════════════════
# Worker 3 — Anthropic backend
# ═══════════════════════════════════════════════════════════════════════════

class AnthropicWorker(QThread):
    """Sends a chat request to Anthropic API and streams the response back.

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
    response_ready  = Signal(str, list)
    error_occurred  = Signal(str)

    def __init__(
        self,
        api_key:  str,
        base_url: str,
        model:    str,
        messages: List[Dict[str, Any]],
        tools:    Optional[List[Dict[str, Any]]] = None,
        parent:   Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._api_key  = api_key
        self._base_url = base_url
        self._model    = model
        self._messages = messages
        self._tools    = tools
        self._full_response = ""
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    # ------------------------------------------------------------------
    def run(self) -> None:
        if not ANTHROPIC_AVAILABLE:
            self.error_occurred.emit(
                "The `anthropic` Python package is not installed.\n"
                "Run:  pip install anthropic"
            )
            return

        if not self._api_key:
            self.error_occurred.emit(
                "No API key set. Please enter your API key in the chat panel."
            )
            return

        # Separate system messages from user/assistant messages
        system_prompt = ""
        filtered_messages = []
        for msg in self._messages:
            if msg["role"] == "system":
                system_prompt += msg["content"] + "\n"
            else:
                filtered_messages.append(msg)

        try:
            client_args = {"api_key": self._api_key}
            if self._base_url:
                client_args["base_url"] = self._base_url

            client = _anthropic.Anthropic(**client_args)
            
            kwargs = {}
            if self._tools:
                kwargs["tools"] = self._tools
                
            # Note: with Anthropic API, streaming with tools can be complex.
            # For simplicity, if tools are provided, we just use messages.create.
            if self._tools:
                resp = client.messages.create(
                    max_tokens=4096,
                    system=system_prompt,
                    messages=filtered_messages, # type: ignore
                    model=self._model,
                    tools=self._tools
                )
                tool_calls = []
                for content_block in resp.content:
                    if content_block.type == 'text':
                        self._full_response += content_block.text
                        self.chunk_received.emit(content_block.text)
                    elif content_block.type == 'tool_use':
                        tool_calls.append({
                            "function": {
                                "name": content_block.name,
                                "arguments": content_block.input
                            }
                        })
                self.response_ready.emit(self._full_response, tool_calls)
            else:
                with client.messages.stream(
                    max_tokens=4096,
                    system=system_prompt,
                    messages=filtered_messages, # type: ignore
                    model=self._model,
                ) as stream:
                    for text in stream.text_stream:
                        if self._is_cancelled:
                            break
                        self._full_response += text
                        self.chunk_received.emit(text)
                self.response_ready.emit(self._full_response, [])

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
        elif self._provider == "Anthropic":
            return ANTHROPIC_AVAILABLE and bool(self._api_key)
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
        if self._provider == "Anthropic":
            # Anthropic doesn't have a standard model listing endpoint in its SDK yet
            return self._fallback_models()
            
        if not OPENAI_AVAILABLE or not self._api_key:
            return self._fallback_models()
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
        messages: List[Dict[str, Any]],
        on_chunk: Callable[[str], None],
        on_done:  Callable[[str, list], None],
        on_error: Callable[[str], None],
        tools:    Optional[List[Dict[str, Any]]] = None,
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
            self._worker = LLMWorker(model, messages, tools, parent)
        elif self._provider == "Anthropic":
            self._worker = AnthropicWorker(
                api_key=self._api_key,
                base_url=self._base_url,
                model=model,
                messages=messages,
                tools=tools,
                parent=parent,
            )
        else:
            # Use stored base_url & api_key; model argument takes precedence
            # over the stored api_model to allow per-request overrides.
            self._worker = APIWorker(
                api_key=self._api_key,
                base_url=self._base_url,
                model=model,
                messages=messages,
                tools=tools,
                parent=parent,
            )

        self._worker.chunk_received.connect(on_chunk)
        self._worker.response_ready.connect(on_done)
        self._worker.error_occurred.connect(on_error)
        self._worker.start()

    def cancel(self) -> None:
        """Abort the current worker thread if one is running."""
        if getattr(self, "_worker", None):
            if hasattr(self._worker, "cancel"):
                self._worker.cancel()
            
            # Safely orphan the thread by disconnecting callbacks so they don't affect UI
            for signal_name in ("chunk_received", "response_ready", "error_occurred"):
                signal = getattr(self._worker, signal_name, None)
                if signal:
                    try:
                        signal.disconnect()
                    except RuntimeError:
                        pass
            self._worker = None

    def is_busy(self) -> bool:
        """Return ``True`` while a worker thread is running."""
        return bool(self._worker and self._worker.isRunning())
