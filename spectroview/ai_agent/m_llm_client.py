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

import json
import os
from typing import Callable, List, Dict, Optional, Any

from PySide6.QtCore import QThread, Signal, QObject

# ---------------------------------------------------------------------------
# Corporate / internal CA support
# ---------------------------------------------------------------------------
_CA_BUNDLE_ENV = (
    os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE") or ""
)

if not _CA_BUNDLE_ENV:
    try:
        import truststore as _truststore
        _truststore.inject_into_ssl()
    except Exception:           # noqa: BLE001 - optional; certifi stays the default
        pass


def _make_http_client() -> Optional[Any]:
    """Return an ``httpx.Client`` verifying against the env-configured CA
    bundle, or ``None`` to use the SDK's default verification.

    Non-``None`` only when ``SSL_CERT_FILE``/``REQUESTS_CA_BUNDLE`` points at an
    existing file — the escape hatch for a private CA that isn't in the OS
    trust store (so ``truststore`` alone wouldn't help), such as a frozen build
    or a non-Windows host.
    """
    if not _CA_BUNDLE_ENV or not os.path.isfile(_CA_BUNDLE_ENV):
        return None
    try:
        import httpx
        return httpx.Client(verify=_CA_BUNDLE_ENV)
    except Exception:           # noqa: BLE001
        return None


#: Provider-side HTTP statuses that mean "not your fault, try again".
_TRANSIENT_STATUS: Dict[int, str] = {
    429: "⏳ Rate limited — too many requests for this model right now.",
    500: "⚠ The provider hit an internal error.",
    502: "⚠ The provider's gateway returned a bad response.",
    503: "⏳ The model is temporarily unavailable (high demand).",
    504: "⏳ The provider timed out.",
}


def _format_exception(exc: BaseException) -> str:
    """Build an error string that includes the underlying cause chain.

    The openai/anthropic SDKs collapse low-level TLS/socket failures into a
    terse ``"Connection error."`` whose ``str()`` hides the real reason (bad
    URL, DNS failure, SSL/proxy issue). Walking ``__cause__``/``__context__``
    surfaces it (e.g. ``CERTIFICATE_VERIFY_FAILED``) so the message is
    actionable instead of a dead end.
    """
    messages: List[str] = []
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        text = str(cur).strip()
        entry = f"{type(cur).__name__}: {text}" if text else type(cur).__name__
        if entry not in messages:
            messages.append(entry)
        cur = cur.__cause__ or cur.__context__

    detail = "\n  Caused by: ".join(messages)

    # A transient provider-side status buried under an SDK traceback reads as
    # "SPECTROview is broken". Say plainly whose problem it is.
    status = getattr(exc, "status_code", None)
    if status in _TRANSIENT_STATUS:
        return (
            f"{_TRANSIENT_STATUS[status]}\n"
            f"This is on the provider's side, not SPECTROview — retry in a moment, "
            f"or switch model/provider.\n\n{detail}"
        )
    return detail


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

def get_ollama_model_info(model: str) -> Optional[Any]:
    """Best-effort ``ollama.show(model)``. Returns ``None`` on any failure
    or if the ``ollama`` package is unavailable. Never raises."""
    if not OLLAMA_AVAILABLE:
        return None
    try:
        return _ollama.show(model)
    except Exception:           # noqa: BLE001
        return None


API_PROVIDERS: Dict[str, Dict[str, str]] = {
    "Gemini": {
        "base_url":      "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.5-flash",
    },
    "DeepSeek": {
        "base_url":      "https://api.deepseek.com",
        "default_model": "deepseek-chat",
    },
    "Mistral": {
        "base_url":      "https://api.mistral.ai/v1",
        "default_model": "mistral-large-latest",
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
    thinking_chunk_received(str)
        Emitted for every streamed *reasoning* token fragment, only when
        ``think`` was requested. Deliberately a separate signal from
        ``chunk_received`` — see the note in ``run()``.
    response_ready(str)
        Emitted once with the full assembled response when streaming ends.
    error_occurred(str)
        Emitted if Ollama is unreachable or returns an error.
    """

    chunk_received  = Signal(str)
    thinking_chunk_received = Signal(str)
    response_ready  = Signal(str, list)
    error_occurred  = Signal(str)

    def __init__(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        parent: Optional[QObject] = None,
        options: Optional[Dict[str, Any]] = None,
        think: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(parent)
        self._model    = model
        self._messages = messages
        self._tools    = tools
        self._options  = options
        self._think    = think
        self._timeout  = timeout
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
            if self._options:
                kwargs["options"] = self._options
            if self._think is not None:
                kwargs["think"] = self._think

            client = _ollama.Client(timeout=self._timeout) if self._timeout else _ollama
            stream = client.chat(
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
                        tool_calls.append(tc.model_dump() if hasattr(tc, "model_dump") else tc)

                thinking_fragment = message.get("thinking") or ""
                if thinking_fragment:
                    self.thinking_chunk_received.emit(thinking_fragment)

                fragment = message.get("content") or ""
                if fragment:
                    self._full_response += fragment
                    self.chunk_received.emit(fragment)

            self.response_ready.emit(self._full_response, tool_calls)

        except Exception as exc:               # noqa: BLE001
            self.error_occurred.emit(_format_exception(exc))


# ═══════════════════════════════════════════════════════════════════════════
# Worker 2 — Cloud / OpenAI-compatible API backend
# ═══════════════════════════════════════════════════════════════════════════

class APIWorker(QThread):
    """Sends a chat request to an OpenAI-compatible cloud API and streams
    the response back.

    Works with Google Gemini, DeepSeek, Mistral, OpenAI, and any compatible endpoint.

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
        timeout:  Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        super().__init__(parent)
        self._api_key  = api_key
        self._base_url = base_url
        self._model    = model
        self._messages = messages
        self._tools    = tools
        self._timeout  = timeout
        self._max_tokens = max_tokens
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
                timeout=self._timeout,
                http_client=_make_http_client(),
            )
            kwargs = {}
            if self._tools:
                kwargs["tools"] = self._tools
            if self._max_tokens:
                kwargs["max_tokens"] = self._max_tokens

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
            self.error_occurred.emit(_format_exception(exc))


# ---------------------------------------------------------------------------
# Anthropic translation — its tool protocol differs from OpenAI's
# ---------------------------------------------------------------------------
#
# The rest of the app speaks OpenAI's dialect: tools are
# {"type": "function", "function": {...}}, a tool call is an assistant message
# with `tool_calls`, and its result is a separate {"role": "tool"} message.
# Anthropic instead takes {"name", "description", "input_schema"} tools, and
# carries calls/results as `tool_use`/`tool_result` content blocks. Passing the
# OpenAI shapes straight through makes the API reject the request, so both are
# translated here — as pure functions, so the mapping is testable without a key.

def anthropic_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Convert OpenAI-style tool schemas to Anthropic's ``input_schema`` form."""
    if not tools:
        return None
    converted: List[Dict[str, Any]] = []
    for tool in tools:
        fn = tool.get("function", tool)
        converted.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", "") or "",
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return converted


def _tool_use_input(raw_arguments: Any) -> Dict[str, Any]:
    """Coerce a recorded tool call's arguments back into a dict."""
    if isinstance(raw_arguments, dict):
        return raw_arguments
    try:
        parsed = json.loads(raw_arguments or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def anthropic_messages(messages: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    """Split *messages* into ``(system_prompt, anthropic_messages)``.

    System messages are concatenated out into the top-level ``system``
    parameter. Assistant messages carrying ``tool_calls`` become ``tool_use``
    content blocks, and the ``{"role": "tool"}`` results answering them are
    merged into a single following user message of ``tool_result`` blocks —
    Anthropic requires every result for one assistant turn in one user message.
    """
    system_parts: List[str] = []
    converted: List[Dict[str, Any]] = []

    for msg in messages:
        role, content = msg.get("role"), msg.get("content") or ""

        if role == "system":
            system_parts.append(content)

        elif role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id") or "",
                "content": content,
            }
            # Append to the previous user message when it is already a
            # tool_result batch, otherwise start a new one.
            if (converted and converted[-1]["role"] == "user"
                    and isinstance(converted[-1]["content"], list)
                    and converted[-1]["content"][0].get("type") == "tool_result"):
                converted[-1]["content"].append(block)
            else:
                converted.append({"role": "user", "content": [block]})

        elif role == "assistant" and msg.get("tool_calls"):
            blocks: List[Dict[str, Any]] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for call in msg["tool_calls"]:
                fn = call.get("function", {})
                blocks.append({
                    "type": "tool_use",
                    "id": call.get("id") or "",
                    "name": fn.get("name", ""),
                    "input": _tool_use_input(fn.get("arguments")),
                })
            converted.append({"role": "assistant", "content": blocks})

        elif content:
            # Anthropic rejects empty-content messages outright.
            converted.append({"role": role, "content": content})

    return "\n".join(system_parts), converted


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
        timeout:  Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        super().__init__(parent)
        self._api_key  = api_key
        self._base_url = base_url
        self._model    = model
        self._messages = messages
        self._tools    = tools
        self._timeout  = timeout
        self._max_tokens = max_tokens
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

        system_prompt, messages = anthropic_messages(self._messages)
        max_tokens = self._max_tokens or 4096

        try:
            client_args = {"api_key": self._api_key, "timeout": self._timeout}
            if self._base_url:
                client_args["base_url"] = self._base_url
            http_client = _make_http_client()
            if http_client is not None:
                client_args["http_client"] = http_client

            client = _anthropic.Anthropic(**client_args)

            kwargs: Dict[str, Any] = {
                "model": self._model,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            tools = anthropic_tools(self._tools)
            if tools:
                kwargs["tools"] = tools

            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    if self._is_cancelled:
                        break
                    self._full_response += text
                    self.chunk_received.emit(text)
                final = stream.get_final_message()

            # Tool calls are only complete once the stream is; read them off
            # the assembled message and re-emit in the OpenAI shape the rest
            # of the app consumes.
            tool_calls = [
                {
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": block.input},
                }
                for block in final.content if block.type == "tool_use"
            ]
            self.response_ready.emit(self._full_response, tool_calls)

        except Exception as exc:               # noqa: BLE001
            self.error_occurred.emit(_format_exception(exc))


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
    * ``"Mistral"`` — Mistral AI via OpenAI-compatible API
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
            One of ``"Ollama"``, ``"Gemini"``, ``"DeepSeek"``, ``"Mistral"``, ``"OpenAI"``,
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
                http_client=_make_http_client(),
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
        request_options: Optional[Dict[str, Any]] = None,
        on_thinking_chunk: Optional[Callable[[str], None]] = None,
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
        request_options:
            Optional dict of generic request tuning values (``num_ctx``,
            ``think``, ``timeout``, ``max_tokens``), translated to the
            right shape per provider below. ``num_ctx``/``think`` are
            Ollama-``options``-only and are never forwarded to the
            OpenAI-compatible or Anthropic workers — structurally
            impossible to affect a cloud provider's behavior.
        on_thinking_chunk:
            Optional callback for streamed reasoning content. Only ever
            fires for the Ollama worker when ``think`` was requested;
            silently unused otherwise (no other worker has this signal).
        """
        self.cancel()
        opts = request_options or {}

        if self._provider == "Ollama":
            ollama_options: Dict[str, Any] = {}
            if "num_ctx" in opts:
                ollama_options["num_ctx"] = opts["num_ctx"]
            if "max_tokens" in opts:
                ollama_options["num_predict"] = opts["max_tokens"]
            self._worker = LLMWorker(
                model, messages, tools, parent,
                options=ollama_options or None,
                think=opts.get("think"),
                timeout=opts.get("timeout"),
            )
        elif self._provider == "Anthropic":
            self._worker = AnthropicWorker(
                api_key=self._api_key,
                base_url=self._base_url,
                model=model,
                messages=messages,
                tools=tools,
                parent=parent,
                timeout=opts.get("timeout"),
                max_tokens=opts.get("max_tokens"),
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
                timeout=opts.get("timeout"),
                max_tokens=opts.get("max_tokens"),
            )

        self._worker.chunk_received.connect(on_chunk)
        self._worker.response_ready.connect(on_done)
        self._worker.error_occurred.connect(on_error)
        if on_thinking_chunk is not None and hasattr(self._worker, "thinking_chunk_received"):
            self._worker.thinking_chunk_received.connect(on_thinking_chunk)
        self._worker.start()

    def cancel(self) -> None:
        """Abort the current worker thread if one is running."""
        worker = self._worker
        self._worker = None
        if worker is None:
            return

        if hasattr(worker, "cancel"):
            worker.cancel()
        for signal_name in ("chunk_received", "thinking_chunk_received",
                            "response_ready", "error_occurred"):
            signal = getattr(worker, signal_name, None)
            if signal:
                try:
                    signal.disconnect()
                except RuntimeError:
                    pass

        if worker.isRunning():
            worker.finished.connect(worker.deleteLater)
        else:
            worker.deleteLater()

    def is_busy(self) -> bool:
        """Return ``True`` while a worker thread is running."""
        return bool(self._worker and self._worker.isRunning())
