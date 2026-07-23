"""
spectroview/ai_agent/vm_chat.py
---------------------------
ViewModel layer for the AI Chat feature.

Responsibilities
----------------
* Build a rich system prompt by delegating to ``PromptManager`` which
  loads modular Markdown files from ``ai_agent/prompts/``, ``rules/``,
  ``knowledge/``, and ``examples/`` — no large prompt strings are
  hard-coded here.
* Manage the multi-turn conversation history.
* Delegate the actual LLM call to ``LLMClient`` (Ollama or cloud API).
* Run the tool-calling loop against the in-process MCP server and emit a
  strongly-typed :class:`ChatResult` so the View never inspects raw JSON.

Pandas expressions requested by the model go through
``utils.safe_eval.evaluate_pandas_expression``, which prefers
``DataFrame.query()`` and only falls back to a namespace-restricted
``eval()`` for expressions ``.query()`` cannot represent (aggregations,
``groupby``). Builtins are stripped in that fallback.

MVVM contract
-------------
The View calls public methods; the ViewModel responds exclusively through
signals.  No Qt widgets are imported here.

Prompt engineering
------------------
All prompts, rules, knowledge, and examples are defined in Markdown files
under ``ai_agent/prompts/``, ``ai_agent/rules/``, ``ai_agent/knowledge/``,
and ``ai_agent/examples/``.  Modify those files to change AI behaviour
without touching this Python file.
"""

from __future__ import annotations

import json
import re
from typing import Optional, List, Dict, Any

import pandas as pd
from PySide6.QtCore import QObject, Signal

from spectroview.ai_agent.m_llm_client import LLMClient, get_ollama_model_info
from spectroview.ai_agent.m_conversation import MConversation
from spectroview.ai_agent.m_conversation_store import MConversationStore
from spectroview.model.m_settings import MSettings
from spectroview.model.m_plot_recipe_store import MPlotRecipeStore
from spectroview.ai_agent.m_prompt_manager import PromptManager
from spectroview.ai_agent.agent.commands import (
    AgentCommand, CreatePlot, DeletePlots, UpdatePlot,
)
from spectroview.ai_agent.agent.ports import RecordingContext
from spectroview.ai_agent.utils.plot_utils import expand_all_plot_configs, normalize_plot_config
from spectroview.ai_agent.utils.df_summary import compact_dataframe_schema
from spectroview.ai_agent.mcp.hub import MCPHub


def _parse_param_size_to_billions(size_str: str) -> Optional[float]:
    """Parse an Ollama ``parameter_size`` string (e.g. ``"8.2B"``, ``"600M"``)
    into a billions-of-parameters float. Returns ``None`` if unparseable."""
    match = re.match(r"^\s*([\d.]+)\s*([BMK])\s*$", str(size_str), re.IGNORECASE)
    if not match:
        return None
    value, unit = float(match.group(1)), match.group(2).upper()
    return value * {"B": 1.0, "M": 1e-3, "K": 1e-6}[unit]


# ═══════════════════════════════════════════════════════════════════════════
# Public result dataclasses (plain dicts would work too, but explicit is better)
# ═══════════════════════════════════════════════════════════════════════════

class ChatResult:
    """Carries the outcome of one completed agent turn back to the View."""
    __slots__ = ("action", "explanation", "text_summary", "plot_config")

    def __init__(
        self,
        action: str                 = "answer",
        explanation: str            = "",
        text_summary: str           = "",
        plot_config: Optional[list] = None,
    ) -> None:
        self.action       = action          # "plot" | "answer"
        self.explanation  = explanation     # human-readable explanation shown in the UI
        self.text_summary = text_summary    # the model's final prose answer
        self.plot_config  = plot_config     # list of graph create/update/delete configs


class VMChatContext(RecordingContext):
    """Exposes a live :class:`VMChat` to the MCP tools as an ``AppContext``.

    Reads are delegated to the ViewModel so the tools always see the current
    DataFrames and graphs; submitted commands are recorded (the inherited
    behaviour) until :meth:`VMChat._emit_final_result` drains them.
    """

    def __init__(self, vm: "VMChat") -> None:
        super().__init__()
        self._vm = vm

    def list_dataframes(self) -> List[str]:
        return list(self._vm._dfs)

    def active_dataframe_name(self) -> str:
        return self._vm._active_df_name

    def get_dataframe(self, name: str = "") -> Optional[pd.DataFrame]:
        return self._vm._dfs.get(name or self._vm._active_df_name)

    def list_graphs(self) -> Dict[int, Dict[str, Any]]:
        return self._vm._graphs


# ═══════════════════════════════════════════════════════════════════════════
# ViewModel
# ═══════════════════════════════════════════════════════════════════════════

class VMChat(QObject):
    """Manages one chat session linked to a pandas DataFrame.

    Signals
    -------
    thinking_changed(bool)
        ``True`` while the LLM is processing, ``False`` when done.
    chunk_received(str)
        Streaming token fragment — connect to a typing-animation label.
    result_ready(object)
        A ``ChatResult`` instance with the parsed action result.
    error_occurred(str)
        Human-readable error message.
    """

    thinking_changed = Signal(bool, str)
    chunk_received   = Signal(str)
    result_ready     = Signal(object)       # ChatResult
    error_occurred   = Signal(str)
    conversation_changed = Signal(str)
    tool_execution_received = Signal(str, str) # name, result text

    #: Hard stop on the tool-calling loop, so a model that keeps requesting
    #: tools instead of answering cannot spin forever.
    MAX_AGENT_TURNS = 5

    # Ollama model-name substrings that trigger small-model mode when the
    # parameter-count check (via `ollama show`) is unavailable/inconclusive.
    _SMALL_MODEL_PATTERNS: List[str] = [
        "qwen2.5-coder:7b", "qwen2.5-coder:3b", "qwen2.5-coder:1.5b",
        "qwen2.5:7b", "qwen2.5:3b", "qwen2.5:1.5b",
        "qwen3:8b", "qwen3:4b", "qwen3:1.7b", "qwen3:0.6b",
        "llama3.2:3b", "llama3.2:1b", "llama3.1:8b",
        "gemma3:4b", "gemma3:1b", "gemma2:9b",
        "phi3:3.8b", "phi3:mini", "phi3.5",
        "deepseek-r1:7b", "deepseek-r1:8b", "deepseek-r1:1.5b",
        "mistral:7b", "codellama:7b", "codellama:13b",
    ]

    # -----------------------------------------------------------------------

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)

        self._client   = LLMClient()
        self._model    = LLMClient.DEFAULT_MODEL
        self._provider: str = LLMClient.DEFAULT_PROVIDER

        # Manage multiple dataframes
        self._dfs: Dict[str, pd.DataFrame] = {}
        self._active_df_name: str = ""

        # Track open graphs so the AI knows their IDs
        self._graphs: Dict[int, Dict[str, Any]] = {}   # {graph_id: {"style": ..., "x": ..., "y": ...}}

        # What the MCP tools see of this application, and where they queue the
        # commands drained at the end of each agent turn.
        self._context = VMChatContext(self)
        self._pending_response: str = ""
        self._loop_count: int = 0

        self._settings = MSettings()
        self._history_folder = self._settings.get_ai_value("history_folder", "", str)
        self._recipe_folder = self._settings.get_plot_recipe_folder()

        self.conversation_store = MConversationStore(self._history_folder)
        self._conversation = self.conversation_store.create_conversation()
        self.recipe_store = MPlotRecipeStore(self._recipe_folder)
        self.max_context_messages: Optional[int] = None # None means no cap

        # ── Prompt engineering ───────────────────────────────────────────
        # PromptManager loads Markdown files from the ai_agent/ subdirectories.
        # To change AI behaviour, edit the .md files — not this file.
        self._prompt_mgr = PromptManager()

        # ── Small-model mode ──────────────────────────────────────────────
        # None = auto-detect; True/False = manual override (set_small_model_mode).
        self._small_model_override: Optional[bool] = None
        self._is_small_model: bool = False
        self._param_count_cache: Dict[str, Optional[bool]] = {}
        # Not refreshed here — refresh happens on set_model()/set_provider(),
        # once a real model/provider is known, to avoid a network call
        # (ollama show) racing construction against a placeholder model.

        # ── MCP ──────────────────────────────────────────────────────────
        # One hub, one background event loop, sessions held open for every
        # server in config/servers.yaml (SPECTROview's own plus any external
        # ones the user enabled). It connects lazily on first tool use, so
        # opening the panel costs nothing until the user actually chats.
        self._hub = MCPHub(self._context)

    @property
    def _mcp_tools_cache(self) -> List[Dict[str, Any]]:
        """Tool schemas offered to the model, across every connected server."""
        return self._hub.list_tools()

    def shutdown(self) -> None:
        """Close the MCP sessions and stop the hub thread."""
        self._client.cancel()
        self._hub.stop()



    # ------------------------------------------------------------------
    # Public API — called by VChatPanel
    # ------------------------------------------------------------------

    def refresh_recipe_store(self) -> None:
        """Rebuild self.recipe_store from the *current* Working Folder
        setting -- same fix as VWorkspaceGraphs._refresh_recipe_and_style_stores()
        for the same bug (a store built once in __init__ never saw a
        Working Folder configured afterward). Call before every save, same
        as the Graphs workspace does."""
        self._recipe_folder = self._settings.get_plot_recipe_folder()
        self.recipe_store = MPlotRecipeStore(self._recipe_folder)

    def set_dataframes(self, dfs: Dict[str, pd.DataFrame], active_name: str = "") -> None:
        """Update the available DataFrames.
        
        The chat history is explicitly preserved across workspace file loads
        and active dataframe changes. The user must explicitly request a new chat.
        """
        self._dfs = dfs.copy() if dfs else {}
        self._active_df_name = active_name

    def update_active_df_name(self, name: str) -> None:
        """Update only the active dataframe name — preserves chat history."""
        self._active_df_name = name

    def set_graphs(self, graphs: Dict[int, Any]) -> None:
        """Update the known open graphs (for inclusion in system prompt)."""
        self._graphs = {
            gid: {
                "style":   getattr(g, 'plot_style', ''),
                "x":       getattr(g, 'x', ''),
                "y":       getattr(g, 'y', []),
                "z":       getattr(g, 'z', ''),
                "df":      getattr(g, 'df_name', ''),
                "filters": getattr(g, 'filters', []),
            }
            for gid, g in graphs.items()
        }

    def set_model(self, model: str) -> None:
        self._model = model
        self._refresh_small_model_mode()

    def set_provider(
        self,
        provider: str,
        api_key:  str = "",
        base_url: str = "",
        model:    str = "",
    ) -> None:
        """Switch the active LLM backend and clear conversation history."""
        self._provider = provider
        self._client.set_provider(provider, api_key=api_key,
                                  base_url=base_url, model=model)
        # Update model to provider default when switching
        if provider != "Ollama" and not model:
            from spectroview.ai_agent.m_llm_client import API_PROVIDERS
            self._model = API_PROVIDERS.get(provider, {}).get("default_model", self._model)
        self._refresh_small_model_mode()
        self.new_conversation()

    def get_provider(self) -> str:
        """Return the currently active provider name."""
        return self._provider

    # ------------------------------------------------------------------
    # Small-model mode — auto-detection with manual override
    # ------------------------------------------------------------------

    def set_small_model_mode(self, enabled: Optional[bool]) -> None:
        """Force simplified-prompt mode on/off, or resume auto-detection.

        Parameters
        ----------
        enabled:
            ``True`` to force the simplified small-model prompt tier,
            ``False`` to force the full prompt tier, or ``None`` to resume
            auto-detection based on the active model.
        """
        self._small_model_override = enabled
        self._refresh_small_model_mode()

    def is_small_model_mode(self) -> bool:
        """Return ``True`` if the simplified small-model prompt tier is active."""
        return self._is_small_model

    def _refresh_small_model_mode(self) -> None:
        is_small = (
            self._small_model_override
            if self._small_model_override is not None
            else self._auto_detect_small_model()
        )
        self._is_small_model = is_small
        cfg = self._prompt_mgr.model_config
        self.max_context_messages = (
            cfg.get("max_context_messages_small", 6) if is_small
            else cfg.get("max_context_messages")
        )

    def _auto_detect_small_model(self) -> bool:
        """Best-effort small-model detection. Ollama-only; defaults to
        ``False`` (full tier) whenever detection is inconclusive, since
        misclassifying an unknown *large* model as small is more harmful
        than the reverse."""
        if self._provider != "Ollama":
            return False

        by_params = self._detect_small_by_param_count(self._model)
        if by_params is not None:
            return by_params

        key = self._model.lower().replace(" ", "")
        return any(p.replace(" ", "") in key for p in self._SMALL_MODEL_PATTERNS)

    def _detect_small_by_param_count(self, model: str) -> Optional[bool]:
        """Return True/False from the model's reported parameter count via
        ``ollama show``, or None if undeterminable. Cached per model name."""
        if model in self._param_count_cache:
            return self._param_count_cache[model]

        result: Optional[bool] = None
        info = get_ollama_model_info(model)
        if info is not None:
            size_str = getattr(getattr(info, "details", None), "parameter_size", None) or ""
            billions = _parse_param_size_to_billions(size_str)
            if billions is not None:
                threshold = self._prompt_mgr.model_config.get("small_model_param_threshold_b", 10.0)
                result = billions < threshold

        self._param_count_cache[model] = result
        return result

    def _build_request_options(self) -> Dict[str, Any]:
        """Assemble the generic request-tuning dict passed to ``LLMClient.chat()``.

        ``num_ctx``/``think`` only ever affect the Ollama worker (see
        ``LLMClient.chat()``); ``timeout``/``max_tokens`` apply to whichever
        backend is active. Small-model mode swaps in the smaller-budget
        values so local models get a correctly-sized context window instead
        of Ollama's silent default.
        """
        cfg = self._prompt_mgr.model_config
        opts: Dict[str, Any] = {}
        if cfg.get("request_timeout_seconds") is not None:
            opts["timeout"] = float(cfg["request_timeout_seconds"])

        if self._is_small_model:
            opts["num_ctx"] = cfg.get("ollama_num_ctx_small", 8192)
            opts["max_tokens"] = cfg.get("max_tokens_small", 4096)
            opts["think"] = cfg.get("ollama_think", False)
        else:
            opts["num_ctx"] = cfg.get("ollama_num_ctx_full", 16384)
            opts["max_tokens"] = cfg.get("max_tokens", 81920)

        return opts

    def process_query(self, user_text: str, reply_to_index: Optional[int] = None) -> None:
        """Send *user_text* to the active LLM backend and emit results when done.

        Safe to call from the main thread; all blocking I/O runs in a
        ``QThread`` via ``LLMClient``.
        """
        if not user_text.strip():
            return

        if not self._dfs:
            self.error_occurred.emit(
                "No DataFrames loaded. Please load data in the Graphs workspace first."
            )
            return

        if not self._client.is_available():
            if self._provider == "Ollama":
                self.error_occurred.emit(
                    "Ollama is not reachable.\n\n"
                    "Make sure Ollama is running:\n"
                    "  • macOS / Linux: ollama serve\n"
                    "  • Then pull a model: ollama pull gemma3:4b"
                )
            else:
                self.error_occurred.emit(
                    f"{self._provider} API is not configured.\n\n"
                    "Please enter your API key in the provider settings above."
                )
            return

        # Track the user turn now; assistant turn added after response
        self._conversation.add_message("user", user_text, reply_to_index=reply_to_index)
        self._pending_response = ""
        self._loop_count = 0

        # Build the full message list for this request
        messages = self._build_messages()

        self.thinking_changed.emit(True, "Thinking")

        self._client.chat(
            model    = self._model,
            messages = messages,
            on_chunk = self._on_chunk,
            on_done  = self._on_done,
            on_error = self._on_error,
            tools    = self._mcp_tools_cache,
            parent   = self,
            request_options = self._build_request_options(),
            on_thinking_chunk = self._on_thinking_chunk,
        )

    def cancel(self) -> None:
        """Abort any in-progress LLM request."""
        self._client.cancel()
        self.thinking_changed.emit(False, "Thinking")
        self._save_history_to_file()

    def clear_history(self) -> None:
        self.new_conversation()

    def load_conversation(self, conv: MConversation) -> None:
        self._save_history_to_file()
        self._conversation = conv
        self.conversation_changed.emit(conv.title)
        
    def new_conversation(self) -> None:
        self._save_history_to_file()
        self._conversation = self.conversation_store.create_conversation()
        self.conversation_changed.emit(self._conversation.title)

    # -- Conversation accessors (the View must not reach into _conversation) --

    @property
    def message_count(self) -> int:
        """Number of messages recorded in the active conversation."""
        return self._conversation.message_count

    @property
    def conversation_title(self) -> str:
        return self._conversation.title

    def visible_messages(self) -> List[tuple[int, Dict[str, Any]]]:
        """``(index, message)`` pairs the UI should render, tool traffic excluded.

        The index is into the *full* history, not the filtered list, because
        ``reply_to_index`` refers to full-history positions.
        """
        return [(i, m) for i, m in enumerate(self._conversation.messages)
                if not m.get("is_hidden")]

    def get_message_content(self, index: int) -> Optional[str]:
        """Content of message *index*, or ``None`` if out of range."""
        messages = self._conversation.messages
        if 0 <= index < len(messages):
            return messages[index].get("content", "")
        return None

    def rename_conversation(self, title: str) -> None:
        """Rename the active conversation and persist it."""
        self._conversation.rename(title)
        self._save_history_to_file()

    def is_busy(self) -> bool:
        return self._client.is_busy()

    def is_available(self) -> bool:
        return self._client.is_available()

    def get_models(self) -> List[str]:
        return self._client.get_models()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    
    def _save_history_to_file(self) -> None:
        """Save the current conversation history to the configured folder."""
        # Only save if there's actual conversation history
        if self._conversation.message_count > 0:
            self._conversation.save(self._history_folder)

    def _build_messages(self) -> List[Dict[str, str]]:
        """Assemble the full message list = system prompt + conversation history."""
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        messages += self._conversation.to_llm_messages(self.max_context_messages)
        return messages

    def _build_system_prompt(self) -> str:
        """Assemble the system prompt from modular Markdown files + dynamic DataFrame/graph context.

        The static portions (identity, tool guidance, rules, knowledge) are
        loaded from the ``ai_agent/`` subdirectories by :class:`PromptManager`.
        Only the dynamic sections — DataFrame schemas and open-graph summaries —
        are computed here at request time and substituted into the placeholders.

        To change AI behaviour, edit the Markdown files in::

            spectroview/ai_agent/prompts/
            spectroview/ai_agent/rules/
            spectroview/ai_agent/knowledge/
            spectroview/ai_agent/examples/
        """
        # ── 1. Build dynamic context from live DataFrames and open graphs ─
        dfs_info_parts: list[str] = [
            f"DATAFRAME: {name!r} ({len(df)} rows, {len(df.columns)} columns)\n"
            f"  Columns: {compact_dataframe_schema(df)}"
            for name, df in self._dfs.items()
        ]

        dataframes_section = "\n\n".join(dfs_info_parts) if dfs_info_parts else "No DataFrames are currently loaded."
        active_df_info = (
            f"\nThe currently active DataFrame in the UI is: {self._active_df_name!r}.\n"
            if self._active_df_name else ""
        )

        # Open-graphs summary
        if self._graphs:
            graph_lines: list[str] = []
            for gid, info in sorted(self._graphs.items()):
                y_str = info['y'][0] if isinstance(info['y'], list) and info['y'] else info['y']
                z_str = f", z={info['z']!r}" if info['z'] else ""
                filt_str = f", filters={info['filters']!r}" if info.get('filters') else ""
                graph_lines.append(
                    f"  Graph ID {gid}: style={info['style']!r}, "
                    f"x={info['x']!r}, y={y_str!r}{z_str}{filt_str}, "
                    f"df={info['df']!r}"
                )
            graphs_info = "CURRENTLY OPEN GRAPHS:\n" + "\n".join(graph_lines) + "\n"
        else:
            graphs_info = "No graphs are currently open.\n"

        # ── 2. Assemble static prompt from Markdown files ────────────
        if self._is_small_model:
            static_prompt = self._prompt_mgr.build_prompt(
                prompts=["system_small"],
                rules=["general"],
                examples=["examples_small"],
            )
        else:
            static_prompt = self._prompt_mgr.build_prompt(
                prompts=["system", "chat", "plotting"],
                rules=["general", "plotting", "spectroview"],
                knowledge=["features"],
                examples=["plotting_examples"],
            )

        # ── 3. Inject dynamic context into the static prompt ──────────────
        return (
            static_prompt
            .replace("{dataframes_section}", dataframes_section)
            .replace("{active_df_info}", active_df_info)
            .replace("{graphs_info}", graphs_info)
        )

    # ------------------------------------------------------------------
    # Worker callbacks (called from the worker QThread via Qt signals)
    # ------------------------------------------------------------------

    def _on_chunk(self, fragment: str) -> None:
        self._pending_response += fragment
        self.chunk_received.emit(fragment)

    def _on_thinking_chunk(self, fragment: str) -> None:
        """Discard a model's reasoning ("thinking") fragment — the app
        never surfaces it in the UI. Still passed to LLMClient.chat() as
        on_thinking_chunk so the worker's channel separation keeps any
        model's `message["thinking"]` content from leaking into the
        visible-answer stream, even for a model that emits it unrequested."""
        pass

    def _emit_final_result(self, full_text: str) -> None:
        """End the turn: drain any queued graph commands and emit a ChatResult.
        """
        self.thinking_changed.emit(False, "Thinking")
        commands = self._context.drain()

        if commands:
            result = ChatResult(
                action="plot",
                explanation=full_text or "Executing graph commands...",
                plot_config=self._commands_to_configs(commands),
                text_summary=full_text,
            )
        else:
            result = ChatResult(action="answer", text_summary=full_text)
        self.result_ready.emit(result)

    @staticmethod
    def _commands_to_configs(commands: List[AgentCommand]) -> List[Dict[str, Any]]:
        """Render queued commands as the config dicts the Graphs workspace takes.
        """
        configs: List[Dict[str, Any]] = []
        for cmd in commands:
            if isinstance(cmd, CreatePlot):
                # Also expands multi-style configs ("plot_style": "box, scatter")
                configs.extend(expand_all_plot_configs([cmd.config]))
            elif isinstance(cmd, UpdatePlot):
                configs.append({"_graph_update": {
                    "graph_id": cmd.graph_id,
                    "properties": normalize_plot_config(dict(cmd.properties)),
                }})
            elif isinstance(cmd, DeletePlots):
                configs.append({"_graph_delete": {
                    "delete_all": cmd.delete_all,
                    "graph_ids": cmd.graph_ids,
                }})
        return configs

    @staticmethod
    def _parse_tool_arguments(raw: Any) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Coerce a tool call's arguments to a dict.

        Returns ``(arguments, None)`` or ``(None, reason)``. Providers send
        either a JSON string (OpenAI, Ollama) or an already-decoded dict
        (Anthropic), and a weak model sometimes sends malformed JSON.
        """
        if isinstance(raw, dict):
            return raw, None
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw) if raw.strip() else {}
            except ValueError as exc:
                return None, str(exc)
            if not isinstance(parsed, dict):
                return None, f"arguments parsed to a {type(parsed).__name__}, not a JSON object"
            return parsed, None
        return None, f"arguments were a {type(raw).__name__}, not a JSON object"

    def _run_tool_call(self, tool_call: Dict[str, Any]) -> tuple[Any, str, str]:
        """Execute one tool call; return ``(call_id, name, result_text)``.

        Never raises — a failure is described in the result text so the model
        can correct itself while it still has turns left.
        """
        function = tool_call.get("function", {})
        name = function.get("name", "")
        arguments, error = self._parse_tool_arguments(function.get("arguments", {}))

        if error is not None:
            text = (
                f"Error: the arguments for '{name}' were not valid JSON ({error}). "
                f"Raw arguments received: {function.get('arguments')!r}. "
                f"Please retry this tool call with a single, well-formed JSON object."
            )
        else:
            text = self._hub.call_tool(name, arguments)

        return tool_call.get("id"), name, text

    def _on_done(self, full_text: str, tool_calls: list) -> None:
        # Record assistant turn in history
        if full_text or tool_calls:
            self._conversation.add_message(
                "assistant", 
                full_text or "", 
                tool_calls=tool_calls if tool_calls else None
            )
            self._save_history_to_file()

        if tool_calls:
            self._loop_count += 1
            if self._loop_count > self.MAX_AGENT_TURNS:
                self.error_occurred.emit(
                    f"Agent loop exceeded maximum turns ({self.MAX_AGENT_TURNS})."
                )
                self._emit_final_result(full_text)
                return

            self.thinking_changed.emit(True, "Executing tools...")
            try:
                results = [self._run_tool_call(tc) for tc in tool_calls]

                # Append tool results to conversation
                for tc_id, name, res_text in results:
                    self._conversation.add_message(
                        "tool", 
                        res_text, 
                        is_hidden=True, 
                        tool_call_id=tc_id
                    )
                    self.tool_execution_received.emit(name, res_text)
                self._save_history_to_file()
                
                # Trigger the next turn
                self._pending_response = ""
                messages = self._build_messages()
                self._client.chat(
                    model=self._model,
                    messages=messages,
                    on_chunk=self._on_chunk,
                    on_done=self._on_done,
                    on_error=self._on_error,
                    tools=self._mcp_tools_cache,
                    parent=self,
                    request_options=self._build_request_options(),
                    on_thinking_chunk=self._on_thinking_chunk,
                )
            except Exception as e:
                self.error_occurred.emit(f"Error calling MCP tools: {e}")
                self._emit_final_result(full_text)
            return

        self._emit_final_result(full_text)

    def _on_error(self, message: str) -> None:
        self.thinking_changed.emit(False, "Thinking")
        self.error_occurred.emit(message)


