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
* Parse the LLM response and emit strongly-typed result signals so the
  View never has to inspect raw JSON.
* Execute *only* safe pandas operations (``df.query()`` and
  ``df.describe()``); never ``eval()`` or ``exec()``.

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
import traceback
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
from PySide6.QtCore import QObject, Signal, QSettings

from spectroview.ai_agent.m_llm_client import LLMClient, API_PROVIDERS
from spectroview.ai_agent.m_conversation import MConversation
from spectroview.ai_agent.m_conversation_store import MConversationStore
from spectroview.ai_agent.m_prompt_manager import PromptManager
from spectroview.ai_agent.utils.plot_utils import expand_all_plot_configs
import asyncio
from spectroview.ai_agent.mcp.server import create_mcp_server
from mcp.shared.memory import create_connected_server_and_client_session



# ═══════════════════════════════════════════════════════════════════════════
# Public result dataclasses (plain dicts would work too, but explicit is better)
# ═══════════════════════════════════════════════════════════════════════════

class ChatResult:
    """Carries the parsed LLM response back to the View."""
    __slots__ = ("action", "explanation", "dataframe", "text_summary",
                 "plot_config", "raw_response", "query", "target_dataframe")

    def __init__(
        self,
        action: str                         = "unknown",
        explanation: str                    = "",
        dataframe: Optional[pd.DataFrame]  = None,
        text_summary: str                   = "",
        plot_config: Optional[dict]         = None,
        raw_response: str                   = "",
        query: str                          = "",
        target_dataframe: str               = "",
    ) -> None:
        self.action       = action          # "filter" | "statistics" | "plot" | "answer" | "query"
        self.explanation  = explanation     # human-readable explanation shown in the UI
        self.dataframe    = dataframe       # filtered DataFrame (Tier 1)
        self.text_summary = text_summary    # statistics / answer text (Tier 2)
        self.plot_config  = plot_config     # {x, y, z, plot_style} suggestion (Tier 3)
        self.raw_response = raw_response    # full LLM text (for debugging)
        self.query        = query           # query string
        self.target_dataframe = target_dataframe # target dataframe name


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

    # Maximum number of message pairs kept in context (user + assistant each)
    MAX_HISTORY_PAIRS = 6

    # -----------------------------------------------------------------------

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)

        self._client   = LLMClient()
        self._model    = LLMClient.DEFAULT_MODEL

        # Manage multiple dataframes
        self._dfs: Dict[str, pd.DataFrame] = {}
        self._active_df_name: str = ""

        # Track open graphs so the AI knows their IDs
        self._graphs: Dict[int, Dict[str, Any]] = {}   # {graph_id: {"style": ..., "x": ..., "y": ...}}

        # Load history folder
        s = QSettings("SPECTROview", "AIChat")
        s.beginGroup("ai_chat")
        self._history_folder = str(s.value("history_folder", ""))
        s.endGroup()
        
        self.conversation_store = MConversationStore(self._history_folder)
        self._conversation = self.conversation_store.create_conversation()
        self.max_context_messages: Optional[int] = None # None means no cap

        # ── Prompt engineering ───────────────────────────────────────────
        # PromptManager loads Markdown files from the ai_agent/ subdirectories.
        # To change AI behaviour, edit the .md files — not this file.
        self._prompt_mgr = PromptManager()



        # ── MCP Server ───────────────────────────────────────────────────
        self._mcp_server = create_mcp_server(self)
        self._mcp_tools_cache = []
        self._fetch_mcp_tools()
        
    def _fetch_mcp_tools(self) -> None:
        async def fetch():
            async with create_connected_server_and_client_session(self._mcp_server._mcp_server) as session:
                await session.initialize()
                res = await session.list_tools()
                return res.tools
        
        try:
            tools = asyncio.run(fetch())
            for t in tools:
                self._mcp_tools_cache.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema
                    }
                })
        except Exception as e:
            print("Failed to initialize MCP tools:", e)



    # ------------------------------------------------------------------
    # Public API — called by VChatPanel
    # ------------------------------------------------------------------

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
        self.new_conversation()

    def get_provider(self) -> str:
        """Return the currently active provider name."""
        return self._provider

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
        messages = self._build_messages(user_text)

        self.thinking_changed.emit(True, "Thinking")

        self._client.chat(
            model    = self._model,
            messages = messages,
            on_chunk = self._on_chunk,
            on_done  = self._on_done,
            on_error = self._on_error,
            tools    = self._mcp_tools_cache,
            parent   = self,
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

    def _build_messages(self, user_text: str) -> List[Dict[str, str]]:
        """Assemble the full message list = system prompt + conversation history."""
        system_prompt = self._build_system_prompt(user_message=user_text)
        messages = [{"role": "system", "content": system_prompt}]

        messages += self._conversation.to_llm_messages(self.max_context_messages)

        return messages

    def _build_system_prompt(self, user_message: str = "") -> str:
        """Assemble the system prompt from modular Markdown files + dynamic DataFrame/graph context.

        The static portions (identity, JSON schema, rules, knowledge) are loaded
        from the ``ai_agent/`` subdirectories by :class:`PromptManager`.  Only the
        dynamic sections — DataFrame schemas and open-graph summaries — are
        computed here at request time and injected via ``str.format()``.

        To change AI behaviour, edit the Markdown files in::

            spectroview/ai_agent/prompts/
            spectroview/ai_agent/rules/
            spectroview/ai_agent/knowledge/
            spectroview/ai_agent/examples/

        Parameters
        ----------
        user_message:
            The latest user message (used for optional intent detection).
        """
        # ── 1. Build dynamic context from live DataFrames and open graphs ─
        dfs_info_parts: list[str] = []
        for name, df in self._dfs.items():
            col_info_lines: list[str] = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                sample_vals = df[col].dropna().unique()[:3].tolist()
                col_info_lines.append(
                    f"    - {col!r} ({dtype}): sample values {sample_vals}"
                )
            col_info = "\n".join(col_info_lines)
            try:
                preview = df.head(3).to_string(max_cols=8)
            except Exception:
                preview = "(preview unavailable)"
            dfs_info_parts.append(
                f"DATAFRAME: {name!r} ({len(df)} rows, {len(df.columns)} columns)\n"
                f"  Columns:\n{col_info}\n"
                f"  Preview:\n{preview}"
            )

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

        # ── 2. Assemble static prompt from Markdown files ─────────────────
        # Pass the user message so PromptManager can auto-detect intent
        # (plotting/fitting/coding) when enable_intent_detection is True.
        static_prompt = self._prompt_mgr.build_prompt(
            intent="chat",
            user_message=user_message,
            prompts=["system", "chat", "plotting"],
            rules=["general", "plotting", "spectroview"],
            knowledge=["features"],
            examples=["plotting_examples"],
        )

        # ── 3. Inject dynamic context into the static prompt ──────────────
        # The static prompt contains {dataframes_section}, {active_df_info},
        # and {graphs_info} placeholders defined in prompts/system.md.
        # We use plain str.replace() rather than str.format() so that JSON
        # examples in other .md files (which contain literal { } braces)
        # do not cause KeyError exceptions.
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

    def _on_done(self, full_text: str, tool_calls: list) -> None:
        self.thinking_changed.emit(False, "Thinking")

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
            if self._loop_count > 5:
                self.error_occurred.emit("Agent loop exceeded maximum turns (5).")
                return
                
            async def run_tools():
                results = []
                async with create_connected_server_and_client_session(self._mcp_server._mcp_server) as session:
                    await session.initialize()
                    for tc in tool_calls:
                        func_name = tc.get("function", {}).get("name")
                        func_args = tc.get("function", {}).get("arguments", {})
                        if isinstance(func_args, str):
                            try:
                                import json
                                func_args = json.loads(func_args)
                            except Exception:
                                func_args = {}
                        try:
                            res = await session.call_tool(func_name, func_args)
                            # Convert result to string
                            res_text = res.content[0].text if res.content and hasattr(res.content[0], 'text') else str(res)
                        except Exception as e:
                            res_text = f"Error executing tool {func_name}: {e}"
                        results.append((tc.get("id"), func_name, res_text))
                return results

            self.thinking_changed.emit(True, "Executing tools...")
            try:
                results = asyncio.run(run_tools())
                
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
                messages = self._build_messages("")
                self._client.chat(
                    model=self._model,
                    messages=messages,
                    on_chunk=self._on_chunk,
                    on_done=self._on_done,
                    on_error=self._on_error,
                    tools=self._mcp_tools_cache,
                    parent=self,
                )
            except Exception as e:
                self.error_occurred.emit(f"Error calling MCP tools: {e}")
            return
            
        # If no tool calls, output text answer.
        # Check if there are any pending plot configs from tool executions
        plot_configs = getattr(self, '_pending_plots', [])
        self._pending_plots = []
        
        if plot_configs:
            # We need to expand any multi-style plot configs (e.g. "plot_style": "box, scatter")
            expanded_configs = []
            for cfg in plot_configs:
                if "_graph_update" in cfg or "_graph_delete" in cfg:
                    expanded_configs.append(cfg)
                else:
                    expanded_configs.extend(expand_all_plot_configs([cfg]))

            result = ChatResult(
                action="plot",
                explanation=full_text or "Executing graph commands...",
                plot_config=expanded_configs,
                text_summary=full_text,
            )
        else:
            result = ChatResult(
                action="answer",
                text_summary=full_text,
                raw_response=full_text,
            )
        self.result_ready.emit(result)

    def _on_error(self, message: str) -> None:
        self.thinking_changed.emit(False, "Thinking")
        self.error_occurred.emit(message)


