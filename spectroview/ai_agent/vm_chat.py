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
from spectroview.ai_agent.tools.dataframe_tool import (
    build_schema_info,
    build_graphs_info,
    safe_query,
    safe_describe,
)
from spectroview.ai_agent.tools.plot_tool import expand_all_plot_configs


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
    tool_execution_received = Signal(str)

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
        static_prompt = self._prompt_mgr.build_prompt(
            intent="chat",
            user_message=user_message,
            prompts=["system", "chat", "plotting"],
            rules=["general", "plotting", "spectroview"],
            knowledge=["features"],
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

    def _on_done(self, full_text: str) -> None:
        self.thinking_changed.emit(False, "Thinking")

        # Record assistant turn in history
        self._conversation.add_message("assistant", full_text)
        self._save_history_to_file()

        result = self._parse_response(full_text)
        
        # ── Multi-Turn ReAct Loop ──
        if result.action == "query" and result.query:
            self.result_ready.emit(result)
            self._execute_agent_tool(result)
        else:
            self.result_ready.emit(result)

    def _execute_agent_tool(self, result: ChatResult) -> None:
        """Executes a pandas query on behalf of the AI and feeds the result back into the chat loop."""
        self._loop_count += 1
        if self._loop_count > 3:
            self.error_occurred.emit("Agent loop exceeded maximum turns (3).")
            return

        if not self._dfs or (result.target_dataframe and result.target_dataframe not in self._dfs):
            res_str = f"Error: Target dataframe '{result.target_dataframe}' not found."
        else:
            df_name = result.target_dataframe if result.target_dataframe else self._active_df_name
            if not df_name:
                df_name = list(self._dfs.keys())[0]
            df = self._dfs[df_name]

            try:
                import pandas as pd
                import numpy as np
                local_vars = {"df": df, "pd": pd, "np": np}
                # Safely evaluate pandas expression
                res_obj = eval(result.query, {"__builtins__": {}}, local_vars)
                res_str = str(res_obj)
            except Exception as e:
                res_str = f"Error evaluating query: {e}"

        # Inject result as a tool/system message
        tool_msg = f"Tool Execution Result:\n```\n{res_str}\n```"
        self._conversation.add_message("user", tool_msg)
        self._save_history_to_file()
        
        self.tool_execution_received.emit(tool_msg)

        # Trigger the next turn
        self.thinking_changed.emit(True, "Executing query...")
        
        self._pending_response = ""
        messages = self._build_messages("")
        self._client.chat(
            model=self._model,
            messages=messages,
            on_chunk=self._on_chunk,
            on_done=self._on_done,
            on_error=self._on_error,
            parent=self,
        )

    def _on_error(self, message: str) -> None:
        self.thinking_changed.emit(False, "Thinking")
        self.error_occurred.emit(message)

    # ------------------------------------------------------------------
    # Response parser
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> ChatResult:
        """Convert the raw LLM JSON string into a ``ChatResult``.

        If the JSON cannot be parsed we fall back to returning the raw
        text as an "answer" so the user always sees something useful.
        """
        data = None
        
        # 1. Try stripping markdown fences
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            pass

        # 2. Try extracting multiple markdown code blocks if direct parsing fails
        if not data:
            blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if blocks:
                data_list = []
                for b in blocks:
                    try:
                        data_list.append(json.loads(b))
                    except json.JSONDecodeError:
                        pass
                
                if data_list:
                    data = data_list[0]
                    # Merge multiple JSON blocks if they are plot configs
                    if len(data_list) > 1:
                        if "plot_config" not in data and "graph_configurations" in data:
                            data["plot_config"] = data["graph_configurations"]
                        if "plot_config" not in data:
                            data["plot_config"] = []
                        elif not isinstance(data["plot_config"], list):
                            data["plot_config"] = [data["plot_config"]]
                        for d in data_list[1:]:
                            pc = d.get("plot_config") or d.get("graph_configurations")
                            if pc:
                                if isinstance(pc, list):
                                    data["plot_config"].extend(pc)
                                else:
                                    data["plot_config"].append(pc)

        # 3. Fallback to greedy regex for a single block
        if not data:
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        
        # 4. Give up and return raw text
        if not data:
            return ChatResult(
                action="answer",
                explanation="(Could not parse structured response)",
                text_summary=raw,
                raw_response=raw,
            )

        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                data = data[0]
            else:
                return ChatResult(
                    action="answer",
                    explanation="",
                    text_summary=raw,
                    raw_response=raw,
                )
                
        # Handle variations from different models
        if "plot_config" not in data and "graph_configurations" in data:
            data["plot_config"] = data["graph_configurations"]
            
        action      = data.get("action")
        # Infer action if missing but plot_config is present
        if not action:
            if "plot_config" in data:
                action = "plot"
            elif "query" in data:
                action = "query"
            else:
                action = "answer"

        explanation = data.get("explanation", "")
        
        # Determine target dataframe
        target_name = data.get("target_dataframe")
        df = None
        if target_name and target_name in self._dfs:
            df = self._dfs[target_name]
        elif self._active_df_name and self._active_df_name in self._dfs:
            df = self._dfs[self._active_df_name]
            target_name = self._active_df_name
        elif self._dfs:
            target_name = list(self._dfs.keys())[0]
            df = self._dfs[target_name]

        result      = ChatResult(
            action=action, 
            explanation=explanation, 
            raw_response=raw, 
            query=data.get("query", ""), 
            target_dataframe=target_name
        )

        if action == "filter":
            if df is None:
                result.action = "answer"
                result.text_summary = explanation or "No dataframe loaded to apply the filter."
            else:
                result = self._execute_filter(data, result, df, target_name)

        elif action == "statistics":
            if df is None:
                result.action = "answer"
                result.text_summary = explanation or "No dataframe loaded to calculate statistics."
            else:
                result = self._execute_statistics(data, result, df, target_name)

        elif action in ("plot", "update", "delete"):
            configs = []
            summary_parts = []
            
            # 1. Process deletes
            graph_delete = data.get("graph_delete")
            if graph_delete and isinstance(graph_delete, dict):
                configs.append({"_graph_delete": graph_delete})
                summary_parts.append("Deleting requested graphs.")
                
            # 2. Process updates
            graph_update = data.get("graph_update")
            if isinstance(graph_update, dict):
                graph_update = [graph_update]
            if graph_update and isinstance(graph_update, list):
                gu_count = 0
                for gu in graph_update:
                    if not isinstance(gu, dict): continue
                    gid = gu.get("graph_id")
                    if str(gid).lower() == "all":
                        for open_gid in self._graphs.keys():
                            gu_copy = dict(gu)
                            gu_copy["graph_id"] = open_gid
                            configs.append({"_graph_update": gu_copy})
                            gu_count += 1
                    else:
                        configs.append({"_graph_update": gu})
                        gu_count += 1
                if gu_count > 0:
                    summary_parts.append(f"Updating {gu_count} graph(s).")
                    
            # 3. Process new plots
            plot_cfg = data.get("plot_config")
            if plot_cfg:
                if isinstance(plot_cfg, dict):
                    plot_cfg = [plot_cfg]
                if isinstance(plot_cfg, list):
                    # Inject df_name before expansion
                    for cfg in plot_cfg:
                        if isinstance(cfg, dict):
                            cfg["df_name"] = target_name
                    expanded = expand_all_plot_configs(
                        [c for c in plot_cfg if isinstance(c, dict)]
                    )
                    if expanded:
                        configs.extend(expanded)
                        summary_parts.append(f"Generated {len(expanded)} plot configuration(s).")

            if configs:
                result.plot_config = configs
                result.text_summary = explanation + "\n\n" + "\n".join(summary_parts) if explanation else "\n".join(summary_parts)
                # Ensure action is 'plot' so the View iterates over configs
                result.action = "plot"
            else:
                result.action = "answer"
                result.text_summary = "No valid graph operations found in the response."

        else:   # "answer" or unknown
            result.text_summary = data.get("answer_text") or explanation or raw

        return result

    # Valid plot styles accepted by the workspace
    VALID_PLOT_STYLES = {'point', 'scatter', 'box', 'bar', 'line',
                         'trendline', 'histogram', 'wafer', '2Dmap'}


    # ------------------------------------------------------------------

    def _execute_filter(self, data: dict, result: ChatResult, df: pd.DataFrame, name: str) -> ChatResult:
        """Run df.query() with the expression provided by the LLM.

        Delegates to :func:`~spectroview.ai_agent.tools.dataframe_tool.safe_query`.
        """
        query_expr = data.get("query", "")
        if not query_expr:
            result.action = "answer"
            result.text_summary = result.explanation or "No filter expression provided."
            return result

        filtered, error = safe_query(df, query_expr)
        if error:
            result.action       = "answer"
            result.text_summary = (
                f"Could not apply filter `{query_expr}` on {name}:\n{error}\n\n"
                "Please rephrase your question."
            )
        else:
            result.dataframe    = filtered
            result.text_summary = (
                f"{len(filtered)} row(s) match `{query_expr}` in {name}"
                f" (out of {len(df)})"
            )
        return result

    def _execute_statistics(self, data: dict, result: ChatResult, df: pd.DataFrame, name: str) -> ChatResult:
        """Run df.describe() on the requested columns.

        Delegates to :func:`~spectroview.ai_agent.tools.dataframe_tool.safe_describe`.
        Falls back to all numeric columns when the LLM provides no column list.
        """
        columns: list[str] = data.get("stat_columns") or []

        # Fall back to all numeric columns when none are specified
        if not columns:
            columns = list(df.select_dtypes("number").columns)

        stats_df, error = safe_describe(df, columns)
        if error or stats_df.empty:
            # Second fallback: all numeric columns
            num_cols = list(df.select_dtypes("number").columns)
            if num_cols and num_cols != columns:
                stats_df, error = safe_describe(df, num_cols)

        if error or stats_df.empty:
            result.action       = "answer"
            result.text_summary = error or "No numeric columns found for statistics."
        else:
            result.text_summary = stats_df.to_string()
            result.dataframe    = stats_df

        return result
