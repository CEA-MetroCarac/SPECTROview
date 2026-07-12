"""
spectroview/llm/vm_chat.py
---------------------------
ViewModel layer for the AI Chat feature.

Responsibilities
----------------
* Build a rich system prompt from the active DataFrame schema.
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

from spectroview.llm.m_llm_client import LLMClient, API_PROVIDERS


# ═══════════════════════════════════════════════════════════════════════════
# Public result dataclasses (plain dicts would work too, but explicit is better)
# ═══════════════════════════════════════════════════════════════════════════

class ChatResult:
    """Carries the parsed LLM response back to the View."""
    __slots__ = ("action", "explanation", "dataframe", "text_summary",
                 "plot_config", "raw_response")

    def __init__(
        self,
        action: str                         = "unknown",
        explanation: str                    = "",
        dataframe: Optional[pd.DataFrame]  = None,
        text_summary: str                   = "",
        plot_config: Optional[dict]         = None,
        raw_response: str                   = "",
    ) -> None:
        self.action       = action          # "filter" | "statistics" | "plot" | "answer"
        self.explanation  = explanation     # human-readable explanation shown in the UI
        self.dataframe    = dataframe       # filtered DataFrame (Tier 1)
        self.text_summary = text_summary    # statistics / answer text (Tier 2)
        self.plot_config  = plot_config     # {x, y, z, plot_style} suggestion (Tier 3)
        self.raw_response = raw_response    # full LLM text (for debugging)


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

    thinking_changed = Signal(bool)
    chunk_received   = Signal(str)
    result_ready     = Signal(object)       # ChatResult
    error_occurred   = Signal(str)

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

        # Chat state
        self._history: List[Dict[str, str]] = []
        self._pending_response: str = ""

        # Active provider label ("Ollama", "Gemini", etc.)
        self._provider: str = LLMClient.DEFAULT_PROVIDER
        
        # Load history folder
        s = QSettings("SPECTROview", "AIChat")
        s.beginGroup("ai_chat")
        self._history_folder = str(s.value("history_folder", ""))
        s.endGroup()

    # ------------------------------------------------------------------
    # Public API — called by VChatPanel
    # ------------------------------------------------------------------

    def set_dataframes(self, dfs: Dict[str, pd.DataFrame], active_name: str = "") -> None:
        """Update the available DataFrames.
        
        History is only cleared when the actual set of dataframes changes
        (e.g. new files loaded/removed).  Simply switching the active
        selection preserves the conversation.
        """
        new_keys = set(dfs.keys()) if dfs else set()
        old_keys = set(self._dfs.keys())

        self._dfs = dfs.copy() if dfs else {}
        self._active_df_name = active_name

        # Only wipe history when the dataframe *set* actually changed
        if new_keys != old_keys:
            self._save_history_to_file()
            self._history.clear()

    def update_active_df_name(self, name: str) -> None:
        """Update only the active dataframe name — preserves chat history."""
        self._active_df_name = name

    def set_graphs(self, graphs: Dict[int, Any]) -> None:
        """Update the known open graphs (for inclusion in system prompt)."""
        self._graphs = {
            gid: {
                "style": getattr(g, 'plot_style', ''),
                "x":     getattr(g, 'x', ''),
                "y":     getattr(g, 'y', []),
                "z":     getattr(g, 'z', ''),
                "df":    getattr(g, 'df_name', ''),
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
            from spectroview.llm.m_llm_client import API_PROVIDERS
            self._model = API_PROVIDERS.get(provider, {}).get("default_model", self._model)
        self._save_history_to_file()
        self._history.clear()

    def get_provider(self) -> str:
        """Return the currently active provider name."""
        return self._provider

    def process_query(self, user_text: str) -> None:
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

        # Build the full message list for this request
        messages = self._build_messages(user_text)

        # Track the user turn now; assistant turn added after response
        self._history.append({"role": "user", "content": user_text})
        self._pending_response = ""

        self.thinking_changed.emit(True)

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
        self.thinking_changed.emit(False)
        self._save_history_to_file()

    def clear_history(self) -> None:
        self._save_history_to_file()
        self._history.clear()

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
        # Only save if there's actual conversation history (more than just one error msg, etc.)
        if not self._history or not self._history_folder:
            return
            
        if not os.path.exists(self._history_folder):
            try:
                os.makedirs(self._history_folder)
            except Exception:
                return
                
        timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}.md"
        filepath = os.path.join(self._history_folder, filename)
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# AI Chat Session - {timestamp}\\n\\n")
                for msg in self._history:
                    role = "User" if msg["role"] == "user" else "AI"
                    content = msg["content"]
                    f.write(f"### {role}\\n{content}\\n\\n")
        except Exception:
            pass

    def _build_messages(self, user_text: str) -> List[Dict[str, str]]:
        """Assemble the full message list = system prompt + capped history + new user turn."""
        system_prompt = self._build_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]

        # Keep only the last MAX_HISTORY_PAIRS pairs to avoid context overflow
        max_msgs = self.MAX_HISTORY_PAIRS * 2
        messages += self._history[-max_msgs:]

        messages.append({"role": "user", "content": user_text})
        return messages

    def _build_system_prompt(self) -> str:
        """Inject DataFrame schemas, a sample, and instructions into the system message."""
        dfs_info = []
        
        for name, df in self._dfs.items():
            col_info_lines = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                sample_vals = df[col].dropna().unique()[:3].tolist()
                col_info_lines.append(f"    - {col!r} ({dtype}): sample values {sample_vals}")
            col_info = "\n".join(col_info_lines)
            
            try:
                preview = df.head(3).to_string(max_cols=8)
            except Exception:
                preview = "(preview unavailable)"
                
            dfs_info.append(
                f"DATAFRAME: {name!r} ({len(df)} rows, {len(df.columns)} columns)\n"
                f"  Columns:\n{col_info}\n"
                f"  Preview:\n{preview}"
            )

        dfs_section = "\n\n".join(dfs_info)
        active_info = f"\nThe currently active DataFrame in the UI is: {self._active_df_name!r}.\n" if self._active_df_name else ""

        # Build open-graphs summary for the AI
        if self._graphs:
            graph_lines = []
            for gid, info in sorted(self._graphs.items()):
                y_str = info['y'][0] if isinstance(info['y'], list) and info['y'] else info['y']
                z_str = f", z={info['z']!r}" if info['z'] else ""
                graph_lines.append(
                    f"  Graph ID {gid}: style={info['style']!r}, x={info['x']!r}, y={y_str!r}{z_str}, df={info['df']!r}"
                )
            graphs_info = "CURRENTLY OPEN GRAPHS:\n" + "\n".join(graph_lines) + "\n"
        else:
            graphs_info = "No graphs are currently open.\n"

        return f"""You are an expert data analyst assistant embedded in SPECTROview, \
a scientific spectroscopy application.
The user has the following pandas DataFrames loaded:

{dfs_section}
{active_info}
{graphs_info}
YOUR JOB: Analyse the user's natural-language question and respond with ONLY a valid \
JSON object — no markdown fences, no explanatory text outside the JSON.

The JSON must have this exact structure:
{{
  "action": "<one of: filter | statistics | plot | update | delete | answer>",
  "explanation": "<short human-readable description of what you are doing>",
  "target_dataframe": "<name of the dataframe to operate on, or null>",
  "query": "<valid pandas .query() expression using column names, or null>",
  "stat_columns": ["<col1>", "<col2>"],
  "graph_update": [
    {{
      "graph_id": <integer graph ID to modify, or "all" to apply to all open graphs>,
      "properties": {{
        "ymin": 3.6,
        "ymax": 4.2,
        "plot_title": "New title",
        "plot_style": "scatter",
        "filters": ["Quadrant != 'Q4'"]
      }}
    }}
  ] or null,
  "graph_delete": {{
    "delete_all": false,
    "graph_ids": [1, 2, 3]
  }} or null,
  "plot_config": [
    {{
      "x": "<column name for X axis>",
      "y": "<column name for primary Y axis>",
      "z": "<column name for hue/color grouping, or null>",
      "filters": ["<valid pandas .query() string, e.g. \"Quadrant != 'Q4'\", or null>"],
      "plot_style": "<one of: point scatter box bar line trendline histogram wafer 2Dmap, or multiple separated by comma like 'box, bar'>",
      "plot_title": "<title text or null>",
      "xlabel": "<X axis label or null>",
      "ylabel": "<Y axis label or null>",
      "zlabel": "<Z/color axis label or null>",
      "xmin": "<min X value (number) or null>",
      "xmax": "<max X value (number) or null>",
      "ymin": "<min Y value (number) or null>",
      "ymax": "<max Y value (number) or null>",
      "zmin": "<min Z value (number) or null>",
      "zmax": "<max Z value (number) or null>",
      "xlogscale": false,
      "ylogscale": false,
      "grid": false,
      "x_rot": 0,
      "legend_visible": true,
      "legend_outside": false,
      "color_palette": "<jet | viridis | plasma | inferno | magma | coolwarm | RdBu | Spectral | tab10 | Set2 | Set3 or null>",
      "scatter_size": 70,
      "join_for_point_plot": false,
      "dodge_point_plot": true,
      "show_bar_plot_error_bar": false,
      "trendline_order": 1,
      "hist_bins": 20,
      "hist_kde": false,
      "plot_width": 480,
      "plot_height": 420,
      "dpi": 100
    }}
  ] or null,
  "answer_text": "<plain text answer for general questions, or null>"
}}

AVAILABLE PLOT STYLES (use EXACTLY these strings): point, scatter, box, bar, line, trendline, histogram, wafer, 2Dmap

RULES:
- For "filter": set "query" to a valid pandas .query() string (e.g. "fwhm_Si > 5.0").
- For "statistics": set "stat_columns" to the list of columns to describe.
- For "plot": fill "plot_config". If plotting multiple styles with identical parameters, output a SINGLE entry in the list and set plot_style to a comma-separated string (e.g., "box, bar, point") for faster generation.
  Only include optional fields (like xmin, ymin, plot_title, grid, etc.) when the user explicitly asks for them.
- For "answer": set "answer_text" to a plain-text explanation.
- For updating an existing graph: use action="update" and set "graph_update" as a list of update objects. Set "graph_id" to the specific ID or "all" to update all open graphs.
- For deleting an existing graph: use action="delete" and set "graph_delete". Set delete_all to true to delete all graphs, or pass specific IDs in graph_ids. For "delete all except 70", pass all open graph IDs EXCEPT 70.
- Set "target_dataframe" to the exact name of the dataframe you are querying. If not specified by user, use the active one.
- NEVER use eval(), exec(), or any Python code other than pandas .query() expressions.
- If the user asks something impossible with this data, use action="answer" to explain why.
- CONVERSATION MEMORY: You have access to the full conversation history. When the user \
makes a follow-up request like "add also a scatter plot" or "do the same but with line plot", \
you MUST reuse the same x, y, z columns, target_dataframe, axis limits, and other settings \
from the previous request. 
  CRITICAL: Do NOT output configuration for plots you have already created in previous turns. ONLY output the newly requested plots.
- Return ONLY the JSON object, no surrounding text."""

    # ------------------------------------------------------------------
    # Worker callbacks (called from the worker QThread via Qt signals)
    # ------------------------------------------------------------------

    def _on_chunk(self, fragment: str) -> None:
        self._pending_response += fragment
        self.chunk_received.emit(fragment)

    def _on_done(self, full_text: str) -> None:
        self.thinking_changed.emit(False)

        # Record assistant turn in history
        self._history.append({"role": "assistant", "content": full_text})

        result = self._parse_response(full_text)
        self.result_ready.emit(result)

    def _on_error(self, message: str) -> None:
        self.thinking_changed.emit(False)
        self.error_occurred.emit(message)

    # ------------------------------------------------------------------
    # Response parser
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> ChatResult:
        """Convert the raw LLM JSON string into a ``ChatResult``.

        If the JSON cannot be parsed we fall back to returning the raw
        text as an "answer" so the user always sees something useful.
        """
        # Strip markdown code fences the model sometimes adds despite instructions
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Try to extract the first {...} block
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return ChatResult(
                        action="answer",
                        explanation="(Could not parse structured response)",
                        text_summary=raw,
                        raw_response=raw,
                    )
            else:
                return ChatResult(
                    action="answer",
                    explanation="",
                    text_summary=raw,
                    raw_response=raw,
                )

        action      = data.get("action", "answer")
        explanation = data.get("explanation", "")
        
        # Determine target dataframe
        target_name = data.get("target_dataframe")
        if target_name and target_name in self._dfs:
            df = self._dfs[target_name]
        elif self._active_df_name and self._active_df_name in self._dfs:
            df = self._dfs[self._active_df_name]
            target_name = self._active_df_name
        elif self._dfs:
            target_name = list(self._dfs.keys())[0]
            df = self._dfs[target_name]
        else:
            return ChatResult(action="answer", explanation="No dataframe available to process this request.", raw_response=raw)

        result      = ChatResult(action=action, explanation=explanation, raw_response=raw)

        if action == "filter":
            result = self._execute_filter(data, result, df, target_name)

        elif action == "statistics":
            result = self._execute_statistics(data, result, df, target_name)

        elif action == "plot":
            result = self._execute_plot(data, result, df, target_name)

        elif action == "update":
            # Graph update — pass graph_update payload through to the View
            graph_update = data.get("graph_update")
            if isinstance(graph_update, dict):
                graph_update = [graph_update]
                
            if graph_update and isinstance(graph_update, list):
                configs = []
                for gu in graph_update:
                    if not isinstance(gu, dict):
                        continue
                    gid = gu.get("graph_id")
                    if str(gid).lower() == "all":
                        for open_gid in self._graphs.keys():
                            gu_copy = dict(gu)
                            gu_copy["graph_id"] = open_gid
                            configs.append({"_graph_update": gu_copy})
                    else:
                        configs.append({"_graph_update": gu})
                        
                if configs:
                    result.plot_config = configs
                    result.text_summary = explanation or f"Updating {len(configs)} graph(s)."
                else:
                    result.action = "answer"
                    result.text_summary = "No valid graph update information provided."
            else:
                result.action = "answer"
                result.text_summary = "No graph update information provided."

        elif action == "delete":
            graph_delete = data.get("graph_delete")
            if graph_delete and isinstance(graph_delete, dict):
                result.plot_config = [{"_graph_delete": graph_delete}]
                result.text_summary = explanation or "Deleting requested graphs."
            else:
                result.action = "answer"
                result.text_summary = "No graph delete information provided."

        else:   # "answer" or unknown
            result.text_summary = data.get("answer_text") or explanation or raw

        return result

    # Valid plot styles accepted by the workspace
    VALID_PLOT_STYLES = {'point', 'scatter', 'box', 'bar', 'line',
                         'trendline', 'histogram', 'wafer', '2Dmap'}

    def _execute_plot(self, data: dict, result: ChatResult, df: pd.DataFrame, name: str) -> ChatResult:
        """Parse plot suggestions — expand comma-separated styles into separate configs."""
        plot_cfg = data.get("plot_config")
        if not plot_cfg:
            result.action = "answer"
            result.text_summary = "No plot configuration provided."
            return result

        # Normalise to list
        if isinstance(plot_cfg, dict):
            plot_cfg = [plot_cfg]

        # Expand entries where plot_style is a comma-separated string
        # (e.g. LLM returns "box, bar, point" in one entry instead of 3)
        expanded: list = []
        for cfg in plot_cfg:
            if not isinstance(cfg, dict):
                continue
            style_val = cfg.get("plot_style", "")
            # Check if it's a comma-separated multi-style string
            if isinstance(style_val, str) and "," in style_val:
                styles = [s.strip() for s in style_val.split(",")]
                for s in styles:
                    if s in self.VALID_PLOT_STYLES:
                        new_cfg = dict(cfg)   # copy all shared params
                        new_cfg["plot_style"] = s
                        new_cfg["df_name"] = name
                        expanded.append(new_cfg)
            else:
                cfg["df_name"] = name
                expanded.append(cfg)

        if not expanded:
            result.action = "answer"
            result.text_summary = "No valid plot configurations could be determined."
            return result

        result.plot_config = expanded
        result.text_summary = f"Generated {len(expanded)} plot configuration(s)."
        return result

    # ------------------------------------------------------------------

    def _execute_filter(self, data: dict, result: ChatResult, df: pd.DataFrame, name: str) -> ChatResult:
        """Run df.query() with the expression provided by the LLM."""
        query_expr = data.get("query", "")
        if not query_expr:
            result.action = "answer"
            result.text_summary = result.explanation or "No filter expression provided."
            return result

        try:
            filtered = df.query(query_expr)
            result.dataframe    = filtered
            result.text_summary = (
                f"{len(filtered)} row(s) match `{query_expr}` in {name}"
                f" (out of {len(df)})"
            )
        except Exception as exc:
            result.action       = "answer"
            result.text_summary = (
                f"Could not apply filter `{query_expr}` on {name}:\n{exc}\n\n"
                "Please rephrase your question."
            )
        return result

    def _execute_statistics(self, data: dict, result: ChatResult, df: pd.DataFrame, name: str) -> ChatResult:
        """Run df.describe() on the requested columns."""
        columns = data.get("stat_columns") or []

        # Validate columns exist
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            # Fall back to numeric columns
            valid_cols = list(df.select_dtypes("number").columns)

        if not valid_cols:
            result.action = "answer"
            result.text_summary = "No numeric columns found for statistics."
            return result

        try:
            stats_df = df[valid_cols].describe()
            result.text_summary = stats_df.to_string()
            result.dataframe    = stats_df
        except Exception as exc:
            result.action       = "answer"
            result.text_summary = f"Statistics failed: {exc}"

        return result
