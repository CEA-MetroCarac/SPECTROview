"""spectroview/ai_agent/agent/ports.py

The boundary between the MCP tools and the running application.

``AppContext`` is everything a tool is allowed to know about SPECTROview: how to
read the loaded DataFrames and open graphs, and how to submit a command. The
MCP server depends on this Protocol, never on ``VMChat`` — which is what lets
the tools be unit-tested against a fake, and lets the same server later run in a
separate process where no ViewModel exists.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import pandas as pd

from spectroview.ai_agent.agent.commands import AgentCommand


@runtime_checkable
class AppContext(Protocol):
    """Read the application's data; submit commands back to it."""

    def list_dataframes(self) -> List[str]:
        """Names of every loaded DataFrame."""
        ...

    def active_dataframe_name(self) -> str:
        """Name of the DataFrame selected in the UI, or ``""``."""
        ...

    def get_dataframe(self, name: str = "") -> Optional[pd.DataFrame]:
        """The named DataFrame, or the active one when *name* is empty.

        Returns ``None`` if it does not exist.
        """
        ...

    def list_graphs(self) -> Dict[int, Dict[str, Any]]:
        """Complete open-graph state keyed by graph ID.

        Implementations should return the canonical ``MGraph.save()`` shape.
        Compact legacy fakes using ``style``/``df`` remain accepted by the
        MCP boundary for backward compatibility.
        """
        ...

    def submit(self, command: AgentCommand) -> Any:
        """Queue or execute *command* and optionally return its domain result."""
        ...

    def load_dataframes(self, file_paths: List[str]) -> List[str]:
        """Load external Excel or CSV dataframes into the application."""
        ...

    def show_graph(self, graph_id: Optional[int] = None) -> None:
        """Bring the application and specified graph window to the foreground."""
        ...


class RecordingContext:
    """In-memory :class:`AppContext` — the fake for tests, and the base the
    ViewModel's own adapter builds on.

    Holds the DataFrames/graphs it is given and records submitted commands in
    ``commands`` instead of executing them.
    """

    def __init__(
        self,
        dataframes: Optional[Dict[str, pd.DataFrame]] = None,
        active_name: str = "",
        graphs: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> None:
        self.dataframes: Dict[str, pd.DataFrame] = dataframes or {}
        self.active_name = active_name
        self.graphs: Dict[int, Dict[str, Any]] = graphs or {}
        self.commands: List[AgentCommand] = []

    def list_dataframes(self) -> List[str]:
        return list(self.dataframes)

    def active_dataframe_name(self) -> str:
        return self.active_name

    def get_dataframe(self, name: str = "") -> Optional[pd.DataFrame]:
        return self.dataframes.get(name or self.active_name)

    def list_graphs(self) -> Dict[int, Dict[str, Any]]:
        return self.graphs

    def submit(self, command: AgentCommand) -> None:
        self.commands.append(command)

    def load_dataframes(self, file_paths: List[str]) -> List[str]:
        import os
        from pathlib import Path
        loaded = []
        for fp in file_paths:
            p = Path(fp)
            if p.is_file():
                try:
                    if p.suffix.lower() in ('.xlsx', '.xls'):
                        dfs = pd.read_excel(p, sheet_name=None)
                        for sname, df in dfs.items():
                            name = f"{p.stem}_{sname}" if len(dfs) > 1 else p.stem
                            self.dataframes[name] = df
                            loaded.append(name)
                    elif p.suffix.lower() in ('.csv', '.tsv'):
                        sep = '\t' if p.suffix.lower() == '.tsv' else ','
                        self.dataframes[p.stem] = pd.read_csv(p, sep=sep)
                        loaded.append(p.stem)
                except Exception:
                    pass
        return loaded

    def show_graph(self, graph_id: Optional[int] = None) -> None:
        pass

    def drain(self) -> List[AgentCommand]:
        """Return the queued commands and clear the queue."""
        queued, self.commands = self.commands, []
        return queued
