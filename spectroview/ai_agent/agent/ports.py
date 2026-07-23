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
        """Open graphs as ``{graph_id: {style, x, y, z, df, filters}}``."""
        ...

    def submit(self, command: AgentCommand) -> None:
        """Queue *command* for the application to carry out."""
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

    def drain(self) -> List[AgentCommand]:
        """Return the queued commands and clear the queue."""
        queued, self.commands = self.commands, []
        return queued
