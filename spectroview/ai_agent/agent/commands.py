"""spectroview/ai_agent/agent/commands.py

Typed commands the agent asks the application to perform.

A tool never touches the UI. It builds one of these and hands it to the
:class:`~spectroview.ai_agent.agent.ports.AppContext`, which queues it until the
agent turn ends. The ViewModel then normalises and forwards the queue to
whichever workspace can carry it out.

Typed commands replace the earlier convention of raw dicts distinguished by
magic keys (``{"_graph_update": ...}``), which four separate layers had to
recognise and none owned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentCommand:
    """Base class for anything the agent asks the application to do."""


@dataclass
class CreatePlot(AgentCommand):
    """Open one new graph window from *config*.

    ``config`` uses the MGraph field names (``x``, ``y``, ``plot_style``,
    ``filters``, ``df_name``, plus any styling option).
    """
    config: Dict[str, Any]


@dataclass
class UpdatePlot(AgentCommand):
    """Apply *properties* to an existing graph.

    ``graph_id`` is a graph's numeric ID as a string, or ``"all"``.
    """
    graph_id: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeletePlots(AgentCommand):
    """Close graphs.

    With ``delete_all`` set, ``graph_ids`` inverts the meaning to "close
    everything *except* these" — the shape the delete_graph tool documents.
    """
    delete_all: bool = False
    graph_ids: List[int] = field(default_factory=list)
