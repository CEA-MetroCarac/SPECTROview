"""Provider- and Qt-agnostic core of the SPECTROview AI Agent.

``commands`` defines what the agent can ask the application to do; ``ports``
defines what the agent is allowed to know about it. Both are plain Python so the
tools and the loop can be tested without Qt, an LLM, or a running application.
"""
from spectroview.ai_agent.agent.commands import (
    AgentCommand, CreatePlot, DeletePlots, UpdatePlot,
)
from spectroview.ai_agent.agent.ports import AppContext, RecordingContext

__all__ = [
    "AgentCommand", "CreatePlot", "UpdatePlot", "DeletePlots",
    "AppContext", "RecordingContext",
]
