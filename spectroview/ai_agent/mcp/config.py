"""spectroview/ai_agent/mcp/config.py

Declarative registry of the MCP servers the agent connects to.

Adding a server is a block in ``config/servers.yaml`` — no Python change:

.. code-block:: yaml

    servers:
      - id: spectroview
        transport: in-process
        factory: spectroview.ai_agent.mcp.server:create_mcp_server
        enabled: true

      - id: filesystem
        transport: stdio
        command: [npx, -y, "@modelcontextprotocol/server-filesystem", "."]
        enabled: false
        tools: [read_file, list_directory]   # optional allowlist

Transports
----------
``in-process``  a FastMCP server built inside this application; ``factory`` is
                an ``module:function`` path taking the AppContext.
``stdio``       an external process spoken to over stdin/stdout.
``http``        a remote server over streamable HTTP; needs ``url``.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

#: Transports this client knows how to open.
TRANSPORTS = ("in-process", "stdio", "http")


@dataclass
class ServerSpec:
    """One configured MCP server."""

    id: str
    transport: str = "in-process"
    enabled: bool = True
    #: "module:function" — in-process only.
    factory: str = ""
    #: argv, stdio only. The first element is the executable.
    command: List[str] = field(default_factory=list)
    #: Extra environment for a stdio child process.
    env: Dict[str, str] = field(default_factory=dict)
    #: Endpoint, http only.
    url: str = ""
    #: When non-empty, only these tool names are exposed to the model.
    tools: List[str] = field(default_factory=list)

    def validate(self) -> Optional[str]:
        """Return why this spec is unusable, or ``None`` if it is fine."""
        if not self.id:
            return "server entry has no 'id'"
        if self.transport not in TRANSPORTS:
            return (f"server {self.id!r}: unknown transport {self.transport!r} "
                    f"(expected one of {', '.join(TRANSPORTS)})")
        if self.transport == "in-process" and ":" not in self.factory:
            return f"server {self.id!r}: in-process needs factory 'module:function'"
        if self.transport == "stdio" and not self.command:
            return f"server {self.id!r}: stdio needs a non-empty 'command'"
        if self.transport == "http" and not self.url:
            return f"server {self.id!r}: http needs a 'url'"
        return None

    def allows(self, tool_name: str) -> bool:
        """Whether *tool_name* passes this server's allowlist."""
        return not self.tools or tool_name in self.tools


def _expand(value: str) -> str:
    """Expand ``${VAR}`` and ``~`` so specs can reference the environment."""
    return os.path.expanduser(os.path.expandvars(value))


def _spec_from_dict(raw: Dict[str, Any]) -> ServerSpec:
    return ServerSpec(
        id=str(raw.get("id", "")),
        transport=str(raw.get("transport", "in-process")),
        enabled=bool(raw.get("enabled", True)),
        factory=str(raw.get("factory", "")),
        command=[_expand(str(c)) for c in raw.get("command", [])],
        env={str(k): _expand(str(v)) for k, v in (raw.get("env") or {}).items()},
        url=_expand(str(raw.get("url", ""))),
        tools=[str(t) for t in raw.get("tools", [])],
    )


def load_server_specs(path: Optional[Path | str] = None) -> List[ServerSpec]:
    """Load the enabled, valid server specs from ``config/servers.yaml``.

    Invalid entries are logged and skipped rather than raising: one bad block
    must not take the whole chat offline.
    """
    config_path = Path(path) if path else Path(__file__).parent.parent / "config" / "servers.yaml"
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        logger.warning("MCP server config not found: %s", config_path)
        return []
    except Exception as exc:                    # noqa: BLE001 - never break startup
        logger.error("Error reading MCP server config %s: %s", config_path, exc)
        return []

    specs: List[ServerSpec] = []
    seen: set[str] = set()
    for entry in raw.get("servers", []) or []:
        if not isinstance(entry, dict):
            logger.warning("Ignoring non-mapping server entry: %r", entry)
            continue
        spec = _spec_from_dict(entry)
        problem = spec.validate()
        if problem:
            logger.warning("Ignoring invalid MCP server config — %s", problem)
            continue
        if spec.id in seen:
            logger.warning("Ignoring duplicate MCP server id %r", spec.id)
            continue
        seen.add(spec.id)
        if spec.enabled:
            specs.append(spec)

    return specs
