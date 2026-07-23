"""spectroview/ai_agent/mcp/hub.py

Client side of MCP: one hub, many servers.

The hub owns a single asyncio event loop running in a daemon thread, and keeps
one initialised :class:`ClientSession` per configured server open for the
application's lifetime. Qt-thread callers use the plain synchronous methods
(:meth:`list_tools`, :meth:`call_tool`, ...), which hand the coroutine to that
loop and wait for the result.

Why a persistent loop: the earlier code called ``asyncio.run()`` directly on the
GUI thread and rebuilt the server session for every tool batch. That blocked the
UI, and it only worked at all because the single server was in-process — a
stdio or HTTP server cannot be reconnected per call.

Tool naming
-----------
Names stay unqualified (``plot_graph``) while they are unique across servers, so
the prompts and the models' learned habits keep working. A name offered by more
than one server is qualified as ``<server_id>__<tool>`` for *every* server
offering it, so the mapping is never ambiguous. ``__`` rather than ``.`` because
OpenAI restricts function names to ``[A-Za-z0-9_-]``.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from spectroview.ai_agent.mcp.config import ServerSpec, load_server_specs

# The ``mcp`` package costs ~0.7 s to import; it is only needed once a session is
# actually opened, so it is imported inside ``_open_session``.
if TYPE_CHECKING:
    from mcp import ClientSession

logger = logging.getLogger(__name__)

#: Seconds to wait for one hub operation before giving up on it.
DEFAULT_TIMEOUT = 60.0

#: Separator between a server id and a tool name when qualification is needed.
QUALIFIER = "__"

#: Synthetic, client-side tool that reads an MCP resource. Not owned by any
#: server — the hub answers it directly (see :meth:`MCPHub._context_tool_schema`).
CONTEXT_TOOL = "get_context"


@dataclass(frozen=True)
class ToolRef:
    """Where an exposed tool actually lives."""
    server_id: str
    raw_name: str


@dataclass(frozen=True)
class ResourceRef:
    """A resource exposed by one server."""
    server_id: str
    uri: str
    name: str = ""
    description: str = ""


class MCPHub:
    """Connects to every configured MCP server and multiplexes their tools."""

    def __init__(
        self,
        context: Any,
        specs: Optional[List[ServerSpec]] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """
        Parameters
        ----------
        context:
            The :class:`~spectroview.ai_agent.agent.ports.AppContext` handed to
            in-process server factories.
        specs:
            Servers to connect. Defaults to ``config/servers.yaml``.
        timeout:
            Seconds any single hub call may take.
        """
        self._context = context
        self._specs = specs if specs is not None else load_server_specs()
        self._timeout = timeout

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stack: Optional[AsyncExitStack] = None
        self._sessions: Dict[str, ClientSession] = {}
        self._ready = threading.Event()

        self._tools: List[Dict[str, Any]] = []       # OpenAI-shaped schemas
        self._tool_refs: Dict[str, ToolRef] = {}     # exposed name -> where it lives
        self._resources: List[ResourceRef] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the loop thread and connect every configured server.

        Idempotent, and called automatically on first use — nothing is spawned
        until the agent actually needs a tool, so merely constructing a
        ViewModel costs no thread and no child process.

        Never raises: a server that fails to connect is logged and skipped, so
        one broken external server cannot disable the whole agent.
        """
        if self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._run_loop, name="mcp-hub", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=self._timeout)

        try:
            self._submit(self._connect_all())
        except Exception as exc:                # noqa: BLE001
            logger.error("MCP hub failed to connect: %s", exc)

    def stop(self) -> None:
        """Close every session and stop the loop thread."""
        if self._loop is None:
            return
        try:
            self._submit(self._disconnect_all())
        except Exception as exc:                # noqa: BLE001
            logger.debug("MCP hub shutdown: %s", exc)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None
        self._loop = None

    def _run_loop(self) -> None:
        """Body of the hub thread: own an event loop until stopped."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()
        self._loop.close()

    def _submit(self, coro) -> Any:
        """Run *coro* on the hub loop from any thread and wait for its result."""
        if self._loop is None:
            raise RuntimeError("MCP hub is not running; call start() first.")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(self._timeout)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def _open_session(self, stack: AsyncExitStack, spec: ServerSpec) -> ClientSession:
        """Open one server's session on the hub loop, per its transport."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.shared.memory import create_connected_server_and_client_session

        if spec.transport == "in-process":
            module_name, _, attr = spec.factory.partition(":")
            factory = getattr(importlib.import_module(module_name), attr)
            server = factory(self._context)
            return await stack.enter_async_context(
                create_connected_server_and_client_session(server._mcp_server))

        if spec.transport == "stdio":
            params = StdioServerParameters(
                command=spec.command[0], args=list(spec.command[1:]),
                env=spec.env or None)
            read, write = await stack.enter_async_context(stdio_client(params))
            return await stack.enter_async_context(ClientSession(read, write))

        # http
        from mcp.client.streamable_http import streamablehttp_client
        read, write, _ = await stack.enter_async_context(
            streamablehttp_client(spec.url))
        return await stack.enter_async_context(ClientSession(read, write))

    async def _connect_all(self) -> None:
        self._stack = AsyncExitStack()
        for spec in self._specs:
            try:
                session = await self._open_session(self._stack, spec)
                await session.initialize()
                self._sessions[spec.id] = session
            except Exception as exc:            # noqa: BLE001
                logger.error("MCP server %r failed to start: %s", spec.id, exc)
        await self._refresh_catalog()

    async def _disconnect_all(self) -> None:
        self._sessions.clear()
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None

    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------

    async def _refresh_catalog(self) -> None:
        """Rebuild the tool/resource catalog from every connected server."""
        per_server: Dict[str, List[Any]] = {}
        self._resources = []

        for server_id, session in self._sessions.items():
            spec = next(s for s in self._specs if s.id == server_id)
            try:
                tools = (await session.list_tools()).tools
            except Exception as exc:            # noqa: BLE001
                logger.error("MCP server %r: list_tools failed: %s", server_id, exc)
                tools = []
            per_server[server_id] = [t for t in tools if spec.allows(t.name)]

            # Resources are optional; a server without them raises rather than
            # returning an empty list, so failure here is expected and quiet.
            try:
                for res in (await session.list_resources()).resources:
                    self._resources.append(ResourceRef(
                        server_id=server_id, uri=str(res.uri),
                        name=res.name or "", description=res.description or ""))
            except Exception:                   # noqa: BLE001
                logger.debug("MCP server %r exposes no resources", server_id)

        # Qualify only names offered by more than one server.
        counts: Dict[str, int] = {}
        for tools in per_server.values():
            for tool in tools:
                counts[tool.name] = counts.get(tool.name, 0) + 1

        self._tools = []
        self._tool_refs = {}
        for server_id, tools in per_server.items():
            for tool in tools:
                exposed = (f"{server_id}{QUALIFIER}{tool.name}"
                           if counts[tool.name] > 1 else tool.name)
                self._tool_refs[exposed] = ToolRef(server_id, tool.name)
                self._tools.append({
                    "type": "function",
                    "function": {
                        "name": exposed,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                })

        if self._resources:
            self._tools.append(self._context_tool_schema())

    def _context_tool_schema(self) -> Dict[str, Any]:
        """Schema for the synthetic ``get_context`` tool.

        MCP resources have no equivalent in any LLM's function-calling API, so
        the hub exposes reading one *as* a tool. The ``uri`` enum lists exactly
        the resources currently available, which also lets grammar-constrained
        local models pick only a real one.
        """
        catalog = "; ".join(
            f"{r.uri} — {r.description or r.name or 'context'}" for r in self._resources)
        return {
            "type": "function",
            "function": {
                "name": CONTEXT_TOOL,
                "description": (
                    "Fetch additional context that is not in your prompt. Call this "
                    "when you need detail you were not given — for example sample "
                    "values or a row preview before choosing columns, or a graph's "
                    f"exact current configuration before editing it. Available: {catalog}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "uri": {
                            "type": "string",
                            "enum": [r.uri for r in self._resources],
                            "description": "Which context to fetch.",
                        },
                    },
                    "required": ["uri"],
                },
            },
        }

    # ------------------------------------------------------------------
    # Public, thread-safe API
    # ------------------------------------------------------------------

    def list_tools(self) -> List[Dict[str, Any]]:
        """Every exposed tool as an OpenAI-style function schema."""
        self.start()
        return list(self._tools)

    def list_resources(self) -> List[ResourceRef]:
        """Every resource exposed by every connected server."""
        self.start()
        return list(self._resources)

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Invoke *name* on whichever server owns it; return its text result.

        Errors are returned as text, never raised: the string goes back to the
        model as the tool result so it can correct itself on the next turn.
        """
        self.start()
        if name == CONTEXT_TOOL:
            return self.read_resource(str(arguments.get("uri", "")))

        ref = self._tool_refs.get(name)
        if ref is None:
            return (f"Error: unknown tool {name!r}. Available tools: "
                    f"{', '.join(sorted(self._tool_refs)) or '(none)'}.")
        session = self._sessions.get(ref.server_id)
        if session is None:
            return f"Error: MCP server {ref.server_id!r} is not connected."

        try:
            result = self._submit(session.call_tool(ref.raw_name, arguments))
        except Exception as exc:                # noqa: BLE001
            return f"Error executing tool {name}: {exc}"
        return _result_text(result)

    def read_resource(self, uri: str) -> str:
        """Read *uri* from the server exposing it; return its text content."""
        self.start()
        ref = next((r for r in self._resources if r.uri == uri), None)
        if ref is None:
            return (f"Error: unknown resource {uri!r}. Available: "
                    f"{', '.join(r.uri for r in self._resources) or '(none)'}.")
        session = self._sessions.get(ref.server_id)
        if session is None:
            return f"Error: MCP server {ref.server_id!r} is not connected."

        try:
            result = self._submit(session.read_resource(uri))
        except Exception as exc:                # noqa: BLE001
            return f"Error reading resource {uri}: {exc}"
        return "\n".join(
            c.text for c in getattr(result, "contents", []) if hasattr(c, "text")
        )

    @property
    def connected_servers(self) -> List[str]:
        """Ids of the servers that are actually connected."""
        return list(self._sessions)


def _result_text(result: Any) -> str:
    """Flatten an MCP tool result into the text handed back to the model."""
    content = getattr(result, "content", None) or []
    parts = [c.text for c in content if hasattr(c, "text")]
    return "\n".join(parts) if parts else str(result)
