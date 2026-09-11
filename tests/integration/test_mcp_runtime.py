"""A real Streamable HTTP client can discover the opt-in loopback server."""

import asyncio
import socket

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from spectroview.ai_agent.agent.ports import RecordingContext
from spectroview.ai_agent.mcp.runtime import LocalMCPRuntime


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_runtime_is_loopback_discoverable_and_stops():
    runtime = LocalMCPRuntime(RecordingContext(), port=_free_port())
    runtime.start()
    try:
        assert runtime.running
        assert runtime.endpoint.startswith("http://127.0.0.1:")

        async def discover():
            async with streamable_http_client(runtime.endpoint) as streams:
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return {tool.name for tool in (await session.list_tools()).tools}

        names = asyncio.run(discover())
        assert "get_application_state" in names
        assert "plot_graph" in names
    finally:
        runtime.stop()
    assert not runtime.running
