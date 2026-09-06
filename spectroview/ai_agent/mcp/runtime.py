"""Lifecycle for SPECTROview's opt-in localhost MCP endpoint."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


class LocalMCPRuntime:
    """Serve the desktop MCP application on loopback using Streamable HTTP.

    The runtime owns one uvicorn thread and supports deterministic shutdown.
    It never accepts a configurable bind address: public-network exposure is
    outside SPECTROview's threat model for this first local integration.
    """

    def __init__(
        self,
        context: Any,
        port: int = DEFAULT_PORT,
        startup_timeout: float = 5.0,
    ) -> None:
        port = int(port)
        if not 1024 <= port <= 65535:
            raise ValueError("MCP port must be between 1024 and 65535.")
        self.context = context
        self.host = LOOPBACK_HOST
        self.port = port
        self.startup_timeout = float(startup_timeout)
        self._thread: Optional[threading.Thread] = None
        self._uvicorn_server = None
        self._failure: Optional[BaseException] = None

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}/mcp"

    @property
    def running(self) -> bool:
        return bool(
            self._thread is not None
            and self._thread.is_alive()
            and self._uvicorn_server is not None
            and self._uvicorn_server.started
        )

    def start(self) -> None:
        """Start once and wait until uvicorn has bound the loopback socket."""
        if self.running:
            return
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("The local MCP server is still starting.")

        # Imports are lazy so MCP/uvicorn add no startup cost while the
        # endpoint remains disabled (the default).
        import uvicorn

        from spectroview.ai_agent.mcp.server import create_desktop_mcp_server

        mcp = create_desktop_mcp_server(
            self.context, host=self.host, port=self.port
        )
        app = mcp.streamable_http_app()
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._uvicorn_server = uvicorn.Server(config)
        self._failure = None

        def run() -> None:
            try:
                self._uvicorn_server.run()
            except BaseException as exc:  # surfaced by start(), never lost
                self._failure = exc
                logger.exception("Local MCP server failed")

        self._thread = threading.Thread(
            target=run, name="spectroview-mcp", daemon=True
        )
        self._thread.start()

        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._failure is not None:
                self.stop()
                raise RuntimeError(f"Local MCP server failed: {self._failure}")
            if self.running:
                logger.info("SPECTROview MCP listening at %s", self.endpoint)
                return
            if self._thread is not None and not self._thread.is_alive():
                break
            time.sleep(0.02)

        self.stop()
        raise RuntimeError(
            f"Local MCP server could not listen on {self.host}:{self.port}. "
            "The port may already be in use."
        )

    def stop(self) -> None:
        """Request graceful ASGI shutdown and join the server thread."""
        server = self._uvicorn_server
        thread = self._thread
        if server is not None:
            server.should_exit = True
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        self._thread = None
        self._uvicorn_server = None

    def restart(self, port: Optional[int] = None) -> None:
        self.stop()
        if port is not None:
            port = int(port)
            if not 1024 <= port <= 65535:
                raise ValueError("MCP port must be between 1024 and 65535.")
            self.port = port
        self.start()
