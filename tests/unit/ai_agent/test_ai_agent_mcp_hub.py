"""
Tests for the multi-server MCP client (mcp/hub.py) and its registry
(mcp/config.py).

The hub replaced a per-tool-batch ``asyncio.run()`` on the GUI thread with one
persistent background loop holding a live session per configured server. What
matters behaviourally: tools from every server are offered under unambiguous
names, calls reach the right server, failures come back as text the model can
act on, and MCP resources are reachable through the synthetic ``get_context``
tool (no LLM API has a native notion of a resource).
"""
import pandas as pd
import pytest

from spectroview.ai_agent.agent.ports import RecordingContext
from spectroview.ai_agent.mcp.config import ServerSpec, load_server_specs
from spectroview.ai_agent.mcp.hub import CONTEXT_TOOL, MCPHub

SPECTROVIEW_SPEC = ServerSpec(
    id="spectroview",
    transport="in-process",
    factory="spectroview.ai_agent.mcp.server:create_mcp_server",
)


def _context() -> RecordingContext:
    return RecordingContext(
        dataframes={"fit_results": pd.DataFrame({
            "Slot": [1, 2, 3],
            "Zone": ["Edge", "Center", "Edge"],
            "fwhm_Si": [1.0, 2.0, 3.0],
        })},
        active_name="fit_results",
        graphs={1: {"style": "box", "x": "Slot", "y": ["fwhm_Si"],
                    "z": "", "df": "fit_results", "filters": []}},
    )


@pytest.fixture
def hub():
    context = _context()
    h = MCPHub(context, specs=[SPECTROVIEW_SPEC])
    h.start()
    h.context = context          # convenience handle for assertions
    yield h
    h.stop()


def _names(hub) -> set[str]:
    return {t["function"]["name"] for t in hub.list_tools()}


class TestLazyStart:
    """Constructing a hub must not spawn anything. Every VMChat owns one, and
    a suite that eagerly connected each of them accumulated dozens of live
    threads and MCP sessions."""

    def test_construction_starts_no_thread(self):
        h = MCPHub(_context(), specs=[SPECTROVIEW_SPEC])
        assert h.connected_servers == []
        assert h._thread is None

    def test_first_use_connects(self):
        h = MCPHub(_context(), specs=[SPECTROVIEW_SPEC])
        try:
            assert "plot_graph" in _names(h)      # list_tools() triggers it
            assert h.connected_servers == ["spectroview"]
        finally:
            h.stop()

    def test_stop_without_start_is_safe(self):
        MCPHub(_context(), specs=[SPECTROVIEW_SPEC]).stop()


class TestConnection:
    def test_configured_server_connects(self, hub):
        assert hub.connected_servers == ["spectroview"]

    def test_all_spectroview_tools_are_offered(self, hub):
        assert {"plot_graph", "query_dataframe", "get_statistics",
                "update_graph", "delete_graph"} <= _names(hub)

    def test_tool_schemas_are_openai_shaped(self, hub):
        tool = next(t for t in hub.list_tools()
                    if t["function"]["name"] == "plot_graph")
        assert tool["type"] == "function"
        assert set(tool["function"]) == {"name", "description", "parameters"}

    def test_a_broken_server_does_not_prevent_the_others(self):
        broken = ServerSpec(id="broken", transport="in-process",
                            factory="spectroview.nonexistent:create")
        h = MCPHub(_context(), specs=[broken, SPECTROVIEW_SPEC])
        h.start()
        try:
            assert h.connected_servers == ["spectroview"]
            assert "plot_graph" in _names(h)
        finally:
            h.stop()

    def test_stop_is_idempotent(self, hub):
        hub.stop()
        hub.stop()


class TestToolCalls:
    def test_call_reaches_the_server(self, hub):
        assert "Statistics:" in hub.call_tool("get_statistics", {"columns": ["fwhm_Si"]})

    def test_graph_tool_submits_a_command_to_the_context(self, hub):
        hub.call_tool("plot_graph", {"x": "Slot", "y": "fwhm_Si", "plot_style": "box"})
        assert [type(c).__name__ for c in hub.context.commands] == ["CreatePlot"]

    def test_unknown_tool_returns_actionable_text_instead_of_raising(self, hub):
        text = hub.call_tool("not_a_tool", {})
        assert "unknown tool" in text.lower()
        assert "plot_graph" in text          # tells the model what does exist

    def test_bad_arguments_come_back_as_text(self, hub):
        text = hub.call_tool("get_statistics", {"columns": "not-a-list"})
        assert "error" in text.lower()


class TestAllowlist:
    def test_only_allowlisted_tools_are_offered(self):
        spec = ServerSpec(id="spectroview", transport="in-process",
                          factory=SPECTROVIEW_SPEC.factory,
                          tools=["plot_graph", "query_dataframe"])
        h = MCPHub(_context(), specs=[spec])
        h.start()
        try:
            assert _names(h) - {CONTEXT_TOOL} == {"plot_graph", "query_dataframe"}
        finally:
            h.stop()


class TestNameQualification:
    def test_unique_names_stay_unqualified(self, hub):
        """Prompts and the models' learned habits both refer to `plot_graph`;
        it must not silently become `spectroview__plot_graph`."""
        assert "plot_graph" in _names(hub)
        assert not any(name.startswith("spectroview__") for name in _names(hub))

    def test_colliding_names_are_qualified_for_every_owner(self):
        """Two servers exposing the same tool: both get prefixed, so neither
        call is ambiguous."""
        specs = [SPECTROVIEW_SPEC,
                 ServerSpec(id="other", transport="in-process",
                            factory=SPECTROVIEW_SPEC.factory)]
        h = MCPHub(_context(), specs=specs)
        h.start()
        try:
            names = _names(h)
            assert "spectroview__plot_graph" in names
            assert "other__plot_graph" in names
            assert "plot_graph" not in names
        finally:
            h.stop()

    def test_a_qualified_call_reaches_its_own_server(self):
        specs = [SPECTROVIEW_SPEC,
                 ServerSpec(id="other", transport="in-process",
                            factory=SPECTROVIEW_SPEC.factory)]
        h = MCPHub(_context(), specs=specs)
        h.start()
        try:
            assert "Statistics:" in h.call_tool(
                "other__get_statistics", {"columns": ["fwhm_Si"]})
        finally:
            h.stop()


class TestResources:
    def test_resources_are_discovered(self, hub):
        assert {r.uri for r in hub.list_resources()} == {
            "spectroview://dataframes/detail",
            "spectroview://graphs/detail",
        }

    def test_get_context_tool_is_offered_when_resources_exist(self, hub):
        assert CONTEXT_TOOL in _names(hub)

    def test_get_context_uri_enum_lists_exactly_the_real_resources(self, hub):
        tool = next(t for t in hub.list_tools()
                    if t["function"]["name"] == CONTEXT_TOOL)
        enum = tool["function"]["parameters"]["properties"]["uri"]["enum"]
        assert set(enum) == {r.uri for r in hub.list_resources()}

    def test_dataframe_detail_carries_what_the_prompt_omits(self, hub):
        """The prompt pushes names and dtypes only; sample values and the row
        preview must be reachable here or the model cannot see them at all."""
        text = hub.call_tool(CONTEXT_TOOL, {"uri": "spectroview://dataframes/detail"})
        assert "fit_results" in text
        assert "Edge" in text            # a sample value
        assert "Preview" in text

    def test_graph_detail_reports_open_graphs(self, hub):
        text = hub.call_tool(CONTEXT_TOOL, {"uri": "spectroview://graphs/detail"})
        assert "fwhm_Si" in text

    def test_unknown_uri_returns_actionable_text(self, hub):
        text = hub.call_tool(CONTEXT_TOOL, {"uri": "spectroview://nope"})
        assert "unknown resource" in text.lower()
        assert "spectroview://dataframes/detail" in text


class TestServerSpecValidation:
    @pytest.mark.parametrize("spec, fragment", [
        (ServerSpec(id=""), "no 'id'"),
        (ServerSpec(id="x", transport="carrier-pigeon"), "unknown transport"),
        (ServerSpec(id="x", transport="in-process", factory="no_colon"), "module:function"),
        (ServerSpec(id="x", transport="stdio"), "command"),
        (ServerSpec(id="x", transport="http"), "url"),
    ])
    def test_invalid_specs_are_reported(self, spec, fragment):
        problem = spec.validate()
        assert problem is not None and fragment in problem

    def test_valid_spec_passes(self):
        assert SPECTROVIEW_SPEC.validate() is None

    def test_allowlist_gating(self):
        assert ServerSpec(id="x").allows("anything")            # empty = allow all
        assert ServerSpec(id="x", tools=["a"]).allows("a")
        assert not ServerSpec(id="x", tools=["a"]).allows("b")


class TestConfigLoading:
    def test_shipped_config_registers_the_spectroview_server(self):
        specs = load_server_specs()
        assert [s.id for s in specs] == ["spectroview"]
        assert specs[0].transport == "in-process"

    def test_missing_file_yields_no_servers(self, tmp_path):
        assert load_server_specs(tmp_path / "absent.yaml") == []

    def test_disabled_and_invalid_entries_are_skipped(self, tmp_path):
        path = tmp_path / "servers.yaml"
        path.write_text(
            "servers:\n"
            "  - id: good\n"
            "    transport: in-process\n"
            "    factory: mod:fn\n"
            "  - id: off\n"
            "    transport: in-process\n"
            "    factory: mod:fn\n"
            "    enabled: false\n"
            "  - id: bad\n"
            "    transport: nonsense\n",
            encoding="utf-8",
        )
        assert [s.id for s in load_server_specs(path)] == ["good"]

    def test_duplicate_ids_are_dropped(self, tmp_path):
        path = tmp_path / "servers.yaml"
        path.write_text(
            "servers:\n"
            "  - id: dup\n    transport: in-process\n    factory: a:b\n"
            "  - id: dup\n    transport: in-process\n    factory: c:d\n",
            encoding="utf-8",
        )
        specs = load_server_specs(path)
        assert len(specs) == 1 and specs[0].factory == "a:b"

    def test_environment_variables_are_expanded_in_commands(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SPECTRO_TEST_DIR", "/data/spectro")
        path = tmp_path / "servers.yaml"
        path.write_text(
            "servers:\n"
            "  - id: fs\n"
            "    transport: stdio\n"
            '    command: [npx, "-y", "server-filesystem", "${SPECTRO_TEST_DIR}"]\n',
            encoding="utf-8",
        )
        assert load_server_specs(path)[0].command[-1] == "/data/spectro"
