"""
Tests for spectroview/ai_agent/mcp/server.py — tool schema shapes.

Verifies the JSON Schema actually generated for each MCP tool (what a
tool-calling LLM, including local models via Ollama's grammar-constrained
decoding, sees and is constrained by) rather than just the Python source.
"""
import asyncio

from mcp.shared.memory import create_connected_server_and_client_session

from spectroview import PLOT_STYLES
from spectroview.ai_agent.agent.ports import RecordingContext
from spectroview.ai_agent.mcp.server import VALID_PLOT_STYLES, create_mcp_server
from spectroview.ai_agent.utils.plot_utils import VALID_PLOT_STYLES as PLOT_UTILS_STYLES


def _list_tools():
    async def _run():
        server = create_mcp_server(RecordingContext())
        async with create_connected_server_and_client_session(server._mcp_server) as session:
            await session.initialize()
            res = await session.list_tools()
            return {t.name: t.inputSchema for t in res.tools}
    return asyncio.run(_run())


class TestPlotStyleSingleSourceOfTruth:
    """The tool schema, the server's validator, and the app's own PLOT_STYLES
    must agree. The Literal in server.py is spelled out (MCP needs it static),
    so only a test can catch drift after a new plot style is added."""

    def test_tool_schema_enum_matches_app_plot_styles(self):
        schema = _list_tools()["plot_graph"]
        assert set(schema["properties"]["plot_style"]["enum"]) == set(PLOT_STYLES)

    def test_server_validator_matches_app_plot_styles(self):
        assert VALID_PLOT_STYLES == frozenset(PLOT_STYLES)

    def test_plot_utils_validator_matches_app_plot_styles(self):
        assert PLOT_UTILS_STYLES == frozenset(PLOT_STYLES)


class TestPlotGraphSchema:
    def test_plot_style_has_enum_of_nine_valid_values(self):
        schema = _list_tools()["plot_graph"]
        style_schema = schema["properties"]["plot_style"]
        assert set(style_schema["enum"]) == {
            "point", "scatter", "box", "bar", "line",
            "trendline", "histogram", "wafer", "2Dmap",
        }

    def test_y_accepts_string_or_array(self):
        schema = _list_tools()["plot_graph"]
        y_schema = schema["properties"]["y"]
        types = {branch.get("type") for branch in y_schema["anyOf"]}
        assert types == {"string", "array"}

    def test_required_fields_are_only_x_y_plot_style(self):
        schema = _list_tools()["plot_graph"]
        assert set(schema["required"]) == {"x", "y", "plot_style"}

    def test_grid_and_plot_title_are_named_top_level_parameters(self):
        """Regression test for the qwen3 run-B bug: `grid` used to only work
        nested inside `other_properties`, an undocumented-in-schema rule a
        weak model got wrong; it's now a real, typed, top-level parameter."""
        schema = _list_tools()["plot_graph"]
        props = schema["properties"]
        for name in ("grid", "plot_title", "xlabel", "ylabel", "zlabel",
                     "xmin", "xmax", "ymin", "ymax", "zmin", "zmax",
                     "color_palette", "xlogscale", "ylogscale"):
            assert name in props, f"{name!r} should be a named top-level parameter"
        assert "description" in props["grid"]

    def test_other_properties_still_present_as_catchall(self):
        schema = _list_tools()["plot_graph"]
        assert "other_properties" in schema["properties"]


class TestUpdateGraphSchema:
    def test_update_graph_has_same_named_parameters_as_plot_graph(self):
        schema = _list_tools()["update_graph"]
        props = schema["properties"]
        for name in ("grid", "plot_title", "color_palette"):
            assert name in props

    def test_update_graph_base_fields_are_all_optional(self):
        schema = _list_tools()["update_graph"]
        assert schema["required"] == ["graph_id"]


class TestDeleteGraphSchema:
    def test_graph_ids_is_nullable_array_not_mismatched(self):
        """Regression test: the original `graph_ids: List[int] = None` type
        hint produced a schema claiming the field must be an array while
        defaulting to null — a structural mismatch."""
        schema = _list_tools()["delete_graph"]
        ids_schema = schema["properties"]["graph_ids"]
        types = {branch.get("type") for branch in ids_schema["anyOf"]}
        assert types == {"array", "null"}
        assert ids_schema["default"] is None


class TestQueryDataframeSchema:
    def test_query_and_df_name_present(self):
        schema = _list_tools()["query_dataframe"]
        assert "query" in schema["properties"]
        assert schema["required"] == ["query"]
