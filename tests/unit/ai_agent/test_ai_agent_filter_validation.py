"""
Tests for filter validation and the other_properties merge precedence in
spectroview/ai_agent/mcp/server.py.

Covers the two concrete qwen3:8b failure shapes from the investigation:
- Run A style: no feedback signal when a filter is malformed (fixed by
  dry-run validation — this file's TestFilterValidation).
- Run B style: `grid` passed as a top-level argument instead of nested in
  `other_properties` was silently dropped (fixed by promoting it to a named
  parameter — this file's TestMergePrecedence, and the regression test
  test_top_level_grid_is_preserved).
"""
import asyncio

import pandas as pd
from mcp import ClientSession
from mcp.client._memory import InMemoryTransport

from spectroview.ai_agent.agent.ports import RecordingContext
from spectroview.ai_agent.mcp.server import create_mcp_server
from spectroview.model.m_graph import MGraph


def _context(graphs=None) -> RecordingContext:
    return RecordingContext(
        dataframes={
            "fit_results": pd.DataFrame({
                "Slot": [1, 2, 3, 5, 6, 7, 8, 10],
                "Zone": ["Edge", "Center", "Edge", "Center", "Edge", "Center", "Edge", "Center"],
                "fwhm_Si": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8],
            })
        },
        active_name="fit_results",
        graphs=graphs or {},
    )


def _call_tool(name, args, graphs=None):
    """Call one tool; return (response text, commands the tool submitted)."""
    async def _run():
        context = _context(graphs)
        server = create_mcp_server(context)
        async with InMemoryTransport(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await session.call_tool(name, args)
                text = res.content[0].text if res.content and hasattr(res.content[0], "text") else str(res)
                return text, context.commands
    return asyncio.run(_run())


class TestFilterValidation:
    def test_unquoted_string_filter_is_rejected_with_actionable_message(self):
        text, pending = _call_tool("plot_graph", {
            "x": "Slot", "y": "fwhm_Si", "plot_style": "point",
            "filters": ["Zone == Edge"],
        })
        assert "quoted" in text.lower()
        assert "NOT created" in text
        assert pending == []

    def test_quoted_string_filter_succeeds(self):
        text, pending = _call_tool("plot_graph", {
            "x": "Slot", "y": "fwhm_Si", "plot_style": "point",
            "filters": ["Zone == 'Edge'"],
        })
        assert "successfully" in text.lower()
        assert [type(c).__name__ for c in pending] == ["CreatePlot"]

    def test_invalid_plot_style_rejected_before_queuing(self):
        text, pending = _call_tool("plot_graph", {
            "x": "Slot", "y": "fwhm_Si", "plot_style": "pie",
        })
        assert pending == []

    def test_multiple_equality_filters_on_same_column_merged_to_in(self):
        """When an LLM passes separate equality filters for multiple values of the
        same column (e.g. ['Slot == 2', 'Slot == 6']), normalize_graph_patch merges
        them into 'Slot in [2, 6]' so conjunctive query evaluation does not produce
        an empty DataFrame."""
        text, pending = _call_tool("plot_graph", {
            "x": "Slot", "y": "fwhm_Si", "plot_style": "point",
            "filters": ["Slot == 2", "Slot == 6", "Slot == 8", "Zone == 'Center'"],
        })
        assert "successfully" in text.lower()
        assert len(pending) == 1
        filters = pending[0].config["filters"]
        assert len(filters) == 2
        assert filters[0]["expression"] == "Slot in [2, 6, 8]"
        assert filters[1]["expression"] == "Zone == 'Center'"

    def test_filters_resulting_in_empty_dataset_rejected_with_actionable_message(self):
        """When filters genuinely match 0 rows in the DataFrame, reject early with
        a descriptive message rather than failing later during plot rendering."""
        text, pending = _call_tool("plot_graph", {
            "x": "Slot", "y": "fwhm_Si", "plot_style": "point",
            "filters": ["Slot > 999"],
        })
        assert "empty dataset (0 matching rows)" in text.lower()
        assert "NOT created" in text
        assert pending == []

    def test_plot_graphs_creates_multiple_plots_in_single_call(self):
        """Verify that plot_graphs validates and creates multiple plots simultaneously in a single call."""
        text, pending = _call_tool("plot_graphs", {
            "plots": [
                {"x": "Slot", "y": "fwhm_Si", "plot_style": "point", "z": "Zone"},
                {"x": "Slot", "y": "fwhm_Si", "plot_style": "box", "filters": ["Slot in [2, 6]"]},
            ]
        })
        assert "validated and queued 2 plot(s)" in text.lower()
        assert len(pending) == 2
        assert pending[0].config["plot_style"] == "point"
        assert pending[0].config["z"] == "Zone"
        assert pending[1].config["plot_style"] == "box"
        assert pending[1].config["filters"][0]["expression"] == "Slot in [2, 6]"


class TestMergePrecedence:
    def test_top_level_grid_is_preserved(self):
        """Direct regression test for the observed qwen3:8b run-B bug: a
        top-level `grid` argument used to be silently dropped because
        `plot_graph`'s signature had no matching parameter."""
        text, pending = _call_tool("plot_graph", {
            "x": "Slot", "y": "fwhm_Si", "plot_style": "point",
            "filters": ["Zone == 'Edge'"], "grid": True,
        })
        assert "successfully" in text.lower()
        assert pending[0].config["grid"] is True

    def test_named_param_wins_over_other_properties_duplicate(self):
        text, pending = _call_tool("plot_graph", {
            "x": "Slot", "y": "fwhm_Si", "plot_style": "point",
            "grid": True, "other_properties": {"grid": False},
        })
        assert pending[0].config["grid"] is True

    def test_other_properties_catchall_key_still_works(self):
        text, pending = _call_tool("plot_graph", {
            "x": "Slot", "y": "fwhm_Si", "plot_style": "point",
            "other_properties": {"x_rot": 45},
        })
        assert pending[0].config["x_rot"] == 45

    def test_advanced_patch_accepts_multiple_customizations_in_one_call(self):
        _, pending = _call_tool("plot_graph", {
            "x": "Slot", "y": "fwhm_Si", "plot_style": "line",
            "other_properties": {
                "title_fontsize": 16,
                "legend_loc": "upper left",
                "tick_direction": "in",
                "figure_margins": [0.1, 0.2],
                "y2": "fwhm_Si",
                "y2color": "purple",
            },
        })
        config = pending[0].config
        assert config["title_fontsize"] == 16
        assert config["legend_loc"] == "upper left"
        assert config["figure_margins"] == [0.1, 0.2]
        assert config["y2"] == "fwhm_Si"

    def test_unknown_advanced_property_is_rejected_by_tool_schema(self):
        text, pending = _call_tool("plot_graph", {
            "x": "Slot", "y": "fwhm_Si", "plot_style": "line",
            "other_properties": {"not_a_graph_property": 1},
        })
        assert "error" in text.lower()
        assert pending == []


class TestUpdateGraphFilterValidation:
    def test_invalid_filter_on_known_graph_is_rejected(self):
        text, pending = _call_tool(
            "update_graph", {"graph_id": "1", "filters": ["Zone == Edge"]},
            graphs={1: {"df": "fit_results"}},
        )
        assert "NOT applied" in text
        assert pending == []

    def test_update_all_with_no_open_graphs_is_rejected(self):
        text, pending = _call_tool("update_graph", {
            "graph_id": "all", "filters": ["Zone == 'Edge'"],
        })
        assert "no graphs" in text.lower()
        assert pending == []

    def test_update_all_validates_and_queues_one_atomic_command(self):
        graph1 = MGraph(graph_id=1, df_name="fit_results", x="Slot", y=["fwhm_Si"])
        graph2 = MGraph(graph_id=2, df_name="fit_results", x="Slot", y=["fwhm_Si"])
        text, pending = _call_tool(
            "update_graph",
            {
                "graph_id": "all",
                "xlogscale": True,
                "other_properties": {"title_fontsize": 16, "legend_loc": "upper left"},
            },
            graphs={1: graph1.save(), 2: graph2.save()},
        )
        assert "successfully" in text.lower()
        assert len(pending) == 1
        assert pending[0].graph_id == "all"
        assert pending[0].properties == {
            "title_fontsize": 16,
            "legend_loc": "upper left",
            "xlogscale": True,
        }
