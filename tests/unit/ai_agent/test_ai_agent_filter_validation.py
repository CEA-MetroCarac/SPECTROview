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
from mcp.shared.memory import create_connected_server_and_client_session

from spectroview.ai_agent.agent.ports import RecordingContext
from spectroview.ai_agent.mcp.server import create_mcp_server


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
        async with create_connected_server_and_client_session(server._mcp_server) as session:
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


class TestUpdateGraphFilterValidation:
    def test_invalid_filter_on_known_graph_is_rejected(self):
        text, pending = _call_tool(
            "update_graph", {"graph_id": "1", "filters": ["Zone == Edge"]},
            graphs={1: {"df": "fit_results"}},
        )
        assert "NOT applied" in text
        assert pending == []

    def test_update_all_skips_dry_run_validation(self):
        """graph_id='all' could span multiple DataFrames — validating
        against one would mislead, so it's intentionally skipped rather
        than silently guessing which DataFrame to check against."""
        text, pending = _call_tool("update_graph", {
            "graph_id": "all", "filters": ["Zone == 'Edge'"],
        })
        assert "successfully" in text.lower()
        assert pending[0].graph_id == "all"
