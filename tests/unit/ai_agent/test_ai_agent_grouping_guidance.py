"""
Tests that the agent is actually taught "group by <column>" -> z, not x.

The observed failure: for "Plot 1: group the data by Zone, exclude slots
10-12, ..." on a graph whose X axis was Slot, the model called
``update_graph(graph_id="1", x="Zone", y="fwhm_Si", ...)`` — replacing the
slot-by-slot view instead of colouring it by Zone.

Nothing in the prompt or the tool schema mapped the word "group" onto ``z``:
``z`` was only ever described as the metric value for wafer/2Dmap, and hue was
mentioned solely in the small tier's examples. These tests pin the guidance
into both tiers and into the schema the model is actually constrained by, so a
future prompt edit cannot quietly drop it again.
"""
import asyncio

import pandas as pd
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from spectroview.ai_agent.agent.ports import RecordingContext
from spectroview.ai_agent.mcp.server import create_mcp_server
from spectroview.ai_agent.vm_chat import VMChat


def _schemas():
    async def _run():
        server = create_mcp_server(RecordingContext())
        async with create_connected_server_and_client_session(server._mcp_server) as session:
            await session.initialize()
            return {t.name: t.inputSchema for t in (await session.list_tools()).tools}
    return asyncio.run(_run())


class TestToolSchemaTeachesGrouping:
    """The parameter descriptions are the only guidance every provider sees,
    however the prompt tier is configured."""

    @pytest.mark.parametrize("tool", ["plot_graph", "update_graph"])
    def test_z_description_names_grouping(self, tool):
        desc = _schemas()[tool]["properties"]["z"]["description"].lower()
        assert "group" in desc
        assert "hue" in desc or "colour" in desc or "color" in desc

    @pytest.mark.parametrize("tool", ["plot_graph", "update_graph"])
    def test_z_description_warns_against_using_x(self, tool):
        """The failure mode is specifically z-vs-x, so the description has to
        contrast them rather than just describe z."""
        desc = _schemas()[tool]["properties"]["z"]["description"].lower()
        assert "x" in desc
        assert "not" in desc or "never" in desc

    def test_update_graph_x_is_documented_as_omittable(self):
        desc = _schemas()["update_graph"]["properties"]["x"]["description"].lower()
        assert "omit" in desc


@pytest.fixture
def vm(qapp):
    v = VMChat()
    v.set_dataframes(
        {"fit_results": pd.DataFrame({
            "Slot": [1, 2], "Zone": ["Edge", "Center"], "fwhm_Si": [1.0, 2.0],
        })},
        "fit_results",
    )
    yield v
    v.shutdown()


def _full(vm, monkeypatch):
    monkeypatch.setattr("spectroview.ai_agent.vm_chat.get_ollama_model_info", lambda m: None)
    vm.set_provider("Ollama")
    vm.set_small_model_mode(False)
    return vm._build_system_prompt().lower()


def _small(vm):
    vm.set_provider("Ollama")
    vm.set_small_model_mode(True)
    return vm._build_system_prompt().lower()


class TestBothPromptTiersTeachGrouping:
    def test_full_tier_maps_group_by_to_z(self, vm, monkeypatch):
        prompt = _full(vm, monkeypatch)
        assert "group by" in prompt
        assert "z" in prompt

    def test_small_tier_maps_group_by_to_z(self, vm):
        prompt = _small(vm)
        assert "group by" in prompt

    def test_full_tier_warns_not_to_use_x_for_grouping(self, vm, monkeypatch):
        """Showing the right answer is not enough — the wrong one has to be
        named, because putting the grouping column on x is the mistake the
        model actually made."""
        prompt = _full(vm, monkeypatch)
        assert 'x="zone"' in prompt
        assert "wrong" in prompt

    def test_small_tier_warns_not_to_use_x_for_grouping(self, vm):
        assert "not `x=" in _small(vm)

    def test_full_tier_says_update_graph_keeps_omitted_values(self, vm, monkeypatch):
        prompt = _full(vm, monkeypatch)
        assert "omit" in prompt
        assert "only the properties the user asked to change" in prompt

    def test_small_tier_says_update_graph_keeps_omitted_values(self, vm):
        prompt = _small(vm)
        assert "omit" in prompt or "keeps" in prompt
