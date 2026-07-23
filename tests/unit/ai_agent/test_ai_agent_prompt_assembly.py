"""
Tests for the AI Agent's assembled system prompts (both tiers).

The core bug under test: the full-tier prompt used to instruct the model
two contradictory ways simultaneously — prompts/system.md said "always use
tool calls, never output JSON", while rules/general.md said "return ONLY a
JSON object with an `action` field" (a protocol the tool-calling code no
longer parses at all). This is a strong suspected cause of the qwen3:8b
run-A failure (a well-formed prose "plan" that never became a real tool
call). These tests assert the contradiction is gone and the small tier is
genuinely, measurably smaller — not just qualitatively "shorter".
"""
import pandas as pd
import pytest

from spectroview.ai_agent.vm_chat import VMChat


@pytest.fixture
def vm(qapp):
    v = VMChat()
    v.set_dataframes({"fit_results": pd.DataFrame({"Slot": [1, 2], "Zone": ["A", "B"]})}, "fit_results")
    return v


def _full_tier_prompt(vm, monkeypatch):
    monkeypatch.setattr("spectroview.ai_agent.vm_chat.get_ollama_model_info", lambda model: None)
    vm.set_provider("Ollama")
    vm.set_small_model_mode(False)
    return vm._build_system_prompt()


def _small_tier_prompt(vm):
    vm.set_provider("Ollama")
    vm.set_small_model_mode(True)
    return vm._build_system_prompt()


class TestFullTierContradictionRemoved:
    def test_no_longer_instructs_raw_json_action_protocol(self, vm, monkeypatch):
        prompt = _full_tier_prompt(vm, monkeypatch)
        assert "Return ONLY the JSON object" not in prompt
        # `system.md` intentionally still shows `{"action": "plot", ...}` as a
        # negative example ("never output this") — that one is correct and
        # must survive. What must NOT survive is the old positive instruction
        # from rules/general.md telling the model to actually set this field.
        assert "Always set `action`" not in prompt
        assert "seven valid values" not in prompt
        assert "JSON protocol" not in prompt

    def test_no_longer_references_dead_target_dataframe_field(self, vm, monkeypatch):
        prompt = _full_tier_prompt(vm, monkeypatch)
        assert "target_dataframe" not in prompt

    def test_still_instructs_native_tool_calling(self, vm, monkeypatch):
        """The correct instruction must survive the cleanup."""
        prompt = _full_tier_prompt(vm, monkeypatch)
        assert "MUST call the provided tools" in prompt
        assert "NEVER output JSON code blocks" in prompt


class TestSmallTierIsGenuinelySmaller:
    def test_small_tier_is_measurably_shorter(self, vm, monkeypatch):
        full = _full_tier_prompt(vm, monkeypatch)
        small = _small_tier_prompt(vm)
        assert len(small) < len(full) * 0.6

    def test_small_tier_still_instructs_tool_calling_only(self, vm):
        small = _small_tier_prompt(vm)
        assert "MUST call the provided tools" in small
        assert "do not narrate" in small.lower() or "never describe" in small.lower()

    def test_small_tier_warns_about_filter_quoting(self, vm):
        small = _small_tier_prompt(vm)
        assert "quoted" in small.lower()

    def test_small_tier_has_no_dataframe_placeholder_left_unfilled(self, vm):
        small = _small_tier_prompt(vm)
        assert "{dataframes_section}" not in small
        assert "{active_df_info}" not in small
        assert "{graphs_info}" not in small
