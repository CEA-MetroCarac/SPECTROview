"""
Tests for the opt-in "show reasoning" feature (spectroview/ai_agent/vm_chat.py,
m_llm_client.py).

Covers the confirmed design: off by default (preserving the small-model
think=False reliability fix), explicit opt-in overrides it when the user
asks, and the underlying Ollama `thinking` stream is kept on a separate
signal from the visible answer channel — never merged in, since merging
is exactly how the original qwen3 "narrates instead of acting" failure
happened.
"""
from unittest.mock import patch

import pandas as pd
import pytest

import spectroview.ai_agent.m_llm_client as m
from spectroview.ai_agent.vm_chat import VMChat


@pytest.fixture
def vm(qapp, monkeypatch):
    monkeypatch.setattr("spectroview.ai_agent.vm_chat.get_ollama_model_info", lambda model: None)
    v = VMChat()
    v.set_dataframes({"df": pd.DataFrame({"A": [1, 2]})}, "df")
    v.set_provider("Ollama")
    return v


class TestShowReasoningDefaultsAndOverride:
    def test_off_by_default(self, vm):
        assert vm.is_show_reasoning() is False

    def test_small_model_default_think_false_preserved_when_reasoning_off(self, vm):
        vm.set_small_model_mode(True)
        opts = vm._build_request_options()
        assert opts["think"] is False

    def test_explicit_opt_in_overrides_small_model_default(self, vm):
        vm.set_small_model_mode(True)
        vm.set_show_reasoning(True)
        opts = vm._build_request_options()
        assert opts["think"] is True

    def test_full_tier_has_no_think_key_when_reasoning_off(self, vm):
        vm.set_small_model_mode(False)
        opts = vm._build_request_options()
        assert "think" not in opts

    def test_full_tier_think_true_when_reasoning_on(self, vm):
        vm.set_small_model_mode(False)
        vm.set_show_reasoning(True)
        opts = vm._build_request_options()
        assert opts["think"] is True

    def test_toggling_off_again_restores_default(self, vm):
        vm.set_small_model_mode(True)
        vm.set_show_reasoning(True)
        vm.set_show_reasoning(False)
        opts = vm._build_request_options()
        assert opts["think"] is False


class TestThinkingChannelNeverMergedIntoAnswer:
    """Worker-level regression test: message["thinking"] must never reach
    chunk_received/response_ready — only thinking_chunk_received."""

    def test_thinking_and_content_stay_on_separate_signals(self):
        def fake_chat(**kwargs):
            return [
                {"message": {"thinking": "Let me plan this out..."}},
                {"message": {"thinking": " I'll call plot_graph."}},
                {"message": {"content": "Done!"}},
            ]

        with patch.object(m, "_ollama") as fake_ollama:
            fake_ollama.chat.side_effect = fake_chat
            m.OLLAMA_AVAILABLE = True

            thinking_fragments = []
            content_fragments = []
            worker = m.LLMWorker(model="qwen3:8b", messages=[], tools=None, think=True)
            worker.thinking_chunk_received.connect(thinking_fragments.append)
            worker.chunk_received.connect(content_fragments.append)
            worker.run()

        assert thinking_fragments == ["Let me plan this out...", " I'll call plot_graph."]
        assert content_fragments == ["Done!"]

    def test_no_thinking_field_produces_no_thinking_signal(self):
        def fake_chat(**kwargs):
            return [{"message": {"content": "Plain answer."}}]

        with patch.object(m, "_ollama") as fake_ollama:
            fake_ollama.chat.side_effect = fake_chat
            m.OLLAMA_AVAILABLE = True

            thinking_fragments = []
            worker = m.LLMWorker(model="qwen2.5-coder:7b", messages=[], tools=None)
            worker.thinking_chunk_received.connect(thinking_fragments.append)
            worker.run()

        assert thinking_fragments == []


class TestLLMClientThinkingCallbackWiring:
    def test_on_thinking_chunk_connected_for_ollama(self):
        with patch.object(m, "LLMWorker") as FakeWorker:
            from unittest.mock import MagicMock
            FakeWorker.return_value = MagicMock()
            client = m.LLMClient()
            client.set_provider("Ollama")
            received = []
            client.chat(
                model="qwen3:8b",
                messages=[],
                on_chunk=lambda *_: None,
                on_done=lambda *_: None,
                on_error=lambda *_: None,
                on_thinking_chunk=received.append,
            )
            client._worker.thinking_chunk_received.connect.assert_called_once_with(received.append)

    def test_omitting_on_thinking_chunk_is_safe(self):
        """Callers that don't care about reasoning (e.g. not yet updated
        code) must not break — the parameter is optional."""
        with patch.object(m, "LLMWorker") as FakeWorker:
            from unittest.mock import MagicMock
            FakeWorker.return_value = MagicMock()
            client = m.LLMClient()
            client.set_provider("Ollama")
            client.chat(
                model="qwen3:8b",
                messages=[],
                on_chunk=lambda *_: None,
                on_done=lambda *_: None,
                on_error=lambda *_: None,
            )
            assert client._worker is not None
