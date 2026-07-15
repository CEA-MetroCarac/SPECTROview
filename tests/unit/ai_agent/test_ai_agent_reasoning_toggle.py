"""
Tests for the reasoning ("thinking") channel-separation infrastructure in
spectroview/ai_agent/m_llm_client.py.

The opt-in "Show model reasoning" UI toggle (VMChat.set_show_reasoning /
is_show_reasoning, and VChatPanel's header button) has been removed
entirely — the app now never explicitly requests `think=True`, restoring
the pre-toggle default (full-tier: no `think` key set at all; small-tier:
`think=False`, preserving the small-model reliability fix).

What remains, and is still tested here, is a correctness guarantee at the
`m_llm_client.py` worker level: even if a model spontaneously emits
`message["thinking"]` content despite not being asked to, it must stay on
its own signal (`thinking_chunk_received`) and never merge into the
visible-answer channel (`chunk_received`) — merging is exactly how the
original qwen3 "narrates instead of acting" failure happened.
`VMChat._on_thinking_chunk` stays wired as the `on_thinking_chunk`
callback for exactly this reason, even though it now just discards the
fragment instead of surfacing it in the UI.
"""
from unittest.mock import patch

import spectroview.ai_agent.m_llm_client as m


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
