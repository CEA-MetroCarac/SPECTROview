"""
Tests for spectroview/ai_agent/m_llm_client.py — request-option plumbing.

Verifies the cross-provider isolation guarantee: `num_ctx`/`think` (Ollama
`options`, meaningless to any other backend) must reach the Ollama worker
and must NEVER be forwarded to the OpenAI-compatible or Anthropic workers,
so wiring up small-model request tuning cannot affect the DeepSeek/cloud
path this project relies on working correctly.

IMPORTANT: `LLMClient.chat()` calls `worker.start()` internally, spawning a
real QThread. These tests must never call `LLMClient.chat()` when a real
worker class is in play without also stubbing that worker class out —
otherwise a real background thread starts and races the test (and can
crash the interpreter doing real network/SSL setup after a mock context
has already exited). Worker-level behavior (do `options`/`think`/`timeout`
reach `ollama.chat()` correctly?) is tested by constructing the worker
directly and calling `.run()` synchronously — never via `LLMClient`.
Client-level behavior (does `LLMClient.chat()` route request_options to
the right worker class?) is tested by stubbing out the worker classes
themselves, so `.start()` becomes a harmless mock call.
"""
import json
from unittest.mock import MagicMock, patch

import spectroview.ai_agent.m_llm_client as m


class _FakeStream:
    def __iter__(self):
        return iter([])


class _FakePydanticToolCall:
    """Mimics ollama._types.Message.ToolCall: a pydantic-model-shaped tool
    call (dict-subscriptable but NOT directly json.dumps()-able), not a
    plain dict."""

    def __init__(self, name, arguments):
        self._data = {"function": {"name": name, "arguments": arguments}}

    def get(self, key, default=None):
        return self._data.get(key, default)

    def model_dump(self):
        return dict(self._data)


class TestOllamaToolCallsAreJsonSerializable:
    """Regression test: ollama's tool_calls arrive as pydantic model objects,
    not plain dicts. json.dump()-ing them directly (as MConversation.save()
    does) raises `Object of type ToolCall is not JSON serializable` —
    discovered via live verification against real qwen3:8b, where every
    successful tool-calling turn failed to persist to conversation history."""

    def test_tool_calls_are_normalized_to_plain_dicts(self):
        fake_tc = _FakePydanticToolCall("plot_graph", {"x": "Slot", "grid": True})

        def fake_chat(**kwargs):
            return [{"message": {"tool_calls": [fake_tc]}}]

        with patch.object(m, "_ollama") as fake_ollama:
            fake_ollama.chat.side_effect = fake_chat
            m.OLLAMA_AVAILABLE = True

            captured = {}
            worker = m.LLMWorker(model="qwen3:8b", messages=[], tools=None)
            worker.response_ready.connect(lambda text, calls: captured.update(tool_calls=calls))
            worker.run()

        tool_calls = captured["tool_calls"]
        assert len(tool_calls) == 1
        # Must not raise — this is the exact failure mode observed live.
        json.dumps(tool_calls)
        assert tool_calls[0] == {"function": {"name": "plot_graph", "arguments": {"x": "Slot", "grid": True}}}

    def test_plain_dict_tool_calls_pass_through_unchanged(self):
        """If a future ollama version ever returns plain dicts directly,
        the normalization must be a no-op, not an error."""
        plain_tc = {"function": {"name": "plot_graph", "arguments": {"x": "Slot"}}}

        def fake_chat(**kwargs):
            return [{"message": {"tool_calls": [plain_tc]}}]

        with patch.object(m, "_ollama") as fake_ollama:
            fake_ollama.chat.side_effect = fake_chat
            m.OLLAMA_AVAILABLE = True

            captured = {}
            worker = m.LLMWorker(model="qwen3:8b", messages=[], tools=None)
            worker.response_ready.connect(lambda text, calls: captured.update(tool_calls=calls))
            worker.run()

        assert captured["tool_calls"] == [plain_tc]


class TestOllamaWorkerAppliesOptions:
    """Constructs LLMWorker directly and calls .run() synchronously —
    no LLMClient, no .start(), no real thread."""

    def test_num_ctx_and_think_forwarded_to_ollama_chat(self):
        captured = {}

        def fake_chat(**kwargs):
            captured.update(kwargs)
            return _FakeStream()

        with patch.object(m, "_ollama") as fake_ollama:
            fake_ollama.chat.side_effect = fake_chat
            m.OLLAMA_AVAILABLE = True

            worker = m.LLMWorker(
                model="qwen3:8b",
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
                options={"num_ctx": 8192, "num_predict": 4096},
                think=False,
            )
            worker.run()

        assert captured["options"] == {"num_ctx": 8192, "num_predict": 4096}
        assert captured["think"] is False

    def test_no_timeout_uses_module_level_chat(self):
        """When timeout is None, the module-level ollama.chat() convenience
        function is used directly rather than constructing a Client."""
        with patch.object(m, "_ollama") as fake_ollama:
            fake_ollama.chat.return_value = _FakeStream()
            m.OLLAMA_AVAILABLE = True

            worker = m.LLMWorker(model="qwen3:8b", messages=[], tools=None)
            worker.run()

            fake_ollama.chat.assert_called_once()
            fake_ollama.Client.assert_not_called()

    def test_timeout_constructs_a_client_with_timeout(self):
        with patch.object(m, "_ollama") as fake_ollama:
            fake_ollama.Client.return_value.chat.return_value = _FakeStream()
            m.OLLAMA_AVAILABLE = True

            worker = m.LLMWorker(model="qwen3:8b", messages=[], tools=None, timeout=120.0)
            worker.run()

            fake_ollama.Client.assert_called_once_with(timeout=120.0)


class TestLLMClientRoutesRequestOptionsPerProvider:
    """Stubs out the worker *classes* so LLMClient.chat()'s dispatch logic
    can be inspected without ever starting a real thread."""

    def test_ollama_provider_constructs_worker_with_translated_options(self):
        with patch.object(m, "LLMWorker") as FakeWorker:
            FakeWorker.return_value = MagicMock()
            client = m.LLMClient()
            client.set_provider("Ollama")
            client.chat(
                model="qwen3:8b",
                messages=[],
                on_chunk=lambda *_: None,
                on_done=lambda *_: None,
                on_error=lambda *_: None,
                request_options={"num_ctx": 8192, "max_tokens": 4096, "think": False, "timeout": 120.0},
            )
            _, kwargs = FakeWorker.call_args
            assert kwargs["options"] == {"num_ctx": 8192, "num_predict": 4096}
            assert kwargs["think"] is False
            assert kwargs["timeout"] == 120.0

    def test_openai_compatible_worker_never_receives_num_ctx_or_think(self):
        with patch.object(m, "APIWorker") as FakeWorker:
            FakeWorker.return_value = MagicMock()
            client = m.LLMClient()
            client.set_provider("DeepSeek", api_key="fake-key")
            client.chat(
                model="deepseek-chat",
                messages=[],
                on_chunk=lambda *_: None,
                on_done=lambda *_: None,
                on_error=lambda *_: None,
                request_options={"num_ctx": 8192, "think": False, "timeout": 120.0, "max_tokens": 4096},
            )
            _, kwargs = FakeWorker.call_args
            assert "num_ctx" not in kwargs
            assert "think" not in kwargs
            assert "options" not in kwargs
            assert kwargs["timeout"] == 120.0
            assert kwargs["max_tokens"] == 4096

    def test_anthropic_worker_never_receives_num_ctx_or_think(self):
        with patch.object(m, "AnthropicWorker") as FakeWorker:
            FakeWorker.return_value = MagicMock()
            client = m.LLMClient()
            client.set_provider("Anthropic", api_key="fake-key")
            client.chat(
                model="claude-3-5-sonnet-20241022",
                messages=[],
                on_chunk=lambda *_: None,
                on_done=lambda *_: None,
                on_error=lambda *_: None,
                request_options={"num_ctx": 8192, "think": False, "timeout": 120.0, "max_tokens": 4096},
            )
            _, kwargs = FakeWorker.call_args
            assert "num_ctx" not in kwargs
            assert "think" not in kwargs
            assert "options" not in kwargs
            assert kwargs["timeout"] == 120.0
            assert kwargs["max_tokens"] == 4096

    def test_no_request_options_is_a_safe_no_op(self):
        """Calling chat() without request_options (e.g. any caller not yet
        updated) must not raise — every new kwarg is optional."""
        with patch.object(m, "APIWorker") as FakeWorker:
            FakeWorker.return_value = MagicMock()
            client = m.LLMClient()
            client.set_provider("DeepSeek", api_key="fake-key")
            client.chat(
                model="deepseek-chat",
                messages=[],
                on_chunk=lambda *_: None,
                on_done=lambda *_: None,
                on_error=lambda *_: None,
            )
            assert client._worker is not None
