"""
Tests for MConversation -> LLM message conversion.

The bug under test (B1): capping the context window with a plain
``messages[-N:]`` slice can cut between an assistant message carrying
``tool_calls`` and the ``role="tool"`` message answering it. Providers
reject that orphan pair. The cap is only active in small-model mode
(``max_context_messages_small: 6``), so it hit exactly the local models
least able to recover from an API error.
"""
from spectroview.ai_agent.m_conversation import MConversation


def _conversation_with_tool_turn() -> MConversation:
    """user, assistant(tool_calls), tool, assistant, user, assistant, user."""
    conv = MConversation()
    conv.add_message("user", "plot it")
    conv.add_message("assistant", "", tool_calls=[
        {"id": "c1", "type": "function",
         "function": {"name": "plot_graph", "arguments": "{}"}}])
    conv.add_message("tool", "Plot command sent to UI successfully.",
                     is_hidden=True, tool_call_id="c1")
    conv.add_message("assistant", "Done.")
    conv.add_message("user", "and another")
    conv.add_message("assistant", "Sure.")
    conv.add_message("user", "thanks")
    return conv


class TestContextWindow:
    def test_no_cap_returns_every_message(self):
        conv = _conversation_with_tool_turn()
        assert len(conv.to_llm_messages(None)) == 7

    def test_window_never_starts_on_an_orphan_tool_result(self):
        conv = _conversation_with_tool_turn()
        # A raw [-5:] slice would start at the "tool" message, orphaning it.
        msgs = conv.to_llm_messages(5)
        assert msgs[0]["role"] != "tool"

    def test_widened_window_keeps_the_parent_tool_calls_message(self):
        conv = _conversation_with_tool_turn()
        msgs = conv.to_llm_messages(5)
        first = msgs[0]
        assert first["role"] == "assistant"
        assert first["tool_calls"][0]["id"] == "c1"
        # ...and the tool result it belongs to is still right behind it.
        assert msgs[1]["role"] == "tool"
        assert msgs[1]["tool_call_id"] == "c1"

    def test_every_tool_message_has_a_preceding_tool_calls_message(self):
        conv = _conversation_with_tool_turn()
        for cap in range(1, 9):
            msgs = conv.to_llm_messages(cap)
            for i, msg in enumerate(msgs):
                if msg["role"] == "tool":
                    assert i > 0, f"orphan tool message at cap={cap}"
                    assert "tool_calls" in msgs[i - 1] or msgs[i - 1]["role"] == "tool", (
                        f"tool message not preceded by its tool_calls at cap={cap}"
                    )

    def test_cap_still_limits_ordinary_history(self):
        conv = MConversation()
        for i in range(10):
            conv.add_message("user" if i % 2 == 0 else "assistant", f"m{i}")
        msgs = conv.to_llm_messages(4)
        assert len(msgs) == 4
        assert msgs[-1]["content"] == "m9"


class TestToolCallMessageShape:
    """Gemini's OpenAI-compatibility endpoint 400s on an assistant message
    that carries tool_calls *and* an empty-string content. OpenAI and DeepSeek
    accept it, so the failure looked provider-specific and only appeared from
    the second turn on — the first request has no assistant message yet.
    """

    def _tool_turn(self, text=""):
        conv = MConversation()
        conv.add_message("user", "plot it")
        conv.add_message("assistant", text, tool_calls=[
            {"id": "c1", "type": "function",
             "function": {"name": "plot_graph", "arguments": "{}"}}])
        conv.add_message("tool", "ok", is_hidden=True, tool_call_id="c1")
        return conv.to_llm_messages(None)

    def test_empty_content_is_omitted_not_sent_as_blank(self):
        assistant = self._tool_turn()[1]
        assert "content" not in assistant
        assert assistant["tool_calls"][0]["id"] == "c1"

    def test_real_content_alongside_tool_calls_is_kept(self):
        assistant = self._tool_turn("Plotting that now.")[1]
        assert assistant["content"] == "Plotting that now."

    def test_ordinary_assistant_message_still_carries_content(self):
        conv = MConversation()
        conv.add_message("assistant", "Here you go.")
        assert conv.to_llm_messages(None)[0]["content"] == "Here you go."

    def test_tool_result_keeps_its_content_and_id(self):
        tool_msg = self._tool_turn()[2]
        assert tool_msg["content"] == "ok"
        assert tool_msg["tool_call_id"] == "c1"


class TestReplyContext:
    def test_reply_prefix_is_injected(self):
        conv = MConversation()
        conv.add_message("user", "hello")
        conv.add_message("assistant", "hi there")
        conv.add_message("user", "explain", reply_to_index=1)
        msgs = conv.to_llm_messages(None)
        assert msgs[-1]["content"].startswith('[Replying to AI message: "hi there"]')

    def test_error_messages_are_excluded(self):
        conv = MConversation()
        conv.add_message("user", "hello")
        conv.add_message("error", "Ollama not reachable")
        assert [m["role"] for m in conv.to_llm_messages(None)] == ["user"]
