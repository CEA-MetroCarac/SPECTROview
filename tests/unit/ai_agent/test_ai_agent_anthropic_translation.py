"""
Tests for the OpenAI -> Anthropic protocol translation in m_llm_client.py.

The bug under test (B2): the app speaks OpenAI's tool dialect everywhere —
``{"type": "function", "function": {...}}`` schemas, an assistant message with
``tool_calls``, and a separate ``{"role": "tool"}`` result. Those were being
handed to Anthropic verbatim, which takes ``input_schema`` tools and carries
calls/results as ``tool_use``/``tool_result`` content blocks. Every Anthropic
run that reached a tool call therefore failed at the API.

No API key or network is needed: the translation is pure functions.
"""
import spectroview.ai_agent.m_llm_client as m


TOOLS = [{
    "type": "function",
    "function": {
        "name": "plot_graph",
        "description": "Create a new graph.",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
    },
}]


class TestToolSchemaTranslation:
    def test_none_and_empty_stay_none(self):
        assert m.anthropic_tools(None) is None
        assert m.anthropic_tools([]) is None

    def test_function_wrapper_is_unwrapped_to_anthropic_shape(self):
        [tool] = m.anthropic_tools(TOOLS)
        assert set(tool) == {"name", "description", "input_schema"}
        assert tool["name"] == "plot_graph"
        assert tool["description"] == "Create a new graph."

    def test_parameters_become_input_schema_unchanged(self):
        [tool] = m.anthropic_tools(TOOLS)
        assert tool["input_schema"] == TOOLS[0]["function"]["parameters"]

    def test_missing_parameters_get_an_empty_object_schema(self):
        [tool] = m.anthropic_tools([{"function": {"name": "ping"}}])
        assert tool["input_schema"] == {"type": "object", "properties": {}}
        assert tool["description"] == ""


class TestMessageTranslation:
    def test_system_messages_are_lifted_out(self):
        system, msgs = m.anthropic_messages([
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "hello"},
        ])
        assert system == "You are an agent."
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_multiple_system_messages_are_joined(self):
        system, _ = m.anthropic_messages([
            {"role": "system", "content": "one"},
            {"role": "system", "content": "two"},
        ])
        assert system == "one\ntwo"

    def test_tool_calls_become_tool_use_blocks(self):
        _, msgs = m.anthropic_messages([
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "function": {"name": "plot_graph",
                                          "arguments": '{"x": "Slot"}'}}]},
        ])
        [block] = msgs[0]["content"]
        assert block == {"type": "tool_use", "id": "c1",
                         "name": "plot_graph", "input": {"x": "Slot"}}

    def test_assistant_text_is_kept_alongside_its_tool_use(self):
        _, msgs = m.anthropic_messages([
            {"role": "assistant", "content": "Plotting now.", "tool_calls": [
                {"id": "c1", "function": {"name": "plot_graph", "arguments": "{}"}}]},
        ])
        types = [b["type"] for b in msgs[0]["content"]]
        assert types == ["text", "tool_use"]

    def test_dict_arguments_are_accepted_as_well_as_json_strings(self):
        _, msgs = m.anthropic_messages([
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "function": {"name": "f", "arguments": {"a": 1}}}]},
        ])
        assert msgs[0]["content"][0]["input"] == {"a": 1}

    def test_malformed_arguments_degrade_to_empty_input(self):
        _, msgs = m.anthropic_messages([
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "function": {"name": "f", "arguments": "{not json"}}]},
        ])
        assert msgs[0]["content"][0]["input"] == {}

    def test_tool_result_becomes_a_user_message_block(self):
        _, msgs = m.anthropic_messages([
            {"role": "tool", "content": "ok", "tool_call_id": "c1"},
        ])
        assert msgs == [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "ok"}]}]

    def test_parallel_tool_results_merge_into_one_user_message(self):
        """Anthropic requires every result answering one assistant turn to
        arrive in a single user message."""
        _, msgs = m.anthropic_messages([
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "function": {"name": "f", "arguments": "{}"}},
                {"id": "c2", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "content": "first", "tool_call_id": "c1"},
            {"role": "tool", "content": "second", "tool_call_id": "c2"},
        ])
        assert len(msgs) == 2
        assert [b["tool_use_id"] for b in msgs[1]["content"]] == ["c1", "c2"]

    def test_empty_messages_are_dropped(self):
        """Anthropic rejects a message whose content is an empty string."""
        _, msgs = m.anthropic_messages([
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "hi"},
        ])
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_full_turn_round_trips_in_order(self):
        _, msgs = m.anthropic_messages([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "plot it"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "function": {"name": "plot_graph", "arguments": "{}"}}]},
            {"role": "tool", "content": "done", "tool_call_id": "c1"},
            {"role": "assistant", "content": "Created."},
        ])
        assert [msg["role"] for msg in msgs] == [
            "user", "assistant", "user", "assistant",
        ]
