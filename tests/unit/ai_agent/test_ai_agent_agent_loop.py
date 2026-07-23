"""
Characterization tests for the AI Agent's tool-calling loop.

These pin down the *observable behaviour* of one full user turn — how many
LLM round trips happen, which messages land in the conversation, when tool
results are fed back, and what the View finally receives — so the loop can
be extracted out of ``VMChat`` (and later driven by a provider adapter)
without silently changing what the user sees.

The LLM is replaced by ``FakeLLMClient``, a scripted stand-in that invokes
``LLMClient.chat()``'s callbacks synchronously. Synchronous callbacks mean
the recursion ``_on_done -> chat() -> _on_done`` unrolls inside the test,
so no Qt event loop spinning or waiting is needed.
"""
import pandas as pd
import pytest

from spectroview.ai_agent.vm_chat import VMChat


# ═══════════════════════════════════════════════════════════════════════════
# Scripted LLM stand-in
# ═══════════════════════════════════════════════════════════════════════════

def tool_call(name: str, arguments: dict | str, call_id: str = "call_1") -> dict:
    """Build one OpenAI-shaped tool call as the workers emit them."""
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


class FakeLLMClient:
    """Replays a scripted list of ``(text, tool_calls)`` responses.

    Mirrors only the ``LLMClient`` surface ``VMChat`` actually uses. Each
    ``chat()`` call consumes one script entry and invokes the callbacks
    immediately; running out of entries is an error, since it means the loop
    made more round trips than the test expected.
    """

    def __init__(self, script: list[tuple[str, list]]) -> None:
        self._script = list(script)
        self.calls: list[dict] = []          # one entry per chat() invocation
        self.cancelled = 0

    # -- surface used by VMChat -------------------------------------------
    def is_available(self) -> bool:
        return True

    def is_busy(self) -> bool:
        return False

    def get_models(self) -> list[str]:
        return ["fake-model"]

    def set_provider(self, provider, api_key="", base_url="", model="") -> None:
        pass

    def cancel(self) -> None:
        self.cancelled += 1

    def chat(self, model, messages, on_chunk, on_done, on_error,
             tools=None, parent=None, request_options=None,
             on_thinking_chunk=None) -> None:
        self.calls.append({"model": model, "messages": messages, "tools": tools})
        if not self._script:
            raise AssertionError(
                f"FakeLLMClient script exhausted after {len(self.calls)} call(s); "
                f"the loop made more round trips than the test scripted."
            )
        text, tool_calls = self._script.pop(0)
        if text:
            on_chunk(text)
        on_done(text, tool_calls)


PLOT_ARGS = {"x": "Slot", "y": "fwhm_Si", "plot_style": "box"}


@pytest.fixture
def vm(qapp):
    v = VMChat()
    v.set_dataframes(
        {"fit_results": pd.DataFrame({
            "Slot": [1, 2, 3],
            "Zone": ["Edge", "Center", "Edge"],
            "fwhm_Si": [1.0, 2.0, 3.0],
        })},
        "fit_results",
    )
    yield v
    # The MCP hub owns a thread and live sessions; without this they pile up
    # across the suite.
    v.shutdown()


def drive(vm, script, user_text="plot it"):
    """Run one user turn against *script*; return (fake_client, results, errors)."""
    fake = FakeLLMClient(script)
    vm._client = fake
    results, errors = [], []
    vm.result_ready.connect(results.append)
    vm.error_occurred.connect(errors.append)
    vm.process_query(user_text)
    return fake, results, errors


# ═══════════════════════════════════════════════════════════════════════════
# Happy paths
# ═══════════════════════════════════════════════════════════════════════════

class TestSingleToolCall:
    def test_plot_tool_produces_one_plot_result(self, vm):
        _, results, errors = drive(vm, [
            ("", [tool_call("plot_graph", PLOT_ARGS)]),
            ("Created the box plot.", []),
        ])
        assert not errors
        assert len(results) == 1
        assert results[0].action == "plot"
        assert len(results[0].plot_config) == 1
        assert results[0].plot_config[0]["plot_style"] == "box"

    def test_loop_makes_exactly_two_round_trips(self, vm):
        fake, _, _ = drive(vm, [
            ("", [tool_call("plot_graph", PLOT_ARGS)]),
            ("Done.", []),
        ])
        assert len(fake.calls) == 2

    def test_tool_result_is_fed_back_to_the_model(self, vm):
        fake, _, _ = drive(vm, [
            ("", [tool_call("plot_graph", PLOT_ARGS)]),
            ("Done.", []),
        ])
        roles = [m["role"] for m in fake.calls[1]["messages"]]
        assert "tool" in roles
        tool_msg = next(m for m in fake.calls[1]["messages"] if m["role"] == "tool")
        assert "successfully" in tool_msg["content"]
        assert tool_msg["tool_call_id"] == "call_1"

    def test_tool_schemas_are_passed_on_every_turn(self, vm):
        fake, _, _ = drive(vm, [
            ("", [tool_call("plot_graph", PLOT_ARGS)]),
            ("Done.", []),
        ])
        for call in fake.calls:
            names = {t["function"]["name"] for t in call["tools"]}
            assert "plot_graph" in names

    def test_plain_answer_needs_no_second_round_trip(self, vm):
        fake, results, errors = drive(vm, [("A wafer plot maps values to die positions.", [])])
        assert not errors
        assert len(fake.calls) == 1
        assert results[0].action == "answer"
        assert "wafer plot" in results[0].text_summary


class TestMultipleToolCalls:
    def test_two_plot_calls_in_one_turn_yield_two_configs(self, vm):
        _, results, _ = drive(vm, [
            ("", [tool_call("plot_graph", PLOT_ARGS, "c1"),
                  tool_call("plot_graph", {**PLOT_ARGS, "plot_style": "scatter"}, "c2")]),
            ("Created both.", []),
        ])
        assert len(results[0].plot_config) == 2
        assert {c["plot_style"] for c in results[0].plot_config} == {"box", "scatter"}

    def test_tool_calls_across_turns_accumulate(self, vm):
        _, results, _ = drive(vm, [
            ("", [tool_call("plot_graph", PLOT_ARGS, "c1")]),
            ("", [tool_call("plot_graph", {**PLOT_ARGS, "plot_style": "bar"}, "c2")]),
            ("Both done.", []),
        ])
        assert len(results[0].plot_config) == 2


class TestConversationRecording:
    def test_roles_recorded_in_order(self, vm):
        drive(vm, [
            ("", [tool_call("plot_graph", PLOT_ARGS)]),
            ("Done.", []),
        ])
        assert [m["role"] for m in vm._conversation.messages] == [
            "user", "assistant", "tool", "assistant",
        ]

    def test_tool_messages_are_hidden_from_the_ui(self, vm):
        drive(vm, [
            ("", [tool_call("plot_graph", PLOT_ARGS)]),
            ("Done.", []),
        ])
        tool_msg = next(m for m in vm._conversation.messages if m["role"] == "tool")
        assert tool_msg["is_hidden"] is True


# ═══════════════════════════════════════════════════════════════════════════
# Failure paths
# ═══════════════════════════════════════════════════════════════════════════

class TestToolFailures:
    def test_unknown_tool_is_reported_back_rather_than_raising(self, vm):
        _, results, errors = drive(vm, [
            ("", [tool_call("no_such_tool", {})]),
            ("Sorry, I could not do that.", []),
        ])
        assert not errors
        tool_msg = next(m for m in vm._conversation.messages if m["role"] == "tool")
        assert "no_such_tool" in tool_msg["content"]
        assert "nknown tool" in tool_msg["content"]

    def test_malformed_json_arguments_ask_the_model_to_retry(self, vm):
        _, _, errors = drive(vm, [
            ("", [tool_call("plot_graph", "{not valid json")]),
            ("Retrying.", []),
        ])
        assert not errors
        tool_msg = next(m for m in vm._conversation.messages if m["role"] == "tool")
        assert "not valid JSON" in tool_msg["content"]
        assert "retry" in tool_msg["content"].lower()

    def test_invalid_filter_is_rejected_without_creating_a_plot(self, vm):
        _, results, _ = drive(vm, [
            ("", [tool_call("plot_graph", {**PLOT_ARGS, "filters": ["Zone == Edge"]})]),
            ("Let me fix that filter.", []),
        ])
        tool_msg = next(m for m in vm._conversation.messages if m["role"] == "tool")
        assert "NOT created" in tool_msg["content"]
        assert results[0].action == "answer"


class TestLoopCap:
    """A model that keeps calling tools forever must be stopped — and must
    not swallow the work it already queued on the way (see B3)."""

    _RUNAWAY = [("", [tool_call("plot_graph", PLOT_ARGS, f"c{i}")]) for i in range(12)]

    def test_loop_is_capped(self, vm):
        fake, _, errors = drive(vm, self._RUNAWAY)
        assert errors, "runaway tool loop must surface an error"
        assert "maximum turns" in errors[0]
        assert len(fake.calls) <= 6

    def test_queued_plots_are_not_lost_when_the_cap_is_hit(self, vm):
        _, results, _ = drive(vm, self._RUNAWAY)
        assert results, "plots queued before the cap must still reach the UI"
        assert results[0].action == "plot"
        assert len(results[0].plot_config) >= 1


class TestCancel:
    def test_cancel_stops_thinking_and_forwards_to_the_client(self, vm):
        fake, _, _ = drive(vm, [("partial", [])])
        states = []
        vm.thinking_changed.connect(lambda busy, _text: states.append(busy))
        vm.cancel()
        assert fake.cancelled >= 1
        assert states[-1] is False
