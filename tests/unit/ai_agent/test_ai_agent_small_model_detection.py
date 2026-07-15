"""
Tests for VMChat's small-model auto-detection and manual override
(spectroview/ai_agent/vm_chat.py).

The detection order is: manual override -> parameter-count check (via a
mocked `ollama show`) -> name-pattern fallback -> default False (full tier).
Unknown/undeterminable cases must default to False (full tier) since
misclassifying an unknown *large* model as small is more harmful than the
reverse.
"""
from types import SimpleNamespace

import pandas as pd
import pytest

from spectroview.ai_agent.vm_chat import VMChat, _parse_param_size_to_billions


def _fake_model_info(parameter_size):
    return SimpleNamespace(details=SimpleNamespace(parameter_size=parameter_size))


@pytest.fixture
def vm(qapp):
    v = VMChat()
    v.set_dataframes({"df": pd.DataFrame({"A": [1, 2]})}, "df")
    return v


class TestParamSizeParsing:
    def test_billions_suffix(self):
        assert _parse_param_size_to_billions("8.2B") == pytest.approx(8.2)

    def test_millions_suffix(self):
        assert _parse_param_size_to_billions("600M") == pytest.approx(0.6)

    def test_unparseable_returns_none(self):
        assert _parse_param_size_to_billions("bogus") is None


class TestParamCountDetection:
    def test_small_model_detected_via_param_count(self, vm, monkeypatch):
        monkeypatch.setattr(
            "spectroview.ai_agent.vm_chat.get_ollama_model_info",
            lambda model: _fake_model_info("8.2B"),
        )
        vm.set_provider("Ollama")
        vm.set_model("qwen3:8b")
        assert vm.is_small_model_mode() is True
        assert vm.max_context_messages == 6

    def test_large_model_not_detected_as_small(self, vm, monkeypatch):
        monkeypatch.setattr(
            "spectroview.ai_agent.vm_chat.get_ollama_model_info",
            lambda model: _fake_model_info("70B"),
        )
        vm.set_provider("Ollama")
        vm.set_model("some-large-model:70b")
        assert vm.is_small_model_mode() is False
        assert vm.max_context_messages is None

    def test_ollama_show_failure_falls_back_to_pattern_list(self, vm, monkeypatch):
        monkeypatch.setattr(
            "spectroview.ai_agent.vm_chat.get_ollama_model_info",
            lambda model: None,
        )
        vm.set_provider("Ollama")
        vm.set_model("qwen3:8b")  # in the pattern list
        assert vm.is_small_model_mode() is True

    def test_unknown_model_with_no_info_defaults_to_full_tier(self, vm, monkeypatch):
        monkeypatch.setattr(
            "spectroview.ai_agent.vm_chat.get_ollama_model_info",
            lambda model: None,
        )
        vm.set_provider("Ollama")
        vm.set_model("some-brand-new-unlisted-model:latest")
        assert vm.is_small_model_mode() is False


class TestProviderGating:
    def test_non_ollama_provider_always_resolves_false(self, vm, monkeypatch):
        monkeypatch.setattr(
            "spectroview.ai_agent.vm_chat.get_ollama_model_info",
            lambda model: _fake_model_info("8.2B"),
        )
        vm.set_provider("DeepSeek", api_key="fake", model="deepseek-chat")
        assert vm.is_small_model_mode() is False


class TestManualOverride:
    def test_override_true_forces_small_tier_even_off_ollama(self, vm):
        vm.set_provider("DeepSeek", api_key="fake", model="deepseek-chat")
        vm.set_small_model_mode(True)
        assert vm.is_small_model_mode() is True

    def test_override_false_forces_full_tier(self, vm, monkeypatch):
        monkeypatch.setattr(
            "spectroview.ai_agent.vm_chat.get_ollama_model_info",
            lambda model: _fake_model_info("8.2B"),
        )
        vm.set_provider("Ollama")
        vm.set_model("qwen3:8b")
        vm.set_small_model_mode(False)
        assert vm.is_small_model_mode() is False

    def test_override_none_resumes_auto_detection(self, vm, monkeypatch):
        monkeypatch.setattr(
            "spectroview.ai_agent.vm_chat.get_ollama_model_info",
            lambda model: _fake_model_info("8.2B"),
        )
        vm.set_provider("Ollama")
        vm.set_model("qwen3:8b")
        vm.set_small_model_mode(False)
        vm.set_small_model_mode(None)
        assert vm.is_small_model_mode() is True
