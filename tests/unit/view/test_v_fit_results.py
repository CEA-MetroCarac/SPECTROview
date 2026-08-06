"""Tests for VFitResults' fit-results decimals control (Fit Results tab)."""
from spectroview.view.components.v_fit_results import VFitResults


class TestFitResultsDecimals:
    def test_decimals_spinbox_defaults_to_3(self, qapp):
        w = VFitResults()
        assert w.spin_results_decimals.value() == 3

    def test_decimals_spinbox_emits_signal(self, qapp):
        w = VFitResults()
        received = []
        w.results_decimals_changed.connect(received.append)
        w.spin_results_decimals.setValue(5)
        assert received == [5]
