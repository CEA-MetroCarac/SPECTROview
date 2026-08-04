"""Tests for VMoreTab's fit-results decimals control and normalization gating."""
from spectroview.view.components.v_moretab import VMoreTab


class _Spec:
    """Minimal spectrum-like object for show_metadata()."""
    label = "s"
    color = None
    xcorrection_value = 0.0
    source_path = None
    intensity_norm_factor = 1.0
    metadata = {}
    baseline = None


class TestMoreTabDecimals:
    def test_decimals_spinbox_defaults_to_3(self, qapp):
        w = VMoreTab()
        assert w.spin_results_decimals.value() == 3

    def test_decimals_spinbox_emits_signal(self, qapp):
        w = VMoreTab()
        received = []
        w.results_decimals_changed.connect(received.append)
        w.spin_results_decimals.setValue(5)
        assert received == [5]

    def test_decimals_control_usable_without_a_selected_spectrum(self, qapp):
        """The decimals option is a global preference: it stays enabled even
        when no spectrum is selected, unlike the normalization controls."""
        w = VMoreTab()
        w.clear_metadata()
        assert w.spin_results_decimals.isEnabled() is True
        assert w.spin_norm_factor.isEnabled() is False
        assert w.btn_normalize.isEnabled() is False

    def test_normalization_controls_enable_for_a_spectrum(self, qapp):
        w = VMoreTab()
        w.clear_metadata()
        w.show_metadata(_Spec())
        assert w.spin_norm_factor.isEnabled() is True
        assert w.spin_results_decimals.isEnabled() is True
