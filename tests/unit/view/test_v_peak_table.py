"""Unit tests for VPeakTable widget lifecycle and rebuild behavior."""
from spectroview.view.components.v_peak_table import VPeakTable


def test_peak_table_rebuild_after_clear_without_cpp_deleted_error(qapp):
    table = VPeakTable()
    fit_model = {
        "peak_models": {
            "0": {
                "PseudoVoigt": {
                    "x0": {"value": 520.0, "vary": True},
                    "ampli": {"value": 100.0, "vary": True},
                    "fwhm": {"value": 10.0, "vary": True},
                    "alpha": {"value": 0.5, "vary": True},
                }
            },
            "1": {
                "Lorentzian": {
                    "x0": {"value": 580.0, "vary": True},
                    "ampli": {"value": 80.0, "vary": True},
                    "fwhm": {"value": 15.0, "vary": True},
                }
            },
        },
        "peak_labels": ["P1", "P2"],
    }

    # 1. Set spectrum with fit_model
    table.set_spectrum({"fit_model": fit_model})
    assert len(table._row_bindings) == 2

    # 2. Set spectrum with empty fit_model (triggers layout clear)
    table.set_spectrum({"fit_model": {}})
    assert len(table._row_bindings) == 0
    assert table._current_topology is None

    # Process events to allow deleteLater() to clean up C++ objects
    qapp.processEvents()

    # 3. Re-set spectrum with same topology - must not crash with deleted C++ object
    table.set_spectrum({"fit_model": fit_model})
    assert len(table._row_bindings) == 2
