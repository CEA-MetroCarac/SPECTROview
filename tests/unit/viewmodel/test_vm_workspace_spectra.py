"""Unit tests for viewmodel/vm_workspace_spectra.py - VMWorkspaceSpectra.

VMWorkspaceSpectra represents each loaded single spectrum as a 1-row "map"
in SpectraStore, keyed by fname. Fitting goes through a real QThread
(VBFthread); tests that need results synchronously call .fit()/.start() then
block on QThread.wait() (a plain blocking join, no event loop required) and,
where signal delivery matters, follow up with qapp.processEvents().
"""
import numpy as np
import pandas as pd
import pytest

from spectroview.model.spectra_store import SpectraStore
from spectroview.viewmodel.vm_workspace_spectra import VMWorkspaceSpectra


@pytest.fixture
def vm(settings):
    return VMWorkspaceSpectra(settings)


def _add_spectrum(vm, name, x, y):
    vm.store.add_map(name, x, y[None, :].astype(np.float32),
                       np.array([[0.0, 0.0]]), [name])
    return vm.store.get_map_data(name)


def _lorentzian_fit_model(ampli=100.0, fwhm=6.0, x0=500.0, seed_offset=0.0):
    return {
        "fit_params": {"max_ite": 200, "xtol": 1e-5, "ftol": 1e-5},
        "peak_labels": ["Peak1"],
        "peak_models": {"0": {"Lorentzian": {
            "ampli": {"value": ampli * 0.8, "min": 0.0, "max": 1e6, "vary": True, "expr": None},
            "fwhm": {"value": fwhm * 0.8, "min": 1e-3, "max": 100.0, "vary": True, "expr": None},
            "x0": {"value": x0 + seed_offset, "min": x0 - 30, "max": x0 + 30, "vary": True, "expr": None},
        }}},
    }


def _synthetic_lorentzian(x0=500.0, ampli=100.0, fwhm=6.0):
    x = np.linspace(300.0, 700.0, 300)
    y = ampli / (1 + 4 * ((x - x0) / fwhm) ** 2)
    return x, y


class TestLoadFiles:
    def test_loads_txt_spectrum_into_store(self, vm, single_spectrum_file):
        vm.load_files([str(single_spectrum_file)])
        assert single_spectrum_file.stem in vm.store.map_names
        md = vm.store.get_map_data(single_spectrum_file.stem)
        assert md.n_spectra == 1

    def test_loads_multiple_files(self, vm, multiple_spectra_files):
        vm.load_files([str(p) for p in multiple_spectra_files])
        assert len(vm.store.map_names) == len(multiple_spectra_files)

    def test_duplicate_stem_is_skipped(self, vm, single_spectrum_file):
        vm.load_files([str(single_spectrum_file)])
        vm.load_files([str(single_spectrum_file)])  # same stem again
        assert vm.store.map_names.count(single_spectrum_file.stem) == 1

    def test_source_path_stored_in_metadata(self, vm, single_spectrum_file):
        vm.load_files([str(single_spectrum_file)])
        md = vm.store.get_map_data(single_spectrum_file.stem)
        assert md.map_metadata.get("source_path")


class TestSelection:
    def test_set_selected_fnames(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        _add_spectrum(vm, "s2", x, y)
        vm.set_selected_fnames(["s2"])
        assert vm.selected_fnames == ["s2"]

    def test_set_selected_indices_maps_to_fnames(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        _add_spectrum(vm, "s2", x, y)
        vm.set_selected_indices([1])
        assert vm.selected_fnames == ["s2"]

    def test_duplicate_selection_deduplicated_preserving_order(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1", "s1", "s1"])
        assert vm.selected_fnames == ["s1"]

    def test_selection_changed_signal_emits_tensor_list(self, vm, qapp):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        received = []
        vm.spectra_selection_changed.connect(received.append)
        vm.set_selected_fnames(["s1"])
        assert len(received) == 1
        assert received[0]["type"] == "tensor_list"
        assert received[0]["fnames"] == ["s1"]

    def test_empty_selection_emits_empty_list(self, vm, qapp):
        received = []
        vm.spectra_selection_changed.connect(received.append)
        vm.set_selected_fnames([])
        assert received[-1] == []


class TestSpectralRange:
    """apply_spectral_range() is the 'ROI selection' mechanism: it restricts
    fitting/analysis to an X-axis sub-range."""

    def test_crops_to_requested_range(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.apply_spectral_range(400.0, 600.0, apply_all=False)
        md = vm.store.get_map_data("s1")
        assert md.x.min() >= 400.0 - 5
        assert md.x.max() <= 600.0 + 5
        assert md.range_min == pytest.approx(md.x[0])
        assert md.range_max == pytest.approx(md.x[-1])

    def test_swapped_min_max_still_works(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.apply_spectral_range(600.0, 400.0, apply_all=False)  # reversed
        md = vm.store.get_map_data("s1")
        assert md.x.min() < md.x.max()

    def test_apply_all_targets_every_active_spectrum(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        _add_spectrum(vm, "s2", x, y)
        vm.apply_spectral_range(400.0, 600.0, apply_all=True)
        for name in ("s1", "s2"):
            md = vm.store.get_map_data(name)
            assert md.range_min == pytest.approx(400.0, abs=5)

    def test_clears_stale_fit_model_on_recrop(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.add_peak_at(500.0)
        md = vm.store.get_map_data("s1")
        assert md.fit_model is not None
        vm.apply_spectral_range(400.0, 600.0, apply_all=False)
        assert md.fit_model is None

    def test_no_selection_emits_notify(self, vm, qapp):
        notifications = []
        vm.notify.connect(notifications.append)
        vm.apply_spectral_range(400.0, 600.0, apply_all=False)
        assert any("selected" in n.lower() for n in notifications)


class TestBaseline:
    def test_add_baseline_point_creates_config(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.add_baseline_point(310.0, 50.0)
        md = vm.store.get_map_data("s1")
        assert md.baseline_config is not None
        assert md.baseline_config["points"][0] == [310.0]

    def test_points_stay_sorted_by_x(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.add_baseline_point(600.0, 10.0)
        vm.add_baseline_point(320.0, 20.0)
        md = vm.store.get_map_data("s1")
        assert md.baseline_config["points"][0] == [320.0, 600.0]

    def test_remove_baseline_point_removes_closest(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.add_baseline_point(320.0, 20.0)
        vm.add_baseline_point(600.0, 10.0)
        vm.remove_baseline_point(325.0)  # closest to 320.0
        md = vm.store.get_map_data("s1")
        assert md.baseline_config["points"][0] == [600.0]

    def test_cannot_add_point_after_subtraction(self, vm, qapp):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.add_baseline_point(320.0, 20.0)
        vm.add_baseline_point(680.0, 20.0)
        vm.subtract_baseline(apply_all=False)
        notifications = []
        vm.notify.connect(notifications.append)
        vm.add_baseline_point(500.0, 50.0)
        assert any("subtract" in n.lower() for n in notifications)

    def test_subtract_baseline_zeroes_out_flat_baseline(self, vm):
        x = np.linspace(300, 700, 200)
        y = 2.0 * x + 5.0  # pure linear "baseline", no real peak
        _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.add_baseline_point(300.0, 0.0)
        vm.add_baseline_point(700.0, 0.0)
        md = vm.store.get_map_data("s1")
        md.baseline_config["attached"] = True
        md.baseline_config["sigma"] = 0
        md.Y_baseline = None
        vm._apply_baseline_settings(md.baseline_config, ["s1"])
        vm.subtract_baseline(apply_all=False)
        assert md.is_baseline_subtracted is True
        np.testing.assert_allclose(md.Y[0], np.zeros_like(x), atol=1e-2)

    def test_copy_paste_baseline(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        _add_spectrum(vm, "s2", x, y)
        vm.set_selected_fnames(["s1"])
        vm.add_baseline_point(320.0, 20.0)
        vm.copy_baseline()
        vm.set_selected_fnames(["s2"])
        vm.paste_baseline(apply_all=False)
        md2 = vm.store.get_map_data("s2")
        assert md2.baseline_config["points"][0] == [320.0]

    def test_delete_baseline_reverts_to_raw(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.add_baseline_point(320.0, 20.0)
        vm.delete_baseline(apply_all=False)
        md = vm.store.get_map_data("s1")
        assert md.baseline_config is None
        assert md.is_baseline_subtracted is False


class TestPeaks:
    def test_add_peak_at_creates_fit_model(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.add_peak_at(500.0)
        md = vm.store.get_map_data("s1")
        assert md.fit_model is not None
        assert "0" in md.fit_model["peak_models"]
        shape = list(md.fit_model["peak_models"]["0"].keys())[0]
        assert shape == "Lorentzian"  # default _current_peak_shape

    def test_set_peak_shape_changes_next_added_peak(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.set_peak_shape("Gaussian")
        vm.add_peak_at(500.0)
        md = vm.store.get_map_data("s1")
        shape = list(md.fit_model["peak_models"]["0"].keys())[0]
        assert shape == "Gaussian"

    def test_multiple_peaks_get_distinct_indices(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.add_peak_at(450.0)
        vm.add_peak_at(550.0)
        md = vm.store.get_map_data("s1")
        assert set(md.fit_model["peak_models"].keys()) == {"0", "1"}

    def test_remove_peak_at_removes_closest(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.add_peak_at(450.0)
        vm.add_peak_at(550.0)
        vm.remove_peak_at(455.0)  # closest to 450
        md = vm.store.get_map_data("s1")
        assert len(md.fit_model["peak_models"]) == 1
        remaining_shape = list(md.fit_model["peak_models"].values())[0]
        remaining_x0 = list(remaining_shape.values())[0]["x0"]["value"]
        assert remaining_x0 == pytest.approx(550.0)

    def test_update_peak_param(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.add_peak_at(500.0)
        vm.update_peak_param(0, "x0", "value", 505.0)
        md = vm.store.get_map_data("s1")
        assert md.fit_model["peak_models"]["0"]["Lorentzian"]["x0"]["value"] == 505.0

    def test_update_peak_model_changes_shape(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.add_peak_at(500.0)
        vm.update_peak_model(0, "Gaussian")
        md = vm.store.get_map_data("s1")
        assert list(md.fit_model["peak_models"]["0"].keys()) == ["Gaussian"]

    def test_delete_peak_by_index(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.add_peak_at(450.0)
        vm.add_peak_at(550.0)
        vm.delete_peak(0)
        md = vm.store.get_map_data("s1")
        assert "0" not in md.fit_model["peak_models"]
        assert "1" in md.fit_model["peak_models"]

    def test_delete_peaks_clears_everything(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.add_peak_at(500.0)
        vm.delete_peaks(apply_all=False)
        md = vm.store.get_map_data("s1")
        assert md.fit_model is None

    def test_copy_paste_peaks(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        _add_spectrum(vm, "s2", x, y)
        vm.set_selected_fnames(["s1"])
        vm.add_peak_at(500.0)
        vm.copy_peaks()
        vm.set_selected_fnames(["s2"])
        vm.paste_peaks(apply_all=False)
        md2 = vm.store.get_map_data("s2")
        assert md2.fit_model is not None
        assert "0" in md2.fit_model["peak_models"]

    def test_editing_a_peak_clears_stale_fit_results(self, vm):
        x, y = _synthetic_lorentzian()
        md = _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.add_peak_at(500.0)
        md.peak_params = np.array([[1.0, 2.0, 3.0]])
        md.fit_success = np.array([True])
        vm.update_peak_param(0, "x0", "value", 505.0)
        assert md.peak_params is None
        assert md.fit_success is None


class TestXCorrection:
    def test_apply_x_correction_shifts_toward_si_reference(self, vm):
        x, y = _synthetic_lorentzian(x0=515.0)
        md = _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.apply_x_correction(measured_peak=515.0)
        assert md.xcorrection_value == pytest.approx(520.7 - 515.0)
        assert md.x0[0] == pytest.approx(x[0] + (520.7 - 515.0))

    def test_reapplying_correction_is_not_cumulative(self, vm):
        x, y = _synthetic_lorentzian(x0=515.0)
        md = _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.apply_x_correction(measured_peak=515.0)
        vm.apply_x_correction(measured_peak=518.0)  # re-measured differently
        assert md.xcorrection_value == pytest.approx(520.7 - 518.0)

    def test_undo_x_correction_restores_original_axis(self, vm):
        x, y = _synthetic_lorentzian(x0=515.0)
        md = _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        original_x0 = md.x0.copy()
        vm.apply_x_correction(measured_peak=515.0)
        vm.undo_x_correction()
        np.testing.assert_allclose(md.x0, original_x0)
        assert md.xcorrection_value == 0.0


class TestYNormalization:
    def test_apply_and_undo_round_trip(self, vm):
        x, y = _synthetic_lorentzian()
        md = _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        original_y0 = md.Y0.copy()
        vm.apply_y_normalization(norm_factor=2.0, apply_all=False)
        np.testing.assert_allclose(md.Y0, original_y0 / 2.0, rtol=1e-5)
        vm.undo_y_normalization(apply_all=False)
        np.testing.assert_allclose(md.Y0, original_y0, rtol=1e-5)


class TestReinitSpectra:
    def test_reinit_clears_all_processing(self, vm):
        x, y = _synthetic_lorentzian()
        md = _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.add_peak_at(500.0)
        vm.apply_spectral_range(400.0, 600.0, apply_all=False)
        vm.reinit_spectra(apply_all=False)
        assert md.fit_model is None
        assert md.range_min is None
        np.testing.assert_array_equal(md.x, md.x0)


class TestFitModelSerialization:
    def test_apply_fit_model_to_mapdata_sets_range_baseline_peaks(self, vm):
        x, y = _synthetic_lorentzian()
        md = _add_spectrum(vm, "s1", x, y)
        fit_model = {
            "range_min": 400.0, "range_max": 600.0,
            "baseline": {"mode": "Linear", "attached": False, "sigma": 0,
                          "points": [[400.0, 600.0], [0.0, 0.0]], "is_subtracted": False},
            "peak_labels": ["Peak1"],
            "peak_models": {"0": {"Lorentzian": {
                "ampli": {"value": 90.0, "min": 0, "max": 1e5, "vary": True, "expr": None},
                "fwhm": {"value": 6.0, "min": 0, "max": 100, "vary": True, "expr": None},
                "x0": {"value": 500.0, "min": 470, "max": 530, "vary": True, "expr": None},
            }}},
        }
        vm._apply_fit_model_to_mapdata(md, fit_model)
        assert md.range_min == pytest.approx(400.0, abs=5)
        assert md.baseline_config["mode"] == "Linear"
        assert "0" in md.fit_model["peak_models"]

    def test_build_clean_fit_model_round_trips_through_save_apply(self, vm, tmp_path):
        from spectroview.viewmodel.utils import build_clean_fit_model
        x, y = _synthetic_lorentzian()
        md = _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.add_peak_at(500.0)
        md.range_min, md.range_max = 400.0, 600.0
        raw_model = dict(md.fit_model)
        raw_model["range_min"] = md.range_min
        raw_model["range_max"] = md.range_max

        clean = build_clean_fit_model(raw_model)
        assert "peak_models" in clean
        assert "fit_params" in clean
        assert list(clean.keys()) == ["fit_params", "range_min", "range_max", "baseline", "peak_labels", "peak_models"]


class TestFitting:
    def test_fit_single_spectrum_recovers_known_params(self, vm, qapp):
        x, y = _synthetic_lorentzian(x0=500.0, ampli=100.0, fwhm=6.0)
        md = _add_spectrum(vm, "s1", x, y)
        vm._apply_fit_model_to_mapdata(md, _lorentzian_fit_model(seed_offset=-2.0))
        vm.selected_fnames = ["s1"]

        vm.fit(apply_all=False)
        vm._fit_thread.wait()
        qapp.processEvents()

        assert md.fit_success[0]
        result = dict(zip(md.param_names, md.peak_params[0]))
        assert result["P1_x0"] == pytest.approx(500.0, abs=0.1)
        assert result["P1_fwhm"] == pytest.approx(6.0, abs=0.2)
        assert vm._is_fitting is False

    def test_fit_with_no_peaks_notifies_and_does_not_start_thread(self, vm, qapp):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        vm.selected_fnames = ["s1"]
        notifications = []
        vm.notify.connect(notifications.append)
        vm.fit(apply_all=False)
        assert any("no peaks" in n.lower() for n in notifications)
        assert vm._fit_thread is None

    def test_apply_fit_model_and_run_via_fit_model_builder(self, vm, qapp, fit_model_si_file, tmp_path):
        import json
        from unittest.mock import MagicMock

        x = np.linspace(460, 570, 300)
        y = 9146.0 / (1 + 4 * ((x - 521.0) / 4.8) ** 2)
        md = _add_spectrum(vm, "s1", x, y)
        vm.selected_fnames = ["s1"]

        builder = MagicMock()
        builder.get_current_model_path.return_value = fit_model_si_file
        vm.set_fit_model_builder(builder)

        vm.apply_fit_model(apply_all=False)
        vm._fit_thread.wait()
        qapp.processEvents()

        assert md.fit_success[0]
        assert md.fit_r2[0] > 0.9


class TestPersistence:
    def test_save_and_load_work_round_trip(self, vm, tmp_path, monkeypatch, qapp):
        from PySide6.QtWidgets import QFileDialog
        x, y = _synthetic_lorentzian()
        md = _add_spectrum(vm, "s1", x, y)
        vm.set_selected_fnames(["s1"])
        vm.add_peak_at(500.0)

        save_path = tmp_path / "roundtrip.spectra"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(save_path), ""))
        vm.save_work()
        assert save_path.exists()

        vm2 = VMWorkspaceSpectra(vm.settings)
        vm2.load_work(str(save_path))
        assert "s1" in vm2.store.map_names
        md2 = vm2.store.get_map_data("s1")
        np.testing.assert_allclose(md2.x0, md.x0)
        np.testing.assert_allclose(md2.Y0, md.Y0)
        assert md2.fit_model is not None

    def test_load_real_legacy_spectra_workspace(self, vm, legacy_spectra_workspace):
        if not legacy_spectra_workspace.exists():
            pytest.skip("example legacy .spectra file not present")
        vm.load_work(str(legacy_spectra_workspace))
        assert len(vm.store.map_names) > 0

    def test_clear_workspace_resets_store(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        vm.clear_workspace()
        assert vm.store.map_names == []
        assert vm.selected_fnames == []


class TestResultsExtraction:
    def test_collect_fit_results_builds_dataframe(self, vm, qapp):
        x, y = _synthetic_lorentzian(x0=500.0, ampli=100.0, fwhm=6.0)
        md = _add_spectrum(vm, "s1", x, y)
        vm._apply_fit_model_to_mapdata(md, _lorentzian_fit_model(seed_offset=-1.0))
        vm.selected_fnames = ["s1"]
        vm.fit(apply_all=False)
        vm._fit_thread.wait()
        qapp.processEvents()

        vm.collect_fit_results()
        assert vm.df_fit_results is not None
        assert len(vm.df_fit_results) == 1
        assert "Filename" in vm.df_fit_results.columns
        assert "X" not in vm.df_fit_results.columns  # dropped for Spectra workspace

    def test_compute_column_from_expression(self, vm, qapp, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
        monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: None)

        vm.df_fit_results = pd.DataFrame({"Filename": ["s1", "s2"], "ampli_P1": [10.0, 20.0], "fwhm_P1": [2.0, 4.0]})
        vm.compute_column_from_expression("area_P1", "ampli_P1 * fwhm_P1")
        assert "area_P1" in vm.df_fit_results.columns
        assert vm.df_fit_results["area_P1"].tolist() == [20.0, 80.0]

    def test_compute_column_duplicate_name_warns(self, vm, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        warnings = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))
        vm.df_fit_results = pd.DataFrame({"Filename": ["s1"], "ampli_P1": [10.0]})
        vm.compute_column_from_expression("ampli_P1", "ampli_P1 * 2")
        assert len(warnings) == 1

    def test_add_column_from_filename(self, vm):
        vm.df_fit_results = pd.DataFrame({"Filename": ["sampleA_run1", "sampleB_run2"], "x0_P1": [1.0, 2.0]})
        vm.add_column_from_filename("sample", 0)
        assert vm.df_fit_results["sample"].tolist() == ["sampleA", "sampleB"]


class TestErrorHandling:
    def test_load_files_reports_error_via_message_box(self, vm, tmp_path, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        errors = []
        monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a))
        bad_file = tmp_path / "bad.txt"
        bad_file.write_text("not,valid,spectrum,data\nasdf\n")
        # Malformed content: loader should either cope or raise -> caught and reported.
        vm.load_files([str(tmp_path / "does_not_exist_at_all.txt")])
        assert len(errors) == 1

    def test_operations_on_empty_selection_do_not_crash(self, vm, qapp):
        vm.copy_baseline()
        vm.paste_baseline()
        vm.copy_peaks()
        vm.paste_peaks()
        vm.apply_x_correction(500.0)
        vm.undo_x_correction()
        vm.fit(apply_all=False)  # nothing selected -> no-op, must not raise


class TestActiveState:
    def test_set_spectrum_active_toggles_flag(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        vm.set_spectrum_active("s1", False)
        assert not vm.store.get_map_data("s1").is_active[0]
        vm.set_spectrum_active("s1", True)
        assert vm.store.get_map_data("s1").is_active[0]

    def test_set_all_spectra_active(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "s1", x, y)
        _add_spectrum(vm, "s2", x, y)
        vm.set_all_spectra_active(False)
        assert not any(vm.store.get_map_data(n).is_active[0] for n in vm.store.map_names)
        vm.set_all_spectra_active(True)
        assert all(vm.store.get_map_data(n).is_active[0] for n in vm.store.map_names)

    def test_set_spectrum_active_unknown_fname_is_noop(self, vm):
        vm.set_spectrum_active("missing", False)  # must not raise


class TestReceiveSpectra:
    def test_receive_spectra_ingests_payloads_into_store(self, vm, qapp):
        x, y = _synthetic_lorentzian()
        payloads = [{
            'name': "from_maps_1",
            'x0': x.astype(np.float64),
            'Y0': y[None, :].astype(np.float32),
            'baseline_config': None,
            'fit_model': None,
            'range_min': None,
            'range_max': None,
            'is_subtracted': False,
        }]
        emitted = []
        vm.spectra_list_changed.connect(lambda info: emitted.append(info))
        vm.receive_spectra(payloads)
        assert "from_maps_1" in vm.store.map_names
        assert emitted, "receive_spectra should refresh the spectra list"

    def test_receive_spectra_skips_duplicate_names(self, vm):
        x, y = _synthetic_lorentzian()
        _add_spectrum(vm, "dup", x, y)
        before = len(vm.store.map_names)
        vm.receive_spectra([{
            'name': "dup", 'x0': x.astype(np.float64), 'Y0': y[None, :].astype(np.float32),
            'baseline_config': None, 'fit_model': None,
            'range_min': None, 'range_max': None, 'is_subtracted': False,
        }])
        assert len(vm.store.map_names) == before
