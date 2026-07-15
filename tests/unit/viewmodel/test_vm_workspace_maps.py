"""Unit tests for viewmodel/vm_workspace_maps.py - VMWorkspaceMaps.

VMWorkspaceMaps extends VMWorkspaceSpectra: each loaded hyperspectral map is
one MapData block with N>1 rows. These tests focus on what's genuinely
Maps-specific (selection/active-state scoped to current_map_name, batch
fitting across many rows of a map, mega-batch fitting across multiple maps,
Zone/Quadrant wafer columns) -- baseline/peak/x-correction mechanics
themselves are already covered by the shared base-class tests.
"""
import numpy as np
import pandas as pd
import pytest

from spectroview.viewmodel.vm_workspace_maps import VMWorkspaceMaps


@pytest.fixture
def vm(settings):
    return VMWorkspaceMaps(settings)


def _make_map_df(map_name, n_rows_side=4, x0=500.0, ampli=100.0, fwhm=6.0, seed=0):
    """Build a synthetic hyperspectral map DataFrame with a single Lorentzian
    peak (slightly jittered per-pixel) plus a trailing 'extra' wavenumber
    column, matching _extract_spectra_from_map's convention of dropping the
    last column."""
    rng = np.random.default_rng(seed)
    x = np.linspace(300.0, 700.0, 200)
    x_cols = list(x) + [x[-1] + (x[1] - x[0])]  # extra col to be dropped

    rows = []
    coords = []
    for i in range(n_rows_side):
        for j in range(n_rows_side):
            a = ampli * rng.uniform(0.8, 1.2)
            y = a / (1 + 4 * ((x - x0) / fwhm) ** 2)
            y_full = np.append(y, 0.0)  # matches the extra dropped column
            rows.append(y_full)
            coords.append((float(i), float(j)))

    data = {"X": [c[0] for c in coords], "Y": [c[1] for c in coords]}
    for k, xc in enumerate(x_cols):
        data[str(xc)] = [row[k] for row in rows]
    return pd.DataFrame(data)


def _lorentzian_fit_model(ampli=100.0, fwhm=6.0, x0=500.0, seed_offset=-2.0, coef_noise=0.0):
    return {
        "fit_params": {"max_ite": 200, "xtol": 1e-5, "ftol": 1e-5, "coef_noise": coef_noise},
        "peak_labels": ["Peak1"],
        "peak_models": {"0": {"Lorentzian": {
            "ampli": {"value": ampli * 0.8, "min": 0.0, "max": 1e6, "vary": True, "expr": None},
            "fwhm": {"value": fwhm * 0.8, "min": 1e-3, "max": 100.0, "vary": True, "expr": None},
            "x0": {"value": x0 + seed_offset, "min": x0 - 30, "max": x0 + 30, "vary": True, "expr": None},
        }}},
    }


def _load_map(vm, map_name, **kwargs):
    df = _make_map_df(map_name, **kwargs)
    vm.maps[map_name] = df
    vm.maps_metadata[map_name] = {}
    vm._extract_spectra_from_map(map_name, df)
    return vm.store.get_map_data(map_name)


class TestLoadAndSelectMap:
    def test_extract_spectra_creates_one_map_data_block(self, vm):
        md = _load_map(vm, "map1", n_rows_side=3)
        assert md.n_spectra == 9
        assert "map1" in vm.store.map_names

    def test_last_wavenumber_column_is_dropped(self, vm):
        df = _make_map_df("map1", n_rows_side=2)
        n_wn_cols = len(df.columns) - 2  # minus X, Y
        md = _load_map(vm, "map1", n_rows_side=2)
        assert md.n_wavenumbers == n_wn_cols - 1

    def test_fnames_encode_coordinates(self, vm):
        md = _load_map(vm, "map1", n_rows_side=2)
        assert all(name.startswith("map1_(") for name in md.fnames)

    def test_select_map_updates_current_map_state(self, vm):
        _load_map(vm, "map1", n_rows_side=2)
        _load_map(vm, "map2", n_rows_side=2)
        vm.select_map("map2")
        assert vm.current_map_name == "map2"
        assert vm.current_map_df is vm.maps["map2"]

    def test_select_unknown_map_is_noop(self, vm):
        _load_map(vm, "map1", n_rows_side=2)
        vm.select_map("does_not_exist")
        assert vm.current_map_name is None


class TestActiveStateAndSelection:
    def test_set_spectrum_active_toggles_flag(self, vm):
        md = _load_map(vm, "map1", n_rows_side=2)
        vm.select_map("map1")
        fname = md.fnames[0]
        vm.set_spectrum_active(fname, False)
        assert md.is_active[0] is np.False_ or md.is_active[0] == False  # noqa: E712

    def test_set_all_current_map_spectra_active(self, vm):
        md = _load_map(vm, "map1", n_rows_side=2)
        vm.select_map("map1")
        vm.set_all_current_map_spectra_active(False)
        assert not md.is_active.any()
        vm.set_all_current_map_spectra_active(True)
        assert md.is_active.all()

    def test_get_active_spectra_scoped_to_current_map(self, vm):
        md1 = _load_map(vm, "map1", n_rows_side=2)
        md2 = _load_map(vm, "map2", n_rows_side=2)
        vm.select_map("map1")
        active = vm._get_active_spectra()
        assert set(active) == set(md1.fnames)

    def test_select_all_current_map_spectra_sets_selection(self, vm):
        md = _load_map(vm, "map1", n_rows_side=2)
        vm.select_map("map1")
        vm.select_all_current_map_spectra()
        assert set(vm.selected_fnames) == set(md.fnames)


class TestGetCurrentMapDataframe:
    def test_filters_out_inactive_rows(self, vm):
        md = _load_map(vm, "map1", n_rows_side=2)
        vm.select_map("map1")
        vm.set_spectrum_active(md.fnames[0], False)
        df = vm.get_current_map_dataframe()
        assert len(df) == md.n_spectra - 1


class TestSpectralRangeROI:
    def test_apply_spectral_range_scoped_to_current_map(self, vm):
        md1 = _load_map(vm, "map1", n_rows_side=2)
        md2 = _load_map(vm, "map2", n_rows_side=2)
        vm.select_map("map1")
        vm.apply_spectral_range(400.0, 600.0, apply_all=False)
        assert md1.x.min() >= 400.0 - 5
        assert md2.x is None  # untouched


class TestBatchFittingWholeMap:
    def test_fit_all_rows_of_current_map(self, vm, qapp):
        md = _load_map(vm, "map1", n_rows_side=4, x0=500.0, ampli=100.0, fwhm=6.0, seed=1)
        vm._apply_fit_model_to_mapdata(md, _lorentzian_fit_model())
        vm.select_map("map1")
        vm.select_all_current_map_spectra()

        vm.fit(apply_all=False)
        vm._fit_thread.wait()
        qapp.processEvents()

        assert md.fit_success.all()
        idx_x0 = md.param_names.index("P1_x0")
        np.testing.assert_allclose(md.peak_params[:, idx_x0], 500.0, atol=0.5)

    def test_fit_subset_of_selected_rows_only(self, vm, qapp):
        md = _load_map(vm, "map1", n_rows_side=3, seed=2)
        vm._apply_fit_model_to_mapdata(md, _lorentzian_fit_model())
        vm.select_map("map1")
        subset = md.fnames[:3]
        vm.set_selected_fnames(subset)

        vm.fit(apply_all=False)
        vm._fit_thread.wait()
        qapp.processEvents()

        assert md.fit_success[:3].all()
        # Untouched rows keep whatever fit_success default it started as (unset)
        assert md.fit_success.shape[0] == md.n_spectra

    def test_apply_fit_model_and_run_apply_all_across_maps_uses_grouped_task(self, vm, qapp):
        md1 = _load_map(vm, "mapA", n_rows_side=3, x0=500.0, seed=3)
        md2 = _load_map(vm, "mapB", n_rows_side=3, x0=500.0, seed=4)
        # Both maps share the same x-axis length (same synthetic grid) so
        # _run_fit_thread(apply_all=True) merges them into one mega-batch task.
        vm._apply_fit_model_and_run(_lorentzian_fit_model(), apply_all=True)
        vm._fit_thread.wait()
        qapp.processEvents()

        assert md1.fit_success.all()
        assert md2.fit_success.all()


class TestFitResultsWithWaferColumns:
    def test_collect_fit_results_adds_zone_and_quadrant_for_wafer_type(self, vm, qapp):
        md = _load_map(vm, "map1", n_rows_side=3, seed=5)
        md.coords = np.array([[10.0, 10.0], [-10.0, 10.0], [-10.0, -10.0],
                                [10.0, -10.0], [0.0, 0.0], [5.0, 5.0],
                                [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        vm._apply_fit_model_to_mapdata(md, _lorentzian_fit_model())
        vm.select_map("map1")
        vm.select_all_current_map_spectra()
        vm.map_type = "wafer_150mm"

        vm.fit(apply_all=False)
        vm._fit_thread.wait()
        qapp.processEvents()

        vm.collect_fit_results()
        assert "Zone" in vm.df_fit_results.columns
        assert "Quadrant" in vm.df_fit_results.columns

    def test_collect_fit_results_2dmap_has_no_wafer_columns(self, vm, qapp):
        md = _load_map(vm, "map1", n_rows_side=2, seed=6)
        vm._apply_fit_model_to_mapdata(md, _lorentzian_fit_model())
        vm.select_map("map1")
        vm.select_all_current_map_spectra()

        vm.fit(apply_all=False)
        vm._fit_thread.wait()
        qapp.processEvents()

        vm.collect_fit_results()
        assert "Zone" not in vm.df_fit_results.columns
        assert "X" in vm.df_fit_results.columns  # kept for Maps (unlike Spectra)


class TestSendSpectraToSpectraWorkspace:
    def test_emits_deep_copied_payloads_for_selected_rows(self, vm, qapp):
        md = _load_map(vm, "map1", n_rows_side=2, seed=7)
        vm.select_map("map1")
        vm.select_all_current_map_spectra()

        received = []
        vm.send_spectra_to_workspace.connect(received.append)
        vm.send_selected_spectra_to_spectra_workspace()

        assert len(received) == 1
        payloads = received[0]
        assert len(payloads) == md.n_spectra
        assert all("x0" in p and "Y0" in p for p in payloads)


class TestRemoveMap:
    def test_remove_map_clears_store_and_maps_dict(self, vm):
        _load_map(vm, "map1", n_rows_side=2)
        vm.remove_map("map1")
        assert "map1" not in vm.store.map_names
        assert "map1" not in vm.maps


class TestClearWorkspace:
    def test_clear_workspace_resets_maps_state(self, vm):
        _load_map(vm, "map1", n_rows_side=2)
        vm.select_map("map1")
        vm.clear_workspace()
        assert vm.maps == {}
        assert vm.current_map_name is None
        assert vm.store.map_names == []


class TestPersistence:
    def test_save_and_load_maps_workspace_round_trip(self, vm, tmp_path, monkeypatch, qapp):
        from PySide6.QtWidgets import QFileDialog
        md = _load_map(vm, "map1", n_rows_side=2, seed=8)
        vm.select_map("map1")

        save_path = tmp_path / "roundtrip.maps"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(save_path), ""))
        vm.save_work()
        assert save_path.exists()

        vm2 = VMWorkspaceMaps(vm.settings)
        vm2.load_work(str(save_path))
        assert "map1" in vm2.store.map_names
        md2 = vm2.store.get_map_data("map1")
        np.testing.assert_allclose(md2.Y0, md.Y0)
        assert "map1" in vm2.maps

    def test_load_real_zip_maps_workspace(self, vm, zip_maps_workspace):
        if not zip_maps_workspace.exists():
            pytest.skip("example .maps file not present")
        vm.load_work(str(zip_maps_workspace))
        assert len(vm.store.map_names) > 0
