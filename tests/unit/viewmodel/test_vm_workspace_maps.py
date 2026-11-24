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

from spectroview.viewmodel.maps.vm_workspace_maps import VMWorkspaceMaps


@pytest.fixture
def vm(settings):
    return VMWorkspaceMaps(settings)


def _make_map_df(map_name, n_rows_side=4, x0=500.0, ampli=100.0, fwhm=6.0, seed=0):
    """Build a synthetic hyperspectral map DataFrame with a single Lorentzian
    peak (slightly jittered per-pixel)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(300.0, 700.0, 200)

    rows = []
    coords = []
    for i in range(n_rows_side):
        for j in range(n_rows_side):
            a = ampli * rng.uniform(0.8, 1.2)
            y = a / (1 + 4 * ((x - x0) / fwhm) ** 2)
            rows.append(y)
            coords.append((float(i), float(j)))

    mat = np.asarray(rows, dtype=np.float64)
    df = pd.DataFrame(mat, columns=[str(xc) for xc in x])
    df.insert(0, "Y", [c[1] for c in coords])
    df.insert(0, "X", [c[0] for c in coords])
    return df


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

    def test_all_numeric_wavenumber_columns_preserved(self, vm):
        df = _make_map_df("map1", n_rows_side=2)
        n_wn_cols = len(df.columns) - 2  # minus X, Y
        md = _load_map(vm, "map1", n_rows_side=2)
        assert md.n_wavenumbers == n_wn_cols

    def test_trailing_non_numeric_unnamed_columns_dropped(self, vm):
        df = _make_map_df("map_unnamed", n_rows_side=2)
        n_wn_cols = len(df.columns) - 2
        df["Unnamed: 202"] = np.nan
        vm.maps["map_unnamed"] = df
        vm.maps_metadata["map_unnamed"] = {}
        vm._extract_spectra_from_map("map_unnamed", df)
        md = vm.store.get_map_data("map_unnamed")
        assert md.n_wavenumbers == n_wn_cols

    def test_duplicate_xy_coordinates_disambiguated_for_time_resolved_maps(self, vm):
        df = pd.DataFrame({
            "X": [0.0, 0.0, 10.0, 10.0],
            "Y": [5.0, 5.0, 5.0, 5.0],
            "500.0": [10.0, 12.0, 20.0, 22.0],
            "510.0": [15.0, 18.0, 25.0, 28.0],
        })
        vm.maps["tr_map"] = df
        vm.maps_metadata["tr_map"] = {}
        vm._extract_spectra_from_map("tr_map", df)
        md = vm.store.get_map_data("tr_map")
        assert md.n_spectra == 4
        assert len(set(md.fnames)) == 4
        assert md.fnames == [
            "tr_map_(0.0, 5.0)",
            "tr_map_(0.0, 5.0)_2",
            "tr_map_(10.0, 5.0)",
            "tr_map_(10.0, 5.0)_2",
        ]
        vm.select_map("tr_map")
        vm.select_all_current_map_spectra()
        assert len(vm.selected_fnames) == 4

    def test_fnames_encode_coordinates(self, vm):
        md = _load_map(vm, "map1", n_rows_side=2)
        assert all(name.startswith("map1_(") for name in md.fnames)

    def test_select_map_updates_current_map_state(self, vm):
        _load_map(vm, "map1", n_rows_side=2)
        md2 = _load_map(vm, "map2", n_rows_side=2)
        vm.select_map("map2")
        assert vm.current_map_name == "map2"
        assert vm.current_map_df is vm.maps["map2"]
        assert vm.selected_fnames == [md2.fnames[0]]

    def test_select_map_auto_selects_first_spectrum(self, vm):
        md = _load_map(vm, "map1", n_rows_side=2)
        selected_batches = []
        vm.spectra_selection_changed.connect(selected_batches.append)

        vm.select_map("map1")
        assert vm.selected_fnames == [md.fnames[0]]
        assert len(selected_batches) == 1
        assert selected_batches[0]["fnames"] == [md.fnames[0]]

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
        _load_map(vm, "map2", n_rows_side=2)
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
        assert "R2" in vm.df_fit_results.columns  # per-spectrum fit quality collected


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
        vm.df_fit_results = pd.DataFrame({"Filename": ["map1"]})
        fit_results_signals = []
        selection_signals = []
        vm.fit_results_updated.connect(fit_results_signals.append)
        vm.spectra_selection_changed.connect(selection_signals.append)

        vm.clear_workspace()
        assert vm.maps == {}
        assert vm.maps_metadata == {}
        assert vm.current_map_name is None
        assert vm.store.map_names == []
        assert vm.df_fit_results is None
        assert fit_results_signals == [None]
        assert selection_signals == [[]]


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


class TestParseCoordsFromFname:
    def test_parses_standard_fname(self):
        from spectroview.viewmodel.common.spectral_utils import parse_coords_from_fname
        assert parse_coords_from_fname("map1_(1.5, -2.0)") == (1.5, -2.0)

    def test_roundtrips_extract_spectra_format(self):
        from spectroview.viewmodel.common.spectral_utils import parse_coords_from_fname
        # Matches _extract_spectra_from_map's f"{name}_({float(x)}, {float(y)})"
        fname = f"wafer_({float(3)}, {float(-4)})"
        assert parse_coords_from_fname(fname) == (3.0, -4.0)

    def test_returns_none_without_coords(self):
        from spectroview.viewmodel.common.spectral_utils import parse_coords_from_fname
        assert parse_coords_from_fname("plain_name") is None
        assert parse_coords_from_fname("bad_(1.0)") is None


class TestLabSpec6WorkspaceAndMoreTabIntegration:
    """Verify .l6s and .l6m files load into Spectra/Maps ViewModels, API, and More tab."""

    def test_load_l6s_and_l6m_into_viewmodels_and_more_tab(self, vm, settings, examples_dir, qapp):
        from spectroview.api import io as api_io
        from spectroview.model.io.m_io import _METADATA_ORDERED_KEYS
        from spectroview.model.spectra.spectra_store import SpectrumProxy
        from spectroview.view.components.lower_tabs import VMoreTab
        from spectroview.viewmodel.spectra.vm_workspace_spectra import VMWorkspaceSpectra

        l6s_path = examples_dir / "spectroscopic_data" / "spectrum_ls6.l6s"
        l6m_path = examples_dir / "spectroscopic_data" / "mapls6.l6m"
        assert l6s_path.exists() and l6m_path.exists()

        # 1. Spectra ViewModel + More tab display
        vm_spec = VMWorkspaceSpectra(settings)
        vm_spec.load_files([str(l6s_path)])
        assert "spectrum_ls6" in vm_spec.store.map_names
        md_spec = vm_spec.store.get_map_data("spectrum_ls6")
        assert md_spec.n_spectra == 1
        assert md_spec.n_wavenumbers == 1024

        more_tab = VMoreTab()
        proxy_spec = SpectrumProxy(md_spec, 0, "spectrum_ls6")
        more_tab.show_metadata(proxy_spec)

        table_keys = [more_tab.table.item(r, 0).text() for r in range(more_tab.table.rowCount())]
        ordered_in_spec = [k for k in _METADATA_ORDERED_KEYS if k in md_spec.map_metadata]
        assert table_keys[: len(ordered_in_spec)] == ordered_in_spec
        assert "Instrument" in table_keys[len(ordered_in_spec) :]
        assert "S/N" in table_keys[len(ordered_in_spec) :]

        # 2. Maps ViewModel + More tab display
        vm.load_map_files([str(l6m_path)])
        assert "mapls6" in vm.maps
        assert vm.current_map_name == "mapls6"
        md_map = vm.store.get_map_data("mapls6")
        assert md_map.n_spectra == 1368
        assert md_map.n_wavenumbers == 1024

        proxy_map = SpectrumProxy(md_map, 0, md_map.fnames[0])
        more_tab.show_metadata(proxy_map)
        map_table_keys = [more_tab.table.item(r, 0).text() for r in range(more_tab.table.rowCount())]
        ordered_in_map = [k for k in _METADATA_ORDERED_KEYS if k in md_map.map_metadata]
        assert map_table_keys[: len(ordered_in_map)] == ordered_in_map
        assert "SWIFT" in map_table_keys[len(ordered_in_map) :]
        assert "Instrument" in map_table_keys[len(ordered_in_map) :]

        # 3. Public API io.load_spectra and io.load_map
        api_spec = api_io.load_spectra(l6s_path)
        assert len(api_spec["items"]) == 1
        assert api_spec["items"][0]["metadata"]["File Format"] == "HORIBA LabSpec 6"

        api_map_df, api_map_meta = api_io.load_map(l6m_path)
        assert api_map_df.shape == (1368, 1026)
        assert api_map_meta["File Format"] == "HORIBA LabSpec 6 Map"

        # 4. Compressed LabSpec 6 (.l6s) file from LabRAM HR Evol into Spectra ViewModel + More tab + API
        l6s_comp_path = examples_dir / "spectroscopic_data" / "spectrum_ls6_2.l6s"
        assert l6s_comp_path.exists()
        vm_spec.load_files([str(l6s_comp_path)])
        assert "spectrum_ls6_2" in vm_spec.store.map_names
        md_comp = vm_spec.store.get_map_data("spectrum_ls6_2")
        assert md_comp.n_spectra == 1
        assert md_comp.n_wavenumbers == 1022
        assert md_comp.map_metadata["Laser Wavelength (nm)"] == "325"

        proxy_comp = SpectrumProxy(md_comp, 0, "spectrum_ls6_2")
        more_tab.show_metadata(proxy_comp)
        comp_table_keys = [more_tab.table.item(r, 0).text() for r in range(more_tab.table.rowCount())]
        assert "Instrument" in comp_table_keys
        assert "S/N" in comp_table_keys

        api_comp_spec = api_io.load_spectra(l6s_comp_path)
        assert len(api_comp_spec["items"]) == 1
        assert api_comp_spec["items"][0]["metadata"]["Instrument"] == "LabRAM HR Evol"

    def test_load_ngs_into_viewmodel_api_and_more_tab(self, settings, examples_dir, qapp):
        from spectroview.api import io as api_io
        from spectroview.model.io.m_io import _METADATA_ORDERED_KEYS
        from spectroview.model.spectra.spectra_store import SpectrumProxy
        from spectroview.view.components.lower_tabs import VMoreTab
        from spectroview.viewmodel.spectra.vm_workspace_spectra import VMWorkspaceSpectra

        ngs_path = examples_dir / "spectroscopic_data" / "spectrum.ngs"
        assert ngs_path.exists()

        vm_spec = VMWorkspaceSpectra(settings)
        vm_spec.load_files([str(ngs_path)])
        assert "spectrum" in vm_spec.store.map_names
        md_spec = vm_spec.store.get_map_data("spectrum")
        assert md_spec.n_spectra == 1
        assert md_spec.n_wavenumbers == 1024

        more_tab = VMoreTab()
        proxy_spec = SpectrumProxy(md_spec, 0, "spectrum")
        more_tab.show_metadata(proxy_spec)

        table_keys = [more_tab.table.item(r, 0).text() for r in range(more_tab.table.rowCount())]
        ordered_in_spec = [k for k in _METADATA_ORDERED_KEYS if k in md_spec.map_metadata]
        assert table_keys[: len(ordered_in_spec)] == ordered_in_spec
        assert "Instrument" in table_keys[len(ordered_in_spec) :]
        assert "Spectrograph" in table_keys[len(ordered_in_spec) :]

        api_spec = api_io.load_spectra(ngs_path)
        assert len(api_spec["items"]) == 1
        assert api_spec["items"][0]["metadata"]["File Format"] == "HORIBA LabSpec 5"


class TestTwoTierMultiSelectionAndBatchDelete:
    def test_select_maps_comparison_does_not_reemit_primary(self, vm):
        _load_map(vm, "m1", n_rows_side=2)
        _load_map(vm, "m2", n_rows_side=2)
        _load_map(vm, "m3", n_rows_side=2)
        vm.select_map("m1")

        map_updates = []
        spec_updates = []
        live_sel = []
        vm.map_data_updated.connect(lambda df: map_updates.append(df))
        vm.spectra_selection_changed.connect(lambda s: spec_updates.append(s))
        vm.live_maps_selection_changed.connect(lambda names, prim: live_sel.append((list(names), prim)))

        # Adding m2 and m3 alongside unchanged Primary m1 emits ONLY live_maps_selection_changed
        vm.select_maps(["m1", "m2", "m3"], primary_name="m1")
        assert vm.current_map_name == "m1"
        assert vm.selected_map_names == ["m1", "m2", "m3"]
        assert map_updates == []
        assert spec_updates == []
        assert live_sel == [(["m1", "m2", "m3"], "m1")]

        # Changing Primary to m2 emits map_data_updated AND live_maps_selection_changed
        vm.select_maps(["m1", "m2", "m3"], primary_name="m2")
        assert vm.current_map_name == "m2"
        assert len(map_updates) == 1
        assert live_sel[-1] == (["m1", "m2", "m3"], "m2")

    def test_delete_selected_maps_and_active_version_bump(self, vm):
        m1 = _load_map(vm, "m1", n_rows_side=2)
        _load_map(vm, "m2", n_rows_side=2)
        _load_map(vm, "m3", n_rows_side=2)

        # set_spectrum_active and set_all_current_map_spectra_active bump data_version
        vm.select_map("m1")
        v0 = m1.data_version
        vm.set_spectrum_active(m1.fnames[0], False)
        assert m1.data_version > v0
        v1 = m1.data_version
        vm.set_all_current_map_spectra_active(True)
        assert m1.data_version > v1

        # Batch delete m1 and m3 -> m2 survives and becomes Primary
        live_sel = []
        vm.live_maps_selection_changed.connect(lambda names, prim: live_sel.append((list(names), prim)))
        vm.select_maps(["m1", "m3"], primary_name="m1")
        vm.delete_selected_maps()
        assert list(vm.maps.keys()) == ["m2"]
        assert vm.current_map_name == "m2"
        assert vm.selected_map_names == ["m2"]

        # Delete last remaining map -> empty selection signals
        vm.delete_selected_maps()
        assert vm.maps == {}
        assert vm.current_map_name is None
        assert vm.selected_map_names == []
        assert live_sel[-1] == ([], None)

    def test_save_reload_unfitted_map_send_to_spectra_and_add_peak(self, vm, qapp, tmp_path, monkeypatch):
        from PySide6.QtWidgets import QFileDialog

        _load_map(vm, "m1", n_rows_side=2)
        save_path = tmp_path / "unfitted.maps"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(save_path), ""))
        vm.save_work()

        vm2 = VMWorkspaceMaps(vm.settings)
        vm2.load_work(str(save_path))
        md2 = vm2.store.get_map_data("m1")
        assert md2.fit_model is None

        # Sending selected spectra from a reloaded unfitted map must preserve fit_model=None
        payloads = []
        vm2.send_spectra_to_workspace.connect(payloads.extend)
        vm2.selected_fnames = [md2.fnames[0]]
        vm2.send_selected_spectra_to_spectra_workspace()
        assert len(payloads) == 1
        assert payloads[0]["fit_model"] is None

        # Clicking to add a peak on the reloaded map must succeed cleanly
        vm2.add_peak_at(500.0)
        assert "0" in md2.fit_model["peak_models"]
        assert len(md2.fit_model["peak_labels"]) == 1

    def test_vm_maps_shares_y0_buffer_and_preserves_float32_across_preprocessing_and_fit(self, vm, tmp_path):
        import pandas as pd
        from spectroview.fit_engine.vbf_thread import VBFthread
        from spectroview.model.spectra.spectra_store import LazyPeakArray

        x = np.linspace(450.0, 550.0, 101)
        peak = 100.0 / (1.0 + 4.0 * ((x - 500.0) / 6.0) ** 2)
        rows = np.vstack([peak + 10.0 for _ in range(4)])
        df = pd.DataFrame(rows, columns=[str(v) for v in x])
        df.insert(0, "Y", [0.0, 0.0, 1.0, 1.0])
        df.insert(0, "X", [0.0, 1.0, 0.0, 1.0])

        txt_path = tmp_path / "c_contig_map.csv"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Header\n")
        df.to_csv(txt_path, sep=";", index=False, mode="a")

        vm.load_map_files([str(txt_path)])
        md = vm.store.get_map_data("c_contig_map")
        assert md.Y0.dtype == np.float32
        assert md.Y0.flags["C_CONTIGUOUS"]
        # vm.maps[name] must share md.Y0 float32 buffer instead of holding a float64 copy
        assert np.shares_memory(vm.maps["c_contig_map"].iloc[:, 2:].to_numpy(), md.Y0)

        # Preprocessing & baseline subtraction keep md.Y in float32
        md.baseline_config = {
            "mode": "Linear",
            "attached": True,
            "sigma": 0,
            "points": [[450.0, 550.0], [0.0, 0.0]],
        }
        vm._on_map_data_changed(md, "subtract_baseline")
        vm.subtract_baseline(apply_all=False)
        assert md.Y is not None and md.Y.dtype == np.float32

        # VBFthread stores float32 Y_bestfit, avoids blank Y_baseline alloc, and uses LazyPeakArray
        md.Y_baseline = None
        fit_model = _lorentzian_fit_model(ampli=100.0, fwhm=6.0, x0=500.0)
        tasks = [{"map_name": "c_contig_map", "indices": np.arange(4), "fit_model": fit_model}]
        VBFthread(vm.store, tasks).run()

        assert md.Y_bestfit is not None and md.Y_bestfit.dtype == np.float32
        assert md.Y_baseline is None
        assert md.Y_peaks is not None and len(md.Y_peaks) == 1
        assert isinstance(md.Y_peaks[0], LazyPeakArray)
        sliced = md.Y_peaks[0][np.array([0, 2])]
        assert sliced.dtype == np.float32 and sliced.shape == (2, 101)
        single = md.Y_peaks[0][1]
        assert single.dtype == np.float32 and single.shape == (101,)
        np.testing.assert_allclose(md.Y_bestfit[1], single, rtol=1e-5, atol=1e-5)


class TestSaveMapsToCsv:
    def test_save_selected_maps_to_csv_multiple_maps_and_roundtrip(self, vm, tmp_path):
        from spectroview.model.io.m_io import load_map_file

        md_a = _load_map(vm, "map_A", n_rows_side=2, seed=10)
        _load_map(vm, "map_B", n_rows_side=2, seed=11)
        md_c = _load_map(vm, "map_C", n_rows_side=2, seed=12)

        # Crop map_A only so we verify processed md.x / md.Y export
        vm.select_map("map_A")
        vm.apply_spectral_range(400.0, 600.0, apply_all=False)

        vm.select_maps(["map_A", "map_C"], primary_name="map_A")

        notifications = []
        vm.notify.connect(notifications.append)

        count = vm.save_selected_maps_to_csv(str(tmp_path))

        assert count == 2
        path_a = tmp_path / "map_A.csv"
        path_c = tmp_path / "map_C.csv"
        assert path_a.exists()
        assert path_c.exists()
        assert not (tmp_path / "map_B.csv").exists()

        with open(path_a, encoding="utf-8") as f:
            assert f.readline() == "SPECTROview Map Data\n"
        with open(path_c, encoding="utf-8") as f:
            assert f.readline() == "SPECTROview Map Data\n"

        reloaded_a = load_map_file(path_a)
        reloaded_c = load_map_file(path_c)

        np.testing.assert_allclose(reloaded_a[["X", "Y"]].to_numpy(), md_a.coords)
        assert [float(c) for c in reloaded_a.columns[2:]] == pytest.approx(
            [float(w) for w in md_a.x_axis]
        )
        np.testing.assert_allclose(
            reloaded_a.iloc[:, 2:].to_numpy(), md_a.y_matrix, rtol=1e-5, atol=1e-5
        )

        np.testing.assert_allclose(reloaded_c[["X", "Y"]].to_numpy(), md_c.coords)
        assert [float(c) for c in reloaded_c.columns[2:]] == pytest.approx(
            [float(w) for w in md_c.x_axis]
        )
        np.testing.assert_allclose(
            reloaded_c.iloc[:, 2:].to_numpy(), md_c.y_matrix, rtol=1e-5, atol=1e-5
        )

        assert vm.settings.get_last_directory() == str(tmp_path)
        assert notifications[-1] == "Successfully saved 2 maps."

    def test_save_selected_maps_to_csv_single_map_and_current_fallback(self, vm, tmp_path):
        _load_map(vm, "map_one", n_rows_side=2, seed=20)
        vm.select_map("map_one")
        # Force fallback to current_map_name when selected_map_names is cleared
        vm.selected_map_names = []

        notifications = []
        vm.notify.connect(notifications.append)

        count = vm.save_selected_maps_to_csv(str(tmp_path))

        assert count == 1
        assert (tmp_path / "map_one.csv").exists()
        assert notifications[-1] == "Successfully saved 1 map."

    def test_save_selected_maps_to_csv_no_selection_notifies(self, vm, tmp_path):
        out_dir = tmp_path / "exports"
        out_dir.mkdir()
        notifications = []
        vm.notify.connect(notifications.append)

        count = vm.save_selected_maps_to_csv(str(out_dir))

        assert count == 0
        assert list(out_dir.iterdir()) == []
        assert notifications == ["No map selected."]

    def test_save_selected_maps_to_csv_sanitizes_and_deduplicates_filenames(self, vm, tmp_path):
        from spectroview.model.io.m_io import load_map_file

        md1 = _load_map(vm, "sample/1", n_rows_side=2, seed=31)
        md2 = _load_map(vm, "sample:1", n_rows_side=2, seed=32)
        vm.select_maps(["sample/1", "sample:1"], primary_name="sample/1")

        count = vm.save_selected_maps_to_csv(str(tmp_path))

        assert count == 2
        p1 = tmp_path / "sample1.csv"
        p2 = tmp_path / "sample1_2.csv"
        assert p1.exists() and p2.exists()

        df1 = load_map_file(p1)
        df2 = load_map_file(p2)
        np.testing.assert_allclose(df1.iloc[:, 2:].to_numpy(), md1.y_matrix, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(df2.iloc[:, 2:].to_numpy(), md2.y_matrix, rtol=1e-5, atol=1e-5)

    def test_save_selected_maps_to_csv_partial_error_continues(self, vm, tmp_path):
        _load_map(vm, "map_broken", n_rows_side=2, seed=41)
        _load_map(vm, "map_ok", n_rows_side=2, seed=42)
        vm.select_maps(["map_broken", "map_ok"], primary_name="map_ok")

        # Remove map_broken from store only so export fails for it while map_ok succeeds
        vm.store.remove_map("map_broken")

        notifications = []
        vm.notify.connect(notifications.append)

        count = vm.save_selected_maps_to_csv(str(tmp_path))

        assert count == 1
        assert not (tmp_path / "map_broken.csv").exists()
        assert (tmp_path / "map_ok.csv").exists()
        assert any("Error saving map_broken.csv" in msg for msg in notifications)
        assert notifications[-1] == "Successfully saved 1 map."

    def test_save_current_map_to_csv_backward_compatibility(self, vm, tmp_path):
        from spectroview.model.io.m_io import load_map_file

        md = _load_map(vm, "map_A", n_rows_side=2, seed=50)
        vm.select_map("map_A")

        notifications = []
        vm.notify.connect(notifications.append)

        out_base = tmp_path / "single"
        vm.save_current_map_to_csv(str(out_base))

        out_csv = tmp_path / "single.csv"
        assert out_csv.exists()
        reloaded = load_map_file(out_csv)
        np.testing.assert_allclose(
            reloaded.iloc[:, 2:].to_numpy(), md.y_matrix, rtol=1e-5, atol=1e-5
        )
        assert notifications[-1] == "Map 'map_A' saved to CSV."

