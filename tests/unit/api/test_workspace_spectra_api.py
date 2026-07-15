"""Tests for spectroview.api.workspace.SpectraWorkspace."""
import numpy as np
import pandas as pd
import pytest

from spectroview.api import workspace
from spectroview.api.exceptions import FitModelError, WorkspaceError
from spectroview.model.workspace_io import WorkspaceIO


class TestErgonomics:
    def test_len_repr_names_on_empty_workspace(self):
        ws = workspace.SpectraWorkspace()
        assert len(ws) == 0
        assert ws.names == []
        assert "SpectraWorkspace" in repr(ws)

    def test_len_repr_names_after_loading(self, multiple_spectra_files):
        ws = workspace.SpectraWorkspace()
        ws.load_files(multiple_spectra_files)
        assert len(ws) == 3
        assert set(ws.names) == set(ws.store.map_names)
        assert repr(ws).startswith("SpectraWorkspace(3 spectra")


class TestLoadFiles:
    def test_loads_each_file_as_one_spectrum(self, multiple_spectra_files):
        ws = workspace.SpectraWorkspace()
        added = ws.load_files(multiple_spectra_files)
        assert len(added) == 3
        for name in added:
            md = ws.store.get_map_data(name)
            assert md.n_spectra == 1

    def test_reloading_same_file_is_skipped(self, single_spectrum_file):
        ws = workspace.SpectraWorkspace()
        first = ws.load_files([single_spectrum_file])
        second = ws.load_files([single_spectrum_file])
        assert len(first) == 1
        assert second == []
        assert len(ws) == 1

    def test_single_path_not_wrapped_in_list_also_works(self, single_spectrum_file):
        ws = workspace.SpectraWorkspace()
        added = ws.load_files(single_spectrum_file)
        assert len(added) == 1


class TestPreprocessing:
    def _make_ws(self, multiple_spectra_files):
        ws = workspace.SpectraWorkspace()
        ws.load_files(multiple_spectra_files)
        return ws

    def test_crop_limits_all_spectra(self, multiple_spectra_files):
        ws = self._make_ws(multiple_spectra_files)
        ws.crop(range_min=400.0, range_max=600.0)
        for name in ws.names:
            x, _ = ws.get_xy(name)
            assert x.min() >= 400.0
            assert x.max() <= 600.0

    def test_crop_unknown_name_raises(self, multiple_spectra_files):
        ws = self._make_ws(multiple_spectra_files)
        with pytest.raises(WorkspaceError):
            ws.crop(range_min=1.0, names=["does_not_exist"])

    def test_normalize_default_scales_each_to_its_own_max(self, multiple_spectra_files):
        ws = self._make_ws(multiple_spectra_files)
        ws.normalize()
        for name in ws.names:
            _, y = ws.get_xy(name)
            assert np.isclose(y.max(), 1.0)

    def test_normalize_with_shared_factor(self, multiple_spectra_files):
        ws = self._make_ws(multiple_spectra_files)
        _, y_before = ws.get_xy(ws.names[0])
        ws.normalize(factor=2.0)
        _, y_after = ws.get_xy(ws.names[0])
        assert np.allclose(y_after, y_before / 2.0)

    def test_set_and_subtract_baseline(self, multiple_spectra_files):
        ws = self._make_ws(multiple_spectra_files)
        ws.set_baseline({"mode": "Polynomial", "order_max": 1})
        md = ws.store.get_map_data(ws.names[0])
        assert md.Y_baseline is not None
        ws.subtract_baseline()
        md = ws.store.get_map_data(ws.names[0])
        assert md.Y_baseline is None
        assert bool(np.asarray(md.is_baseline_subtracted).all())

    def test_subtract_without_configured_baseline_raises(self, multiple_spectra_files):
        ws = self._make_ws(multiple_spectra_files)
        with pytest.raises(WorkspaceError):
            ws.subtract_baseline()

    def test_reinit_clears_crop_and_baseline_state(self, multiple_spectra_files):
        ws = self._make_ws(multiple_spectra_files)
        ws.crop(range_min=450.0, range_max=550.0)
        ws.reinit()
        for name in ws.names:
            x, y = ws.get_xy(name)
            md = ws.store.get_map_data(name)
            assert np.array_equal(x, md.x0)
            assert md.range_min is None and md.range_max is None


class TestFitModelAndFit:
    def test_set_fit_model_then_fit_populates_results(self, multiple_spectra_files, make_fit_model):
        ws = workspace.SpectraWorkspace()
        ws.load_files(multiple_spectra_files)
        name = ws.names[0]
        _, y = ws.get_xy(name)
        peak_idx = int(np.argmax(y))
        x, _ = ws.get_xy(name)
        fm = make_fit_model([("Lorentzian", dict(x0=float(x[peak_idx]), ampli=float(y[peak_idx]), fwhm=5.0))])

        ws.set_fit_model(fm, names=[name])
        md = ws.store.get_map_data(name)
        assert md.fit_model is not None
        assert md.fit_success is None  # not fitted yet

        ws.fit(names=[name])
        md = ws.store.get_map_data(name)
        assert md.has_fit_results()
        assert md.fit_r2 is not None

    def test_fit_without_fit_model_raises(self, multiple_spectra_files):
        ws = workspace.SpectraWorkspace()
        ws.load_files(multiple_spectra_files)
        with pytest.raises(FitModelError):
            ws.fit(names=[ws.names[0]])

    def test_fit_default_targets_only_spectra_with_fit_model(self, multiple_spectra_files, make_fit_model):
        ws = workspace.SpectraWorkspace()
        ws.load_files(multiple_spectra_files)
        name0 = ws.names[0]
        x, y = ws.get_xy(name0)
        fm = make_fit_model([("Lorentzian", dict(x0=float(x[len(x) // 2]), ampli=float(y.max()), fwhm=5.0))])
        ws.set_fit_model(fm, names=[name0])

        ws.fit()  # no names -> only name0 should be attempted
        md0 = ws.store.get_map_data(name0)
        assert md0.has_fit_results()


class TestCollectResults:
    def test_collect_results_drops_x_y_columns(self, multiple_spectra_files, make_fit_model):
        ws = workspace.SpectraWorkspace()
        ws.load_files(multiple_spectra_files)
        for name in ws.names:
            x, y = ws.get_xy(name)
            fm = make_fit_model([("Lorentzian", dict(x0=float(x[len(x) // 2]), ampli=float(y.max()), fwhm=5.0))])
            ws.set_fit_model(fm, names=[name])
        ws.fit()

        df = ws.collect_results()
        assert isinstance(df, pd.DataFrame)
        assert "Filename" in df.columns
        assert "X" not in df.columns and "Y" not in df.columns
        assert ws.get_results_dataframe() is df

    def test_collect_results_with_nothing_fitted_returns_none(self, multiple_spectra_files):
        ws = workspace.SpectraWorkspace()
        ws.load_files(multiple_spectra_files)
        assert ws.collect_results() is None
        assert ws.get_results_dataframe() is None


class TestRemove:
    def test_remove_drops_spectra(self, multiple_spectra_files):
        ws = workspace.SpectraWorkspace()
        ws.load_files(multiple_spectra_files)
        target = ws.names[0]
        ws.remove([target])
        assert target not in ws.names
        assert len(ws) == 2


class TestSaveLoad:
    def test_round_trip_preserves_store_and_results(self, multiple_spectra_files, make_fit_model, tmp_path):
        ws = workspace.SpectraWorkspace()
        ws.load_files(multiple_spectra_files)
        for name in ws.names:
            x, y = ws.get_xy(name)
            fm = make_fit_model([("Lorentzian", dict(x0=float(x[len(x) // 2]), ampli=float(y.max()), fwhm=5.0))])
            ws.set_fit_model(fm, names=[name])
        ws.fit()
        df_before = ws.collect_results()

        path = tmp_path / "roundtrip.spectra"
        ws.save(path)
        assert path.exists()

        ws2 = workspace.SpectraWorkspace.load(path)
        assert set(ws2.names) == set(ws.names)
        df_after = ws2.get_results_dataframe()
        pd.testing.assert_frame_equal(
            df_before.sort_values("Filename").reset_index(drop=True),
            df_after.sort_values("Filename").reset_index(drop=True),
        )

    def test_saved_file_has_format_version_2_structure(self, multiple_spectra_files, tmp_path):
        ws = workspace.SpectraWorkspace()
        ws.load_files(multiple_spectra_files)
        path = tmp_path / "structural.spectra"
        ws.save(path)

        metadata, arrays, dataframes, is_legacy = WorkspaceIO.load_workspace(str(path))
        assert is_legacy is False
        assert metadata["format_version"] == 2
        assert set(metadata["store_meta"].keys()) == set(ws.names)

    def test_loading_legacy_v1_file_raises_workspace_error(self, legacy_spectra_workspace):
        if not legacy_spectra_workspace.exists():
            pytest.skip("legacy .spectra fixture not present")
        with pytest.raises(WorkspaceError):
            workspace.SpectraWorkspace.load(legacy_spectra_workspace)

    def test_loading_missing_file_raises_workspace_error(self, tmp_path):
        with pytest.raises(WorkspaceError):
            workspace.SpectraWorkspace.load(tmp_path / "does_not_exist.spectra")
