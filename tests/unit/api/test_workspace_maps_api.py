"""Tests for spectroview.api.workspace.MapsWorkspace."""
import numpy as np
import pandas as pd
import pytest

from spectroview.api import workspace, fitting
from spectroview.api.exceptions import FitModelError, WorkspaceError


class TestLoadFiles:
    def test_loads_map_with_pixel_fname_convention(self, map_2d_file):
        ws = workspace.MapsWorkspace()
        added = ws.load_files([map_2d_file])
        assert len(added) == 1
        map_name = added[0]
        md = ws.store.get_map_data(map_name)
        assert md.n_spectra > 1
        assert all(f.startswith(f"{map_name}_(") for f in md.fnames)
        assert map_name in ws.maps

    def test_reloading_same_map_is_skipped(self, map_2d_file):
        ws = workspace.MapsWorkspace()
        first = ws.load_files([map_2d_file])
        second = ws.load_files([map_2d_file])
        assert len(first) == 1
        assert second == []


class TestFit:
    def test_fit_whole_map_and_collect_results(self, map_2d_file, fit_model_si_file):
        if not fit_model_si_file.exists():
            pytest.skip("fit_model_Si_.json fixture not present")
        ws = workspace.MapsWorkspace()
        [map_name] = ws.load_files([map_2d_file])
        fm = fitting.load_fit_model_template(fit_model_si_file)

        ws.set_fit_model(fm, names=[map_name])
        ws.fit(map_names=[map_name])

        md = ws.store.get_map_data(map_name)
        assert md.has_fit_results()
        assert bool(md.fit_success.any())

        df = ws.collect_results()
        assert isinstance(df, pd.DataFrame)
        assert "X" in df.columns and "Y" in df.columns  # unlike SpectraWorkspace, maps keep coords
        assert len(df) == md.n_spectra or len(df) <= md.n_spectra  # only_converged default is True here

    def test_fit_without_fit_model_raises(self, map_2d_file):
        ws = workspace.MapsWorkspace()
        [map_name] = ws.load_files([map_2d_file])
        with pytest.raises(FitModelError):
            ws.fit(map_names=[map_name])

    def test_fit_unknown_map_raises(self, map_2d_file):
        ws = workspace.MapsWorkspace()
        ws.load_files([map_2d_file])
        with pytest.raises(WorkspaceError):
            ws.fit(map_names=["not_a_real_map"])


class TestHeatmapAndProfile:
    @pytest.fixture
    def fitted_ws(self, map_2d_file, fit_model_si_file):
        if not fit_model_si_file.exists():
            pytest.skip("fit_model_Si_.json fixture not present")
        ws = workspace.MapsWorkspace()
        [map_name] = ws.load_files([map_2d_file])
        fm = fitting.load_fit_model_template(fit_model_si_file)
        ws.set_fit_model(fm, names=[map_name])
        ws.fit(map_names=[map_name])
        ws.collect_results()
        return ws, map_name

    def test_intensity_heatmap_matches_pivot_shape(self, map_2d_file):
        ws = workspace.MapsWorkspace()
        [map_name] = ws.load_files([map_2d_file])
        map_df = ws.maps[map_name]
        xi, yi, zi = ws.get_heatmap(map_name, "Intensity")
        assert zi.shape == (len(yi), len(xi))
        assert len(xi) == map_df["X"].nunique()
        assert len(yi) == map_df["Y"].nunique()

    def test_fit_param_heatmap_requires_collected_results(self, map_2d_file):
        ws = workspace.MapsWorkspace()
        [map_name] = ws.load_files([map_2d_file])
        with pytest.raises(WorkspaceError):
            ws.get_heatmap(map_name, "ampli_Si")

    def test_fit_param_heatmap_after_collect_results(self, fitted_ws):
        ws, map_name = fitted_ws
        value_col = [c for c in ws.df_fit_results.columns if "ampli" in c][0]
        xi, yi, zi = ws.get_heatmap(map_name, value_col)
        assert zi.shape == (len(yi), len(xi))

    def test_unknown_map_raises(self, map_2d_file):
        ws = workspace.MapsWorkspace()
        ws.load_files([map_2d_file])
        with pytest.raises(WorkspaceError):
            ws.get_heatmap("not_a_real_map", "Intensity")

    def test_extract_profile_between_corners(self, map_2d_file):
        ws = workspace.MapsWorkspace()
        [map_name] = ws.load_files([map_2d_file])
        xi, yi, zi = ws.get_heatmap(map_name, "Intensity")
        profile = ws.extract_profile(map_name, "Intensity", (xi.min(), yi.min()), (xi.max(), yi.max()), n_samples=15)
        assert list(profile.columns) == ["X", "Y", "distance", "values"]
        assert len(profile) == 15
        assert np.all(np.diff(profile["distance"]) > 0)  # monotonically increasing


class TestWaferZoneQuadrant:
    def test_wafer_map_type_adds_zone_and_quadrant_columns(self, wafer_file, fit_model_si_file):
        if not wafer_file.exists() or not fit_model_si_file.exists():
            pytest.skip("wafer fixture files not present")
        ws = workspace.MapsWorkspace(map_type="wafer_300mm")
        [map_name] = ws.load_files([wafer_file])
        fm = fitting.load_fit_model_template(fit_model_si_file)
        ws.set_fit_model(fm, names=[map_name])
        ws.fit(map_names=[map_name])
        df = ws.collect_results()
        assert "Zone" in df.columns
        assert "Quadrant" in df.columns
        assert set(df["Quadrant"].dropna().unique()).issubset({"Q1", "Q2", "Q3", "Q4"})


class TestRemove:
    def test_remove_drops_map_and_metadata(self, map_2d_file):
        ws = workspace.MapsWorkspace()
        [map_name] = ws.load_files([map_2d_file])
        ws.remove([map_name])
        assert map_name not in ws.names
        assert map_name not in ws.maps
        assert map_name not in ws.maps_metadata


class TestSaveLoad:
    def test_round_trip_preserves_maps_metadata_and_results(self, map_2d_file, fit_model_si_file, tmp_path):
        if not fit_model_si_file.exists():
            pytest.skip("fit_model_Si_.json fixture not present")
        ws = workspace.MapsWorkspace()
        [map_name] = ws.load_files([map_2d_file])
        fm = fitting.load_fit_model_template(fit_model_si_file)
        ws.set_fit_model(fm, names=[map_name])
        ws.fit(map_names=[map_name])
        df_before = ws.collect_results()

        path = tmp_path / "roundtrip.maps"
        ws.save(path)
        ws2 = workspace.MapsWorkspace.load(path)

        assert ws2.map_type == ws.map_type
        assert set(ws2.maps.keys()) == {map_name}
        pd.testing.assert_frame_equal(
            df_before.reset_index(drop=True), ws2.get_results_dataframe().reset_index(drop=True),
        )

    def test_loading_real_gui_saved_maps_workspace(self, zip_maps_workspace):
        if not zip_maps_workspace.exists():
            pytest.skip("zip_maps_workspace fixture not present")
        ws = workspace.MapsWorkspace.load(zip_maps_workspace)
        assert len(ws) > 0
        assert ws.map_type != ""
        for name in ws.names:
            assert name in ws.maps
