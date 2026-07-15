"""Unit tests for model/workspace_io.py - WorkspaceIO save/load (ZIP format v2)."""
import json

import numpy as np
import pandas as pd
import pytest

from spectroview.model.workspace_io import WorkspaceIO


class TestSaveLoadRoundTrip:
    def test_metadata_only(self, tmp_path):
        path = tmp_path / "work.spectra"
        metadata = {"format_version": 2, "store_meta": {"foo": "bar"}}
        WorkspaceIO.save_workspace(str(path), metadata)

        loaded_meta, arrays, dataframes, is_legacy = WorkspaceIO.load_workspace(str(path))
        assert is_legacy is False
        assert loaded_meta == metadata
        assert arrays == {}
        assert dataframes == {}

    def test_arrays_round_trip(self, tmp_path):
        path = tmp_path / "work.maps"
        arrays = {
            "store_map1_x0": np.linspace(0, 1, 50),
            "store_map1_y0": np.random.default_rng(0).normal(size=(10, 50)).astype(np.float32),
            "store_map1_is_active": np.ones(10, dtype=bool),
        }
        WorkspaceIO.save_workspace(str(path), {"format_version": 2}, arrays=arrays)

        _, loaded_arrays, _, is_legacy = WorkspaceIO.load_workspace(str(path))
        assert not is_legacy
        assert set(loaded_arrays.keys()) == set(arrays.keys())
        for k in arrays:
            np.testing.assert_array_equal(loaded_arrays[k], arrays[k])

    def test_dataframes_round_trip(self, tmp_path):
        path = tmp_path / "work.spectra"
        df = pd.DataFrame({"Filename": ["a", "b"], "x0_P1": [500.1, 501.2], "fwhm_P1": [5.0, 6.0]})
        WorkspaceIO.save_workspace(str(path), {"format_version": 2}, dataframes={"df_fit_results": df})

        _, _, loaded_dfs, is_legacy = WorkspaceIO.load_workspace(str(path))
        assert not is_legacy
        pd.testing.assert_frame_equal(loaded_dfs["df_fit_results"], df)

    def test_full_round_trip(self, tmp_path):
        path = tmp_path / "full.maps"
        metadata = {"format_version": 2, "store_meta": {"map1": {"fnames": ["a", "b"]}}}
        arrays = {"store_map1_x0": np.array([1.0, 2.0, 3.0])}
        df = pd.DataFrame({"A": [1, 2, 3]})
        WorkspaceIO.save_workspace(str(path), metadata, arrays=arrays, dataframes={"d": df})

        loaded_meta, loaded_arrays, loaded_dfs, is_legacy = WorkspaceIO.load_workspace(str(path))
        assert loaded_meta == metadata
        np.testing.assert_array_equal(loaded_arrays["store_map1_x0"], arrays["store_map1_x0"])
        pd.testing.assert_frame_equal(loaded_dfs["d"], df)

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "dirs" / "work.spectra"
        WorkspaceIO.save_workspace(str(path), {"format_version": 2})
        assert path.exists()


class TestLegacyDetection:
    def test_legacy_json_file_is_detected(self, tmp_path):
        path = tmp_path / "legacy.spectra"
        path.write_text(json.dumps({"spectrums": {}}))
        metadata, arrays, dataframes, is_legacy = WorkspaceIO.load_workspace(str(path))
        assert is_legacy is True
        assert metadata is None and arrays is None and dataframes is None

    def test_real_legacy_spectra_file(self, legacy_spectra_workspace):
        if not legacy_spectra_workspace.exists():
            pytest.skip("example legacy .spectra file not present")
        _, _, _, is_legacy = WorkspaceIO.load_workspace(str(legacy_spectra_workspace))
        assert is_legacy is True

    def test_real_zip_maps_file(self, zip_maps_workspace):
        if not zip_maps_workspace.exists():
            pytest.skip("example .maps file not present")
        metadata, arrays, dataframes, is_legacy = WorkspaceIO.load_workspace(str(zip_maps_workspace))
        assert is_legacy is False
        assert isinstance(metadata, dict)


class TestMissingFile:
    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            WorkspaceIO.load_workspace(str(tmp_path / "does_not_exist.spectra"))
