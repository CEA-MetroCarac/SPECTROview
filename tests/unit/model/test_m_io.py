"""Unit tests for model/m_io.py - file loading functions.

Covers single-spectrum loading (.txt/.csv/.wdf/.spc), 2D map loading
(.txt/.csv/.wdf), delimiter auto-detection, and DataFrame loading.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spectroview.model.m_io import (
    load_spectrum_file,
    load_map_file,
    load_dataframe_file,
    load_wdf_map,
)


class TestLoadSpectrumFile:
    def test_load_txt_spectrum(self, single_spectrum_file):
        spectrum = load_spectrum_file(single_spectrum_file)
        item = spectrum["items"][0]
        assert len(item["x0"]) > 0
        assert len(item["x0"]) == len(item["y0"])
        assert item["name"] == single_spectrum_file.stem

    def test_load_multiple_spectra(self, multiple_spectra_files):
        spectra = [load_spectrum_file(p) for p in multiple_spectra_files]
        assert len(spectra) == len(multiple_spectra_files)
        for spectrum, file_path in zip(spectra, multiple_spectra_files):
            assert spectrum["items"][0]["name"] == file_path.stem
            assert len(spectrum["items"][0]["x0"]) > 0

    def test_spectrum_data_is_sorted_ascending(self, single_spectrum_file):
        spectrum = load_spectrum_file(single_spectrum_file)
        x0 = spectrum["items"][0]["x0"]
        assert np.all(x0[:-1] <= x0[1:])

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_spectrum_file(Path("nonexistent_file.txt"))

    def test_load_unsupported_extension_raises(self, tmp_path):
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("100 200\n200 300\n")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_spectrum_file(bad_file)


class TestDelimiterDetection:
    def test_semicolon(self, tmp_path):
        txt = tmp_path / "semicolon.txt"
        txt.write_text("Wavenumber;Intensity\n100;200\n150;250\n200;300\n")
        spectrum = load_spectrum_file(txt)
        np.testing.assert_array_equal(spectrum["items"][0]["x0"], [100, 150, 200])
        np.testing.assert_array_equal(spectrum["items"][0]["y0"], [200, 250, 300])

    def test_tab(self, tmp_path):
        txt = tmp_path / "tab.txt"
        txt.write_text("Wavenumber\tIntensity\n100\t200\n150\t250\n200\t300\n")
        spectrum = load_spectrum_file(txt)
        np.testing.assert_array_equal(spectrum["items"][0]["x0"], [100, 150, 200])

    def test_space(self, tmp_path):
        txt = tmp_path / "space.txt"
        txt.write_text("Wavenumber Intensity\n100 200\n150 250\n200 300\n")
        spectrum = load_spectrum_file(txt)
        np.testing.assert_array_equal(spectrum["items"][0]["x0"], [100, 150, 200])


class TestLoadMapFile:
    def test_load_txt_map(self, map_2d_file):
        map_df = load_map_file(map_2d_file)
        assert isinstance(map_df, pd.DataFrame)
        assert "X" in map_df.columns and "Y" in map_df.columns
        assert len(map_df.columns) > 2
        assert len(map_df) > 0

    def test_load_csv_wafer_file(self, wafer_file):
        map_df = load_map_file(wafer_file)
        assert isinstance(map_df, pd.DataFrame)
        assert "X" in map_df.columns and "Y" in map_df.columns
        assert len(map_df.columns) > 2

    def test_map_wavelength_columns_are_numeric(self, map_2d_file):
        map_df = load_map_file(map_2d_file)
        wavelength_cols = [c for c in map_df.columns if c not in ("X", "Y")]
        for col in wavelength_cols:
            float(col)  # raises ValueError if not parseable

    def test_map_columns_sorted_by_increasing_wavenumber(self, map_2d_file):
        map_df = load_map_file(map_2d_file)
        wavelength_cols = [float(c) for c in map_df.columns if c not in ("X", "Y")]
        assert wavelength_cols == sorted(wavelength_cols)

    def test_load_unsupported_map_extension_raises(self, tmp_path):
        bad_file = tmp_path / "map.xyz"
        bad_file.write_text("X\tY\t100\t200\n1\t1\t50\t60\n")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_map_file(bad_file)


class TestLoadWdfMap:
    def test_load_real_wdf_map(self, wdf_map_file):
        if not wdf_map_file.exists():
            pytest.skip("WDF benchmark file not present")
        map_df, metadata = load_wdf_map(wdf_map_file)
        assert isinstance(map_df, pd.DataFrame)
        assert "X" in map_df.columns and "Y" in map_df.columns
        assert len(map_df.columns) > 2
        assert isinstance(metadata, dict)

    def test_wdf_wavenumbers_ascending(self, wdf_map_file):
        if not wdf_map_file.exists():
            pytest.skip("WDF benchmark file not present")
        map_df, _ = load_wdf_map(wdf_map_file)
        wavelength_cols = [float(c) for c in map_df.columns if c not in ("X", "Y")]
        assert wavelength_cols == sorted(wavelength_cols)


class TestLoadDataFrameFile:
    def test_load_excel(self, dataframe_excel_file):
        if not dataframe_excel_file.exists():
            pytest.skip("example Excel file not present")
        dfs = load_dataframe_file(dataframe_excel_file)
        assert isinstance(dfs, dict) and len(dfs) > 0
        for df in dfs.values():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0

    def test_load_csv_dataframe(self, tmp_path):
        csv_file = tmp_path / "test_data.csv"
        test_data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        test_data.to_csv(csv_file, index=False)

        dfs = load_dataframe_file(csv_file)
        assert len(dfs) == 1 and "test_data" in dfs
        pd.testing.assert_frame_equal(dfs["test_data"], test_data)

    def test_load_unsupported_extension_raises(self, tmp_path):
        bad = tmp_path / "data.txt"
        bad.write_text("A,B,C\n1,2,3\n")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_dataframe_file(bad)
