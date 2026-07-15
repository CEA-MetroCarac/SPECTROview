"""Tests for spectroview.api.io -- file loading, dataset import/export, and
the headless Renishaw map converter."""
import numpy as np
import pandas as pd
import pytest

from spectroview.api import io
from spectroview.api.exceptions import LoadError


class TestLoadSpectra:
    def test_load_single_spectrum_file(self, single_spectrum_file):
        data = io.load_spectra(single_spectrum_file)
        assert "items" in data
        assert len(data["items"]) >= 1
        item = data["items"][0]
        assert item["x0"].shape == item["y0"].shape

    def test_load_unsupported_extension_raises(self, tmp_path):
        bogus = tmp_path / "spectrum.xyz"
        bogus.write_text("junk")
        with pytest.raises(LoadError):
            io.load_spectra(bogus)


class TestLoadMap:
    def test_load_2d_map_file(self, map_2d_file):
        df = io.load_map(map_2d_file)
        assert isinstance(df, pd.DataFrame)
        assert "X" in df.columns and "Y" in df.columns

    def test_load_unsupported_extension_raises(self, tmp_path):
        bogus = tmp_path / "map.xyz"
        bogus.write_text("junk")
        with pytest.raises(LoadError):
            io.load_map(bogus)


class TestLoadDataset:
    def test_load_excel_dataset(self, dataframe_excel_file):
        if not dataframe_excel_file.exists():
            pytest.skip("dataset_Excel.xlsx not present")
        data = io.load_dataset(dataframe_excel_file)
        assert isinstance(data, dict)
        assert all(isinstance(v, pd.DataFrame) for v in data.values())


class TestExportResults:
    def test_export_dataframe_to_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        out = io.export_results(df, tmp_path / "out.csv")
        assert out.exists()
        reloaded = pd.read_csv(out, sep=";")
        assert list(reloaded.columns) == ["a", "b"]

    def test_export_list_of_dicts_to_excel(self, tmp_path):
        out = io.export_results([{"a": 1}, {"a": 2}], tmp_path / "out.xlsx")
        assert out.exists()
        reloaded = pd.read_excel(out)
        assert len(reloaded) == 2

    def test_export_unsupported_extension_raises(self, tmp_path):
        with pytest.raises(LoadError):
            io.export_results(pd.DataFrame({"a": [1]}), tmp_path / "out.bogus")


class TestLoadSpectraToMatrix:
    def test_stacks_multiple_files_onto_shared_axis(self, multiple_spectra_files):
        data = io.load_spectra_to_matrix(multiple_spectra_files)
        assert data["Y"].shape[0] == len(data["names"])
        assert data["Y"].shape[1] == data["x"].shape[0]
        assert len(data["metadata"]) == len(data["names"])

    def test_single_path_not_in_a_list_also_works(self, single_spectrum_file):
        data = io.load_spectra_to_matrix(single_spectrum_file)
        assert data["Y"].shape[0] == len(data["names"])

    def test_no_spectra_loaded_raises(self, tmp_path):
        with pytest.raises(LoadError):
            io.load_spectra_to_matrix([])


class TestConvertRenishawMap:
    @pytest.fixture
    def renishaw_longformat_txt(self, tmp_path):
        """A tiny synthetic Renishaw InVia long-format export: 2x2 spatial
        points, 3 wavenumbers each."""
        rows = []
        waves = [500.0, 510.0, 520.0]
        for yi in (0, 1):
            for xi in (0, 1):
                for wi, w in enumerate(waves):
                    rows.append({"#X": xi, "#Y": yi, "#Wave": w, "#Intensity": 100.0 + wi + xi + yi})
        df = pd.DataFrame(rows)
        path = tmp_path / "raw_export.txt"
        df.to_csv(path, sep="\t", index=False)
        return path

    def test_converts_and_is_reloadable_as_a_map(self, renishaw_longformat_txt, tmp_path):
        out_path = tmp_path / "converted.txt"
        result = io.convert_renishaw_map(renishaw_longformat_txt, out_path)
        assert result == out_path
        assert out_path.exists()

        df = io.load_map(out_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # 2x2 spatial points

    def test_invalid_input_raises_load_error(self, tmp_path):
        bad = tmp_path / "not_renishaw.txt"
        bad.write_text("this,is,not,the,right,format\n1,2,3,4,5,6\n")
        with pytest.raises(LoadError):
            io.convert_renishaw_map(bad, tmp_path / "out.txt")
