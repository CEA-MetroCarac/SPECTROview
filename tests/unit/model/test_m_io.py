"""Unit tests for model/m_io.py and model/m_spc.py - file detection and loading pipeline.

Covers:
- MVVM purity of Model-layer I/O modules
- Fast header/sample file classification (detect_file_type, peek_wdf_measurement_type, SpcReader.peek_header)
- 1D spectrum loading (.txt, .csv, .wdf single & temporal series, .spc Raman & PL, .dat TRPL)
- 2D / line-scan / wafer / time-resolved map loading (.txt, .csv, .wdf, .spc) and extract_map_arrays
- Renishaw WiRE long-format converter (convert_action)
- Binary SPC edge cases (header_only, 16-bit integer scaling, non-square fallback grid, corrupt data)
"""
import ast
from pathlib import Path
import struct

import numpy as np
import pandas as pd
import pytest

from spectroview.model.m_io import (
    _METADATA_ORDERED_KEYS,
    convert_action,
    detect_file_type,
    extract_map_arrays,
    load_TRPL_data,
    load_dataframe_file,
    load_l6m_map,
    load_l6s_spectrum,
    load_ngc_map,
    load_ngs_spectrum,
    load_map_file,
    load_spc_map,
    load_spc_spectrum,
    load_spectrum_file,
    load_wdf_map,
    load_wdf_spectrum,
    peek_wdf_measurement_type,
)
from spectroview.model.m_l6s import L6sReader
from spectroview.model.m_ngs import NgsReader
from spectroview.model.m_spc import SpcReader
from spectroview.model.m_wdf import (
    WdfDataType,
    WdfMeasurementType,
    WdfReader,
    WdfScanType,
    WdfUnitType,
)
from spectroview.viewmodel.utils import parse_coords_from_fname


class TestLoadSpectrumFile:
    def test_load_txt_spectrum(self, single_spectrum_file):
        spectrum = load_spectrum_file(single_spectrum_file)
        assert spectrum["type"] == "spectra"
        assert len(spectrum["items"]) == 1
        item = spectrum["items"][0]
        assert item["name"] == single_spectrum_file.stem
        assert isinstance(item["x0"], np.ndarray)
        assert isinstance(item["y0"], np.ndarray)
        assert len(item["x0"]) > 100
        assert item["x0"].shape == item["y0"].shape
        assert np.all(np.isfinite(item["x0"]))
        assert np.all(np.isfinite(item["y0"]))
        assert isinstance(item["metadata"], dict)

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

    def test_load_csv_old_format_alternating_rows(self, tmp_path):
        csv_file = tmp_path / "old_format_map.csv"
        # Line 0: title, Line 1: X;Y;, Line 2: coords + wavenumbers, Line 3: coords + intensities
        content = (
            "Map Title\n"
            "X;Y;\n"
            "1.000;1.000;0.500;1.000;1.500;2.000;\n"
            "1.000;1.000;10.0;20.0;30.0;40.0;\n"
            "2.000;1.000;0.500;1.000;1.500;2.000;\n"
            "2.000;1.000;15.0;25.0;35.0;45.0;\n"
        )
        csv_file.write_text(content)
        map_df = load_map_file(csv_file)
        assert isinstance(map_df, pd.DataFrame)
        assert list(map_df.columns[:2]) == ["X", "Y"]
        # Wavenumber 1.0 should NOT collide with X=1.0 or Y=1.0 (no '.1' or '.2' suffix)
        assert "1.0" in map_df.columns
        assert "1.0.1" not in map_df.columns
        assert "1.0.2" not in map_df.columns
        assert len(map_df) == 2
        assert map_df["X"].tolist() == [1.0, 2.0]
        assert map_df["Y"].tolist() == [1.0, 1.0]
        assert map_df["1.0"].tolist() == [20.0, 25.0]

    def test_load_csv_new_format_with_trailing_semicolon(self, tmp_path):
        csv_file = tmp_path / "new_format_map.csv"
        content = (
            "Wafer Map\n"
            "X;Y;100.0;200.0;300.0;\n"
            "1.0;2.0;10.0;20.0;30.0;\n"
            "3.0;4.0;40.0;50.0;60.0;\n"
        )
        csv_file.write_text(content)
        map_df = load_map_file(csv_file)
        assert isinstance(map_df, pd.DataFrame)
        assert list(map_df.columns[:2]) == ["X", "Y"]
        assert list(map_df.columns[2:5]) == ["100.0", "200.0", "300.0"]
        assert len(map_df) == 2

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


class TestMapSpcMetadata:
    def test_map_spc_metadata_t64000(self):
        from spectroview.model.m_io import _map_spc_metadata
        log_meta = {
            "INSTRUMENT": "T64000",
            "TIME": "10",
            "ACCUM": "2",
            "BINNING": "1",
            "SPECTRO": "599.991",
            "HOLE": "200",
            "EXCIT_LINE": "354.7",
            "GRATING": "2400",
            "OBJECTIV": "X100",
            "DATE": "10/02/26 16:23:36",
            "POWER": "",
        }
        res = _map_spc_metadata(log_meta)
        assert res["Exposure Time (s)"] == "10"
        assert res["Accumulations"] == "2"
        assert res["Slit Opening (µm)"] == "200"
        assert res["Laser Wavelength (nm)"] == "354.7"
        assert res["Grating"] == "2400"
        assert res["Objective Used"] == "X100"
        assert res["Date"] == "10/02/26 16:23:36"
        assert "TIME" not in res
        assert "ACCUM" not in res
        assert "HOLE" not in res
        assert "POWER" not in res
        assert res["INSTRUMENT"] == "T64000"

    def test_map_spc_metadata_labspec(self):
        from spectroview.model.m_io import _map_spc_metadata
        log_meta = {
            "ACQ. TIME (S)": "5",
            "ACCUMULATIONS": "3",
            "ND FILTER": "10%",
            "OBJECTIV": "100X",
            "EXCIT_LINE": "532.0",
        }
        res = _map_spc_metadata(log_meta)
        assert res["Exposure Time (s)"] == "5"
        assert res["Accumulations"] == "3"
        assert res["Laser Power (%)"] == "10%"
        assert res["Objective Used"] == "100X"
        assert res["Laser Wavelength (nm)"] == "532.0"


# ═══════════════════════════════════════════════════════════════════════════
# MVVM Purity, Detection, Real Datasets, Binary Formats & Edge Cases
# ═══════════════════════════════════════════════════════════════════════════


def _build_synthetic_spc_bytes(
    *,
    ftflgs: int = 0x00,
    fexp: int = 16,
    fnpts: int = 4,
    ffirst: float = 100.0,
    flast: float = 400.0,
    fnsub: int = 1,
    fxtype: int = 13,
    fdate: int = 0,
    comment: bytes = b"Synthetic SPC",
    catxt: bytes = b"",
    subfiles: list[tuple[int, float, float, list[int | float]]] | None = None,
    log_text: bytes | None = None,
) -> bytes:
    """Create a valid Galactic SPC v0x4B binary buffer for deterministic unit tests."""
    hdr = bytearray(512)
    struct.pack_into("<BBBbiddi", hdr, 0, ftflgs, 0x4B, 0, fexp, fnpts, ffirst, flast, fnsub)
    struct.pack_into("<BBBB", hdr, 28, fxtype, 1, 0, 0)
    struct.pack_into("<I", hdr, 32, fdate)
    hdr[88 : 88 + min(len(comment), 130)] = comment[:130]
    hdr[218 : 218 + min(len(catxt), 30)] = catxt[:30]

    body = bytearray()
    if subfiles is None:
        subfiles = [(fexp, 0.0, 0.0, [100, 200, 300, 400])]

    for sub_exp, sub_time, sub_wlevel, values in subfiles:
        shdr = bytearray(32)
        struct.pack_into("<BBHff", shdr, 0, 0, sub_exp & 0xFF, 0, sub_time, sub_wlevel)
        body.extend(shdr)
        if sub_exp == -128 or sub_exp == 0x80:
            body.extend(struct.pack(f"<{len(values)}f", *values))
        elif ftflgs & 0x01:
            body.extend(struct.pack(f"<{len(values)}h", *[int(v) for v in values]))
        else:
            body.extend(struct.pack(f"<{len(values)}i", *[int(v) for v in values]))

    if log_text is not None:
        flogoff = 512 + len(body)
        struct.pack_into("<I", hdr, 248, flogoff)
        log_hdr = bytearray(64)
        total_log_size = 64 + len(log_text)
        struct.pack_into("<IIIII", log_hdr, 0, total_log_size, total_log_size, 64, 0, len(log_text))
        body.extend(log_hdr)
        body.extend(log_text)

    return bytes(hdr + body)


class TestModelLayerMvvmPurity:
    """Verify Model-layer I/O modules never import View, ViewModel, Qt, or third-party WDF/PIL."""

    @pytest.mark.parametrize(
        "rel_path",
        [
            "spectroview/model/m_io.py",
            "spectroview/model/m_spc.py",
            "spectroview/model/m_wdf.py",
            "spectroview/model/m_l6s.py",
            "spectroview/model/m_ngs.py",
        ],
    )
    def test_no_qt_or_viewmodel_imports(self, project_root, rel_path):
        source = (project_root / rel_path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden_prefixes = (
            "PySide6",
            "spectroview.view",
            "spectroview.viewmodel",
            "renishawWiRE",
            "PIL",
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(forbidden_prefixes), (
                        f"{rel_path} imports forbidden module {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(forbidden_prefixes), (
                    f"{rel_path} imports from forbidden module {node.module}"
                )


class TestDetectFileType:
    """Test universal fast file classification across real and synthetic files."""

    def test_real_spectroscopic_datasets_classification(self, examples_dir, bench_dir):
        spec_dir = examples_dir / "spectroscopic_data"
        plot_dir = examples_dir / "datasets_for_plotting"

        expected = [
            # 1D discrete spectra & series
            (spec_dir / "spectra_1.wdf", "spectra"),
            (spec_dir / "static_repeatability.wdf", "spectra"),
            (spec_dir / "spectra.spc", "spectra"),
            (spec_dir / "MoS2_PL.spc", "spectra"),
            (spec_dir / "MoS2_Raman.spc", "spectra"),
            (spec_dir / "spectrum.l6s", "spectra"),
            (spec_dir / "spectrum_ls6.l6s", "spectra"),
            (spec_dir / "spectrum_ls6_2.l6s", "spectra"),
            (spec_dir / "spectrum.ngs", "spectra"),
            (spec_dir / "specturm_TRPL.dat", "spectra"),
            (spec_dir / "XRDspectra.txt", "spectra"),
            (spec_dir / "Spectra_baseline_1.txt", "spectra"),
            (bench_dir / "spectrum1_1ML.txt", "spectra"),
            # 2D / line-scan / wafer / time-resolved maps
            (spec_dir / "profil.wdf", "map"),
            (bench_dir / "3_3721map.wdf", "map"),
            (spec_dir / "2dmap_spc.spc", "map"),
            (spec_dir / "map.l6m", "map"),
            (spec_dir / "map.ngc", "map"),
            (bench_dir / "2Dmap_Si.txt", "map"),
            (bench_dir / "1_CL_map.txt", "map"),
            (bench_dir / "wafer1_process1.csv", "map"),
            (bench_dir / "wafer4_newformat.csv", "map"),
            (spec_dir / "time_resolved_wafer_maps.csv", "map"),
            # Workspaces, fit models, and datasheets
            (spec_dir / "spectra_baseline_correction.spectra", "spectra_workspace"),
            (spec_dir / "wafers_NEW.maps", "maps_workspace"),
            (plot_dir / "Dataset_ Wafer_1_2.graphs", "graphs_workspace"),
            (bench_dir / "fit_model_Si.json", "fit_model"),
            (plot_dir / "Data1_wafer_maps.csv", "dataframe"),
            (plot_dir / "Dataset_Excel.xlsx", "dataframe"),
        ]
        for path, expected_kind in expected:
            if path.exists():
                assert detect_file_type(path) == expected_kind, f"Misclassified {path.name}"

    def test_csv_comma_spectrum_vs_dataframe(self, tmp_path):
        spec_csv = tmp_path / "spec_comma.csv"
        spec_csv.write_text("Wavenumber,Intensity\n500.0,120.5\n510.0,140.2\n", encoding="utf-8")
        assert detect_file_type(spec_csv) == "spectra"

        headerless_csv = tmp_path / "spec_headerless.csv"
        headerless_csv.write_text("500.0;120.5\n510.0;140.2\n", encoding="utf-8")
        assert detect_file_type(headerless_csv) == "spectra"

        df_csv = tmp_path / "table.csv"
        df_csv.write_text("Sample,Peak,FWHM\nS1,520.5,3.2\nS2,520.8,3.4\n", encoding="utf-8")
        assert detect_file_type(df_csv) == "dataframe"

    def test_detect_errors_on_missing_empty_or_invalid_files(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            detect_file_type(tmp_path / "missing.txt")

        empty = tmp_path / "empty.txt"
        empty.write_text("   \n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Empty file"):
            detect_file_type(empty)

        one_col = tmp_path / "bad_cols.txt"
        one_col.write_text("100.0\n200.0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid number of columns"):
            detect_file_type(one_col)

        unsupported = tmp_path / "archive.zip"
        unsupported.write_bytes(b"PK\x03\x04")
        with pytest.raises(ValueError, match="Unsupported file format"):
            detect_file_type(unsupported)


class TestSpectrumEdgeCasesAndFormats:
    """Validate 1D spectrum loaders across headerless TXT, CSV variants, WDF, SPC, and TRPL."""

    def test_headerless_txt_preserves_first_data_row(self, examples_dir, bench_dir):
        # Regression: previously skiprows=1 silently dropped row 0 on headerless 2-col TXT files
        res_ml = load_spectrum_file(bench_dir / "spectrum1_1ML.txt")
        assert len(res_ml["items"][0]["x0"]) == 572

        xrd_path = examples_dir / "spectroscopic_data" / "XRDspectra.txt"
        assert xrd_path.exists()
        res_xrd = load_spectrum_file(xrd_path)
        assert len(res_xrd["items"][0]["x0"]) == 3890
        np.testing.assert_allclose(res_xrd["items"][0]["x0"][0], 15.1081, atol=1e-4)

    def test_double_tab_comment_header_txt_spectrum(self, examples_dir):
        p = examples_dir / "spectroscopic_data" / "Spectra_baseline_1.txt"
        assert p.exists()
        res = load_spectrum_file(p)
        item = res["items"][0]
        assert len(item["x0"]) == 575
        assert np.all(np.diff(item["x0"]) >= 0)

    @pytest.mark.parametrize(
        "content",
        [
            "500.0;10.0\n400.0;20.0\n300.0;30.0\n",  # headerless ';', descending X
            "300.0,30.0\n400.0,20.0\n500.0,10.0\n",  # headerless ','
            "Wave;Counts\n300.0;30.0\n400.0;20.0\n500.0;10.0\n",  # 1-row header ';'
            "Title\nMeta1\nWave;Counts\n300.0;30.0\n400.0;20.0\n500.0;10.0\n",  # 3-row header ';'
        ],
    )
    def test_csv_spectrum_variants_and_ascending_sort(self, tmp_path, content):
        csv_file = tmp_path / "spec.csv"
        csv_file.write_text(content, encoding="utf-8")
        res = load_spectrum_file(csv_file)
        item = res["items"][0]
        np.testing.assert_allclose(item["x0"], [300.0, 400.0, 500.0])
        np.testing.assert_allclose(item["y0"], [30.0, 20.0, 10.0])

    def test_wdf_single_and_temporal_series(self, examples_dir):
        spec_dir = examples_dir / "spectroscopic_data"
        single_p = spec_dir / "spectra_1.wdf"
        series_p = spec_dir / "static_repeatability.wdf"
        assert single_p.exists() and series_p.exists()

        assert peek_wdf_measurement_type(single_p) == "spectra"
        assert peek_wdf_measurement_type(series_p) == "spectra"

        single = load_wdf_spectrum(single_p)
        assert len(single["items"]) == 1
        assert len(single["items"][0]["x0"]) == 572
        assert single["items"][0]["metadata"]["File Format"] == "Renishaw WDF"

        series = load_wdf_spectrum(series_p)
        assert len(series["items"]) == 50
        times = [float(it["metadata"]["Time (s)"]) for it in series["items"]]
        assert times == sorted(times)
        assert series["items"][0]["name"].startswith("static_repeatability_")

    def test_spc_single_raman_and_pl_units_and_log_metadata(self, examples_dir):
        spec_dir = examples_dir / "spectroscopic_data"
        raman_p = spec_dir / "spectrum_ls5.spc" if (spec_dir / "spectrum_ls5.spc").exists() else spec_dir / "spectra.spc"
        pl_p = spec_dir / "MoS2_PL.spc"
        assert raman_p.exists() and pl_p.exists()

        raman = load_spc_spectrum(raman_p)
        r_meta = raman["items"][0]["metadata"]
        assert r_meta["X-axis Unit"] == "1/cm"
        assert r_meta["INSTRUMENT"] == "LABRAM SOLEIL"

        _, map_meta = load_spc_map(spec_dir / "2dmap_spc.spc")
        # Regression: log keys containing colons (e.g. FULL TIME(HH:MM)) must not be truncated
        assert "FULL TIME(HH:MM)" in map_meta
        assert "MM)" not in map_meta

        pl = load_spc_spectrum(pl_p)
        pl_meta = pl["items"][0]["metadata"]
        assert pl_meta["X-axis Unit"] == "nm"
        assert np.all(np.diff(pl["items"][0]["x0"]) >= 0)

    def test_trpl_dat_spectrum_and_invalid_header(self, examples_dir, tmp_path):
        trpl_p = examples_dir / "spectroscopic_data" / "specturm_TRPL.dat"
        assert trpl_p.exists()
        res = load_TRPL_data(trpl_p)
        item = res["items"][0]
        assert len(item["x0"]) == 1596
        assert item["x0"][0] == pytest.approx(0.0)
        assert item["y0"][0] == float(np.max(item["y0"]))
        assert item["metadata"]["File Format"] == "TRPL (.dat)"
        assert item["metadata"]["X-axis Unit"] == "ns"
        assert item["metadata"]["Date"] == "13/11/2024 14:22:58"

        bad_dat = tmp_path / "invalid.dat"
        bad_dat.write_text("no counts header here\n1\n2\n3\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid TRPL file format"):
            load_TRPL_data(bad_dat)


class TestSpcBinaryReaderEdgeCases:
    """Test low-level SpcReader modes, integer scaling, date decoding, and corrupt inputs."""

    def test_peek_header_skips_payload_allocation(self, examples_dir):
        map_spc = examples_dir / "spectroscopic_data" / "2dmap_spc.spc"
        assert map_spc.exists()
        reader = SpcReader(str(map_spc), header_only=True)
        assert reader.header["fnsub"] == 1681
        assert reader.y_data is None
        hdr = SpcReader.peek_header(map_spc)
        assert hdr["fnsub"] == 1681
        assert hdr["npts"] == 1024

    def test_16bit_integer_spc_and_descending_x_sort(self, tmp_path):
        # fdate packed: year=2025, month=6, day=15, hour=14, min=30
        packed_date = (2025 << 20) | (6 << 16) | (15 << 11) | (14 << 6) | 30
        spc_bytes = _build_synthetic_spc_bytes(
            ftflgs=0x01,  # TSPREC: 16-bit integer Y values
            fexp=16,      # scale = 2^(16 - 16) = 1.0
            fnpts=4,
            ffirst=400.0,
            flast=100.0,  # descending X axis -> loader must sort ascending
            fnsub=1,
            fxtype=14,    # Electron Volts (eV)
            fdate=packed_date,
            subfiles=[(16, 0.0, 0.0, [10, 20, 30, 40])],
            log_text=b"INSTRUMENT = SyntheticSpec\r\nEXCIT_LINE = 532\r\n",
        )
        spc_path = tmp_path / "int16_desc.spc"
        spc_path.write_bytes(spc_bytes)

        res = load_spc_spectrum(spc_path)
        item = res["items"][0]
        np.testing.assert_allclose(item["x0"], [100.0, 200.0, 300.0, 400.0])
        np.testing.assert_allclose(item["y0"], [40.0, 30.0, 20.0, 10.0])
        assert item["metadata"]["X-axis Unit"] == "eV"
        assert item["metadata"]["Date"] == "2025-06-15 14:30"
        assert item["metadata"]["Laser Wavelength (nm)"] == "532"

    def test_non_square_spc_map_fallback_coordinates(self, tmp_path):
        # Regression: fnsub=6 (not a perfect square) with zero sub_time/sub_wlevel
        # previously crashed with UnboundLocalError on y_coords
        subs = [(-128, 0.0, 0.0, [float(i), float(i + 1), float(i + 2)]) for i in range(6)]
        spc_bytes = _build_synthetic_spc_bytes(
            ftflgs=0x04,
            fexp=-128,
            fnpts=3,
            ffirst=500.0,
            flast=520.0,
            fnsub=6,
            subfiles=subs,
        )
        spc_path = tmp_path / "nonsquare_map.spc"
        spc_path.write_bytes(spc_bytes)

        df, meta = load_spc_map(spc_path)
        assert df.shape == (6, 5)  # X, Y + 3 wavenumbers
        assert df["X"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        assert df["Y"].tolist() == [0.0] * 6
        assert int(meta["Total Map Points"]) == 6

    def test_corrupt_or_truncated_spc_raises_value_error(self, tmp_path):
        short_spc = tmp_path / "short.spc"
        short_spc.write_bytes(b"too short")
        with pytest.raises(ValueError, match="Invalid or truncated SPC file"):
            SpcReader(str(short_spc))

        trunc_spc = tmp_path / "truncated_payload.spc"
        full_bytes = _build_synthetic_spc_bytes(fnpts=10, subfiles=[(-128, 0.0, 0.0, [1.0] * 10)])
        trunc_spc.write_bytes(full_bytes[:530])  # header ok, subfile truncated
        with pytest.raises(ValueError):
            SpcReader(str(trunc_spc))


class TestWdfBinaryReaderEdgeCases:
    """Test native WdfReader header peeking, context manager, enum fallbacks, and corrupt inputs."""

    def test_peek_header_and_context_manager(self, wdf_map_file, examples_dir):
        if wdf_map_file.exists():
            hdr = WdfReader.peek_header(wdf_map_file)
            assert hdr["count"] == 3721
            assert hdr["point_per_spectrum"] == 575
            assert str(hdr["measurement_type"]) == "Mapping"
            assert hdr["application_name"] == "WiRE"

            reader_hdr_only = WdfReader(wdf_map_file, header_only=True)
            assert reader_hdr_only.spectra.size == 0
            assert reader_hdr_only.file_obj.closed

        single_p = examples_dir / "spectroscopic_data" / "spectra_1.wdf"
        assert single_p.exists()
        with WdfReader(single_p) as reader:
            assert reader.count == 1
            assert reader.point_per_spectrum == 572
            assert reader.spectra.shape == (572,)
            assert not reader.file_obj.closed
        assert reader.file_obj.closed

    def test_enum_unknown_codes_fallback_gracefully(self):
        assert WdfMeasurementType(999) == WdfMeasurementType.Unspecified
        assert WdfScanType(999) == WdfScanType.Unspecified
        assert WdfUnitType(999) == WdfUnitType.Arbitrary
        assert str(WdfUnitType.Microseconds) == "us"
        assert WdfDataType(999) == WdfDataType.Arbitrary

    def test_corrupt_or_truncated_wdf_raises_errors(self, tmp_path, examples_dir):
        with pytest.raises(OSError):
            WdfReader(tmp_path / "does_not_exist.wdf")

        short_wdf = tmp_path / "short.wdf"
        short_wdf.write_bytes(b"WDF1" + b"\x00" * 20)
        with pytest.raises(ValueError, match="Invalid or truncated WDF file"):
            WdfReader(short_wdf)

        bad_magic = tmp_path / "bad_magic.wdf"
        bad_magic.write_bytes(b"XXXX" + b"\x00" * 508)
        with pytest.raises(ValueError, match="wdf file format is incorrect"):
            WdfReader(bad_magic)

        single_p = examples_dir / "spectroscopic_data" / "spectra_1.wdf"
        raw_bytes = single_p.read_bytes()
        trunc_data = tmp_path / "trunc_data.wdf"
        # Keep 512B WDF1 header + partial DATA block (truncate inside DATA payload)
        trunc_data.write_bytes(raw_bytes[:600])
        # Force DATA block size header to claim only 88 bytes so _locate_all_blocks registers it
        buf = bytearray(trunc_data.read_bytes())
        struct.pack_into("<IQ", buf, 512 + 4, 0, 88)
        trunc_data.write_bytes(bytes(buf))
        with pytest.raises(ValueError, match="Truncated WDF DATA block"):
            WdfReader(trunc_data)


class TestMapsLineProfilesAndTimeResolvedData:
    """Validate WDF line profiles, wafer CSV trailing semicolons, and time-resolved maps."""

    def test_wdf_1d_line_profile_metadata_and_shape(self, examples_dir):
        profil_p = examples_dir / "spectroscopic_data" / "profil.wdf"
        assert profil_p.exists()

        df, meta = load_wdf_map(profil_p)
        assert df.shape == (401, 577)
        # Regression: 1D line-scan WDF maps must still report Renishaw WDF Map + spatial step sizes
        assert meta["File Format"] == "Renishaw WDF Map"
        assert int(meta["Total Map Points"]) == 401
        assert meta["X Step Size (µm)"] == "0.000"
        assert meta["Y Step Size (µm)"] == "0.100"

        x0, Y0, coords, fnames = extract_map_arrays("profil", df)
        assert x0.shape == (575,)
        assert Y0.shape == (401, 575)
        assert coords.shape == (401, 2)
        assert len(fnames) == 401

    def test_wafer_csv_maps_strip_trailing_unnamed_and_preserve_all_channels(
        self, wafer_file, wafer_process1_file, examples_dir
    ):
        for p in (wafer_file, wafer_process1_file):
            if not p.exists():
                continue
            df = load_map_file(p)
            assert not any(str(c).startswith("Unnamed") for c in df.columns)
            x0, Y0, coords, fnames = extract_map_arrays(p.stem, df)
            assert len(x0) == 2048
            assert Y0.shape == (len(df), 2048)
            assert np.all(np.diff(x0) >= 0)

        tr_path = examples_dir / "spectroscopic_data" / "time_resolved_wafer_maps.csv"
        assert tr_path.exists()
        tr_df = load_map_file(tr_path)
        assert tr_df.shape == (50, 2050)
        x0, Y0, coords, fnames = extract_map_arrays("time_resolved_wafer_maps", tr_df)
        assert len(x0) == 2048
        assert Y0.shape == (50, 2048)
        assert len(set(fnames)) == 50
        # Verify coordinate parser still recovers exact (X, Y) from disambiguated fnames
        for i, fname in enumerate(fnames):
            cx, cy = parse_coords_from_fname(fname)
            assert cx == pytest.approx(coords[i, 0])
            assert cy == pytest.approx(coords[i, 1])


class TestConvertActionEdgeCases:
    """Validate vectorized Renishaw long-format converter edge cases."""

    def test_single_spectrum_export_does_not_raise_index_error(self, tmp_path):
        # Regression: previously dfr_spectrum['size'][1] raised IndexError when N == 1
        raw = tmp_path / "single_pixel.txt"
        raw.write_text(
            "#X\t#Y\t#Wave\t#Intensity\n"
            "10.0\t20.0\t500.0\t100.0\n"
            "10.0\t20.0\t520.0\t200.0\n",
            encoding="utf-8",
        )
        out = tmp_path / "single_converted.txt"
        convert_action(raw, out)
        df = load_map_file(out)
        assert len(df) == 1
        assert df["X"].iloc[0] == pytest.approx(10.0)
        assert df["Y"].iloc[0] == pytest.approx(20.0)
        assert df["500.0"].iloc[0] == pytest.approx(100.0)
        assert df["520.0"].iloc[0] == pytest.approx(200.0)

    def test_preserves_acquisition_coordinate_order_and_handles_double_tabs(self, tmp_path):
        # Regression: groupby(['#Y', '#X']) without sort=False scrambled intensities vs coordinates
        # when scan coordinates were descending or snake-ordered, and double-tabs required temp copies
        lines = ["#X\t\t#Y\t\t#Wave\t\t#Intensity"]
        scan_Order = [(10.0, 20.0, 111.0), (0.0, 20.0, 222.0), (0.0, 10.0, 333.0)]
        for x_val, y_val, base_i in scan_Order:
            lines.append(f"{x_val}\t\t{y_val}\t\t500.0\t\t{base_i}")
            lines.append(f"{x_val}\t\t{y_val}\t\t510.0\t\t{base_i + 1.0}")

        raw = tmp_path / "snake_double_tab.txt"
        raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out = tmp_path / "snake_converted.txt"
        convert_action(raw, out)

        df = load_map_file(out)
        assert df["X"].tolist() == [10.0, 0.0, 0.0]
        assert df["Y"].tolist() == [20.0, 20.0, 10.0]
        assert df["500.0"].tolist() == [111.0, 222.0, 333.0]
        assert df["510.0"].tolist() == [112.0, 223.0, 334.0]

    def test_inconsistent_point_counts_raises_value_error(self, tmp_path):
        raw = tmp_path / "ragged.txt"
        raw.write_text(
            "#X\t#Y\t#Wave\t#Intensity\n"
            "0\t0\t500\t10\n"
            "0\t0\t510\t20\n"
            "1\t0\t500\t30\n",  # missing 510 point
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Inconsistent number of wavenumber points"):
            convert_action(raw, tmp_path / "out.txt")


class TestL6sAndL6mBinaryReader:
    """Validate native HORIBA LabSpec 6 (.l6s and .l6m) binary reader and metadata ordering."""

    def test_l6s_spectrum_loading_and_metadata_ordering(self, examples_dir):
        spec_dir = examples_dir / "spectroscopic_data"
        l6s_path = spec_dir / "spectrum_ls6.l6s" if (spec_dir / "spectrum_ls6.l6s").exists() else spec_dir / "spectrum.l6s"
        assert l6s_path.exists()

        hdr = L6sReader.peek_header(l6s_path)
        assert hdr["root_tag"] == "datas"
        assert hdr["version"] == 2
        assert hdr["root_count"] == 11

        res = load_l6s_spectrum(l6s_path)
        assert res["type"] == "spectra"
        assert len(res["items"]) == 1

        item = res["items"][0]
        assert item["name"] in ("spectrum", "spectrum_ls6", l6s_path.stem)
        assert item["x0"].shape == (1024,)
        assert item["y0"].shape == (1024,)
        assert np.all(np.diff(item["x0"]) > 0)

        # Silicon reference peak around 520.46 cm^-1
        peak_idx = int(np.argmax(item["y0"]))
        assert item["x0"][peak_idx] == pytest.approx(520.4608, abs=0.05)
        assert item["y0"][peak_idx] == pytest.approx(4892.6411, abs=0.05)

        meta = item["metadata"]
        keys = list(meta.keys())

        # Verify _METADATA_ORDERED_KEYS appear first in exact canonical order
        ordered_present = [k for k in _METADATA_ORDERED_KEYS if k in meta]
        assert keys[: len(ordered_present)] == ordered_present

        assert meta["File Format"] == "HORIBA LabSpec 6"
        assert meta["Date"] == "21.05.2026 16:39"
        assert meta["Grating"] == "1800 (500nm)"
        assert meta["Objective Used"] == "100x"
        assert meta["Laser Wavelength (nm)"] == "532.43"
        assert meta["Laser Power (%)"] == "1% (0.96mW)"
        assert meta["Exposure Time (s)"] == 5
        assert meta["Accumulations"] == 2
        assert meta["Slit Opening (µm)"] == "100"
        assert meta["X-axis Unit"] == "1/cm"
        assert meta["Points"] == 1024
        assert meta["Title"] == "D"
        assert meta["Application"] == "LabSpec 6"
        assert meta["Measurement Type"] == "Spectrum"
        assert meta["Spectral Unit"] == "Cnt"
        assert meta["X-axis Type"] == "Spectr"

        # Additional LabSpec 6 parameters appended after _METADATA_ORDERED_KEYS
        tail_keys = keys[len(ordered_present) :]
        assert "Instrument" in tail_keys
        assert meta["Instrument"] == "LabRAM Soleil"
        assert meta["Detector"] == "Syncerity OE"
        assert meta["Laser"] == "532nm_Edge"
        assert meta["Detector temperature (°C)"] == pytest.approx(-60.1)
        assert meta["Acquired"] == "21.05.2026 16:39:38"
        assert meta["S/N"] == "57.7"

    def test_compressed_l6s_spectrum_loading_and_metadata_ordering(self, examples_dir):
        l6s_path = examples_dir / "spectroscopic_data" / "spectrum_ls6_2.l6s"
        assert l6s_path.exists()

        assert detect_file_type(l6s_path) == "spectra"

        hdr = L6sReader.peek_header(l6s_path)
        assert hdr["root_tag"] == "datas"
        assert hdr["version"] == 2
        assert hdr["root_count"] == 10
        assert hdr["is_compressed"] is True

        res = load_l6s_spectrum(l6s_path)
        assert res["type"] == "spectra"
        assert len(res["items"]) == 1

        item = res["items"][0]
        assert item["name"] == "spectrum_ls6_2"
        assert item["x0"].shape == (1022,)
        assert item["y0"].shape == (1022,)
        # Raw eV axis in spectrum_ls6_2.l6s is descending [3.6923..1.8634]; loader sorts ascending
        assert np.all(np.diff(item["x0"]) > 0)
        assert item["x0"][0] == pytest.approx(1.86339, abs=1e-4)
        assert item["x0"][-1] == pytest.approx(3.69226, abs=1e-4)

        # GaN NBE photoluminescence peak around 3.422 eV
        peak_idx = int(np.argmax(item["y0"]))
        assert item["x0"][peak_idx] == pytest.approx(3.4221, abs=1e-3)
        assert item["y0"][peak_idx] == pytest.approx(515.0, abs=0.05)

        meta = item["metadata"]
        keys = list(meta.keys())
        ordered_present = [k for k in _METADATA_ORDERED_KEYS if k in meta]
        assert keys[: len(ordered_present)] == ordered_present

        assert meta["File Format"] == "HORIBA LabSpec 6"
        assert meta["Date"] == "12.11.2026 18:44"
        assert meta["Grating"] == "100 gr/mm Vis"
        assert meta["Objective Used"] == "x15 NUV"
        assert meta["Laser Wavelength (nm)"] == "325"
        assert meta["Laser Power (%)"] == "0.01%"
        assert meta["Exposure Time (s)"] == 100
        assert meta["Accumulations"] == 2
        assert meta["Slit Opening (µm)"] == pytest.approx(250.0225)
        assert meta["X-axis Unit"] == "eV"
        assert meta["Points"] == 1022
        assert meta["Title"] == "D_1"
        assert meta["Application"] == "LabSpec 6"
        assert meta["Measurement Type"] == "Spectrum"
        assert meta["Spectral Unit"] == "Cnt"
        assert meta["X-axis Type"] == "Spectr"

        tail_keys = keys[len(ordered_present) :]
        assert "Instrument" in tail_keys
        assert meta["Instrument"] == "LabRAM HR Evol"
        assert meta["Detector"] == "Synapse CCD"
        assert meta["Detector temperature (°C)"] == pytest.approx(-70.16)
        assert meta["Acquired"] == "12.11.2026 18:44:03"
        assert meta["S/N"] == "36.1"
        assert "Laser (nm)" not in meta

    def test_compressed_l6s_fast_peek_for_large_archives(self, examples_dir, monkeypatch):
        from spectroview.model import m_l6s

        l6s_path = examples_dir / "spectroscopic_data" / "spectrum_ls6_2.l6s"
        # Force the large-archive (<1 ms) fast-peek branch used for multi-MB compressed .l6m maps
        monkeypatch.setattr(m_l6s, "_PEEK_DECOMPRESS_LIMIT", 1024)
        hdr = L6sReader.peek_header(l6s_path)
        assert hdr["root_tag"] == "datas"
        assert hdr["is_compressed"] is True
        assert hdr["flags"] == 0x0D
        assert detect_file_type(l6s_path) == "spectra"

    def test_l6m_map_loading_coordinates_and_metadata_ordering(self, examples_dir):
        spec_dir = examples_dir / "spectroscopic_data"
        l6m_path = spec_dir / "mapls6.l6m" if (spec_dir / "mapls6.l6m").exists() else spec_dir / "map.l6m"
        assert l6m_path.exists()

        hdr = L6sReader.peek_header(l6m_path)
        assert hdr["root_tag"] == "datas"
        assert hdr["version"] == 2
        assert hdr["root_count"] == 16

        df, meta = load_l6m_map(l6m_path)
        # 36 Y points * 38 X points = 1368 spectra, 1024 spectral points + X, Y columns = 1026
        assert df.shape == (1368, 1026)
        assert list(df.columns[:2]) == ["X", "Y"]

        x0, Y0, coords, fnames = extract_map_arrays("map", df)
        assert x0.shape == (1024,)
        assert Y0.shape == (1368, 1024)
        assert coords.shape == (1368, 2)
        assert len(fnames) == 1368
        # Raw eV axis in map.l6m is descending [2.9485..1.2948]; loader must sort ascending
        assert np.all(np.diff(x0) > 0)
        assert x0[0] == pytest.approx(1.29479, abs=1e-4)
        assert x0[-1] == pytest.approx(2.94846, abs=1e-4)

        keys = list(meta.keys())
        ordered_present = [k for k in _METADATA_ORDERED_KEYS if k in meta]
        assert keys[: len(ordered_present)] == ordered_present

        assert meta["File Format"] == "HORIBA LabSpec 6 Map"
        assert meta["Date"] == "28.04.2025 15:55"
        assert meta["Grating"] == "150 (500nm)"
        assert meta["Objective Used"] == "100x"
        assert meta["Laser Wavelength (nm)"] == "532.43"
        assert meta["Laser Power (%)"] == "0.5% (0.48mW)"
        assert meta["Exposure Time (s)"] == pytest.approx(0.01)
        assert meta["Accumulations"] == 1
        assert meta["Slit Opening (µm)"] == "100"
        assert meta["X-axis Unit"] == "eV"
        assert meta["Points per Spectrum"] == 1024
        assert meta["Total Map Points"] == 1368
        assert meta["X Step Size (µm)"] == "0.200"
        assert meta["Y Step Size (µm)"] == "0.203"
        assert meta["X Range (µm)"] == "-3.65 to 3.75"
        assert meta["Y Range (µm)"] == "-3.10 to 3.99"

        tail_keys = keys[len(ordered_present) :]
        assert "SWIFT" in tail_keys
        assert meta["SWIFT"] == "On"
        assert meta["Instrument"] == "LabRAM Soleil"
        assert meta["Detector"] == "Syncerity OE"
        assert meta["S/N"] == "70.3"

    def test_corrupt_or_truncated_l6s_raises_value_error(self, tmp_path):
        short_l6s = tmp_path / "short.l6s"
        short_l6s.write_bytes(b"LabSpec6\x00short")
        with pytest.raises(ValueError, match="File too small"):
            detect_file_type(short_l6s)

        bad_magic = tmp_path / "bad.l6m"
        bad_magic.write_bytes(b"NOT_LABSPEC" + b"\x00" * 128)
        with pytest.raises(ValueError, match="Invalid LabSpec 6 magic header"):
            detect_file_type(bad_magic)
        with pytest.raises(ValueError, match="Invalid LabSpec 6 binary file"):
            L6sReader(bad_magic)

        # Corrupt compressed header: uncomp_size not divisible by 4 with 0x04 shuffle flag
        bad_uncomp = tmp_path / "bad_uncomp.l6s"
        bad_uncomp.write_bytes(
            b"LabSpec6\x00\x00\x00\x00" + struct.pack("<IIII", 0x0D, 0, 101, 32) + b"\x00" * 32
        )
        with pytest.raises(ValueError, match="Corrupt LabSpec 6 compression header"):
            L6sReader.peek_header(bad_uncomp)
        with pytest.raises(ValueError, match="Corrupt LabSpec 6 compression header"):
            L6sReader(bad_uncomp)

        # Corrupt compressed header: comp_size exceeds actual file length
        trunc_comp = tmp_path / "trunc_comp.l6s"
        trunc_comp.write_bytes(
            b"LabSpec6\x00\x00\x00\x00" + struct.pack("<IIII", 0x0D, 0, 1024, 5000) + b"\x00" * 32
        )
        with pytest.raises(ValueError, match="Corrupt LabSpec 6 compression header"):
            L6sReader.peek_header(trunc_comp)
        with pytest.raises(ValueError, match="Corrupt LabSpec 6 compression header"):
            L6sReader(trunc_comp)

        # Corrupt compressed payload bytes (non-LabSpec root descriptor after decode)
        bad_payload = tmp_path / "bad_payload.l6s"
        bad_payload.write_bytes(
            b"LabSpec6\x00\x00\x00\x00" + struct.pack("<IIII", 0x0D, 0, 128, 64) + b"\xff" * 64
        )
        with pytest.raises(ValueError, match="Invalid LabSpec 6 root descriptor"):
            L6sReader.peek_header(bad_payload)
        with pytest.raises(ValueError, match="Invalid LabSpec 6 root descriptor"):
            L6sReader(bad_payload)


class TestNgsAndNgcBinaryReader:
    """Validate native HORIBA LabSpec 5 (.ngs and .ngc) binary reader and metadata ordering."""

    def test_ngs_spectrum_loading_and_metadata_ordering(self, examples_dir):
        ngs_path = examples_dir / "spectroscopic_data" / "spectrum.ngs"
        assert ngs_path.exists()

        hdr = NgsReader.peek_header(ngs_path)
        assert hdr["version"] == 1
        assert hdr["is_compressed"] is False
        assert hdr["subtype"] == "Spectrum"
        assert hdr["label"] == "Si_Ref"
        assert hdr["n_dims"] == 1
        assert hdr["is_map"] is False

        res = load_ngs_spectrum(ngs_path)
        assert res["type"] == "spectra"
        assert len(res["items"]) == 1

        item = res["items"][0]
        assert item["name"] in ("spectrum", "spectrum.ngs", ngs_path.stem)
        assert item["x0"].shape == (1024,)
        assert item["y0"].shape == (1024,)
        assert np.all(np.diff(item["x0"]) > 0)
        assert item["x0"][0] == pytest.approx(34.5072, abs=1e-3)
        assert item["x0"][-1] == pytest.approx(1041.7700, abs=1e-3)

        # Silicon reference peak around 524.35 cm^-1
        peak_idx = int(np.argmax(item["y0"]))
        assert item["x0"][peak_idx] == pytest.approx(524.3491, abs=0.05)
        assert item["y0"][peak_idx] == pytest.approx(831.1758, abs=0.05)

        meta = item["metadata"]
        keys = list(meta.keys())

        # Verify _METADATA_ORDERED_KEYS appear first in exact canonical order
        ordered_present = [k for k in _METADATA_ORDERED_KEYS if k in meta]
        assert keys[: len(ordered_present)] == ordered_present

        assert meta["File Format"] == "HORIBA LabSpec 5"
        assert meta["Date"] == "05/30/24 17:00:37"
        assert meta["Grating"] == 2400
        assert meta["Objective Used"] == "x100"
        assert meta["Laser Wavelength (nm)"] == "363.81"
        assert meta["Exposure Time (s)"] == 10
        assert meta["Accumulations"] == 3
        assert meta["Slit Opening (µm)"] == 200
        assert meta["X-axis Unit"] == "1/cm"
        assert meta["Points"] == 1024
        assert meta["Title"] == "Si_Ref"
        assert meta["Application"] == "LabSpec 5"
        assert meta["Measurement Type"] == "Spectrum"
        assert meta["Spectral Unit"] == "cnt"
        assert meta["X-axis Type"] == "Spectr"

        # Additional LabSpec 5 parameters appended after _METADATA_ORDERED_KEYS
        tail_keys = keys[len(ordered_present) :]
        assert "Instrument" in tail_keys
        assert meta["Instrument"] == "T64000"
        assert meta["Detector"] == "Synapse CCD"
        assert meta["Detector size"] == 1024
        assert meta["Binning"] == 1
        assert meta["Spectro"] == "549.978 1/cm"
        assert meta["Spec. width"] == "3.78597 1/cm"
        assert meta["Entrance Slit"] == "100 µm"
        assert meta["First Intermediate Slit"] == "26700 µm"
        assert meta["Second Intermediate Slit"] == "100 µm"
        assert meta["Analysis Mode"] == "Micro"
        assert meta["Spectrograph"] == "Triple"
        assert "Acquired" in tail_keys

    def test_ngc_map_loading_coordinates_and_metadata_ordering(self, examples_dir):
        spec_dir = examples_dir / "spectroscopic_data"
        ngc_path = spec_dir / "mapls5.ngc" if (spec_dir / "mapls5.ngc").exists() else spec_dir / "map.ngc"
        assert ngc_path.exists()

        hdr = NgsReader.peek_header(ngc_path)
        assert hdr["version"] == 3
        assert hdr["is_compressed"] is True
        assert hdr["subtype"] == "SpIm"
        assert hdr["label"] == "MapSample1_1"
        assert hdr["n_dims"] == 3
        assert hdr["is_map"] is True

        df, meta = load_ngc_map(ngc_path)
        # 61 Y points * 61 X points = 3721 spectra, 1024 spectral points + X, Y columns = 1026
        assert df.shape == (3721, 1026)
        assert list(df.columns[:2]) == ["X", "Y"]

        x0, Y0, coords, fnames = extract_map_arrays("map", df)
        assert x0.shape == (1024,)
        assert Y0.shape == (3721, 1024)
        assert coords.shape == (3721, 2)
        assert len(fnames) == 3721
        assert np.all(np.diff(x0) > 0)
        assert x0[0] == pytest.approx(54.0570, abs=1e-3)
        assert x0[-1] == pytest.approx(1120.3964, abs=1e-3)

        keys = list(meta.keys())
        ordered_present = [k for k in _METADATA_ORDERED_KEYS if k in meta]
        assert keys[: len(ordered_present)] == ordered_present

        assert meta["File Format"] == "HORIBA LabSpec 5 Map"
        assert meta["Date"] == "07/28/26 17:23:18"
        assert meta["Grating"] == 2400
        assert meta["Objective Used"] == "x100"
        assert meta["Laser Wavelength (nm)"] == "354.7"
        assert meta["Exposure Time (s)"] == 5
        assert meta["Accumulations"] == 2
        assert meta["Slit Opening (µm)"] == 200
        assert meta["X-axis Unit"] == "1/cm"
        assert meta["Points per Spectrum"] == 1024
        assert meta["Total Map Points"] == 3721
        assert meta["X Step Size (µm)"] == "0.100"
        assert meta["Y Step Size (µm)"] == "0.100"
        assert meta["X Range (µm)"] == "-3.80 to 2.20"
        assert meta["Y Range (µm)"] == "2.50 to 8.50"
        assert meta["Title"] == "MapSample1_1"
        assert meta["Application"] == "LabSpec 5"
        assert meta["Measurement Type"] == "SpIm"

        tail_keys = keys[len(ordered_present) :]
        assert "Instrument" in tail_keys
        assert meta["Instrument"] == "T64000"
        assert meta["Detector"] == "Synapse CCD"
        assert meta["Second Intermediate Slit"] == "50 µm"
        assert meta["Spectrograph"] == "Triple"

    def test_corrupt_or_truncated_ngs_raises_value_error(self, tmp_path):
        short_ngs = tmp_path / "short.ngs"
        short_ngs.write_bytes(b"NGSNextGen")
        with pytest.raises(ValueError, match="Invalid LabSpec 5 NGSNextGen"):
            detect_file_type(short_ngs)

        bad_ver = tmp_path / "badver.ngc"
        bad_ver.write_bytes(b"NGSNextGen" + struct.pack("<I", 99) + b"\x00" * 32)
        with pytest.raises(ValueError, match="Unsupported LabSpec 5 NGSNextGen version"):
            detect_file_type(bad_ver)
        with pytest.raises(ValueError, match="Unsupported LabSpec 5 NGSNextGen version"):
            NgsReader(bad_ver)



