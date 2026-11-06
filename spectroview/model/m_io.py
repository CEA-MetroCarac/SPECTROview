"""Universal spectroscopic file detection and loading pipeline (Model layer).

Handles detection, binary/text parsing, coordinate normalization, and metadata
extraction across all supported SPECTROview formats without any Qt or ViewModel
dependencies.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
import re
import struct

import numpy as np
import pandas as pd

from spectroview.model.m_spc import SpcReader
from spectroview.model.m_wdf import WdfReader

WDFReader = WdfReader
WDF_AVAILABLE = True


# Precompiled regexes for WDF/TRPL metadata extraction
_UM_INT_RE = re.compile(r"(\d+)[\xc2\xb5µ]m")
_UM_FLOAT_RE = re.compile(r"(\d+\.?\d*)[\xc2\xb5µ]m")

_METADATA_ORDERED_KEYS = (
    "File Format",
    "Date",
    "Grating",
    "Objective Used",
    "Laser Wavelength (nm)",
    "Laser Power (%)",
    "Exposure Time (s)",
    "Accumulations",
    "Slit Opening (µm)",
    "Wavenumber Range",
    "X-axis Unit",
    "Points per Spectrum",
    "Points",
    "Total Map Points",
    "X Step Size (µm)",
    "Y Step Size (µm)",
    "Title",
    "Username",
    "Application",
    "Measurement Type",
    "Scan Type",
    "X Range (µm)",
    "Y Range (µm)",
    "Spectral Unit",
    "X-axis Type",
    "Slit Centre (µm)",
)


def _require_wdf():
    """Return WdfReader (native built-in WDF binary reader)."""
    return WdfReader


def _is_float_token(token: str) -> bool:
    """Return True if *token* parses as a finite or NaN float."""
    t = token.strip()
    if not t:
        return False
    try:
        float(t)
        return True
    except ValueError:
        return False


def _split_line(line: str, delimiter: str | None) -> list[str]:
    """Split a text line using *delimiter* (None denotes arbitrary whitespace)."""
    s = line.strip()
    if not s:
        return []
    if delimiter is None:
        return s.split()
    return [tok.strip() for tok in s.split(delimiter)]


def _detect_text_delimiter(sample_line: str, allow_comma: bool = False) -> str | None:
    """Detect delimiter in priority order: ';', ',', '\\t', or None (whitespace)."""
    if ";" in sample_line:
        return ";"
    if allow_comma and "," in sample_line:
        return ","
    if "\t" in sample_line:
        return "\t"
    return None


def peek_wdf_measurement_type(path: str | Path) -> str:
    """Inspect the 512-byte WDF1 binary header to classify 'map' vs 'spectra'.

    Reads only 144 bytes from disk instead of parsing the full spectral cube.
    Falls back to ``WDFReader`` if the header does not match WDF1 layout.
    """
    _require_wdf()
    p = Path(path)
    with open(p, "rb") as f:
        hdr = f.read(144)

    if len(hdr) >= 136 and hdr[:4] == b"WDF1":
        # Offset 132 (uint32): WDFMeasurementType (1=Single, 2=Series, 3=Mapping)
        mtype = struct.unpack_from("<I", hdr, 132)[0]
        if mtype in (1, 2, 3):
            return "map" if mtype == 3 else "spectra"

    # Fallback for non-standard WDF headers (also validates corrupt files)
    reader_cls = _require_wdf()
    reader = reader_cls(str(p))
    try:
        mtype_str = str(reader.measurement_type)
        return "map" if "Mapping" in mtype_str else "spectra"
    finally:
        reader.close()


def detect_file_type(path: str | Path) -> str:
    """Classify any supported SPECTROview file path without loading full datasets.

    Returns one of:
        ``'spectra'``, ``'map'``, ``'dataframe'``, ``'fit_model'``,
        ``'spectra_workspace'``, ``'maps_workspace'``, ``'graphs_workspace'``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the extension is unsupported or the file structure is invalid.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    ext = p.suffix.lower()

    if ext == ".spectra":
        return "spectra_workspace"
    if ext == ".maps":
        return "maps_workspace"
    if ext == ".graphs":
        return "graphs_workspace"
    if ext == ".json":
        return "fit_model"
    if ext == ".dat":
        return "spectra"
    if ext in (".xlsx", ".xls"):
        return "dataframe"
    if ext == ".wdf":
        return peek_wdf_measurement_type(p)
    if ext == ".spc":
        hdr = SpcReader.peek_header(p)
        return "map" if hdr["fnsub"] > 1 else "spectra"

    if ext in (".csv", ".txt"):
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            raw_lines = []
            for _ in range(12):
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    raw_lines.append(line.rstrip("\r\n"))

        if not raw_lines:
            raise ValueError(f"Empty file: {p.name}")

        if ext == ".csv":
            l0 = raw_lines[0].strip()
            l1 = raw_lines[1].strip() if len(raw_lines) > 1 else ""

            # 1. Explicit wafer / SPECTROview map CSV signatures
            if "Dynamic Sitebased Spectral" in l0 or "SPECTROview Map Data" in l0:
                return "map"
            if l1:
                l1_tokens = [t.strip() for t in l1.split(";") if t.strip()]
                if len(l1_tokens) >= 2 and l1_tokens[0].upper() == "X" and l1_tokens[1].upper() == "Y":
                    return "map"

            # 2. Check delimiter on first line (semicolon or comma)
            delim = ";" if ";" in l0 else ("," if "," in l0 else None)
            if delim is not None:
                l0_tokens = [t.strip() for t in l0.split(delim)]
                while len(l0_tokens) > 1 and l0_tokens[-1] == "":
                    l0_tokens.pop()
                if l0_tokens and not _is_float_token(l0_tokens[0]):
                    # Semicolon-header CSVs and multi-column comma-header CSVs are tabular datasheets
                    if delim == ";" or len(l0_tokens) != 2:
                        return "dataframe"
                    # 2-column comma CSV with text header: check if data row is numeric (spectrum) vs text (dataframe)
                    if len(raw_lines) > 1:
                        d_tokens = [t.strip() for t in raw_lines[1].split(",") if t.strip()]
                        if len(d_tokens) == 2 and all(_is_float_token(t) for t in d_tokens):
                            return "spectra"
                    return "dataframe"

            # 3. Legacy 3-header-row or headerless CSV spectrum / map
            for cand in raw_lines:
                c_delim = ";" if ";" in cand else ("," if "," in cand else None)
                tokens = [t.strip() for t in (cand.split(c_delim) if c_delim else cand.split()) if t.strip()]
                if len(tokens) >= 2 and all(_is_float_token(t) for t in tokens):
                    if len(tokens) == 2:
                        return "spectra"
                    if len(tokens) > 3:
                        return "map"
            raise ValueError(f"Invalid number of columns in {p.name}")

        # .txt classification
        # Find first data line after optional comment/header lines
        data_line = None
        for idx, cand in enumerate(raw_lines):
            stripped = cand.strip()
            if stripped.startswith("#"):
                continue
            delim = _detect_text_delimiter(stripped, allow_comma=False)
            tokens = [t for t in _split_line(stripped, delim) if t != ""]
            if tokens and all(_is_float_token(t) for t in tokens):
                # Note: for LabSpec6 maps, row 0 may be wavenumbers only (M floats) while row 1 has Y, X + M floats
                if idx == 0 and len(raw_lines) > 1:
                    next_s = raw_lines[1].strip()
                    next_d = _detect_text_delimiter(next_s, allow_comma=False)
                    next_t = [t for t in _split_line(next_s, next_d) if t != ""]
                    if next_t and all(_is_float_token(t) for t in next_t):
                        tokens = next_t
                data_line = tokens
                break

        if data_line is None:
            raise ValueError(f"No valid numeric spectral data found in {p.name}")

        n_cols = len(data_line)
        if n_cols == 2:
            return "spectra"
        if n_cols > 3:
            return "map"
        raise ValueError(f"Invalid number of columns ({n_cols}) in {p.name}")

    raise ValueError(f"Unsupported file format: {ext}")


def _build_map_dataframe(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    wavenumber_cols: list[str],
    spectra_matrix: np.ndarray,
) -> pd.DataFrame:
    """Construct a consolidated Map DataFrame without per-column dict fragmentation."""
    if not spectra_matrix.flags["C_CONTIGUOUS"] and not spectra_matrix.flags["F_CONTIGUOUS"]:
        mat = np.ascontiguousarray(spectra_matrix)
    else:
        mat = spectra_matrix
    df = pd.DataFrame(mat, columns=wavenumber_cols, copy=False)
    df.insert(0, "Y", np.asarray(y_coords, dtype=np.float64))
    df.insert(0, "X", np.asarray(x_coords, dtype=np.float64))
    return df


def extract_map_arrays(
    map_name: str,
    map_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract ``(x0, Y0, coords, fnames)`` from a map DataFrame for ``SpectraStore``.

    - Filters out any trailing/non-numeric ``Unnamed:`` columns while preserving
      all valid numeric wavenumber channels.
    - Extracts ``Y0`` (`float32[N, M]`) and ``coords`` (`float64[N, 2]`) with
      minimal copying.
    - Generates unique pixel ``fnames`` (``"{map_name}_({x}, {y})"``), appending
      ``"_{k}"`` suffixes only when duplicate ``(X, Y)`` coordinates occur (e.g.,
      time-resolved multi-scan maps at fixed wafer sites).
    """
    col_indices: list[int] = []
    x_values: list[float] = []
    for idx, col in enumerate(map_df.columns):
        if col in ("X", "Y"):
            continue
        try:
            val = float(col)
            if not np.isnan(val):
                col_indices.append(idx)
                x_values.append(val)
        except (ValueError, TypeError):
            continue

    x0 = np.asarray(x_values, dtype=np.float64)

    x_positions = map_df["X"].to_numpy(dtype=np.float64, copy=False)
    y_positions = map_df["Y"].to_numpy(dtype=np.float64, copy=False)
    coords = np.column_stack((x_positions, y_positions))

    # Fast positional slice without forcing a strided Fortran->C transpose on large blocks
    n_total_cols = map_df.shape[1]
    if col_indices and col_indices[0] == 2 and col_indices[-1] == n_total_cols - 1 and len(col_indices) == n_total_cols - 2:
        Y0 = map_df.iloc[:, 2:].to_numpy().astype(np.float32, copy=False)
    else:
        Y0 = map_df.iloc[:, col_indices].to_numpy().astype(np.float32, copy=False)

    # Build pixel fnames in C-backed Python float lists; disambiguate duplicate (X, Y) sites
    x_list = x_positions.tolist()
    y_list = y_positions.tolist()
    n_rows = len(x_list)
    base_fnames = [f"{map_name}_({x_list[i]}, {y_list[i]})" for i in range(n_rows)]

    if len(set(base_fnames)) == n_rows:
        fnames = base_fnames
    else:
        counts: dict[str, int] = {}
        fnames = []
        for base in base_fnames:
            c = counts.get(base, 0) + 1
            counts[base] = c
            fnames.append(base if c == 1 else f"{base}_{c}")

    return x0, Y0, coords, fnames


def load_spectrum_file(path: Path) -> dict:
    """Load 1D discrete spectrum from a TXT or CSV file.

    Automatically detects delimiter (``;``, ``,``, ``\\t``, or whitespace) and
    skips any leading header/comment rows without dropping data row 0 on
    headerless files. Ensures ascending X-axis ordering.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in (".txt", ".csv"):
        raise ValueError(f"Unsupported file type: {ext}")

    allow_comma = ext == ".csv"
    skiprows = 0
    delimiter: str | None = None
    has_repeated_tabs = False

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_idx in range(20):
            raw = f.readline()
            if not raw:
                break
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                skiprows = line_idx + 1
                continue

            cand_delim = _detect_text_delimiter(stripped, allow_comma=allow_comma)
            if cand_delim == "\t" and "\t\t" in stripped:
                tokens = [t.strip() for t in stripped.split("\t") if t.strip()]
                if len(tokens) >= 2 and all(_is_float_token(t) for t in tokens[:2]):
                    delimiter = r"\s+"
                    has_repeated_tabs = True
                    skiprows = line_idx
                    break
            else:
                tokens = _split_line(stripped, cand_delim)
                # Strip trailing empty token caused by trailing ';' or '\t'
                while len(tokens) > 2 and tokens[-1] == "":
                    tokens.pop()
                if len(tokens) >= 2 and _is_float_token(tokens[0]) and _is_float_token(tokens[1]):
                    delimiter = cand_delim
                    skiprows = line_idx
                    break

            skiprows = line_idx + 1

    if has_repeated_tabs:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            skiprows=skiprows,
            comment="#",
            usecols=[0, 1],
            dtype=np.float64,
            engine="python",
        )
    elif delimiter is None:
        df = pd.read_csv(
            path,
            sep=" ",
            skipinitialspace=True,
            header=None,
            skiprows=skiprows,
            comment="#",
            dtype=np.float64,
            engine="c",
        )
        if df.shape[1] > 2:
            df = df.iloc[:, :2]
    else:
        df = pd.read_csv(
            path,
            sep=delimiter,
            header=None,
            skiprows=skiprows,
            comment="#",
            dtype=np.float64,
            engine="c",
        )
        if df.shape[1] > 2:
            df = df.iloc[:, :2]

    if df.empty or df.shape[1] < 2:
        raise ValueError(f"No valid 2-column spectrum data found in {path.name}")

    arr = df.to_numpy(dtype=np.float64, copy=False)
    x0 = arr[:, 0]
    y0 = arr[:, 1]

    # Drop NaN rows if any trailing blank/comment artifacts existed
    if np.isnan(x0).any() or np.isnan(y0).any():
        valid = ~(np.isnan(x0) | np.isnan(y0))
        x0 = x0[valid]
        y0 = y0[valid]

    if len(x0) == 0:
        raise ValueError(f"Spectrum file {path.name} contains no valid numeric points")

    # Sort by ascending X-axis only when not already sorted
    if len(x0) > 1 and not np.all(x0[:-1] <= x0[1:]):
        order = np.argsort(x0, kind="mergesort")
        x0 = np.ascontiguousarray(x0[order])
        y0 = np.ascontiguousarray(y0[order])
    else:
        x0 = np.ascontiguousarray(x0)
        y0 = np.ascontiguousarray(y0)

    return {
        "type": "spectra",
        "items": [
            {
                "name": path.stem,
                "x0": x0,
                "y0": y0,
                "metadata": {},
            }
        ],
    }


def load_map_file(path: Path) -> pd.DataFrame:
    """Load 2D hyperspectral or wafer map file (CSV or TXT).

    Supports:
    - Standard multi-column CSV wafer maps (``Title`` + ``X;Y;wn0;wn1;...``)
    - Alternating-row CSV wafer/time-resolved maps (``Title`` + ``X;Y;`` + alternating wavenumber/intensity rows)
    - LabSpec6 / Renishaw converted tab/semicolon TXT maps (``Y, X`` in cols 0, 1)
    - Space-delimited CL / hyperspectral TXT maps

    Automatically strips trailing empty delimiter columns (``Unnamed: ...``) and
    orders spectral columns by ascending wavenumber.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".csv":
        with open(path, "r", encoding="utf-8", errors="replace") as file:
            lines = [file.readline() for _ in range(3)]

        if not lines[0] or not lines[1]:
            raise ValueError(f"Invalid or truncated CSV map file: {path.name}")

        raw_l1_tokens = [t.strip() for t in lines[1].rstrip("\r\n").split(";")]
        non_empty_l1 = [t for t in raw_l1_tokens if t]

        if len(non_empty_l1) > 2:
            # New format: Line 1 has X;Y;wn_0;wn_1;... (often with trailing ';')
            valid_indices = [0, 1]
            wn_labels: list[str] = []
            for col_idx in range(2, len(raw_l1_tokens)):
                tok = raw_l1_tokens[col_idx]
                if tok and _is_float_token(tok):
                    valid_indices.append(col_idx)
                    wn_labels.append(tok)

            if not wn_labels:
                raise ValueError(f"No numeric wavenumber columns found in {path.name}")

            raw_df = pd.read_csv(
                path,
                skiprows=2,
                sep=";",
                header=None,
                usecols=valid_indices,
                dtype=np.float64,
                engine="c",
            )
        else:
            # Old format: alternating wavenumber / intensity rows starting at line 2
            if not lines[2]:
                raise ValueError(f"Truncated alternating-row CSV map file: {path.name}")

            raw_l2_tokens = [t.strip() for t in lines[2].rstrip("\r\n").split(";")]
            valid_indices = [0, 1]
            wn_labels = []
            for col_idx in range(2, len(raw_l2_tokens)):
                tok = raw_l2_tokens[col_idx]
                if tok and _is_float_token(tok):
                    valid_indices.append(col_idx)
                    # Normalize numeric label string (e.g. '1.000' -> '1.0') matching legacy float parse
                    wn_labels.append(str(float(tok)))

            if not wn_labels:
                raise ValueError(f"No numeric wavenumber values found in {path.name}")

            # Skip lines 0, 1 and all even wavenumber rows (2, 4, 6...) directly in C parser
            raw_df = pd.read_csv(
                path,
                sep=";",
                header=None,
                skiprows=lambda r: r < 3 or (r & 1) == 0,
                usecols=valid_indices,
                dtype=np.float64,
                engine="c",
            )

        # Ensure ascending wavenumber order
        wn_floats = np.array([float(w) for w in wn_labels], dtype=np.float64)
        if len(wn_floats) > 1 and not np.all(wn_floats[:-1] <= wn_floats[1:]):
            order = np.argsort(wn_floats, kind="mergesort")
            perm = np.empty(2 + len(order), dtype=np.intp)
            perm[0] = 0
            perm[1] = 1
            perm[2:] = order + 2
            map_df = raw_df.iloc[:, perm]
            wn_labels = [wn_labels[i] for i in order]
        else:
            map_df = raw_df

        map_df.columns = ["X", "Y"] + wn_labels
        return map_df

    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
            second_line = f.readline()

        if not first_line or not second_line:
            raise ValueError(f"Invalid or truncated TXT map file: {path.name}")

        test_line = second_line if second_line.strip() else first_line
        if ";" in test_line:
            delimiter = ";"
        elif "\t" in test_line:
            delimiter = "\t"
        else:
            delimiter = " "

        if delimiter == " ":
            header_tokens = first_line.strip().split()
            raw_df = pd.read_csv(
                path,
                sep=" ",
                skipinitialspace=True,
                header=None,
                skiprows=1,
                engine="c",
            )
            n_cols = raw_df.shape[1]
            wavenumbers = header_tokens[2:] if len(header_tokens) == n_cols else header_tokens
        else:
            raw_hdr = [t.strip() for t in first_line.rstrip("\r\n").split(delimiter)]
            raw_data = [t.strip() for t in second_line.rstrip("\r\n").split(delimiter)]
            # Strip trailing empty tokens caused by trailing tabs/semicolons
            while len(raw_hdr) > 2 and raw_hdr[-1] == "":
                raw_hdr.pop()
            while len(raw_data) > 2 and raw_data[-1] == "":
                raw_data.pop()

            n_cols = len(raw_data)
            if len(raw_hdr) == n_cols:
                wavenumbers = raw_hdr[2:]
            elif len(raw_hdr) == n_cols - 2:
                wavenumbers = raw_hdr
            else:
                # Fallback for irregular headers
                wavenumbers = [t for t in raw_hdr if _is_float_token(t)]
                if len(wavenumbers) == n_cols:
                    wavenumbers = wavenumbers[2:]
                n_cols = 2 + len(wavenumbers)

            raw_df = pd.read_csv(
                path,
                sep=delimiter,
                header=None,
                skiprows=1,
                engine="c",
            )
            if raw_df.shape[1] > n_cols:
                raw_df = raw_df.iloc[:, :n_cols]

        if len(wavenumbers) != raw_df.shape[1] - 2 or len(wavenumbers) == 0:
            raise ValueError(f"Header wavenumber count does not match data columns in {path.name}")

        # TXT maps store Y in col 0 and X in col 1.
        # When wavenumbers are already ascending, swap cols 0 and 1 in place to avoid copying the 2D spectral block.
        wn_floats = np.array([float(w) for w in wavenumbers], dtype=np.float64)
        if len(wn_floats) > 1 and not np.all(wn_floats[:-1] <= wn_floats[1:]):
            order = np.argsort(wn_floats, kind="mergesort")
            perm = np.empty(raw_df.shape[1], dtype=np.intp)
            perm[0] = 1
            perm[1] = 0
            perm[2:] = order + 2
            map_df = raw_df.iloc[:, perm]
            sorted_wns = [wavenumbers[i] for i in order]
        else:
            y_vals = raw_df.iloc[:, 0].to_numpy().copy()
            x_vals = raw_df.iloc[:, 1].to_numpy().copy()
            raw_df[raw_df.columns[0]] = x_vals
            raw_df[raw_df.columns[1]] = y_vals
            map_df = raw_df
            sorted_wns = list(wavenumbers)

        map_df.columns = ["X", "Y"] + sorted_wns
        return map_df

    raise ValueError(f"Unsupported file type: {ext}")


def parse_wdf_metadata(reader) -> dict:
    """Extract extended instrument metadata from WDF WXIS, WXCS, WXDM, and WDF1 blocks.

    Pure-Python binary block parser (no Qt or ViewModel dependencies).
    """
    metadata: dict = {}
    block_info = getattr(reader, "block_info", None) or {}
    file_obj = getattr(reader, "file_obj", None)
    if file_obj is None:
        return metadata

    # 1. Parse WXIS block for objective, grating, slit, and laser power
    if "WXIS" in block_info:
        _uid, pos, size = block_info["WXIS"]
        if size > 16:
            file_obj.seek(pos + 16)
            wxis_data = file_obj.read(size - 16)
            try:
                text_data = wxis_data.decode("utf-8", errors="ignore")

                for objective in ("x100", "x50", "x20", "x10", "x5", "x2"):
                    if objective in text_data:
                        metadata["objective_name"] = objective
                        break

                for pattern in ("1800 l/mm", "2400 l/mm", "1200 l/mm", "600 l/mm", "300 l/mm", "150 l/mm"):
                    if pattern in text_data:
                        metadata["grating_name"] = pattern
                        break

                all_um_values = _UM_INT_RE.findall(text_data)
                if len(all_um_values) >= 3:
                    try:
                        slit_opening = float(all_um_values[1])
                        slit_centre = float(all_um_values[2])
                        if 5 <= slit_opening <= 500 and 50 <= slit_centre <= 5000:
                            metadata["slit_opening"] = slit_opening
                            metadata["slit_centre"] = slit_centre
                    except (ValueError, IndexError):
                        pass

                # Laser power: 'u' record with id=0x0500 near offset 2300-2500
                needle = b"u\x00\x05\x80"
                search_start = 2300
                search_end = min(len(wxis_data), 2512)
                idx = wxis_data.find(needle, search_start, search_end)
                while idx != -1 and idx + 8 <= len(wxis_data):
                    str_len = struct.unpack_from("<i", wxis_data, idx + 4)[0]
                    if 0 < str_len < 20 and idx + 8 + str_len <= len(wxis_data):
                        power_str = wxis_data[idx + 8 : idx + 8 + str_len].decode("utf-8", errors="ignore").strip("\x00")
                        if power_str and not any(c.isalpha() for c in power_str):
                            try:
                                power_val = float(power_str)
                                if 0.01 <= power_val <= 100:
                                    metadata["laser_power"] = power_val
                                    break
                            except ValueError:
                                pass
                    idx = wxis_data.find(needle, idx + 1, search_end)
            except Exception:
                pass

    # 2. Parse WXCS block for confocal slit values (fallback)
    if "WXCS" in block_info and ("slit_opening" not in metadata or "slit_centre" not in metadata):
        _uid, pos, size = block_info["WXCS"]
        if size > 16:
            file_obj.seek(pos + 16)
            wxcs_data = file_obj.read(size - 16)
            try:
                text_data = wxcs_data.decode("utf-8", errors="ignore")
                all_um_values = _UM_FLOAT_RE.findall(text_data)
                for i in range(len(all_um_values) - 1):
                    try:
                        val1 = float(all_um_values[i])
                        val2 = float(all_um_values[i + 1])
                        if 5 <= val1 <= 500 and 50 <= val2 <= 5000 and "slit_opening" not in metadata:
                            metadata["slit_opening"] = min(val1, val2)
                            metadata["slit_centre"] = max(val1, val2)
                            break
                    except ValueError:
                        pass
            except Exception:
                pass

    # 3. Parse WXDM block for exposure time ('i' record with id=0x2300, value in ms)
    if "WXDM" in block_info:
        _uid, pos, size = block_info["WXDM"]
        if size > 16:
            file_obj.seek(pos + 16)
            wxdm_data = file_obj.read(size - 16)
            try:
                needle = b"i\x00\x23\x80"
                idx = wxdm_data.find(needle)
                while idx != -1 and idx + 8 <= len(wxdm_data):
                    val = struct.unpack_from("<i", wxdm_data, idx + 4)[0]
                    if 10 <= val <= 3600000:
                        metadata["exposure_time"] = val / 1000.0
                        break
                    idx = wxdm_data.find(needle, idx + 1)
            except Exception:
                pass

    # 4. Parse WDF1 header for acquisition timestamp
    found_date = None
    try:
        cached_hdr = getattr(reader, "_wdf1_bytes", b"")
        if cached_hdr and len(cached_hdr) >= 512:
            header_bytes = cached_hdr
        else:
            file_obj.seek(0)
            header_bytes = file_obj.read(1024)

        def decode_filetime(ft_bytes: bytes) -> str | None:
            ft_int = int.from_bytes(ft_bytes, "little")
            if ft_int < 116444736000000000:
                return None
            ts = (ft_int - 116444736000000000) / 10000000
            return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

        if len(header_bytes) >= 0x98:
            found_date = decode_filetime(header_bytes[0x90:0x98])
        if not found_date and len(header_bytes) >= 0x90:
            found_date = decode_filetime(header_bytes[0x88:0x90])

        if not found_date:
            def check_systemtime(buf: bytes, offset: int) -> str | None:
                if offset + 16 > len(buf):
                    return None
                wYear, wMonth, _wDow, wDay, wHour, wMinute, wSecond, _wMs = struct.unpack_from("<8H", buf, offset)
                if 1990 <= wYear <= 2035 and 1 <= wMonth <= 12 and 1 <= wDay <= 31 and wHour <= 23 and wMinute <= 59 and wSecond <= 59:
                    return f"{wYear}-{wMonth:02d}-{wDay:02d} {wHour:02d}:{wMinute:02d}:{wSecond:02d}"
                return None

            for i in range(0xA0, 0xF0, 2):
                found_date = check_systemtime(header_bytes, i)
                if found_date:
                    break
            if not found_date:
                for i in range(0, 512, 2):
                    found_date = check_systemtime(header_bytes, i)
                    if found_date:
                        break
    except Exception:
        pass

    if not found_date:
        try:
            fname = getattr(file_obj, "name", None)
            if fname and os.path.exists(fname):
                ts = os.path.getmtime(fname)
                found_date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

    if found_date:
        metadata["timestamp"] = found_date

    return metadata


def load_wdf_spectrum(path: Path) -> dict:
    """Load single spectrum or temporal series from a Renishaw ``.wdf`` file."""
    path = Path(path)
    reader_cls = _require_wdf()
    reader = reader_cls(str(path))
    try:
        wavenumbers = np.asarray(reader.xdata, dtype=np.float64)
        sort_inds = None
        is_reversed = False
        if len(wavenumbers) > 1 and wavenumbers[0] > wavenumbers[-1]:
            if np.all(wavenumbers[:-1] >= wavenumbers[1:]):
                is_reversed = True
                wavenumbers = np.ascontiguousarray(wavenumbers[::-1])
            else:
                sort_inds = np.argsort(wavenumbers)
                wavenumbers = np.ascontiguousarray(wavenumbers[sort_inds])
        else:
            wavenumbers = np.array(wavenumbers, dtype=np.float64, copy=True)

        wdf_metadata = parse_wdf_metadata(reader)
        metadata = _reorder_metadata(_construct_wdf_metadata(reader, wdf_metadata, is_map=False))

        mtype_str = str(reader.measurement_type)
        is_series = mtype_str == "Series" or "Series" in mtype_str

        if is_series and reader.spectra.ndim > 1:
            timestamps = None
            if hasattr(reader, "origin_list_header") and reader.origin_list_header is not None:
                for item in reader.origin_list_header:
                    if len(item) >= 5 and "Time" in str(item[3]):
                        ts_arr = np.asarray(item[4], dtype=np.float64)
                        # Convert Windows FILETIME (100-ns ticks) to relative seconds if needed
                        if len(ts_arr) > 0 and ts_arr[0] > 1e14:
                            ts_arr = (ts_arr - ts_arr[0]) / 1e7
                        timestamps = ts_arr.tolist()
                        break

            # Vectorized conversion & column sort for the entire series matrix
            series_mat = np.asarray(reader.spectra, dtype=np.float64)
            if series_mat.ndim > 2:
                series_mat = series_mat.reshape(-1, series_mat.shape[-1])
            if is_reversed:
                series_mat = np.ascontiguousarray(series_mat[:, ::-1])
            elif sort_inds is not None:
                series_mat = np.ascontiguousarray(series_mat[:, sort_inds])

            n_items = min(int(reader.count), series_mat.shape[0])
            stem = path.stem
            spectra_list = []
            for i in range(n_items):
                item_metadata = metadata.copy()
                if timestamps is not None and i < len(timestamps):
                    item_metadata["Time (s)"] = round(float(timestamps[i]), 2)
                spectra_list.append(
                    {
                        "name": f"{stem}_{i}",
                        "x0": wavenumbers.copy(),
                        "y0": series_mat[i].copy(),
                        "metadata": item_metadata,
                    }
                )

            return {"type": "spectra", "items": spectra_list}

        if reader.spectra.ndim == 1:
            raw_y = reader.spectra
        else:
            raw_y = reader.spectra[0] if reader.count == 1 else reader.spectra.mean(axis=0)

        intensities = np.asarray(raw_y, dtype=np.float64)
        if is_reversed:
            intensities = np.ascontiguousarray(intensities[::-1])
        elif sort_inds is not None:
            intensities = np.ascontiguousarray(intensities[sort_inds])
        else:
            intensities = intensities.copy()

        return {
            "type": "spectra",
            "items": [
                {
                    "name": path.stem,
                    "x0": wavenumbers,
                    "y0": intensities,
                    "metadata": metadata,
                }
            ],
        }
    finally:
        reader.close()


def load_wdf_map(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load 2D hyperspectral map or 1D line-scan map from a Renishaw ``.wdf`` file."""
    path = Path(path)
    reader_cls = _require_wdf()
    reader = reader_cls(str(path))
    try:
        x_coords = np.round(np.asarray(reader.xpos, dtype=np.float64), 4)
        y_coords = np.round(np.asarray(reader.ypos, dtype=np.float64), 4)
        wavenumbers = np.asarray(reader.xdata)
        spectra_data = np.asarray(reader.spectra)

        if spectra_data.ndim >= 3:
            spectra_matrix = spectra_data.reshape(-1, spectra_data.shape[-1])
        elif spectra_data.ndim == 1:
            spectra_matrix = spectra_data.reshape(1, -1)
        else:
            spectra_matrix = spectra_data

        if len(wavenumbers) > 1 and wavenumbers[0] > wavenumbers[-1]:
            if np.all(wavenumbers[:-1] >= wavenumbers[1:]):
                wavenumbers = wavenumbers[::-1]
                spectra_matrix = np.ascontiguousarray(spectra_matrix[:, ::-1])
            else:
                sort_inds = np.argsort(wavenumbers)
                wavenumbers = wavenumbers[sort_inds]
                spectra_matrix = spectra_matrix[:, sort_inds]

        wavenumber_cols = [str(wn) for wn in wavenumbers]
        df = _build_map_dataframe(x_coords, y_coords, wavenumber_cols, spectra_matrix)

        wdf_metadata = parse_wdf_metadata(reader)

        x_step = None
        y_step = None
        if len(reader.xpos) > 1:
            x_unique = np.unique(reader.xpos)
            x_step = float(x_unique[1] - x_unique[0]) if len(x_unique) > 1 else 0.0
        if len(reader.ypos) > 1:
            y_unique = np.unique(reader.ypos)
            y_step = float(y_unique[1] - y_unique[0]) if len(y_unique) > 1 else 0.0

        metadata = _construct_wdf_metadata(reader, wdf_metadata, x_step, y_step, is_map=True)
        return df, _reorder_metadata(metadata)
    finally:
        reader.close()


def load_dataframe_file(path: Path) -> dict[str, pd.DataFrame]:
    """Load tabular DataFrame(s) from Excel (``.xlsx``/``.xls``) or ``.csv`` file."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".xlsx", ".xls"):
        with pd.ExcelFile(path) as excel_file:
            sheet_names = excel_file.sheet_names
            if len(sheet_names) == 1:
                return {path.stem: pd.read_excel(excel_file)}
            return {
                f"{path.stem}_{sheet_name}": pd.read_excel(excel_file, sheet_name=sheet_name)
                for sheet_name in sheet_names
            }

    if ext == ".csv":
        # Inspect first line to choose primary delimiter without reading the file twice
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
        primary_sep = ";" if first_line.count(";") >= first_line.count(",") and ";" in first_line else ","
        try:
            df = pd.read_csv(path, sep=primary_sep)
            if df.shape[1] == 1:
                alt_sep = "," if primary_sep == ";" else ";"
                if alt_sep in first_line:
                    df_alt = pd.read_csv(path, sep=alt_sep)
                    if df_alt.shape[1] > 1:
                        df = df_alt
        except Exception:
            df = pd.read_csv(path)
        return {path.stem: df}

    raise ValueError(f"Unsupported file type: {ext}")


def load_spc_spectrum(path: Path) -> dict:
    """Load discrete spectrum from a Galactic / HORIBA ``.spc`` file."""
    path = Path(path)
    reader = SpcReader(str(path))

    y_raw = reader.y_data[0] if reader.header["fnsub"] > 1 else reader.y_data
    x0 = np.asarray(reader.x_data, dtype=np.float64)
    y0 = np.asarray(y_raw, dtype=np.float64)

    if len(x0) > 1 and x0[0] > x0[-1]:
        sort_inds = np.argsort(x0)
        x0 = np.ascontiguousarray(x0[sort_inds])
        y0 = np.ascontiguousarray(y0[sort_inds])
    else:
        x0 = x0.copy()
        y0 = y0.copy()

    x_unit = _extract_spc_x_unit(reader)
    metadata: dict = {
        "File Format": "Galactic SPC",
        "Points": reader.header["npts"],
        "X-axis Unit": x_unit,
    }
    if reader.header.get("date"):
        metadata["Date"] = reader.header["date"]

    if reader.log_metadata:
        metadata.update(_map_spc_metadata(reader.log_metadata))

    return {
        "type": "spectra",
        "items": [
            {
                "name": path.stem,
                "x0": x0,
                "y0": y0,
                "metadata": _reorder_metadata(metadata),
            }
        ],
    }


def load_spc_map(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load 2D hyperspectral map from a Galactic / HORIBA ``.spc`` file."""
    path = Path(path)
    reader = SpcReader(str(path))
    num_spectra = reader.header["fnsub"]

    if reader.sub_time is not None and reader.sub_wlevel is not None:
        extracted_x = reader.sub_time
        extracted_y = reader.sub_wlevel
    elif reader.subheaders:
        extracted_x = np.array([sh.get("time", 0.0) for sh in reader.subheaders], dtype=np.float64)
        extracted_y = np.array([sh.get("wlevel", 0.0) for sh in reader.subheaders], dtype=np.float64)
    else:
        extracted_x = np.zeros(num_spectra, dtype=np.float64)
        extracted_y = np.zeros(num_spectra, dtype=np.float64)

    if np.any(extracted_x) or np.any(extracted_y):
        x_coords = extracted_x
        y_coords = extracted_y
    else:
        side = int(np.sqrt(num_spectra))
        if side * side == num_spectra:
            y_indices, x_indices = np.unravel_index(np.arange(num_spectra), (side, side))
            x_coords = x_indices.astype(np.float64)
            y_coords = y_indices.astype(np.float64)
        else:
            x_coords = np.arange(num_spectra, dtype=np.float64)
            y_coords = np.zeros(num_spectra, dtype=np.float64)

    wavenumbers = np.asarray(reader.x_data, dtype=np.float64)
    spectra_matrix = np.asarray(reader.y_data, dtype=np.float64)
    if spectra_matrix.ndim == 1:
        spectra_matrix = spectra_matrix.reshape(1, -1)

    if len(wavenumbers) > 1 and wavenumbers[0] > wavenumbers[-1]:
        sort_inds = np.argsort(wavenumbers)
        wavenumbers = wavenumbers[sort_inds]
        spectra_matrix = spectra_matrix[:, sort_inds]

    wavenumber_cols = [str(wn) for wn in wavenumbers]
    df = _build_map_dataframe(np.round(x_coords, 4), np.round(y_coords, 4), wavenumber_cols, spectra_matrix)

    x_unit = _extract_spc_x_unit(reader)
    metadata: dict = {
        "File Format": "Galactic SPC Map",
        "Total Map Points": num_spectra,
        "Points per Spectrum": reader.header["npts"],
        "X-axis Unit": x_unit,
    }
    if reader.header.get("date"):
        metadata["Date"] = reader.header["date"]

    if reader.log_metadata:
        metadata.update(_map_spc_metadata(reader.log_metadata))

    return df, _reorder_metadata(metadata)


def _extract_spc_x_unit(reader: SpcReader) -> str:
    """Determine X-axis unit from SPC header ``fxtype``, ``fcatxt``, and LabSpec log."""
    x_unit = "Arbitrary"

    # 1. Galactic SPC fxtype codes (spc.h specification):
    # 1=Wavenumber (1/cm), 2=um, 3=nm, 4=s, 13=Raman Shift (1/cm), 14=eV
    fxtype = reader.header.get("fxtype", 0)
    if fxtype in (1, 13):
        x_unit = "1/cm"
    elif fxtype == 2:
        x_unit = "µm"
    elif fxtype == 3:
        x_unit = "nm"
    elif fxtype == 4:
        x_unit = "s"
    elif fxtype == 14:
        x_unit = "eV"

    # 2. Explicit axis text label overrides fxtype when present
    fcatxt = reader.header.get("fcatxt", "")
    if fcatxt:
        fcatxt_lower = fcatxt.lower()
        if "cm-1" in fcatxt_lower or "1/cm" in fcatxt_lower or "wavenumber" in fcatxt_lower or "raman" in fcatxt_lower:
            x_unit = "1/cm"
        elif "nm" in fcatxt_lower or "nanometer" in fcatxt_lower:
            x_unit = "nm"
        elif "ev" in fcatxt_lower:
            x_unit = "eV"

    # 3. LabSpec6 log block overrides (e.g. PL files that leave fxtype=13 while SPECTROMETER (NM) is active)
    log_str = getattr(reader, "log_content", "") or ""
    if log_str:
        if "SPECTROMETER (NM)" in log_str:
            x_unit = "nm"
        elif "SPECTROMETER (CM-)" in log_str:
            x_unit = "1/cm"
        elif x_unit == "Arbitrary":
            if "eV" in log_str:
                x_unit = "eV"
            elif "RAMAN" in log_str:
                x_unit = "1/cm"

    return x_unit


def _construct_wdf_metadata(
    reader,
    wdf_metadata: dict,
    x_step: float | None = None,
    y_step: float | None = None,
    is_map: bool = False,
) -> dict:
    """Construct standardized WDF acquisition metadata dictionary."""
    treat_as_map = is_map or (x_step is not None) or (y_step is not None)

    metadata = {
        "File Format": "Renishaw WDF Map" if treat_as_map else "Renishaw WDF",
        "Grating": wdf_metadata.get("grating_name", "Unknown"),
        "Objective Used": wdf_metadata.get("objective_name", "Unknown"),
        "Laser Wavelength (nm)": f"{reader.laser_length:.2f}",
        "Laser Power (%)": f"{wdf_metadata['laser_power']}" if "laser_power" in wdf_metadata else "Unknown",
        "Exposure Time (s)": f"{wdf_metadata['exposure_time']}" if "exposure_time" in wdf_metadata else "Unknown",
        "Accumulations": reader.accumulation_count,
        "Slit Opening (µm)": f"{wdf_metadata['slit_opening']}" if "slit_opening" in wdf_metadata else "Unknown",
        "Slit Centre (µm)": f"{wdf_metadata['slit_centre']}" if "slit_centre" in wdf_metadata else "Unknown",
        "Title": reader.title,
        "Username": reader.username,
        "Date": wdf_metadata.get("timestamp", "Unknown"),
        "Application": f"{reader.application_name} v{'.'.join(map(str, reader.application_version))}",
        "Measurement Type": str(reader.measurement_type),
        "Scan Type": str(reader.scan_type),
        "Total Map Points": reader.count,
    }

    if treat_as_map and hasattr(reader, "xpos") and hasattr(reader, "ypos") and len(reader.xpos) > 0:
        metadata.update(
            {
                "X Range (µm)": f"{reader.xpos.min():.2f} to {reader.xpos.max():.2f}",
                "Y Range (µm)": f"{reader.ypos.min():.2f} to {reader.ypos.max():.2f}",
                "X Step Size (µm)": f"{x_step:.3f}" if x_step is not None else "Unknown",
                "Y Step Size (µm)": f"{y_step:.3f}" if y_step is not None else "Unknown",
            }
        )

    metadata.update(
        {
            "Points per Spectrum": reader.point_per_spectrum,
            "Spectral Unit": reader.spectral_unit,
            "X-axis Type": reader.xlist_type,
            "X-axis Unit": reader.xlist_unit,
            "Wavenumber Range": f"{reader.xdata[0]:.2f} to {reader.xdata[-1]:.2f} {reader.xlist_unit}",
        }
    )
    return metadata


def _map_spc_metadata(log_metadata: dict) -> dict:
    """Map raw SPC log keys to standardized SPECTROview metadata keys."""
    mapping_rules = (
        ("Grating", ("GRATING",)),
        ("Objective Used", ("OBJECTIV", "OBJECTIVE")),
        ("Laser Wavelength (nm)", ("EXCIT_LINE", "EXCIT LINE", "LASER WAVELENGTH", "WAVELENGTH")),
        ("Laser Power (%)", ("ND FILTER", "POWER", "LASER POWER", "LASER_POWER")),
        ("Exposure Time (s)", ("ACQ. TIME (S)", "TIME", "ACQ TIME (S)", "ACQ TIME", "EXPOSURE TIME", "EXP TIME")),
        ("Accumulations", ("ACCUMULATIONS", "ACCUM", "ACCUMULATION", "NB ACCUM")),
        ("Slit Opening (µm)", ("HOLE (M)", "HOLE", "SLIT OPENING")),
        ("Date", ("DATE",)),
    )

    new_metadata: dict = {}
    mapped_source_keys: set[str] = set()

    for standard_key, candidate_keys in mapping_rules:
        for spc_key in candidate_keys:
            mapped_source_keys.add(spc_key)
            if spc_key in log_metadata and standard_key not in new_metadata:
                val = log_metadata[spc_key]
                if val is not None and str(val).strip() != "":
                    new_metadata[standard_key] = str(val).strip()

    for k, v in log_metadata.items():
        if k not in mapped_source_keys:
            new_metadata[k] = v

    return new_metadata


def _reorder_metadata(metadata: dict) -> dict:
    """Reorder metadata keys to match the standard SPECTROview display order."""
    new_metadata = {k: metadata[k] for k in _METADATA_ORDERED_KEYS if k in metadata}
    for key, value in metadata.items():
        if key not in new_metadata:
            new_metadata[key] = value
    return new_metadata


def load_TRPL_data(path: Path) -> dict:
    """Load Time-Resolved Photoluminescence (TRPL) ``.dat`` decay file.

    Extracts bin resolution after ``#ns/bin`` and photon counts after ``#counts``
    up to the first zero count, aligning ``t = 0`` at the decay peak maximum.
    """
    path = Path(path)
    bin_value: float | None = None
    channels_per_curve: int | None = None
    header_instrument = ""
    header_date = ""
    y_values: list[int] = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        in_counts = False
        expect_bin = False
        expect_channels = False
        first_comment_seen = False
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if expect_bin:
                bin_value = float(line)
                expect_bin = False
                continue
            if expect_channels:
                try:
                    channels_per_curve = int(line)
                except ValueError:
                    pass
                expect_channels = False
                continue
            if in_counts:
                try:
                    value = int(line)
                except ValueError:
                    break
                if value == 0:
                    break
                y_values.append(value)
                continue
            if not first_comment_seen and line.startswith("#") and not line.startswith(("#ns/bin", "#counts", "#channels")):
                first_comment_seen = True
                parts = [p.strip() for p in line[1:].split("\t") if p.strip()]
                if parts:
                    header_instrument = parts[0]
                if len(parts) > 1:
                    header_date = parts[-1]
            elif line.startswith("#channels"):
                expect_channels = True
            elif line.startswith("#ns/bin"):
                expect_bin = True
            elif line.startswith("#counts"):
                in_counts = True

    if bin_value is None or not y_values:
        raise ValueError("Invalid TRPL file format: missing bin value or counts")

    counts_arr = np.asarray(y_values, dtype=np.float64)
    max_y_index = int(np.argmax(counts_arr))
    y0 = np.ascontiguousarray(counts_arr[max_y_index:])
    x0 = np.arange(len(y0), dtype=np.float64) * bin_value

    trpl_metadata: dict = {
        "File Format": "TRPL (.dat)",
        "X-axis Unit": "ns",
        "Bin Resolution (ns)": f"{bin_value:g}",
        "Points": len(x0),
    }
    if header_date:
        trpl_metadata["Date"] = header_date
    if header_instrument:
        trpl_metadata["Instrument"] = header_instrument
    if channels_per_curve is not None:
        trpl_metadata["Channels per Curve"] = channels_per_curve

    return {
        "type": "spectra",
        "items": [
            {
                "name": path.stem,
                "x0": x0,
                "y0": y0,
                "metadata": _reorder_metadata(trpl_metadata),
            }
        ],
    }


def convert_action(input_path: str | Path, output_path: str | Path) -> None:
    """Convert a Renishaw InVia long-format TXT export into wide-matrix TXT format.

    Vectorized Qt-free implementation supporting both single-tab and double-tab
    Renishaw WiRE text exports without temporary file copies.
    """
    in_p = Path(input_path)
    out_p = Path(output_path)

    try:
        with open(in_p, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
            second_line = f.readline()
        has_double_tab = ("\t\t" in first_line) or ("\t\t" in second_line)
        if has_double_tab:
            dfr = pd.read_csv(in_p, sep=r"\t+", engine="python")
        else:
            dfr = pd.read_csv(in_p, sep="\t", engine="c")
        if not {"#X", "#Y", "#Wave", "#Intensity"}.issubset(dfr.columns):
            raise ValueError("Missing required columns (#X, #Y, #Wave, #Intensity)")
    except Exception as e:
        raise ValueError(f"Invalid file format: {e}") from e

    if dfr.empty:
        raise ValueError("Input file contains no data rows")

    # Preserve original scan order (sort=False prevents scrambling snake/descending grids)
    dfr_spectrum = dfr.groupby(["#Y", "#X"], sort=False).size().reset_index(name="size")
    sizes = dfr_spectrum["size"].to_numpy()
    wavenumber_range = int(sizes[0])
    total_spectra_number = len(dfr_spectrum)

    if not np.all(sizes == wavenumber_range):
        raise ValueError("Inconsistent number of wavenumber points across spatial coordinates")

    wavenumbers = [str(w) for w in dfr["#Wave"].iloc[:wavenumber_range].tolist()]
    intensity_matrix = dfr["#Intensity"].to_numpy().reshape(total_spectra_number, wavenumber_range)

    out_df = pd.DataFrame(intensity_matrix, columns=wavenumbers, copy=False)
    out_df.insert(0, "#X", dfr_spectrum["#X"].to_numpy())
    out_df.insert(0, "#Y", dfr_spectrum["#Y"].to_numpy())
    out_df = out_df.rename(columns={"#X": "", "#Y": ""})

    out_df.to_csv(out_p, sep="\t", encoding="utf-8", index=False)
