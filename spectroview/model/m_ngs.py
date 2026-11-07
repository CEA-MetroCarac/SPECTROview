"""Native binary reader for HORIBA LabSpec 5 (.ngs and .ngc) spectroscopic files.

Decodes the proprietary ``NGSNextGen`` binary container:
- Version 1 (``.ngs``): Uncompressed MFC ``CArchive`` block stream (``DataMatrix`` +
  ``Table`` blocks ``Param``, ``Acq``, ``Extra``, ``Hist``).
- Version 3 (``.ngc``): Haruhiko Okumura ``lzhuf`` (Adaptive Huffman + LZSS)
  compressed bitstream followed by byte-level delta coding (``np.cumsum`` over
  ``uint8``), yielding the exact ``NGSNextGen`` block stream.

Implemented in pure Python ``struct`` and NumPy without external dependencies or
Qt imports.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import struct
from typing import Any

import numpy as np

_NGS_MAGIC = b"NGSNextGen"

# Precomputed 14-bit LZSS position lookup tables for Haruhiko Okumura's lzhuf
# d_code / d_len step boundaries: 32 (3b), 80 (4b), 144 (5b), 192 (6b), 240 (7b), 256 (8b)
_POS_DIST: list[int] = [0] * 16384
_POS_BITS: list[int] = [0] * 16384
for _b14 in range(16384):
    _i = _b14 >> 6
    if _i < 32:
        _c, _l = 0, 3
    elif _i < 80:
        _c, _l = 1 + ((_i - 32) >> 4), 4
    elif _i < 144:
        _c, _l = 4 + ((_i - 80) >> 3), 5
    elif _i < 192:
        _c, _l = 12 + ((_i - 144) >> 2), 6
    elif _i < 240:
        _c, _l = 24 + ((_i - 192) >> 1), 7
    else:
        _c, _l = 48 + (_i - 240), 8
    _tb = 6 + _l
    _POS_DIST[_b14] = ((_c << 6) | ((_b14 >> (14 - _tb)) & 0x3F)) + 1
    _POS_BITS[_b14] = _tb


def _decompress_ngc(payload: bytes | memoryview, uncomp_size: int) -> bytes:
    """Decompress a LabSpec 5 ``.ngc`` (v3) payload using ``lzhuf`` + byte delta coding."""
    if uncomp_size <= 0:
        return b""

    N = 4096
    F = 60
    N_CHAR = 314
    T = 627
    R = 626
    MAX_FREQ = 0x8000

    freq = [1] * N_CHAR + [0] * (T + 1 - N_CHAR)
    prnt = [0] * (T + N_CHAR)
    son = [0] * T
    for i in range(N_CHAR):
        son[i] = i + T
        prnt[i + T] = i
    i, j = 0, N_CHAR
    while j <= R:
        freq[j] = freq[i] + freq[i + 1]
        son[j] = i
        prnt[i] = prnt[i + 1] = j
        i += 2
        j += 1
    freq[T] = 0xFFFF
    prnt[R] = 0

    out = bytearray(N + uncomp_size + F)
    out_idx = N
    end_idx = N + uncomp_size
    src_bytes = bytes(payload) + b"\x00" * 16
    src_pos = 4
    bitbuf = (src_bytes[0] << 24) | (src_bytes[1] << 16) | (src_bytes[2] << 8) | src_bytes[3]
    bitcnt = 32
    p_dist = _POS_DIST
    p_bits = _POS_BITS

    while out_idx < end_idx:
        if bitcnt <= 24:
            bitbuf = (
                ((bitbuf & 0xFFFFFF) << 24)
                | (src_bytes[src_pos] << 16)
                | (src_bytes[src_pos + 1] << 8)
                | src_bytes[src_pos + 2]
            )
            src_pos += 3
            bitcnt += 24

        c = son[R]
        while c < T:
            bitcnt -= 1
            c = son[c + ((bitbuf >> bitcnt) & 1)]
        c -= T

        if freq[R] == MAX_FREQ:
            j = 0
            for i in range(T):
                if son[i] >= T:
                    freq[j] = (freq[i] + 1) >> 1
                    son[j] = son[i]
                    j += 1
            i = 0
            for j in range(N_CHAR, T):
                k = i + 1
                f = freq[i] + freq[k]
                freq[j] = f
                k = j - 1
                while f < freq[k]:
                    k -= 1
                k += 1
                freq[k + 1 : j + 1] = freq[k:j]
                freq[k] = f
                son[k + 1 : j + 1] = son[k:j]
                son[k] = i
                i += 2
            for i in range(T):
                k = son[i]
                prnt[k] = i
                if k < T:
                    prnt[k + 1] = i

        u = prnt[c + T]
        while True:
            k = freq[u] + 1
            freq[u] = k
            if k > freq[u + 1]:
                l = u + 2
                while k > freq[l]:
                    l += 1
                l -= 1
                freq[u] = freq[l]
                freq[l] = k
                i = son[u]
                prnt[i] = l
                if i < T:
                    prnt[i + 1] = l
                j = son[l]
                son[l] = i
                prnt[j] = u
                if j < T:
                    prnt[j + 1] = u
                son[u] = j
                u = l
            u = prnt[u]
            if u == 0:
                break

        if c < 256:
            out[out_idx] = c
            out_idx += 1
        else:
            if bitcnt < 14:
                bitbuf = ((bitbuf & 0x3FFF) << 16) | (src_bytes[src_pos] << 8) | src_bytes[src_pos + 1]
                src_pos += 2
                bitcnt += 16
            b14 = (bitbuf >> (bitcnt - 14)) & 0x3FFF
            bitcnt -= p_bits[b14]
            dist = p_dist[b14]
            match_len = c - 253
            if out_idx + match_len > end_idx:
                match_len = end_idx - out_idx
            src = out_idx - dist
            if dist >= match_len:
                out[out_idx : out_idx + match_len] = out[src : src + match_len]
                out_idx += match_len
            else:
                for k in range(match_len):
                    out[out_idx] = out[src + k]
                    out_idx += 1

    arr = np.frombuffer(out, dtype=np.uint8, count=uncomp_size, offset=N)
    return np.cumsum(arr, dtype=np.uint8).tobytes()


def _read_carchive_count(buf: bytes | memoryview, pos: int) -> tuple[int, int]:
    """Read an MFC ``CArchive`` count (``uint16``, or ``0xFFFF`` + ``uint32``)."""
    val = struct.unpack_from("<H", buf, pos)[0]
    pos += 2
    if val == 0xFFFF:
        val = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
    return val, pos


def _read_pstring(buf: bytes | memoryview, pos: int) -> tuple[str, int]:
    """Read a length-prefixed ``CArchive`` string (Windows-1252 / Latin-1)."""
    length = buf[pos]
    pos += 1
    if length == 0xFF:
        length, pos = _read_carchive_count(buf, pos)
    raw = bytes(buf[pos : pos + length])
    return raw.decode("cp1252", errors="replace").strip(), pos + length


def _format_scalar(val: Any) -> Any:
    """Clean scalar string/numeric values for metadata display."""
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return ""
        try:
            fval = float(s)
            if np.isfinite(fval):
                if "." not in s and "e" not in s.lower():
                    return int(s)
                rounded = round(fval, 6)
                return int(rounded) if rounded == int(rounded) and abs(rounded) < 1e12 else rounded
        except ValueError:
            pass
        return s
    return val


class NgsReader:
    """Native binary reader for HORIBA LabSpec 5 ``.ngs`` and ``.ngc`` files."""

    @staticmethod
    def peek_header(filename: str | Path) -> dict[str, Any]:
        """Validate the ``NGSNextGen`` header and inspect the first block in < 1 ms."""
        p = Path(filename)
        with open(p, "rb") as f:
            hdr = f.read(256)

        if len(hdr) < 24 or not hdr.startswith(_NGS_MAGIC):
            raise ValueError(f"Invalid LabSpec 5 NGSNextGen header in {p.name}")

        version = struct.unpack_from("<I", hdr, 10)[0]
        if version == 1:
            stream = hdr
            pos = 18
        elif version == 3:
            uncomp_size, comp_size = struct.unpack_from("<II", hdr, 16)
            if uncomp_size < 16 or comp_size == 0:
                raise ValueError(f"Corrupt LabSpec 5 .ngc compression header in {p.name}")
            stream = _decompress_ngc(hdr[24:], min(96, uncomp_size))
            pos = 2
        else:
            raise ValueError(f"Unsupported LabSpec 5 NGSNextGen version {version} in {p.name}")

        try:
            block_type, pos = _read_pstring(stream, pos)
            subtype, pos = _read_pstring(stream, pos)
            label, pos = _read_pstring(stream, pos)
            pos += 4  # block_size
            n_dims = 1
            if block_type == "DataMatrix" and pos + 18 <= len(stream):
                _, _, _, _, n_dims = struct.unpack_from("<IIiIH", stream, pos)
        except Exception as exc:
            raise ValueError(f"Truncated LabSpec 5 block header in {p.name}: {exc}") from exc

        is_map = subtype.lower() in ("spim", "map", "profile") or n_dims >= 2
        return {
            "version": version,
            "is_compressed": version == 3,
            "block_type": block_type,
            "subtype": subtype,
            "label": label,
            "n_dims": n_dims,
            "is_map": is_map,
        }

    def __init__(self, filename: str | Path) -> None:
        self.filename = str(filename)
        p = Path(filename)
        raw = p.read_bytes()

        if len(raw) < 24 or not raw.startswith(_NGS_MAGIC):
            raise ValueError(f"Invalid LabSpec 5 NGSNextGen binary file: {p.name}")

        self.version: int = struct.unpack_from("<I", raw, 10)[0]
        if self.version == 1:
            stream: bytes | memoryview = memoryview(raw)
            pos = 18
        elif self.version == 3:
            uncomp_size, comp_size = struct.unpack_from("<II", raw, 16)
            if uncomp_size < 16 or comp_size == 0 or 24 + comp_size > len(raw):
                raise ValueError(f"Truncated or corrupt LabSpec 5 .ngc payload in {p.name}")
            stream = _decompress_ngc(memoryview(raw)[24 : 24 + comp_size], uncomp_size)
            pos = 2
        else:
            raise ValueError(f"Unsupported LabSpec 5 NGSNextGen version {self.version} in {p.name}")

        data_matrix: dict[str, Any] | None = None
        self.tables: dict[str, dict[str, Any]] = {}

        stream_len = len(stream)
        while pos < stream_len:
            t_type, pos = _read_pstring(stream, pos)
            t_sub, pos = _read_pstring(stream, pos)
            t_label, pos = _read_pstring(stream, pos)
            if pos + 4 > stream_len:
                break
            bsize = struct.unpack_from("<I", stream, pos)[0]
            pos += 4
            if pos + bsize > stream_len:
                raise ValueError(f"Truncated LabSpec 5 {t_type} block in {p.name}")
            block = stream[pos : pos + bsize]
            pos += bsize

            if t_type == "DataMatrix":
                data_matrix = self._parse_data_matrix(block, t_sub, t_label, p.name)
            elif t_type == "Table":
                self.tables[t_sub] = self._parse_table(block, t_label)

        if data_matrix is None:
            raise ValueError(f"LabSpec 5 file '{p.name}' does not contain a DataMatrix block.")

        self.record_type: str = data_matrix["subtype"] or "Spectrum"
        self.title: str = data_matrix["label"]
        self.dims: tuple[int, ...] = data_matrix["dims"]
        self.axis_names: list[str] = data_matrix["axis_names"]
        self.axis_units: list[str] = data_matrix["axis_units"]
        self.axis_order: tuple[int, ...] = data_matrix["axis_order"]
        self.axes: dict[str, dict[str, Any]] = data_matrix["axes"]

        spectr_info = self.axes.get("Spectr")
        intens_info = self.axes.get("Intens")
        if spectr_info is None or spectr_info.get("values") is None:
            # Fallback to first non-null axis if named differently
            for ax_name, ax_val in self.axes.items():
                if ax_val.get("values") is not None:
                    spectr_info = ax_val
                    self.x_type = ax_name
                    break
            else:
                raise ValueError(f"LabSpec 5 file '{p.name}' has no spectral coordinate axis.")
        else:
            self.x_type = "Spectr"

        self.x_unit: str = str(spectr_info.get("unit") or "1/cm").strip()
        self.spectral_unit: str = str((intens_info or {}).get("unit") or "Cnt").strip()

        self.x_data: np.ndarray = np.asarray(spectr_info["values"], dtype=np.float64)
        y_flat: np.ndarray = data_matrix["intens"].astype(np.float64)

        n_pts = len(self.x_data)
        if n_pts == 0 or y_flat.size == 0 or (y_flat.size % n_pts) != 0:
            raise ValueError(
                f"Inconsistent LabSpec 5 spectral dimensions in '{p.name}': "
                f"x_data={n_pts}, y_data={y_flat.size}"
            )

        self.n_pts: int = n_pts
        self.num_spectra: int = y_flat.size // n_pts

        self.x_axis: np.ndarray | None = None
        self.y_axis: np.ndarray | None = None
        self.x_coords: np.ndarray = np.zeros(self.num_spectra, dtype=np.float64)
        self.y_coords: np.ndarray = np.zeros(self.num_spectra, dtype=np.float64)
        self.x_step: float | None = None
        self.y_step: float | None = None

        if self.num_spectra == 1 and len(self.dims) == 1:
            self.y_data: np.ndarray = y_flat.copy()
        else:
            self.y_data = y_flat.reshape(self.num_spectra, n_pts)
            if "X" in self.axes and self.axes["X"].get("values") is not None:
                self.x_axis = np.asarray(self.axes["X"]["values"], dtype=np.float64)
            if "Y" in self.axes and self.axes["Y"].get("values") is not None:
                self.y_axis = np.asarray(self.axes["Y"]["values"], dtype=np.float64)

            if self.x_axis is not None and self.y_axis is not None:
                nx = len(self.x_axis)
                ny = len(self.y_axis)
                if nx * ny == self.num_spectra:
                    spatial_order = [
                        self.axis_names[idx].upper()
                        for idx in self.axis_order
                        if idx < len(self.axis_names) and self.axis_names[idx].upper() in ("X", "Y")
                    ]
                    if spatial_order == ["X", "Y"]:
                        self.x_coords = np.repeat(self.x_axis, ny)
                        self.y_coords = np.tile(self.y_axis, nx)
                    else:
                        # Standard HORIBA LabSpec 5/6 order: Y outer loop (dims[0]), X inner loop (dims[1])
                        self.x_coords = np.tile(self.x_axis, ny)
                        self.y_coords = np.repeat(self.y_axis, nx)
                if nx > 1:
                    self.x_step = float(np.median(np.abs(np.diff(self.x_axis))))
                if ny > 1:
                    self.y_step = float(np.median(np.abs(np.diff(self.y_axis))))
            elif self.x_axis is not None and len(self.x_axis) == self.num_spectra:
                self.x_coords = self.x_axis.copy()
                self.y_coords = np.zeros(self.num_spectra, dtype=np.float64)
                if len(self.x_axis) > 1:
                    self.x_step = float(np.median(np.abs(np.diff(self.x_axis))))
            elif self.y_axis is not None and len(self.y_axis) == self.num_spectra:
                self.x_coords = np.zeros(self.num_spectra, dtype=np.float64)
                self.y_coords = self.y_axis.copy()
                if len(self.y_axis) > 1:
                    self.y_step = float(np.median(np.abs(np.diff(self.y_axis))))

        # Extract metadata from Table blocks ('Param', 'Acq', 'Extra', 'Hist')
        self.tables_metadata: dict[str, Any] = {}
        self.laser_wavelength_nm: float | None = None
        self._extract_tables_metadata()

    @staticmethod
    def _parse_data_matrix(
        block: bytes | memoryview,
        subtype: str,
        label: str,
        fname: str,
    ) -> dict[str, Any]:
        """Parse a LabSpec 5 ``DataMatrix`` block."""
        _, _, _, total_pts, n_dims = struct.unpack_from("<IIiIH", block, 0)
        dims = struct.unpack_from(f"<{n_dims}I", block, 18)
        off = 18 + 4 * n_dims

        byte_len, off = _read_carchive_count(block, off)
        if off + byte_len > len(block):
            raise ValueError(f"Truncated DataMatrix intensity array in {fname}")
        intens = np.frombuffer(block, dtype="<f4", count=byte_len // 4, offset=off).copy()
        off += byte_len

        n_axes = struct.unpack_from("<H", block, off)[0]
        off += 2
        axis_names: list[str] = []
        for _ in range(n_axes):
            name, off = _read_pstring(block, off)
            axis_names.append(name)

        n_units = struct.unpack_from("<H", block, off)[0]
        off += 2
        axis_units: list[str] = []
        for _ in range(n_units):
            unit, off = _read_pstring(block, off)
            axis_units.append(unit)

        # Per-axis byte flags (uint16 count + count * uint8)
        n_flags = struct.unpack_from("<H", block, off)[0]
        off += 2 + n_flags

        # Axis dimension ordering (uint16 count + count * uint32)
        n_order = struct.unpack_from("<H", block, off)[0]
        off += 2
        axis_order = struct.unpack_from(f"<{n_order}I", block, off)
        off += 4 * n_order

        # Axis min/max bounds (uint16 count + count * float32)
        n_bound_floats = struct.unpack_from("<H", block, off)[0]
        off += 2
        bounds = struct.unpack_from(f"<{n_bound_floats}f", block, off)
        off += 4 * n_bound_floats

        # Axis coordinate array pointers
        n_ptrs = struct.unpack_from("<H", block, off)[0]
        off += 2
        ptrs = struct.unpack_from(f"<{n_ptrs}I", block, off)
        off += 4 * n_ptrs

        axes: dict[str, dict[str, Any]] = {}
        for ax_idx, ptr in enumerate(ptrs):
            ax_name = axis_names[ax_idx] if ax_idx < len(axis_names) else f"Axis_{ax_idx}"
            ax_unit = axis_units[ax_idx] if ax_idx < len(axis_units) else ""
            if ptr != 0:
                ax_bytes, off = _read_carchive_count(block, off)
                ax_arr = np.frombuffer(block, dtype="<f4", count=ax_bytes // 4, offset=off).copy()
                off += ax_bytes
                axes[ax_name] = {"unit": ax_unit, "values": ax_arr}
            else:
                axes[ax_name] = {"unit": ax_unit, "values": None}

        return {
            "subtype": subtype,
            "label": label,
            "total_pts": total_pts,
            "dims": dims,
            "intens": intens,
            "axis_names": axis_names,
            "axis_units": axis_units,
            "axis_order": axis_order,
            "bounds": bounds,
            "axes": axes,
        }

    @staticmethod
    def _parse_table(block: bytes | memoryview, label: str) -> dict[str, Any]:
        """Parse a LabSpec 5 ``Table`` block (``Param``, ``Acq``, ``Extra``, ``Hist``)."""
        off = 0
        n_rows = struct.unpack_from("<H", block, off)[0]
        off += 2
        row_names: list[str] = []
        for _ in range(n_rows):
            rn, off = _read_pstring(block, off)
            row_names.append(rn)

        n_cols = struct.unpack_from("<H", block, off)[0]
        off += 2
        col_names: list[str] = []
        for _ in range(n_cols):
            cn, off = _read_pstring(block, off)
            col_names.append(cn)

        n_col_types = struct.unpack_from("<H", block, off)[0]
        off += 2
        col_types = struct.unpack_from(f"<{n_col_types}I", block, off)
        off += 4 * n_col_types

        cols_data: dict[str, list[Any]] = {}
        for c_idx in range(n_cols):
            c_name = col_names[c_idx]
            c_type = col_types[c_idx] if c_idx < len(col_types) else 0
            if (c_type & 0xFF) == 3:
                b_len, off = _read_carchive_count(block, off)
                n_vals = b_len // 4
                vals = list(struct.unpack_from(f"<{n_vals}I", block, off))
                off += b_len
                cols_data[c_name] = vals
            else:
                n_vals = struct.unpack_from("<H", block, off)[0]
                off += 2
                s_vals: list[str] = []
                for _ in range(n_vals):
                    v, off = _read_pstring(block, off)
                    s_vals.append(v)
                cols_data[c_name] = s_vals

        return {
            "label": label,
            "rows": row_names,
            "cols": col_names,
            "data": cols_data,
        }

    def _extract_tables_metadata(self) -> None:
        """Flatten ``Param``, ``Acq``, ``Extra``, and ``Hist`` tables into ``tables_metadata``."""
        # 1. Param table ('Name', 'Date', 'Operator', 'Sample', 'Remark', 'Power')
        param_tbl = self.tables.get("Param")
        if param_tbl:
            rows = param_tbl["rows"]
            vals = param_tbl["data"].get("Value", [])
            for idx, rname in enumerate(rows):
                if not rname or rname == "Name" or idx >= len(vals):
                    continue
                val = _format_scalar(vals[idx])
                if val != "" and val is not None:
                    self.tables_metadata[rname] = val

        # 2. Acq table ('Instrument', 'Exposition', 'Accumulation', 'Laser', 'Grating', 'Objective', ...)
        acq_tbl = self.tables.get("Acq")
        if acq_tbl:
            rows = acq_tbl["rows"]
            vals = acq_tbl["data"].get("Value", [])
            units = acq_tbl["data"].get("Unit", [])
            # Keys whose unit is already represented in the standardized key name
            unit_in_key_names = {"Exposition", "Hole", "Laser"}
            for idx, rname in enumerate(rows):
                if not rname or idx >= len(vals):
                    continue
                raw_val = vals[idx].strip() if isinstance(vals[idx], str) else str(vals[idx])
                if not raw_val:
                    continue
                unit = units[idx].strip() if idx < len(units) and isinstance(units[idx], str) else ""
                clean_val = _format_scalar(raw_val)

                if rname.lower() == "laser" and isinstance(clean_val, (int, float)):
                    self.laser_wavelength_nm = float(clean_val)

                if unit and rname not in unit_in_key_names:
                    self.tables_metadata[rname] = f"{clean_val} {unit}"
                else:
                    self.tables_metadata[rname] = clean_val

        # 3. Extra table ('ICSResponse', 'ICSSource', ...)
        extra_tbl = self.tables.get("Extra")
        if extra_tbl:
            rows = extra_tbl["rows"]
            vals = extra_tbl["data"].get("Value", [])
            for idx, rname in enumerate(rows):
                if not rname or idx >= len(vals):
                    continue
                val = _format_scalar(vals[idx])
                if val != "" and val is not None:
                    self.tables_metadata[rname] = val

        # 4. Hist table (Unix timestamp + Operation, e.g. 'Acquired')
        hist_tbl = self.tables.get("Hist")
        if hist_tbl:
            dates = hist_tbl["data"].get("Date", [])
            ops = hist_tbl["data"].get("Operation", [])
            for idx, ts in enumerate(dates):
                op = ops[idx].strip() if idx < len(ops) and isinstance(ops[idx], str) else ""
                if isinstance(ts, int) and ts > 0:
                    dt_str = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    if "Date" not in self.tables_metadata:
                        self.tables_metadata["Date"] = dt_str
                    if op and op not in self.tables_metadata:
                        self.tables_metadata[op] = dt_str

