"""Native binary reader for HORIBA LabSpec 6 (.l6s and .l6m) spectroscopic files.

Decodes the proprietary ``LabSpec6`` binary serialization container (76-byte file
header + recursive 24-byte Depth-First Search node descriptors with ``block_id``
pointer deduplication) using pure Python ``struct`` and NumPy without external
dependencies or Qt imports.
"""

from __future__ import annotations

from pathlib import Path
import struct
from typing import Any

import numpy as np

_LABSPEC6_MAGIC = b"LabSpec6\x00"
_HEADER_SIZE = 0x4C  # 76 bytes before root 24-byte node descriptor

_DTYPE_SIZES: dict[int, int] = {
    0x00: 0,
    0x01: 1,   # uint8 / bool
    0x02: 2,   # int16
    0x03: 2,   # uint16
    0x04: 4,   # int32
    0x05: 8,   # float64
    0x06: 4,   # float32 (spectral and coordinate arrays)
    0x07: 1,   # char (UTF-8 / Windows-1252 null-terminated string)
    0x08: 2,   # wchar_t (UTF-16LE null-terminated string)
    0x09: 24,  # nested 24-byte descriptor array
}

_KNOWN_TAGS: dict[bytes, str] = {
    b"\xd7ati": "datas",
    b"typm": "type",
    b"\xd9itl": "title",
    b"\xd6\xd9tz": "data",
    b"axi{": "axis",
    b"uni|": "unit",
    b"\xdb\xd4l}": "values",
    b"\xdbal}": "value",
    b"\xe3tam": "stat",
    b"mean": "mean",
    b"\xe0\xdcmi": "limits",
    b"temx": "temp",
    b"\xd9\xd4bl": "tables",
    b"acq\x08": "acq",
    b"\xd2\xe2s|": "desc",
    b"\xd7\xdb\xec|": "history",
    b"namm": "name",
    b"pos\x08": "pos",
    b"film": "file",
    b"expo": "expo",
    b"holm": "hole",
    b"sitm": "site",
    b"datm": "date",
    b"\xe7\xd7\xe1i": "snr",
    b"H\xd6\xe4\xda": "scan_info",
    b"stex": "step",
    b"sizm": "size",
    b"modm": "mode",
    b"x\x00\x00\x08": "x",
    b"y\x00\x00\x08": "y",
    b"z\x00\x00\x08": "z",
}


def _decode_tag(tag4: bytes) -> str:
    """Return human-readable property name for a 4-byte LabSpec 6 folded tag."""
    if tag4 in _KNOWN_TAGS:
        return _KNOWN_TAGS[tag4]
    idx = struct.unpack_from("<I", tag4, 0)[0]
    if idx < 1024 and tag4[2:] == b"\x00\x00":
        return f"[{idx}]"
    return "".join(chr(b) if 32 <= b < 127 else f"\\x{b:02x}" for b in tag4.rstrip(b"\x00"))


def _decode_cstr(raw: bytes) -> str:
    """Decode null-terminated single-byte string (UTF-8 or Windows-1252)."""
    chunk = raw.split(b"\x00", 1)[0]
    try:
        return chunk.decode("utf-8").strip()
    except UnicodeDecodeError:
        return chunk.decode("cp1252", errors="replace").strip()


def _format_scalar(val: Any) -> Any:
    """Clean scalar metadata value (strip strings and round float64 representation noise)."""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, (float, np.floating)):
        fval = float(val)
        if np.isfinite(fval):
            rounded = round(fval, 6)
            if rounded == int(rounded):
                return int(rounded) if abs(rounded) < 1e12 else rounded
            return rounded
    if isinstance(val, (int, np.integer)):
        return int(val)
    return val


def _parse_inline_value(dtype: int, raw8: bytes) -> Any:
    """Decode an 8-byte inline descriptor value."""
    if dtype == 0x00:
        return None
    if dtype == 0x01:
        return int(raw8[0])
    if dtype == 0x02:
        return int(struct.unpack_from("<h", raw8, 0)[0])
    if dtype == 0x03:
        return int(struct.unpack_from("<H", raw8, 0)[0])
    if dtype == 0x04:
        return int(struct.unpack_from("<i", raw8, 0)[0])
    if dtype == 0x05:
        return float(struct.unpack_from("<d", raw8, 0)[0])
    if dtype == 0x06:
        return float(struct.unpack_from("<f", raw8, 0)[0])
    if dtype == 0x07:
        return _decode_cstr(raw8)
    if dtype == 0x08:
        u16 = np.frombuffer(raw8, dtype="<u2")
        zero_pos = np.where(u16 == 0)[0]
        if len(zero_pos):
            u16 = u16[: zero_pos[0]]
        return u16.tobytes().decode("utf-16le", errors="replace").strip()
    if dtype == 0x09:
        return []
    return raw8.hex()


def _parse_node(
    buf: bytes | memoryview,
    raw24: bytes | memoryview,
    cursor: int,
    block_table: dict[int, Any],
) -> tuple[dict[str, Any], int]:
    """Recursively parse a 24-byte LabSpec 6 descriptor and its DFS payload."""
    dtype_flags, field_id = struct.unpack_from("<II", raw24, 0)
    tag4 = bytes(raw24[8:12])
    dtype = dtype_flags & 0xFF
    is_outline = (dtype_flags & 0x1000) != 0
    is_shared_ref = (dtype_flags & 0x8000) != 0

    if not is_outline:
        return {
            "dtype": dtype,
            "flags": dtype_flags,
            "field_id": field_id,
            "tag_raw": tag4,
            "tag": _decode_tag(tag4),
            "block_id": 0,
            "value": _parse_inline_value(dtype, bytes(raw24[16:24])),
        }, cursor

    block_id, count = struct.unpack_from("<II", raw24, 16)

    # Transient runtime-only "temp" ('temx') node is never persisted to disk
    if tag4 == b"temx":
        return {
            "dtype": dtype,
            "flags": dtype_flags,
            "field_id": field_id,
            "tag_raw": tag4,
            "tag": "temp",
            "block_id": block_id,
            "value": [],
        }, cursor

    # Shared pointer deduplication (flag 0x8000 or previously decoded block_id)
    if is_shared_ref or (block_id != 0 and block_id in block_table):
        return {
            "dtype": dtype,
            "flags": dtype_flags,
            "field_id": field_id,
            "tag_raw": tag4,
            "tag": _decode_tag(tag4),
            "block_id": block_id,
            "value": block_table.get(block_id),
        }, cursor

    if dtype == 0x09:
        desc_end = cursor + count * 24
        if desc_end > len(buf):
            raise ValueError(f"Truncated LabSpec 6 descriptor array at 0x{cursor:X} (count={count})")
        child_raws = [buf[cursor + i * 24 : cursor + (i + 1) * 24] for i in range(count)]
        cursor = desc_end
        children: list[dict[str, Any]] = []
        if block_id != 0:
            block_table[block_id] = children
        for craw in child_raws:
            child_node, cursor = _parse_node(buf, craw, cursor, block_table)
            children.append(child_node)
        val: Any = children
    else:
        elem_size = _DTYPE_SIZES.get(dtype)
        if elem_size is None:
            raise ValueError(f"Unsupported LabSpec 6 dtype 0x{dtype:02X} at 0x{cursor:X}")
        nbytes = count * elem_size
        if cursor + nbytes > len(buf):
            raise ValueError(f"Truncated LabSpec 6 payload at 0x{cursor:X} (dtype=0x{dtype:02X}, count={count})")
        payload = buf[cursor : cursor + nbytes]
        cursor += nbytes

        if dtype == 0x06:
            val = np.frombuffer(payload, dtype="<f4").astype(np.float64)
        elif dtype == 0x05:
            val = np.frombuffer(payload, dtype="<f8").copy()
        elif dtype == 0x04:
            val = np.frombuffer(payload, dtype="<i4").copy()
        elif dtype == 0x07:
            val = _decode_cstr(bytes(payload))
        elif dtype == 0x08:
            val = bytes(payload).decode("utf-16le", errors="replace").split("\x00", 1)[0].strip()
        elif dtype == 0x01:
            val = np.frombuffer(payload, dtype=np.uint8).copy()
        elif dtype == 0x02:
            val = np.frombuffer(payload, dtype="<i2").copy()
        elif dtype == 0x03:
            val = np.frombuffer(payload, dtype="<u2").copy()
        else:
            val = bytes(payload)

        if block_id != 0:
            block_table[block_id] = val

    return {
        "dtype": dtype,
        "flags": dtype_flags,
        "field_id": field_id,
        "tag_raw": tag4,
        "tag": _decode_tag(tag4),
        "block_id": block_id,
        "value": val,
    }, cursor


class L6sReader:
    """Native binary reader for HORIBA LabSpec 6 ``.l6s`` and ``.l6m`` files."""

    @staticmethod
    def peek_header(filename: str | Path) -> dict[str, Any]:
        """Validate the 76-byte LabSpec 6 binary header without reading full data arrays."""
        p = Path(filename)
        with open(p, "rb") as f:
            hdr = f.read(128)

        if len(hdr) < _HEADER_SIZE + 24:
            raise ValueError(f"File too small to be a valid LabSpec 6 binary file: {p.name}")
        if not hdr.startswith(_LABSPEC6_MAGIC):
            raise ValueError(f"Invalid LabSpec 6 magic header in {p.name}")

        root_tag = hdr[20:24]
        cls_id = struct.unpack_from("<I", hdr, 24)[0]
        version = struct.unpack_from("<I", hdr, 0x44)[0]
        root_df, _, _, root_blk, root_cnt = struct.unpack_from("<II8sII", hdr, _HEADER_SIZE)
        if (root_df & 0xFF) != 0x09 or root_cnt == 0:
            raise ValueError(f"Invalid LabSpec 6 root descriptor in {p.name}")

        return {
            "root_tag": _decode_tag(root_tag),
            "class_id": cls_id,
            "version": version,
            "root_count": root_cnt,
            "root_block_id": root_blk,
        }

    def __init__(self, filename: str | Path) -> None:
        self.filename = str(filename)
        p = Path(filename)
        raw = p.read_bytes()

        if len(raw) < _HEADER_SIZE + 24 or not raw.startswith(_LABSPEC6_MAGIC):
            raise ValueError(f"Invalid LabSpec 6 binary file: {p.name}")

        mv = memoryview(raw)
        self.block_table: dict[int, Any] = {}
        self.root_node, self.end_cursor = _parse_node(
            mv, mv[_HEADER_SIZE : _HEADER_SIZE + 24], _HEADER_SIZE + 24, self.block_table
        )

        root_map: dict[str, Any] = {}
        for ch in self.root_node.get("value", []):
            root_map[ch["tag"]] = ch["value"]

        self.record_type: str = str(root_map.get("type") or "Spectrum").strip()
        self.title: str = str(root_map.get("title") or "").strip()
        self.origin_file: str = str(root_map.get("file") or "").strip()

        raw_snr = str(root_map.get("snr") or "").strip()
        if raw_snr.upper().startswith("S/N="):
            self.snr: str = raw_snr.split("=", 1)[1].strip()
        else:
            self.snr = raw_snr

        # Parse axes from data -> axis
        data_children = root_map.get("data") or []
        axis_container = data_children[0]["value"] if data_children and isinstance(data_children[0], dict) else []
        self.axes: list[dict[str, Any]] = []
        for ax_node in axis_container:
            if isinstance(ax_node, dict) and isinstance(ax_node.get("value"), list):
                ax_dict = {c["tag"]: c["value"] for c in ax_node["value"]}
                self.axes.append(ax_dict)

        if len(self.axes) < 2:
            raise ValueError(f"LabSpec 6 file '{p.name}' does not contain valid spectral axes.")

        intens_ax = self.axes[0]
        spectr_ax = self.axes[1]

        self.spectral_unit: str = str(intens_ax.get("unit") or "Cnt").strip()
        self.x_unit: str = str(spectr_ax.get("unit") or "1/cm").strip()
        self.x_type: str = str(spectr_ax.get("type") or "Spectr").strip()

        self.x_data: np.ndarray = np.asarray(spectr_ax.get("values"), dtype=np.float64)
        y_flat: np.ndarray = np.asarray(intens_ax.get("values"), dtype=np.float64)

        n_pts = len(self.x_data)
        if n_pts == 0 or y_flat.size == 0 or (y_flat.size % n_pts) != 0:
            raise ValueError(
                f"Inconsistent LabSpec 6 spectral dimensions in '{p.name}': "
                f"x_data={n_pts}, y_data={y_flat.size}"
            )

        self.n_pts: int = n_pts
        self.num_spectra: int = y_flat.size // n_pts

        # Spatial / series coordinate axes (Axis[2], Axis[3], ...)
        self.x_axis: np.ndarray | None = None
        self.y_axis: np.ndarray | None = None
        self.x_coords: np.ndarray = np.zeros(self.num_spectra, dtype=np.float64)
        self.y_coords: np.ndarray = np.zeros(self.num_spectra, dtype=np.float64)
        self.x_step: float | None = None
        self.y_step: float | None = None

        if len(self.axes) == 2 and self.num_spectra == 1:
            self.y_data: np.ndarray = y_flat.copy()
        else:
            self.y_data = y_flat.reshape(self.num_spectra, n_pts)
            extra_axes = self.axes[2:]
            for eax in extra_axes:
                atype = str(eax.get("type") or "").upper()
                vals = np.asarray(eax.get("values", []), dtype=np.float64)
                if atype == "X" or (self.x_axis is None and atype not in ("Y", "Z")):
                    self.x_axis = vals
                elif atype == "Y" or self.y_axis is None:
                    self.y_axis = vals

            if self.x_axis is not None and self.y_axis is not None:
                nx = len(self.x_axis)
                ny = len(self.y_axis)
                if nx * ny == self.num_spectra:
                    # LabSpec 6 stores Axis[2]=X (inner loop) and Axis[3]=Y (outer loop)
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

        # Extract metadata tables ('acq', 'desc', 'history')
        self.tables_metadata: dict[str, Any] = {}
        self.laser_wavelength_nm: float | None = None

        tables_list = root_map.get("tables") or []
        for tbl_node in tables_list:
            if not isinstance(tbl_node, dict) or not isinstance(tbl_node.get("value"), list):
                continue
            for entry in tbl_node["value"]:
                if not isinstance(entry, dict) or not isinstance(entry.get("value"), list):
                    continue
                ef = {c["tag"]: c["value"] for c in entry["value"]}
                raw_name = ef.get("name")
                name = str(raw_name if raw_name is not None else entry["tag"]).strip()
                if not name:
                    continue
                val = _format_scalar(ef.get("value"))
                pos = _format_scalar(ef.get("pos"))

                # Capture calibrated laser wavelength stored in 'pos' of 'Laser' entry (e.g. 532.43 nm)
                if name.lower() == "laser" and isinstance(pos, (int, float)) and 100.0 <= float(pos) <= 5000.0:
                    self.laser_wavelength_nm = float(pos)

                if val is not None and str(val) != "":
                    self.tables_metadata[name] = val

