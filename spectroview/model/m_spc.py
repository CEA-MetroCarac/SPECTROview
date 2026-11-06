"""Binary reader for Galactic / HORIBA LabSpec SPC spectroscopic files."""

from __future__ import annotations

from pathlib import Path
import re
import struct

import numpy as np


_LOG_PAIR_RE = re.compile(r"([A-Za-z][A-Za-z0-9\s\.\(\)_\-\%\/\:]*?)\s*=\s*(.*)")

# 32-byte SPC subheader layout (little-endian):
# 0x00 subflgs (B), 0x01 subexp (b), 0x02 subindx (h),
# 0x04 subtime (f4), 0x08 subnext (f4), 0x0C subnois (f4),
# 0x10 subnpts (i4), 0x14 subscan (i4), 0x18 subwlevel (f4), 0x1C subresv (4B)
_SUBHDR_DTYPE = np.dtype(
    [
        ("flags", "<u1"),
        ("exp", "<i1"),
        ("index", "<i2"),
        ("time", "<f4"),
        ("next", "<f4"),
        ("noise", "<f4"),
        ("npts", "<i4"),
        ("scans", "<i4"),
        ("wlevel", "<f4"),
        ("_resv", "V4"),
    ]
)


def _decode_spc_date(date_int: int) -> str:
    """Decode packed 32-bit SPC date integer into YYYY-MM-DD HH:MM if valid."""
    if date_int <= 0:
        return ""
    year = (date_int >> 20) & 0xFFF
    month = (date_int >> 16) & 0xF
    day = (date_int >> 11) & 0x1F
    hour = (date_int >> 6) & 0x1F
    minute = date_int & 0x3F
    if 1970 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31 and hour <= 23 and minute <= 59:
        return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"
    return ""


class SpcReader:
    """Reader for Galactic SPC files (New Format 0x4B, 1996+).

    Supports fast 512-byte header-only inspection (``header_only=True``) for
    instant file categorization as well as vectorized bulk reading of uniform
    multi-subfile hyperspectral maps.
    """

    def __init__(self, filepath: str | Path, header_only: bool = False):
        self.filepath = str(filepath)
        self.header_only = header_only
        self.f = None
        self.header: dict = {}
        self.subheaders: list[dict] = []
        self.sub_time: np.ndarray | None = None
        self.sub_wlevel: np.ndarray | None = None
        self.x_data: np.ndarray | None = None
        self.y_data: np.ndarray | None = None
        self.log_content: str = ""
        self.log_metadata: dict[str, str] = {}

        self.read(header_only=header_only)

    @classmethod
    def peek_header(cls, filepath: str | Path) -> dict:
        """Read and validate only the 512-byte main header without loading spectra."""
        return cls(filepath, header_only=True).header

    def read(self, header_only: bool = False) -> None:
        with open(self.filepath, "rb") as f:
            self.f = f
            file_size = f.seek(0, 2)
            if file_size < 512:
                raise ValueError(f"Invalid or truncated SPC file (< 512 bytes): {self.filepath}")

            self._read_header()

            npts = self.header["npts"]
            fnsub = self.header["fnsub"]
            if fnsub <= 0:
                raise ValueError(f"Invalid SPC subfile count (fnsub={fnsub}): {self.filepath}")
            if npts <= 0 and (self.header["ftflgs"] & 0x40) == 0:
                raise ValueError(f"Invalid SPC point count (npts={npts}): {self.filepath}")

            if header_only:
                self.f = None
                return

            ftflgs = self.header["ftflgs"]
            has_txvals = (ftflgs & 0x80) != 0   # 0x80 = TXVALS (shared explicit X array)
            has_txyxys = (ftflgs & 0x40) != 0   # 0x40 = TXYXYS (per-subfile X / variable npts)
            is_16bit = (ftflgs & 0x01) != 0     # 0x01 = TSPREC (16-bit integer Y values vs 32-bit)

            f.seek(512)

            # 1. Read shared X-axis (unless TXYXYS per-subfile X arrays are used)
            if has_txvals and not has_txyxys:
                expected_x_bytes = npts * 4
                x_bytes = f.read(expected_x_bytes)
                if len(x_bytes) < expected_x_bytes:
                    raise ValueError(f"Truncated SPC X-axis data in {self.filepath}")
                self.x_data = np.frombuffer(x_bytes, dtype="<f4").astype(np.float64)
            elif not has_txyxys:
                self.x_data = np.linspace(self.header["first"], self.header["last"], npts, dtype=np.float64)

            # 2. Read Subfiles (fast vectorized path for uniform subfiles, fallback for variable/mixed)
            subfiles_pos = f.tell()
            if not has_txyxys and self._try_read_uniform_subfiles(f, subfiles_pos, file_size, fnsub, npts, is_16bit):
                data_end_pos = f.tell()
            else:
                f.seek(subfiles_pos)
                self._read_subfiles_sequential(f, fnsub, npts, is_16bit, has_txvals, has_txyxys)
                data_end_pos = f.tell()

            # 3. Read Log Block (optional)
            flogoff = self.header["flogoff"]
            log_offset = flogoff if (0 < flogoff < file_size) else data_end_pos
            if 0 < log_offset < file_size:
                self._read_log_block(f, log_offset, file_size)

            self.f = None

    def _try_read_uniform_subfiles(
        self,
        f,
        start_pos: int,
        file_size: int,
        fnsub: int,
        npts: int,
        is_16bit: bool,
    ) -> bool:
        """Vectorized single-read parser for standard uniform SPC subfiles."""
        # Peek first subheader exponent: exp == -128 (0x80) denotes IEEE float32;
        # otherwise values are fixed-point int16 (TSPREC=0x01) or int32 scaled by 2^(exp - bits)
        first_sh = f.read(32)
        if len(first_sh) < 32:
            raise ValueError(f"Truncated SPC subheader in {self.filepath}")
        raw_first_exp = struct.unpack_from("<b", first_sh, 1)[0]
        hdr_exp = self.header["fexp"]
        first_exp = hdr_exp if (raw_first_exp == 0 and hdr_exp != 0) else raw_first_exp
        use_float32 = first_exp == -128
        if use_float32:
            val_dtype = "<f4"
            bytes_per_pt = 4
        elif is_16bit:
            val_dtype = "<i2"
            bytes_per_pt = 2
        else:
            val_dtype = "<i4"
            bytes_per_pt = 4

        record_size = 32 + npts * bytes_per_pt
        total_bytes = fnsub * record_size
        if start_pos + total_bytes > file_size:
            return False

        f.seek(start_pos)
        raw_buf = f.read(total_bytes)
        if len(raw_buf) < total_bytes:
            return False

        rec_dtype = np.dtype([("sh", _SUBHDR_DTYPE), ("y", val_dtype, (npts,))])
        records = np.frombuffer(raw_buf, dtype=rec_dtype, count=fnsub)
        sh_arr = records["sh"]

        # Verify no per-subfile npts overrides and no mixed float32/integer exponent overrides
        sub_npts = sh_arr["npts"]
        if np.any((sub_npts != 0) & (sub_npts != npts)):
            return False
        raw_exps = sh_arr["exp"].astype(np.int32)
        eff_exps = np.where((raw_exps == 0) & (hdr_exp != 0), hdr_exp, raw_exps)
        if use_float32 and not np.all(eff_exps == -128):
            return False
        if (not use_float32) and np.any(eff_exps == -128):
            return False

        # Extract Y matrix in one vectorized operation
        if use_float32:
            y_mat = records["y"].astype(np.float64)
        else:
            bit_width = 16 if is_16bit else 32
            scales = np.ldexp(1.0, eff_exps - bit_width)[:, np.newaxis]
            y_mat = records["y"].astype(np.float64) * scales

        self.y_data = y_mat[0] if fnsub == 1 else y_mat
        self.sub_time = sh_arr["time"].astype(np.float64)
        self.sub_wlevel = sh_arr["wlevel"].astype(np.float64)

        # Populate lightweight subheader dicts for full API compatibility
        flags_l = sh_arr["flags"].tolist()
        exp_l = sh_arr["exp"].tolist()
        idx_l = sh_arr["index"].tolist()
        time_l = sh_arr["time"].tolist()
        next_l = sh_arr["next"].tolist()
        noise_l = sh_arr["noise"].tolist()
        npts_l = sub_npts.tolist()
        scans_l = sh_arr["scans"].tolist()
        wlevel_l = sh_arr["wlevel"].tolist()
        self.subheaders = [
            {
                "flags": flags_l[i],
                "exp": exp_l[i],
                "index": idx_l[i],
                "time": time_l[i],
                "next": next_l[i],
                "noise": noise_l[i],
                "npts": npts_l[i],
                "scans": scans_l[i],
                "wlevel": wlevel_l[i],
            }
            for i in range(fnsub)
        ]
        return True

    def _read_subfiles_sequential(
        self,
        f,
        fnsub: int,
        header_npts: int,
        is_16bit: bool,
        has_txvals: bool,
        has_txyxys: bool,
    ) -> None:
        """Fallback sequential subfile reader for variable-length or per-subfile-X SPC files."""
        y_list = []
        self.subheaders = []

        for i in range(fnsub):
            subheader = self._read_subheader(f)
            if not subheader:
                raise ValueError(f"Truncated SPC subfile {i} in {self.filepath}")
            self.subheaders.append(subheader)

            npts = subheader["npts"] if (has_txyxys and subheader["npts"] > 0) else header_npts
            if npts <= 0:
                raise ValueError(f"Invalid subfile point count ({npts}) in {self.filepath}")

            if has_txyxys and has_txvals:
                x_bytes = f.read(npts * 4)
                if len(x_bytes) < npts * 4:
                    raise ValueError(f"Truncated per-subfile X array in {self.filepath}")
                if self.x_data is None:
                    self.x_data = np.frombuffer(x_bytes, dtype="<f4").astype(np.float64)

            raw_exp = subheader["exp"]
            hdr_exp = self.header["fexp"]
            exponent = hdr_exp if (raw_exp == 0 and hdr_exp != 0) else raw_exp
            if exponent == -128:
                raw_data = f.read(npts * 4)
                if len(raw_data) < npts * 4:
                    raise ValueError(f"Truncated SPC float32 Y data in {self.filepath}")
                y_chunk = np.frombuffer(raw_data, dtype="<f4").astype(np.float64)
            elif is_16bit:
                raw_data = f.read(npts * 2)
                if len(raw_data) < npts * 2:
                    raise ValueError(f"Truncated SPC int16 Y data in {self.filepath}")
                y_chunk = np.frombuffer(raw_data, dtype="<i2").astype(np.float64)
                y_chunk = y_chunk * np.ldexp(1.0, int(exponent) - 16)
            else:
                raw_data = f.read(npts * 4)
                if len(raw_data) < npts * 4:
                    raise ValueError(f"Truncated SPC int32 Y data in {self.filepath}")
                y_chunk = np.frombuffer(raw_data, dtype="<i4").astype(np.float64)
                y_chunk = y_chunk * np.ldexp(1.0, int(exponent) - 32)

            y_list.append(y_chunk)

        if self.x_data is None:
            n0 = len(y_list[0])
            self.x_data = np.linspace(self.header["first"], self.header["last"], n0, dtype=np.float64)

        self.sub_time = np.array([sh.get("time", 0.0) for sh in self.subheaders], dtype=np.float64)
        self.sub_wlevel = np.array([sh.get("wlevel", 0.0) for sh in self.subheaders], dtype=np.float64)
        self.y_data = np.vstack(y_list) if fnsub > 1 else y_list[0]

    def _read_log_block(self, f, log_offset: int, file_size: int) -> None:
        """Read SPC logstc header (64 bytes) and parse ASCII Key=Value metadata."""
        try:
            f.seek(log_offset)
            remaining = file_size - log_offset
            if remaining <= 0:
                return

            text_offset = log_offset
            text_len = remaining
            if remaining >= 64:
                log_hdr = f.read(64)
                logsizd, _logsizm, logtxto = struct.unpack_from("<III", log_hdr, 0)
                if 64 <= logtxto < remaining:
                    text_offset = log_offset + logtxto
                    if logtxto < logsizd <= remaining:
                        text_len = logsizd - logtxto
                    else:
                        text_len = remaining - logtxto

            f.seek(text_offset)
            raw_text = f.read(text_len)
            self.log_content = raw_text.decode("utf-8", errors="ignore").replace("\x00", "")
            self._parse_log_metadata()
        except Exception:
            pass

    def _parse_log_metadata(self) -> None:
        """Parse Key=Value pairs from the SPC ASCII log block."""
        self.log_metadata = {}
        for line in self.log_content.splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue
            match = _LOG_PAIR_RE.search(line)
            if match:
                key = match.group(1).strip()
                val = match.group(2).strip()
                if 1 < len(key) < 64:
                    self.log_metadata[key] = val

    def _read_header(self) -> None:
        """Read the 512-byte Galactic SPC header (New Format 0x4B)."""
        self.f.seek(0)
        hdr = self.f.read(512)

        # 0x00..0x1B: ftflgs(B), fversn(B), fexper(B), fexp(b), fnpts(i), ffirst(d), flast(d), fnsub(i)
        ftflgs, fversn, fexper, fexp, npts, first, last, fnsub = struct.unpack_from("<BBBbiddi", hdr, 0)

        # Validate SPC version byte (0x4B = New LSB, 0x4C = New MSB, 0x4D = Old format)
        if fversn not in (0x4B, 0x4C, 0x4D):
            raise ValueError(f"Unsupported or invalid SPC version byte: 0x{fversn:02X}")

        # 0x1C..0x1F (28..32): fxtype, fytype, fztype, fpost
        fxtype, fytype, fztype, fpost = struct.unpack_from("<BBBB", hdr, 28)

        # 0x20..0x23 (32..36): fdate (packed uint32)
        date_uint = struct.unpack_from("<I", hdr, 32)[0]

        # 0x58..0xDA (88..218): fcmnt (130-byte null-terminated comment)
        fcmnt = hdr[88:218].split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()

        # 0xDA..0xF8 (218..248): fcatxt (30 bytes: null-separated X, Y, Z axis strings)
        fcatxt = hdr[218:248].split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()

        # 0xF8..0xFC (248..252): flogoff (uint32 offset to logstc block)
        flogoff = struct.unpack_from("<I", hdr, 248)[0]

        self.header = {
            "ftflgs": ftflgs,
            "fversn": fversn,
            "fexper": fexper,
            "fexp": fexp,
            "npts": npts,
            "first": first,
            "last": last,
            "fnsub": fnsub,
            "fxtype": fxtype,
            "fytype": fytype,
            "fztype": fztype,
            "fpost": fpost,
            "flogoff": flogoff,
            "fcmnt": fcmnt,
            "fcatxt": fcatxt,
            "date": _decode_spc_date(date_uint),
        }

    def _read_subheader(self, f=None) -> dict:
        """Read a single 32-byte SPC subheader."""
        fh = f if f is not None else self.f
        chunk = fh.read(32)
        if len(chunk) < 32:
            return {}
        vals = struct.unpack_from("<Bbhfffiff", chunk, 0)
        return {
            "flags": vals[0],
            "exp": vals[1],
            "index": vals[2],
            "time": vals[3],
            "next": vals[4],
            "noise": vals[5],
            "npts": vals[6],
            "scans": vals[7],
            "wlevel": vals[8],
        }
