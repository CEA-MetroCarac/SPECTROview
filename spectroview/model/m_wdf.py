"""Native binary reader for Renishaw WiRE (.wdf) spectroscopic files.

Eliminates third-party ``renishawWiRE`` / ``PIL`` runtime overhead by reading
``WDF1``, ``DATA``, ``XLST``, ``YLST``, ``ORGN``, and ``WMAP`` blocks directly
into NumPy arrays via single-pass ``struct.unpack_from`` and ``np.frombuffer``
operations.
"""

from __future__ import annotations

from enum import IntEnum
import math
from pathlib import Path
import struct

import numpy as np


class WdfMeasurementType(IntEnum):
    """Renishaw WDF measurement type codes (offset 0x84 in WDF1)."""

    Unspecified = 0
    Single = 1
    Series = 2
    Mapping = 3

    @classmethod
    def _missing_(cls, value):
        return cls.Unspecified

    def __str__(self) -> str:
        return self._name_


class WdfScanType(IntEnum):
    """Renishaw WDF scan type codes (offset 0x80 in WDF1)."""

    Unspecified = 0
    Static = 1
    Continuous = 2
    StepRepeat = 3
    FilterScan = 4
    FilterImage = 5
    StreamLine = 6
    StreamLineHR = 7
    PointDetector = 8

    @classmethod
    def _missing_(cls, value):
        return cls.Unspecified

    def __str__(self) -> str:
        return self._name_


_UNIT_STR_MAP: dict[str, str] = {
    "Arbitrary": "",
    "RamanShift": "1/cm",
    "Wavenumber": "nm",
    "Nanometre": "nm",
    "ElectronVolt": "eV",
    "Micron": "um",
    "Counts": "counts",
    "Electrons": "electrons",
    "Millimetres": "mm",
    "Metres": "m",
    "Kelvin": "K",
    "Pascal": "Pa",
    "Seconds": "s",
    "Milliseconds": "ms",
    "Hours": "h",
    "Days": "d",
    "Pixels": "px",
    "Intensity": "",
    "RelativeIntensity": "",
    "Degrees": "°",
    "Radians": "rad",
    "Celsius": "°C",
    "Fahrenheit": "°F",
    "KelvinPerMinute": "K/min",
    "FileTime": "s",
    "Microseconds": "us",
}


class WdfUnitType(IntEnum):
    """Renishaw WDF physical unit codes."""

    Arbitrary = 0
    RamanShift = 1
    Wavenumber = 2
    Nanometre = 3
    ElectronVolt = 4
    Micron = 5
    Counts = 6
    Electrons = 7
    Millimetres = 8
    Metres = 9
    Kelvin = 10
    Pascal = 11
    Seconds = 12
    Milliseconds = 13
    Hours = 14
    Days = 15
    Pixels = 16
    Intensity = 17
    RelativeIntensity = 18
    Degrees = 19
    Radians = 20
    Celsius = 21
    Fahrenheit = 22
    KelvinPerMinute = 23
    FileTime = 24
    Microseconds = 25

    @classmethod
    def _missing_(cls, value):
        return cls.Arbitrary

    def __str__(self) -> str:
        return _UNIT_STR_MAP.get(self._name_, "")


class WdfDataType(IntEnum):
    """Renishaw WDF axis and origin list data type codes."""

    Arbitrary = 0
    Frequency = 1
    Intensity = 2
    Spatial_X = 3
    Spatial_Y = 4
    Spatial_Z = 5
    Spatial_R = 6
    Spatial_Theta = 7
    Spatial_Phi = 8
    Temperature = 9
    Pressure = 10
    Time = 11
    Derived = 12
    Polarization = 13
    FocusTrack = 14
    RampRate = 15
    Checksum = 16
    Flags = 17
    ElapsedTime = 18
    Spectral = 19
    Mp_Well_Spatial_X = 22
    Mp_Well_Spatial_Y = 23
    Mp_LocationIndex = 24
    Mp_WellReference = 25
    EndMarker = 26
    ExposureTime = 27

    @classmethod
    def _missing_(cls, value):
        return cls.Arbitrary

    def __str__(self) -> str:
        return self._name_


def _convert_wavenumber_to_nm(wn: float) -> float:
    """Convert laser wavenumber (cm^-1) to wavelength in nm."""
    if wn == 0.0:
        return math.nan
    return 1.0 / (wn * 1e2) / 1e-9


class WdfReader:
    """Native binary reader for Renishaw WiRE ``.wdf`` files.

    Drop-in replacement for ``renishawWiRE.WDFReader`` with:
    - Fast 512-byte header-only inspection (``header_only=True`` / ``peek_header``).
    - Vectorized ``ORGN`` coordinate/timestamp extraction via ``np.frombuffer``
      (eliminating per-point Python ``struct.unpack`` loops).
    - Single-read ``WDF1``, ``XLST``, ``YLST``, ``DATA``, and ``WMAP`` block parsing.
    - Zero ``PIL`` / ``WHTL`` JPEG decoding overhead.
    """

    def __init__(self, file_name: str | Path, header_only: bool = False):
        self.filepath = str(file_name)
        self.header_only = header_only
        try:
            self.file_obj = open(self.filepath, "rb")
        except OSError as exc:
            raise OSError(f"File {file_name} does not exist!") from exc

        self.title: str = ""
        self.username: str = ""
        self.measurement_type: WdfMeasurementType = WdfMeasurementType.Unspecified
        self.scan_type: WdfScanType = WdfScanType.Unspecified
        self.laser_length: float = math.nan
        self.count: int = 0
        self.spectral_unit: WdfUnitType = WdfUnitType.Arbitrary
        self.xlist_type: WdfDataType = WdfDataType.Arbitrary
        self.xlist_unit: WdfUnitType = WdfUnitType.Arbitrary
        self.ylist_type: WdfDataType = WdfDataType.Arbitrary
        self.ylist_unit: WdfUnitType = WdfUnitType.Arbitrary
        self.point_per_spectrum: int = 0
        self.data_origin_count: int = 0
        self.capacity: int = 0
        self.application_name: str = ""
        self.application_version: list[int] = [0, 0, 0, 0]
        self.xlist_length: int = 0
        self.ylist_length: int = 0
        self.accumulation_count: int = 0
        self.block_info: dict[str, tuple[int, int, int]] = {}
        self.is_completed: bool = False
        self.xdata: np.ndarray = np.empty(0, dtype=np.float32)
        self.ydata: np.ndarray = np.empty(0, dtype=np.float32)
        self.spectra: np.ndarray = np.empty(0, dtype=np.float32)
        self.xpos: np.ndarray = np.empty(0, dtype=np.float64)
        self.ypos: np.ndarray = np.empty(0, dtype=np.float64)
        self.zpos: np.ndarray = np.empty(0, dtype=np.float64)
        self.xpos_unit: WdfUnitType = WdfUnitType.Arbitrary
        self.ypos_unit: WdfUnitType = WdfUnitType.Arbitrary
        self.zpos_unit: WdfUnitType = WdfUnitType.Arbitrary
        self.origin_list_header: list[list] = []
        self._wdf1_bytes: bytes = b""

        try:
            self._parse_header()
            if header_only:
                self.close()
                return

            self._locate_all_blocks()
            if "DATA" in self.block_info:
                self._parse_spectra()
            if "XLST" in self.block_info:
                self._parse_xylist("X")
            if "YLST" in self.block_info:
                self._parse_xylist("Y")
            if "ORGN" in self.block_info:
                self._parse_origin_list()
            if "WMAP" in self.block_info:
                self._parse_wmap()

            self._reshape_spectra()
        except Exception:
            self.close()
            raise

    @classmethod
    def peek_header(cls, filepath: str | Path) -> dict:
        """Read and validate only the 512-byte ``WDF1`` header without loading blocks."""
        reader = cls(filepath, header_only=True)
        return {
            "title": reader.title,
            "username": reader.username,
            "measurement_type": reader.measurement_type,
            "scan_type": reader.scan_type,
            "laser_length": reader.laser_length,
            "count": reader.count,
            "capacity": reader.capacity,
            "point_per_spectrum": reader.point_per_spectrum,
            "accumulation_count": reader.accumulation_count,
            "xlist_length": reader.xlist_length,
            "ylist_length": reader.ylist_length,
            "data_origin_count": reader.data_origin_count,
            "application_name": reader.application_name,
            "application_version": reader.application_version,
            "spectral_unit": reader.spectral_unit,
        }

    def close(self) -> None:
        """Close the underlying binary file handle if open."""
        if getattr(self, "file_obj", None) is not None and not self.file_obj.closed:
            self.file_obj.close()

    def __enter__(self) -> WdfReader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _parse_header(self) -> None:
        """Read and unpack the 512-byte ``WDF1`` file header block in a single read."""
        self.file_obj.seek(0)
        hdr = self.file_obj.read(512)
        if len(hdr) < 512:
            raise ValueError(f"Invalid or truncated WDF file (< 512 bytes): {self.filepath}")

        block_id = hdr[:4]
        block_uid, block_len = struct.unpack_from("<IQ", hdr, 4)
        if block_id != b"WDF1" or block_uid not in (0, 1) or block_len != 512:
            raise ValueError("The wdf file format is incorrect!")

        self._wdf1_bytes = hdr

        # 0x3C..0x60: point_per_spectrum(I), capacity(Q), count(Q), accumulation_count(I),
        #             ylist_length(I), xlist_length(I), data_origin_count(I)
        (
            self.point_per_spectrum,
            self.capacity,
            self.count,
            self.accumulation_count,
            self.ylist_length,
            self.xlist_length,
            self.data_origin_count,
        ) = struct.unpack_from("<IQQIIII", hdr, 0x3C)

        if self.count == 0 and self.capacity > 0:
            self.count = self.capacity
        self.is_completed = self.count == self.capacity

        # 0x60..0x78: application_name (24 bytes UTF-8)
        self.application_name = hdr[0x60:0x78].decode("utf-8", errors="ignore").replace("\x00", "")

        # 0x78..0x80: application_version (4 x uint16)
        self.application_version = list(struct.unpack_from("<4H", hdr, 0x78))

        # 0x80..0x88: scan_type(I), measurement_type(I)
        scan_code, meas_code = struct.unpack_from("<II", hdr, 0x80)
        self.scan_type = WdfScanType(scan_code)
        self.measurement_type = WdfMeasurementType(meas_code)

        # 0x98..0xA0: spectral_unit(I), laser_wavenumber(f)
        spec_unit_code, laser_wn = struct.unpack_from("<If", hdr, 0x98)
        self.spectral_unit = WdfUnitType(spec_unit_code)
        self.laser_length = _convert_wavenumber_to_nm(laser_wn)

        # 0xD0..0xF0: username (32 bytes UTF-8)
        self.username = hdr[0xD0:0xF0].decode("utf-8", errors="ignore").replace("\x00", "")

        # 0xF0..0x200: title (272 bytes UTF-8)
        self.title = hdr[0xF0:0x200].decode("utf-8", errors="ignore").replace("\x00", "")

    def _locate_all_blocks(self) -> None:
        """Index all binary blocks ``(uid, offset, size)`` by jumping across block headers."""
        self.file_obj.seek(0, 2)
        file_size = self.file_obj.tell()
        curpos = 0
        while curpos + 16 <= file_size:
            self.file_obj.seek(curpos)
            bhdr = self.file_obj.read(16)
            if len(bhdr) < 16:
                break
            try:
                block_name = bhdr[:4].decode("ascii")
            except UnicodeDecodeError:
                break
            block_uid, block_size = struct.unpack_from("<IQ", bhdr, 4)
            if block_size < 16 or curpos + block_size > file_size:
                break
            self.block_info[block_name] = (block_uid, curpos, block_size)
            curpos += block_size

    def _parse_xylist(self, direction: str) -> None:
        """Read ``XLST`` or ``YLST`` axis block in a single read."""
        d_upper = direction.upper()
        d_lower = direction.lower()
        name = f"{d_upper}LST"
        _uid, pos, blk_size = self.block_info[name]
        length = getattr(self, f"{d_lower}list_length")
        if length == 0:
            raise ValueError(f"{d_upper}-List possibly not initialized!")

        n_bytes = 8 + length * 4
        if blk_size < 16 + n_bytes:
            raise ValueError(f"Truncated {name} block in {self.filepath}")

        self.file_obj.seek(pos + 0x10)
        buf = self.file_obj.read(n_bytes)
        if len(buf) < n_bytes:
            raise ValueError(f"Truncated {name} data in {self.filepath}")

        ltype_code, lunit_code = struct.unpack_from("<II", buf, 0)
        setattr(self, f"{d_lower}list_type", WdfDataType(ltype_code))
        setattr(self, f"{d_lower}list_unit", WdfUnitType(lunit_code))
        arr = np.frombuffer(buf, dtype="<f4", count=length, offset=8).copy()
        setattr(self, f"{d_lower}data", arr)

    def _parse_spectra(self) -> None:
        """Read the ``DATA`` float32 spectral payload in a single bulk read."""
        if self.count <= 0 or self.point_per_spectrum <= 0:
            raise ValueError(f"Invalid WDF spectrum dimensions (count={self.count}, pps={self.point_per_spectrum})")

        _uid, pos, blk_size = self.block_info["DATA"]
        n_floats = self.count * self.point_per_spectrum
        n_bytes = n_floats * 4
        if blk_size < 0x10 + n_bytes:
            raise ValueError(f"Truncated WDF DATA block in {self.filepath}")

        self.file_obj.seek(pos + 0x10)
        raw = self.file_obj.read(n_bytes)
        if len(raw) < n_bytes:
            raise ValueError(f"Truncated WDF spectral payload in {self.filepath}")

        self.spectra = np.frombuffer(raw, dtype="<f4", count=n_floats).copy()

    def _parse_origin_list(self) -> None:
        """Vectorized parser for the ``ORGN`` coordinate and timestamp lists."""
        _uid, pos, _size = self.block_info["ORGN"]
        self.origin_list_header = [[None] * 5 for _ in range(self.data_origin_count)]
        self.xpos = np.zeros(self.count, dtype=np.float64)
        self.ypos = np.zeros(self.count, dtype=np.float64)
        self.zpos = np.zeros(self.count, dtype=np.float64)

        list_increment = 0x18 + 8 * self.capacity
        curpos = pos + 0x14
        row_read_bytes = 0x18 + 8 * self.count

        for i in range(self.data_origin_count):
            self.file_obj.seek(curpos)
            row_buf = self.file_obj.read(row_read_bytes)
            if len(row_buf) < row_read_bytes:
                break

            p1, p2 = struct.unpack_from("<II", row_buf, 0)
            annotation = row_buf[8:0x18].decode("utf-8", errors="ignore").replace("\x00", "")
            is_primary = bool((p1 >> 31) & 0b1)
            dtype = WdfDataType(p1 & ~(0b1 << 31))
            utype = WdfUnitType(p2)

            if dtype == WdfDataType.Time:
                raw_i64 = np.frombuffer(row_buf, dtype="<i8", count=self.count, offset=0x18)
                arr = raw_i64.astype(np.float64) / 1e7
                if len(arr) > 0:
                    arr = arr - arr[0]
            else:
                arr = np.frombuffer(row_buf, dtype="<f8", count=self.count, offset=0x18).copy()

            self.origin_list_header[i][0] = is_primary
            self.origin_list_header[i][1] = dtype
            self.origin_list_header[i][2] = utype
            self.origin_list_header[i][3] = annotation
            self.origin_list_header[i][4] = arr

            if dtype == WdfDataType.Spatial_X:
                self.xpos = arr
                self.xpos_unit = utype
            elif dtype == WdfDataType.Spatial_Y:
                self.ypos = arr
                self.ypos_unit = utype
            elif dtype == WdfDataType.Spatial_Z:
                self.zpos = arr
                self.zpos_unit = utype

            curpos += list_increment

    def _parse_wmap(self) -> None:
        """Read ``WMAP`` mapping grid dimensions and origin offsets."""
        _uid, pos, blk_size = self.block_info["WMAP"]
        if blk_size < 0x18 + 32:
            return

        self.file_obj.seek(pos + 0x18)
        buf = self.file_obj.read(32)
        if len(buf) < 32:
            return

        x_start, y_start, _u1, x_pad, y_pad, _u2, spectra_w, spectra_h = struct.unpack_from("<6fII", buf, 0)
        if len(self.xpos) > 0 and not np.isclose(x_start, self.xpos[0], rtol=1e-4):
            raise ValueError("WMAP Xpos is not same as in ORGN!")
        if len(self.ypos) > 0 and not np.isclose(y_start, self.ypos[0], rtol=1e-4):
            raise ValueError("WMAP Ypos is not same as in ORGN!")

        self.map_shape = (spectra_w, spectra_h)
        self.map_info = {
            "x_start": x_start,
            "y_start": y_start,
            "x_pad": x_pad,
            "y_pad": y_pad,
            "x_span": spectra_w * x_pad,
            "y_span": spectra_h * y_pad,
            "x_unit": self.xpos_unit,
            "y_unit": self.ypos_unit,
        }

    def _reshape_spectra(self) -> None:
        """Reshape ``self.spectra`` to match ``renishawWiRE`` conventions."""
        if not self.is_completed:
            try:
                self.spectra = np.reshape(self.spectra, (self.count, self.point_per_spectrum))
            except ValueError:
                pass
            return

        if hasattr(self, "map_shape"):
            spectra_w, spectra_h = self.map_shape
            if spectra_w * spectra_h != self.count:
                return
            if spectra_w * spectra_h * self.point_per_spectrum != len(self.spectra):
                return
            if spectra_h > 1 and spectra_w > 1:
                self.spectra = np.reshape(self.spectra, (spectra_h, spectra_w, self.point_per_spectrum))
            else:
                self.spectra = np.reshape(self.spectra, (self.count, self.point_per_spectrum))
        elif self.count > 1:
            self.spectra = np.reshape(self.spectra, (self.count, self.point_per_spectrum))

