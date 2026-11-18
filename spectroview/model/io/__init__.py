"""Model I/O subpackage: file detection, binary/text readers, and workspace persistence."""

from spectroview.model.io.m_io import (
    WDF_AVAILABLE,
    WDFReader,
    _METADATA_ORDERED_KEYS,
    _map_spc_metadata,
    convert_action,
    detect_file_type,
    extract_map_arrays,
    load_TRPL_data,
    load_dataframe_file,
    load_l6m_map,
    load_l6s_spectrum,
    load_map_file,
    load_ngc_map,
    load_ngs_spectrum,
    load_spc_map,
    load_spc_spectrum,
    load_spectrum_file,
    load_wdf_map,
    load_wdf_spectrum,
)
from spectroview.model.io.m_l6s import L6sReader
from spectroview.model.io.m_ngs import NgsReader
from spectroview.model.io.m_spc import SpcReader
from spectroview.model.io.m_wdf import WdfReader
from spectroview.model.io.workspace_io import WorkspaceIO

__all__ = [
    "L6sReader",
    "NgsReader",
    "SpcReader",
    "WDFReader",
    "WDF_AVAILABLE",
    "WdfReader",
    "WorkspaceIO",
    "_METADATA_ORDERED_KEYS",
    "_map_spc_metadata",
    "convert_action",
    "detect_file_type",
    "extract_map_arrays",
    "load_TRPL_data",
    "load_dataframe_file",
    "load_l6m_map",
    "load_l6s_spectrum",
    "load_map_file",
    "load_ngc_map",
    "load_ngs_spectrum",
    "load_spc_map",
    "load_spc_spectrum",
    "load_spectrum_file",
    "load_wdf_map",
    "load_wdf_spectrum",
]
