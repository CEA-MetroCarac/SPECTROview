"""File loading, dataset import, and result export."""
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from spectroview.api.exceptions import LoadError
from spectroview.model.m_io import (
    load_spectrum_file,
    load_map_file,
    load_wdf_spectrum,
    load_wdf_map,
    load_dataframe_file,
    load_spc_spectrum,
    load_spc_map,
    load_TRPL_data,
)


def load_spectra(path: Union[str, Path]) -> Dict[str, Any]:
    """Load discrete spectra from TXT, CSV, WDF, SPC, or DAT (TRPL) files.

    Returns:
        dict with an "items" key: a list of {"name", "x0", "y0", "metadata"} dicts.

    Raises:
        LoadError: unsupported extension or unreadable file.
    """
    path = Path(path)
    ext = path.suffix.lower()

    try:
        if ext in (".txt", ".csv"):
            return load_spectrum_file(path)
        elif ext == ".wdf":
            return load_wdf_spectrum(path)
        elif ext == ".spc":
            return load_spc_spectrum(path)
        elif ext == ".dat":
            return load_TRPL_data(path)
        else:
            raise LoadError(f"Unsupported discrete spectrum file type: {ext}")
    except LoadError:
        raise
    except Exception as e:
        raise LoadError(f"Failed to load spectra from {path}: {e}") from e


def load_map(path: Union[str, Path]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """Load a hyperspectral map from TXT, CSV, WDF, or SPC files.

    Returns:
        A wide-format DataFrame (columns X, Y, <wavenumbers...>), or a
        (DataFrame, metadata) tuple for WDF/SPC files that carry acquisition metadata.

    Raises:
        LoadError: unsupported extension or unreadable file.
    """
    path = Path(path)
    ext = path.suffix.lower()

    try:
        if ext in (".txt", ".csv"):
            return load_map_file(path)
        elif ext == ".wdf":
            return load_wdf_map(path)
        elif ext == ".spc":
            return load_spc_map(path)
        else:
            raise LoadError(f"Unsupported map file type: {ext}")
    except LoadError:
        raise
    except Exception as e:
        raise LoadError(f"Failed to load map from {path}: {e}") from e


def load_dataset(path: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """Load a tabular dataset (Excel or CSV) for use in the Graphs API.

    Returns:
        dict of {sheet_or_table_name: DataFrame}.

    Raises:
        LoadError: unreadable file.
    """
    try:
        return load_dataframe_file(Path(path))
    except Exception as e:
        raise LoadError(f"Failed to load dataset from {path}: {e}") from e


def export_results(results: Union[pd.DataFrame, List[dict]], output_path: Union[str, Path]) -> Path:
    """Export fit results or any tabular data to an Excel or CSV file.

    Args:
        results: a DataFrame, or a list of dicts (one per row).
        output_path: destination path, extension must be .xlsx/.xls or .csv.

    Returns:
        The output path.

    Raises:
        LoadError: unsupported output extension.
    """
    path = Path(output_path)
    df = pd.DataFrame(results) if isinstance(results, list) else results

    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        df.to_excel(path, index=False)
    elif ext == ".csv":
        df.to_csv(path, index=False, sep=";")
    else:
        raise LoadError(f"Output must be .xlsx or .csv, got: {ext}")
    return path


def load_spectra_to_matrix(paths: Union[str, Path, List[Union[str, Path]]]) -> Dict[str, Any]:
    """Load one or more spectrum files and return as a ready-to-use x/Y matrix.

    Calls `load_spectra()` for each file and stacks all items into a single
    intensity matrix, interpolating onto a common wavenumber axis (the first
    spectrum's axis) when necessary.

    Args:
        paths: a single file path or a list of file paths.

    Returns:
        dict with keys:
            x         — float64[M] shared wavenumber axis.
            Y         — float64[N, M] intensity matrix (one row per spectrum).
            names     — list[str] of spectrum names (length N).
            metadata  — list[dict] of per-spectrum acquisition metadata.

    Raises:
        LoadError: no spectra could be loaded from the given paths.

    Example::

        data = io.load_spectra_to_matrix(["sample_a.wdf", "sample_b.wdf"])
        x = data["x"]   # shape (M,)
        Y = data["Y"]   # shape (N, M)
    """
    from scipy.interpolate import interp1d

    if isinstance(paths, (str, Path)):
        paths = [paths]

    all_x, all_y, all_names, all_meta = [], [], [], []
    for p in paths:
        raw = load_spectra(p)
        for item in raw.get("items", []):
            all_x.append(item["x0"])
            all_y.append(item["y0"])
            all_names.append(item["name"])
            all_meta.append(item.get("metadata", {}))

    if not all_x:
        raise LoadError("No spectra were loaded from the provided paths.")

    x_ref = all_x[0]
    rows = []
    for xi, yi in zip(all_x, all_y):
        if len(xi) == len(x_ref) and np.allclose(xi, x_ref):
            rows.append(yi.astype(np.float64))
        else:
            f = interp1d(xi, yi, kind="linear", bounds_error=False, fill_value=0.0)
            rows.append(f(x_ref))

    return {
        "x": x_ref.astype(np.float64),
        "Y": np.stack(rows, axis=0),
        "names": all_names,
        "metadata": all_meta,
    }


def convert_renishaw_map(input_path: Union[str, Path], output_path: Union[str, Path]) -> Path:
    """Convert a Renishaw InVia long-format TXT export into the wide-matrix
    TXT format SPECTROview/LabSpec6 expect.

    The input must have tab-separated columns #X, #Y, #Wave, #Intensity
    (the raw export shape from Renishaw's WiRE software). This is a headless
    equivalent of the GUI's "Hyperspectral Data Converter" tool.

    Args:
        input_path: path to the raw long-format TXT export.
        output_path: destination path for the converted wide-matrix TXT.

    Returns:
        output_path, as a Path.

    Raises:
        LoadError: input_path is missing the required columns or is unreadable.

    Example::

        io.convert_renishaw_map("raw_export.txt", "converted_map.txt")
        df = io.load_map("converted_map.txt")
    """
    from spectroview.model.m_file_converter import convert_action

    output_path = Path(output_path)
    try:
        convert_action(str(input_path), str(output_path))
    except Exception as e:
        raise LoadError(f"Renishaw conversion failed for {input_path}: {e}") from e
    return output_path
