"""Data loading and saving."""
from pathlib import Path
from typing import Dict, Union, Any, List, Tuple
import pandas as pd

from spectroview.model.m_io import (
    load_spectrum_file,
    load_map_file,
    load_wdf_spectrum,
    load_wdf_map,
    load_dataframe_file,
    load_spc_spectrum,
    load_spc_map,
    load_TRPL_data
)

def load_spectra(path: Union[str, Path]) -> Dict[str, Any]:
    """Load discrete spectra from TXT, CSV, WDF, SPC, or DAT (TRPL) files.
    Returns a dictionary containing the loaded spectra data and metadata.
    """
    path = Path(path)
    ext = path.suffix.lower()
    
    if ext in [".txt", ".csv"]:
        return load_spectrum_file(path)
    elif ext == ".wdf":
        return load_wdf_spectrum(path)
    elif ext == ".spc":
        return load_spc_spectrum(path)
    elif ext == ".dat":
        return load_TRPL_data(path)
    else:
        raise ValueError(f"Unsupported discrete spectrum file type: {ext}")

def load_map(path: Union[str, Path]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """Load hyperspectral maps from TXT, CSV, WDF, or SPC files.
    Returns a pandas DataFrame (and optionally metadata for WDF/SPC).
    """
    path = Path(path)
    ext = path.suffix.lower()
    
    if ext in [".txt", ".csv"]:
        return load_map_file(path)
    elif ext == ".wdf":
        return load_wdf_map(path)
    elif ext == ".spc":
        return load_spc_map(path)
    else:
        raise ValueError(f"Unsupported map file type: {ext}")

def load_dataset(path: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """Load standard tabular datasets from Excel or CSV files."""
    return load_dataframe_file(Path(path))

def export_results(results: Union[pd.DataFrame, List[dict]], output_path: Union[str, Path]):
    """Export fit results or data to an Excel or CSV file."""
    path = Path(output_path)
    if isinstance(results, list):
        df = pd.DataFrame(results)
    else:
        df = results
        
    ext = path.suffix.lower()
    if ext in ['.xlsx', '.xls']:
        df.to_excel(path, index=False)
    elif ext == '.csv':
        df.to_csv(path, index=False, sep=';')
    else:
        raise ValueError("Output must be .xlsx or .csv")


def load_spectra_to_matrix(
    paths: Union[str, Path, List[Union[str, Path]]]
) -> dict:
    """Load one or more spectrum files and return as a ready-to-use x/Y matrix.

    This convenience wrapper calls `load_spectra()` for each file and stacks
    all items into a single intensity matrix, interpolating onto a common
    wavenumber axis when necessary.

    Args:
        paths: A single file path or a list of file paths.

    Returns:
        dict with keys:
            ``x``      — float64[M] shared wavenumber axis.
            ``Y``      — float64[N, M] intensity matrix (one row per spectrum).
            ``names``  — list[str] of spectrum names (length N).
            ``metadata`` — list[dict] of per-spectrum acquisition metadata.

    Example::

        data = io.load_spectra_to_matrix(["sample_a.wdf", "sample_b.wdf"])
        x = data["x"]     # shape (M,)
        Y = data["Y"]     # shape (N, M)
    """
    import numpy as np
    from scipy.interpolate import interp1d

    if isinstance(paths, (str, Path)):
        paths = [paths]

    all_x = []
    all_y = []
    all_names = []
    all_meta = []

    for p in paths:
        raw = load_spectra(p)
        for item in raw.get("items", []):
            all_x.append(item["x0"])
            all_y.append(item["y0"])
            all_names.append(item["name"])
            all_meta.append(item.get("metadata", {}))

    if not all_x:
        raise ValueError("No spectra were loaded from the provided paths.")

    # Use the first spectrum's axis as the reference
    x_ref = all_x[0]
    rows = []
    for xi, yi in zip(all_x, all_y):
        if len(xi) == len(x_ref) and np.allclose(xi, x_ref):
            rows.append(yi.astype(np.float64))
        else:
            # Interpolate onto the reference axis
            f = interp1d(xi, yi, kind="linear", bounds_error=False, fill_value=0.0)
            rows.append(f(x_ref))

    return {
        "x": x_ref.astype(np.float64),
        "Y": np.stack(rows, axis=0),
        "names": all_names,
        "metadata": all_meta,
    }

