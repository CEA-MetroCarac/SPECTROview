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
