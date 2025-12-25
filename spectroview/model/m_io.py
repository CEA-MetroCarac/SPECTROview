# spectroview/model/m_io.py

import numpy as np
import pandas as pd
from pathlib import Path
from spectroview.model.m_spectrum import MSpectrum


def load_spectrum_file(path: Path) -> MSpectrum:
    """Load TXT or CSV spectrum file."""
    ext = path.suffix.lower()

    if ext == ".txt":
        df = pd.read_csv(path, header=None, skiprows=1, delimiter="\t")
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    df = df.sort_values(df.columns[0])

    s = MSpectrum()
    s.source_path = str(path.resolve()) 
    s.fname = path.stem
    s.x0 = df.iloc[:, 0].to_numpy()
    s.y0 = df.iloc[:, 1].to_numpy()
    s.x = s.x0.copy()
    s.y = s.y0.copy()
    s.baseline.mode = "Linear"
    s.baseline.sigma = 4

    return s


def load_map_file(path: Path) -> MSpectrum:
    """Load TXT or CSV spectrum file."""
    pass


def load_dataframe_file(path: Path) -> pd.DataFrame:
    """Load DataFrame from EXCEL or CSV file."""
    ext = path.suffix.lower()

    if ext == ".txt":
        df = pd.read_csv(path, header=None, skiprows=1, delimiter="\t")
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return df

def save_spectra_workspace(path: Path, spectra: list[MSpectrum]):
    """Save the current spectrum workspace to a .spectra file."""
    pass

def load_spectra_workspace(path: Path) -> list[MSpectrum]:
    """Load a .spectra file into a list of MSpectrum objects."""
    pass

def save_maps_workspace(path: Path, maps: list[MSpectrum]):
    """Save the current maps workspace to a .maps file."""
    pass   

def load_maps_workspace(path: Path) -> list[MSpectrum]:
    """Load a .maps file into a list of MSpectrum objects."""
    pass   

def save_graphs_workspace(path: Path, dataframes: list[pd.DataFrame]):
    """Save the current graphs workspace to a .graphs file."""
    pass   

def load_graphs_workspace(path: Path) -> list[pd.DataFrame]:
    """Load a .graphs file into a list of DataFrame objects."""
    pass