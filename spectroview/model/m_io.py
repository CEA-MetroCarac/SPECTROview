# spectroview/model/m_io.py

import numpy as np
import pandas as pd
from pathlib import Path
from spectroview.model.m_spectrum import SpectrumM


def load_spectrum_file(path: Path) -> SpectrumM:
    """Load TXT or CSV spectrum file."""
    ext = path.suffix.lower()

    if ext == ".txt":
        df = pd.read_csv(path, header=None, skiprows=1, delimiter="\t")
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    df = df.sort_values(df.columns[0])

    s = SpectrumM()
    s.source_path = str(path.resolve()) 
    s.fname = path.stem
    s.x0 = df.iloc[:, 0].to_numpy()
    s.y0 = df.iloc[:, 1].to_numpy()
    s.x = s.x0.copy()
    s.y = s.y0.copy()
    s.baseline.mode = "Linear"

    return s


def load_map_file(path: Path) -> SpectrumM:
    """Load TXT or CSV spectrum file."""
    pass