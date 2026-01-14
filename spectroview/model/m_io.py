# spectroview/model/m_io.py

import numpy as np
import pandas as pd
from pathlib import Path
from spectroview.model.m_spectrum import MSpectrum


def load_spectrum_file(path: Path) -> MSpectrum:
    """Load TXT or CSV spectrum file.
    
    For TXT files, automatically detects delimiter (semicolon, tab, or space).
    For CSV files, uses default comma delimiter.
    """
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


def load_map_file(path: Path) -> pd.DataFrame:
    """Load hyperspectral map file (CSV or TXT). """
    ext = path.suffix.lower()
    
    if ext == '.csv':
        # Read first 3 lines to determine format
        with open(path, 'r') as file:
            lines = [next(file) for _ in range(3)]
        
        # Check 2nd line to determine old and new format
        if len(lines[1].split(';')) > 3:
            # New format: multiple columns
            map_df = pd.read_csv(path, skiprows=1, delimiter=";")
        else:
            # Old format: alternating rows
            df = pd.read_csv(path, skiprows=2, delimiter=";")
            map_df = df.iloc[::2].reset_index(drop=True)
            map_df.rename(columns={
                map_df.columns[0]: "X", 
                map_df.columns[1]: "Y"
            }, inplace=True)
    
    elif ext == '.txt':
        map_df = pd.read_csv(path, delimiter="\t")
        map_df.columns = ['Y', 'X'] + list(map_df.columns[2:])
        # Reorder columns by increasing wavenumber
        sorted_columns = sorted(map_df.columns[2:], key=float)
        map_df = map_df[['X', 'Y'] + sorted_columns]
    
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    return map_df


def load_dataframe_file(path: Path) -> dict[str, pd.DataFrame]:
    """Load DataFrame(s) from EXCEL or CSV file."""
    ext = path.suffix.lower()
    
    if ext in ['.xlsx', '.xls']:
        # Read all sheets from Excel file
        excel_file = pd.ExcelFile(path)
        sheet_names = excel_file.sheet_names
        
        if len(sheet_names) == 1:
            # Single sheet: use filename as key
            df = pd.read_excel(path)
            return {path.stem: df}
        else:
            # Multiple sheets: use filename_sheetname as keys
            dfs = {}
            for sheet_name in sheet_names:
                df = pd.read_excel(path, sheet_name=sheet_name)
                df_name = f"{path.stem}_{sheet_name}"
                dfs[df_name] = df
            return dfs
    elif ext == '.csv':
        df = pd.read_csv(path)
        return {path.stem: df}
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    return df