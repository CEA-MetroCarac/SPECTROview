# spectroview/model/m_io.py

import numpy as np
import pandas as pd
from pathlib import Path
from spectroview.model.m_spectrum import MSpectrum


def load_spectrum_file(path: Path) -> MSpectrum:
    """Load TXT or CSV spectrum file.
    
    For TXT files, automatically detects delimiter (semicolon, tab, or space).
    For CSV files, uses semicolon delimiter and skips 3 header rows.
    """
    ext = path.suffix.lower()

    if ext == ".txt":
        # Auto-detect delimiter by reading the first data line (after header)
        with open(path, 'r') as f:
            first_line = next(f, None)  # Skip first line (header)
            second_line = next(f, None)  # Read second line to detect delimiter
            
        # Use second line for detection if available, otherwise first line
        test_line = second_line if second_line else first_line
        
        if test_line:
            # Check for delimiter in priority order: semicolon, tab, then space/whitespace
            if ';' in test_line:
                delimiter = ';'
            elif '\t' in test_line:
                delimiter = '\t'
            else:
                delimiter = r'\s+'
        else:
            delimiter = '\t'
        
        # Try reading with header (skiprows=1), then without if that fails
        try:
            df = pd.read_csv(path, header=None, skiprows=1, delimiter=delimiter, 
                           engine='python' if delimiter == r'\s+' else 'c')
            # Check if we got valid data
            if df.empty or df.shape[1] < 2:
                # Try without skipping rows (no header)
                df = pd.read_csv(path, header=None, delimiter=delimiter, 
                               engine='python' if delimiter == r'\s+' else 'c')
        except:
            # Fallback: try without skipping rows
            df = pd.read_csv(path, header=None, delimiter=delimiter, 
                           engine='python' if delimiter == r'\s+' else 'c')
    elif ext == ".csv":
        # CSV files use semicolon delimiter and have 3 header rows
        df = pd.read_csv(path, delimiter=";", header=None, skiprows=3)
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


def load_TRPL_data(path: Path) -> MSpectrum:
    """Load TRPL .dat file (Time-Resolved Photoluminescence).
    
    Extracts:
    - Bin value from line after "#ns/bin"
    - Count values from lines after "#counts" (until first zero)
    - Data from max count index onwards
    - X-axis = time in ns (index * bin_value)
    """
    bin_value = None
    y_values = []

    with open(path, 'r') as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Extract bin value from line after "#ns/bin"
        if line.startswith("#ns/bin") and i + 1 < len(lines):
            bin_value = float(lines[i + 1].strip())
        
        # Extract count values from lines after "#counts"
        if line.startswith("#counts"):
            for count_line in lines[i + 1:]:
                try:
                    value = int(count_line.strip())
                    if value == 0:  # Stop at first zero
                        break
                    y_values.append(value)
                except ValueError:
                    # Skip non-integer lines
                    break
            break
    
    
    if bin_value is None or not y_values:
        raise ValueError("Invalid TRPL file format: missing bin value or counts")
    
    # Find max y-value and extract data from that point onwards
    # This aligns t=0 with the peak of the decay curve
    max_y_index = y_values.index(max(y_values))
    extracted_y = y_values[max_y_index:]
    
    # Generate x-values (time in ns), starting from 0 at the peak
    x_values = [i * bin_value for i in range(len(extracted_y))]
    
    # Create MSpectrum object
    s = MSpectrum()
    s.source_path = str(path.resolve())
    s.fname = path.stem
    s.x0 = np.array(x_values)
    s.y0 = np.array(extracted_y)
    s.x = s.x0.copy()
    s.y = s.y0.copy()
    
    # Disable baseline for TRPL decay fitting
    # Decay models have built-in baseline parameter B, so fitspy's baseline
    # would interfere with parameter optimization
    s.baseline.mode = "Linear"
    s.baseline.is_subtracted = True  # Mark as already subtracted to disable baseline fitting
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
        # Use context manager to ensure file is properly closed
        with pd.ExcelFile(path) as excel_file:
            sheet_names = excel_file.sheet_names
            
            if len(sheet_names) == 1:
                # Single sheet: use filename as key
                df = pd.read_excel(excel_file)
                return {path.stem: df}
            else:
                # Multiple sheets: use filename_sheetname as keys
                dfs = {}
                for sheet_name in sheet_names:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    df_name = f"{path.stem}_{sheet_name}"
                    dfs[df_name] = df
                return dfs
    elif ext == '.csv':
        try:
            # Try reading with semicolon delimiter first (our default format)
            df = pd.read_csv(path, sep=';')
            # If only 1 column, it might strictly be comma separated
            if df.shape[1] == 1:
                 # Try comma
                 df_comma = pd.read_csv(path, sep=',')
                 if df_comma.shape[1] > 1:
                     df = df_comma
        except:
             # Fallback to default
             df = pd.read_csv(path)
             
        return {path.stem: df}
    else:
        raise ValueError(f"Unsupported file type: {ext}")
