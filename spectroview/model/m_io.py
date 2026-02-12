# spectroview/model/m_io.py

import numpy as np
import pandas as pd
from renishawWiRE import WDFReader

from pathlib import Path
from spectroview.model.m_spectrum import MSpectrum
from spectroview.viewmodel.utils import parse_wdf_metadata
from spectroview.model.m_spc import SpcReader
from collections import OrderedDict


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
    #s.source_path = str(path.resolve()) 
    s.fname = path.stem
    s.x0 = df.iloc[:, 0].to_numpy()
    s.y0 = df.iloc[:, 1].to_numpy()
    s.x = s.x0.copy()
    s.y = s.y0.copy()
    s.baseline.mode = "Linear"
    s.baseline.sigma = 4

    return s


def load_wdf_spectrum(path: Path) -> MSpectrum:
    """Load single spectrum from Renishaw .wdf file.
    
    Uses renishawWIRE package to read native WiRE software files.
    Prints comprehensive metadata to console.
    
    """
    reader = WDFReader(str(path))
        
    # Extract data
    wavenumbers = reader.xdata  # Wavenumber axis (cm^-1)
    
    # Handle different array dimensions
    # Single spectrum: spectra is 1D array (num_wavenumbers,)
    # Multiple spectra: spectra is 2D array (num_spectra, num_wavenumbers)
    if reader.spectra.ndim == 1:
        intensities = reader.spectra  # Already 1D
    else:
        # Multiple spectra - take the first one or average
        intensities = reader.spectra[0] if reader.count == 1 else reader.spectra.mean(axis=0)
    
    # Ensure x-axis is ascending (Fitspy requirement)
    # WDF files often have descending wavenumbers (e.g. 3200 -> 100)
    wavenumbers = np.array(wavenumbers, dtype=np.float64)
    intensities = np.array(intensities, dtype=np.float64)
    
    if len(wavenumbers) > 1 and wavenumbers[0] > wavenumbers[-1]:
        sort_inds = np.argsort(wavenumbers)
        wavenumbers = wavenumbers[sort_inds]
        intensities = intensities[sort_inds]
    
    
    # Create MSpectrum object
    s = MSpectrum()
    #s.source_path = str(path.resolve())
    s.fname = path.stem
    s.x0 = np.array(wavenumbers, dtype=np.float64)
    s.y0 = np.array(intensities, dtype=np.float64)
    s.x = s.x0.copy()
    s.y = s.y0.copy()
    s.baseline.mode = "Linear"
    s.baseline.sigma = 4
    
    # Extract additional metadata from WDF file blocks
    wdf_metadata = parse_wdf_metadata(reader)
    
    # Build metadata dictionary with proper formatting
    metadata = {
        "File Format": "Renishaw WDF",

        "Grating": wdf_metadata.get('grating_name', 'Unknown'),
        "Objective Used": wdf_metadata.get('objective_name', 'Unknown'),

        "Laser Wavelength (nm)": f"{reader.laser_length:.2f}",
        "Laser Power (%)": f"{wdf_metadata['laser_power']}" if 'laser_power' in wdf_metadata else 'Unknown',

        "Exposure Time (s)": f"{wdf_metadata['exposure_time']}" if 'exposure_time' in wdf_metadata else 'Unknown',
        "Accumulations": reader.accumulation_count,

        "Slit Opening (µm)": f"{wdf_metadata['slit_opening']}" if 'slit_opening' in wdf_metadata else 'Unknown',
        "Slit Centre (µm)": f"{wdf_metadata['slit_centre']}" if 'slit_centre' in wdf_metadata else 'Unknown',

        "Title": reader.title,
        "Username": reader.username,
        "Date": wdf_metadata.get('timestamp', 'Unknown'),
        "Application": f"{reader.application_name} v{'.'.join(map(str, reader.application_version))}",
        "Measurement Type": str(reader.measurement_type),
        "Scan Type": str(reader.scan_type),
        "Number of Spectra": reader.count,
        "Points per Spectrum": reader.point_per_spectrum,
        "Spectral Unit": reader.spectral_unit,
        "X-axis Type": reader.xlist_type,
        "X-axis Unit": reader.xlist_unit,
        "Wavenumber Range": f"{reader.xdata[0]:.2f} to {reader.xdata[-1]:.2f} {reader.xlist_unit}",
    }
    
    # Assign metadata to spectrum
    s.metadata = metadata
    
    reader.close()
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
    #s.source_path = str(path.resolve())
    s.fname = path.stem
    # Explicitly use float64 for both x and y to ensure compatibility with save/load
    # (decompress always uses float64, so we must match that dtype)
    s.x0 = np.array(x_values, dtype=np.float64)
    s.y0 = np.array(extracted_y, dtype=np.float64)
    s.x = s.x0.copy()
    s.y = s.y0.copy()
    s.baseline.mode = "Linear"
    s.baseline.is_subtracted = False  
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


def load_wdf_map(path: Path) -> pd.DataFrame:
    """Load 2D hyperspectral map from Renishaw .wdf file.
    
    Uses renishawWIRE package to read native WiRE software mapping files.
    Prints comprehensive metadata to console.
    Converts to DataFrame format compatible with SPECTROview Maps workspace.
    """
    reader = WDFReader(str(path))
      
    # Extract data
    x_coords = reader.xpos  # X stage positions (µm)
    y_coords = reader.ypos  # Y stage positions (µm)
    wavenumbers = reader.xdata  # Wavenumber axis (cm^-1)
    spectra_data = reader.spectra  # Can be 2D or 3D array
    
    # Handle different array shapes
    # For 2D maps, spectra might be 3D: (x_grid, y_grid, wavenumbers)
    # We need to reshape to 2D: (num_spectra, num_wavenumbers)
    if spectra_data.ndim == 3:
        # Reshape from (x_grid, y_grid, wavenumbers) to (num_spectra, num_wavenumbers)
        x_size, y_size, num_wavenumbers = spectra_data.shape
        spectra_matrix = spectra_data.reshape(x_size * y_size, num_wavenumbers)
    else:
        # Already 2D
        spectra_matrix = spectra_data
        
    # Ensure x-axis is ascending (Fitspy requirement)
    # WDF files often have descending wavenumbers
    wavenumbers = np.array(wavenumbers)
    if len(wavenumbers) > 1 and wavenumbers[0] > wavenumbers[-1]:
        sort_inds = np.argsort(wavenumbers)
        wavenumbers = wavenumbers[sort_inds]
        # Reorder columns of the spectra matrix
        spectra_matrix = spectra_matrix[:, sort_inds]
    
    # Create DataFrame with X, Y columns and wavenumber columns
    # Convert wavenumbers to strings for column names
    wavenumber_cols = [str(wn) for wn in wavenumbers]
    
    # Build dictionary for DataFrame
    data_dict = {
        'X': x_coords,
        'Y': y_coords
    }
    
    # Add spectral data columns
    for i in range(len(wavenumber_cols)):
        data_dict[wavenumber_cols[i]] = spectra_matrix[:, i]
    
    df = pd.DataFrame(data_dict)
    
    # Extract additional metadata from WDF file blocks
    wdf_metadata = parse_wdf_metadata(reader)
    
    # Calculate step size from position arrays (for maps)
    x_step = None
    y_step = None
    
    if len(reader.xpos) > 1:
        x_unique = np.unique(reader.xpos)
        if len(x_unique) > 1:
            x_step = x_unique[1] - x_unique[0]
    
    if len(reader.ypos) > 1:
        y_unique = np.unique(reader.ypos)
        if len(y_unique) > 1:
            y_step = y_unique[1] - y_unique[0]
    
    # Build metadata dictionary with proper formatting and order
    metadata = {
        "File Format": "Renishaw WDF Map",

        "Grating": wdf_metadata.get('grating_name', 'Unknown'),
        "Objective Used": wdf_metadata.get('objective_name', 'Unknown'),

        "Laser Wavelength (nm)": f"{reader.laser_length:.2f}",
        "Laser Power (%)": f"{wdf_metadata['laser_power']}" if 'laser_power' in wdf_metadata else 'Unknown',

        "Exposure Time (s)": f"{wdf_metadata['exposure_time']}" if 'exposure_time' in wdf_metadata else 'Unknown',
        "Accumulations": reader.accumulation_count,

        "Slit Opening (µm)": f"{wdf_metadata['slit_opening']}" if 'slit_opening' in wdf_metadata else 'Unknown',
        "Slit Centre (µm)": f"{wdf_metadata['slit_centre']}" if 'slit_centre' in wdf_metadata else 'Unknown',
        
        "Title": reader.title,
        "Username": reader.username,
        "Date": wdf_metadata.get('timestamp', 'Unknown'),
        "Application": f"{reader.application_name} v{'.'.join(map(str, reader.application_version))}",

        "Measurement Type": str(reader.measurement_type),    
        "Scan Type": str(reader.scan_type),
        "Total Map Points": reader.count,
        "X Range (µm)": f"{reader.xpos.min():.2f} to {reader.xpos.max():.2f}",
        "Y Range (µm)": f"{reader.ypos.min():.2f} to {reader.ypos.max():.2f}",
        "X Step Size (µm)": f"{x_step:.3f}" if x_step is not None else 'Unknown',
        "Y Step Size (µm)": f"{y_step:.3f}" if y_step is not None else 'Unknown',

        "Points per Spectrum": reader.point_per_spectrum,
        "Spectral Unit": reader.spectral_unit,
        "X-axis Type": reader.xlist_type,
        "X-axis Unit": reader.xlist_unit,
        "Wavenumber Range": f"{reader.xdata[0]:.2f} to {reader.xdata[-1]:.2f} {reader.xlist_unit}",
    }
    
    reader.close()
    return df, metadata


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


def load_spc_spectrum(path: Path) -> MSpectrum:
    """Load single spectrum from Galactic .spc file."""
    reader = SpcReader(str(path))
    
    # Create MSpectrum object
    s = MSpectrum()
    s.fname = path.stem
    
    # Use the first spectrum if multiple exist, or the only one
    if reader.header['fnsub'] > 1:
        y_data = reader.y_data[0]
    else:
        y_data = reader.y_data
        
    s.x0 = np.array(reader.x_data, dtype=np.float64)
    s.y0 = np.array(y_data, dtype=np.float64)
    s.x = s.x0.copy()
    s.y = s.y0.copy()
    s.baseline.mode = "Linear"
    s.baseline.sigma = 4
    
    # Metadata
    metadata = {
        "File Format": "Galactic SPC",
        "Title": reader.header.get('fcmnt', ''),
        "Date": reader.header.get('date', ''),
        "Points": reader.header['npts'],
        "X Units": "Wavenumber" if reader.header.get('ftflgs', 0) & 0x80 else "Arbitrary", 
    }
    
    # Add metadata from Log Block if available
    if hasattr(reader, 'log_metadata') and reader.log_metadata:
        metadata.update(reader.log_metadata)
        
    s.metadata = metadata
    
    return s


def load_spc_map(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load 2D hyperspectral map from Galactic .spc file."""
    reader = SpcReader(str(path))
    
    num_spectra = reader.header['fnsub']
    
    # Try to extract X, Y from subheaders
    # "time" usually maps to X (fast axis) and "wlevel" to Y (slow axis) in valid maps
    if reader.subheaders:
        extracted_x = np.array([sh.get('time', 0.0) for sh in reader.subheaders], dtype=float)
        extracted_y = np.array([sh.get('wlevel', 0.0) for sh in reader.subheaders], dtype=float)
        
        # Check if they look like valid coordinates (not all zeros)
        if np.any(extracted_x) or np.any(extracted_y):
             x_coords = extracted_x
             y_coords = extracted_y
        else:
             # Fallback to grid generation
             side = int(np.sqrt(num_spectra))
             if side * side == num_spectra:
                 y_indices, x_indices = np.unravel_index(np.arange(num_spectra), (side, side))
                 x_coords = x_indices.astype(float)
                 y_coords = y_indices.astype(float)
             else:
                 x_coords = np.arange(num_spectra, dtype=float)
    else:
        # Fallback if no subheaders (unlikely for SPC)
        side = int(np.sqrt(num_spectra))
        if side * side == num_spectra:
            y_indices, x_indices = np.unravel_index(np.arange(num_spectra), (side, side))
            x_coords = x_indices.astype(float)
            y_coords = y_indices.astype(float)
        else:
            x_coords = np.arange(num_spectra, dtype=float)
            
    wavenumbers = reader.x_data
    spectra_matrix = reader.y_data # (num_spectra, num_points)
    
    # Ensure dimensions match
    if spectra_matrix.ndim == 1:
         # Single spectrum treated as map?
         spectra_matrix = spectra_matrix.reshape(1, -1)
         
    # Create DataFrame
    data_dict = {
        'X': x_coords,
        'Y': y_coords
    }
    
    wavenumber_cols = [str(wn) for wn in wavenumbers]
    
    for i in range(len(wavenumber_cols)):
        data_dict[wavenumber_cols[i]] = spectra_matrix[:, i]
        
    df = pd.DataFrame(data_dict)
    
    metadata = {
        "File Format": "Galactic SPC Map",
        "Title": reader.header.get('fcmnt', ''),
        "Date": reader.header.get('date', ''),
        "Total Map Points": num_spectra,
        "Points per Spectrum": reader.header['npts'],
    }
    
    # Add metadata from Log Block if available
    if hasattr(reader, 'log_metadata') and reader.log_metadata:
        metadata.update(reader.log_metadata)
        
    return df, metadata
