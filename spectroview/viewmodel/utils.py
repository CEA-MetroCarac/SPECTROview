# spectroview/viewmodel/utils.py
from PySide6.QtGui import QColor, QPixmap, QImage, QTextCursor, QIcon, QPainter, QCursor, QPalette

def get_tinted_icon(path: str, color_str: str) -> QIcon:
    """Return a QIcon created from the image at `path` tinted to `color_str`."""
    pixmap = QPixmap(path)
    painter = QPainter(pixmap)
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), QColor(color_str))
    painter.end()
    return QIcon(pixmap)

from PySide6.QtCore import Qt, QObject, QEvent, QPoint, QTimer
from PySide6.QtWidgets import (
    QMessageBox, QApplication, QListWidgetItem,
    QDialog, QTextBrowser, QVBoxLayout, QToolTip,
    QFrame, QLabel
)    

import os
import datetime
import struct
import re
import numpy as np
import matplotlib
from io import BytesIO

from copy import deepcopy

from spectroview import DEFAULT_COLORS



def parse_wdf_metadata(reader):
    """Extract comprehensive metadata from WDF file's WXIS and WXCS blocks.
    
    This function parses unparsed WDF file blocks to extract instrument settings
   that are not exposed by the renishawWiRE package but are stored in the file.
    
    Args:
        reader: WDFReader object with file already opened
        
    Returns:
        dict with keys: objective_name, grating_name, exposure_time, 
                       slit_opening, slit_centre, laser_power (if available)
    """
    metadata = {}
    
    # Parse WXIS block for objective, grating, slit, and laser power
    if 'WXIS' in reader.block_info:
        uid, pos, size = reader.block_info['WXIS']
        reader.file_obj.seek(pos + 16)
        wxis_data = reader.file_obj.read(size - 16)
        
        try:
            text_data = wxis_data.decode('utf-8', errors='ignore')
            
            # Extract objective (e.g., "x100", "x50", "x20")
            for objective in ['x100', 'x50', 'x20', 'x10', 'x5', 'x2']:
                if objective in text_data:
                    metadata['objective_name'] = objective
                    break
            
            # Extract grating (e.g., "1800 l/mm")
            grating_patterns = [
                '1800 l/mm', '2400 l/mm', '1200 l/mm',
                '600 l/mm', '300 l/mm', '150 l/mm'
            ]
            for pattern in grating_patterns:
                if pattern in text_data:
                    metadata['grating_name'] = pattern
                    break
            
            # Extract SLIT OPENING and centre using labels
            # More robust: find all number-µm pairs
            all_um_values = re.findall(r'(\d+)[\xc2\xb5µ]m', text_data)
            if len(all_um_values) >= 3:
                # Skip first value (bias), take second (opening) and third (centre)
                try:
                    slit_opening = float(all_um_values[1])  # Second value
                    slit_centre = float(all_um_values[2])   # Third value
                    
                    # Sanity check: slit opening typically 10-200, centre typically 100-3000
                    if 5 <= slit_opening <= 500 and 50 <= slit_centre <= 5000:
                        metadata['slit_opening'] = slit_opening
                        metadata['slit_centre'] = slit_centre
                except (ValueError, IndexError):
                    pass
            
            # Extract LASER POWER from WXIS structured records
            # Laser power is stored as a 'u' (string) record with id=0x0500
            # Located near offset 2300-2400 with a simple numeric string value
            # Record format: u(0x75) + 2-byte-id(0x0500) + \x80 + 4-byte-strlen + string
            # Search specifically in the range 2300-2500 for a simple numeric pattern
            for i in range(2300, min(len(wxis_data) - 12, 2500)):
                if (wxis_data[i:i+1] == b'u' and wxis_data[i+3:i+4] == b'\x80'):
                    try:
                        rec_id = struct.unpack('<H', wxis_data[i+1:i+3])[0]
                        str_len = struct.unpack('<i', wxis_data[i+4:i+8])[0]
                        if rec_id == 0x0500 and 0 < str_len < 20:
                            power_str = wxis_data[i+8:i+8+str_len].decode(
                                'utf-8', errors='ignore').strip('\x00')
                            # Check if it's a simple numeric string (laser power)
                            # Should be like "0.5" or "1" (not "30.8786 Degrees")
                            if power_str and not any(c.isalpha() for c in power_str):
                                try:
                                    power_val = float(power_str)
                                    # Laser power typically 0.1% to 100%
                                    if 0.01 <= power_val <= 100:
                                        metadata['laser_power'] = power_val
                                        break
                                except ValueError:
                                    pass
                    except (struct.error, IndexError):
                        pass
            
        except Exception:
            pass
    
    # Parse WXCS block for confocal values (alternative source)
    if 'WXCS' in reader.block_info and ('slit_opening' not in metadata or 'slit_centre' not in metadata):
        uid, pos, size = reader.block_info['WXCS']
        reader.file_obj.seek(pos + 16)
        wxcs_data = reader.file_obj.read(size - 16)
        
        try:
            text_data = wxcs_data.decode('utf-8', errors='ignore')
            
            # WXCS contains "20.0µm" and "1724.7µm" for confocal opening and centre
            all_um_values = re.findall(r'(\d+\.?\d*)[\xc2\xb5µ]m', text_data)
            
            # Look for two values where one is small (~20) and one is large (~1724)
            for i in range(len(all_um_values) - 1):
                try:
                    val1 = float(all_um_values[i])
                    val2 = float(all_um_values[i+1])
                    
                    # Identify which is opening (smaller) and which is centre (larger)
                    if 5 <= val1 <= 500 and 50 <= val2 <= 5000:
                        if val1 < val2 and 'slit_opening' not in metadata:
                            metadata['slit_opening'] = val1
                            metadata['slit_centre'] = val2
                            break
                        elif val2 < val1 and 'slit_opening' not in metadata:
                            metadata['slit_opening'] = val2
                            metadata['slit_centre'] = val1
                            break
                except ValueError:
                    pass
        except Exception:
            pass
    
    # Parse WXDM block for exposure time
    # Exposure time is stored as int32 milliseconds in an 'i' record with id=0x2300
    # Record format: i(0x69) + 2-byte-id(0x2300) + \x80 + 4-byte-int-value
    if 'WXDM' in reader.block_info:
        uid, pos, size = reader.block_info['WXDM']
        reader.file_obj.seek(pos + 16)
        wxdm_data = reader.file_obj.read(size - 16)
        
        try:
            for i in range(len(wxdm_data) - 8):
                if (wxdm_data[i:i+1] == b'i'
                        and wxdm_data[i+3:i+4] == b'\x80'):
                    rec_id = struct.unpack('<H', wxdm_data[i+1:i+3])[0]
                    if rec_id == 0x2300:  # Exposure time record ID
                        val = struct.unpack('<i', wxdm_data[i+4:i+8])[0]
                        if 10 <= val <= 3600000:  # 10ms to 1 hour
                            metadata['exposure_time'] = val / 1000.0  # Convert ms to seconds
                            break
        except Exception:
            pass
    
    # Parse WDF1 block to find Creation Date
    # Priority:
    # 1. Windows FILETIME at 0x90 (End Time)
    # 2. Windows FILETIME at 0x88 (Start Time)
    # 3. Check for SYSTEMTIME structure (legacy fallback)
    # 4. File modification time (last resort)
    
    found_date = None
    
    try:
        reader.file_obj.seek(0)
        header_bytes = reader.file_obj.read(1024) # Read enough to cover header

        # Helper to decode FILETIME (100-ns intervals since Jan 1, 1601 UTC)
        def decode_filetime(ft_bytes):
            try:
                ft_int = int.from_bytes(ft_bytes, 'little')
                if ft_int == 0: return None
                
                EPOCH_AS_FILETIME = 116444736000000000
                HUNDREDS_OF_NANOSECONDS = 10000000
                
                if ft_int < EPOCH_AS_FILETIME: return None
                
                ts = (ft_int - EPOCH_AS_FILETIME) / HUNDREDS_OF_NANOSECONDS
                # Use local time for display consistent with user expectation
                dt = datetime.datetime.fromtimestamp(ts)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                return None

        # 1. Check End Time at 0x90
        if len(header_bytes) >= 0x98:
            d = decode_filetime(header_bytes[0x90:0x98])
            if d:
                found_date = d

        # 2. Check Start Time at 0x88 (if End Time not found)
        if not found_date and len(header_bytes) >= 0x90:
            d = decode_filetime(header_bytes[0x88:0x90])
            if d:
                found_date = d

        # 3. Legacy SYSTEMTIME scan
        if not found_date:
            def check_systemtime(buf, offset):
                if offset + 16 > len(buf): return None
                # SYSTEMTIME: 8 unsigned shorts (Year, Month, Dow, Day, Hour, Min, Sec, Ms)
                vals = struct.unpack('<8H', buf[offset:offset+16])
                wYear, wMonth, wDow, wDay, wHour, wMinute, wSecond, wMs = vals
                
                # Loose validation for reasonable date
                if (1990 <= wYear <= 2030 and 
                    1 <= wMonth <= 12 and 
                    1 <= wDay <= 31 and 
                    0 <= wHour <= 23 and 
                    0 <= wMinute <= 59 and 
                    0 <= wSecond <= 59):
                        # Filter out empty/zero dates
                        if wYear == 0 and wMonth == 0: return None
                        return f"{wYear}-{wMonth:02d}-{wDay:02d} {wHour:02d}:{wMinute:02d}:{wSecond:02d}"
                return None

            # Check known gaps
            for i in range(0xA0, 0xF0, 2):
                 d = check_systemtime(header_bytes, i)
                 if d:
                     found_date = d
                     break
            
            # Scan full header if needed
            if not found_date:
                for i in range(0, 512, 2):
                    d = check_systemtime(header_bytes, i)
                    if d:
                        found_date = d
                        break
                    
    except Exception as e:
        print(f"Error scanning WDF date: {e}")
        
    # 3. Fallback to file modification time
    if not found_date:
        try:
            fname = getattr(reader.file_obj, 'name', None)
            if fname and os.path.exists(fname):
                ts = os.path.getmtime(fname)
                dt = datetime.datetime.fromtimestamp(ts)
                found_date = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
            
    if found_date:
        metadata['timestamp'] = found_date
    
    return metadata


def view_text(ui, title, text):
        """ Create a QTextBrowser to display a text content"""
        report_viewer = QDialog(ui)
        report_viewer.setWindowTitle(title)
        report_viewer.setGeometry(100, 100, 800, 600)
        text_browser = QTextBrowser(report_viewer)
        text_browser.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        text_browser.setOpenExternalLinks(True)
        text_browser.setPlainText(text)
        text_browser.moveCursor(QTextCursor.Start)
        layout = QVBoxLayout(report_viewer)
        layout.addWidget(text_browser)
        report_viewer.show()  

        
def rgba_to_default_color(rgba, default_colors=DEFAULT_COLORS):
    """
    Convert an RGBA tuple to the closest color in DEFAULT_COLORS.
    If no DEFAULT_COLORS are given, falls back to hex.
    """
    import matplotlib.colors as mcolors
    # Convert input to RGB array
    rgb = np.array(mcolors.to_rgb(rgba)) # drops alpha

    # Compute distance to each default color
    best_color = None
    best_dist = float("inf")
    for dc in default_colors:
        dc_rgb = np.array(mcolors.to_rgb(dc))
        dist = np.linalg.norm(rgb - dc_rgb)  # Euclidean distance in RGB
        if dist < best_dist:
            best_dist = dist
            best_color = dc

    return best_color if best_color else mcolors.to_hex(rgba)


def set_spectrum_item_color(item: QListWidgetItem, spectrum_info: dict):
    """Set list item background color based on spectrum status.
    
    Status hierarchy:
    1. Fitted (has_fit):
       - Converged: Color 4 (Soft Success Green)
       - Not Converged: Color 3 (Soft Warning Orange)
    2. Baselined (has_baseline): Color 2 (Soft Purple)
    3. Cropped (is_cropped): Color 1 (Distinct Gray)
    4. Original/Reinit: Transparent (no color)
    """
    is_cropped = spectrum_info.get("is_cropped", False)
    has_baseline = spectrum_info.get("has_baseline", False)
    has_fit = spectrum_info.get("has_fit", False)
    fit_success = spectrum_info.get("fit_success", False)

    if has_fit:
        if fit_success:
            # Color 4: Converged Fit (Distinct Green)
            item.setBackground(QColor(76, 175, 80, 120))
        else:
            # Color 3: Unconverged Fit (Distinct Red)
            item.setBackground(QColor(244, 67, 54, 120))
    elif has_baseline:
        # Color 2: Baselined (Distinct Purple)
        item.setBackground(QColor(156, 39, 176, 120))
    elif is_cropped:
        # Color 1: Cropped (Distinct Gray)
        item.setBackground(QColor(158, 158, 158, 120))
    else:
        # Original/Reinit: Transparent
        item.setBackground(QColor(0, 0, 0, 0))

def show_alert(message):
    """Show alert"""
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setWindowTitle("Alert")
    msg_box.setText(message)
    msg_box.exec_()


def show_toast_notification(parent, message, title=None, duration=3000, preset=None):
    """Show an auto-dismissing toast notification"""
    try:
        from pyqttoast import Toast, ToastPreset
    except ImportError:
        # Fallback to console print if pyqttoast not available
        prefix = f"[{title}] " if title else ""
        print(f"{prefix}{message}")
        return None
    
    toast = Toast(parent)
    toast.setDuration(duration)
    if title:
        toast.setTitle(title)
    toast.setText(message)
    
    # Apply preset style (default to SUCCESS)
    if preset is None:
        preset = ToastPreset.SUCCESS
    toast.applyPreset(preset)
    
    toast.show()
    return toast



def closest_index(array, value):
    diff = np.abs(array - value)
    if np.isnan(diff).any():
        try:
            return int(np.nanargmin(diff))
        except ValueError:
            return 0
    return int(diff.argmin())


def parse_coords_from_fname(fname: str) -> tuple[float, float] | None:
    """Parse (x, y) spatial coordinates from a map spectrum fname.

    Spectra are named ``"{map_name}_({x}, {y})"`` (see
    ``VMWorkspaceMaps._extract_spectra_from_map``). Returns the coordinates as a
    float tuple, or None if the fname carries no parseable ``(x, y)`` suffix.
    """
    if '(' not in fname or ')' not in fname:
        return None
    coords_str = fname[fname.rfind('(') + 1:fname.rfind(')')]
    try:
        x_str, y_str = coords_str.split(',')
        return (float(x_str.strip()), float(y_str.strip()))
    except (ValueError, AttributeError):
        return None


def fano_display_amplitude(internal_ampli, q):
    """Convert a Fano peak's internal fit amplitude to its displayed peak height.

    The Fano lineshape's 'ampli' fit parameter scales the whole curve rather
    than the peak height directly (see fit_engine/models.py's batched_fano);
    the actual maximum intensity is ampli * (q**2 + 1). Used wherever the UI
    shows or edits a Fano peak's amplitude as an intuitive peak height
    instead of the raw fit parameter.
    """
    return internal_ampli * (q ** 2 + 1)


def fano_internal_amplitude(display_ampli, q):
    """Inverse of fano_display_amplitude(): displayed peak height -> internal fit amplitude."""
    return display_ampli / (q ** 2 + 1)


def save_df_to_excel(save_path, df):
    """Saves a DataFrame to an Excel file with colored columns based on prefixes.
    
    Optimized to use xlsxwriter with direct column writing and prefix cell formats
    for high speed (~1-4s for 30k rows vs ~20 minutes with cell-by-cell openpyxl),
    with an openpyxl fallback if xlsxwriter is unavailable.
    """
    if not save_path:
        return False, "No save path provided."

    try:
        import pandas as pd
        if df is None or df.empty:
            return False, "DataFrame is empty. Nothing to save."

        palette = ['#bda16d', '#a27ba0', '#cb5b12', '#23993b', '#008281', '#147ce4']

        # Try fast xlsxwriter path first
        try:
            import xlsxwriter
            wb = xlsxwriter.Workbook(save_path, {'nan_inf_to_errors': False})
            ws = wb.add_worksheet('Results')
            header_fmt = wb.add_format({'bold': True})

            prefix_formats = {}
            for col_idx, col_name in enumerate(df.columns):
                prefix = col_name.split("_")[0] if "_" in col_name else col_name
                if prefix not in prefix_formats:
                    color_index = len(prefix_formats) % len(palette)
                    prefix_formats[prefix] = wb.add_format({
                        'bg_color': palette[color_index],
                        'pattern': 1
                    })
                fmt = prefix_formats[prefix]
                ws.write(0, col_idx, str(col_name), header_fmt)

                # Clean NaNs and infs to None so xlsxwriter writes blank styled cells
                col_series = df[col_name]
                if pd.api.types.is_numeric_dtype(col_series):
                    mask_invalid = pd.isna(col_series) | np.isinf(col_series)
                    clean_vals = np.where(mask_invalid, None, col_series).tolist()
                else:
                    clean_vals = np.where(pd.isna(col_series), None, col_series).tolist()

                ws.write_column(1, col_idx, clean_vals, fmt)

            wb.close()
            return True, "DataFrame saved successfully."
        except ImportError:
            pass

        # Fallback path with openpyxl (iterating columns directly)
        from openpyxl.styles import PatternFill
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
            worksheet = writer.sheets['Results']
            prefix_colors = {}
            col_fills = []
            for col_name in df.columns:
                prefix = col_name.split("_")[0] if "_" in col_name else col_name
                if prefix not in prefix_colors:
                    color_index = len(prefix_colors) % len(palette)
                    prefix_colors[prefix] = PatternFill(
                        start_color=palette[color_index][1:],
                        end_color=palette[color_index][1:],
                        fill_type='solid'
                    )
                col_fills.append(prefix_colors[prefix])

            for col_cells, fill in zip(worksheet.iter_cols(min_row=2, max_row=len(df) + 1, min_col=1, max_col=len(df.columns)), col_fills):
                for cell in col_cells:
                    cell.fill = fill

        return True, "DataFrame saved successfully."

    except Exception as e:
        return False, f"Error when saving DataFrame: {str(e)}"
    
def replace_peak_labels(fit_model, param):
        """Replace prefix 'm01' of peak model by labels designed by user"""
        peak_labels = fit_model["peak_labels"]
        if "_" in param:
            prefix, param = param.split("_", 1)  
            # Convert prefix to peak_label
            peak_index = int(prefix[1:]) - 1
            if 0 <= peak_index < len(peak_labels):
                peak_label = peak_labels[peak_index]
                return f"{param}_{peak_label}"
        return param

def copy_fig_to_clb(canvas, size_ratio=None):
    """Copy matplotlib figure canvas to clipboard with optional resizing. """
    if not canvas:
        QMessageBox.critical(None, "Error", "No plot to copy.")
        return
    
    figure = canvas.figure
    original_size = figure.get_size_inches()
    original_dpi = figure.dpi
    
    try:
        # Optionally resize the figure
        if size_ratio:
            figure.set_size_inches(size_ratio, forward=True)
            canvas.draw()
        buf = BytesIO()
        figure.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        # Load PNG from buffer and copy to clipboard
        qimage = QImage()
        qimage.loadFromData(buf.getvalue())
        
        clipboard = QApplication.clipboard()
        clipboard.setPixmap(QPixmap.fromImage(qimage))
        
    finally:
        # Restore original figure state
        figure.set_size_inches(original_size, forward=True)
        figure.set_dpi(original_dpi)
        canvas.draw()

# Vector formats where embedding real (editable/searchable) text instead of
# outlined paths matters for publication use.
_VECTOR_FORMATS = ('pdf', 'eps', 'svg')

def export_figure_to_file(canvas, filepath, fmt, dpi=300, transparent=False, size_inches=None):
    """Export a matplotlib figure canvas to a file (PNG/TIFF/SVG/PDF/EPS).

    Mirrors copy_fig_to_clb's temporarily-resize-then-restore pattern so a
    caller can request a specific physical export size without permanently
    resizing the on-screen figure. Vector formats get font-embedding
    rcParams (fonttype 42 for PDF/EPS, real <text> elements for SVG) so
    exported text stays editable/searchable rather than outlined -- applied
    only for the duration of this save via matplotlib.rc_context, not
    process-wide.

    Returns True on success, False (after showing an error dialog) on failure.
    """
    if not canvas:
        QMessageBox.critical(None, "Error", "No plot to export.")
        return False

    figure = canvas.figure
    original_size = figure.get_size_inches()
    original_dpi = figure.dpi

    try:
        if size_inches:
            figure.set_size_inches(size_inches, forward=True)
            canvas.draw()

        # bbox_inches='tight' (copy_fig_to_clb's convenience default, avoids
        # excess whitespace) recomputes the saved bbox from rendered content
        # -- which silently ignores an explicitly requested physical size.
        # Only use it when the caller did NOT ask for a specific size; honor
        # the exact requested dimensions otherwise.
        bbox_inches = None if size_inches else 'tight'
        savefig_kwargs = dict(format=fmt, dpi=dpi, transparent=transparent, bbox_inches=bbox_inches)

        if fmt in _VECTOR_FORMATS:
            with matplotlib.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'}):
                figure.savefig(filepath, **savefig_kwargs)
        else:
            figure.savefig(filepath, **savefig_kwargs)
        return True
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Error exporting figure: {e}")
        return False
    finally:
        # Restore original figure state
        figure.set_size_inches(original_size, forward=True)
        figure.set_dpi(original_dpi)
        canvas.draw()

def build_clean_fit_model(fit_model):
    """Build a clean, consistently ordered fit model dict for serialization.

    Key order: fit_params, range_min, range_max, baseline, peak_labels, peak_models.
    Also rounds float values in fit_params to avoid QSettings float32 precision artifacts
    (e.g. 9.999999747378752e-06 -> 1e-05).
    """
    # Round fit_params floats to 6 significant digits to avoid float32 artifacts
    fit_params = fit_model.get("fit_params", {})
    if fit_params:
        cleaned_params = {}
        for k, v in fit_params.items():
            cleaned_params[k] = float(f"{v:.6g}") if isinstance(v, float) else v
        fit_params = cleaned_params

    # Clean baseline: strip fields not needed for the current mode
    baseline = fit_model.get("baseline")
    if baseline and isinstance(baseline, dict):
        baseline = deepcopy(baseline)
        baseline.pop("y_eval", None)
        if baseline.get("mode") == "Linear":
            baseline.pop("coef", None)
            baseline.pop("order_max", None)

    # Build ordered dict
    return {
        "fit_params": fit_params,
        "range_min": fit_model.get("range_min"),
        "range_max": fit_model.get("range_max"),
        "baseline": baseline,
        "peak_labels": fit_model.get("peak_labels", []),
        "peak_models": fit_model.get("peak_models", {}),
    }


def _clean_metadata_value(val):
    if val is None:
        return None
    s = str(val).strip()
    if s.lower() in ("", "unknown", "none", "nan", "n/a", "null"):
        return None
    return s


def extract_primary_metadata(metadata: dict | object | None) -> dict[str, str]:
    """Extract standard acquisition metadata fields from a dictionary or spectrum object.
    
    Returns a dict with keys:
    - 'Laser Wavelength'
    - 'Laser Power'
    - 'Exposure Time'
    - 'Accumulation'
    - 'Objective'
    
    Missing or unavailable values are gracefully represented as 'N/A'.
    """
    if metadata is None:
        raw_meta = {}
    elif isinstance(metadata, dict):
        raw_meta = metadata
    elif hasattr(metadata, "metadata") and isinstance(metadata.metadata, dict):
        raw_meta = metadata.metadata
    elif hasattr(metadata, "map_metadata") and isinstance(metadata.map_metadata, dict):
        raw_meta = metadata.map_metadata
    else:
        raw_meta = {}

    def get_first_match(candidate_keys):
        # 1. Exact match
        for k in candidate_keys:
            if k in raw_meta:
                v = _clean_metadata_value(raw_meta[k])
                if v is not None:
                    return k, v
        # 2. Case-insensitive match
        lower_map = {k.lower(): (k, v) for k, v in raw_meta.items()}
        for k in candidate_keys:
            lk = k.lower()
            if lk in lower_map:
                orig_k, orig_v = lower_map[lk]
                v = _clean_metadata_value(orig_v)
                if v is not None:
                    return orig_k, v
        return None, None

    # 1. Laser Wavelength
    k, v = get_first_match([
        "Laser Wavelength (nm)", "Laser Wavelength", "laser_wavelength",
        "EXCIT_LINE", "EXCIT LINE", "Wavelength (nm)", "Wavelength"
    ])
    if v:
        wavelength = v if v.lower().endswith("nm") else f"{v} nm"
    else:
        wavelength = "N/A"

    # 2. Laser Power
    k, v = get_first_match([
        "Laser Power (%)", "Laser Power (mW)", "Laser Power", "laser_power",
        "ND FILTER", "Power (%)", "Power (mW)", "Power", "POWER"
    ])
    if v:
        has_unit = "%" in v or "mw" in v.lower() or "w" in v.lower()
        if ("(%)" in (k or "") or "%" in (k or "")) and not has_unit:
            power = f"{v} %"
        elif "(mw)" in (k or "").lower() and not has_unit:
            power = f"{v} mW"
        else:
            power = v
    else:
        power = "N/A"

    # 3. Exposure Time
    k, v = get_first_match([
        "Exposure Time (s)", "Exposure Time", "exposure_time",
        "ACQ. TIME (S)", "Acquisition Time (s)", "Acquisition Time",
        "TIME", "TIME (S)", "ACQ. TIME", "ACQ TIME", "EXP TIME", "EXPOSURE TIME"
    ])
    if v:
        exposure = v if (v.lower().endswith("s") or v.lower().endswith("sec")) else f"{v} s"
    else:
        exposure = "N/A"

    # 4. Accumulation
    k, v = get_first_match([
        "Accumulations", "Accumulation", "accumulation_count", "ACCUMULATIONS",
        "ACCUM", "ACC", "NB ACCUM"
    ])
    accum = v if v else "N/A"

    # 5. Objective
    k, v = get_first_match([
        "Objective Used", "Objective", "objective_name", "OBJECTIV", "OBJECTIVE"
    ])
    obj = v if v else "N/A"

    return {
        "Laser Wavelength": wavelength,
        "Laser Power": power,
        "Exposure Time": exposure,
        "Accumulation": accum,
        "Objective": obj,
    }


def format_metadata_tooltip(metadata: dict | object | None) -> str:
    """Format primary metadata into a compact HTML tooltip."""
    fields = extract_primary_metadata(metadata)
    lines = [
        "<b>Metadata</b>",
        f"• Laser Wavelength: {fields['Laser Wavelength']}",
        f"• Laser Power: {fields['Laser Power']}",
        f"• Exposure Time: {fields['Exposure Time']}",
        f"• Accumulation: {fields['Accumulation']}",
        f"• Objective: {fields['Objective']}",
    ]
    return "<br>".join(lines)


class MetadataTooltipPopup(QFrame):
    """Floating tooltip frame with theme-adaptive styling that stays visible as long as needed."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowDoesNotAcceptFocus)
        self.setObjectName("MetadataTooltipPopup")
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(0)

        self.label = QLabel(self)
        self.label.setTextFormat(Qt.RichText)
        self.label.setWordWrap(False)
        layout.addWidget(self.label)

    def show_tip(self, html: str, global_pos: QPoint):
        """Update content and position near cursor, constrained to screen bounds."""
        app = QApplication.instance()
        if app and not app.styleSheet():
            # Classic themes or environments without QSS: style dynamically from palette
            pal = app.palette()
            bg = pal.color(QPalette.ToolTipBase).name()
            fg = pal.color(QPalette.ToolTipText).name()
            border = pal.color(QPalette.Mid).name()
            self.setStyleSheet(f"""
                QFrame#MetadataTooltipPopup {{
                    background-color: {bg};
                    color: {fg};
                    border: 1px solid {border};
                    border-radius: 6px;
                }}
                QLabel {{
                    background: transparent;
                    color: {fg};
                    font-size: 11px;
                    line-height: 1.4;
                }}
            """)
        elif self.styleSheet():
            # Clear any inline stylesheet so the application QSS rules apply
            self.setStyleSheet("")

        self.label.setText(html)
        self.adjustSize()

        pos = global_pos + QPoint(16, 16)
        screen = self.screen()
        if screen:
            sg = screen.availableGeometry()
            if pos.x() + self.width() > sg.right():
                pos.setX(max(sg.left(), global_pos.x() - self.width() - 8))
            if pos.y() + self.height() > sg.bottom():
                pos.setY(max(sg.top(), global_pos.y() - self.height() - 8))

        self.move(pos)
        self.show()
        self.raise_()


class MetadataTooltipManager(QObject):
    """Manages metadata tooltips for QListWidget items.
    
    Requirements:
    - Display tooltip when user hovers an item (with initial show_delay_ms delay, native-like).
    - Can optionally require Ctrl key if require_ctrl=True.
    - Once showing, seamlessly and immediately updates without delay when moving between items.
    - Stays displayed continuously as long as the mouse cursor is over items.
    - Does NOT automatically disappear after 2 seconds (independent of OS tooltip timeout).
    - Disappears when mouse leaves items or widget.
    - Disappears immediately on click, drag, scroll, or window hide so it never interferes.
    """

    def __init__(self, list_widget, metadata_provider=None, require_ctrl: bool = False, show_delay_ms: int = 1000):
        super().__init__(list_widget)
        self.list_widget = list_widget
        self.metadata_provider = metadata_provider
        self.require_ctrl = require_ctrl
        self.show_delay_ms = show_delay_ms
        self._hovered_item = None
        self._pending_item = None
        self._pending_global_pos = None
        self._is_showing = False
        self._app_filtered = False

        self._popup = MetadataTooltipPopup()

        self._delay_timer = QTimer(self)
        self._delay_timer.setSingleShot(True)
        self._delay_timer.timeout.connect(self._on_delay_timeout)

        self.list_widget.setMouseTracking(True)
        viewport = self.list_widget.viewport()
        if viewport:
            viewport.setMouseTracking(True)
            viewport.installEventFilter(self)
        self.list_widget.installEventFilter(self)

        app = QApplication.instance()
        if app:
            app.installEventFilter(self)
            self._app_filtered = True

        self.list_widget.destroyed.connect(self._on_widget_destroyed)

    def set_metadata_provider(self, provider):
        """Update the metadata provider callable."""
        self.metadata_provider = provider

    def set_delay_ms(self, delay: int):
        """Set initial hover delay in milliseconds before showing tooltip."""
        self.show_delay_ms = max(0, delay)

    def _cancel_delay(self):
        """Cancel pending hover delay timer and reset pending tracking."""
        try:
            if hasattr(self, "_delay_timer") and self._delay_timer is not None and self._delay_timer.isActive():
                self._delay_timer.stop()
        except RuntimeError:
            pass
        self._pending_item = None
        self._pending_global_pos = None

    def _on_delay_timeout(self):
        """Triggered when initial hover delay expires."""
        item = self._pending_item
        gpos = self._pending_global_pos
        self._pending_item = None
        self._pending_global_pos = None

        if not item or item.flags() == Qt.NoItemFlags or not item.text():
            return

        if self.require_ctrl and not self._is_ctrl_held():
            return

        try:
            lw = self.list_widget
            if not lw or not lw.isVisible():
                return
        except (RuntimeError, AttributeError):
            return

        if gpos is None:
            gpos = QCursor.pos()

        self._show_tooltip_for_item(item, gpos)

    def _remove_app_filter(self):
        """Cleanly remove application event filter."""
        if self._app_filtered:
            app = QApplication.instance()
            if app:
                try:
                    app.removeEventFilter(self)
                except Exception:
                    pass
            self._app_filtered = False

    def _on_widget_destroyed(self):
        """Cleanly remove application event filter and popup when widget is destroyed."""
        self._cancel_delay()
        self._remove_app_filter()
        if self._popup:
            try:
                self._popup.hide()
                self._popup.deleteLater()
            except Exception:
                pass
            self._popup = None

    def __del__(self):
        self._cancel_delay()
        self._remove_app_filter()

    @staticmethod
    def _is_ctrl_held() -> bool:
        """Check if Ctrl (or Cmd on macOS) modifier is held down."""
        mods = QApplication.keyboardModifiers()
        return bool(mods & (Qt.ControlModifier | Qt.MetaModifier))

    def _hide_tooltip(self):
        """Dismiss active tooltip and reset state."""
        self._cancel_delay()
        if self._popup:
            self._popup.hide()
        self._is_showing = False
        self._hovered_item = None

    def _fetch_metadata(self, item):
        """Retrieve metadata for the given item."""
        if self.metadata_provider:
            try:
                return self.metadata_provider(item)
            except TypeError:
                return self.metadata_provider(item.text())
        return item.data(Qt.UserRole + 2) or {}

    def _show_tooltip_for_item(self, item, global_pos):
        """Render and display tooltip for a valid list item."""
        if not item or item.flags() == Qt.NoItemFlags or not item.text():
            self._hide_tooltip()
            return

        if not self._popup:
            return

        metadata = self._fetch_metadata(item)
        tooltip_html = format_metadata_tooltip(metadata)

        self._popup.show_tip(tooltip_html, global_pos)
        self._hovered_item = item
        self._is_showing = True

    def eventFilter(self, watched, event):
        """Handle mouse movement, leave, and events."""
        try:
            lw = self.list_widget
            if lw is None:
                return False
            viewport = lw.viewport()
            if viewport is None:
                return False
        except (RuntimeError, AttributeError):
            return False

        etype = event.type()

        # 1. Mouse move on viewport
        if etype == QEvent.MouseMove and watched == viewport:
            if self.require_ctrl and not self._is_ctrl_held():
                self._hide_tooltip()
                return False

            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            item = lw.itemAt(pos)
            gpos = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()

            if not item or item.flags() == Qt.NoItemFlags or not item.text():
                self._hide_tooltip()
                return False

            if self._is_showing:
                # WARM STATE: tooltip is already active
                if item != self._hovered_item:
                    # Switched to another item -> display continuously and IMMEDIATELY without delay
                    self._cancel_delay()
                    self._show_tooltip_for_item(item, gpos)
                elif self._popup:
                    # Keep popup following cursor smoothly
                    pos_pop = gpos + QPoint(16, 16)
                    screen = self._popup.screen()
                    if screen:
                        sg = screen.availableGeometry()
                        if pos_pop.x() + self._popup.width() > sg.right():
                            pos_pop.setX(max(sg.left(), gpos.x() - self._popup.width() - 8))
                        if pos_pop.y() + self._popup.height() > sg.bottom():
                            pos_pop.setY(max(sg.top(), gpos.y() - self._popup.height() - 8))
                    self._popup.move(pos_pop)
            else:
                # COLD STATE: tooltip not yet showing -> apply delay
                if self.show_delay_ms <= 0:
                    self._show_tooltip_for_item(item, gpos)
                else:
                    if item != self._pending_item:
                        self._pending_item = item
                        self._pending_global_pos = gpos
                        self._delay_timer.start(self.show_delay_ms)
                    else:
                        # Moving within same pending item: track cursor position
                        self._pending_global_pos = gpos
                        if not self._delay_timer.isActive():
                            self._delay_timer.start(self.show_delay_ms)
            return False

        # 2. Suppress default Qt ToolTip event so QAbstractItemView does not clear tooltip
        elif etype == QEvent.ToolTip and watched in (viewport, lw):
            return True

        # 3. Mouse leave on viewport
        elif etype == QEvent.Leave and watched == viewport:
            self._hide_tooltip()
            return False

        # 4. Mouse click, wheel, or drag interaction - immediately hide tooltip
        elif etype in (QEvent.MouseButtonPress, QEvent.DragEnter, QEvent.Wheel):
            self._hide_tooltip()
            return False

        # 5. Key release - if require_ctrl is True, release of Ctrl immediately dismisses tooltip
        elif etype == QEvent.KeyRelease:
            if self.require_ctrl and event.key() in (Qt.Key_Control, Qt.Key_Meta):
                self._hide_tooltip()
            return False

        # 6. Key press - pressing Ctrl shows tooltip
        elif etype == QEvent.KeyPress:
            if event.key() in (Qt.Key_Control, Qt.Key_Meta):
                try:
                    if viewport and viewport.isVisible():
                        gpos = QCursor.pos()
                        vpos = viewport.mapFromGlobal(gpos)
                        if viewport.rect().contains(vpos):
                            item = lw.itemAt(vpos)
                            if item and item.flags() != Qt.NoItemFlags and item.text():
                                self._cancel_delay()
                                self._show_tooltip_for_item(item, gpos)
                except (RuntimeError, AttributeError):
                    pass
            return False

        # 7. Widget hide or window deactivate (ONLY if it is the list widget or its parent window)
        elif etype == QEvent.Hide and watched in (lw, viewport, lw.window()):
            self._hide_tooltip()
            return False
        elif etype == QEvent.WindowDeactivate and watched == lw.window():
            self._hide_tooltip()
            return False

        return False
