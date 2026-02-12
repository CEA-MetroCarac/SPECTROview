# spectroview/viewmodel/utils.py
from PySide6.QtGui import QPalette, QColor, QIcon, QPixmap, QImage, QTextCursor
from PySide6.QtCore import Qt, Signal, QThread, QSize
from PySide6.QtWidgets import (
    QComboBox, QMessageBox, QApplication, QListWidgetItem,
    QDialog, QTextBrowser, QVBoxLayout
)    
import struct
import re
    
import base64
import json
import numpy as np
import re
import zlib
import zlib
from openpyxl.styles import PatternFill
import pandas as pd
from copy import deepcopy
import os
import datetime

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from fitspy.core.baseline import BaseLine
from fitspy.core.utils_mp import fit_mp


from spectroview import PALETTE, DEFAULT_COLORS

try:
    from pyqttoast import Toast, ToastPreset
    TOAST_AVAILABLE = True
except ImportError:
    TOAST_AVAILABLE = False


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
    import struct
    
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
            
            # Extract slit opening and centre using labels
            # Look for labeled patterns: "Opening" followed by value, "SlitBeamCentre" followed by value
            # The pattern in WXIS is: ...Opening...<value>µm...SlitBeamCentre...<value>µm...
            
            # Find all µm values with their preceding context
            import re
            # Find pattern: word/label, then µm value
            # Looking for: "20µm" after something like "Opening" and "1725µm" after "SlitBeamCentre" or similar
            
            # More robust: find all number-µm pairs
            all_um_values = re.findall(r'(\d+)[\xc2\xb5µ]m', text_data)
            
            # In WXIS, we know from analysis that the pattern is:
            # ...'0µm'...'20µm'...'1725µm'...
            # where 20 is slit opening and 1725 is slit centre
            # We need to skip the first "0" value and take the next two
            
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
            
            # Extract laser power from WXIS structured records
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
    
    # Parse WDF1 block to find Creation Date (SYSTEMTIME structure)
    # Search priority:
    # 1. File Info region (0xD0 - 0xF0)
    # 2. Entire Header (first 512 bytes)
    # 3. File modification time (fallback)
    
    found_date = None
    try:
        reader.file_obj.seek(0)
        header_bytes = reader.file_obj.read(1024) # Read enough to cover header
        
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

        # 1. Check exact gap at 0x88 (between Measurement Info and Spectral Info)
        # This is exactly 16 bytes (sizeof SYSTEMTIME) and most probable location
        d = check_systemtime(header_bytes, 0x88)
        if d:
            found_date = d
            
        # 2. If not found, check Gap 2 (0xA0 to 0xD0) 
        if not found_date:
            for i in range(0xA0, 0xD0, 2):
                 d = check_systemtime(header_bytes, i)
                 if d:
                     found_date = d
                     break

        # 3. Check File Info region (0xD0 to 0xF0) - Legacy check
        if not found_date:
            for i in range(0xD0, 0xF0, 2):
                d = check_systemtime(header_bytes, i)
                if d:
                    found_date = d
                    break
        
        # 4. Last resort: scan full header
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


def set_spectrum_item_color(item: QListWidgetItem, spectrum):
    """Set list item background color based on spectrum status."""
    if spectrum.baseline.is_subtracted:
        if not hasattr(spectrum.result_fit, 'success'):
            # Baseline subtracted but no fit result
            item.setBackground(QColor("gray"))
        elif spectrum.result_fit.success:
            # Fit succeeded
            item.setBackground(QColor("green"))
        else:
            # Fit failed
            item.setBackground(QColor("orange"))
    else:
        # Baseline not subtracted - transparent background
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
    if not TOAST_AVAILABLE:
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


class CustomizedPalette(QComboBox):
    """Custom QComboBox to show color palette previews along with their names."""
    def __init__(self, palette_list=None, parent=None, icon_size=(99, 12)):
        super().__init__(parent)
        self.icon_width, self.icon_height = icon_size
        self.setIconSize(QSize(*icon_size))
        self.setMinimumWidth(100)

        self.palette_list = palette_list or PALETTE
        self._populate_with_previews()

    def _populate_with_previews(self):
        self.clear()
        for cmap_name in self.palette_list:
            icon = QIcon(self._create_colormap_preview(cmap_name))
            self.addItem(icon, cmap_name)

    def _create_colormap_preview(self, cmap_name):
        """Generate a horizontal gradient preview image for the colormap."""
        width, height = self.icon_width, self.icon_height
        gradient = np.linspace(0, 1, 20).reshape(1, -1)

        fig = Figure(figsize=(width / 100, height / 100), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.imshow(gradient, aspect='auto', cmap=cm.get_cmap(cmap_name))
        ax.set_axis_off()
        canvas.draw()

        image = np.array(canvas.buffer_rgba())
        qimage = QImage(image.data, image.shape[1], image.shape[0],
                        QImage.Format_RGBA8888)
        return QPixmap.fromImage(qimage)

    def get_selected_palette(self):
        return self.currentText()

class ApplyFitModelThread(QThread):
    """Class to perform fitting in a separate Thread."""
    progress_changed = Signal(int, int, int, float)  # (current, total, percentage, elapsed_time)
    
    def __init__(self, spectrums, fit_model, fnames, ncpus=1):
        super().__init__()
        self.spectrums = spectrums
        self.fit_model = fit_model
        self.fnames = fnames
        self.ncpus = ncpus

    def run(self):
        """Execute fitting with progress tracking."""
        import time
        
        fit_model = deepcopy(self.fit_model)
        total = len(self.fnames)
        
        # Start timing
        start_time = time.time()
        
        # Emit initial progress
        self.progress_changed.emit(0, total, 0, 0.0)
         
        # Apply model (this will update progress through queue)
        from multiprocessing import Queue
        queue_incr = Queue()
        
        # Start monitoring progress in background
        from threading import Thread
        monitor_thread = Thread(target=self._monitor_progress, args=(queue_incr, total, start_time))
        monitor_thread.start()
        
        # Perform fitting - pass our queue to apply_model
        self.spectrums.apply_model(
            fit_model, 
            fnames=self.fnames,
            ncpus=self.ncpus, 
            show_progressbar=False,
            queue_incr=queue_incr
        )
        
        # Wait for progress monitor to finish
        monitor_thread.join()
        
        # Emit final completion with total elapsed time
        elapsed_time = time.time() - start_time
        self.progress_changed.emit(total, total, 100, elapsed_time)
    
    def _monitor_progress(self, queue_incr, total, start_time):
        """Monitor fitting progress from queue."""
        import time
        
        count = 0
        while count < total:
            try:
                # Wait for progress update from fitting process
                queue_incr.get(timeout=0.1)
                count += 1
                percentage = int((count / total) * 100)
                elapsed_time = time.time() - start_time
                self.progress_changed.emit(count, total, percentage, elapsed_time)
            except:
                # Timeout or queue empty, continue waiting
                continue

class FitThread(QThread):
    """Thread for fitting spectra with their own existing peak models (no model copying)."""
    progress_changed = Signal(int, int, int, float)  # (current, total, percentage, elapsed_time)
    
    def __init__(self, spectra):
        super().__init__()
        self.spectra = spectra

    def run(self):
        """Fit each spectrum with its own peak models, tracking progress."""
        import time
        
        total = len(self.spectra)
        start_time = time.time()
        
        # Emit initial progress
        self.progress_changed.emit(0, total, 0, 0.0)
        
        # Fit each spectrum sequentially
        for i, spectrum in enumerate(self.spectra, 1):
            if spectrum.peak_models:
                try:
                    # Check if any peak model is a decay model
                    # Decay models have built-in B (baseline) parameter
                    has_decay_model = any(
                        pm.name2 in ["DecaySingleExp", "DecayBiExp"] 
                        for pm in spectrum.peak_models
                    )
                    
                    # For decay models: Mark baseline as already subtracted
                    # This prevents preprocess() from subtracting it (which would 
                    # conflict with the decay model's B parameter)
                    if has_decay_model and not spectrum.baseline.is_subtracted:
                        spectrum.baseline.is_subtracted = True
                    
                    spectrum.preprocess()
                    
                    # For decay models: Skip reinit logic and noisy area detection
                    # Both expect ampli/x0 parameters which decay models don't have
                    if has_decay_model:
                        spectrum.fit(reinit_guess=False, coef_noise=0)
                    else:
                        spectrum.fit()
                except Exception:
                    # Continue fitting other spectra even if one fails
                    pass
            
            # Update progress
            percentage = int((i / total) * 100)
            elapsed_time = time.time() - start_time
            self.progress_changed.emit(i, total, percentage, elapsed_time)

def closest_index(array, value):
    return int(np.abs(array - value).argmin())





def baseline_to_dict(spectrum):
    dict_baseline = dict(vars(spectrum.baseline).items())
    return dict_baseline

def dict_to_baseline(dict_baseline, spectrums):
    for spectrum in spectrums:
        # Create a fresh BaselineModel instance
        new_baseline =  BaseLine()
        for key, value in dict_baseline.items():
            setattr(new_baseline, key, deepcopy(value))
        spectrum.baseline = new_baseline
        
def save_df_to_excel(save_path, df):
    """Saves a DataFrame to an Excel file with colored columns based on prefixes."""
    if not save_path:
        return False, "No save path provided."

    try:
        if df.empty:
            return False, "DataFrame is empty. Nothing to save."
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
            workbook = writer.book
            worksheet = writer.sheets['Results']
            palette = ['#bda16d', '#a27ba0', '#cb5b12', '#23993b', '#008281', '#147ce4']
            prefix_colors = {}

            # Apply colors based on column prefixes
            for col_idx, col_name in enumerate(df.columns, start=1):
                prefix = col_name.split("_")[0] if "_" in col_name else col_name
                if prefix not in prefix_colors:
                    color_index = len(prefix_colors) % len(palette)
                    prefix_colors[prefix] = PatternFill(start_color=palette[color_index][1:], 
                                                        end_color=palette[color_index][1:], 
                                                        fill_type='solid')
                for row in range(2, len(df) + 2):  # Apply the fill color to the entire column
                    worksheet.cell(row=row, column=col_idx).fill = prefix_colors[prefix]

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
        
        from io import BytesIO
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
        
def calc_area(model_name, params):
    """Calculate area under the peak based on model type and parameters."""
    # params: dict with parameter values like ampli, fwhm, alpha, etc.
    if model_name == "Gaussian":
        ampli = params.get('ampli')
        fwhm = params.get('fwhm')
        if ampli is not None and fwhm is not None:
            return ampli * fwhm * np.sqrt(np.pi / (4 * np.log(2)))
    elif model_name == "Lorentzian":
        ampli = params.get('ampli')
        fwhm = params.get('fwhm')
        if ampli is not None and fwhm is not None:
            return np.pi * ampli * fwhm / 2
    elif model_name == "PseudoVoigt":
        ampli = params.get('ampli')
        fwhm = params.get('fwhm')
        alpha = params.get('alpha', 0.5)
        if ampli is not None and fwhm is not None:
            gauss_area = ampli * fwhm * np.sqrt(np.pi / (4 * np.log(2)))
            lorentz_area = np.pi * ampli * fwhm / 2
            return alpha * gauss_area + (1 - alpha) * lorentz_area
    elif model_name == "GaussianAsym":
        # asymmetric Gaussian has left and right FWHM
        ampli = params.get('ampli')
        fwhm_l = params.get('fwhm_l')
        fwhm_r = params.get('fwhm_r')
        if ampli is not None and fwhm_l is not None and fwhm_r is not None:
            area_l = (ampli * fwhm_l * np.sqrt(np.pi / (4 * np.log(2))))/2
            area_r = (ampli * fwhm_r * np.sqrt(np.pi / (4 * np.log(2))))/2
            return area_l + area_r
    elif model_name == "LorentzianAsym":
        ampli = params.get('ampli')
        fwhm_l = params.get('fwhm_l')
        fwhm_r = params.get('fwhm_r')
        if ampli is not None and fwhm_l is not None and fwhm_r is not None:
            area_l = (np.pi * ampli * fwhm_l / 2)/2
            area_r = (np.pi * ampli * fwhm_r / 2)/2
            return area_l + area_r
    return None  # default if parameters missing
    
def dark_palette():
    """Dark palette tuned for SPECTROview UI"""

    p = QPalette()

    # ---------- Base surfaces ----------
    p.setColor(QPalette.Window, QColor(53, 53, 53))          # main background
    p.setColor(QPalette.Base, QColor(42, 42, 42))            # lists, tables, editors
    p.setColor(QPalette.AlternateBase, QColor(48, 48, 48))   # alternating rows

    # ---------- Text ----------
    p.setColor(QPalette.WindowText, Qt.white)
    p.setColor(QPalette.Text, Qt.white)
    p.setColor(QPalette.ButtonText, Qt.white)
    p.setColor(QPalette.PlaceholderText, QColor(140, 140, 140))

    # ---------- Buttons / controls ----------
    p.setColor(QPalette.Button, QColor(64, 64, 64))
    p.setColor(QPalette.Light, QColor(90, 90, 90))
    p.setColor(QPalette.Mid, QColor(72, 72, 72))
    p.setColor(QPalette.Dark, QColor(40, 40, 40))
    p.setColor(QPalette.Shadow, QColor(20, 20, 20))

    # ---------- Tooltips ----------
    p.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    p.setColor(QPalette.ToolTipText, Qt.black)

    # ---------- Highlights / accent ----------
    accent = QColor(42, 130, 218)  # Qt blue (matches screenshot)
    p.setColor(QPalette.Highlight, accent)
    p.setColor(QPalette.HighlightedText, Qt.white)
    p.setColor(QPalette.Link, accent)

    # ---------- Disabled ----------
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(130, 130, 130))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(130, 130, 130))
    p.setColor(QPalette.Disabled, QPalette.WindowText, QColor(130, 130, 130))

    return p

def light_palette():
    """Light palette with soft blue accent"""

    p = QPalette()

    # ---- Base colors ----
    p.setColor(QPalette.Window, QColor(245, 246, 248))        # main background
    p.setColor(QPalette.Base, QColor(255, 255, 255))          # inputs, tables
    p.setColor(QPalette.AlternateBase, QColor(238, 240, 243)) # alternate rows

    # ---- Text ----
    p.setColor(QPalette.WindowText, QColor(30, 30, 30))
    p.setColor(QPalette.Text, QColor(30, 30, 30))
    p.setColor(QPalette.ButtonText, QColor(30, 30, 30))
    p.setColor(QPalette.PlaceholderText, QColor(150, 150, 150))

    # ---- Buttons ----
    p.setColor(QPalette.Button, QColor(235, 236, 239))
    p.setColor(QPalette.Light, QColor(255, 255, 255))
    p.setColor(QPalette.Midlight, QColor(220, 220, 220))
    p.setColor(QPalette.Mid, QColor(200, 200, 200))
    p.setColor(QPalette.Dark, QColor(160, 160, 160))

    # ---- Blue accent ----
    accent = QColor(64, 156, 255)  # soft modern blue
    accent_hover = QColor(90, 170, 255)

    p.setColor(QPalette.Highlight, accent)
    p.setColor(QPalette.HighlightedText, Qt.white)
    p.setColor(QPalette.Link, accent)

    # ---- Tooltips ----
    p.setColor(QPalette.ToolTipBase, QColor(255, 255, 240))
    p.setColor(QPalette.ToolTipText, QColor(20, 20, 20))

    # ---- Disabled state ----
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(160, 160, 160))
    p.setColor(QPalette.Disabled, QPalette.WindowText, QColor(160, 160, 160))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(160, 160, 160))

    return p


