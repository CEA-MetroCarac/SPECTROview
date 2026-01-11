# spectroview/viewmodel/utils.py
from PySide6.QtGui import QPalette, QColor, QIcon, QPixmap, QImage
from PySide6.QtCore import Qt, Signal, QThread, QSize
from PySide6.QtWidgets import QComboBox, QMessageBox, QApplication, QListWidgetItem    
    
import base64
import numpy as np
import zlib
from openpyxl.styles import PatternFill
import pandas as pd
from copy import deepcopy

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from fitspy.core.baseline import BaseLine


from spectroview import PALETTE, DEFAULT_COLORS

try:
    from pyqttoast import Toast, ToastPreset
    TOAST_AVAILABLE = True
except ImportError:
    TOAST_AVAILABLE = False


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
            item.setBackground(QColor("red"))
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
        
        # Perform fitting
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
                    spectrum.preprocess()
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


def compress(array):
    """Compress and encode a numpy array to a base64 string."""
    compressed = zlib.compress(array.tobytes())
    encoded = base64.b64encode(compressed).decode('utf-8')
    return encoded

def decompress(data, dtype):
    """Decode and decompress a base64 string to a numpy array."""
    decoded = base64.b64decode(data.encode('utf-8'))
    decompressed = zlib.decompress(decoded)
    return np.frombuffer(decompressed, dtype=dtype)


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
        
def spectrum_to_dict(spectrums, is_map=False):
    """Custom 'save' method to save 'Spectrum' object in a dictionary"""
    spectrums_data = spectrums.save()
    # Iterate over the saved spectrums data and update x0 and y0
    for i, spectrum in enumerate(spectrums):
        spectrum_dict = {}
        
        # Save x0 and y0 only if it's not a 2DMAP
        if not is_map:
            spectrum_dict.update({
                "x0": compress(spectrum.x0),
                "y0": compress(spectrum.y0)
            })
        
        # Update the spectrums_data with the new dictionary values
        spectrums_data[i].update(spectrum_dict)
    return spectrums_data

def dict_to_spectrum(spectrum, spectrum_data, is_map=True, maps=None):
    """Set attributes of Spectrum object from JSON dict"""
    spectrum.set_attributes(spectrum_data)
    
    if is_map: 
        if maps is None:
            raise ValueError("maps must be provided when map=True.")
        
        # Retrieve map_name and coord from spectrum.fname
        fname = spectrum.fname
        map_name, coord_str = fname.rsplit('_', 1)
        coord_str = coord_str.strip('()')  # Remove parentheses
        coord = tuple(map(float, coord_str.split(',')))  # Convert to float tuple
        
        # Retrieve x0 and y0 from the corresponding map_df using map_name and coord
        if map_name in maps:
            map_df = maps[map_name]
            map_df = map_df.iloc[:, :-1]  # Drop the last column from map_df (NaN)
            coord_x, coord_y = coord

            row = map_df[(map_df['X'] == coord_x) & (map_df['Y'] == coord_y)]
            
            if not row.empty:
                x0 = map_df.columns[2:].astype(float).values  # retreive original x0
                spectrum.x0 = x0 + spectrum.xcorrection_value # apply xcorrection_value
                spectrum.y0 = row.iloc[0, 2:].values  
            else:
                spectrum.x0 = None
                spectrum.y0 = None
        else:
            spectrum.x0 = None
            spectrum.y0 = None
    else:
        # Handle single spectrum case
        if 'x0' in spectrum_data:
            spectrum.x0 = decompress(spectrum_data['x0'], dtype=np.float64)
        if 'y0' in spectrum_data:
            spectrum.y0 = decompress(spectrum_data['y0'], dtype=np.float64)

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
    """Copy matplotlib figure canvas to clipboard with optional resizing"""
    if not canvas:
        QMessageBox.critical(None, "Error", "No plot to copy.")
        return
    
    figure = canvas.figure
    original_size = figure.get_size_inches()
    original_dpi = figure.dpi
    
    try:
        # Temporarily set high DPI and custom size for clipboard export
        if size_ratio:
            figure.set_size_inches(size_ratio, forward=True)
        figure.set_dpi(300)  # High resolution for clipboard
        canvas.draw()
        
        # Get rendered buffer and convert to QPixmap
        buffer = canvas.buffer_rgba()
        width, height = canvas.get_width_height()
        qimage = QImage(buffer, width, height, QImage.Format_RGBA8888)
        
        # Copy to clipboard
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


