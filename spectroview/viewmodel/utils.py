# spectroview/view/v_utils.py
from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt

import numpy as np

from copy import deepcopy

from fitspy.core.baseline import BaseLine
from fitspy.core.utils_mp import fit_mp
    
from PySide6.QtCore import Signal, QThread, Qt
from PySide6.QtGui import QPalette, QColor, Qt

class FitThread(QThread):
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


