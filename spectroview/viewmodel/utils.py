# spectroview/view/v_utils.py
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
import os
import base64
import markdown
import zlib
import numpy as np
import pandas as pd
import platform

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from io import BytesIO
from PIL import Image
from threading import Thread
from multiprocessing import Queue
from copy import deepcopy
from openpyxl.styles import PatternFill

from fitspy.core.spectrum import Spectrum as FitspySpectrum
from fitspy.core.spectra import Spectra as FitspySpectra
from fitspy.core.baseline import BaseLine
from fitspy.core.utils_mp import fit_mp

from spectroview import PALETTE, DEFAULT_COLORS
from spectroview.modules.df_table import DataframeTable

from PySide6.QtWidgets import QDialog, QTableWidgetItem, QVBoxLayout,  QTextBrowser, \
    QComboBox, QListWidgetItem, QMessageBox, QDialog, QVBoxLayout, QListWidget, QAbstractItemView
    
from PySide6.QtCore import Signal, QThread, Qt, QSize
from PySide6.QtGui import QPalette, QColor, QTextCursor, QIcon, Qt, QPixmap, QImage

if platform.system() == 'Darwin':
    import AppKit 
if platform.system() == 'Windows':
    import win32clipboard


class FitThread(QThread):
    """ Class to perform fitting in a separate Thread """
    progress_changed = Signal(int)
    def __init__(self, spectrums, fit_model, fnames, ncpus=1):
        super().__init__()
        self.spectrums = spectrums
        self.fit_model = fit_model
        self.fnames = fnames
        self.ncpus = ncpus

    def run(self):
        fit_model = deepcopy(self.fit_model)
        self.spectrums.apply_model(fit_model, fnames=self.fnames,
                                   ncpus=self.ncpus, show_progressbar=False)

        self.progress_changed.emit(100)

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


