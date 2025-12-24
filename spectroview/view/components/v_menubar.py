#view/toolbar.py
import os

import numpy as np
import matplotlib.pyplot as plt

from PySide6.QtWidgets import QToolBar, QWidget, QSizePolicy
from PySide6.QtCore import Qt, QSize, QPoint, Signal
from PySide6.QtGui import  QIcon, QAction, Qt, QCursor

from spectroview import ICON_DIR


class VMenuBar(QToolBar):
    open_requested = Signal()
    save_requested = Signal()
    clear_requested = Signal()
    convert_requested = Signal()
    settings_requested = Signal()
    theme_requested = Signal()
    manual_requested = Signal()
    about_requested = Signal()
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setIconSize(QSize(30, 30))
        
        self.actionOpen = self.addAction(QIcon(os.path.join(ICON_DIR, "open.png")), "Open")
        self.actionOpen.triggered.connect(self.open_requested.emit)

        self.actionSave = self.addAction(QIcon(os.path.join(ICON_DIR, "save.png")), "Save")
        self.actionSave.triggered.connect(self.save_requested.emit)

        self.actionClearWS= self.addAction(QIcon(os.path.join(ICON_DIR, "clear.png")), "Clear current workspace")
        self.actionClearWS.triggered.connect(self.clear_requested.emit)

        self.addSeparator()
        self.actionConvert= self.addAction(QIcon(os.path.join(ICON_DIR, "FileConvert.png")), "Convert 2Dmap format")
        self.actionConvert.triggered.connect(self.convert_requested.emit)

        self.addSeparator()
        self.actionSettings= self.addAction(QIcon(os.path.join(ICON_DIR, "settings.png")), "Settings")
        self.actionSettings.triggered.connect(self.settings_requested.emit)
        
        # ----- Expanding spacer -----
        spacer = QWidget(self)
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.addWidget(spacer)
        
        self.actionTheme= self.addAction(QIcon(os.path.join(ICON_DIR, "dark-light.png")), "Toggle Dark/Light Theme")
        self.actionTheme.triggered.connect(self.theme_requested.emit)

        self.actionManual= self.addAction(QIcon(os.path.join(ICON_DIR, "manual.png")), "Open User Manual")
        self.actionManual.triggered.connect(self.manual_requested.emit)

        self.actionAbout= self.addAction(QIcon(os.path.join(ICON_DIR, "about.png")), "About SPECTROview")
        self.actionAbout.triggered.connect(self.about_requested.emit)
        self.addSeparator()
        
    