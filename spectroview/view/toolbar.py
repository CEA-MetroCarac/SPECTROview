#view/toolbar.py
import os

import numpy as np
import matplotlib.pyplot as plt

from PySide6.QtWidgets import QToolBar, QWidget, QSizePolicy
from PySide6.QtCore import Qt, QSize, QPoint
from PySide6.QtGui import  QIcon, QAction, Qt, QCursor

from spectroview import ICON_DIR


class MenuBar(QToolBar):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("SPECTROview Toolbar")
        self.setMovable(False)
        self.setIconSize(QSize(24, 24))
        
        # Example: Add a sample action with an icon
        self.actionOpen = self.addAction(QIcon(os.path.join(ICON_DIR, "open.png")), "Open")
        self.actionSave = self.addAction(QIcon(os.path.join(ICON_DIR, "save.png")), "Save")
        self.actionClearWS= self.addAction(QIcon(os.path.join(ICON_DIR, "clear.png")), "Clear current workspace")
        self.addSeparator()
        self.actionConvert= self.addAction(QIcon(os.path.join(ICON_DIR, "FileConvert.png")), "Convert 2Dmap format")
        self.addSeparator()
        self.actionSettings= self.addAction(QIcon(os.path.join(ICON_DIR, "settings.png")), "Settings")
        
        # ----- Expanding spacer -----
        spacer = QWidget(self)
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.addWidget(spacer)
        
        self.actionTheme= self.addAction(QIcon(os.path.join(ICON_DIR, "dark-light.png")), "Toggle Dark/Light Theme")
        self.actionManual= self.addAction(QIcon(os.path.join(ICON_DIR, "manual.png")), "Open User Manual")
        self.actionAbout= self.addAction(QIcon(os.path.join(ICON_DIR, "about.png")), "About SPECTROview")
        self.addSeparator()
        