#view/main_view.py
import os
from pathlib import Path


from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from spectroview.view.v_toolbar import MenuBar
from spectroview.view.v_spectra_workspace import SpectraWorkspace
from spectroview.view.v_maps_workspace import MapsWorkspace
from spectroview.view.v_graphs_workspace import GraphsWorkspace


from spectroview import LOGO_APPLI


class MainView(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        self.setWindowTitle(
            "SPECTROview (Tool for Spectroscopic Data Processing and Visualization)"
        )
        self.setGeometry(100, 100, 1500, 900)
        self.setWindowIcon(QIcon(LOGO_APPLI))

        # Central widget
        central = QWidget(self)
        layout = QVBoxLayout(central)

        # Main Tab Widget
        self.tabWidget = QTabWidget(central)

        self.spectra_view = SpectraWorkspace()
        self.graphs_view = GraphsWorkspace()
        self.maps_view = MapsWorkspace()

        self.tabWidget.addTab(self.spectra_view, "Spectra")
        self.tabWidget.addTab(self.maps_view, "Maps")
        self.tabWidget.addTab(self.graphs_view, "Graphs")

        layout.addWidget(self.tabWidget)
        self.setCentralWidget(central)

        # Toolbar
        self.menu_bar = MenuBar()
        self.addToolBar(Qt.TopToolBarArea, self.menu_bar)
        
    def setup_connections(self):
        self.menu_bar.open_requested.connect(self.spectra_view.vm.open_dialog)

