# main.py
import sys
import os
from pathlib import Path


from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QIcon

from spectroview.view.v_menubar import VMenuBar
from spectroview.view.v_spectra_workspace import VSpectraWorkspace
from spectroview.view.v_maps_workspace import VMapsWorkspace
from spectroview.view.v_graphs_workspace import VGraphsWorkspace
from spectroview.view.components.v_utils import dark_palette, light_palette

from spectroview import LOGO_APPLI

class VMain(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("CEA-Leti", "SPECTROview")

        self.init_ui()
        self.toggle_theme(self.settings.value("theme"))
        self.setup_connections()


    def init_ui(self):
        self.setWindowTitle(
            "SPECTROview (Tool for Spectroscopic Data Processing and Visualization)"
        )
        self.setGeometry(100, 100, 1400, 900)
        self.setWindowIcon(QIcon(LOGO_APPLI))

        # Central widget
        central = QWidget(self)
        layout = QVBoxLayout(central)

        # Main Tab Widget
        self.tabWidget = QTabWidget(central)

        self.v_spectra_workspace = VSpectraWorkspace()
        self.v_graphs_workspace = VGraphsWorkspace()
        self.v_maps_workspace = VMapsWorkspace()

        self.tabWidget.addTab(self.v_spectra_workspace, "Spectra")
        self.tabWidget.addTab(self.v_maps_workspace, "Maps")
        self.tabWidget.addTab(self.v_graphs_workspace, "Graphs")

        layout.addWidget(self.tabWidget)
        self.setCentralWidget(central)

        # Toolbar
        self.menu_bar = VMenuBar()
        self.addToolBar(Qt.TopToolBarArea, self.menu_bar)
        
    def setup_connections(self):
        self.menu_bar.open_requested.connect(self.v_spectra_workspace.vm.file_open_dialog)
        self.menu_bar.theme_requested.connect(self.toggle_theme)


    def toggle_theme(self, theme=None):
        app = QApplication.instance()
        if theme is None:
            theme = "light" if self.settings.value("theme", "dark") == "dark" else "dark"
        if theme == "dark":
            app.setPalette(dark_palette())
            self.settings.setValue("theme", "dark")
        else:
            app.setPalette(light_palette())
            self.settings.setValue("theme", "light")

def launcher():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(LOGO_APPLI))
    app.setStyle("Fusion")

    window = VMain()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launcher()
