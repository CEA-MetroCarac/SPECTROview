# main.py
import sys
import os
from pathlib import Path


from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget, QFileDialog
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QIcon

from spectroview.view.v_menubar import VMenuBar
from spectroview.view.workspace_spectra import WorkspaceSpectra
from spectroview.view.workspace_maps import WorkspaceMaps
from spectroview.view.workspace_graphs import WorkspaceGraphs

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

        self.v_spectra_workspace = WorkspaceSpectra()
        self.v_graphs_workspace = WorkspaceGraphs()
        self.v_maps_workspace = WorkspaceMaps()

        self.tabWidget.addTab(self.v_spectra_workspace, "Spectra")
        self.tabWidget.addTab(self.v_maps_workspace, "Maps")
        self.tabWidget.addTab(self.v_graphs_workspace, "Graphs")

        layout.addWidget(self.tabWidget)
        self.setCentralWidget(central)

        # Toolbar
        self.menu_bar = VMenuBar()
        self.addToolBar(Qt.TopToolBarArea, self.menu_bar)
        
    def setup_connections(self):
        self.menu_bar.open_requested.connect(self.open_files)
        self.menu_bar.save_requested.connect(self.save)
        self.menu_bar.clear_requested.connect(self.clear_workspace)
        self.menu_bar.convert_requested.connect(self.file_converter)

        self.menu_bar.about_requested.connect(self.about)
        self.menu_bar.manual_requested.connect(self.manual) 
        self.menu_bar.theme_requested.connect(self.toggle_theme)
        

    def open_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            None,
            "Open spectra",
            "",
            "Data (*.txt *.csv)"
        )
        if paths:
            self.v_spectra_workspace.vm.load_files(paths)   # Load files into spectra workspace

    def save(self):
        pass  # Implement save functionality as needed

    def clear_workspace(self):
        pass  # Implement clear functionality as needed

    def file_converter(self):
        pass  # Implement file conversion functionality as needed

    def about(self):
        pass  # Implement about dialog as needed

    def manual(self):
        pass  # Implement manual/help dialog as needed

    def toggle_theme(self, theme=None):
        app = QApplication.instance()
        if theme is None:
            theme = "light" if self.settings.value("theme") == "dark" else "dark"
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
