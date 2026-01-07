# main.py
import sys
import os
import webbrowser
from pathlib import Path

import pandas as pd

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget, QFileDialog, QMessageBox
from PySide6.QtCore import Qt, QSettings, QFileInfo
from PySide6.QtGui import QIcon

from spectroview.model.m_file_converter import MFileConverter

from spectroview.viewmodel.vm_settings import VMSettings
from spectroview.view.components.v_settings import VSettingsDialog
from spectroview.view.components.v_about import VAboutDialog

from spectroview.view.components.v_menubar import VMenuBar
from spectroview.view.v_workspace_spectra import VWorkspaceSpectra
from spectroview.view.v_workspace_maps import WorkspaceMaps
from spectroview.view.v_workspace_graphs import VWorkspaceGraphs

from spectroview.viewmodel.utils import dark_palette, light_palette

from spectroview import LOGO_APPLI, USER_MANUAL_PDF

class Main(QMainWindow):
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
        self.setGeometry(100, 100, 1400, 930)
        self.setWindowIcon(QIcon(LOGO_APPLI))

        # Central widget
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        # Main Tab Widget
        self.tabWidget = QTabWidget(central)

        self.v_spectra_workspace = VWorkspaceSpectra()
        self.v_graphs_workspace = VWorkspaceGraphs()
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
        self.menu_bar.settings_requested.connect(self._open_settings)
        self.menu_bar.convert_requested.connect(self.file_converter)

        self.menu_bar.about_requested.connect(self.about)
        self.menu_bar.manual_requested.connect(self.manual) 
        self.menu_bar.theme_requested.connect(self.toggle_theme)
        

    def open_files(self):
        """Universal file opener supporting all SPECTROview formats."""
        last_dir = self.settings.value("last_directory", "/")
        paths, _ = QFileDialog.getOpenFileNames(
            None,
            "Open file(s)",
            last_dir,
            "SPECTROview formats (*.csv *.txt *.spectra *.maps *.graphs *.xlsx)"
        )
        
        if not paths:
            return
        
        # Save last directory
        last_dir = QFileInfo(paths[0]).absolutePath()
        self.settings.setValue("last_directory", last_dir)
        
        # Categorize files by type
        spectra_files = []
        hyperspectral_files = []
        dataframes = []
        spectra_work_file = None
        maps_work_file = None
        graphs_work_file = None
        
        for file_path in paths:
            path = Path(file_path)
            ext = path.suffix.lower()
            
            # Saved workspace files
            if ext == '.spectra':
                spectra_work_file = str(path)
            elif ext == '.maps':
                maps_work_file = str(path)
            elif ext == '.graphs':
                graphs_work_file = str(path)
            elif ext == '.xlsx':
                dataframes.append(str(path))
            elif ext in ['.csv', '.txt']:
                # Detect if it's spectrum or hyperspectral data
                try:
                    delimiter = ";" if ext == '.csv' else "\t"
                    df = pd.read_csv(path, delimiter=delimiter, header=None, skiprows=3)
                    
                    if df.shape[1] == 2:
                        spectra_files.append(str(path))
                    elif df.shape[1] > 3:
                        hyperspectral_files.append(str(path))
                    else:
                        QMessageBox.warning(self, "Invalid File", f"Invalid number of columns in {path.name}")
                except Exception as e:
                    QMessageBox.warning(self, "Read Error", f"Failed to read {path.name}: {e}")
            else:
                QMessageBox.warning(self, "Unsupported Format", f"Unsupported file format: {ext}")
        
        # Load files into appropriate workspaces
        if spectra_files:
            self.v_spectra_workspace.vm.load_files(spectra_files)
            self.tabWidget.setCurrentWidget(self.v_spectra_workspace)
        
        if hyperspectral_files:
            # TODO: Implement when maps workspace is converted to MVVM
            # self.v_maps_workspace.vm.load_files(hyperspectral_files)
            self.tabWidget.setCurrentWidget(self.v_maps_workspace)
        
        if dataframes:
            self.v_graphs_workspace.vm.load_dataframes(dataframes)
            self.tabWidget.setCurrentWidget(self.v_graphs_workspace)
        
        # Load saved work files
        if spectra_work_file:
            self.v_spectra_workspace.load_work(spectra_work_file)
            self.tabWidget.setCurrentWidget(self.v_spectra_workspace)
        
        if maps_work_file:
            # TODO: Implement when maps workspace is converted to MVVM
            # self.v_maps_workspace.load_work(maps_work_file)
            self.tabWidget.setCurrentWidget(self.v_maps_workspace)
        
        if graphs_work_file:
            self.v_graphs_workspace.load_workspace(graphs_work_file)
            self.tabWidget.setCurrentWidget(self.v_graphs_workspace)

    def save(self):
        """Save current workspace based on active tab."""
        current_tab = self.tabWidget.currentWidget()
        
        if current_tab == self.v_spectra_workspace:
            self.v_spectra_workspace.save_work()
        elif current_tab == self.v_maps_workspace:
            # TODO: Implement when maps workspace is converted to MVVM
            # self.v_maps_workspace.save_work()
            QMessageBox.information(self, "Not Implemented", "Maps workspace save not yet implemented in MVVM.")
        elif current_tab == self.v_graphs_workspace:
            self.v_graphs_workspace.save_workspace()
        else:
            QMessageBox.warning(self, "No Tab Selected", "No valid tab is selected for saving.")

    def clear_workspace(self):
        """Clear current workspace based on active tab."""
        current_tab = self.tabWidget.currentWidget()
        
        if current_tab == self.v_spectra_workspace:
            reply = QMessageBox.question(
                self,
                "Clear Workspace",
                "Are you sure you want to clear all spectra?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.v_spectra_workspace.clear_workspace()
        elif current_tab == self.v_maps_workspace:
            # TODO: Implement when maps workspace is converted to MVVM
            # self.v_maps_workspace.clear_workspace()
            QMessageBox.information(self, "Not Implemented", "Maps workspace clear not yet implemented in MVVM.")
        elif current_tab == self.v_graphs_workspace:
            reply = QMessageBox.question(
                self,
                "Clear Workspace",
                "Are you sure you want to clear all graphs and dataframes?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.v_graphs_workspace.clear_workspace()
        else:
            QMessageBox.warning(self, "No Tab Selected", "Nothing to clear.")

    def _open_settings(self):
        """   Open settings dialog. """
        vm = VMSettings()
        dlg = VSettingsDialog(vm, self)
        dlg.exec()

    def file_converter(self):
        """Open file converter dialog for hyperspectral data."""
        dlg = MFileConverter(self.settings, self)
        dlg.exec()

    def about(self):
        """Show About dialog."""
        dlg = VAboutDialog(self)
        dlg.exec()

    def manual(self):
        """Open user manual PDF in default browser."""
        if os.path.exists(USER_MANUAL_PDF):
            webbrowser.open(USER_MANUAL_PDF)
        else:
            QMessageBox.warning(self, "Manual Not Found", f"User manual not found at:\n{USER_MANUAL_PDF}")

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

    window = Main()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launcher()
