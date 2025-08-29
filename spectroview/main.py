# app/main.py
import sys
import os
import pandas as pd
import datetime
import logging

from pathlib import Path

from PySide6.QtWidgets import QApplication, QFileDialog
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QFileInfo, QCoreApplication, Qt
from PySide6.QtGui import QIcon

from spectroview import UI_FILE, LOGO_APPLI, USER_MANUAL
from spectroview.config.gui import resources

from spectroview.components.utils import show_alert
from spectroview.components.utils import CommonUtilities, FitModelManager
from spectroview.components.widget_mapview import MapViewWidget
from spectroview.config.ui_connector import UiConnector

from spectroview.main_tabs.maps import Maps
from spectroview.main_tabs.spectrums import Spectrums
from spectroview.main_tabs.graphs import Graphs
from spectroview.main_tabs.file_converter import FileConverter

from spectroview.config.about import About
from spectroview.config.app_settings import AppSettings
from spectroview.config.app_shortcuts import setup_shortcuts

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Main:
    def __init__(self):
        # Load the UI file
        loader = QUiLoader()
        ui_file = QFile(UI_FILE)
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file)
        ui_file.close()

        self.common = CommonUtilities()

        #### APP SETTINGS
        self.app_settings = AppSettings()
        self.app_settings.load()
        qsettings = self.app_settings.qsettings

        # Create subsystem instances
        self.visu = Graphs(qsettings, self.ui, self.common)
        self.spectrums = Spectrums(qsettings, self.ui, self.common, self.visu, self.app_settings)
        self.maps = Maps(qsettings, self.ui, self.spectrums, self.common, self.visu, self.app_settings)
        self.fitmodel_manager = FitModelManager(qsettings)
        self.mapview_widget = MapViewWidget(self, self.app_settings)
        self.convertfile = FileConverter(self.ui, qsettings)

        # Apply stored settings to UI
        self.app_settings.apply_to_ui(self.ui)
        
        # Centralize GUI wiring
        self.gui = UiConnector(self.app_settings, self.ui, self, self.maps, self.spectrums, self.visu)

        ### SETUP SHORTCUTS 
        setup_shortcuts(self)
        
    def open(self, file_paths=None):
        """
        Universal action to open all supported files of SPECTROview:
         - spectroscopic data which can be hyperspectral or spectra (CSV, TXT).
         - dataframes (Excel)
         - saved work of SPECTROview (.maps, .spectra, .graphs)
        """
        if file_paths is None:
            last_dir = self.app_settings.qsettings.value("last_directory", "/")
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            file_paths, _ = QFileDialog.getOpenFileNames(
                self.ui.tabWidget, "Open spectra file(s)", last_dir,
                "SPECTROview formats (*.csv *.txt *.spectra *.maps * "
                ".graphs)", options=options)

        if file_paths:
            last_dir = QFileInfo(file_paths[0]).absolutePath()
            self.app_settings.qsettings.setValue("last_directory", last_dir)

            spectra_files = []
            hyperspectral_files = []
            dataframes = []
            spectra_file = None
            maps_file = None
            graphs_file = None
            df = None

            for file_path in file_paths:
                file_path = Path(file_path)
                extension = file_path.suffix.lower()
                if extension == '.spectra':
                    spectra_file = str(file_path)
                elif extension == '.maps':
                    maps_file = str(file_path)
                elif extension == '.graphs':
                    graphs_file = str(file_path)

                elif extension == '.xlsx':
                    dataframes.append(str(file_path))

                elif extension == '.csv':
                    try:
                        df = pd.read_csv(file_path, delimiter=";", header=None, skiprows=3)
                    except Exception as e:
                        show_alert(f"Failed to read CSV {file_path}: {e}")
                        df = None
                elif extension == '.txt':
                    try:
                        df = pd.read_csv(file_path, delimiter="\t", header=None, skiprows=3)
                    except Exception as e:
                        show_alert(f"Failed to read TXT {file_path}: {e}")
                        df = None
                else:
                    show_alert(f"Unsupported file format: {extension}")
                    continue

                if df is not None:
                    if df.shape[1] == 2:
                        spectra_files.append(str(file_path))
                    elif df.shape[1] > 3:
                        hyperspectral_files.append(str(file_path))
                    else:
                        show_alert(f"Invalid number of columns in file: {file_path}")

            if spectra_files:
                self.spectrums.open_spectra(file_paths=spectra_files)
            if hyperspectral_files:
                self.maps.open_hyperspectra(file_paths=hyperspectral_files)
            if dataframes:
                self.ui.tabWidget.setCurrentWidget(self.ui.tab_graphs)
                self.visu.open_dfs(file_paths=dataframes)

            if spectra_file:
                self.ui.tabWidget.setCurrentWidget(self.ui.tab_spectra)
                self.spectrums.load_work(spectra_file)
            if maps_file:
                self.ui.tabWidget.setCurrentWidget(self.ui.tab_maps)
                self.maps.load_work(maps_file)
            if graphs_file:
                self.ui.tabWidget.setCurrentWidget(self.ui.tab_graphs)
                self.visu.load(graphs_file)

    def save(self):
        """Saves the current work depending on the active tab"""
        current_tab = self.ui.tabWidget.currentWidget()
        if current_tab == self.ui.tab_spectra:
            self.spectrums.save_work()
        elif current_tab == self.ui.tab_maps:
            self.maps.save_work()
        elif current_tab == self.ui.tab_graphs:
            self.visu.save()
        else:
            show_alert("No valid tab is selected for saving.")

    def clear_env(self):
        """Clear working enviroments"""
        current_tab = self.ui.tabWidget.currentWidget()
        if current_tab == self.ui.tab_graphs:
            self.visu.clear_env()
        elif current_tab == self.ui.tab_maps:
            self.maps.clear_env()
        elif current_tab == self.ui.tab_spectra:
            self.spectrums.clear_env()
        else:
            show_alert("No thing to clear.")
    def show_about(self):
        """Show about dialog """
        show_about = About(self.ui)
        show_about.exec_()

    def open_manual(self):
        """Open doc detail about query function of pandas dataframe"""
        title = "SPECTROview User Manual"
        self.common.view_markdown(self.ui, title, USER_MANUAL, 1200, 900, "/resources/doc/")

expiration_date = datetime.datetime(2050, 6, 1)

def launcher():
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    if datetime.datetime.now() > expiration_date:
        text = (
            "The current SPECTROview version has expired. Checkout the SPECTROview's "
            "Github page (cf. About) to update newest version."
        )
        app = QApplication(sys.argv)
        app.setWindowIcon(QIcon(LOGO_APPLI))
        window = Main()
        window.ui.centralwidget.setEnabled(False)
        app.setStyle("Fusion")
        window.ui.show()
        show_alert(text)
        sys.exit(app.exec())

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(LOGO_APPLI))
    window = Main()
    window.ui.centralwidget.setEnabled(True)
    app.setStyle("Fusion")
    window.ui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    launcher()