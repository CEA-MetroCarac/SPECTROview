# app/main.py
import sys
import os
import pandas as pd
import datetime
from pathlib import Path
from functools import partial
import logging

from PySide6.QtWidgets import QApplication, QFileDialog
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QFileInfo, QCoreApplication, Qt
from PySide6.QtGui import QIcon

from app.common import CommonUtilities, FitModelManager, MapViewWidget, ConvertFile
from app.common import show_alert

from app.ui import resources 
from app.maps import Maps
from app.spectrums import Spectrums
from app.visualisation import Visualization

from app.app_settings import AppSettings
from app.app_shortcuts import setup_shortcuts

# --- constants ---
DIRNAME = os.path.dirname(__file__)
UI_FILE = os.path.join(DIRNAME, "ui", "gui.ui")
ICON_APPLI = os.path.join(DIRNAME, "ui", "iconpack", "icon3.png")
USER_MANUAL = os.path.join(DIRNAME, "doc", "user_manual.md")
ABOUT = os.path.join(DIRNAME, "resources", "about.md")

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
        sync_settings = partial(self.app_settings.sync_app_settings, self.ui)

        def watch(widget):
            for sig_name in ("valueChanged", "stateChanged", "textChanged", "currentIndexChanged", "toggled"):
                sig = getattr(widget, sig_name, None)
                if sig:
                    sig.connect(sync_settings)

        qsettings = self.app_settings.qsettings

        # Theme selection based on stored mode
        if self.app_settings.mode == "light":
            self.toggle_light_mode()
        else:
            self.toggle_dark_mode()

        # Create subsystem instances
        self.visu = Visualization(qsettings, self.ui, self.common)
        self.spectrums = Spectrums(qsettings, self.ui, self.common, self.visu, self.app_settings)
        self.maps = Maps(qsettings, self.ui, self.spectrums, self.common, self.visu, self.app_settings)
        self.fitmodel_manager = FitModelManager(qsettings)
        self.mapview_widget = MapViewWidget(self, self.app_settings)
        self.convertfile = ConvertFile(self.ui, qsettings)

        # Apply stored settings to UI
        self.app_settings.apply_to_ui(self.ui)

        ### SETUP SHORTCUTS 
        setup_shortcuts(self)
        
        # TOOLBAR
        self.ui.actionOpen.triggered.connect(lambda: self.open())
        self.ui.actionSave.triggered.connect(self.save)
        self.ui.actionClear_env.triggered.connect(self.clear_env)

        self.ui.actionDarkMode.triggered.connect(self.toggle_dark_mode)
        self.ui.actionLightMode.triggered.connect(self.toggle_light_mode)
        self.ui.actionAbout.triggered.connect(self.show_about)
        self.ui.actionHelps.triggered.connect(self.open_manual)

        ## MAPS module:
        watch(self.ui.ncpu)
        watch(self.ui.cb_fit_negative)
        watch(self.ui.max_iteration)
        watch(self.ui.cbb_fit_methods)
        watch(self.ui.xtol)

        watch(self.ui.cb_attached)
        watch(self.ui.rbtn_linear)
        watch(self.ui.rbtn_polynomial)
        watch(self.ui.noise)
        watch(self.ui.degre)
        
        ## SPECTRA module:
        watch(self.ui.ncpu_2)
        watch(self.ui.cb_fit_negative_2)
        watch(self.ui.max_iteration_2)
        watch(self.ui.cbb_fit_methods_2)
        watch(self.ui.xtol_2)

        watch(self.ui.cb_attached_2)
        watch(self.ui.rbtn_linear_2)

        watch(self.ui.rbtn_polynomial_2)
        watch(self.ui.noise_2)
        watch(self.ui.degre_2)
        
        ## GRAPH module:
        watch(self.ui.cb_grid)


        self.ui.spectrums_listbox.files_dropped.connect(self.open)
        self.ui.spectra_listbox.files_dropped.connect(self.open)

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

    def toggle_dark_mode(self):
        self.ui.setPalette(self.common.dark_palette())
        self.app_settings.mode = "dark"
        self.app_settings.save()

    def toggle_light_mode(self):
        self.ui.setPalette(self.common.light_palette())
        self.app_settings.mode = "light"
        self.app_settings.save()

    def open_manual(self):
        """Open doc detail about query function of pandas dataframe"""
        title = "SPECTROview User Manual"
        self.common.view_markdown(self.ui, title, USER_MANUAL, 1200, 900, "doc/")

    def show_about(self):
        """Show about dialog """
        self.common.view_markdown(self.ui, "About", ABOUT, 650, 480, "resources/")


expiration_date = datetime.datetime(2050, 6, 1)


def launcher():
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    if datetime.datetime.now() > expiration_date:
        text = (
            "The current SPECTROview version has expired. Checkout the SPECTROview's "
            "Github page (cf. About) to update newest version."
        )
        app = QApplication(sys.argv)
        app.setWindowIcon(QIcon(ICON_APPLI))
        window = Main()
        window.ui.centralwidget.setEnabled(False)
        app.setStyle("Fusion")
        window.ui.show()
        show_alert(text)
        sys.exit(app.exec())

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(ICON_APPLI))
    window = Main()
    window.ui.centralwidget.setEnabled(True)
    app.setStyle("Fusion")
    window.ui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    launcher()
