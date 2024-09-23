"""
Main module for SPECTROview application.

This module initializes the main window of the SPECTROview application,
loads necessary UI components, connects GUI elements to backend methods,
and manages application settings.

"""

import sys
import os
import pandas as pd
import datetime
from pathlib import Path

from PySide6.QtWidgets import QApplication, QFileDialog
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QSettings, QFileInfo, QCoreApplication, Qt
from PySide6.QtGui import QIcon, QKeySequence, QShortcut

from .common import CommonUtilities, FitModelManager
from .common import PEAK_MODELS
from .common import show_alert
    
from .ui import resources 
from .maps import Maps
from .spectrums import Spectrums
from .visualisation import Visualization

DIRNAME = os.path.dirname(__file__)
UI_FILE = os.path.join(DIRNAME, "ui", "gui.ui")
ICON_APPLI = os.path.join(DIRNAME, "ui", "iconpack", "icon3.png")
USER_MANUAL = os.path.join(DIRNAME, "doc", "user_manual.md")
ABOUT = os.path.join(DIRNAME, "resources", "about.md")


class Main:
    def __init__(self):
        # Load the UI file
        loader = QUiLoader()
        ui_file = QFile(UI_FILE)
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file)
        ui_file.close()

        self.setup_shortcuts()
        self.common = CommonUtilities()

        # Initialize QSettings
        QSettings.setDefaultFormat(QSettings.IniFormat)
        self.settings = QSettings("CEA-Leti", "SPECTROview")

        if self.settings.value("mode") == "light":
            self.toggle_light_mode()
        else:
            self.toggle_dark_mode()

        # Create an instance of Dataframe and pass the self.ui object
        self.visu = Visualization(self.settings, self.ui, self.common)
        self.spectrums = Spectrums(self.settings, self.ui, self.common,
                                   self.visu)
        self.maps = Maps(self.settings, self.ui, self.spectrums, self.common,
                         self.visu)
        self.fitmodel_manager = FitModelManager(self.settings)

        # TOOLBAR
        self.ui.actionOpen.triggered.connect(lambda: self.open())
        self.ui.actionSave.triggered.connect(self.save)
        self.ui.actionClear_env.triggered.connect(self.clear_env)

        self.ui.actionDarkMode.triggered.connect(self.toggle_dark_mode)
        self.ui.actionLightMode.triggered.connect(self.toggle_light_mode)
        self.ui.actionAbout.triggered.connect(self.show_about)
        self.ui.actionHelps.triggered.connect(self.open_manual)
        
        # Save GUI states to settings
        self.ui.ncpus.valueChanged.connect(self.save_settings)
        ## Maps module:
        self.load_settings()
        self.ui.cb_fit_negative.stateChanged.connect(self.save_settings)
        self.ui.max_iteration.valueChanged.connect(self.save_settings)
        self.ui.cbb_fit_methods.currentIndexChanged.connect(
            self.save_settings)
        self.ui.xtol.textChanged.connect(self.save_settings)
        self.ui.cb_attached.stateChanged.connect(self.save_settings)
        
        self.ui.noise.valueChanged.connect(self.save_settings)
        self.ui.rbtn_linear.toggled.connect(self.save_settings)
        self.ui.degre.valueChanged.connect(self.save_settings)
        
        ## Spectra module:
        self.ui.cb_fit_negative_2.stateChanged.connect(self.save_settings)
        self.ui.max_iteration_2.valueChanged.connect(self.save_settings)
        self.ui.cbb_fit_methods_2.currentIndexChanged.connect(
            self.save_settings)
        self.ui.xtol_2.textChanged.connect(self.save_settings)
        self.ui.cb_attached_2.stateChanged.connect(self.save_settings)
        
        self.ui.cb_grid.stateChanged.connect(self.save_settings)

        ########################################################
        ############## GUI for Maps Processing tab #############
        ########################################################
        self.ui.btn_remove_wafer.clicked.connect(self.maps.remove_map)

        self.ui.btn_sel_all.clicked.connect(self.maps.select_all_spectra)
        self.ui.btn_sel_verti.clicked.connect(self.maps.select_verti)
        self.ui.btn_sel_horiz.clicked.connect(self.maps.select_horiz)
        self.ui.btn_sel_q1.clicked.connect(self.maps.select_Q1)
        self.ui.btn_sel_q2.clicked.connect(self.maps.select_Q2)
        self.ui.btn_sel_q3.clicked.connect(self.maps.select_Q3)
        self.ui.btn_sel_q4.clicked.connect(self.maps.select_Q4)

        self.ui.btn_load_model.clicked.connect(self.maps.load_fit_model)
        self.ui.btn_apply_model.clicked.connect(
            self.maps.apply_model_fnc_handler)
        self.ui.btn_init.clicked.connect(self.maps.reinit_fnc_handler)
        self.ui.btn_collect_results.clicked.connect(self.maps.collect_results)
        self.ui.btn_view_df_2.clicked.connect(self.maps.view_fit_results_df)
        self.ui.btn_show_stats.clicked.connect(self.maps.view_stats)
        self.ui.btn_save_fit_results.clicked.connect(
            self.maps.save_fit_results)
        self.ui.btn_view_wafer.clicked.connect(self.maps.view_map_data)

        self.ui.btn_open_fitspy.clicked.connect(self.maps.fitspy_launcher)
        self.ui.btn_cosmis_ray.clicked.connect(self.maps.cosmis_ray_detection)
        
        self.ui.btn_split_fname_2.clicked.connect(self.maps.split_fname)
        self.ui.btn_add_col_2.clicked.connect(self.maps.add_column)

        self.ui.range_max.returnPressed.connect(self.maps.set_x_range)
        self.ui.range_min.returnPressed.connect(self.maps.set_x_range)
        self.ui.range_apply.clicked.connect(self.maps.set_x_range_handler)
        
        self.ui.btn_fit.clicked.connect(self.maps.fit_fnc_handler)
        self.ui.save_model.clicked.connect(self.maps.save_fit_model)
        self.ui.clear_peaks.clicked.connect(self.maps.clear_peaks_handler)
        self.ui.btn_copy_fit_model.clicked.connect(self.maps.copy_fit_model)
        self.ui.btn_paste_fit_model.clicked.connect(
            self.maps.paste_fit_model_fnc_handler)
        self.ui.cbb_fit_models.addItems(PEAK_MODELS)

        self.ui.btn_undo_baseline.clicked.connect(self.maps.set_x_range_handler)

        self.ui.btn_send_to_compare.clicked.connect(
            self.maps.send_spectrum_to_compare)
        self.ui.btn_default_folder_model.clicked.connect(
            self.maps.set_default_model_folder)

        ########################################################
        ############## GUI for Spectrums Processing tab #############
        ########################################################
        self.ui.cbb_fit_models_2.addItems(PEAK_MODELS)
        self.ui.range_apply_2.clicked.connect(
            self.spectrums.set_x_range_handler)
        self.ui.range_max_2.returnPressed.connect(self.spectrums.set_x_range)
        self.ui.range_min_2.returnPressed.connect(self.spectrums.set_x_range)

        self.ui.sub_baseline_2.clicked.connect(
            self.spectrums.subtract_baseline_handler)
        self.ui.btn_undo_baseline_2.clicked.connect(
            self.spectrums.set_x_range_handler)
        self.ui.clear_peaks_2.clicked.connect(
            self.spectrums.clear_peaks_handler)
        self.ui.btn_fit_3.clicked.connect(
            self.spectrums.fit_fnc_handler)
        self.ui.btn_copy_fit_model_2.clicked.connect(
            self.spectrums.copy_fit_model)
        self.ui.btn_paste_fit_model_2.clicked.connect(
            self.spectrums.paste_fit_model_fnc_handler)
        self.ui.save_model_2.clicked.connect(self.spectrums.save_fit_model)

        self.ui.btn_load_model_3.clicked.connect(self.spectrums.load_fit_model)
        self.ui.btn_apply_model_3.clicked.connect(
            self.spectrums.apply_model_fnc_handler)
        self.ui.btn_open_fitspy_3.clicked.connect(
            self.spectrums.fitspy_launcher)
        self.ui.btn_cosmis_ray_3.clicked.connect(
            self.spectrums.cosmis_ray_detection)
        self.ui.btn_init_3.clicked.connect(self.spectrums.reinit_fnc_handler)
        self.ui.btn_show_stats_3.clicked.connect(self.spectrums.view_stats)
        self.ui.btn_sel_all_3.clicked.connect(self.spectrums.select_all_spectra)
        self.ui.btn_remove_spectrum.clicked.connect(
            self.spectrums.remove_spectrum)
        self.ui.btn_collect_results_3.clicked.connect(
            self.spectrums.collect_results)
        self.ui.btn_view_df_5.clicked.connect(
            self.spectrums.view_fit_results_df)
        self.ui.btn_save_fit_results_3.clicked.connect(
            self.spectrums.save_fit_results)

        self.ui.btn_split_fname.clicked.connect(self.spectrums.split_fname)
        self.ui.btn_add_col.clicked.connect(self.spectrums.add_column)

        self.ui.btn_default_folder_model_3.clicked.connect(
            self.spectrums.set_default_model_folder)

    def open(self, file_paths=None):
        """
        Universal action to open all supported files of SPECTROview:
         - spectroscopic data which can be hyperspectral or spectra (CSV, TXT).
         - dataframes (Excel)
         - saved work of SPECTROview (.maps, .spectra, .graphs)
        """
        if file_paths is None:
            # Initialize the last used directory from QSettings
            last_dir = self.settings.value("last_directory", "/")
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            file_paths, _ = QFileDialog.getOpenFileNames(
                self.ui.tabWidget, "Open spectra file(s)", last_dir,
                "SPECTROview formats (*.csv *.txt *.spectra *.maps * "
                ".graphs)", options=options)

        if file_paths:
            last_dir = QFileInfo(file_paths[0]).absolutePath()
            self.settings.setValue("last_directory", last_dir)

            spectra_files = []
            hyperspectral_files = []
            dataframes = []
            spectra_file = None
            maps_file = None
            graphs_file = None
            df = None

            for file_path in file_paths:
                file_path = Path(file_path)
                extension = file_path.suffix.lower()  # get file extension
                if extension == '.spectra':
                    spectra_file = str(file_path)
                elif extension == '.maps':
                    maps_file = str(file_path)
                elif extension == '.graphs':
                    graphs_file = str(file_path)

                elif extension == '.xlsx':
                    dataframes.append(str(file_path))

                elif extension == '.csv':
                    df = pd.read_csv(file_path, delimiter=";", header=None,
                                     skiprows=3)
                elif extension == '.txt':
                    df = pd.read_csv(file_path, delimiter="\t", header=None,
                                     skiprows=3)
                else:
                    show_alert(f"Unsupported file format: {extension}")
                    continue
                if df is not None:
                    if df.shape[1] == 2:
                        spectra_files.append(str(file_path))
                    elif df.shape[1] > 3:
                        hyperspectral_files.append(str(file_path))
                    else:
                        show_alert(
                            f"Invalid number of columns in file: {file_path}")

            # Open files with corresponding method
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

    def setup_shortcuts(self):
        """Setup key shortcuts of the application"""
        # Use MetaModifier for macOS and ControlModifier for other platforms
        modifier = Qt.MetaModifier if sys.platform == "darwin" else Qt.ControlModifier

        # Shortcut for Cmd+1 (or Ctrl+1 on other platforms) to switch to tab_spectra
        shortcut_spectra = QShortcut(QKeySequence(modifier | Qt.Key_1), self.ui)
        shortcut_spectra.activated.connect(lambda: self.ui.tabWidget.setCurrentWidget(self.ui.tab_spectra))
        shortcut_spectra_amp = QShortcut(QKeySequence(modifier | Qt.Key_Ampersand), self.ui)
        shortcut_spectra_amp.activated.connect(lambda: self.ui.tabWidget.setCurrentWidget(self.ui.tab_spectra))

        # Shortcut for Cmd+2 (or Ctrl+2 on other platforms) to switch to tab_maps
        shortcut_maps = QShortcut(QKeySequence(modifier | Qt.Key_2), self.ui)
        shortcut_maps.activated.connect(lambda: self.ui.tabWidget.setCurrentWidget(self.ui.tab_maps))
        shortcut_graphs_e = QShortcut(QKeySequence(modifier | Qt.Key_Eacute), self.ui)
        shortcut_graphs_e.activated.connect(lambda: self.ui.tabWidget.setCurrentWidget(self.ui.tab_maps))

        # Shortcut for Cmd+3 (or Ctrl+3 on other platforms) to switch to tab_graphs
        shortcut_graphs = QShortcut(QKeySequence(modifier | Qt.Key_3), self.ui)
        shortcut_graphs.activated.connect(lambda: self.ui.tabWidget.setCurrentWidget(self.ui.tab_graphs))
        shortcut_spectra_quote = QShortcut(QKeySequence(modifier | Qt.Key_QuoteDbl), self.ui)
        shortcut_spectra_quote.activated.connect(lambda: self.ui.tabWidget.setCurrentWidget(self.ui.tab_graphs))

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
        self.settings.setValue("mode", "dark")  # Save to settings

    def toggle_light_mode(self):
        self.ui.setPalette(self.common.light_palette())
        self.settings.setValue("mode", "light")  # Save to settings

    def open_manual(self):
        """Open doc detail about query function of pandas dataframe"""
        title = "SPECTROview User Manual"
        self.common.view_markdown(self.ui, title, USER_MANUAL, 1200, 900, "doc/")

    def show_about(self):
        """Show about dialog """
        self.common.view_markdown(self.ui, "About", ABOUT, 650, 440, "resources/")
        
    def save_settings(self):
        """
        Save all settings to persistent storage (QSettings).
        """        
        gui_states = {
            'ncpu': self.ui.ncpus.text(),
            # Maps module
            'fit_negative': self.ui.cb_fit_negative.isChecked(),
            'max_ite': self.ui.max_iteration.text(),
            'method': self.ui.cbb_fit_methods.currentText(),
            'xtol': float(self.ui.xtol.text()),
            'attached': self.ui.cb_attached.isChecked(),
            
            # Spectra module
            'fit_negative2': self.ui.cb_fit_negative_2.isChecked(),
            'max_ite2': self.ui.max_iteration_2.text(),
            'method2': self.ui.cbb_fit_methods_2.currentText(),
            'xtol2': float(self.ui.xtol_2.text()),
            'attached2': self.ui.cb_attached_2.isChecked(),
            
            # Visualization module
            'grid': self.ui.cb_grid.isChecked()
        }
        # Save the gui states to QSettings
        for key, value in gui_states.items():
            self.settings.setValue(key, value)

    def load_settings(self):
        """
        Load last used fitting settings from persistent storage (QSettings).
        """
        gui_states = {
            'ncpu': self.settings.value('ncpu', defaultValue=1,
                                           type=int),
            # Maps module
            'fit_negative': self.settings.value('fit_negative',
                                                defaultValue=False, type=bool),
            'max_ite': self.settings.value('max_ite', defaultValue=500,
                                           type=int),
            'method': self.settings.value('method', defaultValue='leastsq',
                                          type=str),
            'xtol': self.settings.value('xtol', defaultValue=1.e-4, type=float), 
            # 'attached': self.settings.value('attached',
            #                                     defaultValue=True, type=bool),
            
            # Spectra module
            'fit_negative2': self.settings.value('fit_negative2',
                                                defaultValue=False, type=bool),
            'max_ite2': self.settings.value('max_ite2', defaultValue=500,
                                           type=int),
            'method2': self.settings.value('method2', defaultValue='leastsq',
                                          type=str),
            'xtol2': self.settings.value('xtol2', defaultValue=1.e-4, type=float),
            # 'attached2': self.settings.value('attached2',
            #                                     defaultValue=True, type=bool),
            
            # Visualization module
            'grid': self.settings.value('grid', defaultValue=False, type=bool),
        }

        # Update GUI elements with the loaded values
        self.ui.ncpus.setValue(gui_states['ncpu'])
        self.ui.cb_fit_negative.setChecked(gui_states['fit_negative'])
        self.ui.max_iteration.setValue(gui_states['max_ite'])
        self.ui.cbb_fit_methods.setCurrentText(gui_states['method'])
        self.ui.xtol.setText(str(gui_states['xtol']))
        # self.ui.cb_attached.setChecked(gui_states['attached'])
        
        self.ui.cb_fit_negative_2.setChecked(gui_states['fit_negative2'])
        self.ui.max_iteration_2.setValue(gui_states['max_ite2'])
        self.ui.cbb_fit_methods_2.setCurrentText(gui_states['method2'])
        self.ui.xtol_2.setText(str(gui_states['xtol2']))
        # self.ui.cb_attached_2.setChecked(gui_states['attached2'])
        
        self.ui.cb_grid.setChecked(gui_states['grid'])
        
expiration_date = datetime.datetime(2025, 6, 1)

def launcher():
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    if datetime.datetime.now() > expiration_date:
        text = f"The current SPECTROview version has expired. Checkout the SPECTROview's Github page (cf. About) to update newest version."
        # If expired, disable the central widget
        app = QApplication(sys.argv)
        app.setWindowIcon(QIcon(ICON_APPLI))
        window = Main()
        window.ui.centralwidget.setEnabled(False)
        app.setStyle("Fusion")
        window.ui.show()
        show_alert(text)
        sys.exit(app.exec())

    # If not expired, continue launching the application as usual
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

