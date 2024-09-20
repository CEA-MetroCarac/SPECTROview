import os
import time
import numpy as np
import pandas as pd
import json
from copy import deepcopy
from pathlib import Path

from .common import view_df, show_alert, spectrum_to_dict, dict_to_spectrum, baseline_to_dict, dict_to_baseline, clear_layout
from .common import FitThread, PeakTable, DataframeTable, \
    FitModelManager, CustomListWidget, SpectraViewWidget
from .common import FIT_METHODS,PALETTE

from lmfit import fit_report
from fitspy.spectra import Spectra
from fitspy.spectrum import Spectrum
from fitspy.app.gui import Appli
from fitspy.utils import closest_index

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from PySide6.QtWidgets import QFileDialog, QMessageBox, QApplication,QAbstractItemView, QListWidgetItem, QSlider, QHBoxLayout, QLabel, QComboBox
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt, QFileInfo, QTimer, QObject
from tkinter import Tk, END

from superqt import QRangeSlider, QLabeledSlider , QLabeledRangeSlider

class Maps(QObject):
    """
    Class manages the GUI interactions and operations related to spectra
    fittings,
    and visualization of fitted data within 'Maps' TAB of the application.
    """

    def __init__(self, settings, ui, spectrums, common, visu):
        super().__init__()
        self.settings = settings
        self.ui = ui
        self.visu = visu
        self.spectrums_tab = spectrums
        self.common = common

        self.loaded_fit_model = None
        self.current_fit_model = None
        self.maps = {}  # list of opened maps data
        self.spectrums = Spectra()
        
        
         # Initialize SpectraViewWidget
        self.spectra_widget = SpectraViewWidget(self)
        self.ui.fig_canvas_layout_2.addWidget(self.spectra_widget.canvas)
        self.ui.toolbar_layout_2.addWidget(self.spectra_widget.control_widget) 
        self.ui.cbb_fit_models.currentIndexChanged.connect(self.update_peak_model)
        
         # Initialize PeakTable
        self.peak_table = PeakTable(self, self.ui.peak_table1, self.ui.cbb_layout_2)
        
         # Initialize Dataframe table
        self.df_fit_results = None
        self.df_table = DataframeTable(self.ui.layout_df_table)
        
        # Update spectra_listbox when selecting maps via MAPS LIST
        self.ui.maps_listbox.itemSelectionChanged.connect(
            self.upd_spectra_list)
        
        # Create a customized QListWidget
        self.ui.spectra_listbox = CustomListWidget()
        self.ui.spectra_listbox.setDragDropMode(QAbstractItemView.NoDragDrop)
        self.ui.listbox_layout2.addWidget(self.ui.spectra_listbox)
        self.ui.spectra_listbox.itemSelectionChanged.connect(self.refresh_gui)
        self.ui.checkBox_2.stateChanged.connect(self.check_uncheck_all)
        
        # Set a delay for the function "plot1"
        self.delay_timer = QTimer()
        self.delay_timer.setSingleShot(True)
        self.delay_timer.timeout.connect(self.plot)

        self.plot_styles = ["box plot", "point plot", "bar plot"]
        self.create_2Dmap_widget()
        self.zoom_pan_active = False

        self.ui.cbb_fit_methods.addItems(FIT_METHODS)
        self.ui.btn_send_to_viz2.clicked.connect(self.send_df_to_viz)
        
        # Connect the stateChanged signal of the legend CHECKBOX
        self.ui.cbb_wafer_size.currentIndexChanged.connect(self.refresh_gui)
        self.ui.rdbt_show_wafer.toggled.connect(self.refresh_gui)
        self.ui.cb_interpolation.stateChanged.connect(self.refresh_gui)
        self.ui.cb_remove_outliters.stateChanged.connect(self.update_z_range_slider)
        
        self.ui.cbb_map_color.addItems(PALETTE)
        self.ui.cbb_map_color.currentIndexChanged.connect(self.refresh_gui)
        
        # BASELINE
        self.setup_baseline_controls()
        
        # Load default folder path from QSettings during application startup
        self.fit_model_manager = FitModelManager(self.settings)
        self.fit_model_manager.default_model_folder = self.settings.value(
            "default_model_folder", "")
        self.ui.l_defaut_folder_model.setText(
            self.fit_model_manager.default_model_folder)
        # Show available fit models
        QTimer.singleShot(0, self.populate_available_models)
        self.ui.btn_refresh_model_folder.clicked.connect(
            self.populate_available_models)

    def setup_baseline_controls(self):
        self.ui.cb_attached.clicked.connect(self.refresh_gui)
        self.ui.noise.valueChanged.connect(self.refresh_gui)
        self.ui.rbtn_linear.clicked.connect(self.refresh_gui)
        self.ui.rbtn_polynomial.clicked.connect(self.refresh_gui)
        self.ui.degre.valueChanged.connect(self.refresh_gui)
        
        self.ui.cb_attached.stateChanged.connect(self.get_baseline_settings)
        self.ui.noise.valueChanged.connect(self.get_baseline_settings)
        self.ui.rbtn_linear.toggled.connect(self.get_baseline_settings)
        self.ui.degre.valueChanged.connect(self.get_baseline_settings)
        
        self.ui.btn_copy_bl.clicked.connect(self.copy_baseline)
        self.ui.btn_paste_bl.clicked.connect(self.paste_baseline_handler)
        self.ui.sub_baseline.clicked.connect(self.subtract_baseline_handler)
    
    def open_hyperspectra(self, maps=None, file_paths=None):
        """Open hyperspectral data"""

        if self.maps is None:
            self.maps = {}
        if maps:
            self.maps = maps
        else:
            if file_paths is None:
                # Initialize the last used directory from QSettings
                last_dir = self.settings.value("last_directory", "/")
                options = QFileDialog.Options()
                options |= QFileDialog.ReadOnly
                file_paths, _ = QFileDialog.getOpenFileNames(
                    self.ui.tabWidget, "Open spectra file(s)", last_dir,
                    "SPECTROview formats (*.csv *.txt)", options=options)

            # Load RAW spectra data from CSV files
            if file_paths:
                last_dir = QFileInfo(file_paths[0]).absolutePath()
                self.settings.setValue("last_directory", last_dir)

                for file_path in file_paths:
                    file_path = Path(file_path)
                    fname = file_path.stem  # get fname w/o extension
                    extension = file_path.suffix.lower()  # get file extension

                    if extension == '.csv':
                        # Read 3 first line of CSV files
                        with open(file_path, 'r') as file:
                            lines = [next(file) for _ in range(3)]

                        # Check 2nd line to determine old and new format
                        if len(lines[1].split(';')) > 3:
                            # If contains more than 3 columns
                            map_df = pd.read_csv(file_path, skiprows=1,
                                                 delimiter=";")
                        else:
                            map_df = pd.read_csv(file_path, header=None,
                                                 skiprows=2,
                                                 delimiter=";")
                    elif extension == '.txt':
                        map_df = pd.read_csv(file_path, delimiter="\t")
                        map_df.columns = ['Y', 'X'] + list(
                            map_df.columns[2:])
                        # Reorder df as increasing wavenumber
                        sorted_columns = sorted(map_df.columns[2:], key=float)
                        map_df = map_df[['X', 'Y'] + sorted_columns]
                    else:
                        show_alert(f"Unsupported file format: {extension}")
                        continue

                    map_name = fname
                    if map_name in self.maps:
                        msg = f"Map '{map_name}' is already opened"
                        show_alert(msg)
                    else:
                        self.maps[map_name] = map_df
        self.extract_spectra()
        self.ui.tabWidget.setCurrentWidget(self.ui.tab_maps)

    def extract_spectra(self):
        """Extract all spectra from each map dataframe."""
        for map_name, map_df in self.maps.items():
            if len(map_df.columns) > 2 and 'X' in map_df.columns and 'Y' \
                    in map_df.columns:
                self.process_old_format(map_df, map_name)
            else:
                self.process_new_format(map_df, map_name)
        self.upd_maps_list()

    def process_old_format(self, map_df, map_name):
        """Process old format wafer dataframe"""
        for _, row in map_df.iterrows():
            coord = tuple(row[['X', 'Y']])
            x_values = map_df.columns[2:].tolist()
            x_values = pd.to_numeric(x_values, errors='coerce').tolist()
            y_values = row[2:].tolist()
            # Skip the last value
            if len(x_values) > 1:
                x_values = x_values[:-1]
            if len(y_values) > 1:
                y_values = y_values[:-1]
            fname = f"{map_name}_{coord}"
            if not any(spectrum.fname == fname for spectrum in self.spectrums):
                spectrum = Spectrum()
                spectrum.fname = fname
                spectrum.x = np.asarray(x_values)
                spectrum.x0 = np.asarray(x_values)
                spectrum.y = np.asarray(y_values)
                spectrum.y0 = np.asarray(y_values)
                spectrum.baseline.mode = "Linear"
                self.spectrums.append(spectrum)

    def process_new_format(self, map_df, map_name):
        """Process new format wafer dataframe."""
        for i in range(0, len(map_df), 2):
            coord_row = map_df.iloc[i]
            intensity_row = map_df.iloc[i + 1]
            coord = (float(coord_row.iloc[0]), float(coord_row.iloc[1]))
            x_values = coord_row.iloc[2:].tolist()
            x_values = pd.to_numeric(x_values, errors='coerce').tolist()
            y_values = intensity_row.iloc[2:].tolist()
            # Skip the last value
            if len(x_values) > 1:
                x_values = x_values[:-1]
            if len(y_values) > 1:
                y_values = y_values[:-1]
            fname = f"{map_name}_{coord}"
            if not any(spectrum.fname == fname for spectrum in self.spectrums):
                spectrum = Spectrum()
                spectrum.fname = fname
                spectrum.x = np.asarray(x_values)
                spectrum.x0 = np.asarray(x_values)
                spectrum.y = np.asarray(y_values)
                spectrum.y0 = np.asarray(y_values)
                spectrum.baseline.mode = "Linear"
                self.spectrums.append(spectrum)


    def view_fit_results_df(self):
        """To view selected dataframe in GUI"""
        df = getattr(self, 'df_fit_results', None)  
        if df is not None and not df.empty: 
            view_df(self.ui.tabWidget, df)
        else:
            show_alert("No fit dataframe to display")

    def collect_results(self):
        """Collect fit results to create a consolidated dataframe"""
        # Add all dicts into a list, then convert to a dataframe.
        self.copy_fit_model()
        fit_results_list = []
        self.df_fit_results = None

        for spectrum in self.spectrums:
            if hasattr(spectrum, 'peak_models'):
                map_name, coord = self.spectrum_object_id(spectrum)
                x, y = coord
                fit_result = {'Filename': map_name, 'X': x, 'Y': y}

                for model in spectrum.peak_models:
                    if hasattr(model, 'param_names') and hasattr(model,
                                                                 'param_hints'):
                        for param_name in model.param_names:
                            key = param_name.split('_')[1]
                            if key in model.param_hints and 'value' in \
                                    model.param_hints[key]:
                                fit_result[param_name] = model.param_hints[key][
                                    'value']

                if len(fit_result) > 3:
                    fit_results_list.append(fit_result)
        self.df_fit_results = pd.DataFrame(fit_results_list).round(3)

        if self.df_fit_results is not None and not self.df_fit_results.empty:
            # reindex columns according to the parameters names
            self.df_fit_results = self.df_fit_results.reindex(
                sorted(self.df_fit_results.columns), axis=1)
            names = []
            for name in self.df_fit_results.columns:
                if name in ["Filename", "X", 'Y', "success"]:
                    # to be in the 3 first columns
                    name = '0' + name
                elif '_' in name:
                    # model peak parameters to be at the end
                    name = 'z' + name[5:]
                names.append(name)
            self.df_fit_results = self.df_fit_results.iloc[:,
                                  list(np.argsort(names, kind='stable'))]
            columns = [
                self.common.translate_param(self.current_fit_model, column) for
                column in self.df_fit_results.columns]
            self.df_fit_results.columns = columns

            
            if self.ui.rdbt_show_wafer.isChecked():
                # DIAMETER
                diameter = float(self.ui.cbb_wafer_size.currentText())
                # QUADRANT
                self.df_fit_results['Quadrant'] = self.df_fit_results.apply(
                    self.common.quadrant, axis=1)
                # ZONE
                self.df_fit_results['Zone'] = self.df_fit_results.apply(
                    lambda row: self.common.zone(row, diameter), axis=1)
            else: 
                pass
        self.df_table.show(self.df_fit_results)


    def update_peak_model(self):
        """Update the peak model in the SpectraViewWidget based on combobox selection."""
        selected_model = self.ui.cbb_fit_models.currentText()
        self.spectra_widget.set_peak_model(selected_model)

    def set_default_model_folder(self, folder_path=None):
        """Define a default folder containing fit models."""
        if not folder_path:
            folder_path = QFileDialog.getExistingDirectory(None,
                                                           "Select Default "
                                                           "Folder",
                                                           options=QFileDialog.ShowDirsOnly)

        if folder_path:
            self.fit_model_manager.set_default_model_folder(folder_path)
            # Save selected folder path back to QSettings
            self.settings.setValue("default_model_folder", folder_path)
            self.ui.l_defaut_folder_model.setText(
                self.fit_model_manager.default_model_folder)
            QTimer.singleShot(0, self.populate_available_models)

    def populate_available_models(self):
        """Populate available fit models in the combobox"""
        # Scan default folder and populate available models in the combobox
        self.fit_model_manager.scan_models()
        self.available_models = self.fit_model_manager.get_available_models()
        self.ui.cbb_fit_model_list.clear()
        self.ui.cbb_fit_model_list.addItems(self.available_models)

    def upd_model_cbb_list(self):
        """
        Update and populate the model list in the combobox.

        Updates the list of models in the combobox based on the default model
        folder.
        """
        current_path = self.fit_model_manager.default_model_folder
        self.set_default_model_folder(current_path)

    def load_fit_model(self, fname_json=None):
        """Load a pre-created fit model"""
        self.fname_json = fname_json
        self.upd_model_cbb_list()
        if not self.fname_json:
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            selected_file, _ = QFileDialog.getOpenFileName(self.ui,
                                                           "Select JSON Model "
                                                           "File",
                                                           "",
                                                           "JSON Files ("
                                                           "*.json);;All "
                                                           "Files (*)",
                                                           options=options)
            if not selected_file:
                return
            self.fname_json = selected_file
        display_name = QFileInfo(self.fname_json).fileName()
        # Add the display name to the combobox only if it doesn't already exist
        if display_name not in [self.ui.cbb_fit_model_list.itemText(i) for i in
                                range(self.ui.cbb_fit_model_list.count())]:
            self.ui.cbb_fit_model_list.addItem(display_name)
            self.ui.cbb_fit_model_list.setCurrentText(display_name)
        else:
            show_alert('Fit model is already available in the model list')

    def get_loaded_fit_model(self):
        """Define loaded fit model"""
        if self.ui.cbb_fit_model_list.currentIndex() == -1:
            self.loaded_fit_model = None
            return
        try:
            # If the file is not found in the selected path, try finding it
            # in the default folder
            folder_path = self.fit_model_manager.default_model_folder
            model_name = self.ui.cbb_fit_model_list.currentText()
            path = os.path.join(folder_path, model_name)
            self.loaded_fit_model = self.spectrums.load_model(path, ind=0)
        except FileNotFoundError:
            try:
                self.loaded_fit_model = self.spectrums.load_model(
                    self.fname_json, ind=0)
            except FileNotFoundError:
                show_alert('Fit model file not found in the default folder.')

    def save_fit_model(self):
        """Save the fit model of the selected spectrum."""
        sel_spectrum, sel_spectra = self.get_spectrum_object()
        path = self.fit_model_manager.default_model_folder
        save_path, _ = QFileDialog.getSaveFileName(
            self.ui.tabWidget, "Save fit model", path,
            "JSON Files (*.json)")
        if save_path and sel_spectrum:
            self.spectrums.save(save_path, [sel_spectrum.fname])
            show_alert("Fit model is saved (JSON file)")
        else:
            show_alert("No fit model to save.")
        self.upd_model_cbb_list()

    def read_x_range(self):
        """Read x range of selected spectrum"""
        sel_spectrum, sel_spectra = self.get_spectrum_object()
        self.ui.range_min.setText(str(sel_spectrum.x[0]))
        self.ui.range_max.setText(str(sel_spectrum.x[-1]))

    def set_x_range(self, fnames=None):
        """Sets a new x-axis range for the selected spectrum(s)"""
        new_x_min = float(self.ui.range_min.text())
        new_x_max = float(self.ui.range_max.text())
        if fnames is None:
            map_name, coords = self.spectra_id()
            fnames = [f"{map_name}_{coord}" for coord in coords]
        self.common.reinit_spectrum(fnames, self.spectrums)
        for fname in fnames:
            spectrum, _ = self.spectrums.get_objects(fname)
            spectrum.range_min = float(self.ui.range_min.text())
            spectrum.range_max = float(self.ui.range_max.text())

            ind_min = closest_index(spectrum.x0, new_x_min)
            ind_max = closest_index(spectrum.x0, new_x_max)
            spectrum.x = spectrum.x0[ind_min:ind_max + 1].copy()
            spectrum.y = spectrum.y0[ind_min:ind_max + 1].copy()
        QTimer.singleShot(50, self.upd_spectra_list)
        QTimer.singleShot(300, self.spectra_widget.rescale)

    def set_x_range_all(self):
        """Set new x range for all spectrum"""
        fnames = self.spectrums.fnames
        self.set_x_range(fnames=fnames)


    def get_baseline_settings(self):
        """Get baseline settings from GUI and set to selected spectrum"""
        spectrum_object = self.get_spectrum_object()
        if spectrum_object is None:
            return
        else:
            sel_spectrum, sel_spectra = spectrum_object  
            sel_spectrum.baseline.attached = self.ui.cb_attached.isChecked()
            sel_spectrum.baseline.sigma = self.ui.noise.value()
            if self.ui.rbtn_linear.isChecked():
                sel_spectrum.baseline.mode = "Linear"
            else:
                sel_spectrum.baseline.mode = "Polynomial"
                sel_spectrum.baseline.order_max = self.ui.degre.value()

    

    def copy_baseline(self):
        """Copy baseline of the selected spectrum"""
        sel_spectrum, _ = self.get_spectrum_object()
        self.current_baseline = baseline_to_dict(sel_spectrum)

    
    def paste_baseline(self, sel_spectra=None):
        """Paste baseline to the selected spectrum(s)"""
        if sel_spectra is None:
            _, sel_spectra = self.get_spectrum_object()
        dict_to_baseline(self.current_baseline, sel_spectra)
        
        QTimer.singleShot(50, self.refresh_gui)

    def subtract_baseline(self, sel_spectra=None):
        """Subtract baseline action for the selected spectrum(s)."""
        if sel_spectra is None:
            _, sel_spectra = self.get_spectrum_object()
        for spectrum in sel_spectra:
            if not spectrum.baseline.is_subtracted:
                spectrum.subtract_baseline()
            else: 
                continue
        QTimer.singleShot(50, self.refresh_gui)
        QTimer.singleShot(300, self.spectra_widget.rescale)

    def subtract_baseline_all(self):
        """Subtracts baseline points for all spectra"""
        self.subtract_baseline(self.spectrums)
    
    def paste_baseline_all(self):
        """Paste baseline to the all spectrum(s)"""
        self.paste_baseline(self.spectrums)

    def get_fit_settings(self):
        """Get all settings for the fitting action."""
        sel_spectrum, sel_spectra = self.get_spectrum_object()
        fit_params = sel_spectrum.fit_params.copy()
        fit_params['fit_negative'] = self.ui.cb_fit_negative.isChecked()
        fit_params['max_ite'] = self.ui.max_iteration.value()
        fit_params['method'] = self.ui.cbb_fit_methods.currentText()
        fit_params['ncpus'] = self.ui.ncpus.value()
        fit_params['xtol'] = float(self.ui.xtol.text())
        sel_spectrum.fit_params = fit_params

    def fit(self, fnames=None):
        """Fit selected spectrum(s) with current parameters"""
        self.get_fit_settings()
        if fnames is None:
            map_name, coords = self.spectra_id()
            fnames = [f"{map_name}_{coord}" for coord in coords]
        for fname in fnames:
            spectrum, _ = self.spectrums.get_objects(fname)
            if len(spectrum.peak_models) != 0:
                spectrum.fit()
            else:
                continue
        QTimer.singleShot(100, self.upd_spectra_list)

    def fit_all(self):
        """Apply all fit parameters to all spectrum(s)"""
        fnames = self.spectrums.fnames
        self.fit(fnames)

    def clear_peaks(self, fnames=None):
        """Clear existing peak models of the selected spectrum(s)"""
        if fnames is None:
            map_name, coords = self.spectra_id()
            fnames = [f"{map_name}_{coord}" for coord in coords]
        for fname in fnames:
            spectrum, _ = self.spectrums.get_objects(fname)
            if len(spectrum.peak_models) != 0:
                spectrum.remove_models()
            else:
                continue
        QTimer.singleShot(100, self.upd_spectra_list)

    def clear_peaks_all(self):
        """Clear peak models of all spectra"""
        fnames = self.spectrums.fnames
        self.clear_peaks(fnames)

    def copy_fit_model(self):
        """Copy fit_model from current selected spectrum"""
        # Get only 1 spectrum among several selected spectrum:
        self.get_fit_settings()
        sel_spectrum, _ = self.get_spectrum_object()
        if len(sel_spectrum.peak_models) == 0:
            self.ui.lbl_copied_fit_model.setText("")
            msg = ("Select spectrum is not fitted or No fit results to collect")
            show_alert(msg)
            self.current_fit_model = None
            return
        else:
            self.current_fit_model = None
            self.current_fit_model = deepcopy(sel_spectrum.save())
        self.ui.lbl_copied_fit_model.setText("copied")

    def paste_fit_model(self, fnames=None):
        """Apply the copied fit model to selected spectra."""

        self.ui.centralwidget.setEnabled(False)  # Disable GUI
        if fnames is None:
            map_name, coords = self.spectra_id()
            fnames = [f"{map_name}_{coord}" for coord in coords]

        self.common.reinit_spectrum(fnames, self.spectrums)
        fit_model = deepcopy(self.current_fit_model)

        self.ntot = len(fnames)
        ncpus = int(self.ui.ncpus.text())

        if fit_model is not None:
            self.spectrums.pbar_index = 0
            self.thread = FitThread(self.spectrums, fit_model, fnames, ncpus)
            self.thread.finished.connect(self.fit_completed)
            self.thread.start()
        else:
            show_alert("Nothing to paste")
            self.ui.centralwidget.setEnabled(True)

        # Update progress bar & text
        self.start_time = time.time()
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress_bar)
        self.progress_timer.start(100)

    def paste_fit_model_all(self):
        """Apply the copied fit model to all spectra"""
        fnames = self.spectrums.fnames
        self.paste_fit_model(fnames)

    def apply_loaded_fit_model(self, fnames=None):
        """Fit selected spectrum(s) with the loaded fit model."""
        self.get_loaded_fit_model()
        self.ui.centralwidget.setEnabled(False)  # Disable GUI
        if self.loaded_fit_model is None:
            show_alert("Select from the list or load a fit model.")
            self.ui.centralwidget.setEnabled(True)
            return

        if fnames is None:
            map_name, coords = self.spectra_id()
            fnames = [f"{map_name}_{coord}" for coord in coords]

        self.ntot = len(fnames)
        ncpus = int(self.ui.ncpus.text())
        fit_model = self.loaded_fit_model
        spectra = self.spectrums
        self.spectrums.pbar_index = 0

        self.thread = FitThread(spectra, fit_model, fnames, ncpus)
        self.thread.finished.connect(self.fit_completed)
        self.thread.start()

        # Update progress bar & text
        self.start_time = time.time()
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress_bar)
        self.progress_timer.start(100)

    def update_progress_bar(self):
        """Update fitting progress in GUI"""
        index = self.spectrums.pbar_index
        percent = 100 * (index + 1) / self.ntot
        elapsed_time = time.time() - self.start_time
        text = f"{index}/{self.ntot} ({elapsed_time:.2f}s)"
        self.ui.progressBar.setValue(percent)
        self.ui.progressText.setText(text)
        if self.spectrums.pbar_index >= self.ntot - 1:
            self.progress_timer.stop()

    def fit_completed(self):
        """Update GUI after completing fitting process."""
        self.upd_spectra_list()
        QTimer.singleShot(200, self.spectra_widget.rescale)
        self.ui.progressBar.setValue(100)
        self.ui.centralwidget.setEnabled(True)

    def apply_loaded_fit_model_all(self):
        """Apply loaded fit model to all selected spectra"""
        fnames = self.spectrums.fnames
        self.apply_loaded_fit_model(fnames=fnames)

    def save_fit_results(self):
        """Save fitted results in an Excel file"""

        last_dir = self.settings.value("last_directory", "/")
        save_path, _ = QFileDialog.getSaveFileName(
            self.ui.tabWidget, "Save DF fit results", last_dir,
            "Excel Files (*.xlsx)")
        if save_path:
            try:
                if not self.df_fit_results.empty:
                    self.df_fit_results.to_excel(save_path, index=False)
                    QMessageBox.information(
                        self.ui.tabWidget, "Success",
                        "DataFrame saved successfully.")
                else:
                    QMessageBox.warning(
                        self.ui.tabWidget, "Warning",
                        "DataFrame is empty. Nothing to save.")
            except Exception as e:
                QMessageBox.critical(
                    self.ui.tabWidget, "Error",
                    f"Error saving DataFrame: {str(e)}")
    
    def split_fname(self):
        """Split 'Filename' column and populate the combobox"""
        dfr = self.df_fit_results
        try:
            fname_parts = dfr.loc[0, 'Filename'].split('_')
        except Exception as e:
            print(f"Error splitting column header: {e}")
        self.ui.cbb_split_fname_2.clear()
        for part in fname_parts:
            self.ui.cbb_split_fname_2.addItem(part)

    def add_column(self):
        """Add a column to `df_fit_results` based on split_fname method"""

        dfr = self.df_fit_results
        col_name = self.ui.ent_col_name_2.text()
        selected_part_index = self.ui.cbb_split_fname_2.currentIndex()
        if selected_part_index < 0 or not col_name:
            show_alert("Select a part and enter a column name.")
            return
        # Check if column with the same name already exists
        if col_name in dfr.columns:
            text = (
                f"Column '{col_name}' already exists. Please choose a "
                f"different name")
            show_alert(text)
            return
        try:
            parts = dfr['Filename'].str.split('_')
        except Exception as e:
            print(f"Error adding new column to fit results dataframe: {e}")

        dfr[col_name] = [part[selected_part_index] if len(
            part) > selected_part_index else None for part in parts]

        self.df_fit_results = dfr
        self.df_table.show(self.df_fit_results)


    def reinit(self, fnames=None):
        """Reinitialize the selected spectrum(s)."""
        if fnames is None:
            map_name, coords = self.spectra_id()
            fnames = [f"{map_name}_{coord}" for coord in coords]
        self.common.reinit_spectrum(fnames, self.spectrums)
        self.upd_spectra_list()
        QTimer.singleShot(200, self.spectra_widget.rescale)

    def reinit_all(self):
        """Reinitialize all spectra"""
        fnames = self.spectrums.fnames
        self.reinit(fnames)

    def plot(self):
        """Plot spectra or fit results in the main plot area."""
        map_name, coords = self.spectra_id()  # current selected spectra ID
        selected_spectrums = []

        for spectrum in self.spectrums:
            map_name_fs, coord_fs = self.spectrum_object_id(spectrum)
            if map_name_fs == map_name and coord_fs in coords:
                selected_spectrums.append(spectrum)

        # Limit the number of spectra to avoid crashes
        selected_spectrums = selected_spectrums[:30]

        if not selected_spectrums:
            return

        self.spectra_widget.plot(selected_spectrums)
        self.plot_2Dmap()
        self.read_x_range()
        self.peak_table.show(selected_spectrums[0])
        
    def upd_spectra_list(self):
        """Show spectrums in a listbox"""
        map_name, _ = self.spectra_id()
        map_df = self.maps.get(map_name)
            
        if map_df is not None:
            column_labels = map_df.columns[2:-1].astype(float)
            min_value = float(column_labels.min())
            max_value = float(column_labels.max())
            self.update_xrange_slider(min_value, max_value)

        checked_states = {}
        for index in range(self.ui.spectra_listbox.count()):
            item = self.ui.spectra_listbox.item(index)
            checked_states[item.text()] = item.checkState()
          
        self.current_row = self.ui.spectra_listbox.currentRow()
        self.ui.spectra_listbox.clear()
        current_item = self.ui.maps_listbox.currentItem()

        if current_item is not None:
            map_name = current_item.text()
            
            for spectrum in self.spectrums:
                map_name_fs, coord_fs = self.spectrum_object_id(spectrum)
                if map_name == map_name_fs:
                    item = QListWidgetItem(str(coord_fs))
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(checked_states.get(coord_fs, Qt.Checked))
                    if hasattr(spectrum.result_fit,
                               'success') and spectrum.result_fit.success:
                        item.setBackground(QColor("green"))
                    elif hasattr(spectrum.result_fit,
                                 'success') and not \
                            spectrum.result_fit.success:
                        item.setBackground(QColor("orange"))
                    else:
                        item.setBackground(QColor(0, 0, 0, 0))
                    self.ui.spectra_listbox.addItem(item)

        # Update the item count label
        item_count = self.ui.spectra_listbox.count()
        self.ui.item_count_label.setText(f"{item_count} points")

        # Reselect the previously selected item
        if self.current_row >= 0 and self.current_row < item_count:
            self.ui.spectra_listbox.setCurrentRow(self.current_row)
        else:
            if self.ui.spectra_listbox.count() > 0:
                self.ui.spectra_listbox.setCurrentRow(0)
        QTimer.singleShot(50, self.refresh_gui)
        
    def create_2Dmap_widget(self):
        """Create 2Dmap plot widgets"""
        fig = plt.figure(dpi=70)
        self.ax = fig.add_subplot(111)
        
        self.ax.tick_params(axis='x', which='both')
        self.ax.tick_params(axis='y', which='both')
        self.canvas = FigureCanvas(fig)
        self.toolbar =NavigationToolbar2QT(self.canvas)
        for action in self.toolbar.actions():
            if action.text() in ['Home','Zoom','Save', 'Pan', 'Back', 'Forward', 'Subplots']:
                action.setVisible(False)
                
        # Variables to keep track of highlighted points and Ctrl key status
        self.selected_points = []
        self.ctrl_pressed = False
        
        # Connect the mouse and key events to the handler functions
        fig.canvas.mpl_connect('button_press_event', self.on_click_2Dmap)
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        
        self.ui.map_layout.addWidget(self.canvas)
        self.ui.toolbar_layout_3.addWidget(self.toolbar)
        self.canvas.draw()
        self.create_range_sliders(0,100)
        
    def create_range_sliders(self, xmin, xmax):
        """Create xrange and intensity-range sliders"""
        self.x_range_slider = QRangeSlider(Qt.Horizontal)
        self.x_range_slider.setRange(xmin, xmax)  
        self.x_range_slider.setValue((xmin, xmax)) 
        self.x_range_slider.setTracking(True)
        self.x_range_slider_label = QLabel('X range:')
        self.x_range_label = QLabel(f'[{xmin}; {xmax}]')
        self.ui.xrange_slider_layout.addWidget(self.x_range_slider_label)
        self.ui.xrange_slider_layout.addWidget(self.x_range_slider)
        self.ui.xrange_slider_layout.addWidget(self.x_range_label)
        # Connect to update function
        self.x_range_slider.valueChanged.connect(self.update_xrange_slider_label)
        self.x_range_slider.valueChanged.connect(self.update_z_range_slider)
        
        self.z_range_slider = QRangeSlider(Qt.Horizontal)
        self.z_range_slider.setRange(0, 100) 
        self.z_range_slider.setValue((0, 100)) 
        self.z_range_slider.setTracking(True)
        self.z_slider_cbb = QComboBox()
        self.z_slider_cbb.addItems(['Area', 'Intensity'])
        self.z_slider_cbb.currentIndexChanged.connect(self.update_z_range_slider)
        
        self.intensity_range_label = QLabel(f'[{0}; {100}]')
        self.ui.intensity_slider_layout.addWidget(self.z_slider_cbb)
        self.ui.intensity_slider_layout.addWidget(self.z_range_slider)
        self.ui.intensity_slider_layout.addWidget(self.intensity_range_label)
        
        self.z_range_slider.valueChanged.connect(self.update_z_range_label)
        self.z_range_slider.valueChanged.connect(self.refresh_gui)
    
    def update_xrange_slider(self, xmin, xmax):
        """Update the range of the slider based on new min and max values."""
        xmin_label = round(xmin, 3)
        xmax_label = round(xmax, 3)
        self.x_range_slider.setRange(xmin, xmax)
        self.x_range_slider.setValue((xmin, xmax))
        self.x_range_label.setText(f'[{xmin_label}; {xmax_label}]')
    
    def update_xrange_slider_label(self):
        """Update the QLabel text with the current values."""
        xmin_val, max_val = self.x_range_slider.value()
        self.x_range_label.setText(f'[{xmin_val}; {max_val}]')
        
    def update_z_range_slider(self):
        _,_, vmin, vmax, =self.get_data_for_heatmap()
        self.z_range_slider.setRange(vmin, vmax)
        self.z_range_slider.setValue((vmin, vmax))
        self.intensity_range_label.setText(f'[{vmin}; {vmax}]')

    def update_z_range_label(self):
        """Update the QLabel text with the current values."""
        imin_val, imax_val = self.z_range_slider.value()
        self.intensity_range_label.setText(f'[{imin_val}; {imax_val}]')
        
    def get_data_for_heatmap(self):
        """Prepare data for heatmap based on range sliders values"""
        map_name, _ = self.spectra_id()
        map_df = self.maps.get(map_name)

        # Default return values in case of no valid map_df or filtered columns
        heatmap_pivot = pd.DataFrame()  # Empty DataFrame for heatmap
        extent = [0, 0, 0, 0]  # Default extent values
        vmin = 0
        vmax = 0
        
        if map_df is not None:
            min_range, max_range = self.x_range_slider.value()
            column_labels = map_df.columns[2:-1]  # Keep labels as strings

            # Convert slider range values to strings for comparison
            filtered_columns = column_labels[(column_labels.astype(float) >= min_range) &
                                            (column_labels.astype(float) <= max_range)]
            
            if len(filtered_columns) > 0:
                # Create a filtered DataFrame including X, Y, and the selected range of columns
                filtered_map_df = map_df[['X', 'Y'] + list(filtered_columns)]
                x_col = filtered_map_df['X'].values
                y_col = filtered_map_df['Y'].values
                final_z_col = []
                parameter = self.z_slider_cbb.currentText()
                if parameter == 'Area':
                    # Calculate the intensity sums for the selected range
                    z_col = filtered_map_df[filtered_columns].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0).sum(axis=1)
                if parameter == 'Intensity':
                    # Min and max intensity values for each spectrum
                    z_col = filtered_map_df[filtered_columns].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0).max(axis=1)
                
                if self.ui.cb_remove_outliters.isChecked():
                    # Remove outliers using IQR method and replace them with interpolated values
                    Q1 = z_col.quantile(0.05)
                    Q3 = z_col.quantile(0.95)
                    IQR = Q3 - Q1

                    # Identify the outliers
                    outlier_mask = (z_col < (Q1 - 1.5 * IQR)) | (z_col > (Q3 + 1.5 * IQR))

                    # Interpolate values for the outliers using linear interpolation
                    z_col_interpolated = z_col.copy()
                    z_col_interpolated[outlier_mask] = np.nan  # Mark outliers as NaN for interpolation
                    z_col_interpolated = z_col_interpolated.interpolate(method='linear', limit_direction='both')
                    final_z_col=z_col_interpolated
                else:
                    final_z_col=z_col       
                # Update vmin and vmax after interpolation
                vmin = round(final_z_col.min(), 2)
                vmax = round(final_z_col.max(), 2)

                # Heatmap data 
                heatmap_data = pd.DataFrame({'X': x_col, 'Y': y_col, 'Z': final_z_col})
                heatmap_pivot = heatmap_data.pivot(index='Y', columns='X', values='Z')
                xmin, xmax = x_col.min(), x_col.max()
                ymin, ymax = y_col.min(), y_col.max()
                extent=[xmin, xmax, ymin, ymax]
                
        return heatmap_pivot, extent, vmin, vmax
    
    def plot_2Dmap(self):
        """Plot 2D maps of measurement points"""
        r = int(self.ui.cbb_wafer_size.currentText()) / 2

        self.ax.clear()

        if self.ui.rdbt_show_wafer.isChecked():
            wafer_circle = patches.Circle((0, 0), radius=r, fill=False,
                                        color='black', linewidth=1)
            self.ax.add_patch(wafer_circle)
            self.ax.set_yticklabels([])

            all_x, all_y = self.get_mes_sites_coord()
            self.ax.scatter(all_x, all_y, marker='x', color='gray', s=10)
            self.ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

        # Plot heatmap for 2D map
        if self.ui.rdbt_show_2Dmap.isChecked():    
            heatmap_pivot, extent, _, _ = self.get_data_for_heatmap()
            
            color = self.ui.cbb_map_color.currentText()
            interpolation_option = 'bilinear' if self.ui.cb_interpolation.isChecked() else 'none'
            vmin, vmax = self.z_range_slider.value()

            self.img = self.ax.imshow(heatmap_pivot, extent=extent, vmin=vmin, vmax=vmax,
                                origin='lower', aspect='auto', cmap=color, interpolation=interpolation_option)
            
            # Update or create the colorbar
            if hasattr(self, 'cbar') and self.cbar is not None:
                self.cbar.update_normal(self.img)
            else:
                self.cbar = self.ax.figure.colorbar(self.img, ax=self.ax)

        # Highlighted measurement sites
        map_name, coords = self.spectra_id()
        if coords:
            x, y = zip(*coords)
            self.ax.scatter(x, y, marker='o', color='red', s=20)

        title = self.z_slider_cbb.currentText()
        self.ax.set_title(title, fontsize=13)
        self.ax.get_figure().tight_layout()
        self.canvas.draw()
        
    def on_click_2Dmap(self, event):
        """select the measurement points via 2Dmap plot"""
        all_x, all_y = self.get_mes_sites_coord()
        self.ui.spectra_listbox.clearSelection()
        if event.inaxes == self.ax:
            x_clicked, y_clicked = event.xdata, event.ydata
            if event.button == 1:  # Left mouse button
                all_x = np.array(all_x)
                all_y = np.array(all_y)
                distances = np.sqrt(
                    (all_x - x_clicked) ** 2 + (all_y - y_clicked) ** 2)
                nearest_index = np.argmin(distances)
                nearest_x, nearest_y = all_x[nearest_index], all_y[
                    nearest_index]

                # Check if Ctrl key is pressed
                modifiers = QApplication.keyboardModifiers()
                if modifiers == Qt.ControlModifier:
                    self.selected_points.append((nearest_x, nearest_y))
                else:
                    # Clear the selected points list and add the current one
                    self.selected_points = [(nearest_x, nearest_y)]

        # Set the current selection in the spectra_listbox
        for index in range(self.ui.spectra_listbox.count()):
            item = self.ui.spectra_listbox.item(index)
            item_text = item.text()
            x, y = map(float, item_text.strip('()').split(','))
            if (x, y) in self.selected_points:
                item.setSelected(True)
                self.current_row= index
                self.ui.spectra_listbox.setCurrentRow(self.current_row)
            else:
                item.setSelected(False)
            
            
    def on_key_press(self, event):
        """Handler function for key press event"""
        if event.key == 'ctrl':
            self.ctrl_pressed = True

    def on_key_release(self, event):
        """Handler function for key release event"""
        if event.key == 'ctrl':
            self.ctrl_pressed = False

    def get_mes_sites_coord(self):
        """
        Get all coordinates of measurement sites of the selected map.
        """
        map_name, coords = self.spectra_id()
        all_x = []
        all_y = []
        for spectrum in self.spectrums:
            map_name_fs, coord_fs = self.spectrum_object_id(spectrum)
            if map_name == map_name_fs:
                x, y = coord_fs
                all_x.append(x)
                all_y.append(y)
        return all_x, all_y

    def upd_maps_list(self):
        """
        Update the Maps listbox.

        Action:
        - Retrieves the current row selection from the map listbox.
        - Clears the Maps listbox and updates it with current map names.
        - Handles selection of items in the listbox.
        """
        current_row = self.ui.maps_listbox.currentRow()
        self.ui.maps_listbox.clear()
        map_names = list(self.maps.keys())
        for map_name in map_names:
            item = QListWidgetItem(map_name)
            self.ui.maps_listbox.addItem(item)
        item_count = self.ui.maps_listbox.count()
        # Management of selecting item of listbox
        if current_row >= item_count:
            current_row = item_count - 1
        if current_row >= 0:
            self.ui.maps_listbox.setCurrentRow(current_row)
        else:
            if item_count > 0:
                self.ui.maps_listbox.setCurrentRow(0)
        QTimer.singleShot(100, self.upd_spectra_list)

    def check_uncheck_all(self, state):
        """
        Check or uncheck all items in the listbox based on the state of the
        main checkbox.
        """
        check_state = Qt.Unchecked if state == 0 else Qt.Checked
        for index in range(self.ui.spectra_listbox.count()):
            item = self.ui.spectra_listbox.item(index)
            item.setCheckState(check_state)
    
    

    def remove_map(self):
        """
        Remove a selected map from the application
        """
        map_name, coords = self.spectra_id()
        if map_name in self.maps:
            del self.maps[map_name]
            self.spectrums = Spectra(
                spectrum for spectrum in self.spectrums if
                not spectrum.fname.startswith(map_name))
            self.upd_maps_list()
        self.ui.spectra_listbox.clear()
        self.spectra_widget.sel_spectrums = None  
        self.spectra_widget.refresh_plot() 
        self.ax.clear()
        self.canvas.draw_idle()

    def select_all_spectra(self):
        """Select all spectra listed in the spectra listbox"""
        item_count = self.ui.spectra_listbox.count()
        for i in range(item_count):
            item = self.ui.spectra_listbox.item(i)
            item.setSelected(True)

    def select_verti(self):
        """Select all spectra listed vertically"""
        self._select_spectra(lambda x, y: x == 0)

    def select_horiz(self):
        """Select all spectra listed horizontally"""
        self._select_spectra(lambda x, y: y == 0)

    def select_Q1(self):
        """Select all spectra listed in quadrant 1"""
        self._select_spectra(lambda x, y: x < 0 and y < 0)

    def select_Q2(self):
        """Select all spectra listed in quadrant 2"""
        self._select_spectra(lambda x, y: x < 0 and y > 0)

    def select_Q3(self):
        """Select all spectra listed in quadrant 3"""
        self._select_spectra(lambda x, y: x > 0 and y > 0)

    def select_Q4(self):
        """Select all spectra listed in quadrant 4 """
        self._select_spectra(lambda x, y: x > 0 and y < 0)

    def _select_spectra(self, condition):
        """
        Helper function to select spectra based on a given condition.
        """
        self.ui.spectra_listbox.clearSelection()  # Clear all selection
        item_count = self.ui.spectra_listbox.count()
        for i in range(item_count):
            item = self.ui.spectra_listbox.item(i)
            coord_str = item.text()
            x_coord, y_coord = map(float, coord_str.strip('()').split(','))
            if condition(x_coord, y_coord):
                item.setSelected(True)

    def get_spectrum_object(self):
        """
        Get the selected spectrum object from the UI.-

        Returns:
        - sel_spectrum: The primary selected spectrum object.
        - sel_spectra: List of all selected spectrum objects.

        """
        map_name, coords = self.spectra_id()
        sel_spectra = []
        for spectrum in self.spectrums:
            map_name_fs, coord_fs = self.spectrum_object_id(spectrum)
            if map_name_fs == map_name and coord_fs in coords:
                sel_spectra.append(spectrum)
        if len(sel_spectra) == 0:
            return
        sel_spectrum = sel_spectra[0]
        return sel_spectrum, sel_spectra

    def spectra_id(self):
        """
        Get selected spectra IDs from the GUI Maps and spectra listboxes.

        Returns:
        - map_name: Name of the selected maps.
        - coords: List of selected spectrum coordinates.

        """
        map_item = self.ui.maps_listbox.currentItem()
        if map_item is not None:
            map_name = map_item.text()
            selected_spectra = self.ui.spectra_listbox.selectedItems()
            coords = []
            if selected_spectra:
                for selected_item in selected_spectra:
                    text = selected_item.text()
                    x, y = map(float, text.strip('()').split(','))
                    coord = (x, y)
                    coords.append(coord)
            return map_name, coords
        return None, None

    def spectrum_object_id(self, spectrum=None):
        """
        Get the ID of a selected spectrum from a fitspy spectra object.

        Returns:
        - map_name_fs: Map name extracted from the spectrum object.
        - coord_fs: Coordinates extracted from the spectrum object.

        """
        fname_parts = spectrum.fname.split("_")
        map_name_fs = "_".join(fname_parts[:-1])
        coord_str = fname_parts[-1]  # Last part contains the coordinates
        coord_fs = tuple(
            map(float, coord_str.split('(')[1].split(')')[0].split(',')))
        return map_name_fs, coord_fs

    def refresh_gui(self):
        """Trigger a function to plot spectra after a delay"""
        self.delay_timer.start(100)

    def view_map_data(self):
        """View data of the selected map in the map list"""
        map_name, coords = self.spectra_id()
        view_df(self.ui.tabWidget, self.maps[map_name])

    def send_df_to_viz(self):
        """Send the collected spectral data dataframe to the visualization
        tab."""
        dfs_new = self.visu.original_dfs
        df_name = self.ui.ent_send_df_to_viz2.text()
        
        dfs_new[df_name] = self.df_fit_results
        self.visu.open_dfs(dfs=dfs_new, file_paths=None)

    def send_spectrum_to_compare(self):
        """
        Send selected spectra to the 'Spectrums' tab for comparison.
        """
        sel_spectrum, sel_spectra = self.get_spectrum_object()
        for spectrum in sel_spectra:
            sent_spectrum = deepcopy(spectrum)
            self.spectrums_tab.spectrums.append(sent_spectrum)
            self.spectrums_tab.upd_spectra_list()

    def cosmis_ray_detection(self):
        """Perform cosmic ray detection on the spectra data."""
        self.spectrums.outliers_limit_calculation()

    
    def view_stats(self):
        """
        Show statistical fitting results of the selected spectrum.
        """
        map_name, coords = self.spectra_id()
        selected_spectrums = []
        for spectrum in self.spectrums:
            map_name_fs, coord_fs = self.spectrum_object_id(spectrum)
            if map_name_fs == map_name and coord_fs in coords:
                selected_spectrums.append(spectrum)
        if len(selected_spectrums) == 0:
            return

        ui = self.ui.tabWidget
        title = f"Fitting Report - {map_name} - {coords}"
        # Show the 'report' of the first selected spectrum
        spectrum = selected_spectrums[0]
        if spectrum.result_fit:
            try:
                text = fit_report(spectrum.result_fit)
                self.common.view_text(ui, title, text)
            except:
                return


    def fitspy_launcher(self):
        """
        Launch FITSPY with selected spectra from the application.
        """
        if self.spectrums:
            plt.style.use('default')
            root = Tk()
            appli = Appli(root, force_terminal_exit=False)

            appli.spectra = self.spectrums
            for spectrum in appli.spectra:
                fname = spectrum.fname
                appli.fileselector.filenames.append(fname)
                appli.fileselector.lbox.insert(END, os.path.basename(fname))
            appli.fileselector.select_item(0)
            appli.update()
            root.mainloop()
        else:
            show_alert("No spectrum is loaded; FITSPY cannot open")
            return

    

    def set_x_range_handler(self):
        """
        Handler for setting X range based on keyboard modifiers.
        """
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.set_x_range_all()
        else:
            self.set_x_range()
    
    def paste_baseline_handler(self):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.paste_baseline_all()
        else:
            self.paste_baseline()

    def subtract_baseline_handler(self):
        """
        Handler for subtracting baseline based on keyboard modifiers.
        """
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.subtract_baseline_all()
        else:
            self.subtract_baseline()

    def clear_peaks_handler(self):
        """
        Handler for clearing peaks based on Ctrl key.
        """
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.clear_peaks_all()
        else:
            self.clear_peaks()

    def fit_fnc_handler(self):
        """
        Handler for fitting function based on Ctrl key.
        """
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.fit_all()
        else:
            self.fit()

    def apply_model_fnc_handler(self):
        """
        Handler for applying loaded fit model based on keyboard modifiers.
        """
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.apply_loaded_fit_model_all()
        else:
            self.apply_loaded_fit_model()

    def paste_fit_model_fnc_handler(self):
        """
        Handler for pasting fit model based on keyboard modifiers.
        """
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.paste_fit_model_all()
        else:
            self.paste_fit_model()

    def reinit_fnc_handler(self):
        """
        Handler for reinitializing fit function based on keyboard modifiers.
        """
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.reinit_all()
        else:
            self.reinit()
            
    def save_work(self):
        """Save the current application states to a JSON file."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(None,
                                                    "Save work",
                                                    "",
                                                    "SPECTROview Files (*.maps)")
            if file_path:
                spectrums_data = spectrum_to_dict(self.spectrums)
                data_to_save = {
                    'spectrums': spectrums_data,
                    'maps': {k: v.to_dict() for k, v in self.maps.items()},
                }

                with open(file_path, 'w') as f:
                    json.dump(data_to_save, f, indent=4)
                show_alert("Work saved successfully.")
        except Exception as e:
            show_alert(f"Error saving work: {e}")


    def load_work(self, file_path):
        """Load a previously saved application state from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                load = json.load(f)
                try:
                    self.spectrums = Spectra()
                    for spectrum_id, spectrum_data in load.get('spectrums', {}).items():
                        spectrum = Spectrum()
                        dict_to_spectrum(spectrum, spectrum_data)
                        spectrum.preprocess()
                        self.spectrums.append(spectrum)

                    self.maps = {k: pd.DataFrame(v) for k, v in
                                 load.get('maps', {}).items()}

                    QTimer.singleShot(300, self.collect_results)
                    self.upd_maps_list()

                except Exception as e:
                    show_alert(f"Error loading work: {e}")
        except Exception as e:
            show_alert(f"Error loading saved work (Maps Tab): {e}")
            
    def clear_env(self):
        """Clear the environment and reset the application state"""
        # Clear loaded maps and spectra
        self.maps.clear()
        self.spectrums = Spectra()
        self.loaded_fit_model = None
        self.current_fit_model = None
        self.df_fit_results = None

        # Clear UI elements that display data
        self.ui.maps_listbox.clear()
        self.ui.spectra_listbox.clear()
        self.ui.item_count_label.setText("0 points")

        # Clear spectra plot view and reset selected spectrums
        self.spectra_widget.sel_spectrums = None 
        self.spectra_widget.refresh_plot() 
        # Clear plot ofmeasurement sites 
        if hasattr(self, 'canvas2'):
            self.ax.clear()
            self.canvas.draw()

        self.df_table.clear()
        self.peak_table.clear()
        
        # Refresh the UI to reflect the cleared state
        QTimer.singleShot(50, self.spectra_widget.rescale)
        QTimer.singleShot(100, self.upd_maps_list)
        print("'Maps' Tab environment has been cleared.")
