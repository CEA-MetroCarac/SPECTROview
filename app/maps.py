"""This modules build the GUI and functionality of "MAPS" Tab""" 
import os
import time
import numpy as np
import pandas as pd
import json
import gzip
from io import StringIO

from copy import deepcopy
from pathlib import Path

from app.common import view_df, show_alert, spectrum_to_dict, dict_to_spectrum, baseline_to_dict, dict_to_baseline, save_df_to_excel
from app.common import FitThread, PeakTableWidget, DataframeTableWidget, \
    FitModelManager, CustomListWidget, SpectraViewWidget, MapViewWidget, Graph
from app.common import FIT_METHODS,PALETTE
from app.visualisation import MdiSubWindow
from lmfit import fit_report
from fitspy.spectra import Spectra
from fitspy.spectrum import Spectrum
from fitspy.app.gui import Appli
from fitspy.utils import closest_index

import matplotlib.pyplot as plt

from PySide6.QtWidgets import QFileDialog, QMessageBox, QApplication,QAbstractItemView, QListWidgetItem, QHBoxLayout, QLineEdit, QSpacerItem, QPushButton, QSizePolicy, QDialog, QVBoxLayout
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt, QFileInfo, QTimer, QObject
from tkinter import Tk, END

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
        self.peak_table = PeakTableWidget(self, self.ui.peak_table1, self.ui.cbb_layout_2)
        
         # Initialize Dataframe Table
        self.df_fit_results = None
        self.df_table = DataframeTableWidget(self.ui.layout_df_table)
        
        # Initialize QListWidget for spectra list
        self.ui.spectra_listbox = CustomListWidget()
        self.ui.spectra_listbox.setDragDropMode(QAbstractItemView.NoDragDrop)
        self.ui.listbox_layout2.addWidget(self.ui.spectra_listbox)
        self.ui.spectra_listbox.itemSelectionChanged.connect(self.refresh_gui)
        self.ui.checkBox_2.stateChanged.connect(self.check_uncheck_all)
        
        # Set a delay for the function "plot action"
        self.delay_timer = QTimer()
        self.delay_timer.setSingleShot(True)
        self.delay_timer.timeout.connect(self.plot)

        # Map_view_Widget
        self.map_plot = MapViewWidget(self)
        self.map_plot.spectra_listbox= self.ui.spectra_listbox
        self.ui.map_layout.addWidget(self.map_plot.widget)
        self.map_plot.btn_extract_profile.clicked.connect(self.plot_extracted_profile)
        
        ## Update spectra_listbox when selecting maps via MAPS LIST
        self.ui.maps_listbox.itemSelectionChanged.connect(self.upd_spectra_list)
        
        # BASELINE
        self.setup_baseline_controls()
        self.ui.cbb_fit_methods.addItems(FIT_METHODS)
        self.ui.btn_send_to_viz2.clicked.connect(self.send_df_to_viz)

        # Peak correction
        self.ui.btn_xrange_correction_2.clicked.connect(self.xrange_correction)
        self.ui.btn_undo_correction_2.clicked.connect(lambda: self.undo_xrange_correction())
        

        # FITMODEL FOLDER
        self.fit_model_manager = FitModelManager(self.settings)
        self.fit_model_manager.default_model_folder = self.settings.value(
            "default_model_folder", "")
        self.ui.l_defaut_folder_model.setText(
            self.fit_model_manager.default_model_folder)
        ## Show available fit models
        QTimer.singleShot(0, self.populate_available_models)
        self.ui.btn_refresh_model_folder.clicked.connect(
            self.populate_available_models)

    def setup_baseline_controls(self):
        self.ui.cb_attached.stateChanged.connect(self.refresh_gui)
        self.ui.noise.valueChanged.connect(self.refresh_gui)
        self.ui.rbtn_linear.clicked.connect(self.refresh_gui)
        self.ui.rbtn_polynomial.clicked.connect(self.refresh_gui)
        self.ui.degre.valueChanged.connect(self.refresh_gui)
        
        self.ui.cb_attached.stateChanged.connect(self.get_baseline_settings)
        self.ui.noise.valueChanged.connect(self.get_baseline_settings)
        self.ui.rbtn_linear.toggled.connect(self.get_baseline_settings)
        self.ui.rbtn_polynomial.clicked.connect(self.get_baseline_settings)
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
                            df = pd.read_csv(file_path, skiprows=2,
                                                 delimiter=";")
                            map_df = df.iloc[::2].reset_index(drop=True)
                            map_df.rename(columns={map_df.columns[0]: "X", map_df.columns[1]: "Y"}, inplace=True)

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
            self.create_spectrum_objects(map_df, map_name)
        self.upd_maps_list()

    def create_spectrum_objects(self, map_df, map_name):
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
                spectrum.is_corrected = False
                spectrum.correction_value = 0
                spectrum.baseline.mode = "Linear"
                spectrum.baseline.sigma = 5
                self.spectrums.append(spectrum)

    def xrange_correction(self, ref_value=None, sel_spectra=None):
        """Correct peak shift based on Si reference sample."""
        try:
            text = self.ui.ent_si_peak_2.text()
            ref_value = round(float(text), 3)                
            if sel_spectra is None:
                _, sel_spectra = self.get_spectrum_object()
            
            # Restore to original values
            self.undo_xrange_correction(sel_spectra)
            for spectrum in sel_spectra:
                # Correction action
                correction_value = (520.7 - ref_value)
                uncorrectted_x = deepcopy(spectrum.x)
                uncorrectted_x0 = deepcopy(spectrum.x0)
                spectrum.x = uncorrectted_x + correction_value
                spectrum.x0 = uncorrectted_x0 + correction_value
                spectrum.correction_value = correction_value
                spectrum.is_corrected = True

            QTimer.singleShot(100, self.upd_spectra_list)

        except ValueError:
            QMessageBox.warning(self.ui.tabWidget, "Input Error", "Please enter a valid numeric Si peak reference.")

    
    def undo_xrange_correction(self, sel_spectra=None):
        """Undo peak shift correction for the given spectra."""
        if sel_spectra is None:
            _, sel_spectra = self.get_spectrum_object()
        for spectrum in sel_spectra:
            if spectrum.is_corrected:
                # Restore original X values
                correctted_x=deepcopy(spectrum.x)
                correctted_x0=deepcopy(spectrum.x0)
                spectrum.x = correctted_x - spectrum.correction_value
                spectrum.x0 = correctted_x0 - spectrum.correction_value
                spectrum.correction_value = 0
                spectrum.is_corrected = False
        QTimer.singleShot(100, self.upd_spectra_list)

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
                    if hasattr(model, 'param_names') and hasattr(model,'param_hints'):
                        for param_name in model.param_names:
                            key = param_name.split('_')[1]
                            if key in model.param_hints and 'value' in \
                                    model.param_hints[key]:
                                fit_result[param_name] = model.param_hints[key]['value']

                if len(fit_result) > 3:
                    fit_results_list.append(fit_result)
        self.df_fit_results = pd.DataFrame(fit_results_list).round(3)

        if self.df_fit_results is not None and not self.df_fit_results.empty:
            # reindex columns according to the parameters names
            self.df_fit_results = self.df_fit_results.reindex(sorted(self.df_fit_results.columns), axis=1)
            names = []
            for name in self.df_fit_results.columns:
                if name in ["Filename", "X", 'Y']:
                    name = '0' + name
                elif '_' in name:
                    name = 'z' + name[5:]
                names.append(name)
            self.df_fit_results = self.df_fit_results.iloc[:,list(np.argsort(names, kind='stable'))]
            # Replace peak_label
            columns = [self.common.replace_peak_labels(self.current_fit_model, column) for column in self.df_fit_results.columns]
            self.df_fit_results.columns = columns

            map_type = self.map_plot.cbb_map_type.currentText()
            if map_type == 'Wafer':
                # DIAMETER
                diameter = float(self.map_plot.cbb_wafer_size.currentText())
                # QUADRANT
                self.df_fit_results['Quadrant'] = self.df_fit_results.apply(
                    self.common.quadrant, axis=1)
                # ZONE
                self.df_fit_results['Zone'] = self.df_fit_results.apply(
                    lambda row: self.common.zone(row, diameter), axis=1)
            else: 
                pass
        self.df_table.show(self.df_fit_results)
        
        self.map_plot.df_fit_results = self.df_fit_results
        self.map_plot.populate_z_values_cbb()
        
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
        QTimer.singleShot(100, self.upd_spectra_list)
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
        """Function to save fitted results in an Excel file with colored columns."""
        last_dir = self.settings.value("last_directory", "/")
        save_path, _ = QFileDialog.getSaveFileName(
            self.ui.tabWidget, "Save DF fit results", last_dir,
            "Excel Files (*.xlsx)"
        )
        
        # Call save_df_to_excel and handle feedback
        success, message = save_df_to_excel(save_path, self.df_fit_results)
        if success:
            QMessageBox.information(self.ui.tabWidget, "Success", message)
        else:
            QMessageBox.warning(self.ui.tabWidget, "Warning", message)
    
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
        else:
            # Extract map_name and coords from fnames
            map_name = None
            coords = []
            for fname in fnames:
                parts = fname.split('_')
                if map_name is None:
                    map_name = parts[0]  # Assume map_name is the first part
                coords.append('_'.join(parts[1:]))
            
        selected_spectrums = []
        for spectrum in self.spectrums:
            map_name_fs, coord_fs = self.spectrum_object_id(spectrum)
            
            if map_name_fs == map_name and coord_fs in coords:
                selected_spectrums.append(spectrum)

        # Restore spectrums if they were x-range corrected
        self.undo_xrange_correction(selected_spectrums)

        self.common.reinit_spectrum(fnames, self.spectrums)
        self.upd_spectra_list()
        QTimer.singleShot(200, self.spectra_widget.rescale)

    def reinit_all(self):
        """Reinitialize all spectra"""
        self.undo_xrange_correction(self.spectrums)
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
        
        df = self.maps.get(map_name)
        self.map_plot.map_df_name=map_name
        self.map_plot.map_df=df
        self.map_plot.plot(coords)

        # Show correction value of the last selected item
        correction_value = round(selected_spectrums[-1].correction_value, 3)
        text = f"[{correction_value}]"
        self.ui.lbl_correction_value_2.setText(text)

        self.read_x_range()
        self.peak_table.show(selected_spectrums[0])
        
    def upd_spectra_list(self):
        """Show spectrums in a listbox"""
        
        #Update the Xrange slider based on selected map
        map_name, _ = self.spectra_id()
        map_df = self.maps.get(map_name)

        if map_df is not None:
            self.map_plot.map_df_name=map_name
            self.map_plot.map_df=map_df
            column_labels = map_df.columns[2:-1].astype(float)
            min_value = float(column_labels.min())
            max_value = float(column_labels.max())
            self.map_plot.update_xrange_slider(min_value, max_value)

        checked_states = {}
        for index in range(self.ui.spectra_listbox.count()):
            item = self.ui.spectra_listbox.item(index)
            checked_states[item.text()] = item.checkState()
          
        self.current_row = self.ui.spectra_listbox.currentRow()
        current_selection = [self.ui.spectra_listbox.item(i).text() for i in range(self.ui.spectra_listbox.count()) if self.ui.spectra_listbox.item(i).isSelected()]

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
                    
                    if spectrum.baseline.is_subtracted:
                        if not hasattr(spectrum.result_fit, 'success'):
                            item.setBackground(QColor("red"))
                        elif spectrum.result_fit.success:
                            item.setBackground(QColor("green"))
                        else:
                            item.setBackground(QColor("orange"))
                    else:
                        item.setBackground(QColor(0, 0, 0, 0))
                    self.ui.spectra_listbox.addItem(item)

        # Update the item count label
        item_count = self.ui.spectra_listbox.count()
        self.ui.item_count_label.setText(f"{item_count} points")

        # Restore selection states
        for index in range(self.ui.spectra_listbox.count()):
            item = self.ui.spectra_listbox.item(index)
            if item.text() in current_selection:
                item.setSelected(True)

        # Reselect the previously selected item
        if self.current_row >= 0 and self.current_row < item_count:
            self.ui.spectra_listbox.setCurrentRow(self.current_row)
        else:
            if self.ui.spectra_listbox.count() > 0:
                self.ui.spectra_listbox.setCurrentRow(0)

        QTimer.singleShot(50, self.refresh_gui)

    def upd_maps_list(self):
        """Update the Maps listbox"""
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
        self.map_plot.ax.clear()
        self.map_plot.canvas.draw_idle()

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
        """Get the selected spectrum(s) object"""
        map_name, coords = self.spectra_id()
        sel_spectra = []
        for spectrum in self.spectrums:
            map_name_fs, coord_fs = self.spectrum_object_id(spectrum)
            if map_name_fs == map_name and coord_fs in coords:
                sel_spectra.append(spectrum)
        if len(sel_spectra) == 0:
            return
        sel_spectrum = sel_spectra[0]
        if sel_spectrum is not None and sel_spectra:
            return sel_spectrum, sel_spectra
        else:
            return None, None

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

    def view_map_df(self):
        """View data of the selected map in the map list"""
        map_name, coords = self.spectra_id()
        view_df(self.ui.tabWidget, self.maps[map_name], simplified_df=True, fill_colors=False)

    def send_df_to_viz(self):
        """Send the collected spectral data dataframe to the visualization
        tab."""
        dfs_new = self.visu.original_dfs
        df_name = self.ui.ent_send_df_to_viz2.text()
        
        dfs_new[df_name] = self.df_fit_results
        self.visu.open_dfs(dfs=dfs_new, file_paths=None)
    
    def plot_extracted_profile(self):
        """Extract profile from map plot and Plot in VIS TAB"""
        
        profile_name = self.map_plot.profile_name.text()
        profil_df = self.map_plot.extract_profile()
        
        if profil_df is not None and profile_name is not None:
            dfs_new = self.visu.original_dfs
            dfs_new[profile_name] = profil_df
            self.visu.open_dfs(dfs=dfs_new, file_paths=None)

            #Add a line plot
            available_ids = [i for i in range(1, len(self.visu.plots) + 2) if i not in self.visu.plots]
            graph_id = min(available_ids) if available_ids else len(self.visu.plots) + 1
            graph = Graph(graph_id=graph_id)
            self.visu.plots[graph.graph_id] = graph

            graph.plot_style='line'
            graph.df_name = profile_name
            graph.x = 'distance'
            graph.y = ['values']
            graph.z = None
            graph.create_plot_widget(100)
            graph_dialog = QDialog(self.visu)
            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(graph)
            graph_dialog.setLayout(layout)
            graph_dialog.setContentsMargins(2, 2, 2, 0)

            # Add the QDialog to a QMdiSubWindow
            sub_window = MdiSubWindow(graph_id, self.ui.lbl_figsize,mdi_area=self.ui.mdiArea)
            sub_window.setWidget(graph_dialog)
            sub_window.closed.connect(self.visu.delete_graph)
            sub_window.resize(graph.plot_width, graph.plot_height)
            self.ui.mdiArea.addSubWindow(sub_window)
            sub_window.show()
            self.visu.add_graph_list_to_combobox()
            text = f"{graph.graph_id}-{graph.plot_style}_plot: {{'distance'}} - {{'values'}} - {{'None'}}"
            graph_dialog.setWindowTitle(text)
            # Plot action
            QTimer.singleShot(100, self.visu.plot_action)
            QTimer.singleShot(200, self.visu.customize_legend)
            self.ui.tabWidget.setCurrentWidget(self.ui.tab_graphs)

            self.map_plot.options_menu.close() #Close option menu
        else:
            msg="Profile extraction failed or insufficient points selected."
            show_alert(msg)

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
        """Save current results to a JSON file."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(None,
                                                    "Save work",
                                                    "",
                                                    "SPECTROview Files (*.maps)")
            if file_path:
                spectrums_data = spectrum_to_dict(self.spectrums, is_map=True)
                
                compressed_maps = {}
                for k, v in self.maps.items():
                    # Convert DataFrame to a CSV string and compress it
                    compressed_data = v.to_csv(index=False).encode('utf-8')
                    compressed_maps[k] = gzip.compress(compressed_data)

                data_to_save = {
                    'spectrums_data': spectrums_data,
                    'maps': {k: v.hex() for k, v in compressed_maps.items()},
                }
                with open(file_path, 'w') as f:
                    json.dump(data_to_save, f, indent=4)
                show_alert("Work saved successfully.")
        except Exception as e:
            show_alert(f"Error saving work: {e}")

    def load_work(self, file_path):
        """Load a saved results from a JSON file with compressed DataFrames."""
        try:
            with open(file_path, 'r') as f:
                load = json.load(f)
                try:
                    self.spectrums = Spectra()
                    self.maps = {}
                    # Decode hex and decompress the dataframe
                    for k, v in load.get('maps', {}).items():
                        compressed_data = bytes.fromhex(v)
                        csv_data = gzip.decompress(compressed_data).decode('utf-8')
                        self.maps[k] = pd.read_csv(StringIO(csv_data)) 
                        
                    for spectrum_id, spectrum_data in load.get('spectrums_data', {}).items():
                        spectrum = Spectrum()
                        dict_to_spectrum(spectrum=spectrum, spectrum_data=spectrum_data, maps=self.maps, is_map=True)
                        spectrum.preprocess()
                        self.spectrums.append(spectrum)

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
        self.map_plot.ax.clear()
        self.map_plot.canvas.draw()

        self.df_table.clear()
        self.peak_table.clear()
        
        # Refresh the UI to reflect the cleared state
        QTimer.singleShot(50, self.spectra_widget.rescale)
        QTimer.singleShot(100, self.upd_maps_list)
        print("'Maps' Tab environment has been cleared.")
