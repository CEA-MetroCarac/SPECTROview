import os
import time
import numpy as np
import pandas as pd
import json
from copy import deepcopy
from pathlib import Path

from common import view_df, show_alert, spectrum_to_dict, set_attributes, \
    clear_layout
from common import FitThread, WaferPlot, ShowParameters, DataframeTable, \
    FitModelManager, Filter
from common import FIT_METHODS, NCPUS, PLOT_POLICY

from lmfit import fit_report
from fitspy.spectra import Spectra
from fitspy.spectrum import Spectrum
from fitspy.app.gui import Appli
from fitspy.utils import closest_index

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from PySide6.QtWidgets import (QFileDialog, QMessageBox, QApplication,
                               QListWidgetItem, QCheckBox)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt, QFileInfo, QTimer, QObject, Signal
from tkinter import Tk, END


class Maps(QObject):
    """
    Class manages the GUI interactions and operations related to spectra
    fittings,
    and visualization of fitted data within 'Maps' TAB of the application.

    Attributes:
        settings (QSettings):
            Application settings object.
        ui (QWidget):
            Main user interface widget.
        visu (Visualization):
            Visualization module instance.
        spectrums_tab (Spectra):
            Collection of opened spectra.
        common (CommonFunctions):
            Utility functions and common operations.

        maps (dict):
            Dictionary of opened maps containing raw spectra data.
        loaded_fit_model (object):
            Currently loaded fit model.
        current_fit_model (object):
            Fit model currently applied for fitting operations.

        spectrums (Spectra):
            Instance of the Spectra class for managing spectrum objects.
        df_fit_results (pd.DataFrame):
            DataFrame containing fit results.
        filter (Filter):
            Instance of the Filter class for data filtering operations.
        filtered_df (pd.DataFrame):
            DataFrame containing filtered fit results.
        delay_timer (QTimer):
            Timer for delaying plot updates.
        plot_styles (list):
            List of available plot styles.
        zoom_pan_active (bool):
            Flag indicating whether zoom and pan operations are active.
        available_models (list):
            List of available fit model names.
        fit_model_manager (FitModelManager):
            Manager for handling fit model operations.
    """

    def __init__(self, settings, ui, spectrums, common, visu):
        super().__init__()
        self.settings = settings
        self.ui = ui
        self.visu = visu
        self.spectrums_tab = spectrums
        self.common = common

        self.maps = {}  # list of opened maps data
        self.toolbar = None
        self.loaded_fit_model = None
        self.current_fit_model = None
        self.spectrums = Spectra()
        self.df_fit_results = None

        # FILTER: Create an instance of the FILTER class
        self.filter = Filter(self.ui.ent_filter_query_3,
                             self.ui.filter_listbox_2,
                             self.df_fit_results)
        self.filtered_df = None
        # Connect filter signals to filter methods
        self.ui.btn_add_filter_3.clicked.connect(self.filter.add_filter)
        self.ui.ent_filter_query_3.returnPressed.connect(self.filter.add_filter)
        self.ui.btn_remove_filters_3.clicked.connect(self.filter.remove_filter)
        self.ui.btn_apply_filters_3.clicked.connect(self.apply_filters)

        # Update spectra_listbox when selecting maps via MAPS LIST
        self.ui.maps_listbox.itemSelectionChanged.connect(
            self.upd_spectra_list)

        # Connect and plot_spectra of selected SPECTRUM LIST
        self.ui.spectra_listbox.itemSelectionChanged.connect(self.refresh_gui)

        # Connect the stateChanged signal of the legend CHECKBOX
        self.ui.cb_legend.stateChanged.connect(self.refresh_gui)
        self.ui.cb_raw.stateChanged.connect(self.refresh_gui)
        self.ui.cb_bestfit.stateChanged.connect(self.refresh_gui)
        self.ui.cb_colors.stateChanged.connect(self.refresh_gui)
        self.ui.cb_residual.stateChanged.connect(self.refresh_gui)
        self.ui.cb_filled.stateChanged.connect(self.refresh_gui)
        self.ui.cb_peaks.stateChanged.connect(self.refresh_gui)
        self.ui.cb_attached.stateChanged.connect(self.refresh_gui)
        self.ui.cb_normalize.stateChanged.connect(self.refresh_gui)

        self.ui.cbb_wafer_size.currentIndexChanged.connect(self.refresh_gui)
        self.ui.cbb_xaxis_unit2.currentIndexChanged.connect(self.refresh_gui)
        self.ui.rdbt_show_wafer.toggled.connect(self.refresh_gui)

        self.ui.cb_limits.stateChanged.connect(self.refresh_gui)
        self.ui.cb_expr.stateChanged.connect(self.refresh_gui)
        # Set a delay for the function "plot1"
        self.delay_timer = QTimer()
        self.delay_timer.setSingleShot(True)
        self.delay_timer.timeout.connect(self.plot1)

        self.plot_styles = ["box plot", "point plot", "bar plot"]
        self.create_plot_widget()
        self.create_spectra_plot_widget()
        self.zoom_pan_active = False

        self.ui.cbb_fit_methods.addItems(FIT_METHODS)
        self.ui.cbb_cpu_number.addItems(NCPUS)

        # FIT SETTINGS
        self.load_fit_settings()
        self.ui.cb_fit_negative.stateChanged.connect(self.save_fit_settings)
        self.ui.max_iteration.valueChanged.connect(self.save_fit_settings)
        self.ui.cbb_fit_methods.currentIndexChanged.connect(
            self.save_fit_settings)
        self.ui.cbb_cpu_number.currentIndexChanged.connect(
            self.save_fit_settings)
        self.ui.xtol.textChanged.connect(self.save_fit_settings)
        self.ui.sb_dpi_spectra.valueChanged.connect(
            self.create_spectra_plot_widget)

        # BASELINE
        self.ui.cb_attached.clicked.connect(self.upd_spectra_list)
        self.ui.noise.valueChanged.connect(self.upd_spectra_list)
        self.ui.rbtn_linear.clicked.connect(self.upd_spectra_list)
        self.ui.rbtn_polynomial.clicked.connect(self.upd_spectra_list)
        self.ui.degre.valueChanged.connect(self.upd_spectra_list)

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
            fname = f"{map_name}_{coord}"
            if not any(spectrum.fname == fname for spectrum in self.spectrums):
                spectrum = Spectrum()
                spectrum.fname = fname
                spectrum.x = np.asarray(x_values)
                spectrum.x0 = np.asarray(x_values)
                spectrum.y = np.asarray(y_values)
                spectrum.y0 = np.asarray(y_values)
                self.spectrums.append(spectrum)

    def process_new_format(self, map_df, map_name):
        """Process new format wafer dataframe."""
        for i in range(0, len(map_df), 2):
            coord_row = map_df.iloc[i]
            intensity_row = map_df.iloc[i + 1]
            coord = (coord_row.iloc[0], coord_row.iloc[1])
            x_values = coord_row.iloc[2:].tolist()
            x_values = pd.to_numeric(x_values, errors='coerce').tolist()
            y_values = intensity_row.iloc[2:].tolist()
            fname = f"{map_name}_{coord}"
            if not any(spectrum.fname == fname for spectrum in self.spectrums):
                spectrum = Spectrum()
                spectrum.fname = fname
                spectrum.x = np.asarray(x_values)
                spectrum.x0 = np.asarray(x_values)
                spectrum.y = np.asarray(y_values)
                spectrum.y0 = np.asarray(y_values)
                self.spectrums.append(spectrum)

    def apply_filters(self):
        """
        Apply all checked filters to the current selected dataframe.
        """
        self.filter.set_dataframe(self.df_fit_results)
        self.filtered_df = self.filter.apply_filters()
        self.display_df_in_GUI(self.filtered_df)

    def view_fit_results_df(self):
        """
        Checks if a filtered dataframe exists and displays it in
        a table in the GUI.
        If no filtered dataframe exists, it displays an alert indicating no
        data is available.
        """
        if self.filtered_df is None:
            df = self.df_fit_results
        else:
            df = self.filtered_df

        if df is not None:
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

            # QUADRANT
            self.df_fit_results['Quadrant'] = self.df_fit_results.apply(
                self.common.quadrant, axis=1)
            # DIAMETER
            diameter = float(self.ui.wafer_size.text())

            # ZONE
            self.df_fit_results['Zone'] = self.df_fit_results.apply(
                lambda row: self.common.zone(row, diameter), axis=1)

        self.display_df_in_GUI(self.df_fit_results)

        self.filtered_df = self.df_fit_results
        self.upd_cbb_param()
        self.upd_cbb_map()
        self.send_df_to_viz()

    def display_df_in_GUI(self, df):
        """Display a given df in the GUI via QTableWidget"""
        df_table = DataframeTable(df, self.ui.layout_df_table)

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
            spectrum.attractors_calculation()
        QTimer.singleShot(50, self.upd_spectra_list)
        QTimer.singleShot(300, self.rescale)

    def set_x_range_all(self):
        """Set new x range for all spectrum"""
        fnames = self.spectrums.fnames
        self.set_x_range(fnames=fnames)

    def on_click(self, event):
        """
        On click action to add a "peak models" or "baseline points"
        """
        sel_spectrum, sel_spectra = self.get_spectrum_object()
        fit_model = self.ui.cbb_fit_models.currentText()
        # Add a new peak_model for current selected peak
        if self.zoom_pan_active == False and self.ui.rdbtn_peak.isChecked():
            if event.button == 1 and event.inaxes:
                x = event.xdata
                y = event.ydata
            sel_spectrum.add_peak_model(fit_model, x)
            self.upd_spectra_list()

        # Add a new baseline point for current selected peak
        if self.zoom_pan_active == False and self.ui.rdbtn_baseline.isChecked():
            if event.button == 1 and event.inaxes:
                x1 = event.xdata
                y1 = event.ydata
                if sel_spectrum.baseline.is_subtracted:
                    show_alert(
                        "Already subtracted before. Reinitialize spectrum to "
                        "perform new baseline")
                else:
                    sel_spectrum.baseline.add_point(x1, y1)
                    self.upd_spectra_list()

    def get_baseline_settings(self):
        """Get baseline settings from GUI and set to spectrum objects"""
        sel_spectrum, sel_spectra = self.get_spectrum_object()
        if sel_spectrum is None:
            return
        sel_spectrum.baseline.attached = self.ui.cb_attached.isChecked()
        sel_spectrum.baseline.sigma = self.ui.noise.value()
        if self.ui.rbtn_linear.isChecked():
            sel_spectrum.baseline.mode = "Linear"
        else:
            sel_spectrum.baseline.mode = "Polynomial"
            sel_spectrum.baseline.order_max = self.ui.degre.value()

    def plot_baseline_dynamically(self, ax, spectrum):
        """Evaluate and plot baseline points and line dynamically"""
        self.get_baseline_settings()
        if not spectrum.baseline.is_subtracted:
            x_bl = spectrum.x
            y_bl = spectrum.y if spectrum.baseline.attached else None
            if len(spectrum.baseline.points[0]) == 0:
                return
            # Clear any existing baseline plot
            for line in ax.lines:
                if line.get_label() == "Baseline":
                    line.remove()
            # Evaluate the baseline
            baseline_values = spectrum.baseline.eval(x_bl, y_bl)
            ax.plot(x_bl, baseline_values, 'r')
            # Plot the attached baseline points
            if spectrum.baseline.attached and y_bl is not None:
                attached_points = spectrum.baseline.attach_points(x_bl, y_bl)
                ax.plot(attached_points[0], attached_points[1], 'ko',
                        mfc='none')
            else:
                ax.plot(spectrum.baseline.points[0],
                        spectrum.baseline.points[1], 'ko', mfc='none', ms=5)

    def subtract_baseline(self, sel_spectra=None):
        """Subtract baseline for the selected spectrum(s)."""
        sel_spectrum, _ = self.get_spectrum_object()
        points = deepcopy(sel_spectrum.baseline.points)
        mode = sel_spectrum.baseline.mode  
        coef = sel_spectrum.baseline.coef  
        distance = sel_spectrum.baseline.distance
        sigma = sel_spectrum.baseline.sigma 
        attached = sel_spectrum.baseline.attached 
        is_subtracted = sel_spectrum.baseline.is_subtracted
        
        if len(points[0]) == 0:
            return
        if sel_spectra is None:
            _, sel_spectra = self.get_spectrum_object()
        for spectrum in sel_spectra:
            spectrum.baseline.points = points.copy()
            spectrum.baseline.mode = mode
            spectrum.baseline.coef = coef
            spectrum.baseline.distance = distance
            spectrum.baseline.sigma = sigma
            spectrum.baseline.attached = attached 
            spectrum.baseline.is_subtracted = is_subtracted
            
            spectrum.subtract_baseline()
        QTimer.singleShot(50, self.upd_spectra_list)
        QTimer.singleShot(300, self.rescale)

    def subtract_baseline_all(self):
        """Subtracts baseline points for all spectra"""
        self.subtract_baseline(self.spectrums)

    def get_fit_settings(self):
        """Get all settings for the fitting action."""
        sel_spectrum, sel_spectra = self.get_spectrum_object()
        fit_params = sel_spectrum.fit_params.copy()
        fit_params['fit_negative'] = self.ui.cb_fit_negative.isChecked()
        fit_params['max_ite'] = self.ui.max_iteration.value()
        fit_params['method'] = self.ui.cbb_fit_methods.currentText()
        fit_params['ncpus'] = self.ui.cbb_cpu_number.currentText()
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
        self.progress_timer.start(400)

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
        self.progress_timer.start(400)

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
        QTimer.singleShot(200, self.rescale)
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

    def load_fit_results(self, file_paths=None):
        """Load fitted results from an Excel file"""

        # Initialize the last used directory from QSettings
        last_dir = self.settings.value("last_directory", "/")
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_paths, _ = QFileDialog.getOpenFileNames(
            self.ui.tabWidget, "Open File(s)", last_dir,
            "Excel Files (*.xlsx *.xls)", options=options)
        # Load dataframes from Excel files
        if file_paths:
            last_dir = QFileInfo(file_paths[0]).absolutePath()
            self.settings.setValue("last_directory", last_dir)
            # Load DataFrame from the first selected Excel file
            excel_file_path = file_paths[0]
            try:
                dfr = pd.read_excel(excel_file_path)
                self.df_fit_results = None
                self.df_fit_results = dfr
                self.filtered_df = dfr
            except Exception as e:
                show_alert("Error loading DataFrame:", e)

        self.display_df_in_GUI(self.df_fit_results)

        self.upd_cbb_param()
        self.upd_cbb_map()
        self.send_df_to_viz()

    def upd_cbb_map(self):
        """Update the combobox with unique values from 'Map' column"""
        self.ui.cbb_wafer_1.clear()
        try:
            map_names = self.df_fit_results['Filename'].unique()
            for map_name in map_names:
                self.ui.cbb_wafer_1.addItem(map_name)
        except Exception as e:
            print(f"Error updating combobox with 'Filename' values: {e}")

    def upd_cbb_param(self):
        """Update comboboxes with all values of `df_fit_results`"""

        if self.df_fit_results is not None:
            columns = self.df_fit_results.columns.tolist()
            self.ui.cbb_param_1.clear()
            self.ui.cbb_x.clear()
            self.ui.cbb_y.clear()
            self.ui.cbb_z.clear()
            self.ui.cbb_x.addItem("None")
            self.ui.cbb_y.addItem("None")
            self.ui.cbb_z.addItem("None")
            for column in columns:
                self.ui.cbb_param_1.addItem(column)
                self.ui.cbb_x.addItem(column)
                self.ui.cbb_y.addItem(column)
                self.ui.cbb_z.addItem(column)

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
        # Check if filters are applied
        if self.filtered_df is None:
            df = self.df_fit_results
        else:
            df = self.filtered_df
        self.display_df_in_GUI(df)
        self.send_df_to_viz()
        self.upd_cbb_param()
        self.upd_cbb_map()

    def reinit(self, fnames=None):
        """Reinitialize the selected spectrum(s)."""
        if fnames is None:
            map_name, coords = self.spectra_id()
            fnames = [f"{map_name}_{coord}" for coord in coords]
        self.common.reinit_spectrum(fnames, self.spectrums)
        self.upd_spectra_list()
        QTimer.singleShot(200, self.rescale)

    def reinit_all(self):
        """Reinitialize all spectra"""
        fnames = self.spectrums.fnames
        self.reinit(fnames)

    def rescale(self):
        """Rescale the spectra plot"""
        self.ax.autoscale()
        self.canvas1.draw()

    def create_spectra_plot_widget(self):
        """
        Create canvas and toolbar for plotting in the GUI.

        Action:
        - Sets the plot style.
        - Clears existing layouts in the UI.
        - Updates the spectra list in the UI.
        - Creates a new FigureCanvas for plotting.
        - Configures plot elements such as axes labels, grid, and connections.
        - Adds canvas and toolbar to the UI layout.
        - Draws the plot on the canvas.
        """
        plt.style.use(PLOT_POLICY)
        self.common.clear_layout(self.ui.QVBoxlayout.layout())
        self.common.clear_layout(self.ui.toolbar_frame.layout())
        self.upd_spectra_list()
        dpi = float(self.ui.sb_dpi_spectra.text())

        fig1 = plt.figure(dpi=dpi)
        self.ax = fig1.add_subplot(111)
        txt = self.ui.cbb_xaxis_unit2.currentText()
        self.ax.set_xlabel(txt)
        self.ax.set_ylabel("Intensity (a.u)")
        self.ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
        self.canvas1 = FigureCanvas(fig1)
        self.canvas1.mpl_connect('button_press_event', self.on_click)

        # Toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas1)
        for action in self.toolbar.actions():
            if action.text() in ['Save', 'Pan', 'Back', 'Forward', 'Subplots']:
                action.setVisible(False)
            if action.text() == 'Pan' or action.text() == 'Zoom':
                action.toggled.connect(self.toggle_zoom_pan)

        rescale = next(
            a for a in self.toolbar.actions() if a.text() == 'Home')
        rescale.triggered.connect(self.rescale)

        self.ui.QVBoxlayout.addWidget(self.canvas1)
        self.ui.toolbar_frame.addWidget(self.toolbar)
        self.canvas1.figure.tight_layout()
        self.canvas1.draw()

    def plot1(self):
        """Plot selected spectra"""
        map_name, coords = self.spectra_id()  # current selected spectra ID
        selected_spectrums = []

        for spectrum in self.spectrums:
            map_name_fs, coord_fs = self.spectrum_object_id(spectrum)
            if map_name_fs == map_name and coord_fs in coords:
                selected_spectrums.append(spectrum)

        # Only plot 10 first spectra to advoid crash
        selected_spectrums = selected_spectrums[:50]
        if len(selected_spectrums) == 0:
            return

        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.clear()

        # reassign previous axis limits (related to zoom)
        if not xlim == ylim == (0.0, 1.0):
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

        for spectrum in selected_spectrums:
            fname, coord = self.spectrum_object_id(spectrum)
            x_values = spectrum.x
            y_values = spectrum.y

            # NORMALIZE
            if self.ui.cb_normalize.isChecked():
                max_intensity = 0.0
                max_intensity = max(max_intensity, max(spectrum.y))
                y_values = y_values / max_intensity
            self.ax.plot(x_values, y_values, label=f"{coord}", ms=3, lw=2)

            # BASELINE
            self.plot_baseline_dynamically(ax=self.ax, spectrum=spectrum)

            # RAW
            if self.ui.cb_raw.isChecked():
                x0_values = spectrum.x0
                y0_values = spectrum.y0
                self.ax.plot(x0_values, y0_values, 'ko-', label='raw', ms=3,
                             lw=1)
            # Background
            y_bkg = np.zeros_like(x_values)
            if spectrum.bkg_model is not None:
                y_bkg = spectrum.bkg_model.eval(
                    spectrum.bkg_model.make_params(), x=x_values)

            # BEST-FIT and PEAK_MODELS
            y_peaks = np.zeros_like(x_values)
            if self.ui.cb_bestfit.isChecked():
                peak_labels = spectrum.peak_labels
                for i, peak_model in enumerate(spectrum.peak_models):
                    peak_label = peak_labels[i]

                    # remove temporarily 'expr'
                    param_hints_orig = deepcopy(peak_model.param_hints)
                    for key, _ in peak_model.param_hints.items():
                        peak_model.param_hints[key]['expr'] = ''
                    params = peak_model.make_params()
                    # rassign 'expr'
                    peak_model.param_hints = param_hints_orig
                    y_peak = peak_model.eval(params, x=x_values)
                    y_peaks += y_peak
                    if self.ui.cb_filled.isChecked():
                        self.ax.fill_between(x_values, 0, y_peak, alpha=0.5,
                                             label=f"{peak_label}")
                        if self.ui.cb_peaks.isChecked():
                            position = peak_model.param_hints['x0']['value']
                            intensity = peak_model.param_hints['ampli']['value']
                            position = round(position, 2)
                            text = f"{peak_label}\n({position})"
                            self.ax.text(position, intensity, text,
                                         ha='center', va='bottom',
                                         color='black', fontsize=12)
                    else:

                        self.ax.plot(x_values, y_peak, '--',
                                     label=f"{peak_label}")
                if hasattr(spectrum.result_fit,
                           'success') and self.ui.cb_bestfit.isChecked():
                    y_fit = y_bkg + y_peaks
                    self.ax.plot(x_values, y_fit, label=f"bestfit")
            # RESIDUAL
            if hasattr(spectrum.result_fit,
                       'residual') and self.ui.cb_residual.isChecked():
                residual = spectrum.result_fit.residual
                self.ax.plot(x_values, residual, 'ko-', ms=3, label='residual')

            if self.ui.cb_colors.isChecked() is False:
                self.ax.set_prop_cycle(None)
            # R-SQUARED
            if hasattr(spectrum.result_fit, 'rsquared'):
                rsquared = round(spectrum.result_fit.rsquared, 4)
                self.ui.rsquared_1.setText(f"R2={rsquared}")
            else:
                self.ui.rsquared_1.setText("R2=0")

        # self.ax.set_xlabel("Raman shift (cm$^{-1}$)")
        txt = self.ui.cbb_xaxis_unit2.currentText()
        self.ax.set_xlabel(txt)
        self.ax.set_ylabel("Intensity (a.u)")
        if self.ui.cb_legend.isChecked():
            self.ax.legend(loc='upper right')
        self.ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
        self.ax.get_figure().tight_layout()

        self.canvas1.draw()
        self.plot2()
        self.read_x_range()
        self.show_peak_table()

    def create_plot_widget(self):
        """
        Create plot widgets for other plots: measurement sites,
        waferdataview, plotview.
        """
        # plot 2: Measurement sites view
        fig2 = plt.figure(dpi=100)
        self.ax2 = fig2.add_subplot(111)
        self.ax2.spines['right'].set_visible(False)
        self.ax2.spines['top'].set_visible(False)
        self.ax2.spines['left'].set_visible(False)
        self.ax2.tick_params(axis='x', which='both', bottom=True, top=False)
        self.ax2.tick_params(axis='y', which='both', right=False, left=False)
        self.canvas2 = FigureCanvas(fig2)

        # Variables to keep track of highlighted points and Ctrl key status
        self.selected_points = []
        self.ctrl_pressed = False
        # Connect the mouse and key events to the handler functions
        fig2.canvas.mpl_connect('button_press_event',
                                self.on_click_sites_mesurements)
        fig2.canvas.mpl_connect('key_press_event', self.on_key_press)
        fig2.canvas.mpl_connect('key_release_event', self.on_key_release)
        layout = self.ui.measurement_sites.layout()
        layout.addWidget(self.canvas2)
        self.canvas2.draw()

        # plot4: graph
        fig4 = plt.figure(dpi=90)
        self.ax4 = fig4.add_subplot(111)
        self.canvas4 = FigureCanvas(fig4)
        self.ui.frame_graph.addWidget(self.canvas4)
        self.canvas4.draw()

    def on_click_sites_mesurements(self, event):
        """
        To select the measurement points directly in the plot.

        Parameters:
        - event (MouseEvent): The mouse event object.

        Action:
        - Retrieves all measurement sites coordinates.
        - Clears selection in the spectra listbox.
        - Selects nearest measurement point based on mouse click.
        - Handles Ctrl key to allow multi-selection of points.
        - Updates selected points in the spectra listbox.
        """
        all_x, all_y = self.get_mes_sites_coord()
        self.ui.spectra_listbox.clearSelection()
        if event.inaxes == self.ax2:
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

    def plot2(self):
        """
        Plot 2D maps of measurement points
        """
        r = int(self.ui.cbb_wafer_size.currentText()) / 2

        self.ax2.clear()
        if self.ui.rdbt_show_wafer.isChecked():
            wafer_circle = patches.Circle((0, 0), radius=r, fill=False,
                                          color='black', linewidth=1)
            self.ax2.add_patch(wafer_circle)
            self.ax2.set_yticklabels([])

        all_x, all_y = self.get_mes_sites_coord()
        self.ax2.scatter(all_x, all_y, marker='x', color='gray', s=10)

        map_name, coords = self.spectra_id()
        if coords:
            x, y = zip(*coords)
            self.ax2.scatter(x, y, marker='o', color='red', s=40)

        self.ax2.grid(True, linestyle='--', linewidth=0.5, color='gray')
        self.ax2.get_figure().tight_layout()
        self.canvas2.draw()

    def plot3(self):
        """
        Plot WaferDataFrame.
        """
        self.common.clear_layout(self.ui.frame_wafer.layout())
        dfr = self.df_fit_results
        map_name = self.ui.cbb_wafer_1.currentText()
        color = self.ui.cbb_color_pallete.currentText()
        wafer_size = float(self.ui.wafer_size.text())

        if map_name is not None:
            selected_df = dfr.query('Filename == @map_name')
        sel_param = self.ui.cbb_param_1.currentText()
        self.canvas3 = self.plot3_action(selected_df, sel_param, wafer_size,
                                         color)

        self.ui.frame_wafer.addWidget(self.canvas3)

    def plot3_action(self, selected_df, sel_param, wafer_size, color):
        """
        Plot wafer map of a selected parameter.
        """
        x = selected_df['X']
        y = selected_df['Y']
        param = selected_df[sel_param]
        vmin = float(
            self.ui.int_vmin.text()) if self.ui.int_vmin.text() else None
        vmax = float(
            self.ui.int_vmax.text()) if self.ui.int_vmax.text() else None
        stats = self.ui.cb_stats.isChecked()

        plt.close('all')
        fig = plt.figure(dpi=80)
        ax = fig.add_subplot(111)

        wdf = WaferPlot()
        wdf.plot(ax, x=x, y=y, z=param, cmap=color, vmin=vmin, vmax=vmax,
                 stats=stats, r=(wafer_size / 2))

        text = self.ui.plot_title.text()
        title = sel_param if not text else text
        ax.set_title(f"{title}")

        fig.tight_layout()
        canvas = FigureCanvas(fig)
        return canvas

    def plot4(self):
        """
        Plot graph.
        """
        if self.filtered_df is not None:
            dfr = self.filtered_df
        else:
            dfr = self.df_fit_results
        x = self.ui.cbb_x.currentText()
        y = self.ui.cbb_y.currentText()

        z = self.ui.cbb_z.currentText()
        if z == "None":
            hue = None
        else:
            hue = z if z != "" else None

        style = self.ui.cbb_plot_style.currentText()
        xmin = self.ui.xmin.text()
        ymin = self.ui.ymin.text()
        xmax = self.ui.xmax.text()
        ymax = self.ui.ymax.text()

        title = self.ui.ent_plot_title_2.text()
        x_text = self.ui.ent_xaxis_lbl.text()
        y_text = self.ui.ent_yaxis_lbl.text()

        text = self.ui.ent_x_rot.text()
        xlabel_rot = 0  # Default rotation angle
        if text:
            xlabel_rot = float(text)
        ax = self.ax4
        self.common.plot_graph(ax, dfr, x, y, hue, style, xmin, xmax, ymin,
                               ymax,
                               title,
                               x_text, y_text, xlabel_rot)
        self.ax4.get_figure().tight_layout()
        self.canvas4.draw()

    def get_mes_sites_coord(self):
        """
        Get all coordinates of measurement sites of selected maps.

        Returns:
        - all_x (list of float): List of x-coordinates of measurement sites.
        - all_y (list of float): List of y-coordinates of measurement sites.

        Action:
        - Retrieves the map name and coordinates.
        - Iterates through spectra to find measurement sites belonging to the
        selected map.
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

    def upd_spectra_list(self):
        """
        Update the spectra list based on the currently selected map.
        """
        current_row = self.ui.spectra_listbox.currentRow()
        self.ui.spectra_listbox.clear()
        current_item = self.ui.maps_listbox.currentItem()

        if current_item is not None:
            map_name = current_item.text()
            for spectrum in self.spectrums:
                map_name_fs, coord_fs = self.spectrum_object_id(spectrum)
                if map_name == map_name_fs:
                    item = QListWidgetItem(str(coord_fs))
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
        if current_row >= 0 and current_row < item_count:
            self.ui.spectra_listbox.setCurrentRow(current_row)
        else:
            if self.ui.spectra_listbox.count() > 0:
                self.ui.spectra_listbox.setCurrentRow(0)
        QTimer.singleShot(50, self.refresh_gui)

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
        self.ax.clear()
        self.ax2.clear()
        self.canvas1.draw()
        self.canvas2.draw()

    def copy_fig(self):
        """Copy figure to clipboard"""
        self.common.copy_fig_to_clb(canvas=self.canvas1)

    def copy_fig_wafer(self):
        """Copy the figure of the waferplot"""
        self.common.copy_fig_to_clb(canvas=self.canvas3)

    def copy_fig_graph(self):
        """Copy the figure of the graph plot to the clipboard"""
        self.common.copy_fig_to_clb(canvas=self.canvas4)

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
        dfs_new["MAPS_best_fit"] = self.df_fit_results
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

    def toggle_zoom_pan(self, checked):
        """Toggle zoom and pan functionality for the application."""
        self.zoom_pan_active = checked
        if not checked:
            self.zoom_pan_active = False

    def show_peak_table(self):
        """
        Show all fitted parameters of the selected spectrum in the GUI.
        """
        sel_spectrum, sel_spectra = self.get_spectrum_object()
        main_layout = self.ui.peak_table1
        cb_limits = self.ui.cb_limits
        cb_expr = self.ui.cb_expr
        update = self.upd_spectra_list
        show_params = ShowParameters(main_layout, sel_spectrum, cb_limits,
                                     cb_expr, update)
        show_params.show_peak_table(main_layout, sel_spectrum, cb_limits,
                                    cb_expr)

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

    def save_work(self):
        """Save the current application states to a JSON file."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(None,
                                                       "Save work",
                                                       "",
                                                       "SPECTROview Files ("
                                                       "*.maps)")
            if file_path:
                data_to_save = {
                    'spectrums': [spectrum_to_dict(spectrum) for spectrum in
                                  self.spectrums],
                    'maps': {k: v.to_dict() for k, v in self.maps.items()},
                    # 'df_fit_results': self.df_fit_results.to_dict() if
                    # self.df_fit_results is not None else None,
                    'filters': self.filter.filters,
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
                    for spectrum_data in load.get('spectrums', []):
                        spectrum = Spectrum()
                        set_attributes(spectrum, spectrum_data)
                        self.spectrums.append(spectrum)

                    self.maps = {k: pd.DataFrame(v) for k, v in
                                   load.get('maps', {}).items()}
                    self.filter.filters = load.get('filters', [])
                    self.filter.upd_filter_listbox()

                    QTimer.singleShot(300, self.collect_results)
                    self.upd_maps_list()

                except Exception as e:
                    show_alert(f"Error loading work: {e}")
        except Exception as e:
            show_alert(f"Error loading saved work (Maps Tab): {e}")

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

    def save_fit_settings(self):
        """
        Save all fitting settings to persistent storage (QSettings).
        """
        fit_params = {
            'fit_negative': self.ui.cb_fit_negative.isChecked(),
            'max_ite': self.ui.max_iteration.value(),
            'method': self.ui.cbb_fit_methods.currentText(),
            'ncpus': self.ui.cbb_cpu_number.currentText(),
            'xtol': float(self.ui.xtol.text())
        }
        print("settings are saved")
        # Save the fit_params to QSettings
        for key, value in fit_params.items():
            self.settings.setValue(key, value)

    def load_fit_settings(self):
        """
        Load last used fitting settings from persistent storage (QSettings).
        """
        fit_params = {
            'fit_negative': self.settings.value('fit_negative',
                                                defaultValue=False, type=bool),
            'max_ite': self.settings.value('max_ite', defaultValue=500,
                                           type=int),
            'method': self.settings.value('method', defaultValue='leastsq',
                                          type=str),
            'ncpus': self.settings.value('ncpus', defaultValue='auto',
                                         type=str),
            'xtol': self.settings.value('xtol', defaultValue=1.e-4, type=float)
        }

        # Update GUI elements with the loaded values
        self.ui.cb_fit_negative.setChecked(fit_params['fit_negative'])
        self.ui.max_iteration.setValue(fit_params['max_ite'])
        self.ui.cbb_fit_methods.setCurrentText(fit_params['method'])
        self.ui.cbb_cpu_number.setCurrentText(fit_params['ncpus'])
        self.ui.xtol.setText(str(fit_params['xtol']))

    def set_x_range_handler(self):
        """
        Handler for setting X range based on keyboard modifiers.
        """
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.set_x_range_all()
        else:
            self.set_x_range()

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

    def clear_env(self):
        """Clear the environment and reset the application state"""
        # Clear loaded maps and spectra
        self.maps.clear()
        self.spectrums = Spectra()
        self.loaded_fit_model = None
        self.current_fit_model = None

        # Clear DataFrames and Filters
        self.df_fit_results = None
        self.filtered_df = None

        # Clear UI elements that display data
        self.ui.maps_listbox.clear()
        self.ui.spectra_listbox.clear()
        self.ui.rsquared_1.clear()
        self.ui.item_count_label.setText("0 points")

        # Clear plot areas
        self.ax.clear()
        self.ax2.clear()
        self.ax4.clear()
        if hasattr(self, 'canvas1'):
            self.canvas1.draw()
        if hasattr(self, 'canvas2'):
            self.canvas2.draw()
        if hasattr(self, 'canvas4'):
            self.canvas4.draw()

        # Refresh the UI to reflect the cleared state
        QTimer.singleShot(50, self.rescale)
        QTimer.singleShot(100, self.upd_maps_list)
        print("'Maps' Tab environment has been cleared and reset.")
