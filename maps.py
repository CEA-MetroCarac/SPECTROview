# maps.py module
import os
import numpy as np
import pandas as pd
import json
from copy import deepcopy
from pathlib import Path
import dill
from utils import view_df, show_alert, quadrant, zone, view_text, \
    copy_fig_to_clb, \
    translate_param, clear_layout, reinit_spectrum, plot_graph
from utils import FitThread, PEAK_MODELS
from lmfit import fit_report
from fitspy.spectra import Spectra
from fitspy.spectrum import Spectrum
from fitspy.app.gui import Appli
from fitspy.utils import closest_index

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from wafer_view import WaferView
from PySide6.QtWidgets import (QFileDialog, QMessageBox, QApplication,
                               QListWidgetItem)
from PySide6.QtWidgets import QLabel, QComboBox, QLineEdit, QCheckBox, \
    QHBoxLayout, QSpacerItem, QSizePolicy, QPushButton, QVBoxLayout
from PySide6.QtGui import QColor, QPalette
from PySide6.QtCore import Qt, QFileInfo, QTimer, QObject, Signal
from tkinter import Tk, END

DIRNAME = os.path.dirname(__file__)
PLOT_POLICY = os.path.join(DIRNAME, "resources", "plotpolicy_spectre.mplstyle")


def update_param_hint_value(pm, key, text):
    #print(f"Peak Model: {pm.prefix}, Param Hint Key: {key}, Value: {text}")
    pm.param_hints[key]['value'] = float(text)


class Maps(QObject):
    # Define a signal for progress updates
    fit_progress_changed = Signal(int)

    def __init__(self, settings, ui, dataframe):
        super().__init__()
        self.settings = settings
        self.ui = ui
        self.dataframe = dataframe

        self.wafers = {}  # list of opened wafers
        self.toolbar = None
        self.model_fs = None  # FITSPY
        self.spectra_fs = Spectra()  # FITSPY

        # Update spectra_listbox when selecting wafer via WAFER LIST
        self.ui.wafers_listbox.itemSelectionChanged.connect(
            self.upd_spectra_list)

        # Connect and plot_spectre of selected SPECTRUM LIST
        self.ui.spectra_listbox.itemSelectionChanged.connect(self.delay_plot)

        # Connect the stateChanged signal of the legend CHECKBOX
        self.ui.cb_legend.stateChanged.connect(self.delay_plot)

        self.ui.cb_raw.stateChanged.connect(self.delay_plot)
        self.ui.cb_bestfit.stateChanged.connect(self.delay_plot)
        self.ui.cb_colors.stateChanged.connect(self.delay_plot)
        self.ui.cb_residual.stateChanged.connect(self.delay_plot)
        self.ui.cb_filled.stateChanged.connect(self.delay_plot)
        self.ui.cb_peaks.stateChanged.connect(self.delay_plot)
        self.ui.limits.stateChanged.connect(self.delay_plot)
        self.ui.expr.stateChanged.connect(self.delay_plot)
        # Set a delay for the function "plot1"
        self.delay_timer = QTimer()
        self.delay_timer.setSingleShot(True)
        self.delay_timer.timeout.connect(self.plot1)

        # Connect the progress signal to update_progress_bar slot
        self.fit_progress_changed.connect(self.update_pbar)

        self.plot_styles = ["box plot", "point plot", "bar plot"]
        self.create_plot_widget()

    def open_data(self, wafers=None, file_paths=None):
        """Open CSV files containing RAW spectra of each wafer"""

        if self.wafers is None:
            self.wafers = {}
        if wafers:
            self.wafers = wafers
        else:
            if file_paths is None:
                # Initialize the last used directory from QSettings
                last_dir = self.settings.value("last_directory", "/")
                options = QFileDialog.Options()
                options |= QFileDialog.ReadOnly
                file_paths, _ = QFileDialog.getOpenFileNames(
                    self.ui.tabWidget, "Open RAW spectra CSV File(s)", last_dir,
                    "CSV Files (*.csv);;Text Files (*.txt)", options=options)
            # Load RAW spectra data from CSV files
            if file_paths:
                last_dir = QFileInfo(file_paths[0]).absolutePath()
                self.settings.setValue("last_directory", last_dir)

                for file_path in file_paths:
                    file_path = Path(file_path)
                    fname = file_path.stem  # get fname w/o extension
                    extension = file_path.suffix.lower()  # get file extension

                    if extension == '.csv':
                        wafer_df = pd.read_csv(file_path, skiprows=1,
                                               delimiter=";")
                    elif extension == '.txt':
                        wafer_df = pd.read_csv(file_path, delimiter="\t")
                        wafer_df.columns = ['Y', 'X'] + list(
                            wafer_df.columns[2:])
                        # Reorder df as increasing wavenumber
                        sorted_columns = sorted(wafer_df.columns[2:], key=float)
                        wafer_df = wafer_df[['X', 'Y'] + sorted_columns]
                    else:
                        show_alert(f"Unsupported file format: {extension}")
                        continue
                    wafer_name = fname
                    if wafer_name in self.wafers:
                        print(f"Wafer '{wafer_name}' is already opened")
                    else:
                        self.wafers[wafer_name] = wafer_df
        self.extract_spectra()

    def extract_spectra(self):
        """Extract all spectra of each wafer dataframe"""
        for wafer_name, wafer_df in self.wafers.items():
            coord_columns = wafer_df.columns[:2]
            for _, row in wafer_df.iterrows():

                # Extract XY coords, wavenumber, and intensity values
                coord = tuple(row[coord_columns])
                x_values = wafer_df.columns[2:].tolist()
                x_values = pd.to_numeric(x_values, errors='coerce').tolist()

                y_values = row[2:].tolist()

                fname = f"{wafer_name}_{coord}"

                if not any(spectrum_fs.fname == fname for spectrum_fs in
                           self.spectra_fs):
                    # create FITSPY object
                    spectrum_fs = Spectrum()
                    spectrum_fs.fname = fname
                    spectrum_fs.x = np.asarray(x_values)[:-1]
                    spectrum_fs.x0 = np.asarray(x_values)[:-1]
                    spectrum_fs.y = np.asarray(y_values)[:-1]
                    spectrum_fs.y0 = np.asarray(y_values)[:-1]
                    self.spectra_fs.append(spectrum_fs)

        self.upd_wafers_list()

    def read_spectrum_model(self):
        """To read fitted params and peak_models of selected spectrum"""
        sel_spectrum = self.get_spectrum_objet()
        # Read spectral range
        self.ui.range_min.setText(str(sel_spectrum.x[0]))
        self.ui.range_max.setText(str(sel_spectrum.x[-1]))

        # Read peak_models

    def set_x_range(self):
        """Update sel_spectrum's range from QLineEdit"""
        sel_spectrum = self.get_spectrum_objet()
        fnames = []
        fnames.append(sel_spectrum.fname)
        reinit_spectrum(fnames, self.spectra_fs)
        new_x_min = float(self.ui.range_min.text())
        new_x_max = float(self.ui.range_max.text())
        # Set x range for selected spectrum
        ind_min = closest_index(sel_spectrum.x0, new_x_min)
        ind_max = closest_index(sel_spectrum.x0, new_x_max)
        sel_spectrum.x = sel_spectrum.x0[ind_min:ind_max + 1].copy()
        sel_spectrum.y = sel_spectrum.y0[ind_min:ind_max + 1].copy()
        sel_spectrum.range_max = float(self.ui.range_max.text())
        self.delay_plot()
        QTimer.singleShot(100, self.upd_spectra_list)

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())

    def show_peak_table(self):
        sel_spectrum = self.get_spectrum_objet()
        self.clear_layout(self.ui.peak_table1)

        labelsss = sel_spectrum.peak_labels
        peak_modelsss = sel_spectrum.peak_models
        print("Length of labels:", len(labelsss))
        print("Length of peak_models:", len(peak_modelsss))


        header_labels = ["Delete", "Label", "Model"]
        param_hint_order = ['x0', 'fwhm', 'ampli', 'alpha', 'fwhm_l', 'fwhm_r']

        # Create and add headers to list
        for param_hint_key in param_hint_order:
            if any(param_hint_key in peak_model.param_hints for peak_model in sel_spectrum.peak_models):
                header_labels.append(param_hint_key.title())
                header_labels.append(f"fix {param_hint_key.title()}")
                if self.ui.limits.isChecked():
                    header_labels.append(f"min {param_hint_key.title()}")
                    header_labels.append(f"max {param_hint_key.title()}")
                if self.ui.expr.isChecked():
                    header_labels.append(f"expression {param_hint_key.title()}")

        # Create vertical layouts for each column type
        delete_layout = QVBoxLayout()
        label_layout = QVBoxLayout()
        model_layout = QVBoxLayout()
        param_hint_layouts = {param_hint: {var: QVBoxLayout() for var in ['value', 'min', 'max', 'expr', 'vary']} for
                              param_hint in param_hint_order}

        # Add header labels to each layout
        for header_label in header_labels:
            label = QLabel(header_label)
            label.setAlignment(Qt.AlignCenter)
            if header_label == "Delete":
                delete_layout.addWidget(label)
            elif header_label == "Label":
                label_layout.addWidget(label)
            elif header_label == "Model":
                model_layout.addWidget(label)
            elif header_label.startswith("fix"):
                param_hint_key = header_label.split()[1].lower()
                param_hint_layouts[param_hint_key]['vary'].addWidget(label)
            elif "min" in header_label:
                param_hint_key = header_label.split()[1].lower()
                param_hint_layouts[param_hint_key]['min'].addWidget(label)
            elif "max" in header_label:
                param_hint_key = header_label.split()[1].lower()
                param_hint_layouts[param_hint_key]['max'].addWidget(label)
            elif "expression" in header_label:
                param_hint_key = header_label.split()[1].lower()
                param_hint_layouts[param_hint_key]['expr'].addWidget(label)
            else:
                param_hint_key = header_label.lower()
                param_hint_layouts[param_hint_key]['value'].addWidget(label)

        for i, peak_model in enumerate(sel_spectrum.peak_models):
            # Button to delete peak_model
            delete = QPushButton(peak_model.prefix)
            delete.setFixedWidth(50)
            delete.clicked.connect(lambda idx=i, spectrum=sel_spectrum: self.delete_peak_model(spectrum, idx))
            delete_layout.addWidget(delete)

            # Peak_label
            label = QLineEdit(sel_spectrum.peak_labels[i])
            label.setFixedWidth(80)
            label.textChanged.connect(
                lambda text, idx=i, spectrum=sel_spectrum: self.update_peak_label(spectrum, idx, text))
            label_layout.addWidget(label)

            # Peak model : Lorentizan, Gaussien, etc...
            model = QComboBox()
            model.addItems(PEAK_MODELS)
            current_model_index = PEAK_MODELS.index(peak_model.name2) if peak_model.name2 in PEAK_MODELS else 0
            model.setCurrentIndex(current_model_index)
            model.setFixedWidth(120)
            model.currentIndexChanged.connect(lambda index, pm=peak_model: self.update_model_name(pm, index))
            model_layout.addWidget(model)

            # variables of peak_model
            param_hints = peak_model.param_hints
            for param_hint_key in param_hint_order:
                if param_hint_key in param_hints:
                    param_hint_value = param_hints[param_hint_key]

                    # 4.1 VALUE
                    value_val = round(param_hint_value.get('value', 0.0), 2)
                    value = QLineEdit(str(value_val))
                    value.setFixedWidth(70)
                    value.setFixedHeight(24)
                    value.setAlignment(Qt.AlignRight)
                    value.textChanged.connect(
                        lambda text, pm=peak_model, key=param_hint_key: self.update_param_hint_value(pm, key, text))
                    param_hint_layouts[param_hint_key]['value'].addWidget(value)

                    # 4.2 FIXED or NOT
                    vary = QCheckBox()
                    vary.setChecked(not param_hint_value.get('vary', False))
                    vary.setFixedHeight(24)
                    vary.stateChanged.connect(
                        lambda state, pm=peak_model, key=param_hint_key: self.update_param_hint_vary(pm, key,
                                                                                                     not state))
                    param_hint_layouts[param_hint_key]['vary'].addWidget(vary)

                    # 4.3 MIN MAX
                    if self.ui.limits.isChecked():
                        min_val = round(param_hint_value.get('min', 0.0), 2)
                        min_lineedit = QLineEdit(str(min_val))
                        min_lineedit.setFixedWidth(70)
                        min_lineedit.setFixedHeight(24)
                        min_lineedit.setAlignment(Qt.AlignRight)
                        min_lineedit.textChanged.connect(
                            lambda text, pm=peak_model, key=param_hint_key: self.update_param_hint_min(pm, key, text))
                        param_hint_layouts[param_hint_key]['min'].addWidget(min_lineedit)

                        max_val = round(param_hint_value.get('max', 0.0), 2)
                        max_lineedit = QLineEdit(str(max_val))
                        max_lineedit.setFixedWidth(70)
                        max_lineedit.setFixedHeight(24)
                        max_lineedit.setAlignment(Qt.AlignRight)
                        max_lineedit.textChanged.connect(
                            lambda text, pm=peak_model, key=param_hint_key: self.update_param_hint_max(pm, key, text))
                        param_hint_layouts[param_hint_key]['max'].addWidget(max_lineedit)

                    # 4.4 EXPRESSION
                    if self.ui.expr.isChecked():
                        expr_val = str(param_hint_value.get('expr', ''))
                        expr = QLineEdit(expr_val)
                        expr.setFixedWidth(150)
                        expr.setFixedHeight(24)  # Set a fixed height for QLineEdit
                        expr.setAlignment(Qt.AlignRight)
                        expr.textChanged.connect(
                            lambda text, pm=peak_model, key=param_hint_key: self.update_param_hint_expr(pm, key, text))
                        param_hint_layouts[param_hint_key]['expr'].addWidget(expr)

                else:
                    # Add empty labels for alignment
                    empty_label = QLabel()
                    empty_label.setFixedHeight(24)
                    param_hint_layouts[param_hint_key]['value'].addWidget(empty_label)
                    param_hint_layouts[param_hint_key]['vary'].addWidget(empty_label)
                    if self.ui.limits.isChecked():
                        param_hint_layouts[param_hint_key]['min'].addWidget(empty_label)
                        param_hint_layouts[param_hint_key]['max'].addWidget(empty_label)
                    if self.ui.expr.isChecked():
                        param_hint_layouts[param_hint_key]['expr'].addWidget(empty_label)

        # Add vertical layouts to main layout
        self.ui.peak_table1.addLayout(delete_layout)
        self.ui.peak_table1.addLayout(label_layout)
        self.ui.peak_table1.addLayout(model_layout)

        for param_hint_key, param_hint_layout in param_hint_layouts.items():
            for var_layout in param_hint_layout.values():
                self.ui.peak_table1.addLayout(var_layout)
    def add_peak(self):
        """To add a peak_model for the selected spectrum"""
        sel_spectrum = self.get_spectrum_objet()
        pos = float(self.ui.peak_pos.text())
        fit_model = self.ui.fit_model.currentText()
        sel_spectrum.add_peak_model(fit_model, pos)
        self.upd_spectra_list()
    def delete_peak_model(self, spectrum, idx):
        """"To delete a peak model"""
        del spectrum.peak_models[idx]
        del spectrum.peak_labels[idx]
        self.upd_spectra_list()
    def clear_all_peaks(self):
        """To clear all current peak_models of the selected spectrum"""
        sel_spectrum = self.get_spectrum_objet()
        sel_spectrum.remove_models()
        self.upd_spectra_list()

    def save_fit_model(self):
        """To save the current fit model of the selected spectrum"""
        sel_spectrum = self.get_spectrum_objet()

        last_dir = self.settings.value("last_directory", "/")
        save_path, _ = QFileDialog.getSaveFileName(
            self.ui.tabWidget, "Save fit model", last_dir,
            "JSON Files (*.json)")
        if save_path:
            try:
                if sel_spectrum:
                    self.spectra_fs.save(save_path, [sel_spectrum.fname])
                    QMessageBox.information(
                        self.ui.tabWidget, "Success",
                        "Fit model is saved (JSON file).")
                else:
                    QMessageBox.warning(
                        self.ui.tabWidget, "Warning",
                        "No fit model to save.")
            except Exception as e:
                QMessageBox.critical(
                    self.ui.tabWidget, "Error",
                    f"Error saving model dictionary: {str(e)}")

    def update_peak_label(self, spectrum, idx, text):
        spectrum.peak_labels[idx] = text
    def update_model_name(self, pm, index):
        print(f"Model Name: {pm.name2[index]}")
    def update_param_hint_value(self, pm, key, text):
        pm.param_hints[key]['value'] = float(text)
    def update_param_hint_min(self, pm, key, text):
        pm.param_hints[key]['min'] = float(text)
    def update_param_hint_max(self, pm, key, text):
        pm.param_hints[key]['max'] = float(text)
    def update_param_hint_vary(self, pm, key, state):
        pm.param_hints[key]['vary'] = state
        self.upd_spectra_list()
    def update_param_hint_expr(self, pm, key, text):
        pm.param_hints[key]['expr'] = text

    def apply_fit_model(self, sel_spectrum):
        """To apply all parameters of a fit model"""
        sel_spectrum = self.get_spectrum_objet()
        sel_spectrum.fit()
        self.delay_plot()
        QTimer.singleShot(100, self.upd_spectra_list)
    def open_model(self, fname_json=None):
        """Load a fit model pre-created by FITSPY tool"""
        if not fname_json:
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
            fname_json = selected_file
        self.model_fs = self.spectra_fs.load_model(fname_json, ind=0)
        display_name = QFileInfo(fname_json).baseName()
        self.ui.lb_loaded_model.setText(f"'{display_name}' is loaded !")
        # self.ui.lb_loaded_model.setStyleSheet("color: yellow;")

    def fit(self, fnames=None):
        """Fit selected spectrum(s)"""
        # Disable the button to prevent multiple clicks leading to a crash
        self.ui.btn_fit.setEnabled(False)
        if self.model_fs is None:
            show_alert("Load a fit model before fitting.")
            self.ui.btn_fit.setEnabled(True)
            return

        if fnames is None:
            wafer_name, coords = self.spectre_id()
            fnames = [f"{wafer_name}_{coord}" for coord in coords]

        # Start fitting process in a separate thread
        self.fit_thread = FitThread(self.spectra_fs, self.model_fs, fnames)
        # To update progress bar
        self.fit_thread.fit_progress_changed.connect(self.update_pbar)
        # To display progress in GUI
        self.fit_thread.fit_progress.connect(
            lambda num, elapsed_time: self.fit_progress(num, elapsed_time,
                                                        fnames))
        # To update spectra list + plot fitted specturm once fitting finished
        self.fit_thread.fit_completed.connect(self.fit_completed)
        self.fit_thread.finished.connect(
            lambda: self.ui.btn_fit.setEnabled(True))
        self.fit_thread.start()

    def fit_all(self):
        """ Apply loaded fit model to all selected spectra"""
        fnames = self.spectra_fs.fnames
        self.fit(fnames=fnames)

    def collect_results(self):
        """Function to collect best-fit results and append in a dataframe"""
        # Add all dict into a list, then convert to a dataframe.
        fit_results_list = []
        self.df_fit_results = None

        for spectrum_fs in self.spectra_fs:
            if hasattr(spectrum_fs.result_fit, 'best_values'):
                wafer_name, coord = self.spectre_id_fs(spectrum_fs)
                x, y = coord
                success = spectrum_fs.result_fit.success
                rsquared = spectrum_fs.result_fit.rsquared
                best_values = spectrum_fs.result_fit.best_values
                best_values["Wafer"] = wafer_name
                best_values["X"] = x
                best_values["Y"] = y
                best_values["success"] = success
                best_values["Rsquared"] = rsquared

                fit_results_list.append(best_values)
        self.df_fit_results = (pd.DataFrame(fit_results_list)).round(3)

        # reindex columns according to the parameters names
        self.df_fit_results = self.df_fit_results.reindex(
            sorted(self.df_fit_results.columns), axis=1)
        names = []
        for name in self.df_fit_results.columns:
            if name in ["Wafer", "X", 'Y', "success"]:
                name = '0' + name  # to be in the 3 first columns
            elif '_' in name:
                name = 'z' + name[5:]  # model peak parameters to be at the end
            names.append(name)
        self.df_fit_results = self.df_fit_results.iloc[:,
                              list(np.argsort(names, kind='stable'))]
        columns = [translate_param(self.model_fs, column) for column in
                   self.df_fit_results.columns]
        self.df_fit_results.columns = columns

        # Add "Quadrant" columns
        self.df_fit_results['Quadrant'] = self.df_fit_results.apply(quadrant,
                                                                    axis=1)
        diameter = float(self.ui.wafer_size.text())
        # Use a lambda function to pass the row argument to the zone function
        self.df_fit_results['Zone'] = self.df_fit_results.apply(
            lambda row: zone(row, diameter), axis=1)

        self.upd_cbb_param()
        self.upd_cbb_wafer()
        self.send_df_to_viz()

    def save_fit_results(self):
        """Functon to save fitted results in an excel file"""
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
        """Functon to load fitted results to view"""
        self.df_fit_results = None

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
                self.df_fit_results = dfr
            except Exception as e:
                show_alert("Error loading DataFrame:", e)

        self.upd_cbb_param()
        self.upd_cbb_wafer()
        self.send_df_to_viz()

    def upd_cbb_wafer(self):
        """to append all values of df_fit_results to comoboxses"""
        self.ui.cbb_wafer_1.clear()
        wafer_names = self.df_fit_results['Wafer'].unique()
        for wafer_name in wafer_names:
            self.ui.cbb_wafer_1.addItem(wafer_name)

    def upd_cbb_param(self):
        """to append all values of df_fit_results to comoboxses"""
        if self.df_fit_results is not None:
            columns = self.df_fit_results.columns.tolist()
            self.ui.cbb_param_1.clear()
            self.ui.cbb_x.clear()
            self.ui.cbb_y.clear()
            self.ui.cbb_z.clear()
            for column in columns:
                # remove_special_chars = re.sub(r'\$[^$]+\$', '', column)
                self.ui.cbb_param_1.addItem(column)
                self.ui.cbb_x.addItem(column)
                self.ui.cbb_y.addItem(column)
                self.ui.cbb_z.addItem(column)

    def split_fname(self):
        """Split fname and populate the combobox"""
        dfr = self.df_fit_results
        fname_parts = dfr.loc[0, 'Wafer'].split('_')
        self.ui.cbb_split_fname_2.clear()  # Clear existing items in combobox
        for part in fname_parts:
            self.ui.cbb_split_fname_2.addItem(part)

    def add_column(self):
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
        parts = dfr['Wafer'].str.split('_')
        dfr[col_name] = [part[selected_part_index] if len(
            part) > selected_part_index else None for part in parts]
        show_alert(f"Column added successfully:'{col_name}'")
        self.df_fit_results = dfr
        self.send_df_to_viz()
        self.upd_cbb_param()
        self.upd_cbb_wafer()

    def reinit(self, fnames=None):
        """Reinitialize the selected spectrum(s)"""
        if fnames is None:
            wafer_name, coords = self.spectre_id()
            fnames = [f"{wafer_name}_{coord}" for coord in coords]
        reinit_spectrum(fnames, self.spectra_fs)
        self.delay_plot()
        self.upd_spectra_list()
        QTimer.singleShot(200, self.rescale)

    def reinit_all(self):
        """Reinitialize all spectra"""
        fnames = self.spectra_fs.fnames
        self.reinit(fnames)

    def reinit_fnc_handler(self):
        """Switch between 2 save fit fnc with the Ctrl key"""
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.reinit_all()
        else:
            self.reinit()

    def rescale(self):
        """Rescale the figure."""
        self.ax.autoscale()
        self.canvas1.draw()

    def create_plot_widget(self):
        """Create canvas and toolbar for plotting in the GUI"""
        plt.style.use(PLOT_POLICY)
        fig1 = plt.figure()
        self.ax = fig1.add_subplot(111)
        self.ax.set_xlabel("Raman shift (cm$^{-1}$)")
        self.ax.set_ylabel("Intensity (a.u)")
        self.canvas1 = FigureCanvas(fig1)
        self.toolbar = NavigationToolbar2QT(self.canvas1)
        rescale = next(
            a for a in self.toolbar.actions() if a.text() == 'Home')
        rescale.triggered.connect(self.rescale)

        self.ui.QVBoxlayout.addWidget(self.canvas1)
        self.ui.toolbar_frame.addWidget(self.toolbar)
        self.canvas1.figure.tight_layout()
        self.canvas1.draw()

        # plot 2: Measurement sites view
        fig2 = plt.figure()
        self.ax2 = fig2.add_subplot(111)
        self.canvas2 = FigureCanvas(fig2)
        layout = self.ui.wafer_plot.layout()
        layout.addWidget(self.canvas2)
        self.canvas2.draw()

        # Plot4: graph
        fig4 = plt.figure()
        self.ax4 = fig4.add_subplot(111)
        self.canvas4 = FigureCanvas(fig4)
        self.ui.frame_graph.addWidget(self.canvas4)
        self.canvas4.draw()

    def plot1(self):
        """Plot all selected spectra"""

        wafer_name, coords = self.spectre_id()  # current selected spectra ID
        selected_spectra_fs = []
        for spectrum_fs in self.spectra_fs:
            wafer_name_fs, coord_fs = self.spectre_id_fs(spectrum_fs)
            if wafer_name_fs == wafer_name and coord_fs in coords:
                selected_spectra_fs.append(spectrum_fs)
        if len(selected_spectra_fs) == 0:
            return
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.clear()
        # reassign previous axis limits (related to zoom)
        if not xlim == ylim == (0.0, 1.0):
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

        for spectrum_fs in selected_spectra_fs:
            fname, coord = self.spectre_id_fs(spectrum_fs)
            x_values = spectrum_fs.x
            y_values = spectrum_fs.y
            self.ax.plot(x_values, y_values, label=f"{coord}", ms=3, lw=2)

            if self.ui.cb_raw.isChecked():
                x0_values = spectrum_fs.x0
                y0_values = spectrum_fs.y0
                self.ax.plot(x0_values, y0_values, 'ko-', label='raw', ms=3,
                             lw=1)

            if hasattr(spectrum_fs.result_fit, 'components') and self.ui.cb_bestfit.isChecked():
                bestfit = spectrum_fs.result_fit.best_fit
                self.ax.plot(x_values, bestfit, label=f"bestfit")

            for peak_model in spectrum_fs.peak_models:
                # Convert prefix to peak_labels
                peak_labels = spectrum_fs.peak_labels
                prefix = str(peak_model.prefix)
                peak_index = int(prefix[1:-1]) - 1
                if 0 <= peak_index < len(peak_labels):
                    peak_label = peak_labels[peak_index]

                # remove temporarily 'expr'
                param_hints_orig = deepcopy(peak_model.param_hints)
                for key, _ in peak_model.param_hints.items():
                    peak_model.param_hints[key]['expr'] = ''
                params = peak_model.make_params()
                # rassign 'expr'
                peak_model.param_hints = param_hints_orig

                y_peak = peak_model.eval(params, x=x_values)

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

            if hasattr(spectrum_fs.result_fit,
                       'residual') and self.ui.cb_residual.isChecked():
                residual = spectrum_fs.result_fit.residual
                self.ax.plot(x_values, residual, 'ko-', ms=3, label='residual')
            if self.ui.cb_colors.isChecked() is False:
                self.ax.set_prop_cycle(None)

            if hasattr(spectrum_fs.result_fit, 'rsquared'):
                rsquared = round(spectrum_fs.result_fit.rsquared, 4)
                self.ui.rsquared_1.setText(f"R2={rsquared}")
            else:
                self.ui.rsquared_1.setText("R2=0")

        self.ax.set_xlabel("Raman shift (cm$^{-1}$)")
        self.ax.set_ylabel("Intensity (a.u)")
        if self.ui.cb_legend.isChecked():
            self.ax.legend(loc='upper right')

        self.ax.get_figure().tight_layout()
        self.canvas1.draw()
        self.plot2()
        self.read_spectrum_model()
        self.show_peak_table()

    def plot2(self):
        """Plot wafer maps of measurement sites"""
        wafer_name, coords = self.spectre_id()
        all_x = []
        all_y = []
        for spectrum_fs in self.spectra_fs:
            wafer_name_fs, coord_fs = self.spectre_id_fs(spectrum_fs)
            if wafer_name == wafer_name_fs:
                x, y = coord_fs
                all_x.append(x)
                all_y.append(y)

        self.ax2.clear()
        self.ax2.scatter(all_x, all_y, marker='x', color='gray', s=10)
        if coords:
            x, y = zip(*coords)
            self.ax2.scatter(x, y, marker='o', color='red', s=40)
        self.ax2.get_figure().tight_layout()
        self.canvas2.draw()

    def plot3(self):
        """Plot WaferDataFrame"""
        clear_layout(self.ui.frame_wafer.layout())
        dfr = self.df_fit_results
        wafer_name = self.ui.cbb_wafer_1.currentText()
        color = self.ui.cbb_color_pallete.currentText()
        wafer_size = float(self.ui.wafer_size.text())

        if wafer_name is not None:
            selected_df = dfr.query('Wafer == @wafer_name')
        sel_param = self.ui.cbb_param_1.currentText()
        self.canvas3 = self.plot3_action(selected_df, sel_param, wafer_size,
                                         color)
        self.ui.frame_wafer.addWidget(self.canvas3)

    def plot3_action(self, selected_df, sel_param, wafer_size, color):
        """PLot wafer map of a selected parameters"""
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

        wdf = WaferView()
        wdf.plot(ax, x=x, y=y, z=param, cmap=color, vmin=vmin, vmax=vmax,
                 stats=stats,
                 r=(wafer_size / 2))

        text = self.ui.plot_title.text()
        title = sel_param if not text else text
        ax.set_title(f"{title}")

        fig.tight_layout()
        canvas = FigureCanvas(fig)
        return canvas

    def plot4(self):
        """Plot graph """
        dfr = self.df_fit_results
        x = self.ui.cbb_x.currentText()
        y = self.ui.cbb_y.currentText()
        z = self.ui.cbb_z.currentText()
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
        plot_graph(ax, dfr, x, y, z, style, xmin, xmax, ymin, ymax, title,
                   x_text, y_text, xlabel_rot)

        self.ax4.get_figure().tight_layout()
        self.canvas4.draw()

    def upd_wafers_list(self):
        """ To update the wafer listbox"""
        current_row = self.ui.wafers_listbox.currentRow()

        self.ui.wafers_listbox.clear()
        wafer_names = list(self.wafers.keys())
        for wafer_name in wafer_names:
            item = QListWidgetItem(wafer_name)
            self.ui.wafers_listbox.addItem(item)
        item_count = self.ui.wafers_listbox.count()

        # Management of selecting item of listbox
        if current_row >= item_count:
            current_row = item_count - 1
        if current_row >= 0:
            self.ui.wafers_listbox.setCurrentRow(current_row)
        else:
            if item_count > 0:
                self.ui.wafers_listbox.setCurrentRow(0)
        QTimer.singleShot(100, self.upd_spectra_list)

    def upd_spectra_list(self):
        """to update the spectra list"""
        current_row = self.ui.spectra_listbox.currentRow()

        self.ui.spectra_listbox.clear()
        current_item = self.ui.wafers_listbox.currentItem()

        if current_item is not None:
            wafer_name = current_item.text()
            for spectrum_fs in self.spectra_fs:
                wafer_name_fs, coord_fs = self.spectre_id_fs(spectrum_fs)
                if wafer_name == wafer_name_fs:
                    item = QListWidgetItem(str(coord_fs))
                    if hasattr(spectrum_fs.result_fit,
                               'success') and spectrum_fs.result_fit.success:
                        item.setBackground(QColor("green"))
                    elif hasattr(spectrum_fs.result_fit,
                                 'success') and not \
                            spectrum_fs.result_fit.success:
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
        QTimer.singleShot(50, self.delay_plot)

    def remove_wafer(self):
        """To remove a wafer from the listbox and wafers df"""
        wafer_name, coords = self.spectre_id()
        if wafer_name in self.wafers:
            del self.wafers[wafer_name]
            self.spectra_fs = Spectra(
                spectrum_fs for spectrum_fs in self.spectra_fs if
                not spectrum_fs.fname.startswith(wafer_name))
            self.upd_wafers_list()
        self.ui.spectra_listbox.clear()
        self.ax.clear()
        self.ax2.clear()
        self.canvas1.draw()
        self.canvas2.draw()

    def copy_fig(self):
        """To copy figure canvas to clipboard"""
        copy_fig_to_clb(canvas=self.canvas1)

    def copy_fig_wafer(self):
        """To copy figure canvas to clipboard"""
        copy_fig_to_clb(canvas=self.canvas3)

    def copy_fig_graph(self):
        """To copy figure canvas to clipboard"""
        copy_fig_to_clb(canvas=self.canvas4)

    def select_all_spectra(self):
        """ To quickly select all spectra within the spectra listbox"""
        item_count = self.ui.spectra_listbox.count()
        for i in range(item_count):
            item = self.ui.spectra_listbox.item(i)
            item.setSelected(True)

    def select_verti(self):
        """ To select all spectra vertically within the spectra listbox"""
        self.ui.spectra_listbox.clearSelection()  # Clear all selection
        item_count = self.ui.spectra_listbox.count()
        for i in range(item_count):
            item = self.ui.spectra_listbox.item(i)
            coord_str = item.text()
            x_coord, y_coord = map(float, coord_str.strip('()').split(','))
            if x_coord == 0:
                item.setSelected(True)

    def select_horiz(self):
        """ To quickly select all spectra vertically the spectra listbox"""
        self.ui.spectra_listbox.clearSelection()  # Clear all selection
        item_count = self.ui.spectra_listbox.count()
        for i in range(item_count):
            item = self.ui.spectra_listbox.item(i)
            coord_str = item.text()
            x_coord, y_coord = map(float, coord_str.strip('()').split(','))
            if y_coord == 0:
                item.setSelected(True)

    def get_spectrum_objet(self):
        """ to get the selected spectrum object"""
        wafer_name, coords = self.spectre_id()
        selected_spectra_fs = []
        for spectrum_fs in self.spectra_fs:
            wafer_name_fs, coord_fs = self.spectre_id_fs(spectrum_fs)
            if wafer_name_fs == wafer_name and coord_fs in coords:
                selected_spectra_fs.append(spectrum_fs)
        if len(selected_spectra_fs) == 0:
            return
        sel_spectrum = selected_spectra_fs[0]
        return sel_spectrum

    def spectre_id(self):
        """Get selected spectre id(s)"""
        wafer_item = self.ui.wafers_listbox.currentItem()
        if wafer_item is not None:
            wafer_name = wafer_item.text()
            selected_spectra = self.ui.spectra_listbox.selectedItems()
            coords = []
            if selected_spectra:
                for selected_item in selected_spectra:
                    text = selected_item.text()
                    x, y = map(float, text.strip('()').split(','))
                    coord = (x, y)
                    coords.append(coord)
            return wafer_name, coords
        return None, None

    def spectre_id_fs(self, spectrum_fs=None):
        """Get selected spectre id(s) of FITSPY object"""

        fname_parts = spectrum_fs.fname.split("_")
        wafer_name_fs = "_".join(fname_parts[:-1])
        coord_str = fname_parts[-1]  # Last part contains the coordinates
        coord_fs = tuple(
            map(float, coord_str.split('(')[1].split(')')[0].split(',')))
        return wafer_name_fs, coord_fs

    def delay_plot(self):
        """Trigger the fnc to plot spectre"""
        self.delay_timer.start(100)

    def view_fit_results_df(self):
        """To view selected dataframe"""
        view_df(self.ui.tabWidget, self.df_fit_results)

    def view_wafer_data(self):
        """To view data of selected wafer """
        wafer_name, coords = self.spectre_id()
        view_df(self.ui.tabWidget, self.wafers[wafer_name])

    def fit_fnc_handler(self):
        """Switch between 2 save fit fnc with the Ctrl key"""
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.fit_all()
        else:
            self.fit()

    def send_df_to_viz(self):
        """Send the collected spectral data dataframe to visu tab"""
        dfs = self.dataframe.original_dfs
        dfs["2Dmaps_bestfit_results"] = self.df_fit_results
        self.dataframe.action_open_df(file_paths=None, original_dfs=dfs)

    def cosmis_ray_detection(self):
        self.spectra_fs.outliers_limit_calculation()

    def view_stats(self):
        """Show the statistique fitting results of the selected spectrum"""
        wafer_name, coords = self.spectre_id()
        selected_spectra_fs = []
        for spectrum_fs in self.spectra_fs:
            wafer_name_fs, coord_fs = self.spectre_id_fs(spectrum_fs)
            if wafer_name_fs == wafer_name and coord_fs in coords:
                selected_spectra_fs.append(spectrum_fs)
        if len(selected_spectra_fs) == 0:
            return

        ui = self.ui.tabWidget
        title = f"Fitting Report - {wafer_name} - {coords}"
        # Show the 'report' of the first selected spectrum
        spectrum_fs = selected_spectra_fs[0]
        if spectrum_fs.result_fit:
            text = fit_report(spectrum_fs.result_fit)
            view_text(ui, title, text)

    def save_work(self):
        """Save the current work/results."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(None,
                                                       "Save work",
                                                       "",
                                                       "SPECTROview Files ("
                                                       "*.svmap)")
            if file_path:
                data_to_save = {
                    'spectra_fs': self.spectra_fs,
                    'wafers': self.wafers,
                    'model_fs': self.model_fs,
                    'model_name': self.ui.lb_loaded_model.text(),
                    'df_fit_results': self.df_fit_results,

                    'cbb_x': self.ui.cbb_x.currentIndex(),
                    'cbb_y': self.ui.cbb_y.currentIndex(),
                    'cbb_z': self.ui.cbb_z.currentIndex(),
                    "cbb_param_1": self.ui.cbb_param_1.currentIndex(),
                    "cbb_wafer_1": self.ui.cbb_wafer_1.currentIndex(),

                    'plot_title': self.ui.plot_title.text(),
                    'plot_title2': self.ui.ent_plot_title_2.text(),
                    'xmin': self.ui.xmin.text(),
                    'xmax': self.ui.xmax.text(),
                    'ymax': self.ui.ymax.text(),
                    'ymin': self.ui.ymin.text(),
                    'xaxis_lbl': self.ui.ent_xaxis_lbl.text(),
                    'yaxis_lbl': self.ui.ent_yaxis_lbl.text(),
                    'x_rot': self.ui.ent_x_rot.text(),
                    "plot_style": self.ui.cbb_plot_style.currentIndex(),
                    'vmin': self.ui.int_vmin.text(),
                    'vmax': self.ui.int_vmax.text(),

                    'wafer_size': self.ui.wafer_size.text(),
                    "color_pal": self.ui.cbb_color_pallete.currentIndex(),
                }
                with open(file_path, 'wb') as f:
                    dill.dump(data_to_save, f)
                show_alert("Work saved successfully.")
        except Exception as e:
            show_alert(f"Error saving work: {e}")

    def load_work(self):
        """Load a previously saved work."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(None,
                                                       "Load work",
                                                       "",
                                                       "SPECTROview Files ("
                                                       "*.svmap)")
            if file_path:
                with open(file_path, 'rb') as f:
                    load = dill.load(f)
                    self.spectra_fs = load.get('spectra_fs')
                    self.wafers = load.get('wafers')
                    self.model_fs = load.get('model_fs')
                    model_name = load.get('model_name', '')
                    self.ui.lb_loaded_model.setText(model_name)
                    self.ui.lb_loaded_model.setStyleSheet("color: yellow;")

                    self.df_fit_results = load.get('df_fit_results')
                    self.upd_cbb_param()
                    self.upd_cbb_wafer()
                    self.send_df_to_viz()
                    self.upd_wafers_list()

                    # Set default values or None for missing keys
                    self.ui.cbb_x.setCurrentIndex(load.get('cbb_x', -1))
                    self.ui.cbb_y.setCurrentIndex(load.get('cbb_y', -1))
                    self.ui.cbb_z.setCurrentIndex(load.get('cbb_z', -1))
                    self.ui.cbb_param_1.setCurrentIndex(
                        load.get('cbb_param_1', -1))
                    self.ui.cbb_wafer_1.setCurrentIndex(
                        load.get('cbb_wafer_1', -1))
                    self.ui.plot_title.setText(load.get('plot_title', ''))
                    self.ui.ent_plot_title_2.setText(
                        load.get('plot_title2', ''))
                    self.ui.xmin.setText(load.get('xmin', ''))
                    self.ui.xmax.setText(load.get('xmax', ''))
                    self.ui.ymax.setText(load.get('ymax', ''))
                    self.ui.ymin.setText(load.get('ymin', ''))
                    self.ui.ent_xaxis_lbl.setText(load.get('xaxis_lbl', ''))
                    self.ui.ent_yaxis_lbl.setText(load.get('yaxis_lbl', ''))
                    self.ui.ent_x_rot.setText(load.get('x_rot', ''))

                    self.ui.cbb_color_pallete.setCurrentIndex(
                        load.get('color_pal', -1))
                    self.ui.wafer_size.setText(load.get('wafer_size', ''))
                    self.ui.int_vmin.setText(load.get('vmin', ''))
                    self.ui.int_vmax.setText(load.get('vmax', ''))

                    self.plot4()
                    self.plot3()
        except Exception as e:
            show_alert(f"Error loading work: {e}")

    def fitspy_launcher(self):
        """To Open FITSPY with selected spectra"""
        if self.spectra_fs:
            plt.style.use('default')
            root = Tk()
            appli = Appli(root, force_terminal_exit=False)

            appli.spectra = self.spectra_fs
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

    def fit_progress(self, num, elapsed_time, fnames):
        """Called when fitting process is completed"""
        self.ui.progress.setText(
            f"{num}/{len(fnames)} fitted ({elapsed_time:.2f}s)")

    def fit_completed(self):
        """Called when fitting process is completed"""
        self.delay_plot()
        self.upd_spectra_list()
        QTimer.singleShot(200, self.rescale)

    def update_pbar(self, progress):
        self.ui.progressBar.setValue(progress)
