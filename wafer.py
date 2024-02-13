# wafer.py module
import os
import copy
# import win32clipboard
from io import BytesIO
import numpy as np
import pandas as pd
from pathlib import Path
from lmfit import Model, fit_report
from fitspy.spectra import Spectra
from fitspy.spectrum import Spectrum
from threading import Thread
from multiprocessing import Queue
from PySide6.QtCore import Qt, QThread, Signal

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.collections import PathCollection

from wafer_view import WaferView

from PySide6.QtWidgets import (
    QFileDialog, QVBoxLayout, QMessageBox, QFrame, QPushButton, QTableWidget,
    QTableWidgetItem,
    QHBoxLayout, QApplication, QSpacerItem, QSizePolicy, QDialog,
    QListWidgetItem, QCheckBox, QTextBrowser
)
from PySide6.QtGui import QStandardItemModel, QStandardItem, QColor

from PySide6.QtGui import QIcon, QTextCursor
from PySide6.QtCore import Qt, QSize, QCoreApplication, QSettings, QFileInfo, \
    QTimer

DIRNAME = os.path.dirname(__file__)
PLOT_POLICY = os.path.join(DIRNAME, "resources", "plotpolicy_spectre.mplstyle")


class Wafer:
    def __init__(self, ui, callbacks_df):
        self.ui = ui
        self.callbacks_df = callbacks_df
        QSettings.setDefaultFormat(QSettings.IniFormat)
        self.settings = QSettings("CEA-Leti", "DaProViz")

        self.file_paths = []  # Store file_paths of all raw data wafers
        self.wafers = {}  # list of opened wafers

        self.ax = None
        self.canvas = None
        self.model_fs = None  # FITSPY
        self.spectra_fs = Spectra()  # FITSPY

        # Update spectra_listbox when selecting wafer via WAFER LIST
        self.ui.wafers_listbox.itemSelectionChanged.connect(
            self.upd_spectra_list)

        # Connect and plot_spectre of selected SPECTRUM LIST
        self.ui.spectra_listbox.itemSelectionChanged.connect(
            self.plot_sel_spectre)

        # Connect the stateChanged signal of the legend CHECKBOX
        self.ui.cb_legend.stateChanged.connect(self.plot_sel_spectre)

        self.ui.cb_raw.stateChanged.connect(self.plot_sel_spectre)
        self.ui.cb_bestfit.stateChanged.connect(self.plot_sel_spectre)
        self.ui.cb_colors.stateChanged.connect(self.plot_sel_spectre)
        self.ui.cb_residual.stateChanged.connect(self.plot_sel_spectre)
        self.ui.cb_filled.stateChanged.connect(self.plot_sel_spectre)

        # Set a 200ms delay for the function plot_sel_spectra
        self.delay_timer = QTimer()
        self.delay_timer.setSingleShot(True)
        self.delay_timer.timeout.connect(self.plot_sel_spectra)

    def open_csv(self, file_paths=None, wafers=None):
        """Open CSV files contaning RAW spectra of each wafer"""
        # Initialize the last used directory from QSettings
        if self.wafers is None:
            self.wafers = {}
        if wafers:
            self.wafers = wafers
        else:
            if file_paths is None:
                last_dir = self.settings.value("last_directory", "/")
                options = QFileDialog.Options()
                options |= QFileDialog.ReadOnly
                file_paths, _ = QFileDialog.getOpenFileNames(
                    self.ui.tabWidget, "Open RAW spectra CSV File(s)", last_dir,
                    "All Files (*)", options=options)
            # Load RAW spectra data from CSV files
            if file_paths:
                # Update the last used directory in QSettings
                last_dir = QFileInfo(file_paths[0]).absolutePath()
                self.settings.setValue("last_directory", last_dir)

                self.file_paths += file_paths
                for file_path in file_paths:
                    file_path = Path(file_path)
                    fname = file_path.stem

                    wafer_df = pd.read_csv(file_path, skiprows=1, delimiter=";")
                    wafer_name = fname
                    # Append or update the existing data in the self.wafers
                    # dictionary
                    if wafer_name in self.wafers:
                        print("wafer is already opened")
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
        self.ui.lb_loaded_model.setStyleSheet("color: yellow;")

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
        wafer_name_fs = "_".join(fname_parts[:2])
        coord_str = fname_parts[-1].split('(')[1].split(')')[0]
        coord_fs = tuple(map(float, coord_str.split(',')))
        return wafer_name_fs, coord_fs

    def fit_selected(self):
        """Fit only selected spectrum(s)"""
        if self.model_fs is None:
            self.show_alert("Please load a fit model before fitting.")
            return
        wafer_name, coords = self.spectre_id()  # Get current selected coords
        fnames = []
        for coord in coords:
            fname = f"{wafer_name}_{coord}"
            fnames.append(fname)
        self.spectra_fs.apply_model(self.model_fs, fnames=fnames)
        self.plot_sel_spectre()
        self.upd_spectra_list()

    def upd_spectra_list(self):
        """to update the spectra list"""
        self.ui.spectra_listbox.clear()
        self.clear_wafer_plot()
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
        self.ui.item_count_label.setText(f"Number of points: {item_count}")
        # Select the first item by default
        if self.ui.spectra_listbox.count() > 0:
            self.ui.spectra_listbox.setCurrentRow(0)
            QTimer.singleShot(50, self.plot_sel_spectre)

    def fit_all(self):
        """ Apply loaded fit model to all selected spectra"""
        if self.model_fs is None:
            self.show_alert("Load a fit model first!")
            return
        self.spectra_fs.apply_model(self.model_fs)
        self.plot_sel_spectre()
        self.upd_spectra_list()

    def collect_results(self):
        """Function to collect best-fit results and append in a dataframe"""
        # Add all dict into a list, then convert to a dataframe.
        fit_results_list = []
        self.df_fit_results = None

        for spectrum_fs in self.spectra_fs:
            if hasattr(spectrum_fs.result_fit, 'best_values'):
                wafer_name, coord = self.spectre_id_fs(spectrum_fs)
                x, y = coord
                best_values = spectrum_fs.result_fit.best_values

                best_values["Wafer"] = wafer_name
                best_values["X"] = x
                best_values["Y"] = y
                fit_results_list.append(best_values)

        self.df_fit_results = (pd.DataFrame(fit_results_list)).round(3)
        # Reordering columns and rename headers
        self.df_fit_results = self.df_reorder_rename(self.df_fit_results)

        # Add "Quadrant" columns
        self.df_fit_results['Quadrant'] = self.df_fit_results.apply(
            self.determine_quadrant, axis=1)

        self.apprend_cbb_param()
        self.apprend_cbb_wafer()

    def save_fit_results(self):
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

    def df_reorder_rename(self, df):
        """To reorder (x0, fwhm, ampli) and rename headers of dataframe"""
        # Reorder columns
        reordered_columns = []
        for param in ["x0", "fwhm", "ampli"]:
            for peak_label in sorted(
                    set(col.split("_")[0] for col in df.columns[:-3])):
                col_name = f"{peak_label}_{param}"
                reordered_columns.append(col_name)
        df = df[["Wafer", "X", "Y"] + reordered_columns]

        # Rename columns headers
        param_unit_mapping = {"ampli": "Intensity", "fwhm": "FWHM",
                              "x0": "Position"}
        for column in df.columns:
            if "_" in column:
                peak_label, param = column.split("_", 1)
                if param in param_unit_mapping:
                    unit = "(a.u)" if param == "ampli" else "(cm⁻¹)"
                    new_column_name = f"{param_unit_mapping[param]} of peak " \
                                      f"{peak_label} {unit}"
                    df.rename(columns={column: new_column_name}, inplace=True)
        return df

    def view_param_1(self):
        """ Plot WaferDataFrame"""
        self.clear_layout(self.ui.frame_wafer_1.layout())
        dfr = self.df_fit_results
        wafer_name = self.ui.cbb_wafer_1.currentText()
        color = self.ui.cbb_color_pallete.currentText()
        wafer_size = float(self.ui.wafer_size.text())
        if wafer_name is not None:
            selected_df = dfr.query('Wafer == @wafer_name')
        sel_param = self.ui.cbb_param_1.currentText()
        canvas = self.view_param_helper(selected_df, sel_param, wafer_size,
                                        color)
        self.ui.frame_wafer_1.addWidget(canvas)

    def view_param_2(self):
        """ Plot WaferDataFrame"""
        self.clear_layout(self.ui.frame_wafer_2.layout())
        dfr = self.df_fit_results
        wafer_name = self.ui.cbb_wafer_2.currentText()
        color = self.ui.cbb_color_pallete.currentText()
        wafer_size = float(self.ui.wafer_size.text())
        if wafer_name is not None:
            selected_df = dfr.query('Wafer == @wafer_name')
        sel_param = self.ui.cbb_param_2.currentText()
        canvas = self.view_param_helper(selected_df, sel_param, wafer_size,
                                        color)
        self.ui.frame_wafer_2.addWidget(canvas)

    def view_param_helper(self, selected_df, sel_param, wafer_size, color):
        x = selected_df['X']
        y = selected_df['Y']
        param = selected_df[sel_param]

        vmin = float(
            self.ui.int_vmin.text()) if self.ui.int_vmin.text() else None
        vmax = float(
            self.ui.int_vmax.text()) if self.ui.int_vmax.text() else None
        stats = self.ui.cb_stats.isChecked()
        plt.close('all')
        fig = plt.figure()
        ax = fig.add_subplot(111)

        wdf = WaferView()
        wdf.plot(ax, x=x, y=y, z=param, cmap=color, vmin=vmin, vmax=vmax,
                 stats=stats,
                 r=(wafer_size / 2))

        text = self.ui.plot_title.text()
        title = (f"{sel_param}") if not text else text
        ax.set_title(f"{title}")

        # Color scale

        fig.tight_layout()
        canvas = FigureCanvas(fig)
        return canvas

    def apprend_cbb_wafer(self, wafer_names=None):
        """to append all values of df_fit_results to comoboxses"""
        self.ui.cbb_wafer_1.clear()
        self.ui.cbb_wafer_2.clear()
        wafer_names = list(self.wafers.keys())
        for wafer_name in wafer_names:
            self.ui.cbb_wafer_1.addItem(wafer_name)
            self.ui.cbb_wafer_2.addItem(wafer_name)

    def apprend_cbb_param(self, df_fit_results=None):
        """to append all values of df_fit_results to comoboxses"""
        df_fit_results = self.df_fit_results
        columns = df_fit_results.columns.tolist()
        if df_fit_results is not None:
            self.ui.cbb_param_1.clear()
            self.ui.cbb_param_2.clear()
            for column in columns:
                self.ui.cbb_param_1.addItem(column)
                self.ui.cbb_param_2.addItem(column)

    def reinit_spectrum(self, spectrum):
        """Reinitialize the given spectrum"""
        spectrum.range_min = None
        spectrum.range_max = None
        spectrum.x = spectrum.x0.copy()
        spectrum.y = spectrum.y0.copy()
        spectrum.norm_mode = None
        spectrum.result_fit = lambda: None
        spectrum.remove_models()
        spectrum.baseline.points = [[], []]
        spectrum.baseline.is_subtracted = False

    def reinit_sel(self, fnames=None):
        """Reinitialize the selected spectrum(s)"""
        wafer_name, coords = self.spectre_id()  # Get current selected coords
        fnames = []
        for coord in coords:
            fname = f"{wafer_name}_{coord}"
            fnames.append(fname)
        for fname in fnames:
            spectrum, _ = self.spectra_fs.get_objects(fname)
            self.reinit_spectrum(spectrum)
        self.plot_sel_spectre()
        self.upd_spectra_list()

    def reinit_all(self):
        """Reinitialize all spectra"""
        fnames = [
            f"{self.spectre_id_fs(spectrum_fs)[0]}_" \
            f"{self.spectre_id_fs(spectrum_fs)[1]}"
            for spectrum_fs in self.spectra_fs]
        for fname in fnames:
            spectrum, _ = self.spectra_fs.get_objects(fname)
            self.reinit_spectrum(spectrum)
        self.plot_sel_spectre()
        self.upd_spectra_list()

    def reinit_fnc_handler(self):
        """Switch between 2 save fit fnc with the Ctrl key"""
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.reinit_all()
        else:
            self.reinit_sel()

    def plot_sel_spectra(self):
        """Plot all selected spectra"""
        plt.style.use(PLOT_POLICY)
        wafer_name, coords = self.spectre_id()  # current selected spectra ID
        selected_spectra_fs = []
        for spectrum_fs in self.spectra_fs:
            wafer_name_fs, coord_fs = self.spectre_id_fs(spectrum_fs)
            if wafer_name_fs == wafer_name and coord_fs in coords:
                selected_spectra_fs.append(spectrum_fs)
        if len(selected_spectra_fs) == 0:
            return
        self.clear_spectre_view()
        plt.close('all')
        fig = plt.figure()
        self.ax = fig.add_subplot(111)

        for spectrum_fs in selected_spectra_fs:
            fname, coord = self.spectre_id_fs(spectrum_fs)
            x_values = spectrum_fs.x
            y_values = spectrum_fs.y
            self.ax.plot(x_values, y_values)  # label=f"{fname}-{coord}"

            if self.ui.cb_raw.isChecked():
                x0_values = spectrum_fs.x0
                y0_values = spectrum_fs.y0
                self.ax.plot(x0_values, y0_values, 'ko-', label='raw', ms=3,
                             lw=1)

            if hasattr(spectrum_fs.result_fit, 'components') and hasattr(
                    spectrum_fs.result_fit, 'components') and \
                    self.ui.cb_bestfit.isChecked():
                bestfit = spectrum_fs.result_fit.best_fit
                self.ax.plot(x_values, bestfit, label=f"bestfit")

                for peak_model in spectrum_fs.result_fit.components:
                    prefix = str(peak_model.prefix)
                    params = peak_model.make_params()
                    y_peak = peak_model.eval(params, x=x_values)
                    if self.ui.cb_filled.isChecked():
                        self.ax.fill_between(x_values, 0, y_peak, alpha=0.5,
                                             label=f"{prefix}")
                    else:
                        self.ax.plot(x_values, y_peak, '--', label=f"{prefix}")

            if hasattr(spectrum_fs.result_fit,
                       'residual') and self.ui.cb_residual.isChecked():
                residual = spectrum_fs.result_fit.residual
                self.ax.plot(x_values, residual, 'ko-', ms=3, label='residual')

            if self.ui.cb_colors.isChecked() is False:
                self.ax.set_prop_cycle(None)

        self.ax.set_xlabel("Raman shift (cm$^{-1}$)")
        self.ax.set_ylabel("Intensity (a.u)")
        if self.ui.cb_legend.isChecked():
            self.ax.legend(loc='upper right')
        fig.tight_layout()
        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self.ui)
        self.ui.spectre_view_frame.addWidget(self.canvas)
        self.ui.toolbar_frame.addWidget(self.toolbar)

        self.plot_wafer()

    def plot_wafer(self):
        """Plot wafer maps of measurement sites"""
        plt.style.use(PLOT_POLICY)
        self.clear_wafer_plot()
        wafer_name, coords = self.spectre_id()
        all_x = []
        all_y = []
        for spectrum_fs in self.spectra_fs:
            wafer_name_fs, coord_fs = self.spectre_id_fs(spectrum_fs)
            if wafer_name == wafer_name_fs:
                x, y = coord_fs
                all_x.append(x)
                all_y.append(y)
        fig, ax = plt.subplots()
        ax.scatter(all_x, all_y, marker='x', color='black', s=10)
        # Highlight selected spectra in red
        if coords:
            selected_x, selected_y = zip(*coords)
            ax.scatter(selected_x, selected_y, marker='o', color='red')
        canvas = FigureCanvas(fig)
        layout = self.ui.wafer_plot.layout()
        if layout:
            layout.addWidget(canvas)

    def upd_wafers_list(self):
        """ To update the wafer listbox"""
        self.ui.wafers_listbox.clear()
        wafer_names = list(self.wafers.keys())
        for wafer_name in wafer_names:
            item = QListWidgetItem(wafer_name)
            self.ui.wafers_listbox.addItem(item)
            self.clear_wafer_plot()  # Clear the wafer_plot

        # Select the first item by default
        if self.ui.wafers_listbox.count() > 0:
            self.ui.wafers_listbox.setCurrentRow(0)
            QTimer.singleShot(100, self.upd_spectra_list)

    def remove_wafer(self):
        """To remove a wafer"""
        wafer_name, coords = self.spectre_id()
        if wafer_name in self.wafers:
            del self.wafers[wafer_name]
            self.spectra_fs = Spectra(
                spectrum_fs for spectrum_fs in self.spectra_fs if
                not spectrum_fs.fname.startswith(wafer_name))
            self.upd_wafers_list()
        self.ui.spectra_listbox.clear()
        self.clear_spectre_view()
        self.clear_wafer_plot()

    def clear_layout(self, layout):
        """Clear everything within a given Qlayout"""
        if layout is not None:
            for i in reversed(range(layout.count())):
                item = layout.itemAt(i)
                if isinstance(item.widget(),
                              (FigureCanvas, NavigationToolbar2QT)):
                    widget = item.widget()
                    layout.removeWidget(widget)
                    widget.close()

    def clear_spectre_view(self):
        """ Clear plot and toolbar within the spectre_view"""
        self.clear_layout(self.ui.spectre_view_frame.layout())
        self.clear_layout(self.ui.toolbar_frame.layout())

    def clear_wafer_plot(self):
        """ To clear wafer plot"""
        self.clear_layout(self.ui.wafer_plot.layout())

    def copy_fig(self):
        self.save_dpi = float(self.ui.ent_plot_save_dpi.text())
        if self.canvas:
            figure = self.canvas.figure
            with BytesIO() as buf:
                figure.savefig(buf, format='png', dpi=400)
                data = buf.getvalue()
            format_id = win32clipboard.RegisterClipboardFormat('PNG')
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(format_id, data)
            win32clipboard.CloseClipboard()
        else:
            QMessageBox.critical(None, "Error", "No plot to copy.")

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

    def plot_sel_spectre(self):
        """Trigger the fnc to plot spectre"""
        self.delay_timer.start(200)

    def show_alert(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("Alert")
        msg_box.setText(message)
        msg_box.exec_()

    def view_df(self, df):
        """To view selected dataframe"""
        # Create a QDialog to contain the table
        df_viewer = QDialog(self.ui.tabWidget)
        df_viewer.setWindowTitle("DataFrame Viewer")
        # Create a QTableWidget and populate it with data from the DataFrame
        table_widget = QTableWidget(df_viewer)
        table_widget.setColumnCount(df.shape[1])
        table_widget.setRowCount(df.shape[0])
        table_widget.setHorizontalHeaderLabels(df.columns)
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                item = QTableWidgetItem(str(df.iat[row, col]))
                table_widget.setItem(row, col, item)
        # Set the resizing mode for the QTableWidget to make it resizable
        table_widget.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
        # Use a QVBoxLayout to arrange the table within a scroll area
        layout = QVBoxLayout(df_viewer)
        layout.addWidget(table_widget)
        df_viewer.exec_()

    def view_fit_results_df(self):
        """To view selected dataframe"""
        self.view_df(self.df_fit_results)

    def view_wafer_data(self):
        """To view data of selected wafer """
        wafer_name, coords = self.spectre_id()
        self.view_df(self.wafers[wafer_name])

    def fit_fnc_handler(self):
        """Switch between 2 save fit fnc with the Ctrl key"""
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.fit_all()
        else:
            self.fit_selected()

    def send_df_to_vis(self):
        dfs = {}
        dfs["fitted_results"] = self.df_fit_results
        self.callbacks_df.action_open_df(file_paths=None,
                                         original_dfs=dfs)

    def determine_quadrant(self, row):
        if row['X'] < 0 and row['Y'] < 0:
            return 'Q1'
        elif row['X'] < 0 and row['Y'] > 0:
            return 'Q2'
        elif row['X'] > 0 and row['Y'] > 0:
            return 'Q3'
        elif row['X'] > 0 and row['Y'] < 0:
            return 'Q4'
        else:
            return np.nan

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

        # Show the 'report' of the first selected spectrum
        spectrum_fs = selected_spectra_fs[0]
        if spectrum_fs.result_fit:
            report = fit_report(spectrum_fs.result_fit)
            # Create a QDialog to display the report content
            report_viewer = QDialog(self.ui.tabWidget)
            report_viewer.setWindowTitle("Fitting Report")
            report_viewer.setGeometry(100, 100, 800, 600)

            # Create a QTextBrowser to display the report content
            text_browser = QTextBrowser(report_viewer)
            text_browser.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            text_browser.setOpenExternalLinks(True)

            # Display the report text in QTextBrowser
            text_browser.setPlainText(report)

            text_browser.moveCursor(
                QTextCursor.Start)  # Scroll to top of document

            layout = QVBoxLayout(report_viewer)
            layout.addWidget(text_browser)
            report_viewer.exec()  # Show the Report viewer dialog
