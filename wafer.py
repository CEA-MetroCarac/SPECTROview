# wafer.py module
import os
import copy
# import win32clipboard
from io import BytesIO
import numpy as np
import pandas as pd
from pathlib import Path
from lmfit import Model
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

from wafer_plot import WaferPlot

from PySide6.QtWidgets import (
    QFileDialog, QVBoxLayout, QMessageBox, QFrame, QPushButton,
    QHBoxLayout, QApplication, QSpacerItem, QSizePolicy, QListWidgetItem
)
from PySide6.QtGui import QStandardItemModel, QStandardItem

from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt, QSize, QCoreApplication, QSettings, QFileInfo, \
    QTimer

DIRNAME = os.path.dirname(__file__)
PLOT_POLICY = os.path.join(DIRNAME, "resources", "plotpolicy_spectre.mplstyle")


class Wafer:
    def __init__(self, ui):
        self.ui = ui
        QSettings.setDefaultFormat(QSettings.IniFormat)
        self.settings = QSettings("CEA-Leti", "DaProViz")

        self.file_paths = []  # Store file_paths of all raw data wafers
        self.wafers = {}  # list of opened wafers

        self.current_scale = None
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
        self.ui.display_loaded_model.setText(display_name)

    def spectre_id(self):
        """Get selected spectre id(s)"""
        wafer_name = self.ui.wafers_listbox.currentItem().text()
        selected_spectra = self.ui.spectra_listbox.selectedItems()
        coords = []
        if selected_spectra:
            for selected_item in selected_spectra:
                text = selected_item.text()
                x, y = map(float, text.strip('()').split(','))
                coord = (x, y)
                coords.append(coord)
        return wafer_name, coords

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

    def fit_all(self):
        """ Apply loaded fit model to all selected spectra"""
        if self.model_fs is None:
            self.show_alert("Load a fit model first!")
            return
        self.spectra_fs.apply_model(self.model_fs)
        self.plot_sel_spectre()

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

    def plot_sel_spectra(self):
        """Plot all selected spectra"""
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
            self.ax.plot(x_values, y_values, label=f"{fname}-{coord}")

            if self.ui.cb_raw.isChecked():
                x0_values = spectrum_fs.x0
                y0_values = spectrum_fs.y0
                self.ax.plot(x0_values, y0_values, label='raw')

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

        self.ax.set_xlabel("Raman shift (cm-1)")
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
                    self.ui.spectra_listbox.addItem(item)

        # Update the item count label
        item_count = self.ui.spectra_listbox.count()
        self.ui.item_count_label.setText(f"Number of points: {item_count}")
        # Select the first item by default
        if self.ui.spectra_listbox.count() > 0:
            self.ui.spectra_listbox.setCurrentRow(0)
            QTimer.singleShot(50, self.plot_sel_spectre)

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
