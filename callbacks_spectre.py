# callbacks_spectre.py module
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


class CallbacksSpectre:
    def __init__(self, ui):
        self.ui = ui
        QSettings.setDefaultFormat(QSettings.IniFormat)
        self.settings = QSettings("CEA-Leti", "DaProViz")

        self.file_paths = []  # Store file_paths of all raw data wafers
        self.wafers = {}  # list of opened wafers
        self.spectra = {}  # list of all spectra within all wafer
        self.current_scale = None
        self.ax = None
        self.canvas = None

        self.spectra_fs = None  # FITSPY
        self.model_fs = None  # FITSPY
        self.spectra_fs = None  # FITSPY

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
        self.ui.cb_components.stateChanged.connect(self.plot_sel_spectre)
        self.ui.cb_residual.stateChanged.connect(self.plot_sel_spectre)
        self.ui.cb_raw_bl.stateChanged.connect(self.plot_sel_spectre)

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
        self.upd_wafers_list()
        self.extract_spectra()  # extract spectra of all wafers df

    def extract_spectra(self, spectra=None):
        """Extract all spectra of each wafer dataframe"""
        self.spectra_fs = Spectra()  # Create an Fitspy object

        if self.spectra is None:
            self.spectra = {}
        if spectra:
            self.spectra = spectra

        for wafer_name, wafer_df in self.wafers.items():
            coord_columns = wafer_df.columns[:2]  # Extract XY coordination
            wafer_spectra = {}  # Dict to store all spectrum of a given wafer

            for _, row in wafer_df.iterrows():
                # Extract XY coordinate, wavenumber and intensity values
                coords = tuple(row[coord_columns])
                x_values = wafer_df.columns[2:].tolist()
                x_values = pd.to_numeric(x_values, errors='coerce').tolist()
                y_values = row[2:].tolist()

                spectrum = {'wavenumber': x_values, 'intensity': y_values}
                # Add or update all spectrum into the dict of given wafer
                wafer_spectra[coords] = spectrum

                if wafer_name in self.spectra:
                    if coords in self.spectra[wafer_name]:
                        self.spectra[wafer_name][coords].update(spectrum)
                    else:
                        self.spectra[wafer_name][coords] = spectrum
                else:
                    self.spectra[wafer_name] = wafer_spectra

                # FITSPY
                spectrum_fs = Spectrum()
                spectrum_fs.x = np.asarray(x_values)
                spectrum_fs.x0 = np.asarray(x_values)
                spectrum_fs.y = np.asarray(y_values)
                spectrum_fs.y0 = np.asarray(y_values)
                spectrum_fs.fname = f"{wafer_name}_{coords}"

                self.spectra_fs.append(spectrum_fs)
            print(self.spectra_fs)

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
        # Convert the coordinates to a tuple of floats
        coord_fs = tuple(map(float, coord_str.split(',')))

        return wafer_name_fs, coord_fs

    def plot_spectre(self, x=None, y=None, coord=None):
        """To plot raw spectra"""

        self.clear_spectre_view()
        if x is not None and y is not None:
            plt.close('all')
            # Create a figure with initial size
            fig = plt.figure()
            self.ax = fig.add_subplot(111)

            if isinstance(x, (list, np.ndarray)) and isinstance(y, (
                    list, np.ndarray)) and isinstance(coord,
                                                      (list, np.ndarray)):
                for wavenumbers, intensities, coords in zip(x, y, coord):
                    self.ax.plot(wavenumbers, intensities, label=f"{coord}_raw")

            self.ax.set_xlabel("Raman shift (cm-1)")
            self.ax.set_ylabel("Intensity (a.u)")
            if self.ui.cb_legend.isChecked():
                self.ax.legend(loc='upper right')
        fig.tight_layout()

        # Create a FigureCanvas & NavigationToolbar2QT
        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self.ui)
        home_action = next(
            a for a in self.toolbar.actions() if a.text() == 'Home')
        home_action.triggered.connect(self.rescale)

        # Add fig canvas and toolbar into frame
        self.ui.spectre_view_frame.addWidget(self.canvas)
        self.ui.toolbar_frame.addWidget(self.toolbar)

        # Connect the get current scale
        self.canvas.mpl_connect("draw_event",
                                lambda event: self.get_current_scale(self.ax))

    def plot_sel_spectra(self):
        """Plot all selected spectra"""
        wafer_name, coords = self.spectre_id()

        if wafer_name is not None and coords is not None:
            wafer_data = self.spectra[wafer_name]
            if coords:
                all_wavenumbers = []
                all_intensities = []
                all_coords = []

                for coord in coords:
                    sel_spectrum = wafer_data.get(
                        coord)  # Use get to handle missing keys
                    wavenumbers = sel_spectrum.get('wavenumber', [])
                    intensities = sel_spectrum.get('intensity', [])
                    all_wavenumbers.append(wavenumbers)
                    all_intensities.append(intensities)
                    all_coords.append(coord)
                self.plot_spectre(all_wavenumbers, all_intensities, all_coords)
                self.plot_wafer(wafer_name, coords)  # Wafer plot
            else:
                self.clear_spectre_view()
        else:
            self.clear_spectre_view()
            self.ui.spectra_listbox.clear()
            self.clear_wafer_plot()

    def plot_wafer(self, wafer_name, selected_coords=None):
        """Plot wafer maps of measurement sites"""
        self.clear_wafer_plot()

        if wafer_name in self.spectra:
            wafer_spectra = self.spectra[wafer_name]

            # Extract x_coords and y_coords from the keys of wafer_spectra
            all_coords = list(wafer_spectra.keys())
            all_x_coords, all_y_coords = zip(*all_coords)
            plt.close('all')

            fig, ax = plt.subplots()
            ax.scatter(all_x_coords, all_y_coords, marker='x', color='gray',
                       s=10)

            # Highlight selected spectra in red
            if selected_coords:
                selected_x, selected_y = zip(*selected_coords)
                ax.scatter(selected_x, selected_y, marker='o', color='red')

            self.canvas = FigureCanvas(fig)
            layout = self.ui.wafer_plot.layout()
            if layout:
                layout.addWidget(self.canvas)

    ###########################################
    def set_scale(self, x_limits=None, y_limits=None):
        """Set the zoom level of the plot."""
        ax = self.ax
        if x_limits is not None:
            ax.set_xlim(x_limits)
        if y_limits is not None:
            ax.set_ylim(y_limits)
        self.canvas.draw()

    def get_current_scale(self, ax):
        """Update the current scale with the last zoom state."""
        self.current_scale = ax._get_view()
        # print('current scale is collected')
        # print(self.current_scale)

    def rescale(self):
        """Rescale the figure."""
        self.ax.autoscale()
        self.canvas.draw()
        self.toolbar.home()

    def fitting_all_wafer(self):
        """ Apply loaded fit model to all selected spectra"""
        if self.model_fs is None:
            self.show_alert("Please load a fit model before fitting.")
            return

        self.selected_spectra_fs = Spectra()
        for spectrum_fs in self.spectra_fs:
            current_spectrum_fs = copy.deepcopy(spectrum_fs)
            self.selected_spectra_fs.append(current_spectrum_fs)

        self.selected_spectra_fs.apply_model(self.model_fs, ncpu=4,
                                             fit_only=False)
        self.fitted_spectra_fs = copy.deepcopy(self.selected_spectra_fs)

        self.update_wafer_data()
        self.plot_sel_spectre()
        print("All spectra are fitted")

    def fitting_sel_spectrum(self):
        """Fit only selected spectrum(s)"""
        if self.model_fs is None:
            self.show_alert("Please load a fit model before fitting.")
            return
        wafer_name, coords = self.spectre_id()

        self.selected_spectra_fs = Spectra()

        for spectrum_fs in self.spectra_fs:
            wafer_name_fs, coord_fs = self.spectre_id_fs(spectrum_fs)
            # Fit the selected spectrum
            if wafer_name_fs == wafer_name and coord_fs == coords:
                current_spectrum_fs = copy.deepcopy(spectrum_fs)
                self.selected_spectra_fs.append(current_spectrum_fs)

        self.selected_spectra_fs.apply_model(self.model_fs, ncpu=4,
                                             fit_only=True)
        self.fitted_spectra_fs = copy.deepcopy(self.selected_spectra_fs)

        self.update_wafer_data()
        self.plot_sel_spectre()
        print("Selected spectrum is fitted")

    def update_wafer_data(self):
        """Update wafer data (self.spectra) with fitted results"""
        for fitted_spectrum_fs in self.fitted_spectra_fs:
            fname_parts = fitted_spectrum_fs.fname.split("_")
            wafer_name_fs = fname_parts[0] + "_" + fname_parts[1]
            coord_values_str = fname_parts[-1].split('(')[1].split(')')[
                0].split(',')
            x_coord_fs = float(coord_values_str[0])
            y_coord_fs = float(coord_values_str[1])

            # Extracting fitted results
            x = fitted_spectrum_fs.x.tolist()
            y = fitted_spectrum_fs.y.tolist()
            best_fit = fitted_spectrum_fs.result_fit.best_fit.tolist()
            residual = fitted_spectrum_fs.result_fit.residual.tolist()

            # Update the self.spectra dataframe with the fitted results
            coords = (x_coord_fs, y_coord_fs)
            wafer_data = self.spectra[wafer_name_fs]
            # Initialize the keys before updating
            wafer_data[coords]['bestfit'] = {'wavenumber': [], 'intensity': []}
            wafer_data[coords]['residual'] = {'wavenumber': [], 'intensity': []}
            wafer_data[coords]['raw_bl'] = {'wavenumber': [], 'intensity': []}

            # Update the 'bestfit' spectrum with fitted results
            wafer_data[coords]['bestfit'] = {'wavenumber': x,
                                             'intensity': best_fit}
            wafer_data[coords]['residual'] = {'wavenumber': x,
                                              'intensity': residual}
            wafer_data[coords]['raw_bl'] = {'wavenumber': x, 'intensity': y}

            for model in fitted_spectrum_fs.result_fit.components:
                params = model.make_params()
                prefix = model.prefix
                y_model = (model.eval(params, x=x)).tolist()
                if f'{prefix}' not in wafer_data[coords]:
                    wafer_data[coords][f'{prefix}'] = {}
                    wafer_data[coords][f'{prefix}']['wavenumber'] = x
                    wafer_data[coords][f'{prefix}']['intensity'] = y_model

    def get_selected_keys_to_plot(self):
        """Get the selected keys to plot based on checkbox states."""
        selected_keys = []
        if self.ui.cb_raw.isChecked():
            selected_keys.append('raw')
        if self.ui.cb_raw_bl.isChecked():
            selected_keys.append('raw_bl')
        if self.ui.cb_bestfit.isChecked():
            selected_keys.append('bestfit')
        if self.ui.cb_residual.isChecked():
            selected_keys.append('residual')
        return selected_keys

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
        # Get the selected_wafer_name
        wafer_name = self.ui.wafers_listbox.currentItem().text()
        if wafer_name in self.spectra:
            wafer_spectra = self.spectra[wafer_name]
            if wafer_spectra:
                for coord_values, spectrum in wafer_spectra.items():
                    coord_str = f"({coord_values[0]},{coord_values[1]})"
                    item = QListWidgetItem(coord_str)
                    self.ui.spectra_listbox.addItem(item)
        # Update the item count label
        item_count = self.ui.spectra_listbox.count()
        self.ui.item_count_label.setText(f"Number of points: {item_count}")
        # Select the first item by default
        if self.ui.spectra_listbox.count() > 0:
            self.ui.spectra_listbox.setCurrentRow(0)
            QTimer.singleShot(100, self.plot_sel_spectre)

    def remove_wafer(self):
        """To remove a wafer from wafer_listbox"""
        selected_item = self.ui.wafers_listbox.currentItem()
        if selected_item:
            selected_wafer_name = selected_item.text()
            if selected_wafer_name in self.wafers:
                del self.spectra[selected_wafer_name]
                del self.wafers[selected_wafer_name]
                # Remove spectra from self.spectra_fs
                self.spectra_fs = [spectrum_fs for spectrum_fs in
                                   self.spectra_fs if
                                   not spectrum_fs.fname.startswith(
                                       selected_wafer_name)]

                self.upd_wafers_list()
        # Clear spectra_listbox and spectre_view
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

    def copy_fig(self, canvas):
        self.save_dpi = float(self.ui.ent_plot_save_dpi.text())
        # canvas = self.ui.spectre_view_frame.layout().itemAt(0).widget()
        if canvas:
            figure = canvas.figure
            with BytesIO() as buf:
                figure.savefig(buf, format='png', dpi=self.save_dpi)
                data = buf.getvalue()
            format_id = win32clipboard.RegisterClipboardFormat('PNG')
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(format_id, data)
            win32clipboard.CloseClipboard()
        else:
            QMessageBox.critical(None, "Error", "No plot to copy.")

    def select_all_spectra(self):
        """ To quickly select all spectra within the listbox"""
        item_count = self.ui.spectra_listbox.count()
        for i in range(item_count):
            item = self.ui.spectra_listbox.item(i)
            item.setSelected(True)

    def select_spectra_vertical(self):
        """ To select all spectra vertically within the listbox"""
        self.ui.spectra_listbox.clearSelection()  # Clear all selection
        item_count = self.ui.spectra_listbox.count()
        for i in range(item_count):
            item = self.ui.spectra_listbox.item(i)
            coord_str = item.text()
            # Extract X and Y coordinates from the item's text
            parts = coord_str.split(", ")
            x_coord = float(parts[0].split(":")[1])
            if x_coord == 0:
                item.setSelected(True)

    def select_spectra_horizontal(self):
        """ To quickly select all spectra vertically within the listbox"""
        self.ui.spectra_listbox.clearSelection()  # Clear all selection
        item_count = self.ui.spectra_listbox.count()
        for i in range(item_count):
            item = self.ui.spectra_listbox.item(i)
            coord_str = item.text()
            # Extract X and Y coordinates from the item's text
            parts = coord_str.split(", ")
            y_coord = float(parts[1].split(":")[1])

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
