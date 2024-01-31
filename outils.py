# wafer.py module
import os
import copy
import win32clipboard
from io import BytesIO
import numpy as np
import pandas as pd
from pathlib import Path
from lmfit import Model

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
        self.ui = ui  # connect to main.py

        self.ax = None
        self.canvas = None
        self.current_scale = None
        QSettings.setDefaultFormat(QSettings.IniFormat)
        self.settings = QSettings("CEA-Leti", "DaProViz")

        self.wafers = {}  # list of opened wafers
        self.working_wafers = {}  # deep copy

        self.spectra = {}  # list of measurements sites within each wafer
        self.working_spectra = {}  # deep copy

        self.ax = None
        self.file_paths = []  # Store file_paths of all raw data wafers

        # Update spectra_listbox when selecting wafer via wafers_list
        self.ui.wafers_listbox.itemSelectionChanged.connect(
            self.upd_spectra_list)

        # Connect and plot_spectre of selected measurement_sites via list
        self.ui.spectra_listbox.itemSelectionChanged.connect(
            self.plot_sel_spectre)

        # Connect the stateChanged signal of the legend checkbox
        self.ui.checkbox_legend.stateChanged.connect(self.plot_sel_spectre)
        self.ui.checkbox_raw.stateChanged.connect(self.plot_sel_spectre)
        self.ui.checkbox_bestfit.stateChanged.connect(
            self.plot_sel_spectre)
        self.ui.checkbox_components.stateChanged.connect(
            self.plot_sel_spectre)
        self.ui.checkbox_residual.stateChanged.connect(
            self.plot_sel_spectre)
        self.ui.checkbox_fitlegend.stateChanged.connect(
            self.plot_sel_spectre)

        # Set a 200ms delay for the function plot_sel_spectre_delayed
        self.delay_timer = QTimer()
        self.delay_timer.setSingleShot(True)
        self.delay_timer.timeout.connect(self.plot_sel_spectre_delayed)

        self.current_scale = None

    def open_csv(self, file_paths=None, wafers=None):
        """To open RAW spectrum CSV files"""
        # Initialize the last used directory from QSettings
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
                    # Apprendre to the dictionary "wafers":
                    self.wafers[wafer_name] = wafer_df

        self.working_wafers = copy.deepcopy(self.wafers)
        self.upd_wafers_list()
        self.extract_spectre()  # extract spectra of all wafers dataframe

    def extract_spectre(self):
        """Extract individual spectra of each wafer"""
        self.spectra = {}

        # Iterate over each wafer within  dictionary "wafers"
        for wafer_name, wafer_df in self.wafers.items():
            # Extract two columns X, Y coordination
            coord_columns = wafer_df.columns[:2]

            # Create a dict to store all spectra of current wafer
            wafer_spectra = {}

            # Iterate over rows in dataframe of the current wafer
            for _, row in wafer_df.iterrows():
                # Extract X, Y coordination values
                coord_values = tuple(row[coord_columns])

                # Extract wavenumber values and convert to numeric
                wavenumbers = wafer_df.columns[2:].tolist()
                wavenumbers = pd.to_numeric(wavenumbers,
                                            errors='coerce').tolist()

                # Extract intensity values for the current row
                intensities = row[2:].tolist()

                # Create a dictionary to store all spectrum data at given coords
                spectrum = {'raw': {'wavenumber': wavenumbers,
                                    'intensity': intensities},
                            'bestfit': {'wavenumber': [], 'intensity': []},
                            'component': {'wavenumber': [], 'intensity': []},
                            'residual': {'wavenumber': [], 'intensity': []}}

                # Add the measurement site to the dictionary
                wafer_spectra[coord_values] = spectrum

            self.spectra[wafer_name] = wafer_spectra
        self.working_spectra = copy.deepcopy(self.spectra)

    def upd_wafers_list(self):
        """ To update the wafer listbox"""
        self.ui.wafers_listbox.clear()
        wafer_names = list(self.working_wafers.keys())
        for wafer_name in wafer_names:
            item = QListWidgetItem(wafer_name)
            self.ui.wafers_listbox.addItem(item)
            self.clear_wafer_plot()  # Clear the wafer_plot

    def upd_spectra_list(self):
        """to update the spectra list"""
        self.ui.spectra_listbox.clear()
        self.clear_wafer_plot()
        # Get the selected_wafer_name
        selected_item = self.ui.wafers_listbox.currentItem()
        if selected_item:
            selected_wafer_name = selected_item.text()

            if selected_wafer_name in self.spectra:
                wafer_spectra = self.spectra[selected_wafer_name]

                if wafer_spectra:
                    # get coord_values from wafer_spectra dictionary
                    for coord_values, spectrum in wafer_spectra.items():
                        coord_str = f"X: {coord_values[0]}, Y: " \
                                    f"{coord_values[1]}"

                        item = QListWidgetItem(coord_str)
                        self.ui.spectra_listbox.addItem(item)

    def remove_wafer(self):
        """To remove a wafer from wafer_listbox"""
        selected_item = self.ui.wafers_listbox.currentItem()
        if selected_item:
            selected_wafer_name = selected_item.text()
            if selected_wafer_name in self.wafers:
                del self.wafers[selected_wafer_name]
                del self.working_wafers[selected_wafer_name]
                self.upd_wafers_list()

        # Clear spectra_listbox and spectre_view
        self.ui.spectra_listbox.clear()
        self.clear_spectre_view()
        self.clear_wafer_plot()

    def clear_spectra_list(self):
        """To clear spectra listbox"""
        self.ui.spectra_listbox.clear()

    def clear_layout(self, layout):
        """ Clear everything within a given layout"""
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

    ###########################################
    def set_scale(self, x_limits=None, y_limits=None):
        """Set the zoom level of the plot."""
        ax = self.ui.spectre_view_frame.layout().itemAt(0).widget().figure.gca()
        if x_limits is not None:
            ax.set_xlim(x_limits)
        if y_limits is not None:
            ax.set_ylim(y_limits)
        # Redraw the plot with the updated limits
        self.ui.spectre_view_frame.layout().itemAt(0).widget().draw()

    def get_current_scale(self, ax):
        """Update the current scale with the last zoom state."""
        self.current_scale = ax._get_view()
        print(self.current_scale)

    def rescale(self):
        """Rescale the figure."""
        self.ax.autoscale(enable=True, axis='x')
        self.canvas.draw()

    def plot_spectre(self, x=None, y=None, coord=None):
        """To plot and show (fitted) spectre(s)"""
        plt.style.use(PLOT_POLICY)
        self.clear_spectre_view()

        if x is not None and y is not None:
            plt.close(
                'all')  # Explicitly close all open figures before creating.

            # Create a figure with initial size
            fig = plt.figure()
            self.ax = fig.add_subplot(111)

            if isinstance(x, (list, np.ndarray)) and isinstance(y, (
                    list, np.ndarray)) and isinstance(coord,
                                                      (list, np.ndarray)):
                for wavenumbers, intensities, (x_coord, y_coord, key) in zip(x,
                                                                             y,
                                                                             coord):
                    # Check the state of the checkboxes and plot accordingly
                    if key in self.get_selected_keys_to_plot():
                        self.ax.plot(wavenumbers, intensities,
                                     label=f"X: {x_coord}, Y: {y_coord}_{key}")

            self.ax.set_xlabel("Raman shift (cm-1)")
            self.ax.set_ylabel("Intensity (a.u)")

            # Check the state of the legend checkbox
            if self.ui.checkbox_legend.isChecked():
                self.ax.legend(loc='upper right')

        fig.tight_layout()

        # Create a FigureCanvas & NavigationToolbar2QT
        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self.ui)

        home_action = next(
            a for a in self.toolbar.actions() if a.text() == 'Home')
        home_action.triggered.connect(self.rescale)

        # add fig canvas et toolbar into frame
        self.ui.spectre_view_frame.addWidget(self.canvas)
        self.ui.toolbar_frame.addWidget(self.toolbar)

        # Connect the draw_event signal to a function that updates the
        # zoom state
        self.canvas.mpl_connect("draw_event",
                                lambda event: self.get_current_scale(self.ax))

        if self.current_scale is not None:
            self.set_scale(x_limits=self.current_scale['xlim'],
                           y_limits=self.current_scale['ylim'])

    def plot_sel_spectre(self):
        """Trigger the fnc to plot spectre"""
        self.delay_timer.start(200)

    def plot_sel_spectre_delayed(self):
        """" Plot all selected spectra"""
        # Get the selected wafer name
        selected_wafer_item = self.ui.wafers_listbox.currentItem()

        if selected_wafer_item:
            selected_wafer_name = selected_wafer_item.text()

            # Check if selected wafer name exists in sites dict
            if selected_wafer_name in self.spectra:
                wafer_spectra = self.spectra[selected_wafer_name]

                # Get the selected items in the measurement_sites list
                selected_items = \
                    self.ui.spectra_listbox.selectedItems()

                if selected_items:
                    # lists to store spectra and coordinate of selected sites
                    all_wavenumbers = []
                    all_intensities = []
                    all_coords = []

                    for selected_item in selected_items:
                        selected_text = selected_item.text()

                        # Extract XY coordinates from the selected spectra
                        parts = selected_text.split(", ")
                        x_coord = float(parts[0].split(":")[1])
                        y_coord = float(parts[1].split(":")[1])

                        if (x_coord, y_coord) in wafer_spectra:
                            selected_spectre = wafer_spectra[(x_coord, y_coord)]

                            for key, spectrum_data in selected_spectre.items():
                                wavenumbers = spectrum_data['wavenumber']
                                intensities = spectrum_data['intensity']

                                all_wavenumbers.append(wavenumbers)
                                all_intensities.append(intensities)
                                all_coords.append((x_coord, y_coord, key))

                    # Call plot_spectre to plot the selected spectra
                    self.plot_spectre(all_wavenumbers, all_intensities,
                                      all_coords)

                    # Wafer plot
                    selected_coords = self.get_selected_coords()
                    self.plot_wafer(selected_wafer_name, selected_coords)

                else:
                    # If no measurement site is selected, clear plot view
                    self.clear_spectre_view()
            else:
                # If the selected wafer name does not exist, clear plot view
                self.clear_spectre_view()
                self.clear_spectra_list()
                self.clear_wafer_plot()
        else:
            # If no wafer is selected, clear the view
            self.clear_spectre_view()
            self.clear_spectra_list()
            self.clear_wafer_plot()

    def get_selected_keys_to_plot(self):
        """Get the selected keys to plot based on checkbox states."""
        selected_keys = []
        if self.ui.checkbox_raw.isChecked():
            selected_keys.append('raw')
        if self.ui.checkbox_bestfit.isChecked():
            selected_keys.append('bestfit')
        if self.ui.checkbox_residual.isChecked():
            selected_keys.append('residual')
        if self.ui.checkbox_components.isChecked():
            selected_keys.append('component')
        return selected_keys

    def plot_wafer(self, wafer_name, selected_coords=None):
        # Clear the existing plot
        self.clear_wafer_plot()

        # Check if the wafer_name exists in the 'spectra' dictionary
        if wafer_name in self.spectra:
            wafer_spectra = self.spectra[wafer_name]

            # Extract X, Y coordinates
            x_coords, y_coords = zip(*wafer_spectra.keys())
            plt.close('all')
            fig, ax = plt.subplots()
            ax.scatter(x_coords, y_coords, marker='x', color='gray', s=10)

            # Highlight selected coordinates in red
            if selected_coords:
                selected_x, selected_y = zip(*selected_coords)
                ax.scatter(selected_x, selected_y, marker='o', color='red')

            # fig.tight_layout()
            self.canvas = FigureCanvas(fig)
            layout = self.ui.wafer_plot.layout()

            if layout:
                layout.addWidget(self.canvas)

    def get_selected_coords(self):
        """"Collect all coordinate of selected spectra"""
        selected_coords = []

        # Get the selected_wafer_name
        selected_item = self.ui.wafers_listbox.currentItem()
        if selected_item:
            selected_wafer_name = selected_item.text()

            # Check if a wafer is selected
            if selected_wafer_name in self.spectra:
                wafer_spectra = self.spectra[selected_wafer_name]

                # Get the selected items in the measurement_sites list
                selected_items = self.ui.spectra_listbox.selectedItems()

                for selected_item in selected_items:
                    selected_text = selected_item.text()

                    # Extract X and Y coordinates from the selected item's text
                    parts = selected_text.split(", ")
                    x_coord = float(parts[0].split(":")[1])
                    y_coord = float(parts[1].split(":")[1])

                    if (x_coord, y_coord) in wafer_spectra:
                        selected_coords.append((x_coord, y_coord))

        return selected_coords

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

    def fitting(self):
        """Fit selected spectra using lmfit."""
        # Get the selected wafer name
        selected_wafer_item = self.ui.wafers_listbox.currentItem()

        if selected_wafer_item:
            selected_wafer_name = selected_wafer_item.text()

            # Check if selected wafer name exists in sites dict
            if selected_wafer_name in self.spectra:
                wafer_spectra = self.spectra[selected_wafer_name]

                # Get the selected items in the measurement_sites list
                selected_items = \
                    self.ui.spectra_listbox.selectedItems()

                if selected_items:
                    for selected_item in selected_items:
                        selected_text = selected_item.text()

                        # Extract XY coordinates from the selected spectra
                        parts = selected_text.split(", ")
                        x_coord = float(parts[0].split(":")[1])
                        y_coord = float(parts[1].split(":")[1])

                        if (x_coord, y_coord) in wafer_spectra:
                            selected_spectre = wafer_spectra[(x_coord, y_coord)]

                            # Perform fitting on the selected spectrum
                            self.fit_spectrum(selected_spectre)

    def fit_spectrum(self, spectrum_data):
        """Fit a spectrum using lmfit."""

        # Define the fitting model (Lorentzians + linear baseline)
        def lorentzian(x, amp, cen, wid):
            return (amp / np.pi) * (wid / ((x - cen) ** 2 + wid ** 2))

        def baseline(x, slope, intercept):
            return slope * x + intercept

        def composite_model(x, amp1, cen1, wid1, amp2, cen2, wid2,
                            amp3, cen3, wid3, amp4, cen4, wid4,
                            amp5, cen5, wid5, slope, intercept):
            return (lorentzian(x, amp1, cen1, wid1) +
                    lorentzian(x, amp2, cen2, wid2) +
                    lorentzian(x, amp3, cen3, wid3) +
                    lorentzian(x, amp4, cen4, wid4) +
                    lorentzian(x, amp5, cen5, wid5) +
                    baseline(x, slope, intercept))

        model = Model(composite_model)

        # Get the spectral data
        wavenumbers = spectrum_data['raw']['wavenumber']
        intensities = spectrum_data['raw']['intensity']

        params = model.make_params(
            amp1=1, cen1=390, wid1=10,
            amp2=1, cen2=408, wid2=10,
            amp3=1, cen3=375, wid3=10,
            amp4=1, cen4=411, wid4=10,
            amp5=1, cen5=453, wid5=10,
            slope=0, intercept=np.min(intensities)
        )

        # Set bounds for fitting parameters
        params['wid1'].min = params['wid2'].min = params['wid3'].min = \
            params['wid4'].min = params['wid5'].min = 0

        # Set the fitting range (350 - 470 cm-1)
        fitting_range = (350, 470)
        wavenumbers_array = np.array(wavenumbers)  # Convert to NumPy array
        mask = np.where((wavenumbers_array >= fitting_range[0]) & (
                wavenumbers_array <= fitting_range[1]))

        # Perform the fit using the indices from the mask
        result = model.fit(np.array(intensities)[mask], params,
                           x=wavenumbers_array[mask], nan_policy='omit')

        # Extract the best-fit values
        bestfit_params = result.best_values

        # Update the 'bestfit' key in the spectrum_data dictionary
        spectrum_data['bestfit'] = {'wavenumber': wavenumbers,
                                    'intensity': result.best_fit}

        # You can also access the best-fit parameters like bestfit_params
        # bestfit_params['amp1'], bestfit_params['cen1'], etc.

        # Plot the raw spectrum and best-fit curve
        self.plot_spectre([wavenumbers, wavenumbers],
                          [intensities, result.best_fit],
                          [(0, 0, 'raw'), (0, 0, 'bestfit')])

        # You may want to update the UI or store the fit results as needed
