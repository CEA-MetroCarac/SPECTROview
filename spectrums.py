"""
Module dedicated to the 'Spectrum(s)' TAB of the main GUI
"""
import os
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
import dill

from common import view_df, show_alert, Filter
from common import FitThread, FitModelManager, ShowParameters, DataframeTable
from common import PLOT_POLICY, NCPUS, FIT_METHODS

from lmfit import fit_report
from fitspy.spectra import Spectra
from fitspy.spectrum import Spectrum
from fitspy.app.gui import Appli
from fitspy.utils import closest_index
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from PySide6.QtWidgets import (QFileDialog, QMessageBox, QApplication,
                               QListWidgetItem, QCheckBox, QListWidget,
                               QAbstractItemView)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt, QFileInfo, QTimer, QObject, Signal
from tkinter import Tk, END


class Spectrums(QObject):
    """
    Class manages the GUI interactions and operations related to spectra fittings,
    and visualization of fitted data within "SPACTRA" TAB of the application.


    Signals:
        fit_progress_changed:
            Signal emitted to indicate progress during fitting operations.

    Attributes:
        fit_progress_changed (Signal):
            Signal emitted to update the progress during fitting operations.
        settings (QSettings):
            Application settings object for storing persistent settings.
        ui (QObject):
            UI object providing access to GUI elements.
        common (object):
            Object for common utility functions used across the application.
        visu (object):
            Visualization object for managing plot-related functionalities.
        loaded_fit_model (object):
            Currently loaded fit model object.
        current_fit_model (object):
            Currently selected fit model object.
        spectrums (Spectra):
            Collection of Spectrum objects containing spectral data.
        df_fit_results (DataFrame):
            DataFrame containing fit results.
        filter (Filter):
            Object managing filtering operations on fit results.
        filtered_df (DataFrame):
            Filtered DataFrame based on applied filters.
    """
    # Define a signal for progress updates
    fit_progress_changed = Signal(int)

    def __init__(self, settings, ui, common, visu):
        super().__init__()
        self.settings = settings
        self.ui = ui
        self.common = common
        self.visu = visu

        self.loaded_fit_model = None
        self.current_fit_model = None
        self.spectrums = Spectra()
        self.df_fit_results = None

        # Create a customized QListWidget
        self.ui.spectrums_listbox = CustomListWidget()
        self.ui.spectrums_listbox.setDragDropMode(QListWidget.InternalMove)
        self.ui.listbox_layout.addWidget(self.ui.spectrums_listbox)
        self.ui.spectrums_listbox.items_reordered.connect(
            self.update_spectrums_order)

        # FILTER: Create an instance of the FILTER class
        self.filter = Filter(self.ui.ent_filter_query_2,
                             self.ui.filter_listbox,
                             self.df_fit_results)
        self.filtered_df = None
        # Connect filter signals to filter methods
        self.ui.btn_add_filter_2.clicked.connect(self.filter.add_filter)
        self.ui.ent_filter_query_2.returnPressed.connect(self.filter.add_filter)
        self.ui.btn_remove_filters_2.clicked.connect(self.filter.remove_filter)
        self.ui.btn_apply_filters_2.clicked.connect(self.apply_filters)

        # Connect and plot_spectra of selected SPECTRUM LIST
        self.ui.spectrums_listbox.itemSelectionChanged.connect(
            self.plot_delay)
        # Connect the stateChanged signal of the legend CHECKBOX
        self.ui.cb_legend_3.stateChanged.connect(self.plot_delay)
        self.ui.cb_raw_3.stateChanged.connect(self.plot_delay)
        self.ui.cb_bestfit_3.stateChanged.connect(self.plot_delay)
        self.ui.cb_colors_3.stateChanged.connect(self.plot_delay)
        self.ui.cb_residual_3.stateChanged.connect(self.plot_delay)
        self.ui.cb_filled_3.stateChanged.connect(self.plot_delay)
        self.ui.cb_peaks_3.stateChanged.connect(self.plot_delay)
        self.ui.cb_attached_3.stateChanged.connect(self.plot_delay)
        self.ui.cb_normalize_3.stateChanged.connect(self.plot_delay)

        self.ui.cb_limits_2.stateChanged.connect(self.plot_delay)
        self.ui.cb_expr_2.stateChanged.connect(self.plot_delay)

        # Set a delay for the function plot1
        self.delay_timer = QTimer()
        self.delay_timer.setSingleShot(True)
        self.delay_timer.timeout.connect(self.plot1)
        # Connect the progress signal to update_progress_bar slot
        self.fit_progress_changed.connect(self.update_pbar)

        self.plot_styles = ["point plot", "scatter plot", "box plot",
                            "bar plot"]
        self.create_plot_widget()
        self.create_spectra_plot_widget()
        self.zoom_pan_active = False

        self.ui.cbb_fit_methods_2.addItems(FIT_METHODS)
        self.ui.cbb_cpu_number_2.addItems(NCPUS)

        # FIT SETTINGS
        self.load_fit_settings()

        self.ui.cb_fit_negative_2.stateChanged.connect(self.save_fit_settings)
        self.ui.max_iteration_2.valueChanged.connect(self.save_fit_settings)
        self.ui.cbb_fit_methods_2.currentIndexChanged.connect(
            self.save_fit_settings)
        self.ui.cbb_cpu_number_2.currentIndexChanged.connect(
            self.save_fit_settings)
        self.ui.xtol_2.textChanged.connect(self.save_fit_settings)
        self.ui.sb_dpi_spectra_2.valueChanged.connect(
            self.create_spectra_plot_widget)

        # BASELINE
        self.ui.cb_attached_3.clicked.connect(self.upd_spectra_list)
        self.ui.noise_2.valueChanged.connect(self.upd_spectra_list)
        self.ui.rbtn_linear_2.clicked.connect(self.upd_spectra_list)
        self.ui.rbtn_polynomial_2.clicked.connect(self.upd_spectra_list)
        self.ui.degre_2.valueChanged.connect(self.upd_spectra_list)

        # Load default folder path from QSettings during application startup
        self.fit_model_manager = FitModelManager(self.settings)
        self.fit_model_manager.default_model_folder = self.settings.value(
            "default_model_folder", "")
        self.ui.l_defaut_folder_model_3.setText(
            self.fit_model_manager.default_model_folder)
        QTimer.singleShot(0, self.populate_available_models)

    def update_spectrums_order(self):
        """
        Update the order of spectrums when user rearranges them via listbox drag-and-drop.

        This method retrieves the new order of spectrums from the customized
        QListWidget and updates the internal list of spectrums accordingly.
        """

        new_order = []
        for index in range(self.ui.spectrums_listbox.count()):
            item_text = self.ui.spectrums_listbox.item(index).text()
            # Find the corresponding spectrum in the original list
            for spectrum in self.spectrums:
                if spectrum.fname == item_text:
                    new_order.append(spectrum)
                    break
        # Clear the existing list and update with the new order
        self.spectrums.clear()
        self.spectrums.extend(new_order)

    def open_data(self, spectra=None, file_paths=None):
        """
        Open and load raw spectral data from file paths or provided Spectra object.

        If file_paths is not provided, a file dialog is opened to select text files.
        The loaded spectra are added to the internal Spectra object.

        Args:
            spectra (Spectra, optional): Spectra object containing pre-loaded data.
            file_paths (list of str, optional): List of file paths to load raw spectra data.
        """
        if self.spectrums is None:
            self.spectrums = Spectra()
        if spectra:
            self.spectrums = spectra
        else:
            if file_paths is None:
                # Initialize the last used directory from QSettings
                last_dir = self.settings.value("last_directory", "/")
                options = QFileDialog.Options()
                options |= QFileDialog.ReadOnly
                file_paths, _ = QFileDialog.getOpenFileNames(
                    self.ui.tabWidget, "Open RAW spectra TXT File(s)", last_dir,
                    "Text Files (*.txt)", options=options)

            if file_paths:
                last_dir = QFileInfo(file_paths[0]).absolutePath()
                self.settings.setValue("last_directory", last_dir)

                for file_path in file_paths:
                    file_path = Path(file_path)
                    fname = file_path.stem

                    # Check if fname is already opened
                    if any(spectrum.fname == fname for spectrum in
                           self.spectrums):
                        print(f"Spectrum '{fname}' is already opened.")
                        continue

                    dfr = pd.read_csv(file_path, header=None, skiprows=1,
                                      delimiter="\t")
                    dfr_sorted = dfr.sort_values(by=0)  # increasing order
                    # Convert values to float
                    dfr_sorted.iloc[:, 0] = dfr_sorted.iloc[:, 0].astype(float)
                    dfr_sorted.iloc[:, 1] = dfr_sorted.iloc[:, 1].astype(float)

                    x_values = dfr_sorted.iloc[:, 0].tolist()
                    y_values = dfr_sorted.iloc[:, 1].tolist()

                    # create FITSPY object
                    spectrum = Spectrum()
                    spectrum.fname = fname
                    spectrum.x = np.asarray(x_values)
                    spectrum.x0 = np.asarray(x_values)
                    spectrum.y = np.asarray(y_values)
                    spectrum.y0 = np.asarray(y_values)
                    self.spectrums.append(spectrum)
        QTimer.singleShot(100, self.upd_spectra_list)

    def upd_spectra_list(self):
        """
        Update the spectrums list in the UI based on the current data.

        This method updates the QListWidget in the UI to display the list
        of loaded spectra filenames. It manages the visibility of baseline
        and noise parameters based on user selections.
        """

        current_row = self.ui.spectrums_listbox.currentRow()
        self.ui.spectrums_listbox.clear()
        for spectrum in self.spectrums:
            fname = spectrum.fname
            item = QListWidgetItem(fname)
            if hasattr(spectrum.result_fit,
                       'success') and spectrum.result_fit.success:
                item.setBackground(QColor("green"))
            elif hasattr(spectrum.result_fit,
                         'success') and not \
                    spectrum.result_fit.success:
                item.setBackground(QColor("orange"))
            else:
                item.setBackground(QColor(0, 0, 0, 0))
            self.ui.spectrums_listbox.addItem(item)

        item_count = self.ui.spectrums_listbox.count()
        self.ui.item_count_label_3.setText(f"{item_count} points")
        # Management of selecting item of listbox
        if current_row >= item_count:
            current_row = item_count - 1
        if current_row >= 0:
            self.ui.spectrums_listbox.setCurrentRow(current_row)
        else:
            if item_count > 0:
                self.ui.spectrums_listbox.setCurrentRow(0)
        QTimer.singleShot(50, self.plot_delay)

    def on_click(self, event):
        """
        Handle click events on spectra plot canvas for adding peak models or baseline points.

        This method interprets click events on the spectra plot canvas to add peak models
        or baseline points. It manages the UI interaction based on the type of click (left
        click for adding peaks, right click for baseline points).

        Args:
            event (QMouseEvent): Mouse click event containing information about the click position.
        """
        sel_spectrum, sel_spectra = self.get_spectrum_object()
        fit_model = self.ui.cbb_fit_models_2.currentText()

        # Add a new peak_model for current selected peak
        if self.zoom_pan_active == False and self.ui.rdbtn_peak_2.isChecked():
            if event.button == 1 and event.inaxes:
                x = event.xdata
                y = event.ydata
            sel_spectrum.add_peak_model(fit_model, x)
            self.upd_spectra_list()

        # Add a new baseline point for current selected peak
        if self.zoom_pan_active == False and \
                self.ui.rdbtn_baseline_2.isChecked():
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

    def apply_filters(self, filters=None):
        """
        Apply currently checked filters to the fit results DataFrame.

        This method applies the currently checked filters in the UI to the fit results
        DataFrame. It updates the filtered_df attribute with the filtered results.

        Args:
            filters (list of str, optional): List of filter names to apply. Defaults to None.
        """
        self.filter.set_dataframe(self.df_fit_results)
        self.filtered_df = self.filter.apply_filters()
        self.display_df_in_GUI(self.filtered_df)

    def set_default_model_folder(self, folder_path=None):
        """
        Set the default folder path for fit models.

        This method sets the default folder path for loading and saving fit models.
        It updates the settings and UI elements accordingly.

        Args:
            folder_path (str, optional): Folder path to set as default. Defaults to None.
        """
        if not folder_path:
            folder_path = QFileDialog.getExistingDirectory(None,
                                                           "Select Default "
                                                           "Folder",
                                                           options=QFileDialog.ShowDirsOnly)

        if folder_path:
            self.fit_model_manager.set_default_model_folder(folder_path)
            # Save selected folder path back to QSettings
            self.settings.setValue("default_model_folder", folder_path)
            self.ui.l_defaut_folder_model_3.setText(
                self.fit_model_manager.default_model_folder)
            QTimer.singleShot(0, self.populate_available_models)

    def populate_available_models(self):
        """
        Populate the available fit models in the UI combobox.

        This method populates the available fit models from the default model folder
        into the UI combobox for model selection.
        """
        # Scan default folder and populate available models in the combobox
        self.available_models = self.fit_model_manager.get_available_models()
        self.ui.cbb_fit_model_list_3.clear()
        self.ui.cbb_fit_model_list_3.addItems(self.available_models)

    def load_fit_model(self, fname_json=None):
        """
        Load a fit model from a JSON file or from the UI selection.

        This method loads a fit model from a specified JSON file or from the current
        selection in the UI combobox. It updates the loaded_fit_model and current_fit_model
        attributes accordingly.

        Args:
            fname_json (str, optional): File name of the JSON file to load. Defaults to None.
        """
        self.fname_json = fname_json
        self.upd_model_cbb_list()
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
            self.fname_json = selected_file
        display_name = QFileInfo(self.fname_json).fileName()
        # Add the display name to the combobox only if it doesn't already exist
        if display_name not in [self.ui.cbb_fit_model_list_3.itemText(i) for i
                                in
                                range(self.ui.cbb_fit_model_list_3.count())]:
            self.ui.cbb_fit_model_list_3.addItem(display_name)
            self.ui.cbb_fit_model_list_3.setCurrentText(display_name)
        else:
            show_alert('Fit model is already available in the model list')

    def get_loaded_fit_model(self):
        """
        Retrieve the currently loaded fit model from the UI.

        Returns:
            object: Currently loaded fit model object.
        """
        if self.ui.cbb_fit_model_list_3.currentIndex() == -1:
            self.loaded_fit_model = None
            return
        try:
            # If the file is not found in the selected path, try finding it
            # in the default folder
            folder_path = self.fit_model_manager.default_model_folder
            model_name = self.ui.cbb_fit_model_list_3.currentText()
            path = os.path.join(folder_path, model_name)
            self.loaded_fit_model = self.spectrums.load_model(path, ind=0)
        except FileNotFoundError:
            try:
                self.loaded_fit_model = self.spectrums.load_model(
                    self.fname_json, ind=0)
            except FileNotFoundError:
                show_alert('Fit model file not found in the default folder.')

    def save_fit_model(self):
        """
        Save the fit model of the currently selected spectrum to a JSON file.

        This method saves the fit model of the currently selected spectrum to
        a JSON file in the default model folder.
        """
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

    def upd_model_cbb_list(self):
        """
        Update and populate the model list in the UI combobox.

        This method updates the model list in the UI combobox based on the available
        fit models loaded from the default model folder.
        """
        current_path = self.fit_model_manager.default_model_folder
        self.set_default_model_folder(current_path)

    def apply_loaded_fit_model(self, fnames=None):
        """
        Apply the loaded fit model to selected spectra.

        This method applies the loaded fit model to the selected spectra
        based on the provided file names.

        Args:
            fnames (list of str, optional): List of filenames to apply the fit model. Defaults to None.
        """
        self.get_loaded_fit_model()
        self.ui.btn_apply_model_3.setEnabled(False)
        if self.loaded_fit_model is None:
            show_alert(
                "Select from the list or load a fit model before fitting.")
            self.ui.btn_apply_model_3.setEnabled(True)
            return

        if fnames is None:
            fnames = self.get_spectrum_fnames()

        # Start fitting process in a separate thread
        self.apply_model_thread = FitThread(self.spectrums,
                                            self.loaded_fit_model,
                                            fnames)
        # To update progress bar
        self.apply_model_thread.fit_progress_changed.connect(self.update_pbar)
        # To display progress in GUI
        self.apply_model_thread.fit_progress.connect(
            lambda num, elapsed_time: self.fit_progress(num, elapsed_time,
                                                        fnames))
        # To update spectra list + plot fitted spectrum once fitting finished
        self.apply_model_thread.fit_completed.connect(self.fit_completed)
        self.apply_model_thread.finished.connect(
            lambda: self.ui.btn_apply_model_3.setEnabled(True))
        self.apply_model_thread.start()

    def apply_loaded_fit_model_all(self):
        """
        Apply the loaded fit model to all spectra.

        This method applies the loaded fit model to all spectra in the current
        Spectra object.
        """
        fnames = self.spectrums.fnames
        self.apply_loaded_fit_model(fnames=fnames)

    def get_spectrum_fnames(self):
        """
        Get the filenames of currently selected spectra in the UI.

        Returns:
            list of str: List of filenames of currently selected spectra.
        """
        items = self.ui.spectrums_listbox.selectedItems()
        fnames = []
        for item in items:
            text = item.text()
            fnames.append(text)
        return fnames

    def get_spectrum_object(self):
        """
        Get the Spectrum object of currently selected spectra.

        Returns:
            Spectrum: Spectrum object of currently selected spectra.
        """
        fnames = self.get_spectrum_fnames()
        sel_spectra = []
        for spectrum in self.spectrums:
            if spectrum.fname in fnames:
                sel_spectra.append(spectrum)
        if len(sel_spectra) == 0:
            return
        sel_spectrum = sel_spectra[0]
        return sel_spectrum, sel_spectra

    def create_spectra_plot_widget(self):
        """
        Create the widget for displaying spectra plots.

        This method creates the widget for displaying spectra plots and
        initializes the necessary settings for interactive plotting.
        """
        plt.style.use(PLOT_POLICY)
        self.common.clear_layout(self.ui.QVBoxlayout_2.layout())
        self.common.clear_layout(self.ui.toolbar_frame_3.layout())
        self.upd_spectra_list()
        dpi = float(self.ui.sb_dpi_spectra_2.text())

        fig1 = plt.figure(dpi=dpi)
        self.ax = fig1.add_subplot(111)
        self.ax.set_xlabel("Raman shift (cm$^{-1}$)")
        self.ax.set_ylabel("Intensity (a.u)")
        self.ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
        self.canvas1 = FigureCanvas(fig1)
        self.canvas1.mpl_connect('button_press_event', self.on_click)

        # Toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas1, self.ui)
        rescale = next(
            a for a in self.toolbar.actions() if a.text() == 'Home')
        rescale.triggered.connect(self.rescale)
        for action in self.toolbar.actions():
            if action.text() == 'Pan' or action.text() == 'Zoom':
                action.toggled.connect(self.toggle_zoom_pan)

        self.ui.QVBoxlayout_2.addWidget(self.canvas1)
        self.ui.toolbar_frame_3.addWidget(self.toolbar)
        self.canvas1.figure.tight_layout()
        self.canvas1.draw()

    def rescale(self):
        """
        Rescale the spectra plot to fit within the axes.

        This method rescales the spectra plot to fit within the axes
        while preserving the aspect ratio.
        """
        self.ax.autoscale()
        self.canvas1.draw()

    def toggle_zoom_pan(self, checked):
        """
        Toggle zoom and pan functionality for spectra plot.

        Args:
            checked (bool): True if zoom and pan are enabled; False otherwise.
        """
        self.zoom_pan_active = checked
        if not checked:
            self.zoom_pan_active = False

    def create_plot_widget(self):
        """
        Create canvas and toolbar for plotting in the GUI.

        This method creates the canvas and toolbar for plotting in the GUI.
        """
        # Plot2: graph1
        fig2 = plt.figure(dpi=90)
        self.ax2 = fig2.add_subplot(111)
        self.canvas2 = FigureCanvas(fig2)
        self.ui.frame_graph_3.addWidget(self.canvas2)
        self.canvas2.draw()

        # Plot3: graph2
        fig3 = plt.figure(dpi=90)
        self.ax3 = fig3.add_subplot(111)
        self.canvas3 = FigureCanvas(fig3)
        self.ui.frame_graph_7.addWidget(self.canvas3)
        self.canvas3.draw()

    def plot1(self):
        """
        Plot spectra or fit results in the main plot area.

        This method plots the spectra or fit results in the main plot area
        based on user interactions or selections.
        """
        fnames = self.get_spectrum_fnames()
        selected_spectrums = []

        for spectrum in self.spectrums:
            fname = spectrum.fname
            if fname in fnames:
                selected_spectrums.append(spectrum)
        if len(selected_spectrums) == 0:
            return

        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.clear()
        # reassign previous axis limits (related to zoom)
        if not xlim == ylim == (0.0, 1.0):
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

        for spectrum in selected_spectrums:
            fname = spectrum.fname
            x_values = spectrum.x
            y_values = spectrum.y

            # NORMALIZE
            if self.ui.cb_normalize_3.isChecked():
                max_intensity = 0.0
                max_intensity = max(max_intensity, max(spectrum.y))
                y_values = y_values / max_intensity
            self.ax.plot(x_values, y_values, label=f"{fname}", ms=3, lw=2)

            # BASELINE
            self.plot_baseline_dynamically(ax=self.ax, spectrum=spectrum)

            # RAW
            if self.ui.cb_raw_3.isChecked():
                x0_values = spectrum.x0
                y0_values = spectrum.y0
                self.ax.plot(x0_values, y0_values, 'ko-', label='raw', ms=3,
                             lw=1)
            # BEST-FIT and PEAK_MODELS
            if hasattr(spectrum.result_fit, 'components') and hasattr(
                    spectrum.result_fit, 'components') and \
                    self.ui.cb_bestfit_3.isChecked():
                bestfit = spectrum.result_fit.best_fit
                self.ax.plot(x_values, bestfit, label=f"bestfit")
            if self.ui.cb_bestfit_3.isChecked():
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

                    if self.ui.cb_filled_3.isChecked():
                        self.ax.fill_between(x_values, 0, y_peak, alpha=0.5,
                                             label=f"{peak_label}")
                        if self.ui.cb_peaks_3.isChecked():
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
            # RESIDUAL
            if hasattr(spectrum.result_fit,
                       'residual') and self.ui.cb_residual_3.isChecked():
                residual = spectrum.result_fit.residual
                self.ax.plot(x_values, residual, 'ko-', ms=3, label='residual')
            if self.ui.cb_colors_3.isChecked() is False:
                self.ax.set_prop_cycle(None)

            # R-SQUARED
            if hasattr(spectrum.result_fit, 'rsquared'):
                rsquared = round(spectrum.result_fit.rsquared, 4)
                self.ui.rsquared_2.setText(f"R2={rsquared}")
            else:
                self.ui.rsquared_2.setText("R2=0")

        self.ax.set_xlabel("Raman shift (cm$^{-1}$)")
        self.ax.set_ylabel("Intensity (a.u)")
        if self.ui.cb_legend_3.isChecked():
            self.ax.legend(loc='upper right')
        self.ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
        self.ax.get_figure().tight_layout()
        self.canvas1.draw()
        self.read_x_range()
        self.show_peak_table()

    def read_x_range(self):
        """Read the x range of the selected spectrum.

        This method retrieves the selected spectrum object, reads its x-axis range,
        and sets the minimum and maximum x values in the corresponding GUI text fields.
        """
        sel_spectrum, sel_spectra = self.get_spectrum_object()
        self.ui.range_min_2.setText(str(sel_spectrum.x[0]))
        self.ui.range_max_2.setText(str(sel_spectrum.x[-1]))

    def set_x_range(self, fnames=None):
        """Set a new x range for the selected spectrum.

        This method updates the x range of the selected spectrum(s) based on the values
        provided in the GUI. It recalculates the spectrum data within the specified range
        and updates the GUI accordingly.

        Args:
            fnames (list, optional): List of filenames of the spectra to be updated.
                                     If None, the filenames are retrieved from the GUI selection.
        """
        new_x_min = float(self.ui.range_min_2.text())
        new_x_max = float(self.ui.range_max_2.text())
        if fnames is None:
            fnames = self.get_spectrum_fnames()
        self.common.reinit_spectrum(fnames, self.spectrums)
        for fname in fnames:
            spectrum, _ = self.spectrums.get_objects(fname)
            spectrum.range_min = float(self.ui.range_min_2.text())
            spectrum.range_max = float(self.ui.range_max_2.text())

            ind_min = closest_index(spectrum.x0, new_x_min)
            ind_max = closest_index(spectrum.x0, new_x_max)
            spectrum.x = spectrum.x0[ind_min:ind_max + 1].copy()
            spectrum.y = spectrum.y0[ind_min:ind_max + 1].copy()
            spectrum.attractors_calculation()
        QTimer.singleShot(50, self.upd_spectra_list)
        QTimer.singleShot(300, self.rescale)

    def set_x_range_all(self):
        """Set a new x range for all spectra.

        This method updates the x range for all spectra using the range values
        provided in the GUI and recalculates the spectrum data within the specified range.
        """
        fnames = self.spectrums.fnames
        self.set_x_range(fnames=fnames)

    def get_baseline_settings(self):
        """Retrieve baseline settings from the GUI and apply to the selected spectrum.

        This method extracts the baseline settings (e.g., attached baseline, noise level,
        baseline mode, polynomial order) from the GUI and applies them to the selected spectrum object.
        """
        sel_spectrum, sel_spectra = self.get_spectrum_object()
        if sel_spectrum is None:
            return
        sel_spectrum.baseline.attached = self.ui.cb_attached_3.isChecked()
        sel_spectrum.baseline.sigma = self.ui.noise_2.value()
        if self.ui.rbtn_linear_2.isChecked():
            sel_spectrum.baseline.mode = "Linear"
        else:
            sel_spectrum.baseline.mode = "Polynomial"
            sel_spectrum.baseline.order_max = self.ui.degre_2.value()

    def plot_baseline_dynamically(self, ax, spectrum):
        """Dynamically evaluate and plot the baseline for a given spectrum.

        This method retrieves the baseline settings, evaluates the baseline points and line,
        and plots them dynamically on the provided axis.

        Args:
            ax (matplotlib.axes.Axes): The axis on which to plot the baseline.
            spectrum (object): The spectrum object for which the baseline is being plotted.
        """
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
        """Subtract the baseline for the selected spectrum(s).

        This method performs baseline subtraction on the selected spectrum(s)
        using the points defined in the baseline settings.

        Args:
            sel_spectra (list, optional): List of selected spectra objects. If None,
                                          the selected spectra are retrieved from the GUI.
        """
        sel_spectrum, _ = self.get_spectrum_object()
        points = deepcopy(sel_spectrum.baseline.points)
        if len(points[0]) == 0:
            return
        if sel_spectra is None:
            _, sel_spectra = self.get_spectrum_object()
        for spectrum in sel_spectra:
            spectrum.baseline.points = points.copy()
            spectrum.subtract_baseline()
        QTimer.singleShot(50, self.upd_spectra_list)
        QTimer.singleShot(300, self.rescale)

    def subtract_baseline_all(self):
        """Subtract the baseline for all spectra.

        This method performs baseline subtraction on all spectra using the points
        defined in the baseline settings.
        """
        self.subtract_baseline(self.spectrums)

    def clear_peaks(self, fnames=None):
        """Clear all existing peak models of the selected spectrum(s).

        This method removes all peak models from the selected spectra.

        Args:
            fnames (list, optional): List of filenames of the spectra to be cleared.
                                     If None, the filenames are retrieved from the GUI selection.
        """
        if fnames is None:
            fnames = self.get_spectrum_fnames()
        for fname in fnames:
            spectrum, _ = self.spectrums.get_objects(fname)
            if len(spectrum.peak_models) != 0:
                spectrum.remove_models()
            else:
                continue
        QTimer.singleShot(100, self.upd_spectra_list)

    def clear_peaks_all(self):
        """Clear peaks for all spectra.

        This method removes all peak models from all spectra.
        """
        fnames = self.spectrums.fnames
        self.clear_peaks(fnames)

    def get_fit_settings(self):
        """Retrieve all settings for the fitting action from the GUI.

        This method extracts the fitting parameters (e.g., fit_negative, max_iterations,
        fitting method, number of CPUs, tolerance) from the GUI and applies them to the
        selected spectrum object.
        """
        sel_spectrum, sel_spectra = self.get_spectrum_object()
        fit_params = sel_spectrum.fit_params.copy()
        fit_params['fit_negative'] = self.ui.cb_fit_negative_2.isChecked()
        fit_params['max_ite'] = self.ui.max_iteration_2.value()
        fit_params['method'] = self.ui.cbb_fit_methods_2.currentText()
        fit_params['ncpus'] = self.ui.cbb_cpu_number_2.currentText()
        fit_params['xtol'] = float(self.ui.xtol_2.text())
        sel_spectrum.fit_params = fit_params

    def fit(self, fnames=None):
        """Fit the selected spectrum(s) with current parameters.

        This method performs fitting on the selected spectrum(s) using the parameters
        defined in the GUI and updates the GUI accordingly.

        Args:
            fnames (list, optional): List of filenames of the spectra to be fitted.
                                     If None, the filenames are retrieved from the GUI selection.
        """
        self.get_fit_settings()
        if fnames is None:
            fnames = self.get_spectrum_fnames()
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

    def copy_fit_model(self):
        """Copy the model dictionary of the selected spectrum.

        This method copies the model dictionary of the first spectrum in the
        selected spectra list, if it exists, and sets the copied model in the GUI.
        """
        # Get only 1 spectrum among several selected spectrum:
        self.get_fit_settings()
        sel_spectrum, _ = self.get_spectrum_object()
        if len(sel_spectrum.peak_models) == 0:
            self.ui.lbl_copied_fit_model_2.setText("")
            show_alert(
                "The selected spectrum does not have fit model to be copied!")
            self.current_fit_model = None
            return
        else:
            self.current_fit_model = None
            self.current_fit_model = deepcopy(sel_spectrum.save())
        self.ui.lbl_copied_fit_model_2.setText("copied")

    def paste_fit_model(self, fnames=None):
        """Apply the copied fit model to the selected spectra.

        This method pastes the copied fit model onto the selected spectra and
        performs fitting in a separate thread.

        Args:
            fnames (list, optional): List of filenames of the spectra to be fitted.
                                     If None, the filenames are retrieved from the GUI selection.
        """
        # Get fnames of all selected spectra
        self.ui.btn_paste_fit_model_2.setEnabled(False)

        if fnames is None:
            fnames = self.get_spectrum_fnames()

        self.common.reinit_spectrum(fnames, self.spectrums)
        fit_model = deepcopy(self.current_fit_model)
        if self.current_fit_model is not None:
            # Starting fit process in a seperate thread
            self.paste_model_thread = FitThread(self.spectrums, fit_model,
                                                fnames)
            self.paste_model_thread.fit_progress_changed.connect(
                self.update_pbar)
            self.paste_model_thread.fit_progress.connect(
                lambda num, elapsed_time: self.fit_progress(num, elapsed_time,
                                                            fnames))
            self.paste_model_thread.fit_completed.connect(self.fit_completed)
            self.paste_model_thread.finished.connect(
                lambda: self.ui.btn_paste_fit_model_2.setEnabled(True))
            self.paste_model_thread.start()
        else:
            show_alert("Nothing to paste")
            self.ui.btn_paste_fit_model_2.setEnabled(True)

    def paste_fit_model_all(self):
        """Paste the copied fit model (in clipboard) and apply to selected spectrum(s)."""

        fnames = self.spectrums.fnames
        self.paste_fit_model(fnames)

    def show_peak_table(self):
        """Show all fitted parameters in the GUI."""

        sel_spectrum, sel_spectra = self.get_spectrum_object()
        main_layout = self.ui.peak_table1_2
        cb_limits = self.ui.cb_limits_2
        cb_expr = self.ui.cb_expr_2
        update = self.upd_spectra_list
        show_params = ShowParameters(main_layout, sel_spectrum, cb_limits,
                                     cb_expr, update)
        show_params.show_peak_table(main_layout, sel_spectrum, cb_limits,
                                    cb_expr)

    def collect_results(self):
        """Collect best-fit results and append them to a dataframe."""

        # Add all dict into a list, then convert to a dataframe.
        self.copy_fit_model()
        fit_results_list = []
        self.df_fit_results = None

        for spectrum in self.spectrums:
            if hasattr(spectrum.result_fit, 'best_values'):
                success = spectrum.result_fit.success
                rsquared = spectrum.result_fit.rsquared
                best_values = spectrum.result_fit.best_values
                best_values["Filename"] = spectrum.fname
                best_values["success"] = success
                best_values["Rsquared"] = rsquared
                fit_results_list.append(best_values)
        self.df_fit_results = (pd.DataFrame(fit_results_list)).round(3)

        if self.df_fit_results is not None and not self.df_fit_results.empty:
            # reindex columns according to the parameters names
            self.df_fit_results = self.df_fit_results.reindex(
                sorted(self.df_fit_results.columns),
                axis=1)
            names = []
            for name in self.df_fit_results.columns:
                if name in ["Filename", "success"]:
                    name = '0' + name
                elif '_' in name:
                    name = 'z' + name[5:]
                names.append(name)
            self.df_fit_results = self.df_fit_results.iloc[:,
                                  list(np.argsort(names, kind='stable'))]
            columns = [
                self.common.translate_param(self.current_fit_model, column) for
                column
                in self.df_fit_results.columns]
            self.df_fit_results.columns = columns

            self.display_df_in_GUI(self.df_fit_results)
        else:
            self.ui.fit_results_table_2.clear()

        self.filtered_df = self.df_fit_results
        self.upd_cbb_param()
        self.send_df_to_viz()

    def display_df_in_GUI(self, df):
        """Display a given DataFrame in the GUI via QTableWidget.

        Args:
            df (pd.DataFrame): The DataFrame to be displayed in the GUI.
        """
        df_table = DataframeTable(df, self.ui.layout_df_table2)

    def split_fname(self):
        """Split the filename and populate the combobox."""

        dfr = self.df_fit_results
        fname_parts = dfr.loc[0, 'Filename'].split('_')
        self.ui.cbb_split_fname.clear()  # Clear existing items in combobox
        for part in fname_parts:
            self.ui.cbb_split_fname.addItem(part)

    def add_column(self):
        """Add a column to the dataframe of fit results based on the split_fname method."""

        dfr = self.df_fit_results
        col_name = self.ui.ent_col_name.text()
        selected_part_index = self.ui.cbb_split_fname.currentIndex()
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
        self.display_df_in_GUI(self.df_fit_results)
        self.send_df_to_viz()
        self.upd_cbb_param()

    def upd_cbb_param(self):
        """Append all values of df_fit_results to comboboxes."""

        if self.df_fit_results is not None:
            columns = self.df_fit_results.columns.tolist()
            self.ui.cbb_x_3.clear()
            self.ui.cbb_y_3.clear()
            self.ui.cbb_z_3.clear()
            self.ui.cbb_x_7.clear()
            self.ui.cbb_y_7.clear()
            self.ui.cbb_z_7.clear()
            self.ui.cbb_x_3.addItem("None")
            self.ui.cbb_y_3.addItem("None")
            self.ui.cbb_z_3.addItem("None")
            self.ui.cbb_x_7.addItem("None")
            self.ui.cbb_y_7.addItem("None")
            self.ui.cbb_z_7.addItem("None")
            for column in columns:
                self.ui.cbb_x_3.addItem(column)
                self.ui.cbb_y_3.addItem(column)
                self.ui.cbb_z_3.addItem(column)
                self.ui.cbb_x_7.addItem(column)
                self.ui.cbb_y_7.addItem(column)
                self.ui.cbb_z_7.addItem(column)

    def send_df_to_viz(self):
        """Send the collected spectral data dataframe to the visualization tab."""

        dfs_new = self.visu.original_dfs
        dfs_new["SPECTRUMS_best_fit"] = self.df_fit_results
        self.visu.open_dfs(dfs=dfs_new, fnames=None)

    def plot2(self):
        """Plot graph """
        if self.filtered_df is not None:
            dfr = self.filtered_df
        else:
            dfr = self.df_fit_results
        x = self.ui.cbb_x_3.currentText()
        y = self.ui.cbb_y_3.currentText()

        z = self.ui.cbb_z_3.currentText()
        if z == "None":
            hue = None
        else:
            hue = z if z != "" else None

        style = self.ui.cbb_plot_style_3.currentText()
        xmin = self.ui.xmin_3.text()
        ymin = self.ui.ymin_3.text()
        xmax = self.ui.xmax_3.text()
        ymax = self.ui.ymax_3.text()

        title = self.ui.ent_plot_title_5.text()
        x_text = self.ui.ent_xaxis_lbl_3.text()
        y_text = self.ui.ent_yaxis_lbl_3.text()

        text = self.ui.ent_x_rot_3.text()
        xlabel_rot = 0  # Default rotation angle
        if text:
            xlabel_rot = float(text)

        ax = self.ax2
        self.common.plot_graph(ax, dfr, x, y, hue, style, xmin, xmax, ymin,
                               ymax,
                               title,
                               x_text, y_text, xlabel_rot)
        self.ax2.get_figure().tight_layout()
        self.canvas2.draw()

    def plot3(self):
        if self.filtered_df is not None:
            dfr = self.filtered_df
        else:
            dfr = self.df_fit_results
        x = self.ui.cbb_x_7.currentText()
        y = self.ui.cbb_y_7.currentText()
        z = self.ui.cbb_z_7.currentText()

        if z == "None":
            hue = None
        else:
            hue = z if z != "" else None
        style = self.ui.cbb_plot_style_7.currentText()
        xmin = self.ui.xmin_3.text()
        ymin = self.ui.ymin_3.text()
        xmax = self.ui.xmax_3.text()
        ymax = self.ui.ymax_3.text()

        title = self.ui.ent_plot_title_5.text()
        x_text = self.ui.ent_xaxis_lbl_3.text()
        y_text = self.ui.ent_yaxis_lbl_3.text()

        text = self.ui.ent_x_rot_3.text()
        xlabel_rot = 0  # Default rotation angle
        if text:
            xlabel_rot = float(text)

        ax = self.ax3
        self.common.plot_graph(ax, dfr, x, y, hue, style, xmin, xmax, ymin,
                               ymax,
                               title,
                               x_text, y_text, xlabel_rot)

        self.ax3.get_figure().tight_layout()
        self.canvas3.draw()

    def plot_delay(self):
        """Trigger the fnc to plot spectra"""
        self.delay_timer.start(100)

    def fit_progress(self, num, elapsed_time, fnames):
        """Update progress when the fitting process is completed.

        Args:
            num (int): Number of fitted spectra.
            elapsed_time (float): Time taken to complete the fitting.
            fnames (list): List of filenames of the fitted spectra.
        """
        self.ui.progress_3.setText(
            f"{num}/{len(fnames)} fitted ({elapsed_time:.2f}s)")

    def fit_completed(self):
        """
        Actions to perform upon completion of fitting process.

        Updates the spectra list, triggers a delayed rescaling of plots, and
        performs necessary GUI updates after the fitting process completes.
        """
        self.plot_delay()
        self.upd_spectra_list()
        QTimer.singleShot(200, self.rescale)

    def update_pbar(self, progress):
        """
        Update the progress bar with current progress.

        Sets the value of the progress bar on the GUI to indicate the current
        progress of an ongoing operation.

        Parameters:
        - progress: Current progress value (0-100).
        """
        self.ui.progressBar_3.setValue(progress)

    def cosmis_ray_detection(self):
        self.spectrums.outliers_limit_calculation()

    def reinit(self, fnames=None):
        """Reinitialize the selected spectrum(s).

        Args:
            fnames (list, optional): List of filenames of the spectra to be reinitialized. Defaults to None.
        """
        if fnames is None:
            fnames = self.get_spectrum_fnames()
        self.common.reinit_spectrum(fnames, self.spectrums)
        self.plot_delay()
        self.upd_spectra_list()
        QTimer.singleShot(200, self.rescale)

    def reinit_all(self):
        """Reinitialize all spectra"""
        fnames = self.spectrums.fnames
        self.reinit(fnames)

    def view_fit_results_df(self):
        """To view selected dataframe"""
        if self.filtered_df is None:
            df = self.df_fit_results
        else:
            df = self.filtered_df

        if df is not None:
            view_df(self.ui.tabWidget, df)
        else:
            show_alert("No fit dataframe to display")

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
                self.df_fit_results = None
                self.df_fit_results = dfr
                self.filtered_df = dfr
            except Exception as e:
                show_alert("Error loading DataFrame:", e)
        self.display_df_in_GUI(self.df_fit_results)
        self.upd_cbb_param()
        self.send_df_to_viz()

    def view_stats(self):
        """
        Show statistical fitting results of the selected spectrum.

        Displays statistical fitting results of the selected spectrum, such as
        fit reports and fitting statistics.

        """
        fnames = self.get_spectrum_fnames()
        selected_spectrums = []
        for spectrum in self.spectrums:
            if spectrum.fname in fnames:
                selected_spectrums.append(spectrum)
        if len(selected_spectrums) == 0:
            return
        ui = self.ui.tabWidget
        title = f"Fitting Report - {fnames}"
        # Show the 'report' of the first selected spectrum
        spectrum = selected_spectrums[0]
        if spectrum.result_fit:
            text = fit_report(spectrum.result_fit)
            self.common.view_text(ui, title, text)

    def select_all_spectra(self):
        """ To quickly select all spectra within the spectra listbox"""
        item_count = self.ui.spectrums_listbox.count()
        for i in range(item_count):
            item = self.ui.spectrums_listbox.item(i)
            item.setSelected(True)

    def remove_spectrum(self):
        fnames = self.get_spectrum_fnames()
        self.spectrums = Spectra(
            spectrum for spectrum in self.spectrums if
            spectrum.fname not in fnames)
        self.upd_spectra_list()
        self.ax.clear()
        self.canvas1.draw()

    def copy_fig(self):
        """To copy figure canvas to clipboard"""
        self.common.copy_fig_to_clb(canvas=self.canvas1)

    def copy_fig_graph1(self):
        """To copy figure canvas to clipboard"""
        self.common.copy_fig_to_clb(canvas=self.canvas2)

    def copy_fig_graph2(self):
        """To copy figure canvas to clipboard"""
        self.common.copy_fig_to_clb(canvas=self.canvas3)

    def save_work(self):
        """
        Save the current application state to a file.

        Opens a file dialog to select a destination and saves the current state
        of the application, including spectra, wafers, fit models, fit results,
        filters, GUI settings, and plot settings.

        Shows an alert upon successful saving or displays an error message if
        saving fails.
        """
        try:
            file_path, _ = QFileDialog.getSaveFileName(None,
                                                       "Save work",
                                                       "",
                                                       "SPECTROview Files ("
                                                       "*.svspectra)")
            if file_path:
                data_to_save = {
                    'spectrums': self.spectrums,
                    'loaded_fit_model': self.loaded_fit_model,
                    'current_fit_model': self.current_fit_model,
                    'df_fit_results': self.df_fit_results,
                    'filters': self.filter.filters,

                    'cbb_x_1': self.ui.cbb_x_3.currentIndex(),
                    'cbb_y_1': self.ui.cbb_y_3.currentIndex(),
                    'cbb_z_1': self.ui.cbb_z_3.currentIndex(),
                    'cbb_x_2': self.ui.cbb_x_7.currentIndex(),
                    'cbb_y_2': self.ui.cbb_y_7.currentIndex(),
                    'cbb_z_2': self.ui.cbb_z_7.currentIndex(),

                    "plot_style_1": self.ui.cbb_plot_style_3.currentIndex(),
                    "plot_style_2": self.ui.cbb_plot_style_7.currentIndex(),
                }
                with open(file_path, 'wb') as f:
                    dill.dump(data_to_save, f)
                show_alert("Work saved successfully.")
        except Exception as e:
            show_alert(f"Error saving work: {e}")

    def load_work(self):
        """
        Load a previously saved application state from a file.

        Opens a file dialog to select a saved file and loads its contents to
        restore the application state, including spectra, wafers, fit models,
        fit results, filters, GUI settings, and plot settings.

        Updates the GUI and application state based on the loaded data. Shows
        an alert upon successful loading or displays an error message if loading
        fails.
        """
        try:
            file_path, _ = QFileDialog.getOpenFileName(None,
                                                       "Load work",
                                                       "",
                                                       "SPECTROview Files ("
                                                       "*.svspectra)")
            if file_path:
                with open(file_path, 'rb') as f:
                    load = dill.load(f)
                    try:
                        self.spectrums = load.get('spectrums')
                        self.current_fit_model = load.get('current_fit_model')
                        self.loaded_fit_model = load.get('loaded_fit_model')

                        self.df_fit_results = load.get('df_fit_results')
                        self.filter.filters = load.get('filters')
                        self.filter.upd_filter_listbox()

                        self.upd_cbb_param()
                        self.send_df_to_viz()
                        self.upd_spectra_list()

                        self.ui.cbb_x_3.setCurrentIndex(load.get('cbb_x_1', -1))
                        self.ui.cbb_y_3.setCurrentIndex(load.get('cbb_y_1', -1))
                        self.ui.cbb_z_3.setCurrentIndex(load.get('cbb_z_1', -1))
                        self.ui.cbb_x_7.setCurrentIndex(load.get('cbb_x_2', -1))
                        self.ui.cbb_y_7.setCurrentIndex(load.get('cbb_y_2', -1))
                        self.ui.cbb_z_7.setCurrentIndex(load.get('cbb_z_2', -1))

                        self.ui.cbb_plot_style_3.setCurrentIndex(
                            load.get('plot_style_1', -1))
                        self.ui.cbb_plot_style_7.setCurrentIndex(
                            load.get('plot_style_2', -1))
                        #
                        # self.plot2()
                        # self.plot3()
                        self.display_df_in_GUI(self.df_fit_results)
                    except Exception as e:
                        show_alert(f"Error loading work: {e}")
        except Exception as e:
            show_alert(f"Error loading work: {e}")

    def load_fit_settings(self):
        """Reload last used fitting settings from QSettings"""
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
        self.ui.cb_fit_negative_2.setChecked(fit_params['fit_negative'])
        self.ui.max_iteration_2.setValue(fit_params['max_ite'])
        self.ui.cbb_fit_methods_2.setCurrentText(fit_params['method'])
        self.ui.cbb_cpu_number_2.setCurrentText(fit_params['ncpus'])
        self.ui.xtol_2.setText(str(fit_params['xtol']))

    def save_fit_settings(self):
        """Save all fitting settings to QSettings when interface element's
        states changed"""
        fit_params = {
            'fit_negative': self.ui.cb_fit_negative_2.isChecked(),
            'max_ite': self.ui.max_iteration_2.value(),
            'method': self.ui.cbb_fit_methods_2.currentText(),
            'ncpus': self.ui.cbb_cpu_number_2.currentText(),
            'xtol': float(self.ui.xtol_2.text())
        }
        print("settings are saved")
        # Save the fit_params to QSettings
        for key, value in fit_params.items():
            self.settings.setValue(key, value)

    def fitspy_launcher(self):
        """To Open FITSPY with selected spectra"""
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
            show_alert("No spectrum is loaded, FITSPY cannot open")
            return

    def fit_fnc_handler(self):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.fit_all()
        else:
            self.fit()

    def reinit_fnc_handler(self):
        """Switch between 2 save fit fnc with the Ctrl key"""
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.reinit_all()
        else:
            self.reinit()

    def subtract_baseline_handler(self):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.subtract_baseline_all()
        else:
            self.subtract_baseline()

    def paste_fit_model_fnc_handler(self):
        """Switch between 2 save fit fnc with the Ctrl key"""
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.paste_fit_model_all()
        else:
            self.paste_fit_model()

    def set_x_range_handler(self):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.set_x_range_all()
        else:
            self.set_x_range()

    def apply_model_fnc_handler(self):
        """Switch between 2 save fit fnc with the Ctrl key"""
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.apply_loaded_fit_model_all()
        else:
            self.apply_loaded_fit_model()

    def clear_peaks_handler(self):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.clear_peaks_all()
        else:
            self.clear_peaks()


class CustomListWidget(QListWidget):
    """
    Customized QListWidget with drag-and-drop functionality for rearranging items.

    This class inherits from QListWidget and provides extended functionality
    for reordering items via drag-and-drop operations.

    Signals:
        items_reordered:
            Emitted when items in the list widget are reordered by the user
            using drag-and-drop.
    """
    items_reordered = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragDropMode(QListWidget.InternalMove)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def dropEvent(self, event):
        """
        Overrides the dropEvent method to emit the items_reordered signal
            after an item is dropped into a new position.
        """
        super().dropEvent(event)
        self.items_reordered.emit()
