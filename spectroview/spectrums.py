import os
import time
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
import json

from common import view_df, show_alert, spectrum_to_dict, dict_to_spectrum, baseline_to_dict, dict_to_baseline, populate_spectrum_listbox
from common import FitThread, FitModelManager, ShowParameters, DataframeTable, CustomListWidget
from common import PLOT_POLICY, FIT_METHODS

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
    Class manages the GUI interactions and operations related to spectra
    fittings,
    and visualization of fitted data within "SPACTRA" TAB of the application.
    """

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
        self.ui.listbox_layout.addWidget(self.ui.spectrums_listbox)
        self.ui.spectrums_listbox.items_reordered.connect(
            self.update_spectrums_order)

        # Connect and plot_spectra of selected SPECTRUM LIST
        self.ui.spectrums_listbox.itemSelectionChanged.connect(
            self.refresh_gui)
        # Connect the checkbox signal to the method
        self.ui.checkBox.stateChanged.connect(self.check_uncheck_all)

        # Connect the stateChanged signal of the legend CHECKBOX
        self.ui.cb_legend_3.stateChanged.connect(self.refresh_gui)
        self.ui.cb_raw_3.stateChanged.connect(self.refresh_gui)
        self.ui.cb_bestfit_3.stateChanged.connect(self.refresh_gui)
        self.ui.cb_colors_3.stateChanged.connect(self.refresh_gui)
        self.ui.cb_residual_3.stateChanged.connect(self.refresh_gui)
        self.ui.cb_filled_3.stateChanged.connect(self.refresh_gui)
        self.ui.cb_peaks_3.stateChanged.connect(self.refresh_gui)
        self.ui.cb_attached_2.stateChanged.connect(self.refresh_gui)
        self.ui.cb_normalize_3.stateChanged.connect(self.refresh_gui)
        self.ui.cb_limits_2.stateChanged.connect(self.refresh_gui)
        self.ui.cb_expr_2.stateChanged.connect(self.refresh_gui)

        # Set a delay for the function plot1
        self.delay_timer = QTimer()
        self.delay_timer.setSingleShot(True)
        self.delay_timer.timeout.connect(self.plot)
        self.ui.cbb_xaxis_unit.currentIndexChanged.connect(self.refresh_gui)

        self.create_spectra_plot_widget()
        self.zoom_pan_active = False

        self.ui.cbb_fit_methods_2.addItems(FIT_METHODS)
        self.ui.sb_dpi_spectra_2.valueChanged.connect(
            self.create_spectra_plot_widget)
        
        self.ui.btn_send_to_viz.clicked.connect(self.send_df_to_viz)

        # BASELINE
        self.ui.cb_attached_2.clicked.connect(self.refresh_gui)
        self.ui.noise_2.valueChanged.connect(self.refresh_gui)
        self.ui.rbtn_linear_2.clicked.connect(self.refresh_gui)
        self.ui.rbtn_polynomial_2.clicked.connect(self.refresh_gui)
        self.ui.degre_2.valueChanged.connect(self.refresh_gui)
        
        self.ui.cb_attached_2.clicked.connect(self.get_baseline_settings)
        self.ui.noise_2.valueChanged.connect(self.get_baseline_settings)
        self.ui.rbtn_linear_2.clicked.connect(self.get_baseline_settings)
        self.ui.rbtn_polynomial_2.clicked.connect(self.get_baseline_settings)
        self.ui.degre_2.valueChanged.connect(self.get_baseline_settings)
        
        self.ui.btn_copy_bl_2.clicked.connect(self.copy_baseline)
        self.ui.btn_paste_bl_2.clicked.connect(self.paste_baseline_handler)
        self.ui.sub_baseline_2.clicked.connect(self.subtract_baseline_handler)

        # Load default folder path from QSettings during application startup
        self.fit_model_manager = FitModelManager(self.settings)
        self.fit_model_manager.default_model_folder = self.settings.value(
            "default_model_folder", "")
        self.ui.l_defaut_folder_model_3.setText(
            self.fit_model_manager.default_model_folder)
        QTimer.singleShot(0, self.populate_available_models)
        self.ui.btn_refresh_model_folder_3.clicked.connect(
            self.populate_available_models)

    def open_spectra(self, spectra=None, file_paths=None):
        """Open and load raw spectral data"""

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
                    spectrum.baseline.mode = "Linear"
                    self.spectrums.append(spectrum)

        QTimer.singleShot(100, self.upd_spectra_list)
        self.ui.tabWidget.setCurrentWidget(self.ui.tab_spectra)

    def upd_spectra_list(self):
        """Show spectrums in a listbox"""

        # Store the checked state of each item
        checked_states = {}
        for index in range(self.ui.spectrums_listbox.count()):
            item = self.ui.spectrums_listbox.item(index)
            checked_states[item.text()] = item.checkState()

        current_row = self.ui.spectrums_listbox.currentRow()
        self.ui.spectrums_listbox.clear()
        
        for spectrum in self.spectrums:
            fname = spectrum.fname
            item = populate_spectrum_listbox(spectrum, fname,checked_states)
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
        QTimer.singleShot(50, self.refresh_gui)
        
    def get_checked_spectra(self):
        """
        Get a list of selected spectra based on listbox's checkbox states.
        """
        checked_spectra = Spectra()
        for index in range(self.ui.spectrums_listbox.count()):
            item = self.ui.spectrums_listbox.item(index)
            if item.checkState() == Qt.Checked:
                fname = item.text()
                spectrum = next((s for s in self.spectrums if s.fname == fname),
                                None)
                if spectrum:
                    checked_spectra.append(spectrum)
        return checked_spectra

    def check_uncheck_all(self, state):
        """
        Check or uncheck all items in the listbox based on the state of the
        main checkbox.
        """
        check_state = Qt.Unchecked if state == 0 else Qt.Checked
        for index in range(self.ui.spectrums_listbox.count()):
            item = self.ui.spectrums_listbox.item(index)
            item.setCheckState(check_state)

    def on_click(self, event):
        """
        Handle click events on spectra plot canvas for adding peak models or
        baseline points.
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

    def update_spectrums_order(self):
        """
        Update the order of spectra when user rearranges them via listbox
        drag-and-drop.
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


    def set_default_model_folder(self, folder_path=None):
        """
        Set the default folder where contain fit_models.
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

    def upd_model_cbb_list(self):
        """Update and populate the model list in the UI combobox"""
        current_path = self.fit_model_manager.default_model_folder
        self.set_default_model_folder(current_path)

    def populate_available_models(self):
        """Populate the available fit models in the UI combobox"""
        self.fit_model_manager.scan_models()
        self.available_models = self.fit_model_manager.get_available_models()
        self.ui.cbb_fit_model_list_3.clear()
        self.ui.cbb_fit_model_list_3.addItems(self.available_models)

    def load_fit_model(self, fname_json=None):
        """
        Load a fit model from a JSON file or from the UI selection.
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
        """Retrieve the currently loaded fit model from the UI"""
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
                show_alert('Fit model file not found in the default folder')

    def save_fit_model(self):
        """
        Save the fit model of the currently selected spectrum to a JSON file.
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

    def get_spectrum_fnames(self):
        """Get the filenames of currently selected spectra in the UI"""
        if self.ui.spectrums_listbox.count() == 0:
            return []
        items = self.ui.spectrums_listbox.selectedItems()
        fnames = []
        for item in items:
            text = item.text()
            fnames.append(text)
        return fnames

    def get_spectrum_object(self):
        """Get the Spectrum object of currently selected spectra"""
        
        fnames = self.get_spectrum_fnames()
        sel_spectra = []
        for spectrum in self.spectrums:
            if spectrum.fname in fnames:
                sel_spectra.append(spectrum)
                
        if len(sel_spectra) == 0:
            return
        sel_spectrum = sel_spectra[0]
        if sel_spectrum is not None and sel_spectra:
            return sel_spectrum, sel_spectra
        else:
            return None, None

    def create_spectra_plot_widget(self):
        """Create the widget for displaying spectra plots"""

        plt.style.use(PLOT_POLICY)
        self.common.clear_layout(self.ui.QVBoxlayout_2.layout())
        self.common.clear_layout(self.ui.toolbar_frame_3.layout())
        self.upd_spectra_list()
        dpi = float(self.ui.sb_dpi_spectra_2.text())

        fig1 = plt.figure(dpi=dpi)
        self.ax = fig1.add_subplot(111)
        txt = self.ui.cbb_xaxis_unit.currentText()
        self.ax.set_xlabel(txt)
        self.ax.set_ylabel("Intensity (a.u)")
        self.ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
        self.canvas1 = FigureCanvas(fig1)
        self.canvas1.mpl_connect('button_press_event', self.on_click)

        # Toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas1, self.ui)
        for action in self.toolbar.actions():
            if action.text() in ['Save', 'Pan', 'Back', 'Forward', 'Subplots']:
                action.setVisible(False)
            if action.text() == 'Pan' or action.text() == 'Zoom':
                action.toggled.connect(self.toggle_zoom_pan)
        rescale = next(
            a for a in self.toolbar.actions() if a.text() == 'Home')
        rescale.triggered.connect(self.rescale)

        self.ui.QVBoxlayout_2.addWidget(self.canvas1)
        self.ui.toolbar_frame_3.addWidget(self.toolbar)
        self.canvas1.figure.tight_layout()
        self.canvas1.draw()

    def rescale(self):
        """Rescale the spectra plot to fit within the axes"""
        self.ax.autoscale()
        self.canvas1.draw()

    def toggle_zoom_pan(self, checked):
        """Toggle zoom and pan functionality for spectra plot"""
        self.zoom_pan_active = checked
        if not checked:
            self.zoom_pan_active = False

    def plot(self):
        """Plot spectra or fit results in the main plot area"""
        fnames = self.get_spectrum_fnames()
        selected_spectrums = []

        for spectrum in self.spectrums:
            fname = spectrum.fname
            if fname in fnames:
                selected_spectrums.append(spectrum)

        # Only plot 100 first spectra to advoid crash
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

            # Background
            y_bkg = np.zeros_like(x_values)
            if spectrum.bkg_model is not None:
                y_bkg = spectrum.bkg_model.eval(
                    spectrum.bkg_model.make_params(), x=x_values)

            # BEST-FIT and PEAK_MODELS
            y_peaks = np.zeros_like(x_values)
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
                    y_peaks += y_peak
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

                if hasattr(spectrum.result_fit,
                           'success') and self.ui.cb_bestfit_3.isChecked():
                    y_fit = y_bkg + y_peaks
                    self.ax.plot(x_values, y_fit, label=f"bestfit")

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

        # self.ax.set_xlabel("Raman shift (cm$^{-1}$)")
        txt = self.ui.cbb_xaxis_unit.currentText()
        self.ax.set_xlabel(txt)

        self.ax.set_ylabel("Intensity (a.u)")
        if self.ui.cb_legend_3.isChecked():
            self.ax.legend(loc='upper right')
        self.ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
        self.ax.get_figure().tight_layout()
        self.canvas1.draw()
        self.read_x_range()
        self.show_peak_table()

    def read_x_range(self):
        """Read the x range of the selected spectrum"""
        sel_spectrum, sel_spectra = self.get_spectrum_object()
        self.ui.range_min_2.setText(str(sel_spectrum.x[0]))
        self.ui.range_max_2.setText(str(sel_spectrum.x[-1]))

    def set_x_range(self, fnames=None):
        """Set a new x range for the selected spectrum"""
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
            
        QTimer.singleShot(50, self.upd_spectra_list)
        QTimer.singleShot(300, self.rescale)

    def set_x_range_all(self):
        """Set a new x range for all spectra"""

        checked_spectra = self.get_checked_spectra()
        fnames = checked_spectra.fnames
        self.set_x_range(fnames=fnames)

    def get_baseline_settings(self):
        """Get baseline settings from GUI and set to selected spectrum"""
        spectrum_object = self.get_spectrum_object()
        if spectrum_object is None:
            return
        else:
            sel_spectrum, sel_spectra = spectrum_object  
            sel_spectrum.baseline.attached = self.ui.cb_attached_2.isChecked()
            sel_spectrum.baseline.sigma = self.ui.noise_2.value()
            if self.ui.rbtn_linear_2.isChecked():
                sel_spectrum.baseline.mode = "Linear"
            else:
                sel_spectrum.baseline.mode = "Polynomial"
                sel_spectrum.baseline.order_max = self.ui.degre_2.value()

    
    def plot_baseline_dynamically(self, ax, spectrum):
        """Evaluate and plot baseline points and line dynamically"""
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
            attached = spectrum.baseline.attached
            baseline_values = spectrum.baseline.eval(x_bl, y_bl,
                                                     attached=attached)
            ax.plot(x_bl, baseline_values, 'r')

            # Plot the attached baseline points
            if spectrum.baseline.attached and y_bl is not None:
                attached_points = spectrum.baseline.attached_points(x_bl, y_bl)
                ax.plot(attached_points[0], attached_points[1], 'ko',
                        mfc='none')
            else:
                ax.plot(spectrum.baseline.points[0],
                        spectrum.baseline.points[1], 'ko', mfc='none', ms=5)
                
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
        QTimer.singleShot(300, self.rescale)

    def paste_baseline_all(self):
        """Paste baseline to the all spectrum(s)"""
        checked_spectra = self.get_checked_spectra()
        self.paste_baseline(checked_spectra)
        
    def subtract_baseline_all(self):
        """Subtract the baseline for all spectra"""
        checked_spectra = self.get_checked_spectra()
        self.subtract_baseline(checked_spectra)

    def clear_peaks(self, fnames=None):
        """Clear all existing peak models of the selected spectrum(s)"""
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
        """Clear peaks for all spectra"""
        checked_spectra = self.get_checked_spectra()
        fnames = checked_spectra.fnames
        self.clear_peaks(fnames)

    def get_fit_settings(self):
        """Retrieve all settings for the fitting action from the GUI"""
        sel_spectrum, sel_spectra = self.get_spectrum_object()
        fit_params = sel_spectrum.fit_params.copy()
        fit_params['fit_negative'] = self.ui.cb_fit_negative_2.isChecked()
        fit_params['max_ite'] = self.ui.max_iteration_2.value()
        fit_params['method'] = self.ui.cbb_fit_methods_2.currentText()
        fit_params['ncpus'] = self.ui.ncpus.value()
        fit_params['xtol'] = float(self.ui.xtol_2.text())
        sel_spectrum.fit_params = fit_params

    def fit(self, fnames=None):
        """Fit the selected spectrum(s) with current parameters"""
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
        checked_spectra = self.get_checked_spectra()
        fnames = checked_spectra.fnames
        self.fit(fnames)

    def apply_loaded_fit_model(self, fnames=None):
        """Apply the loaded fit model to selected spectra"""
        self.get_loaded_fit_model()
        self.ui.centralwidget.setEnabled(False)  # Disable GUI
        if self.loaded_fit_model is None:
            show_alert("Select from the list or load a fit model.")
            self.ui.centralwidget.setEnabled(True)
            return

        if fnames is None:
            fnames = self.get_spectrum_fnames()

        self.ntot = len(fnames)
        ncpus = int(self.ui.ncpus.text())
        fit_model = self.loaded_fit_model
        self.spectrums.pbar_index = 0

        self.thread = FitThread(self.spectrums, fit_model, fnames, ncpus)
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
        """Apply the loaded fit model to all spectra"""

        checked_spectra = self.get_checked_spectra()
        fnames = checked_spectra.fnames
        self.apply_loaded_fit_model(fnames=fnames)

    def copy_fit_model(self):
        """Copy the model dictionary of the selected spectrum"""
        # Get only 1 spectrum among several selected spectrum:
        self.get_fit_settings()
        sel_spectrum, _ = self.get_spectrum_object()
        if len(sel_spectrum.peak_models) == 0:
            self.ui.lbl_copied_fit_model_2.setText("")
            msg = ("Select spectrum is not fitted or No fit results to collect")
            show_alert(msg)
            self.current_fit_model = None
            return
        else:
            self.current_fit_model = None
            self.current_fit_model = deepcopy(sel_spectrum.save())
        self.ui.lbl_copied_fit_model_2.setText("copied")

    def paste_fit_model(self, fnames=None):
        """Apply the copied fit model to the selected spectra"""
        self.ui.centralwidget.setEnabled(False)  # Disable GUI
        if fnames is None:
            fnames = self.get_spectrum_fnames()
        self.common.reinit_spectrum(fnames, self.spectrums)
        self.ntot = len(fnames)
        fit_model = deepcopy(self.current_fit_model)
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
        """Apply the copied fit model to selected spectrum(s)"""
        checked_spectra = self.get_checked_spectra()
        fnames = checked_spectra.fnames
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

        # Only work on selected spectra (via checkboxes)
        checked_spectra = self.get_checked_spectra()

        for spectrum in checked_spectra:
            if hasattr(spectrum, 'peak_models'):
                fit_result = {'Filename': spectrum.fname}
                for model in spectrum.peak_models:
                    if hasattr(model, 'param_names') and hasattr(model,
                                                                 'param_hints'):
                        for param_name in model.param_names:
                            key = param_name.split('_')[1]
                            if key in model.param_hints and 'value' in \
                                    model.param_hints[key]:
                                fit_result[param_name] = model.param_hints[key][
                                    'value']

                if len(fit_result) > 1:
                    fit_results_list.append(fit_result)
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


    def display_df_in_GUI(self, df):
        """Display a given DataFrame in the GUI via QTableWidget"""
        df_table = DataframeTable(df, self.ui.layout_df_table2)

    def split_fname(self):
        """Split the filename and populate the combobox."""

        dfr = self.df_fit_results
        fname_parts = dfr.loc[0, 'Filename'].split('_')
        self.ui.cbb_split_fname.clear()  # Clear existing items in combobox
        for part in fname_parts:
            self.ui.cbb_split_fname.addItem(part)

    def add_column(self):
        """Add a column to the fit results dfr based on split_fname method"""

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

    def send_df_to_viz(self):
        """Send the collected spectral data to the visualization tab"""
        dfs_new = self.visu.original_dfs
        df_name = self.ui.ent_send_df_to_viz.text()
        dfs_new[df_name] = self.df_fit_results
        self.visu.open_dfs(dfs=dfs_new, file_paths=None)


    def refresh_gui(self):
        """Trigger the fnc to plot spectra"""
        self.delay_timer.start(100)

    def cosmis_ray_detection(self):
        self.spectrums.outliers_limit_calculation()

    def reinit(self, fnames=None):
        """Reinitialize the selected spectrum(s)"""
        if fnames is None:
            fnames = self.get_spectrum_fnames()
        self.common.reinit_spectrum(fnames, self.spectrums)
        self.refresh_gui()
        self.upd_spectra_list()
        QTimer.singleShot(200, self.rescale)

    def reinit_all(self):
        """Reinitialize all spectra"""
        checked_spectra = self.get_checked_spectra()
        fnames = checked_spectra.fnames
        self.reinit(fnames)

    def view_fit_results_df(self):
        """To view selected dataframe in GUI"""
        df = getattr(self, 'df_fit_results', None)  
        
        if df is not None and not df.empty: 
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
            except Exception as e:
                show_alert("Error loading DataFrame:", e)
        self.display_df_in_GUI(self.df_fit_results)


    def view_stats(self):
        """Show statistical fitting results of the selected spectrum."""
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
        """To quickly select all spectra within the spectra listbox"""
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
    
    def paste_baseline_handler(self):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.paste_baseline_all()
        else:
            self.paste_baseline()

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

    def save_work(self):
        """Save the current application state to a file"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(None,
                                                       "Save work",
                                                       "",
                                                       "SPECTROview Files ("
                                                       "*.spectra)")
            if file_path:
                data_to_save = {
                    'spectrums': spectrum_to_dict(self.spectrums)
                }
                with open(file_path, 'w') as f:
                    json.dump(data_to_save, f, indent=4)
                show_alert("Work saved successfully.")
                
        except Exception as e:
            show_alert(f"Error saving work: {e}")

    def load_work(self, file_path):
        """Load a previously saved application state from a file."""
        try:
            with open(file_path, 'r') as f:
                load = json.load(f)
                try:
                    # Load all spectra
                    self.spectrums = Spectra()
                    for spectrum_id, spectrum_data in load.get('spectrums', {}).items():
                        spectrum = Spectrum()
                        dict_to_spectrum(spectrum, spectrum_data)
                        spectrum.preprocess()
                        self.spectrums.append(spectrum)

                    QTimer.singleShot(300, self.collect_results)
                    self.upd_spectra_list()
                except Exception as e:
                    show_alert(f"Error loading work: {e}")
        except Exception as e:
            show_alert(f"Error loading saved work (Spectrums Tab): {e}")

    def clear_env(self):
        """Clear the environment and reset the application state"""
        # Clear loaded spectrums
        self.spectrums = Spectra()
        self.loaded_fit_model = None
        self.current_fit_model = None

        # Clear DataFrames 
        self.df_fit_results = None

        # Clear UI elements that display data
        self.ui.spectrums_listbox.clear()
        self.ui.rsquared_2.clear()
        self.ui.item_count_label_3.setText("0 points")

        # Clear plot areas
        self.ax.clear()
        if hasattr(self, 'canvas1'):
            self.canvas1.draw()
        

        # Refresh the UI to reflect the cleared state
        QTimer.singleShot(50, self.rescale)
        QTimer.singleShot(100, self.upd_spectra_list)
        print("'Spectrums' Tab environment has been cleared and reset.")



