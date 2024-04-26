# spectrum.py module
import os
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
import dill

from utils import view_df, show_alert, view_text, copy_fig_to_clb, \
    translate_param, reinit_spectrum, plot_graph, clear_layout, \
    display_df_in_table
from utils import FitThread, ShowParameters, FIT_METHODS, NCPUS, PEAK_MODELS, \
    FIT_PARAMS
from lmfit import fit_report
from fitspy.spectra import Spectra
from fitspy.spectrum import Spectrum
from fitspy.app.gui import Appli
from fitspy.utils import closest_index
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from PySide6.QtWidgets import (QFileDialog, QMessageBox, QApplication,
                               QListWidgetItem, QCheckBox)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt, QFileInfo, QTimer, QObject, Signal
from tkinter import Tk, END

DIRNAME = os.path.dirname(__file__)
PLOT_POLICY = os.path.join(DIRNAME, "resources", "plotpolicy_spectre.mplstyle")


class Spectrums(QObject):
    # Define a signal for progress updates
    fit_progress_changed = Signal(int)

    def __init__(self, settings, ui, dataframe):
        super().__init__()
        self.settings = settings
        self.ui = ui
        self.dataframe = dataframe

        self.loaded_model_fs = None
        self.current_fit_model = None
        self.spectra_fs = Spectra()
        self.df_fit_results = None
        self.filters = []
        self.filtered_df = None  # FITSPY

        # Connect and plot_spectre of selected SPECTRUM LIST
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

    def load_fit_settings(self):
        """Load last used fitting settings from QSettings"""
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

    def open_data(self, spectra=None, file_paths=None, ):
        if self.spectra_fs is None:
            self.spectra_fs = Spectra()
        if spectra:
            self.spectra_fs = spectra
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
                           self.spectra_fs):
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
                    spectrum_fs = Spectrum()
                    spectrum_fs.fname = fname
                    spectrum_fs.x = np.asarray(x_values)
                    spectrum_fs.x0 = np.asarray(x_values)
                    spectrum_fs.y = np.asarray(y_values)
                    spectrum_fs.y0 = np.asarray(y_values)
                    self.spectra_fs.append(spectrum_fs)
        QTimer.singleShot(100, self.upd_spectra_list)

    def upd_spectra_list(self):
        current_row = self.ui.spectrums_listbox.currentRow()
        self.ui.spectrums_listbox.clear()
        for spectrum_fs in self.spectra_fs:
            fname = spectrum_fs.fname
            item = QListWidgetItem(fname)
            if hasattr(spectrum_fs.result_fit,
                       'success') and spectrum_fs.result_fit.success:
                item.setBackground(QColor("green"))
            elif hasattr(spectrum_fs.result_fit,
                         'success') and not \
                    spectrum_fs.result_fit.success:
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

    def get_spectrum_fnames(self):
        """Get fname of the selected spectra"""
        items = self.ui.spectrums_listbox.selectedItems()
        fnames = []
        for item in items:
            text = item.text()
            fnames.append(text)
        return fnames

    def get_spectrum_object(self):
        """Get selected spectrum/spectra object"""
        fnames = self.get_spectrum_fnames()
        sel_spectra = []
        for spectrum in self.spectra_fs:
            if spectrum.fname in fnames:
                sel_spectra.append(spectrum)
        if len(sel_spectra) == 0:
            return
        sel_spectrum = sel_spectra[0]
        return sel_spectrum, sel_spectra

    def create_spectra_plot_widget(self):
        """Create widget of the spectra plot"""
        plt.style.use(PLOT_POLICY)
        clear_layout(self.ui.QVBoxlayout_2.layout())
        clear_layout(self.ui.toolbar_frame_3.layout())
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

    def on_click(self, event):
        """On click action to add a "peak models" or "baseline points" """
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

    def toggle_zoom_pan(self, checked):
        """Toggle zoom and pan functionality"""
        self.zoom_pan_active = checked
        if not checked:
            self.zoom_pan_active = False

    def create_plot_widget(self):
        """Create canvas and toolbar for plotting in the GUI"""
        # Plot2: graph1
        fig2 = plt.figure(dpi=80)
        self.ax2 = fig2.add_subplot(111)
        self.canvas2 = FigureCanvas(fig2)
        self.ui.frame_graph_3.addWidget(self.canvas2)
        self.canvas2.draw()

        # Plot3: graph2
        fig3 = plt.figure(dpi=80)
        self.ax3 = fig3.add_subplot(111)
        self.canvas3 = FigureCanvas(fig3)
        self.ui.frame_graph_7.addWidget(self.canvas3)
        self.canvas3.draw()

    def rescale(self):
        """Rescale the figure."""
        self.ax.autoscale()
        self.canvas1.draw()

    def plot1(self):
        """Plot all selected spectra"""
        fnames = self.get_spectrum_fnames()
        selected_spectra_fs = []

        for spectrum_fs in self.spectra_fs:
            fname = spectrum_fs.fname
            if fname in fnames:
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
            fname = spectrum_fs.fname
            x_values = spectrum_fs.x
            y_values = spectrum_fs.y

            # NORMALIZE
            if self.ui.cb_normalize_3.isChecked():
                max_intensity = 0.0
                max_intensity = max(max_intensity, max(spectrum_fs.y))
                y_values = y_values / max_intensity
            self.ax.plot(x_values, y_values, label=f"{fname}", ms=3, lw=2)

            # BASELINE
            self.plot_baseline_dynamically(ax=self.ax, spectrum=spectrum_fs)

            # RAW
            if self.ui.cb_raw_3.isChecked():
                x0_values = spectrum_fs.x0
                y0_values = spectrum_fs.y0
                self.ax.plot(x0_values, y0_values, 'ko-', label='raw', ms=3,
                             lw=1)
            # BEST-FIT and PEAK_MODELS
            if hasattr(spectrum_fs.result_fit, 'components') and hasattr(
                    spectrum_fs.result_fit, 'components') and \
                    self.ui.cb_bestfit_3.isChecked():
                bestfit = spectrum_fs.result_fit.best_fit
                self.ax.plot(x_values, bestfit, label=f"bestfit")
            if self.ui.cb_bestfit_3.isChecked():
                peak_labels = spectrum_fs.peak_labels
                for i, peak_model in enumerate(spectrum_fs.peak_models):
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
            if hasattr(spectrum_fs.result_fit,
                       'residual') and self.ui.cb_residual_3.isChecked():
                residual = spectrum_fs.result_fit.residual
                self.ax.plot(x_values, residual, 'ko-', ms=3, label='residual')
            if self.ui.cb_colors_3.isChecked() is False:
                self.ax.set_prop_cycle(None)

            # R-SQUARED
            if hasattr(spectrum_fs.result_fit, 'rsquared'):
                rsquared = round(spectrum_fs.result_fit.rsquared, 4)
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

    def plot_baseline_dynamically(self, ax, spectrum):
        """ To evaluate and plot baseline points and line dynamically"""
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
        """ Subtract baseline for the selected spectrum(s) """
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
        """ Subtract baseline for all spectrum(s) """
        self.subtract_baseline(self.spectra_fs)

    def subtract_baseline_handler(self):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.subtract_baseline_all()
        else:
            self.subtract_baseline()

    def read_x_range(self):
        """Read x range of selected spectrum"""
        sel_spectrum, sel_spectra = self.get_spectrum_object()
        self.ui.range_min_2.setText(str(sel_spectrum.x[0]))
        self.ui.range_max_2.setText(str(sel_spectrum.x[-1]))

    def set_x_range(self, fnames=None):
        """ Set new x range for selected spectrum"""
        new_x_min = float(self.ui.range_min_2.text())
        new_x_max = float(self.ui.range_max_2.text())
        if fnames is None:
            fnames = self.get_spectrum_fnames()
        reinit_spectrum(fnames, self.spectra_fs)
        for fname in fnames:
            spectrum, _ = self.spectra_fs.get_objects(fname)
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
        """ Set new x range for all spectrum"""
        fnames = self.spectra_fs.fnames
        self.set_x_range(fnames=fnames)

    def set_x_range_handler(self):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.set_x_range_all()
        else:
            self.set_x_range()

    def show_peak_table(self):
        """ To show all fitted parameters in GUI"""
        sel_spectrum, sel_spectra = self.get_spectrum_object()
        main_layout = self.ui.peak_table1_2
        cb_limits = self.ui.cb_limits_2
        cb_expr = self.ui.cb_expr_2
        update = self.upd_spectra_list
        show_params = ShowParameters(main_layout, sel_spectrum, cb_limits,
                                     cb_expr, update)
        show_params.show_peak_table(main_layout, sel_spectrum, cb_limits,
                                    cb_expr)

    def clear_peaks(self, fnames=None):
        """Clear all existing peak models of the selected spectrum(s)"""
        if fnames is None:
            fnames = self.get_spectrum_fnames()
        for fname in fnames:
            spectrum, _ = self.spectra_fs.get_objects(fname)
            if len(spectrum.peak_models) != 0:
                spectrum.remove_models()
            else:
                continue
        QTimer.singleShot(100, self.upd_spectra_list)

    def clear_peaks_all(self):
        """Clear peaks of all spectra"""
        fnames = self.spectra_fs.fnames
        self.clear_peaks(fnames)

    def clear_peaks_handler(self):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.clear_peaks_all()
        else:
            self.clear_peaks()

    def save_fit_settings(self):
        """Save all fitting settings to QSettings"""
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

    def get_baseline_settings(self):
        """ Pass baseline settings from GUI to spectrum objects for baseline
        subtraction"""
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

    def open_fit_model(self, fname_json=None):
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
        self.loaded_model_fs = self.spectra_fs.load_model(fname_json, ind=0)
        display_name = QFileInfo(fname_json).baseName()
        self.ui.lb_loaded_model_3.setText(f"'{display_name}' is loaded !")

    def apply_loaded_fit_model(self, fnames=None):
        """Fit selected spectrum(s)"""
        self.ui.btn_fit_3.setEnabled(False)
        if self.loaded_model_fs is None:
            show_alert("Load a fit model before fitting.")
            self.ui.btn_fit_3.setEnabled(False)
            return
        if fnames is None:
            fnames = self.get_spectrum_fnames()

        # Start fitting process in a separate thread
        self.fit_thread = FitThread(self.spectra_fs, self.loaded_model_fs,
                                    fnames)
        # To update progress bar
        self.fit_thread.fit_progress_changed.connect(self.update_pbar)
        # To display progress in GUI
        self.fit_thread.fit_progress.connect(
            lambda num, elapsed_time: self.fit_progress(num, elapsed_time,
                                                        fnames))
        # To update spectra list + plot fitted specturm once fitting finished
        self.fit_thread.fit_completed.connect(self.fit_completed)

        self.fit_thread.finished.connect(
            lambda: self.ui.btn_fit_3.setEnabled(True))
        self.fit_thread.start()

    def apply_loaded_fit_model_all(self):
        """ Apply loaded fit model to all selected spectra"""
        fnames = self.spectra_fs.fnames
        self.apply_loaded_fit_model(fnames=fnames)

    def apply_loaded_fit_model_fnc_handler(self):
        """Switch between 2 save fit fnc with the Ctrl key"""
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.apply_loaded_fit_model_all()
        else:
            self.apply_loaded_fit_model()

    def get_fit_settings(self):
        """Getall settings for the fitting action"""
        sel_spectrum, sel_spectra = self.get_spectrum_object()
        fit_params = sel_spectrum.fit_params.copy()
        fit_params['fit_negative'] = self.ui.cb_fit_negative_2.isChecked()
        fit_params['max_ite'] = self.ui.max_iteration_2.value()
        fit_params['method'] = self.ui.cbb_fit_methods_2.currentText()
        fit_params['ncpus'] = self.ui.cbb_cpu_number_2.currentText()
        fit_params['xtol'] = float(self.ui.xtol_2.text())
        sel_spectrum.fit_params = fit_params

    def fit(self, fnames=None):
        """Fit selected spectrum(s) with current parameters"""
        self.get_fit_settings()
        if fnames is None:
            fnames = self.get_spectrum_fnames()
        for fname in fnames:
            spectrum, _ = self.spectra_fs.get_objects(fname)
            if len(spectrum.peak_models) != 0:
                spectrum.fit()
            else:
                continue
        QTimer.singleShot(100, self.upd_spectra_list)

    def fit_all(self):
        """Apply all fit parameters to all spectrum(s)"""
        fnames = self.spectra_fs.fnames
        self.fit(fnames)

    def apply_fit_model_handler(self):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.fit_all()
        else:
            self.fit()

    def copy_fit_model(self):
        """ To copy the model dict of the selected spectrum. If several
        spectrums are selected → copy the model dict of first spectrum in
        list"""
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
        fname = sel_spectrum.fname
        self.ui.lbl_copied_fit_model_2.setText(
            f"The fit model of '{fname}' spectrum is copied to the clipboard.")

    def paste_fit_model(self, fnames=None):
        """ To apply the copied fit model to selected spectrums"""
        # Get fnames of all selected spectra
        self.ui.btn_paste_fit_model_2.setEnabled(False)

        if fnames is None:
            fnames = self.get_spectrum_fnames()

        reinit_spectrum(fnames, self.spectra_fs)
        fit_model = deepcopy(self.current_fit_model)
        if self.current_fit_model is not None:
            # Starting fit process in a seperate thread
            self.paste_model_thread = FitThread(self.spectra_fs, fit_model,
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
        """ To paste the copied fit model (in clipboard) and apply to
        selected spectrum(s"""
        fnames = self.spectra_fs.fnames
        self.paste_fit_model(fnames)

    def paste_fit_model_fnc_handler(self):
        """Switch between 2 save fit fnc with the Ctrl key"""
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.paste_fit_model_all()
        else:
            self.paste_fit_model()

    def save_fit_model(self):
        """To save the fit model of the current selected spectrum"""
        sel_spectrum, sel_spectra = self.get_spectrum_object()
        last_dir = self.settings.value("last_directory", "/")
        save_path, _ = QFileDialog.getSaveFileName(
            self.ui.tabWidget, "Save fit model", last_dir,
            "JSON Files (*.json)")
        if save_path and sel_spectrum:
            self.spectra_fs.save(save_path, [sel_spectrum.fname])
            show_alert("Fit model is saved (JSON file)")
        else:
            show_alert("No fit model to save.")

    def collect_results(self):
        """Function to collect best-fit results and append in a dataframe"""
        # Add all dict into a list, then convert to a dataframe.
        fit_results_list = []
        self.df_fit_results = None

        for spectrum_fs in self.spectra_fs:
            if hasattr(spectrum_fs.result_fit, 'best_values'):
                success = spectrum_fs.result_fit.success
                rsquared = spectrum_fs.result_fit.rsquared
                best_values = spectrum_fs.result_fit.best_values
                best_values["Filename"] = spectrum_fs.fname
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
                if name in ["Sample", "success"]:
                    name = '0' + name  # to be in the 2 first columns
                elif '_' in name:
                    name = 'z' + name[
                                 5:]  # model peak parameters to be at the end
                names.append(name)
            self.df_fit_results = self.df_fit_results.iloc[:,
                                  list(np.argsort(names, kind='stable'))]
            columns = [translate_param(self.loaded_model_fs, column) for column
                       in
                       self.df_fit_results.columns]
            self.df_fit_results.columns = columns
            display_df_in_table(self.ui.fit_results_table_2,
                                self.df_fit_results)
        else:
            self.ui.fit_results_table_2.clear()

        self.filtered_df = self.df_fit_results
        self.upd_cbb_param()
        self.send_df_to_viz()

    def split_fname(self):
        """Split fname and populate the combobox"""
        dfr = self.df_fit_results
        fname_parts = dfr.loc[0, 'Filename'].split('_')
        self.ui.cbb_split_fname.clear()  # Clear existing items in combobox
        for part in fname_parts:
            self.ui.cbb_split_fname.addItem(part)

    def add_column(self):
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
        display_df_in_table(self.ui.fit_results_table_2, self.df_fit_results)
        self.send_df_to_viz()
        self.upd_cbb_param()

    def add_filter(self):
        filter_expression = self.ui.ent_filter_query_2.text().strip()
        if filter_expression:
            filter = {"expression": filter_expression, "state": False}
            self.filters.append(filter)
        # Add the filter expression to QListWidget as a checkbox item
        item = QListWidgetItem()
        checkbox = QCheckBox(filter_expression)
        item.setSizeHint(checkbox.sizeHint())
        self.ui.filter_listbox.addItem(item)
        self.ui.filter_listbox.setItemWidget(item, checkbox)

    def filters_ischecked(self):
        """Collect selected filters from the UI"""
        checked_filters = []
        for i in range(self.ui.filter_listbox.count()):
            item = self.ui.filter_listbox.item(i)
            checkbox = self.ui.filter_listbox.itemWidget(item)
            expression = checkbox.text()
            state = checkbox.isChecked()
            checked_filters.append({"expression": expression, "state": state})
        return checked_filters

    def apply_filters(self, filters=None):
        if filters:
            self.filters = filters
        else:
            checked_filters = self.filters_ischecked()
            self.filters = checked_filters

        # Apply all filters at once
        self.filtered_df = self.df_fit_results.copy()  # Initialize with a
        # copy of the original DataFrame

        for filter_data in self.filters:
            filter_expr = filter_data["expression"]
            is_checked = filter_data["state"]

            if is_checked:
                try:
                    # Ensure filter_expr is a string
                    filter_expr = str(filter_expr)
                    print(f"Applying filter expression: {filter_expr}")

                    # Apply the filter
                    self.filtered_df = self.filtered_df.query(filter_expr)
                except Exception as e:
                    QMessageBox.critical(self.ui, "Error",
                                         f"Filter error: {str(e)}")
                    print(f"Error applying filter: {str(e)}")
                    print(f"Filter expression causing the error: {filter_expr}")

    def upd_filter_listbox(self):
        """To update filter listbox"""
        self.ui.filter_listbox.clear()
        for filter_data in self.filters:
            filter_expression = filter_data["expression"]
            item = QListWidgetItem()
            checkbox = QCheckBox(filter_expression)
            item.setSizeHint(checkbox.sizeHint())
            self.ui.filter_listbox.addItem(item)
            self.ui.filter_listbox.setItemWidget(item, checkbox)
            checkbox.setChecked(filter_data["state"])

    def remove_filter(self):
        """To remove a filter from listbox"""
        selected_items = [item for item in
                          self.ui.filter_listbox.selectedItems()]
        for item in selected_items:
            checkbox = self.ui.filter_listbox.itemWidget(item)
            filter_expression = checkbox.text()
            for filter in self.filters[:]:
                if filter.get("expression") == filter_expression:
                    self.filters.remove(filter)
            self.ui.filter_listbox.takeItem(self.ui.filter_listbox.row(item))

    def upd_cbb_param(self):
        """to append all values of df_fit_results to comoboxses"""
        if self.df_fit_results is not None:
            columns = self.df_fit_results.columns.tolist()
            self.ui.cbb_x_3.clear()
            self.ui.cbb_y_3.clear()
            self.ui.cbb_z_3.clear()
            self.ui.cbb_x_7.clear()
            self.ui.cbb_y_7.clear()
            self.ui.cbb_z_7.clear()
            for column in columns:
                self.ui.cbb_x_3.addItem(column)
                self.ui.cbb_y_3.addItem(column)
                self.ui.cbb_z_3.addItem(column)
                self.ui.cbb_x_7.addItem(column)
                self.ui.cbb_y_7.addItem(column)
                self.ui.cbb_z_7.addItem(column)

    def send_df_to_viz(self):
        """Send the collected spectral data dataframe to visu tab"""
        dfs = self.dataframe.original_dfs
        dfs["fit_results"] = self.df_fit_results
        self.dataframe.action_open_df(file_paths=None, original_dfs=dfs)

    def plot2(self):
        """Plot graph """
        if self.filtered_df is not None:
            dfr = self.filtered_df
        else:
            dfr = self.df_fit_results
        x = self.ui.cbb_x_3.currentText()
        y = self.ui.cbb_y_3.currentText()
        z = self.ui.cbb_z_3.currentText()
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
        plot_graph(ax, dfr, x, y, z, style, xmin, xmax, ymin, ymax,
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
        plot_graph(ax, dfr, x, y, z, style, xmin, xmax, ymin, ymax, title,
                   x_text, y_text, xlabel_rot)

        self.ax3.get_figure().tight_layout()
        self.canvas3.draw()

    def plot_delay(self):
        """Trigger the fnc to plot spectre"""
        self.delay_timer.start(100)

    def fit_progress(self, num, elapsed_time, fnames):
        """Called when fitting process is completed"""
        self.ui.progress_3.setText(
            f"{num}/{len(fnames)} fitted ({elapsed_time:.2f}s)")

    def fit_completed(self):
        """ Called when fitting process is completed"""
        self.plot_delay()
        self.upd_spectra_list()
        QTimer.singleShot(200, self.rescale)

    def update_pbar(self, progress):
        """ Called when a spectrum is fitted to update progress bar"""
        self.ui.progressBar_3.setValue(progress)

    def cosmis_ray_detection(self):
        self.spectra_fs.outliers_limit_calculation()

    def reinit(self, fnames=None):
        """Reinitialize the selected spectrum(s)"""
        if fnames is None:
            fnames = self.get_spectrum_fnames()
        reinit_spectrum(fnames, self.spectra_fs)
        self.plot_delay()
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

    def view_fit_results_df(self):
        """To view selected dataframe"""
        if self.filtered_df is None:
            df = self.df_fit_results
        else:
            df = self.filtered_df
        view_df(self.ui.tabWidget, df)

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
                self.filtered_df = dfr
            except Exception as e:
                show_alert("Error loading DataFrame:", e)
        display_df_in_table(self.ui.fit_results_table_2, self.df_fit_results)
        self.upd_cbb_param()
        self.send_df_to_viz()

    def view_stats(self):
        """Show the statistique fitting results of the selected spectrum"""
        fnames = self.get_spectrum_fnames()
        selected_spectra_fs = []
        for spectrum_fs in self.spectra_fs:
            if spectrum_fs.fname in fnames:
                selected_spectra_fs.append(spectrum_fs)
        if len(selected_spectra_fs) == 0:
            return
        ui = self.ui.tabWidget
        title = f"Fitting Report - {fnames}"
        # Show the 'report' of the first selected spectrum
        spectrum_fs = selected_spectra_fs[0]
        if spectrum_fs.result_fit:
            text = fit_report(spectrum_fs.result_fit)
            view_text(ui, title, text)

    def select_all_spectra(self):
        """ To quickly select all spectra within the spectra listbox"""
        item_count = self.ui.spectrums_listbox.count()
        for i in range(item_count):
            item = self.ui.spectrums_listbox.item(i)
            item.setSelected(True)

    def remove_spectrum(self):
        fnames = self.get_spectrum_fnames()
        # Filter out selected spectra from self.spectra_fs
        self.spectra_fs = Spectra(
            spectrum_fs for spectrum_fs in self.spectra_fs if
            spectrum_fs.fname not in fnames)
        self.upd_spectra_list()
        self.ax.clear()
        self.canvas1.draw()

    def copy_fig(self):
        """To copy figure canvas to clipboard"""
        copy_fig_to_clb(canvas=self.canvas1)

    def copy_fig_graph1(self):
        """To copy figure canvas to clipboard"""
        copy_fig_to_clb(canvas=self.canvas2)

    def copy_fig_graph2(self):
        """To copy figure canvas to clipboard"""
        copy_fig_to_clb(canvas=self.canvas3)

    def save_work(self):
        """Save the current work/results."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(None,
                                                       "Save work",
                                                       "",
                                                       "SPECTROview Files ("
                                                       "*.svspectra)")
            if file_path:
                data_to_save = {
                    'spectra_fs': self.spectra_fs,
                    'model_fs': self.loaded_model_fs,
                    'model_name': self.ui.lb_loaded_model_3.text(),
                    'df_fit_results': self.df_fit_results,
                    'filters': self.filters,

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
        """Load a previously saved work."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(None,
                                                       "Load work",
                                                       "",
                                                       "SPECTROview Files ("
                                                       "*.svspectra)")
            if file_path:
                with open(file_path, 'rb') as f:
                    load = dill.load(f)
                    self.spectra_fs = load.get('spectra_fs')
                    self.loaded_model_fs = load.get('model_fs')
                    model_name = load.get('model_name', '')
                    self.ui.lb_loaded_model_3.setText(model_name)
                    self.ui.lb_loaded_model_3.setStyleSheet("color: yellow;")

                    self.df_fit_results = load.get('df_fit_results')
                    self.upd_cbb_param()
                    self.send_df_to_viz()
                    self.upd_spectra_list()

                    self.filters = load.get('filters')
                    self.upd_filter_listbox()

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

                    self.plot2()
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
            show_alert("No spectrum is loaded, FITSPY cannot open")
            return
