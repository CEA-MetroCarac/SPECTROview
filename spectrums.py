# wafer.py module
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import dill

from utils import view_df, show_alert, quadrant, view_text, copy_fig_to_clb

from lmfit import fit_report
from fitspy.spectra import Spectra
from fitspy.spectrum import Spectrum
from fitspy.app.gui import Appli
from multiprocessing import Queue
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from wafer_view import WaferView
from PySide6.QtWidgets import (QFileDialog, QMessageBox, QApplication,
                               QListWidgetItem)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt, QSettings, QFileInfo, QTimer, QObject, Signal, \
    QThread
from tkinter import Tk, END

DIRNAME = os.path.dirname(__file__)
PLOT_POLICY = os.path.join(DIRNAME, "resources", "plotpolicy_spectre.mplstyle")


class Spectrums(QObject):
    # Define a signal for progress updates
    fit_progress_changed = Signal(int)

    def __init__(self, ui, callbacks_df):
        super().__init__()
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

        # Connect and plot_spectre of selected SPECTRUM LIST
        self.ui.spectrums_listbox.itemSelectionChanged.connect(
            self.plot_sel_spectre)

        # Connect the stateChanged signal of the legend CHECKBOX
        self.ui.cb_legend_3.stateChanged.connect(self.plot_sel_spectre)

        self.ui.cb_raw_3.stateChanged.connect(self.plot_sel_spectre)
        self.ui.cb_bestfit_3.stateChanged.connect(self.plot_sel_spectre)
        self.ui.cb_colors_3.stateChanged.connect(self.plot_sel_spectre)
        self.ui.cb_residual_3.stateChanged.connect(self.plot_sel_spectre)
        self.ui.cb_filled_3.stateChanged.connect(self.plot_sel_spectre)

        # Set a delay for the function plot_sel_spectra
        self.delay_timer = QTimer()
        self.delay_timer.setSingleShot(True)
        self.delay_timer.timeout.connect(self.plot_sel_spectra)

        # Connect the progress signal to update_progress_bar slot
        self.fit_progress_changed.connect(self.update_pbar)

        self.plot_styles = ["box plot", "point plot", "bar plot"]

    def open_data(self, file_paths=None, spectra=None):
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
                for file_path in file_paths:
                    file_path = Path(file_path)
                    fname = file_path.stem

                    # Check if fname is already opened
                    if any(spectrum.fname == fname for spectrum in
                           self.spectra_fs):
                        print(
                            f"Spectrum '{fname}' is already opened. "
                            f"Skipping...")
                        continue

                    dfr = pd.read_csv(file_path, header=None, skiprows=1,
                                      delimiter="\t")
                    dfr_sorted = dfr.sort_values(by=0)
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

        QTimer.singleShot(100, self.upd_spectrums_list)

    def upd_spectrums_list(self):
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
        QTimer.singleShot(50, self.plot_sel_spectre)

    def plot_sel_spectre(self):
        """Trigger the fnc to plot spectre"""
        self.delay_timer.start(100)

    def get_selected_spectra(self):
        items = self.ui.spectrums_listbox.selectedItems()
        fnames = []
        for item in items:
            text = item.text()
            fnames.append(text)
        return fnames

    def plot_sel_spectra(self):
        """Plot all selected spectra"""
        plt.style.use(PLOT_POLICY)
        fnames = self.get_selected_spectra()
        selected_spectra_fs = []

        for spectrum_fs in self.spectra_fs:
            fname = spectrum_fs.fname
            if fname in fnames:
                selected_spectra_fs.append(spectrum_fs)
        if len(selected_spectra_fs) == 0:
            return

        self.clear_spectre_view()
        plt.close('all')
        fig = plt.figure()
        self.ax = fig.add_subplot(111)

        for spectrum_fs in selected_spectra_fs:
            fname = spectrum_fs.fname
            x_values = spectrum_fs.x
            y_values = spectrum_fs.y
            self.ax.plot(x_values, y_values, label=f"{fname}", ms=3, lw=2)

            if self.ui.cb_raw_3.isChecked():
                x0_values = spectrum_fs.x0
                y0_values = spectrum_fs.y0
                self.ax.plot(x0_values, y0_values, 'ko-', label='raw', ms=3,
                             lw=1)

            if hasattr(spectrum_fs.result_fit, 'components') and hasattr(
                    spectrum_fs.result_fit, 'components') and \
                    self.ui.cb_bestfit_3.isChecked():
                bestfit = spectrum_fs.result_fit.best_fit
                self.ax.plot(x_values, bestfit, label=f"bestfit")

                for peak_model in spectrum_fs.result_fit.components:
                    # Convert prefix to peak_labels
                    peak_labels = spectrum_fs.peak_labels
                    prefix = str(peak_model.prefix)
                    peak_index = int(prefix[1:-1]) - 1
                    if 0 <= peak_index < len(peak_labels):
                        peak_label = peak_labels[peak_index]

                    params = peak_model.make_params()
                    y_peak = peak_model.eval(params, x=x_values)
                    if self.ui.cb_filled_3.isChecked():
                        self.ax.fill_between(x_values, 0, y_peak, alpha=0.5,
                                             label=f"{peak_label}")
                    else:
                        self.ax.plot(x_values, y_peak, '--',
                                     label=f"{peak_label}")

            if hasattr(spectrum_fs.result_fit,
                       'residual') and self.ui.cb_residual_3.isChecked():
                residual = spectrum_fs.result_fit.residual
                self.ax.plot(x_values, residual, 'ko-', ms=3, label='residual')

            if self.ui.cb_colors_3.isChecked() is False:
                self.ax.set_prop_cycle(None)

        self.ax.set_xlabel("Raman shift (cm$^{-1}$)")
        self.ax.set_ylabel("Intensity (a.u)")
        if self.ui.cb_legend_3.isChecked():
            self.ax.legend(loc='upper right')
        fig.tight_layout()

        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self.ui)
        self.ui.spectre_view_frame_4.addWidget(self.canvas)
        self.ui.toolbar_frame_3.addWidget(self.toolbar)

    def clear_spectre_view(self):
        """ Clear plot and toolbar within the spectre_view"""
        self.clear_layout(self.ui.spectre_view_frame_4.layout())
        self.clear_layout(self.ui.toolbar_frame_3.layout())

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
        self.ui.lb_loaded_model_3.setText(f"'{display_name}' is loaded !")
        self.ui.lb_loaded_model_3.setStyleSheet("color: yellow;")

    def fit(self, fnames=None):
        """Fit selected spectrum(s)"""
        if self.model_fs is None:
            show_alert("Load a fit model before fitting.")
            return
        if fnames is None:
            fnames = self.get_selected_spectra()

        # Start fitting process in a separate thread
        self.fit_thread = FitThread(self.spectra_fs, self.model_fs, fnames)
        self.fit_thread.fit_progress_changed.connect(self.update_pbar)
        self.fit_thread.fit_progress.connect(
            lambda num, elapsed_time: self.fit_progress(num, elapsed_time,
                                                        fnames))
        self.fit_thread.fit_completed.connect(self.fit_completed)
        self.fit_thread.start()

    def fit_all(self):
        """ Apply loaded fit model to all selected spectra"""
        fnames = self.spectra_fs.fnames
        self.fit(fnames=fnames)

    def fit_fnc_handler(self):
        """Switch between 2 save fit fnc with the Ctrl key"""
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.fit_all()
            print('ctril hold)')
        else:
            self.fit()
            print('no hold')

    def collect_results(self):
        """Function to collect best-fit results and append in a dataframe"""
        # Add all dict into a list, then convert to a dataframe.
        fit_results_list = []
        self.df_fit_results = None

        for spectrum_fs in self.spectra_fs:
            if hasattr(spectrum_fs.result_fit, 'best_values'):
                # wafer_name, coord = self.spectre_id_fs(spectrum_fs)
                # x, y = coord
                success = spectrum_fs.result_fit.success
                best_values = spectrum_fs.result_fit.best_values
                best_values["Sample"] = spectrum_fs.fname
                # best_values["X"] = x
                # best_values["Y"] = y
                best_values["success"] = success
                fit_results_list.append(best_values)
        self.df_fit_results = (pd.DataFrame(fit_results_list)).round(3)

        # reindex columns according to the parameters names
        self.df_fit_results = self.df_fit_results.reindex(
            sorted(self.df_fit_results.columns),
            axis=1)
        names = []
        for name in self.df_fit_results.columns:
            if name in ["Sample", "success"]:
                name = '0' + name  # to be in the 2 first columns
            elif '_' in name:
                name = 'z' + name[2:]  # model peak parameters to be at the end
            names.append(name)
        self.df_fit_results = self.df_fit_results.iloc[:,
                              list(np.argsort(names, kind='stable'))]

        columns = [self.translate_param(column) for column in
                   self.df_fit_results.columns]
        self.df_fit_results.columns = columns
        print(self.df_fit_results)
        # self.apprend_cbb_param()
        # self.apprend_cbb_wafer()
        # self.send_df_to_vis()

    def translate_param(self, param):
        """Translate parameter names to plot title"""
        peak_labels = self.model_fs["peak_labels"]
        param_unit_mapping = {"ampli": "Intensity", "fwhm": "FWHM",
                              "fwhm_l": "FWHM_left", "fwhm_r": "FWHM_right",
                              "alpha": "L/G ratio",
                              "x0": "Position"}
        if "_" in param:
            prefix, param = param.split("_", 1)
            if param in param_unit_mapping:
                if param == "alpha":
                    unit = ""  # Set unit to empty string for "alpha"
                else:
                    unit = "(a.u)" if param == "ampli" else "(cm⁻¹)"
                label = param_unit_mapping[param]
                # Convert prefix to peak_label
                peak_index = int(prefix[1:]) - 1
                if 0 <= peak_index < len(peak_labels):
                    peak_label = peak_labels[peak_index]
                    return f"{label} of peak {peak_label} {unit}"
        return param

    def fit_progress(self, num, elapsed_time, fnames):
        """Called when fitting process is completed"""
        self.ui.progress_3.setText(
            f"{num}/{len(fnames)} fitted ({elapsed_time:.2f}s)")

    def fit_completed(self):
        """Called when fitting process is completed"""
        self.plot_sel_spectre()
        self.upd_spectrums_list()

    def update_pbar(self, progress):
        self.ui.progressBar_3.setValue(progress)

    def fitspy_launcher(self):
        """To Open FITSPY with selected spectra"""
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

    def cosmis_ray_detection(self):
        self.spectra_fs.outliers_limit_calculation()

    def reinit(self, fnames=None):
        """Reinitialize the selected spectrum(s)"""
        if fnames is None:
            fnames = self.get_selected_spectra()
        for fname in fnames:
            spectrum, _ = self.spectra_fs.get_objects(fname)
            spectrum.range_min = None
            spectrum.range_max = None
            spectrum.x = spectrum.x0.copy()
            spectrum.y = spectrum.y0.copy()
            spectrum.norm_mode = None
            spectrum.result_fit = lambda: None
            spectrum.remove_models()
            spectrum.baseline.points = [[], []]
            spectrum.baseline.is_subtracted = False
        self.plot_sel_spectre()
        self.upd_spectrums_list()

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
        view_df(self.ui.tabWidget, self.df_fit_results)

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
                print("Error loading DataFrame:", e)

        # self.apprend_cbb_param()
        # self.apprend_cbb_wafer()
        # self.send_df_to_vis()

    def view_stats(self):
        """Show the statistique fitting results of the selected spectrum"""
        fnames = self.get_selected_spectra()
        selected_spectra_fs = []
        for spectrum_fs in self.spectra_fs:
            if spectrum_fs.fname in fnames:
                selected_spectra_fs.append(spectrum_fs)
        if len(selected_spectra_fs) == 0:
            return
        ui = self.ui.tabWidget
        title = "Fitting Report"
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
        sel_fnames = self.get_selected_spectra()
        # Filter out selected spectra from self.spectra_fs
        self.spectra_fs = Spectra(
            spectrum_fs for spectrum_fs in self.spectra_fs if
            spectrum_fs.fname not in sel_fnames)
        # Clear the listbox and re-populate it with remaining spectra
        self.ui.spectrums_listbox.clear()
        self.upd_spectrums_list()
        # Clear the plot view
        self.clear_spectre_view()

    def save_work(self):
        """Save the current work/results."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(None,
                                                       "Save fitted spectrum "
                                                       "data",
                                                       "",
                                                       "SPECTROview Files ("
                                                       "*.svs)")
            if file_path:
                data_to_save = {
                    'spectra_fs': self.spectra_fs,

                    'model_fs': self.model_fs,
                    'df_fit_results': self.df_fit_results,
                    'cbb_x': self.ui.cbb_x.currentIndex(),
                    'cbb_y': self.ui.cbb_y.currentIndex(),
                    'cbb_z': self.ui.cbb_z.currentIndex(),
                    "cbb_param_1": self.ui.cbb_param_1.currentIndex(),
                    "cbb_wafer_1": self.ui.cbb_wafer_1.currentIndex(),

                    'plot_title': self.ui.ent_plot_title_5.text(),
                    'xmin': self.ui.xmin_3.text(),
                    'xmax': self.ui.xmax_3.text(),
                    'ymax': self.ui.ymax_3.text(),
                    'ymin': self.ui.ymin_3.text(),
                    'ent_xaxis_lbl': self.ui.ent_xaxis_lbl_3.text(),
                    'ent_yaxis_lbl': self.ui.ent_yaxis_lbl_3.text(),
                    'ent_x_rot': self.ui.ent_x_rot_3.text(),
                }
                with open(file_path, 'wb') as f:
                    dill.dump(data_to_save, f)
                print("Work saved successfully.")
        except Exception as e:
            print(f"Error saving work: {e}")

    def load_work(self):
        """Load a previously saved work."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(None,
                                                       "Save fitted spectrum "
                                                       "data",
                                                       "",
                                                       "SPECTROview Files ("
                                                       "*.svs)")
            if file_path:
                with open(file_path, 'rb') as f:
                    loaded_data = dill.load(f)
                    self.spectra_fs = loaded_data['spectra_fs']
                    self.model_fs = loaded_data['model_fs']
                    self.df_fit_results = loaded_data['df_fit_results']
                    self.upd_spectrums_list()
                    self.ui.ent_plot_title_5.setText(loaded_data['plot_title'])
                    self.ui.xmin_3.setText(loaded_data['xmin'])
                    self.ui.xmax_3.setText(loaded_data['xmax'])
                    self.ui.ymax_3.setText(loaded_data['ymax'])
                    self.ui.ymin_3.setText(loaded_data['ymin'])
                    self.ui.ent_xaxis_lbl_3.setText(
                        loaded_data['ent_xaxis_lbl'])
                    self.ui.ent_yaxis_lbl_3.setText(
                        loaded_data['ent_yaxis_lbl'])
                    self.ui.ent_x_rot_3.setText(loaded_data['ent_x_rot'])

                    # Plot the graph and wafer after loading the work
                    # self.plot_graph()

                print("Work loaded successfully.")
        except Exception as e:
            print(f"Error loading work: {e}")


class FitThread(QThread):
    fit_progress_changed = Signal(int)
    fit_progress = Signal(int, float)  # number and elapsed time
    fit_completed = Signal()

    def __init__(self, spectra_fs, model_fs, fnames):
        super().__init__()
        self.spectra_fs = spectra_fs
        self.model_fs = model_fs
        self.fnames = fnames

    def run(self):
        start_time = time.time()  # Record start time
        num = 0
        for index, fname in enumerate(self.fnames):
            progress = int((index + 1) / len(self.fnames) * 100)
            self.fit_progress_changed.emit(progress)
            self.spectra_fs.apply_model(self.model_fs, fnames=[fname])
            num += 1
            elapsed_time = time.time() - start_time
            self.fit_progress.emit(num, elapsed_time)
        self.fit_progress_changed.emit(100)
        self.fit_completed.emit()
