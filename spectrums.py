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

<<<<<<< HEAD
                    # spectrum_fs.load_profile(file_path)
                    # create FITSPY object
                    spectrum_fs = Spectrum()

                    spectrum_fs.fname = fname
                    spectrum_fs.x = np.asarray(x_values)
                    spectrum_fs.x0 = np.asarray(x_values)
                    spectrum_fs.y = np.asarray(y_values)
                    spectrum_fs.y0 = np.asarray(y_values)
=======
                    dfr = pd.read_csv(file_path, header=None, skiprows=1,
                                      delimiter="\t")
                    arr = dfr.to_numpy()

                    x_values = arr[:, 0]  # Extract first column as x_values
                    y_values = arr[:, 1]  # Extract second column as y_values

                    # create FITSPY object
                    spectrum_fs = Spectrum()
                    spectrum_fs.fname = fname
                    spectrum_fs.x = x_values
                    print(x_values)
                    print(spectrum_fs.x)
                    spectrum_fs.x0 = x_values
                    spectrum_fs.y = y_values
                    spectrum_fs.y0 = y_values
>>>>>>> github/main
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
        fnames = self.get_selected_spectra()
        self.spectra_fs.apply_model(self.model_fs, fnames)

        # # Start fitting process in a separate thread
        # self.fit_thread = FitThread(self.spectra_fs, self.model_fs, fnames)
        # self.fit_thread.fit_progress_changed.connect(self.update_pbar)
        # self.fit_thread.fit_progress.connect(
        #     lambda num, elapsed_time: self.fit_progress(num, elapsed_time,
        #                                                 fnames))
        # self.fit_thread.fit_completed.connect(self.fit_completed)
        # self.fit_thread.start()

    def fit_fnc_handler(self):
        """Switch between 2 save fit fnc with the Ctrl key"""
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.fit_all()
        else:
            self.fit()

    def fit_all(self):
        """ Apply loaded fit model to all selected spectra"""
        fnames = self.spectra_fs.fnames
        self.fit(fnames=fnames)

    def fit_progress(self, num, elapsed_time, fnames):
        """Called when fitting process is completed"""
        self.ui.progress_3.setText(
            f"{num}/{len(fnames)} fitted ({elapsed_time:.2f}s)")

    def fit_completed(self):
        """Called when fitting process is completed"""
        self.plot_sel_spectre()
        self.upd_spectra_list()

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
