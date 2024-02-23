# maps.py module
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import dill

from utils import view_df, show_alert, quadrant, view_text, copy_fig_to_clb, \
    translate_param, clear_layout, reinit_spectrum, plot_graph
from utils import FitThread
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

        self.ax = None
        self.canvas1 = None
        self.canvas2 = None
        self.canvas3 = None
        self.model_fs = None
        self.spectra_fs = Spectra()  # FITSPY

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

        # Set a delay for the function plot_sel_spectra
        self.delay_timer = QTimer()
        self.delay_timer.setSingleShot(True)
        self.delay_timer.timeout.connect(self.plot_sel_spectra)

        # Connect the progress signal to update_progress_bar slot
        self.fit_progress_changed.connect(self.update_pbar)

        self.plot_styles = ["point plot", "scatter plot", "box plot",
                            "bar plot"]
        self.create_plot_widget()

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
        QTimer.singleShot(50, self.plot_delay)

    def get_selected_spectra(self):
        items = self.ui.spectrums_listbox.selectedItems()
        fnames = []
        for item in items:
            text = item.text()
            fnames.append(text)
        return fnames

    def create_plot_widget(self):
        """Create canvas and toolbar for plotting in the GUI"""
        plt.style.use(PLOT_POLICY)
        fig1 = plt.figure()
        self.ax = fig1.add_subplot(111)
        self.ax.set_xlabel("Raman shift (cm$^{-1}$)")
        self.ax.set_ylabel("Intensity (a.u)")
        if self.ui.cb_legend.isChecked():
            self.ax.legend(loc='upper right')
        self.canvas1 = FigureCanvas(fig1)
        self.toolbar = NavigationToolbar2QT(self.canvas1, self.ui)
        # Connect Home button to rescale function
        home_action = next(
            a for a in self.toolbar.actions() if a.text() == 'Home')
        home_action.triggered.connect(self.rescale)
        self.ui.spectre_view_frame_4.addWidget(self.canvas1)
        self.ui.toolbar_frame_3.addWidget(self.toolbar)
        self.canvas1.figure.tight_layout()
        self.canvas1.draw()

    def rescale(self):
        """Rescale the figure."""
        self.ax.autoscale()
        self.canvas1.draw()

    def plot_sel_spectra(self):
        """Plot all selected spectra"""
        fnames = self.get_selected_spectra()
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

            if hasattr(spectrum_fs.result_fit,
                       'residual') and self.ui.cb_residual_3.isChecked():
                residual = spectrum_fs.result_fit.residual
                self.ax.plot(x_values, residual, 'ko-', ms=3, label='residual')
            if self.ui.cb_colors_3.isChecked() is False:
                self.ax.set_prop_cycle(None)

        self.ax.set_xlabel("Raman shift (cm$^{-1}$)")
        self.ax.set_ylabel("Intensity (a.u)")
        if self.ui.cb_legend.isChecked():
            self.ax.legend(loc='upper right')
        self.ax.get_figure().tight_layout()
        self.canvas1.draw()

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
        else:
            self.fit()

    def collect_results(self):
        """Function to collect best-fit results and append in a dataframe"""
        # Add all dict into a list, then convert to a dataframe.
        fit_results_list = []
        self.df_fit_results = None

        for spectrum_fs in self.spectra_fs:
            if hasattr(spectrum_fs.result_fit, 'best_values'):
                success = spectrum_fs.result_fit.success
                best_values = spectrum_fs.result_fit.best_values
                best_values["Sample"] = spectrum_fs.fname
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
                name = 'z' + name[4:]  # model peak parameters to be at the end
            names.append(name)
        self.df_fit_results = self.df_fit_results.iloc[:,
                              list(np.argsort(names, kind='stable'))]
        columns = [translate_param(self.model_fs, column) for column in
                   self.df_fit_results.columns]
        self.df_fit_results.columns = columns

        self.upd_cbb_param()
        self.send_df_to_viz()

    def split_fname(self):
        """Split fname and populate the combobox"""
        dfr = self.df_fit_results
        fname_parts = dfr.loc[0, 'Sample'].split('_')
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
        parts = dfr['Sample'].str.split('_')
        dfr[col_name] = [part[selected_part_index] if len(
            part) > selected_part_index else None for part in parts]
        show_alert(f"Column added successfully:'{col_name}'")
        self.df_fit_results = dfr
        self.send_df_to_viz()
        self.upd_cbb_param()

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
        dfs = self.callbacks_df.original_dfs
        dfs["fit_results"] = self.df_fit_results
        self.callbacks_df.action_open_df(file_paths=None, original_dfs=dfs)

    def plot_graph(self, view=None):
        """Plot graph """
        clear_layout(self.ui.frame_graph_3.layout())

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

        self.canvas2 = plot_graph(dfr, x, y, z, style, xmin, xmax, ymin, ymax,
                                  title,
                                  x_text, y_text, xlabel_rot)

        self.ui.frame_graph_3.addWidget(self.canvas2)

    def plot_graph2(self, view=None):
        clear_layout(self.ui.frame_graph_7.layout())

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

        self.canvas3 = plot_graph(dfr, x, y, z, style, xmin, xmax, ymin, ymax,
                                  title,
                                  x_text, y_text, xlabel_rot)

        self.ui.frame_graph_7.addWidget(self.canvas3)

    def plot_delay(self):
        """Trigger the fnc to plot spectre"""
        self.delay_timer.start(100)

    def fit_progress(self, num, elapsed_time, fnames):
        """Called when fitting process is completed"""
        self.ui.progress_3.setText(
            f"{num}/{len(fnames)} fitted ({elapsed_time:.2f}s)")

    def fit_completed(self):
        """Called when fitting process is completed"""
        self.plot_delay()
        self.upd_spectrums_list()
        QTimer.singleShot(200, self.rescale)

    def update_pbar(self, progress):
        self.ui.progressBar_3.setValue(progress)

    def cosmis_ray_detection(self):
        self.spectra_fs.outliers_limit_calculation()

    def reinit(self, fnames=None):
        """Reinitialize the selected spectrum(s)"""
        if fnames is None:
            fnames = self.get_selected_spectra()
        reinit_spectrum(fnames, self.spectra_fs)
        self.plot_delay()
        self.upd_spectrums_list()
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
                show_alert("Error loading DataFrame:", e)

        self.upd_cbb_param()
        self.send_df_to_viz()

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
        self.upd_spectrums_list()
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
                                                       "Save fitted spectrum "
                                                       "data",
                                                       "",
                                                       "SPECTROview Files ("
                                                       "*.svs)")
            if file_path:
                data_to_save = {
                    'spectra_fs': self.spectra_fs,
                    'model_fs': self.model_fs,
                    'model_name': self.ui.lb_loaded_model_3.text(),
                    'df_fit_results': self.df_fit_results,

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
                                                       "Save fitted spectrum "
                                                       "data",
                                                       "",
                                                       "SPECTROview Files ("
                                                       "*.svs)")
            if file_path:
                with open(file_path, 'rb') as f:
                    load = dill.load(f)
                    self.spectra_fs = load.get('spectra_fs')
                    self.model_fs = load.get('model_fs')
                    model_name = load.get('model_name', '')
                    self.ui.lb_loaded_model_3.setText(model_name)
                    self.ui.lb_loaded_model_3.setStyleSheet("color: yellow;")

                    self.df_fit_results = load.get('df_fit_results')
                    self.upd_cbb_param()
                    self.send_df_to_viz()
                    self.upd_spectrums_list()

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

                    self.plot_graph()
                    self.plot_graph2()
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
