# maps.py module
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import dill
import multiprocessing
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


class Maps(QObject):
    # Define a signal for progress updates
    fit_progress_changed = Signal(int)

    def __init__(self, ui, callbacks_df):
        super().__init__()
        self.ui = ui
        self.callbacks_df = callbacks_df
        QSettings.setDefaultFormat(QSettings.IniFormat)
        self.settings = QSettings("CEA-Leti", "DaProViz")

        self.wafers = {}  # list of opened wafers
        self.ax = None
        self.canvas1 = None
        self.canvas2 = None
        self.canvas2 = None
        self.model_fs = None  # FITSPY
        self.spectra_fs = Spectra()  # FITSPY

        # Update spectra_listbox when selecting wafer via WAFER LIST
        self.ui.wafers_listbox.itemSelectionChanged.connect(
            self.upd_spectra_list)

        # Connect and plot_spectre of selected SPECTRUM LIST
        self.ui.spectra_listbox.itemSelectionChanged.connect(
            self.delay_plot)

        # Connect the stateChanged signal of the legend CHECKBOX
        self.ui.cb_legend.stateChanged.connect(self.delay_plot)

        self.ui.cb_raw.stateChanged.connect(self.delay_plot)
        self.ui.cb_bestfit.stateChanged.connect(self.delay_plot)
        self.ui.cb_colors.stateChanged.connect(self.delay_plot)
        self.ui.cb_residual.stateChanged.connect(self.delay_plot)
        self.ui.cb_filled.stateChanged.connect(self.delay_plot)
        self.ui.cb_peaks.stateChanged.connect(self.delay_plot)

        # Set a delay for the function plot_sel_spectra
        self.delay_timer = QTimer()
        self.delay_timer.setSingleShot(True)
        self.delay_timer.timeout.connect(self.plot_sel_spectra)

        # Connect the progress signal to update_progress_bar slot
        self.fit_progress_changed.connect(self.update_pbar)

        self.plot_styles = ["box plot", "point plot", "bar plot"]

    def open_data(self, file_paths=None, wafers=None):
        """Open CSV files containing RAW spectra of each wafer"""

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
                    "CSV Files (*.csv);;Text Files (*.txt)", options=options)
            # Load RAW spectra data from CSV files
            if file_paths:
                last_dir = QFileInfo(file_paths[0]).absolutePath()
                self.settings.setValue("last_directory", last_dir)

                for file_path in file_paths:
                    file_path = Path(file_path)
                    fname = file_path.stem  # get fname w/o extension
                    extension = file_path.suffix.lower()  # get file extension

                    if extension == '.csv':
                        wafer_df = pd.read_csv(file_path, skiprows=1,
                                               delimiter=";")
                    elif extension == '.txt':
                        wafer_df = pd.read_csv(file_path, delimiter="\t")
                        wafer_df.columns = ['Y', 'X'] + list(
                            wafer_df.columns[2:])
                        # Reorder df as increasing wavenumber
                        sorted_columns = sorted(wafer_df.columns[2:], key=float)
                        wafer_df = wafer_df[['X', 'Y'] + sorted_columns]
                    else:
                        print(f"Unsupported file format: {extension}")
                        continue

                    wafer_name = fname
                    if wafer_name in self.wafers:
                        print("Wafer is already opened")
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

    def fit(self, fnames=None):
        """Fit selected spectrum(s)"""
        if self.model_fs is None:
            show_alert("Load a fit model before fitting.")
            return

        if fnames is None:
            wafer_name, coords = self.spectre_id()
            fnames = [f"{wafer_name}_{coord}" for coord in coords]

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

    def collect_results(self):
        """Function to collect best-fit results and append in a dataframe"""
        # Add all dict into a list, then convert to a dataframe.
        fit_results_list = []
        self.df_fit_results = None

        for spectrum_fs in self.spectra_fs:
            if hasattr(spectrum_fs.result_fit, 'best_values'):
                wafer_name, coord = self.spectre_id_fs(spectrum_fs)
                x, y = coord
                success = spectrum_fs.result_fit.success
                best_values = spectrum_fs.result_fit.best_values
                best_values["Wafer"] = wafer_name
                best_values["X"] = x
                best_values["Y"] = y
                best_values["success"] = success
                fit_results_list.append(best_values)
        self.df_fit_results = (pd.DataFrame(fit_results_list)).round(3)

        # reindex columns according to the parameters names
        self.df_fit_results = self.df_fit_results.reindex(
            sorted(self.df_fit_results.columns),
            axis=1)
        names = []
        for name in self.df_fit_results.columns:
            if name in ["Wafer", "X", 'Y', "success"]:
                name = '0' + name  # to be in the 3 first columns
            elif '_' in name:
                name = 'z' + name[4:]  # model peak parameters to be at the end
            names.append(name)
        self.df_fit_results = self.df_fit_results.iloc[:,
                              list(np.argsort(names, kind='stable'))]
        columns = [translate_param(self.model_fs, column) for column in
                   self.df_fit_results.columns]
        self.df_fit_results.columns = columns

        # Add "Quadrant" columns
        self.df_fit_results['Quadrant'] = self.df_fit_results.apply(quadrant,
                                                                    axis=1)
        self.apprend_cbb_param()
        self.apprend_cbb_wafer()
        self.send_df_to_viz()

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

        self.apprend_cbb_param()
        self.apprend_cbb_wafer()
        self.send_df_to_viz()

    def apprend_cbb_wafer(self):
        """to append all values of df_fit_results to comoboxses"""
        self.ui.cbb_wafer_1.clear()
        wafer_names = self.df_fit_results['Wafer'].unique()
        for wafer_name in wafer_names:
            self.ui.cbb_wafer_1.addItem(wafer_name)

    def apprend_cbb_param(self):
        """to append all values of df_fit_results to comoboxses"""
        if self.df_fit_results is not None:
            columns = self.df_fit_results.columns.tolist()
            self.ui.cbb_param_1.clear()
            self.ui.cbb_x.clear()
            self.ui.cbb_y.clear()
            self.ui.cbb_z.clear()
            for column in columns:
                # remove_special_chars = re.sub(r'\$[^$]+\$', '', column)
                self.ui.cbb_param_1.addItem(column)
                self.ui.cbb_x.addItem(column)
                self.ui.cbb_y.addItem(column)
                self.ui.cbb_z.addItem(column)

    def reinit(self, fnames=None):
        """Reinitialize the selected spectrum(s)"""
        if fnames is None:
            wafer_name, coords = self.spectre_id()
            fnames = [f"{wafer_name}_{coord}" for coord in coords]
        reinit_spectrum(fnames, self.spectra_fs)
        self.delay_plot()
        self.upd_spectra_list()

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
            self.ax.plot(x_values, y_values, label=f"{coord}", ms=3, lw=2)

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
                    # Convert prefix to peak_labels
                    peak_labels = spectrum_fs.peak_labels
                    prefix = str(peak_model.prefix)
                    peak_index = int(prefix[1:-1]) - 1
                    if 0 <= peak_index < len(peak_labels):
                        peak_label = peak_labels[peak_index]

                    params = peak_model.make_params()
                    y_peak = peak_model.eval(params, x=x_values)
                    if self.ui.cb_filled.isChecked():
                        self.ax.fill_between(x_values, 0, y_peak, alpha=0.5,
                                             label=f"{peak_label}")
                        if self.ui.cb_peaks.isChecked():
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

        self.canvas1 = FigureCanvas(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas1, self.ui)
        self.ui.spectre_view_frame.addWidget(self.canvas1)
        self.ui.toolbar_frame.addWidget(self.toolbar)

        self.plot_measurement_sites()

    def plot_wafer(self):
        """Plot WaferDataFrame for view 1"""
        clear_layout(self.ui.frame_wafer.layout())
        dfr = self.df_fit_results
        wafer_name = self.ui.cbb_wafer_1.currentText()
        color = self.ui.cbb_color_pallete.currentText()
        wafer_size = float(self.ui.wafer_size.text())

        if wafer_name is not None:
            selected_df = dfr.query('Wafer == @wafer_name')
        sel_param = self.ui.cbb_param_1.currentText()
        self.canvas2 = self.plot_wafer_helper(selected_df, sel_param,
                                              wafer_size,
                                              color)
        self.ui.frame_wafer.addWidget(self.canvas2)

    def plot_wafer_helper(self, selected_df, sel_param, wafer_size, color):
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
        title = sel_param if not text else text
        ax.set_title(f"{title}")

        fig.tight_layout()
        canvas = FigureCanvas(fig)
        return canvas

    def plot_graph(self):
        """Plot graph """
        clear_layout(self.ui.frame_graph.layout())

        dfr = self.df_fit_results
        x = self.ui.cbb_x.currentText()
        y = self.ui.cbb_y.currentText()
        z = self.ui.cbb_z.currentText()
        style = self.ui.cbb_plot_style.currentText()
        xmin = self.ui.xmin.text()
        ymin = self.ui.ymin.text()
        xmax = self.ui.xmax.text()
        ymax = self.ui.ymax.text()

        title = self.ui.ent_plot_title_2.text()
        x_text = self.ui.ent_xaxis_lbl.text()
        y_text = self.ui.ent_yaxis_lbl.text()

        text = self.ui.ent_x_rot.text()
        xlabel_rot = 0  # Default rotation angle
        if text:
            xlabel_rot = float(text)

        self.canvas3 = plot_graph(dfr, x, y, z, style, xmin, xmax, ymin, ymax,
                                  title,
                                  x_text, y_text, xlabel_rot)

        self.ui.frame_graph.addWidget(self.canvas3)

    def plot_measurement_sites(self):
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
        ax.scatter(all_x, all_y, marker='x', color='gray', s=10)

        # Highlight selected spectra in red
        if coords:
            selected_x, selected_y = zip(*coords)
            ax.scatter(selected_x, selected_y, marker='o', color='red', s=40)

        canvas = FigureCanvas(fig)
        layout = self.ui.wafer_plot.layout()
        if layout:
            layout.addWidget(canvas)

    def upd_wafers_list(self):
        """ To update the wafer listbox"""
        current_row = self.ui.wafers_listbox.currentRow()

        self.ui.wafers_listbox.clear()
        wafer_names = list(self.wafers.keys())
        for wafer_name in wafer_names:
            item = QListWidgetItem(wafer_name)
            self.ui.wafers_listbox.addItem(item)
            self.clear_wafer_plot()  # Clear the wafer_plot
        item_count = self.ui.wafers_listbox.count()

        # Management of selecting item of listbox
        if current_row >= item_count:
            current_row = item_count - 1
        if current_row >= 0:
            self.ui.wafers_listbox.setCurrentRow(current_row)
        else:
            if item_count > 0:
                self.ui.wafers_listbox.setCurrentRow(0)
        QTimer.singleShot(100, self.upd_spectra_list)

    def upd_spectra_list(self):
        """to update the spectra list"""
        current_row = self.ui.spectra_listbox.currentRow()

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
        self.ui.item_count_label.setText(f"{item_count} points")

        # Reselect the previously selected item
        if current_row >= 0 and current_row < item_count:
            self.ui.spectra_listbox.setCurrentRow(current_row)
        else:
            if self.ui.spectra_listbox.count() > 0:
                self.ui.spectra_listbox.setCurrentRow(0)
        QTimer.singleShot(50, self.delay_plot)

    def remove_wafer(self):
        """To remove a wafer from the listbox and wafers df"""
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

    def clear_spectre_view(self):
        """ Clear plot and toolbar within the spectre_view"""
        clear_layout(self.ui.spectre_view_frame.layout())
        clear_layout(self.ui.toolbar_frame.layout())

    def clear_wafer_plot(self):
        """ To clear wafer plot"""
        clear_layout(self.ui.wafer_plot.layout())

    def copy_fig(self):
        """To copy figure canvas to clipboard"""
        copy_fig_to_clb(canvas=self.canvas1)

    def copy_fig_wafer(self):
        """To copy figure canvas to clipboard"""
        copy_fig_to_clb(canvas=self.canvas2)

    def copy_fig_graph(self):
        """To copy figure canvas to clipboard"""
        copy_fig_to_clb(canvas=self.canvas3)

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

    def delay_plot(self):
        """Trigger the fnc to plot spectre"""
        self.delay_timer.start(100)

    def view_fit_results_df(self):
        """To view selected dataframe"""
        view_df(self.ui.tabWidget, self.df_fit_results)

    def view_wafer_data(self):
        """To view data of selected wafer """
        wafer_name, coords = self.spectre_id()
        view_df(self.ui.tabWidget, self.wafers[wafer_name])

    def fit_fnc_handler(self):
        """Switch between 2 save fit fnc with the Ctrl key"""
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.fit_all()
        else:
            self.fit()

    def send_df_to_viz(self):
        """Send the collected spectral data dataframe to visu tab"""
        dfs = self.callbacks_df.original_dfs
        dfs["2Dmaps_fit_results"] = self.df_fit_results
        self.callbacks_df.action_open_df(file_paths=None, original_dfs=dfs)

    def cosmis_ray_detection(self):
        self.spectra_fs.outliers_limit_calculation()

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
        ui = self.ui.tabWidget
        title = "Fitting Report"
        # Show the 'report' of the first selected spectrum
        spectrum_fs = selected_spectra_fs[0]
        if spectrum_fs.result_fit:
            text = fit_report(spectrum_fs.result_fit)
            view_text(ui, title, text)

    def save_work(self):
        """Save the current work/results."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(None,
                                                       "Save fitted wafer data",
                                                       "",
                                                       "SPECTROview Files ("
                                                       "*.sv2dmap)")
            if file_path:
                data_to_save = {
                    'spectra_fs': self.spectra_fs,
                    'wafers': self.wafers,
                    'model_fs': self.model_fs,
                    'model_name': self.ui.lb_loaded_model.text(),
                    'df_fit_results': self.df_fit_results,

                    'cbb_x': self.ui.cbb_x.currentIndex(),
                    'cbb_y': self.ui.cbb_y.currentIndex(),
                    'cbb_z': self.ui.cbb_z.currentIndex(),
                    "cbb_param_1": self.ui.cbb_param_1.currentIndex(),
                    "cbb_wafer_1": self.ui.cbb_wafer_1.currentIndex(),

                    'plot_title': self.ui.plot_title.text(),
                    'plot_title2': self.ui.ent_plot_title_2.text(),
                    'xmin': self.ui.xmin.text(),
                    'xmax': self.ui.xmax.text(),
                    'ymax': self.ui.ymax.text(),
                    'ymin': self.ui.ymin.text(),
                    'xaxis_lbl': self.ui.ent_xaxis_lbl.text(),
                    'yaxis_lbl': self.ui.ent_yaxis_lbl.text(),
                    'x_rot': self.ui.ent_x_rot.text(),
                    "plot_style": self.ui.cbb_plot_style.currentIndex(),
                    'vmin': self.ui.int_vmin.text(),
                    'vmax': self.ui.int_vmax.text(),

                    'wafer_size': self.ui.wafer_size.text(),
                    "color_pal": self.ui.cbb_color_pallete.currentIndex(),
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
                                                       "Save fitted wafer data",
                                                       "",
                                                       "SPECTROview Files ("
                                                       "*.sv2dmap)")
            if file_path:
                with open(file_path, 'rb') as f:
                    load = dill.load(f)
                    self.spectra_fs = load.get('spectra_fs')
                    self.wafers = load.get('wafers')
                    self.model_fs = load.get('model_fs')
                    model_name = load.get('model_name', '')
                    self.ui.lb_loaded_model.setText(model_name)
                    self.ui.lb_loaded_model.setStyleSheet("color: yellow;")

                    self.df_fit_results = load.get('df_fit_results')
                    self.apprend_cbb_param()
                    self.apprend_cbb_wafer()
                    self.send_df_to_viz()
                    self.upd_wafers_list()

                    # Set default values or None for missing keys
                    self.ui.cbb_x.setCurrentIndex(load.get('cbb_x', -1))
                    self.ui.cbb_y.setCurrentIndex(load.get('cbb_y', -1))
                    self.ui.cbb_z.setCurrentIndex(load.get('cbb_z', -1))
                    self.ui.cbb_param_1.setCurrentIndex(load.get('cbb_param_1', -1))
                    self.ui.cbb_wafer_1.setCurrentIndex(load.get('cbb_wafer_1', -1))
                    self.ui.plot_title.setText(load.get('plot_title', ''))
                    self.ui.ent_plot_title_2.setText(load.get('plot_title2', ''))
                    self.ui.xmin.setText(load.get('xmin', ''))
                    self.ui.xmax.setText(load.get('xmax', ''))
                    self.ui.ymax.setText(load.get('ymax', ''))
                    self.ui.ymin.setText(load.get('ymin', ''))
                    self.ui.ent_xaxis_lbl.setText(load.get('xaxis_lbl', ''))
                    self.ui.ent_yaxis_lbl.setText(load.get('yaxis_lbl', ''))
                    self.ui.ent_x_rot.setText(load.get('x_rot', ''))

                    self.ui.cbb_color_pallete.setCurrentIndex(load.get('color_pal', -1))
                    self.ui.wafer_size.setText(load.get('wafer_size', ''))
                    self.ui.int_vmin.setText(load.get('vmin', ''))
                    self.ui.int_vmax.setText(load.get('vmax', ''))

                    self.plot_graph()
                    self.plot_wafer()
                print("Work loaded successfully.")
        except Exception as e:
            print(f"Error loading work: {e}")

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
            show_alert("No spectrum is loaded; FITSPY cannot open")
            return

    def fit_progress(self, num, elapsed_time, fnames):
        """Called when fitting process is completed"""
        self.ui.progress.setText(
            f"{num}/{len(fnames)} fitted ({elapsed_time:.2f}s)")

    def fit_completed(self):
        """Called when fitting process is completed"""
        self.delay_plot()
        self.upd_spectra_list()

    def update_pbar(self, progress):
        self.ui.progressBar.setValue(progress)

    def fit(self, fnames=None):
        """Fit selected spectrum(s)"""
        if self.model_fs is None:
            show_alert("Load a fit model before fitting.")
            return

        if fnames is None:
            wafer_name, coords = self.spectre_id()
            fnames = [f"{wafer_name}_{coord}" for coord in coords]

        # Start fitting process in a separate thread
        self.fit_thread = FitThread(self.spectra_fs, self.model_fs, fnames)

        self.fit_thread.fit_progress_changed.connect(self.update_pbar)
        self.fit_thread.fit_progress.connect(
            lambda num, elapsed_time: self.fit_progress(num, elapsed_time,
                                                        fnames))
        self.fit_thread.fit_completed.connect(self.fit_completed)
        self.fit_thread.start()