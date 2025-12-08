"""This modules build the GUI and functionality of "SPECTRA" Tab"""
import os
import time
import numpy as np
import pandas as pd
import json
from copy import deepcopy
from pathlib import Path

from spectroview.modules.utils import view_df, view_text, show_alert, spectrum_to_dict, dict_to_spectrum, baseline_to_dict, dict_to_baseline, populate_spectrum_listbox, save_df_to_excel, calc_area, replace_peak_labels

from spectroview.modules.utils import FitThread, CustomizedListWidget, Spectra, Spectrum
from spectroview.modules.df_table import DataframeTable
from spectroview.modules.peak_table import PeakTable
from spectroview.modules.spectra_viewer import SpectraViewer
from spectroview.modules.fit_model_manager import FitModelManager

from lmfit import fit_report
from fitspy.core.utils import closest_index

from PySide6.QtWidgets import QFileDialog, QMessageBox, QApplication
from PySide6.QtCore import Qt, QFileInfo, QTimer, QObject

class Spectrums(QObject):
    """Class manages the GUI interactions of Spectra Tab"""
    def __init__(self, settings, ui, graphs):
        super().__init__()
        self.settings = settings

        self.ui = ui
        self.graphs = graphs

        self.loaded_fit_model = None
        self.copied_fit_model = None
        self.current_peaks = None #Current peaks available of a given spectrum
        self.spectrums = Spectra()

        # Initialize SpectraViewWidget
        self.spectra_viewer = SpectraViewer(self)
        self.ui.fig_canvas_layout.addWidget(self.spectra_viewer.canvas)
        self.ui.toolbar_layout.addWidget(self.spectra_viewer.control_widget) 
        self.ui.cbb_fit_models_2.currentIndexChanged.connect(self.update_peak_model)
        
        # Initialize Peak Table
        self.peak_table = PeakTable(self, self.ui.peak_table1_2, self.ui.cbb_layout)
        
        # Initialize Dataframe Table
        self.df_fit_results = None
        self.df_table = DataframeTable(self.ui.layout_df_table2)

        # Initialize QListWidget for spectra list
        self.ui.spectrums_listbox = CustomizedListWidget()
        self.ui.listbox_layout.addWidget(self.ui.spectrums_listbox)
        self.ui.spectrums_listbox.items_reordered.connect(self.update_spectrums_order)
        self.ui.spectrums_listbox.itemSelectionChanged.connect(self.refresh_gui)
        self.ui.checkBox.stateChanged.connect(self.check_uncheck_all)

        # Set a delay for the function "plot action"
        self.delay_timer = QTimer()
        self.delay_timer.setSingleShot(True)
        self.delay_timer.timeout.connect(self.plot)
        
        self.ui.btn_send_to_viz.clicked.connect(self.send_df_to_viz)

        # Peak correction
        self.ui.btn_xrange_correction.clicked.connect(self.xrange_correction)
        self.ui.btn_undo_correction.clicked.connect(lambda: self.undo_xrange_correction())

        # BASELINE
        self.setup_baseline_controls()
        
        # FIT MODEL MANAGER
        self.fit_model_manager = FitModelManager()
        self.ui.horizontalLayout_106.addWidget(self.fit_model_manager)
        self.fit_model_manager.connect_apply(self.apply_model_fnc_handler)

    def setup_baseline_controls(self):
        """Set up baseline controls and their signal connections."""
        self.ui.cb_attached_2.stateChanged.connect(self.refresh_gui)
        self.ui.noise_2.valueChanged.connect(self.refresh_gui)
        self.ui.rbtn_linear_2.clicked.connect(self.refresh_gui)
        self.ui.rbtn_polynomial_2.clicked.connect(self.refresh_gui)
        self.ui.degre_2.valueChanged.connect(self.refresh_gui)

        self.ui.cb_attached_2.stateChanged.connect(self.get_baseline_settings)
        self.ui.noise_2.valueChanged.connect(self.get_baseline_settings)
        self.ui.rbtn_linear_2.clicked.connect(self.get_baseline_settings)
        self.ui.rbtn_polynomial_2.clicked.connect(self.get_baseline_settings)
        self.ui.degre_2.valueChanged.connect(self.get_baseline_settings)

        self.ui.btn_copy_bl_2.clicked.connect(self.copy_baseline)
        self.ui.btn_paste_bl_2.clicked.connect(self.paste_baseline_handler)
        self.ui.sub_baseline_2.clicked.connect(self.subtract_baseline_handler)
        self.get_baseline_settings()
        
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

                    dfr = pd.read_csv(file_path, header=None, skiprows=1, delimiter="\t", dtype={0: float, 1: float})
                    dfr_sorted = dfr.sort_values(by=0)  # increasing order

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
                    spectrum.baseline.sigma = 10
                    self.spectrums.append(spectrum)

        QTimer.singleShot(100, self.upd_spectra_list)
        self.ui.tabWidget.setCurrentWidget(self.ui.tab_spectra)

    def xrange_correction(self, ref_value=None, sel_spectra=None):
        """Correct peak shift based on Si reference sample."""
        try:
            text = self.ui.ent_si_peak.text()
            ref_value = round(float(text), 3)                
            if sel_spectra is None:
                _, sel_spectra = self.get_spectrum_object()
               
            for spectrum in sel_spectra:
                new_xcorrection_value = 520.7 - ref_value
                print(f"Applying x range correction of {new_xcorrection_value} for spectrum {spectrum.fname}")
                spectrum.apply_xcorrection(new_xcorrection_value)
            QTimer.singleShot(100, self.upd_spectra_list)

        except ValueError:
            QMessageBox.warning(self.ui.tabWidget, "Input Error", "Please enter a valid numeric Si peak reference.")
    
    def undo_xrange_correction(self, sel_spectra=None):
        """Undo peak shift correction for the given spectra."""
        if sel_spectra is None:
            _, sel_spectra = self.get_spectrum_object()
        for spectrum in sel_spectra:
            spectrum.undo_xcorrection()

        QTimer.singleShot(100, self.upd_spectra_list)       
        
    def copy_fit_model(self):
        """Copy the model dictionary of the selected spectrum"""
        # Get only 1 spectrum among several selected spectrum:
        self.get_fit_settings()
        sel_spectrum, _ = self.get_spectrum_object()
        if len(sel_spectrum.peak_models) == 0:
            self.ui.lbl_copied_fit_model_2.setText("")
            msg = ("No fit results to collect or copy")
            print(msg)
            self.copied_fit_model = None
            return
        else:
            self.copied_fit_model = None
            self.copied_fit_model = deepcopy(sel_spectrum.save())
        self.ui.lbl_copied_fit_model_2.setText("copied")     

    def paste_fit_model(self, fnames=None):
        """Apply the copied fit model to the selected spectra"""
        self.ui.centralwidget.setEnabled(False)  # Disable GUI
        
        fnames = fnames or self.get_spectrum_fnames()
            
        for fname in fnames:
            spectrum, _ = self.spectrums.get_objects(fname)
            spectrum.reinit()
            
        self.ntot = len(fnames)
        ncpu = self.settings.value("fit_settings/ncpu", 1, type=int)
        fit_model = deepcopy(self.copied_fit_model)

        if fit_model is not None:
            self.spectrums.pbar_index = 0
            self.thread = FitThread(self.spectrums, fit_model, fnames, ncpu)
            self.thread.finished.connect(self.fit_completed)
            self.thread.start()
        else:
            show_alert("Nothing to paste")
            self.ui.centralwidget.setEnabled(True)

        # Update progress bar & text
        self.start_time = time.time()
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress_bar)
        self.progress_timer.start(100)
        
    def paste_peaks(self, sel_spectra=None):
        """Copy and paste only peak labels and peak models to the selected spectra."""
        if not self.copied_fit_model:
            show_alert("No fit model copied")
            return
        # Extract data from the correct location
        fit_data = self.copied_fit_model
        
        self.current_peaks = {
            "peak_labels": fit_data.get("peak_labels", []),
            "peak_models": deepcopy(fit_data.get("peak_models", {}))  
        }
        
        if not self.current_peaks["peak_labels"] and not self.current_peaks["peak_models"]:
            show_alert("No peak data to paste")
            return

        if sel_spectra is None:
            _, sel_spectra = self.get_spectrum_object()

        # Assign peak labels and models to selected spectra
        for spectrum in sel_spectra:
            spectrum.set_attributes(self.current_peaks)

        QTimer.singleShot(50, self.refresh_gui)
        
    def paste_peaks_all(self):
        checked_spectra = self.get_checked_spectra()
        self.paste_peaks(checked_spectra)
        
        
    def get_fit_settings(self):
        """Retrieve all settings for the fitting action from the GUI"""
        sel_spectrum, sel_spectra = self.get_spectrum_object()
        fit_params = sel_spectrum.fit_params.copy()
        
        fit_params['fit_negative'] = self.settings.value("fit_settings/fit_negative", False, type=bool)
        fit_params['max_ite'] = self.settings.value("fit_settings/max_ite", 200, type=int)
        fit_params['method'] = self.settings.value("fit_settings/method", "Leastsq")
        fit_params['ncpu'] = self.settings.value("fit_settings/ncpu", 1, type=int)
        fit_params['xtol'] = self.settings.value("fit_settings/xtol", 1e-4, type=float)
        
        sel_spectrum.fit_params = fit_params
            
    def fit(self, fnames=None):
        """Fit the selected spectrum(s) with current parameters"""
        self.get_fit_settings()
        fnames = fnames or self.get_spectrum_fnames()
            
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
        self.get_fit_model()
        
        self.ui.centralwidget.setEnabled(False)  # Disable GUI
        if self.loaded_fit_model is None:
            show_alert("Select from the list or load a fit model.")
            self.ui.centralwidget.setEnabled(True)
            return

        fnames = fnames or self.get_spectrum_fnames()
            
        for fname in fnames:
            spectrum, _ = self.spectrums.get_objects(fname)
            spectrum.reinit()

        self.ntot = len(fnames)
        ncpu = self.settings.value("fit_settings/ncpu", 1, type=int)
        fit_model = self.loaded_fit_model
        
        self.spectrums.pbar_index = 0
        self.thread = FitThread(self.spectrums, fit_model, fnames, ncpu)
        self.thread.finished.connect(self.fit_completed)
        self.thread.start()

        # Update progress bar & text
        self.start_time = time.time()
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress_bar)
        self.progress_timer.start(100)

    def update_progress_bar(self):
        """Update fitting progress in GUI"""
        index = self.spectrums.pbar_index
        percent = 100 * (index + 1) / self.ntot
        elapsed_time = time.time() - self.start_time
        text = f"{index}/{self.ntot} ({elapsed_time:.2f}s)"
        self.ui.progressBar_2.setValue(percent)
        self.ui.progressText_2.setText(text)
        if self.spectrums.pbar_index >= self.ntot - 1:
            self.progress_timer.stop()

    def fit_completed(self):
        """Update GUI after completing fitting process."""
        self.upd_spectra_list()
        QTimer.singleShot(200,  self.spectra_viewer.rescale)
        self.ui.progressBar_2.setValue(100)
        self.ui.centralwidget.setEnabled(True)

    def apply_loaded_fit_model_all(self):
        """Apply the loaded fit model to all spectra"""
        checked_spectra = self.get_checked_spectra()
        fnames = checked_spectra.fnames
        self.apply_loaded_fit_model(fnames=fnames)

    def upd_spectra_list(self):
        """Show spectrums in a listbox"""
        # Store the checked state of each item
        checked_states = {}
        for index in range(self.ui.spectrums_listbox.count()):
            item = self.ui.spectrums_listbox.item(index)
            checked_states[item.text()] = item.checkState()

        current_row = self.ui.spectrums_listbox.currentRow()
        current_selection = {item.text() for item in self.ui.spectrums_listbox.selectedItems()}

        self.ui.spectrums_listbox.clear()
        
        for spectrum in self.spectrums:
            fname = spectrum.fname
            item = populate_spectrum_listbox(spectrum, fname, checked_states)
            self.ui.spectrums_listbox.addItem(item)

        item_count = self.ui.spectrums_listbox.count()
        self.ui.item_count_label_3.setText(f"{item_count} points")
        
        # Restore selection states
        for index in range(self.ui.spectrums_listbox.count()):
            item = self.ui.spectrums_listbox.item(index)
            if item.text() in current_selection:
                item.setSelected(True)

        # Management of selecting item of listbox
        if current_row >= item_count:
            current_row = item_count - 1
        if current_row >= 0:
            self.ui.spectrums_listbox.setCurrentRow(current_row)
        else:
            if item_count > 0:
                self.ui.spectrums_listbox.setCurrentRow(0)
        
        QTimer.singleShot(50, self.refresh_gui)

    def plot(self):
        """Plot spectra or fit results in the main plot area."""
        fnames = self.get_spectrum_fnames()
        
        selected_spectrums = Spectra()
        selected_spectrums = [spectrum for spectrum in self.spectrums if spectrum.fname in fnames]
        # Limit the number of spectra to avoid crashes
        selected_spectrums = selected_spectrums[:30]
        if not selected_spectrums:
            return
        self.spectra_viewer.plot(selected_spectrums)

        # Show correction value of the last selected item
        xcorrection_value = round(selected_spectrums[-1].xcorrection_value, 3)
        text = f"[{xcorrection_value}]"
        self.ui.lbl_correction_value.setText(text)
        
        self.read_x_range()
        self.peak_table.show(selected_spectrums[0])

    def refresh_gui(self):
        """Trigger the fnc to plot spectra"""
        self.delay_timer.start(100)

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

    def update_peak_model(self):
        """Update the peak model in the Spectra viewer based on combobox selection."""
        selected_model = self.ui.cbb_fit_models_2.currentText()
        self.spectra_viewer.set_peak_model(selected_model)


    def get_fit_model(self):
        """Get the selected fit model from the combobox or last loaded file."""
        if self.fit_model_manager.combo_models.currentIndex() == -1:
            self.loaded_fit_model = None
            return
        
        model_name = self.fit_model_manager.combo_models.currentText()
        folder_path = self.fit_model_manager.default_model_folder
        path_in_folder = os.path.join(folder_path, model_name)
    
        # Try default folder first
        if os.path.exists(path_in_folder):
            self.loaded_fit_model = self.spectrums.load_model(path_in_folder, ind=0)
            return

        # Fallback: ask user to load a file if not found in default folder
        loaded_model_path = self.fit_model_manager.loaded_fit_model
        if loaded_model_path and os.path.exists(loaded_model_path):
            self.loaded_fit_model = self.spectrums.load_model(loaded_model_path, ind=0)
            return
        
        # If all fails
        show_alert('Fit model file not found')
        self.loaded_fit_model = None

    def save_fit_model(self):
        """Save the fit model of the currently selected spectrum to a JSON file."""

        sel_spectrum, sel_spectra = self.get_spectrum_object()
        path = self.fit_model_manager.default_model_folder
        save_path, _ = QFileDialog.getSaveFileName(
            self.ui.tabWidget, "Save fit model", path,
            "JSON Files (*.json)")
        
        if save_path and sel_spectrum:
            self.spectrums.save(save_path, [sel_spectrum.fname])
            show_alert("Fit model is saved (JSON file)")
        else:
            print("Nothing is saved.")
        self.fit_model_manager.scan_and_udp_model_cbb() # Update model list in combobox

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
        """Get the selected spectrum(s) object"""
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
    
    def read_x_range(self):
        """Read the x range of the selected spectrum"""
        sel_spectrum, sel_spectra = self.get_spectrum_object()
        xmin = round(sel_spectrum.x[0], 3)
        xmax = round(sel_spectrum.x[-1], 3)
        self.ui.range_min_2.setText(str(xmin))
        self.ui.range_max_2.setText(str(xmax))

    def set_x_range(self, fnames=None):
        """Set a new x range for the selected spectrum"""
        new_x_min = float(self.ui.range_min_2.text())
        new_x_max = float(self.ui.range_max_2.text())
        
        fnames = fnames or self.get_spectrum_fnames()
        for fname in fnames:
            self.spectrums.get_objects(fname)[0].reinit()

        for fname in fnames:
            spectrum, _ = self.spectrums.get_objects(fname)
            spectrum.range_min = float(self.ui.range_min_2.text())
            spectrum.range_max = float(self.ui.range_max_2.text())

            ind_min = closest_index(spectrum.x0, new_x_min)
            ind_max = closest_index(spectrum.x0, new_x_max)
            spectrum.x = spectrum.x0[ind_min:ind_max + 1].copy()
            spectrum.y = spectrum.y0[ind_min:ind_max + 1].copy()
            
        QTimer.singleShot(50, self.upd_spectra_list)
        QTimer.singleShot(300, self.spectra_viewer.rescale)

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
                
    def copy_baseline(self):
        """Copy baseline of the selected spectrum"""
        sel_spectrum, _ = self.get_spectrum_object()
        self.current_baseline = deepcopy(baseline_to_dict(sel_spectrum))
    
    def paste_baseline(self, sel_spectra=None):
        """Paste baseline to the selected spectrum(s)"""
        if sel_spectra is None:
            _, sel_spectra = self.get_spectrum_object()
        baseline_data = deepcopy(self.current_baseline)    
        dict_to_baseline(baseline_data, sel_spectra)
        QTimer.singleShot(50, self.refresh_gui)
    
    def paste_baseline_all(self):
        """Paste baseline to the all spectrum(s)"""
        checked_spectra = self.get_checked_spectra()
        self.paste_baseline(checked_spectra)

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
        QTimer.singleShot(100, self.upd_spectra_list)
        QTimer.singleShot(300, self.spectra_viewer.rescale)
        
    def subtract_baseline_all(self):
        """Subtract the baseline for all spectra"""
        checked_spectra = self.get_checked_spectra()
        self.subtract_baseline(checked_spectra)

    def clear_peaks(self, fnames=None):
        """Clear all existing peak models of the selected spectrum(s)"""
        fnames = fnames or self.get_spectrum_fnames()
                    
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

    

    def paste_fit_model_all(self):
        """Apply the copied fit model to selected spectrum(s)"""
        checked_spectra = self.get_checked_spectra()
        fnames = checked_spectra.fnames
        self.paste_fit_model(fnames)

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
                params = {}
                fit_result = {'Filename': spectrum.fname}
                for model in spectrum.peak_models:
                    if hasattr(model, 'param_names') and hasattr(model,'param_hints'):
                        for param_name in model.param_names:
                            peak_id = param_name.split('_', 1)[0]
                            key = param_name.split('_', 1)[1]
                            
                            if key in model.param_hints and 'value' in model.param_hints[key]:
                                val = model.param_hints[key]['value']        
                                fit_result[param_name] = val
                                params[key] = val
                                
                        # Calculate peak area
                        model_type = model.name2  # Get the type of peak model : Lorentizan, Gaussian, etc...
                        area = calc_area(model_type, params)
                        if area is not None:
                            area_key = f"{peak_id}_area"
                            fit_result[area_key] = area
                                
                if len(fit_result) > 1:
                    fit_results_list.append(fit_result)
                    
        self.df_fit_results = (pd.DataFrame(fit_results_list)).round(3)

        if self.df_fit_results is not None and not self.df_fit_results.empty:
            # reindex columns according to the parameters names
            self.df_fit_results = self.df_fit_results.reindex(sorted(self.df_fit_results.columns), axis=1)
            names = []
            for name in self.df_fit_results.columns:
                if name in ["Filename"]:
                    name = '0' + name
                elif '_' in name:
                    name = 'z' + name[5:]
                names.append(name)
            self.df_fit_results = self.df_fit_results.iloc[:,list(np.argsort(names, kind='stable'))]
            # Replace peak_label
            columns = [replace_peak_labels(self.copied_fit_model, column) for column in self.df_fit_results.columns]
            self.df_fit_results.columns = columns
        
        self.df_table.show(self.df_fit_results)

    def split_fname(self):
        """Split the filename and populate the combobox."""

        dfr = self.df_fit_results
        fname_parts = dfr.loc[0, 'Filename'].split('_')
        self.ui.cbb_split_fname.clear()  # Clear existing items in combobox
        for part in fname_parts:
            self.ui.cbb_split_fname.addItem(part)

    def is_number(self, s):
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False
    
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
            # Convert each selected part to float if it's a number, else keep as string
            new_col = [
                float(part[selected_part_index]) if len(part) > selected_part_index and self.is_number(part[selected_part_index])
                else (part[selected_part_index] if len(part) > selected_part_index else None)
                for part in parts
            ]
            dfr[col_name] = new_col

        except Exception as e:
            print(f"Error adding new column to fit results dataframe: {e}")
            return

        self.df_fit_results = dfr
        self.df_table.show(self.df_fit_results)

    def send_df_to_viz(self):
        """Send the collected spectral data to the visualization tab"""
        dfs_new = self.graphs.original_dfs
        df_name = self.ui.ent_send_df_to_viz.text()
        dfs_new[df_name] = self.df_fit_results
        self.graphs.open_dfs(dfs=dfs_new, file_paths=None)


    def cosmis_ray_detection(self):
        self.spectrums.outliers_limit_calculation()

    def reinit(self, fnames=None):
        """Reinitialize the selected spectrum(s)"""
        fnames = fnames or self.get_spectrum_fnames()
        
        for fname in fnames:
            self.spectrums.get_objects(fname)[0].reinit()
            
        self.upd_spectra_list()
        QTimer.singleShot(200, self.spectra_viewer.rescale)

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
        """Function to save fitted results in an Excel file with colored columns."""
        last_dir = self.settings.value("last_directory", "/")
        save_path, _ = QFileDialog.getSaveFileName(
            self.ui.tabWidget, "Save DF fit results", last_dir,
            "Excel Files (*.xlsx)"
        )
        success, message = save_df_to_excel(save_path, self.df_fit_results)
        
        if success:
            QMessageBox.information(self.ui.tabWidget, "Success", message)
        else:
            QMessageBox.warning(self.ui.tabWidget, "Warning", message)

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
            view_text(ui, title, text)

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
        self.spectra_viewer.ax.clear()
        self.spectra_viewer.canvas.draw()


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
            
    def paste_peaks_fnc_handler(self):
        """Switch between 2 save fit fnc with the Ctrl key"""
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.paste_peaks_all()
        else:
            self.paste_peaks()

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
                    'spectrums': spectrum_to_dict(self.spectrums, is_map=False)
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
                        dict_to_spectrum(spectrum=spectrum, spectrum_data=spectrum_data, is_map=False)
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
        self.spectrums = Spectra()
        self.loaded_fit_model = None
        self.copied_fit_model = None
        self.df_fit_results = None

        # Clear Listbox
        self.ui.spectrums_listbox.clear()
        self.ui.item_count_label_3.setText("0 points")

        # Clear spectra plot view and reset selected spectrums
        self.spectra_viewer.sel_spectrums = None  
        self.spectra_viewer.refresh_plot() 
        
        self.df_table.clear()
        self.peak_table.clear()

        # Refresh the UI to reflect the cleared state
        QTimer.singleShot(50, self.spectra_viewer.rescale)
        QTimer.singleShot(100, self.upd_spectra_list)
        print("'Spectrums' Tab environment has been cleared.")

