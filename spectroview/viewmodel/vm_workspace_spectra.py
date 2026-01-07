"""ViewModel for Spectra Workspace - handles business logic and data management."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog

from spectroview.model.m_io import load_spectrum_file
from spectroview.model.m_settings import MSettings
from spectroview.model.m_spectra import MSpectra
from spectroview.model.m_spectrum import MSpectrum
from spectroview.viewmodel.utils import (
    FitThread,
    baseline_to_dict,
    calc_area,
    closest_index,
    dict_to_baseline,
    spectrum_to_dict,
    dict_to_spectrum,
    replace_peak_labels,
    save_df_to_excel
)


class VMWorkspaceSpectra(QObject):
    # ───── ViewModel → View signals ─────
    spectra_list_changed = Signal(list)      # list[str]
    spectra_selection_changed = Signal(list) # list[dict] → plot data
    count_changed = Signal(int)
    show_xcorrection_value = Signal(float)  # ΔX of first selected spectrum
    spectral_range_changed = Signal(float, float)
    
    fit_in_progress = Signal(bool)  # Enable/disable fit buttons
    fit_progress_updated = Signal(int, int, int, float)  # To show fitting progress in GUI
    
    # Fit results signals
    fit_results_updated = Signal(object)  # pd.DataFrame
    split_parts_updated = Signal(list)    # list[str] for combobox

    notify = Signal(str)  # general notifications
    
    def __init__(self, settings: MSettings):
        super().__init__()
        self.settings = settings
        self.spectra = MSpectra()

        self.selected_indices = []
        self._baseline_clipboard = None  # for copy/paste baseline
        self._peaks_clipboard = None    # for copy/paste peaks
        self._loaded_fit_model = None  # for applying loaded fit model
        self._current_peak_shape = "Lorentzian"
        self._fit_thread = None  # Track active fit thread
        self._is_fitting = False  # Track if fitting is in progress
        
        # Fit results data
        self.df_fit_results = None
        self._fitmodel_clipboard = None


    # View → ViewModel slots
    def load_files(self, paths: list[str]):
        existing_paths = {s.source_path for s in self.spectra}

        skipped = []

        for p in paths:
            path = str(Path(p).resolve())
            if path in existing_paths:
                skipped.append(Path(p).name)
                continue

            spectrum = load_spectrum_file(Path(p))
            self.spectra.add(spectrum)

        if skipped:
            self.notify.emit(
                f"Already loaded and skipped:\n" + "\n".join(skipped)
            )
        self._emit_list_update()


    def set_selected_indices(self, indices: list[int]):
        """Set currently selected spectra (via Listwidget) by their indices."""
        self.selected_indices = indices
        self._emit_selected_spectra()

    def _emit_selected_spectra(self):
        """Prepare and emit data for plotting the selected spectra."""
        selected_spectra = self.spectra.get(self.selected_indices)

        if not selected_spectra:
            self.spectra_selection_changed.emit([])
            return
        # emit list of the selected spectra to plot in View
        self.spectra_selection_changed.emit(selected_spectra)    

        # emit x-correction of first spectrum to show in GUI
        self.show_xcorrection_value.emit(selected_spectra[0].xcorrection_value)

        # emit spectral range of first selected spectrum to show in GUI
        s = selected_spectra[0]
        xmin = float(s.x[0])
        xmax = float(s.x[-1])
        self.spectral_range_changed.emit(xmin, xmax)
    
    def reorder_spectra(self, new_order: list[int]):
        """new_order = list of old indices in new visual order"""
        self.spectra.reorder(new_order)

        # After reorder, selection must be re-emitted
        self._emit_list_update()
        self._emit_selected_spectra()

    def remove_selected_spectra(self):
        """Remove currently selected spectra."""
        if not self.selected_indices:
            self.notify.emit("No spectra selected.")
            return
        old_selection = set(self.selected_indices)
        old_count = len(self.spectra)
        # Remove from model
        self.spectra.remove(self.selected_indices)
        
        new_count = len(self.spectra)
        self._emit_list_update()

        if new_count == 0:
            self.selected_indices = []
            self.spectra_selection_changed.emit([])
            return
        # Find closest valid index
        min_removed = min(old_selection)
        new_index = min(min_removed, new_count - 1)

        self.selected_indices = [new_index]
        self._emit_selected_spectra()
        
    # Internal helpers
    def _emit_list_update(self):
        """Emit updated list of spectra names and count."""
        names = [s.fname for s in self.spectra]
        self.spectra_list_changed.emit(names)
        self.count_changed.emit(len(self.spectra))

    def add_peak_at(self, x: float):
        if not self.selected_indices:
            return

        spectrum = self.spectra.get(self.selected_indices)[0]
        
        fit_settings = self.settings.load_fit_settings()

        maxshift = fit_settings.get("maxshift", 20.0)
        maxfwhm = fit_settings.get("maxfwhm", 200.0)
        peak_shape = self._current_peak_shape or "Lorentzian"

        spectrum.add_peak_model(peak_shape,x,dx0=(maxshift, maxshift),dfwhm=maxfwhm)
        self._emit_selected_spectra()

    def remove_peak_at(self, x: float):
        if not self.selected_indices:
            return

        spectrum = self.spectra.get(self.selected_indices)[0]

        if not spectrum.peak_models:
            return

        idx = min(
            range(len(spectrum.peak_models)),
            key=lambda i: abs(
                spectrum.peak_models[i].param_hints["x0"]["value"] - x
            )
        )

        del spectrum.peak_models[idx]
        del spectrum.peak_labels[idx]
        self._emit_selected_spectra()

    def set_baseline_settings(self, settings: dict):
        if not self.selected_indices:
            return

        for spectrum in self.spectra.get(self.selected_indices):
            bl = spectrum.baseline
            bl.attached = settings["attached"]
            bl.sigma = settings["noise"]

            if settings["mode"] == "Linear":
                bl.mode = "Linear"
            else:
                bl.mode = "Polynomial"
                bl.order_max = settings["order"]

        self._emit_selected_spectra()
 

    def add_baseline_point(self, x: float, y: float):
        if not self.selected_indices:
            return

        spectrum = self.spectra.get(self.selected_indices)[0]

        if spectrum.baseline.is_subtracted:
            self.notify.emit("Baseline already subtracted.")
            return

        spectrum.baseline.add_point(x, y)
        self._emit_selected_spectra()


    def remove_baseline_point(self, x: float):
        if not self.selected_indices:
            return

        spectrum = self.spectra.get(self.selected_indices)[0]

        if not spectrum.baseline.points:
            return

        xs, ys = spectrum.baseline.points
        if not xs:
            return

        idx = min(range(len(xs)), key=lambda i: abs(xs[i] - x))
        xs.pop(idx)
        ys.pop(idx)

        self._emit_selected_spectra()

    def apply_x_correction(self, measured_peak: float):
        if not self.selected_indices:
            self.notify.emit("No spectrum selected.")
            return

        spectra = self.spectra.get(self.selected_indices)

        SI_REF = 520.7
        delta_x = SI_REF - measured_peak 

        for s in spectra:
            s.apply_xcorrection(delta_x)

        # Trigger plot refresh
        self.show_xcorrection_value.emit(spectra[0].xcorrection_value)
        self._emit_selected_spectra()


    def undo_x_correction(self):
        """Undo X-axis correction for selected spectra."""
        if not self.selected_indices:
            self.notify.emit("No spectrum selected.")
            return

        spectra = self.spectra.get(self.selected_indices)

        for spectrum in spectra:
            spectrum.undo_xcorrection()

        self.show_xcorrection_value.emit(spectra[0].xcorrection_value)
        self._emit_selected_spectra()

    def reinit_spectra(self, apply_all: bool = False):
        """Reinitialize spectra to original data."""
        if apply_all:
            spectra = self.spectra
        else:
            if not self.selected_indices:
                self.notify.emit("No spectrum selected.")
                return
            spectra = self.spectra.get(self.selected_indices)

        for spectrum in spectra:
            spectrum.reinit()
        
        self._emit_selected_spectra() # Refresh plot 


    def apply_spectral_range(self, xmin: float, xmax: float, apply_all: bool):
        if not self.selected_indices:
            return

        if xmin > xmax:
            xmin, xmax = xmax, xmin

        spectra = (
            self.spectra
            if apply_all
            else self.spectra.get(self.selected_indices)
        )

        for spectrum in spectra:
            spectrum.reinit()

            spectrum.range_min = xmin
            spectrum.range_max = xmax

            i_min = closest_index(spectrum.x0, xmin)
            i_max = closest_index(spectrum.x0, xmax)

            spectrum.x = spectrum.x0[i_min:i_max + 1].copy()
            spectrum.y = spectrum.y0[i_min:i_max + 1].copy()

        self._emit_selected_spectra()

    def copy_baseline(self):
        if not self.selected_indices:
            self.notify.emit("No spectrum selected.")
            return

        spectrum = self.spectra.get(self.selected_indices)[0]
        self._baseline_clipboard = deepcopy(baseline_to_dict(spectrum))

    def paste_baseline(self, apply_all: bool = False):
        if self._baseline_clipboard is None:
            self.notify.emit("No baseline copied.")
            return

        if apply_all:
            spectra = self.spectra
        else:
            if not self.selected_indices:
                self.notify.emit("No spectrum selected.")
                return
            spectra = self.spectra.get(self.selected_indices)

        dict_to_baseline(
            deepcopy(self._baseline_clipboard),
            spectra
        )

        self._emit_selected_spectra()

    def subtract_baseline(self, apply_all: bool = False):
        if apply_all:
            spectra = self.spectra
        else:
            if not self.selected_indices:
                self.notify.emit("No spectrum selected.")
                return
            spectra = self.spectra.get(self.selected_indices)

        for spectrum in spectra:
            if not spectrum.baseline.is_subtracted:
                spectrum.subtract_baseline()

        self._emit_selected_spectra()

    def delete_baseline(self, apply_all: bool = False):
        """Delete baseline (points + subtraction state)."""

        if apply_all:
            spectra = self.spectra
        else:
            if not self.selected_indices:
                self.notify.emit("No spectrum selected.")
                return
            spectra = self.spectra.get(self.selected_indices)

        for spectrum in spectra:
            bl = spectrum.baseline

            # Clear baseline points
            if bl.points:
                xs, ys = bl.points
                xs.clear()
                ys.clear()

            # Reset subtraction state
            bl.is_subtracted = False

        self._emit_selected_spectra()

    def copy_peaks(self):
        if not self.selected_indices:
            self.notify.emit("No spectrum selected.")
            return

        spectrum = self.spectra.get(self.selected_indices)[0]

        if not spectrum.peak_models:
            self.notify.emit("No peaks to copy.")
            return
        self._peaks_clipboard = deepcopy(spectrum.save())

    def paste_peaks(self, apply_all: bool = False):
        if not hasattr(self, "_peaks_clipboard") or self._peaks_clipboard is None:
            self.notify.emit("No peaks copied.")
            return

        spectra = (
            self.spectra
            if apply_all
            else self.spectra.get(self.selected_indices)
        )

        for spectrum in spectra:
            spectrum.set_attributes(
                {
                    "peak_labels": self._peaks_clipboard.get("peak_labels", []),
                    "peak_models": deepcopy(
                        self._peaks_clipboard.get("peak_models", {})
                    ),
                }
            )

        self._emit_selected_spectra()


    def delete_peaks(self, apply_all: bool = False):
        spectra = (
            self.spectra
            if apply_all
            else self.spectra.get(self.selected_indices)
        )

        if not spectra:
            self.notify.emit("No spectrum selected.")
            return

        for spectrum in spectra:
            if spectrum.peak_models:
                spectrum.remove_models()

        self._emit_selected_spectra()


    def fit(self, apply_all: bool = False):
        # Prevent concurrent fit operations
        if self._is_fitting:
            self.notify.emit("Fit already in progress. Please wait...")
            return

        spectra = self.spectra if apply_all else self.spectra.get(self.selected_indices)
        
        if not spectra:
            return

        # Check if any spectrum has peak models
        has_peaks = any(s.peak_models for s in spectra)
        if not has_peaks:
            self.notify.emit("No peaks to fit.")
            return

        self._is_fitting = True
        self.fit_in_progress.emit(True)

        try:
            for s in spectra:
                if s.peak_models:
                    s.fit()
        except Exception as e:
            self.notify.emit(f"Fit error: {e}")
        finally:
            self._is_fitting = False
            self.fit_in_progress.emit(False)
            self._emit_selected_spectra()
    
    def copy_fit_model(self):
        if not self.selected_indices:
            self.notify.emit("No spectrum selected.")
            return

        spectrum = self.spectra.get(self.selected_indices)[0]
        if not spectrum.peak_models:
            self.notify.emit("No fit results to copy.")
            return

        self._fitmodel_clipboard = deepcopy(spectrum.save())

    def paste_fit_model(self, apply_all: bool = False):
        if not hasattr(self, "_fitmodel_clipboard"):
            self.notify.emit("No fit model copied.")
            return
        spectra = self.spectra if apply_all else self.spectra.get(self.selected_indices)

        for s in spectra:
            s.reinit()

        self._run_fit_thread(deepcopy(self._fitmodel_clipboard), spectra)

    def save_fit_model(self):
        if not self.selected_indices:
            self.notify.emit("No spectrum selected.")
            return

        spectrum = self.spectra.get(self.selected_indices)[0]

        if not spectrum.peak_models:
            self.notify.emit("No fit model to save.")
            return

        path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Fit Model",
            "",
            "JSON Files (*.json)"
        )

        if not path:
            return

        self.spectra.save(path, [spectrum.fname])
        self.notify.emit("Fit model saved successfully.")

    def apply_loaded_fit_model(self, apply_all: bool = False):
        if not hasattr(self, "_vm_fit_model_builder"):
            self.notify.emit("Fit model manager not connected.")
            return

        model_path = self._vm_fit_model_builder.get_current_model_path()
        
        if model_path is None or not model_path.exists():
            self.notify.emit("No fit model selected.")
            return
        #Load fit model from JSON file
        try:
            fit_model = self.spectra.load_model(str(model_path), ind=0)
        except Exception as e:
            self.notify.emit(f"Failed to load fit model:\n{e}")
            return

        spectra = self.spectra if apply_all else self.spectra.get(self.selected_indices)
        if not spectra:
            self.notify.emit("No spectrum selected.")
            return

        for s in spectra:
            s.reinit()

        self._run_fit_thread(fit_model, spectra)

    def _run_fit_thread(self, fit_model: dict, spectra):
        # Prevent concurrent fit operations
        if self._is_fitting:
            self.notify.emit("Fit already in progress. Please wait...")
            return

        if not spectra:
            self.notify.emit("No spectra selected.")
            return

        # Cancel any existing thread
        if self._fit_thread and self._fit_thread.isRunning():
            self._fit_thread.terminate()
            self._fit_thread.wait()

        fnames = [s.fname for s in spectra]
        ncpu = self.settings.load_fit_settings().get("ncpu", 1)

        self.spectra.pbar_index = 0

        self._is_fitting = True
        self.fit_in_progress.emit(True)

        self._fit_thread = FitThread(
            self.spectra,
            fit_model,
            fnames,
            ncpu
        )
        self._fit_thread.progress_changed.connect(self.fit_progress_updated.emit)
        self._fit_thread.finished.connect(self._on_fit_finished)
        self._fit_thread.start()

    def _on_fit_finished(self):
        """Handle fit thread completion."""
        self._is_fitting = False
        self.fit_in_progress.emit(False)
        
        # Don't reset progress bar - let final state (X/X 100%) remain visible
        self._emit_selected_spectra()
        
        # Cleanup thread
        if self._fit_thread:
            self._fit_thread.deleteLater()
            self._fit_thread = None


    def set_fit_model_builder(self, vm_fit_model_builder):
        self._vm_fit_model_builder = vm_fit_model_builder
 
    def set_peak_shape(self, shape: str):
        """Receive peak shape from View."""
        self._current_peak_shape = shape


    def update_peak_label(self, index, text):
        s = self.spectra.get(self.selected_indices)[0]
        s.peak_labels[index] = text
        self._emit_selected_spectra()

    def update_peak_model(self, index, model_name):
        s = self.spectra.get(self.selected_indices)[0]
        pm = s.peak_models[index]

        x0 = pm.param_hints["x0"]["value"]
        ampli = pm.param_hints["ampli"]["value"]

        new_pm = s.create_peak_model(
            index + 1,
            model_name,
            x0=x0,
            ampli=ampli,
            dx0=(20.0, 20.0)  # ✅ FIXED
        )

        s.peak_models[index] = new_pm
        s.result_fit = None
        self._emit_selected_spectra()


    def update_peak_param(self, index, key, field, value):
        s = self.spectra.get(self.selected_indices)[0]
        s.peak_models[index].param_hints[key][field] = value
        self._emit_selected_spectra()

    def delete_peak(self, index):
        s = self.spectra.get(self.selected_indices)[0]
        del s.peak_models[index]
        del s.peak_labels[index]
        self._emit_selected_spectra()

    def update_dragged_peak(self, x: float, y: float):
        """Update peak position during dragging (real-time update).
        
        Args:
            x: New x position (center)
            y: New y value (amplitude/intensity)
        """
        if not self.selected_indices:
            return

        spectrum = self.spectra.get(self.selected_indices)[0]
        
        if not spectrum.peak_models:
            return

        # Find the peak model being dragged (closest to new x position)
        # Note: The View already updates the model directly for immediate visual feedback
        # This method is here for any additional processing needed
        # The actual update happens in the View for performance
        pass

    def finalize_peak_drag(self):
        """Finalize peak drag operation - ensure model is synchronized."""
        if not self.selected_indices:
            return

        # Re-emit to ensure everything is synchronized
        self._emit_selected_spectra()


    def copy_spectrum_data_to_clipboard(self):
        """Copy X, Y, and peak model data of the first selected spectrum to clipboard as DataFrame."""
        self._copy_spectrum_data()

    def _copy_spectrum_data(self):
        """Copy X, Y, and peak model data of the first selected spectrum to clipboard as DataFrame."""
        import pandas as pd
        import numpy as np

        if not self.selected_indices:
            return

        selected_spectra = self.spectra.get(self.selected_indices)
        if not selected_spectra:
            return

        spectrum = selected_spectra[0]
        x_values = spectrum.x
        y_values = spectrum.y

        # Create a dictionary for the DataFrame
        data = {
            "X values": x_values,
            "Y values": y_values
        }

        # Add each peak model's evaluated Y values as a new column
        for i, peak_model in enumerate(spectrum.peak_models):
            # Evaluate peak model
            try:
                from copy import deepcopy
                param_hints_orig = deepcopy(peak_model.param_hints)
                for key in peak_model.param_hints.keys():
                    peak_model.param_hints[key]["vary"] = False
                
                params = peak_model.make_params()
                peak_model.param_hints = param_hints_orig
                
                y_peak = peak_model.eval(params, x=x_values)

                if hasattr(spectrum, 'peak_labels') and i < len(spectrum.peak_labels):
                    label = spectrum.peak_labels[i]
                else:
                    label = f"Peak {i + 1}"

                data[label] = y_peak
            except Exception as e:
                # Skip peaks that fail to evaluate
                continue

        # Create DataFrame and copy to clipboard
        df = pd.DataFrame(data)
        df.to_clipboard(index=False)

    def save_work(self):
        """Save current workspace to .spectra file."""
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save work",
            "",
            "SPECTROview Files (*.spectra)"
        )
        
        if not file_path:
            return
        
        try:
            data_to_save = {
                'spectrums': spectrum_to_dict(self.spectra, is_map=False)
            }
            with open(file_path, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            self.notify.emit("Work saved successfully.")
        except Exception as e:
            self.notify.emit(f"Error saving work: {e}")

    def load_work(self, file_path: str):
        """Load previously saved workspace from .spectra file."""
        try:
            with open(file_path, 'r') as f:
                load = json.load(f)
            
            # Clear existing data
            self.spectra = MSpectra()
            
            # Load all spectra
            for spectrum_id, spectrum_data in load.get('spectrums', {}).items():
                spectrum = MSpectrum()
                dict_to_spectrum(spectrum=spectrum, spectrum_data=spectrum_data, is_map=False)
                spectrum.preprocess()
                self.spectra.append(spectrum)
            
            # Update UI
            self._emit_list_update()
            if len(self.spectra) > 0:
                self.selected_indices = [0]
                self._emit_selected_spectra()
            else:
                self.selected_indices = []
                self.spectra_selection_changed.emit([])
            
        except Exception as e:
            self.notify.emit(f"Error loading work: {e}")

    def clear_workspace(self):
        """Clear all spectra and reset workspace to initial state."""
        # Clear data model
        self.spectra = MSpectra()
        self.selected_indices = []
        
        # Clear clipboard data
        self._baseline_clipboard = None
        self._peaks_clipboard = None
        self._fitmodel_clipboard = None
        self._loaded_fit_model = None
        
        # Stop any running fit thread
        if self._fit_thread and self._fit_thread.isRunning():
            self._fit_thread.terminate()
            self._fit_thread.wait()
            self._fit_thread = None
        
        self._is_fitting = False
        
        # Emit updates to View
        self._emit_list_update()
        self.spectra_selection_changed.emit([])
        self.fit_in_progress.emit(False)
        
        # Clear fit results
        self.df_fit_results = None
        self.fit_results_updated.emit(None)
    # ═════════════════════════════════════════════════════════════════════
    # Fit Results Methods
    # ═════════════════════════════════════════════════════════════════════
    
    def collect_fit_results(self):
        """Collect best-fit results from all spectra and create DataFrame."""
        if not self.spectra:
            self.notify.emit("No spectra loaded.")
            return
        
        # Copy current fit model for reference
        if self.selected_indices and self.spectra.get(self.selected_indices):
            spectrum = self.spectra.get(self.selected_indices)[0]
            if spectrum.peak_models:
                self._fitmodel_clipboard = deepcopy(spectrum.save())
        
        fit_results_list = []
        
        for spectrum in self.spectra:
            if not hasattr(spectrum, 'peak_models') or not spectrum.peak_models:
                continue
            
            params = {}
            fit_result = {'Filename': spectrum.fname}
            
            for model in spectrum.peak_models:
                model_name = model.name
                
                # Get result parameters if fit was performed
                # Check that result_fit exists, is not None, and has params attribute (not a function)
                if (hasattr(spectrum, 'result_fit') and 
                    spectrum.result_fit and 
                    hasattr(spectrum.result_fit, 'params')):
                    for param_name in model.param_names:
                        # Extract peak-specific parameter value
                        if param_name in spectrum.result_fit.params:
                            param_value = spectrum.result_fit.params[param_name].value
                            params[param_name] = param_value
                else:
                    # Use param_hints if no fit result
                    # Extract prefix from model (e.g., "m01" from "Model(lorentzian, prefix='m01')")
                    # Use model.prefix if available, otherwise parse from name
                    if hasattr(model, 'prefix') and model.prefix:
                        prefix = model.prefix.rstrip('_')  # Remove trailing underscore
                    else:
                        # Fallback: use model_name directly
                        prefix = model_name
                    
                    for key in model.param_hints:
                        param_name = f"{prefix}_{key}"
                        param_value = model.param_hints[key].get('value')
                        params[param_name] = param_value
                
                # Calculate peak area
                model_type = model.name2  # Get the type of peak model: Lorentzian, Gaussian, etc.
                # Extract prefix for this model
                if hasattr(model, 'prefix') and model.prefix:
                    peak_id = model.prefix.rstrip('_')
                else:
                    peak_id = model_name
                
                # Build params dict for this specific peak (filter by prefix)
                peak_params = {}
                for param_name, param_value in params.items():
                    if param_name.startswith(peak_id + '_'):
                        # Remove prefix to get parameter name (e.g., 'ampli', 'fwhm')
                        param_key = param_name.replace(peak_id + '_', '')
                        peak_params[param_key] = param_value
                
                area = calc_area(model_type, peak_params)
                if area is not None:
                    area_key = f"{peak_id}_area"
                    params[area_key] = area
            
            # Add all parameters to fit_result
            fit_result.update(params)
            
            if len(fit_result) > 1:  # Has more than just filename
                fit_results_list.append(fit_result)
        
        if not fit_results_list:
            self.notify.emit("No fit results to collect.")
            return
        
        # Create DataFrame
        self.df_fit_results = pd.DataFrame(fit_results_list).round(3)
        
        # Replace peak labels if clipboard has model (before sorting)
        if self._fitmodel_clipboard:
            columns = [
                replace_peak_labels(self._fitmodel_clipboard, col) 
                for col in self.df_fit_results.columns
            ]
            self.df_fit_results.columns = columns
        
        # Sort columns: Filename first, then grouped by parameter type (x0_, fwhm_, ampli_, etc.)
        cols = list(self.df_fit_results.columns)
        filename_col = ['Filename'] if 'Filename' in cols else []
        other_cols = [c for c in cols if c != 'Filename']
        
        # Define priority order for parameter types
        param_priority = {
            'x0': 0,
            'fwhm': 1,
            'ampli': 2,
            'area': 3,
            'sigma': 4,
            'gamma': 5,
            'fraction': 6,
            'height': 7,
        }
        
        # Sort by parameter type (prefix) then by peak identifier (suffix)
        def sort_key(col_name):
            if '_' in col_name:
                parts = col_name.split('_', 1)  # Split on first underscore
                param_type = parts[0]  # e.g., "x0", "ampli", "fwhm"
                peak_id = parts[1] if len(parts) > 1 else ''  # e.g., "p1", "p2"
                # Use priority if defined, otherwise use high number (appears last)
                priority = param_priority.get(param_type, 999)
                return (priority, param_type, peak_id)
            else:
                return (999, col_name, '')
        
        sorted_cols = sorted(other_cols, key=sort_key)
        final_cols = filename_col + sorted_cols
        
        self.df_fit_results = self.df_fit_results[final_cols]
        
        # Emit signal to update View
        self.fit_results_updated.emit(self.df_fit_results)
        #self.notify.emit(f"Collected results from {len(fit_results_list)} spectra.")
    
    def split_filename(self):
        """Split the first filename by underscore and emit parts for combobox."""
        if self.df_fit_results is None or self.df_fit_results.empty:
            self.notify.emit("No fit results available. Collect results first.")
            return
        
        fname = self.df_fit_results.loc[0, 'Filename']
        parts = fname.split('_')
        
        self.split_parts_updated.emit(parts)
    
    def add_column_from_filename(self, col_name: str, part_index: int):
        """Add a new column to fit results by extracting part from filename."""
        if self.df_fit_results is None or self.df_fit_results.empty:
            self.notify.emit("No fit results available.")
            return
        
        if not col_name:
            self.notify.emit("Please enter a column name.")
            return
        
        # Check if column already exists
        if col_name in self.df_fit_results.columns:
            self.notify.emit(f"Column '{col_name}' already exists. Please choose a different name.")
            return
        
        try:
            parts = self.df_fit_results['Filename'].str.split('_')
            
            # Extract selected part and convert to float if possible
            new_col = []
            for part in parts:
                if len(part) > part_index:
                    value = part[part_index]
                    # Try to convert to float
                    try:
                        new_col.append(float(value))
                    except (ValueError, TypeError):
                        new_col.append(value)
                else:
                    new_col.append(None)
            
            self.df_fit_results[col_name] = new_col
            
            # Emit updated dataframe
            self.fit_results_updated.emit(self.df_fit_results)
            self.notify.emit(f"Added column '{col_name}' in the fit results.")
            
        except Exception as e:
            self.notify.emit(f"Error adding column: {e}")
    
    def save_fit_results(self):
        """Save fit results DataFrame to Excel file."""
        if self.df_fit_results is None or self.df_fit_results.empty:
            self.notify.emit("No fit results to save.")
            return
        
        last_dir = self.settings.load_fit_settings().get("last_directory", "/")
        save_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Fit Results",
            last_dir,
            "Excel Files (*.xlsx)"
        )
        
        if not save_path:
            return
        
        success, message = save_df_to_excel(save_path, self.df_fit_results)
        
        if success:
            self.notify.emit("Fit results saved successfully.")
        else:
            self.notify.emit(f"Error saving results: {message}")
    
    def send_results_to_graphs(self, df_name: str):
        """Send fit results to visualization tab (placeholder for future implementation)."""
        if self.df_fit_results is None or self.df_fit_results.empty:
            self.notify.emit("No fit results to send.")
            return
        
        # TODO: Implement when Graphs workspace is converted to MVVM
        # For now, just notify
        self.notify.emit(f"Send to Viz feature will be implemented when Graphs workspace is converted to MVVM.")
