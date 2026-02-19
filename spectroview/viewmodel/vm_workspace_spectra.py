"""ViewModel for Spectra Workspace - handles business logic and data management."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from lmfit import fit_report
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox

from spectroview.model.m_io import load_spectrum_file, load_TRPL_data, load_wdf_spectrum, load_spc_spectrum
from spectroview.model.m_settings import MSettings
from spectroview.model.m_spectra import MSpectra
from spectroview.model.m_spectrum import MSpectrum
from spectroview.viewmodel.utils import (
    ApplyFitModelThread, FitThread,
    baseline_to_dict,
    calc_area,
    closest_index,
    dict_to_baseline,

    replace_peak_labels,
    save_df_to_excel,
    view_text,
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
    send_df_to_graphs = Signal(str, object)  # (df_name, pd.DataFrame)

    notify = Signal(str)  # general notifications
    
    def __init__(self, settings: MSettings):
        super().__init__()
        self.settings = settings
        self.spectra = MSpectra()

        self.selected_fnames = []  # Changed from indices to fnames for robust identification
        self._baseline_clipboard = None  # for copy/paste baseline
        self._peaks_clipboard = None    # for copy/paste peaks
        self._loaded_fit_model = None  # for applying loaded fit model
        self._current_peak_shape = "Lorentzian"
        self._fit_thread = None  # Track active fit thread
        self._is_fitting = False  # Track if fitting is in progress
        
        # Fit results data
        self.df_fit_results = None
        self._fitmodel_clipboard = None
    
    # ═════════════════════════════════════════════════════════════════════
    # Helper methods for fname-based spectrum retrieval
    # ═════════════════════════════════════════════════════════════════════
    
    def _get_spectrum_by_fname(self, fname: str) -> MSpectrum | None:
        """Get a single spectrum by its fname (unique identifier)."""
        for spectrum in self.spectra:
            if spectrum.fname == fname:
                return spectrum
        return None
    
    def _get_spectra_by_fnames(self, fnames: list[str]) -> list[MSpectrum]:
        """Get multiple spectra by their fnames."""
        result = []
        for fname in fnames:
            spectrum = self._get_spectrum_by_fname(fname)
            if spectrum is not None:
                result.append(spectrum)
        return result
    
    def _get_selected_spectra(self) -> list[MSpectrum]:
        """Get currently selected spectra that are also active (checked).
        
        Filters selected spectra to only include those with is_active=True,
        so operations respect checkbox state.
        """
        selected = self._get_spectra_by_fnames(self.selected_fnames)
        # Filter to only active (checked) spectra
        return [s for s in selected if s.is_active]


    # View → ViewModel slots
    def load_files(self, paths: list[str]):
        """Load spectrum files from disk."""
        # Build sets of existing identifiers for fast lookup
        existing_paths = {s.source_path for s in self.spectra if s.source_path}
        existing_fnames = {s.fname for s in self.spectra}
        
        loaded_files = []
        last_valid_path = None

        for p in paths:
            path = Path(p)
            resolved_path = str(path.resolve())
            
            # Check for duplicate path OR duplicate filename
            # This prevents loading the same file twice, or different files with same name
            if resolved_path in existing_paths:
                self.notify.emit(f"Spectrum '{path.name}' already loaded (path match), skipping.")
                continue
                
            if path.stem in existing_fnames:
                self.notify.emit(f"Spectrum '{path.name}' already loaded (name match), skipping.")
                continue

            try:
                # Use appropriate loader based on file extension
                if path.suffix.lower() == '.dat':
                    spectrum = load_TRPL_data(path)
                elif path.suffix.lower() == '.wdf':
                    spectrum = load_wdf_spectrum(path)
                elif path.suffix.lower() == '.spc':
                    spectrum = load_spc_spectrum(path)
                else:
                    spectrum = load_spectrum_file(path)
                self.spectra.add(spectrum)
                loaded_files.append(path.name)
                last_valid_path = path  # Track last successfully loaded file
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Error loading {path.name}: {str(e)}")

        if loaded_files:
            self._emit_list_update()
            
            # Update last_directory setting
            if last_valid_path:
                self.settings.set_last_directory(str(last_valid_path.parent))


    def set_selected_indices(self, indices: list[int]):
        """Set currently selected spectra by list widget indices.
        
        Converts list indices to fnames for robust identification.
        """
        # Convert indices to fnames (unique identifiers)
        fnames = []
        for idx in indices:
            if 0 <= idx < len(self.spectra):
                fnames.append(self.spectra[idx].fname)
        
        # Store fnames (ensure uniqueness while preserving order)
        self.selected_fnames = list(dict.fromkeys(fnames))
        self._emit_selected_spectra()
    
    def set_selected_fnames(self, fnames: list[str]):
        """Set currently selected spectra by their fnames directly."""
        # Store fnames (ensure uniqueness while preserving order)
        self.selected_fnames = list(dict.fromkeys(fnames))
        self._emit_selected_spectra()
    
    def _get_active_spectra(self) -> list:
        """Get list of active spectra (used for apply_all operations).
        
        Returns:
            List of active MSpectrum objects (where is_active=True)
        """
        return [s for s in self.spectra if s.is_active]

    def _emit_selected_spectra(self):
        """Prepare and emit data for plotting the selected spectra."""
        selected_spectra = self._get_selected_spectra()

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
        if not self.selected_fnames:
            self.notify.emit("No spectra selected.")
            return
        
        # Find indices to remove (fname-based)
        indices_to_remove = []
        for idx, spectrum in enumerate(self.spectra):
            if spectrum.fname in self.selected_fnames:
                indices_to_remove.append(idx)
        
        if not indices_to_remove:
            return
        
        # Store first removed index for re-selection
        min_removed_idx = min(indices_to_remove)
        
        # Remove from model
        self.spectra.remove(indices_to_remove)
        
        new_count = len(self.spectra)
        self._emit_list_update()

        if new_count == 0:
            self.selected_fnames = []
            self.spectra_selection_changed.emit([])
            return
        
        # Select closest spectrum by index
        new_index = min(min_removed_idx, new_count - 1)
        self.selected_fnames = [self.spectra[new_index].fname]
        self._emit_selected_spectra()
        
    # Internal helpers
    def _emit_list_update(self):
        """Emit updated list of spectra (full objects) and count."""
        self.spectra_list_changed.emit(list(self.spectra))
        self.count_changed.emit(len(self.spectra))

    def add_peak_at(self, x: float):
        if not self.selected_fnames:
            return

        spectrum = self._get_selected_spectra()[0]
        
        fit_settings = self.settings.load_fit_settings()

        maxshift = fit_settings.get("maxshift", 20.0)
        maxfwhm = fit_settings.get("maxfwhm", 200.0)
        peak_shape = self._current_peak_shape or "Lorentzian"

        spectrum.add_peak_model(peak_shape,x,dx0=(maxshift, maxshift),dfwhm=maxfwhm)
        
        # Initialize decay model parameters with reasonable values
        if peak_shape in ["DecaySingleExp", "DecayBiExp"]:
            self._initialize_decay_params(spectrum.peak_models[-1], spectrum)
        
        self._emit_selected_spectra()
    
    def _initialize_decay_params(self, peak_model, spectrum):
        """Initialize decay model parameters with reasonable values for TRPL fitting.
        
        Sets proper initial values and bounds for exponential decay parameters:
        - A, A1, A2: Amplitudes (based on max intensity)
        - tau, tau1, tau2: Decay time constants (lifetimes in ns)
        - B: Baseline offset (based on min intensity)
        """
        y_max = np.max(spectrum.y)
        y_min = np.min(spectrum.y)
        
        if peak_model.name2 == "DecaySingleExp":
            # Single exponential: A * exp(-t/tau) + B
            peak_model.set_param_hint("A", value=y_max, min=0, vary=True)
            peak_model.set_param_hint("tau", value=5.0, min=0.1, max=100, vary=True)
            peak_model.set_param_hint("B", value=y_min, min=0, vary=True)
        
        elif peak_model.name2 == "DecayBiExp":
            # Bi-exponential: A1*exp(-t/tau1) + A2*exp(-t/tau2) + B
            peak_model.set_param_hint("A1", value=y_max * 0.7, min=0, vary=True)
            peak_model.set_param_hint("tau1", value=2.0, min=0.1, max=50, vary=True)
            peak_model.set_param_hint("A2", value=y_max * 0.3, min=0, vary=True)
            peak_model.set_param_hint("tau2", value=10.0, min=0.1, max=100, vary=True)
            peak_model.set_param_hint("B", value=y_min, min=0, vary=True)

    def remove_peak_at(self, x: float):
        if not self.selected_fnames:
            return

        spectrum = self._get_selected_spectra()[0]

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
        if not self.selected_fnames:
            return

        for spectrum in self._get_selected_spectra():
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
        if not self.selected_fnames:
            return

        spectrum = self._get_selected_spectra()[0]

        if spectrum.baseline.is_subtracted:
            self.notify.emit("Baseline already subtracted.")
            return

        spectrum.baseline.add_point(x, y)
        self._emit_selected_spectra()


    def remove_baseline_point(self, x: float):
        if not self.selected_fnames:
            return

        spectrum = self._get_selected_spectra()[0]

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
        if not self.selected_fnames:
            self.notify.emit("No spectrum selected.")
            return

        spectra = self._get_selected_spectra()

        SI_REF = 520.7
        delta_x = SI_REF - measured_peak 

        for s in spectra:
            s.apply_xcorrection(delta_x)

        # Trigger plot refresh
        self.show_xcorrection_value.emit(spectra[0].xcorrection_value)
        self._emit_selected_spectra()


    def undo_x_correction(self):
        """Undo X-axis correction for selected spectra."""
        if not self.selected_fnames:
            self.notify.emit("No spectrum selected.")
            return

        spectra = self._get_selected_spectra()

        for spectrum in spectra:
            spectrum.undo_xcorrection()

        self.show_xcorrection_value.emit(spectra[0].xcorrection_value)
        self._emit_selected_spectra()

    def reinit_spectra(self, apply_all: bool = False):
        """Reinitialize spectra to original data."""
        if apply_all:
            spectra = self._get_active_spectra()
        else:
            if not self.selected_fnames:
                self.notify.emit("No spectrum selected.")
                return
            spectra = self._get_selected_spectra()

        for spectrum in spectra:
            spectrum.reinit()
        
        self._emit_selected_spectra() # Refresh plot 
        self._emit_list_update()  # Refresh list colors after reinit


    def apply_spectral_range(self, xmin: float, xmax: float, apply_all: bool):
        if not self.selected_fnames:
            return

        if xmin > xmax:
            xmin, xmax = xmax, xmin

        spectra = (
            self.spectra
            if apply_all
            else self._get_selected_spectra()
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
        self._emit_list_update()  # Refresh list colors after range change

    def copy_baseline(self):
        if not self.selected_fnames:
            self.notify.emit("No spectrum selected.")
            return

        spectrum = self._get_selected_spectra()[0]
        self._baseline_clipboard = deepcopy(baseline_to_dict(spectrum))

    def paste_baseline(self, apply_all: bool = False):
        if self._baseline_clipboard is None:
            self.notify.emit("No baseline copied.")
            return

        if apply_all:
            spectra = self._get_active_spectra()
        else:
            if not self.selected_fnames:
                self.notify.emit("No spectrum selected.")
                return
            spectra = self._get_selected_spectra()

        # Apply baseline to selected spectra
        baseline_data = deepcopy(self._baseline_clipboard)
        dict_to_baseline(baseline_data, spectra)
        
        self._emit_selected_spectra()

    def subtract_baseline(self, apply_all: bool = False):
        if apply_all:
            spectra = self._get_active_spectra()
        else:
            if not self.selected_fnames:
                self.notify.emit("No spectrum selected.")
                return
            spectra = self._get_selected_spectra()

        for spectrum in spectra:
            if not spectrum.baseline.is_subtracted:
                spectrum.eval_baseline()
                spectrum.subtract_baseline()

        self._emit_selected_spectra()
        self._emit_list_update()  # Refresh list colors after baseline subtraction

    def delete_baseline(self, apply_all: bool = False):
        """Delete baseline (points + subtraction state)."""

        if apply_all:
            spectra = self.spectra
        else:
            if not self.selected_fnames:
                self.notify.emit("No spectrum selected.")
                return
            spectra = self._get_selected_spectra()

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
        self._emit_list_update()  # Refresh list colors after baseline deletion

    def copy_peaks(self):
        if not self.selected_fnames:
            self.notify.emit("No spectrum selected.")
            return

        spectrum = self._get_selected_spectra()[0]

        if not spectrum.peak_models:
            self.notify.emit("No peaks to copy.")
            return
        self._peaks_clipboard = deepcopy(spectrum.save())

    def paste_peaks(self, apply_all: bool = False):
        if not hasattr(self, "_peaks_clipboard") or self._peaks_clipboard is None:
            self.notify.emit("No peaks copied.")
            return

        spectra = (
            self._get_active_spectra()
            if apply_all
            else self._get_selected_spectra()
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
        self._emit_list_update()  # Refresh list colors after paste peaks


    def delete_peaks(self, apply_all: bool = False):
        spectra = (
            self._get_active_spectra()
            if apply_all
            else self._get_selected_spectra()
        )

        if not spectra:
            self.notify.emit("No spectrum selected.")
            return

        for spectrum in spectra:
            if spectrum.peak_models:
                spectrum.remove_models()

        self._emit_selected_spectra()
        self._emit_list_update()  # Refresh list colors after clear peaks


    def fit(self, apply_all: bool = False):
        # Prevent concurrent fit operations
        if self._is_fitting:
            self.notify.emit("Fit already in progress. Please wait...")
            return

        spectra = self._get_active_spectra() if apply_all else self._get_selected_spectra()
        
        if not spectra:
            return

        # Check if any spectrum has peak models
        has_peaks = any(s.peak_models for s in spectra)
        if not has_peaks:
            self.notify.emit("No peaks to fit.")
            return

        # Cancel any existing thread
        if self._fit_thread and self._fit_thread.isRunning():
            self._fit_thread.terminate()
            self._fit_thread.wait()

        self._is_fitting = True
        self.fit_in_progress.emit(True)

        # Use SimpleFitThread to fit each spectrum with its own models
        self._fit_thread = FitThread(spectra)
        self._fit_thread.progress_changed.connect(self.fit_progress_updated.emit)
        self._fit_thread.finished.connect(self._on_fit_finished)
        self._fit_thread.start()
    
    def copy_fit_model(self):
        if not self.selected_fnames:
            self.notify.emit("No spectrum selected.")
            return

        spectrum = self._get_selected_spectra()[0]
        if not spectrum.peak_models:
            self.notify.emit("No fit results to copy.")
            return

        self._fitmodel_clipboard = deepcopy(spectrum.save())

    def paste_fit_model(self, apply_all: bool = False):
        if not hasattr(self, "_fitmodel_clipboard"):
            self.notify.emit("No fit model copied.")
            return
        spectra = self._get_active_spectra() if apply_all else self._get_selected_spectra()

        for s in spectra:
            s.reinit()

        self._run_fit_thread(deepcopy(self._fitmodel_clipboard), spectra)

    def save_fit_model(self):
        if not self.selected_fnames:
            self.notify.emit("No spectrum selected.")
            return

        spectrum = self._get_selected_spectra()[0]

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
            QMessageBox.critical(None, "Error", f"Failed to load fit model:\n{e}")
            return

        spectra = self._get_active_spectra() if apply_all else self._get_selected_spectra()
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

        self._fit_thread = ApplyFitModelThread(
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
        self._emit_list_update()  # Refresh list colors after fitting
        
        # Cleanup thread
        if self._fit_thread:
            self._fit_thread.deleteLater()
            self._fit_thread = None

    def stop_fit(self):
        """Stop the currently running fit thread."""
        if self._fit_thread and self._fit_thread.isRunning():
            # Store reference before terminating (terminate may trigger finished signal)
            thread = self._fit_thread
            self._fit_thread = None
            
            thread.terminate()
            thread.wait()
            thread.deleteLater()
            
            self._is_fitting = False
            self.fit_in_progress.emit(False)
            self.notify.emit("Fitting stopped by user.")

    def set_fit_model_builder(self, vm_fit_model_builder):
        self._vm_fit_model_builder = vm_fit_model_builder
 
    def set_peak_shape(self, shape: str):
        """Receive peak shape from View."""
        self._current_peak_shape = shape


    def update_peak_label(self, index, text):
        s = self._get_selected_spectra()[0]
        s.peak_labels[index] = text
        self._emit_selected_spectra()

    def update_peak_model(self, index, model_name):
        s = self._get_selected_spectra()[0]
        pm = s.peak_models[index]

        # Check if this is a decay model or spectroscopy model
        is_decay_model = model_name in ["DecaySingleExp", "DecayBiExp"]
        
        if is_decay_model:
            # Decay models don't have x0/ampli - create fresh and reinitialize
            # Use a dummy x0 value (middle of data range)
            x0_dummy = (s.x[0] + s.x[-1]) / 2
            new_pm = s.create_peak_model(
                index + 1,
                model_name,
                x0=x0_dummy,
                ampli=1.0,  # Dummy value, will be overwritten
                dx0=(20.0, 20.0)
            )
            # Initialize decay parameters properly
            s.peak_models[index] = new_pm
            self._initialize_decay_params(new_pm, s)
        else:
            # Spectroscopy models - preserve x0 and ampli if available
            x0 = pm.param_hints.get("x0", {}).get("value", (s.x[0] + s.x[-1]) / 2)
            ampli = pm.param_hints.get("ampli", {}).get("value", 1.0)
            
            new_pm = s.create_peak_model(
                index + 1,
                model_name,
                x0=x0,
                ampli=ampli,
                dx0=(20.0, 20.0)
            )
            s.peak_models[index] = new_pm

        s.result_fit = None
        self._emit_selected_spectra()


    def update_peak_param(self, index, key, field, value):
        s = self._get_selected_spectra()[0]
        s.peak_models[index].param_hints[key][field] = value
        self._emit_selected_spectra()

    def delete_peak(self, index):
        s = self._get_selected_spectra()[0]
        del s.peak_models[index]
        del s.peak_labels[index]
        self._emit_selected_spectra()

    def update_dragged_peak(self, x: float, y: float):
        """Update peak position during dragging (real-time update).
        
        Args:
            x: New x position (center)
            y: New y value (amplitude/intensity)
        """
        if not self.selected_fnames:
            return

        spectrum = self._get_selected_spectra()[0]
        
        if not spectrum.peak_models:
            return

        # Find the peak model being dragged (closest to new x position)
        # Note: The View already updates the model directly for immediate visual feedback
        # This method is here for any additional processing needed
        # The actual update happens in the View for performance
        pass

    def finalize_peak_drag(self):
        """Finalize peak drag operation - ensure model is synchronized."""
        if not self.selected_fnames:
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

        if not self.selected_fnames:
            return

        selected_spectra = self._get_selected_spectra()
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
        from PySide6.QtWidgets import QMessageBox
        
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save work",
            "",
            "SPECTROview Files (*.spectra)"
        )
        
        if not file_path:
            return
        
        try:
            data_to_save = {'spectrums': self.spectra.save(is_map=False)}
            
            # Save fit results DataFrame (including computed columns)
            if self.df_fit_results is not None and not self.df_fit_results.empty:
                data_to_save['df_fit_results'] = self.df_fit_results.to_dict('records')
            else:
                data_to_save['df_fit_results'] = None
            
            with open(file_path, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            self.notify.emit("Work saved successfully.")
        except Exception as e:
            QMessageBox.critical(None, "Save Error", f"Error saving work:\n{str(e)}")

    def load_work(self, file_path: str):
        """Load previously saved workspace from .spectra file."""
        
        try:
            with open(file_path, 'r') as f:
                load = json.load(f)
            
            # Clear existing data
            self.spectra = MSpectra()
            
            # Load all spectra
            for spectrum_id, spectrum_data in load.get('spectrums', {}).items():
                spectrum = MSpectra.load_from_dict(
                    spectrum_class=MSpectrum,
                    spectrum_data=spectrum_data,
                    is_map=False
                )
                spectrum.preprocess()
                self.spectra.append(spectrum)
            
            # Restore fit results DataFrame (including computed columns)
            if 'df_fit_results' in load and load['df_fit_results'] is not None:
                self.df_fit_results = pd.DataFrame(load['df_fit_results'])

                # Emit signal to update fit results table
                self.fit_results_updated.emit(self.df_fit_results)
            else:
                self.df_fit_results = None
            
            # Update UI
            self._emit_list_update()
            if len(self.spectra) > 0:
                self.selected_fnames = [self.spectra[0].fname]
                self._emit_selected_spectra()
            else:
                self.selected_fnames = []
                self.spectra_selection_changed.emit([])
            
        except Exception as e:
            QMessageBox.critical(None, "Load Error", f"Error loading spectra workspace:\n{str(e)}")

    def clear_workspace(self):
        """Clear all spectra and reset workspace to initial state."""
        # Clear data model
        self.spectra = MSpectra()
        self.selected_fnames = []
        
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
        """Collect best-fit results from active spectra and create DataFrame."""
        active_spectra = self._get_active_spectra()
        if not active_spectra:
            self.notify.emit("No active spectra to collect results from.")
            return
        
        # Copy current fit model for reference
        selected = self._get_selected_spectra()
        if selected:
            spectrum = selected[0]
            if spectrum.peak_models:
                self._fitmodel_clipboard = deepcopy(spectrum.save())
        
        fit_results_list = []
        
        for spectrum in active_spectra:
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
        
        # Build peak_id order mapping to preserve original order
        peak_id_order = {}
        peak_id_index = 0
        for col in other_cols:
            if '_' in col:
                parts = col.split('_', 1)
                if len(parts) > 1:
                    peak_id = parts[1]
                    if peak_id not in peak_id_order:
                        peak_id_order[peak_id] = peak_id_index
                        peak_id_index += 1
        
        # Sort by parameter type (prefix) then by peak identifier (in original order)
        def sort_key(col_name):
            if '_' in col_name:
                parts = col_name.split('_', 1)  # Split on first underscore
                param_type = parts[0]  # e.g., "x0", "ampli", "fwhm"
                peak_id = parts[1] if len(parts) > 1 else ''  # e.g., "p1", "p2"
                # Use priority if defined, otherwise use high number (appears last)
                priority = param_priority.get(param_type, 999)
                # Use original order index for peak_id
                peak_order = peak_id_order.get(peak_id, 999)
                return (priority, param_type, peak_order)
            else:
                return (999, col_name, 999)
        
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
    
    def compute_column_from_expression(self, col_name: str, expression: str):
        """Add a new column to fit results by evaluating a mathematical expression."""
        
        if self.df_fit_results is None or self.df_fit_results.empty:
            QMessageBox.warning(None, "No Fit Results", "No fit results available. Collect results first.")
            return
        
        if not col_name:
            QMessageBox.warning(None, "Missing Column Name", "Please enter a column name.")
            return
        
        if not expression:
            QMessageBox.warning(None, "Missing Expression", "Please enter a mathematical expression.")
            return
        
        # Check if column already exists
        if col_name in self.df_fit_results.columns:
            QMessageBox.warning(
                None, 
                "Duplicate Column Name",
                f"Column '{col_name}' already exists. Please choose a different name."
            )
            return
        
        try:
            # Use pandas eval for safe expression evaluation
            # This handles mathematical operations safely without eval()
            result = self.df_fit_results.eval(expression)
            
            # Check for inf and NaN values (from division by zero, etc.)
            warnings = []
            if pd.isna(result).any():
                nan_count = pd.isna(result).sum()
                warnings.append(
                    f"Expression resulted in {nan_count} NaN value(s). "
                    "This may be due to division by zero or invalid operations."
                )
            
            if np.isinf(result).any():
                inf_count = np.isinf(result).sum()
                warnings.append(
                    f"Expression resulted in {inf_count} infinite value(s). "
                    "This may be due to division by zero."
                )
            
            # Add the computed column to the dataframe
            self.df_fit_results[col_name] = result
            
            # Round to 3 decimals for consistency
            if pd.api.types.is_numeric_dtype(self.df_fit_results[col_name]):
                self.df_fit_results[col_name] = self.df_fit_results[col_name].round(3)
            
            # Emit updated dataframe
            self.fit_results_updated.emit(self.df_fit_results)
            
            # Show success message with warnings if any
            if warnings:
                message = f"Successfully added computed column '{col_name}'.\n\nWarnings:\n" + "\n".join(f"• {w}" for w in warnings)
                QMessageBox.warning(None, "Column Added with Warnings", message)
            else:
                QMessageBox.information(
                    None, 
                    "Success",
                    f"Successfully added computed column '{col_name}'."
                )
            
        except pd.errors.UndefinedVariableError as e:
            # Column name in expression doesn't exist
            QMessageBox.critical(
                None,
                "Invalid Column Name",
                f"Invalid column name in expression.\n\n"
                f"Error: {str(e)}\n\n"
                f"Available columns:\n{', '.join(self.df_fit_results.columns)}\n\n"
                "Note: Use backticks for names with special characters: `x0_LO(M)`"
            )
        except SyntaxError as e:
            # Invalid syntax in expression
            QMessageBox.critical(
                None,
                "Syntax Error",
                f"Invalid expression syntax.\n\n"
                f"Error: {str(e)}\n\n"
                "Examples:\n"
                "• column1 - column2\n"
                "• (col1 + col2) * 2\n\n"
                "Note: Use backticks for names with special characters: `x0_LO(M)`"
            )
        except ZeroDivisionError:
            # Explicit division by zero (though pandas.eval usually handles this)
            QMessageBox.critical(
                None,
                "Division by Zero",
                "Division by zero detected in expression.\n\n"
                "Please check your formula."
            )
        except Exception as e:
            # Catch-all for other errors
            QMessageBox.critical(
                None,
                "Expression Error",
                f"Error evaluating expression:\n\n"
                f"{str(e)}\n\n"
                f"Available columns:\n{', '.join(self.df_fit_results.columns)}\n\n"
                "Note: Use backticks for names with special characters or when column's header contain spaces: `x0_LO(M)`"
            )
    
    def save_fit_results(self):
        """Save fit results to Excel or CSV file."""
        if self.df_fit_results is None or self.df_fit_results.empty:
            self.notify.emit("No fit results to save.")
            return
        
        last_dir = self.settings.get_last_directory()
        file_path, selected_filter = QFileDialog.getSaveFileName(
            None,
            "Save Fit Results",
            str(Path(last_dir) / "fit_results.xlsx"),
            "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                # Determine format from file extension or filter
                ext = Path(file_path).suffix.lower()
                
                if ext == '.csv' or 'CSV' in selected_filter:
                    # Save as CSV with semicolon delimiter
                    self.df_fit_results.to_csv(file_path, index=False, sep=';')
                    self.notify.emit(f"Fit results saved: {Path(file_path).name}")
                else:
                    # Save as Excel (default) using custom function with colored columns
                    if not ext:
                        file_path += '.xlsx'
                    success, message = save_df_to_excel(file_path, self.df_fit_results)
                    if success:
                        self.notify.emit(f"Fit results saved: {Path(file_path).name}")
                    else:
                        QMessageBox.critical(None, "Error", f"Error saving fit results: {message}")
                        return
                
                # Update last_directory setting
                self.settings.set_last_directory(str(Path(file_path).parent))
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Error saving fit results: {e}")
    
    def send_results_to_graphs(self, df_name: str):
        """Send fit results DataFrame to Graphs workspace."""
        if self.df_fit_results is None or self.df_fit_results.empty:
            self.notify.emit("No fit results to send.")
            return
        
        if not df_name:
            self.notify.emit("Please enter a DataFrame name.")
            return
            
        self.send_df_to_graphs.emit(df_name, self.df_fit_results)
        self.notify.emit(f"Sent fit results to Graphs workspace as '{df_name}'.")
    
    def view_stats(self, parent_widget=None):
        """Show statistical fitting results of the selected spectrum."""
        selected_spectra = self._get_selected_spectra()
        
        if not selected_spectra:
            self.notify.emit("No spectrum selected.")
            return
        
        # Show the 'report' of the first selected spectrum
        spectrum = selected_spectra[0]
        fnames = [s.fname for s in selected_spectra]
        title = f"Fitting Report - {fnames}"
        
        # Check if result_fit exists and has the necessary params attribute
        if (hasattr(spectrum, 'result_fit') and 
            spectrum.result_fit is not None and
            hasattr(spectrum.result_fit, 'params') and
            spectrum.result_fit.params is not None):
            try:
                text = fit_report(spectrum.result_fit)
                view_text(parent_widget, title, text)
            except Exception as e:
                self.notify.emit(f"Error generating fit report: {str(e)}")
        else:
            self.notify.emit("No fit results available for the selected spectrum. Please fit the spectrum first.")
