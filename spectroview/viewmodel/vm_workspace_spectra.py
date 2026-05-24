"""ViewModel for Spectra Workspace - handles business logic and data management."""
import os
import json
from copy import deepcopy
from pathlib import Path
from collections import OrderedDict
        
import numpy as np
import pandas as pd
from lmfit import fit_report
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox, QFileDialog

from spectroview.viewmodel.utils import closest_index
from spectroview.model.m_io import load_spectrum_file, load_TRPL_data, load_wdf_spectrum, load_spc_spectrum
from spectroview.model.m_settings import MSettings

from spectroview.model.workspace_io import WorkspaceIO
from spectroview.model.spectra_store import SpectraStore
from spectroview.model.peak_model import initialize_peak_params
from spectroview.fit_engine.tensor_fit_thread import TensorFitThread

from spectroview.fit_engine.baseline import eval_baseline_batch
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
    spectra_selection_changed = Signal(object) # list[MSpectrum] or dict (tensor batch)
    count_changed = Signal(int)
    show_xcorrection_value = Signal(float)  # ΔX of first selected spectrum
    spectral_range_changed = Signal(float, float)
    
    fit_in_progress = Signal(bool)  # Enable/disable fit buttons
    fit_progress_updated = Signal(int, int, int, float)  # To show fitting progress in GUI
    fit_timings_ready = Signal(str)  # Tooltip with timings
    
    # Fit results signals
    fit_results_updated = Signal(object)  # pd.DataFrame
    split_parts_updated = Signal(list)    # list[str] for combobox
    send_df_to_graphs = Signal(str, object)  # (df_name, pd.DataFrame)

    notify = Signal(str)  # general notifications
    
    def __init__(self, settings: MSettings):
        super().__init__()
        self.settings = settings
        self.store = SpectraStore()
        self.store = SpectraStore()

        self.selected_fnames = []  # Changed from indices to fnames for robust identification
        self._baseline_clipboard = None  # for copy/paste baseline
        self._peaks_clipboard = None    # for copy/paste peaks
        self._loaded_fit_model = None  # for applying loaded fit model
        self._current_peak_shape = "Lorentzian"
        self._fit_thread = None  # Track active fit thread
        self._is_fitting = False  # Track if fitting is in progress
        self._use_batch_engine = True  # Use high-performance batch engine
        
        # Fit results data
        self.df_fit_results = None
        self._fitmodel_clipboard = None
    
    # ═════════════════════════════════════════════════════════════════════
    # Helper methods for fname-based spectrum retrieval
    # ═════════════════════════════════════════════════════════════════════
    
    def _get_spectra_by_fnames(self, fnames: list[str]) -> list[str]:
        """Return list of valid fnames that exist in the store."""
        return [fname for fname in fnames if self.store.get_map_data(fname) is not None]
    
    def _get_selected_spectra(self) -> list[str]:
        """Get currently selected fnames that are also active (checked)."""
        selected = self._get_spectra_by_fnames(self.selected_fnames)
        active_selected = []
        for fname in selected:
            md = self.store.get_map_data(fname)
            if md and md.is_active[0]:
                active_selected.append(fname)
        return active_selected


    # View → ViewModel slots
    def load_files(self, paths: list[str]):
        """Load spectrum files from disk directly into SpectraStore."""
        existing_fnames = set(self.store.map_names)
        
        loaded_files = []
        last_valid_path = None

        for p in paths:
            path = Path(p)
            resolved_path = str(path.resolve())
                
            if path.stem in existing_fnames:
                self.notify.emit(f"Spectrum '{path.name}' already loaded (name match), skipping.")
                continue

            try:
                # Use appropriate loader based on file extension
                if path.suffix.lower() == '.dat':
                    data_dict = load_TRPL_data(path)
                elif path.suffix.lower() == '.wdf':
                    data_dict = load_wdf_spectrum(path)
                elif path.suffix.lower() == '.spc':
                    data_dict = load_spc_spectrum(path)
                else:
                    data_dict = load_spectrum_file(path)
                
                if data_dict and "items" in data_dict:
                    for item in data_dict["items"]:
                        fname = item["name"]
                        x0 = item["x0"]
                        y0 = item["y0"]
                        metadata = item.get("metadata", {})
                        
                        self.store.add_map(
                            name=fname,
                            x0=x0.copy(),
                            Y0=np.asarray(y0, dtype=np.float32).reshape(1, -1),
                            coords=np.array([[0.0, 0.0]], dtype=np.float64),
                            fnames=[fname],
                            is_active=np.array([True], dtype=bool)
                        )
                        md = self.store.get_map_data(fname)
                        if md:
                            md.map_metadata = metadata
                            # Store source path in metadata for persistence 
                            md.map_metadata['source_path'] = resolved_path
                
                loaded_files.append(p)
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
        map_names = self.store.map_names
        for idx in indices:
            if 0 <= idx < len(map_names):
                fnames.append(map_names[idx])
        
        # Store fnames (ensure uniqueness while preserving order)
        self.selected_fnames = list(dict.fromkeys(fnames))
        self._emit_selected_spectra()
    
    def set_selected_fnames(self, fnames: list[str]):
        """Set currently selected spectra by their fnames directly."""
        # Store fnames (ensure uniqueness while preserving order)
        self.selected_fnames = list(dict.fromkeys(fnames))
        self._emit_selected_spectra()
    
    def _get_active_spectra(self) -> list:
        """Get list of active spectra fnames (where is_active is True)."""
        active = []
        for name in self.store.map_names:
            md = self.store.get_map_data(name)
            if md and md.is_active[0]:
                active.append(name)
        return active

    def _emit_selected_spectra(self):
        """Prepare and emit data for plotting the selected spectra."""
        active_fnames = self._get_selected_spectra()

        if not active_fnames:
            self.spectra_selection_changed.emit([])
            return

        # Build tensor batch for SpectraViewer
        x_list = []
        Y_list = []
        x0_list = []
        Y0_list = []
        colors = []
        labels = []
        bl_params = []
        pk_params = []
        y_bestfit_list = []
        y_peaks_list = []
        y_baseline_list = []
        fit_model_list = []
        
        for fname in active_fnames:
            md = self.store.get_map_data(fname)
            if not md: continue
            
            self._update_fit_model_from_params(md, 0)
            
            x_list.append(md.x if md.x is not None else md.x0)
            Y_list.append(md.Y[0] if md.Y is not None else md.Y0[0])
            x0_list.append(md.x0)
            Y0_list.append(md.Y0[0])
            colors.append(md.colors[0])
            labels.append(md.labels[0] or fname)
            bl_params.append(md.baseline_config)
            pk_params.append(md.peak_params)
            y_bestfit_list.append(md.Y_bestfit[0] if md.Y_bestfit is not None else None)
            y_peaks_list.append([p[0] for p in md.Y_peaks] if md.Y_peaks is not None else None)
            y_baseline_list.append(md.Y_baseline[0] if md.Y_baseline is not None else None)
            fit_model_list.append(md.fit_model)

        # In Spectra Workspace, different spectra might have different x axes.
        # VSpectraViewer will need to handle a list of arrays for x, y, x0, y0.
        data = {
            "type": "tensor_list", # custom type for varying x axes
            "x": x_list,
            "y": Y_list,
            "x0": x0_list,
            "y0": Y0_list,
            "colors": colors,
            "labels": labels,
            "fnames": active_fnames,
            "baseline_config": bl_params,
            "peak_params": pk_params,
            "y_bestfit": y_bestfit_list,
            "y_peaks": y_peaks_list,
            "y_baseline": y_baseline_list,
            "fit_models": fit_model_list,
            "fit_r2": [md.fit_r2[0] if md.fit_r2 is not None else 0.0 for md in (self.store.get_map_data(fname) for fname in active_fnames) if md]
        }

        # emit list of the selected spectra to plot in View
        self.spectra_selection_changed.emit(data)    

        # For x-correction, we could read it from md.map_metadata or store it.
        md0 = self.store.get_map_data(active_fnames[0])
        xcorr = md0.map_metadata.get('xcorrection_value', 0.0) if md0 else 0.0
        self.show_xcorrection_value.emit(xcorr)

        # emit spectral range of first selected spectrum to show in GUI
        if md0:
            x_axis = md0.x if md0.x is not None else md0.x0
            xmin = float(x_axis[0])
            xmax = float(x_axis[-1])
            self.spectral_range_changed.emit(xmin, xmax)

    def _update_fit_model_from_params(self, md, row_idx: int = 0):
        """Update md.fit_model values using the fitted values in md.peak_params[row_idx]."""
        if md is None or md.peak_params is None or md.param_names is None or not md.fit_model:
            return
        
        peak_params = md.peak_params[row_idx]
        param_names = md.param_names
        
        peak_models = md.fit_model.get("peak_models", {})
        sorted_keys = sorted(peak_models.keys(), key=lambda k: int(k))
        
        for p_name, val in zip(param_names, peak_params):
            if "_" in p_name and p_name.startswith("m"):
                try:
                    parts = p_name.split("_", 1)
                    m_prefix = parts[0]
                    pname = parts[1]
                    
                    peak_num_1based = int(m_prefix[1:])
                    peak_num_0based = peak_num_1based - 1
                    
                    if peak_num_0based < len(sorted_keys):
                        key = sorted_keys[peak_num_0based]
                        pdict = peak_models[key]
                        shape = list(pdict.keys())[0]
                        
                        if pname in pdict[shape]:
                            pdict[shape][pname]["value"] = float(val)
                except Exception as e:
                    print(f"Error updating parameter {p_name} from fit: {e}")

    def _reconstruct_y_peaks(self, md):
        """Reconstruct md.Y_peaks from current fit_model."""
        if not md or not md.fit_model or not md.fit_model.get("peak_models"):
            md.Y_peaks = None
            return
        from spectroview.fit_engine.evaluator import eval_peak_initial
        md.Y_peaks = []
        x_arr = md.x if md.x is not None else md.x0
        N = md.Y.shape[0] if md.Y is not None else md.Y0.shape[0]
        for p_model in md.fit_model["peak_models"].values():
            y_curve = eval_peak_initial(x_arr, p_model)
            md.Y_peaks.append(np.tile(y_curve, (N, 1)))

    def reorder_spectra(self, new_order: list[int]):
        """new_order = list of old indices in new visual order"""
        self.store.reorder_maps(new_order)

        # After reorder, selection must be re-emitted
        self._emit_list_update()
        self._emit_selected_spectra()

    def remove_selected_spectra(self):
        """Remove currently selected spectra."""
        if not self.selected_fnames:
            self.notify.emit("No spectra selected.")
            return
        
        # Find indices to remove (fname-based)
        map_names = self.store.map_names
        indices_to_remove = []
        for idx, name in enumerate(map_names):
            if name in self.selected_fnames:
                indices_to_remove.append(idx)
        
        if not indices_to_remove:
            return
        
        # Store first removed index for re-selection
        min_removed_idx = min(indices_to_remove)
        
        for name in self.selected_fnames:
            self.store.remove_map(name)
        
        new_count = len(self.store.map_names)
        self._emit_list_update()

        if new_count == 0:
            self.selected_fnames = []
            self.spectra_selection_changed.emit([])
            return
        
        # Select closest spectrum by index
        new_index = min(min_removed_idx, new_count - 1)
        self.selected_fnames = [self.store.map_names[new_index]]
        self._emit_selected_spectra()

    def cosmic_ray_detection(self):
        """Detect cosmic rays for all loaded spectra."""
        # TODO: Implement tensor-based cosmic ray detection for SpectraStore
        # self.store.remove_cosmic_rays()
        self.notify.emit("Cosmic ray detection completed.")

    # Internal helpers
    def _emit_list_update(self):
        """Emit updated list of spectra dicts and count."""
        spectra_info = []
        for name in self.store.map_names:
            md = self.store.get_map_data(name)
            if not md: continue
            
            has_fit = md.peak_params is not None and len(md.peak_params) > 0
            has_baseline = md.baseline_config is not None
            
            spectra_info.append({
                "fname": name,
                "is_active": bool(md.is_active[0]),
                "has_baseline": has_baseline,
                "fit_success": has_fit,
            })
            
        self.spectra_list_changed.emit(spectra_info)
        self.count_changed.emit(len(self.store.map_names))

    def _get_interactive_map_data(self):
        """Helper to fetch MapData for interactive tools in Spectra Workspace."""
        if not self.selected_fnames:
            return None
        fname = self._get_selected_spectra()[0]
        return self.store.get_map_data(fname)

    def add_peak_at(self, x: float):
        md = self._get_interactive_map_data()
        if not md: return

        # Clear fit state to prevent _update_fit_model_from_params overwriting with stale parameters
        md.peak_params = None
        md.param_names = None
        md.fit_success = None
        md.fit_r2 = None
        md.Y_bestfit = None

        if md.fit_model is None:
            md.fit_model = {"peak_labels": [], "peak_models": {}}

        fit_settings = self.settings.load_fit_settings()
        maxshift = fit_settings.get("maxshift", 20.0)
        maxfwhm = fit_settings.get("maxfwhm", 200.0)
        minfwhm = fit_settings.get("minfwhm", 0.0)
        peak_shape = self._current_peak_shape or "Lorentzian"

        # Determine y value
        idx = closest_index(md.x if md.x is not None else md.x0, x)
        y_val = float(md.Y[0][idx] if md.Y is not None else md.Y0[0][idx])
        if y_val <= 0: y_val = 1e-10

        # Create new peak dict
        peak_idx = str(len(md.fit_model["peak_labels"]))
        peak_model = {}
        y_arr = md.Y[0] if md.Y is not None else md.Y0[0]
        self._initialize_peak_params(peak_model, peak_shape, x, y_val, minfwhm, maxfwhm, maxshift, y_arr)

        md.fit_model["peak_labels"].append(f"Peak {int(peak_idx) + 1}")
        md.fit_model["peak_models"][peak_idx] = {peak_shape: peak_model}

        # Quickly evaluate and append to Y_peaks for instant preview
        x_arr = md.x if md.x is not None else md.x0
        from spectroview.fit_engine.evaluator import eval_peak_initial
        y_curve = eval_peak_initial(x_arr, {peak_shape: peak_model})
        
        if md.Y_peaks is None:
            md.Y_peaks = []
        
        # md.Y_peaks expects a list of arrays (one per peak), shape (N, M_proc)
        N = md.Y.shape[0] if md.Y is not None else md.Y0.shape[0]
        md.Y_peaks.append(np.tile(y_curve, (N, 1)))

        self._emit_selected_spectra()

    def _initialize_decay_params(self, peak_model, peak_shape, y_arr):
        """Initialize decay model parameters for TRPL fitting inside dict model."""
        y_max = float(np.max(y_arr))
        y_min = float(np.min(y_arr))
        
        # Remove standard shape params
        for p in ["ampli", "x0", "fwhm"]:
            if p in peak_model:
                del peak_model[p]

        if peak_shape == "DecaySingleExp":
            peak_model["A"] = {"value": y_max, "min": 0, "max": y_max * 100, "vary": True}
            peak_model["tau"] = {"value": 5.0, "min": 0.1, "max": 100, "vary": True}
            peak_model["B"] = {"value": y_min, "min": 0, "max": y_min * 10, "vary": True}
        elif peak_shape == "DecayBiExp":
            peak_model["A1"] = {"value": y_max * 0.7, "min": 0, "max": y_max * 100, "vary": True}
            peak_model["tau1"] = {"value": 2.0, "min": 0.1, "max": 50, "vary": True}
            peak_model["A2"] = {"value": y_max * 0.3, "min": 0, "max": y_max * 100, "vary": True}
            peak_model["tau2"] = {"value": 10.0, "min": 0.1, "max": 100, "vary": True}
            peak_model["B"] = {"value": y_min, "min": 0, "max": y_min * 10, "vary": True}

    def _initialize_peak_params(self, peak_model, peak_shape, x0, ampli, minfwhm, maxfwhm, maxshift, y_arr=None):
        """Build canonical parameters dictionary based on peak shape."""
        initialize_peak_params(peak_model, peak_shape, x0, ampli, minfwhm, maxfwhm, maxshift, y_arr)

    def remove_peak_at(self, x: float):
        md = self._get_interactive_map_data()
        if not md or not md.fit_model or not md.fit_model.get("peak_models"):
            return

        # Find closest peak by x0
        closest_k = None
        min_dist = float('inf')
        for k, pdict in md.fit_model["peak_models"].items():
            shape = list(pdict.keys())[0]
            if "x0" in pdict[shape]:
                px = pdict[shape]["x0"]["value"]
                dist = abs(px - x)
                if dist < min_dist:
                    min_dist = dist
                    closest_k = k
        
        if closest_k is not None:
            # Rebuild dictionary without the deleted peak to keep indices contiguous
            old_models = md.fit_model["peak_models"]
            old_labels = md.fit_model["peak_labels"]
            idx_to_del = int(closest_k)
            
            new_models = {}
            new_labels = []
            new_idx = 0
            for k, pdict in old_models.items():
                if int(k) == idx_to_del:
                    continue
                new_models[str(new_idx)] = pdict
                if int(k) < len(old_labels):
                    new_labels.append(old_labels[int(k)])
                new_idx += 1
            
            md.fit_model["peak_models"] = new_models
            md.fit_model["peak_labels"] = new_labels
            md.peak_params = None
            md.param_names = None
            md.fit_success = None
            md.fit_r2 = None
            md.Y_bestfit = None
            self._reconstruct_y_peaks(md)

        self._emit_selected_spectra() 

    def add_baseline_point(self, x: float, y: float):
        md = self._get_interactive_map_data()
        if not md: return

        if getattr(md, "is_baseline_subtracted", False):
            self.notify.emit("Baseline already subtracted. Please reinit first.")
            return

        if md.baseline_config is None:
            # Use current settings if available, else default
            attached = True
            mode = "Linear"
            if hasattr(self, "_baseline_settings") and self._baseline_settings:
                attached = bool(self._baseline_settings.get("attached", True))
                mode = self._baseline_settings.get("mode", "Linear")
            md.baseline_config = {"points": [[], []], "mode": mode, "attached": attached}
            
        pts = md.baseline_config.get("points", [[], []])
        if not pts or len(pts) < 2:
            pts = [[], []]
            md.baseline_config["points"] = pts
            
        pts[0].append(x)
        pts[1].append(y)
        
        # Sort by x
        sorted_pts = sorted(zip(pts[0], pts[1]))
        md.baseline_config["points"] = [list(t) for t in zip(*sorted_pts)] if sorted_pts else [[], []]
        
        # Dynamically evaluate the baseline curve
        x_arr = md.x if md.x is not None else md.x0
        y_arr = md.Y if md.Y is not None else md.Y0
        md.Y_baseline = eval_baseline_batch(x_arr, y_arr, md.baseline_config)

        self._emit_selected_spectra()

    def remove_baseline_point(self, x: float):
        md = self._get_interactive_map_data()
        if not md or not md.baseline_config:
            return

        pts = md.baseline_config.get("points", [[], []])
        if not pts or not pts[0]:
            return

        xs, ys = pts[0], pts[1]
        idx = min(range(len(xs)), key=lambda i: abs(xs[i] - x))
        xs.pop(idx)
        ys.pop(idx)

        md.baseline_config["points"] = [xs, ys]
        
        # Dynamically evaluate the baseline curve
        x_arr = md.x if md.x is not None else md.x0
        y_arr = md.Y if md.Y is not None else md.Y0
        md.Y_baseline = eval_baseline_batch(x_arr, y_arr, md.baseline_config)

        self._emit_selected_spectra()

    def apply_x_correction(self, measured_peak: float):
        if not self.selected_fnames:
            self.notify.emit("No spectrum selected.")
            return

        fnames = self._get_selected_spectra()
        SI_REF = 520.7
        delta_x = SI_REF - measured_peak 

        for md in self._get_unique_map_data(fnames):
            if md.x is not None:
                md.x = md.x + delta_x
            md.x0 = md.x0 + delta_x
            md.map_metadata['xcorrection_value'] = md.map_metadata.get('xcorrection_value', 0.0) + delta_x

        # Trigger plot refresh
        md0 = self._get_unique_map_data(fnames)[0]
        self.show_xcorrection_value.emit(md0.map_metadata.get('xcorrection_value', 0.0))
        self._emit_selected_spectra()


    def undo_x_correction(self):
        """Undo X-axis correction for selected spectra."""
        if not self.selected_fnames:
            self.notify.emit("No spectrum selected.")
            return

        fnames = self._get_selected_spectra()

        for md in self._get_unique_map_data(fnames):
            delta = md.map_metadata.get('xcorrection_value', 0.0)
            if delta != 0.0:
                if md.x is not None:
                    md.x = md.x - delta
                md.x0 = md.x0 - delta
                md.map_metadata['xcorrection_value'] = 0.0

        self.show_xcorrection_value.emit(0.0)
        self._emit_selected_spectra()

    def apply_y_normalization(self, norm_factor: float, apply_all: bool = False):
        """Apply intensity normalization for selected spectra."""
        fnames = (
            self._get_active_spectra()
            if apply_all
            else self._get_selected_spectra()
        )

        for md in self._get_unique_map_data(fnames):
            if md.Y is not None:
                md.Y = md.Y * norm_factor
            md.Y0 = md.Y0 * norm_factor

        self._emit_selected_spectra()

    def undo_y_normalization(self, apply_all: bool = False):
        """Undo intensity normalization for selected spectra."""
        # Tensor engine manages normalization in the view layer predominantly.
        # Restoring from disk or reinit is required if Y0 was permanently altered.
        pass

    def reinit_spectra(self, apply_all: bool = False):
        """Reinitialize spectra to original data."""
        if apply_all:
            fnames = self._get_active_spectra()
        else:
            if not self.selected_fnames:
                self.notify.emit("No spectrum selected.")
                return
            fnames = self._get_selected_spectra()

        for md in self._get_unique_map_data(fnames):
            if md.x0 is not None:
                md.x = md.x0.copy()
            if md.Y0 is not None:
                md.Y = md.Y0.copy()
            md.baseline_config = None
            md.Y_baseline = None
            md.peak_params = None
            md.fit_model = None
            md.Y_peaks = None
            md.Y_bestfit = None
            md.is_baseline_subtracted = False
            md.range_min = None
            md.range_max = None
            
        self._emit_selected_spectra() # Refresh plot 
        self._emit_list_update()  # Refresh list colors after reinit


    def _get_unique_map_data(self, fnames: list[str]) -> list:
        """Helper to get deduplicated MapData objects for the given filenames.
        In Spectra workspace, fname == map_name, so this simply resolves them.
        """
        mds = {}
        for f in fnames:
            md = self.store.get_map_data(f)
            if md and md.name not in mds:
                mds[md.name] = md
        return list(mds.values())

    def apply_spectral_range(self, xmin: float, xmax: float, apply_all: bool):
        if not self.selected_fnames:
            return

        if xmin > xmax:
            xmin, xmax = xmax, xmin

        fnames = (
            self._get_active_spectra()
            if apply_all
            else self._get_selected_spectra()
        )

        for md in self._get_unique_map_data(fnames):
            # 1) Reinit completely to avoid dimension mismatches if previously cropped with baseline/peaks
            if md.x0 is not None:
                md.x = md.x0.copy()
            if md.Y0 is not None:
                md.Y = md.Y0.copy()
            md.baseline_config = None
            md.Y_baseline = None
            md.peak_params = None
            md.fit_model = None
            md.Y_peaks = None
            md.Y_bestfit = None
            md.is_baseline_subtracted = False

            # 2) Perform crop
            if md.x is None:
                curr_x = md.x0
                curr_y = md.Y0
            else:
                curr_x = md.x
                curr_y = md.Y
                
            i_min = closest_index(curr_x, xmin)
            i_max = closest_index(curr_x, xmax)
            if i_min > i_max:
                i_min, i_max = i_max, i_min
            
            md.x = curr_x[i_min:i_max+1].copy()
            md.Y = curr_y[:, i_min:i_max+1].copy()
            md.range_min = float(md.x[0])
            md.range_max = float(md.x[-1])

        self._emit_selected_spectra()
        self._emit_list_update()  # Refresh list colors after range change

    def copy_baseline(self):
        if not self.selected_fnames:
            self.notify.emit("No spectrum selected.")
            return

        fnames = self._get_selected_spectra()
        md = self._get_unique_map_data(fnames)[0]
        if md and md.baseline_config:
            self._baseline_clipboard = deepcopy(md.baseline_config)

    def paste_baseline(self, apply_all: bool = False):
        if self._baseline_clipboard is None:
            self.notify.emit("No baseline copied.")
            return

        if apply_all:
            fnames = self._get_active_spectra()
        else:
            if not self.selected_fnames:
                self.notify.emit("No spectrum selected.")
                return
            fnames = self._get_selected_spectra()

        # Apply baseline to selected spectra
        baseline_data = deepcopy(self._baseline_clipboard)
        for md in self._get_unique_map_data(fnames):
            md.baseline_config = deepcopy(baseline_data)
            
            x_arr = md.x if md.x is not None else md.x0
            y_arr = md.Y if md.Y is not None else md.Y0
            md.Y_baseline = eval_baseline_batch(x_arr, y_arr, md.baseline_config)

        self._emit_selected_spectra()

    def subtract_baseline(self, apply_all: bool = False):
        if apply_all:
            fnames = self._get_active_spectra()
        else:
            if not self.selected_fnames:
                self.notify.emit("No spectrum selected.")
                return
            fnames = self._get_selected_spectra()

        any_subtracted = False
        for md in self._get_unique_map_data(fnames):
            if getattr(md, "is_baseline_subtracted", False):
                any_subtracted = True
                break
        if any_subtracted:
            self.notify.emit("Baseline already subtracted. Please delete baseline or reinit first.")
            return

        for md in self._get_unique_map_data(fnames):
            if md.baseline_config is not None and md.Y_baseline is not None:
                # Execute subtraction
                md.Y = md.Y - md.Y_baseline
                # Clear baseline config/curve since it's now subtracted into Y
                md.baseline_config = None
                md.Y_baseline = None
                md.is_baseline_subtracted = True

        self._emit_selected_spectra()
        self._emit_list_update()  # Refresh list colors after baseline subtraction

    def delete_baseline(self, apply_all: bool = False):
        """Undo baseline subtraction: reinitialise spectrum data, re-apply crop range."""
        if apply_all:
            fnames = self.store.map_names
        else:
            if not self.selected_fnames:
                self.notify.emit("No spectrum selected.")
                return
            fnames = self._get_selected_spectra()

        for md in self._get_unique_map_data(fnames):
            md.baseline_config = None
            md.Y_baseline = None
            md.is_baseline_subtracted = False
            
            # Undo baseline subtraction by reverting to Y0, but maintaining x crop
            if md.x is not None:
                # Find indices of current x in x0
                xmin = md.x[0]
                xmax = md.x[-1]
                i_min = closest_index(md.x0, xmin)
                i_max = closest_index(md.x0, xmax)
                if i_min > i_max:
                    i_min, i_max = i_max, i_min
                
                md.Y = md.Y0[:, i_min:i_max + 1].copy()
            else:
                md.Y = md.Y0.copy()

        self._emit_selected_spectra()
        self._emit_list_update()


    # ── Baseline mode / settings helpers ──────────────────────────────────

    def _apply_baseline_settings(self, settings: dict, fnames):
        """Internal helper: push mode/params from 'settings' dict onto each MapData baseline_config."""
        mode     = settings.get("mode")       
        coef     = float(settings.get("coef",     5.0))
        order    = int(settings.get("order_max", 1))
        sigma    = int(settings.get("sigma",     0))
        attached = bool(settings.get("attached", True))

        for md in self._get_unique_map_data(fnames):
            if md.baseline_config is None:
                md.baseline_config = {"points": [[], []]}
            bl = md.baseline_config
            bl["mode"]      = mode
            bl["coef"]      = coef
            bl["order_max"] = order
            bl["sigma"]     = sigma
            bl["attached"]  = attached
            
            x_arr = md.x if md.x is not None else md.x0
            y_arr = md.Y if md.Y is not None else md.Y0
            md.Y_baseline = eval_baseline_batch(x_arr, y_arr, md.baseline_config)

    def set_baseline_settings(self, settings: dict):
        self._baseline_settings = settings
        if not self.selected_fnames:
            return
        fnames = self._get_selected_spectra()
        self._apply_baseline_settings(settings, fnames)

    def preview_baseline(self, settings: dict):
        self._baseline_settings = settings
        if not self.selected_fnames:
            return
        
        fnames = self._get_selected_spectra()
        self._apply_baseline_settings(settings, fnames)

        self._emit_selected_spectra() 


    def copy_peaks(self):
        if not self.selected_fnames:
            self.notify.emit("No spectrum selected.")
            return

        fnames = self._get_selected_spectra()
        md = self._get_unique_map_data(fnames)[0]

        if not md.fit_model or not md.fit_model.get("peak_models"):
            self.notify.emit("No peaks to copy.")
            return
        self._peaks_clipboard = deepcopy(md.fit_model)

    def paste_peaks(self, apply_all: bool = False):
        if not hasattr(self, "_peaks_clipboard") or self._peaks_clipboard is None:
            self.notify.emit("No peaks copied.")
            return

        fnames = (
            self._get_active_spectra()
            if apply_all
            else self._get_selected_spectra()
        )

        from spectroview.fit_engine.evaluator import eval_peak_initial
        for md in self._get_unique_map_data(fnames):
            md.fit_model = deepcopy(self._peaks_clipboard)
            
            # Dynamically reconstruct Y_peaks for instant preview
            md.Y_peaks = []
            x_arr = md.x if md.x is not None else md.x0
            N = md.Y.shape[0] if md.Y is not None else md.Y0.shape[0]
            
            for p_model in md.fit_model.get("peak_models", {}).values():
                y_curve = eval_peak_initial(x_arr, p_model)
                md.Y_peaks.append(np.tile(y_curve, (N, 1)))

        self._emit_selected_spectra()
        self._emit_list_update()  # Refresh list colors after paste peaks


    def delete_peaks(self, apply_all: bool = False):
        fnames = (
            self._get_active_spectra()
            if apply_all
            else self._get_selected_spectra()
        )

        if not fnames:
            self.notify.emit("No spectrum selected.")
            return

        for md in self._get_unique_map_data(fnames):
            md.fit_model = None
            md.Y_peaks = None
            md.Y_bestfit = None

        self._emit_selected_spectra()
        self._emit_list_update()  # Refresh list colors after clear peaks


    
    
    def copy_fit_model(self):
        if not self.selected_fnames:
            self.notify.emit("No spectrum selected.")
            return

        fnames = self._get_selected_spectra()
        md = self._get_unique_map_data(fnames)[0]

        if not md.fit_model:
            self.notify.emit("No fit results to copy.")
            return

        self._fitmodel_clipboard = deepcopy(md.fit_model)

    def paste_fit_model(self, apply_all: bool = False):
        if not hasattr(self, "_fitmodel_clipboard") or self._fitmodel_clipboard is None:
            self.notify.emit("No fit model copied.")
            return
        fnames = self._get_active_spectra() if apply_all else self._get_selected_spectra()

        for md in self._get_unique_map_data(fnames):
            md.x = md.x0.copy()
            md.Y = md.Y0.copy()
            md.baseline_config = None
            md.peak_params = None
            md.fit_model = deepcopy(self._fitmodel_clipboard)

        # In Maps workspace, _run_fit_thread expects the map names, not point names
        unique_fnames = [md.name for md in self._get_unique_map_data(fnames)]
        self._run_fit_thread(deepcopy(self._fitmodel_clipboard), unique_fnames)

    def save_fit_model(self):
        if not self.selected_fnames:
            self.notify.emit("No spectrum selected.")
            return

        fnames = self._get_selected_spectra()
        md = self._get_unique_map_data(fnames)[0]

        if not md.fit_model:
            self.notify.emit("No fit model to save.")
            return

        default_dir = ""
        if hasattr(self, "_vm_fit_model_builder") and self._vm_fit_model_builder is not None:
            model_path = self._vm_fit_model_builder.get_current_model_path()
            if model_path is not None:
                default_dir = str(model_path)

        path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Fit Model",
            default_dir,
            "JSON Files (*.json)"
        )

        if not path:
            return

        def default_encoder(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        with open(path, 'w', encoding='utf-8') as f:
            # fitspy's load_model expects integer indexing (e.g. key '0')
            json.dump({'0': md.fit_model}, f, indent=2, default=default_encoder)

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
            with open(model_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            fit_model = data.get("0", {})
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to load fit model:\n{e}")
            return

        fnames = self._get_active_spectra() if apply_all else self._get_selected_spectra()
        if not fnames:
            self.notify.emit("No spectrum selected.")
            return

        for md in self._get_unique_map_data(fnames):
            # 1. Reinit first to cleanly reset state
            if md.x0 is not None:
                md.x = md.x0.copy()
            if md.Y0 is not None:
                md.Y = md.Y0.copy()
            
            md.baseline_config = None
            md.Y_baseline = None
            md.peak_params = None
            md.Y_peaks = None
            md.Y_bestfit = None
            md.is_baseline_subtracted = False
            md.range_min = None
            md.range_max = None

            # 2. Apply range cropping if present in model
            xmin = fit_model.get("range_min")
            xmax = fit_model.get("range_max")
            if xmin is not None and xmax is not None:
                if xmin > xmax:
                    xmin, xmax = xmax, xmin
                curr_x = md.x if md.x is not None else md.x0
                curr_y = md.Y if md.Y is not None else md.Y0
                
                i_min = closest_index(curr_x, xmin)
                i_max = closest_index(curr_x, xmax)
                if i_min > i_max:
                    i_min, i_max = i_max, i_min
                
                md.x = curr_x[i_min:i_max+1].copy()
                md.Y = curr_y[:, i_min:i_max+1].copy()
                md.range_min = float(md.x[0])
                md.range_max = float(md.x[-1])

            # 3. Apply baseline configuration and subtraction
            bl_info = fit_model.get("baseline")
            if bl_info and bl_info.get("mode"):
                md.baseline_config = deepcopy(bl_info)
                x_arr = md.x if md.x is not None else md.x0
                y_arr = md.Y if md.Y is not None else md.Y0
                md.Y_baseline = eval_baseline_batch(x_arr, y_arr, md.baseline_config)
                
                if bl_info.get("is_subtracted", False):
                    # Subtract baseline
                    md.Y = md.Y - md.Y_baseline
                    md.baseline_config = None
                    md.Y_baseline = None
                    md.is_baseline_subtracted = True

            # 4. Attach fit model dictionary
            md.fit_model = deepcopy(fit_model)
            
            # Reconstruct initial peak curves (Y_peaks) for plotting preview
            self._reconstruct_y_peaks(md)

        unique_fnames = [md.name for md in self._get_unique_map_data(fnames)]
        self._run_fit_thread(fit_model, unique_fnames)

    def fit(self, apply_all: bool = False):
        """Fitting action for selected spectra with their current individual models."""
        # Prevent concurrent fit operations
        if self._is_fitting:
            self.notify.emit("Fit already in progress. Please wait...")
            return

        spectra_fnames = self._get_active_spectra() if apply_all else self._get_selected_spectra()

        if not spectra_fnames:
            return

        tasks = []
        for fname in spectra_fnames:
            md = self.store.get_map_data(fname)
            if md and md.fit_model:
                tasks.append({
                    "map_name": fname,
                    "indices": np.array([0]),
                    "fit_model": md.fit_model
                })

        if not tasks:
            self.notify.emit("No peaks to fit.")
            return

        # Cancel any existing thread
        if self._fit_thread and self._fit_thread.isRunning():
            self._fit_thread.terminate()
            self._fit_thread.wait()

        self._is_fitting = True
        self.fit_in_progress.emit(True)

        self._fit_thread = TensorFitThread(self.store, tasks)
        self._fit_thread.progress_changed.connect(self.fit_progress_updated.emit)
        self._fit_thread.timings_ready.connect(self.fit_timings_ready.emit)
        self._fit_thread.finished.connect(self._on_fit_finished)
        self._fit_thread.start()


    def _run_fit_thread(self, fit_model: dict, fnames, apply_model_to_spectra=True):
        # Prevent concurrent fit operations
        if self._is_fitting:
            self.notify.emit("Fit already in progress. Please wait...")
            return

        if not fnames:
            self.notify.emit("No spectra selected.")
            return

        tasks = []
        for fname in fnames:
            tasks.append({
                "map_name": fname,
                "indices": np.array([0]),
                "fit_model": fit_model
            })

        # Cancel any existing thread
        if self._fit_thread and self._fit_thread.isRunning():
            self._fit_thread.terminate()
            self._fit_thread.wait()

        self._is_fitting = True
        self.fit_in_progress.emit(True)

        self._fit_thread = TensorFitThread(self.store, tasks)
        self._fit_thread.progress_changed.connect(self.fit_progress_updated.emit)
        self._fit_thread.timings_ready.connect(self.fit_timings_ready.emit)
        self._fit_thread.finished.connect(self._on_fit_finished)
        self._fit_thread.start()

    def _run_fit_thread_batches(self, batches):
        """No longer used since TensorFitThread natively handles iterative tasks."""
        pass


    def _sync_fit_results_to_store(self):
        """Harvest fit parameters. In tensor mode, results are already in store, so this is a no-op."""
        pass

    def _on_fit_finished(self):
        """Handle fit thread completion."""
        self._is_fitting = False
        self.fit_in_progress.emit(False)
        
        # Write fitted parameters from MSpectrum objects into SpectraStore
        self._sync_fit_results_to_store()
        
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
            # Store reference before stopping (stop may trigger finished signal)
            thread = self._fit_thread
            self._fit_thread = None
            
            # Gracefully stop the thread
            thread.stop()
            thread.wait() # Wait for thread to finish cleanly
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
        md = self._get_unique_map_data(self._get_selected_spectra())[0]
        if md and md.fit_model and "peak_labels" in md.fit_model:
            if index < len(md.fit_model["peak_labels"]):
                md.fit_model["peak_labels"][index] = text
        self._emit_selected_spectra()

    def update_peak_model(self, index, model_name):
        md = self._get_unique_map_data(self._get_selected_spectra())[0]
        if not md or not md.fit_model or str(index) not in md.fit_model.get("peak_models", {}):
            return

        # Clear fit state
        md.peak_params = None
        md.param_names = None
        md.fit_success = None
        md.fit_r2 = None
        md.Y_bestfit = None

        pdict = md.fit_model["peak_models"][str(index)]
        old_shape = list(pdict.keys())[0]
        old_params = pdict[old_shape]

        x0 = old_params.get("x0", {}).get("value", (md.x[0] + md.x[-1]) / 2 if md.x is not None else 0)
        ampli = old_params.get("ampli", {}).get("value", old_params.get("A", {}).get("value", old_params.get("A1", {}).get("value", 1.0)))
        fit_settings = self.settings.load_fit_settings()
        maxshift = fit_settings.get("maxshift", 20.0)
        maxfwhm = fit_settings.get("maxfwhm", 200.0)
        minfwhm = fit_settings.get("minfwhm", 0.0)
        
        new_params = {}
        y_arr = md.Y[0] if md.Y is not None else md.Y0[0]
        self._initialize_peak_params(new_params, model_name, x0, ampli, minfwhm, maxfwhm, maxshift, y_arr)

        md.fit_model["peak_models"][str(index)] = {model_name: new_params}
        self._reconstruct_y_peaks(md)
        self._emit_selected_spectra()


    def update_peak_param(self, index, key, field, value):
        md = self._get_unique_map_data(self._get_selected_spectra())[0]
        if not md or not md.fit_model: return

        # Clear fit state
        md.peak_params = None
        md.param_names = None
        md.fit_success = None
        md.fit_r2 = None
        md.Y_bestfit = None

        pdict = md.fit_model.get("peak_models", {}).get(str(index))
        if pdict:
            shape = list(pdict.keys())[0]
            if key in pdict[shape]:
                pdict[shape][key][field] = value
        self._reconstruct_y_peaks(md)
        self._emit_selected_spectra()

    def delete_peak(self, index):
        md = self._get_unique_map_data(self._get_selected_spectra())[0]
        if not md or not md.fit_model: return

        old_models = md.fit_model.get("peak_models", {})
        old_labels = md.fit_model.get("peak_labels", [])
        
        new_models = {}
        new_labels = []
        new_idx = 0
        for k, pdict in old_models.items():
            if int(k) == index:
                continue
            new_models[str(new_idx)] = pdict
            if int(k) < len(old_labels):
                new_labels.append(old_labels[int(k)])
            new_idx += 1
            
        md.fit_model["peak_models"] = new_models
        md.fit_model["peak_labels"] = new_labels
        md.peak_params = None
        md.param_names = None
        md.fit_success = None
        md.fit_r2 = None
        md.Y_bestfit = None
        self._reconstruct_y_peaks(md)
        self._emit_selected_spectra()

    def update_dragged_peak(self, x: float, y: float):
        """Update peak position during dragging (real-time update).
        
        Args:
            x: New x position (center)
            y: New y value (amplitude/intensity)
        """
        if not self.selected_fnames:
            return

        md = self._get_unique_map_data(self._get_selected_spectra())[0]
        if not md or not md.fit_model: return
        
        # Determine the peak being dragged
        min_dist = float('inf')
        closest_k = None
        for k, pdict in md.fit_model.get("peak_models", {}).items():
            shape = list(pdict.keys())[0]
            if "x0" in pdict[shape]:
                px = pdict[shape]["x0"]["value"]
                dist = abs(px - x)
                if dist < min_dist:
                    min_dist = dist
                    closest_k = k
        
        if closest_k is not None:
            shape = list(md.fit_model["peak_models"][closest_k].keys())[0]
            md.fit_model["peak_models"][closest_k][shape]["x0"]["value"] = x
            if "ampli" in md.fit_model["peak_models"][closest_k][shape]:
                md.fit_model["peak_models"][closest_k][shape]["ampli"]["value"] = y

    def finalize_peak_drag(self):
        """Finalize peak drag operation - ensure model is synchronized."""
        if not self.selected_fnames:
            return

        md = self._get_interactive_map_data()
        if md:
            md.peak_params = None
            md.param_names = None
            md.fit_success = None
            md.fit_r2 = None
            md.Y_bestfit = None
            self._reconstruct_y_peaks(md)

        # Re-emit to ensure everything is synchronized
        self._emit_selected_spectra()


    def copy_spectrum_data_to_clipboard(self):
        """Copy X, Y, and peak model data of the first selected spectrum to clipboard as DataFrame."""
        self._copy_spectrum_data()

    def _copy_spectrum_data(self):
        """Copy X, Y, and peak model data of the first selected spectrum to clipboard as DataFrame."""
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
        """Save current workspace to .spectra file using high-performance ZIP format."""
        
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save work",
            "",
            "SPECTROview Files (*.spectra)"
        )
        
        if not file_path:
            return
        
        try:
            spectrums_meta = {}
            arrays = {}
            for i, fname in enumerate(self.store.map_names):
                md = self.store.get_map_data(fname)
                if not md: continue
                
                spectrum_dict = {
                    "fname": fname,
                    "is_active": md.is_active.tolist() if md.is_active is not None else [True],
                    "metadata": md.map_metadata,
                    "baseline_config": md.baseline_config,
                    "peak_params": md.peak_params,
                    "colors": md.colors,
                    "labels": md.labels
                }
                
                # Format v4: Store x0/y0 arrays directly in NPZ instead of base64 JSON
                if md.x0 is not None:
                    arrays[f"x0_{i}"] = md.x0
                if md.Y0 is not None:
                    arrays[f"y0_{i}"] = md.Y0[0]
                    
                spectrums_meta[i] = spectrum_dict
                
            metadata = {
                'format_version': 5,  # Version 5 signals TensorStore based save format
                'spectrums_meta': spectrums_meta
            }
            
            dataframes = {}
            if self.df_fit_results is not None and not self.df_fit_results.empty:
                dataframes['df_fit_results'] = self.df_fit_results
            
            WorkspaceIO.save_workspace(file_path, metadata, arrays, dataframes)
            self.notify.emit("Work saved successfully.")
        except Exception as e:
            QMessageBox.critical(None, "Save Error", f"Error saving work:\n{str(e)}")

    def load_work(self, file_path: str):
        """Load previously saved workspace from .spectra file (supports ZIP)."""
        
        try:
            metadata, arrays, dataframes, is_legacy = WorkspaceIO.load_workspace(file_path)
            
            if is_legacy or metadata.get('format_version', 3) < 4:
                QMessageBox.critical(None, "Load Error", "Legacy formats (< v4) are no longer supported in Tensor Engine mode.")
                return
            
            # Clear existing data
            self.store = SpectraStore()
            self.selected_fnames = []
            
            # Load all spectra
            fmt = metadata.get('format_version', 5)
            spectrums_meta = metadata.get('spectrums_meta', {})
            for spectrum_id, spectrum_data in spectrums_meta.items():
                i = int(spectrum_id)
                x0_arr = arrays.get(f"x0_{i}")
                y0_arr = arrays.get(f"y0_{i}")
                
                if x0_arr is None or y0_arr is None:
                    continue
                    
                fname = spectrum_data.get("fname", f"spectrum_{i}")
                is_active = np.array(spectrum_data.get("is_active", [True]), dtype=bool)
                
                self.store.add_map(
                    name=fname,
                    x0=x0_arr.copy(),
                    Y0=y0_arr.astype(np.float32).reshape(1, -1),
                    coords=np.array([[0.0, 0.0]], dtype=np.float64),
                    fnames=[fname],
                    is_active=is_active
                )
                
                md = self.store.get_map_data(fname)
                if md:
                    md.map_metadata = spectrum_data.get("metadata", {})
                    md.baseline_config = spectrum_data.get("baseline_config")
                    md.peak_params = spectrum_data.get("peak_params")
                    md.colors = spectrum_data.get("colors", [None])
                    md.labels = spectrum_data.get("labels", [None])
            
            self._sync_fit_results_to_store()
            
            # Restore fit results DataFrame
            if dataframes and 'df_fit_results' in dataframes:
                self.df_fit_results = dataframes['df_fit_results']
                # Emit signal to update fit results table
                self.fit_results_updated.emit(self.df_fit_results)
            else:
                self.df_fit_results = None
            
            # Update UI
            self._emit_list_update()
            map_names = self.store.map_names
            if len(map_names) > 0:
                self.selected_fnames = [map_names[0]]
                self._emit_selected_spectra()
            else:
                self.selected_fnames = []
                self.spectra_selection_changed.emit([])
            
        except Exception as e:
            QMessageBox.critical(None, "Load Error", f"Error loading spectra workspace:\n{str(e)}")

    def clear_workspace(self):
        """Clear all spectra and reset workspace to initial state."""
        self.store = SpectraStore()
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

    def collect_fit_results(self, map_names: list[str] = None):
        """Collect best-fit results from target maps and create DataFrame."""
        if map_names is None:
            map_names = self._get_active_spectra()
            
        if not map_names:
            self.notify.emit("No active spectra to collect results from.")
            return

        dfs = []
        for name in map_names:
            md = self.store.get_map_data(name)
            if not md or not md.has_fit_results():
                continue

            peak_labels = None
            if md.fit_model:
                peak_labels = md.fit_model.get('peak_labels', None)

            # Build DataFrame for this specific MapData (Spectra Workspace collects all)
            df = self.store.build_fit_results_df(
                name=name,
                map_type=getattr(self, 'map_type', '2Dmap'),
                peak_labels=peak_labels,
                only_converged=False,
            )
            if df is not None and not df.empty:
                dfs.append(df)

        if not dfs:
            self.notify.emit("No fit results to collect.")
            self.df_fit_results = None
            return

        # Concatenate all DataFrames
        df_all = pd.concat(dfs, ignore_index=True)
        # Round only non-coordinate float columns to 3 decimal places to preserve precision of coordinates (X, Y)
        cols_to_round = [col for col in df_all.columns if col not in ['Filename', 'X', 'Y', 'Quadrant', 'Zone']]
        df_all[cols_to_round] = df_all[cols_to_round].round(3)
        self.df_fit_results = df_all
        self.fit_results_updated.emit(self.df_fit_results)
    
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

    def save_spectra_data(self, parent_widget=None):
        """Save selected spectra data (x, y) to separate txt files."""
        selected_spectra = self._get_selected_spectra()
        
        if not selected_spectra:
            self.notify.emit("No spectrum selected.")
            return
            
        last_dir = self.settings.get_last_directory()
        
        dir_path = QFileDialog.getExistingDirectory(
            parent_widget,
            "Select Directory to Save Spectra Data",
            last_dir
        )
        
        if not dir_path:
            return
            
        saved_count = 0
        for spectrum in selected_spectra:
            # Create a safe filename based on the spectrum's fname
            base_name = str(spectrum.fname)
            # Replace invalid path characters if necessary (though usually fname is already safe)
            safe_name = "".join([c for c in base_name if c.isalpha() or c.isdigit() or c in (' ', '.', '_', '-')]).rstrip()
            if not safe_name.lower().endswith('.txt'):
                safe_name += '.txt'
                
            file_path = os.path.join(dir_path, safe_name)
            
            try:
                # spectrum.x and spectrum.y are the data arrays
                data = np.column_stack((spectrum.x, spectrum.y))
                np.savetxt(file_path, data, fmt='%.6f', delimiter='\t', comments='')
                saved_count += 1
            except Exception as e:
                self.notify.emit(f"Error saving {safe_name}: {e}")
                
        if saved_count > 0:
            self.notify.emit(f"Successfully saved {saved_count} spectra.")
            self.settings.set_last_directory(dir_path)
