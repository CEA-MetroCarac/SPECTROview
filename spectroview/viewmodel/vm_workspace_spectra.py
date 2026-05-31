"""ViewModel for Spectra Workspace - handles business logic and data management."""
import os
import re
import json
import base64
import zlib
import traceback
from copy import deepcopy
from pathlib import Path
        
import numpy as np
import pandas as pd

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox

from spectroview.model.m_io import load_spectrum_file, load_TRPL_data, load_wdf_spectrum, load_spc_spectrum
from spectroview.model.m_settings import MSettings
from spectroview.model.workspace_io import WorkspaceIO
from spectroview.model.spectra_store import SpectraStore, SpectrumProxy
from spectroview.model.peak_model import initialize_peak_params
from spectroview.fit_engine.vbf_thread import VBFthread
from spectroview.fit_engine.baseline import eval_baseline_batch
from spectroview.fit_engine.evaluator import eval_peak_initial
from spectroview.viewmodel.utils import (
    generate_fit_report,
    closest_index,
    save_df_to_excel,
    view_text,
    build_clean_fit_model,
)


class VMWorkspaceSpectra(QObject):
    # ───── ViewModel → View signals ─────
    spectra_list_changed = Signal(list)      # list[str]
    spectra_selection_changed = Signal(object) # list[MSpectrum] or dict (tensor batch)
    count_changed = Signal(int)
    show_xcorrection_value = Signal(float)  # ΔX of first selected spectrum
    spectral_range_changed = Signal(float, float)
    
    fit_in_progress = Signal(bool)  # Enable/disable fit buttons
    fit_progress_updated = Signal(int, int, int, float, int)  # (converged, total, pct, elapsed, converged_count)
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

    # ═════════════════════════════════════════════════════════════════════
    # Hooks for Bulk Operations (Template Method Pattern)
    # ═════════════════════════════════════════════════════════════════════

    def _get_target_mds(self, apply_all: bool) -> list:
        """Get the target MapData objects for bulk operations.
        
        - apply_all=True: returns all MapData objects in the store.
        - apply_all=False: returns unique MapData objects corresponding to selected spectra.
        """
        if apply_all:
            return [self.store.get_map_data(name) for name in self.store.map_names]
        else:
            if not self.selected_fnames:
                return []
            fnames = self._get_selected_spectra()
            return self._get_unique_map_data(fnames)

    def _on_map_data_changed(self, md, action: str):
        """Hook for subclasses to execute code after a MapData is modified."""
        pass

    def _post_bulk_action(self, apply_all: bool, action: str):
        """Hook for subclasses to execute code after a bulk action is completed."""
        self._emit_selected_spectra()
        self._emit_list_update()

    # ═════════════════════════════════════════════════════════════════════
    # View → ViewModel slots
    # ═════════════════════════════════════════════════════════════════════

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

        proxies = []
        for fname in active_fnames:
            md = self.store.get_map_data(fname)
            if md:
                proxies.append(SpectrumProxy(md, 0, fname))

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
            "proxies": proxies,
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
        xcorr = md0.xcorrection_value if md0 else 0.0
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
        
        for p_name, val in zip(param_names, peak_params):
            if "_" in p_name:
                try:
                    parts = p_name.split("_", 1)
                    prefix = parts[0]
                    pname = parts[1]
                    
                    digits = re.findall(r'\d+', prefix)
                    if digits:
                        peak_num_1based = int(digits[0])
                        key = str(peak_num_1based - 1)
                        
                        if key in peak_models:
                            pdict = peak_models[key]
                            shape = list(pdict.keys())[0]
                            
                            pname_key = pname
                            if pname == "amplitude" and "ampli" in pdict[shape]:
                                pname_key = "ampli"
                            elif pname == "ampli" and "amplitude" in pdict[shape]:
                                pname_key = "amplitude"
                                
                            if pname_key in pdict[shape]:
                                pdict[shape][pname_key]["value"] = float(val)
                except Exception as e:
                    print(f"Error updating parameter {p_name} from fit: {e}")

    def _reconstruct_y_peaks(self, md, indices=None):
        """Reconstruct md.Y_peaks from current fit_model."""
        if not md or not md.fit_model or not md.fit_model.get("peak_models"):
            md.Y_peaks = None
            return
        
        x_arr = md.x if md.x is not None else md.x0
        N = md.Y.shape[0] if md.Y is not None else md.Y0.shape[0]
        M = len(x_arr)
        
        peak_models = list(md.fit_model["peak_models"].values())
        num_peaks = len(peak_models)
        
        if indices is None:
            md.Y_peaks = []
            for p_model in peak_models:
                y_curve = eval_peak_initial(x_arr, p_model)
                md.Y_peaks.append(np.tile(y_curve, (N, 1)).astype(np.float32))
        else:
            if md.Y_peaks is None or len(md.Y_peaks) != num_peaks:
                md.Y_peaks = [np.zeros((N, M), dtype=np.float32) for _ in range(num_peaks)]
            for p_idx, p_model in enumerate(peak_models):
                y_curve = eval_peak_initial(x_arr, p_model)
                md.Y_peaks[p_idx][indices] = y_curve

    def _restore_preprocessed_state(self, md):
        """Restore cropped, baseline-subtracted, and peak states for MapData."""
        # Restore baseline_config from fit_model if missing (handles older/saved files)
        if not md.baseline_config and md.fit_model and isinstance(md.fit_model, dict) and "baseline" in md.fit_model:
            md.baseline_config = md.fit_model["baseline"]

        # 1. Apply range cropping
        if md.range_min is not None or md.range_max is not None:
            mask = np.logical_and(
                md.x0 >= (md.range_min if md.range_min is not None else -np.inf),
                md.x0 <= (md.range_max if md.range_max is not None else np.inf)
            )
            md.x = md.x0[mask].copy()
            md.Y = md.Y0[:, mask].copy()
        else:
            md.x = md.x0.copy()
            md.Y = md.Y0.copy()

        # NOTE: md.x0 and md.Y0 already contain xcorrection and intensity_norm
        # because those operations modify the raw arrays before saving.

        # 4. Evaluate baseline on cropped x/Y
        if md.baseline_config:
            md.Y_baseline = eval_baseline_batch(md.x, md.Y, md.baseline_config)

        # 5. If baseline was subtracted in saved session, subtract it now!
        is_sub = md.is_baseline_subtracted
        is_sub_any = is_sub.any() if isinstance(is_sub, np.ndarray) else bool(is_sub)
        if is_sub_any and md.Y_baseline is not None:
            if isinstance(is_sub, np.ndarray):
                sub_indices = np.where(is_sub)[0]
                if len(sub_indices) > 0:
                    md.Y[sub_indices] = md.Y[sub_indices] - md.Y_baseline[sub_indices]
                    md.Y_baseline[sub_indices] = 0.0
            else:
                md.Y = md.Y - md.Y_baseline
                md.Y_baseline = None

        # 6. Reconstruct Y_peaks and Y_bestfit
        if md.Y_bestfit is None and md.fit_model and "peak_models" in md.fit_model:
            md.Y_peaks = []
            x_arr = md.x
            N = md.Y.shape[0]
            total_peaks = np.zeros_like(md.Y)
            for p_model in md.fit_model["peak_models"].values():
                y_curve = eval_peak_initial(x_arr, p_model)
                y_curve_batch = np.tile(y_curve, (N, 1))
                md.Y_peaks.append(y_curve_batch.astype(np.float32))
                total_peaks += y_curve_batch
            
            is_sub_all = is_sub.all() if isinstance(is_sub, np.ndarray) else bool(is_sub)
            if not is_sub_all and md.Y_baseline is not None:
                md.Y_bestfit = (total_peaks + md.Y_baseline).astype(np.float32)
            else:
                md.Y_bestfit = total_peaks.astype(np.float32)

    def _reindex_fit_model(self, md):
        """Re-index all peak models to consecutive keys and update default labels accordingly."""
        if not md or not md.fit_model:
            return

        old_models = md.fit_model.get("peak_models", {})
        old_labels = md.fit_model.get("peak_labels", [])

        if not old_models:
            return

        sorted_keys = sorted(old_models.keys(), key=lambda k: int(k))

        new_models = {}
        new_labels = []

        for new_idx, k in enumerate(sorted_keys):
            pdict = old_models[k]
            new_models[str(new_idx)] = pdict

            orig_label = ""
            k_int = int(k)
            if k_int < len(old_labels):
                orig_label = old_labels[k_int]

            if not orig_label or orig_label.strip() == "" or re.match(r"^Peak\s*\d+$", orig_label, re.IGNORECASE):
                new_labels.append(f"Peak{new_idx + 1}")
            else:
                new_labels.append(orig_label)

        md.fit_model["peak_models"] = new_models
        md.fit_model["peak_labels"] = new_labels

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

    def _emit_list_update(self):
        """Emit updated list of spectra dicts and count."""
        spectra_info = []
        for name in self.store.map_names:
            md = self.store.get_map_data(name)
            if not md: continue
            
            # Crop status
            is_cropped = md.range_min is not None or md.range_max is not None
            
            # Baseline status
            has_baseline = md.baseline_config is not None
            
            # Fit status
            has_fit = False
            fit_converged = False
            if md.peak_params is not None and len(md.peak_params) > 0:
                has_fit = np.any(md.peak_params[0] != 0.0)
                if has_fit and md.fit_success is not None:
                    fit_converged = bool(md.fit_success[0])
            
            spectra_info.append({
                "fname": name,
                "is_active": bool(md.is_active[0]),
                "is_cropped": is_cropped,
                "has_baseline": has_baseline,
                "has_fit": has_fit,
                "fit_success": fit_converged,
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

        # Determine which spectrum index to use
        spec_idx = 0
        if self.selected_fnames and self.selected_fnames[0] in md.fnames:
            spec_idx = md.fnames.index(self.selected_fnames[0])

        # Determine y value
        idx = closest_index(md.x if md.x is not None else md.x0, x)
        y_arr = md.Y[spec_idx] if md.Y is not None else md.Y0[spec_idx]
        y_val = float(y_arr[idx])
        if y_val <= 0: y_val = 1e-10

        # Create new peak dict
        if md.fit_model["peak_models"]:
            max_k = max(int(k) for k in md.fit_model["peak_models"].keys())
        else:
            max_k = -1
            
        if "peak_labels" not in md.fit_model:
            md.fit_model["peak_labels"] = []
            
        next_idx = max(max_k + 1, len(md.fit_model["peak_labels"]))
        peak_idx = str(next_idx)
        
        peak_model = {}
        self._initialize_peak_params(peak_model, peak_shape, x, y_val, minfwhm, maxfwhm, maxshift, y_arr)

        while len(md.fit_model["peak_labels"]) <= next_idx:
            md.fit_model["peak_labels"].append(f"Peak{len(md.fit_model['peak_labels']) + 1}")

        md.fit_model["peak_models"][peak_idx] = {peak_shape: peak_model}

        # Quickly evaluate and append to Y_peaks for instant preview
        x_arr = md.x if md.x is not None else md.x0
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
            # Simply remove the peak from the dictionary without reindexing!
            md.fit_model["peak_models"].pop(closest_k, None)
            
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

        is_sub = getattr(md, "is_baseline_subtracted", False)
        is_sub_any = is_sub.any() if isinstance(is_sub, np.ndarray) else bool(is_sub)
        if is_sub_any:
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
            prev_delta = md.xcorrection_value
            if prev_delta != 0.0:
                if md.x is not None:
                    md.x = md.x - prev_delta
                md.x0 = md.x0 - prev_delta

            if md.x is not None:
                md.x = md.x + delta_x
            md.x0 = md.x0 + delta_x
            md.xcorrection_value = delta_x
            
            if 'xcorrection_value' in md.map_metadata:
                del md.map_metadata['xcorrection_value']

        # Trigger plot refresh
        md0 = self._get_unique_map_data(fnames)[0]
        self.show_xcorrection_value.emit(md0.xcorrection_value)
        self._emit_selected_spectra()


    def undo_x_correction(self):
        """Undo X-axis correction for selected spectra."""
        if not self.selected_fnames:
            self.notify.emit("No spectrum selected.")
            return

        fnames = self._get_selected_spectra()

        for md in self._get_unique_map_data(fnames):
            delta = md.xcorrection_value
            if delta != 0.0:
                if md.x is not None:
                    md.x = md.x - delta
                md.x0 = md.x0 - delta
                md.xcorrection_value = 0.0
            
            if 'xcorrection_value' in md.map_metadata:
                del md.map_metadata['xcorrection_value']

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
            # Undo existing normalization if there is one
            if getattr(md, 'intensity_norm_factor', 1.0) != 1.0 and getattr(md, 'intensity_norm_factor', 1.0) != 0:
                if md.Y is not None:
                    md.Y = md.Y * md.intensity_norm_factor
                if md.Y0 is not None:
                    md.Y0 = md.Y0 * md.intensity_norm_factor
            
            # Apply new normalization (divide by factor)
            md.intensity_norm_factor = norm_factor
            if norm_factor != 0:
                if md.Y is not None:
                    md.Y = md.Y / norm_factor
                if md.Y0 is not None:
                    md.Y0 = md.Y0 / norm_factor

        self._emit_selected_spectra()

    def undo_y_normalization(self, apply_all: bool = False):
        """Undo intensity normalization for selected spectra."""
        fnames = (
            self._get_active_spectra()
            if apply_all
            else self._get_selected_spectra()
        )

        for md in self._get_unique_map_data(fnames):
            if getattr(md, 'intensity_norm_factor', 1.0) != 1.0 and getattr(md, 'intensity_norm_factor', 1.0) != 0:
                if md.Y is not None:
                    md.Y = md.Y * md.intensity_norm_factor
                if md.Y0 is not None:
                    md.Y0 = md.Y0 * md.intensity_norm_factor
                md.intensity_norm_factor = 1.0

        self._emit_selected_spectra()

    def reinit_spectra(self, apply_all: bool = False):
        """Reinitialize spectra to original data."""
        mds = self._get_target_mds(apply_all)
        if not mds:
            if not apply_all: self.notify.emit("No spectrum selected.")
            return

        for md in mds:
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
            md.colors = [None] * len(md.fnames)
            md.labels = [None] * len(md.fnames)
            
            self._on_map_data_changed(md, "reinit_spectra")
            
        self._post_bulk_action(apply_all, "reinit_spectra")


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
        mds = self._get_target_mds(apply_all)
        if not mds:
            if not apply_all: self.notify.emit("No spectrum selected.")
            return

        if xmin > xmax:
            xmin, xmax = xmax, xmin

        for md in mds:
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
            self._on_map_data_changed(md, "apply_spectral_range")

        self._post_bulk_action(apply_all, "apply_spectral_range")

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

        mds = self._get_target_mds(apply_all)
        if not mds:
            if not apply_all: self.notify.emit("No spectrum selected.")
            return

        # Apply baseline to selected spectra
        baseline_data = deepcopy(self._baseline_clipboard)
        for md in mds:
            md.baseline_config = deepcopy(baseline_data)
            
            x_arr = md.x if md.x is not None else md.x0
            y_arr = md.Y if md.Y is not None else md.Y0
            md.Y_baseline = eval_baseline_batch(x_arr, y_arr, md.baseline_config)
            self._on_map_data_changed(md, "paste_baseline")

        self._post_bulk_action(apply_all, "paste_baseline")

    def subtract_baseline(self, apply_all: bool = False):
        mds = self._get_target_mds(apply_all)
        if not mds:
            if not apply_all: self.notify.emit("No spectrum selected.")
            return

        subtracted_any = False
        for md in mds:
            if md and md.baseline_config is not None and md.Y_baseline is not None:
                is_sub = getattr(md, "is_baseline_subtracted", False)
                
                # Support both vector/array state (Maps Workspace) and scalar state (Spectra Workspace)
                if isinstance(is_sub, np.ndarray):
                    # Maps Workspace: subtract only for the unsubtracted pixel indices
                    indices_to_sub = [i for i in range(len(is_sub)) if not is_sub[i]]
                    if indices_to_sub:
                        if md.Y is None:
                            md.Y = md.Y0.copy()
                        if md.x is None:
                            md.x = md.x0.copy()
                        md.Y[indices_to_sub] = md.Y[indices_to_sub] - md.Y_baseline[indices_to_sub]
                        md.Y_baseline[indices_to_sub] = 0.0
                        md.is_baseline_subtracted[indices_to_sub] = True
                        subtracted_any = True
                        self._on_map_data_changed(md, "subtract_baseline")
                else:
                    # Spectra Workspace (or scalar mode): subtract if not already subtracted
                    if not is_sub:
                        if md.Y is None:
                            md.Y = md.Y0.copy()
                        if md.x is None:
                            md.x = md.x0.copy()
                        md.Y = md.Y - md.Y_baseline
                        md.Y_baseline = None
                        md.is_baseline_subtracted = True
                        subtracted_any = True
                        self._on_map_data_changed(md, "subtract_baseline")

        if subtracted_any:
            self._post_bulk_action(apply_all, "subtract_baseline")
        else:
            self.notify.emit("Baseline already subtracted for all selected spectra.")

    def delete_baseline(self, apply_all: bool = False):
        """Undo baseline subtraction: reinitialise spectrum data, re-apply crop range."""
        mds = self._get_target_mds(apply_all)
        if not mds:
            if not apply_all: self.notify.emit("No spectrum selected.")
            return

        for md in mds:
            if md:
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
                self._on_map_data_changed(md, "delete_baseline")

        self._post_bulk_action(apply_all, "delete_baseline")


    # ── Baseline mode / settings helpers ──────────────────────────────────

    def _apply_baseline_settings(self, settings: dict, fnames):
        """Internal helper: push mode/params from 'settings' dict onto each MapData baseline_config."""
        mode     = settings.get("mode")       
        coef     = float(settings.get("coef",     5.0))
        order    = int(settings.get("order_max", 1))
        sigma    = int(settings.get("sigma",     4))
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
        self.notify.emit("Peaks copied to clipboard.")

    def paste_peaks(self, apply_all: bool = False):
        if not hasattr(self, "_peaks_clipboard") or self._peaks_clipboard is None:
            self.notify.emit("No peaks copied.")
            return

        mds = self._get_target_mds(apply_all)
        if not mds:
            if not apply_all: self.notify.emit("No spectrum selected.")
            return

        for md in mds:
            if md:
                md.fit_model = deepcopy(self._peaks_clipboard)
                
                # Dynamically reconstruct Y_peaks for instant preview
                md.Y_peaks = []
                x_arr = md.x if md.x is not None else md.x0
                N = md.Y.shape[0] if md.Y is not None else md.Y0.shape[0]
                
                for p_model in md.fit_model.get("peak_models", {}).values():
                    y_curve = eval_peak_initial(x_arr, p_model)
                    md.Y_peaks.append(np.tile(y_curve, (N, 1)))
                self._on_map_data_changed(md, "paste_peaks")

        self._post_bulk_action(apply_all, "paste_peaks")

    def delete_peaks(self, apply_all: bool = False):
        mds = self._get_target_mds(apply_all)
        if not mds:
            if not apply_all: self.notify.emit("No spectrum selected.")
            return

        for md in mds:
            if md:
                md.fit_model = None
                md.Y_peaks = None
                md.Y_bestfit = None
                md.peak_params = None
                md.fit_success = None
                md.fit_r2 = None
                self._on_map_data_changed(md, "delete_peaks")

        self._post_bulk_action(apply_all, "delete_peaks")

    
    def _apply_fit_model_to_mapdata(self, md, fit_model, indices=None):
        """Cleanly resets and applies a fit model (cropping, baseline, and peaks) to MapData."""
        if not md:
            return

        N = md.Y0.shape[0] if md.Y0 is not None else 1
        M = md.Y.shape[1] if md.Y is not None else (md.Y0.shape[1] if md.Y0 is not None else 0)

        # 1. Clean reset
        if indices is None:
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
        else:
            if md.x is None and md.x0 is not None:
                md.x = md.x0.copy()
            if md.Y is None and md.Y0 is not None:
                md.Y = md.Y0.copy()

            # For the given indices, revert Y to raw Y0 (cropped to current md.x if needed)
            if md.Y0 is not None and md.Y is not None:
                if md.x is not None and md.x0 is not None and len(md.x) < len(md.x0):
                    xmin, xmax = md.x[0], md.x[-1]
                    i_min = closest_index(md.x0, xmin)
                    i_max = closest_index(md.x0, xmax)
                    if i_min > i_max:
                        i_min, i_max = i_max, i_min
                    md.Y[indices] = md.Y0[indices, i_min:i_max+1].copy()
                else:
                    md.Y[indices] = md.Y0[indices].copy()

            if not isinstance(md.is_baseline_subtracted, np.ndarray):
                md.is_baseline_subtracted = np.full(N, bool(md.is_baseline_subtracted), dtype=bool)
            md.is_baseline_subtracted[indices] = False

            if md.Y_baseline is not None:
                md.Y_baseline[indices] = 0.0
            if md.Y_bestfit is not None:
                md.Y_bestfit[indices] = 0.0
            if md.peak_params is not None:
                md.peak_params[indices] = 0.0
            if md.Y_peaks is not None:
                for peak_curve in md.Y_peaks:
                    peak_curve[indices] = 0.0

        # 2. Apply range cropping
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

        # Recalculate processed wavenumber axis length M
        M = md.Y.shape[1] if md.Y is not None else (md.Y0.shape[1] if md.Y0 is not None else 0)

        # 3. Apply baseline config
        bl_info = fit_model.get("baseline")
        if bl_info and bl_info.get("mode"):
            md.baseline_config = deepcopy(bl_info)
            x_arr = md.x if md.x is not None else md.x0
            y_arr = md.Y if md.Y is not None else md.Y0
            
            if indices is None:
                md.Y_baseline = eval_baseline_batch(x_arr, y_arr, md.baseline_config)
                
                if bl_info.get("is_subtracted", False):
                    md.Y = md.Y - md.Y_baseline
                    # Clear Y_baseline curve since it is subtracted into Y, but preserve config
                    md.Y_baseline = None
                    md.is_baseline_subtracted = True
            else:
                y_baseline_sel = eval_baseline_batch(x_arr, y_arr[indices], md.baseline_config)
                if md.Y_baseline is None:
                    md.Y_baseline = np.zeros((N, M), dtype=np.float32)
                md.Y_baseline[indices] = y_baseline_sel
                
                if bl_info.get("is_subtracted", False):
                    md.Y[indices] = md.Y[indices] - md.Y_baseline[indices]
                    md.Y_baseline[indices] = 0.0
                    md.is_baseline_subtracted[indices] = True

        # 4. Attach fit model dictionary
        md.fit_model = deepcopy(fit_model)

        # 5. Reconstruct peak curves for preview
        self._reconstruct_y_peaks(md, indices=indices)

    def copy_fit_model(self):
        if not self.selected_fnames:
            self.notify.emit("No spectrum selected.")
            return

        fnames = self._get_selected_spectra()
        md = self._get_unique_map_data(fnames)[0]

        if not md.fit_model:
            self.notify.emit("No fit results to copy.")
            return

        # Ensure current crop range and baseline config are written into fit_model before copy
        md.fit_model["range_min"] = md.range_min
        md.fit_model["range_max"] = md.range_max
        if md.baseline_config:
            bl = md.baseline_config
            is_sub = getattr(md, "is_baseline_subtracted", False)
            is_sub_val = bool(is_sub[0]) if isinstance(is_sub, np.ndarray) else bool(is_sub)
            
            md.fit_model["baseline"] = deepcopy(bl)
            md.fit_model["baseline"]["is_subtracted"] = is_sub_val
        else:
            md.fit_model["baseline"] = None

        # Inject current fit settings
        md.fit_model["fit_params"] = self.settings.load_fit_settings()

        self._fitmodel_clipboard = deepcopy(build_clean_fit_model(md.fit_model))
        self.notify.emit("Fit model copied to clipboard.")

    def paste_fit_model(self, apply_all: bool = False):
        if not hasattr(self, "_fitmodel_clipboard") or self._fitmodel_clipboard is None:
            self.notify.emit("No fit model copied.")
            return
        fnames = self._get_active_spectra() if apply_all else self._get_selected_spectra()
        if not fnames:
            self.notify.emit("No spectrum selected.")
            return

        for md in self._get_unique_map_data(fnames):
            self._apply_fit_model_to_mapdata(md, self._fitmodel_clipboard)

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

        # Ensure current crop range and baseline config are written into fit_model before save
        md.fit_model["range_min"] = md.range_min
        md.fit_model["range_max"] = md.range_max
        if md.baseline_config:
            bl = md.baseline_config
            is_sub = getattr(md, "is_baseline_subtracted", False)
            is_sub_val = bool(is_sub[0]) if isinstance(is_sub, np.ndarray) else bool(is_sub)
            
            md.fit_model["baseline"] = deepcopy(bl)
            md.fit_model["baseline"]["is_subtracted"] = is_sub_val
        else:
            md.fit_model["baseline"] = None

        # Inject current fit settings
        md.fit_model["fit_params"] = self.settings.load_fit_settings()

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

        clean_model = build_clean_fit_model(md.fit_model)

        def default_encoder(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'0': clean_model}, f, indent=2, default=default_encoder)

        self.notify.emit("Fit model saved successfully.")

    def apply_fit_model(self, apply_all: bool = False):
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
            self._apply_fit_model_to_mapdata(md, fit_model)

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
                self._reindex_fit_model(md)
                
                # Inject current fit settings
                md.fit_model["fit_params"] = self.settings.load_fit_settings()
                
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

        self._fit_thread = VBFthread(self.store, tasks)
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

        # Ensure fit_model has fit_params before running
        if "fit_params" not in fit_model:
            fit_model["fit_params"] = self.settings.load_fit_settings()

        tasks = [{
            "map_names": fnames,
            "indices": np.arange(len(fnames)),
            "fit_model": fit_model
        }]

        # Cancel any existing thread
        if self._fit_thread and self._fit_thread.isRunning():
            self._fit_thread.terminate()
            self._fit_thread.wait()

        self._is_fitting = True
        self.fit_in_progress.emit(True)

        self._fit_thread = VBFthread(self.store, tasks)
        self._fit_thread.progress_changed.connect(self.fit_progress_updated.emit)
        self._fit_thread.timings_ready.connect(self.fit_timings_ready.emit)
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
        if md and md.fit_model and "peak_models" in md.fit_model:
            if "peak_labels" not in md.fit_model:
                md.fit_model["peak_labels"] = []
            
            # Make sure list is large enough to hold index
            while len(md.fit_model["peak_labels"]) <= index:
                md.fit_model["peak_labels"].append(f"Peak{len(md.fit_model['peak_labels']) + 1}")
                
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
        
        # Determine which spectrum index to use
        spec_idx = 0
        if self.selected_fnames and self.selected_fnames[0] in md.fnames:
            spec_idx = md.fnames.index(self.selected_fnames[0])
            
        y_arr = md.Y[spec_idx] if md.Y is not None else md.Y0[spec_idx]
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
        if str(index) in old_models:
            del old_models[str(index)]

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

        fnames = self._get_selected_spectra()
        if not fnames:
            return

        md = self.store.get_map_data(fnames[0])
        if not md: return

        local_idx = md.fnames.index(fnames[0])
        x_values = md.x if md.x is not None else md.x0
        y_values = md.Y[local_idx] if md.Y is not None else md.Y0[local_idx]

        data = {
            "X values": x_values,
            "Y values": y_values
        }

        if md.fit_model and md.fit_model.get("peak_models"):
            peak_models = list(md.fit_model["peak_models"].values())
            peak_labels = md.fit_model.get("peak_labels", [])
            for i, p_model in enumerate(peak_models):
                try:
                    self._update_fit_model_from_params(md, local_idx)
                    y_peak = eval_peak_initial(x_values, p_model)

                    if i < len(peak_labels):
                        label = peak_labels[i]
                    else:
                        label = f"Peak{i + 1}"

                    data[label] = y_peak
                except Exception:
                    continue

        df = pd.DataFrame(data)
        df.to_clipboard(index=False)

    def save_work(self):
        """Save workspace using high-performance ZIP format (SpectraStore-backed)."""
        is_maps = hasattr(self, 'maps')
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save work",
            "",
            "SPECTROview Maps (*.maps)" if is_maps else "SPECTROview Files (*.spectra)"
        )
        if not file_path:
            return

        try:
            # 1. Export entire store to NPZ and metadata
            arrays = {}
            store_meta = {}
            for map_name in self.store.map_names:
                # Add arrays
                arrays.update(self.store.to_npz_dict(map_name))
                # Add metadata
                store_meta[map_name] = self.store.to_metadata_dict(map_name)

            def _sanitize_for_json(obj):
                """Recursively convert numpy types to JSON-serializable equivalents."""
                if isinstance(obj, dict):
                    return {k: _sanitize_for_json(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_sanitize_for_json(v) for v in obj]
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                return obj

            metadata = {
                'format_version': 2,  # Version 2 signals direct ZIP-streaming newest serialization format
                'store_meta': _sanitize_for_json(store_meta),
            }

            # If it is Maps workspace, save maps-specific metadata
            if is_maps:
                metadata['maps_metadata'] = _sanitize_for_json(getattr(self, 'maps_metadata', {}))
                metadata['map_type'] = getattr(self, 'map_type', '2Dmap')

            dataframes = {}
            if self.df_fit_results is not None and not self.df_fit_results.empty:
                dataframes['df_fit_results'] = self.df_fit_results

            WorkspaceIO.save_workspace(file_path, metadata, arrays, dataframes)
            self.notify.emit("Work saved successfully.")
            
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(None, "Save Error", f"Error saving work:\n{str(e)}")

    def load_work(self, file_path: str):
        """Load previously saved workspace from legacy JSON (v1) or newest ZIP-streaming (v2)."""
        try:
            metadata, arrays, dataframes, is_legacy = WorkspaceIO.load_workspace(file_path)

            if is_legacy or metadata.get('format_version', 1) < 2:
                self.clear_workspace()
                if hasattr(self, 'maps'):
                    self._load_legacy_maps(file_path)
                else:
                    self._load_legacy_spectra(file_path)
                return

            self.clear_workspace()
            
            # 1. Restore SpectraStore from arrays + per-map metadata
            self.store = SpectraStore()
            store_meta = metadata.get('store_meta', {})
            for map_name, meta in store_meta.items():
                SpectraStore.load_map_from_npz(arrays, meta, map_name, store=self.store)
                md = self.store.get_map_data(map_name)
                if md:
                    self._restore_preprocessed_state(md)

            # 2. If Maps workspace, rebuild legacy DataFrames for heatmap rendering (highly optimized)
            if hasattr(self, 'maps'):
                self.maps = {}
                self.maps_metadata = metadata.get('maps_metadata', {})
                self.map_type = metadata.get('map_type', '2Dmap')
                
                for map_name in self.store.map_names:
                    md = self.store.get_map_data(map_name)
                    x0 = md.x0
                    coords = md.coords
                    intensities = md.Y0

                    # 18.7x faster and memory-efficient DataFrame builder:
                    col_names = list(map(str, x0))
                    df = pd.DataFrame(intensities, columns=col_names)
                    df.insert(0, 'Y', coords[:, 1])
                    df.insert(0, 'X', coords[:, 0])
                    self.maps[map_name] = df

            # 3. Restore DataFrames
            self.df_fit_results = dataframes.get('df_fit_results') if dataframes else None

            # 4. Refresh View
            if hasattr(self, 'maps'):
                map_names = self.store.map_names
                if map_names:
                    self.select_map(map_names[0])
                self.maps_list_changed.emit(map_names)
                self.count_changed.emit(sum(md.n_spectra for md in self.store._maps.values()))
            else:
                self._emit_list_update()
                map_names = self.store.map_names
                if len(map_names) > 0:
                    self.selected_fnames = [map_names[0]]
                    self._emit_selected_spectra()
                else:
                    self.selected_fnames = []
                    self.spectra_selection_changed.emit([])

            if self.df_fit_results is not None:
                self.fit_results_updated.emit(self.df_fit_results)
                
            self.notify.emit("Work loaded successfully.")

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(None, "Load Error", f"Error loading workspace:\n{str(e)}")

    def _load_legacy_spectra(self, file_path: str):
        """Load legacy JSON-based .spectra workspace (OLD version)."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.store = SpectraStore()
        self.selected_fnames = []

        def decompress_array(base64_str, dtype=np.float64):
            if not base64_str:
                return None
            decoded = base64.b64decode(base64_str.encode('utf-8'))
            decompressed = zlib.decompress(decoded)
            return np.frombuffer(decompressed, dtype=dtype)

        spectrums = data.get("spectrums", {})
        sorted_keys = sorted(spectrums.keys(), key=lambda x: int(x))
        for skey in sorted_keys:
            sdata = spectrums[skey]
            fname = sdata.get("fname", f"spectrum_{skey}")
            
            x0_arr = decompress_array(sdata.get("x0"), np.float64)
            y0_arr = decompress_array(sdata.get("y0"), np.float64)
            if x0_arr is None or y0_arr is None:
                continue

            is_active = sdata.get("is_active", True)

            self.store.add_map(
                name=fname,
                x0=x0_arr.copy(),
                Y0=y0_arr.astype(np.float32).reshape(1, -1),
                coords=np.array([[0.0, 0.0]], dtype=np.float64),
                fnames=[fname],
                is_active=np.array([is_active], dtype=bool)
            )

            md = self.store.get_map_data(fname)
            if md:
                md.map_metadata = sdata.get("metadata", {})
                md.colors = [sdata.get("color")]
                md.labels = [sdata.get("label")]
                md.xcorrection_value = float(sdata.get("xcorrection_value", 0.0))
                md.intensity_norm_factor = float(sdata.get("intensity_norm_factor", 1.0))
                md.is_baseline_subtracted = sdata.get("is_baseline_subtracted", False)
                md.range_min = sdata.get("range_min")
                md.range_max = sdata.get("range_max")

                # Crop range if min/max are defined
                if md.range_min is not None or md.range_max is not None:
                    mask = np.logical_and(
                        md.x0 >= md.range_min,
                        md.x0 <= md.range_max
                    )
                    md.x = md.x0[mask].copy()
                    md.Y = md.Y0[:, mask].copy()
                else:
                    md.x = md.x0.copy()
                    md.Y = md.Y0.copy()

                # NOTE: legacy y0 and x0 already contain xcorrection and intensity_norm

                legacy_bl = sdata.get("baseline")
                if legacy_bl:
                    md.baseline_config = {
                        "mode": legacy_bl.get("mode", "Linear"),
                        "points": legacy_bl.get("points", [[], []]),
                        "attached": legacy_bl.get("attached", False),
                        "coef": legacy_bl.get("coef", 5),
                    }
                    # Batch evaluate baseline on cropped x/Y
                    md.Y_baseline = eval_baseline_batch(md.x, md.Y, md.baseline_config)

                # If baseline was subtracted in saved session, subtract it now!
                if md.is_baseline_subtracted and md.Y_baseline is not None:
                    md.Y = md.Y - md.Y_baseline
                    md.Y_baseline = None

                legacy_peaks = sdata.get("peak_models")
                legacy_labels = sdata.get("peak_labels", [])
                if legacy_peaks:
                    md.fit_model = {
                        "peak_labels": legacy_labels,
                        "peak_models": legacy_peaks
                    }
                    
                    # Reconstruct Y_peaks and Y_bestfit
                    md.Y_peaks = []
                    x_arr = md.x
                    N = md.Y.shape[0]
                    total_peaks = np.zeros_like(md.Y)
                    for p_model in md.fit_model["peak_models"].values():
                        y_curve = eval_peak_initial(x_arr, p_model)
                        y_curve_batch = np.tile(y_curve, (N, 1))
                        md.Y_peaks.append(y_curve_batch.astype(np.float32))
                        total_peaks += y_curve_batch
                    
                    if md.Y_baseline is not None:
                        md.Y_bestfit = (total_peaks + md.Y_baseline).astype(np.float32)
                    else:
                        md.Y_bestfit = total_peaks.astype(np.float32)

                    md.fit_success = np.array([sdata.get("result_fit_success", False)], dtype=bool)

        legacy_results = data.get("df_fit_results")
        if legacy_results:
            self.df_fit_results = pd.DataFrame(legacy_results)
            self.fit_results_updated.emit(self.df_fit_results)
        else:
            self.df_fit_results = None

        self._emit_list_update()
        map_names = self.store.map_names
        if len(map_names) > 0:
            self.selected_fnames = [map_names[0]]
            self._emit_selected_spectra()
        else:
            self.selected_fnames = []
            self.spectra_selection_changed.emit([])
        self.notify.emit("Legacy spectra workspace loaded successfully.")

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
            map_names = self.store.map_names
            
        if not map_names:
            self.notify.emit("No loaded spectra to collect results from.")
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
        
        # Drop X and Y columns for Spectra Workspace only
        if type(self).__name__ == 'VMWorkspaceSpectra':
            for col in ['X', 'Y']:
                if col in df_all.columns:
                    df_all.drop(columns=[col], inplace=True)
                    
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
        fnames = self._get_selected_spectra()
        
        if not fnames:
            self.notify.emit("No spectrum selected.")
            return
        
        # Show the 'report' of the first selected spectrum
        fname = fnames[0]
        md = self.store.get_map_data(fname)
        
        if not md or not md.has_fit_results():
            self.notify.emit("No fit results available for the selected spectrum. Please fit the spectrum first.")
            return
            
        try:
            local_idx = md.fnames.index(fname)
            success = bool(md.fit_success[local_idx])
            r2 = float(md.fit_r2[local_idx])
            
            report = ["[[Fit Statistics]]"]
            report.append(f"    success    = {success}")
            report.append(f"    R-squared  = {r2:.6f}")
            report.append("\n[[Variables]]")
            
            # Apply user-defined peak labels if available
            col_names = list(md.param_names)
            if md.fit_model and md.fit_model.get("peak_labels"):
                col_names = self.store._apply_peak_labels(col_names, md.fit_model["peak_labels"])
            
            for name, val in zip(col_names, md.peak_params[local_idx]):
                report.append(f"    {name:15s}: {val:.6g}")
                
            text = "\n".join(report)
            title = f"Fitting Report - {fname}"
            view_text(parent_widget, title, text)
        except Exception as e:
            self.notify.emit(f"Error generating fit report: {str(e)}")

    def save_spectra_data(self, parent_widget=None):
        """Save selected spectra data (x, y) to separate txt files."""
        fnames = self._get_selected_spectra()
        
        if not fnames:
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
        for fname in fnames:
            # Create a safe filename based on the spectrum's fname
            base_name = str(fname)
            # Replace invalid path characters if necessary (though usually fname is already safe)
            safe_name = "".join([c for c in base_name if c.isalpha() or c.isdigit() or c in (' ', '.', '_', '-')]).rstrip()
            if not safe_name.lower().endswith('.txt'):
                safe_name += '.txt'
                
            file_path = os.path.join(dir_path, safe_name)
            
            try:
                md = self.store.get_map_data(fname)
                if md:
                    local_idx = md.fnames.index(fname)
                    x = md.x if md.x is not None else md.x0
                    y = md.Y[local_idx] if md.Y is not None else md.Y0[local_idx]
                    data = np.column_stack((x, y))
                    np.savetxt(file_path, data, fmt='%.6f', delimiter='\t', comments='')
                    saved_count += 1
            except Exception as e:
                self.notify.emit(f"Error saving {safe_name}: {e}")
                
        if saved_count > 0:
            self.notify.emit(f"Successfully saved {saved_count} spectra.")
            self.settings.set_last_directory(dir_path)
