"""Peak editing, fit-model management, and VBFthread orchestration mixin."""

from __future__ import annotations

from copy import deepcopy
import json
import re

import numpy as np
from PySide6.QtWidgets import QFileDialog

from spectroview.fit_engine.baseline import eval_baseline_batch
from spectroview.fit_engine.evaluator import eval_peak_initial
from spectroview.fit_engine.vbf_thread import VBFthread
from spectroview.i18n import tr
from spectroview.model.spectra.peak_model import initialize_peak_params
from spectroview.viewmodel.common.spectral_utils import (
    build_clean_fit_model,
    closest_index,
)


class SpectraFittingMixin:
    """Peak model manipulation, fit-model serialization, and VBF thread execution."""

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

                    digits = re.findall(r"\d+", prefix)
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

    def _reconstruct_y_peaks(self, md, indices=None, include_bestfit: bool = False):
        """Reconstruct md.Y_peaks lazily (and optionally md.Y_bestfit) from current fit_model."""
        if not md or not md.fit_model or not md.fit_model.get("peak_models"):
            md.Y_peaks = None
            return

        from spectroview.model.spectra.spectra_store import LazyPeakArray

        x_arr = md.x_axis
        n_rows = md.y_matrix.shape[0]
        m_cols = len(x_arr)

        peak_models = list(md.fit_model["peak_models"].values())
        num_peaks = len(peak_models)

        md.Y_peaks = [LazyPeakArray(md, i) for i in range(num_peaks)]

        if include_bestfit:
            if md.peak_params is not None and md.has_fit_results():
                total_peaks = np.zeros((n_rows, m_cols), dtype=np.float32)
                for p_lazy in md.Y_peaks:
                    total_peaks += p_lazy[:]
            else:
                total_curve = np.zeros(m_cols, dtype=np.float32)
                for p_model in peak_models:
                    total_curve += eval_peak_initial(x_arr, p_model).astype(np.float32)
                total_peaks = np.tile(total_curve, (n_rows, 1))
            is_sub = md.is_baseline_subtracted
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

            if (
                not orig_label
                or orig_label.strip() == ""
                or re.match(r"^Peak\s*\d+$", orig_label, re.IGNORECASE)
            ):
                new_labels.append(f"Peak{new_idx + 1}")
            else:
                new_labels.append(orig_label)

        md.fit_model["peak_models"] = new_models
        md.fit_model["peak_labels"] = new_labels

    @staticmethod
    def _clear_fit_state(md):
        """Discard stale fit results on a MapData block.

        Called whenever peaks are added/removed/edited so
        ``_update_fit_model_from_params`` doesn't later overwrite the edited
        peak with values from a fit that no longer matches the current model.
        """
        md.peak_params = None
        md.param_names = None
        md.fit_success = None
        md.fit_r2 = None
        md.Y_bestfit = None

    def add_peak_at(self, x: float):
        """Add a peak centered at *x* using the active peak shape."""
        md = self._get_interactive_map_data()
        if not md:
            return

        # Clear fit state to prevent _update_fit_model_from_params overwriting with stale parameters
        self._clear_fit_state(md)

        if not isinstance(md.fit_model, dict):
            md.fit_model = {"peak_labels": [], "peak_models": {}}
        else:
            if not isinstance(md.fit_model.get("peak_models"), dict):
                md.fit_model["peak_models"] = {}
            if not isinstance(md.fit_model.get("peak_labels"), list):
                md.fit_model["peak_labels"] = []

        fit_settings = self.settings.load_fit_settings()
        maxshift = fit_settings.get("maxshift", 20.0)
        maxfwhm = fit_settings.get("maxfwhm", 200.0)
        minfwhm = fit_settings.get("minfwhm", 0.0)
        peak_shape = self._current_peak_shape or "Lorentzian"

        # Determine which spectrum index to use
        spec_idx = 0
        if self.selected_fnames:
            fname_idx = md.get_fname_index(self.selected_fnames[0])
            if fname_idx is not None:
                spec_idx = fname_idx

        # Determine y value
        idx = closest_index(md.x_axis, x)
        y_arr = md.y_matrix[spec_idx]
        y_val = float(y_arr[idx])
        if y_val <= 0:
            y_val = 1e-10

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
        self._initialize_peak_params(
            peak_model, peak_shape, x, y_val, minfwhm, maxfwhm, maxshift, y_arr
        )

        while len(md.fit_model["peak_labels"]) <= next_idx:
            md.fit_model["peak_labels"].append(
                f"Peak{len(md.fit_model['peak_labels']) + 1}"
            )

        md.fit_model["peak_models"][peak_idx] = {peak_shape: peak_model}

        # Expose lazy per-peak evaluator for instant preview without full-map tiling
        from spectroview.model.spectra.spectra_store import LazyPeakArray

        n_peaks = len(md.fit_model["peak_models"])
        md.Y_peaks = [LazyPeakArray(md, i) for i in range(n_peaks)]

        self._emit_selected_spectra()

    def _initialize_peak_params(
        self, peak_model, peak_shape, x0, ampli, minfwhm, maxfwhm, maxshift, y_arr=None
    ):
        """Build canonical parameters dictionary based on peak shape."""
        initialize_peak_params(
            peak_model, peak_shape, x0, ampli, minfwhm, maxfwhm, maxshift, y_arr
        )

    def remove_peak_at(self, x: float):
        """Remove the peak whose center is closest to *x*."""
        md = self._get_interactive_map_data()
        if not md or not md.fit_model or not md.fit_model.get("peak_models"):
            return

        closest_k = None
        min_dist = float("inf")
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

            self._clear_fit_state(md)
            self._reconstruct_y_peaks(md)

        self._emit_selected_spectra()

    def copy_peaks(self) -> bool:
        """Copy peak definitions from the primary selected spectrum."""
        if not self.selected_fnames:
            self.notify.emit(tr("No spectrum selected."))
            return False

        md = self._get_primary_selected_map_data()
        if not md or not md.fit_model or not md.fit_model.get("peak_models"):
            self.notify.emit(tr("No peaks to copy."))
            return False
        self._peaks_clipboard = deepcopy(md.fit_model)
        self.clipboard_action_succeeded.emit("peaks_copy")
        return True

    def paste_peaks(self, apply_all: bool = False) -> bool:
        """Paste copied peak definitions onto target spectra."""
        if not hasattr(self, "_peaks_clipboard") or self._peaks_clipboard is None:
            self.notify.emit(tr("No peaks copied."))
            return False

        mds = self._get_target_mds(apply_all)
        if not mds:
            if not apply_all:
                self.notify.emit(tr("No spectrum selected."))
            return False

        for md in mds:
            if md:
                md.fit_model = deepcopy(self._peaks_clipboard)
                self._reconstruct_y_peaks(md)
                self._on_map_data_changed(md, "paste_peaks")

        self._post_bulk_action(apply_all, "paste_peaks")
        self.clipboard_action_succeeded.emit("peaks_paste")
        return True

    def delete_peaks(self, apply_all: bool = False):
        """Delete all peaks and fit results on target spectra."""
        mds = self._get_target_mds(apply_all)
        if not mds:
            if not apply_all:
                self.notify.emit(tr("No spectrum selected."))
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

        n_rows = md.Y0.shape[0] if md.Y0 is not None else 1

        # 1. Clean reset
        if indices is None:
            self._reset_mapdata_arrays(md)
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
                    md.Y[indices] = md.Y0[indices, i_min : i_max + 1].copy()
                else:
                    md.Y[indices] = md.Y0[indices].copy()

            if not isinstance(md.is_baseline_subtracted, np.ndarray):
                md.is_baseline_subtracted = np.full(
                    n_rows, bool(md.is_baseline_subtracted), dtype=bool
                )
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
            curr_x = md.x_axis
            curr_y = md.y_matrix

            i_min = closest_index(curr_x, xmin)
            i_max = closest_index(curr_x, xmax)
            if i_min > i_max:
                i_min, i_max = i_max, i_min

            md.x = curr_x[i_min : i_max + 1].copy()
            md.Y = curr_y[:, i_min : i_max + 1].copy()
            md.range_min = float(md.x[0])
            md.range_max = float(md.x[-1])

        # Recalculate processed wavenumber axis length M
        m_cols = (
            md.Y.shape[1]
            if md.Y is not None
            else (md.Y0.shape[1] if md.Y0 is not None else 0)
        )

        # 3. Apply baseline config
        bl_info = fit_model.get("baseline")
        if bl_info and bl_info.get("mode"):
            md.baseline_config = deepcopy(bl_info)
            x_arr = md.x_axis
            y_arr = md.y_matrix

            if indices is None:
                md.Y_baseline = eval_baseline_batch(x_arr, y_arr, md.baseline_config)

                if bl_info.get("is_subtracted", False):
                    md.Y = (md.Y - md.Y_baseline).astype(np.float32, copy=False)
                    md.Y_baseline = None
                    md.is_baseline_subtracted = True
            else:
                y_baseline_sel = eval_baseline_batch(
                    x_arr, y_arr[indices], md.baseline_config
                )
                if md.Y_baseline is None:
                    md.Y_baseline = np.zeros((n_rows, m_cols), dtype=np.float32)
                md.Y_baseline[indices] = y_baseline_sel

                if bl_info.get("is_subtracted", False):
                    md.Y[indices] = (md.Y[indices] - md.Y_baseline[indices]).astype(
                        np.float32, copy=False
                    )
                    md.Y_baseline[indices] = 0.0
                    md.is_baseline_subtracted[indices] = True

        # 4. Attach fit model dictionary
        md.fit_model = deepcopy(fit_model)

        # 5. Reconstruct peak curves for preview
        self._reconstruct_y_peaks(md, indices=indices)

    def _sync_fit_model_metadata(self, md):
        """Write current crop range, baseline state, and fit settings into md.fit_model."""
        md.fit_model["range_min"] = md.range_min
        md.fit_model["range_max"] = md.range_max
        if md.baseline_config:
            is_sub = getattr(md, "is_baseline_subtracted", False)
            is_sub_val = (
                bool(is_sub[0]) if isinstance(is_sub, np.ndarray) else bool(is_sub)
            )
            md.fit_model["baseline"] = deepcopy(md.baseline_config)
            md.fit_model["baseline"]["is_subtracted"] = is_sub_val
        else:
            md.fit_model["baseline"] = None
        md.fit_model["fit_params"] = self.settings.load_fit_settings()

    def copy_fit_model(self) -> bool:
        """Copy full fit model (preprocessing + peaks + settings) to internal clipboard."""
        if not self.selected_fnames:
            self.notify.emit(tr("No spectrum selected."))
            return False

        md = self._get_primary_selected_map_data()
        if not md or not md.fit_model:
            self.notify.emit(tr("No fit results to copy."))
            return False

        self._sync_fit_model_metadata(md)

        self._fitmodel_clipboard = deepcopy(build_clean_fit_model(md.fit_model))
        self.clipboard_action_succeeded.emit("fitmodel_copy")
        return True

    def paste_fit_model(self, apply_all: bool = False) -> bool:
        """Paste the copied fit model onto target spectra and launch fitting."""
        if not hasattr(self, "_fitmodel_clipboard") or self._fitmodel_clipboard is None:
            self.notify.emit(tr("No fit model copied."))
            return False

        self._fitmodel_clipboard["fit_params"] = self.settings.load_fit_settings()

        fnames = (
            self._get_active_spectra() if apply_all else self._get_selected_spectra()
        )
        if not fnames:
            self.notify.emit(tr("No spectrum selected."))
            return False

        for md in self._get_unique_map_data(fnames):
            self._apply_fit_model_to_mapdata(md, self._fitmodel_clipboard)

        unique_fnames = [md.name for md in self._get_unique_map_data(fnames)]
        self._run_fit_thread(deepcopy(self._fitmodel_clipboard), unique_fnames)
        self.clipboard_action_succeeded.emit("fitmodel_paste")
        return True

    def save_fit_model(self):
        """Save the primary spectrum's fit model to a JSON file."""
        if not self.selected_fnames:
            self.notify.emit(tr("No spectrum selected."))
            return

        md = self._get_primary_selected_map_data()
        if not md or not md.fit_model:
            self.notify.emit(tr("No fit model to save."))
            return

        self._sync_fit_model_metadata(md)

        default_dir = ""
        if hasattr(self, "_vm_fit_model_builder") and self._vm_fit_model_builder is not None:
            model_path = self._vm_fit_model_builder.get_current_model_path()
            if model_path is not None:
                default_dir = str(model_path)

        path, _ = QFileDialog.getSaveFileName(
            None,
            tr("Save Fit Model"),
            default_dir,
            tr("JSON Files (*.json)"),
        )

        if not path:
            return

        clean_model = build_clean_fit_model(md.fit_model)

        def default_encoder(obj):
            if hasattr(obj, "tolist"):
                return obj.tolist()
            raise TypeError(
                f"Object of type {obj.__class__.__name__} is not JSON serializable"
            )

        with open(path, "w", encoding="utf-8") as f:
            json.dump({"0": clean_model}, f, indent=2, default=default_encoder)

        self.notify.emit(tr("Fit model saved successfully."))

    def _load_selected_fit_model_dict(self) -> dict | None:
        """Load the fit model dictionary currently selected in FitModelBuilder."""
        if not hasattr(self, "_vm_fit_model_builder") or self._vm_fit_model_builder is None:
            self.notify.emit(tr("Fit model manager not connected."))
            return None

        model_path = self._vm_fit_model_builder.get_current_model_path()
        if model_path is None or not model_path.exists():
            self.notify.emit(tr("No fit model selected."))
            return None

        try:
            with open(model_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("0", {})
        except Exception as e:
            self.notify.emit(tr("Failed to load fit model: {error}", error=e))
            return None

    def apply_fit_model(self, apply_all: bool = False):
        """Apply the currently selected FitModelBuilder JSON model and fit."""
        fit_model = self._load_selected_fit_model_dict()
        if fit_model is None:
            return

        fnames = (
            self._get_active_spectra() if apply_all else self._get_selected_spectra()
        )
        if not fnames:
            self.notify.emit(tr("No spectrum selected."))
            return

        for md in self._get_unique_map_data(fnames):
            self._apply_fit_model_to_mapdata(md, fit_model)

        unique_fnames = [md.name for md in self._get_unique_map_data(fnames)]
        self._run_fit_thread(fit_model, unique_fnames)

    def fit(self, apply_all: bool = False):
        """Fitting action for selected spectra with their current individual models."""
        if self._is_fitting:
            self.notify.emit(tr("Fit already in progress. Please wait..."))
            return

        spectra_fnames = (
            self._get_active_spectra() if apply_all else self._get_selected_spectra()
        )

        if not spectra_fnames:
            return

        tasks = []
        for fname in spectra_fnames:
            md = self.store.get_map_data(fname)
            if md and md.fit_model and md.fit_model.get("peak_models"):
                self._reindex_fit_model(md)
                md.fit_model["fit_params"] = self.settings.load_fit_settings()

                tasks.append({
                    "map_name": fname,
                    "indices": np.array([0]),
                    "fit_model": md.fit_model,
                })

        if not tasks:
            self.notify.emit(tr("No peaks to fit."))
            return

        self._launch_fit_thread(tasks)

    def _run_fit_thread(self, fit_model: dict, fnames):
        if self._is_fitting:
            self.notify.emit(tr("Fit already in progress. Please wait..."))
            return

        if not fnames:
            self.notify.emit(tr("No spectra selected."))
            return

        if "fit_params" not in fit_model:
            fit_model["fit_params"] = self.settings.load_fit_settings()

        tasks = [{
            "map_names": fnames,
            "indices": np.arange(len(fnames)),
            "fit_model": fit_model,
        }]

        self._launch_fit_thread(tasks)

    def _cancel_fit_thread(self):
        """Gracefully stop and dispose of the active fit thread, if any.

        Disconnects ``finished`` first so a late signal from the cancelled thread
        can't clobber a freshly started one. Safer than QThread.terminate(),
        which could kill the worker mid-write into the store.
        """
        thread = self._fit_thread
        if thread is None:
            return
        self._fit_thread = None
        try:
            thread.finished.disconnect(self._on_fit_finished)
        except (RuntimeError, TypeError):
            pass
        if thread.isRunning():
            thread.stop()
            thread.wait()
        thread.deleteLater()

    def _launch_fit_thread(self, tasks: list):
        """Cancel any running fit, then start a VBFthread for ``tasks``."""
        self._cancel_fit_thread()
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

        self._emit_selected_spectra()
        self._emit_list_update()

        if self._fit_thread:
            self._fit_thread.deleteLater()
            self._fit_thread = None

    def stop_fit(self):
        """Stop the currently running fit thread."""
        if self._fit_thread and self._fit_thread.isRunning():
            thread = self._fit_thread
            self._fit_thread = None

            thread.stop()
            thread.wait()
            thread.deleteLater()

            self._is_fitting = False
            self.fit_in_progress.emit(False)
            self.notify.emit(tr("Fitting stopped by user."))

    def set_fit_model_builder(self, vm_fit_model_builder):
        """Inject the VMFitModelBuilder dependency."""
        self._vm_fit_model_builder = vm_fit_model_builder

    def set_peak_shape(self, shape: str):
        """Receive peak shape from View."""
        self._current_peak_shape = shape

    def update_peak_label(self, index, text):
        """Update display label of peak *index* on the primary selected spectrum."""
        md = self._get_primary_selected_map_data()
        if md and md.fit_model and "peak_models" in md.fit_model:
            if "peak_labels" not in md.fit_model:
                md.fit_model["peak_labels"] = []

            while len(md.fit_model["peak_labels"]) <= index:
                md.fit_model["peak_labels"].append(
                    f"Peak{len(md.fit_model['peak_labels']) + 1}"
                )

            md.fit_model["peak_labels"][index] = text
        self._emit_selected_spectra()

    def update_peak_model(self, index, model_name):
        """Change peak shape for peak *index* while preserving center and amplitude."""
        md = self._get_primary_selected_map_data()
        if not md or not md.fit_model or str(index) not in md.fit_model.get("peak_models", {}):
            return

        self._clear_fit_state(md)

        pdict = md.fit_model["peak_models"][str(index)]
        old_shape = list(pdict.keys())[0]
        old_params = pdict[old_shape]

        x0 = old_params.get("x0", {}).get(
            "value", (md.x[0] + md.x[-1]) / 2 if md.x is not None else 0
        )
        ampli = old_params.get("ampli", {}).get(
            "value",
            old_params.get("A", {}).get(
                "value", old_params.get("A1", {}).get("value", 1.0)
            ),
        )
        fit_settings = self.settings.load_fit_settings()
        maxshift = fit_settings.get("maxshift", 20.0)
        maxfwhm = fit_settings.get("maxfwhm", 200.0)
        minfwhm = fit_settings.get("minfwhm", 0.0)

        new_params = {}

        spec_idx = 0
        if self.selected_fnames:
            fname_idx = md.get_fname_index(self.selected_fnames[0])
            if fname_idx is not None:
                spec_idx = fname_idx

        y_arr = md.y_matrix[spec_idx]
        self._initialize_peak_params(
            new_params, model_name, x0, ampli, minfwhm, maxfwhm, maxshift, y_arr
        )

        md.fit_model["peak_models"][str(index)] = {model_name: new_params}
        self._reconstruct_y_peaks(md)
        self._emit_selected_spectra()

    def update_peak_param(self, index, key, field, value):
        """Update a single peak parameter field (value/min/max/vary/expr)."""
        md = self._get_primary_selected_map_data()
        if not md or not md.fit_model:
            return

        self._clear_fit_state(md)

        pdict = md.fit_model.get("peak_models", {}).get(str(index))
        if pdict:
            shape = list(pdict.keys())[0]
            if key in pdict[shape]:
                pdict[shape][key][field] = value
        self._reconstruct_y_peaks(md)
        self._emit_selected_spectra()

    def delete_peak(self, index):
        """Delete a single peak by dictionary key *index*."""
        md = self._get_primary_selected_map_data()
        if not md or not md.fit_model:
            return

        old_models = md.fit_model.get("peak_models", {})
        if str(index) in old_models:
            del old_models[str(index)]

        self._clear_fit_state(md)
        self._reconstruct_y_peaks(md)
        self._emit_selected_spectra()

    def update_dragged_peak(self, x: float, y: float):
        """Update peak position during dragging (real-time update)."""
        if not self.selected_fnames:
            return

        md = self._get_primary_selected_map_data()
        if not md or not md.fit_model:
            return

        min_dist = float("inf")
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
            self._clear_fit_state(md)
            self._reconstruct_y_peaks(md)

        self._emit_selected_spectra()
