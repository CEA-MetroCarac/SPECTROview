"""ViewModel for Spectra Workspace - handles business logic and data management."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog

from spectroview.model.io.m_io import (
    load_l6s_spectrum,
    load_ngs_spectrum,
    load_spc_spectrum,
    load_spectrum_file,
    load_TRPL_data,
    load_wdf_spectrum,
)
from spectroview.model.m_settings import MSettings
from spectroview.model.spectra.spectra_store import SpectraStore, SpectrumProxy
from spectroview.viewmodel.spectra.vm_spectra_fitting import SpectraFittingMixin
from spectroview.viewmodel.spectra.vm_spectra_persistence import SpectraPersistenceMixin
from spectroview.viewmodel.spectra.vm_spectra_preprocessing import (
    SpectraPreprocessingMixin,
)
from spectroview.viewmodel.spectra.vm_spectra_results import SpectraResultsMixin


class VMWorkspaceSpectra(
    QObject,
    SpectraPreprocessingMixin,
    SpectraFittingMixin,
    SpectraResultsMixin,
    SpectraPersistenceMixin,
):
    """Base ViewModel for Spectra Workspace (subclassed by VMWorkspaceMaps)."""

    # ───── ViewModel → View signals ─────
    spectra_list_changed = Signal(list)  # list[str]
    spectra_selection_changed = Signal(object)  # dict (tensor batch) or empty list
    count_changed = Signal(int)
    show_xcorrection_value = Signal(float)  # ΔX of first selected spectrum
    spectral_range_changed = Signal(float, float)

    fit_in_progress = Signal(bool)  # Enable/disable fit buttons
    fit_progress_updated = Signal(
        int, int, int, float, int
    )  # (converged, total, pct, elapsed, converged_count)
    fit_timings_ready = Signal(str)  # Tooltip with timings

    # Fit results signals
    fit_results_updated = Signal(object)  # pd.DataFrame
    split_parts_updated = Signal(list)  # list[str] for combobox
    send_df_to_graphs = Signal(
        str, object, bool
    )  # (df_name, pd.DataFrame, force_replace)

    clipboard_action_succeeded = Signal(
        str
    )  # "baseline_copy", "baseline_paste", "peaks_copy", "peaks_paste", "fitmodel_copy", "fitmodel_paste"

    notify = Signal(str)  # general notifications

    def __init__(self, settings: MSettings):
        super().__init__()
        self.settings = settings
        self.store = SpectraStore()

        self.selected_fnames = []
        self._baseline_clipboard = None
        self._peaks_clipboard = None
        self._current_peak_shape = "Lorentzian"
        self._fit_thread = None
        self._is_fitting = False

        # Fit results data
        self.df_fit_results = None
        self._fitmodel_clipboard = None
        self.results_decimals = 3

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

    def _get_active_spectra(self) -> list:
        """Get list of active spectra fnames (where is_active is True)."""
        active = []
        for name in self.store.map_names:
            md = self.store.get_map_data(name)
            if md and md.is_active[0]:
                active.append(name)
        return active

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

    def _get_primary_selected_map_data(self):
        """Return MapData for the first active selected spectrum, or None if unavailable."""
        selected = self._get_selected_spectra()
        if not selected:
            return None
        mds = self._get_unique_map_data(selected[:1])
        return mds[0] if mds else None

    def _get_interactive_map_data(self):
        """Helper to fetch MapData for interactive tools in Spectra Workspace."""
        return self._get_primary_selected_map_data()

    # ═════════════════════════════════════════════════════════════════════
    # Hooks for Bulk Operations (Template Method Pattern)
    # ═════════════════════════════════════════════════════════════════════

    def _get_target_mds(self, apply_all: bool) -> list:
        """Get the target MapData objects for bulk operations.

        - apply_all=True: returns MapData objects for all *checked* (is_active) spectra only.
        - apply_all=False: returns unique MapData objects corresponding to selected spectra.
        """
        if apply_all:
            active_fnames = self._get_active_spectra()
            return [self.store.get_map_data(name) for name in active_fnames]
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
                self.notify.emit(
                    f"Spectrum '{path.name}' already loaded (name match), skipping."
                )
                continue

            try:
                if path.suffix.lower() == ".dat":
                    data_dict = load_TRPL_data(path)
                elif path.suffix.lower() == ".wdf":
                    data_dict = load_wdf_spectrum(path)
                elif path.suffix.lower() == ".spc":
                    data_dict = load_spc_spectrum(path)
                elif path.suffix.lower() == ".l6s":
                    data_dict = load_l6s_spectrum(path)
                elif path.suffix.lower() in (".ngs", ".ngc"):
                    data_dict = load_ngs_spectrum(path)
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
                            is_active=np.array([True], dtype=bool),
                        )
                        md = self.store.get_map_data(fname)
                        if md:
                            md.map_metadata = metadata
                            md.map_metadata["source_path"] = resolved_path

                loaded_files.append(p)
                last_valid_path = path
            except Exception as e:
                self.notify.emit(f"Error loading {path.name}: {e}")

        if loaded_files:
            self._emit_list_update()

            if last_valid_path:
                self.settings.set_last_directory(str(last_valid_path.parent))

    def set_selected_indices(self, indices: list[int]):
        """Set currently selected spectra by list widget indices."""
        fnames = []
        map_names = self.store.map_names
        for idx in indices:
            if 0 <= idx < len(map_names):
                fnames.append(map_names[idx])

        self.selected_fnames = list(dict.fromkeys(fnames))
        self._emit_selected_spectra()

    def set_selected_fnames(self, fnames: list[str]):
        """Set currently selected spectra by their fnames directly."""
        self.selected_fnames = list(dict.fromkeys(fnames))
        self._emit_selected_spectra()

    def set_spectrum_active(self, fname: str, is_active: bool):
        """Toggle one spectrum's checked/active state (Spectra: N=1 per block)."""
        md = self.store.get_map_data(fname)
        if md is not None:
            md.is_active[0] = is_active

    def set_all_spectra_active(self, is_active: bool):
        """Check/uncheck every loaded spectrum."""
        for name in self.store.map_names:
            md = self.store.get_map_data(name)
            if md is not None:
                md.is_active[0] = is_active

    def receive_spectra(self, payloads: list):
        """Ingest spectra payloads sent from another workspace into this store.

        Wired in main.py to VMWorkspaceMaps.send_spectra_to_workspace so the
        Maps → Spectra transfer stays within the ViewModel layer.
        """
        added = 0
        for data in payloads:
            name = data["name"]
            if name in self.store.map_names:
                continue
            self.store.add_map(
                name=name,
                x0=data["x0"],
                Y0=data["Y0"],
                coords=np.array([[0.0, 0.0]], dtype=np.float64),
                fnames=[name],
                is_active=np.array([True], dtype=bool),
            )
            md = self.store.get_map_data(name)
            if md:
                md.baseline_config = data["baseline_config"]
                raw_fm = data.get("fit_model")
                md.fit_model = (
                    raw_fm if isinstance(raw_fm, dict) and raw_fm else None
                )
                md.range_min = data["range_min"]
                md.range_max = data["range_max"]
                if data["is_subtracted"]:
                    md.is_baseline_subtracted = True
                self.store.batch_preprocess(
                    name, md.baseline_config, md.range_min, md.range_max
                )
            added += 1
        if added:
            self._emit_list_update()

    def _emit_selected_spectra(self):
        """Prepare and emit data for plotting the selected spectra."""
        active_fnames = self._get_selected_spectra()

        if not active_fnames:
            self.spectra_selection_changed.emit([])
            return

        x_list = []
        y_list = []
        x0_list = []
        y0_list = []
        colors = []
        labels = []
        bl_params = []
        pk_params = []
        y_bestfit_list = []
        y_peaks_list = []
        y_baseline_list = []
        fit_model_list = []
        fit_r2_list = []
        proxies = []

        for fname in active_fnames:
            md = self.store.get_map_data(fname)
            if not md:
                continue

            self._update_fit_model_from_params(md, 0)

            x_list.append(md.x_axis)
            y_list.append(md.y_matrix[0])
            x0_list.append(md.x0)
            y0_list.append(md.Y0[0])
            colors.append(md.colors[0])
            labels.append(md.labels[0] or fname)
            bl_params.append(md.baseline_config)
            pk_params.append(md.peak_params)
            y_bestfit_list.append(md.Y_bestfit[0] if md.Y_bestfit is not None else None)
            y_peaks_list.append(
                [p[0] for p in md.Y_peaks] if md.Y_peaks is not None else None
            )
            y_baseline_list.append(
                md.Y_baseline[0] if md.Y_baseline is not None else None
            )
            fit_model_list.append(md.fit_model)
            fit_r2_list.append(md.fit_r2[0] if md.fit_r2 is not None else 0.0)
            proxies.append(SpectrumProxy(md, 0, fname))

        data = {
            "type": "tensor_list",
            "x": x_list,
            "y": y_list,
            "x0": x0_list,
            "y0": y0_list,
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
            "fit_r2": fit_r2_list,
        }

        self.spectra_selection_changed.emit(data)

        md0 = self.store.get_map_data(active_fnames[0])
        xcorr = md0.xcorrection_value if md0 else 0.0
        self.show_xcorrection_value.emit(xcorr)

        if md0:
            x_axis = md0.x_axis
            xmin = float(x_axis[0])
            xmax = float(x_axis[-1])
            self.spectral_range_changed.emit(xmin, xmax)

    def _emit_list_update(self):
        """Emit updated list of spectra dicts and count."""
        spectra_info = []
        for name in self.store.map_names:
            md = self.store.get_map_data(name)
            if not md:
                continue

            is_cropped = md.range_min is not None or md.range_max is not None
            has_baseline = md.baseline_config is not None

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
                "metadata": getattr(md, "map_metadata", {}),
            })

        self.spectra_list_changed.emit(spectra_info)
        self.count_changed.emit(len(self.store.map_names))

    def reorder_spectra(self, new_order: list[int]):
        """Reorder loaded spectra according to visual index permutation."""
        self.store.reorder_maps(new_order)
        self._emit_list_update()
        self._emit_selected_spectra()

    def remove_selected_spectra(self):
        """Remove currently selected spectra."""
        if not self.selected_fnames:
            self.notify.emit("No spectra selected.")
            return

        map_names = self.store.map_names
        indices_to_remove = [
            idx for idx, name in enumerate(map_names) if name in self.selected_fnames
        ]

        if not indices_to_remove:
            return

        min_removed_idx = min(indices_to_remove)

        for name in self.selected_fnames:
            self.store.remove_map(name)

        new_count = len(self.store.map_names)
        self._emit_list_update()

        if new_count == 0:
            self.selected_fnames = []
            self.spectra_selection_changed.emit([])
            return

        new_index = min(min_removed_idx, new_count - 1)
        self.selected_fnames = [self.store.map_names[new_index]]
        self._emit_selected_spectra()

    def clear_workspace(self):
        """Clear all spectra and reset workspace to initial state."""
        self.store = SpectraStore()
        self.selected_fnames = []
        self.df_fit_results = None

        self._baseline_clipboard = None
        self._peaks_clipboard = None
        self._fitmodel_clipboard = None

        self._cancel_fit_thread()
        self._is_fitting = False
        self.fit_in_progress.emit(False)
        self.fit_progress_updated.emit(0, 0, 0, 0.0, 0)

        self.show_xcorrection_value.emit(0.0)
        self.spectral_range_changed.emit(0.0, 0.0)

        self._emit_list_update()
        self._emit_selected_spectra()
        self.fit_results_updated.emit(None)

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
            last_dir,
        )

        if not dir_path:
            return

        saved_count = 0
        for fname in fnames:
            base_name = str(fname)
            safe_name = "".join(
                c
                for c in base_name
                if c.isalpha() or c.isdigit() or c in (" ", ".", "_", "-")
            ).rstrip()
            if not safe_name.lower().endswith(".txt"):
                safe_name += ".txt"

            file_path = os.path.join(dir_path, safe_name)

            try:
                md = self.store.get_map_data(fname)
                if md:
                    local_idx = md.get_fname_index(fname)
                    if local_idx is None:
                        continue
                    x = md.x_axis
                    y = md.y_matrix[local_idx]
                    data = np.column_stack((x, y))
                    np.savetxt(file_path, data, fmt="%.6f", delimiter="\t", comments="")
                    saved_count += 1
            except Exception as e:
                self.notify.emit(f"Error saving {safe_name}: {e}")

        if saved_count > 0:
            self.notify.emit(f"Successfully saved {saved_count} spectra.")
            self.settings.set_last_directory(dir_path)
