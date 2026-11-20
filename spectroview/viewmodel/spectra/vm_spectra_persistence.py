"""Workspace serialization (save/load and legacy v1 .spectra loader) mixin."""

from __future__ import annotations

import base64
import json
import traceback
import zlib

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QFileDialog

from spectroview.fit_engine.baseline import eval_baseline_batch
from spectroview.i18n import tr
from spectroview.model.io.workspace_io import WorkspaceIO
from spectroview.model.spectra.spectra_store import SpectraStore


class SpectraPersistenceMixin:
    """Save/load SpectraStore-backed workspaces and restore preprocessed state."""

    @staticmethod
    def _build_map_df_from_md(md) -> pd.DataFrame:
        """Build a legacy [X, Y, <wavenumbers...>] DataFrame from a MapData object."""
        col_names = list(map(str, md.x0))
        df = pd.DataFrame(md.Y0, columns=col_names)
        coords = (
            md.coords
            if md.coords is not None
            else np.zeros((md.n_spectra, 2), dtype=np.float64)
        )
        df.insert(0, "Y", coords[:, 1])
        df.insert(0, "X", coords[:, 0])
        return df

    def _restore_preprocessed_state(self, md):
        """Restore cropped, baseline-subtracted, and peak states for MapData."""
        if (
            not md.baseline_config
            and md.fit_model
            and isinstance(md.fit_model, dict)
            and "baseline" in md.fit_model
        ):
            md.baseline_config = md.fit_model["baseline"]

        # 1. Apply range cropping
        if md.range_min is not None or md.range_max is not None:
            mask = np.logical_and(
                md.x0 >= (md.range_min if md.range_min is not None else -np.inf),
                md.x0 <= (md.range_max if md.range_max is not None else np.inf),
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
        if md.Y_bestfit is None and md.fit_model and md.fit_model.get("peak_models"):
            self._reconstruct_y_peaks(md, include_bestfit=True)

    def save_work(self):
        """Save workspace using high-performance ZIP format (SpectraStore-backed)."""
        is_maps = hasattr(self, "maps")
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            tr("Save work"),
            "",
            tr("SPECTROview Maps (*.maps)") if is_maps else tr("SPECTROview Files (*.spectra)"),
        )
        if not file_path:
            return

        try:
            arrays = {}
            store_meta = {}
            for map_name in self.store.map_names:
                arrays.update(self.store.to_npz_dict(map_name))
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
                "format_version": 2,
                "store_meta": _sanitize_for_json(store_meta),
            }

            if is_maps:
                metadata["maps_metadata"] = _sanitize_for_json(
                    getattr(self, "maps_metadata", {})
                )
                metadata["map_type"] = getattr(self, "map_type", "2Dmap")

            dataframes = {}
            if self.df_fit_results is not None and not self.df_fit_results.empty:
                dataframes["df_fit_results"] = self.df_fit_results

            WorkspaceIO.save_workspace(file_path, metadata, arrays, dataframes)
            self.notify.emit(tr("Work saved successfully."))

        except Exception as e:
            traceback.print_exc()
            self.notify.emit(tr("Error saving workspace: {error}", error=e))

    def load_work(self, file_path: str):
        """Load previously saved workspace from legacy JSON (v1) or newest ZIP-streaming (v2)."""
        try:
            metadata, arrays, dataframes, is_legacy = WorkspaceIO.load_workspace(
                file_path
            )

            if is_legacy or metadata.get("format_version", 1) < 2:
                self.clear_workspace()
                if hasattr(self, "maps"):
                    self._load_legacy_maps(file_path)
                else:
                    self._load_legacy_spectra(file_path)
                return

            self.clear_workspace()

            # 1. Restore SpectraStore from arrays + per-map metadata
            self.store = SpectraStore()
            store_meta = metadata.get("store_meta", {})
            for map_name, meta in store_meta.items():
                SpectraStore.load_map_from_npz(arrays, meta, map_name, store=self.store)
                md = self.store.get_map_data(map_name)
                if md:
                    self._restore_preprocessed_state(md)

            # 2. If Maps workspace, rebuild legacy DataFrames for heatmap rendering
            if hasattr(self, "maps"):
                self.maps = {}
                self.maps_metadata = metadata.get("maps_metadata", {})
                self.map_type = metadata.get("map_type", "2Dmap")

                for map_name in self.store.map_names:
                    md = self.store.get_map_data(map_name)
                    self.maps[map_name] = self._build_map_df_from_md(md)

            # 3. Restore DataFrames
            self.df_fit_results = (
                dataframes.get("df_fit_results") if dataframes else None
            )

            # 4. Refresh View
            if hasattr(self, "maps"):
                map_names = self.store.map_names
                if map_names:
                    self.select_map(map_names[0])
                self.maps_list_changed.emit(map_names)
                self.count_changed.emit(
                    sum(md.n_spectra for md in self.store._maps.values())
                )
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

            self.notify.emit(tr("Work loaded successfully."))

        except Exception as e:
            traceback.print_exc()
            self.notify.emit(tr("Error loading workspace: {error}", error=e))

    def _load_legacy_spectra(self, file_path: str):
        """Load legacy JSON-based .spectra workspace (OLD version)."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.store = SpectraStore()
        self.selected_fnames = []

        def decompress_array(base64_str, dtype=np.float64):
            if not base64_str:
                return None
            decoded = base64.b64decode(base64_str.encode("utf-8"))
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
                is_active=np.array([is_active], dtype=bool),
            )

            md = self.store.get_map_data(fname)
            if md:
                md.map_metadata = sdata.get("metadata", {})
                md.colors = [sdata.get("color")]
                md.labels = [sdata.get("label")]
                md.xcorrection_value = float(sdata.get("xcorrection_value", 0.0))
                md.intensity_norm_factor = float(
                    sdata.get("intensity_norm_factor", 1.0)
                )
                md.is_baseline_subtracted = sdata.get("is_baseline_subtracted", False)
                md.range_min = sdata.get("range_min")
                md.range_max = sdata.get("range_max")

                if md.range_min is not None or md.range_max is not None:
                    mask = np.logical_and(
                        md.x0 >= md.range_min,
                        md.x0 <= md.range_max,
                    )
                    md.x = md.x0[mask].copy()
                    md.Y = md.Y0[:, mask].copy()
                else:
                    md.x = md.x0.copy()
                    md.Y = md.Y0.copy()

                legacy_bl = sdata.get("baseline")
                if legacy_bl:
                    md.baseline_config = {
                        "mode": legacy_bl.get("mode", "Linear"),
                        "points": legacy_bl.get("points", [[], []]),
                        "attached": legacy_bl.get("attached", False),
                        "coef": legacy_bl.get("coef", 5),
                    }
                    md.Y_baseline = eval_baseline_batch(
                        md.x, md.Y, md.baseline_config
                    )

                if md.is_baseline_subtracted and md.Y_baseline is not None:
                    md.Y = md.Y - md.Y_baseline
                    md.Y_baseline = None

                legacy_peaks = sdata.get("peak_models")
                legacy_labels = sdata.get("peak_labels", [])
                if legacy_peaks:
                    md.fit_model = {
                        "peak_labels": legacy_labels,
                        "peak_models": legacy_peaks,
                    }
                    self._reconstruct_y_peaks(md, include_bestfit=True)
                    md.fit_success = np.array(
                        [sdata.get("result_fit_success", False)], dtype=bool
                    )

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
        self.notify.emit(tr("Legacy spectra workspace loaded successfully."))
