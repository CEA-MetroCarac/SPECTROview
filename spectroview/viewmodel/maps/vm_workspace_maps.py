"""ViewModel for Maps Workspace - extends Spectra Workspace with hyperspectral map functionality."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Signal

from spectroview.i18n import tr, tr_plural
from spectroview.model.io.m_io import (
    extract_map_arrays,
    load_l6m_map,
    load_map_file,
    load_ngc_map,
    load_spc_map,
    load_wdf_map,
)
from spectroview.model.m_settings import MSettings
from spectroview.model.spectra.spectra_store import SpectrumProxy
from spectroview.viewmodel.common.spectral_utils import parse_coords_from_fname
from spectroview.viewmodel.maps.vm_maps_fitting import MapsFittingMixin
from spectroview.viewmodel.maps.vm_maps_persistence import MapsPersistenceMixin
from spectroview.viewmodel.spectra.vm_workspace_spectra import VMWorkspaceSpectra


class VMWorkspaceMaps(MapsFittingMixin, MapsPersistenceMixin, VMWorkspaceSpectra):
    """Maps Workspace ViewModel."""

    maps_list_changed = Signal(list)
    map_data_updated = Signal(object)
    live_maps_selection_changed = Signal(list, object)  # (selected_names, primary_name)
    send_spectra_to_workspace = Signal(list)
    clear_map_cache_requested = Signal(str)
    switch_to_graphs_tab = Signal()  # Request to switch to Graphs tab

    def __init__(self, settings: MSettings):
        super().__init__(settings)
        self.maps = {}
        self.current_map_name: str | None = None
        self.selected_map_names: list[str] = []
        self.current_map_df = None

        # Store metadata for each map (for WDF files)
        self.maps_metadata: dict[str, dict] = {}
        self.map_type = "2Dmap"

        # Reference to Graphs workspace (injected after construction)
        self.graphs_workspace = None

    def load_map_files(self, paths: list[str]):
        """Load hyperspectral map files and extract spectra."""
        loaded_maps = []
        last_valid_path = None
        suffix_loaders = {
            ".wdf": load_wdf_map,
            ".spc": load_spc_map,
            ".l6m": load_l6m_map,
            ".ngc": load_ngc_map,
            ".ngs": load_ngc_map,
        }

        for p in paths:
            path = Path(p)
            map_name = path.stem

            if map_name in self.maps:
                self.notify.emit(tr("Map '{name}' already loaded, skipping.", name=map_name))
                continue

            try:
                loader = suffix_loaders.get(path.suffix.lower())
                if loader is not None:
                    result = loader(path)
                    if isinstance(result, tuple):
                        map_df, metadata = result
                        self.maps_metadata[map_name] = metadata
                    else:
                        map_df = result
                else:
                    map_df = load_map_file(path)
                    self.maps_metadata[map_name] = {}
                self._extract_spectra_from_map(map_name, map_df)
                md = self.store.get_map_data(map_name)
                self.maps[map_name] = (
                    self._build_map_df_from_md(md) if md is not None else map_df
                )
                loaded_maps.append(map_name)
                last_valid_path = path
            except Exception as e:
                self.notify.emit(tr("Error loading {name}: {error}", name=path.name, error=e))

        if loaded_maps:
            self._emit_maps_list_update()
            self.select_map(loaded_maps[-1])

            if last_valid_path:
                self.settings.set_last_directory(str(last_valid_path.parent))

    def select_map(self, map_name: str):
        """Select a single map as Primary and synchronize spectra/map viewers."""
        if map_name not in self.maps:
            return
        self.select_maps([map_name], primary_name=map_name)

    def select_maps(self, map_names: list[str], primary_name: str | None = None):
        """Select one or more maps using the Two-Tier Selection Model.

        * ``current_map_name`` (Primary Map) drives single-map viewers and tools.
        * ``selected_map_names`` drives multi-map comparison in Live Maps.
        Changing only the comparison set while keeping ``primary_name`` fixed
        avoids resetting ``VMapViewer``, ``VSpectraViewer``, or fit tools.
        """
        valid_names = [n for n in dict.fromkeys(map_names) if n in self.maps]
        if valid_names:
            resolved_primary = (
                primary_name if primary_name in valid_names else valid_names[-1]
            )
        else:
            resolved_primary = None

        self.selected_map_names = valid_names

        if resolved_primary is None:
            had_primary = self.current_map_name is not None or bool(
                self.selected_fnames
            )
            self.current_map_name = None
            self.current_map_df = None
            self.selected_fnames = []
            if had_primary:
                self.spectra_list_changed.emit([])
                self.count_changed.emit(0)
                self.spectra_selection_changed.emit([])
                self.map_data_updated.emit(pd.DataFrame())
            self.live_maps_selection_changed.emit([], None)
            return

        if resolved_primary != self.current_map_name:
            self.current_map_name = resolved_primary
            self.current_map_df = self.maps[resolved_primary]
            self._show_map_spectra(resolved_primary)

            md = self.store.get_map_data(resolved_primary)
            if md is not None and len(md.fnames) > 0:
                self.selected_fnames = [md.fnames[0]]
                self._emit_selected_spectra()
            else:
                self.selected_fnames = []
                self.spectra_selection_changed.emit([])

            self.map_data_updated.emit(self.current_map_df)

        self.live_maps_selection_changed.emit(
            list(self.selected_map_names), self.current_map_name
        )

    def _get_unique_map_data(self, fnames: list[str]) -> list:
        """Override parent to fetch MapData of the active map for interactive tools."""
        if not self.current_map_name:
            return []
        md = self.store.get_map_data(self.current_map_name)
        return [md] if md else []

    def _emit_list_update(self):
        """Override parent to refresh the current map's spectra list."""
        if self.current_map_name:
            self._show_map_spectra(self.current_map_name)

    def _get_interactive_map_data(self):
        """Override parent to provide the MapData for the active map."""
        if not self.current_map_name:
            return None
        return self.store.get_map_data(self.current_map_name)

    def _emit_selected_spectra(self):
        """Override parent to emit tensor batch data for Maps."""
        if not self.selected_fnames:
            self.spectra_selection_changed.emit([])
            return

        if (
            self.current_map_name
            and self.store.get_map_data(self.current_map_name) is not None
        ):
            map_name = self.current_map_name
            md = self.store.get_map_data(map_name)

            idx_map = md.fname_to_index_map()
            indices = [idx_map[name] for name in self.selected_fnames if name in idx_map]

            if indices:
                indices = np.array(indices)
                self._update_fit_model_from_params(md, indices[0])
                x, y_batch = self.store.get_xy_batch(map_name, indices)

                x0 = md.x0
                y0_batch = md.Y0[indices]

                proxies = [SpectrumProxy(md, idx, md.fnames[idx]) for idx in indices]

                batch_data = {
                    "type": "tensor",
                    "x": x,
                    "y": y_batch,
                    "x0": x0,
                    "y0": y0_batch,
                    "indices": indices,
                    "fnames": [md.fnames[i] for i in indices],
                    "colors": [md.colors[i] for i in indices] if md.colors else None,
                    "labels": [md.labels[i] for i in indices] if md.labels else None,
                    "proxies": proxies,
                    "y_bestfit": (
                        md.Y_bestfit[indices] if md.Y_bestfit is not None else None
                    ),
                    "y_peaks": (
                        [p[indices] for p in md.Y_peaks]
                        if md.Y_peaks is not None
                        else None
                    ),
                    "y_baseline": (
                        md.Y_baseline[indices] if md.Y_baseline is not None else None
                    ),
                    "baseline_config": [md.baseline_config] * len(indices),
                    "fit_models": (
                        [md.fit_model] * len(indices) if md.fit_model else None
                    ),
                    "fit_r2": md.fit_r2[indices] if md.fit_r2 is not None else None,
                    "map_name": map_name,
                }

                self.spectra_selection_changed.emit(batch_data)

                if x is not None and len(x) > 0:
                    self.show_xcorrection_value.emit(md.xcorrection_value)
                    self.spectral_range_changed.emit(float(x[0]), float(x[-1]))

                return

        self.spectra_selection_changed.emit([])

    def _extract_spectra_from_map(self, map_name: str, map_df: pd.DataFrame):
        """Extract all individual spectra from a hyperspectral map dataframe."""
        x_data, y0_mat, coords, fnames = extract_map_arrays(map_name, map_df)
        map_metadata = self.maps_metadata.get(map_name, {})

        self.store.add_map(
            name=map_name,
            x0=x_data,
            Y0=y0_mat,
            coords=coords,
            fnames=fnames,
            is_active=np.ones(len(fnames), dtype=bool),
        )
        md = self.store.get_map_data(map_name)
        if md:
            md.map_metadata = map_metadata

    def _show_map_spectra(self, map_name: str):
        """Display spectra for the selected map in the spectra list."""
        md = self.store.get_map_data(map_name)
        if md is None:
            self.spectra_list_changed.emit([])
            self.count_changed.emit(0)
            return

        is_cropped = md.range_min is not None or md.range_max is not None
        has_baseline_config = md.baseline_config is not None

        if has_baseline_config and isinstance(md.is_baseline_subtracted, np.ndarray):
            baseline_status = md.is_baseline_subtracted
        else:
            baseline_status = np.full(
                md.n_spectra, bool(has_baseline_config and md.is_baseline_subtracted)
            )

        if md.peak_params is not None:
            has_fit_array = np.any(md.peak_params != 0.0, axis=1)
        else:
            has_fit_array = np.zeros(md.n_spectra, dtype=bool)

        if md.fit_success is not None:
            if isinstance(md.fit_success, np.ndarray):
                fit_success_array = md.fit_success
            else:
                fit_success_array = np.array([bool(x) for x in md.fit_success])
        else:
            fit_success_array = np.zeros(md.n_spectra, dtype=bool)

        spectra_info = [
            {
                "fname": fname,
                "is_active": bool(is_act),
                "is_cropped": is_cropped,
                "has_baseline": bool(has_bl),
                "has_fit": bool(hf),
                "fit_success": bool(fs),
            }
            for fname, is_act, has_bl, hf, fs in zip(
                md.fnames,
                md.is_active,
                baseline_status,
                has_fit_array,
                fit_success_array,
            )
        ]

        self.spectra_list_changed.emit(spectra_info)
        self.count_changed.emit(len(spectra_info))

    def get_current_map_dataframe(self) -> pd.DataFrame | None:
        """Get the DataFrame of the currently selected map (filtered by checked spectra)."""
        if not self.current_map_name or self.current_map_name not in self.maps:
            return None

        md = self.store.get_map_data(self.current_map_name)
        df = self.maps[self.current_map_name]
        if df is None and md is not None:
            df = self._build_map_df_from_md(md)
            self.maps[self.current_map_name] = df
        if md is None or not hasattr(md, "is_active") or md.is_active is None:
            return df

        if np.all(md.is_active):
            return df

        if len(md.is_active) == len(df):
            if not np.any(md.is_active):
                return df.iloc[0:0]
            return df.iloc[md.is_active]

        active_fnames = {
            fname for fname, active in zip(md.fnames, md.is_active) if active
        }
        fname_prefix = f"{self.current_map_name}_("
        active_coords = set()
        for fname in active_fnames:
            if fname.startswith(fname_prefix):
                xy = parse_coords_from_fname(fname)
                if xy is not None:
                    active_coords.add(xy)

        if not active_coords:
            return df.iloc[0:0]

        idx = pd.MultiIndex.from_arrays([df["X"], df["Y"]])
        mask = idx.isin(list(active_coords))
        return df[mask]

    def get_selected_map_names(self) -> list[str]:
        """Return ordered list of selected map names, falling back to primary map."""
        targets = [n for n in self.selected_map_names if n in self.maps]
        if not targets and self.current_map_name and self.current_map_name in self.maps:
            targets = [self.current_map_name]
        return targets

    @staticmethod
    def _sanitize_map_filename(map_name: str, used_names: set[str]) -> str:
        """Return a filesystem-safe, collision-free CSV filename for *map_name*."""
        safe_stem = "".join(
            c
            for c in str(map_name)
            if c.isalpha() or c.isdigit() or c in (" ", ".", "_", "-")
        ).rstrip(" .")
        if safe_stem.lower().endswith(".csv"):
            safe_stem = safe_stem[:-4].rstrip(" .")
        if not safe_stem:
            safe_stem = "map"

        candidate = f"{safe_stem}.csv"
        suffix_idx = 2
        while candidate.lower() in used_names:
            candidate = f"{safe_stem}_{suffix_idx}.csv"
            suffix_idx += 1
        used_names.add(candidate.lower())
        return candidate

    @staticmethod
    def _write_map_data_to_csv(md, file_path: str) -> str:
        """Write *md* (processed if available, else raw) as a SPECTROview CSV file."""
        x_axis = md.x_axis
        y_matrix = md.y_matrix
        coords = (
            md.coords
            if md.coords is not None
            else np.zeros((y_matrix.shape[0], 2), dtype=np.float64)
        )

        col_names = ["X", "Y"] + [str(float(w)) for w in x_axis]
        data = np.hstack([coords, y_matrix.astype(np.float64)])
        df = pd.DataFrame(data, columns=col_names)

        if not file_path.lower().endswith(".csv"):
            file_path += ".csv"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("SPECTROview Map Data\n")
        df.to_csv(file_path, sep=";", index=False, mode="a")
        return file_path

    def save_current_map_to_csv(self, file_path: str):
        """Save the currently selected map to a CSV file (readable by m_io.py)."""
        if not self.current_map_name or self.current_map_name not in self.maps:
            self.notify.emit(tr("No map selected."))
            return

        try:
            md = self.store.get_map_data(self.current_map_name)
            if not md:
                self.notify.emit(tr("Map data not found."))
                return

            self._write_map_data_to_csv(md, file_path)
            self.notify.emit(tr("Map '{name}' saved to CSV.", name=self.current_map_name))
        except Exception as e:
            self.notify.emit(tr("Error saving map: {error}", error=e))

    def save_selected_maps_to_csv(self, dir_path: str) -> int:
        """Save all selected maps as separate CSV files inside *dir_path*."""
        targets = self.get_selected_map_names()
        if not targets:
            self.notify.emit(tr("No map selected."))
            return 0
        if not dir_path:
            return 0

        used_names: set[str] = set()
        saved_count = 0
        for map_name in targets:
            safe_name = self._sanitize_map_filename(map_name, used_names)
            file_path = str(Path(dir_path) / safe_name)

            md = self.store.get_map_data(map_name)
            if not md:
                self.notify.emit(
                    tr("Error saving {name}: {error}", name=safe_name, error=tr("Map data not found."))
                )
                continue

            try:
                self._write_map_data_to_csv(md, file_path)
                saved_count += 1
            except Exception as e:
                self.notify.emit(tr("Error saving {name}: {error}", name=safe_name, error=e))

        if saved_count > 0:
            self.notify.emit(
                tr_plural(
                    "Successfully saved {count} map.",
                    "Successfully saved {count} maps.",
                    saved_count,
                )
            )
            self.settings.set_last_directory(dir_path)

        return saved_count

    def delete_current_map(self):
        """Delete the currently selected Primary map."""
        if not self.current_map_name:
            self.notify.emit(tr("No map selected."))
            return

        map_name = self.current_map_name
        self.remove_map(map_name)
        self.notify.emit(tr("Map '{name}' deleted.", name=map_name))

    def delete_selected_maps(self):
        """Delete all currently selected maps (batch delete when multiple selected)."""
        targets = self.get_selected_map_names()
        if not targets:
            self.notify.emit(tr("No map selected."))
            return

        if len(targets) == 1:
            self.remove_map(targets[0])
            self.notify.emit(tr("Map '{name}' deleted.", name=targets[0]))
            return

        existing_order = list(self.maps.keys())
        deleted_indices = [
            existing_order.index(n) for n in targets if n in existing_order
        ]
        min_idx = min(deleted_indices) if deleted_indices else 0

        for name in targets:
            del self.maps[name]
            self.maps_metadata.pop(name, None)
            self.store.remove_map(name)

        self.selected_map_names = []
        self.current_map_name = None
        self.current_map_df = None
        self.selected_fnames = []

        self._emit_maps_list_update()
        remaining = list(self.maps.keys())
        if remaining:
            next_name = remaining[min(min_idx, len(remaining) - 1)]
            self.select_maps([next_name], primary_name=next_name)
        else:
            self.spectra_list_changed.emit([])
            self.count_changed.emit(0)
            self.spectra_selection_changed.emit([])
            self.map_data_updated.emit(pd.DataFrame())
            self.live_maps_selection_changed.emit([], None)

        self.notify.emit(
            tr_plural("Deleted {count} map.", "Deleted {count} maps.", len(targets))
        )

    def select_all_current_map_spectra(self):
        """Select all spectra from the currently displayed map."""
        if not self.current_map_name:
            return

        md = self.store.get_map_data(self.current_map_name)
        if md:
            self.selected_fnames = list(md.fnames)
        self._emit_selected_spectra()

    def set_spectrum_active(self, fname: str, is_active: bool):
        """Toggle one spectrum's checked/active state and refresh the map view."""
        if not self.current_map_name:
            return
        md = self.store.get_map_data(self.current_map_name)
        if md is None:
            return
        idx = md.get_fname_index(fname)
        if idx is None:
            return
        if bool(md.is_active[idx]) == is_active:
            return
        md.is_active[idx] = is_active
        md.touch()
        self.clear_map_cache_requested.emit(self.current_map_name)
        self.map_data_updated.emit(self.get_current_map_dataframe())

    def set_all_current_map_spectra_active(self, is_active: bool):
        """Check/uncheck every spectrum in the current map and refresh the map view."""
        if not self.current_map_name:
            return
        md = self.store.get_map_data(self.current_map_name)
        if md is None:
            return
        md.is_active[:] = is_active
        md.touch()
        self.clear_map_cache_requested.emit(self.current_map_name)
        self.map_data_updated.emit(self.get_current_map_dataframe())

    def apply_y_normalization(self, norm_factor: float, apply_all: bool = False):
        """Override to update map visualizer after applying normalization."""
        super().apply_y_normalization(norm_factor, apply_all)
        if self.current_map_name:
            self.clear_map_cache_requested.emit(self.current_map_name)
        self.map_data_updated.emit(self.get_current_map_dataframe())

    def undo_y_normalization(self, apply_all: bool = False):
        """Override to update map visualizer after undoing normalization."""
        super().undo_y_normalization(apply_all)
        if self.current_map_name:
            self.clear_map_cache_requested.emit(self.current_map_name)
        self.map_data_updated.emit(self.get_current_map_dataframe())

    def reinit_current_map_spectra(self, apply_all: bool = False):
        """Reinitialize spectra and emit map update."""
        super().reinit_spectra(apply_all)
        if self.current_map_name:
            self.clear_map_cache_requested.emit(self.current_map_name)
        self.map_data_updated.emit(self.get_current_map_dataframe())

    # ═════════════════════════════════════════════════════════════════════
    # Hooks for Bulk Operations (Template Method Pattern) inherited from Spectra
    # ═════════════════════════════════════════════════════════════════════

    def _get_target_mds(self, apply_all: bool) -> list:
        """Override to operate on the entire map instead of selected spectra."""
        maps_to_target = (
            self.store.map_names
            if apply_all
            else ([self.current_map_name] if self.current_map_name else [])
        )
        return [
            self.store.get_map_data(name)
            for name in maps_to_target
            if self.store.get_map_data(name)
        ]

    def _on_map_data_changed(self, md, action: str):
        """Override to execute map-specific cache clearing and preprocessing updates."""
        if action == "reinit_spectra":
            self.store.clear_preprocess(md.name)
        elif action in (
            "delete_baseline",
            "subtract_baseline",
            "apply_spectral_range",
        ):
            self.store.batch_preprocess(
                md.name, md.baseline_config, md.range_min, md.range_max
            )

        self.clear_map_cache_requested.emit(md.name)

    def _post_bulk_action(self, apply_all: bool, action: str):
        """Override to refresh fit results and map view."""
        if apply_all:
            if action in ("delete_peaks", "delete_baseline", "reinit_spectra"):
                self.df_fit_results = None
                self.fit_results_updated.emit(None)
        else:
            if action in (
                "delete_peaks",
                "delete_baseline",
                "paste_peaks",
                "reinit_spectra",
            ):
                self.collect_fit_results()

        if self.current_map_name:
            self._show_map_spectra(self.current_map_name)
            self._emit_selected_spectra()

    def validate_cosmic_ray_erase(self):
        """Override: commit erasures and refresh the map heatmap cache."""
        super().validate_cosmic_ray_erase()

        if self.current_map_name:
            self.clear_map_cache_requested.emit(self.current_map_name)
            self.map_data_updated.emit(self.get_current_map_dataframe())

    def send_selected_spectra_to_spectra_workspace(self):
        """Send selected spectra to the Spectra workspace tab for comparison."""
        if not self.selected_fnames:
            self.notify.emit(tr("No spectrum selected."))
            return

        md = self.store.get_map_data(self.current_map_name)
        if not md:
            return

        fname_set = set(self.selected_fnames)
        indices = [i for i, f in enumerate(md.fnames) if f in fname_set]

        spectra_payloads = []
        for i in indices:
            fname = md.fnames[i]
            is_sub = (
                md.is_baseline_subtracted[i]
                if isinstance(md.is_baseline_subtracted, np.ndarray)
                else bool(md.is_baseline_subtracted)
            )

            spectra_payloads.append({
                "name": fname,
                "x0": md.x0.copy(),
                "Y0": md.Y0[i : i + 1].copy(),
                "baseline_config": deepcopy(md.baseline_config),
                "fit_model": deepcopy(md.fit_model) if md.fit_model else None,
                "range_min": md.range_min,
                "range_max": md.range_max,
                "is_subtracted": is_sub,
            })

        if spectra_payloads:
            self.send_spectra_to_workspace.emit(spectra_payloads)
            self.notify.emit(
                tr_plural(
                    "Sent {count} spectrum to Spectra workspace.",
                    "Sent {count} spectra to Spectra workspace.",
                    len(spectra_payloads),
                )
            )

    def remove_map(self, map_name: str):
        """Remove a map and its spectra from the loaded maps."""
        if map_name not in self.maps:
            return

        old_order = list(self.maps.keys())
        old_idx = old_order.index(map_name)
        del self.maps[map_name]
        self.maps_metadata.pop(map_name, None)
        self.store.remove_map(map_name)
        self.selected_map_names = [
            n for n in self.selected_map_names if n != map_name
        ]

        was_primary = self.current_map_name == map_name
        if was_primary:
            self.current_map_name = None
            self.current_map_df = None
            self.selected_fnames = []

        self._emit_maps_list_update()
        remaining = list(self.maps.keys())

        if remaining and (was_primary or not self.selected_map_names):
            new_sel = (
                self.selected_map_names
                if self.selected_map_names
                else [remaining[min(old_idx, len(remaining) - 1)]]
            )
            next_primary = new_sel[-1] if was_primary else self.current_map_name
            self.select_maps(new_sel, primary_name=next_primary)
        elif not remaining:
            self.spectra_list_changed.emit([])
            self.count_changed.emit(0)
            self.spectra_selection_changed.emit([])
            self.map_data_updated.emit(pd.DataFrame())
            self.live_maps_selection_changed.emit([], None)
        else:
            self.live_maps_selection_changed.emit(
                list(self.selected_map_names), self.current_map_name
            )

    def _emit_maps_list_update(self):
        """Emit updated list of map names."""
        self.maps_list_changed.emit(list(self.maps.keys()))

    def reorder_maps(self, new_order: list[int]):
        """Reorder loaded maps according to a visual index permutation."""
        keys = list(self.maps.keys())
        if len(new_order) != len(keys) or sorted(new_order) != list(range(len(keys))):
            return
        if new_order == list(range(len(keys))):
            return

        ordered_names = [keys[idx] for idx in new_order]
        self.maps = {name: self.maps[name] for name in ordered_names}

        store_names = self.store.map_names
        if store_names == keys:
            self.store.reorder_maps(new_order)
        elif len(store_names) == len(ordered_names) and set(store_names) == set(
            ordered_names
        ):
            store_index = {name: i for i, name in enumerate(store_names)}
            self.store.reorder_maps([store_index[name] for name in ordered_names])

        selected_set = set(self.selected_map_names)
        self.selected_map_names = [
            name for name in ordered_names if name in selected_set
        ]

        self._emit_maps_list_update()
        if self.selected_map_names or self.current_map_name:
            self.live_maps_selection_changed.emit(
                list(self.selected_map_names), self.current_map_name
            )

    def _extract_coords_for_spectra(self, spectra):
        """Extract (X, Y) spatial coordinates for a list of map spectra."""
        coords = []
        for spectrum in spectra:
            xy = parse_coords_from_fname(spectrum.fname)
            if xy is None:
                return None
            coords.append(list(xy))

        return np.array(coords, dtype=np.float64) if coords else None

    def _get_active_spectra(self):
        """Override: Return active spectra fnames for the CURRENT map only."""
        if not self.current_map_name:
            return []
        md = self.store.get_map_data(self.current_map_name)
        if not md:
            return []
        return [f for i, f in enumerate(md.fnames) if md.is_active[i]]

    def _get_selected_spectra(self):
        """Override: Return selected fnames that are active for the CURRENT map."""
        if not self.current_map_name:
            return []
        md = self.store.get_map_data(self.current_map_name)
        if not md:
            return []
        fname_set = set(self.selected_fnames)
        return [
            f for i, f in enumerate(md.fnames) if md.is_active[i] and f in fname_set
        ]

    def clear_workspace(self):
        """Clear all maps, spectra, and reset workspace to initial state."""
        self.maps.clear()
        self.maps_metadata.clear()
        self.current_map_name = None
        self.selected_map_names = []
        self.current_map_df = None

        super().clear_workspace()

        if hasattr(self, "store"):
            self.store.clear()

        self.maps_list_changed.emit([])
        self.spectra_list_changed.emit([])
        self.map_data_updated.emit(pd.DataFrame())
        self.live_maps_selection_changed.emit([], None)
        self.count_changed.emit(0)

    def set_graphs_workspace(self, graphs_workspace):
        """Inject Graphs workspace reference for cross-workspace communication."""
        self.graphs_workspace = graphs_workspace

    def extract_and_send_profile_to_graphs(
        self, profile_name: str, profile_df: pd.DataFrame
    ):
        """Extract profile data and send to Graphs workspace for plotting."""
        if profile_df is None or profile_df.empty:
            self.notify.emit(
                tr(
                    "No profile data to extract. Select exactly 2 points in 2D map mode."
                )
            )
            return

        if not profile_name or profile_name.strip() == "":
            self.notify.emit(tr("Please provide a name for the profile."))
            return

        if self.graphs_workspace is None:
            self.notify.emit(tr("Graphs workspace not available."))
            return

        self.graphs_workspace.vm.add_dataframe(profile_name, profile_df)

        plot_config = {
            "df_name": profile_name,
            "plot_style": "line",
            "x": "distance",
            "y": ["values"],
            "z": None,
            "filters": [],
        }

        success = self.graphs_workspace.create_plot_from_config(
            profile_name, plot_config
        )

        if success:
            self.switch_to_graphs_tab.emit()
            self.notify.emit(
                tr("Profile '{name}' sent to Graphs workspace", name=profile_name)
            )
        else:
            self.notify.emit(tr("Failed to create profile plot."))
