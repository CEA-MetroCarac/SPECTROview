"""ViewModel for Maps Workspace - extends Spectra Workspace with hyperspectral map functionality."""
import traceback
import io
import json
import gzip
from io import StringIO
from copy import deepcopy
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import KDTree

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox

from spectroview.viewmodel.utils import zone, quadrant, closest_index, LazyMapDict

from spectroview.model.m_settings import MSettings


from spectroview.model.m_io import load_map_file, load_wdf_map, load_spc_map
from spectroview.model.spectra_store import SpectraStore
from spectroview.viewmodel.vm_workspace_spectra import VMWorkspaceSpectra
from spectroview.fit_engine.tensor_fit_thread import TensorFitThread
from spectroview.fit_engine.scalar_models import FitResult
from spectroview.model.workspace_io import WorkspaceIO
from spectroview.fit_engine.baseline import eval_baseline_batch



class VMWorkspaceMaps(VMWorkspaceSpectra):
    """Maps Workspace ViewModel."""
    
    maps_list_changed = Signal(list)
    map_data_updated = Signal(object)
    send_spectra_to_workspace = Signal(list)
    clear_map_cache_requested = Signal(str)
    switch_to_graphs_tab = Signal()  # Request to switch to Graphs tab
    
    def __init__(self, settings: MSettings):
        super().__init__(settings)
        self.maps = LazyMapDict(self.store)
        self.current_map_name: str | None = None
        self.current_map_df = None
        
        # Store metadata for each map (for WDF files)
        self.maps_metadata: dict[str, dict] = {}  # {map_name: metadata_dict}
        self.map_type = '2Dmap'

        # Reference to Graphs workspace (injected after construction)
        self.graphs_workspace = None
        
        # We share self.store with Spectra workspace through inheritance
        

    
    def load_map_files(self, paths: list[str]):
        """Load hyperspectral map files and extract spectra."""
        loaded_maps = []
        last_valid_path = None
        
        for p in paths:
            path = Path(p)
            map_name = path.stem
            
            if map_name in self.maps:
                self.notify.emit(f"Map '{map_name}' already loaded, skipping.")
                continue
            
            try:
                # Use appropriate loader based on file extension
                if path.suffix.lower() == '.wdf':
                    result = load_wdf_map(path)
                    # Handle tuple return (DataFrame, metadata)
                    if isinstance(result, tuple):
                        map_df, metadata = result
                        self.maps_metadata[map_name] = metadata
                    else:
                        map_df = result
                elif path.suffix.lower() == '.spc':
                     result = load_spc_map(path)
                     if isinstance(result, tuple):
                         map_df, metadata = result
                         self.maps_metadata[map_name] = metadata
                     else:
                         map_df = result
                else:
                    map_df = load_map_file(path)
                    self.maps_metadata[map_name] = {}  # Empty metadata for non-WDF maps
                self.maps[map_name] = map_df
                self._extract_spectra_from_map(map_name, map_df)
                loaded_maps.append(map_name)
                last_valid_path = path  # Track last successfully loaded file
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Error loading {path.name}: {str(e)}")
        
        if loaded_maps:
            self._emit_maps_list_update()
            
            # Update last_directory setting
            if last_valid_path:
                self.settings.set_last_directory(str(last_valid_path.parent))
    
    def select_map(self, map_name: str):
        """Select a map and filter/show its spectra (fast - no extraction)."""
        if map_name not in self.maps:
            return
        
        self.current_map_name = map_name
        self.current_map_df = self.maps[map_name]
        
        # Filter and show spectra for this map (already extracted during load)
        self._show_map_spectra(map_name)
        
        # Emit single signal to update view
        self.map_data_updated.emit(self.current_map_df)

    def _get_unique_map_data(self, fnames: list[str]) -> list:
        """Override parent to fetch MapData of the active map for interactive tools.
        In Maps workspace, fnames are points (map_name_pointX), so we resolve them
        all to the current map since all points share the same MapData container.
        """
        if not self.current_map_name:
            return []
        md = self.store.get_map_data(self.current_map_name)
        return [md] if md else []

    def _emit_list_update(self):
        """Override parent to refresh the current map's spectra list instead of global map names."""
        if self.current_map_name:
            self._show_map_spectra(self.current_map_name)

    def _get_interactive_map_data(self):
        """Override parent to provide the MapData for the active map during baseline/peak modifications."""
        if not self.current_map_name:
            return None
        return self.store.get_map_data(self.current_map_name)

    def _emit_selected_spectra(self):
        """Override parent to emit tensor batch data for Maps instead of just MSpectrum lists."""
        if not self.selected_fnames:
            self.spectra_selection_changed.emit([])
            return

        if self.current_map_name and self.store.get_map_data(self.current_map_name) is not None:
            map_name = self.current_map_name
            md = self.store.get_map_data(map_name)
            
            # Find indices for the selected fnames
            idx_map = {name: i for i, name in enumerate(md.fnames)}
            indices = [idx_map[name] for name in self.selected_fnames if name in idx_map]
            
            if indices:
                indices = np.array(indices)
                self._update_fit_model_from_params(md, indices[0])
                x, Y = self.store.get_xy_batch(map_name, indices)
                
                # Fetch raw data for the viewer
                x0 = md.x0
                Y0 = md.Y0[indices]
                
                from spectroview.model.spectra_store import SpectrumProxy
                proxies = []
                for idx in indices:
                    proxies.append(SpectrumProxy(md, idx, md.fnames[idx]))

                # Create a dict structure containing the batch arrays
                # This will be intercepted by the SpectraViewer to plot fast
                batch_data = {
                    'type': 'tensor',
                    'x': x,
                    'y': Y,
                    'x0': x0,
                    'y0': Y0,
                    'indices': indices,
                    'fnames': [md.fnames[i] for i in indices],
                    'colors': [md.colors[i] for i in indices] if md.colors else None,
                    'labels': [md.labels[i] for i in indices] if md.labels else None,
                    'proxies': proxies,
                    'y_bestfit': md.Y_bestfit[indices] if md.Y_bestfit is not None else None,
                    'y_peaks': [p[indices] for p in md.Y_peaks] if md.Y_peaks is not None else None,
                    'y_baseline': md.Y_baseline[indices] if md.Y_baseline is not None else None,
                    'baseline_config': [md.baseline_config] * len(indices),
                    'fit_models': [md.fit_model] * len(indices) if md.fit_model else None,
                    'fit_r2': md.fit_r2[indices] if md.fit_r2 is not None else None,
                    'map_name': map_name,
                }
                
                self.spectra_selection_changed.emit(batch_data)
                
                if x is not None and len(x) > 0:
                    # Maps don't have xcorrection implemented yet, emit 0.0
                    self.show_xcorrection_value.emit(0.0)
                    self.spectral_range_changed.emit(float(x[0]), float(x[-1]))
                    
                return

        self.spectra_selection_changed.emit([])
            
    def _extract_spectra_from_map(self, map_name: str, map_df: pd.DataFrame):
        """Extract all individual spectra from a hyperspectral map dataframe."""
        # Extract wavenumber columns (all columns except X, Y)
        wavenumber_cols = [col for col in map_df.columns if col not in ['X', 'Y']]
        
        # Convert to numeric and skip last value (following legacy behavior)
        x_values = pd.to_numeric(wavenumber_cols, errors='coerce').tolist()
        if len(x_values) > 1:
            x_values = x_values[:-1]  # Skip last value
            wavenumber_cols = wavenumber_cols[:-1]
               
        x_data = np.asarray(x_values, dtype=np.float64)
        
        # Pre-extract all spatial coordinates and intensity data (faster)
        x_positions = map_df['X'].values
        y_positions = map_df['Y'].values
        intensity_data = map_df[wavenumber_cols].values  # 2D numpy array
        
        # Check if this map has metadata (WDF files)
        map_metadata = self.maps_metadata.get(map_name, {})

        N = len(map_df)
        coords = np.column_stack([x_positions, y_positions]).astype(np.float64)
        Y0 = intensity_data.astype(np.float32)
        fnames = [
            f"{map_name}_({float(x_positions[i])}, {float(y_positions[i])})"
            for i in range(N)
        ]
        self.store.add_map(
            name=map_name,
            x0=x_data,
            Y0=Y0,
            coords=coords,
            fnames=fnames,
            is_active=np.ones(N, dtype=bool),
        )
        md = self.store.get_map_data(map_name)
        if md:
            md.map_metadata = map_metadata
    
    def _show_map_spectra(self, map_name: str):
        """Display spectra for the selected map in the spectra list."""
        # Use SpectraStore as the source of truth — emit lightweight dicts
        # that are compatible with set_spectrum_item_color and v_map_list.
        md = self.store.get_map_data(map_name)
        if md is None:
            self.spectra_list_changed.emit([])
            self.count_changed.emit(0)
            return

        # Crop status
        is_cropped = md.range_min is not None or md.range_max is not None

        # Baseline status
        has_baseline_config = md.baseline_config is not None

        spectra_info = []
        for i, fname in enumerate(md.fnames):
            has_baseline = has_baseline_config
            if has_baseline and isinstance(md.is_baseline_subtracted, np.ndarray):
                has_baseline = bool(md.is_baseline_subtracted[i])
            elif has_baseline:
                has_baseline = bool(md.is_baseline_subtracted)
            
            # Fit status
            has_fit = False
            fit_converged = False
            if md.peak_params is not None:
                has_fit = np.any(md.peak_params[i] != 0.0)
                if has_fit and md.fit_success is not None:
                    fit_converged = bool(md.fit_success[i])

            spectra_info.append({
                "fname": fname,
                "is_active": bool(md.is_active[i]),
                "is_cropped": is_cropped,
                "has_baseline": has_baseline,
                "has_fit": has_fit,
                "fit_success": fit_converged,
            })

        self.spectra_list_changed.emit(spectra_info)
        self.count_changed.emit(len(spectra_info))

    
    def get_current_map_dataframe(self) -> pd.DataFrame | None:
        """Get the DataFrame of the currently selected map (filtered by checked spectra).
        
        Returns only rows corresponding to checked spectra in the list.
        """
        if not self.current_map_name or self.current_map_name not in self.maps:
            return None
        
        df = self.maps[self.current_map_name]
        
        # Filter by active spectra — use SpectraStore as source of truth
        md = self.store.get_map_data(self.current_map_name)
        if md is not None:
            active_fnames = {
                fname for fname, active in zip(md.fnames, md.is_active) if active
            }
        else:
            active_fnames = set()
        # Filter DataFrame to only include active spectra
        # Match based on fname format: "map_name_(x, y)"
        fname_prefix = f"{self.current_map_name}_("
        
        # Build set of (X, Y) tuples from active fnames
        active_coords = set()
        for fname in active_fnames:
            if fname.startswith(fname_prefix):
                # Extract coordinates from fname
                coords_str = fname[fname.rfind('(')+1:fname.rfind(')')]
                try:
                    x_str, y_str = coords_str.split(',')
                    active_coords.add((float(x_str.strip()), float(y_str.strip())))
                except (ValueError, AttributeError):
                    continue
        
        if not active_coords:
            # No active spectra for this map, return empty DataFrame with same structure
            return df.iloc[0:0]
        
        # Filter DataFrame rows by (X, Y) coordinates (Vectorized & much faster than apply)
        idx = pd.MultiIndex.from_arrays([df['X'], df['Y']])
        mask = idx.isin(list(active_coords))
        return df[mask]
    
    def save_current_map_to_excel(self, file_path: str):
        """Save the currently selected map to an Excel file."""
        if not self.current_map_name or self.current_map_name not in self.maps:
            self.notify.emit("No map selected.")
            return
        
        try:
            df = self.maps[self.current_map_name]
            df.to_excel(file_path, index=False)
            self.notify.emit(f"Map '{self.current_map_name}' saved to Excel.")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error saving map: {e}")
    
    def delete_current_map(self):
        """Delete the currently selected map."""
        if not self.current_map_name:
            self.notify.emit("No map selected.")
            return
        
        map_name = self.current_map_name
        self.remove_map(map_name)
        self.notify.emit(f"Map '{map_name}' deleted.")
    
    def select_all_current_map_spectra(self):
        """Select all spectra from the currently displayed map."""
        if not self.current_map_name:
            return
        
        md = self.store.get_map_data(self.current_map_name)
        if md:
            self.selected_fnames = list(md.fnames)
        self._emit_selected_spectra()
    
    def reinit_current_map_spectra(self, apply_all: bool = False):
        """Reinitialize selected spectra or all spectra from all maps."""
        # Delegate to our reinit_spectra which handles fname-based selection
        self.reinit_spectra(apply_all)
        
    def reinit_spectra(self, apply_all: bool = False):
        """Override parent to selectively or globally reinitialize spectra."""
        if apply_all:
            for name in self.store.map_names:
                md = self.store.get_map_data(name)
                if md:
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
                    md.fit_success = None
                    md.fit_r2 = None
                    self.store.clear_preprocess(name)
        else:
            if not self.selected_fnames:
                self.notify.emit("No spectrum selected.")
                return
            md = self.store.get_map_data(self.current_map_name)
            if not md:
                return
            
            fnames = self._get_selected_spectra()
            indices = [md.fnames.index(f) for f in fnames if f in md.fnames]
            if not indices:
                return

            N = md.Y0.shape[0] if md.Y0 is not None else 1
            if md.x is None and md.x0 is not None:
                md.x = md.x0.copy()
            if md.Y is None and md.Y0 is not None:
                md.Y = md.Y0.copy()

            # Revert Y for selected indices to cropped/raw Y0
            if md.Y0 is not None and md.Y is not None:
                if md.x is not None and md.x0 is not None and len(md.x) < len(md.x0):
                    xmin, xmax = md.x[0], md.x[-1]
                    i_min = closest_index(md.x0, xmin)
                    i_max = closest_index(md.x0, xmax)
                    if i_min > i_max: i_min, i_max = i_max, i_min
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
            if md.fit_success is not None:
                md.fit_success[indices] = False
            if md.fit_r2 is not None:
                md.fit_r2[indices] = 0.0
            if md.Y_peaks is not None:
                for peak_curve in md.Y_peaks:
                    peak_curve[indices] = 0.0

        if self.current_map_name:
            self._show_map_spectra(self.current_map_name)
            self._emit_selected_spectra()
    
    def subtract_baseline(self, apply_all: bool = False):
        """Override parent to selectively or globally subtract baseline."""
        if apply_all:
            for name in self.store.map_names:
                md = self.store.get_map_data(name)
                if md and md.baseline_config is not None:
                    x_arr = md.x if md.x is not None else md.x0
                    y_arr = md.Y if md.Y is not None else md.Y0
                    md.Y_baseline = eval_baseline_batch(x_arr, y_arr, md.baseline_config)
                    if md.Y is None:
                        md.Y = md.Y0.copy()
                    md.Y = md.Y - md.Y_baseline
                    md.Y_baseline = None
                    md.is_baseline_subtracted = True
                    self.store.batch_preprocess(name, md.baseline_config, md.range_min, md.range_max)
        else:
            if not self.selected_fnames:
                self.notify.emit("No spectrum selected.")
                return
            md = self.store.get_map_data(self.current_map_name)
            if not md or md.baseline_config is None:
                self.notify.emit("No baseline configured.")
                return

            fnames = self._get_selected_spectra()
            indices = [md.fnames.index(f) for f in fnames if f in md.fnames]
            if not indices:
                return

            N = md.Y0.shape[0] if md.Y0 is not None else 1
            if not isinstance(md.is_baseline_subtracted, np.ndarray):
                md.is_baseline_subtracted = np.full(N, bool(md.is_baseline_subtracted), dtype=bool)

            indices_to_sub = [i for i in indices if not md.is_baseline_subtracted[i]]
            if not indices_to_sub:
                self.notify.emit("Baseline already subtracted. Please delete baseline or reinit first.")
                return

            if md.Y is None:
                md.Y = md.Y0.copy()
            if md.x is None:
                md.x = md.x0.copy()

            x_arr = md.x
            y_arr = md.Y
            
            if md.Y_baseline is None:
                md.Y_baseline = eval_baseline_batch(x_arr, y_arr, md.baseline_config)

            # Perform baseline subtraction for selected indices
            md.Y[indices_to_sub] = md.Y[indices_to_sub] - md.Y_baseline[indices_to_sub]
            md.Y_baseline[indices_to_sub] = 0.0
            md.is_baseline_subtracted[indices_to_sub] = True

        if self.current_map_name:
            self._emit_selected_spectra()
            self._emit_list_update()

    def preview_baseline(self, settings: dict):
        """Override parent to preview baseline on the tensor data."""
        if not self.selected_fnames:
            return
        
        md = self.store.get_map_data(self.current_map_name) if self.current_map_name else None
        if not md:
            return
            
        self._baseline_settings = settings
        fnames = self._get_selected_spectra()
        self._apply_baseline_settings(settings, fnames)
        self._emit_selected_spectra()

    def delete_baseline(self, apply_all: bool = False):
        """Override parent to selectively or globally delete baseline."""
        if apply_all:
            for name in self.store.map_names:
                md = self.store.get_map_data(name)
                if md:
                    md.baseline_config = None
                    md.Y_baseline = None
                    md.is_baseline_subtracted = False
                    if md.range_min is not None or md.range_max is not None:
                        mask = np.logical_and(md.x0 >= md.range_min, md.x0 <= md.range_max)
                        md.x = md.x0[mask].copy()
                        md.Y = md.Y0[:, mask].copy()
                    else:
                        md.x = md.x0.copy()
                        md.Y = md.Y0.copy()
                    self.store.clear_preprocess(name)
        else:
            if not self.selected_fnames:
                self.notify.emit("No spectrum selected.")
                return
            md = self.store.get_map_data(self.current_map_name)
            if not md:
                return

            fnames = self._get_selected_spectra()
            indices = [md.fnames.index(f) for f in fnames if f in md.fnames]
            if not indices:
                return

            N = md.Y0.shape[0] if md.Y0 is not None else 1
            if not isinstance(md.is_baseline_subtracted, np.ndarray):
                md.is_baseline_subtracted = np.full(N, bool(md.is_baseline_subtracted), dtype=bool)

            # Revert selected indices to cropped/raw Y0
            if md.Y is None:
                md.Y = md.Y0.copy()
            if md.x is None:
                md.x = md.x0.copy()

            if md.x is not None and md.x0 is not None and len(md.x) < len(md.x0):
                xmin, xmax = md.x[0], md.x[-1]
                i_min = closest_index(md.x0, xmin)
                i_max = closest_index(md.x0, xmax)
                if i_min > i_max: i_min, i_max = i_max, i_min
                md.Y[indices] = md.Y0[indices, i_min:i_max+1].copy()
            else:
                md.Y[indices] = md.Y0[indices].copy()

            md.is_baseline_subtracted[indices] = False
            if md.Y_baseline is not None:
                md.Y_baseline[indices] = 0.0

        if self.current_map_name:
            self._show_map_spectra(self.current_map_name)
            self._emit_selected_spectra()
    
    def paste_peaks(self, apply_all: bool = False):
        """Override parent to selectively or globally paste peaks."""
        if not hasattr(self, "_peaks_clipboard") or self._peaks_clipboard is None:
            self.notify.emit("No peaks copied.")
            return

        if apply_all:
            for name in self.store.map_names:
                md = self.store.get_map_data(name)
                if md:
                    md.fit_model = deepcopy(self._peaks_clipboard)
                    self._reconstruct_y_peaks(md)
        else:
            if not self.selected_fnames:
                self.notify.emit("No spectrum selected.")
                return
            md = self.store.get_map_data(self.current_map_name)
            if not md:
                return

            fnames = self._get_selected_spectra()
            indices = [md.fnames.index(f) for f in fnames if f in md.fnames]
            if not indices:
                return

            md.fit_model = deepcopy(self._peaks_clipboard)
            self._reconstruct_y_peaks(md, indices=indices)

        self._emit_selected_spectra()
        self._emit_list_update()

    def delete_peaks(self, apply_all: bool = False):
        """Override parent to selectively or globally delete peaks."""
        if apply_all:
            for name in self.store.map_names:
                md = self.store.get_map_data(name)
                if md:
                    md.fit_model = None
                    md.Y_peaks = None
                    md.Y_bestfit = None
                    md.peak_params = None
                    md.fit_success = None
                    md.fit_r2 = None
        else:
            if not self.selected_fnames:
                self.notify.emit("No spectrum selected.")
                return
            md = self.store.get_map_data(self.current_map_name)
            if not md:
                return

            fnames = self._get_selected_spectra()
            indices = [md.fnames.index(f) for f in fnames if f in md.fnames]
            if not indices:
                return

            if md.Y_peaks is not None:
                for peak_curve in md.Y_peaks:
                    peak_curve[indices] = 0.0

            if md.Y_bestfit is not None:
                if md.Y_baseline is not None:
                    md.Y_bestfit[indices] = md.Y_baseline[indices]
                else:
                    md.Y_bestfit[indices] = 0.0

            if md.peak_params is not None:
                md.peak_params[indices] = 0.0
            if md.fit_success is not None:
                md.fit_success[indices] = False
            if md.fit_r2 is not None:
                md.fit_r2[indices] = 0.0

        if self.current_map_name:
            self._show_map_spectra(self.current_map_name)
            self._emit_selected_spectra()



    def apply_x_range_all(self, apply_all: bool = False):
        """Override to update store preprocessing range."""
        super().apply_x_range_all(apply_all)
        if self.current_map_name:
            self._update_store_preprocessing()
            self._emit_selected_spectra()
            
    def _update_store_preprocessing(self):
        """Helper to sync the first spectrum's baseline and crop settings to the Store."""
        if not self.current_map_name: return
        
        md = self.store.get_map_data(self.current_map_name)
        if not md or not md.is_active.any():
            self.store.clear_preprocess(self.current_map_name)
            return
            
        b_dict = md.baseline_config or {}
        rmin, rmax = md.range_min, md.range_max
        is_sub = getattr(md, "is_baseline_subtracted", False)
        is_subtracted = is_sub.any() if isinstance(is_sub, np.ndarray) else bool(is_sub)
        
        if not is_subtracted and rmin is None and rmax is None:
            self.store.clear_preprocess(self.current_map_name)
        else:
            if not is_subtracted:
                # pass an empty baseline dict so it only crops
                b_dict = {}
            self.store.batch_preprocess(self.current_map_name, b_dict, rmin, rmax)
    
    def send_selected_spectra_to_spectra_workspace(self):
        """Send selected spectra to the Spectra workspace tab for comparison."""
        if not self.selected_fnames:
            self.notify.emit("No spectrum selected.")
            return
        
        # For Phase 6: We will send fnames and map_names to Spectra Workspace.
        # But for now, we just pass since Spectra workspace will also use the store.
        self.notify.emit(f"Sent {len(self.selected_fnames)} spectra to Spectra workspace.")
    
    def remove_map(self, map_name: str):
        """Remove a map and its spectra from the loaded maps."""
        if map_name not in self.maps:
            return
        
        del self.maps[map_name]
        if hasattr(self, '_maps_arrays_cache') and map_name in self._maps_arrays_cache:
            del self._maps_arrays_cache[map_name]
            
        self.store.remove_map(map_name)
        
        # If removed map was selected, clear view
        if self.current_map_name == map_name:
            self.current_map_name = None
            self.current_map_df = None
            self.selected_fnames = []
            self.spectra_list_changed.emit([])
            self.count_changed.emit(0)
            self.spectra_selection_changed.emit([])
            # Clear map plot by emitting empty DataFrame
            self.map_data_updated.emit(pd.DataFrame())
        
        # Update maps list in View
        self._emit_maps_list_update()
    
    def _emit_maps_list_update(self):
        """Emit updated list of map names."""
        map_names = list(self.maps.keys())
        self.maps_list_changed.emit(map_names)


    def _extract_coords_for_spectra(self, spectra):
        """Extract (X, Y) spatial coordinates for a list of map spectra.
        
        Parses coordinates from the spectrum fname format: "map_name_(x, y)"
        
        Returns:
            np.ndarray of shape (N, 2) or None if parsing fails
        """
        coords = []
        for spectrum in spectra:
            fname = spectrum.fname
            try:
                coord_str = fname[fname.rfind('(') + 1:fname.rfind(')')]
                x_str, y_str = coord_str.split(',')
                coords.append([float(x_str.strip()), float(y_str.strip())])
            except (ValueError, AttributeError, IndexError):
                # If any spectrum has unparseable coords, disable propagation
                return None

        return np.array(coords, dtype=np.float64) if coords else None

    def _get_active_spectra(self):
        """Override: Return active spectra fnames for the CURRENT map only."""
        if not self.current_map_name: return []
        md = self.store.get_map_data(self.current_map_name)
        if not md: return []
        return [f for i, f in enumerate(md.fnames) if md.is_active[i]]

    def _get_selected_spectra(self):
        """Override: Return selected fnames that are active for the CURRENT map."""
        if not self.current_map_name: return []
        md = self.store.get_map_data(self.current_map_name)
        if not md: return []
        fname_set = set(self.selected_fnames)
        return [f for i, f in enumerate(md.fnames) if md.is_active[i] and f in fname_set]

    def apply_fit_model(self, apply_all: bool = False):
        """Override parent to apply loaded fit model and sync preprocessing/fit thread."""
        if not hasattr(self, "_vm_fit_model_builder"):
            self.notify.emit("Fit model manager not connected.")
            return

        model_path = self._vm_fit_model_builder.get_current_model_path()
        if model_path is None or not model_path.exists():
            self.notify.emit("No fit model selected.")
            return
            
        import json
        from PySide6.QtWidgets import QMessageBox
        
        try:
            with open(model_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            fit_model = data.get("0", {})
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to load fit model:\n{e}")
            return

        # Apply to MapData(s)
        if apply_all:
            mds = [self.store.get_map_data(name) for name in self.store.map_names]
            for md in mds:
                if md:
                    self._apply_fit_model_to_mapdata(md, fit_model)
        else:
            md = self.store.get_map_data(self.current_map_name) if self.current_map_name else None
            if md:
                fnames = self._get_selected_spectra()
                fname_set = set(fnames)
                indices = [i for i, f in enumerate(md.fnames) if f in fname_set]
                self._apply_fit_model_to_mapdata(md, fit_model, indices=indices)

        if self.current_map_name:
            self._update_store_preprocessing()
            self._emit_selected_spectra()
            self._emit_list_update()

        fnames = self._get_active_spectra() if apply_all else self._get_selected_spectra()
        self._run_fit_thread(fit_model, fnames, apply_all=apply_all)

    def paste_fit_model(self, apply_all: bool = False):
        """Override parent to apply clipboard fit model and sync preprocessing/fit thread."""
        if not hasattr(self, "_fitmodel_clipboard") or self._fitmodel_clipboard is None:
            self.notify.emit("No fit model copied.")
            return

        # Apply to MapData(s)
        if apply_all:
            mds = [self.store.get_map_data(name) for name in self.store.map_names]
            for md in mds:
                if md:
                    self._apply_fit_model_to_mapdata(md, self._fitmodel_clipboard)
        else:
            md = self.store.get_map_data(self.current_map_name) if self.current_map_name else None
            if md:
                fnames = self._get_selected_spectra()
                fname_set = set(fnames)
                indices = [i for i, f in enumerate(md.fnames) if f in fname_set]
                self._apply_fit_model_to_mapdata(md, self._fitmodel_clipboard, indices=indices)

        if self.current_map_name:
            self._update_store_preprocessing()
            self._emit_selected_spectra()
            self._emit_list_update()

        fnames = self._get_active_spectra() if apply_all else self._get_selected_spectra()
        self._run_fit_thread(deepcopy(self._fitmodel_clipboard), fnames, apply_all=apply_all)

    def fit(self, apply_all: bool = False):
        """Override: Fitting action for the Maps workspace."""
        if self._is_fitting:
            self.notify.emit("Fit already in progress. Please wait...")
            return

        tasks = []
        if apply_all:
            for map_name in self.store.map_names:
                md = self.store.get_map_data(map_name)
                if md and md.fit_model:
                    self._reindex_fit_model(md)
                    tasks.append({
                        "map_name": map_name,
                        "indices": np.arange(md.n_spectra),
                        "fit_model": md.fit_model
                    })
        else:
            md = self.store.get_map_data(self.current_map_name)
            if not md or not md.fit_model:
                self.notify.emit("No peaks to fit.")
                return
            self._reindex_fit_model(md)
            fnames = self._get_selected_spectra()
            fname_set = set(fnames)
            indices = np.array([i for i, f in enumerate(md.fnames) if f in fname_set])
            if len(indices) == 0:
                return
            tasks = [{
                "map_name": self.current_map_name,
                "indices": indices,
                "fit_model": md.fit_model
            }]

        if not tasks:
            return

        self._is_fitting = True
        self.fit_in_progress.emit(True)
        self._fit_thread = TensorFitThread(self.store, tasks)
        self._fit_thread.progress_changed.connect(self.fit_progress_updated.emit)
        self._fit_thread.timings_ready.connect(self.fit_timings_ready.emit)
        self._fit_thread.finished.connect(self._on_fit_finished)
        self._fit_thread.start()

    def _run_fit_thread(self, fit_model: dict, fnames: list[str], apply_all: bool = False):
        """Override: Run fit thread for Maps workspace loaded fit models."""
        if self._is_fitting:
            self.notify.emit("Fit already in progress. Please wait...")
            return

        tasks = []
        if apply_all:
            for map_name in self.store.map_names:
                md = self.store.get_map_data(map_name)
                if md:
                    tasks.append({
                        "map_name": map_name,
                        "indices": np.arange(md.n_spectra),
                        "fit_model": fit_model
                    })
        else:
            md = self.store.get_map_data(self.current_map_name)
            if not md: return
            fname_set = set(fnames)
            indices = np.array([i for i, f in enumerate(md.fnames) if f in fname_set])
            if len(indices) == 0:
                return
            tasks = [{
                "map_name": self.current_map_name,
                "indices": indices,
                "fit_model": fit_model
            }]

        if not tasks:
            return

        self._is_fitting = True
        self.fit_in_progress.emit(True)
        self._fit_thread = TensorFitThread(self.store, tasks)
        self._fit_thread.progress_changed.connect(self.fit_progress_updated.emit)
        self._fit_thread.timings_ready.connect(self.fit_timings_ready.emit)
        self._fit_thread.finished.connect(self._on_fit_finished)
        self._fit_thread.start()

    def _sync_fit_results_to_store(self):
        """Override: No-op for Maps since TensorFitThread writes directly to Store."""
        pass


    def get_fit_results_dataframe(self):
        """Get fit results DataFrame for the current map.
        
        This method returns a simplified version of the fit results for heatmap visualization.
        It reuses self.df_fit_results (created by collect_fit_results) to avoid redundancy.
        """
        # If we have collected fit results (from collect_fit_results), use them
        if self.df_fit_results is not None and not self.df_fit_results.empty:
            return self.df_fit_results
        return pd.DataFrame()
    
    
    def collect_fit_results(self):
        """Collect fit results, delegating to SpectraStore via parent class and post-processing wafer coords."""
        map_names = self.store.map_names
        if not map_names:
            self.notify.emit("No maps loaded to collect results from.")
            return

        super().collect_fit_results(map_names=map_names)

        if self.df_fit_results is None or self.df_fit_results.empty:
            return

        # Wafer post-processing
        if self.map_type != '2Dmap':
            radius = 150
            if '300' in self.map_type:
                radius = 150
            elif '200' in self.map_type:
                radius = 100
            elif '100' in self.map_type:
                radius = 50

            x = self.df_fit_results['X']
            y = self.df_fit_results['Y']

            quadrant_arr = np.full(len(self.df_fit_results), np.nan, dtype=object)
            quadrant_arr[(x < 0) & (y < 0)] = 'Q1'
            quadrant_arr[(x < 0) & (y > 0)] = 'Q2'
            quadrant_arr[(x > 0) & (y > 0)] = 'Q3'
            quadrant_arr[(x > 0) & (y < 0)] = 'Q4'
            self.df_fit_results['Quadrant'] = quadrant_arr

            dist = np.sqrt(x**2 + y**2)
            zone_arr = np.full(len(self.df_fit_results), np.nan, dtype=object)
            zone_arr[dist <= radius * 0.35] = 'Center'
            zone_arr[(dist > radius * 0.35) & (dist < radius * 0.8)] = 'Mid-Radius'
            zone_arr[dist >= radius * 0.8] = 'Edge'
            self.df_fit_results['Zone'] = zone_arr

            # Re-emit the wafer updated results
            self.fit_results_updated.emit(self.df_fit_results)

        # Trigger heatmap refresh with new fit results
        # This ensures the map viewer uses fresh data instead of cache
        if self.current_map_name:
            self.clear_map_cache_requested.emit(self.current_map_name)
            self.map_data_updated.emit(self.current_map_df)
    
    # ─────────────────────────────────────────────────────────────────
    # SAVE/LOAD WORKSPACE
    # ─────────────────────────────────────────────────────────────────
    
    def _map_df_to_binary(self, map_df: pd.DataFrame) -> dict:
        """Serialize a map DataFrame to a compact binary form (numpy + gzip).
        
        Much faster than to_csv() which formats each float as ASCII text.
        Returns a dict with 'cols' (column names) and 'data' (gzip+hex bytes).
        """
        # Use native float64 (no precision loss). copy=False avoids unnecessary data duplication.
        arr = map_df.to_numpy(copy=False)
        cols = list(map_df.columns.astype(str))  # preserve column names
        buf = io.BytesIO()
        np.save(buf, arr)
        # Use compresslevel=1 for massive speedup (~70s -> ~3s) with barely any size penalty
        return {'cols': cols, 'data': gzip.compress(buf.getvalue(), compresslevel=1).hex()}
    
    @staticmethod
    def _binary_to_map_df(entry: dict | str) -> pd.DataFrame:
        """Deserialize a map DataFrame from binary form.
        
        Supports both the new binary format (dict with 'cols'+'data')
        and the legacy CSV+gzip+hex format (plain hex string).
        """
        if isinstance(entry, str):
            # Legacy format: gzip-compressed CSV encoded as hex
            csv_data = gzip.decompress(bytes.fromhex(entry)).decode('utf-8')
            return pd.read_csv(StringIO(csv_data))
        else:
            # New binary format: {'cols': [...], 'data': hex}
            raw = gzip.decompress(bytes.fromhex(entry['data']))
            arr = np.load(io.BytesIO(raw))
            # Cast back to float64 to match standard dataframe precision
            return pd.DataFrame(arr, columns=entry['cols'], dtype=np.float64)



    def _load_legacy_maps(self, file_path: str):
        """Load legacy JSON-based .maps workspace (OLD version)."""
        import gzip, io, re
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.store = SpectraStore()
        self.maps = LazyMapDict(self.store)
        
        spectrums_data = data.get("spectrums_data", {})
        
        # Build coordinate lookup dictionary from individual spectrum data
        lookup = {}
        for skey, sdata in spectrums_data.items():
            fname = sdata.get("fname", "")
            match = re.match(r"(.+?)_\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)", fname)
            if match:
                mname = match.group(1)
                xc = float(match.group(2))
                yc = float(match.group(3))
                lookup[(mname, round(xc, 3), round(yc, 3))] = sdata

        # Process each 2D map stored in the file
        for map_name, hex_string in data.get("maps", {}).items():
            # Hex-parse and decompress gzip CSV
            hex_bytes = bytes.fromhex(hex_string)
            decompressed = gzip.decompress(hex_bytes)
            df = pd.read_csv(io.BytesIO(decompressed))
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            # Extract coordinates, wavenumbers, and intensity arrays
            coords = df[['X', 'Y']].values.astype(np.float64)
            x0 = np.array([float(col) for col in df.columns[2:]], dtype=np.float64)
            Y0 = df.iloc[:, 2:].values.astype(np.float32)

            N = len(df)
            fnames = []
            is_active_list = []
            colors = []
            labels = []
            map_spectra_sdata = []

            for i in range(N):
                row = df.iloc[i]
                xc, yc = row['X'], row['Y']
                sdata = lookup.get((map_name, round(xc, 3), round(yc, 3)))
                
                if sdata:
                    fnames.append(sdata.get("fname", f"{map_name}_({xc}, {yc})"))
                    is_active_list.append(sdata.get("is_active", True))
                    colors.append(sdata.get("color"))
                    labels.append(sdata.get("label"))
                    map_spectra_sdata.append(sdata)
                else:
                    fnames.append(f"{map_name}_({xc}, {yc})")
                    is_active_list.append(True)
                    colors.append(None)
                    labels.append(None)
                    map_spectra_sdata.append({})

            # Register map into store
            self.store.add_map(
                name=map_name,
                x0=x0,
                Y0=Y0,
                coords=coords,
                fnames=fnames,
                is_active=np.array(is_active_list, dtype=bool),
                colors=colors,
                labels=labels
            )

            # Reconstruct parameter names and peak_params matrix
            param_names_set = set()
            for sdata in map_spectra_sdata:
                pmodels = sdata.get("peak_models", {}) or {}
                for k, pdict in pmodels.items():
                    shape = list(pdict.keys())[0]
                    for pname in pdict[shape].keys():
                        param_names_set.add(f"{pname}_{int(k) + 1}")

            def param_key(p):
                parts = p.split("_")
                idx = int(parts[-1])
                name = "_".join(parts[:-1])
                return (idx, name)

            param_names = sorted(list(param_names_set), key=param_key)

            peak_params_matrix = np.zeros((N, len(param_names)), dtype=np.float64)
            success_list = []
            for i, sdata in enumerate(map_spectra_sdata):
                success_list.append(sdata.get("result_fit_success", False))
                pmodels = sdata.get("peak_models", {}) or {}
                for k, pdict in pmodels.items():
                    shape = list(pdict.keys())[0]
                    for pname, pvals in pdict[shape].items():
                        p_fullname = f"{pname}_{int(k) + 1}"
                        if p_fullname in param_names:
                            col_idx = param_names.index(p_fullname)
                            peak_params_matrix[i, col_idx] = pvals.get("value", 0.0)

            # Extract a representative fit model from the first spectrum that has one
            fit_model_dict = {}
            for sdata in map_spectra_sdata:
                pmodels = sdata.get("peak_models", {})
                plabels = sdata.get("peak_labels", [])
                if pmodels:
                    fit_model_dict = {"peak_labels": plabels, "peak_models": pmodels}
                    break

            if param_names:
                self.store.set_fit_results(
                    map_name=map_name,
                    indices=np.arange(N),
                    peak_params=peak_params_matrix,
                    success=np.array(success_list, dtype=bool),
                    r2=np.zeros(N, dtype=np.float64),
                    param_names=param_names,
                    fit_model=fit_model_dict
                )

            # Restore configurations per map
            md = self.store.get_map_data(map_name)
            if md:
                # Find baseline config
                for sdata in map_spectra_sdata:
                    legacy_bl = sdata.get("baseline")
                    if legacy_bl:
                        md.baseline_config = {
                            "mode": legacy_bl.get("mode", "Linear"),
                            "points": legacy_bl.get("points", [[], []]),
                            "attached": legacy_bl.get("attached", False),
                            "coef": legacy_bl.get("coef", 5),
                        }
                        break

                if map_spectra_sdata:
                    first_s = map_spectra_sdata[0]
                    md.range_min = first_s.get("range_min")
                    md.range_max = first_s.get("range_max")
                    md.xcorrection_value = float(first_s.get("xcorrection_value", 0.0))
                    md.intensity_norm_factor = float(first_s.get("intensity_norm_factor", 1.0))
                    md.is_baseline_subtracted = first_s.get("is_baseline_subtracted", False)

                self._restore_preprocessed_state(md)

            # Rebuild legacy maps entry for visual heatmap representation (highly optimized)
            col_names = list(map(str, x0))
            df = pd.DataFrame(Y0, columns=col_names)
            df.insert(0, 'Y', coords[:, 1])
            df.insert(0, 'X', coords[:, 0])
            self.maps[map_name] = df

        # Restore df_fit_results
        legacy_results = data.get("df_fit_results")
        if legacy_results:
            self.df_fit_results = pd.DataFrame(legacy_results)
            self.fit_results_updated.emit(self.df_fit_results)
        else:
            self.df_fit_results = None

        self.maps_metadata = data.get("maps_metadata", {})
        self.map_type = "2Dmap"

        map_names = self.store.map_names
        if map_names:
            self.select_map(map_names[0])
        self.maps_list_changed.emit(map_names)
        self.count_changed.emit(sum(md.n_spectra for md in self.store._maps.values()))
        self.notify.emit("Legacy maps workspace loaded successfully.")



    def clear_workspace(self):
        """Clear all maps, spectra, and reset workspace to initial state."""
        # Stop any running fit thread first (via parent)
        super().clear_workspace()

        # Clear Maps-specific data structures
        self.maps.clear()
        self.current_map_name = None
        self.current_map_df = None

        # Reset SpectraStore
        if hasattr(self, 'store'):
            self.store.clear()

        # Emit updates to View
        self.maps_list_changed.emit([])
        self.map_data_updated.emit(pd.DataFrame())


    def set_graphs_workspace(self, graphs_workspace):
        """Inject Graphs workspace reference for cross-workspace communication."""
        self.graphs_workspace = graphs_workspace
    
    def extract_and_send_profile_to_graphs(self, profile_name: str, profile_df: pd.DataFrame):
        """Extract profile data and send to Graphs workspace for plotting."""
        
        if profile_df is None or profile_df.empty:
            self.notify.emit("No profile data to extract. Select exactly 2 points in 2D map mode.")
            return
        
        if not profile_name or profile_name.strip() == "":
            self.notify.emit("Please provide a name for the profile.")
            return
        
        if self.graphs_workspace is None:
            self.notify.emit("Graphs workspace not available.")
            return
        
        # Add DataFrame to Graphs workspace
        self.graphs_workspace.vm.add_dataframe(profile_name, profile_df)
        
        # Create plot configuration for line plot
        plot_config = {
            'df_name': profile_name,
            'plot_style': 'line',
            'x': 'distance',
            'y': ['values'],
            'z': None,
            'filters': [],
            'plot_width': 480,
            'plot_height': 420,
            'dpi': 100
        }
        
        # Create plot using Graphs workspace method
        success = self.graphs_workspace.create_plot_from_config(profile_name, plot_config)
        
        if success:
            # Emit signal to request tab switch
            self.switch_to_graphs_tab.emit()
            self.notify.emit(f"Profile '{profile_name}' sent to Graphs workspace")
        else:
            self.notify.emit("Failed to create profile plot.")


