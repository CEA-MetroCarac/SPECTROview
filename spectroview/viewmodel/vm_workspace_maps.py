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

from spectroview.viewmodel.utils import zone, quadrant

from spectroview.model.m_settings import MSettings
from spectroview.model.m_spectra import MSpectra
from spectroview.model.m_spectrum import MSpectrum
from spectroview.model.m_io import load_map_file, load_wdf_map, load_spc_map
from spectroview.model.spectra_store import SpectraStore
from spectroview.viewmodel.vm_workspace_spectra import VMWorkspaceSpectra
from spectroview.fit_engine.tensor_fit_thread import TensorFitThread
from spectroview.fit_engine.scalar_models import FitResult
from spectroview.model.workspace_io import WorkspaceIO



class VMWorkspaceMaps(VMWorkspaceSpectra):
    """Maps Workspace ViewModel."""
    
    maps_list_changed = Signal(list)
    map_data_updated = Signal(object)
    send_spectra_to_workspace = Signal(list)
    clear_map_cache_requested = Signal(str)
    switch_to_graphs_tab = Signal()  # Request to switch to Graphs tab
    
    def __init__(self, settings: MSettings):
        super().__init__(settings)
        # Maps storage: {map_name: DataFrame}
        self.maps: dict[str, pd.DataFrame] = {}
        self.current_map_name: str | None = None
        self.current_map_df = None
        
        # Store metadata for each map (for WDF files)
        self.maps_metadata: dict[str, dict] = {}  # {map_name: metadata_dict}
        self.map_type = '2Dmap'

        # Reference to Graphs workspace (injected after construction)
        self.graphs_workspace = None
        self._maps_arrays_cache = {}

        # ── Phase-1: Tensor-centric data store ──────────────────────────
        # SpectraStore owns all heavy numerical data as contiguous NumPy arrays.
        # self.spectra (legacy MSpectrum list) is kept in parallel for GUI
        # compatibility until Phase 4 removes it entirely.
        self.store = SpectraStore()
        
    def _ensure_spectrum_loaded(self, spectrum: MSpectrum):
        """Lazily load x0 and y0 arrays for a spectrum from the map cache if not already loaded."""
        if getattr(spectrum, '_is_lazy_loaded', False):
            return
            
        if spectrum.x0 is not None and spectrum.y0 is not None and not hasattr(spectrum, '_lazy_meta'):
            spectrum._is_lazy_loaded = True
            return

        # 1. Lazily apply heavy metadata (peak models, etc.)
        if hasattr(spectrum, '_lazy_meta'):
            spectrum.set_attributes(spectrum._lazy_meta)
            del spectrum._lazy_meta

        fname = spectrum.fname
        try:
            # Parse map name and coordinates from fname: "map_name_(x, y)"
            map_name, coord_str = fname.rsplit('_', 1)
            coord_str = coord_str.strip('()')
            coord = tuple(map(float, coord_str.split(',')))
            
            # Check if we have cached arrays for this map
            if hasattr(self, '_maps_arrays_cache') and map_name in self._maps_arrays_cache:
                cache = self._maps_arrays_cache[map_name]
                # Find matching coordinate index
                coords = cache['coords']
                distances = np.sum((coords - np.array(coord))**2, axis=1)
                idx = np.argmin(distances)
                if distances[idx] < 1e-4:
                    spectrum.x0 = cache['x0'].copy() + spectrum.xcorrection_value
                    spectrum.y0 = cache['y0'][idx].copy()
                    
                    # Apply preprocessing now that arrays are loaded
                    spectrum.is_preprocessed = False
                    spectrum.preprocess()

                    # ── Restore fit result and peak_models from SpectraStore ──────
                    # The store holds fitted params; reconstruct a lightweight
                    # FitResult so downstream code (.result_fit.success, .params) works.
                    md = self.store.get_map_data(map_name)
                    if md is not None and md.has_fit_results() and md.fit_success is not None:
                        if md.fit_success[idx]:
                            params_dict = {
                                pname: float(md.peak_params[idx, j])
                                for j, pname in enumerate(md.param_names)
                            }
                            spectrum.result_fit = FitResult(
                                success=True,
                                params_dict=params_dict,
                                best_fit=spectrum.y,  # approximate
                                rsquared=float(md.fit_r2[idx]),
                            )

                        # Restore peak_models from stored fit_model definition
                        if md.fit_model:
                            spectrum.set_attributes(md.fit_model)

                    spectrum._is_lazy_loaded = True
        except Exception as e:
            pass

    def _get_spectra_by_fnames(self, fnames: list[str]) -> list[MSpectrum]:
        """Override to ensure requested spectra are lazily loaded."""
        spectra = super()._get_spectra_by_fnames(fnames)
        for s in spectra:
            self._ensure_spectrum_loaded(s)
        return spectra
        
    def _get_active_spectra(self) -> list:
        """Override to ensure active spectra are lazily loaded."""
        spectra = super()._get_active_spectra()
        for s in spectra:
            self._ensure_spectrum_loaded(s)
        return spectra
    
    def _get_selected_spectra(self) -> list[MSpectrum]:
        """Get currently selected spectra that are also active (checked).
        
        Override parent to filter by is_active so that operations on selected
        spectra (non-Ctrl click) also respect checkbox state.
        """
        selected = self._get_spectra_by_fnames(self.selected_fnames)
        # Filter to only active (checked) spectra
        return [s for s in selected if s.is_active]
    
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
    
    def _extract_spectra_from_map(self, map_name: str, map_df: pd.DataFrame):
        """Extract all individual spectra from a hyperspectral map dataframe.

        Phase 1: Dual-write — populates both self.spectra (legacy MSpectrum list)
        and self.store (new SpectraStore) in parallel.
        """
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

        # ── Phase-1: Register in SpectraStore (vectorized) ──────────────
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
        # Also populate the lazy-load arrays cache (used by _ensure_spectrum_loaded)
        self._maps_arrays_cache[map_name] = {
            'coords': coords,
            'x0': x_data,
            'y0': Y0,
        }

        # ── Legacy MSpectrum objects (kept for GUI compat in Phase 1) ───
        for idx in range(N):
            x_pos = float(x_positions[idx])
            y_pos = float(y_positions[idx])
            y_data = np.asarray(intensity_data[idx], dtype=np.float64)
            
            # Create MSpectrum object
            spectrum = MSpectrum()
            spectrum.fname = f"{map_name}_({x_pos}, {y_pos})"
            spectrum.x = x_data.copy()
            spectrum.x0 = x_data.copy()
            spectrum.y = y_data.copy()
            spectrum.y0 = y_data.copy()
            
            # Set default baseline settings (matching legacy)
            spectrum.baseline.mode = "Linear"
            spectrum.baseline.sigma = 4
            
            # Add metadata if available (from WDF maps)
            if map_metadata:
                spectrum.metadata = map_metadata.copy()
            
            # Add to main collection
            self.spectra.add(spectrum)
    
    def _show_map_spectra(self, map_name: str):
        """Display spectra for the selected map in the spectra list."""
        # Filter spectra by fname prefix: "{map_name}_("
        fname_prefix = f"{map_name}_("
        map_spectra = [
            s for s in self.spectra 
            if s.fname.startswith(fname_prefix)
        ]
        
        # Single batched signal emission to update view (pass spectrum objects)
        self.spectra_list_changed.emit(map_spectra)
        self.count_changed.emit(len(map_spectra))

    
    def get_current_map_dataframe(self) -> pd.DataFrame | None:
        """Get the DataFrame of the currently selected map (filtered by checked spectra).
        
        Returns only rows corresponding to checked spectra in the list.
        """
        if not self.current_map_name or self.current_map_name not in self.maps:
            return None
        
        df = self.maps[self.current_map_name]
        
        # Filter by active spectra without triggering full spectrum instantiation
        active_fnames = {s.fname for s in self.spectra if s.is_active and s.fname.startswith(f"{self.current_map_name}_(")}
        
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
        
        fname_prefix = f"{self.current_map_name}_("
        self.selected_fnames = [
            s.fname for s in self.spectra 
            if s.fname.startswith(fname_prefix)
        ]
        self._emit_selected_spectra()
    
    def reinit_current_map_spectra(self, apply_all: bool = False):
        """Reinitialize selected spectra or all spectra from all maps."""
        # Delegate to parent's reinit_spectra which already handles fname-based selection
        self.reinit_spectra(apply_all)
    
    def reinit_spectra(self, apply_all: bool = False):
        """Override parent to refresh map-specific list after reinit."""
        # Call parent implementation
        super().reinit_spectra(apply_all)
        
        # Refresh current map's spectra list with updated colors
        if self.current_map_name:
            self._show_map_spectra(self.current_map_name)
    
    def delete_baseline(self, apply_all: bool = False):
        """Override parent to refresh map-specific list after baseline deletion."""
        # Call parent implementation
        super().delete_baseline(apply_all)
        
        # Refresh current map's spectra list with updated colors
        if self.current_map_name:
            self._show_map_spectra(self.current_map_name)
    
    def delete_peaks(self, apply_all: bool = False):
        """Override parent to refresh map-specific list after peaks deletion."""
        # Call parent implementation
        super().delete_peaks(apply_all)
        
        # Refresh current map's spectra list with updated colors
        if self.current_map_name:
            self._show_map_spectra(self.current_map_name)
    
    def send_selected_spectra_to_spectra_workspace(self):
        """Send selected spectra to the Spectra workspace tab for comparison."""
        if not self.selected_fnames:
            self.notify.emit("No spectrum selected.")
            return
        
        selected_spectra = self._get_spectra_by_fnames(self.selected_fnames)
        if not selected_spectra:
            return
        
        # Create deep copies to avoid modifying the original spectra
        spectra_copies = [deepcopy(spectrum) for spectrum in selected_spectra]
        
        # Emit signal with the copies
        self.send_spectra_to_workspace.emit(spectra_copies)
        self.notify.emit(f"Sent {len(spectra_copies)} spectra to Spectra workspace.")
    
    def remove_map(self, map_name: str):
        """Remove a map and its spectra from the loaded maps."""
        if map_name not in self.maps:
            return
        
        del self.maps[map_name]
        
        # Remove all spectra belonging to this map (filter by fname prefix)
        fname_prefix = f"{map_name}_("
        indices_to_remove = [
            idx for idx, spectrum in enumerate(self.spectra)
            if spectrum.fname.startswith(fname_prefix)
        ]
        
        if indices_to_remove:
            self.spectra.remove(indices_to_remove)
        
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
    
    
    def get_fit_results_dataframe(self):
        """Get fit results DataFrame for the current map.
        
        This method returns a simplified version of the fit results for heatmap visualization.
        It reuses self.df_fit_results (created by collect_fit_results) to avoid redundancy.
        """
        # If we have collected fit results (from collect_fit_results), use them
        if self.df_fit_results is not None and not self.df_fit_results.empty:
            return self.df_fit_results
        
        # Otherwise return empty DataFrame
        return pd.DataFrame()
    



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

    def _sync_fit_results_to_store(self):
        """Override: harvest fitted parameters into SpectraStore for the current map."""
        if self.current_map_name and self.store.get_map_info(self.current_map_name):
            self._writeback_fit_results_to_store(self.current_map_name)

    def _on_fit_finished(self):
        """Override: harvest fitted parameters into SpectraStore, then update heatmap."""
        super()._on_fit_finished()

        # Re-emit map data to update heatmap with new fit results
        if self.current_map_name:
            self.map_data_updated.emit(self.current_map_df)
            self.clear_map_cache_requested.emit(self.current_map_name)

    def _writeback_fit_results_to_store(self, map_name: str):
        """Harvest fit parameters from MSpectrum objects into SpectraStore arrays.

        Iterates only over spectra belonging to `map_name` and collects their
        `result_fit.params` into contiguous (N, K) matrices so that
        collect_fit_results() can use the fast SpectraStore tensor path.
        """
        fname_prefix = f"{map_name}_("
        map_spectra = [
            s for s in self.spectra
            if s.fname.startswith(fname_prefix)
        ]

        # Determine parameter schema from first successfully fitted spectrum
        param_names = []
        fit_model_ref = None
        for s in map_spectra:
            if hasattr(s, 'result_fit') and hasattr(s.result_fit, 'params') and s.result_fit.params:
                param_names = list(s.result_fit.params.keys())
                # Also capture the fit model for serialization
                if hasattr(s, 'peak_models') and s.peak_models:
                    fit_model_ref = s.save(save_data=False)
                break

        if not param_names:
            return  # No fitted spectra — nothing to write back

        N = len(map_spectra)
        K = len(param_names)
        peak_params = np.zeros((N, K), dtype=np.float64)
        fit_success = np.zeros(N, dtype=bool)
        fit_r2 = np.zeros(N, dtype=np.float64)

        for i, spectrum in enumerate(map_spectra):
            rf = getattr(spectrum, 'result_fit', None)
            if rf is not None and hasattr(rf, 'params') and rf.params:
                for j, pname in enumerate(param_names):
                    pval = rf.params.get(pname)
                    if pval is not None:
                        # Support both ParamValue(.value) and raw float/lmfit Parameter
                        if hasattr(pval, 'value'):
                            peak_params[i, j] = float(pval.value)
                        else:
                            try:
                                peak_params[i, j] = float(pval)
                            except (TypeError, ValueError):
                                pass
                fit_success[i] = bool(getattr(rf, 'success', False))
                fit_r2[i] = float(getattr(rf, 'rsquared', 0.0))

        # Build local indices (0..N-1) for this map
        info = self.store.get_map_info(map_name)
        indices = np.arange(info.n_spectra)

        # Sync is_active flags from MSpectrum objects into store
        md = self.store.get_map_data(map_name)
        if md is not None:
            for i, spectrum in enumerate(map_spectra):
                md.is_active[i] = spectrum.is_active

        self.store.set_fit_results(
            map_name=map_name,
            indices=indices,
            peak_params=peak_params,
            success=fit_success,
            r2=fit_r2,
            param_names=param_names,
            fit_model=fit_model_ref or {},
        )
    
    def collect_fit_results(self):
        """Collect fit results, delegating to SpectraStore when possible.

        Phase-1 fast path: if the store has fit results for the current map,
        build the DataFrame entirely from tensor arrays (no Python object loop).
        Fallback: call parent's collect_fit_results (per-object loop) for the
        Spectra workspace or when the store has no results yet.
        """
        if (
            self.current_map_name
            and self.store.has_fit_results(self.current_map_name)
            and self.store.get_map_info(self.current_map_name) is not None
        ):
            # ── Fast tensor path ──────────────────────────────────────────
            # Ensure _fitmodel_clipboard is populated (so peak labels are available)
            # Use the first fitted spectrum in the current map
            if not self._fitmodel_clipboard:
                map_prefix = f"{self.current_map_name}_("
                for s in self.spectra:
                    if s.fname.startswith(map_prefix):
                        # Trigger lazy load so peak_models are available
                        self._ensure_spectrum_loaded(s)
                        if hasattr(s, 'peak_models') and s.peak_models:
                            self._fitmodel_clipboard = deepcopy(s.save())
                            break

            peak_labels = None
            if self._fitmodel_clipboard:
                peak_labels = self._fitmodel_clipboard.get('peak_labels', None)

            df = self.store.build_fit_results_df(
                name=self.current_map_name,
                map_type=self.map_type,
                peak_labels=peak_labels,
            )
            if df is None or df.empty:
                self.notify.emit("No fit results to collect.")
                return

            self.df_fit_results = df
        else:
            # ── Legacy per-object path (Spectra workspace or no store results) ──
            # Call parent's collect_fit_results to generate base DataFrame
            super().collect_fit_results()

            # If no fit results, nothing to do
            if self.df_fit_results is None or self.df_fit_results.empty:
                return

            # Extract map_name, X, and Y using vectorized regex on Filename column
            # Expected format: "map_name_(X, Y)"
            extracted = self.df_fit_results['Filename'].str.extract(r'^(.*?)\_\(([^,]+),\s*([^)]+)\)$')

            map_names = extracted[0].fillna(self.df_fit_results['Filename'])
            x_coords = pd.to_numeric(extracted[1], errors='coerce')
            y_coords = pd.to_numeric(extracted[2], errors='coerce')

            self.df_fit_results['Filename'] = map_names
            self.df_fit_results.insert(1, 'X', x_coords)
            self.df_fit_results.insert(2, 'Y', y_coords)

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

        # Emit updated DataFrame
        self.fit_results_updated.emit(self.df_fit_results)

        # Trigger heatmap refresh with new fit results
        # This ensures the map viewer uses fresh data instead of cache
        if self.current_map_name:
            # Clear VMapViewer's griddata cache to force recomputation
            self.clear_map_cache_requested.emit(self.current_map_name)
            # Emit map data update to trigger plot refresh
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

    def save_work(self):
        """Save maps workspace using ultra-fast ZIP+NPZ+Pickle architecture (format v4).

        v4 change: Spectral data is stored exclusively in the SpectraStore arrays.
        The heavy per-spectrum metadata dict (spectrums_meta) is replaced by
        lightweight per-map metadata stored in store_meta_{map_name}.
        """
        file_path, _ = QFileDialog.getSaveFileName(
            None, "Save work", "", "SPECTROview Files (*.maps)"
        )
        if not file_path:
            return

        try:
            # 1. Build per-map lightweight metadata for JSON
            store_meta = {}
            for map_name in self.store.map_names:
                store_meta[map_name] = self.store.to_metadata_dict(map_name)

            # Per-spectrum state not in store (xcorrection, color, label) is
            # captured from MSpectrum objects and stored in store_meta already
            # (colors/labels are synced into the store during normal operation).

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
                'format_version': 4,
                'store_meta': _sanitize_for_json(store_meta),
                'maps_metadata': _sanitize_for_json(self.maps_metadata),
                'map_type': self.map_type,
            }

            # 2. Store arrays: SpectraStore owns all heavy numerical data
            arrays = {}
            for map_name in self.store.map_names:
                arrays.update(self.store.to_npz_dict(map_name))

            # 3. Pickle DataFrames (fit results)
            dataframes = {}
            if self.df_fit_results is not None and not self.df_fit_results.empty:
                dataframes['df_fit_results'] = self.df_fit_results

            WorkspaceIO.save_workspace(file_path, metadata, arrays, dataframes)
            self.notify.emit("Maps workspace saved successfully.")

        except Exception as e:
            QMessageBox.critical(None, "Save Error", f"Error saving maps workspace:\n{str(e)}")

    def load_work(self, file_path: str):
        """Load maps workspace — auto-detects format version (v4, v3, or legacy)."""
        try:
            metadata, arrays, dataframes, is_legacy = WorkspaceIO.load_workspace(file_path)

            if is_legacy:
                self.load_work_legacy(file_path)
                return

            self.clear_workspace()
            fmt = metadata.get('format_version', 3)

            if fmt >= 4:
                self._load_work_v4(metadata, arrays, dataframes)
            else:
                self._load_work_v3(metadata, arrays, dataframes)

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(None, "Load Error", f"Error loading maps workspace:\n{str(e)}")

    def _load_work_v4(self, metadata: dict, arrays: dict, dataframes: dict):
        """Restore workspace from format v4 (SpectraStore-backed)."""
        self.maps_metadata = metadata.get('maps_metadata', {})
        self.map_type = metadata.get('map_type', '2Dmap')
        store_meta = metadata.get('store_meta', {})

        # 1. Restore SpectraStore from arrays + per-map metadata
        self.store = SpectraStore()
        for map_name, meta in store_meta.items():
            SpectraStore.load_map_from_npz(arrays, meta, map_name, store=self.store)

        # 2. Rebuild legacy DataFrames for heatmap rendering using per-map MapData
        self.maps = {}
        self._maps_arrays_cache = {}
        for map_name in self.store.map_names:
            md = self.store.get_map_data(map_name)
            x0 = md.x0
            coords = md.coords
            intensities = md.Y0

            col_names = ['X', 'Y'] + [str(float(w)) for w in x0]
            data_combined = np.hstack([coords, intensities.astype(np.float64)])
            self.maps[map_name] = pd.DataFrame(data_combined, columns=col_names)

            # Populate lazy-load cache for _ensure_spectrum_loaded.
            # In v4, the store x0 is already the correctly-cropped wavenumber axis
            # (the last-element-drop happened in _extract_spectra_from_map), so
            # use it directly without further slicing.
            self._maps_arrays_cache[map_name] = {
                'coords': coords,
                'x0': x0,
                'y0': intensities,
            }

        # 3. Reconstruct lightweight MSpectrum shells (lazy-loaded on demand)
        self.spectra = MSpectra()
        for map_name, meta in store_meta.items():
            md = self.store.get_map_data(map_name)
            fnames = md.fnames
            colors = md.colors
            labels = md.labels
            is_active_arr = md.is_active
            map_metadata = self.maps_metadata.get(map_name, {})

            for i, fname in enumerate(fnames):
                spectrum = MSpectrum()
                spectrum.fname = fname
                spectrum.is_active = bool(is_active_arr[i])
                spectrum.color = colors[i]
                spectrum.label = labels[i]
                if map_metadata:
                    spectrum.metadata = map_metadata.copy()
                spectrum.is_preprocessed = False
                spectrum._is_lazy_loaded = False
                # Attach a minimal _lazy_meta so _ensure_spectrum_loaded can
                # inject arrays from the cache when needed
                spectrum._lazy_meta = {'fname': fname}
                self.spectra.append(spectrum)

        # 4. Restore DataFrames
        self.df_fit_results = dataframes.get('df_fit_results') if dataframes else None

        # 5. Refresh View
        map_names = self.store.map_names
        if map_names:
            self.select_map(map_names[0])
        self.maps_list_changed.emit(map_names)
        self.count_changed.emit(len(self.spectra))
        if self.df_fit_results is not None:
            self.fit_results_updated.emit(self.df_fit_results)
        self.notify.emit("Maps workspace loaded successfully.")

    def _load_work_v3(self, metadata: dict, arrays: dict, dataframes: dict):
        """Restore workspace from format v3 (legacy spectrums_meta dict)."""
        self.maps = {}
        self._maps_arrays_cache = {}

        for k in arrays.keys():
            if k.endswith('_coords'):
                map_name = k[4:-7]
                coords = arrays[f'map_{map_name}_coords']
                wavenumbers = arrays[f'map_{map_name}_x0']
                intensities = arrays[f'map_{map_name}_y0']

                if len(wavenumbers) > 1:
                    x0_spectra = wavenumbers[:-1]
                    y0_spectra = intensities[:, :-1]
                else:
                    x0_spectra = wavenumbers
                    y0_spectra = intensities

                self._maps_arrays_cache[map_name] = {
                    'coords': coords,
                    'x0': x0_spectra,
                    'y0': y0_spectra,
                }

                col_names = ['X', 'Y'] + [str(w) for w in wavenumbers]
                data_combined = np.hstack([coords, intensities.astype(np.float64)])
                self.maps[map_name] = pd.DataFrame(data_combined, columns=col_names)

                # Also populate SpectraStore for consistency
                N = len(coords)
                fnames = [
                    f"{map_name}_({float(coords[i, 0])}, {float(coords[i, 1])})"
                    for i in range(N)
                ]
                self.store.add_map(
                    name=map_name,
                    x0=x0_spectra,
                    Y0=y0_spectra,
                    coords=coords,
                    fnames=fnames,
                    is_active=np.ones(N, dtype=bool),
                )

        self.maps_metadata = metadata.get('maps_metadata', {})

        # Reconstruct lazy MSpectrum shells
        self.spectra = MSpectra()
        spectrums_meta = metadata.get('spectrums_meta', {})

        for i, (spec_id, spec_meta) in enumerate(spectrums_meta.items()):
            spec_meta.pop('x0', None)
            spec_meta.pop('y0', None)

            spectrum = MSpectrum()
            spectrum.fname = spec_meta.get('fname', '')
            spectrum.is_active = spec_meta.get('is_active', True)
            spectrum.color = spec_meta.get('color', None)
            spectrum.label = spec_meta.get('label', None)
            spectrum._lazy_meta = spec_meta

            map_name = spectrum.fname.rsplit('_', 1)[0]
            map_metadata = self.maps_metadata.get(map_name, {})
            if map_metadata:
                spectrum.metadata = map_metadata.copy()

            spectrum.is_preprocessed = False
            spectrum._is_lazy_loaded = False
            self.spectra.append(spectrum)

        self.df_fit_results = dataframes.get('df_fit_results') if dataframes else None

        map_names = list(self.maps.keys())
        if map_names:
            self.select_map(map_names[0])
        self.maps_list_changed.emit(map_names)
        self.count_changed.emit(len(self.spectra))
        if self.df_fit_results is not None:
            self.fit_results_updated.emit(self.df_fit_results)
        self.notify.emit("Maps workspace loaded successfully.")

    def load_work_legacy(self, file_path: str):
        """Legacy JSON loader for maps workspace."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Clear current workspace first
            self.clear_workspace()
            
            # Load and decompress map DataFrames
            self.maps = {}
            self.maps_metadata = {}
            
            for map_name, entry in data.get('maps', {}).items():
                self.maps[map_name] = self._binary_to_map_df(entry)
            
            # Restore maps metadata (for WDF files)
            if 'maps_metadata' in data and data['maps_metadata'] is not None:
                self.maps_metadata = data['maps_metadata']
            
            # Reconstruct spectrum objects from saved data using vectorized KDTree queries
            self.spectra = MSpectra()
            spectrums_data = data.get('spectrums_data', {})
            
            # 1. Group coordinates by map for batched querying
            map_queries = {}
            for spectrum_id, spectrum_data in spectrums_data.items():
                fname = spectrum_data.get('fname', '')
                try:
                    map_name, coord_str = fname.rsplit('_', 1)
                    coord_str = coord_str.strip('()')
                    coord = tuple(map(float, coord_str.split(',')))
                    if map_name not in map_queries:
                        map_queries[map_name] = {'coords': [], 'ids': []}
                    map_queries[map_name]['coords'].append(coord)
                    map_queries[map_name]['ids'].append(spectrum_id)
                except Exception:
                    continue
            
            # 2. Batch query KDTree and pre-assign mapping
            precalculated_data = {}
            for map_name, map_df_full in self.maps.items():
                if map_name not in map_queries:
                    continue
                
                # Build tree
                tree_df = map_df_full.iloc[:, :-1]
                tree_coords = tree_df[['X', 'Y']].values
                tree = KDTree(tree_coords)
                df_x0 = tree_df.columns[2:].astype(float).values
                mapped_y = tree_df.iloc[:, 2:].values.astype(float)
                
                # Query in batch
                qdata = map_queries[map_name]
                coords_arr = np.array(qdata['coords'])
                distances, indices = tree.query(coords_arr)
                
                # squared distances for threshold
                dist_sq = distances**2
                
                for i, (d_sq, idx_val) in enumerate(zip(dist_sq, indices)):
                    if d_sq < 1e-4:
                        precalculated_data[qdata['ids'][i]] = (df_x0, mapped_y[idx_val])
            
            # 3. Instantiate spectra with pre-matched variables directly
            for spectrum_id, spectrum_data in spectrums_data.items():
                x0_base, y0_base = precalculated_data.get(spectrum_id, (None, None))
                spectrum = self._create_spectrum_from_dict(spectrum_data, x0_base, y0_base)
                self.spectra.append(spectrum)
            
            # Restore metadata from maps_metadata (stored per-map, not per-spectrum)
            for spectrum in self.spectra:
                map_name = spectrum.fname.rsplit('_', 1)[0]
                map_metadata = self.maps_metadata.get(map_name, {})
                if map_metadata:
                    spectrum.metadata = map_metadata.copy()
            
            # Restore fit results DataFrame (including computed columns)
            if 'df_fit_results_binary' in data and data['df_fit_results_binary'] is not None:
                self.df_fit_results = self._binary_to_map_df(data['df_fit_results_binary'])
            elif 'df_fit_results' in data and data['df_fit_results'] is not None:
                # Fallback for old save files
                self.df_fit_results = pd.DataFrame(data['df_fit_results'])
            else:
                self.df_fit_results = None
            
            # Select first map by default
            map_names = list(self.maps.keys())
            if map_names:
                self.select_map(map_names[0])
            
            # Emit updates to View
            self.maps_list_changed.emit(list(self.maps.keys()))
            self.count_changed.emit(len(self.spectra))
            
            # Populate fit results table and map viewer parameter lists
            if self.df_fit_results is not None and not self.df_fit_results.empty:
                self.fit_results_updated.emit(self.df_fit_results)
            
            self.notify.emit("Legacy maps workspace loaded successfully.")
            
        except Exception as e:
            QMessageBox.critical(None, "Load Error", f"Error loading legacy maps workspace:\n{str(e)}")

    def clear_workspace(self):
        """Clear all maps, spectra, and reset workspace to initial state."""
        # Stop any running fit thread first (via parent)
        super().clear_workspace()

        # Clear Maps-specific data structures
        self.maps.clear()
        if hasattr(self, '_maps_arrays_cache'):
            self._maps_arrays_cache.clear()
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

