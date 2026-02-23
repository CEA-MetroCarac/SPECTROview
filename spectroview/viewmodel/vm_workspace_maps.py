"""ViewModel for Maps Workspace - extends Spectra Workspace with hyperspectral map functionality."""
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

from spectroview.model.m_settings import MSettings
from spectroview.model.m_spectra import MSpectra
from spectroview.model.m_spectrum import MSpectrum
from spectroview.model.m_io import load_map_file, load_wdf_map, load_spc_map
from spectroview.viewmodel.vm_workspace_spectra import VMWorkspaceSpectra



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
        

        
        # Reference to Graphs workspace (injected after construction)
        self.graphs_workspace = None
    
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
        
        # Create spectra for each row (spatial point) and add to main collection
        for idx in range(len(map_df)):
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
        
        # Filter by active spectra
        active_spectra = self._get_active_spectra()
        active_fnames = {s.fname for s in active_spectra}
        
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
        
        # Filter DataFrame rows by (X, Y) coordinates
        mask = df.apply(lambda row: (row['X'], row['Y']) in active_coords, axis=1)
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
    
    def _on_fit_finished(self):
        """Override to invalidate cache when fitting completes."""
        super()._on_fit_finished()
        
        # Re-emit map data to update heatmap with new fit results
        if self.current_map_name:
            self.map_data_updated.emit(self.current_map_df)
            # Signal View to clear griddata cache for this map (data has changed)
            self.clear_map_cache_requested.emit(self.current_map_name)
    
    def collect_fit_results(self):
        """Override parent to add X and Y coordinate columns for map data."""

        
        # Call parent's collect_fit_results to generate base DataFrame
        super().collect_fit_results()
        
        # If no fit results, nothing to do
        if self.df_fit_results is None or self.df_fit_results.empty:
            return
        
        map_names = []
        x_coords = []
        y_coords = []
        
        for fname in self.df_fit_results['Filename']:
            # Extract coordinates from fname: "map_name_(x, y)"
            if '(' in fname and ')' in fname:
                # Extract map_name (everything before last '_(')
                map_name = fname[:fname.rfind('_(')]
                map_names.append(map_name)
                
                # Extract coordinates
                coords_str = fname[fname.rfind('(')+1:fname.rfind(')')]
                try:
                    x_str, y_str = coords_str.split(',')
                    x_coords.append(float(x_str.strip()))
                    y_coords.append(float(y_str.strip()))
                except (ValueError, AttributeError):
                    x_coords.append(None)
                    y_coords.append(None)
            else:
                # Fallback if format is unexpected
                map_names.append(fname)
                x_coords.append(None)
                y_coords.append(None)
        
        # Replace Filename column with map_name only
        self.df_fit_results['Filename'] = map_names
        
        # Insert X and Y columns after Filename (at positions 1 and 2)
        self.df_fit_results.insert(1, 'X', x_coords)
        self.df_fit_results.insert(2, 'Y', y_coords)
        
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
        """Save current maps workspace to .maps file."""
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save work",
            "",
            "SPECTROview Files (*.maps)"
        )
        
        if not file_path:
            return
        
        try:
            # Convert all spectra to dict format (is_map=True for map spectra)
            spectrums_data = self.spectra.save(is_map=True)
            
            # Compress map DataFrames using fast numpy binary storage
            maps_binary = {
                map_name: self._map_df_to_binary(map_df)
                for map_name, map_df in self.maps.items()
            }
            
            # Prepare data structure
            data_to_save = {
                'format_version': 2,              # signals new binary map format
                'spectrums_data': spectrums_data,
                'maps': maps_binary,
                'maps_metadata': self.maps_metadata,
            }
            
            # Save fit results DataFrame (including computed columns)
            if self.df_fit_results is not None and not self.df_fit_results.empty:
                # Revert to standard dictionary records for df_fit_results to prevent NumPy Pickling errors on Strings
                data_to_save['df_fit_results'] = self.df_fit_results.to_dict('records')
            else:
                data_to_save['df_fit_results'] = None
            
            # Save to JSON file instantly via memory buffer (avoids 4M+ python I/O write stalls)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(data_to_save))
            
            self.notify.emit("Maps workspace saved successfully.")
            
        except Exception as e:
            QMessageBox.critical(None, "Save Error", f"Error saving maps workspace:\n{str(e)}")
    
    def load_work(self, file_path: str):
        """Load maps workspace from .maps file (supports v1 CSV and v2 binary formats)."""
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Clear current workspace first
            self.clear_workspace()
            
            # Load and decompress map DataFrames
            # Supports both legacy CSV+gzip (format_version absent or 1)
            # and new numpy binary format (format_version == 2)
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
                # Pop metadata before setting attributes to prevent Fitspy crashes on dict
                saved_metadata = spectrum_data.pop('metadata', None)
                
                spectrum = MSpectrum()
                spectrum.set_attributes(spectrum_data)
                
                if saved_metadata:
                    spectrum.metadata = saved_metadata
                
                # Assign precalculated spatial data
                x0_base, y0_base = precalculated_data.get(spectrum_id, (None, None))
                if x0_base is not None and y0_base is not None:
                    spectrum.x0 = x0_base + spectrum.xcorrection_value
                    spectrum.y0 = y0_base
                    
                    # Ensure base arrays exist for preprocess() to use or GUI to render
                    spectrum.x = spectrum.x0.copy()
                    spectrum.y = spectrum.y0.copy()
                else:
                    spectrum.x0 = None
                    spectrum.y0 = None
                    spectrum.x = None
                    spectrum.y = None
                    
                spectrum.preprocess()
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
            
            self.notify.emit(f"Loaded {len(self.maps)} map(s) with {len(self.spectra)} spectra")
            
        except Exception as e:
            QMessageBox.critical(None, "Load Error", f"Error loading maps workspace:\n{str(e)}")
    
    def clear_workspace(self):
        """Clear all maps, spectra, and reset workspace to initial state."""
        # Stop any running fit thread first (via parent)
        super().clear_workspace()
        
        # Clear Maps-specific data structures
        self.maps.clear()
        self.current_map_name = None
        self.current_map_df = None
        
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

