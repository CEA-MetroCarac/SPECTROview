"""ViewModel for Maps Workspace - extends Spectra Workspace with hyperspectral map functionality."""
import json
import gzip
from io import StringIO
import numpy as np
import pandas as pd
import pandas as pd
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox

from spectroview.model.m_settings import MSettings
from spectroview.model.m_spectra import MSpectra
from spectroview.model.m_spectrum import MSpectrum
from spectroview.model.m_io import load_map_file
from spectroview.viewmodel.vm_workspace_spectra import VMWorkspaceSpectra
from spectroview.viewmodel.utils import spectrum_to_dict, dict_to_spectrum


class VMWorkspaceMaps(VMWorkspaceSpectra):
    """Maps Workspace ViewModel."""
    
    maps_list_changed = Signal(list)
    map_data_updated = Signal(object)
    send_spectra_to_workspace = Signal(list)
    clear_map_cache_requested = Signal(str)
    
    def __init__(self, settings: MSettings):
        super().__init__(settings)
        self.maps = {}  # {map_name: map_dataframe}
        self.current_map_name = None
        self.current_map_df = None
        
        self._fit_results_cache: pd.DataFrame | None = None
        self._fit_results_cache_dirty: bool = True
    
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
        
        for p in paths:
            path = Path(p)
            map_name = path.stem
            
            if map_name in self.maps:
                self.notify.emit(f"Map '{map_name}' already loaded, skipping.")
                continue
            
            try:
                map_df = load_map_file(path)
                self.maps[map_name] = map_df
                self._extract_spectra_from_map(map_name, map_df)
                loaded_maps.append(map_name)
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Error loading {path.name}: {str(e)}")
        
        if loaded_maps:
            self._emit_maps_list_update()
    
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
        
        x_data = np.asarray(x_values)
        
        # Pre-extract all spatial coordinates and intensity data (faster)
        x_positions = map_df['X'].values
        y_positions = map_df['Y'].values
        intensity_data = map_df[wavenumber_cols].values  # 2D numpy array
        
        # Create spectra for each row (spatial point) and add to main collection
        for idx in range(len(map_df)):
            x_pos = float(x_positions[idx])
            y_pos = float(y_positions[idx])
            y_data = intensity_data[idx]  # Already numpy array, no conversion needed
            
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
    
    def send_selected_spectra_to_spectra_workspace(self):
        """Send selected spectra to the Spectra workspace tab for comparison."""
        if not self.selected_fnames:
            self.notify.emit("No spectrum selected.")
            return
        
        from copy import deepcopy
        
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
        """Get fit results DataFrame for the current map (cached for heatmap display).
        
        This method returns a simplified version of the fit results for heatmap visualization.
        It reuses self.df_fit_results (created by collect_fit_results) to avoid redundancy.
        """
        # Return cached results if available and not dirty
        if not self._fit_results_cache_dirty and self._fit_results_cache is not None:
            return self._fit_results_cache
        
        # If we have collected fit results (from collect_fit_results), use them
        if self.df_fit_results is not None and not self.df_fit_results.empty:
            self._fit_results_cache = self.df_fit_results.copy()
            self._fit_results_cache_dirty = False
            return self._fit_results_cache
        
        # Otherwise return empty DataFrame
        self._fit_results_cache = pd.DataFrame()
        self._fit_results_cache_dirty = False
        return self._fit_results_cache
    
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
        # Invalidate cache so get_fit_results_dataframe will use fresh data
        self._fit_results_cache_dirty = True
        
        # Call parent's collect_fit_results to generate base DataFrame
        super().collect_fit_results()
        
        # If no fit results, nothing to do
        if self.df_fit_results is None or self.df_fit_results.empty:
            return
        
        # For Maps workspace, we need to restructure the DataFrame:
        # - Extract map_name from fname (everything before "_(")
        # - Extract X, Y coordinates from fname
        # - Replace Filename column with just map_name
        
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
    
    def clear_workspace(self):
        """Clear all maps, spectra, and reset workspace to initial state."""
        # Stop any running fit thread first (via parent)
        super().clear_workspace()
        
        # Clear Maps-specific data structures
        self.maps.clear()
        self.current_map_name = None
        self.current_map_df = None
        
        # Clear all caches
        self._fit_results_cache = None
        self._fit_results_cache_dirty = True
        
        # Emit updates to View
        self.maps_list_changed.emit([])
        self.map_data_updated.emit(pd.DataFrame())
        # Parent already emits: spectra_list_changed, spectra_selection_changed, 
        # count_changed, fit_in_progress, fit_results_updated
    
    # ─────────────────────────────────────────────────────────────────
    # SAVE/LOAD WORKSPACE
    # ─────────────────────────────────────────────────────────────────
    
    def save_work(self):
        """Save current maps workspace to .maps file (100% compatible with legacy format)."""  
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
            spectrums_data = spectrum_to_dict(self.spectra, is_map=True)
            
            # Compress map DataFrames (gzip + hex encoding for JSON compatibility)
            compressed_maps = {}
            for map_name, map_df in self.maps.items():
                csv_data = map_df.to_csv(index=False).encode('utf-8')
                compressed_maps[map_name] = gzip.compress(csv_data)
            
            # Prepare data structure matching legacy format exactly
            data_to_save = {
                'spectrums_data': spectrums_data,
                'maps': {k: v.hex() for k, v in compressed_maps.items()},
            }
            
            # Save to JSON file
            with open(file_path, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            
            self.notify.emit("Maps workspace saved successfully.")
            
        except Exception as e:
            QMessageBox.critical(None, "Save Error", f"Error saving maps workspace:\n{str(e)}")
    
    def load_work(self, file_path: str):
        """Load maps workspace from .maps file (100% compatible with legacy format)."""
        
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Clear current workspace first
            self.clear_workspace()
            
            # Load and decompress map DataFrames
            self.maps = {}
            for map_name, hex_data in data.get('maps', {}).items():
                compressed_data = bytes.fromhex(hex_data)
                csv_data = gzip.decompress(compressed_data).decode('utf-8')
                self.maps[map_name] = pd.read_csv(StringIO(csv_data))
            
            # Reconstruct spectrum objects from saved data
            self.spectra = MSpectra()
            for spectrum_id, spectrum_data in data.get('spectrums_data', {}).items():
                spectrum = MSpectrum()
                dict_to_spectrum(
                    spectrum=spectrum,
                    spectrum_data=spectrum_data,
                    maps=self.maps,
                    is_map=True
                )
                spectrum.preprocess()
                self.spectra.append(spectrum)
            
            # Select first map by default
            map_names = list(self.maps.keys())
            if map_names:
                self.select_map(map_names[0])
            
            # Emit updates to View
            self.maps_list_changed.emit(list(self.maps.keys()))
            self.count_changed.emit(len(self.spectra))
            self.notify.emit(f"Loaded {len(self.maps)} map(s) with {len(self.spectra)} spectra")
            
        except Exception as e:
            QMessageBox.critical(None, "Load Error", f"Error loading maps workspace:\n{str(e)}")


