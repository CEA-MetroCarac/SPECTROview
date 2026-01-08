"""ViewModel for Maps Workspace - extends Spectra Workspace with hyperspectral map functionality."""
import numpy as np
import pandas as pd
from pathlib import Path
from PySide6.QtCore import Signal

from spectroview.model.m_settings import MSettings
from spectroview.model.m_spectra import MSpectra
from spectroview.model.m_spectrum import MSpectrum
from spectroview.model.m_io import load_map_file
from spectroview.viewmodel.vm_workspace_spectra import VMWorkspaceSpectra


class VMWorkspaceMaps(VMWorkspaceSpectra):
    """Maps Workspace ViewModel - handles hyperspectral data and map visualization."""
    
    # ───── Additional Maps-specific signals ─────
    maps_list_changed = Signal(list)  # list[str] - map names
    map_selected = Signal(str)  # selected map name
    map_data_updated = Signal(object)  # pd.DataFrame for map visualization
    selection_indices_to_restore = Signal(list)  # List indices to select in widget after map switch
    send_spectra_to_workspace = Signal(list)  # list[MSpectrum] - spectra to send to Spectra workspace
    
    def __init__(self, settings: MSettings):
        super().__init__(settings)
        self.maps = {}  # dict[str, pd.DataFrame] - {map_name: map_dataframe}
        self.map_spectra = {}  # dict[str, list[MSpectrum]] - {map_name: [spectra...]}
        self.current_map_name = None
        self.current_map_df = None
        self.current_map_indices = []  # Global indices of currently displayed map's spectra
        self.selected_list_indices = []  # List indices to persist across map switches
        
        # Cache for fit results
        self._fit_results_cache: pd.DataFrame | None = None
        self._fit_results_cache_dirty: bool = True
        
        # Cache spectrum to index mapping to avoid list conversions
        self._spectrum_to_index: dict = {}  # {id(spectrum): index}
    
    # ─────────────────────────────────────────────────────────────────
    # MAPS LOADING AND MANAGEMENT
    # ─────────────────────────────────────────────────────────────────
    
    def load_map_files(self, paths: list[str]):
        """Load hyperspectral map files and extract spectra immediately (matches legacy)."""
        loaded_maps = []
        
        for p in paths:
            path = Path(p)
            map_name = path.stem
            
            # Skip if already loaded
            if map_name in self.maps:
                self.notify.emit(f"Map '{map_name}' already loaded, skipping.")
                continue
            
            # Load map dataframe
            try:
                map_df = self._load_map_dataframe(path)
                self.maps[map_name] = map_df
                
                # IMMEDIATELY extract spectra (legacy behavior - no lazy loading)
                self._extract_spectra_from_map(map_name, map_df)
                
                loaded_maps.append(map_name)
            except Exception as e:
                self.notify.emit(f"Error loading {path.name}: {str(e)}")
        
        if loaded_maps:
            self._emit_maps_list_update()
            self.notify.emit(f"Loaded {len(loaded_maps)} map(s)")
    
    def _load_map_dataframe(self, path: Path) -> pd.DataFrame:
        """Load hyperspectral dataframe from CSV or TXT file."""
        return load_map_file(path)
    
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
        
        # Restore selection based on list indices (or select first item if no previous selection)
        self._restore_selection_after_map_switch()
        
    
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
        
        # Create list to store spectra for this map (pre-allocate)
        num_spectra = len(map_df)
        map_spectra_list = [None] * num_spectra
        
        # Create spectrum for each row (spatial point) - vectorized operations
        for idx in range(num_spectra):
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
            
            # Store map metadata
            if not hasattr(spectrum, 'metadata'):
                spectrum.metadata = {}
            spectrum.metadata['map_name'] = map_name
            spectrum.metadata['x_position'] = x_pos
            spectrum.metadata['y_position'] = y_pos
            spectrum.metadata['point_index'] = idx
            
            map_spectra_list[idx] = spectrum
        
        # Store the extracted spectra with this map
        self.map_spectra[map_name] = map_spectra_list
        
        # Add these spectra to the main collection and update index cache
        start_idx = len(self.spectra)
        for i, spectrum in enumerate(map_spectra_list):
            self.spectra.add(spectrum)
            self._spectrum_to_index[id(spectrum)] = start_idx + i
        
        # Note: Don't call _show_map_spectra here - extraction happens during load,
        # filtering happens during select_map() when user clicks a map
    
    def _show_map_spectra(self, map_name: str):
        """Display spectra for the selected map in the spectra list."""
        if map_name not in self.map_spectra:
            return
        
        # Get spectra for this map
        map_spectra = self.map_spectra[map_name]
        num_spectra = len(map_spectra)
        
        # Single pass: extract names AND indices together (optimize!)
        spectra_names = []
        self.current_map_indices = []
        
        for spectrum in map_spectra:
            # Extract name
            spectra_names.append(spectrum.fname)
            
            # Get index from cache (O(1) lookup)
            idx = self._spectrum_to_index.get(id(spectrum))
            if idx is not None:
                self.current_map_indices.append(idx)
        
        # Single batched signal emission to update view
        self.spectra_list_changed.emit(spectra_names)
        self.count_changed.emit(num_spectra)
    
    def _restore_selection_after_map_switch(self):
        """Restore selection based on list indices after switching maps."""
        num_spectra = len(self.current_map_indices)
        
        if num_spectra == 0:
            self.selected_indices = []
            self._emit_selected_spectra()
            return
        
        # If we have saved list indices, restore them (if valid for this map)
        if self.selected_list_indices:
            # Convert list indices to global indices for the new map
            restored_global_indices = []
            restored_list_indices = []
            for list_idx in self.selected_list_indices:
                if 0 <= list_idx < num_spectra:
                    global_idx = self.current_map_indices[list_idx]
                    restored_global_indices.append(global_idx)
                    restored_list_indices.append(list_idx)
            
            if restored_global_indices:
                self.selected_indices = restored_global_indices
                self._emit_selected_spectra()
                # Signal View to update list widget selection
                self.selection_indices_to_restore.emit(restored_list_indices)
                return
        
        # No previous selection or all indices invalid → auto-select first item
        first_global_idx = self.current_map_indices[0]
        self.selected_indices = [first_global_idx]
        self.selected_list_indices = [0]
        self._emit_selected_spectra()
        # Signal View to select first item in list widget
        self.selection_indices_to_restore.emit([0])
        if self.selected_list_indices:
            # Convert list indices to global indices for the new map
            restored_global_indices = []
            for list_idx in self.selected_list_indices:
                if 0 <= list_idx < num_spectra:
                    global_idx = self.current_map_indices[list_idx]
                    restored_global_indices.append(global_idx)
            
            if restored_global_indices:
                self.selected_indices = restored_global_indices
                self._emit_selected_spectra()
                # Signal View to update list widget selection
                self.selection_indices_to_restore.emit(self.selected_list_indices)
                return
        
        # No previous selection or all indices invalid → auto-select first item
        first_global_idx = self.current_map_indices[0]
        self.selected_indices = [first_global_idx]
        self.selected_list_indices = [0]
        self._emit_selected_spectra()
        # Signal View to select first item in list widget
        self.selection_indices_to_restore.emit([0])
    
    def get_current_map_dataframe(self) -> pd.DataFrame | None:
        """Get the DataFrame of the currently selected map."""
        if self.current_map_name and self.current_map_name in self.maps:
            return self.maps[self.current_map_name]
        return None
    
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
            self.notify.emit(f"Error saving map: {e}")
    
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
        if not self.current_map_indices:
            return
        
        # Select all indices from current map
        self.selected_indices = self.current_map_indices.copy()
        self._emit_selected_spectra()
    
    def reinit_current_map_spectra(self, apply_all: bool = False):
        """Reinitialize selected spectra or all spectra from all maps.
        
        Args:
            apply_all: If True, reinit ALL spectra from ALL maps. If False, only selected spectra.
        """
        if apply_all:
            # Reinit all spectra from all loaded maps
            for spectra_list in self.map_spectra.values():
                for spectrum in spectra_list:
                    spectrum.reinit()
        else:
            # Reinit only selected spectra
            if not self.selected_indices:
                self.notify.emit("No spectrum selected.")
                return
            
            selected_spectra = self.spectra.get(self.selected_indices)
            for spectrum in selected_spectra:
                spectrum.reinit()
        
        self._emit_selected_spectra()
    
    def send_selected_spectra_to_spectra_workspace(self):
        """Send selected spectra to the Spectra workspace tab for comparison."""
        if not self.selected_indices:
            self.notify.emit("No spectrum selected.")
            return
        
        from copy import deepcopy
        
        selected_spectra = self.spectra.get(self.selected_indices)
        if not selected_spectra:
            return
        
        # Create deep copies to avoid modifying the original spectra
        spectra_copies = [deepcopy(spectrum) for spectrum in selected_spectra]
        
        # Emit signal with the copies
        self.send_spectra_to_workspace.emit(spectra_copies)
        self.notify.emit(f"Sent {len(spectra_copies)} spectra to Spectra workspace.")
    
    def remove_map(self, map_name: str):
        """Remove a map and its spectra from the loaded maps."""
        if map_name in self.maps:
            del self.maps[map_name]
            
            # Remove spectra from main collection
            if map_name in self.map_spectra:
                # Find indices of spectra to remove
                all_spectra_list = list(self.spectra)
                indices_to_remove = []
                
                for spectrum in self.map_spectra[map_name]:
                    try:
                        idx = all_spectra_list.index(spectrum)
                        indices_to_remove.append(idx)
                    except ValueError:
                        pass  # Spectrum not found, skip
                
                # Remove by indices
                if indices_to_remove:
                    self.spectra.remove(indices_to_remove)
                
                del self.map_spectra[map_name]
            
            # If removed map was selected, clear view
            if self.current_map_name == map_name:
                self.current_map_name = None
                self.current_map_df = None
                self.spectra_list_changed.emit([])
                self.count_changed.emit(0)
                self.spectra_selection_changed.emit([])
            
            self._emit_maps_list_update()
    
    def _emit_maps_list_update(self):
        """Emit updated list of map names."""
        map_names = list(self.maps.keys())
        self.maps_list_changed.emit(map_names)
    
    def set_selected_indices(self, indices: list[int]):
        """Override to handle index selection.
        
        Note: This receives GLOBAL indices directly from View layer.
        The View layer does the list-to-global conversion before calling this.
        """
        # Save list indices for persistence across map switches
        self.selected_list_indices = []
        for global_idx in indices:
            if global_idx in self.current_map_indices:
                list_idx = self.current_map_indices.index(global_idx)
                self.selected_list_indices.append(list_idx)
        
        # Pass through to parent - indices are already global
        super().set_selected_indices(indices)
    
    # ─────────────────────────────────────────────────────────────────
    # MAP VISUALIZATION HELPERS
    # ─────────────────────────────────────────────────────────────────
    
    def get_map_heatmap_data(self, parameter: str = 'Intensity') -> dict:
        """
        Get heatmap data for the current map.
        
        Args:
            parameter: 'Intensity', 'Area', or fit parameter name
            
        Returns:
            dict with keys: 'X', 'Y', 'Z', 'map_type'
        """
        if self.current_map_df is None:
            return None
        
        df = self.current_map_df
        
        # Extract X, Y positions
        x_positions = df['X'].values
        y_positions = df['Y'].values
        
        # Calculate Z values based on parameter
        if parameter == 'Intensity':
            # Use maximum intensity for each spectrum
            wavenumber_cols = [col for col in df.columns if col not in ['X', 'Y']]
            z_values = df[wavenumber_cols].max(axis=1).values
        
        elif parameter == 'Area':
            # Calculate area under curve for each spectrum
            wavenumber_cols = [col for col in df.columns if col not in ['X', 'Y']]
            x_data = [float(col) for col in wavenumber_cols]
            
            z_values = []
            for _, row in df.iterrows():
                y_data = row[wavenumber_cols].values.astype(float)
                area = np.trapz(y_data, x_data)
                z_values.append(area)
            z_values = np.array(z_values)
        
        else:
            # Fit parameter - extract from fit results DataFrame
            if self.df_fit_results is not None and parameter in self.df_fit_results.columns:
                z_values = self.df_fit_results[parameter].values
            else:
                return None
        
        return {
            'X': x_positions,
            'Y': y_positions,
            'Z': z_values,
            'map_name': self.current_map_name
        }
    
    def get_spectra_at_position(self, x: float, y: float, tolerance: float = 1.0) -> list[int]:
        """
        Get spectrum indices at a given spatial position.
        
        Args:
            x, y: Spatial coordinates
            tolerance: Maximum distance to consider a match
            
        Returns:
            List of spectrum indices matching the position
        """
        if self.current_map_df is None:
            return []
        
        matches = []
        for idx, spectrum in enumerate(self.spectra):
            if 'metadata' in spectrum.__dict__:
                meta = spectrum.metadata
                x_pos = meta.get('x_position', 0)
                y_pos = meta.get('y_position', 0)
                
                distance = ((x - x_pos)**2 + (y - y_pos)**2)**0.5
                if distance <= tolerance:
                    matches.append(idx)
        
        return matches
    
    def get_fit_results_dataframe(self):
        """Get fit results DataFrame for the current map (cached).
        
        Returns:
            pd.DataFrame: DataFrame with columns [Filename, X, Y, ...fit_parameters]
        """
        # Return cached results if available and not dirty
        if not self._fit_results_cache_dirty and self._fit_results_cache is not None:
            return self._fit_results_cache
        
        import pandas as pd
        
        if not self.spectra or len(self.spectra) == 0:
            self._fit_results_cache = pd.DataFrame()
            self._fit_results_cache_dirty = False
            return self._fit_results_cache
        
        # Quick check: do ANY spectra have fit results? If not, skip entirely
        has_any_fits = any(
            hasattr(s, 'result_fit') and s.result_fit 
            for s in self.spectra
        )
        
        if not has_any_fits:
            self._fit_results_cache = pd.DataFrame()
            self._fit_results_cache_dirty = False
            return self._fit_results_cache
        
        # Collect all fit results
        results = []
        for spectrum in self.spectra:
            # Only include spectra that have been fitted
            if not hasattr(spectrum, 'result_fit') or not spectrum.result_fit:
                continue
            
            # Get metadata
            map_name = spectrum.metadata.get('map_name', '')
            x_pos = spectrum.metadata.get('x_position')
            y_pos = spectrum.metadata.get('y_position')
            
            if x_pos is None or y_pos is None:
                continue
            
            # Start row with identification
            row = {
                'Filename': map_name,
                'X': float(x_pos),
                'Y': float(y_pos)
            }
            
            # Add peak parameters
            if hasattr(spectrum, 'peak_models') and spectrum.peak_models:
                for i, peak_model in enumerate(spectrum.peak_models):
                    peak_label = (spectrum.peak_labels[i] 
                                 if i < len(spectrum.peak_labels) 
                                 else f"Peak_{i+1}")
                    
                    # Get parameters from param_hints
                    for param_name, param_hint in peak_model.param_hints.items():
                        # Extract the parameter key (remove peak prefix)
                        parts = param_name.split('_', 1)
                        if len(parts) == 2:
                            key = parts[1]  # e.g., 'x0', 'amplitude', 'fwhm'
                        else:
                            key = param_name
                        
                        # Create column name
                        col_name = f"{peak_label}_{key}"
                        row[col_name] = param_hint.get('value', 0)
            
            results.append(row)
        
        if not results:
            self._fit_results_cache = pd.DataFrame()
        else:
            self._fit_results_cache = pd.DataFrame(results)
        
        self._fit_results_cache_dirty = False
        return self._fit_results_cache
    
    def invalidate_fit_results_cache(self):
        """Mark fit results cache as dirty (call after fitting)."""
        self._fit_results_cache_dirty = True
    
    def _on_fit_finished(self):
        """Override to invalidate cache when fitting completes."""
        super()._on_fit_finished()
        self.invalidate_fit_results_cache()
        # Re-emit map data to update heatmap with new fit results
        if self.current_map_name:
            self.map_data_updated.emit(self.current_map_df)

