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
    send_spectra_to_workspace = Signal(list)  # list[MSpectrum] - spectra to send to Spectra workspace
    
    def __init__(self, settings: MSettings):
        super().__init__(settings)
        self.maps = {}  # dict[str, pd.DataFrame] - {map_name: map_dataframe}
        self.map_spectra = {}  # dict[str, list[MSpectrum]] - {map_name: [spectra...]}
        self.current_map_name = None
        self.current_map_df = None
        self.current_map_indices = []  # Global indices of currently displayed map's spectra
    
    # ─────────────────────────────────────────────────────────────────
    # MAPS LOADING AND MANAGEMENT
    # ─────────────────────────────────────────────────────────────────
    
    def load_map_files(self, paths: list[str]):
        """Load hyperspectral map files (CSV, TXT formats)."""
        loaded_maps = []
        
        for p in paths:
            path = Path(p)
            map_name = path.stem
            
            # Skip if already loaded
            if map_name in self.maps:
                self.notify.emit(f"Map '{map_name}' already loaded, skipping.")
                continue
            
            # Load map dataframe based on file extension
            try:
                map_df = self._load_map_dataframe(path)
                self.maps[map_name] = map_df
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
        """Select a map and show its spectra."""
        if map_name not in self.maps:
            return
        
        self.current_map_name = map_name
        self.current_map_df = self.maps[map_name]
        
        # Extract spectra if not already extracted for this map
        if map_name not in self.map_spectra:
            self._extract_spectra_from_map(map_name, self.current_map_df)
        else:
            # Use existing spectra (preserves fitting results)
            self._show_map_spectra(map_name)
        
        # Emit signals
        self.map_selected.emit(map_name)
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
        
        # Create list to store spectra for this map
        map_spectra_list = []
        
        # Create spectrum for each row (spatial point)
        for idx, row in map_df.iterrows():
            x_pos = float(row['X'])
            y_pos = float(row['Y'])
            
            # Extract y values and skip last one
            y_values = row[wavenumber_cols].tolist()
            y_data = np.asarray(y_values)
            
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
            
            map_spectra_list.append(spectrum)
        
        # Store the extracted spectra with this map
        self.map_spectra[map_name] = map_spectra_list
        
        # Add these spectra to the main collection
        for spectrum in map_spectra_list:
            self.spectra.add(spectrum)
        
        # Update View to show only this map's spectra
        self._show_map_spectra(map_name)
    
    def _show_map_spectra(self, map_name: str):
        """Display spectra for the selected map in the spectra list."""
        if map_name not in self.map_spectra:
            return
        
        # Get spectra for this map
        map_spectra = self.map_spectra[map_name]
        spectra_names = [s.fname for s in map_spectra]
        
        # Find global indices of this map's spectra in self.spectra
        all_spectra_list = list(self.spectra)
        self.current_map_indices = []
        for spectrum in map_spectra:
            try:
                idx = all_spectra_list.index(spectrum)
                self.current_map_indices.append(idx)
            except ValueError:
                pass  # Spectrum not found
        
        # Update the view to show only this map's spectra
        self.spectra_list_changed.emit(spectra_names)
        self.count_changed.emit(len(map_spectra))
        
        # Auto-select first spectrum if exists
        if self.current_map_indices:
            self.selected_indices = [self.current_map_indices[0]]
            self._emit_selected_spectra()
    
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
        """Override to map displayed list indices to global spectra indices."""
        if not self.current_map_indices:
            # No map selected, use parent behavior
            super().set_selected_indices(indices)
            return
        
        # Map from displayed list indices to global indices
        global_indices = []
        for idx in indices:
            if 0 <= idx < len(self.current_map_indices):
                global_indices.append(self.current_map_indices[idx])
        
        # Call parent with global indices
        super().set_selected_indices(global_indices)
    
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

