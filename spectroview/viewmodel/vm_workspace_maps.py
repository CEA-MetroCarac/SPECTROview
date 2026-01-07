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
    
    def __init__(self, settings: MSettings):
        super().__init__(settings)
        self.maps = {}  # dict[str, pd.DataFrame] - {map_name: map_dataframe}
        self.current_map_name = None
        self.current_map_df = None
    
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
        """Select a map and extract its spectra."""
        if map_name not in self.maps:
            return
        
        self.current_map_name = map_name
        self.current_map_df = self.maps[map_name]
        
        # Extract all spectra from the selected map
        self._extract_spectra_from_map(map_name, self.current_map_df)
        
        # Emit signals
        self.map_selected.emit(map_name)
        self.map_data_updated.emit(self.current_map_df)
    
    def _extract_spectra_from_map(self, map_name: str, map_df: pd.DataFrame):
        """Extract all individual spectra from a hyperspectral map dataframe."""
        # Clear current spectra
        self.spectra = MSpectra()
        
        # Extract wavenumber columns (all columns except X, Y)
        wavenumber_cols = [col for col in map_df.columns if col not in ['X', 'Y']]
        
        # Convert to numeric and skip last value (following legacy behavior)
        x_values = pd.to_numeric(wavenumber_cols, errors='coerce').tolist()
        if len(x_values) > 1:
            x_values = x_values[:-1]  # Skip last value
            wavenumber_cols = wavenumber_cols[:-1]
        
        x_data = np.asarray(x_values)
        
        # Create spectrum for each row (spatial point)
        for idx, row in map_df.iterrows():
            x_pos = row['X']
            y_pos = row['Y']
            
            # Extract y values and skip last one
            y_values = row[wavenumber_cols].tolist()
            y_data = np.asarray(y_values)
            
            # Create MSpectrum object
            spectrum = MSpectrum()
            spectrum.fname = f"{map_name}_{tuple([x_pos, y_pos])}"
            spectrum.x = x_data.copy()
            spectrum.x0 = x_data.copy()
            spectrum.y = y_data.copy()
            spectrum.y0 = y_data.copy()
            
            # Set default baseline settings (matching legacy)
            spectrum.baseline.mode = "Linear"
            spectrum.baseline.sigma = 5
            
            # Store map metadata
            if not hasattr(spectrum, 'metadata'):
                spectrum.metadata = {}
            spectrum.metadata['map_name'] = map_name
            spectrum.metadata['x_position'] = x_pos
            spectrum.metadata['y_position'] = y_pos
            spectrum.metadata['point_index'] = idx
            
            self.spectra.add(spectrum)
        
        # Update View
        self._emit_list_update()
        
        # Auto-select first spectrum if exists
        if len(self.spectra) > 0:
            self.selected_indices = [0]
            self._emit_selected_spectra()
    
    def remove_map(self, map_name: str):
        """Remove a map from the loaded maps."""
        if map_name in self.maps:
            del self.maps[map_name]
            
            # If removed map was selected, clear spectra
            if self.current_map_name == map_name:
                self.current_map_name = None
                self.current_map_df = None
                self.spectra = MSpectra()
                self._emit_list_update()
                self.spectra_selection_changed.emit([])
            
            self._emit_maps_list_update()
    
    def _emit_maps_list_update(self):
        """Emit updated list of map names."""
        map_names = list(self.maps.keys())
        self.maps_list_changed.emit(map_names)
    
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
    
    def select_quadrant(self, quadrant: str):
        """
        Select spectra in a specific wafer quadrant.
        
        Args:
            quadrant: 'Q1', 'Q2', 'Q3', 'Q4', 'H' (horizontal), 'V' (vertical), 'All'
        """
        if self.current_map_df is None:
            return
        
        df = self.current_map_df
        x_center = df['X'].mean()
        y_center = df['Y'].mean()
        
        selected = []
        for idx, row in df.iterrows():
            x = row['X']
            y = row['Y']
            
            if quadrant == 'All':
                selected.append(idx)
            elif quadrant == 'Q1':  # Upper right
                if x >= x_center and y >= y_center:
                    selected.append(idx)
            elif quadrant == 'Q2':  # Upper left
                if x < x_center and y >= y_center:
                    selected.append(idx)
            elif quadrant == 'Q3':  # Lower left
                if x < x_center and y < y_center:
                    selected.append(idx)
            elif quadrant == 'Q4':  # Lower right
                if x >= x_center and y < y_center:
                    selected.append(idx)
            elif quadrant == 'H':  # Horizontal center
                if abs(y - y_center) < df['Y'].std() * 0.5:
                    selected.append(idx)
            elif quadrant == 'V':  # Vertical center
                if abs(x - x_center) < df['X'].std() * 0.5:
                    selected.append(idx)
        
        self.selected_indices = selected
        self._emit_selected_spectra()
