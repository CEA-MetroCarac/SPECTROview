"""ViewModel for Maps Workspace - extends Spectra Workspace with hyperspectral map functionality."""
import json
import gzip
from io import StringIO
import numpy as np
import pandas as pd
from pathlib import Path
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QMessageBox

from spectroview.model.m_settings import MSettings
from spectroview.model.m_spectra import MSpectra
from spectroview.model.m_spectrum import MSpectrum
from spectroview.model.m_io import load_map_file
from spectroview.viewmodel.vm_workspace_spectra import VMWorkspaceSpectra
from spectroview.viewmodel.utils import spectrum_to_dict, dict_to_spectrum


class VMWorkspaceMaps(VMWorkspaceSpectra):
    """Maps Workspace ViewModel."""
    
    maps_list_changed = Signal(list)
    map_selected = Signal(str)
    map_data_updated = Signal(object)
    selection_indices_to_restore = Signal(list)
    send_spectra_to_workspace = Signal(list)
    clear_map_cache_requested = Signal(str)
    
    def __init__(self, settings: MSettings):
        super().__init__(settings)
        self.maps = {}  # {map_name: map_dataframe}
        self.current_map_name = None
        self.current_map_df = None
        self.current_map_indices = []  # Global indices for current map
        self.selected_list_indices = []  # Persists across map switches
        
        self._fit_results_cache: pd.DataFrame | None = None
        self._fit_results_cache_dirty: bool = True
    
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
        
        spectra_names = []
        self.current_map_indices = []
        
        for idx, spectrum in enumerate(self.spectra):
            if spectrum.fname.startswith(fname_prefix):
                spectra_names.append(spectrum.fname)
                self.current_map_indices.append(idx)
        
        # Single batched signal emission to update view
        self.spectra_list_changed.emit(spectra_names)
        self.count_changed.emit(len(spectra_names))
    
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
        """Reinitialize selected spectra or all spectra from all maps """
        if apply_all:
            # Reinit all spectra
            for spectrum in self.spectra:
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
            self.current_map_indices = []
            self.selected_indices = []
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
            
            # Parse map_name and coordinates from fname: "map_name_(x, y)"
            fname = spectrum.fname
            if '(' not in fname or ')' not in fname:
                continue  # Skip if fname format is unexpected
            
            # Extract map_name (everything before last '(')
            map_name = fname[:fname.rfind('(')].rstrip('_')
            
            # Extract coordinates
            coords_str = fname[fname.rfind('(')+1:fname.rfind(')')]
            try:
                x_str, y_str = coords_str.split(',')
                x_pos = float(x_str.strip())
                y_pos = float(y_str.strip())
            except (ValueError, AttributeError):
                continue  # Skip if parsing fails
            
            # Start row with identification
            row = {
                'Filename': map_name,
                'X': x_pos,
                'Y': y_pos
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
    
    def _rebuild_current_map_indices(self):
        """Rebuild current_map_indices by finding all spectra for the current map.
        
        This is necessary after operations that might modify the spectra list
        (like fitting) to ensure coordinate-based lookups still work.
        """
        if not self.current_map_name:
            return
        
        # Rebuild current_map_indices by filtering fname prefix
        fname_prefix = f"{self.current_map_name}_("
        self.current_map_indices = [
            idx for idx, spectrum in enumerate(self.spectra)
            if spectrum.fname.startswith(fname_prefix)
        ]
    
    def _on_fit_finished(self):
        """Override to invalidate cache when fitting completes."""
        super()._on_fit_finished()
        
        # Rebuild current_map_indices to ensure coordinate lookups work
        # (fitting might have modified the spectra list indices)
        if self.current_map_name:
            self._rebuild_current_map_indices()
            # Re-emit map data to update heatmap with new fit results
            self.map_data_updated.emit(self.current_map_df)
            
            # Signal View to clear griddata cache for this map (data has changed)
            self.clear_map_cache_requested.emit(self.current_map_name)
    
    def clear_workspace(self):
        """Clear all maps, spectra, and reset workspace to initial state."""
        # Stop any running fit thread first (via parent)
        super().clear_workspace()
        
        # Clear Maps-specific data structures
        self.maps.clear()
        self.current_map_name = None
        self.current_map_df = None
        self.current_map_indices = []
        self.selected_list_indices = []
        
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
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        
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


