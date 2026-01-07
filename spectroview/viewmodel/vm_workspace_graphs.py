# viewmodel/vm_workspace_graphs.py
"""ViewModel for Graphs Workspace - handles business logic and data management."""

import json
import pandas as pd
import gzip
from io import StringIO

from pathlib import Path
from typing import Dict, List, Optional


from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog

from spectroview.model.m_graph import MGraph
from spectroview.model.m_settings import MSettings
from spectroview.model.m_io import load_dataframe_file


class VMWorkspaceGraphs(QObject):
    """ViewModel for managing graphs, DataFrames, and plotting logic.
    
    Responsibilities:
    - Manage multiple DataFrames
    - Handle graph creation and updates
    - Apply filters to DataFrames
    - Save/load workspace
    """
    
    # ───── ViewModel → View signals ─────
    dataframes_changed = Signal(list)  # List of DataFrame names
    dataframe_selected = Signal(str)  # Selected DataFrame name
    dataframe_columns_changed = Signal(list)  # List of column names
    
    graphs_changed = Signal(list)  # List of graph IDs
    graph_selected = Signal(int)  # Selected graph ID
    graph_properties_changed = Signal(object)  # MGraph object
    
    filtered_data_ready = Signal(object)  # Filtered DataFrame
    
    notify = Signal(str)  # General notifications
    
    def __init__(self, settings: MSettings):
        """Initialize the ViewModel."""
        super().__init__()
        self.settings = settings
        
        # Data storage
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.graphs: Dict[int, MGraph] = {}
        
        # Current selections
        self.selected_df_name: Optional[str] = None
        self.selected_graph_id: Optional[int] = None
        
        # Graph ID counter
        self._next_graph_id = 1
    
    # ═════════════════════════════════════════════════════════════════════
    # DataFrame Management
    # ═════════════════════════════════════════════════════════════════════
    
    def load_dataframes(self, file_paths: List[str] = None):
        """Load DataFrames from Excel/CSV files."""
        if not file_paths:
            file_paths, _ = QFileDialog.getOpenFileNames(
                None,
                "Open DataFrame Files",
                "",
                "Excel/CSV Files (*.xlsx *.xls *.csv)"
            )
        
        if not file_paths:
            return
        
        skipped = []
        
        for file_path in file_paths:
            try:
                path = Path(file_path)
                
                # Load DataFrame(s) using helper function
                # Returns dict: {name: DataFrame}
                dfs_dict = load_dataframe_file(path)
                
                # Add each DataFrame to workspace
                for df_name, df in dfs_dict.items():
                    # Check if already loaded
                    if df_name in self.dataframes:
                        skipped.append(df_name)
                        continue
                    
                    self.dataframes[df_name] = df
                
            except Exception as e:
                self.notify.emit(f"Error loading {Path(file_path).name}: {e}")
        
        if skipped:
            self.notify.emit(f"Already loaded and skipped:\n" + "\n".join(skipped))
        
        self._emit_dataframes_list()
    
    def add_dataframe(self, df_name: str, df: pd.DataFrame):
        """Add a DataFrame programmatically (e.g., from fit results)."""
        if df_name in self.dataframes:
            self.notify.emit(f"DataFrame '{df_name}' already exists.")
            return
        
        self.dataframes[df_name] = df
        self._emit_dataframes_list()
        self.notify.emit(f"Added DataFrame: {df_name}")
    
    def remove_dataframe(self, df_name: str):
        """Remove a DataFrame."""
        if df_name in self.dataframes:
            del self.dataframes[df_name]
            self._emit_dataframes_list()
            
            # Clear selection if this was selected
            if self.selected_df_name == df_name:
                self.selected_df_name = None
                self.dataframe_columns_changed.emit([])
    
    def select_dataframe(self, df_name: str):
        """Select a DataFrame and emit its columns."""
        if df_name not in self.dataframes:
            return
        
        self.selected_df_name = df_name
        self.dataframe_selected.emit(df_name)
        
        # Emit column names
        df = self.dataframes[df_name]
        self.dataframe_columns_changed.emit(list(df.columns))
    
    def get_dataframe(self, df_name: str) -> Optional[pd.DataFrame]:
        """Get a DataFrame by name."""
        return self.dataframes.get(df_name)
    
    def save_dataframe_to_excel(self, df_name: str):
        """Save a DataFrame to Excel file."""
        if df_name not in self.dataframes:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save DataFrame",
            f"{df_name}.xlsx",
            "Excel Files (*.xlsx)"
        )
        
        if file_path:
            try:
                self.dataframes[df_name].to_excel(file_path, index=False)
                self.notify.emit(f"DataFrame saved: {Path(file_path).name}")
            except Exception as e:
                self.notify.emit(f"Error saving DataFrame: {e}")
    
    # ═════════════════════════════════════════════════════════════════════
    # Filter Management
    # ═════════════════════════════════════════════════════════════════════
    
    def apply_filters(self, df_name: str, filters: List[Dict]) -> Optional[pd.DataFrame]:
        """Apply filters to a DataFrame."""
        if df_name not in self.dataframes:
            return None
        
        df = self.dataframes[df_name].copy()
        
        # Apply checked filters
        for filter_data in filters:
            if filter_data.get("state", False):
                expression = filter_data["expression"]
                try:
                    df = df.query(expression)
                except Exception as e:
                    self.notify.emit(f"Filter error: {e}")
                    return None
        
        self.filtered_data_ready.emit(df)
        return df
    
    def has_slot_column(self, df_name: str) -> bool:
        """Check if DataFrame has a 'Slot' column."""
        if df_name not in self.dataframes:
            return False
        return 'Slot' in self.dataframes[df_name].columns
    
    def get_unique_slots(self, df_name: str) -> List:
        """Get unique slot values from DataFrame."""
        if not self.has_slot_column(df_name):
            return []
        
        df = self.dataframes[df_name]
        return sorted(df['Slot'].dropna().unique())
    
    def create_multi_wafer_graphs(self, df_name: str, slot_numbers: List, 
                                   plot_config: Dict, base_filters: List[Dict]) -> List[MGraph]:
        """Create multiple wafer graphs, one for each slot.
        
        Args:
            df_name: Name of the DataFrame
            slot_numbers: List of slot numbers to plot
            plot_config: Base plot configuration
            base_filters: Base filters to apply (will be merged with slot filters)
        
        Returns:
            List of created MGraph objects
        """
        created_graphs = []
        
        for slot_num in slot_numbers:
            # Merge base filters with slot-specific filter
            slot_filters = self._merge_filters_with_slot(base_filters, slot_num)
            
            # Create new graph
            graph = MGraph(graph_id=self._next_graph_id)
            
            # Apply base configuration
            for key, value in plot_config.items():
                if hasattr(graph, key):
                    setattr(graph, key, value)
            
            # Set slot-specific filters
            graph.filters = slot_filters
            
            # Store graph
            self.graphs[self._next_graph_id] = graph
            self._next_graph_id += 1
            
            created_graphs.append(graph)
        
        self._emit_graphs_list()
        return created_graphs
    
    def _merge_filters_with_slot(self, base_filters: List[Dict], slot_num: int) -> List[Dict]:
        """Merge base filters with slot-specific filter.
        
        Ensures that only the specified slot filter is active.
        """
        import copy
        # Deep copy to avoid modifying original filter dictionaries
        merged = copy.deepcopy(base_filters)
        slot_expr = f"Slot == {slot_num}"
        
        # Check if a Slot filter already exists
        slot_found = False
        for f in merged:
            expr = f.get("expression", "")
            if "Slot ==" in expr:
                # Replace with current slot and activate
                f["expression"] = slot_expr
                f["state"] = True
                slot_found = True
                break
        
        # If no Slot filter exists, add one
        if not slot_found:
            merged.append({
                "expression": slot_expr,
                "state": True
            })
        
        return merged
    
    # ═════════════════════════════════════════════════════════════════════
    # Graph Management
    # ═════════════════════════════════════════════════════════════════════
    
    def create_graph(self, plot_config: Dict = None) -> MGraph:
        """Create a new graph with properties from config."""
        graph = MGraph(graph_id=self._next_graph_id)
        
        # Apply configuration if provided
        if plot_config:
            for key, value in plot_config.items():
                if hasattr(graph, key):
                    setattr(graph, key, value)
        
        self.graphs[self._next_graph_id] = graph
        self._next_graph_id += 1
        
        self._emit_graphs_list()
        return graph
    
    def get_graph_ids(self) -> List[int]:
        """Get list of all graph IDs."""
        return list(self.graphs.keys())
    
    def get_graph(self, graph_id: int) -> Optional[MGraph]:
        """Get a graph by ID."""
        return self.graphs.get(graph_id)
    
    def select_graph(self, graph_id: int):
        """Select a graph and emit its properties."""
        if graph_id not in self.graphs:
            return
        
        self.selected_graph_id = graph_id
        self.graph_selected.emit(graph_id)
        self.graph_properties_changed.emit(self.graphs[graph_id])
    
    def update_graph(self, graph_id: int, properties: Dict):
        """Update graph properties."""
        if graph_id not in self.graphs:
            return
        
        graph = self.graphs[graph_id]
        for key, value in properties.items():
            if hasattr(graph, key):
                setattr(graph, key, value)
        
        self.graph_properties_changed.emit(graph)
    
    def delete_graph(self, graph_id: int):
        """Delete a graph."""
        if graph_id in self.graphs:
            del self.graphs[graph_id]
            self._emit_graphs_list()
            
            if self.selected_graph_id == graph_id:
                self.selected_graph_id = None
    
    # ═════════════════════════════════════════════════════════════════════
    # Save/Load Workspace
    # ═════════════════════════════════════════════════════════════════════
    
    def save_workspace(self):
        """Save graphs workspace to .graphs file."""
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Graphs Workspace",
            "",
            "Graphs Files (*.graphs)"
        )
        
        if not file_path:
            return
        
        try:
            # Convert Graph objects to serializable format (use 'plots' key for compatibility)
            plots_data = {}
            for graph_id, graph in self.graphs.items():
                graph_data = graph.save()
                plots_data[graph_id] = graph_data
            
            # Compress DataFrames using gzip (same as legacy code)
            compressed_dfs = {}
            for k, v in self.dataframes.items():
                # Convert DataFrame to a CSV string and compress it
                compressed_df = v.to_csv(index=False).encode('utf-8')
                compressed_dfs[k] = gzip.compress(compressed_df)
            
            # Prepare data to save (use 'plots' and 'original_dfs' keys for compatibility)
            data_to_save = {
                'plots': plots_data,
                'original_dfs': {k: v.hex() for k, v in compressed_dfs.items()},
            }
            
            # Save to JSON file
            with open(file_path, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            
            self.notify.emit(f"Workspace saved: {Path(file_path).name}")
        except Exception as e:
            self.notify.emit(f"Error saving workspace: {e}")
    
    def load_workspace(self, file_path: str):
        """Load graphs workspace from .graphs file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Clear current data
            self.graphs.clear()
            self.dataframes.clear()
            
            # Load DataFrames from compressed format (use 'original_dfs' key for compatibility)
            for k, v in data.get('original_dfs', {}).items():
                compressed_data = bytes.fromhex(v)
                csv_data = gzip.decompress(compressed_data).decode('utf-8')
                self.dataframes[k] = pd.read_csv(StringIO(csv_data))
            
            # Load graphs (use 'plots' key for compatibility)
            plots_data = data.get('plots', {})
            for graph_id_str, graph_data in plots_data.items():
                graph_id = int(graph_id_str)
                graph = MGraph(graph_id=graph_id)
                graph.load(graph_data)
                self.graphs[graph_id] = graph
            
            # Update next graph ID
            if self.graphs:
                self._next_graph_id = max(self.graphs.keys()) + 1
            else:
                self._next_graph_id = 1
            
            # Emit updates
            self._emit_dataframes_list()
            self._emit_graphs_list()
            
            self.notify.emit(f"Loaded {len(self.graphs)} graphs, {len(self.dataframes)} DataFrames")
        except Exception as e:
            self.notify.emit(f"Error loading workspace: {e}")
    
    def clear_workspace(self):
        """Clear all graphs and DataFrames."""
        self.graphs.clear()
        self.dataframes.clear()
        self.selected_df_name = None
        self.selected_graph_id = None
        self._next_graph_id = 1
        
        self._emit_dataframes_list()
        self._emit_graphs_list()
        self.dataframe_columns_changed.emit([])
    
    # ═════════════════════════════════════════════════════════════════════
    # Internal Helpers
    # ═════════════════════════════════════════════════════════════════════
    
    def _emit_dataframes_list(self):
        """Emit list of DataFrame names."""
        self.dataframes_changed.emit(list(self.dataframes.keys()))
    
    def _emit_graphs_list(self):
        """Emit list of graph IDs."""
        self.graphs_changed.emit(list(self.graphs.keys()))
