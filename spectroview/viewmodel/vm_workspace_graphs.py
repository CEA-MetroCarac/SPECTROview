# viewmodel/vm_workspace_graphs.py
"""ViewModel for Graphs Workspace - handles business logic and data management."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog

from spectroview.model.m_graph import MGraph
from spectroview.model.m_settings import MSettings


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
        """Initialize the ViewModel.
        
        Args:
            settings: Application settings manager
        """
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
        """Load DataFrames from Excel/CSV files.
        
        Args:
            file_paths: List of file paths. If None, opens file dialog.
        """
        if not file_paths:
            file_paths, _ = QFileDialog.getOpenFileNames(
                None,
                "Open DataFrame Files",
                "",
                "Excel/CSV Files (*.xlsx *.xls *.csv)"
            )
        
        if not file_paths:
            return
        
        for file_path in file_paths:
            try:
                path = Path(file_path)
                
                # Read file based on extension
                if path.suffix.lower() in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path)
                elif path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path)
                else:
                    self.notify.emit(f"Unsupported file type: {path.suffix}")
                    continue
                
                # Use filename as DataFrame name
                df_name = path.stem
                
                # Check if already loaded
                if df_name in self.dataframes:
                    self.notify.emit(f"DataFrame '{df_name}' already loaded.")
                    continue
                
                self.dataframes[df_name] = df
                self.notify.emit(f"Loaded DataFrame: {df_name} ({len(df)} rows)")
                
            except Exception as e:
                self.notify.emit(f"Error loading {Path(file_path).name}: {e}")
        
        self._emit_dataframes_list()
    
    def add_dataframe(self, df_name: str, df: pd.DataFrame):
        """Add a DataFrame programmatically (e.g., from fit results).
        
        Args:
            df_name: Name for the DataFrame
            df: pandas DataFrame
        """
        if df_name in self.dataframes:
            self.notify.emit(f"DataFrame '{df_name}' already exists.")
            return
        
        self.dataframes[df_name] = df
        self._emit_dataframes_list()
        self.notify.emit(f"Added DataFrame: {df_name}")
    
    def remove_dataframe(self, df_name: str):
        """Remove a DataFrame.
        
        Args:
            df_name: Name of DataFrame to remove
        """
        if df_name in self.dataframes:
            del self.dataframes[df_name]
            self._emit_dataframes_list()
            
            # Clear selection if this was selected
            if self.selected_df_name == df_name:
                self.selected_df_name = None
                self.dataframe_columns_changed.emit([])
    
    def select_dataframe(self, df_name: str):
        """Select a DataFrame and emit its columns.
        
        Args:
            df_name: Name of DataFrame to select
        """
        if df_name not in self.dataframes:
            return
        
        self.selected_df_name = df_name
        self.dataframe_selected.emit(df_name)
        
        # Emit column names
        df = self.dataframes[df_name]
        self.dataframe_columns_changed.emit(list(df.columns))
    
    def get_dataframe(self, df_name: str) -> Optional[pd.DataFrame]:
        """Get a DataFrame by name.
        
        Args:
            df_name: Name of DataFrame
            
        Returns:
            DataFrame or None if not found
        """
        return self.dataframes.get(df_name)
    
    def save_dataframe_to_excel(self, df_name: str):
        """Save a DataFrame to Excel file.
        
        Args:
            df_name: Name of DataFrame to save
        """
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
        """Apply filters to a DataFrame.
        
        Args:
            df_name: Name of DataFrame to filter
            filters: List of filter dicts with 'expression' and 'state'
            
        Returns:
            Filtered DataFrame or None
        """
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
    
    # ═════════════════════════════════════════════════════════════════════
    # Graph Management
    # ═════════════════════════════════════════════════════════════════════
    
    def create_graph(self) -> MGraph:
        """Create a new graph with default properties.
        
        Returns:
            New MGraph instance
        """
        graph = MGraph(graph_id=self._next_graph_id)
        self.graphs[self._next_graph_id] = graph
        self._next_graph_id += 1
        
        self._emit_graphs_list()
        return graph
    
    def get_graph(self, graph_id: int) -> Optional[MGraph]:
        """Get a graph by ID.
        
        Args:
            graph_id: Graph ID
            
        Returns:
            MGraph or None
        """
        return self.graphs.get(graph_id)
    
    def select_graph(self, graph_id: int):
        """Select a graph and emit its properties.
        
        Args:
            graph_id: Graph ID to select
        """
        if graph_id not in self.graphs:
            return
        
        self.selected_graph_id = graph_id
        self.graph_selected.emit(graph_id)
        self.graph_properties_changed.emit(self.graphs[graph_id])
    
    def update_graph(self, graph_id: int, properties: Dict):
        """Update graph properties.
        
        Args:
            graph_id: Graph ID
            properties: Dictionary of properties to update
        """
        if graph_id not in self.graphs:
            return
        
        graph = self.graphs[graph_id]
        for key, value in properties.items():
            if hasattr(graph, key):
                setattr(graph, key, value)
        
        self.graph_properties_changed.emit(graph)
    
    def delete_graph(self, graph_id: int):
        """Delete a graph.
        
        Args:
            graph_id: Graph ID to delete
        """
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
            # Prepare data
            workspace_data = {
                'graphs': {gid: g.save() for gid, g in self.graphs.items()},
                'dataframes': {name: df.to_dict('list') for name, df in self.dataframes.items()},
                'next_graph_id': self._next_graph_id
            }
            
            with open(file_path, 'w') as f:
                json.dump(workspace_data, f, indent=4)
            
            self.notify.emit(f"Workspace saved: {Path(file_path).name}")
        except Exception as e:
            self.notify.emit(f"Error saving workspace: {e}")
    
    def load_workspace(self, file_path: str):
        """Load graphs workspace from .graphs file.
        
        Args:
            file_path: Path to .graphs file
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Clear current data
            self.graphs.clear()
            self.dataframes.clear()
            
            # Load DataFrames
            for name, df_data in data.get('dataframes', {}).items():
                self.dataframes[name] = pd.DataFrame(df_data)
            
            # Load graphs
            for gid_str, graph_data in data.get('graphs', {}).items():
                gid = int(gid_str)
                graph = MGraph(gid)
                graph.load(graph_data)
                self.graphs[gid] = graph
            
            self._next_graph_id = data.get('next_graph_id', len(self.graphs) + 1)
            
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
        
        self.notify.emit("Workspace cleared.")
    
    # ═════════════════════════════════════════════════════════════════════
    # Internal Helpers
    # ═════════════════════════════════════════════════════════════════════
    
    def _emit_dataframes_list(self):
        """Emit list of DataFrame names."""
        self.dataframes_changed.emit(list(self.dataframes.keys()))
    
    def _emit_graphs_list(self):
        """Emit list of graph IDs."""
        self.graphs_changed.emit(list(self.graphs.keys()))
