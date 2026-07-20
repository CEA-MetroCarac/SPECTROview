"""ViewModel for Graphs Workspace - handles business logic and data management."""
import json
import pandas as pd
import gzip
import copy
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox

from spectroview.model.m_graph import MGraph
from spectroview.model.m_settings import MSettings
from spectroview.model.m_io import load_dataframe_file
from spectroview.model.workspace_io import WorkspaceIO


class VMWorkspaceGraphs(QObject):
    """ViewModel for managing graphs, DataFrames, and plotting logic."""
    
    # ───── ViewModel → View signals ─────
    dataframes_changed = Signal(list)
    dataframe_columns_changed = Signal(list)
    graphs_changed = Signal(list)
    notify = Signal(str)
    undo_state_changed = Signal()

    def __init__(self, settings: MSettings):
        super().__init__()
        self.settings = settings

        # Data storage
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.dataframe_sources: Dict[str, str] = {}  # Track source file paths for refresh
        self.graphs: Dict[int, MGraph] = {}

        # Current selection
        self.selected_df_name: Optional[str] = None

        # Graph ID counter
        self._next_graph_id = 1

        # ───── Undo/redo: a stack of whole-workspace graph snapshots ─────
        self._undo_stack: List[Dict[int, dict]] = []
        self._redo_stack: List[Dict[int, dict]] = []
        self._max_undo_depth = 50
        # begin/end_undo_batch() collapse several graph calls from one user
        # action into a single undo step; depth-counted so nested batches compose.
        self._undo_batch_depth = 0
        self._undo_batch_pending = False

    # ------------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------------
    def begin_undo_batch(self) -> None:
        """Mark the start of a sequence of mutations that represent ONE
        logical user action. Call `end_undo_batch()` (via try/finally) when
        done. Only the first actual mutation inside the outermost
        begin/end pair records an undo snapshot; further mutations before
        the matching end_undo_batch() collapse into that same step."""
        self._undo_batch_depth += 1
        if self._undo_batch_depth == 1:
            self._undo_batch_pending = True

    def end_undo_batch(self) -> None:
        self._undo_batch_depth = max(0, self._undo_batch_depth - 1)
        if self._undo_batch_depth == 0:
            self._undo_batch_pending = False

    def _record_undo_point_if_needed(self) -> None:
        """Called at the start of every graph-set mutation
        (create_graph/update_graph/delete_graph/create_multi_wafer_graphs).
        Outside any begin/end_undo_batch() pair, every call is its own undo
        step (correct for the common case: one Apply = one update_graph()
        call). Inside a batch, only the first call captures a snapshot."""
        if self._undo_batch_depth == 0:
            self._push_undo_snapshot()
        elif self._undo_batch_pending:
            self._push_undo_snapshot()
            self._undo_batch_pending = False

    def _capture_snapshot(self) -> Dict[int, dict]:
        """Deep-copied snapshot of every graph's current saved state
        (MGraph.save() already deep-copies its mutable fields)."""
        return {graph_id: graph.save() for graph_id, graph in self.graphs.items()}

    def _push_undo_snapshot(self) -> None:
        self._undo_stack.append(self._capture_snapshot())
        if len(self._undo_stack) > self._max_undo_depth:
            self._undo_stack.pop(0)
        self._redo_stack.clear()  # a new action invalidates prior redo history
        self.undo_state_changed.emit()

    def _restore_snapshot(self, snapshot: Dict[int, dict]) -> None:
        """Replace self.graphs wholesale with `snapshot`'s contents (mirrors
        load_workspace()'s graph-restoration loop). The View is responsible
        for reconciling its own VGraph widgets/MDI subwindows against the
        new self.graphs afterward -- this only updates ViewModel state."""
        self.graphs.clear()
        for graph_id, graph_data in snapshot.items():
            graph = MGraph(graph_id=graph_id)
            graph.load(graph_data)
            self.graphs[graph_id] = graph
        self._next_graph_id = max(self.graphs.keys()) + 1 if self.graphs else 1
        self._emit_graphs_list()

    def _reset_undo_history(self) -> None:
        """Clear undo/redo history -- called whenever the workspace's graph
        set is replaced wholesale (load/clear), since "undo" should never
        reach back into a *previous*, now-discarded workspace's state."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._undo_batch_depth = 0
        self._undo_batch_pending = False
        self.undo_state_changed.emit()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def undo(self) -> bool:
        """Revert to the workspace state before the last undo-tracked
        action. Returns False (no-op) if there's nothing to undo."""
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._capture_snapshot())
        snapshot = self._undo_stack.pop()
        self._restore_snapshot(snapshot)
        self.undo_state_changed.emit()
        return True

    def redo(self) -> bool:
        """Re-apply the last undone action. Returns False (no-op) if
        there's nothing to redo."""
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._capture_snapshot())
        snapshot = self._redo_stack.pop()
        self._restore_snapshot(snapshot)
        self.undo_state_changed.emit()
        return True

    # ------------------------------------------------------------------------
    # Dataframe management
    # ------------------------------------------------------------------------
    def load_dataframes(self, file_paths: List[str] = None):
        """Load DataFrames from files."""
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
        last_valid_path = None
        
        for file_path in file_paths:
            try:
                path = Path(file_path)
                dfs_dict = load_dataframe_file(path)
                
                # Add each DataFrame to workspace
                for df_name, df in dfs_dict.items():
                    # Check if already loaded
                    if df_name in self.dataframes:
                        skipped.append(df_name)
                        continue
                    
                    self.dataframes[df_name] = df
                    # Store source file path for refresh functionality
                    self.dataframe_sources[df_name] = str(path)
                    last_valid_path = path  # Track last successfully loaded file
                
            except Exception as e:
                self.notify.emit(f"Error loading {Path(file_path).name}: {e}")
        
        if skipped:
            self.notify.emit("Already loaded and skipped:\n" + "\n".join(skipped))
        
        self._emit_dataframes_list()
        
        # Update last_directory setting
        if last_valid_path:
            self.settings.set_last_directory(str(last_valid_path.parent))
    
    def add_dataframe(self, df_name: str, df: pd.DataFrame, force_replace: bool = False):
        """Add a DataFrame to workspace."""
        is_replace = df_name in self.dataframes
        if is_replace and not force_replace:
            self.notify.emit(f"DataFrame '{df_name}' already exists.")
            return
        
        self.dataframes[df_name] = df
        self._emit_dataframes_list()
        
        if is_replace and force_replace:
            self.notify.emit(f"DataFrame '{df_name}' successfully replaced.")
    
    def remove_dataframe(self, df_name: str):
        """Remove a DataFrame."""
        if df_name in self.dataframes:
            del self.dataframes[df_name]
            # Also remove source file reference
            if df_name in self.dataframe_sources:
                del self.dataframe_sources[df_name]
            
            self._emit_dataframes_list()
            
            # Clear selection if this was selected
            if self.selected_df_name == df_name:
                self.selected_df_name = None
                self.dataframe_columns_changed.emit([])
    
    def select_dataframe(self, df_name: str):
        """Select a DataFrame."""
        if df_name not in self.dataframes:
            return
        
        self.selected_df_name = df_name
        
        # Emit column names
        df = self.dataframes[df_name]
        self.dataframe_columns_changed.emit(list(df.columns))
    
    def get_dataframe(self, df_name: str) -> Optional[pd.DataFrame]:
        """Get a DataFrame by name."""
        return self.dataframes.get(df_name)
    
    def refresh_dataframe(self, df_name: str) -> bool:
        """Refresh DataFrame from source file"""
        # Check if DataFrame exists and has source file
        if df_name not in self.dataframes:
            return False
        
        if df_name not in self.dataframe_sources:
            return False
        
        source_path = Path(self.dataframe_sources[df_name])
        
        # Check if source file still exists
        if not source_path.exists():
            return False
        
        try:
            # Reload DataFrame from source file
            dfs_dict = load_dataframe_file(source_path)
            
            # Find the matching DataFrame (should have same name)
            if df_name in dfs_dict:
                # Update the DataFrame in place
                self.dataframes[df_name] = dfs_dict[df_name]
                
                # If this is the selected DataFrame, re-emit columns
                if self.selected_df_name == df_name:
                    self.dataframe_columns_changed.emit(list(self.dataframes[df_name].columns))
                
                return True
            else:
                return False
                
        except Exception as e:
            self.notify.emit(f"Error refreshing DataFrame: {e}")
            return False
    
    def save_dataframe(self, df_name: str):
        """Save DataFrame to Excel or CSV file."""
        if df_name not in self.dataframes:
            return
        
        last_dir = self.settings.get_last_directory()
        file_path, selected_filter = QFileDialog.getSaveFileName(
            None,
            "Save DataFrame",
            str(Path(last_dir) / f"{df_name}.xlsx"),
            "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                # Determine format from file extension
                ext = Path(file_path).suffix.lower()
                
                if ext == '.csv' or 'CSV' in selected_filter:
                    # Save as CSV with semicolon delimiter
                    self.dataframes[df_name].to_csv(file_path, index=False, sep=';')
                else:
                    # Save as Excel (default)
                    if not ext:
                        file_path += '.xlsx'
                    self.dataframes[df_name].to_excel(file_path, index=False)
                
                self.notify.emit(f"DataFrame saved: {Path(file_path).name}")
                # Update last_directory setting
                self.settings.set_last_directory(str(Path(file_path).parent))
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Error saving DataFrame: {e}")
    
    # ═════════════════════════════════════════════════════════════════════
    # Filter Management
    # ═════════════════════════════════════════════════════════════════════
    
    def apply_filters(self, df_name: str, filters: List[Dict]) -> Optional[pd.DataFrame]:
        """Apply filters to a DataFrame.

        Always returns a frame independent of internal storage (never a live
        reference callers could use to mutate the stored DataFrame). When at
        least one filter is checked, `.query()` already produces a fresh
        frame, so no extra defensive `.copy()` is taken beforehand — avoids
        copying the full (potentially large) DataFrame twice.
        """
        if df_name not in self.dataframes:
            return None

        df = self.dataframes[df_name]
        active_filters = [f for f in filters if f.get("state", False)]

        if not active_filters:
            return df.copy()

        for filter_data in active_filters:
            expression = filter_data["expression"]
            try:
                df = df.query(expression)
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Filter error: {e}")
                return None

        return df
    
    def has_slot_column(self, df_name: str) -> bool:
        """Check if DataFrame has 'Slot' column."""
        if df_name not in self.dataframes:
            return False
        return 'Slot' in self.dataframes[df_name].columns
    
    def get_unique_slots(self, df_name: str) -> List:
        """Get unique slot values."""
        if not self.has_slot_column(df_name):
            return []
        
        df = self.dataframes[df_name]
        return sorted(df['Slot'].dropna().unique())
    
    def create_multi_wafer_graphs(self, df_name: str, slot_numbers: List,
                                   plot_config: Dict, base_filters: List[Dict]) -> List[MGraph]:
        """Create multiple wafer graphs for each slot."""
        self._record_undo_point_if_needed()
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
        """Merge base filters with slot filter."""
        
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
        """Create a new graph."""
        self._record_undo_point_if_needed()
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
        """Get all graph IDs."""
        return list(self.graphs.keys())
    
    def get_graph(self, graph_id: int) -> Optional[MGraph]:
        """Get a graph by ID."""
        return self.graphs.get(graph_id)
    
    def update_graph(self, graph_id: int, properties: Dict):
        """Update graph properties."""
        if graph_id not in self.graphs:
            return

        self._record_undo_point_if_needed()
        graph = self.graphs[graph_id]
        for key, value in properties.items():
            if hasattr(graph, key):
                setattr(graph, key, value)
    
    def delete_graph(self, graph_id: int):
        """Delete a graph."""
        if graph_id in self.graphs:
            self._record_undo_point_if_needed()
            del self.graphs[graph_id]
            self._emit_graphs_list()
    
    # ═════════════════════════════════════════════════════════════════════
    # Save/Load Workspace
    # ═════════════════════════════════════════════════════════════════════
    
    def save_workspace(self):
        """Save workspace to file using high-performance ZIP + Pickle architecture."""
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Graphs Workspace",
            "",
            "Graphs Files (*.graphs)"
        )
        
        if not file_path:
            return
        
        try:
            # Serialize graphs
            plots_data = {graph_id: graph.save() for graph_id, graph in self.graphs.items()}
            
            # Prepare metadata (light config)
            metadata = {
                'format_version': 3,  # signals ZIP binary format
                'plots': plots_data,
                'dataframe_sources': self.dataframe_sources,
            }
            
            # Save using WorkspaceIO
            WorkspaceIO.save_workspace(file_path, metadata, dataframes=self.dataframes)
            
            self.notify.emit(f"Workspace saved: {Path(file_path).name}")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error saving workspace: {e}")
    
    def load_workspace(self, file_path: str):
        """Load workspace supporting both legacy JSON and new ZIP formats."""
        try:
            # Attempt to load ZIP workspace
            metadata, _, dataframes, is_legacy = WorkspaceIO.load_workspace(file_path)
            
            if is_legacy:
                self.load_workspace_legacy(file_path)
                return

            self._reset_undo_history()
            self.graphs.clear()
            self.dataframes.clear()
            self.dataframe_sources.clear()

            # Restore DataFrames
            if dataframes:
                self.dataframes = dataframes
            
            # Load source file paths
            self.dataframe_sources = metadata.get('dataframe_sources', {})
            
            # Load graphs
            for graph_id_str, graph_data in metadata.get('plots', {}).items():
                graph_id = int(graph_id_str)
                graph = MGraph(graph_id=graph_id)
                graph.load(graph_data)
                self.graphs[graph_id] = graph
            
            # Update next graph ID
            self._next_graph_id = max(self.graphs.keys()) + 1 if self.graphs else 1
            
            # Emit updates
            self._emit_dataframes_list()
            self._emit_graphs_list()
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error loading workspace: {e}")
            
    def load_workspace_legacy(self, file_path: str):
        """Legacy JSON loader for backward compatibility with old .graphs files."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            self._reset_undo_history()
            self.graphs.clear()
            self.dataframes.clear()
            self.dataframe_sources.clear()

            # Load DataFrames
            for k, v in data.get('original_dfs', {}).items():
                compressed_data = bytes.fromhex(v)
                csv_data = gzip.decompress(compressed_data).decode('utf-8')
                self.dataframes[k] = pd.read_csv(StringIO(csv_data))
            
            # Load source file paths
            if 'dataframe_sources' in data:
                self.dataframe_sources = data['dataframe_sources']
            
            # Load graphs
            for graph_id_str, graph_data in data.get('plots', {}).items():
                graph_id = int(graph_id_str)
                graph = MGraph(graph_id=graph_id)
                graph.load(graph_data)
                self.graphs[graph_id] = graph
            
            # Update next graph ID
            self._next_graph_id = max(self.graphs.keys()) + 1 if self.graphs else 1
            
            # Emit updates
            self._emit_dataframes_list()
            self._emit_graphs_list()
            
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error loading legacy workspace: {e}")
    
    def clear_workspace(self):
        """Clear workspace."""
        self._reset_undo_history()
        self.graphs.clear()
        self.dataframes.clear()
        self.dataframe_sources.clear()  # Clear source file references
        self.selected_df_name = None
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
