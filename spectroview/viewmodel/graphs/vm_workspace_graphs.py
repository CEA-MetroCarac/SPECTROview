"""ViewModel for Graphs Workspace - handles business logic and data management."""

from __future__ import annotations

import copy
import gzip
from io import StringIO
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog

from spectroview.model.graphs.graph_control import (
    GRAPH_PROPERTY_SET,
    normalize_graph_patch,
)
from spectroview.model.graphs.m_graph import MGraph
from spectroview.model.io.m_io import load_dataframe_file
from spectroview.model.io.workspace_io import WorkspaceIO
from spectroview.model.m_settings import MSettings
from spectroview.viewmodel.graphs.graph_undo import GraphUndoMixin


class VMWorkspaceGraphs(QObject, GraphUndoMixin):
    """ViewModel for managing graphs, DataFrames, and plotting logic."""

    # ───── ViewModel → View signals ─────
    dataframes_changed = Signal(list)
    dataframe_columns_changed = Signal(list)
    graphs_changed = Signal(list)
    # Emitted for *property* changes as well as graph-list changes so AI context stays synced.
    graph_state_changed = Signal()
    notify = Signal(str)
    undo_state_changed = Signal()

    def __init__(self, settings: MSettings):
        super().__init__()
        self.settings = settings

        # Data storage
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.dataframe_sources: Dict[str, str] = {}
        self.graphs: Dict[int, MGraph] = {}

        # Current selection
        self.selected_df_name: Optional[str] = None

        # Graph ID counter
        self._next_graph_id = 1

        # Undo/redo snapshot stack
        self._init_undo_state(max_undo_depth=50)

    def _populate_graphs_from_dict(
        self, plots_dict: dict, *, legacy_x_as_numeric_false: Optional[bool] = None
    ) -> None:
        """Populate ``self.graphs`` and ``self._next_graph_id`` from a serialized plots mapping."""
        self.graphs.clear()
        for graph_id_key, graph_data in plots_dict.items():
            graph_id = int(graph_id_key)
            graph = MGraph(graph_id=graph_id)
            if legacy_x_as_numeric_false is None:
                graph.load(graph_data)
            else:
                graph.load(
                    graph_data, legacy_x_as_numeric_false=legacy_x_as_numeric_false
                )
            self.graphs[graph_id] = graph
        self._next_graph_id = max(self.graphs.keys()) + 1 if self.graphs else 1

    @staticmethod
    def _filter_creation_config(plot_config: Dict) -> Dict:
        """Return a deep-copied creation patch containing only valid mutable MGraph fields."""
        patch = copy.deepcopy(plot_config)
        patch.pop("graph_id", None)
        return {key: value for key, value in patch.items() if key in GRAPH_PROPERTY_SET}

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
                "Excel/CSV Files (*.xlsx *.xls *.csv)",
            )

        if not file_paths:
            return

        skipped = []
        last_valid_path = None

        for file_path in file_paths:
            try:
                path = Path(file_path)
                dfs_dict = load_dataframe_file(path)

                for df_name, df in dfs_dict.items():
                    if df_name in self.dataframes:
                        skipped.append(df_name)
                        continue

                    self.dataframes[df_name] = df
                    self.dataframe_sources[df_name] = str(path)
                    last_valid_path = path

            except Exception as e:
                self.notify.emit(f"Error loading {Path(file_path).name}: {e}")

        if skipped:
            self.notify.emit("Already loaded and skipped:\n" + "\n".join(skipped))

        self._emit_dataframes_list()

        if last_valid_path:
            self.settings.set_last_directory(str(last_valid_path.parent))

    def add_dataframe(
        self, df_name: str, df: pd.DataFrame, force_replace: bool = False
    ):
        """Add a DataFrame to workspace."""
        is_replace = df_name in self.dataframes
        if is_replace and not force_replace:
            self.notify.emit(f"DataFrame '{df_name}' already exists.")
            return

        self.dataframes[df_name] = df
        self._emit_dataframes_list()

        if is_replace and force_replace:
            self.notify.emit(f"DataFrame '{df_name}' successfully replaced.")
        else:
            self.notify.emit(
                f"DataFrame '{df_name}' sent successfully to Graphs workspace."
            )

    def remove_dataframe(self, df_name: str):
        """Remove a DataFrame."""
        if df_name in self.dataframes:
            del self.dataframes[df_name]
            if df_name in self.dataframe_sources:
                del self.dataframe_sources[df_name]

            self._emit_dataframes_list()

            if self.selected_df_name == df_name:
                self.selected_df_name = None
                self.dataframe_columns_changed.emit([])

    def select_dataframe(self, df_name: str):
        """Select a DataFrame."""
        if df_name not in self.dataframes:
            return

        self.selected_df_name = df_name

        df = self.dataframes[df_name]
        self.dataframe_columns_changed.emit(list(df.columns))

    def get_dataframe(self, df_name: str) -> Optional[pd.DataFrame]:
        """Get a DataFrame by name."""
        return self.dataframes.get(df_name)

    def refresh_dataframe(self, df_name: str) -> bool:
        """Refresh DataFrame from source file."""
        if df_name not in self.dataframes:
            return False

        if df_name not in self.dataframe_sources:
            return False

        source_path = Path(self.dataframe_sources[df_name])

        if not source_path.exists():
            return False

        try:
            dfs_dict = load_dataframe_file(source_path)

            if df_name in dfs_dict:
                self.dataframes[df_name] = dfs_dict[df_name]

                if self.selected_df_name == df_name:
                    self.dataframe_columns_changed.emit(
                        list(self.dataframes[df_name].columns)
                    )

                return True
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
            "Excel Files (*.xlsx);;CSV Files (*.csv)",
        )

        if file_path:
            try:
                ext = Path(file_path).suffix.lower()

                if ext == ".csv" or "CSV" in selected_filter:
                    self.dataframes[df_name].to_csv(file_path, index=False, sep=";")
                else:
                    if not ext:
                        file_path += ".xlsx"
                    self.dataframes[df_name].to_excel(file_path, index=False)

                self.notify.emit(f"DataFrame saved: {Path(file_path).name}")
                self.settings.set_last_directory(str(Path(file_path).parent))
            except Exception as e:
                self.notify.emit(f"Error saving DataFrame: {e}")

    # ═════════════════════════════════════════════════════════════════════
    # Filter Management
    # ═════════════════════════════════════════════════════════════════════

    def apply_filters(
        self, df_name: str, filters: List[Dict]
    ) -> Optional[pd.DataFrame]:
        """Apply filters to a DataFrame, returning an independent frame."""
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
                self.notify.emit(f"Filter error: {e}")
                return None

        return df

    def has_slot_column(self, df_name: str) -> bool:
        """Check if DataFrame has 'Slot' column."""
        if df_name not in self.dataframes:
            return False
        return "Slot" in self.dataframes[df_name].columns

    def get_unique_slots(self, df_name: str) -> List:
        """Get unique slot values."""
        if not self.has_slot_column(df_name):
            return []

        df = self.dataframes[df_name]
        return sorted(df["Slot"].dropna().unique())

    def create_multi_wafer_graphs(
        self,
        df_name: str,
        slot_numbers: List,
        plot_config: Dict,
        base_filters: List[Dict],
    ) -> List[MGraph]:
        """Create multiple validated wafer graphs for each slot."""
        pending = []
        dataframe = self.dataframes.get(df_name)
        for offset, slot_num in enumerate(slot_numbers):
            slot_filters = self._merge_filters_with_slot(base_filters, slot_num)
            graph = MGraph(graph_id=self._next_graph_id + offset)
            patch = self._filter_creation_config(plot_config)
            patch["df_name"] = df_name
            patch["filters"] = slot_filters
            normalized = normalize_graph_patch(
                patch, current=graph, dataframe=dataframe
            )
            for name, value in normalized.items():
                setattr(graph, name, value)
            pending.append(graph)

        self._record_undo_point_if_needed()
        created_graphs = []
        for graph in pending:
            self.graphs[self._next_graph_id] = graph
            self._next_graph_id += 1
            created_graphs.append(graph)

        self._emit_graphs_list()
        return created_graphs

    def _merge_filters_with_slot(
        self, base_filters: List[Dict], slot_num: int
    ) -> List[Dict]:
        """Merge base filters with slot filter."""
        merged = copy.deepcopy(base_filters)
        slot_expr = f"Slot == {slot_num}"

        slot_found = False
        for f in merged:
            expr = f.get("expression", "")
            if "Slot ==" in expr:
                f["expression"] = slot_expr
                f["state"] = True
                slot_found = True
                break

        if not slot_found:
            merged.append({"expression": slot_expr, "state": True})

        return merged

    # ═════════════════════════════════════════════════════════════════════
    # Graph Management
    # ═════════════════════════════════════════════════════════════════════

    def create_graph(self, plot_config: Dict = None) -> MGraph:
        """Create a graph through the shared validated graph-command layer."""
        graph = MGraph(graph_id=self._next_graph_id)

        if plot_config:
            patch = self._filter_creation_config(plot_config)
            target_df = self.dataframes.get(patch.get("df_name"))
            normalized = normalize_graph_patch(
                patch, current=graph, dataframe=target_df
            )
            for name, value in normalized.items():
                setattr(graph, name, value)

        self._record_undo_point_if_needed()
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

    def update_graph(self, graph_id: int, properties: Dict) -> Dict:
        """Validate and atomically update graph properties."""
        if graph_id not in self.graphs:
            return {}

        if not properties:
            return {}

        graph = self.graphs[graph_id]
        prospective_df_name = properties.get("df_name", graph.df_name)
        dataframe = self.dataframes.get(prospective_df_name)
        patch = normalize_graph_patch(properties, current=graph, dataframe=dataframe)
        self._record_undo_point_if_needed()
        for name, value in patch.items():
            setattr(graph, name, copy.deepcopy(value))
        self.graph_state_changed.emit()
        return patch

    def restore_graph_state(self, graph_id: int, state: Dict) -> bool:
        """Restore a graph after a render failure without creating undo data."""
        graph = self.graphs.get(graph_id)
        if graph is None:
            return False
        graph.load(copy.deepcopy(state))
        self.graph_state_changed.emit()
        return True

    def sync_derived_graph_properties(self, graph_id: int, properties: Dict) -> None:
        """Persist renderer-derived bookkeeping without an undo snapshot."""
        graph = self.graphs.get(graph_id)
        if graph is None:
            return
        for key, value in properties.items():
            if hasattr(graph, key):
                setattr(graph, key, copy.deepcopy(value))
        self.graph_state_changed.emit()

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
            "Graphs Files (*.graphs)",
        )

        if not file_path:
            return

        try:
            plots_data = {
                graph_id: graph.save() for graph_id, graph in self.graphs.items()
            }

            metadata = {
                "format_version": 4,
                "plots": plots_data,
                "dataframe_sources": self.dataframe_sources,
            }

            WorkspaceIO.save_workspace(
                file_path, metadata, dataframes=self.dataframes
            )

            self.notify.emit(f"Workspace saved: {Path(file_path).name}")
        except Exception as e:
            self.notify.emit(f"Error saving workspace: {e}")

    def load_workspace(self, file_path: str):
        """Load workspace supporting both legacy JSON and new ZIP formats."""
        try:
            metadata, _, dataframes, is_legacy = WorkspaceIO.load_workspace(file_path)

            if is_legacy:
                self.load_workspace_legacy(file_path)
                return

            self._reset_undo_history()
            self.graphs.clear()
            self.dataframes.clear()
            self.dataframe_sources.clear()

            if dataframes:
                self.dataframes = dataframes

            self.dataframe_sources = metadata.get("dataframe_sources", {})

            legacy_axis_type = int(metadata.get("format_version", 0) or 0) < 4
            self._populate_graphs_from_dict(
                metadata.get("plots", {}), legacy_x_as_numeric_false=legacy_axis_type
            )

            self._emit_dataframes_list()
            self._emit_graphs_list()
        except Exception as e:
            self.notify.emit(f"Error loading workspace: {e}")

    def load_workspace_legacy(self, file_path: str):
        """Legacy JSON loader for backward compatibility with old .graphs files."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            self._reset_undo_history()
            self.graphs.clear()
            self.dataframes.clear()
            self.dataframe_sources.clear()

            for k, v in data.get("original_dfs", {}).items():
                compressed_data = bytes.fromhex(v)
                csv_data = gzip.decompress(compressed_data).decode("utf-8")
                self.dataframes[k] = pd.read_csv(StringIO(csv_data))

            if "dataframe_sources" in data:
                self.dataframe_sources = data["dataframe_sources"]

            self._populate_graphs_from_dict(
                data.get("plots", {}), legacy_x_as_numeric_false=True
            )

            self._emit_dataframes_list()
            self._emit_graphs_list()

        except Exception as e:
            self.notify.emit(f"Error loading legacy workspace: {e}")

    def clear_workspace(self):
        """Clear workspace."""
        self._reset_undo_history()
        self.graphs.clear()
        self.dataframes.clear()
        self.dataframe_sources.clear()
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
        self.graph_state_changed.emit()
