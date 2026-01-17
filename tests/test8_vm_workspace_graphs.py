"""
Tests for viewmodel/vm_workspace_graphs.py - Graphs workspace ViewModel

Tests cover:
- DataFrame loading from Excel/CSV
- DataFrame management (add/remove/select)
- Data filtering
- Graph creation and management
- Workspace persistence (save/load)
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

from PySide6.QtWidgets import QFileDialog

from spectroview.viewmodel.vm_workspace_graphs import VMWorkspaceGraphs
from spectroview.model.m_graph import MGraph


class TestVMWorkspaceGraphsInitialization:
    """Tests for VMWorkspaceGraphs initialization."""
    
    def test_initialization(self, qapp, mock_settings):
        """Test ViewModel initializes correctly."""
        vm = VMWorkspaceGraphs(mock_settings)
        
        assert vm.settings == mock_settings
        assert vm.dataframes == {}
        assert vm.graphs == {}
        assert vm.selected_df_name is None


class TestVMWorkspaceGraphsDataFrameLoading:
    """Tests for DataFrame loading."""
    
    def test_load_excel_file(self, qapp, mock_settings, dataframe_excel_file, monkeypatch):
        """Test loading Excel file."""
        if not dataframe_excel_file.exists():
            pytest.skip("Excel test file not available")
        
        vm = VMWorkspaceGraphs(mock_settings)
        
        # Mock file dialog
        def mock_get_open_filenames(*args, **kwargs):
            return [str(dataframe_excel_file)], ""
        monkeypatch.setattr(QFileDialog, "getOpenFileNames", mock_get_open_filenames)
        
        # Load DataFrame
        vm.load_dataframes()
        
        # Verify DataFrame was loaded
        assert len(vm.dataframes) > 0
    
    def test_load_csv_file(self, qapp, mock_settings, tmp_path, monkeypatch):
        """Test loading CSV file."""
        # Create test CSV
        csv_file = tmp_path / "test.csv"
        test_df = pd.DataFrame({
            'X': [1, 2, 3, 4, 5],
            'Y': [10, 20, 15, 25, 30],
            'Category': ['A', 'B', 'A', 'B', 'A']
        })
        test_df.to_csv(csv_file, index=False)
        
        vm = VMWorkspaceGraphs(mock_settings)
        
        # Mock file dialog
        def mock_get_open_filenames(*args, **kwargs):
            return [str(csv_file)], ""
        monkeypatch.setattr(QFileDialog, "getOpenFileNames", mock_get_open_filenames)
        
        # Load DataFrame
        vm.load_dataframes()
        
        # Verify DataFrame was loaded
        assert 'test' in vm.dataframes
        loaded_df = vm.dataframes['test']
        assert len(loaded_df) == 5
        assert list(loaded_df.columns) == ['X', 'Y', 'Category']


class TestVMWorkspaceGraphsDataFrameManagement:
    """Tests for DataFrame management operations."""
    
    @pytest.fixture
    def vm_with_dataframe(self, qapp, mock_settings, sample_dataframe):
        """Create ViewModel with sample DataFrame."""
        vm = VMWorkspaceGraphs(mock_settings)
        vm.add_dataframe("test_df", sample_dataframe)
        return vm
    
    def test_add_dataframe(self, qapp, mock_settings, sample_dataframe):
        """Test adding DataFrame programmatically."""
        vm = VMWorkspaceGraphs(mock_settings)
        
        # Add DataFrame
        vm.add_dataframe("my_data", sample_dataframe)
        
        # Verify DataFrame was added
        assert "my_data" in vm.dataframes
        pd.testing.assert_frame_equal(vm.dataframes["my_data"], sample_dataframe)
    
    def test_remove_dataframe(self, vm_with_dataframe):
        """Test removing DataFrame."""
        vm = vm_with_dataframe
        
        # Remove DataFrame
        vm.remove_dataframe("test_df")
        
        # Verify DataFrame was removed
        assert "test_df" not in vm.dataframes
    
    def test_select_dataframe(self, vm_with_dataframe):
        """Test selecting DataFrame."""
        vm = vm_with_dataframe
        
        # Connect signal to capture emission
        emitted_columns = []
        vm.dataframe_columns_changed.connect(lambda cols: emitted_columns.append(cols))
        
        # Select DataFrame
        vm.select_dataframe("test_df")
        
        # Verify selection and signal emission
        assert vm.selected_df_name == "test_df"
        assert len(emitted_columns) > 0
    
    def test_get_dataframe(self, vm_with_dataframe):
        """Test getting DataFrame by name."""
        vm = vm_with_dataframe
        
        df = vm.get_dataframe("test_df")
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)


class TestVMWorkspaceGraphsFiltering:
    """Tests for data filtering functionality."""
    
    def test_apply_filters(self, qapp, mock_settings):
        """Test applying filters to DataFrame."""
        vm = VMWorkspaceGraphs(mock_settings)
        
        # Create DataFrame with numeric data
        df = pd.DataFrame({
            'X': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Y': [10, 20, 15, 25, 30, 35, 40, 45, 50, 55],
            'Category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
        })
        vm.add_dataframe("data", df)
        
        # Apply filter: X > 5 (correct filter format with expression and state)
        filters = [
            {'expression': 'X > 5', 'state': True}
        ]
        filtered_df = vm.apply_filters("data", filters)
        
        # Verify filtering
        assert len(filtered_df) == 5  # Values 6-10
        assert all(filtered_df['X'] > 5)
    
    def test_has_slot_column(self, qapp, mock_settings):
        """Test detecting Slot column."""
        vm = VMWorkspaceGraphs(mock_settings)
        
        # DataFrame with Slot column
        df_with_slot = pd.DataFrame({
            'Slot': [1, 2, 3],
            'Value': [10, 20, 30]
        })
        vm.add_dataframe("with_slot", df_with_slot)
        
        # DataFrame without Slot column
        df_without_slot = pd.DataFrame({
            'X': [1, 2, 3],
            'Y': [10, 20, 30]
        })
        vm.add_dataframe("without_slot", df_without_slot)
        
        # Verify detection
        assert vm.has_slot_column("with_slot") is True
        assert vm.has_slot_column("without_slot") is False
    
    def test_get_unique_slots(self, qapp, mock_settings):
        """Test getting unique slot values."""
        vm = VMWorkspaceGraphs(mock_settings)
        
        df = pd.DataFrame({
            'Slot': [1, 2, 1, 3, 2, 3, 1],
            'Value': [10, 20, 15, 30, 25, 35, 12]
        })
        vm.add_dataframe("data", df)
        
        # Get unique slots
        unique_slots = vm.get_unique_slots("data")
        
        # Verify unique values
        assert set(unique_slots) == {1, 2, 3}


class TestVMWorkspaceGraphsGraphManagement:
    """Tests for graph creation and management."""
    
    def test_create_graph(self, qapp, mock_settings, sample_dataframe):
        """Test creating a new graph."""
        vm = VMWorkspaceGraphs(mock_settings)
        vm.add_dataframe("data", sample_dataframe)
        
        # Create graph with configuration
        plot_config = {
            'df_name': 'data',
            'plot_style': 'line',
            'x': 'X',
            'y': ['Y']
        }
        
        graph = vm.create_graph(plot_config)
        
        # Verify graph was created (create_graph returns MGraph object)
        assert graph is not None
        assert graph.graph_id in vm.graphs
        assert isinstance(graph, MGraph)
    
    def test_update_graph(self, qapp, mock_settings, sample_dataframe):
        """Test updating graph properties."""
        vm = VMWorkspaceGraphs(mock_settings)
        vm.add_dataframe("data", sample_dataframe)
        
        # Create graph
        plot_config = {'df_name': 'data'}
        graph = vm.create_graph(plot_config)
        graph_id = graph.graph_id
        
        # Update properties
        updates = {
            'plot_style': 'scatter',
            'xlabel': 'New X Label',
            'grid': True
        }
        vm.update_graph(graph_id, updates)
        
        # Verify updates
        graph = vm.graphs[graph_id]
        assert graph.plot_style == 'scatter'
        assert graph.xlabel == 'New X Label'
        assert graph.grid is True
    
    def test_delete_graph(self, qapp, mock_settings, sample_dataframe):
        """Test deleting graph."""
        vm = VMWorkspaceGraphs(mock_settings)
        vm.add_dataframe("data", sample_dataframe)
        
        # Create graph
        graph_id = vm.create_graph({'df_name': 'data'})
        
        # Delete graph
        vm.delete_graph(graph_id)
        
        # Verify deletion
        assert graph_id not in vm.graphs
    
    def test_get_graph(self, qapp, mock_settings, sample_dataframe):
        """Test getting graph by ID."""
        vm = VMWorkspaceGraphs(mock_settings)
        vm.add_dataframe("data", sample_dataframe)
        
        # Create graph (returns MGraph object)
        graph_obj = vm.create_graph({'df_name': 'data'})
        graph_id = graph_obj.graph_id
        
        # Get graph
        graph = vm.get_graph(graph_id)
        
        # Verify retrieval
        assert graph is not None
        assert isinstance(graph, MGraph)
        assert graph.graph_id == graph_id


class TestVMWorkspaceGraphsPersistence:
    """Tests for workspace save/load."""
    
    def test_save_workspace(self, qapp, mock_settings, sample_dataframe, temp_workspace, monkeypatch):
        """Test saving workspace."""
        vm = VMWorkspaceGraphs(mock_settings)
        vm.add_dataframe("data", sample_dataframe)
        vm.create_graph({'df_name': 'data', 'plot_style': 'line'})
        
        # Mock file dialog
        save_path = temp_workspace / "test.graphs"
        
        def mock_get_save_filename(*args, **kwargs):
            return str(save_path), ""
        monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_get_save_filename)
        
        # Save workspace
        vm.save_workspace()
        
        # Verify file was created
        assert save_path.exists()
    
    def test_load_workspace(self, qapp, mock_settings, saved_graphs_workspace):
        """Test loading workspace."""
        if not saved_graphs_workspace.exists():
            pytest.skip("Saved graphs workspace not available")
        
        vm = VMWorkspaceGraphs(mock_settings)
        
        # Load workspace
        vm.load_workspace(str(saved_graphs_workspace))
        
        # Verify data was loaded
        assert len(vm.dataframes) > 0 or len(vm.graphs) > 0
    
    def test_save_load_roundtrip(self, qapp, mock_settings, sample_dataframe, temp_workspace, monkeypatch):
        """Test save then load preserves data."""
        # Create workspace with data
        vm1 = VMWorkspaceGraphs(mock_settings)
        vm1.add_dataframe("data", sample_dataframe)
        graph_id = vm1.create_graph({
            'df_name': 'data',
            'plot_style': 'scatter',
            'x': 'X',
            'y': ['Y']
        })
        
        # Save workspace
        save_path = temp_workspace / "roundtrip.graphs"
        
        def mock_get_save_filename(*args, **kwargs):
            return str(save_path), ""
        monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_get_save_filename)
        
        vm1.save_workspace()
        
        # Load into new ViewModel
        vm2 = VMWorkspaceGraphs(mock_settings)
        vm2.load_workspace(str(save_path))
        
        # Verify data matches
        assert "data" in vm2.dataframes
        assert len(vm2.graphs) == 1
        
        # Verify graph properties
        loaded_graph = list(vm2.graphs.values())[0]
        assert loaded_graph.plot_style == 'scatter'
        assert loaded_graph.x == 'X'
        assert loaded_graph.y == ['Y']


class TestVMWorkspaceGraphsClearWorkspace:
    """Tests for clearing workspace."""
    
    def test_clear_workspace(self, qapp, mock_settings, sample_dataframe):
        """Test clearing workspace removes all data."""
        vm = VMWorkspaceGraphs(mock_settings)
        vm.add_dataframe("data", sample_dataframe)
        vm.create_graph({'df_name': 'data'})
        
        # Clear workspace
        vm.clear_workspace()
        
        # Verify everything is cleared
        assert len(vm.dataframes) == 0
        assert len(vm.graphs) == 0
        assert vm.selected_df_name is None
