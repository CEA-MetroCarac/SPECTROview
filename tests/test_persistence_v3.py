import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from spectroview.model.workspace_io import WorkspaceIO
from spectroview.viewmodel.vm_workspace_spectra import VMWorkspaceSpectra
from spectroview.viewmodel.vm_workspace_maps import VMWorkspaceMaps
from spectroview.viewmodel.vm_workspace_graphs import VMWorkspaceGraphs
from spectroview.model.m_spectrum import MSpectrum
from spectroview.model.m_spectra import MSpectra

def test_workspace_io_roundtrip(tmp_path):
    # Setup mock data
    metadata = {
        'version': '1.0',
        'settings': {'color': 'blue'}
    }
    arrays = {
        'x0': np.linspace(0, 10, 100),
        'y0': np.sin(np.linspace(0, 10, 100))
    }
    dataframes = {
        'df1': pd.DataFrame({'A': [1, 2, 3], 'B': [4.5, 5.5, 6.5]})
    }
    
    file_path = str(tmp_path / "test_workspace.zip")
    
    # Save
    WorkspaceIO.save_workspace(file_path, metadata, arrays, dataframes)
    
    # Load
    loaded_meta, loaded_arrays, loaded_dfs, is_legacy = WorkspaceIO.load_workspace(file_path)
    
    assert not is_legacy
    assert loaded_meta == metadata
    np.testing.assert_array_equal(loaded_arrays['x0'], arrays['x0'])
    np.testing.assert_array_equal(loaded_arrays['y0'], arrays['y0'])
    pd.testing.assert_frame_equal(loaded_dfs['df1'], dataframes['df1'])

def test_vm_workspace_spectra_persistence(qapp, mock_settings, mock_file_dialog, tmp_path, sample_spectrum):
    # Setup VM
    vm = VMWorkspaceSpectra(mock_settings)
    vm.spectra.add(sample_spectrum)
    
    # Mock save file path
    save_file = tmp_path / "test_spectra.spectra"
    mock_file_dialog.return_file = str(save_file)
    
    # Save
    vm.save_work()
    assert save_file.exists()
    
    # Load in new VM
    vm_loaded = VMWorkspaceSpectra(mock_settings)
    vm_loaded.load_work(str(save_file))
    
    assert len(vm_loaded.spectra) == 1
    loaded_spec = vm_loaded.spectra[0]
    assert loaded_spec.fname == sample_spectrum.fname
    np.testing.assert_array_equal(loaded_spec.x0, sample_spectrum.x0)
    np.testing.assert_array_equal(loaded_spec.y0, sample_spectrum.y0)
    assert loaded_spec.baseline.mode == sample_spectrum.baseline.mode

def test_vm_workspace_maps_lazy_persistence(qapp, mock_settings, mock_file_dialog, tmp_path, sample_map_dataframe):
    # Setup VM
    vm = VMWorkspaceMaps(mock_settings)
    vm.maps["Si_map"] = sample_map_dataframe
    vm._extract_spectra_from_map("Si_map", sample_map_dataframe)
    
    # Mock save file path
    save_file = tmp_path / "test_map.maps"
    mock_file_dialog.return_file = str(save_file)
    
    # Save
    vm.save_work()
    assert save_file.exists()
    
    # Load in new VM
    vm_loaded = VMWorkspaceMaps(mock_settings)
    vm_loaded.load_work(str(save_file))
    
    assert len(vm_loaded.spectra) == len(vm.spectra)
    
    # Check that spectra are initially lazy-loaded
    first_spec = vm_loaded.spectra[0]
    assert not getattr(first_spec, '_is_lazy_loaded', False)
    assert first_spec.x0 is None or first_spec.y0 is None
    
    # Access spectrum - should trigger lazy loading!
    # Retrieve it through VM which guarantees load resolution
    loaded_spectra_list = vm_loaded._get_spectra_by_fnames([first_spec.fname])
    assert len(loaded_spectra_list) == 1
    resolved_spec = loaded_spectra_list[0]
    
    assert getattr(resolved_spec, '_is_lazy_loaded', True)
    assert resolved_spec.x0 is not None
    assert resolved_spec.y0 is not None
    # minus X, Y, and last value skip in extraction
    expected_len = len(sample_map_dataframe.columns) - 3
    assert len(resolved_spec.x0) == expected_len

def test_vm_workspace_graphs_persistence(qapp, mock_settings, mock_file_dialog, tmp_path, sample_dataframe):
    # Setup VM
    vm = VMWorkspaceGraphs(mock_settings)
    vm.dataframes["data1"] = sample_dataframe
    vm.dataframe_sources["data1"] = "some_source_path.csv"
    
    # Mock save file path
    save_file = tmp_path / "test_graphs.graphs"
    mock_file_dialog.return_file = str(save_file)
    
    # Save
    vm.save_workspace()
    assert save_file.exists()
    
    # Load in new VM
    vm_loaded = VMWorkspaceGraphs(mock_settings)
    vm_loaded.load_workspace(str(save_file))
    
    assert "data1" in vm_loaded.dataframes
    pd.testing.assert_frame_equal(vm_loaded.dataframes["data1"], sample_dataframe)
    assert vm_loaded.dataframe_sources["data1"] == "some_source_path.csv"
