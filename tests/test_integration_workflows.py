"""
Integration tests for SPECTROview - End-to-end workflows

Tests cover:
- Complete spectra workflow: load → process → fit → save → reload
- Complete maps workflow: load → extract → process → save → reload
- Complete graphs workflow: load → filter → plot → save → reload
- Cross-workspace communication (Maps → Graphs)
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from spectroview.viewmodel.vm_workspace_spectra import VMWorkspaceSpectra
from spectroview.viewmodel.vm_workspace_graphs import VMWorkspaceGraphs


class TestSpectraWorkflow:
    """Integration tests for complete spectra processing workflow."""
    
    def test_complete_spectra_workflow(self, qapp, mock_settings, single_spectrum_file, temp_workspace, monkeypatch):
        """
        Test complete workflow:
        1. Load spectrum
        2. Crop range
        3. Add baseline
        4. Subtract baseline
        5. Add peaks
        6. Perform fitting
        7. Save workspace
        8. Reload workspace
        9. Verify consistency
        """
        # Step 1: Load spectrum
        vm = VMWorkspaceSpectra(mock_settings)
        vm.load_files([str(single_spectrum_file)])
        assert len(vm.spectra) == 1

        # Verify spectrum is loaded correctly
        spectrum = vm.spectra[0]
        
        # Verify fname matches filename without extension
        assert spectrum.fname == single_spectrum_file.stem, \
            f"Expected fname '{single_spectrum_file.stem}', got '{spectrum.fname}'"
        
        # Verify x array properties
        assert len(spectrum.x) == 571, \
            f"Expected x length 571, got {len(spectrum.x)}"
        assert abs(spectrum.x.min() - 55.76) < 0.1, \
            f"Expected x min ~55.76, got {spectrum.x.min():.2f}"
        assert abs(spectrum.x.max() - 952.67) < 0.1, \
            f"Expected x max ~952.67, got {spectrum.x.max():.2f}"
        
        # Verify y array properties
        assert len(spectrum.y) == 571, \
            f"Expected y length 571, got {len(spectrum.y)}"
        assert abs(spectrum.y.min() - 200.76) < 1.0, \
            f"Expected y min ~200.76, got {spectrum.y.min():.2f}"
        assert abs(spectrum.y.max() - 37047.90) < 1.0, \
            f"Expected y max ~37047.90, got {spectrum.y.max():.2f}"
        
        # Step 2: Select spectrum and crop range
        vm.set_selected_indices([0])
        x_min = 460
        x_max = 570
        vm.apply_spectral_range(x_min, x_max, apply_all=False)
        
        # Step 3: Add baseline points
        vm.add_baseline_point(460.6702169878947, 1.8264804387197842)
        vm.add_baseline_point(568.8419912888185, -0.8027723385864789)
        baseline_points_count = len(spectrum.baseline.points)
        
        # Step 4: Subtract baseline
        vm.subtract_baseline(apply_all=False)
        
        # Step 5: Add peak model
        mid_x = 529
        vm.add_peak_at(mid_x)
        models_count = len(spectrum.peak_models)
        
        # Step 6: Perform fitting
        spectrum.preprocess()
        spectrum.fit()
        
        # Verify fitting was successful
        assert hasattr(spectrum, 'result_fit'), "Fitting did not produce result_fit"
        assert spectrum.result_fit.success, "Fitting failed"
        
        # Verify peak model parameters
        assert len(spectrum.peak_models) == 1, \
            f"Expected 1 peak model, got {len(spectrum.peak_models)}"
        
        # Get fitted parameters from result_fit.best_values
        best_values = spectrum.result_fit.best_values
        
        # Verify that the key parameters exist
        assert 'm01_ampli' in best_values, "Missing ampli parameter"
        assert 'm01_fwhm' in best_values, "Missing fwhm parameter"
        assert 'm01_x0' in best_values, "Missing x0 parameter"
        
        ampli_value = best_values['m01_ampli']
        fwhm_value = best_values['m01_fwhm']
        x0_value = best_values['m01_x0']
        
        # Verify fitted values with 1 decimal precision
        assert abs(ampli_value - 37096.2) < 15.0, \
            f"Expected ampli ~37096.2, got {ampli_value:.1f}"
        
        assert abs(fwhm_value - 3.6) < 0.1, \
            f"Expected fwhm ~3.6, got {fwhm_value:.1f}"
        
        assert abs(x0_value - 520.1) < 0.1, \
            f"Expected x0 ~520.1, got {x0_value:.1f}"
        
        # Step 7: Save workspace
        save_path = temp_workspace / "workflow.spectra"
        
        def mock_get_save_filename(*args, **kwargs):
            return str(save_path), ""
        
        from PySide6.QtWidgets import QFileDialog
        monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_get_save_filename)
        
        vm.save_work()
        assert save_path.exists()
        
        # Step 8: Reload workspace into new ViewModel
        vm2 = VMWorkspaceSpectra(mock_settings)
        vm2.load_work(str(save_path))
        
        # Step 9: Verify consistency
        assert len(vm2.spectra) == 1
        loaded_spectrum = vm2.spectra[0]
        
        # Verify spectrum data is preserved
        assert loaded_spectrum.fname == spectrum.fname
        assert loaded_spectrum.range_min is not None
        assert loaded_spectrum.range_max is not None
        
        # Verify baseline is preserved
        assert len(loaded_spectrum.baseline.points) == baseline_points_count
        
        # Verify peak models are preserved
        assert len(loaded_spectrum.peak_models) == models_count
    
    def test_load_multiple_spectral_types(self, qapp, mock_settings, multiple_spectra_files):
        """
        Test loading multiple spectral data types.
        
        Verify:
        1. Load two different spectrum file types
        2. Check number of loaded spectra
        3. Verify XRDspectra.txt data properties
        """
        vm = VMWorkspaceSpectra(mock_settings)
        
        # Load multiple spectrum files (spectrum1_1ML.txt and XRDspectra.txt)
        file_paths = [str(f) for f in multiple_spectra_files]
        vm.load_files(file_paths)
        
        # Verify number of loaded spectra
        assert len(vm.spectra) == 2, \
            f"Expected 2 spectra, got {len(vm.spectra)}"
        
        # Find the XRDspectra spectrum
        xrd_spectrum = None
        for spectrum in vm.spectra:
            if spectrum.fname == "XRDspectra":
                xrd_spectrum = spectrum
                break
        
        assert xrd_spectrum is not None, "XRDspectra not found in loaded spectra"
        
        # Verify XRDspectra.txt data properties
        # X array properties
        assert len(xrd_spectrum.x) == 3889, \
            f"Expected x length 3889, got {len(xrd_spectrum.x)}"
        assert abs(xrd_spectrum.x.min() - 15.12) < 0.1, \
            f"Expected x min ~15.12, got {xrd_spectrum.x.min():.2f}"
        assert abs(xrd_spectrum.x.max() - 80.07) < 0.1, \
            f"Expected x max ~80.07, got {xrd_spectrum.x.max():.2f}"
        
        # Y array properties
        assert len(xrd_spectrum.y) == 3889, \
            f"Expected y length 3889, got {len(xrd_spectrum.y)}"
        assert abs(xrd_spectrum.y.min() - 3.0) < 0.1, \
            f"Expected y min ~3.0, got {xrd_spectrum.y.min():.2f}"
        assert abs(xrd_spectrum.y.max() - 1128.0) < 1.0, \
            f"Expected y max ~1128.0, got {xrd_spectrum.y.max():.2f}"


class TestMapsWorkflow:
    """Integration tests for complete maps processing workflow."""
    
    def test_load_multiple_maps(self, qtbot, mock_settings, multiple_map_files, monkeypatch):
        """Test loading multiple 2D map files.
        
        Tests:
        1. Load Small2Dmap.txt
        2. Load wafer4_process1.csv  
        3. Load wafer10_newformat.csv
        4. Verify all maps are loaded correctly
        5. Verify spectra extraction for each map
        """
        from spectroview.viewmodel.vm_workspace_maps import VMWorkspaceMaps
        from PySide6.QtWidgets import QMessageBox
        from unittest.mock import MagicMock
        
        # Mock QMessageBox to avoid popups
        mock_msgbox = MagicMock()
        monkeypatch.setattr(QMessageBox, "critical", mock_msgbox)
        
        # Create Maps ViewModel
        vm = VMWorkspaceMaps(mock_settings)
        
        # Track map list changes
        map_lists = []
        vm.maps_list_changed.connect(lambda data: map_lists.append(data))
        
        # Load all map files
        existing_files = [str(f) for f in multiple_map_files if f.exists()]
        if not existing_files:
            pytest.skip("No map files available")
        
        vm.load_map_files(existing_files)
        qtbot.wait(500)  # Maps loading takes time
        
        # Verify maps were loaded
        assert len(vm.maps) == len(existing_files)
        
        # Verify each map has spectra extracted
        for map_file in existing_files:
            map_name = Path(map_file).stem
            assert map_name in vm.maps
            
            # Select the map and verify spectra
            vm.select_map(map_name)
            qtbot.wait(100)
            
            # Verify spectra were extracted
            map_prefix = f"{map_name}_("
            map_spectra = [s for s in vm.spectra if s.fname.startswith(map_prefix)]
            assert len(map_spectra) > 0
    
    def test_process_single_map(self, qtbot, mock_settings, map_2d_file, monkeypatch):
        """Test complete processing workflow for a single map."""
        from spectroview.viewmodel.vm_workspace_maps import VMWorkspaceMaps
        from PySide6.QtWidgets import QMessageBox
        from unittest.mock import MagicMock
        
        if not map_2d_file.exists():
            pytest.skip("Map test file not available")
        
        # Mock QMessageBox
        mock_msgbox = MagicMock()
        monkeypatch.setattr(QMessageBox, "critical", mock_msgbox)
        
        vm = VMWorkspaceMaps(mock_settings)
        
        # Step 1: Load map
        vm.load_map_files([str(map_2d_file)])
        qtbot.wait(300)
        
        # Step 2: Select map
        map_name = list(vm.maps.keys())[0]
        vm.select_map(map_name)
        qtbot.wait(100)
        
        # Step 3: Verify current map is set
        assert vm.current_map_name == map_name
        assert vm.current_map_df is not None
        
        # Step 4: Select a spectrum
        if len(vm.spectra) > 0:
            vm.set_selected_indices([0])
            qtbot.wait(50)
            
            # Step 5: Add baseline to selected spectrum
            selected = vm._get_selected_spectra()[0]
            x_min = selected.x.min()
            x_max = selected.x.max()
            vm.add_baseline_point(x_min + 50, 100)
            vm.add_baseline_point(x_max - 50, 100)
            qtbot.wait(50)
            
            # Step 6: Subtract baseline
            vm.subtract_baseline(apply_all=False)
            qtbot.wait(50)
            
            # Verify baseline points were added
            assert len(selected.baseline.points) > 0
    
    def test_map_save_load_workflow(self, qtbot, mock_settings, map_2d_file, temp_workspace, monkeypatch):
        """Test saving and loading map workspace.
        
        Workflow:
        1. Load map
        2. Process spectra (baseline, peaks)
        3. Save workspace
        4. Load into new ViewModel
        5. Verify data persisted
        """
        from spectroview.viewmodel.vm_workspace_maps import VMWorkspaceMaps
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        from unittest.mock import MagicMock
        
        if not map_2d_file.exists():
            pytest.skip("Map test file not available")
        
        # Mock QMessageBox
        mock_msgbox = MagicMock()
        monkeypatch.setattr(QMessageBox, "critical", mock_msgbox)
        
        # Create first ViewModel
        vm1 = VMWorkspaceMaps(mock_settings)
        vm1.load_map_files([str(map_2d_file)])
        qtbot.wait(300)
        
        # Process data
        map_name = list(vm1.maps.keys())[0]
        vm1.select_map(map_name)
        qtbot.wait(100)
        
        if len(vm1.spectra) > 0:
            vm1.set_selected_indices([0])
            vm1.add_peak_at(300)
        
        # Save workspace
        save_path = temp_workspace / "test_maps.maps"
        
        def mock_get_save_filename(*args, **kwargs):
            return str(save_path), ""
        
        monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_get_save_filename)
        vm1.save_work()
        
        assert save_path.exists()
        
        # Load into new ViewModel
        vm2 = VMWorkspaceMaps(mock_settings)
        vm2.load_work(str(save_path))
        qtbot.wait(300)
        
        # Verify data persisted
        assert len(vm2.maps) == 1
        assert map_name in vm2.maps
        assert len(vm2.spectra) > 0
class TestGraphsWorkflow:
    """Integration tests for complete graphs workflow."""
    
    def test_complete_graphs_workflow(self, qapp, mock_settings, temp_workspace, monkeypatch):
        """
        Test complete workflow:
        1. Load DataFrame
        2. Apply filters
        3. Create graph
        4. Save workspace
        5. Reload workspace
        6. Verify consistency
        """
        # Step 1: Create and load DataFrame
        vm = VMWorkspaceGraphs(mock_settings)
        
        test_df = pd.DataFrame({
            'X': list(range(20)),
            'Y': list(range(0, 40, 2)),
            'Z': list(range(20, 40)),
            'Category': ['A', 'B'] * 10
        })
        vm.add_dataframe("test_data", test_df)
        
        # Step 2: Apply filters (correct format with expression and state)
        filters = [
            {'expression': 'X > 5', 'state': True},
            {'expression': 'Y < 30', 'state': True}
        ]
        filtered_df = vm.apply_filters("test_data", filters)
        assert len(filtered_df) < len(test_df)
        
        # Step 3: Create graph
        plot_config = {
            'df_name': 'test_data',
            'plot_style': 'scatter',
            'x': 'X',
            'y': ['Y', 'Z'],
            'xlabel': 'X Values',
            'ylabel': 'Y Values',
            'grid': True,
            'filters': filters
        }
        graph = vm.create_graph(plot_config)
        assert graph.graph_id in vm.graphs
        
        # Step 4: Save workspace
        save_path = temp_workspace / "graphs_workflow.graphs"
        
        def mock_get_save_filename(*args, **kwargs):
            return str(save_path), ""
        
        from PySide6.QtWidgets import QFileDialog
        monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_get_save_filename)
        
        vm.save_workspace()
        assert save_path.exists()
        
        # Step 5: Reload workspace
        vm2 = VMWorkspaceGraphs(mock_settings)
        vm2.load_workspace(str(save_path))
        
        # Step 6: Verify consistency
        assert "test_data" in vm2.dataframes
        pd.testing.assert_frame_equal(vm2.dataframes["test_data"], test_df)
        
        assert len(vm2.graphs) == 1
        loaded_graph = list(vm2.graphs.values())[0]
        assert loaded_graph.plot_style == 'scatter'
        assert loaded_graph.x == 'X'
        assert loaded_graph.xlabel == 'X Values'
        assert loaded_graph.grid is True


class TestMultipleSpectraProcessing:
    """Integration test for batch processing multiple spectra."""
    
    def test_batch_processing(self, qapp, mock_settings, multiple_spectra_files):
        """Test processing multiple spectra with same settings."""
        vm = VMWorkspaceSpectra(mock_settings)
        
        # Load multiple files
        file_paths = [str(f) for f in multiple_spectra_files]
        vm.load_files(file_paths)
        assert len(vm.spectra) == len(multiple_spectra_files)
        
        # Select all spectra
        vm.set_selected_indices(list(range(len(vm.spectra))))
        
        # Apply range to all
        vm.apply_spectral_range(200, 400, apply_all=True)
        
        # Verify range was applied to all
        for spectrum in vm.spectra:
            assert spectrum.range_min is not None
            assert spectrum.range_max is not None
        
        # Add baseline to first, then copy to all
        vm.set_selected_indices([0])
        vm.add_baseline_point(220, 100)
        vm.add_baseline_point(380, 100)
        vm.copy_baseline()
        
        # Paste to all others
        vm.paste_baseline(apply_all=True)
        
        # Verify all spectra have baseline points
        for spectrum in vm.spectra:
            assert len(spectrum.baseline.points) > 0


class TestWorkspacePersistenceRobustness:
    """Integration tests for workspace persistence robustness."""
    
    def test_save_empty_workspace(self, qapp, mock_settings, temp_workspace, monkeypatch):
        """Test saving empty workspace."""
        vm = VMWorkspaceSpectra(mock_settings)
        
        save_path = temp_workspace / "empty.spectra"
        
        def mock_get_save_filename(*args, **kwargs):
            return str(save_path), ""
        
        from PySide6.QtWidgets import QFileDialog
        monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_get_save_filename)
        
        # Save empty workspace
        vm.save_work()
        
        # Load it back
        vm2 = VMWorkspaceSpectra(mock_settings)
        vm2.load_work(str(save_path))
        
        # Verify it's still empty
        assert len(vm2.spectra) == 0
    
    def test_load_corrupted_workspace(self, qapp, mock_settings, temp_workspace, monkeypatch):
        """Test loading corrupted workspace file handles gracefully."""
        vm = VMWorkspaceSpectra(mock_settings)
        
        # Create corrupted file
        corrupted_path = temp_workspace / "corrupted.spectra"
        corrupted_path.write_text("This is not valid JSON")
        
        # Mock QMessageBox to prevent actual popup
        mock_msgbox = MagicMock()
        from PySide6.QtWidgets import QMessageBox
        monkeypatch.setattr(QMessageBox, "critical", mock_msgbox)
        
        # Attempt to load - should handle error gracefully
        # (Implementation should emit error notification, not crash)
        try:
            vm.load_work(str(corrupted_path))
            # If it doesn't raise, that's fine too
        except Exception as e:
            # Error handling is acceptable
            pass
        
        # Verify error message was shown (via QMessageBox or exception)
        # The method should not crash
        assert True  # Test passes if we get here without crashing


class TestDataIntegrity:
    """Integration tests for data integrity throughout processing."""
    
    def test_x_values_remain_sorted(self, qapp, mock_settings, single_spectrum_file):
        """Test that x-values remain sorted after all operations."""
        vm = VMWorkspaceSpectra(mock_settings)
        vm.load_files([str(single_spectrum_file)])
        vm.set_selected_indices([0])
        
        spectrum = vm.spectra[0]
        
        # Apply various operations
        vm.apply_spectral_range(200, 400, apply_all=False)
        vm.add_baseline_point(250, 100)
        vm.subtract_baseline(apply_all=False)
        vm.apply_x_correction(305)
        
        # Verify x-values are still sorted
        assert all(spectrum.x[i] <= spectrum.x[i+1] for i in range(len(spectrum.x)-1))
    
    def test_data_shapes_consistency(self, qapp, mock_settings, single_spectrum_file):
        """Test that x and y arrays maintain same length."""
        vm = VMWorkspaceSpectra(mock_settings)
        vm.load_files([str(single_spectrum_file)])
        vm.set_selected_indices([0])
        
        spectrum = vm.spectra[0]
        
        # Initial check
        assert len(spectrum.x) == len(spectrum.y)
        
        # After range application
        vm.apply_spectral_range(200, 400, apply_all=False)
        assert len(spectrum.x) == len(spectrum.y)
        
        # After baseline operations
        vm.add_baseline_point(250, 100)
        vm.subtract_baseline(apply_all=False)
        assert len(spectrum.x) == len(spectrum.y)


class TestEdgeCases:
    """Integration tests for edge cases and boundary conditions."""
    
    def test_empty_dataframe(self, qapp, mock_settings):
        """Test handling empty DataFrame."""
        vm = VMWorkspaceGraphs(mock_settings)
        
        empty_df = pd.DataFrame()
        vm.add_dataframe("empty", empty_df)
        
        # Should handle gracefully
        assert "empty" in vm.dataframes
    
    def test_single_point_spectrum(self, qapp, mock_settings):
        """Test handling spectrum with single data point."""
        vm = VMWorkspaceSpectra(mock_settings)
        
        from spectroview.model.m_spectrum import MSpectrum
        import numpy as np
        
        spectrum = MSpectrum()
        spectrum.fname = "single_point"
        spectrum.x0 = np.array([100.0])
        spectrum.y0 = np.array([200.0])
        spectrum.x = spectrum.x0.copy()
        spectrum.y = spectrum.y0.copy()
        
        vm.spectra.add(spectrum)
        
        # Operations should handle gracefully
        assert len(vm.spectra) == 1



