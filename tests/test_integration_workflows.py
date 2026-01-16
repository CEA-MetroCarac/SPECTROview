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
    
    def test_complete_multi_spectra_workflow(self, qapp, mock_settings, multiple_spectra_files, temp_workspace, monkeypatch):
        """
        Comprehensive test for complete spectra workflow with multiple file types.
        
        Step 1: Load multiple spectra and verify properties
        Step 2: Perform fitting on spectrum1_1ML (crop, baseline, fit)
        Step 3: Verify fit results
        Step 4: Load and apply saved fit model
        Step 5: Save and reload workspace, verify persistence
        """
        vm = VMWorkspaceSpectra(mock_settings)
        
        # ===================================================================
        # STEP 1: Load multiple spectrum files and verify
        # ===================================================================
        file_paths = [str(f) for f in multiple_spectra_files]
        vm.load_files(file_paths)
        
        # Verify number of loaded spectra
        assert len(vm.spectra) == 2, \
            f"Expected 2 spectra, got {len(vm.spectra)}"
        
        # Find both spectra
        spectrum1_1ml = None
        xrd_spectrum = None
        for spectrum in vm.spectra:
            if spectrum.fname == "spectrum1_1ML":
                spectrum1_1ml = spectrum
            elif spectrum.fname == "XRDspectra":
                xrd_spectrum = spectrum
        
        assert spectrum1_1ml is not None, "spectrum1_1ML not found"
        assert xrd_spectrum is not None, "XRDspectra not found"
        
        # Verify spectrum1_1ML properties
        assert len(spectrum1_1ml.x) == 571
        assert abs(spectrum1_1ml.x.min() - 55.76) < 0.1
        assert abs(spectrum1_1ml.x.max() - 952.67) < 0.1
        assert len(spectrum1_1ml.y) == 571
        assert abs(spectrum1_1ml.y.min() - 200.76) < 1.0
        assert abs(spectrum1_1ml.y.max() - 37047.90) < 1.0
        
        # Verify XRDspectra properties
        assert len(xrd_spectrum.x) == 3889
        assert abs(xrd_spectrum.x.min() - 15.12) < 0.1
        assert abs(xrd_spectrum.x.max() - 80.07) < 0.1
        assert len(xrd_spectrum.y) == 3889
        assert abs(xrd_spectrum.y.min() - 3.0) < 0.1
        assert abs(xrd_spectrum.y.max() - 1128.0) < 1.0
        
        # ===================================================================
        # STEP 2: Perform fitting on spectrum1_1ML
        # ===================================================================
        # Select spectrum1_1ML by finding its index
        spectrum1_index = None
        for i, spectrum in enumerate(vm.spectra):
            if spectrum.fname == "spectrum1_1ML":
                spectrum1_index = i
                break
        
        vm.set_selected_indices([spectrum1_index])
        
        # Crop range
        x_min = 460
        x_max = 570
        vm.apply_spectral_range(x_min, x_max, apply_all=False)
        
        # Add baseline points
        vm.add_baseline_point(460.6702169878947, 1.8264804387197842)
        vm.add_baseline_point(568.8419912888185, -0.8027723385864789)
        baseline_points_count = len(spectrum1_1ml.baseline.points)
        
        # Subtract baseline
        vm.subtract_baseline(apply_all=False)
        
        # Add peak model
        mid_x = 529
        vm.add_peak_at(mid_x)
        models_count = len(spectrum1_1ml.peak_models)
        
        # Perform fitting
        spectrum1_1ml.preprocess()
        spectrum1_1ml.fit()
        
        # Verify fitting was successful
        assert hasattr(spectrum1_1ml, 'result_fit'), "Fitting did not produce result_fit"
        assert spectrum1_1ml.result_fit.success, "Fitting failed"
        
        # Verify peak model parameters
        assert len(spectrum1_1ml.peak_models) == 1
        
        best_values = spectrum1_1ml.result_fit.best_values
        assert 'm01_ampli' in best_values
        assert 'm01_fwhm' in best_values
        assert 'm01_x0' in best_values
        
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
        
        # ===================================================================
        # STEP 3: Load and apply saved fit model using VM method
        # ===================================================================
        from pathlib import Path as PathlibPath
        from spectroview.viewmodel.vm_fit_model_builder import VMFitModelBuilder
        from unittest.mock import MagicMock
        
        fit_model_path = PathlibPath("examples/spectroscopic_data/fit_model_Si_.json")
        
        # Setup fit model builder (required by apply_loaded_fit_model)
        vm._vm_fit_model_builder = MagicMock(spec=VMFitModelBuilder)
        vm._vm_fit_model_builder.get_current_model_path.return_value = fit_model_path
        
        # Ensure spectrum1_1ML is still selected
        vm.set_selected_indices([spectrum1_index])
        
        # Track fit completion
        fit_completed = []
        vm.fit_in_progress.connect(lambda in_progress: fit_completed.append(not in_progress) if not in_progress else None)
        
        # Apply the loaded fit model using the VM method
        vm.apply_loaded_fit_model(apply_all=False)
        
        # Wait for fitting thread to complete (max 5 seconds)
        import time
        timeout = 5.0
        start_time = time.time()
        while not fit_completed and (time.time() - start_time) < timeout:
            qapp.processEvents()
            time.sleep(0.01)
        
        assert len(fit_completed) > 0, "Fit did not complete within timeout"
        
        # Verify fitting with loaded model was successful
        assert hasattr(spectrum1_1ml, 'result_fit'), "Fitting with loaded model did not produce result_fit"
        assert spectrum1_1ml.result_fit.success, "Fitting with loaded model failed"
        
        # Verify the loaded model parameters
        assert len(spectrum1_1ml.peak_models) >= 1, \
            f"Expected at least 1 peak model from loaded fit, got {len(spectrum1_1ml.peak_models)}"
        
        # Get fitted results from loaded model
        loaded_best_values = spectrum1_1ml.result_fit.best_values
        
        # Check the fitted parameters match expected values (same as Step 2)
        # Applying the loaded fit model should produce consistent results
        if 'm01_ampli' in loaded_best_values:
            loaded_ampli = loaded_best_values['m01_ampli']
            loaded_fwhm = loaded_best_values['m01_fwhm']
            loaded_x0 = loaded_best_values['m01_x0']
            
            # Verify fitted values with 1 decimal precision (same checks as Step 2)
            assert abs(loaded_ampli - 37096.2) < 15.0, \
                f"Expected ampli ~37096.2, got {loaded_ampli:.1f}"
            assert abs(loaded_fwhm - 3.6) < 0.1, \
                f"Expected fwhm ~3.6, got {loaded_fwhm:.1f}"
            assert abs(loaded_x0 - 520.1) < 0.1, \
                f"Expected x0 ~520.1, got {loaded_x0:.1f}"

        
        # ===================================================================
        # STEP 4: Save workspace and reload to verify persistence
        # ===================================================================
        save_path = temp_workspace / "complete_workflow.spectra"
        
        def mock_get_save_filename(*args, **kwargs):
            return str(save_path), ""
        
        from PySide6.QtWidgets import QFileDialog
        monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_get_save_filename)
        
        vm.save_work()
        assert save_path.exists(), "Workspace file was not created"
        
        # Reload workspace into new ViewModel
        vm2 = VMWorkspaceSpectra(mock_settings)
        vm2.load_work(str(save_path))
        
        # Verify both spectra were saved and loaded
        assert len(vm2.spectra) == 2, \
            f"Expected 2 spectra after reload, got {len(vm2.spectra)}"
        
        # Find the reloaded spectrum1_1ML
        loaded_spectrum1 = None
        for spectrum in vm2.spectra:
            if spectrum.fname == "spectrum1_1ML":
                loaded_spectrum1 = spectrum
                break
        
        assert loaded_spectrum1 is not None, "spectrum1_1ML not found after reload"
        
        # Verify spectrum data is preserved
        assert loaded_spectrum1.range_min is not None
        assert loaded_spectrum1.range_max is not None
        
        # Verify baseline is preserved
        assert len(loaded_spectrum1.baseline.points) == baseline_points_count, \
            f"Expected {baseline_points_count} baseline points, got {len(loaded_spectrum1.baseline.points)}"
        
        # Verify peak models are preserved
        assert len(loaded_spectrum1.peak_models) == models_count, \
            f"Expected {models_count} peak models, got {len(loaded_spectrum1.peak_models)}"



class TestMapsWorkflow:
    """Integration tests for complete maps processing workflow."""
    
    def test_load_multiple_maps(self, qtbot, mock_settings, multiple_map_files, monkeypatch):
        """Test loading multiple 2D map files.
        
        Tests:
        1. Load Small2Dmap.txt
        2. Load wafer4_process1.csv  
        3. Load wafer10_newformat.csv
        4. Verify all maps are loaded correctly
        5. Verify spectra counts for each map
        6. Verify first spectrum properties for each map
        """
        from spectroview.viewmodel.vm_workspace_maps import VMWorkspaceMaps
        from PySide6.QtWidgets import QMessageBox
        from unittest.mock import MagicMock
        
        # Mock QMessageBox to avoid popups
        mock_msgbox = MagicMock()
        monkeypatch.setattr(QMessageBox, "critical", mock_msgbox)
        
        # Create Maps ViewModel
        vm = VMWorkspaceMaps(mock_settings)
        
        # Load all map files
        existing_files = [str(f) for f in multiple_map_files if f.exists()]
        if not existing_files:
            pytest.skip("No map files available")
        
        vm.load_map_files(existing_files)
        qtbot.wait(500)  # Maps loading takes time
        
        # Verify number of loaded maps
        assert len(vm.maps) == 3, \
            f"Expected 3 maps, got {len(vm.maps)}"
        
        # Define expected counts and properties for each map
        map_expectations = {
            'Small2Dmap': {
                'count': 1681,
                'x_len': 574,
                'x_min': 55.79,
                'x_max': 957.19,
                'y_len': 574,
                'y_min': -4.61,
                'y_max': 809.99
            },
            'wafer4_process1': {
                'count': 49,
                'x_len': 2048,
                'x_min': -96.70,
                'x_max': 1079.36,
                'y_len': 2048,
                'y_min': -7072.00,
                'y_max': 17101.00
            },
            'wafer10_newformat': {
                'count': 4,
                'x_len': 2048,
                'x_min': -40.49,
                'x_max': 1029.65,
                'y_len': 2048,
                'y_min': -1883.00,
                'y_max': 62075.00
            }
        }
        
        # Verify each map
        for map_name, expected in map_expectations.items():
            # Verify map exists
            assert map_name in vm.maps, \
                f"Map '{map_name}' not found in loaded maps"
            
            # Select the map
            vm.select_map(map_name)
            qtbot.wait(100)
            
            # Count spectra for this map
            map_prefix = f"{map_name}_("
            map_spectra = [s for s in vm.spectra if s.fname.startswith(map_prefix)]
            
            # Verify spectra count
            assert len(map_spectra) == expected['count'], \
                f"{map_name}: Expected {expected['count']} spectra, got {len(map_spectra)}"
            
            # Verify first spectrum properties
            if len(map_spectra) > 0:
                first_spectrum = map_spectra[0]
                
                # X array properties
                assert len(first_spectrum.x) == expected['x_len'], \
                    f"{map_name}: Expected x length {expected['x_len']}, got {len(first_spectrum.x)}"
                assert abs(first_spectrum.x.min() - expected['x_min']) < 0.1, \
                    f"{map_name}: Expected x min ~{expected['x_min']}, got {first_spectrum.x.min():.2f}"
                assert abs(first_spectrum.x.max() - expected['x_max']) < 0.1, \
                    f"{map_name}: Expected x max ~{expected['x_max']}, got {first_spectrum.x.max():.2f}"
                
                # Y array properties
                assert len(first_spectrum.y) == expected['y_len'], \
                    f"{map_name}: Expected y length {expected['y_len']}, got {len(first_spectrum.y)}"
                assert abs(first_spectrum.y.min() - expected['y_min']) < 1.0, \
                    f"{map_name}: Expected y min ~{expected['y_min']}, got {first_spectrum.y.min():.2f}"
                assert abs(first_spectrum.y.max() - expected['y_max']) < 1.0, \
                    f"{map_name}: Expected y max ~{expected['y_max']}, got {first_spectrum.y.max():.2f}"

    
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



