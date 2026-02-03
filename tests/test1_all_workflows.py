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
import numpy as np
import time
from pathlib import Path
from pathlib import Path as PathlibPath
from unittest.mock import patch, MagicMock

from PySide6.QtWidgets import QFileDialog, QMessageBox

from spectroview.viewmodel.vm_workspace_spectra import VMWorkspaceSpectra
from spectroview.viewmodel.vm_workspace_graphs import VMWorkspaceGraphs
from spectroview.viewmodel.vm_workspace_maps import VMWorkspaceMaps
from spectroview.viewmodel.vm_fit_model_builder import VMFitModelBuilder
from spectroview.model.m_spectrum import MSpectrum


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
        fit_model_path = PathlibPath("examples/predefined_fit_models/fit_model_Si_.json")
        
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
        
        # Get fitted results
        best_values = spectrum1_1ml.result_fit.best_values
        
        # Check the fitted parameters match expected values (same as Step 2)
        # Applying the loaded fit model should produce consistent results
        if 'm01_ampli' in best_values:
            ampli = best_values['m01_ampli']
            fwhm = best_values['m01_fwhm']
            x0 = best_values['m01_x0']
            
            # Verify fitted values with 1 decimal precision (same checks as Step 2)
            assert abs(ampli - 37096.2) < 15.0, \
                f"Expected ampli ~37096.2, got {ampli:.1f}"
            assert abs(fwhm - 3.6) < 0.1, \
                f"Expected fwhm ~3.6, got {fwhm:.1f}"
            assert abs(x0 - 520.1) < 0.1, \
                f"Expected x0 ~520.1, got {x0:.1f}"

        
        # ===================================================================
        # STEP 4: Save workspace and reload to verify persistence
        # ===================================================================
        save_path = temp_workspace / "complete_workflow.spectra"
        
        def mock_get_save_filename(*args, **kwargs):
            return str(save_path), ""
        
        
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
        1. Load 2Dmap_Si.txt
        2. Load wafer1_process1.csv  
        3. Load wafer4_newformat.csv
        4. Verify all maps are loaded correctly
        5. Verify spectra counts for each map
        6. Verify first spectrum properties for each map
        """
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
            '2Dmap_Si': {
                'count': 1681,
                'x_len': 574,
                'x_min': 55.79,
                'x_max': 957.19,
                'y_len': 574,
                'y_min': -4.61,
                'y_max': 809.99
            },
            'wafer1_process1': {
                'count': 49,
                'x_len': 2048,
                'x_min': -96.70,
                'x_max': 1079.36,
                'y_len': 2048,
                'y_min': -7072.00,
                'y_max': 17101.00
            },
            'wafer4_newformat': {
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

    
    
    def test_process_single_map(self, qapp, qtbot, mock_settings, temp_workspace, monkeypatch):
        """
        Test complete map processing workflow.
        
        Steps:
        1. Load 2Dmap_Si
        2. Select map and first spectrum
        3. Load and apply fit model
        4. Verify fitted results of first spectrum
        5. Collect fit results and verify DataFrame
        6. Save workspace
        7. Reload workspace and verify persistence
        """
        from pathlib import Path as PathlibPath
        
        map_file = PathlibPath("examples/spectroscopic_data/2Dmap_Si.txt")
        if not map_file.exists():
            pytest.skip("Map test file not available")
        
        # Mock QMessageBox
        mock_msgbox = MagicMock()
        monkeypatch.setattr(QMessageBox, "critical", mock_msgbox)
        
        vm = VMWorkspaceMaps(mock_settings)
        
        # ===================================================================
        # STEP 1: Load 2Dmap_Si
        # ===================================================================
        vm.load_map_files([str(map_file)])
        qtbot.wait(500)
        
        # Verify map was loaded
        assert "2Dmap_Si" in vm.maps, "2Dmap_Si not loaded"
        
        # ===================================================================
        # STEP 2: Select map
        # ===================================================================
        vm.select_map("2Dmap_Si")
        qtbot.wait(200)
        
        # Verify map selection
        assert vm.current_map_name == "2Dmap_Si"
        assert vm.current_map_df is not None
        assert len(vm.spectra) == 1681, f"Expected 1681 spectra, got {len(vm.spectra)}"
        
        # ===================================================================
        # STEP 3: Load and apply fit model to first spectrum
        # ===================================================================
        fit_model_path = PathlibPath("examples/predefined_fit_models/fit_model_Si_.json")
        
        # Setup fit model builder
        vm._vm_fit_model_builder = MagicMock(spec=VMFitModelBuilder)
        vm._vm_fit_model_builder.get_current_model_path.return_value = fit_model_path
        
        # Select first spectrum
        vm.set_selected_indices([0])
        qtbot.wait(50)
        
        # Track fit completion
        fit_completed = []
        vm.fit_in_progress.connect(lambda in_progress: fit_completed.append(not in_progress) if not in_progress else None)
        
        # Apply the loaded fit model
        vm.apply_loaded_fit_model(apply_all=False)
        
        # Wait for fitting thread to complete (max 5 seconds)
        timeout = 5.0
        start_time = time.time()
        while not fit_completed and (time.time() - start_time) < timeout:
            qapp.processEvents()
            time.sleep(0.01)
        
        assert len(fit_completed) > 0, "Fit did not complete within timeout"
        
        # ===================================================================
        # STEP 4: Verify fitted results of first spectrum
        # ===================================================================
        first_spectrum = vm.spectra[0]
        
        # Verify fitting was successful
        assert hasattr(first_spectrum, 'result_fit'), "Fitting did not produce result_fit"
        assert first_spectrum.result_fit.success, "Fitting failed"
        
        # Verify peak models were applied
        assert len(first_spectrum.peak_models) >= 1, \
            f"Expected at least 1 peak model, got {len(first_spectrum.peak_models)}"
        
        # Get fitted parameters
        best_values = first_spectrum.result_fit.best_values
        assert 'm01_ampli' in best_values, "Missing ampli parameter"
        assert 'm01_fwhm' in best_values, "Missing fwhm parameter"
        assert 'm01_x0' in best_values, "Missing x0 parameter"
        
        # Verify fitted values are reasonable
        ampli = best_values['m01_ampli']
        fwhm = best_values['m01_fwhm']
        x0 = best_values['m01_x0']
        
        assert 820 < ampli < 830, f"Ampli should be 821.233, got {ampli:.1f}"
        assert 3.2 < fwhm < 3.4, f"fwhm should be 3.3632, got {fwhm:.1f}"
        assert 520 <x0 < 520.3, f"x0 should be 520.221, got {x0:.1f}"
        
        # ===================================================================
        # STEP 5: Collect fit results and verify DataFrame
        # ===================================================================
        vm.collect_fit_results()
        
        # Verify fit results DataFrame
        assert vm.df_fit_results is not None, "Fit results DataFrame is None"
        assert len(vm.df_fit_results) == 1, \
            f"Expected 1 row in fit results, got {len(vm.df_fit_results)}"
        
        # Verify DataFrame has map-specific columns (Filename, X, Y)
        assert 'Filename' in vm.df_fit_results.columns
        assert 'X' in vm.df_fit_results.columns, "Missing X coordinate column"
        assert 'Y' in vm.df_fit_results.columns, "Missing Y coordinate column"
        assert 'x0_Si' in vm.df_fit_results.columns, "Missing x0_Si column"
        assert 'fwhm_Si' in vm.df_fit_results.columns, "Missing fwhm_Si column"
        assert 'ampli_Si' in vm.df_fit_results.columns, "Missing ampli_Si column"
        assert 'ampli_Si' in vm.df_fit_results.columns, "Missing ampli_Si column"
        
        # Verify the DataFrame contains fit parameters
        assert 'm01_ampli' in vm.df_fit_results.columns or any('ampli' in col for col in vm.df_fit_results.columns), \
            "Missing amplitude column in fit results"
        
        # ===================================================================
        # STEP 6: Save workspace
        # ===================================================================
        save_path = temp_workspace / "processed_map.maps"
        
        def mock_get_save_filename(*args, **kwargs):
            return str(save_path), ""
        
        monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_get_save_filename)
        vm.save_work()
        
        assert save_path.exists(), "Workspace file was not created"
        
        # ===================================================================
        # STEP 7: Reload workspace and verify persistence
        # ===================================================================
        vm2 = VMWorkspaceMaps(mock_settings)
        vm2.load_work(str(save_path))
        qtbot.wait(500)
        
        # Verify map persisted
        assert len(vm2.maps) == 1, f"Expected 1 map, got {len(vm2.maps)}"
        assert "2Dmap_Si" in vm2.maps
        
        # Verify spectra persisted
        assert len(vm2.spectra) > 0, "No spectra loaded from workspace"
        
        # Verify first spectrum fit results persisted
        loaded_first_spectrum = vm2.spectra[0]
        if hasattr(loaded_first_spectrum, 'result_fit') and loaded_first_spectrum.result_fit:
            assert loaded_first_spectrum.result_fit.success, "Loaded spectrum fit result not successful"
            assert len(loaded_first_spectrum.peak_models) >= 1, "Peak models not persisted"


class TestGraphsWorkflow:
    """Integration tests for complete graphs workflow."""
    
    def test_complete_graphs_workflow(self, qapp, mock_settings, dataframe_excel_file, temp_workspace, monkeypatch):
        """
        Test complete graphs workflow with real Excel file.
        
        Steps:
        1. Load real Excel file (data_inline.xlsx) with 2 sheets
        2. Verify both dataframes (sheet1, sheet2) are loaded
        3. Verify dimensions of each dataframe
        4. Select sheet1 and apply filter "Zone != Edge"
        5. Create box plot with X=Slot, Y=x0_Si, Z=Zone
        6. Create bar plot with same X, Y, Z
        7. Save workspace
        8. Reload workspace and verify consistency
        """
        if not dataframe_excel_file.exists():
            pytest.skip("Excel test file not available")
        
        vm = VMWorkspaceGraphs(mock_settings)
        
        # ===================================================================
        # STEP 1: Load real Excel file with 2 sheets
        # ===================================================================
        def mock_get_open_filenames(*args, **kwargs):
            return [str(dataframe_excel_file)], ""
        
        monkeypatch.setattr(QFileDialog, "getOpenFileNames", mock_get_open_filenames)
        vm.load_dataframes()
        
        # ===================================================================
        # STEP 2: Verify both dataframes are loaded
        # ===================================================================
        assert "dataset_Excel_sheet1" in vm.dataframes, "dataset_Excel_sheet1 not loaded from Excel file"
        assert "dataset_Excel_sheet2" in vm.dataframes, "dataset_Excel_sheet2 not loaded from Excel file"
        assert len(vm.dataframes) == 2, f"Expected 2 dataframes, got {len(vm.dataframes)}"
        
        # ===================================================================
        # STEP 3: Verify dimensions of each dataframe
        # ===================================================================
        sheet1_df = vm.dataframes["dataset_Excel_sheet1"]
        sheet2_df = vm.dataframes["dataset_Excel_sheet2"]
        
        assert len(sheet1_df) > 580, "sheet1 is empty"
        assert len(sheet1_df.columns) > 11, "sheet1 has no columns"
        assert len(sheet2_df) > 580, "sheet2 is empty"
        assert len(sheet2_df.columns) > 12, "sheet2 has no columns"
        
        # Verify required columns exist in sheet1
        assert 'Slot' in sheet1_df.columns, "Missing Slot column in sheet1"
        assert 'x0_Si' in sheet1_df.columns, "Missing x0_Si column in sheet1"
        assert 'Zone' in sheet1_df.columns, "Missing Zone column in sheet1"
        
        # ===================================================================
        # STEP 4: Select sheet1 and apply filter "Zone != Edge"
        # ===================================================================
        vm.select_dataframe("dataset_Excel_sheet1")
        
        filters = [
            {'expression': 'Zone != "Edge"', 'state': True}
        ]
        filtered_df = vm.apply_filters("dataset_Excel_sheet1", filters)
        
        assert len(filtered_df) < len(sheet1_df), "Filter did not reduce dataframe size"
        assert all(filtered_df['Zone'] != 'Edge'), "Filter did not exclude Edge zone"
        
        # ===================================================================
        # STEP 5: Create box plot with X=Slot, Y=x0_Si, Z=Zone
        # ===================================================================
        box_plot_config = {
            'df_name': 'dataset_Excel_sheet1',
            'plot_style': 'box',
            'x': 'Slot',
            'y': ['x0_Si'],
            'z': 'Zone',
            'xlabel': 'Slot',
            'ylabel': 'x0_Si',
            'filters': filters
        }
        box_graph = vm.create_graph(box_plot_config)
        
        assert box_graph is not None, "Box graph creation failed"
        assert box_graph.graph_id in vm.graphs
        
        # ===================================================================
        # STEP 6: Create bar plot with same X, Y, Z
        # ===================================================================
        bar_plot_config = {
            'df_name': 'dataset_Excel_sheet1',
            'plot_style': 'bar',
            'x': 'Slot',
            'y': ['x0_Si'],
            'z': 'Zone',
            'xlabel': 'Slot',
            'ylabel': 'x0_Si',
            'filters': filters
        }
        bar_graph = vm.create_graph(bar_plot_config)
        
        assert bar_graph is not None, "Bar graph creation failed"
        assert bar_graph.graph_id in vm.graphs
        assert len(vm.graphs) == 2, f"Expected 2 graphs, got {len(vm.graphs)}"
        
        # ===================================================================
        # STEP 7: Save workspace
        # ===================================================================
        save_path = temp_workspace / "graphs_with_excel.graphs"
        
        def mock_get_save_filename(*args, **kwargs):
            return str(save_path), ""

        monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_get_save_filename)
        vm.save_workspace()
        
        assert save_path.exists(), "Workspace file was not created"
        
        # ===================================================================
        # STEP 8: Reload workspace and verify consistency
        # ===================================================================
        vm2 = VMWorkspaceGraphs(mock_settings)
        vm2.load_workspace(str(save_path))
        
        # Verify dataframes persisted
        assert "dataset_Excel_sheet1" in vm2.dataframes, "dataset_Excel_sheet1 not loaded from workspace"
        assert "dataset_Excel_sheet2" in vm2.dataframes, "dataset_Excel_sheet2 not loaded from workspace"
        assert len(vm2.dataframes) == 2, f"Expected 2 dataframes, got {len(vm2.dataframes)}"
        
        pd.testing.assert_frame_equal(vm2.dataframes["dataset_Excel_sheet1"], sheet1_df)
        pd.testing.assert_frame_equal(vm2.dataframes["dataset_Excel_sheet2"], sheet2_df)
        
        # Verify graphs persisted
        assert len(vm2.graphs) == 2, f"Expected 2 graphs, got {len(vm2.graphs)}"
        
        loaded_graphs = list(vm2.graphs.values())
        box_graph_loaded = next((g for g in loaded_graphs if g.plot_style == 'box'), None)
        assert box_graph_loaded is not None, "Box graph not found in loaded workspace"
        assert box_graph_loaded.x == 'Slot'
        assert box_graph_loaded.y == ['x0_Si']
        assert box_graph_loaded.z == 'Zone'
        
        bar_graph_loaded = next((g for g in loaded_graphs if g.plot_style == 'bar'), None)
        assert bar_graph_loaded is not None, "Bar graph not found in loaded workspace"
        assert bar_graph_loaded.x == 'Slot'
        assert bar_graph_loaded.y == ['x0_Si']
        assert bar_graph_loaded.z == 'Zone'

class TestSaveLoadWorkspace:
    """Test save and load a save workpsace files (.spectra, .maps, .graphs)"""
    
    
    def test_load_saved_spectra_workspace(self, qapp, mock_settings, saved_spectra_workspace):
        """
        Test loading saved spectra workspace.
        
        Verifies:
        - 26 spectra loaded
        - First spectrum fit results: x0=414.126, fwhm=5.68787, ampli=2696.76
        """
        if not saved_spectra_workspace.exists():
            pytest.skip("Saved spectra workspace not available")
        
        vm = VMWorkspaceSpectra(mock_settings)
        vm.load_work(str(saved_spectra_workspace))
        
        # Verify number of loaded spectra
        assert len(vm.spectra) == 26, f"Expected 26 spectra, got {len(vm.spectra)}"
        
        # Select first spectrum
        vm.set_selected_indices([0])
        
        # Get first spectrum
        first_spectrum = vm.spectra[0]
        assert hasattr(first_spectrum, 'x'), "First spectrum missing x data"
        assert hasattr(first_spectrum, 'y'), "First spectrum missing y data"
        assert len(first_spectrum.x) > 100, "First spectrum x data is empty"
        assert len(first_spectrum.y) > 100, "First spectrum y data is empty"
        
        # Perform fit on first spectrum
        first_spectrum.fit()
        
        # Verify fit results
        assert hasattr(first_spectrum, 'result_fit'), "Spectrum has no result_fit after fitting"
        assert first_spectrum.result_fit is not None, "result_fit is None after fitting"
        assert first_spectrum.result_fit.success, "Fit was not successful"
        
        # Get fit parameters
        params = first_spectrum.result_fit.params
        
        # Check x0 parameter
        x0_key = next((k for k in params.keys() if 'x0' in k.lower()), None)
        assert x0_key is not None, "No x0 parameter found in fit results"
        x0_value = params[x0_key].value
        assert abs(x0_value - 414.126) < 0.1, \
            f"Expected x0 ~414.126, got {x0_value:.3f}"
        
        # Check fwhm parameter
        fwhm_key = next((k for k in params.keys() if 'fwhm' in k.lower()), None)
        assert fwhm_key is not None, "No fwhm parameter found in fit results"
        fwhm_value = params[fwhm_key].value
        assert abs(fwhm_value - 5.68787) < 0.1, \
            f"Expected fwhm ~5.68787, got {fwhm_value:.5f}"
        
        # Check ampli parameter
        ampli_key = next((k for k in params.keys() if 'ampli' in k.lower()), None)
        assert ampli_key is not None, "No ampli parameter found in fit results"
        ampli_value = params[ampli_key].value
        assert abs(ampli_value - 2696.76) < 10.0, \
            f"Expected ampli ~2696.76, got {ampli_value:.2f}"
    
    def test_load_saved_maps_workspace(self, qapp, qtbot, mock_settings, saved_maps_workspace, monkeypatch):
        """
        Test loading saved maps workspace.
        
        Verifies:
        - 4 maps loaded
        - First map is wafer4_process1 with 49 spectra
        """
        if not saved_maps_workspace.exists():
            pytest.skip("Saved maps workspace not available")
        
        # Mock QMessageBox
        mock_msgbox = MagicMock()
        monkeypatch.setattr(QMessageBox, "critical", mock_msgbox)
        
        vm = VMWorkspaceMaps(mock_settings)
        vm.load_work(str(saved_maps_workspace))
        qtbot.wait(300)
        
        # Verify number of loaded maps
        assert len(vm.maps) == 4, f"Expected 4 maps, got {len(vm.maps)}"
        
        # Get map names
        map_names = list(vm.maps.keys())
        
        # Verify wafer4_process1 is in the maps
        assert "wafer4_process1" in map_names, \
            f"wafer4_process1 not found in maps: {map_names}"
        
        # Select wafer4_process1
        vm.select_map("wafer4_process1")
        qtbot.wait(100)
        
        # Verify wafer4_process1 has 49 spectra
        wafer4_map = vm.maps["wafer4_process1"]
        assert len(wafer4_map) == 49, \
            f"Expected 49 entries in wafer4_process1 map, got {len(wafer4_map)}"
    
    def test_load_saved_graphs_workspace(self, qapp, mock_settings, saved_graphs_workspace):
        """
        Test loading saved graphs workspace.
        
        Verifies:
        - 2 dataframes loaded
        - 7 graphs loaded
        """
        if not saved_graphs_workspace.exists():
            pytest.skip("Saved graphs workspace not available")
        
        vm = VMWorkspaceGraphs(mock_settings)
        vm.load_workspace(str(saved_graphs_workspace))
        
        # Verify number of loaded dataframes
        assert len(vm.dataframes) == 2, \
            f"Expected 2 dataframes, got {len(vm.dataframes)}"
        
        # Verify number of loaded graphs
        assert len(vm.graphs) == 7, \
            f"Expected 7 graphs, got {len(vm.graphs)}"
    


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

        
        spectrum = MSpectrum()
        spectrum.fname = "single_point"
        spectrum.x0 = np.array([100.0])
        spectrum.y0 = np.array([200.0])
        spectrum.x = spectrum.x0.copy()
        spectrum.y = spectrum.y0.copy()
        
        vm.spectra.add(spectrum)
        
        # Operations should handle gracefully
        assert len(vm.spectra) == 1



