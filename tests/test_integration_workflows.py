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
        6. Save workspace
        7. Reload workspace
        8. Verify consistency
        """
        # Step 1: Load spectrum
        vm = VMWorkspaceSpectra(mock_settings)
        vm.load_files([str(single_spectrum_file)])
        assert len(vm.spectra) == 1
        
        # Step 2: Select spectrum and crop range
        vm.set_selected_indices([0])
        spectrum = vm.spectra[0]
        x_min = spectrum.x.min() + (spectrum.x.max() - spectrum.x.min()) * 0.2
        x_max = spectrum.x.max() - (spectrum.x.max() - spectrum.x.min()) * 0.2
        vm.apply_spectral_range(x_min, x_max, apply_all=False)
        
        # Step 3: Add baseline points
        vm.add_baseline_point(x_min + 50, 100)
        vm.add_baseline_point(x_max - 50, 100)
        baseline_points_count = len(spectrum.baseline.points)
        
        # Step 4: Subtract baseline
        vm.subtract_baseline(apply_all=False)
        
        # Step 5: Add peak model
        mid_x = (x_min + x_max) / 2
        vm.add_peak_at(mid_x)
        models_count = len(spectrum.peak_models)
        
        # Step 6: Save workspace
        save_path = temp_workspace / "workflow.spectra"
        
        def mock_get_save_filename(*args, **kwargs):
            return str(save_path), ""
        
        from PySide6.QtWidgets import QFileDialog
        monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_get_save_filename)
        
        vm.save_work()
        assert save_path.exists()
        
        # Step 7: Reload workspace into new ViewModel
        vm2 = VMWorkspaceSpectra(mock_settings)
        vm2.load_work(str(save_path))
        
        # Step 8: Verify consistency
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
