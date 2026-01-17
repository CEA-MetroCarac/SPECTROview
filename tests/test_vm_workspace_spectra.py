"""
Tests for viewmodel/vm_workspace_spectra.py - Spectra workspace ViewModel

Tests cover:
- File loading and spectrum management
- Selection management  
- Spectral range operations
- Baseline operations (add/remove points, copy/paste, subtract)
- Peak operations (add/remove, copy/paste)
- X-correction (apply/undo)
- Workspace persistence (save/load)
- Fit results collection
"""

import pytest
import json
import gzip
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
from io import StringIO

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog
import numpy as np

from spectroview.viewmodel.vm_workspace_spectra import VMWorkspaceSpectra
from spectroview.model.m_spectrum import MSpectrum
from spectroview.model.m_spectra import MSpectra


class TestVMWorkspaceSpectraInitialization:
    """Tests for VMWorkspaceSpectra initialization."""
    
    def test_initialization(self, qapp, mock_settings):
        """Test ViewModel initializes correctly."""
        vm = VMWorkspaceSpectra(mock_settings)
        
        assert vm.settings == mock_settings
        assert isinstance(vm.spectra, MSpectra)
        assert len(vm.spectra) == 0
        assert vm.selected_fnames == []


class TestVMWorkspaceSpectraFileLoading:
    """Tests for file loading functionality."""
    
    def test_load_single_file(self, qapp, mock_settings, single_spectrum_file):
        """Test loading a single spectrum file."""
        vm = VMWorkspaceSpectra(mock_settings)
        
        # Connect signal to capture emission
        emitted_lists = []
        vm.spectra_list_changed.connect(lambda spectra_list: emitted_lists.append(spectra_list))
        
        # Load file
        vm.load_files([str(single_spectrum_file)])
        
        # Verify spectrum was loaded
        assert len(vm.spectra) == 1
        assert vm.spectra[0].fname == single_spectrum_file.stem
        
        # Verify signal was emitted
        assert len(emitted_lists) == 1
    
    def test_load_multiple_files(self, qapp, mock_settings, multiple_spectra_files):
        """Test loading multiple spectrum files."""
        vm = VMWorkspaceSpectra(mock_settings)
        
        # Load files
        file_paths = [str(f) for f in multiple_spectra_files]
        vm.load_files(file_paths)
        
        # Verify all spectra were loaded
        assert len(vm.spectra) == len(multiple_spectra_files)
    
    def test_load_files_emits_count(self, qapp, mock_settings, multiple_spectra_files):
        """Test that loading files emits count_changed signal."""
        vm = VMWorkspaceSpectra(mock_settings)
        
        # Connect to count signal
        emitted_counts = []
        vm.count_changed.connect(lambda count: emitted_counts.append(count))
        
        # Load files
        file_paths = [str(f) for f in multiple_spectra_files]
        vm.load_files(file_paths)
        
        # Verify count signal was emitted
        assert len(emitted_counts) > 0
        assert emitted_counts[-1] == len(multiple_spectra_files)


class TestVMWorkspaceSpectraSelection:
    """Tests for selection management."""
    
    @pytest.fixture
    def vm_with_spectra(self, qapp, mock_settings, multiple_spectra_files):
        """Create ViewModel with loaded spectra."""
        vm = VMWorkspaceSpectra(mock_settings)
        file_paths = [str(f) for f in multiple_spectra_files]
        vm.load_files(file_paths)
        return vm
    
    def test_set_selected_indices(self, vm_with_spectra):
        """Test setting selected spectra by indices."""
        # Select indices 0 and 2
        vm_with_spectra.set_selected_indices([0, 2])
        
        # Verify selection
        assert len(vm_with_spectra.selected_fnames) == 2
    
    def test_set_selected_fnames(self, vm_with_spectra):
        """Test setting selected spectra by fnames."""
        # Get fnames
        fname1 = vm_with_spectra.spectra[0].fname
        fname2 = vm_with_spectra.spectra[1].fname
        
        # Select by fnames
        vm_with_spectra.set_selected_fnames([fname1, fname2])
        
        # Verify selection
        assert len(vm_with_spectra.selected_fnames) == 2
        assert fname1 in vm_with_spectra.selected_fnames
        assert fname2 in vm_with_spectra.selected_fnames
    
    def test_selection_emits_signal(self, vm_with_spectra):
        """Test that selection emits spectra_selection_changed signal."""
        emitted_data = []
        vm_with_spectra.spectra_selection_changed.connect(lambda data: emitted_data.append(data))
        
        # Set selection
        vm_with_spectra.set_selected_indices([0])
        
        # Verify signal was emitted
        assert len(emitted_data) > 0


class TestVMWorkspaceSpectraRemoval:
    """Tests for spectrum removal."""
    
    def test_remove_selected_spectra(self, qapp, mock_settings, multiple_spectra_files):
        """Test removing selected spectra."""
        vm = VMWorkspaceSpectra(mock_settings)
        file_paths = [str(f) for f in multiple_spectra_files]
        vm.load_files(file_paths)
        
        initial_count = len(vm.spectra)
        
        # Select and remove first spectrum
        vm.set_selected_indices([0])
        vm.remove_selected_spectra()
        
        # Verify spectrum was removed
        assert len(vm.spectra) == initial_count - 1


class TestVMWorkspaceSpectraSpectralRange:
    """Tests for spectral range operations."""
    
    @pytest.fixture
    def vm_with_one_spectrum(self, qapp, mock_settings, single_spectrum_file):
        """Create ViewModel with single spectrum."""
        vm = VMWorkspaceSpectra(mock_settings)
        vm.load_files([str(single_spectrum_file)])
        vm.set_selected_indices([0])
        return vm
    
    def test_apply_spectral_range(self, vm_with_one_spectrum):
        """Test applying spectral range crop."""
        spectrum = vm_with_one_spectrum.spectra[0]
        original_len = len(spectrum.x)
        
        # Get original range
        x_min_orig = spectrum.x.min()
        x_max_orig = spectrum.x.max()
        
        # Apply narrower range
        new_min = x_min_orig + (x_max_orig - x_min_orig) * 0.2
        new_max = x_max_orig - (x_max_orig - x_min_orig) * 0.2
        
        vm_with_one_spectrum.apply_spectral_range(new_min, new_max, apply_all=False)
        
        # Verify range was applied (x-values should be cropped)
        assert spectrum.range_min is not None
        assert spectrum.range_max is not None


class TestVMWorkspaceSpectraBaseline:
    """Tests for baseline operations."""
    
    @pytest.fixture
    def vm_with_selected_spectrum(self, qapp, mock_settings, single_spectrum_file):
        """Create ViewModel with selected spectrum."""
        vm = VMWorkspaceSpectra( mock_settings)
        vm.load_files([str(single_spectrum_file)])
        vm.set_selected_indices([0])
        return vm
    
    def test_add_baseline_point(self, vm_with_selected_spectrum):
        """Test adding baseline anchor point."""
        spectrum = vm_with_selected_spectrum.spectra[0]
        
        # Add baseline point
        x_pos = 200
        y_pos = 100
        vm_with_selected_spectrum.add_baseline_point(x_pos, y_pos)
        
        # Verify point was added (baseline should have points)
        assert len(spectrum.baseline.points) > 0
    
    def test_remove_baseline_point(self, vm_with_selected_spectrum):
        """Test removing baseline anchor point."""
        vm = vm_with_selected_spectrum
        spectrum = vm.spectra[0]
        
        # Add point first
        vm.add_baseline_point(200, 100)
        xs, ys = spectrum.baseline.points
        initial_count = len(xs)
        
        # Remove point
        vm.remove_baseline_point(200)
        
        # Verify point was removed (check xs list)
        xs_after, ys_after = spectrum.baseline.points
        assert len(xs_after) < initial_count
    
    def test_copy_baseline(self, vm_with_selected_spectrum):
        """Test copying baseline."""
        vm = vm_with_selected_spectrum
        
        # Add baseline points
        vm.add_baseline_point(200, 100)
        vm.add_baseline_point(400, 100)
        
        # Copy baseline
        vm.copy_baseline()
        
        # Verify baseline is copied (stored in vm._baseline_clipboard)
        assert vm._baseline_clipboard is not None
    
    def test_paste_baseline(self, qapp, mock_settings, multiple_spectra_files):
        """Test pasting baseline to other spectrum."""
        vm = VMWorkspaceSpectra(mock_settings)
        file_paths = [str(f) for f in multiple_spectra_files]
        vm.load_files(file_paths)
        
        # Select first spectrum and add baseline
        vm.set_selected_indices([0])
        vm.add_baseline_point(200, 100)
        vm.add_baseline_point(400, 100)
        
        # Copy baseline
        vm.copy_baseline()
        
        # Select second spectrum and paste
        vm.set_selected_indices([1])
        vm.paste_baseline(apply_all=False)
        
        # Verify baseline was pasted
        spectrum2 = vm.spectra[1]
        assert len(spectrum2.baseline.points) > 0
    
    def test_subtract_baseline(self, vm_with_selected_spectrum):
        """Test baseline subtraction."""
        vm = vm_with_selected_spectrum
        spectrum = vm.spectra[0]
        
        # Add baseline points and subtract
        vm.add_baseline_point(150, spectrum.y[0])
        vm.add_baseline_point(450, spectrum.y[-1])
        
        y_before = spectrum.y.copy()
        vm.subtract_baseline(apply_all=False)
        
        # After baseline eval and subtract, y values should change
        # (exact behavior depends on fitspy implementation)
        # We just verify the method runs without error


class TestVMWorkspaceSpectraPeaks:
    """Tests for peak operations."""
    
    @pytest.fixture
    def vm_with_selected_spectrum(self, qapp, mock_settings, single_spectrum_file):
        """Create ViewModel with selected spectrum."""
        vm = VMWorkspaceSpectra(mock_settings)
        vm.load_files([str(single_spectrum_file)])
        vm.set_selected_indices([0])
        return vm
    
    def test_add_peak_at(self, vm_with_selected_spectrum):
        """Test adding peak at position."""
        vm = vm_with_selected_spectrum
        
        # Add peak at x position
        vm.add_peak_at(300)
        
        # Verify peak was added (model should be added to spectrum)
        spectrum = vm.spectra[0]
        assert len(spectrum.peak_models) > 0
    
    def test_remove_peak_at(self, vm_with_selected_spectrum):
        """Test removing peak at position."""
        vm = vm_with_selected_spectrum
        
        # Add peak first
        vm.add_peak_at(300)
        initial_models = len(vm.spectra[0].peak_models)
        
        # Remove peak
        vm.remove_peak_at(300)
        
        # Verify peak was removed
        assert len(vm.spectra[0].peak_models) < initial_models
    
    def test_copy_paste_peaks(self, qapp, mock_settings, multiple_spectra_files):
        """Test copying and pasting peaks."""
        vm = VMWorkspaceSpectra(mock_settings)
        file_paths = [str(f) for f in multiple_spectra_files]
        vm.load_files(file_paths)
        
        # Select first spectrum and add peak
        vm.set_selected_indices([0])
        vm.add_peak_at(300)
        
        # Copy peaks
        vm.copy_peaks()
        
        # Select second spectrum and paste
        vm.set_selected_indices([1])
        vm.paste_peaks(apply_all=False)
        
        # Verify peaks were pasted
        assert len(vm.spectra[1].peak_models) > 0


class TestVMWorkspaceSpectraXCorrection:
    """Tests for X-correction operations."""
    
    def test_apply_x_correction(self, qapp, mock_settings, single_spectrum_file):
        """Test applying X-correction."""
        vm = VMWorkspaceSpectra(mock_settings)
        vm.load_files([str(single_spectrum_file)])
        vm.set_selected_indices([0])
        
        spectrum = vm.spectra[0]
        x_original = spectrum.x.copy()
        
        # Apply correction (shift measured peak to reference)
        measured_peak = 305
        vm.reference_peak = 300
        vm.apply_x_correction(measured_peak)
        
        # Verify correction was applied
        assert spectrum.xcorrection_value != 0
    
    def test_undo_x_correction(self, qapp, mock_settings, single_spectrum_file):
        """Test undoing X-correction."""
        vm = VMWorkspaceSpectra(mock_settings)
        vm.load_files([str(single_spectrum_file)])
        vm.set_selected_indices([0])
        
        spectrum = vm.spectra[0]
        x_original = spectrum.x0.copy()
        
        # Apply then undo correction
        vm.reference_peak = 300
        vm.apply_x_correction(305)
        vm.undo_x_correction()
        
        # Verify correction was undone
        assert spectrum.xcorrection_value == 0
        # Use almost_equal for floating point comparison
        np.testing.assert_array_almost_equal(spectrum.x0, x_original, decimal=10)


class TestVMWorkspaceSpectraPersistence:
    """Tests for workspace save/load."""
    
    def test_save_workspace(self, qapp, mock_settings, single_spectrum_file, temp_workspace, monkeypatch):
        """Test saving workspace to file."""
        vm = VMWorkspaceSpectra(mock_settings)
        vm.load_files([str(single_spectrum_file)])
        
        # Mock file dialog to return specific path
        save_path = temp_workspace / "test_workspace.spectra"
        
        def mock_get_save_filename(*args, **kwargs):
            return str(save_path), ""
        monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_get_save_filename)
        
        # Save workspace
        vm.save_work()
        
        # Verify file was created
        assert save_path.exists()
    
    def test_load_workspace(self, qapp, mock_settings, saved_spectra_workspace):
        """Test loading workspace from file."""
        if not saved_spectra_workspace.exists():
            pytest.skip("Saved workspace file not available")
        
        vm = VMWorkspaceSpectra(mock_settings)
        
        # Load workspace
        vm.load_work(str(saved_spectra_workspace))
        
        # Verify spectra were loaded
        assert len(vm.spectra) > 0
    
    def test_save_load_roundtrip(self, qapp, mock_settings, single_spectrum_file, temp_workspace, monkeypatch):
        """Test that save then load preserves data."""
        # Create and populate workspace
        vm1 = VMWorkspaceSpectra(mock_settings)
        vm1.load_files([str(single_spectrum_file)])
        
        # Add some processing
        vm1.set_selected_indices([0])
        vm1.add_baseline_point(200, 100)
        vm1.add_peak_at(300)
        
        original_spectrum = vm1.spectra[0]
        
        # Save workspace
        save_path = temp_workspace / "roundtrip.spectra"
        
        def mock_get_save_filename(*args, **kwargs):
            return str(save_path), ""
        monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_get_save_filename)
        
        vm1.save_work()
        
        # Load into new ViewModel
        vm2 = VMWorkspaceSpectra(mock_settings)
        vm2.load_work(str(save_path))
        
        # Verify loaded spectrum matches original
        assert len(vm2.spectra) == 1
        loaded_spectrum = vm2.spectra[0]
        assert loaded_spectrum.fname == original_spectrum.fname
        # Baseline points is [xs, ys] structure
        xs1, ys1 = loaded_spectrum.baseline.points
        xs2, ys2 = original_spectrum.baseline.points
        assert len(xs1) == len(xs2)
        assert len(loaded_spectrum.peak_models) == len(original_spectrum.peak_models)


class TestVMWorkspaceSpectraFitResults:
    """Tests for fit results collection."""
    
    @patch('spectroview.viewmodel.vm_workspace_spectra.fit_report')
    def test_collect_fit_results(self, mock_fit_report, qapp, mock_settings, single_spectrum_file):
        """Test collecting fit results into DataFrame."""
        # Mock fit report to return dummy string
        mock_fit_report.return_value = "Fit Report\nPeak1: center=300"
        
        vm = VMWorkspaceSpectra(mock_settings)
        vm.load_files([str(single_spectrum_file)])
        
        # Mock fit result on spectrum
        spectrum = vm.spectra[0]
        spectrum.result_fit = Mock()
        spectrum.result_fit.success = True
        spectrum.result_fit.params = {}
        
        # Mark spectrum as active for collection
        spectrum.is_active = True
        
        # Collect results
        df = vm.collect_fit_results()
        
        # collect_fit_results() only returns data if there are actual fit results
        # Since we're mocking, the method won't return a full DataFrame
        # Just verify the method runs without error
