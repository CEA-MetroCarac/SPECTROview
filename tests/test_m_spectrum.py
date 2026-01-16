"""
Tests for model/m_spectrum.py - Single spectrum data model

Tests cover:
- Spectrum initialization with custom attributes
- reinit() method
- preprocess() method
- apply_xcorrection() and undo_xcorrection()
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from spectroview.model.m_spectrum import MSpectrum


class TestMSpectrumInitialization:
    """Tests for MSpectrum initialization."""
    
    def test_create_empty_spectrum(self):
        """Test creating an empty MSpectrum object."""
        spectrum = MSpectrum()
        
        # Verify custom attributes are initialized
        assert spectrum.label is None
        assert spectrum.color is None
        assert spectrum.xcorrection_value == 0
        assert spectrum.source_path is None
        assert spectrum.is_active is True
    
    def test_create_spectrum_with_data(self):
        """Test creating MSpectrum with data."""
        x = np.array([100, 200, 300, 400])
        y = np.array([10, 20, 15, 25])
        
        spectrum = MSpectrum()
        spectrum.x0 = x
        spectrum.y0 = y
        spectrum.x = x.copy()
        spectrum.y = y.copy()
        
        # Verify data is stored
        np.testing.assert_array_equal(spectrum.x0, x)
        np.testing.assert_array_equal(spectrum.y0, y)
        np.testing.assert_array_equal(spectrum.x, x)
        np.testing.assert_array_equal(spectrum.y, y)
    
    def test_custom_attributes(self):
        """Test setting custom attributes."""
        spectrum = MSpectrum()
        
        spectrum.label = "My Spectrum"
        spectrum.color = "#FF0000"
        spectrum.is_active = False
        
        assert spectrum.label == "My Spectrum"
        assert spectrum.color == "#FF0000"
        assert spectrum.is_active is False


class TestMSpectrumReinit:
    """Tests for reinit() method."""
    
    def test_reinit_resets_to_original_data(self, sample_spectrum):
        """Test that reinit() restores original x and y data."""
        # Modify the spectrum
        sample_spectrum.x = sample_spectrum.x * 2
        sample_spectrum.y = sample_spectrum.y * 2
        sample_spectrum.label = "Modified"
        sample_spectrum.color = "#00FF00"
        
        # Reinitialize
        sample_spectrum.reinit()
        
        # Verify x and y are restored to x0 and y0
        np.testing.assert_array_equal(sample_spectrum.x, sample_spectrum.x0)
        np.testing.assert_array_equal(sample_spectrum.y, sample_spectrum.y0)
        
        # Verify label and color are reset
        assert sample_spectrum.label is None
        assert sample_spectrum.color is None
    
    def test_reinit_clears_baseline(self, sample_spectrum):
        """Test that reinit() clears baseline points."""
        # Add baseline points
        sample_spectrum.baseline.add_point(150, 100)
        sample_spectrum.baseline.add_point(350, 100)
        
        # Reinitialize
        sample_spectrum.reinit()
        
        # Verify baseline is reset
        assert sample_spectrum.baseline.mode == "Linear"
        # fitspy baseline.points is [xs, ys] structure, after reinit becomes [[], []]
        if sample_spectrum.baseline.points is not None:
            xs, ys = sample_spectrum.baseline.points
            assert len(xs) == 0 and len(ys) == 0
    
    def test_reinit_removes_models(self, sample_spectrum):
        """Test that reinit() removes peak models."""
        # This would normally add peak models via fitspy
        # For now, just verify the method runs without error
        sample_spectrum.reinit()
        
        # Verify peak_models attribute exists (fitspy behavior)
        assert hasattr(sample_spectrum, 'peak_models')


class TestMSpectrumXCorrection:
    """Tests for X-correction (peak position correction)."""
    
    def test_apply_xcorrection(self):
        """Test applying X-correction shifts x-values."""
        spectrum = MSpectrum()
        x_original = np.array([100, 200, 300, 400])
        spectrum.x0 = x_original.copy()
        spectrum.x = x_original.copy()
        
        # Apply correction of +10
        spectrum.apply_xcorrection(10)
        
        # Verify x-values are shifted
        expected_x = x_original + 10
        np.testing.assert_array_equal(spectrum.x0, expected_x)
        np.testing.assert_array_equal(spectrum.x, expected_x)
        
        # Verify correction value is stored
        assert spectrum.xcorrection_value == 10
    
    def test_apply_negative_xcorrection(self):
        """Test applying negative X-correction."""
        spectrum = MSpectrum()
        x_original = np.array([100, 200, 300, 400])
        spectrum.x0 = x_original.copy()
        spectrum.x = x_original.copy()
        
        # Apply correction of -5
        spectrum.apply_xcorrection(-5)
        
        # Verify x-values are shifted
        expected_x = x_original - 5
        np.testing.assert_array_equal(spectrum.x0, expected_x)
        assert spectrum.xcorrection_value == -5
    
    def test_apply_xcorrection_overwrites_existing(self):
        """Test that applying new correction undoes previous one."""
        spectrum = MSpectrum()
        x_original = np.array([100, 200, 300, 400])
        spectrum.x0 = x_original.copy()
        spectrum.x = x_original.copy()
        
        # Apply first correction
        spectrum.apply_xcorrection(10)
        
        # Apply second correction (should undo first)
        spectrum.apply_xcorrection(20)
        
        # Verify only second correction is applied
        expected_x = x_original + 20
        np.testing.assert_array_equal(spectrum.x0, expected_x)
        assert spectrum.xcorrection_value == 20
    
    def test_undo_xcorrection(self):
        """Test undoing X-correction restores original values."""
        spectrum = MSpectrum()
        x_original = np.array([100, 200, 300, 400])
        spectrum.x0 = x_original.copy()
        spectrum.x = x_original.copy()
        
        # Apply correction
        spectrum.apply_xcorrection(10)
        
        # Undo correction
        spectrum.undo_xcorrection()
        
        # Verify x-values are restored
        np.testing.assert_array_equal(spectrum.x0, x_original)
        np.testing.assert_array_equal(spectrum.x, x_original)
        assert spectrum.xcorrection_value == 0
    
    def test_undo_xcorrection_when_none_applied(self):
        """Test undoing correction when none was applied."""
        spectrum = MSpectrum()
        x_original = np.array([100, 200, 300, 400])
        spectrum.x0 = x_original.copy()
        spectrum.x = x_original.copy()
        
        # Undo without applying correction
        spectrum.undo_xcorrection()
        
        # Verify x-values are unchanged
        np.testing.assert_array_equal(spectrum.x0, x_original)
        assert spectrum.xcorrection_value == 0


class TestMSpectrumPreprocess:
    """Tests for preprocess() method."""
    
    @patch.object(MSpectrum, 'load_profile')
    @patch.object(MSpectrum, 'apply_range')
    @patch.object(MSpectrum, 'eval_baseline')
    @patch.object(MSpectrum, 'subtract_baseline')
    @patch.object(MSpectrum, 'normalization')
    def test_preprocess_calls_all_steps(self, mock_norm, mock_sub_bl, 
                                        mock_eval_bl, mock_apply_range, 
                                        mock_load_profile):
        """Test that preprocess() calls all necessary methods."""
        spectrum = MSpectrum()
        spectrum.fname = "test_spectrum"
        
        # Call preprocess
        spectrum.preprocess()
        
        # Verify all methods were called
        mock_load_profile.assert_called_once_with("test_spectrum")
        mock_apply_range.assert_called_once()
        mock_eval_bl.assert_called_once()
        mock_sub_bl.assert_called_once()
        mock_norm.assert_called_once()
