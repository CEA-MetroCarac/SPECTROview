"""
Tests for model/m_io.py - File loading functions

Tests cover:
- Loading single spectrum files (.txt, .csv)
- Loading 2D map files (.txt, .csv)
- Loading DataFrame files (.xlsx, .csv)
- Delimiter detection for TXT files
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from spectroview.model.m_io import (
    load_spectrum_file,
    load_map_file,
    load_dataframe_file
)
from spectroview.model.m_spectrum import MSpectrum


class TestLoadSpectrumFile:
    """Tests for load_spectrum_file function."""
    
    def test_load_txt_spectrum(self, single_spectrum_file):
        """Test loading a .txt spectrum file."""
        spectrum = load_spectrum_file(single_spectrum_file)
        
        # Verify it returns MSpectrum object
        assert isinstance(spectrum, MSpectrum)
        
        # Verify data is loaded
        assert len(spectrum.x0) > 0
        assert len(spectrum.y0) > 0
        assert len(spectrum.x0) == len(spectrum.y0)
        
        # Verify fname is set correctly
        assert spectrum.fname == single_spectrum_file.stem
        
        # Verify source_path is set
        assert spectrum.source_path == str(single_spectrum_file.resolve())
        
        # Verify x and y are copies of x0 and y0
        np.testing.assert_array_equal(spectrum.x, spectrum.x0)
        np.testing.assert_array_equal(spectrum.y, spectrum.y0)
        
        # Verify baseline mode is set
        assert spectrum.baseline.mode == "Linear"
    
    def test_load_multiple_spectra(self, multiple_spectra_files):
        """Test loading multiple spectrum files."""
        spectra = [load_spectrum_file(path) for path in multiple_spectra_files]
        
        assert len(spectra) == len(multiple_spectra_files)
        
        # Verify each spectrum is loaded correctly
        for spectrum, file_path in zip(spectra, multiple_spectra_files):
            assert isinstance(spectrum, MSpectrum)
            assert spectrum.fname == file_path.stem
            assert len(spectrum.x0) > 0
    
    def test_spectrum_data_is_sorted(self, single_spectrum_file):
        """Test that spectrum x-values are sorted."""
        spectrum = load_spectrum_file(single_spectrum_file)
        
        # Verify x-values are sorted
        assert np.all(spectrum.x[:-1] <= spectrum.x[1:])
    
    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file raises error."""
        nonexistent = Path("nonexistent_file.txt")
        
        with pytest.raises(FileNotFoundError):
            load_spectrum_file(nonexistent)
    
    def test_load_unsupported_extension(self, tmp_path):
        """Test loading file with unsupported extension raises error."""
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_text("100 200\n200 300\n")
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_spectrum_file(unsupported_file)


class TestLoadMapFile:
    """Tests for load_map_file function."""
    
    def test_load_txt_map(self, map_2d_file):
        """Test loading a .txt 2D map file."""
        map_df = load_map_file(map_2d_file)
        
        # Verify it returns DataFrame
        assert isinstance(map_df, pd.DataFrame)
        
        # Verify it has X, Y columns
        assert 'X' in map_df.columns
        assert 'Y' in map_df.columns
        
        # Verify it has wavelength columns (more than just X, Y)
        assert len(map_df.columns) > 2
        
        # Verify data is not empty
        assert len(map_df) > 0
    
    def test_load_csv_wafer_file(self, wafer_file):
        """Test loading a .csv wafer file."""
        map_df = load_map_file(wafer_file)
        
        # Verify it returns DataFrame
        assert isinstance(map_df, pd.DataFrame)
        
        # Verify it has X, Y columns
        assert 'X' in map_df.columns
        assert 'Y' in map_df.columns
        
        # Verify wavelength columns are present
        assert len(map_df.columns) > 2
    
    def test_map_has_numeric_wavelength_columns(self, map_2d_file):
        """Test that wavelength columns can be parsed as floats."""
        map_df = load_map_file(map_2d_file)
        
        # Get wavelength columns (all except X, Y)
        wavelength_cols = [col for col in map_df.columns if col not in ['X', 'Y']]
        
        # Verify all wavelength columns can be converted to float
        for col in wavelength_cols:
            try:
                float(col)
            except ValueError:
                pytest.fail(f"Wavelength column '{col}' cannot be converted to float")
    
    def test_load_unsupported_map_extension(self, tmp_path):
        """Test loading map with unsupported extension raises error."""
        unsupported_file = tmp_path / "map.xyz"
        unsupported_file.write_text("X\tY\t100\t200\n1\t1\t50\t60\n")
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_map_file(unsupported_file)


class TestLoadDataFrameFile:
    """Tests for load_dataframe_file function."""
    
    def test_load_excel_single_sheet(self, dataframe_excel_file):
        """Test loading Excel file with single/multiple sheets."""
        dfs_dict = load_dataframe_file(dataframe_excel_file)
        
        # Verify it returns a dictionary
        assert isinstance(dfs_dict, dict)
        
        # Verify dictionary is not empty
        assert len(dfs_dict) > 0
        
        # Verify all values are DataFrames
        for df in dfs_dict.values():
            assert isinstance(df, pd.DataFrame)
    
    def test_excel_dataframe_has_data(self, dataframe_excel_file):
        """Test that loaded DataFrame has data."""
        dfs_dict = load_dataframe_file(dataframe_excel_file)
        
        # Get first DataFrame
        first_df = list(dfs_dict.values())[0]
        
        # Verify it has rows and columns
        assert len(first_df) > 0
        assert len(first_df.columns) > 0
    
    def test_load_csv_dataframe(self, tmp_path):
        """Test loading CSV DataFrame file."""
        # Create a test CSV file
        csv_file = tmp_path / "test_data.csv"
        test_data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        test_data.to_csv(csv_file, index=False)
        
        # Load the file
        dfs_dict = load_dataframe_file(csv_file)
        
        # Verify single entry with filename as key
        assert len(dfs_dict) == 1
        assert 'test_data' in dfs_dict
        
        # Verify data matches
        loaded_df = dfs_dict['test_data']
        pd.testing.assert_frame_equal(loaded_df, test_data)
    
    def test_load_unsupported_dataframe_extension(self, tmp_path):
        """Test loading DataFrame with unsupported extension raises error."""
        unsupported_file = tmp_path / "data.txt"
        unsupported_file.write_text("A,B,C\n1,2,3\n")
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_dataframe_file(unsupported_file)


class TestDelimiterDetection:
    """Tests for automatic delimiter detection in TXT files."""
    
    def test_detect_semicolon_delimiter(self, tmp_path):
        """Test detection of semicolon delimiter."""
        txt_file = tmp_path / "semicolon.txt"
        txt_file.write_text("Wavenumber;Intensity\n100;200\n150;250\n200;300\n")
        
        spectrum = load_spectrum_file(txt_file)
        
        # Verify data is loaded correctly
        assert len(spectrum.x0) == 3
        np.testing.assert_array_equal(spectrum.x0, [100, 150, 200])
        np.testing.assert_array_equal(spectrum.y0, [200, 250, 300])
    
    def test_detect_tab_delimiter(self, tmp_path):
        """Test detection of tab delimiter."""
        txt_file = tmp_path / "tab.txt"
        txt_file.write_text("Wavenumber\tIntensity\n100\t200\n150\t250\n200\t300\n")
        
        spectrum = load_spectrum_file(txt_file)
        
        # Verify data is loaded correctly
        assert len(spectrum.x0) == 3
        np.testing.assert_array_equal(spectrum.x0, [100, 150, 200])
    
    def test_detect_space_delimiter(self, tmp_path):
        """Test detection of space/whitespace delimiter."""
        txt_file = tmp_path / "space.txt"
        txt_file.write_text("Wavenumber Intensity\n100 200\n150 250\n200 300\n")
        
        spectrum = load_spectrum_file(txt_file)
        
        # Verify data is loaded correctly
        assert len(spectrum.x0) == 3
        np.testing.assert_array_equal(spectrum.x0, [100, 150, 200])
