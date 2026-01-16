"""
Pytest configuration and shared fixtures for SPECTROview tests.

This module provides:
- Qt application setup for pytest-qt
- Mock settings and GUI components
- Test data path fixtures
- Temporary workspace fixtures
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock
import pytest
import numpy as np
import pandas as pd

from PySide6.QtWidgets import QApplication
from spectroview.model.m_settings import MSettings
from spectroview.model.m_spectrum import MSpectrum


# ============================================================================
# Qt Application Setup
# ============================================================================

@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


# ============================================================================
# Path Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def project_root():
    """Return path to project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def examples_dir(project_root):
    """Return path to examples directory."""
    return project_root / "examples"


@pytest.fixture(scope="session")
def spectroscopic_data_dir(examples_dir):
    """Return path to spectroscopic_data directory."""
    return examples_dir / "spectroscopic_data"


@pytest.fixture
def temp_workspace(tmp_path):
    """Create temporary workspace directory for save/load tests."""
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    return workspace_dir


# ============================================================================
# Test Data Files
# ============================================================================

@pytest.fixture(scope="session")
def single_spectrum_file(spectroscopic_data_dir):
    """Return path to a single spectrum test file."""
    return spectroscopic_data_dir / "spectrum1_1ML.txt"


@pytest.fixture(scope="session")
def multiple_spectra_files(spectroscopic_data_dir):
    """Return list of spectrum file paths."""
    return [
        spectroscopic_data_dir / "spectrum1_1ML.txt",
        spectroscopic_data_dir / "spectrum2_1ML.txt",
        spectroscopic_data_dir / "spectrum3_3ML.txt",
    ]


@pytest.fixture(scope="session")
def map_2d_file(spectroscopic_data_dir):
    """Return path to a 2D map test file."""
    return spectroscopic_data_dir / "Small2Dmap.txt"


@pytest.fixture(scope="session")
def wafer_file(spectroscopic_data_dir):
    """Return path to a wafer data test file."""
    return spectroscopic_data_dir / "wafer10_newformat.csv"


@pytest.fixture(scope="session")
def fit_model_file(spectroscopic_data_dir):
    """Return path to a fit model JSON file."""
    return spectroscopic_data_dir / "model.json"


@pytest.fixture(scope="session")
def dataframe_excel_file(examples_dir):
    """Return path to an Excel DataFrame file."""
    return examples_dir / "data_inline.xlsx"


@pytest.fixture(scope="session")
def saved_spectra_workspace(examples_dir):
    """Return path to a saved spectra workspace file."""
    return examples_dir / "spectra.spectra"


@pytest.fixture(scope="session")
def saved_maps_workspace(examples_dir):
    """Return path to a saved maps workspace file."""
    return examples_dir / "wafers.maps"


@pytest.fixture(scope="session")
def saved_graphs_workspace(examples_dir):
    """Return path to a saved graphs workspace file."""
    return examples_dir / "graphs.graphs"


# ============================================================================
# Mock Settings
# ============================================================================

@pytest.fixture
def mock_settings(tmp_path):
    """Create a mock MSettings object with temporary paths."""
    settings = MSettings()
    
    # Set temporary paths for file operations
    settings.default_open_path = str(tmp_path)
    settings.default_save_path = str(tmp_path)
    settings.fit_models_folder = str(tmp_path / "fit_models")
    
    # Default fit parameters
    settings.fit_method = "leastsq"
    settings.fit_negative = False
    settings.fit_outliers = False
    settings.max_ite = 200
    settings.ncpus = 1  # Always single-threaded for tests
    
    return settings


# ============================================================================
# Mock GUI Components
# ============================================================================

@pytest.fixture
def mock_file_dialog(monkeypatch):
    """Mock QFileDialog for file selection tests."""
    mock_dialog = MagicMock()
    
    def mock_get_open_filenames(parent=None, caption="", directory="", filter=""):
        """Mock getOpenFileNames to return predefined files."""
        return mock_dialog.return_files, ""
    
    def mock_get_save_filename(parent=None, caption="", directory="", filter=""):
        """Mock getSaveFileName to return predefined path."""
        return mock_dialog.return_file, ""
    
    def mock_get_existing_directory(parent=None, caption="", directory=""):
        """Mock getExistingDirectory to return predefined path."""
        return mock_dialog.return_directory
    
    # Set default return values
    mock_dialog.return_files = []
    mock_dialog.return_file = ""
    mock_dialog.return_directory = ""
    
    # Patch QFileDialog methods
    from PySide6.QtWidgets import QFileDialog
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", mock_get_open_filenames)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_get_save_filename)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", mock_get_existing_directory)
    
    return mock_dialog


@pytest.fixture
def mock_message_box(monkeypatch):
    """Mock QMessageBox for user notification tests."""
    mock_box = MagicMock()
    mock_box.return_value = MagicMock()
    
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", mock_box)
    monkeypatch.setattr(QMessageBox, "warning", mock_box)
    monkeypatch.setattr(QMessageBox, "critical", mock_box)
    monkeypatch.setattr(QMessageBox, "question", mock_box)
    
    return mock_box


# ============================================================================
# Mock Data Generators
# ============================================================================

@pytest.fixture
def sample_spectrum():
    """Create a sample MSpectrum object for testing."""
    x = np.linspace(100, 500, 100)
    y = np.exp(-((x - 300) ** 2) / 1000) + 0.1 * np.random.randn(100)
    
    spectrum = MSpectrum()
    spectrum.fname = "test_spectrum"
    spectrum.x0 = x
    spectrum.y0 = y
    spectrum.x = x.copy()
    spectrum.y = y.copy()
    spectrum.baseline.mode = "Linear"
    
    return spectrum


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    data = {
        'X': np.arange(10),
        'Y': np.random.randn(10),
        'Z': np.random.randn(10),
        'Category': ['A', 'B'] * 5,
        'Value': np.random.uniform(0, 100, 10)
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_map_dataframe():
    """Create a sample map DataFrame for testing."""
    # Create a simple 5x5 grid
    x_coords = []
    y_coords = []
    for i in range(5):
        for j in range(5):
            x_coords.append(i)
            y_coords.append(j)
    
    # Add wavelength columns (e.g., Raman shift values)
    wavelengths = np.linspace(100, 500, 20)
    data = {'X': x_coords, 'Y': y_coords}
    
    for wl in wavelengths:
        # Generate random intensity values
        data[str(wl)] = np.random.uniform(0, 100, 25)
    
    return pd.DataFrame(data)


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup resources after each test."""
    yield
    # Add any cleanup code here if needed
    # e.g., close file handles, reset global state
