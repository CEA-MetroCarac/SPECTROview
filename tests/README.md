# SPECTROview Test Suite

This directory contains automated tests for the SPECTROview project using pytest and pytest-qt.

## Test Organization

```
tests/
├── conftest.py                        # Shared fixtures and test configuration
├── test_m_io.py                       # Model: File I/O functions
├── test_m_spectrum.py                 # Model: Single spectrum
├── test_m_spectra.py                  # Model: Spectra collection
├── test_m_graph.py                    # Model: Graph configuration
├── test_vm_workspace_spectra.py       # ViewModel: Spectra workspace
├── test_vm_workspace_graphs.py        # ViewModel: Graphs workspace
└── test_integration_workflows.py      # Integration & end-to-end tests
```

## Running Tests

### Run All Tests

```bash
# From project root
pytest tests/ -v
```

### Run Specific Test File

```bash
pytest tests/test_m_spectrum.py -v
```

### Run Specific Test

```bash
pytest tests/test_m_spectrum.py::TestMSpectrumInitialization::test_create_empty_spectrum -v
```

### Run Tests with Coverage

```bash
pytest tests/ --cov=spectroview --cov-report=html
```

The coverage report will be generated in `htmlcov/index.html`.

### Run Tests in Parallel (faster)

```bash
pytest tests/ -n auto
```

Requires: `pip install pytest-xdist`

## Test Categories

### Model Layer Tests

Test pure data models without Qt dependencies:

- **test_m_io.py**: File loading for spectra, maps, and DataFrames
- **test_m_spectrum.py**: Single spectrum operations (initialization, x-correction, reinit)
- **test_m_spectra.py**: Spectrum collection operations (add/remove, reorder, apply_model)
- **test_m_graph.py**: Graph configuration (save/load, properties)

### ViewModel Layer Tests

Test business logic with mocked GUI components:

- **test_vm_workspace_spectra.py**: Spectra workspace functionality
  - File loading and management
  - Selection management
  - Spectral range operations
  - Baseline operations (add/remove/copy/paste)
  - Peak operations (add/remove/copy/paste)
  - X-correction (apply/undo)
  - Workspace persistence
  - Fit results collection

- **test_vm_workspace_graphs.py**: Graphs workspace functionality
  - DataFrame loading and management
  - Data filtering
  - Graph creation and updates
  - Workspace persistence

### Integration Tests

End-to-end workflow tests:

- **test_integration_workflows.py**:
  - Complete spectra workflow (load → process → fit → save → reload)
  - Complete graphs workflow (load → filter → plot → save → reload)
  - Batch processing multiple spectra
  - Data integrity tests
  - Edge cases and robustness tests

## Test Fixtures

Common fixtures are defined in `conftest.py`:

### Qt Application
- `qapp`: QApplication instance for Qt tests

### Settings
- `mock_settings`: Mocked MSettings object with temporary paths

### Test Data Paths
- `single_spectrum_file`: Path to single spectrum test file
- `multiple_spectra_files`: List of spectrum file paths
- `map_2d_file`: Path to 2D map file
- `wafer_file`: Path to wafer data file
- `dataframe_excel_file`: Path to Excel DataFrame file
- `saved_spectra_workspace`: Path to saved .spectra workspace
- `saved_maps_workspace`: Path to saved .maps workspace
- `saved_graphs_workspace`: Path to saved .graphs workspace

### Temporary Directories
- `temp_workspace`: Temporary workspace directory for save/load tests

### Mock GUI Components
- `mock_file_dialog`: Mocked QFileDialog
- `mock_message_box`: Mocked QMessageBox

### Sample Data Generators
- `sample_spectrum`: Generate MSpectrum object
- `sample_dataframe`: Generate pandas DataFrame
- `sample_map_dataframe`: Generate map DataFrame

## Testing Strategy

### Mocking Approach

1. **GUI Components**: All View layer components are mocked (QFileDialog, QMessageBox, progress bars)
2. **Threading**: Thread execution is mocked; we test threading logic without actual parallelization
3. **Matplotlib**: Test plot configuration, not rendering

### Test Data

Tests use example data from `examples/spectroscopic_data/`:
- Single spectra files
- 2D map files  
- Wafer data files
- Saved workspace files

### Principles

- **Deterministic**: Tests produce consistent results
- **Isolated**: Each test is independent, no shared state
- **Fast**: Expensive operations are mocked
- **Focused**: Test one thing at a time

## Troubleshooting

### Qt Platform Plugin Error

If you see: `qt.qpa.plugin: Could not find the Qt platform plugin "windows"`

Solution: Set environment variable:
```bash
set QT_QPA_PLATFORM=offscreen
pytest tests/ -v
```

### Missing Test Data

If example data files are missing, some tests will be skipped automatically using `pytest.skip()`.

### Import Errors

Ensure SPECTROview is installed in development mode:
```bash
pip install -e .
```

## Adding New Tests

### 1. Model Layer Test

```python
# tests/test_m_mymodel.py
import pytest
from spectroview.model.m_mymodel import MyModel

class TestMyModel:
    def test_something(self):
        model = MyModel()
        assert model.property == expected_value
```

### 2. ViewModel Test with Fixtures

```python
# tests/test_vm_myviewmodel.py
import pytest
from spectroview.viewmodel.vm_myviewmodel import MyViewModel

class TestMyViewModel:
    def test_operation(self, qapp, mock_settings):
        vm = MyViewModel(mock_settings)
        vm.do_something()
        assert vm.state == expected_state
```

### 3. Integration Test

```python
# tests/test_integration_workflows.py
class TestMyWorkflow:
    def test_complete_workflow(self, qapp, mock_settings, test_file):
        # Load
        vm = ViewModel(mock_settings)
        vm.load(test_file)
        
        # Process
        vm.process()
        
        # Verify
        assert vm.result == expected
```

## Continuous Integration

To run tests in CI/CD pipelines:

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    pip install -e ".[test]"
    pytest tests/ --cov=spectroview --cov-report=xml
```

## Coverage Goals

Target coverage for different layers:
- **Model layer**: >80%
- **ViewModel layer**: >70%
- **View layer**: Not tested (GUI components)

## Contact

For issues with tests, refer to:
- Main documentation: `doc/dev_intructions.md`
- Implementation plan: `.gemini/antigravity/brain/*/implementation_plan.md`
