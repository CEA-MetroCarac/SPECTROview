# SPECTROview AI Coding Agent Instructions

## Project Overview
SPECTROview is a Qt/PySide6 desktop application for spectroscopic data processing (Raman, PL) built using **strict MVVM architecture**. The app processes discrete spectra, 2D maps, and wafer-scale data with fitting capabilities powered by `fitspy` and `lmfit`.

## Architecture: MVVM Pattern (Critical)

### File Naming Convention
- **View**: `v_*.py` (e.g., `v_workspace_spectra.py`, `v_fit_model_builder.py`)
- **ViewModel**: `vm_*.py` (e.g., `vm_workspace_spectra.py`, `vm_fit_model_builder.py`)
- **Model**: `m_*.py` (e.g., `m_spectra.py`, `m_spectrum.py`, `m_settings.py`)

### Responsibilities
- **View (`view/`)**: PySide6 widgets, UI layout, NO business logic. Connects to ViewModel signals/slots.
- **ViewModel (`viewmodel/`)**: QObject subclasses with `Signal` definitions. Handles business logic, data transformation, file I/O, threading. Emits signals to update View.
- **Model (`model/`)**: Data structures. `MSpectrum` and `MSpectra` extend `fitspy` base classes.

### Communication Flow
```
User Action → View → ViewModel (via slot) → Model (data) → ViewModel (emit signal) → View (update UI)
```

**Example**: [v_workspace_spectra.py](spectroview/view/v_workspace_spectra.py#L24) instantiates `VMWorkspaceSpectra`, connects ViewModel signals to View update methods. See [vm_workspace_spectra.py](spectroview/viewmodel/vm_workspace_spectra.py#L21-L30) for signal definitions.

## Key Modules

### Three Workspace Tabs
1. **Spectra** (`spectroview/workspaces/spectrums.py`, `view/v_workspace_spectra.py`): Process discrete 1D spectra
2. **Maps** (`spectroview/workspaces/maps.py`, `view/v_workspace_maps.py`): Process 2D hyperspectral maps and wafer data
3. **Graphs** (`spectroview/workspaces/graphs.py`, `view/v_workspace_graphs.py`): Visualize/plot DataFrames

### Fitting Engine
- **Primary**: Uses `fitspy` library (see `model/m_spectrum.py`, `model/m_spectra.py` extending `FitspySpectrum`/`FitspySpectra`)
- **Fast parallel fitting**: `fitfaster/` module with `engine.py` using `scipy.optimize.least_squares` + `joblib.Parallel`
- **Threading**: Long-running fits use `FitThread` (see `viewmodel/utils.py`) to prevent UI blocking. Progress updates via signals.

### File Formats
- **Input**: `.csv`, `.txt`, `.xlsx` (spectral data)
- **Saved workspace**: `.spectra`, `.maps`, `.graphs` (pickled Python objects containing processed data + models)
- **Fit models**: `.json` files stored in user-defined folder (see Settings)

## Development Workflows

### Running the App
```bash
# From project root
python -m spectroview.main
# Or via entry point (after pip install -e .)
spectroview
```

### Testing
- **Framework**: pytest with `pytest-qt` for Qt testing
- **Tests location**: `tests/test_*.py`
- **Run tests**: `pytest` (requires qtbot fixture)
- Example: [test_1_reading_all_supported_data.py](tests/test_1_reading_all_supported_data.py)

### Building
- Uses `setuptools` (see [pyproject.toml](pyproject.toml))
- Entry point: `spectroview.main:launcher`
- Version defined in `spectroview/__init__.py` as `VERSION` string

## Project-Specific Conventions

### Constants & Configuration
- **Constants**: Defined in [spectroview/__init__.py](spectroview/__init__.py) (`PEAK_MODELS`, `FIT_PARAMS`, `PALETTE`, `PLOT_STYLES`, `DEFAULT_COLORS`, etc.)
- **Settings**: `MSettings` class (model) + `VMSettings` (viewmodel) + `VSettingsDialog` (view). Persisted via `QSettings("CEA-Leti", "SPECTROview")`
- **Paths**: `RESOURCES_DIR`, `ICON_DIR`, `UI_FILE`, `PLOT_POLICY` defined in `__init__.py`

### Signal/Slot Pattern
- ViewModel defines `Signal` objects (e.g., `spectra_list_changed = Signal(list)`)
- View connects signals to UI update methods (e.g., `self.vm.spectra_list_changed.connect(self.update_list)`)
- **Never** call View methods directly from ViewModel; always emit signals

### Data Flow Example (Loading Files)
1. User clicks "Open" → [main.py](spectroview/main.py#L85) `open_files()` called
2. Files categorized by extension → routed to workspace (e.g., `.spectra` → `VWorkspaceSpectra`)
3. Workspace's ViewModel loads data → `vm.load_files(paths)` ([vm_workspace_spectra.py](spectroview/viewmodel/vm_workspace_spectra.py#L43))
4. ViewModel updates model (`MSpectra.add()`) → emits `spectra_list_changed` signal
5. View receives signal → updates `QListWidget` with spectrum names

### Threading for Fits
- Fitting multiple spectra is CPU-intensive → use `FitThread` (QThread subclass)
- Progress updates: ViewModel emits `fit_progress_updated(current, total, success_count, elapsed_time)`
- View updates `QProgressBar` in response
- See [viewmodel/vm_workspace_spectra.py](spectroview/viewmodel/vm_workspace_spectra.py) `fit_selected()` method

## External Dependencies
- **fitspy**: Core fitting engine (version pinned to `2025.6`)
- **lmfit**: Nonlinear least-squares minimization
- **PySide6**: Qt bindings for Python (not PyQt5!)
- **matplotlib**: Plotting (with custom style from `resources/plotpolicy.mplstyle`)
- **pandas**: DataFrame handling in Graphs workspace
- **superqt**: Enhanced Qt widgets (check usage in `view/components/`)

## Common Pitfalls
- **Don't mix Qt framework**: This is **PySide6**, not PyQt5/PyQt6
- **Don't put business logic in View**: Move to ViewModel
- **Don't forget signal/slot connections**: Missing connection = UI won't update
- **Threading**: Use `QThread` for long operations; direct processing blocks UI
- **File paths**: Always use `Path` from `pathlib`, not raw strings

## References
- Main entry point: [main.py](spectroview/main.py)
- MVVM example: [v_workspace_spectra.py](spectroview/view/v_workspace_spectra.py) + [vm_workspace_spectra.py](spectroview/viewmodel/vm_workspace_spectra.py)
- Model extensions: [m_spectra.py](spectroview/model/m_spectra.py), [m_spectrum.py](spectroview/model/m_spectrum.py)
