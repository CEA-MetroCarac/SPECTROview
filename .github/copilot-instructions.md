# SPECTROview Coding Instructions

## Project Overview
SPECTROview is a Qt/PySide6 desktop application for spectroscopic data processing (Raman, PL) built using **strict MVVM architecture**. The app processes discrete spectra, 2D hyperspectral maps, and wafer-scale data.

## Architecture: MVVM Pattern (Strict Enforcement)

### File Naming Convention
- **View**: `v_*.py` (e.g., `v_workspace_spectra.py`, `v_fit_model_builder.py`, `v_map_viewer.py`)
- **ViewModel**: `vm_*.py` (e.g., `vm_workspace_spectra.py`, `vm_fit_model_builder.py`, `vm_settings.py`)
- **Model**: `m_*.py` (e.g., `m_spectra.py`, `m_spectrum.py`, `m_settings.py`, `m_graph.py`)

### Layer Responsibilities

#### View Layer (`view/` and `view/components/`)
- **Purpose**: PySide6 widgets, UI layout, user interaction handlers
- **Rules**: 
  - NO business logic
  - NO data manipulation
  - NO file I/O
  - Only UI state management and event handling
- **Communication**: Connects to ViewModel signals/slots, emits signals for user actions
- **Files**: 
  - Main workspaces: `v_workspace_spectra.py`, `v_workspace_maps.py`, `v_workspace_graphs.py`
  - Components: `v_map_viewer.py`, `v_spectra_viewer.py`, `v_graph.py`, `v_fit_model_builder.py`, etc.

#### ViewModel Layer (`viewmodel/`)
- **Purpose**: Business logic, data transformation, orchestration
- **Rules**:
  - QObject subclasses with `Signal` definitions
  - Handles all business logic and data processing
  - Manages file I/O operations
  - Creates and manages worker threads (e.g., `FitThread`)
  - Emits signals to update View
  - **Never** imports or calls View methods directly
- **Communication**: Receives method calls from View, emits signals back to View
- **Files**: `vm_workspace_spectra.py`, `vm_workspace_maps.py`, `vm_workspace_graphs.py`, `vm_fit_model_builder.py`, `vm_settings.py`
- **Utilities**: `utils.py` (contains `FitThread`, helper functions, custom widgets)

#### Model Layer (`model/`)
- **Purpose**: Pure data structures and domain models
- **Rules**:
  - No Qt dependencies (except where extending fitspy classes that use Qt)
  - Represents application state and domain entities
  - No UI logic, no signals/slots
- **Files**: 
  - `m_spectrum.py`: Extends `fitspy.Spectrum` for single spectrum data
  - `m_spectra.py`: Extends `fitspy.Spectra` for spectrum collections
  - `m_graph.py`: Graph/plot configuration model
  - `m_settings.py`: Application settings model
  - `m_io.py`: File loading utilities
  - `m_file_converter.py`: File format conversion utilities
  - `m_fit_model_manager.py`: Fit model management

### Communication Flow
```
User Action → View (emit signal) → ViewModel (slot/method) → Model (data manipulation)
                ↑                                              ↓
                └──────────── ViewModel (emit signal) ─────────┘
```

**Key Principle**: Views and ViewModels communicate ONLY via signals/slots. Never direct method calls from ViewModel to View.

### Example: Signal/Slot Pattern
```python
# ViewModel defines signals
class VMWorkspaceSpectra(QObject):
    spectra_list_changed = Signal(list)  # Emits list of spectra
    fit_progress_updated = Signal(int, int, int, float)  # current, total, success, time
    
# View connects to ViewModel signals
class VWorkspaceSpectra(QWidget):
    def __init__(self):
        self.vm = VMWorkspaceSpectra(settings)
        self.vm.spectra_list_changed.connect(self._update_spectra_list)
        self.vm.fit_progress_updated.connect(self._update_progress_bar)
```

## Application Structure

### Main Window (`main.py`)
- Entry point: `spectroview.main:launcher`
- Creates three workspace tabs: Spectra, Maps, Graphs
- Handles cross-workspace communication (dependency injection)
- Manages menubar, settings, theme switching
- Routes file opening to appropriate workspace based on file type

### Three Workspace Tabs

#### 1. Spectra Workspace
- **View**: `view/v_workspace_spectra.py`
- **ViewModel**: `viewmodel/vm_workspace_spectra.py`
- **Purpose**: Process discrete 1D spectra
- **Features**:
  - Load individual spectrum files (`.txt`, `.csv`)
  - Baseline correction, peak fitting
  - Batch processing with parallel fitting
  - Export fit results to Excel
  - Save/load workspace (`.spectra` files)

#### 2. Maps Workspace
- **View**: `view/v_workspace_maps.py`
- **ViewModel**: `viewmodel/vm_workspace_maps.py` (extends `VMWorkspaceSpectra`)
- **Purpose**: Process 2D hyperspectral maps and wafer data
- **Features**:
  - Load hyperspectral map files (`.txt`, `.csv` with X, Y columns)
  - Interactive heatmap visualization with map viewer
  - Profile extraction (2 points → line plot in Graphs workspace)
  - Multiple map viewers (main + floating dialogs)
  - Wafer-scale data visualization (300mm, 200mm, 100mm)
  - Map type selection: 2D maps or wafer formats
  - Export map data and fit results
  - Save/load workspace (`.maps` files)
- **Components**:
  - `v_map_viewer.py`: Matplotlib canvas with heatmap controls
  - `v_map_viewer_dialog.py`: Floating map viewer window
  - `v_map_list.py`: Map and spectra list with controls

#### 3. Graphs Workspace
- **View**: `view/v_workspace_graphs.py`
- **ViewModel**: `viewmodel/vm_workspace_graphs.py`
- **Model**: `model/m_graph.py`
- **Purpose**: Visualize/plot DataFrames
- **Features**:
  - Load DataFrame files (`.xlsx`, `.csv`)
  - Multiple plot types: line, scatter, bar, box, heatmap, wafer, trendline
  - MDI (Multiple Document Interface) for multiple plots
  - Data filtering with complex conditions
  - Axis limits, labels, legends customization
  - Export plots and data
  - Save/load workspace (`.graphs` files)
- **Components**:
  - `v_graph.py`: Individual plot widget using seaborn/matplotlib
  - `v_data_filter.py`: DataFrame filtering UI
  - `v_dataframe_table.py`: Tabular data viewer

### Shared Components (`view/components/`)

- **`v_spectra_viewer.py`**: Matplotlib canvas for plotting spectra (used in Spectra and Maps)
- **`v_fit_model_builder.py`**: Peak model builder UI (baseline + peaks configuration)
- **`v_peak_table.py`**: Peak parameters table editor
- **`v_fit_results.py`**: Fit results display table
- **`v_menubar.py`**: Application menubar (File, Tools, Settings, Help)
- **`v_settings.py`**: Settings dialog
- **`v_about.py`**: About dialog
- **`v_spectra_list.py`**: Spectra list widget with checkboxes

## Fitting Engine

### Primary Fitting (fitspy)
- **Models**: Extend `fitspy.Spectrum` and `fitspy.Spectra`
- **Files**: `model/m_spectrum.py`, `model/m_spectra.py`
- **Features**: Baseline correction, peak fitting, parameter constraints
- **Peak models**: Gaussian, Lorentzian, Pseudo-Voigt, etc.

### Threading for Long Operations
- **FitThread**: QThread subclass in `viewmodel/utils.py`
- **Pattern**:
  1. ViewModel creates FitThread with worker function
  2. Thread emits progress signals
  3. ViewModel forwards signals to View
  4. View updates progress bar
- **Example**: See `vm_workspace_spectra.py` → `fit_selected()` method

## File Formats

### Input Formats
- **Spectra**: `.csv`, `.txt` (2 columns: wavelength, intensity)
- **Maps**: `.csv`, `.txt` (first 2 columns: X, Y, remaining columns: wavelengths)
- **DataFrames**: `.xlsx`, `.csv` (any tabular data)

### Workspace Save Files
- **`.spectra`**: JSON with compressed spectra data and fit models
- **`.maps`**: JSON with compressed map data, spectra, and fit models
- **`.graphs`**: JSON with DataFrames and plot configurations

### Fit Models
- **`.json`**: User-defined peak models (baseline + peaks)
- **Location**: User-configurable folder in Settings

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
- **Run tests**: `pytest`
- **Examples**: 
  - `test_1_reading_all_supported_data.py`: File loading tests
  - `test_2_loading_saved_files.py`: Workspace save/load tests

### Building
- **Build system**: setuptools (see `pyproject.toml`)
- **Entry point**: `spectroview.main:launcher`
- **Version**: Defined in `spectroview/__init__.py` as `VERSION` string

## Configuration & Constants

### Constants (`spectroview/__init__.py`)
- `VERSION`: Application version
- `PEAK_MODELS`: Available peak model types
- `FIT_PARAMS`: Default fit parameters
- `PALETTE`: Color palettes for heatmaps
- `PLOT_STYLES`: Available plot types
- `DEFAULT_COLORS`: Default color scheme
- `RESOURCES_DIR`, `ICON_DIR`: Resource paths
- `USER_MANUAL_PDF`: User manual path

### Settings Persistence
- **Storage**: `QSettings("CEA-Leti", "SPECTROview")`
- **Model**: `model/m_settings.py`
- **ViewModel**: `viewmodel/vm_settings.py`
- **View**: `view/components/v_settings.py`
- **Settings**: Theme, default paths, plot styles, fit parameters, etc.

## Cross-Workspace Communication

### Dependency Injection Pattern
- **Where**: `main.py` → `setup_connections()`
- **Purpose**: Enable workspaces to communicate without tight coupling
- **Example**: Maps workspace needs to send profiles to Graphs workspace

```python
# In main.py
def setup_connections(self):
    # Inject Graphs workspace into Maps ViewModel
    self.v_maps_workspace.vm.set_graphs_workspace(self.v_graphs_workspace)
    
    # Connect tab switching signal
    self.v_maps_workspace.vm.switch_to_graphs_tab.connect(
        lambda: self.tabWidget.setCurrentWidget(self.v_graphs_workspace)
    )
```

### Example: Profile Extraction (Maps → Graphs)
1. User selects 2 points on map → extracts profile
2. Maps ViewModel calls `graphs_workspace.create_plot_from_config()`
3. Graphs workspace creates line plot with profile data
4. Maps ViewModel emits `switch_to_graphs_tab` signal
5. Main window switches to Graphs tab

## External Dependencies

- **fitspy** `==2025.6`: Core fitting engine
- **lmfit**: Nonlinear least-squares minimization
- **PySide6**: Qt bindings for Python (**not PyQt5/PyQt6**)
- **matplotlib** `==3.8.4`: Plotting backend
- **seaborn** `==0.13.2`: Statistical plotting
- **pandas**: DataFrame handling
- **scipy**: Scientific computing (interpolation, optimization)
- **superqt**: Enhanced Qt widgets (e.g., `QLabeledDoubleRangeSlider`)
- **openpyxl**: Excel file support
- **numpy**: Numerical operations

## Common Pitfalls & Best Practices

### ❌ Don't:
- Mix Qt frameworks (this is **PySide6**, not PyQt5/PyQt6)
- Put business logic in View classes
- Call View methods directly from ViewModel
- Forget to connect signals (missing connection = UI won't update)
- Use synchronous operations for long-running tasks (always use QThread)
- Use raw strings for file paths (use `Path` from `pathlib`)
- Import View classes in ViewModel files

### ✅ Do:
- Follow MVVM file naming convention (`v_*.py`, `vm_*.py`, `m_*.py`)
- Use signals for all ViewModel → View communication
- Use dependency injection for cross-workspace communication
- Block signals during programmatic UI updates (`blockSignals(True)`)
- Use `QTimer.singleShot()` for deferred UI updates
- Cache expensive computations (e.g., griddata in map viewer)
- Validate user input in ViewModel before processing
- Emit informative notification signals for user feedback

## Key Implementation Patterns

### 1. Loading Files
```
User → Open Files → main.py categorizes by extension
         ↓
    Routes to workspace (e.g., .txt → Maps if >3 columns, Spectra if 2 columns)
         ↓
    ViewModel.load_files(paths) → Model updates → Emit signals → View updates
```

### 2. Fitting Workflow
```
User → Click Fit → ViewModel creates FitThread
         ↓
    Thread runs fitting (parallel processing)
         ↓
    Thread emits progress → ViewModel forwards → View updates progress bar
         ↓
    Thread completes → ViewModel collects results → Emit signals → View updates
```

### 3. Map Viewer Selection
```
User → Click on map → v_map_viewer emits spectra_selected signal
         ↓
    v_workspace_maps updates spectra list selection
         ↓
    ViewModel.set_selected_fnames() → Emit spectra_selection_changed
         ↓
    v_spectra_viewer receives signal → Updates plot
```

## Resources & References

### Main Entry Points
- **Application**: `spectroview/main.py` → `Main` class
- **Launcher**: `spectroview/main.py` → `launcher()` function

### MVVM Examples
- **Spectra Workspace**: `view/v_workspace_spectra.py` + `viewmodel/vm_workspace_spectra.py`
- **Maps Workspace**: `view/v_workspace_maps.py` + `viewmodel/vm_workspace_maps.py`
- **Graphs Workspace**: `view/v_workspace_graphs.py` + `viewmodel/vm_workspace_graphs.py`

### Model Extensions
- **Spectrum**: `model/m_spectrum.py` (extends `fitspy.Spectrum`)
- **Spectra Collection**: `model/m_spectra.py` (extends `fitspy.Spectra`)
- **Graph**: `model/m_graph.py` (plot configuration)

### Complex Components
- **Map Viewer**: `view/components/v_map_viewer.py` (interactive heatmap with profile extraction)
- **Fit Model Builder**: `view/components/v_fit_model_builder.py` + `viewmodel/vm_fit_model_builder.py`
- **Graph Plotting**: `view/components/v_graph.py` (seaborn/matplotlib integration)

## Debugging Tips

- **Signal not received?** Check connections in `setup_connections()` or `_connect_signals()`
- **UI frozen?** Long operation running on main thread → move to QThread
- **Plot not updating?** Check if signals are blocked or cache needs clearing
- **Import errors?** Verify MVVM layer separation (no View imports in ViewModel)
- **Threading issues?** Ensure thread-safe operations (use signals for cross-thread communication)
