# SPECTROview Developer Instructions

## Project Overview
SPECTROview is a Qt/PySide6 desktop application for spectroscopic data processing (Raman, PL, XRD, XPS, TRPL) built using **strict MVVM architecture**. The app processes discrete spectra, 2D hyperspectral maps, and wafer-scale data. It features a high-performance tensor-based fitting engine for maps, multivariate analysis (PCA/NMF), and an integrated data visualization workspace.

**Repository**: [https://github.com/CEA-MetroCarac/SPECTROview](https://github.com/CEA-MetroCarac/SPECTROview)

---

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

#### ViewModel Layer (`viewmodel/`)
- **Purpose**: Business logic, data transformation, orchestration
- **Rules**:
  - QObject subclasses with `Signal` definitions
  - Handles all business logic and data processing
  - Manages file I/O operations
  - Creates and manages worker threads (e.g., `TensorFitThread`, `FitThread`)
  - Emits signals to update View
  - **Never** imports or calls View methods directly
- **Communication**: Receives method calls from View, emits signals back to View

#### Model Layer (`model/`)
- **Purpose**: Pure data structures and domain models
- **Rules**:
  - No Qt dependencies (except where extending fitspy classes that use Qt)
  - Represents application state and domain entities
  - No UI logic, no signals/slots

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

---

## Application Structure

### Main Window (`main.py`)
- Entry point: `spectroview.main:launcher`
- Creates three workspace tabs: Spectra, Maps, Graphs
- Handles cross-workspace communication (dependency injection)
- Manages menubar, settings, theme switching
- Routes file opening to appropriate workspace based on file type

### Package Constants (`spectroview/__init__.py`)
- `VERSION`: Application version string
- `PEAK_MODELS`: Available peak model types (Gaussian, Lorentzian, PseudoVoigt, GaussianAsym, LorentzianAsym, Fano, DecaySingleExp, DecayBiExp)
- `FIT_PARAMS`: Default fit parameters
- `PALETTE`: Color palettes for heatmaps
- `PLOT_STYLES`: Available plot types (point, scatter, box, bar, line, trendline, wafer, 2Dmap)
- `DEFAULT_COLORS`, `MARKERS`, `LEGEND_LOCATION`: Plot styling constants
- `X_AXIS_UNIT`, `Y_AXIS_UNIT`: Predefined axis label options
- `RESOURCES_DIR`, `ICON_DIR`: Resource paths (PyInstaller-safe via `resource_path()`)
- `USER_MANUAL_PDF`: User manual path

---

## Three Workspace Tabs

### 1. Spectra Workspace
- **View**: `view/v_workspace_spectra.py`
- **ViewModel**: `viewmodel/vm_workspace_spectra.py`
- **Purpose**: Process discrete 1D spectra
- **Features**:
  - Load individual spectrum files (`.txt`, `.csv`, `.wdf`, `.spc`, `.dat`)
  - Baseline correction (Manual: Linear/Polynomial, Auto: airPLS/asLS)
  - Peak fitting (legacy fitspy/lmfit or Tensor engine)
  - X-axis correction (Si reference)
  - Y-axis normalization
  - Batch processing with parallel fitting
  - Copy/paste fit models between spectra
  - Fit results collection with computed columns
  - Export fit results to Excel
  - Multivariate Analysis (PCA/NMF)
  - Save/load workspace (`.spectra` files)

### 2. Maps Workspace
- **View**: `view/v_workspace_maps.py` (extends `VWorkspaceSpectra`)
- **ViewModel**: `viewmodel/vm_workspace_maps.py` (extends `VMWorkspaceSpectra`)
- **Purpose**: Process 2D hyperspectral maps and wafer data
- **Features**:
  - Load hyperspectral map files (`.txt`, `.csv`, `.wdf`, `.spc`)
  - Interactive heatmap visualization with map viewer
  - High-performance tensor fitting (< 3s for typical maps)
  - Profile extraction (2 points → line plot in Graphs workspace)
  - Multiple map viewers (main + floating dialogs)
  - Wafer-scale data visualization (300mm, 200mm, 100mm)
  - Map type selection: 2D maps or wafer formats
  - Mask feature for filtering heatmap regions
  - Spectrum selection via map click (synced with spectra list)
  - Send selected spectra to Spectra workspace for comparison
  - Zone and quadrant classification for wafer data
  - Export map data and fit results
  - Save/load workspace (`.maps` files) with fast numpy binary serialization
- **Key Components**:
  - `v_map_viewer.py`: Matplotlib canvas with heatmap, profile extraction, selection tools
  - `v_map_viewer_dialog.py`: Floating map viewer window
  - `v_map_list.py`: Map and spectra list with controls

### 3. Graphs Workspace
- **View**: `view/v_workspace_graphs.py`
- **ViewModel**: `viewmodel/vm_workspace_graphs.py`
- **Model**: `model/m_graph.py`
- **Purpose**: Visualize/plot DataFrames
- **Features**:
  - Load DataFrame files (`.xlsx`, `.csv`)
  - Multiple plot types: line, scatter, bar, box, heatmap, wafer, trendline, 2Dmap
  - MDI (Multiple Document Interface) for multiple plots
  - Data filtering with pandas query expressions
  - Axis limits, labels, legends customization
  - Singleton `CustomizeGraphDialog` with auto-switch (see below)
  - Annotations (vertical/horizontal lines, text)
  - Broken axis support (X and Y)
  - Export plots and data
  - Save/load workspace (`.graphs` files)
- **Key Components**:
  - `v_graph.py`: Individual plot widget using seaborn/matplotlib
  - `v_data_filter.py`: DataFrame filtering UI
  - `v_dataframe_table.py`: Tabular data viewer
  - `customize_graph_dialog.py`: Singleton dialog for graph customization

---

## Shared View Components (`view/components/`)

| File | Purpose |
|------|---------|
| `v_spectra_viewer.py` | Matplotlib canvas for plotting spectra (used in Spectra and Maps) |
| `v_fit_model_builder.py` | Peak model builder UI (baseline + peaks configuration) |
| `v_peak_table.py` | Peak parameters table editor with constraints (fix, limits, expressions) |
| `v_fit_results.py` | Fit results display table with computed columns |
| `v_menubar.py` | Application menubar (File, Tools, Settings, Help) |
| `v_settings.py` | Settings dialog |
| `v_about.py` | About dialog |
| `v_spectra_list.py` | Spectra list widget with checkboxes |
| `v_map_viewer.py` | Heatmap viewer with profile extraction and selection tools |
| `v_map_viewer_dialog.py` | Floating map viewer window |
| `v_map_list.py` | Map list and spectra list with controls |
| `v_moretab.py` | Metadata viewer, normalization controls, cosmic ray detection |
| `v_mva.py` | MVA (PCA/NMF) UI with plots |
| `v_data_filter.py` | DataFrame filtering UI |
| `v_dataframe_table.py` | Tabular data viewer |
| `v_graph.py` | Individual plot widget (seaborn/matplotlib) |
| `customize_graph_dialog.py` | Graph customization dialog (axis, legend, annotations) |
| `customized_widgets.py` | Custom Qt widgets (e.g., `CustomizedPalette`) |

---

## Model Layer (`model/`)

| File | Purpose |
|------|---------|
| `m_spectrum.py` | Extends `fitspy.Spectrum` for single spectrum data |
| `m_spectra.py` | Extends `fitspy.Spectra` for spectrum collections |
| `m_graph.py` | Graph/plot configuration model |
| `m_settings.py` | Application settings model |
| `m_io.py` | File loading utilities (TXT, CSV, WDF, SPC, TRPL, DAT) |
| `m_file_converter.py` | File format conversion utilities |
| `m_fit_model_manager.py` | Fit model management |
| `m_fit_models.py` | Custom peak model functions (Fano, DecaySingleExp, DecayBiExp) registered into fitspy |
| `m_spc.py` | Custom reader for Galactic SPC files (binary format) |
| `m_mva.py` | Multivariate analysis engine (PCA via SVD, NMF via multiplicative update rules) |
| `m_quick_calc.py` | Quick calculators (Spot Size, Penetration Depth, Unit Converter) |

---

## Fitting Engine

SPECTROview supports two fitting engines. The Tensor engine is the default for both workspaces.

### Legacy Engine (fitspy + lmfit)
- **Models**: Extend `fitspy.Spectrum` and `fitspy.Spectra`
- **Files**: `model/m_spectrum.py`, `model/m_spectra.py`, `viewmodel/utils.py` (`FitThread`)
- **How it works**: Per-spectrum fitting via `scipy.optimize.least_squares` orchestrated by lmfit
- **When used**: Fallback when `_use_batch_engine = False` in the ViewModel

### Tensor Fit Engine (`fit_engine/`)
The high-performance engine that fits **all spectra simultaneously** using a batched Levenberg-Marquardt optimizer with analytical Jacobians. Achieves ~10–15× speedup over the legacy engine.

> **Deep-dive documentation**: See [`doc/dev_Tensor_Fit_Engine.md`](dev_Tensor_Fit_Engine.md)

#### Module Structure

| File | Class/Function | Purpose |
|------|----------------|---------|
| `tensor_engine.py` | `TensorFittingEngine` | Orchestrator: preprocessing, evaluator construction, optimization, result writeback |
| `evaluator.py` | `TensorEvaluator` | Maps `fit_model` dict → flat parameter tensors. Handles free/fixed params, bounds, expressions, model evaluation, Jacobian computation |
| `optimizer.py` | `batched_levenberg_marquardt()` | Core numerical workhorse. Solves N independent least-squares problems simultaneously via tensor operations |
| `models.py` | `batched_gaussian`, `batched_lorentzian`, `batched_pseudovoigt` + their `_jac` counterparts | Batched peak model functions with analytical Jacobians. Registry: `BATCHED_MODELS` |
| `scalar_models.py` | `FitResult`, `ParamValue`, `PEAK_MODEL_REGISTRY` | Lightweight result objects (lmfit-compatible), scalar fallback models for peak shapes without analytical Jacobians |
| `tensor_fit_thread.py` | `TensorFitThread` | QThread wrapper for async execution with progress signals |

#### Processing Pipeline
```
User clicks Fit → ViewModel._run_fit_thread()
    → TensorFitThread (QThread)
        → TensorFittingEngine.fit_spectra()
            1. Apply fit model to all spectra
            2. TensorEvaluator.from_fit_model() — parse model dict → param layout
            3. Preprocess all spectra (spectral range, baseline)
            4. Build weights matrix (fit_negative, fit_outliers, coef_noise)
            5. Extract data matrix Y (N × M) and initial params p0 (N × K)
            6. batched_levenberg_marquardt() — iterate until convergence
            7. Write results back to MSpectrum objects
        → Emit progress_changed signal → View updates progress bar
```

#### Adding a New Peak Model
1. Define `batched_newshape(x, params)` in `models.py` — params shape is `(N, n_params)`
2. **(Crucial for speed)** Derive and define `batched_newshape_jac(x, params)` — return shape `(N, M, n_params)`
3. Register in `BATCHED_MODELS` dict: `"NewShape": (batched_newshape, batched_newshape_jac, ["param1", "param2", ...])`
4. If no analytical Jacobian is provided, the engine falls back to `numerical_jacobian()` which is significantly slower

#### Key Tuning Parameters
- `max_ite` (default: 200): Max Levenberg-Marquardt iterations
- `xtol` (default: 1e-4): Relative parameter step tolerance
- `ftol` (default: 1e-4): Relative cost function tolerance
- `coef_noise`: Noise threshold coefficient (0 = disabled)

---

## Multivariate Analysis (MVA)

> **Deep-dive documentation**: See [`doc/dev_MVA.md`](dev_MVA.md)

### Architecture
- **Model**: `model/m_mva.py` — `MMVA` class with pure NumPy PCA (SVD) and NMF (multiplicative update rules)
- **ViewModel**: `viewmodel/vm_mva.py` — Orchestrates data extraction, computation, caching
- **View**: `view/components/v_mva.py` — UI with parameter controls and matplotlib plots

### Implemented Methods
- **PCA**: `numpy.linalg.svd` on mean-centered data → scree plot, loadings, scores
- **NMF**: Lee & Seung multiplicative update rules → loadings (endmembers), scores (concentrations)

### Data Pipeline
1. Operates on **active** (checked) spectra
2. Uses preprocessed `spectrum.y` (baseline-subtracted, normalized, cropped)
3. Interpolates onto common grid if x-axes differ
4. Results exportable to Graphs workspace

---

## Quick Calculators (`model/m_quick_calc.py`)

Three built-in calculators accessible via Tools menu:

| Calculator | Purpose | Key Formulas |
|-----------|---------|-------------|
| **Spot Size** | Laser spot size, DOF, power density | `spot = 1.22λ/NA`, `DOF = 4nλ/NA²` |
| **Penetration Depth** | Optical penetration depth | `d = λ/(4πk)`, `α = 4πk/λ` |
| **Unit Converter** | Wavelength ↔ Energy ↔ Wavenumber, Raman shift | `E = 1239.84/λ`, `ν = 10⁷/λ` |

---

## Graph Customization Dialog (Singleton Pattern)

The `CustomizeGraphDialog` follows a singleton pattern managed by `VWorkspaceGraphs`:

### Signal Flow
```
User clicks "Customize" on VGraph
    → VGraph emits customize_requested(graph_id)
        → VWorkspaceGraphs._show_or_switch_customize_dialog(graph_id)
            → Creates dialog (first time) or calls dialog.switch_graph(widget, id)

User clicks different MDI subwindow
    → VWorkspaceGraphs._on_subwindow_activated()
        → If dialog is visible → dialog.switch_graph(new_widget, new_id)
```

### Key Methods
- `CustomizeGraphDialog.switch_graph(graph_widget, graph_id)` — re-binds all child widgets
- `CustomizeLegend.switch_graph(graph_widget)` — reloads legend properties
- `CustomizeAnnotations.switch_graph(graph_widget)` — disconnects/reconnects signals, reloads
- `CustomizeAxis.switch_graph(graph_widget)` — reloads axis settings

### Cleanup
- Closing the active graph → closes dialog
- Delete all / clear workspace → closes and nullifies dialog

---

## Cross-Workspace Communication

### Dependency Injection Pattern
- **Where**: `main.py` → `setup_connections()`
- **Purpose**: Enable workspaces to communicate without tight coupling

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

### Cross-Workspace Features

| Feature | From → To | Mechanism |
|---------|-----------|-----------|
| Profile extraction | Maps → Graphs | `vm_maps.extract_and_send_profile_to_graphs()` |
| Fit results to plot | Spectra/Maps → Graphs | `send_df_to_graphs` signal |
| Send spectra | Maps → Spectra | `send_spectra_to_workspace` signal |
| Tab switching | Maps → Main | `switch_to_graphs_tab` signal |

### Computed Columns (Fit Results)

> **Detailed documentation**: See [`doc/computed_columns_feature.md`](computed_columns_feature.md)

Users can add computed columns to fit results tables using `pandas.DataFrame.eval()`. Supports mathematical expressions referencing existing columns (e.g., `area_p1 / area_p2`). Special characters in column names require backtick quoting.

---

## File Formats

### Input Formats
- **Spectra**: 
  - `.csv`, `.txt` (2 columns: wavelength, intensity)
  - `.wdf` (Renishaw WiRE files) — includes metadata
  - `.spc` (Galactic SPC files) — includes metadata
  - `.dat` (Time-Resolved Photoluminescence - TRPL)
- **Maps**: 
  - `.csv`, `.txt` (first 2 columns: X, Y; remaining columns: wavelengths)
  - `.wdf` (Renishaw WiRE mapping files)
  - `.spc` (Galactic SPC mapping files)
- **DataFrames**: `.xlsx`, `.csv` (any tabular data)

### Workspace Save Files
- **`.spectra`**: JSON with compressed spectra data and fit models
- **`.maps`**: JSON with numpy binary-compressed map data (v2 format), spectra, metadata, and fit results
- **`.graphs`**: JSON with DataFrames and plot configurations

### Fit Models
- **`.json`**: User-defined peak models (baseline + peaks)
- **Location**: User-configurable folder in Settings

---

## Threading for Long Operations

### Pattern
1. ViewModel creates thread with worker function
2. Thread emits progress signals
3. ViewModel forwards signals to View
4. View updates progress bar

### Thread Classes

| Thread | Location | Purpose |
|--------|----------|---------|
| `TensorFitThread` | `fit_engine/tensor_fit_thread.py` | High-performance tensor fitting (default for Maps) |
| `FitThread` | `viewmodel/utils.py` | Legacy per-spectrum fitting (fitspy/lmfit) |
| `ApplyFitModelThread` | `viewmodel/utils.py` | Applies fit model to spectra in background |

### macOS Stack Size Note
`TensorFitThread` sets stack size to 8MB on macOS (`setStackSize(8 * 1024 * 1024)`) because macOS defaults to 512KB for QThread, which can cause segfaults when LAPACK allocates large arrays during batched tensor operations.

---

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
- **PyInstaller**: See `main.spec` for standalone binary configuration

---

## Configuration & Constants

### Settings Persistence
- **Storage**: `QSettings("CEA-Leti", "SPECTROview")`
- **Model**: `model/m_settings.py`
- **ViewModel**: `viewmodel/vm_settings.py`
- **View**: `view/components/v_settings.py`
- **Settings**: Theme, default paths, plot styles, fit parameters (xtol, ftol, max_ite, coef_noise), etc.

---

## External Dependencies

| Package | Version Constraint | Purpose |
|---------|-------------------|---------|
| `fitspy` | `< 2026.4` | Core fitting engine (Spectrum/Spectra base classes) |
| `lmfit` | — | Nonlinear least-squares minimization (legacy engine) |
| `PySide6` | — | Qt bindings for Python (**not PyQt5/PyQt6**) |
| `matplotlib` | `< 3.10.9` | Plotting backend |
| `seaborn` | `< 0.13.3` | Statistical plotting |
| `numpy` | `< 2.0.0` | Numerical operations, tensor engine core |
| `pandas` | — | DataFrame handling |
| `scipy` | — | Scientific computing (interpolation, optimization, KDTree) |
| `openpyxl` | `>= 3.1.5` | Excel file support |
| `superqt` | — | Enhanced Qt widgets (e.g., `QLabeledDoubleRangeSlider`) |
| `renishawwire` | — | Reading Renishaw WDF files |
| `pywin32` | Windows only | Windows-specific functionality |

---

## Common Pitfalls & Best Practices

### ❌ Don't:
- Mix Qt frameworks (this is **PySide6**, not PyQt5/PyQt6)
- Put business logic in View classes
- Call View methods directly from ViewModel
- Forget to connect signals (missing connection = UI won't update)
- Use synchronous operations for long-running tasks (always use QThread)
- Use raw strings for file paths (use `Path` from `pathlib`)
- Import View classes in ViewModel files
- Create per-widget dialog instances when a singleton pattern is appropriate

### ✅ Do:
- Follow MVVM file naming convention (`v_*.py`, `vm_*.py`, `m_*.py`)
- Use signals for all ViewModel → View communication
- Use dependency injection for cross-workspace communication
- Block signals during programmatic UI updates (`blockSignals(True)`)
- Use `QTimer.singleShot()` for deferred UI updates
- Cache expensive computations (e.g., griddata in map viewer)
- Validate user input in ViewModel before processing
- Emit informative notification signals for user feedback
- Use analytical Jacobians for peak models in the tensor engine
- Disconnect signals before re-binding in `switch_graph()` patterns

---

## Key Implementation Patterns

### 1. Loading Files
```
User → Open Files → main.py categorizes by extension
         ↓
    Routes to workspace (e.g., .txt → Maps if >3 columns, Spectra if 2 columns)
         ↓
    ViewModel.load_files(paths) → Model updates → Emit signals → View updates
```

### 2. Fitting Workflow (Tensor Engine)
```
User → Click Fit → ViewModel extracts fit_model from reference spectrum
         ↓
    Creates TensorFitThread with spectra + fit_model
         ↓
    TensorFittingEngine.fit_spectra():
        1. Apply model → 2. Preprocess → 3. Build evaluator
        4. Extract Y matrix → 5. Build p0 → 6. Batched LM optimize
        7. Write back results
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

### 4. Singleton Dialog Pattern
```
User → Click "Customize" on any graph → VGraph emits customize_requested(graph_id)
         ↓
    VWorkspaceGraphs._show_or_switch_customize_dialog(graph_id)
         ↓
    If dialog is None → Create new CustomizeGraphDialog
    If dialog exists → dialog.switch_graph(widget, id)
         ↓
    User switches MDI subwindow → _on_subwindow_activated auto-switches dialog
```

---

## Resources & References

### Main Entry Points
- **Application**: `spectroview/main.py` → `Main` class
- **Launcher**: `spectroview/main.py` → `launcher()` function

### MVVM Examples
- **Spectra Workspace**: `view/v_workspace_spectra.py` + `viewmodel/vm_workspace_spectra.py`
- **Maps Workspace**: `view/v_workspace_maps.py` + `viewmodel/vm_workspace_maps.py`
- **Graphs Workspace**: `view/v_workspace_graphs.py` + `viewmodel/vm_workspace_graphs.py`

### Deep-Dive Documentation
- **Tensor Fit Engine**: [`doc/dev_Tensor_Fit_Engine.md`](dev_Tensor_Fit_Engine.md)
- **MVA Feature**: [`doc/dev_MVA.md`](dev_MVA.md)
- **Computed Columns**: [`doc/computed_columns_feature.md`](computed_columns_feature.md)

---

## Debugging Tips

- **Signal not received?** Check connections in `setup_connections()` or `_connect_signals()`
- **UI frozen?** Long operation running on main thread → move to QThread
- **Plot not updating?** Check if signals are blocked or cache needs clearing
- **Import errors?** Verify MVVM layer separation (no View imports in ViewModel)
- **Threading issues?** Ensure thread-safe operations (use signals for cross-thread communication)
- **Tensor engine segfault on macOS?** Check stack size setting in `TensorFitThread`
- **Map data not refreshing?** Clear griddata cache via `clear_map_cache_requested` signal
- **Dialog shows wrong graph?** Check `switch_graph()` reconnects signals properly
