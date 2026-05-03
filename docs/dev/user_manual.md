# SPECTROview User Manual

**Last updated: May 2026**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Supported Data Formats](#3-supported-data-formats)
4. [User Interface Overview](#4-user-interface-overview)
5. [Spectra Workspace](#5-spectra-workspace)
6. [Maps Workspace](#6-maps-workspace)
7. [Graphs Workspace](#7-graphs-workspace)
8. [Tensor Fit Engine](#8-tensor-fit-engine)
9. [Multivariate Analysis (MVA)](#9-multivariate-analysis-mva)
10. [Quick Calculators](#10-quick-calculators)
11. [Settings & Preferences](#11-settings--preferences)
12. [Save & Load Workspace](#12-save--load-workspace)
13. [Keyboard Shortcuts & Tips](#13-keyboard-shortcuts--tips)

---

## 1. Introduction

Spectroscopy techniques (such as Raman, Photoluminescence, XRD, XPS, etc.) are widely used in various fields, including materials science, chemistry, biology, and geology. In recent years, these techniques have increasingly found their place in cleanroom environments, particularly within the microelectronics industry, where they serve as critical metrology tools for wafer-scale measurements.

The data collected from these in-line measurements (wafer data) require specific processing, but existing software solutions are often not optimized for this type of data and typically lack advanced plotting and visualization capabilities. Additionally, the licensing requirements of these software solutions can restrict access for a broader community of users.

**SPECTROview** addresses these gaps by offering free, open-source software that is compatible with both in-line data (wafer maps) as well as standard spectroscopic data (discrete spectra, 2D maps). It also features a built-in visualization tool, enabling users to streamline both data processing and visualization in a single application.

> **GitHub Repository**: [https://github.com/CEA-MetroCarac/SPECTROview](https://github.com/CEA-MetroCarac/SPECTROview)

> **Citation**: Le, V.-H., & Quéméré, P. (2025). SPECTROview: A Tool for Spectroscopic Data Processing and Visualization. Zenodo. [https://doi.org/10.5281/zenodo.14147172](https://doi.org/10.5281/zenodo.14147172)

---

## 2. Installation

### Requirements
- **Python**: Version 3.8 to 3.12

### From PyPI (recommended)
```bash
pip install spectroview
```

### From GitHub (latest development version)
```bash
pip install git+https://github.com/CEA-MetroCarac/SPECTROview.git
```

### Launch SPECTROview
```bash
spectroview
```

---

## 3. Supported Data Formats

Example files for all supported formats can be found in the [`/examples`](https://github.com/CEA-MetroCarac/SPECTROview/tree/main/examples) folder of the GitHub repository.

### 3.1 .wdf (Renishaw) and .spc (HORIBA) Formats

SPECTROview natively supports:
- **`.wdf`** — Recorded with WiRE software from Renishaw instruments
- **`.spc`** — Recorded with LabSpec 6 software from HORIBA instruments

Both discrete spectra and hyperspectral datasets (2D maps) are fully supported. These files can be opened directly without any prior conversion.

In addition to spectroscopic data, SPECTROview reads and imports associated **metadata**, including:
- Excitation wavelength (nm)
- Gratings (gr/mm)
- Objective used
- Laser power
- Slit/hole
- Acquisition timestamp

### 3.2 Spectroscopic Data (.txt, .csv)

SPECTROview supports spectroscopic data in TXT or CSV format. Files must consist of **two columns** separated by semicolons, spaces, or tabs:

| Column | Content |
|--------|---------|
| Column 1 | X-axis values (e.g., Raman shift in cm⁻¹, wavelength in nm) |
| Column 2 | Corresponding intensity values |

Files can contain column headers or not. Multiple spectrum files can be loaded simultaneously.

### 3.3 Hyperspectral Data (.txt, .csv)

For 2D maps or wafer maps, files require **X, Y coordinate columns** plus spectral data:

| Column | Content |
|--------|---------|
| Column 1 | X position |
| Column 2 | Y position |
| Columns 3...N | Intensity values at each wavelength/wavenumber |

The header row should contain the wavenumber/wavelength values as column names.

### 3.4 Datasheet (Excel or CSV)

Excel files (`.xlsx`, `.xls`) containing one or multiple sheets, or CSV files can be directly loaded into the **Graphs Workspace** for visualization.

### 3.5 TRPL Data (.dat)

Time-resolved photoluminescence (TRPL) data files are supported in `.dat` format.

### 3.6 Files Saved by SPECTROview

Depending on the active workspace, SPECTROview saves files with specific extensions so work can be resumed later:

| Workspace | Extension |
|-----------|-----------|
| Maps | `.maps` |
| Spectra | `.spectra` |
| Graphs | `.graphs` |

---

## 4. User Interface Overview

The SPECTROview application features three main workspaces, each designed for a specific purpose:

| Workspace | Purpose |
|-----------|---------|
| **Spectra** | Processing one or multiple discrete spectra |
| **Maps** | Processing one or multiple hyperspectral datasets (2D maps, wafer data) |
| **Graphs** | Plotting and visualizing data |

<!-- TODO: Add screenshot of main application window -->

### Toolbar

A horizontal toolbar is located at the top of the application with the following buttons:

| Button | Description |
|--------|-------------|
| **Open** | Loads all supported data types. The application automatically switches to the appropriate workspace based on the file type |
| **Save** | Saves the current active workspace to a file (`.maps`, `.spectra`, or `.graphs`) |
| **User Manual** | Opens this user manual |
| **Settings** | Opens the Settings dialog |
| **Dark/Light Mode** | Toggles between dark and light themes |
| **Quick Calc** | Opens the Quick Calculators dialog (laser spot size, penetration depth, unit converter) |

### Tooltips

Most GUI elements in SPECTROview (buttons, text boxes, dropdowns, etc.) feature **tooltips**. Hover the mouse cursor over any element for 1 second to see a brief explanation of its function.

---

## 5. Spectra Workspace

The Spectra Workspace is designed for processing individual or multiple discrete spectra.

<!-- TODO: Add screenshot of Spectra Workspace overview -->

### 5.1 Spectra List & Progress Bar

The **Spectra List** panel (left side) displays all loaded spectra with checkboxes:

- **Checkboxes** ☑: Enable/disable spectra for batch operations. Only checked (active) spectra are included in "Apply All" operations
- **Selection**: Click to select one spectrum, Shift+Click for range selection, Ctrl+Click for multi-selection
- **Color indicators**: Visual indicators show the processing state (raw, baseline subtracted, fitted)
- **Drag & drop**: Reorder spectra by dragging
- **Right-click**: Context menu for operations on selected spectra

**Progress Bar**: Located under the Spectra List, displays:
- Number of available spectra
- Fitting progress (fitted / total, percentage, elapsed time)
- **Stop** button to cancel fitting at any time

### 5.2 Spectra Viewer

The Spectra Viewer plots all spectra (and their best-fit curves) selected via the Spectra List.

<!-- TODO: Add screenshot of Spectra Viewer -->

#### Toolbar Buttons

| Button | Description |
|--------|-------------|
| **Rescale** | Rescale the spectra plot. Shortcut: `Ctrl+R` |
| **Zoom** | When active, enables zoom using left mouse click & drag |
| **Baseline** | When active, allows defining baseline anchor points using left mouse click |
| **Peaks** | When active, allows defining peaks on spectra using left mouse click |

#### View Options

| Option | Description |
|--------|-------------|
| **Normalization** | Normalize all selected spectra to maximum peak intensity. Type spectral range into 'min' and 'max' fields to normalize to a specific range |
| **Legend** | Display legend for selected spectra. Click on legend box to change color or labels (requires Zoom to be disabled) |
| **Copy** | Copy spectra plot to clipboard as image. `Ctrl+Click` (or `Cmd+Click` on macOS): copies numerical data to clipboard |
| **More Options** | Opens additional view options (X/Y axis units, log scale, spectrum plot style, broken axis) |

#### More Options Panel

| Setting | Options |
|---------|---------|
| **X-axis unit** | Wavenumber (cm⁻¹), Wavelength (nm), Emission energy (eV), Binding energy (eV), Frequency (Hz), 2θ (°), Time (ns) |
| **Y-axis unit** | Intensity (a.u.), Intensity (a.u./s) |
| **Y axis-scale** | Linear or Logarithmic |
| **Spectrum plot style** | Line, scatter, or point |
| **Broken axis** | X-axis and Y-axis break ranges |

#### Mouse Interactions

| Action | Effect |
|--------|--------|
| **Left click** (Baseline mode) | Add baseline anchor point |
| **Left click** (Peaks mode) | Add a new peak |
| **Right click** (Peaks mode) | Remove nearest peak |
| **Mouse wheel** | Quick rescale of Y-axis (intensity scale) |
| **Click & drag on peak** | Adjust peak position/height interactively |

### 5.3 Fit Model Builder

The Fit Model Builder tab contains three main panels: **Fitting**, **Peak Table**, and **Fit Model Control**.

<!-- TODO: Add screenshot of Fit Model Builder -->

#### 5.3.1 Fitting Panel

The Fitting Panel is divided into four sections, each corresponding to key steps in the fitting process:

**Step 1: X-axis Correction** (optional)

Used to perform x-axis correction based on a well-known reference sample (currently implemented for Silicon with a theoretical Raman peak at 520.7 cm⁻¹).

Procedure:
1. Record the silicon reference spectrum during the same experimental session
2. Fit the Si-Ref spectrum to determine the measured peak position
3. Enter the measured position into the text box
4. Select spectra and apply the correction

**Step 2: Spectral Range**

Define the spectral range for analysis by entering minimum and maximum values. Click "Apply" to crop the spectrum to the defined range.

**Step 3: Baseline**

Two modes available:

| Mode | Sub-options | Description |
|------|-------------|-------------|
| **Manual** | Linear, Polynomial | Define baseline anchor points by clicking in the Spectra Viewer. Options: *Attached* (points snap to spectrum), *Correct noise* (average over neighboring points) |
| **Auto** | airPLS, asLS | Automatic baseline estimation. Adjust the slider to adapt the baseline curve |

- **airPLS** (Adaptive Iteratively Reweighted PLS): Aggressive method, forces baseline to the bottom. Good for high-fluorescence samples. Developed and validated for Raman data.
- **asLS** (Asymmetric Least Squares): More conservative/flexible approach.

**Step 4: Peak Model**

Select peak shapes and add peaks by clicking on the spectrum:

| Model | Parameters | Use Case |
|-------|-----------|----------|
| Gaussian | ampli, fwhm, x0 | Symmetric Gaussian peaks |
| Lorentzian | ampli, fwhm, x0 | Symmetric Lorentzian peaks |
| PseudoVoigt | ampli, fwhm, x0, alpha | Mixed Gaussian + Lorentzian |
| GaussianAsym | ampli, fwhm_l, fwhm_r, x0 | Asymmetric Gaussian |
| LorentzianAsym | ampli, fwhm_l, fwhm_r, x0 | Asymmetric Lorentzian |
| Fano | ampli, fwhm, x0, q | Fano resonance |
| DecaySingleExp | A, tau, B | Single exponential decay (TRPL) |
| DecayBiExp | A1, tau1, A2, tau2, B | Bi-exponential decay (TRPL) |

#### 5.3.2 Peak Table Panel

The Peak Table displays all peaks defined for the selected spectrum with dynamically updated properties: **label**, **model**, **position**, **FWHM**, **intensity**.

<!-- TODO: Add screenshot of Peak Table -->

**Constraints** — For each peak, parameters can be constrained:

| Constraint Type | Description |
|----------------|-------------|
| **Fix** checkbox | Keep parameter constant at its current value |
| **Limits** checkbox | Restrict parameter within min/max range |
| **Expression** checkbox | Link parameters via mathematical expressions |

**Expression Examples**:
- Constrain peak position: `m01_x0 - 17` (peak m02 is 17 units less than m01)
- Link intensities: `m02_ampli / 2` (peak m03 intensity is half of m02)
- Link FWHM: `m01_fwhm` (peak m02 shares same width as m01)

#### 5.3.3 Fit Model Control Panel

| Button | Action |
|--------|--------|
| **Fit** | Start fitting the selected spectra |
| **Ctrl+Fit** | Fit all active (checked) spectra |
| **Copy Fit Model** | Copy the entire fit model (spectral range, baseline, peaks) |
| **Paste Fit Model** | Paste fit model to selected spectra |
| **Save Fit Model** | Save fit model to `.json` file |
| **Load/Apply Fit Model** | Load a saved model and apply to selected spectra |

### 5.4 Fit Results

Once spectra have been fitted, the results can be collected and analyzed:

#### 5.4.1 Collecting Results

Click **"Collect Fit Results"** to populate the fit results table with all fitted parameters from active spectra. The table includes:
- Filename
- Peak positions (x0), amplitudes (ampli), FWHM values
- Peak areas (calculated)
- Any additional model-specific parameters

#### 5.4.2 Computed Columns

Create new columns from mathematical expressions:

1. Enter a **Column Name** (e.g., `Peak_Ratio`)
2. Enter a **Mathematical Expression** (e.g., `area_p1 / area_p2`)
3. Click **"Compute & Add"**

Supported operations: `+`, `-`, `*`, `/`, `**`, `%`, `()`

> **Important**: If column names contain special characters like parentheses `()`, spaces, or hyphens, wrap them in backticks: `` `x0_LO(M)` ``

#### 5.4.3 Saving or Visualizing

- **Send to Graphs**: Name the dataset and click "Send" to transfer to the Graphs workspace for plotting
- **Export to Excel**: Save the fit results table as an Excel file

### 5.5 More Tab

The More Tab contains three main sections:

| Section | Content |
|---------|---------|
| **Left panel** | Metadata viewer showing all settings/parameters from loaded files (`.wdf` or `.spc` format only): excitation wavelength, gratings, objective, laser power, slit/hole, acquisition timestamp |
| **Middle panel** | Additional information about the selected spectrum |
| **Right panel** | Additional data processing features: intensity normalization, cosmic ray detection |

---

## 6. Maps Workspace

The Maps Workspace handles hyperspectral datasets (2D maps and wafer maps). It extends all Spectra Workspace capabilities with map-specific features.

<!-- TODO: Add screenshot of Maps Workspace overview -->

### 6.1 Loading Maps

Load hyperspectral map files through the **Open** button in the toolbar. Supported formats: `.csv`, `.txt`, `.wdf`, `.spc`.

Multiple maps can be loaded simultaneously. All loaded maps appear in a list, from which you can:
- Select a map to display
- Delete a map
- Save individual maps to Excel

### 6.2 Map Viewer

The Map Viewer displays an interactive heatmap of the selected fitted parameter.

<!-- TODO: Add screenshot of Map Viewer -->

**Key Features**:

| Feature | Description |
|---------|-------------|
| **Parameter selection** | Choose which fitted parameter to display (via dropdown after collecting fit results) |
| **Click-to-select** | Click on a map point to select the corresponding spectrum |
| **Shift+Click** | Select multiple spectra by clicking on multiple map points |
| **Select All** | Select all spectra in the map |
| **Profile extraction** | Select two points to extract an intensity profile line → sent to Graphs workspace |
| **Mask feature** | Define regions to filter the heatmap based on user-defined conditions |
| **Copy** | Copy the heatmap to clipboard |
| **Multiple viewers** | Add additional floating map viewer windows for side-by-side parameter comparison |

### 6.3 Map Types

| Type | Description |
|------|-------------|
| **2D map** | Standard rectangular 2D scan |
| **Wafer 300mm** | 300mm wafer map with zone/quadrant classification |
| **Wafer 200mm** | 200mm wafer map |
| **Wafer 100mm** | 100mm wafer map |

### 6.4 Spectra List (Map Mode)

When a map is selected, its spectra are displayed in the Spectra List. Features:
- Checkbox activation/deactivation for batch fitting
- Color indicators for processing state
- Spectrum selection synced with map viewer clicks

### 6.5 Fitting Maps

Maps use the **Tensor Fit Engine** for high-performance fitting:

1. Define peaks and baseline on a representative spectrum
2. Click **Fit** → fits selected spectra, or **Ctrl+Fit** → fits all active spectra
3. The Tensor Engine fits all spectra simultaneously (typically < 3 seconds for a full map)
4. After fitting, collect fit results and visualize on the map viewer

### 6.6 Sending Spectra to Other Workspaces

- **Send to Spectra Workspace**: Select spectra and send copies to the Spectra tab for detailed comparison
- **Send to Graphs Workspace**: Fit results DataFrame can be sent directly for plotting

---

## 7. Graphs Workspace

The Graphs Workspace is dedicated to data visualization, with emphasis on simplicity and speed.

<!-- TODO: Add screenshot of Graphs Workspace -->

### 7.1 Loading Data

Datasets (DataFrames) can be loaded from:
- **Other workspaces**: Fit results sent from Spectra or Maps workspaces
- **Excel files** (`.xlsx`): Each worksheet is loaded as an independent dataset
- **CSV files** (`.csv`)

All loaded datasets appear in a list. Controls:

| Button | Action |
|--------|--------|
| **View** | Display selected dataset in a table |
| **Delete** | Remove selected dataset |
| **Save** | Save selected dataset to file |
| **Refresh** | Reload dataset from original file (useful after external edits) |

### 7.2 Adding a New Plot

1. **Select a dataset** from the loaded datasets list
2. **Choose X, Y, Z columns** from the dropdown menus (populated from dataset columns)
3. **Select plot style**: scatter, point, bar, box, line, trendline, 2Dmap, wafer
4. **(Optional)** Set axis labels and plot title — these can also be set later via "Update"
5. **(Optional)** Set axis limits and Z-axis range (for heatmap types)
6. **For wafer plots**: Define wafer size (100, 200, 300 mm)
7. Click **"Add Plot"**

### 7.3 Modifying Existing Plots

Once a plot is added, it appears as a subwindow in the MDI area. Each plot has toolbar buttons:

| Button | Description |
|--------|-------------|
| **Customize** | Opens the Customize Dialog (Legend, Annotations, Axis, General settings) |
| **Copy** | Copy the figure to clipboard |

#### Customize Dialog

The Customize Dialog is a **singleton** — only one instance exists. It automatically switches to the active plot when you click a different subwindow.

| Tab | Controls |
|-----|---------|
| **Legend** | Modify legend labels, colors, markers, line styles, and positions for each curve |
| **Annotations** | Add vertical/horizontal lines and text annotations |
| **Axis** | Customize axis labels, limits, tick formatting, rotation, grid, broken axis ranges |
| **General** | DPI, figure size, and export options |

**Interactive editing**:
- Double-click on the **legend box** to edit its properties
- Double-click on any **annotation** to edit its content
- Change settings in the right **Control Panel** and click **"Update"** to apply

### 7.4 Data Filtering

Filter your dataset before plotting using the **Filter** text field:

**Syntax**: `column_name operator value`

| Operator | Meaning |
|----------|---------|
| `==` | Equal to |
| `!=` | Not equal to |
| `<`, `>`, `<=`, `>=` | Comparison |
| `and`, `or`, `not` | Logical operators |

**Examples**:

| Filter Expression | Meaning |
|------------------|---------|
| `Confocal != "high"` | Select all values where "Confocal" is not "high" |
| `Thickness == "1ML" or Thickness == "3ML"` | Select "1ML" or "3ML" thickness values |
| `a3_LOM >= 1000` | Select values ≥ 1000 in column "a3_LOM" |
| `` `Laser Power` <= 5 `` | Select values ≤ 5 (backticks required for column names with spaces) |

### 7.5 Annotations & Other Customization

Additional view and customization options are available in the **More Options** tab:
- Add/customize annotations (lines and text) for each curve
- Click on the legend box to customize annotation properties
- Set grid, DPI, axis rotations, and figure export settings

---

## 8. Tensor Fit Engine

### What is it?

The Tensor Fit Engine is SPECTROview's high-performance fitting backend that replaces the traditional per-spectrum fitting approach. Instead of fitting spectra one at a time, it **fits all spectra simultaneously** using batched matrix operations.

### Why is it faster?

| Traditional Approach | Tensor Engine |
|---------------------|---------------|
| Fits spectra one-by-one | Fits all N spectra at once |
| Numerical Jacobians (slow) | Analytical Jacobians (fast) |
| Python function call overhead | Vectorized NumPy/LAPACK operations |
| Sequential execution | Batched tensor math |

**Result**: Typically **10–15× faster**. A 1000-spectrum map that would take 30+ seconds now fits in < 3 seconds.

### How to use it

The Tensor Engine is used **automatically** — no special action needed. When you click **Fit** in either the Spectra or Maps workspace, SPECTROview uses the Tensor Engine by default.

### Tuning for Performance vs. Accuracy

Access fitting parameters in **Settings → Fit Parameters**:

| Setting | Default | Fast Preview | Precision Fitting |
|---------|---------|-------------|-------------------|
| `xtol` | 1e-4 | 1e-2 | 1e-6 |
| `ftol` | 1e-4 | 1e-2 | 1e-6 |
| `max_ite` | 200 | 50 | 500 |

---

## 9. Multivariate Analysis (MVA)

SPECTROview includes built-in Multivariate Analysis tools accessible from the **MVA tab** in the Spectra and Maps workspaces.

### Supported Methods

#### Principal Component Analysis (PCA)
- Reduces spectral data dimensions to reveal main patterns of variance
- **Visualizations**: Scree plot (variance per component), Loadings (spectral contributions), Scores (2D scatter of spectra)

#### Non-negative Matrix Factorization (NMF)
- Decomposes data into additive, non-negative components (endmembers)
- **Visualizations**: Loadings (endmember spectra), Scores (concentration/abundance map)

### How to Use

1. **Load spectra** in the Spectra or Maps workspace
2. **Check (activate)** the spectra you want to analyze
3. Open the **MVA tab**
4. Select method (**PCA** or **NMF**) and number of components
5. Click **Run**
6. View results in the embedded plots
7. **(Optional)** Export scores to the Graphs workspace for further visualization

### Data Preparation Notes
- MVA operates on **preprocessed data** (after baseline subtraction, normalization, cropping)
- If spectra have different x-axis ranges, they are **automatically interpolated** onto a common grid
- At least **2 active spectra** are required

---

## 10. Quick Calculators

Accessible from the **Quick Calc** button in the toolbar, or from **Tools → Quick Calculators**.

### 10.1 Laser Spot Size Calculator

Calculates the diffraction-limited spot size, depth of focus, and power density.

| Input | Description |
|-------|-------------|
| Laser Wavelength | λ in nm |
| Objective NA | Numerical aperture |
| Working Distance | WD in mm |
| Refractive Index | n (default: 1.0 for air) |
| Laser Power | Power in mW |

| Output | Formula |
|--------|---------|
| Spot Size | `1.22 × λ / NA` (μm) |
| Depth of Focus | `4 × n × λ / NA²` (μm) |
| Angle of View | `2 × arcsin(NA)` (°) |
| Lens Diameter | `2 × WD × tan(arcsin(NA))` (mm) |
| Power Density | `Power / Area` (kW/cm², mW/μm², W/m²) |

### 10.2 Penetration Depth Calculator

Calculates optical penetration depth from the extinction coefficient.

| Input | Description |
|-------|-------------|
| Laser Wavelength | λ in nm |
| Extinction Coefficient | k (dimensionless) |

| Output | Formula |
|--------|---------|
| Absorption Coefficient α | `4πk / (λ × 10⁻⁷)` (cm⁻¹) |
| Penetration Depth d | `λ / (4πk)` (nm) |

Reference values for k can be found at [refractiveindex.info](https://refractiveindex.info).

### 10.3 Unit Converter

Converts between spectroscopic units:

**Absolute Conversion**:
| From | To | Formula |
|------|-----|---------|
| Wavelength (nm) | Energy (eV) | `E = 1239.84193 / λ` |
| Wavelength (nm) | Wavenumber (cm⁻¹) | `ν = 10⁷ / λ` |

**Relative Conversion** (Raman Shift):
| Input | Output | Formula |
|-------|--------|---------|
| Laser λ₀ + Raman Shift Δω | Scattered λ | `λ = 10⁷ / (10⁷/λ₀ - Δω)` |

---

## 11. Settings & Preferences

Access settings via the **Settings** button in the toolbar.

### Available Settings

| Category | Options |
|----------|---------|
| **Theme** | Dark mode / Light mode |
| **Default Directory** | Default folder for file dialogs |
| **Fit Parameters** | xtol, ftol, max_ite, coef_noise |
| **Fit Model Path** | Default folder for saved fit models |
| **Peak Limits** | Minimum FWHM, maximum FWHM, maximum shift |

---

## 12. Save & Load Workspace

### Saving

Click the **Save** button in the toolbar to save the current workspace:

| Workspace | File Extension | Contents |
|-----------|---------------|----------|
| **Spectra** | `.spectra` | All loaded spectra, fit models, baseline settings, fit results |
| **Maps** | `.maps` | Map data (compressed), all spectra with fit results, metadata |
| **Graphs** | `.graphs` | All datasets and plot configurations |

### Loading

Click the **Open** button and select a previously saved workspace file. SPECTROview will automatically switch to the correct workspace tab and restore all data.

---

## 13. Keyboard Shortcuts & Tips

### Global Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+R` | Rescale spectra plot |
| `Ctrl+Click` on Copy | Copy numerical data to clipboard |
| `Mouse wheel` | Quick Y-axis rescale in Spectra Viewer |

### Tips

- **Tooltips everywhere**: Hover over any GUI element for 1 second to see its function
- **Drag peaks**: Click and drag peaks directly in the Spectra Viewer to adjust position and height
- **Quick re-fit**: After adjusting parameters, click Fit again — the engine warm-starts from previous results
- **Data filtering**: Use pandas-style expressions in the Graphs workspace to filter data before plotting
- **Column names with spaces**: Use backticks (`` ` ``) around column names containing special characters
- **Multiple maps**: Load and switch between multiple maps; each map's spectra are managed independently
- **Profile extraction**: In Maps workspace, select exactly 2 points to extract an intensity profile along the connecting line

---

*For additional help, feature requests, or to report issues, visit the [GitHub Issues page](https://github.com/CEA-MetroCarac/SPECTROview/issues).*
