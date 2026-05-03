# Spectra Workspace

The Spectra Workspace is designed for processing individual or multiple discrete spectra.

<!-- TODO: Add screenshot of Spectra Workspace overview -->

## Spectra List & Progress Bar

The **Spectra List** panel (left side) displays all loaded spectra with checkboxes:

- **Checkboxes** ☑: Enable/disable spectra for batch operations
- **Selection**: Click to select, Shift+Click for range, Ctrl+Click for multi-selection
- **Color indicators**: Visual state (raw, baseline subtracted, fitted)
- **Drag & drop**: Reorder spectra by dragging

**Progress Bar** (under the Spectra List):

- Spectra count
- Fitting progress: fitted / total, percentage, elapsed time
- **Stop** button to cancel fitting

## Spectra Viewer

Plots all selected spectra with their best-fit curves.

### Toolbar

| Button | Description | Shortcut |
|--------|-------------|----------|
| **Rescale** | Rescale the plot | `Ctrl+R` |
| **Zoom** | Enable zoom with left click & drag | |
| **Baseline** | Add baseline anchor points by clicking | |
| **Peaks** | Add peaks by clicking | |

### View Options

| Option | Description |
|--------|-------------|
| **Normalization** | Normalize to max intensity. Use min/max fields for specific range |
| **Legend** | Show/hide legend. Click legend box to edit (requires Zoom off) |
| **Copy** | Copy plot as image. `Ctrl+Click`: copy numerical data |
| **More Options** | X/Y units, log scale, plot style, broken axis |

### Mouse Interactions

| Action | Effect |
|--------|--------|
| Left click (Baseline mode) | Add baseline anchor point |
| Left click (Peaks mode) | Add a new peak |
| Right click (Peaks mode) | Remove nearest peak |
| Mouse wheel | Quick Y-axis rescale |
| Click & drag on peak | Adjust peak position/height |

## Fit Model Builder

Three panels: **Fitting**, **Peak Table**, **Fit Model Control**.

### Step 1: X-axis Correction (optional)

Correct the x-axis using a silicon reference sample (520.7 cm⁻¹):

1. Fit the Si reference spectrum
2. Enter the measured position
3. Select spectra and apply correction

### Step 2: Spectral Range

Enter min/max values and click **Apply** to crop the spectrum.

### Step 3: Baseline

=== "Manual"

    | Option | Description |
    |--------|-------------|
    | **Linear** | Linear interpolation between anchor points |
    | **Polynomial** | Polynomial fit through anchor points |
    | **Attached** | Points snap to the spectrum curve |
    | **Correct noise** | Average over neighboring points |

=== "Auto"

    | Option | Description |
    |--------|-------------|
    | **airPLS** | Aggressive, forces baseline to bottom. Good for fluorescence |
    | **asLS** | Conservative, flexible approach |

    Adjust the slider to adapt the baseline curve.

### Step 4: Peak Models

| Model | Parameters | Use Case |
|-------|-----------|----------|
| Gaussian | ampli, fwhm, x0 | Symmetric Gaussian |
| Lorentzian | ampli, fwhm, x0 | Symmetric Lorentzian |
| PseudoVoigt | ampli, fwhm, x0, alpha | Mixed Gaussian+Lorentzian |
| GaussianAsym | ampli, fwhm_l, fwhm_r, x0 | Asymmetric Gaussian |
| LorentzianAsym | ampli, fwhm_l, fwhm_r, x0 | Asymmetric Lorentzian |
| Fano | ampli, fwhm, x0, q | Fano resonance |
| DecaySingleExp | A, tau, B | Single exponential (TRPL) |
| DecayBiExp | A1, tau1, A2, tau2, B | Bi-exponential (TRPL) |

### Peak Table

Displays all peaks with dynamically updated properties. Constraint options:

| Constraint | Description |
|-----------|-------------|
| **Fix** | Keep parameter constant |
| **Limits** | Restrict to min/max range |
| **Expression** | Link parameters mathematically |

!!! example "Expression Examples"
    - Peak position: `m01_x0 - 17` (peak m02 is 17 units less than m01)
    - Link intensities: `m02_ampli / 2` (peak m03 is half of m02)
    - Link FWHM: `m01_fwhm` (peak m02 shares same width)

### Fit Model Control

| Button | Action |
|--------|--------|
| **Fit** | Fit selected spectra |
| **Ctrl+Fit** | Fit all active spectra |
| **Copy/Paste** | Transfer fit models between spectra |
| **Save/Load** | Persist fit models as `.json` files |

## Fit Results

### Collecting Results

Click **"Collect Fit Results"** to populate the results table with all fitted parameters.

### Computed Columns

Create derived columns from mathematical expressions:

1. Enter **Column Name** (e.g., `Peak_Ratio`)
2. Enter **Expression** (e.g., `area_p1 / area_p2`)
3. Click **"Compute & Add"**

!!! warning "Column Names with Special Characters"
    Wrap column names containing `()`, spaces, or `-` in backticks: `` `x0_LO(M)` ``

### Export

- **Send to Graphs**: Transfer to Graphs workspace for visualization
- **Export to Excel**: Save as `.xlsx` file

## More Tab

| Section | Content |
|---------|---------|
| **Metadata** | Settings from `.wdf`/`.spc` files (wavelength, gratings, etc.) |
| **Spectrum info** | Details about selected spectrum |
| **Processing** | Intensity normalization, cosmic ray detection |
