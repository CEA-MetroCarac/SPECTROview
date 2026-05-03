# Maps Workspace

The Maps Workspace handles hyperspectral datasets (2D maps and wafer maps). It extends all Spectra Workspace capabilities with map-specific features.

<!-- TODO: Add screenshot of Maps Workspace overview -->

## Loading Maps

Load hyperspectral map files via the **Open** button. Supported formats: `.csv`, `.txt`, `.wdf`, `.spc`.

Multiple maps can be loaded simultaneously. Controls:

- **Select** a map from the list to display
- **Delete** a map
- **Save** individual maps to Excel

## Map Viewer

Interactive heatmap visualization of fitted parameters.

| Feature | Description |
|---------|-------------|
| **Parameter selection** | Choose parameter to display (dropdown) |
| **Click-to-select** | Click map point to select corresponding spectrum |
| **Shift+Click** | Select multiple spectra |
| **Select All** | Select all spectra in the map |
| **Profile extraction** | Select 2 points → extract intensity profile → Graphs workspace |
| **Mask** | Filter heatmap by user-defined conditions |
| **Copy** | Copy heatmap to clipboard |
| **Multiple viewers** | Add floating windows for side-by-side comparison |

## Map Types

| Type | Description |
|------|-------------|
| **2D map** | Standard rectangular 2D scan |
| **Wafer 300mm** | 300mm wafer with zone/quadrant classification |
| **Wafer 200mm** | 200mm wafer |
| **Wafer 100mm** | 100mm wafer |

## Fitting Maps

Maps use the **[Tensor Fit Engine](fitting.md)** for high-performance fitting:

1. Define peaks and baseline on a representative spectrum
2. Click **Fit** (selected) or **Ctrl+Fit** (all active)
3. Fitting runs simultaneously for all spectra (typically < 3 seconds)
4. Collect fit results and visualize on the map viewer

!!! tip "Re-fitting"
    Click Fit again after adjusting parameters. The engine warm-starts from previous results for faster convergence.

## Cross-Workspace Features

| Action | Description |
|--------|-------------|
| **Send to Spectra** | Send selected spectra to Spectra workspace for comparison |
| **Send to Graphs** | Transfer fit results DataFrame for visualization |
| **Profile → Graphs** | Extract profile line and send to Graphs workspace |
