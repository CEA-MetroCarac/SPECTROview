# Graphs Workspace

The Graphs Workspace is dedicated to data visualization, with emphasis on simplicity and speed.

<!-- TODO: Add screenshot of Graphs Workspace -->

## Loading Data

Datasets can be loaded from:

- **Other workspaces**: Fit results sent from Spectra or Maps
- **Excel files** (`.xlsx`): Each worksheet becomes an independent dataset
- **CSV files** (`.csv`)

Dataset controls:

| Button | Action |
|--------|--------|
| **View** | Display dataset in a table |
| **Delete** | Remove dataset |
| **Save** | Save to file |
| **Refresh** | Reload from original file (useful after external edits) |

## Adding a New Plot

1. Select a **dataset** from the list
2. Choose **X, Y, Z columns** from dropdown menus
3. Select **plot style**: scatter, point, bar, box, line, trendline, 2Dmap, wafer
4. (Optional) Set axis labels and title — can be changed later via "Update"
5. (Optional) Set axis limits and Z-range
6. For wafer plots: select wafer size (100, 200, 300 mm)
7. Click **"Add Plot"**

## Modifying Plots

Each plot is an MDI subwindow with toolbar buttons:

| Button | Description |
|--------|-------------|
| **Customize** | Open the Customize Dialog |
| **Copy** | Copy figure to clipboard |

### Customize Dialog

The dialog is a **singleton** — it automatically switches to the active plot when you click a different subwindow.

| Tab | Controls |
|-----|---------|
| **Legend** | Labels, colors, markers, line styles, positions |
| **Annotations** | Vertical/horizontal lines, text annotations |
| **Axis** | Labels, limits, ticks, rotation, grid, broken axis |
| **General** | DPI, figure size, export |

**Interactive editing**:

- Double-click **legend box** → edit properties
- Double-click **annotation** → edit content
- Modify settings in **Control Panel** → click **"Update"**

## Data Filtering

Filter datasets before plotting using pandas query syntax:

**Syntax**: `column_name operator value`

| Filter | Meaning |
|--------|---------|
| `Confocal != "high"` | Exclude "high" confocal values |
| `Thickness == "1ML" or Thickness == "3ML"` | Select specific thicknesses |
| `a3_LOM >= 1000` | Values ≥ 1000 |
| `` `Laser Power` <= 5 `` | Backticks for column names with spaces |

!!! info "Logical Operators"
    Combine conditions with `and`, `or`, `not`:
    ```
    Thickness == "1ML" and `Laser Power` <= 5
    ```

## Annotations & Customization

Additional options in the **More Options** tab:

- Add/customize annotations (lines and text)
- Click legend box to customize per-curve annotation properties
- Set grid, DPI, axis rotations, figure export settings
