# Purpose

This file contains factual information about SPECTROview's capabilities. It is referenced when the AI Agent needs to answer questions about what the application can and cannot do.

---

# Supported File Formats

## Spectral Data

| Format | Extension | Description |
|--------|-----------|-------------|
| Text file | `.txt` | Two-column (x, y) plain text, various delimiters |
| CSV | `.csv` | Comma-separated values |
| Renishaw WDF | `.wdf` | Renishaw WiRE spectrometer format (single spectra and maps) |
| Galactic SPC | `.spc` | Galactic/Thermo Scientific GRAMS format |
| TRPL Data | `.txt` | Time-resolved photoluminescence data |

## Data Tables (Graphs Workspace)

| Format | Extension | Description |
|--------|-----------|-------------|
| Excel | `.xlsx`, `.xls` | Multi-sheet workbooks (each sheet → one DataFrame) |
| CSV | `.csv` | Single DataFrame |

## Project Files

| Format | Extension | Description |
|--------|-----------|-------------|
| SPECTROview project | `.specview` | Full workspace state including fits, graphs, settings |

---

# Workspaces

| Workspace | Tab | Description |
|-----------|-----|-------------|
| **File Loading** | First tab | Import and preview spectral data files |
| **Spectra** | Spectra tab | Interactive peak fitting, baseline correction, spectrum processing |
| **Graphs** | Graphs tab | DataFrame-based plotting with 9 plot styles and AI integration |
| **Maps** | Maps tab | 2D hyperspectral map visualisation and spatial analysis |
| **Spectra Store** | Store tab | Batch management of multiple spectra collections |
| **MVA** | MVA tab | Multivariate analysis (PCA, NMF) on spectral datasets |

---

# Plot Capabilities (Graphs Workspace)

## Supported Plot Styles

| Style | String | Use case |
|-------|--------|----------|
| Point | `point` | Statistical point plot with error bars (95% CI) |
| Scatter | `scatter` | Individual data points, optional color grouping |
| Box | `box` | Box-and-whisker distribution by category |
| Bar | `bar` | Bar chart with optional error bars |
| Line | `line` | Connected line plot, sorted by X |
| Trendline | `trendline` | Polynomial regression fit (configurable order) |
| Histogram | `histogram` | Distribution histogram with optional KDE |
| Wafer | `wafer` | Semiconductor wafer spatial map |
| 2D Map | `2Dmap` | General 2D heatmap (X column, Y row, Z value) |

## Color Palettes

Available palettes: `jet` (default), `viridis`, `plasma`, `magma`, `cividis`, `cool`, `hot`, `YlGnBu`, `YlOrRd`, `seismic`, `bwr`, `Spectral`

## Plot Properties (Configurable)

- Axis labels and limits (X, Y, Z)
- Log scale (X and Y axes independently)
- Color palette; colormap normalization for wafer/2D map (`colormap_norm`:
  linear/log/centered, `colormap_center` for the centered mode)
- Legend visibility and placement
- Scatter point size
- Trendline polynomial order
- Histogram bin count and KDE overlay
- Plot dimensions (width, height in pixels) and DPI
- Rotation of X-axis tick labels
- Filters (pandas `.query()` expressions)
- Annotations: vline, hline, text, arrow, vspan/hspan (shaded region), box
  (rectangle), callout (text + arrow to a point) — draggable, editable via
  the Annotations tab of the Customize Graph dialog
- Broken axis (`axis_breaks`): skip an uninteresting range on X *or* Y (not
  both at once) via a real two-panel layout — works for every plot style
- Inset (zoom) axes: one optional inset per graph showing the same series
  at its own X/Y limits (`inset_enabled`, `inset_bounds`,
  `inset_xmin`/`xmax`/`ymin`/`ymax`, `inset_show_zoom_indicator`)
- Not exposed as named `plot_graph`/`update_graph` tool parameters (see AI
  Agent Tools below) — set via each tool's `other_properties` dict using
  the MGraph field names above

---

# Peak Models

Available in the Spectra workspace and VBF engine:

- **Gaussian** — symmetric bell curve
- **Lorentzian** — symmetric Lorentzian (Cauchy) peak
- **PseudoVoigt** — linear combination of Gaussian and Lorentzian (α mixing parameter)
- **GaussianAsym** — asymmetric Gaussian with separate left/right FWHM
- **LorentzianAsym** — asymmetric Lorentzian with separate left/right FWHM
- **Fano** — Fano resonance profile (interference asymmetry parameter q)
- **DecaySingleExp** — single exponential decay A·exp(−t/τ) + B
- **DecayBiExp** — bi-exponential decay A₁·exp(−t/τ₁) + A₂·exp(−t/τ₂) + B

---

# Export Capabilities

- **Graph export**: PNG, TIFF, SVG, PDF, EPS with DPI, transparent
  background, physical size (mm/in, journal presets), and an export-time
  theme override, via each graph's own Export dialog
- **Batch graph export**: "Export All" exports every open graph to a
  folder in one pass
- **Multi-panel figure composer**: "Compose Figure" combines several open
  graphs into one exported figure (a grid of subplots), with optional
  shared axis labels (hide interior tick/axis labels) and panel labels
  (a/b/c, A/B/C, i/ii/iii, or numeric)
- **Table export**: Copy to clipboard as TSV from the AI chat DataFrame preview
- **Project export**: Save full workspace as `.specview` file
- **Batch fit results**: Export VBF results as CSV or Excel

---

# AI Agent Tools

The AI Agent supports five tools, called via native function/tool calling:

| Tool | Description |
|------|-------------|
| `query_dataframe` | Filter rows or evaluate a pandas expression (including aggregations) and return the result |
| `get_statistics` | Run `df.describe()` on specified columns |
| `plot_graph` | Create one or more new graphs in the Graphs workspace |
| `update_graph` | Modify an existing graph's properties by ID |
| `delete_graph` | Close one or more open graphs |

For a general question that doesn't require a data operation, reply with plain text instead of calling a tool.
