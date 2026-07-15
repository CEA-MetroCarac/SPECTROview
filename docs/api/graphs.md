# Data Visualization (Graphs)

The `spectroview.api.graphs` module lets you programmatically generate publication-quality plots that pixel-for-pixel replicate the native aesthetic of the SPECTROview Graphs workspace, for all 9 supported plot styles. It delegates to the exact same rendering code the GUI's own Graphs widget uses (no separate re-implementation, no approximation), so scripted plots genuinely match what the GUI draws — same color/marker cycling, same 95% CI computation, same box/point/trendline/wafer/2D-map styling.

---

## Preparing Data

Graphing functions expect a standard Pandas DataFrame — typically the output of `SpectraWorkspace.collect_results()` / `MapsWorkspace.collect_results()`, or loaded via `io.load_dataset()`.

```python
import pandas as pd
from spectroview.api import io

df = io.load_dataset("fit_results.csv")["fit_results"]
print(df.columns)
```

Every `plot_*` function accepts an optional `ax=` (draws onto it instead of creating a new figure) and returns the `Axes` it drew on, so you can compose subplots yourself.

---

## Categorical / XY Plots

### Statistical Point Plots

Mean + 95% confidence interval error bars grouped by a categorical variable — good for comparing peak parameters across experimental conditions.

```python
import matplotlib.pyplot as plt
from spectroview.api import graphs

fig, ax = plt.subplots(figsize=(8, 6))
graphs.plot_point(df, x="Sample", y="peak_1_x0", hue="Condition",
                   join=False, dodge=True, title="Peak Shift by Process Condition", ax=ax)
plt.tight_layout()
plt.savefig("point_plot.png", dpi=300)
```

### Line Plots

Like `plot_point`, but connects the per-x means with a line and a shaded 95% CI band — useful for a variable that has a natural order (e.g. depth, time, dose).

```python
graphs.plot_line(df, x="Depth_nm", y="peak_1_ampli", hue="Condition", title="Amplitude vs Depth")
```

### Bar Plots

```python
graphs.plot_bar(df, x="Sample", y="peak_1_ampli", hue="Condition", show_error_bar=True)
```

### Box Plots

```python
graphs.plot_box(df, x="Sample", y="peak_1_fwhm", hue="Condition", title="FWHM Distribution Analysis")
```

### Histograms

```python
graphs.plot_histogram(df, x="peak_1_fwhm", hue="Condition", bins=20, kde=True)
```

### Scatter Plots & Trendlines

```python
graphs.plot_scatter(df, x="peak_1_ampli", y="peak_2_ampli", hue="Condition",
                     title="Peak 1 vs Peak 2 Amplitude Correlation")

graphs.plot_trendline(df, x="peak_1_x0", y="peak_1_fwhm", order=1, hue="Condition",
                       title="Correlation: FWHM vs Peak Position")
```

---

## Spatial Plots

These take a tidy DataFrame with one row per (x, y) sample and a numeric value column `z` — exactly the shape `MapsWorkspace.collect_results()` produces.

### 2D Maps

Fast, exact heatmap on the acquisition's own regular grid (raster/2D-map acquisitions).

```python
graphs.plot_2dmap(df, x="X", y="Y", z="ampli_Si", cmap="jet", title="Amplitude Map")
```

### Wafer Maps

Interpolated heatmap over a circular wafer outline, with optional summary statistics annotated on the plot.

```python
graphs.plot_wafer(df, x="X", y="Y", z="ampli_Si", wafer_size=300.0, show_stats=True)
```

---

## Plot Templates

A plot template is a named, reusable set of plot configurations (each shaped like the dict `MGraph.save()` produces in the GUI) saved as a JSON file in a folder — the same mechanism the GUI's "Save as Template" feature uses.

```python
from spectroview.api import graphs

configs = [
    {"plot_style": "scatter", "x": "peak_1_ampli", "y": ["peak_2_ampli"], "z": "Condition"},
    {"plot_style": "box", "x": "Sample", "y": ["peak_1_fwhm"], "z": "Condition"},
]
template_id = graphs.save_plot_template("./templates", "Standard QC Plots", configs)

# List and reload later
for summary in graphs.list_plot_templates("./templates"):
    print(summary["name"], summary["graph_count"])

configs = graphs.load_plot_template("./templates", template_id)
graphs.delete_plot_template("./templates", template_id)
```
