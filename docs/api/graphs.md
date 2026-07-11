# Data Visualization (Graphs)

The `spectroview.api.graphs` module allows you to programmatically generate publication-quality plots that perfectly replicate the native aesthetic of the SPECTROview Graphs workspace. It provides tailored wrappers around `seaborn` and `matplotlib`.

---

## 1. Preparing Data

Graphing functions expect a standard Pandas DataFrame. This DataFrame can be the output of your fit results (exported via `io.export_results` or saved directly).

```python
import pandas as pd
from spectroview.api import io

# Load fitted data results
df = io.load_dataset("fit_results.csv")
print(df.columns)
# Example output: ['Sample', 'Condition', 'peak_1_x0', 'peak_1_ampli']
```

---

## 2. Statistical Point Plots

The `plot_point` function creates statistical summaries (mean + 95% confidence interval error bars) grouped by categorical variables. This is excellent for comparing peak parameters across different experimental conditions.

```python
import matplotlib.pyplot as plt
from spectroview.api import graphs

# Create the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Plot Peak Position (x0) grouped by Sample, colored by Condition
graphs.plot_point(
    df=df, 
    x="Sample", 
    y="peak_1_x0", 
    hue="Condition",
    join=False,      # Set True to connect the means with lines
    dodge=True,      # Offsets the error bars to prevent overlapping
    title="Peak Shift by Process Condition",
    ax=ax
)

plt.tight_layout()
plt.savefig("point_plot.png", dpi=300)
plt.show()
```

---

## 3. Box Plots

Box plots provide a detailed view of the statistical distribution (median, quartiles, and outliers).

```python
fig, ax = plt.subplots(figsize=(8, 6))

graphs.plot_box(
    df=df, 
    x="Sample", 
    y="peak_1_fwhm", 
    hue="Condition",
    title="FWHM Distribution Analysis",
    ax=ax
)

plt.tight_layout()
plt.show()
```

---

## 4. Scatter Plots & Trendlines

Scatter plots visualize the correlation between two continuous parameters. You can overlay regression trendlines using `plot_trendline`.

```python
# Scatter Plot
fig, ax = plt.subplots(figsize=(8, 6))
graphs.plot_scatter(
    df=df,
    x="peak_1_ampli",
    y="peak_2_ampli",
    hue="Condition",
    title="Peak 1 vs Peak 2 Amplitude Correlation",
    ax=ax
)
plt.show()

# Trendline (Linear Regression)
fig, ax = plt.subplots(figsize=(8, 6))
graphs.plot_trendline(
    df=df,
    x="peak_1_x0",
    y="peak_1_fwhm",
    order=1,           # 1 for Linear, 2 for Quadratic, etc.
    hue="Condition",
    title="Correlation: FWHM vs Peak Position",
    ax=ax
)
plt.show()
```
