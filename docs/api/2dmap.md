# Maps Workspace & 2D Analysis

This section demonstrates how to handle hyperspectral map data (like Raman wafer maps): loading, fitting every pixel, building heatmaps and line profiles, and running Multivariate Analysis — through the stateful `MapsWorkspace` session (recommended) and, for MVA, the lower-level array-based functions.

---

## Stateful Workspaces

`spectroview.api.workspace.MapsWorkspace` mirrors the GUI's own Maps Workspace: one map per loaded file, all of whose pixels share a fit model, with pixel identifiers named `f"{map_name}_({x}, {y})"` exactly like the GUI. It reads/writes the same `.maps` file format the GUI's "Save work" produces.

```python
from spectroview.api import workspace, fitting

ws = workspace.MapsWorkspace()          # or MapsWorkspace(map_type="wafer_300mm") for wafer maps
[map_name] = ws.load_files(["2Dmap_Si.txt"])

fit_model = fitting.load_fit_model_template("fit_model_Si.json")
ws.set_fit_model(fit_model, names=[map_name])
ws.fit(map_names=[map_name])            # one vectorized fit call over every pixel in the map

df = ws.collect_results()               # one row per pixel, includes X/Y coordinates
print(df.head())

# Build a heatmap of a fitted parameter
xi, yi, zi = ws.get_heatmap(map_name, "ampli_Si")

import matplotlib.pyplot as plt
plt.imshow(zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()], origin="lower", cmap="jet")
plt.colorbar(label="Amplitude")
plt.savefig("heatmap.png", dpi=200)

# Extract a line profile between two points on the heatmap
profile = ws.extract_profile(map_name, "ampli_Si", point1=(xi.min(), yi.min()), point2=(xi.max(), yi.max()))
print(profile)  # columns: X, Y, distance, values

# Save — this file opens directly in the SPECTROview GUI
ws.save("session.maps")
```

`get_heatmap(map_name, value_col, ...)` accepts `value_col="Intensity"` or `"Area"` (computed directly from the raw intensities, no fit required) or any fit-parameter column present in `df_fit_results` (call `collect_results()` first). For wafer map types (`map_type != "2Dmap"`), the grid is built by scattered-point interpolation over the wafer radius implied by `map_type`; for `"2Dmap"` it's a fast exact pivot on the acquisition's own regular grid.

Reload a session later (built by this API or saved from the GUI):

```python
ws2 = workspace.MapsWorkspace.load("session.maps")
```

`MapsWorkspace` inherits all of `SpectraWorkspace`'s preprocessing methods (`crop`, `set_baseline`, `subtract_baseline`, `normalize`, `reinit`) — each applies across every pixel of the targeted map(s) at once.

---

## Loading Map Data Directly

For array-level control without a `MapsWorkspace`, `spectroview.api.io.load_map()` returns the raw map DataFrame (this is what `MapsWorkspace.load_files()` uses internally):

```python
from spectroview.api import io
import numpy as np

# WDF and SPC files return (DataFrame, metadata_dict); TXT/CSV return a DataFrame only.
result = io.load_map("my_wafer_map.wdf")
df_map, metadata = result if isinstance(result, tuple) else (result, {})
```

The returned DataFrame's first two columns are always spatial coordinates (`X`, `Y`); all remaining columns are wavenumber values (as strings):

```python
x_coords = df_map["X"].to_numpy(dtype=float)
y_coords = df_map["Y"].to_numpy(dtype=float)

wn_cols  = [c for c in df_map.columns if c not in ("X", "Y")]
x_axis   = np.array([float(c) for c in wn_cols])
Y_matrix = df_map[wn_cols].to_numpy(dtype=float)
```

`Y_matrix` can be passed directly to `spectroview.api.fitting.fit_batch()` — see [Spectra Workspace: Batch Fitting](spectra.md#batch-fitting).

---

## Multivariate Analysis (MVA)

Multivariate Analysis explores large, complex datasets without requiring prior knowledge of peak positions. `spectroview.api.analysis` provides PCA and NMF, operating on any `(n_spectra, n_wavenumbers)` matrix — e.g. the `Y_matrix` above, or `Y0` from a `MapsWorkspace`'s underlying store.

### Principal Component Analysis (PCA)

```python
from spectroview.api import analysis, preprocessing

Y_norm = preprocessing.normalize_spectra(Y_matrix)
pca_result = analysis.pca(Y_norm, n_components=3, center=True)

print("Explained Variance Ratios:", pca_result.explained_variance_ratio)
scores   = pca_result.scores     # (n_spectra, n_components)
loadings = pca_result.loadings   # (n_components, n_wavenumbers)
```

### Non-negative Matrix Factorization (NMF)

NMF forces components and scores to be strictly positive, often giving more physically interpretable "pure component" spectra than PCA.

```python
nmf_result = analysis.nmf(Y_norm, n_components=3, max_iter=1000)
scores_w   = nmf_result.W
loadings_h = nmf_result.H
```

### Reconstruction Error

```python
errors = analysis.reconstruction_error(Y_norm, pca_result.scores, pca_result.loadings, pca_result.mean_spectrum)
# errors: (n_spectra,) per-spectrum L2 residual -- useful for flagging outlier spectra
```

### Exporting MVA Results

```python
import pandas as pd

df_pca = pd.DataFrame({"X": x_coords, "Y": y_coords})
for i in range(pca_result.n_components):
    df_pca[f"PCA_Score_PC{i+1}"] = scores[:, i]

df_pca.to_csv("map_with_pca_scores.csv", index=False)
```

To visualize `df_pca` as a heatmap the same way `MapsWorkspace.get_heatmap()` does, use `spectroview.model.heatmap.build_heatmap_grid(x_coords, y_coords, scores[:, 0], map_type="2Dmap")`.
