# 2D Map & Hyperspectral Analysis

This section demonstrates how to handle hyperspectral map data (like Raman wafer maps) and apply Multivariate Analysis techniques programmatically.

---

## 1. Loading Map Data

When loading map data (from `.wdf`, `.spc`, `.txt`, or `.csv` files), the `spectroview.api.io` module handles the complexities of extracting both the spatial coordinates and the spectral intensities.

```python
from spectroview.api import io
import numpy as np

# Load hyperspectral map data
# WDF and SPC files return (DataFrame, metadata_dict)
# TXT and CSV files return a DataFrame only
result = io.load_map("my_wafer_map.wdf")

if isinstance(result, tuple):
    df_map, metadata = result
else:
    df_map, metadata = result, {}

print("Acquisition metadata:")
for k, v in metadata.items():
    print(f"  {k}: {v}")
```

### Extracting Spectral Data from the DataFrame

The returned DataFrame always has this layout:

| `X` | `Y` | `'100.5'` | `'101.0'` | `'101.5'` | ... |
|-----|-----|-----------|-----------|-----------|-----|
| Stage X (µm) | Stage Y (µm) | Intensity at 100.5 cm⁻¹ | ... | ... | ... |

The **first two columns are always spatial coordinates** (`X`, `Y`). All remaining columns are wavenumber values (as strings). You must separate them before processing:

```python
# Separate spatial coordinates from spectral data
x_coords = df_map['X'].to_numpy(dtype=float)    # shape (N,) — stage positions
y_coords = df_map['Y'].to_numpy(dtype=float)    # shape (N,) — stage positions

# Spectral data columns — all columns except 'X' and 'Y'
wn_cols  = [c for c in df_map.columns if c not in ('X', 'Y')]

# Reconstruct the wavenumber axis and intensity matrix
x_axis   = np.array([float(c) for c in wn_cols])      # float64[M] — wavenumbers
Y_matrix = df_map[wn_cols].to_numpy(dtype=float)       # float64[N, M] — intensities

print(f"Map size: {Y_matrix.shape[0]} spectra × {Y_matrix.shape[1]} wavenumber points")
print(f"Wavenumber range: {x_axis[0]:.1f} – {x_axis[-1]:.1f} cm-1")
```

---

## 2. Batch Fitting on 2D Maps

Because `Y_matrix` from a hyperspectral map is a 2D matrix of shape `(N_spectra, M_wavenumbers)`, you can pass it directly to the Vectorized Batch Fit (VBF) engine — the same engine that powers the GUI's Maps workspace.

```python
import json
import pandas as pd
from spectroview.api import fitting, processing

# 1. Optionally preprocess: crop and subtract baseline first
x_crop, Y_crop = processing.crop_spectra(x_axis, Y_matrix, range_min=450.0, range_max=600.0)
Y_corrected, _ = processing.subtract_baseline(x_crop, Y_crop, {"mode": "arpls", "smoothing_factor": 1e5})

# 2. Load a pre-defined fit model (e.g., exported from the GUI)
with open("my_gui_model.json", "r") as f:
    fit_model = json.load(f)

# — Or define the model directly using the helper —
fit_model = fitting.build_fit_model(
    peaks=[
        {
            "model": "Lorentzian",
            "x0":    {"value": 520.0, "min": 515.0, "max": 525.0},
            "ampli": {"value": 1000.0, "min": 0.0, "max": 1e9},
            "fwhm":  {"value": 3.0, "min": 0.5, "max": 15.0}
        }
    ]
)

# 3. Fit all N spectra simultaneously
print(f"Fitting {Y_corrected.shape[0]} spectra from the map...")
results = fitting.fit_batch(x_crop, Y_corrected, fit_model)

print(f"Success rate: {results['success'].mean() * 100:.1f}%")

# 4. Merge the fit results back with spatial coordinates
df_results = pd.DataFrame(results['params'], columns=results['param_names'])
df_results.insert(0, 'X', x_coords)
df_results.insert(1, 'Y', y_coords)
df_results['Fit_R_squared'] = results['r_squared']
df_results['Fit_Success']   = results['success']

# 5. Save the full results map
df_results.to_csv("map_with_fit_results.csv", index=False)
```

---

## 3. Multivariate Analysis (MVA)

Multivariate Analysis is crucial for exploring large, complex datasets without requiring prior knowledge of peak positions. The `spectroview.api.analysis` module provides robust implementations of PCA and NMF.

### Principal Component Analysis (PCA)

PCA reduces the dimensionality of your data, allowing you to visualize variance and identify clusters.

```python
from spectroview.api import analysis, processing

# It is highly recommended to normalize data before running PCA
Y_norm = processing.normalize_spectra(Y_matrix)

# Run PCA requesting the top 3 principal components
pca_result = analysis.pca(Y_norm, n_components=3, center=True)

# Access the results
print("Explained Variance Ratios:", pca_result.explained_variance)

# Extract Scores (weights of each spectrum on the components)
# Shape: (n_spectra, n_components)
scores = pca_result.scores

# Extract Loadings (the "spectral shape" of the components)
# Shape: (n_components, n_wavenumbers)
loadings = pca_result.loadings
```

### Non-negative Matrix Factorization (NMF)

NMF is often preferred over PCA for spectroscopic data because it forces the resulting components (loadings) and scores to be strictly positive, leading to more physically interpretable "pure component" spectra.

```python
# Run NMF requesting 3 components
# Note: NMF requires strictly non-negative data (ensure baseline is subtracted and min >= 0)
nmf_result = analysis.nmf(Y_norm, n_components=3, max_iter=1000)

# Extract Scores (W matrix)
scores_w = nmf_result.W

# Extract Loadings (H matrix)
loadings_h = nmf_result.H
```

### Exporting MVA Results

You can export the PCA/NMF scores back into your DataFrame to plot them as heatmaps.

```python
import pandas as pd

# Build a results DataFrame with spatial coordinates + PCA scores
df_pca = pd.DataFrame({
    'X': x_coords,
    'Y': y_coords,
})

for i in range(pca_result.n_components):
    df_pca[f'PCA_Score_PC{i+1}'] = scores[:, i]

# Save the augmented DataFrame
df_pca.to_csv("map_with_pca_scores.csv", index=False)
```

---

