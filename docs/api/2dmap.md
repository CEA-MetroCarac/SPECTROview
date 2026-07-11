# 2D Map & Hyperspectral Analysis

This section demonstrates how to handle hyperspectral map data (like Raman wafer maps) and apply Multivariate Analysis techniques programmatically.

---

## 1. Loading Map Data

When loading map data (from `.wdf` or `.spc` files), the `spectroview.api.io` module handles the complexities of extracting both the spatial coordinates and the spectral intensities.

```python
from spectroview.api import io
import numpy as np

# Load hyperspectral map data
# Returns a Pandas DataFrame and a dictionary of metadata
df_map, metadata = io.load_map("my_wafer_map.wdf")

# The DataFrame contains coordinates as an index (or specific columns)
# The columns are the wavenumber (X-axis) points.
x_axis = df_map.columns.to_numpy(dtype=float)
Y_matrix = df_map.to_numpy(dtype=float)

print(f"Loaded map with {Y_matrix.shape[0]} spectra and {Y_matrix.shape[1]} data points.")
print(f"Map coordinates are available in: {df_map.index.names}")
```

---

## 2. Multivariate Analysis (MVA)

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

# Add PCA scores as new columns to the original map DataFrame
for i in range(pca_result.n_components):
    df_map[f'PCA_Score_PC{i+1}'] = scores[:, i]

# Save the augmented DataFrame
df_map.to_csv("map_with_pca_scores.csv")
```
