# Multivariate Analysis (MVA)

SPECTROview includes built-in Multivariate Analysis tools accessible from the **MVA tab** in the Spectra and Maps workspaces.

## Supported Methods

### Principal Component Analysis (PCA)

Reduces spectral data dimensions to reveal main patterns of variance.

**Visualizations**:

- **Scree Plot** — Variance explained per component + cumulative variance
- **Loadings** — Spectral contribution to each principal component
- **Scores** — 2D scatter of spectra projected onto principal components

### Non-negative Matrix Factorization (NMF)

Decomposes data into additive, non-negative components (endmembers). Useful for spectroscopic data as components often correspond to physical constituents.

**Parameters**:

| Parameter | Description |
|-----------|-------------|
| Components | Number of non-negative components to extract |
| Max Iterations | Limit on update iterations (default: 500) |
| Tolerance | Convergence threshold (default: 1e-4) |

**Visualizations**:

- **Loadings** — Non-negative spectral components (endmembers)
- **Scores** — Concentration/abundance of components across dataset

## How to Use

1. **Load spectra** in the Spectra or Maps workspace
2. **Check (activate)** the spectra to analyze
3. Open the **MVA tab**
4. Select method (**PCA** or **NMF**) and number of components
5. Click **Run**
6. View results in embedded plots
7. (Optional) **Export scores** to Graphs workspace

!!! warning "Data Preparation"
    - MVA operates on **preprocessed data** (after baseline subtraction, normalization)
    - Spectra with different x-axes are **automatically interpolated** onto a common grid
    - At least **2 active spectra** are required
