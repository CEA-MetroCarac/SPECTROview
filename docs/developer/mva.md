# Multivariate Analysis (MVA) of Raman Spectroscopic Data

This document provides a technical overview of the Multivariate Analysis (MVA) features implemented in SPECTROview.

## Architecture

The MVA feature follows the MVVM (Model-View-ViewModel) architectural pattern:

- **Model** (`spectroview/model/m_mva.py`): Contains the core mathematics for MVA techniques, specifically Principal Component Analysis (PCA) and Non-negative Matrix Factorization (NMF). Uses standard numerical libraries (`numpy`, `scipy`) without requiring extra dependencies like `scikit-learn`.
- **ViewModel** (`spectroview/viewmodel/vm_mva.py`): Orchestrates the data flow. It extracts preprocessed data from active spectra, invokes the Model to perform calculations, and caches results.
- **View** (`spectroview/view/components/v_mva.py`): Provides the user interface, including parameter input controls, method selection, and `matplotlib`-based plotting capabilities (scree plots, loadings, scores). 

## Implemented Methods

### 1. Principal Component Analysis (PCA)
PCA is a dimensionality reduction technique that transforms spectra into a new coordinate system such that the greatest variance by any projection of the data comes to lie on the first coordinate (the first principal component).

- **Implementation Details**: Implemented using Singular Value Decomposition (`numpy.linalg.svd`) on mean-centered data.
- **Visualizations**: 
  - *Scree Plot*: Displays the percentage of variance explained by each principal component and the cumulative variance.
  - *Loadings*: Visualizes the spectral contribution to each principal component.
  - *Scores*: Provides a 2D scatter plot of the spectra projected onto the principal components.

### 2. Non-negative Matrix Factorization (NMF)
NMF factors the data matrix into two non-negative matrices (typically W and H). This is particularly useful for spectroscopic data as it yields additive, non-negative components that often correspond to physical endmembers.

- **Implementation Details**: Uses the Lee & Seung multiplicative update rules. It iteratively updates the matrices to minimize the Frobenius norm of the reconstruction error.
- **Parameters**: 
  - *Components*: Number of non-negative components to extract.
  - *Max Iterations*: Limit on the number of update iterations.
  - *Tolerance*: Convergence threshold based on the relative change in reconstruction error.
- **Visualizations**:
  - *Loadings*: Displays the non-negative spectral components (endmembers).
  - *Scores*: Provides a 2D scatter plot of the concentration/abundances of these components across the dataset.

## Data Pipeline

1. **Selection**: MVA operates exclusively on the *active* (checked) spectra in the SPECTROview workspace.
2. **Preprocessing**: The analysis uses the `spectrum.y` attribute, meaning it operates on data that has already undergone preprocessing steps (baseline subtraction, normalization, cropping) configured by the user.
3. **Interpolation**: If the selected spectra have varying wavenumber axes, they are automatically interpolated onto a common grid (the grid of the first active spectrum) to build a consistent data matrix.
4. **Export**: The results (scores for PCA, W matrix for NMF) can be exported directly to the Graphs workspace for further visualization and comparison.
