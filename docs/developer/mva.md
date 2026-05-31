# **Multivariate Analysis (MVA) of Raman Spectroscopic Data**

This document provides a technical overview of the Multivariate Analysis (MVA) features implemented in `SPECTROview`.

## **Architecture**

The MVA feature follows the MVVM (Model-View-ViewModel) architectural pattern:

- **Model** (`spectroview/model/m_mva.py`): Contains the core mathematics for MVA techniques, specifically Principal Component Analysis (PCA) and Non-negative Matrix Factorization (NMF). Uses standard numerical libraries (`numpy`, `scipy`) without requiring extra dependencies like `scikit-learn`.
- **ViewModel** (`spectroview/viewmodel/vm_mva.py`): Orchestrates the data flow. It extracts preprocessed data from active spectra, invokes the Model to perform calculations, caches results, and computes per-spectrum reconstruction errors.
- **View** (`spectroview/view/components/v_mva.py`): Provides the user interface with a tabbed plot panel (Scree Plot, Loadings, Scores, Residuals), interactive parameter controls, and `matplotlib` navigation toolbars for each plot.

## **Algorithm Implementation**

The PCA implementation in `SPECTROview` uses Singular Value Decomposition (SVD) on mean-centered data.

### **Visualization Enhancements**

`SPECTROview` features a comprehensive tabbed plot interface for visualizing results:

- **Scree Plot**: Bar chart of individual variance + cumulative line
- **Loadings**: Spectral loading overlays with fill-between for visual clarity
- **Scores**: 2D scatter with interactive axis selection
- **Residuals**: Per-spectrum reconstruction error for model quality assessment

## **Implemented Methods**

### **1. Principal Component Analysis (PCA)**
PCA is a dimensionality reduction technique that transforms spectra into a new coordinate system such that the greatest variance by any projection of the data lies on the first coordinate (the first principal component).

- **Implementation Details**: Implemented using Singular Value Decomposition (`numpy.linalg.svd`) on optionally mean-centered data.
- **Parameters**:
  - *Components*: Number of principal components to retain (2–50).
  - *Mean centering*: Whether to subtract the mean spectrum before SVD (default: on). This is standard practice for spectral PCA.
- **Visualizations**:
  - *Scree Plot*: Displays the percentage of variance explained by each principal component and the cumulative variance.
  - *Loadings*: Visualizes the spectral contribution to each principal component, with percentage of explained variance in the legend.
  - *Scores*: Provides a 2D scatter plot of the spectra projected onto the principal components, with interactive axis selection.
  - *Residuals*: Per-spectrum L2 reconstruction error for model quality assessment.

### **2. Non-negative Matrix Factorization (NMF)**
NMF factors the data matrix into two non-negative matrices (typically W and H). This is particularly useful for spectroscopic data, as it yields additive, non-negative components that often correspond to physical endmembers.

- **Implementation Details**: Uses the Lee & Seung multiplicative update rules. It iteratively updates the matrices to minimize the Frobenius norm of the reconstruction error.
- **Parameters**:
  - *Components*: The number of non-negative components (or endmembers) to extract from the dataset.
  - *Max Iterations*: The maximum number of multiplicative update iterations the algorithm is allowed to perform. If the algorithm does not converge within this limit, it will stop.
  - *Tolerance*: The convergence threshold. The algorithm stops early if the relative change in the reconstruction error between successive iterations falls below this value.
  - *Random Seed*: A seed value for the random number generator used to initialize the matrices. Setting a fixed seed ensures that the NMF results are reproducible across runs (default: 42).
- **Visualizations**:
  - *Loadings*: Displays the non-negative spectral components (endmembers).
  - *Scores*: Provides a 2D scatter plot of the concentration/abundances of these components across the dataset.
  - *Residuals*: Per-spectrum L2 reconstruction error.

> **Why doesn't NMF have a Scree Plot?**  
> Unlike PCA, which strictly orders orthogonal components by the maximum variance they explain, NMF finds non-orthogonal, additive components (endmembers) that mix together. Because NMF components are not orthogonal and are not sequentially ordered by variance, an isolated percentage of "variance explained" cannot be calculated for each individual component. To evaluate an NMF model, rely on the **Residuals** plot to assess reconstruction quality, and the **Loadings** to ensure components correspond to physically meaningful spectra.

## **User Interface**

The MVA view uses a split-panel layout:

### **Left Panel (Controls)**
- **Method Selection**: Radio buttons to switch between `PCA` and `NMF`.
- **Parameter Groups**: Separate group boxes for PCA and NMF parameters, showing only the relevant parameters for the selected method.
- **Run Button**: Executes the analysis.
- **Export Section**: Sends scores to the `Graphs` workspace.

### **Right Panel (Tabbed Plots)**
Each plot type occupies its own tab in a `QTabWidget`:
- **Scree Plot** (PCA only): Variance explained per component.
- **Loadings**: Spectral loading profiles.
- **Scores**: Interactive 2D scatter with axis selection.
- **Residuals**: Per-spectrum reconstruction quality.

Each tab includes a `matplotlib` `NavigationToolbar` for zoom, pan, and save operations.

## **Data Pipeline**

1. **Selection**: MVA operates exclusively on the *active* (checked) spectra in the `SPECTROview` workspace.
2. **Preprocessing**: The analysis uses the `spectrum.y` attribute, meaning it operates on data that has already undergone preprocessing steps (`baseline` subtraction, normalization, cropping) configured by the user.
3. **Interpolation**: If the selected spectra have varying wavenumber axes, they are automatically interpolated onto a common grid (the grid of the first active spectrum) to build a consistent data matrix.
4. **Export**: The results (scores for PCA, W matrix for NMF) can be exported directly to the `Graphs` workspace for further visualization and comparison.

## **Citations and References**

If these features are used to generate results for publication, users are kindly requested to cite the relevant literature for the implemented algorithms:

[1] **PCA algorithm**: Wold, S., Esbensen, K., and Geladi, P., "Principal component analysis." *Chemometrics and intelligent laboratory systems* 2.1-3 (1987): 37-52.

[2] **NMF algorithm**: Lee, Daniel D., and H. Sebastian Seung, "Learning the parts of objects by non-negative matrix factorization." *Nature* 401.6755 (1999): 788-791.
