## **Multivariate Analysis (MVA)**

`SPECTROview` features built-in Multivariate Analysis (MVA) tools, easily accessible from the dedicated **MVA tab** located within both the `Spectra` and `Maps` workspaces.

> **Note**: MVA algorithms require a minimum of 2 active spectra to run.

### **1. Supported Algorithms**

#### **1.1. Principal Component Analysis (PCA)**
A statistical procedure that orthogonally transforms and reduces the dimensionality of complex spectral datasets, revealing the most significant patterns of variance. The implementation uses Singular Value Decomposition (SVD).

  - **Parameters**:
    - *Components* (2–50): Number of principal components to retain.
    - *Mean centering* (on/off): Whether to subtract the mean spectrum before analysis. Default: on.
  - **Available Visualizations**: Scree plots (variance explained), Principal Component Loadings, Score plots, and Residual reconstruction error plots.

#### **1.2. Non-negative Matrix Factorization (NMF)**
A powerful decomposition algorithm that factors the spectral data into distinct, additive, non-negative components, often representing pure chemical endmembers.

  - **Parameters**:
    - *Components* (2–50): Number of non-negative components to extract.
    - *Max Iterations* (50–5000): Maximum number of update iterations.
    - *Tolerance*: Convergence threshold for early stopping.
    - *Random Seed*: For reproducible results.
  - **Available Visualizations**: Component Loadings, Score plots, and Residual reconstruction error plots.

> **Why doesn't NMF have a Scree Plot?**  
> Unlike PCA, which strictly orders orthogonal components by the maximum variance they explain, NMF finds non-orthogonal, additive components (endmembers) that mix together. Because NMF components are not orthogonal and are not sequentially ordered by variance, an isolated percentage of "variance explained" cannot be calculated for each individual component. To evaluate an NMF model, rely on the **Residuals** plot to assess reconstruction quality, and the **Loadings** to ensure components correspond to physically meaningful spectra.
------

### **2. User Interface**

The MVA tab features a **split-panel layout**:

#### **2.1. Left Panel - Controls**
- **Method Selection**: Choose between `PCA` and `NMF` using radio buttons.
- **Parameter Controls**: All adjustable parameters are available as interactive GUI elements (spinboxes, checkboxes). Only the parameters relevant to the selected method are displayed.
- **Run Button**: Click "▶ Run Analysis" to execute the analysis with the current parameters.
- **Export**: Send results (scores) to the `Graphs` workspace for further analysis.

#### **2.2. Right Panel - Tabbed Plots**
Results are displayed across multiple tabs, each with its own interactive `matplotlib` toolbar (zoom, pan, save):

| Tab | PCA | NMF | Description |
|-----|-----|-----|-------------|
| **Summary** | ✅ | ✅ | A combined dashboard plot showing Scree, Loadings, and Scores all at once |
| **Scree Plot** | ✅ | - | Bar chart of explained variance per component with cumulative line |
| **Loadings** | ✅ | ✅ | Spectral loading profiles overlaid on the wavenumber axis |
| **Scores** | ✅ | ✅ | 2D scatter plot with selectable X/Y component axes |
| **Residuals** | ✅ | ✅ | Per-spectrum reconstruction error for assessing model quality |

### **3. Tips**
- Use the **Scree Plot** to determine the optimal number of components (look for the "elbow" in the cumulative variance curve).
- The **Loadings** plot shows which spectral features contribute most to each component.
- The **Scores** plot reveals groupings and outliers in your dataset.
- The **Residuals** tab helps identify spectra that are poorly reconstructed by the model.
- Each plot tab includes a **`matplotlib` toolbar** for zooming, panning, and saving figures.


-----

### **4. Citations and References**

If you use these MVA features to generate results for a publication, we kindly ask that you cite the following papers describing the underlying implementations and core algorithms:

- **PCA algorithm**: Wold, S., Esbensen, K., and Geladi, P., "Principal component analysis." *Chemometrics and intelligent laboratory systems* 2.1-3 (1987): 37-52.
- **NMF algorithm**: Lee, Daniel D., and H. Sebastian Seung, "Learning the parts of objects by non-negative matrix factorization." *Nature* 401.6755 (1999): 788-791.
