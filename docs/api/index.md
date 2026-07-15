# API

The SPECTROview Python API lets you automate workflows and interact programmatically with the core spectroscopic processing engines of SPECTROview. It is completely decoupled from the Qt GUI (no `QApplication` required), making it well suited to Jupyter notebooks, batch scripts, and CI pipelines.

All modules are exposed through the `spectroview.api` package.

## API Modules

- **[`spectroview.api.workspace`](spectra.md#stateful-workspaces)** — `SpectraWorkspace` / `MapsWorkspace`: stateful sessions that load, preprocess, fit, collect results, and save/load in one place. Files they write/read use the *exact same format* as the GUI's own "Save work" — a workspace built entirely from a script opens directly in the GUI, and vice versa.
- **[`spectroview.api.io`](spectra.md#loading-data)** — File readers (WDF, SPC, CSV, TXT, DAT), dataset import, result export, and a headless Renishaw map converter.
- **[`spectroview.api.preprocessing`](spectra.md#preprocessing)** — Stateless array-level baseline subtraction, cropping, and normalization.
- **[`spectroview.api.fitting`](spectra.md#batch-fitting)** — The interface to the Vectorized Batch Fit (VBF) engine: model construction, batch fitting, and fit-model template CRUD.
- **[`spectroview.api.analysis`](2dmap.md#multivariate-analysis-mva)** — Multivariate analysis: PCA and NMF.
- **[`spectroview.api.graphs`](graphs.md)** — Publication-quality plots that pixel-for-pixel match the GUI's own rendering, for all 9 supported plot styles, plus plot-template CRUD.
- **[`spectroview.api.calculators`](calculators.md)** — Physical calculators for optics and unit conversions.
- **`spectroview.api.settings`** — Headless read/write access to persistent configuration (fit defaults, model folder, last directory).
- **`spectroview.api.exceptions`** — The `SpectroviewError` hierarchy raised across this package (`LoadError`, `FitModelError`, `FitError`, `WorkspaceError`, `TemplateError`).

## Exploring the API

- [Spectra Workspace](spectra.md): Load discrete spectra, preprocess them, and run the batch fitting engine — both the stateful `SpectraWorkspace` way and the low-level array-based way.
- [Maps Workspace & 2D Analysis](2dmap.md): Load hyperspectral/wafer maps, fit every pixel, build heatmaps and line profiles, and run PCA/NMF.
- [Data Visualization (Graphs)](graphs.md): Replicate SPECTROview's native plots in scripts, for all plot styles, plus reusable plot templates.
- [Quick Calculators](calculators.md): Programmatic optical and spectroscopic-unit converters.
- [Extending SPECTROview](extending.md): The real, file-based extension points (custom baseline methods, fit-model/plot-template folders) — there is no plugin framework, and this page explains what's available instead.

## Error handling

Every function in `spectroview.api` raises a subclass of `spectroview.api.exceptions.SpectroviewError` on failure — never a raw `KeyError`/`ValueError` from the underlying model layer, and never a Qt dialog (which would hang in a headless script). Catch `SpectroviewError` broadly, or a specific subtype (`LoadError`, `FitModelError`, `FitError`, `WorkspaceError`, `TemplateError`) narrowly:

```python
from spectroview.api import io
from spectroview.api.exceptions import LoadError

try:
    data = io.load_spectra("missing_file.wdf")
except LoadError as e:
    print(f"Could not load spectra: {e}")
```
