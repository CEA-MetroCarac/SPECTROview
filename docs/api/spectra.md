# Spectra Workspace

This section demonstrates how to handle discrete 1D spectra: loading, preprocessing, and fitting them with the Vectorized Batch Fit (VBF) engine — both through the stateful `SpectraWorkspace` session (recommended) and through the low-level array-based functions it's built on.

---

## Stateful Workspaces

`spectroview.api.workspace.SpectraWorkspace` mirrors the GUI's own Spectra Workspace: it holds your spectra, tracks preprocessing and fit-model state per spectrum, and reads/writes the exact same `.spectra` file format the GUI's "Save work" button produces. A workspace built entirely from a script opens directly in the GUI, and a `.spectra` file saved from the GUI loads directly here.

```python
from spectroview.api import workspace, fitting

ws = workspace.SpectraWorkspace()
ws.load_files(["sample_a.wdf", "sample_b.wdf", "sample_c.txt"])
print(ws)  # SpectraWorkspace(3 spectra: ['sample_a', 'sample_b', 'sample_c'])

# Preprocess every spectrum in the workspace
ws.crop(range_min=400.0, range_max=800.0)
ws.set_baseline({"mode": "arpls", "smoothing_factor": 1e5})
ws.subtract_baseline()

# Attach a fit model and fit
fit_model = fitting.build_fit_model(
    peaks=[
        {"model": "Lorentzian",
         "x0": {"value": 520.0, "min": 515.0, "max": 525.0},
         "ampli": {"value": 1000.0, "min": 0.0, "max": 1e9},
         "fwhm": {"value": 3.0, "min": 1.0, "max": 10.0}},
    ]
)
ws.set_fit_model(fit_model)
ws.fit()

df = ws.collect_results()   # pandas DataFrame, one row per spectrum
print(df)

# Save — this file opens directly in the SPECTROview GUI
ws.save("session.spectra")
```

To reload a session later (built by this API or saved from the GUI):

```python
ws2 = workspace.SpectraWorkspace.load("session.spectra")
```

Each preprocessing/fit method accepts an optional `names=[...]` list to target a subset of spectra instead of all of them, e.g. `ws.crop(range_min=400, range_max=800, names=["sample_a"])`.

!!! note "Legacy files"
    Very old `.spectra` files (format v1, pre-dating the current ZIP-based save format) are GUI-only. `SpectraWorkspace.load()` raises `WorkspaceError` on them — open the file once in the GUI and re-save it to upgrade it to the current format.

---

## Loading Data

The `spectroview.api.io` module provides unified functions to load spectra from multiple formats (WDF, SPC, CSV, TXT, DAT) — this is what `SpectraWorkspace.load_files()` uses internally, and it's also useful on its own for the low-level array-based workflow below.

```python
from spectroview.api import io
import numpy as np

spectra_dict = io.load_spectra("my_sample_spectra.wdf")

# Each item in the list is one spectrum (WDF series files can contain many)
items = spectra_dict["items"]
print(f"Number of spectra loaded: {len(items)}")

# Each item has: "name", "x0" (wavenumber axis), "y0" (intensities), "metadata"
first = items[0]
print("Wavenumber axis (x0):", first["x0"].shape)   # shape (M,)
print("Intensities (y0):    ", first["y0"].shape)   # shape (M,)
print("Laser Wavelength:", first["metadata"].get("Laser Wavelength (nm)"))
```

### Loading Multiple Files as a Batch

To work on a collection of spectra together as a single 2D matrix (rather than a `SpectraWorkspace`), use `io.load_spectra_to_matrix()`:

```python
import glob

file_paths = sorted(glob.glob("./measurements/*.wdf"))
data = io.load_spectra_to_matrix(file_paths)

x     = data["x"]      # float64[M]   — shared wavenumber axis
Y     = data["Y"]      # float64[N,M] — intensity matrix (N spectra)
names = data["names"]  # list[str]    — spectrum names
```

!!! note
    `load_spectra_to_matrix()` interpolates all spectra onto the x-axis of the first file if their axes differ slightly.

---

## Preprocessing

For array-level control without a `SpectraWorkspace`, `spectroview.api.preprocessing` exposes the same crop/baseline/normalize operations as stateless functions over `(x, Y)` arrays.

```python
from spectroview.api import preprocessing

# Crop to the region of interest
x_crop, Y_crop = preprocessing.crop_spectra(x, Y, range_min=400.0, range_max=800.0)

# Baseline subtraction (same algorithms as the GUI, e.g. arpls, airpls, asls, Linear, Polynomial)
baseline_config = {"mode": "arpls", "smoothing_factor": 1e5}
Y_corrected, Y_baseline = preprocessing.subtract_baseline(x_crop, Y_crop, baseline_config)

# Normalize each spectrum to its own maximum
Y_norm = preprocessing.normalize_spectra(Y_corrected)
```

---

## Batch Fitting

The core power of SPECTROview is its Vectorized Batch Fit (VBF) engine, which can fit thousands of spectra in a fraction of a second. `spectroview.api.fitting` exposes it directly for array-level workflows (a `SpectraWorkspace.fit()` call uses this under the hood, per spectrum).

### Step 1: Define or Load the Fit Model

**Option A: Load a template exported from the SPECTROview GUI (or saved via this API)**

```python
from spectroview.api import fitting

fit_model = fitting.load_fit_model_template("my_gui_model.json")
```

**Option B: Build the model with `fitting.build_fit_model()`**

```python
fit_model = fitting.build_fit_model(
    peaks=[
        {"model": "Lorentzian",
         "x0": {"value": 520.0, "min": 515.0, "max": 525.0},
         "ampli": {"value": 1000.0, "min": 0.0, "max": 1e9},
         "fwhm": {"value": 3.0, "min": 1.0, "max": 10.0}},
        {"model": "Gaussian",
         "x0": {"value": 600.0, "min": 580.0, "max": 620.0},
         "ampli": {"value": 500.0, "min": 0.0, "max": 1e9},
         "fwhm": {"value": 15.0, "min": 5.0, "max": 50.0}},
    ]
)
```

Each peak parameter dict accepts `value`, `min`, `max`, and optionally `vary` (bool, default `True`) or `fix` (its inverse) and `expr` (a string expression linking it to another parameter).

### Step 2: Execute the Fit

```python
fit_params = {"xtol": 1e-4, "ftol": 1e-4, "max_ite": 500}

results = fitting.fit_batch(x_crop, Y_corrected, fit_model, fit_params=fit_params)

print(f"Success rate: {results['success'].mean() * 100:.1f}%")
print(f"Average R-squared: {results['r_squared'].mean():.4f}")

fitted_params_matrix = results["params"]        # shape (N, K)
parameter_names      = results["param_names"]   # e.g. ['P1_x0', 'P1_fwhm', ...]
```

By default, `fit_batch` derives fit weights from `fit_params` exactly as the GUI does before every fit — negative-intensity points are excluded (unless `fit_params["fit_negative"]` is `True`), and points below a noise floor set by `fit_params["coef_noise"]` are excluded. Pass `auto_weights=False` to fit every point with equal weight, or pass your own `weights` array.

For a raw spectrum matrix that still needs the crop/baseline steps from a fit model applied first, use `fitting.apply_fit_model(x, Y, fit_model)`, which combines all three steps (crop, baseline, fit) in one call and also returns the actually-fitted `x`/`Y`.

### Step 3: Export Results

```python
import pandas as pd
from spectroview.api import io

df_results = pd.DataFrame(results["params"], columns=results["param_names"])
df_results["R_squared"]   = results["r_squared"]
df_results["Fit_Success"] = results["success"]

io.export_results(df_results, "my_fit_results.csv")
```

### Saving a Fit Model for Reuse

```python
fitting.save_fit_model_template(fit_model, "my_model.json")

# List and reload templates from a folder later
names = fitting.list_fit_model_templates("./models")
fit_model = fitting.load_fit_model_template("./models/my_model.json")
```
