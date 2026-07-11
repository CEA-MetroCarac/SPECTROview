# Spectra Processing & Fitting

This section demonstrates how to handle discrete 1D spectra, perform standard preprocessing steps, and leverage the Vectorized Batch Fit (VBF) engine programmatically.

---

## 1. Loading Data

The `spectroview.api.io` module provides unified functions to load spectra from multiple formats (WDF, SPC, CSV, TXT, DAT).

```python
from spectroview.api import io
import numpy as np

# Load a discrete spectrum file
# Returns a dict with "type" and "items" keys
spectra_dict = io.load_spectra("my_sample_spectra.wdf")

# Each item in the list is one spectrum (WDF series files can contain many)
items = spectra_dict["items"]   # list of dicts
print(f"Number of spectra loaded: {len(items)}")

# Each item has: "name", "x0" (wavenumber axis), "y0" (intensities), "metadata"
first = items[0]
print("Spectrum name:", first["name"])
print("Wavenumber axis (x0):", first["x0"].shape)   # shape (M,)
print("Intensities (y0):    ", first["y0"].shape)   # shape (M,)

# Acquisition metadata (populated for WDF and SPC files)
print("Laser Wavelength:", first["metadata"].get("Laser Wavelength (nm)"))
print("Accumulations:",    first["metadata"].get("Accumulations"))
```

### Loading Multiple Files as a Batch

To work on a collection of spectra together (e.g., a repeatability study), stack them into a single 2D matrix using `io.load_spectra_to_matrix()`:

```python
import glob

# Load all .wdf files from a folder and stack into a single matrix
file_paths = sorted(glob.glob("./measurements/*.wdf"))

data = io.load_spectra_to_matrix(file_paths)

x    = data["x"]         # float64[M]   — shared wavenumber axis
Y    = data["Y"]         # float64[N,M] — intensity matrix (N spectra)
names = data["names"]    # list[str]    — spectrum names

print(f"Loaded {len(names)} spectra, {x.shape[0]} wavenumber points each.")
```

> [!NOTE]
> `load_spectra_to_matrix()` interpolates all spectra onto the x-axis of the first file if their axes differ slightly. For perfectly consistent acquisitions (same instrument settings) the axes will always match.

---

## 2. Preprocessing

The `spectroview.api.processing` module allows you to apply baseline subtractions, crop spectral ranges, and normalize data.

### Cropping

```python
from spectroview.api import processing

# Crop to the region of interest (e.g., 400 to 800 cm-1)
x_crop, Y_crop = processing.crop_spectra(
    x,
    Y,
    range_min=400.0,
    range_max=800.0
)
```

### Baseline Subtraction

You can programmatically apply the same advanced baseline algorithms found in the SPECTROview GUI (such as `arpls` or `airpls`).

```python
# Option A: Automatic baseline (e.g. arpls)
baseline_config_auto = {
    "mode": "arpls",
    "smoothing_factor": 1e5  # Adjust this to make the baseline stiffer or looser
}

# Option B: Linear baseline using anchor points
baseline_config_linear = {
    "mode": "Linear",
    "attached": False,
    "points": [
        [410.0, 790.0],  # X anchor points (wavenumbers)
        [0.0, 0.0]       # Y anchor points (intensities)
    ]
}

# Subtract baseline (using the automatic option as an example)
Y_corrected, Y_baseline = processing.subtract_baseline(x_crop, Y_crop, baseline_config_auto)

# Y_corrected now contains the pure Raman/PL signal without the background
```

---

## 3. Batch Fitting

The core power of SPECTROview is its Vectorized Batch Fit (VBF) engine, which can fit thousands of spectra in a fraction of a second. The API exposes this through `spectroview.api.fitting`.

### Step 1: Define or Load the Fit Model

The fit model is a dictionary describing the peak models to fit.

**Option A: Load from a pre-defined JSON file (exported from the SPECTROview GUI)**

```python
import json

with open("my_gui_model.json", "r") as f:
    fit_model = json.load(f)
```

**Option B: Build the model using the `fitting.build_fit_model()` helper**

This is the recommended way to define models in scripts. It produces the exact format expected by the VBF engine:

```python
from spectroview.api import fitting

fit_model = fitting.build_fit_model(
    peaks=[
        {
            "model": "Lorentzian",
            "x0":    {"value": 520.0, "min": 515.0, "max": 525.0},
            "ampli": {"value": 1000.0, "min": 0.0,  "max": 1e9},
            "fwhm":  {"value": 3.0,   "min": 1.0,   "max": 10.0}
        },
        {
            "model": "Gaussian",
            "x0":    {"value": 600.0, "min": 580.0, "max": 620.0},
            "ampli": {"value": 500.0, "min": 0.0,   "max": 1e9},
            "fwhm":  {"value": 15.0,  "min": 5.0,   "max": 50.0}
        }
    ]
)
```

> [!NOTE]
> Each peak parameter dict accepts `value`, `min`, `max`, and optionally `vary` (bool, default `True`) and `expr` (string expression for linked parameters).

### Step 2: Execute the Fit

```python
# Optional: Set optimizer constraints and tolerance
fit_params = {
    "xtol": 1e-4,     # Parameter step tolerance
    "ftol": 1e-4,     # Cost function tolerance
    "max_ite": 500    # Maximum number of iterations per spectrum
}

# Fit the batch (x is 1D array of wavenumbers, Y is 2D matrix of intensities)
results = fitting.fit_batch(
    x_crop,
    Y_corrected,
    fit_model,
    fit_params=fit_params
)

print(f"Success rate: {np.mean(results['success']) * 100:.1f}%")
print(f"Average R-squared: {np.mean(results['r_squared']):.4f}")

# The fitted parameters are returned as a dense 2D NumPy array
fitted_params_matrix = results['params']    # shape (N, K)
parameter_names      = results['param_names']  # list of K names e.g. ['P1_x0', 'P1_fwhm', ...]
```

### Step 3: Export Results

You can export the results immediately back to CSV or Excel for further analysis using `io.export_results`.

```python
import pandas as pd

# Create a Pandas DataFrame for the results
df_results = pd.DataFrame(results['params'], columns=results['param_names'])

# Add R-squared and Success flags as new columns
df_results['R_squared']   = results['r_squared']
df_results['Fit_Success'] = results['success']

# Export to CSV
io.export_results(df_results, "my_fit_results.csv")
```
