# Spectra Processing & Fitting

This section demonstrates how to handle discrete 1D spectra, perform standard preprocessing steps, and leverage the Vectorized Batch Fit (VBF) engine programmatically.

---

## 1. Loading Data

The `spectroview.api.io` module provides unified functions to load spectra from multiple formats (WDF, SPC, CSV, TXT, DAT). 

```python
from spectroview.api import io

# Load a discrete spectrum file
spectra_dict = io.load_spectra("my_sample_spectra.wdf")

# The returned dictionary contains the data arrays and metadata
print("Wavenumber axis (X):", spectra_dict['x'].shape)
print("Intensity matrix (Y):", spectra_dict['Y'].shape)

# Metadata (if available from WDF/SPC)
print("Laser Wavelength:", spectra_dict.get("Laser Wavelength (nm)"))
print("Accumulations:", spectra_dict.get("Accumulations"))
```

---

## 2. Preprocessing

The `spectroview.api.processing` module allows you to apply baseline subtractions, crop spectral ranges, and normalize data.

### Cropping and Normalization

```python
from spectroview.api import processing

# Crop to the region of interest (e.g., 400 to 800 cm-1)
x_crop, Y_crop = processing.crop_spectra(
    spectra_dict['x'], 
    spectra_dict['Y'], 
    range_min=400.0, 
    range_max=800.0
)

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

The fit model is a dictionary describing the baseline configuration and the individual peaks. 

**Option A: Load from a pre-defined JSON file (e.g., exported from the SPECTROview GUI)**

```python
import json

with open("my_gui_model.json", "r") as f:
    fit_model = json.load(f)
```

**Option B: Define the model manually**

```python
import numpy as np

fit_model = {
    # You can configure the baseline to be fit simultaneously with the peaks
    "baseline_config": {
        "mode": "Linear", 
        "attached": False,
        "points": [
            [410.0, 790.0],  # X anchor points
            [0.0, 0.0]       # Y anchor points
        ]
    },
    # Define as many peaks as you need
    "peaks": {
        "peak_1": {
            "model": "Lorentzian",
            "x0": {"value": 520.0, "min": 515.0, "max": 525.0, "fix": False},
            "ampli": {"value": 1000.0, "min": 0.0, "max": np.inf, "fix": False},
            "fwhm": {"value": 3.0, "min": 1.0, "max": 10.0, "fix": False}
        },
        "peak_2": {
            "model": "Gaussian",
            "x0": {"value": 600.0, "min": 580.0, "max": 620.0, "fix": False},
            "ampli": {"value": 500.0, "min": 0.0, "max": np.inf, "fix": False},
            "fwhm": {"value": 15.0, "min": 5.0, "max": 50.0, "fix": False}
        }
    }
}
```

### Step 2: Execute the Fit

```python
from spectroview.api import fitting

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
fitted_params_matrix = results['params']
parameter_names = results['param_names']
```

### Step 3: Export Results

You can export the results immediately back to CSV or Excel for further analysis using `io.export_results`.

```python
import pandas as pd

# Create a Pandas DataFrame for the results
df_results = pd.DataFrame(results['params'], columns=results['param_names'])

# Add R-squared and Success flags as new columns
df_results['R_squared'] = results['r_squared']
df_results['Fit_Success'] = results['success']

# Export to CSV
io.export_results(df_results, "my_fit_results.csv")
```
