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

# 1. Crop to the region of interest (e.g., 400 to 800 cm-1)
x_crop, Y_crop = processing.crop_spectra(
    spectra_dict['x'], 
    spectra_dict['Y'], 
    range_min=400.0, 
    range_max=800.0
)

# 2. Min-Max normalization (scale max intensity to 1.0)
Y_norm = processing.normalize_spectra(Y_crop)
```

### Baseline Subtraction

You can programmatically apply the same advanced baseline algorithms found in the SPECTROview GUI (such as `arpls` or `airpls`).

```python
# Configure the baseline algorithm
baseline_config = {
    "mode": "arpls",
    "smoothing_factor": 1e5  # Adjust this to make the baseline stiffer or looser
}

# Subtract baseline
Y_corrected, Y_baseline = processing.subtract_baseline(x_crop, Y_crop, baseline_config)

# Y_corrected now contains the pure Raman/PL signal without the background
```

---

## 3. Batch Fitting

The core power of SPECTROview is its Vectorized Batch Fit (VBF) engine, which can fit thousands of spectra in a fraction of a second. The API exposes this through `spectroview.api.fitting`.

### Step 1: Define the Model

The fit model is passed as a Python dictionary describing the baseline configuration and the individual peaks.

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

# Fit the batch (x is 1D array of wavenumbers, Y is 2D matrix of intensities)
results = fitting.fit_batch(x_crop, Y_corrected, fit_model)

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
