# SPECTROview High-Performance Fitting Engine

> **Maintenance Rule**: This document MUST be updated whenever the fitting engine
> logic, architecture, or module structure is modified. Any PR touching
> `spectroview/core/` should include corresponding updates to this file.

## Overview

The `spectroview/core/` package provides a high-performance fitting engine for
spectroscopic data. It replaces the per-spectrum fitspy/lmfit fitting loop with
a direct `scipy.optimize.least_squares` pipeline, achieving ~2× speedup for
large hyperspectral datasets while maintaining identical numerical accuracy.

### Performance Benchmarks (1681-spectrum Si map, 2 Lorentzian peaks)

| Step | Time |
|------|------|
| Apply fit model to spectra | 0.74s |
| Preprocessing (range + baseline) | 0.38s |
| **Fitting (1681 spectra)** | **16.4s** |
| Write results back to spectra | 0.09s |
| **Total** | **17.6s** |
| Old fitspy engine (same data) | ~35s |
| **Speedup** | **~2×** |

---

## Module Structure

```
spectroview/core/
├── __init__.py          # Public API exports
├── models.py            # Peak model functions + PeakModelEvaluator + FitResult
├── baseline.py          # Batch baseline preprocessing (delegates to fitspy)
├── spatial.py           # Spiral traversal + NeighborPropagator (KDTree)
├── optimizer.py         # scipy.optimize.least_squares wrappers
├── batch_engine.py      # Main orchestrator (BatchFittingEngine)
└── hyper_fit_thread.py  # QThread wrapper (HyperFitThread)
```

---

## Core Modules

### `models.py` — Peak Model Functions & Evaluator

**Peak functions**: Pure-NumPy implementations of all supported peak shapes,
matching fitspy's output exactly:

| Function | Parameters | Notes |
|----------|-----------|-------|
| `gaussian` | ampli, fwhm, x0 | Standard Gaussian |
| `lorentzian` | ampli, fwhm, x0 | Standard Lorentzian |
| `pseudovoigt` | ampli, fwhm, x0, alpha | Linear mix of G + L |
| `gaussian_asym` | ampli, fwhm1, fwhm2, x0 | Asymmetric Gaussian |
| `lorentzian_asym` | ampli, fwhm1, fwhm2, x0 | Asymmetric Lorentzian |
| `fano` | ampli, fwhm, x0, q | Fano lineshape |
| `decay_single_exp` | A, tau, B | Single exponential decay |
| `decay_bi_exp` | A1, tau1, A2, tau2, B | Bi-exponential decay |

**`PeakModelEvaluator`**: Maps a structured fit model dict (from `spectrum.save()`
or JSON) into a flat parameter vector for scipy optimization.

Key responsibilities:
- Parse `peak_models` dict → extract model functions, parameter values, bounds
- Maintain mapping between flat vector indices and named parameters
- Track which parameters are free (vary=True) vs fixed
- `evaluate(x, p_full)` → sum of all peak contributions
- `residual(p_free, x, y)` → `evaluate(x, ...) - y`
- `build_result(p_opt, x, y, success)` → `FitResult` object
- `write_back_to_spectrum(spectrum, fit_result)` → update MSpectrum attributes

**`FitResult`**: Compatibility class matching lmfit's `MinimizerResult` interface:
- `.success` — convergence flag
- `.params["name"].value` — parameter access (lmfit pattern)
- `.best_values` — dict of `{name: value}`
- `.best_fit` — model evaluated at optimal parameters

### `spatial.py` — Spatial Traversal & Neighbor Propagation

**`build_traversal_order(coords, strategy)`**: Determines the order in which
map pixels are fitted.

- `"spiral"` (default for maps): Starts from the center of the map and spirals
  outward. Each pixel's fitted neighbor is likely already available as initial guess.
- `"sequential"`: Simple index order (for non-map data).

Implementation: Find the pixel closest to the centroid, then greedily select the
nearest unvisited pixel using a KDTree.

**`NeighborPropagator`**: Stores fitted results and provides initial guesses
for neighboring pixels.

- `store_result(idx, p_opt)` — cache a fitted result
- `get_initial_guess(idx, default)` — return the nearest fitted neighbor's
  parameters, or `default` if no neighbors are fitted yet
- Uses KDTree with `k_neighbors=4` for fast spatial lookup

### `optimizer.py` — Fitting Functions

**`fit_single_spectrum(x, y, evaluator, p0, bounds, method, xtol, max_nfev)`**:
Wraps `scipy.optimize.least_squares` for a single spectrum.

- Handles infinite bounds safely (clips p0 within bounds)
- Falls back from `"lm"` to `"trf"` when finite bounds exist
- Returns `(p_opt, success, cost)` tuple

**`fit_batch_sequential(...)`**: Fits all spectra sequentially with optional
neighbor propagation.

- Follows `traversal_order` (spiral for maps, sequential otherwise)
- Uses `NeighborPropagator` to seed initial guesses from fitted neighbors
- Reports progress via callback
- Supports cancellation via `cancel_check` callable

**`fit_batch_threaded(...)`**: Parallel fitting using `ThreadPoolExecutor`.

> **Note**: Threading does NOT help for map fitting because each fit is only
> ~5ms and thread overhead dominates. Profiling showed 4-thread was **0.6× slower**
> than sequential. Threading is available for non-map batches where spatial
> propagation is not used, but is not currently faster for typical workloads.

### `batch_engine.py` — Main Orchestrator

**`BatchFittingEngine`**: The public API of the fitting engine.

```python
engine = BatchFittingEngine()
results = engine.fit_spectra(
    spectra=list_of_MSpectrum,
    fit_model=model_dict,
    coords=np.array([[x1,y1], ...]),  # None for non-map
    fit_params={"method": "leastsq", "xtol": 1e-4, "max_ite": 500},
    ncpus=1,
    progress_callback=lambda current, total: ...,
    cancel_check=lambda: False,
    apply_model_to_spectra=True,  # False for re-fitting
)
```

### `hyper_fit_thread.py` — QThread Wrapper

**`HyperFitThread`**: Drop-in replacement for `ApplyFitModelThread`.

```python
thread = HyperFitThread(
    spectrums=spectra_collection,
    fit_model=model_dict,
    fnames=["fname1", "fname2", ...],
    ncpus=1,
    coords=np.array([[x1,y1], ...]),  # None for non-map
    apply_model_to_spectra=True,
)
thread.progress_changed.connect(on_progress)
thread.finished.connect(on_done)
thread.start()
```

Signal: `progress_changed(current, total, percentage, elapsed_time)`

---

## Data Flow Pipeline

```
User clicks "Apply Fit Model" or "Fit"
         │
         ▼
VMWorkspace._run_fit_thread(fit_model, spectra)
         │
         ▼
HyperFitThread.run()                          [QThread]
  ├── Build fname → spectrum lookup (O(1) dict)
  ├── Extract fit_params from spectrum
  └── Call BatchFittingEngine.fit_spectra()
         │
         ▼
BatchFittingEngine.fit_spectra()
  │
  ├── Step 1: apply_custom_fit_model()        [0.74s for 1681 spectra]
  │     └── For each spectrum: deepcopy model, set peak_models/baseline/range
  │
  ├── Step 2: spectrum.preprocess()           [0.38s]
  │     └── For each spectrum: apply_range → eval_baseline → subtract_baseline
  │
  ├── Step 3: Build PeakModelEvaluator        [<1ms]
  │     └── Parse fit_model → flat param vector + bounds + model functions
  │
  ├── Step 4: Extract data matrix             [<1ms]
  │     └── Detect shared x-axis, build (N, M) Y_matrix
  │
  ├── Step 5: Reinitialize amplitudes         [<1ms]
  │     └── Adjust model's initial amplitudes to match actual data intensity
  │
  ├── Step 6: Choose strategy & FIT           [16.4s]
  │     ├── If coords provided (maps):
  │     │     └── fit_batch_sequential with spiral traversal + propagation
  │     ├── Elif ncpus > 1:
  │     │     └── fit_batch_threaded (no propagation)
  │     └── Else:
  │           └── fit_batch_sequential (no propagation)
  │
  └── Step 7: Write results back              [0.09s]
        └── For each spectrum: build FitResult, set result_fit + peak param hints
```

---

## Performance Strategy

### Why Direct scipy Instead of lmfit

Each `lmfit.Model.fit()` call creates Parameter objects, builds the composite
model, and initializes the minimizer. For 1681 spectra, this overhead adds up
to ~1.7 seconds of pure Python object creation. The batch engine creates the
model evaluator **once** and reuses it for all spectra.

### Why Spatial Propagation Works

Adjacent pixels in a 2D map have nearly identical spectra. When fitting in
spiral order from the center, each pixel's neighbor is already fitted. Using
the neighbor's optimized parameters as the initial guess means the optimizer
starts very close to the solution, converging in 5-20 iterations instead of
50-100+ from a generic initial guess.

### Why Threading Doesn't Help for Maps

Each spectrum fit takes only ~5ms (70 data points, 4 free parameters). With
`ThreadPoolExecutor`, the per-task overhead (GIL acquisition, future creation,
result retrieval) is ~3ms — a 60% overhead. Profiling showed 4-thread parallel
fitting was **0.6× slower** than sequential.

Additionally, spatial propagation is inherently sequential: each pixel depends
on its neighbor's result. Parallelizing would break this dependency chain and
lose the propagation benefit.

### Why Two-Stage Was Removed

The original design used a coarse-then-refined approach (Stage 1: relaxed xtol,
Stage 2: full xtol for outliers only). Profiling revealed this fitted every
spectrum **twice**, doubling the total time from ~9s to ~18s. Since propagation
already provides excellent initial guesses, a single pass with full tolerance
converges just as quickly.

---

## Integration Points

### Maps Workspace (`VMWorkspaceMaps`)

- **`_run_fit_thread()`**: Overridden to use `HyperFitThread` with spatial
  coordinates extracted from the map DataFrame
- **`fit()`**: Overridden to use the batch engine with `apply_model_to_spectra=False`
  (models already assigned from previous apply/paste)
- **Fallback**: `_use_batch_engine = False` falls back to parent's `ApplyFitModelThread`

### Spectra Workspace (`VMWorkspaceSpectra`)

- **`_run_fit_thread()`**: Uses `HyperFitThread` with `coords=None`
  (sequential fitting without spatial propagation)
- **`fit()`**: Uses `HyperFitThread` for the "Fit" button
- **Fallback**: `_use_batch_engine = False` falls back to old `FitThread`

### Compatibility

The engine writes results back to `MSpectrum` objects using the same attributes
the GUI expects:
- `spectrum.result_fit` — `FitResult` (compatible with lmfit's MinimizerResult)
- `spectrum.peak_models[i].param_hints` — updated with fitted values
- `spectrum.x`, `spectrum.y` — unchanged (preprocessed data)

---

## Fitspy Dependency Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Fitting** | ✅ Replaced | BatchFittingEngine + scipy.optimize |
| **Preprocessing** | ⚠️ Still uses fitspy | `spectrum.preprocess()` delegates to fitspy |
| **Baseline management** | ⚠️ Still uses fitspy | `BaseLine` class, eval, anchor points |
| **Peak model management** | ⚠️ Still uses fitspy | `add_peak_model`, `remove_models` |
| **Save/Load** | ⚠️ Still uses fitspy | `spectrum.save()`, `set_attributes()` |
| **MSpectrum inheritance** | ⚠️ Still inherits | `MSpectrum(FitspySpectrum)` |

Full fitspy decoupling is planned for a future iteration.
