# SPECTROview High-Performance Fitting Engine

> **Maintenance Rule**: This document MUST be updated whenever the fitting engine
> logic, architecture, or module structure is modified. Any PR touching
> `spectroview/core/` or `spectroview/core2/` should include corresponding updates
> to this file.

## Overview

SPECTROview includes **two** fitting engines:

| Engine | Package | Strategy | Best For |
|--------|---------|----------|----------|
| **BatchFittingEngine** (v1) | `spectroview/core/` | Per-spectrum `scipy.optimize.least_squares` | Discrete spectra, small batches |
| **TensorFittingEngine** (v2) | `spectroview/core2/` | All-at-once batched Levenberg-Marquardt | 2D hyperspectral maps (≥100 spectra) |

The **v2 tensor engine** fits ALL spectra in a map simultaneously using
vectorised NumPy tensor operations (`np.einsum`, `np.linalg.solve`),
achieving **~3s for 1521 spectra** — comparable to commercial software
like LabSpec6.

### Performance Comparison (1521-spectrum MoS2 map, 3 Lorentzian peaks)

| Engine | Fit Time | Total Pipeline |
|--------|----------|---------------|
| Old fitspy engine | ~35s | ~35s |
| BatchFittingEngine (v1, core/) | ~16s | ~18s |
| **TensorFittingEngine (v2, core2/)** | **~2-3s** | **~3-4s** |

---

## Engine Architecture

### V2: Tensor Engine (`spectroview/core2/`)

```
spectroview/core2/
├── __init__.py            # Public API exports
├── models.py              # Batched peak functions + analytical Jacobians
├── optimizer.py           # Batched Levenberg-Marquardt solver
├── evaluator.py           # Maps fit_model dict → tensor parameter matrices
├── tensor_engine.py       # Main orchestrator (TensorFittingEngine)
└── tensor_fit_thread.py   # QThread wrapper (TensorFitThread)
```

### V1: Batch Engine (`spectroview/core/`)

```
spectroview/core/
├── __init__.py            # Public API exports
├── models.py              # Scalar peak functions + PeakModelEvaluator + FitResult
├── spatial.py             # Spiral traversal + NeighborPropagator
├── optimizer.py           # scipy.optimize.least_squares wrappers
├── batch_engine.py        # Main orchestrator (BatchFittingEngine)
└── hyper_fit_thread.py    # QThread wrapper (HyperFitThread)
```

---

## V2 Tensor Engine — Core Modules

### `core2/models.py` — Batched Peak Models + Analytical Jacobians

All functions operate on **tensors**:
- Input: `x (M,)`, `params (N, n_p)` — N spectra, n_p parameters per peak
- Output: `Y (N, M)` for evaluation, `J (N, M, n_p)` for Jacobians

| Model | Function | Jacobian | Parameters |
|-------|----------|----------|-----------|
| Lorentzian | `batched_lorentzian` | `batched_lorentzian_jac` | ampli, fwhm, x0 |
| Gaussian | `batched_gaussian` | `batched_gaussian_jac` | ampli, fwhm, x0 |
| PseudoVoigt | `batched_pseudovoigt` | `batched_pseudovoigt_jac` | ampli, fwhm, x0, alpha |
| *Other* | *scalar fallback* | `numerical_jacobian` | *varies* |

**Key design**: Analytical Jacobians eliminate the need for finite-difference
approximation (which would require 2×K extra model evaluations per iteration).
This is the primary reason the tensor engine is ~5× faster than v1.

### `core2/optimizer.py` — Batched Levenberg-Marquardt

**`batched_levenberg_marquardt(x, Y_data, evaluate_fn, jacobian_fn, p0, ...)`**

A custom LM solver that optimises all N spectra simultaneously:

1. **Compute Jacobian**: `J = jacobian_fn(x, p)` → `(N, M, K)` tensor
2. **Normal equations**: `JᵀJ = einsum('nmk,nml→nkl', J, J)` → `(N, K, K)`
3. **Solve**: `np.linalg.solve(JᵀJ + λ·diag, -Jᵀr)` → `(N, K)` step
4. **Accept/reject**: Per-spectrum cost comparison with LM damping update
5. **Convergence**: Per-spectrum tracking, early exit for converged spectra

**Bound handling**: Projected gradient (clip to bounds after each step).
Simple, robust, and avoids the gradient-squashing issues of sigmoid transforms.

**Convergence tracking**: Each spectrum converges independently. The optimizer
skips converged spectra in subsequent iterations, progressively reducing work.

### `core2/evaluator.py` — Tensor Evaluator

**`TensorEvaluator`**: Maps a `fit_model` dict to the tensor API.

Key methods:
- `from_fit_model(dict)` → parse peaks, build parameter layout
- `evaluate(x, p_free)` → `(N, M)` composite model for all spectra
- `jacobian(x, p_free)` → `(N, M, K_free)` Jacobian
- `build_p0_matrix(spectra, x)` → `(N, K_free)` initial guess with per-spectrum amplitude scaling
- `extract_p0_from_spectrum(s)` → `(K_free,)` warm-start from previous fit
- `build_result(p, x, y, ok)` → `FitResult` (GUI-compatible)
- `write_back_to_spectrum(s, fr)` → update `MSpectrum` attributes

**Mixed model support**: Different peaks can use different model types
(e.g. peak 1 = Lorentzian, peak 2 = Gaussian). Each peak type is routed
to its own batched function; results are summed.

**Fixed parameters**: Parameters with `vary=False` are excluded from the
free-parameter vector. The evaluator maintains `_free_idx` / `_fixed_idx`
masks and reconstructs the full vector before evaluation.

### `core2/tensor_engine.py` — Orchestrator

**`TensorFittingEngine.fit_spectra()`**: The main entry point.

Pipeline:
1. Apply fit model to spectra (set peak_models, baseline, range)
2. Build `TensorEvaluator` from model dict
3. Preprocess all spectra (range + baseline subtraction)
4. Extract `(N, M)` data matrix from preprocessed spectra
5. Build `(N, K)` initial parameter matrix (amplitude-scaled or warm-start)
6. Call `batched_levenberg_marquardt()`
7. Write results back to `MSpectrum` objects

### `core2/tensor_fit_thread.py` — QThread Wrapper

**`TensorFitThread`**: Drop-in replacement for `HyperFitThread`.
Same signal interface: `progress_changed(current, total, percent, elapsed)`.

---

## Data Flow Pipeline (V2 Tensor Engine)

```
User clicks "Apply Fit Model" or "Fit"
         │
         ▼
VMWorkspaceMaps._run_fit_thread(fit_model, spectra)
         │
         ▼
TensorFitThread.run()                              [QThread]
  ├── Build fname → spectrum lookup
  └── Call TensorFittingEngine.fit_spectra()
         │
         ▼
TensorFittingEngine.fit_spectra()
  │
  ├── Step 1: apply_custom_fit_model()              [~0.7s]
  │     └── For each spectrum: set peak_models/baseline/range
  │
  ├── Step 2: spectrum.preprocess()                 [~0.4s]
  │     └── For each spectrum: apply_range → eval_baseline → subtract
  │
  ├── Step 3: Build p0 matrix                       [<1ms]
  │     └── Scale amplitudes per spectrum from actual data
  │
  ├── Step 4: TENSOR FIT — batched_levenberg_marquardt()  [~2-3s]
  │     ├── All N spectra optimised simultaneously
  │     ├── Analytical Jacobians (no finite differences)
  │     ├── np.einsum for JᵀJ, Jᵀr assembly
  │     ├── np.linalg.solve for all N normal equations at once
  │     └── Per-spectrum convergence tracking + early exit
  │
  └── Step 5: Write results back                    [~0.1s]
        └── For each spectrum: build FitResult, update param hints
```

---

## Performance Strategy

### Why Tensor (All-at-Once) Instead of Per-Spectrum

The v1 engine calls `scipy.optimize.least_squares` individually for each
spectrum, incurring ~10ms of Python overhead per call. For 1521 spectra,
that's ~15s of overhead alone.

The v2 tensor engine:
- Builds the Jacobian for ALL spectra in a single NumPy call
- Assembles ALL normal equations via `np.einsum` (runs in C/BLAS)
- Solves ALL normal equations via `np.linalg.solve` (LAPACK `dgesv`)
- Per-iteration cost: ~30-50ms for 1521 spectra → 30 iterations = ~1.5s

### Why Analytical Jacobians Matter

Finite-difference Jacobians require `2×K` model evaluations per iteration
(central differences). For K=9 parameters, that's 18 evaluations vs 1
analytical evaluation — an 18× reduction in the costliest operation.

### Why No Spatial Propagation

The v1 engine used spatial propagation (spiral traversal + neighbor seeding)
to get good initial guesses. However, this:
1. Forces **sequential** fitting (each pixel depends on its neighbor)
2. **Contaminates** heterogeneous regions (substrate params propagated into flake pixels)

The v2 engine fits all pixels independently but simultaneously. Each pixel's
initial guess comes from the fit model template + per-spectrum amplitude scaling.

---

## Integration Points

### Maps Workspace (`VMWorkspaceMaps`)

- **`_run_fit_thread()`**: Uses `TensorFitThread` (core2)
- **`fit()`**: Uses `TensorFitThread` with `apply_model_to_spectra=False` (warm-start)
- **Fallback chain**: TensorFitThread → HyperFitThread → parent's FitThread

### Spectra Workspace (`VMWorkspaceSpectra`)

- **`_run_fit_thread()`**: Uses `HyperFitThread` (core/) with `coords=None`
- **`fit()`**: Uses `HyperFitThread` for the "Fit" button
- **Fallback**: `_use_batch_engine = False` falls back to old `FitThread`

### GUI Compatibility

Both engines write results back to `MSpectrum` objects using the same interface:
- `spectrum.result_fit` — `FitResult` (lmfit-compatible `.params[key].value`)
- `spectrum.peak_models[i].param_hints` — updated with fitted values
- `spectrum.bkg_model.param_hints` — updated if background present

---

## Fitspy Dependency Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Fitting (maps)** | ✅ Replaced (core2) | TensorFittingEngine |
| **Fitting (spectra)** | ✅ Replaced (core/) | BatchFittingEngine |
| **Preprocessing** | ⚠️ Still uses fitspy | `spectrum.preprocess()` |
| **Baseline management** | ⚠️ Still uses fitspy | `BaseLine` class |
| **Peak model management** | ⚠️ Still uses fitspy | `add_peak_model` |
| **Save/Load** | ⚠️ Still uses fitspy | `spectrum.save()` |
| **MSpectrum inheritance** | ⚠️ Still inherits | `MSpectrum(FitspySpectrum)` |

Full fitspy decoupling is planned for a future iteration.
