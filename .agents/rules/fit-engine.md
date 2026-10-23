---
name: fit-engine
description: Vectorized Batch Fit (VBF) engine invariants, batched Levenberg-Marquardt optimizer, analytical Jacobians, and thread safety rules.
trigger: model_decision
---

# Vectorized Batch Fit (VBF) Engine

Deep narrative, math, and tuning rationale are in `docs/developer/vbf_engine.md`. This file contains the invariants and contracts to preserve when modifying `spectroview/fit_engine/`.

---

## 1. Core Purpose & Architectural Invariants

The VBF engine is a custom **batched Levenberg-Marquardt** optimizer that simultaneously fits all $N$ spectra in a map as multi-dimensional tensors (`(N, M)` or `(N, M, K)`), with **zero Python loops over spectra**. This is the core performance engine of `SPECTROview`.

### Invariants to Preserve
1. **Fully Vectorized Hot Path**: Never introduce per-spectrum Python loops in `vbf_engine.py`, `evaluator.py`, `models.py`, or `optimizer.py`. Independent convergence is tracked via boolean masks (`active` and `dirty`). Respect these masks: they prevent redundant evaluation of already-converged spectra.
2. **`np.matmul` Over `np.einsum`**: The normal equations are solved using `J.transpose(0, 2, 1) @ J`. **Do not replace this with `np.einsum`**; `einsum` does not dispatch to BLAS for batched 3D matrix products on CPU and was measured to be 13–20$\times$ slower.
3. **Analytical Jacobians**: Every model in `BATCHED_MODELS` must provide an exact analytical Jacobian. When adding a peak shape:
   - Implement the batched function and its analytical Jacobian in `spectroview/fit_engine/models.py`.
   - Register in `BATCHED_MODELS` and register the scalar reference in `scalar_models.py`.
   - Add the model name to `PEAK_MODELS` in `spectroview/__init__.py`.
   - Validate the Jacobian against `numerical_jacobian()` on random parameters across the parameter domain.
4. **In-Place Arithmetic in `models.py`**:
   - Compute reciprocal terms (`1 / width`) once and broadcast.
   - Accumulate Jacobian slices in place (`*=`, `+=`, `out=`).
   - Validate rewrites against reference implementations to $\le 10^{-10}$ relative error.

---

## 2. Module Map

| Module | Role |
|---|---|
| `vbf_thread.py` (`VBFthread`) | `QThread` wrapper. Prepares weights (`_prepare_weights`), executes engine, writes results back into `SpectraStore`. Emits `progress_changed` and `timings_ready`. |
| `vbf_engine.py` (`VBFengine`) | Orchestrates `fit_spectra()`: builds evaluator, initial parameter guessing (`p0`), noise thresholding, optimizer execution, and post-fit cleanup. |
| `evaluator.py` (`VBFevaluator`) | Bridges dictionary `fit_model` to flat tensor parameter arrays. Handles free/fixed parameter indexing and chain-rule expression resolution. |
| `optimizer.py` | Pure NumPy `batched_levenberg_marquardt()`. Independent of Qt. |
| `models.py` | Batched peak functions and analytical Jacobians; `BATCHED_MODELS` registry. |
| `scalar_models.py` | Single-spectrum reference functions; `PEAK_MODEL_REGISTRY` for UI preview curves. |
| `baseline.py` / `noise.py` / `weights.py` | Baseline subtraction algorithms, noise estimation (`detect_noise_level`), and weight calculations. |

---

## 3. Thread Safety & OS Traps

- **Worker Thread Write-Back**: `VBFthread` directly updates `SpectraStore` (writing `md.Y_bestfit` and `md.Y_peaks`) from a non-GUI background thread. This is safe **only** because these are pure NumPy/Python structures. **Never touch a Qt widget from inside `VBFthread`**. UI updates must occur on slots connected to `progress_changed` or `finished`.
- **macOS Stack Budget**: `VBFthread` explicitly configures an 8 MB stack (`setStackSize(8 * 1024 * 1024)`). macOS's default 512 KB `QThread` stack overflows LAPACK's internal on-stack workspace during large-parameter fits, causing immediate segmentation faults. Keep this configuration.
- **Noise Thresholding**: `coef_noise` applies noise filtering twice (pre-fit and post-fit) using a single `compute_noise_stats()` pass. Do not split this into multiple median passes.
- **Parameters**: `max_ite`, `xtol`, `ftol`, `fit_negative`, `coef_noise` are active parameters. `fit_outliers` is a legacy no-op preserved for backward compatibility with older session files.

---

## 4. Testing

Tests are located in `tests/unit/fit_engine/`. Any modification to optimization logic or peak formulas must pass all unit tests and maintain scalar parity with `tests/unit/fit_engine/test_scalar_models.py`.
