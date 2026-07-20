---
paths:
  - "spectroview/fit_engine/**"
---

# Vectorized Batch Fit (VBF) Engine

Deep narrative (math, per-optimization rationale, tuning): `docs/developer/vbf_engine.md`.
This file is the module map + the invariants to preserve when editing `fit_engine/`.

## What it is

A **batched Levenberg-Marquardt** optimizer that fits all N spectra of a map
**simultaneously** as `(N, M[, K])` tensors — no Python loop over spectra. This is
the app's core performance story; keep everything here vectorized.

## Module map

| Module | Role |
|---|---|
| `vbf_thread.py` (`VBFthread`) | `QThread` wrapper. Groups tasks, builds the weights matrix (`_prepare_weights`), calls the engine, **writes results back into `SpectraStore`**. Emits `progress_changed` / `timings_ready`. |
| `vbf_engine.py` (`VBFengine`) | Orchestrates one `fit_spectra()` call: build evaluator → `p0` + noise threshold → optimizer → post-fit noise cleanup → build result arrays. |
| `evaluator.py` (`VBFevaluator`) | Bridge between the dict-based `fit_model` and the flat tensor param space. Free/fixed indexing, expression resolution (chain rule), model routing. |
| `optimizer.py` | Pure `batched_levenberg_marquardt()`. GUI-agnostic. |
| `models.py` | Batched peak functions + **analytical Jacobians**; `BATCHED_MODELS` registry; `numerical_jacobian()` fallback. |
| `scalar_models.py` | Single-spectrum reference fns + `PEAK_MODEL_REGISTRY` (UI preview curves; scalar fallback). |
| `baseline.py` / `noise.py` / `weights.py` / `evaluator*` | Baseline algorithms, noise estimation (`detect_noise_level`), weighting. |

## Invariants to preserve

- **Vectorized, all-at-once.** Manipulate `(N, M)`/`(N, M, K)` tensors; never add a
  per-spectrum Python loop in the hot path. Independent convergence is handled by
  the `active`/`dirty` boolean masks — respect them (they skip converged and
  rejected-step spectra).
- **`np.matmul`, not `np.einsum`** for the normal equations (`J.transpose(0,2,1) @ J`)
  — einsum doesn't dispatch to BLAS here and was 13–20× slower. Don't "tidy" it back.
- **Analytical Jacobians** for every registered shape. When adding a model, provide
  the batched fn **and** its Jacobian, register in `BATCHED_MODELS`, and add the name
  to `PEAK_MODELS` in `spectroview/__init__.py`. Validate the Jacobian against
  `numerical_jacobian()` on random params before trusting it. (Full recipe:
  `vbf_engine.md` §11.)
- **Efficiency pattern in `models.py`**: compute each repeated `1/w`-style term once
  as a reciprocal, reuse via broadcast multiply, write Jacobian slices in place
  (`*=`, `+=`, `out=`). Validate rewrites to ≤1e-10 relative error.

## Traps

- **Worker-thread write-back**: `VBFthread` calls `store.set_fit_results()` and writes
  `md.Y_bestfit`/`md.Y_peaks` from the non-GUI thread. This is safe **only** because
  those are pure Python/NumPy objects — **never touch a Qt widget from inside the
  thread**; do UI updates on the `finished`/`progress_changed` signals.
- **macOS stack size**: `VBFthread` sets an 8 MB stack (`setStackSize`). macOS's
  default 512 KB `QThread` stack overflows LAPACK's on-stack workspace for large K
  and segfaults. Keep this.
- **`fit_params` keys**: `max_ite`, `xtol`, `ftol`, `fit_negative`, `coef_noise` are
  read; `fit_outliers` is a **legacy no-op** kept only for old-file compatibility.
- **`coef_noise` runs noise thresholding twice** (pre- and post-fit) sharing one
  `compute_noise_stats()` result — don't split them into two median passes.

Tests live in `tests/unit/fit_engine/`. Any change to model math or the optimizer
must keep them green and stay validated against the scalar reference implementations.
