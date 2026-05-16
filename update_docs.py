import re

with open('docs/developer/tensor-engine.md', 'r') as f:
    content = f.read()

# Replace Section 1 (Why the Tensor Engine is Much Faster)
# Actually, I will add Performance Optimizations section at the end of section 1.

perf_section = """
## 1.5 Performance Optimizations

Recent optimizations have achieved a 2.2x to 5x speedup across various datasets:

1. **Template Model Application (11.9x speedup)**: `_prepare_fit_model_template()` pre-computes lmfit Models and bounds sanitization *once*. Per-spectrum application only clones the lightweight `param_hints` dictionary.
2. **Batch Preprocessing (3.0x speedup)**: `_batch_preprocess()` computes range masks and baseline indices once for all spectra sharing the same x-axis, using vectorized subtraction instead of per-spectrum calls.
3. **Adaptive Batched Solver (up to 5x speedup)**: The solver dynamically chooses between NumPy's batched `np.linalg.solve` (best for large N) and SciPy's `cho_solve` (Cholesky decomposition, best for small N / large K, exploiting symmetric positive-definite normal equations).
4. **Mean-Based Convergence Criteria**: The convergence check uses `mean(|Δp| / |p|) < xtol` rather than `max()`, preventing a single slowly-converging parameter in multi-peak models from stalling the entire batch.
5. **Vectorized Write-Back (3.5x speedup)**: `build_results_batch()` evaluates all best-fit curves and computes R² simultaneously across all spectra using a single `_to_full()` expansion.
6. **Zero-Weight Early Exit**: Spectra identified as purely noise (all weights zero) are marked as converged instantly, saving all Levenberg-Marquardt iterations.
"""

content = content.replace("## 2. Code Logic and Core Implementation Principles", perf_section + "\n## 2. Code Logic and Core Implementation Principles")


# Replace Step 6 in Section 2
old_step6 = "6.  **Solve**: Solve \((J^T J + \lambda \text{diag}(J^T J)) \delta \mathbf{p} = -J^T r\) for all \(N\) spectra simultaneously."
new_step6 = "6.  **Solve**: Solve \((J^T J + \lambda \text{diag}(J^T J)) \delta \mathbf{p} = -J^T r\). The engine adaptively selects between batched NumPy `solve` and SciPy `cho_solve` (Cholesky decomposition) depending on \(N\) and \(K\) to maximize speed."
content = content.replace(old_step6, new_step6)


# Update Class Table
old_opt = "| `optimizer.py` | `batched_levenberg_marquardt()` | Pure numerical optimizer. Solves N independent least-squares problems simultaneously using `np.einsum` for normal equations and `np.linalg.solve` for the linear system. GUI-agnostic. |"
new_opt = "| `optimizer.py` | `batched_levenberg_marquardt()` | Pure numerical optimizer. Solves N independent least-squares problems simultaneously using `np.einsum`. Uses an adaptive solver (`cho_solve` or `np.linalg.solve`) depending on matrix size. GUI-agnostic. |"
content = content.replace(old_opt, new_opt)


# Update Section 4 Processing Pipeline
old_sec4 = """**Step 1 — Model Application** (`apply_model_to_spectra=True`):
The `fit_model` dictionary is applied to all `MSpectrum` objects via `apply_custom_fit_model()`, ensuring they have the correct number and type of peaks. When `False` (Spectra workspace with per-spectrum models), the `TensorFitThread` pre-groups spectra by model signature and processes each group as a separate batch.

**Step 2 — Evaluator Construction**:
`TensorEvaluator.from_fit_model()` iterates over `peak_models` in the fit model dict. For each peak, it:

- Looks up the model name in `BATCHED_MODELS` (fast path) or `PEAK_MODEL_REGISTRY` (scalar fallback)
- Extracts `param_hints` (value, min, max, vary, expr) for each parameter
- Assigns a sequential prefix (`m01_`, `m02_`, ...) to parameter names
- Builds `_free_idx` / `_fixed_idx` arrays for the free ↔ full parameter mapping
- Parses expression strings (e.g., `m01_fwhm`) and marks linked parameters as fixed

**Step 3 — Preprocessing**:
Calls `spectrum.preprocess()` on spectra that haven't been preprocessed yet (baseline evaluation, spectral range cropping)."""

new_sec4 = """**Step 1 — Model Application** (`apply_model_to_spectra=True`):
Instead of a slow per-spectrum deepcopy, `_prepare_fit_model_template()` creates a highly optimized template. It builds lmfit Model objects and sanitizes parameter bounds exactly once. Then, `_apply_template_to_spectrum()` merely copies the lightweight `param_hints` dictionary. When `False` (Spectra workspace with per-spectrum models), spectra are grouped by model signature and processed as separate batches.

**Step 2 — Evaluator Construction**:
`TensorEvaluator.from_fit_model()` parses the model. For each peak:
- Looks up the model name in `BATCHED_MODELS` (fast path) or `PEAK_MODEL_REGISTRY` (scalar fallback).
- Assigns a sequential prefix (`m01_`, `m02_`, ...) and extracts `param_hints`.
- Builds `_free_idx` / `_fixed_idx` mappings between optimized free parameters and the full parameter set.
- Parses expression strings (e.g., `m01_fwhm = m02_fwhm`) and maps dependencies.

**Step 3 — Preprocessing**:
`_batch_preprocess()` uses a batched strategy. For spectra sharing the same x-axis (e.g. hyperspectral maps), it computes the range crop mask and evaluates the baseline indexing **once**. This mask and baseline are then rapidly applied via NumPy array slicing and subtraction to all spectra. For complex modes (like arpls) or variable lengths, it falls back to per-spectrum `spectrum.preprocess()`."""

content = content.replace(old_sec4, new_sec4)

# Update Step 7 Result Writeback
old_step7 = """**Step 7 — Result Writeback**:
For each spectrum, `build_result()` reconstructs the full parameter vector, evaluates the best-fit curve, and computes R². Then `write_back_to_spectrum()` writes the optimized values back to each `peak_model.param_hints` and sets `spectrum.result_fit`."""

new_step7 = """**Step 7 — Result Writeback**:
`evaluator.build_results_batch()` uses a vectorized approach. It performs a single `_to_full()` expansion for the entire \(N \times K\) parameter matrix, calls `evaluate()` once to generate all best-fit curves, and calculates \(R^2\) via array operations. The data is then packaged into `FitResult` objects and assigned to `spectrum.result_fit`."""

content = content.replace(old_step7, new_step7)

# Update xtol explanation
old_xtol = "*   **`xtol` (default: 1e-4)**: The relative tolerance for the parameter step size \(\delta p\). If the relative change in all parameters is less than `xtol`, the spectrum is considered converged."
new_xtol = "*   **`xtol` (default: 1e-4)**: The relative tolerance for the parameter step size \(\delta p\). Convergence is reached when the **mean** relative change across all parameters (\(\operatorname{mean}(|\delta p| / |p|)\)) is less than `xtol`. This mean-based criterion ensures that a single slowly oscillating parameter does not artificially delay convergence for the entire spectrum."
content = content.replace(old_xtol, new_xtol)

# Update Noise Threshold (Mechanism B)
old_mechb = """#### Mechanism B — Peak Suppression (`apply_noise_threshold`)

In the `TensorEvaluator`, any peak whose **center position** (`x0`) falls in a below-threshold region has its initial parameters forcibly set to zero:

```python
for each peak:
    x0_val = peak center position
    if ymean[at x0] < noise_level:
        ampli = 0.0    # force amplitude to zero
        fwhm  = 0.0    # force width to zero
```"""

new_mechb = """#### Mechanism B — Peak Suppression (`apply_noise_threshold`)

In the `TensorEvaluator`, any peak whose **center position** (`x0`) falls in a below-threshold region has its amplitude and shape parameters suppressed, and its positional parameters restored to the initial guess:

```python
for each peak:
    x0_val = peak center position
    if ymean[at x0] < noise_level:
        ampli = 0.0    # force amplitude to zero
        fwhm  = 0.0    # force width to zero
        # Additionally, x0 and other shape parameters are restored 
        # to p0_matrix (initial guesses) to prevent random fluctuation mapping
```"""

content = content.replace(old_mechb, new_mechb)

with open('docs/developer/tensor-engine.md', 'w') as f:
    f.write(content)
print("Updated tensor-engine.md")
