# Developer Guide: Tensor Fit Engine

This document provides a deep dive into the inner workings, architecture, and performance characteristics of the Tensor Fit Engine located in `spectroview/fit_engine/`.

## 1. Why the Tensor Engine is Much Faster

The legacy fit engines (based on `lmfit`/`scipy.optimize.least_squares`) operate on a **per-spectrum** basis. For a hyperspectral map containing thousands of spectra, this approach introduces significant overhead:
- **Python Function Call Overhead**: Calling the objective function and Jacobian estimator thousands of times per iteration.
- **Finite-Difference Jacobians**: Approximating the Jacobian numerically requires `2 * K` (where K is the number of parameters) additional function evaluations per iteration, per spectrum.
- **Sequential Execution**: Even with multiprocessing, the overhead of serialization and inter-process communication creates bottlenecks.

The **Tensor Fit Engine** achieves massive speedups (~10x to 15x faster) through the following core principles:

1.  **All-at-Once Optimization**: It optimizes all \(N\) spectra simultaneously. The parameter matrices, data arrays, and residuals are manipulated as large 2D or 3D tensors.
2.  **Vectorized Operations (BLAS/LAPACK)**: By framing the problem as tensors, the heavy lifting is offloaded to highly optimized C/Fortran libraries.
    - Matrix multiplications and transpositions for the normal equations (\(J^T J\) and \(J^T r\)) are performed using `np.einsum`.
    - The linear systems for all spectra are solved in a single call to `np.linalg.solve`, which dispatches to LAPACK.
3.  **Analytical Jacobians**: Instead of estimating derivatives numerically, the engine uses exact analytical formulas for common peak shapes (Lorentzian, Gaussian, PseudoVoigt). This eliminates the \(2K\) extra evaluations entirely.
4.  **No Spatial Propagation**: Unlike older map-fitting approaches that used spiral traversal to propagate guesses from neighbor to neighbor (forcing sequential execution), the tensor engine initializes all pixels independently using amplitude scaling, allowing purely parallel tensor math.
5.  **Variable Length Support**: Spectra with different lengths (e.g. differently cropped) are padded with zeros to fit into a uniform 2D tensor, while a boolean `weights` mask ensures padded regions are ignored during optimization.
6.  **Expression Support**: Supports complex mathematical relationships between parameters across the batch by evaluating mathematical constraints symbolically before mapping to free parameters.

---

## 2. Code Logic and Core Implementation Principles

The engine implements a **Batched Levenberg-Marquardt** algorithm.

### The Mathematics of Batched LM
For \(N\) spectra, each with \(M\) wavelength points and \(K\) free parameters:

1.  **Evaluate Model**: \(\mathbf{Y}_{pred} = f(\mathbf{x}, \mathbf{p})\), returning an \((N, M)\) tensor.
2.  **Calculate Residuals**: \(\mathbf{r} = \mathbf{W} \circ (\mathbf{Y}_{pred} - \mathbf{Y}_{data})\), returning an \((N, M)\) tensor.
3.  **Calculate Jacobian**: \(\mathbf{J} = \frac{\partial f}{\partial \mathbf{p}}\), returning an \((N, M, K)\) tensor.
4.  **Normal Equations**: Assemble \(J^T J\) (size \(N \times K \times K\)) and \(J^T r\) (size \(N \times K\)).
5.  **Damping (Marquardt step)**: Add a damping factor \(\lambda_i\) to the diagonal of \(J^T J\) for each spectrum \(i\).
6.  **Solve**: Solve \((J^T J + \lambda \text{diag}(J^T J)) \delta \mathbf{p} = -J^T r\) for all \(N\) spectra simultaneously.
7.  **Evaluate Step**: Update \(\mathbf{p} \leftarrow \mathbf{p} + \delta \mathbf{p}\) (with projection to bounds) and evaluate the new cost. Adjust \(\lambda\) per spectrum based on success/failure.

### Independent Convergence
Even though the math is batched, each spectrum converges independently. The optimizer uses a boolean mask (`active = ~converged`) to skip Jacobian calculations and linear solves for spectra that have already reached the tolerance limits, progressively speeding up the later iterations.

---

## 3. Folder and Class Structure

The engine is contained within `spectroview/fit_engine/` and consists of the following modules:

*   **`tensor_engine.py`**: Contains the orchestrator class `TensorFittingEngine`. It manages the high-level workflow: preprocessing spectra, extracting matrices, calling the optimizer, and writing results back to the GUI objects.
*   **`optimizer.py`**: Contains the core mathematical workhorse `batched_levenberg_marquardt`. This is purely numerical and agnostic to the GUI objects.
*   **`evaluator.py`**: Contains `TensorEvaluator`. This class acts as the bridge between the flexible dictionary-based `fit_model` (which supports fixed parameters, bounds, and mixed peak types) and the rigid, flat, free-parameter tensors required by the optimizer.
*   **`models.py`**: Defines the batched model evaluation functions (e.g., `batched_lorentzian`) and their corresponding analytical Jacobian functions (e.g., `batched_lorentzian_jac`).
*   **`scalar_models.py`**: Contains fallback scalar models and the `FitResult` data class.
*   **`tensor_fit_thread.py`**: Contains `TensorFitThread`, a `QThread` wrapper that runs the engine asynchronously to prevent GUI freezing, and emits progress signals.

---

## 4. Processing Pipeline / Execution Flow

When a user triggers a fit (e.g., via the "Fit" button in the Maps workspace), the following pipeline executes:

1.  **Thread Invocation**: `VMWorkspaceMaps._run_fit_thread()` instantiates and starts `TensorFitThread`.
2.  **Engine Initialization**: `TensorFittingEngine.fit_spectra()` is called.
3.  **Model Application (Optional)**: If `apply_model_to_spectra=True`, the `fit_model` dictionary is applied to all `MSpectrum` objects, ensuring they have the correct number and type of peaks. If `False` (like in the Spectra workspace), the engine groups spectra by their existing model signatures and processes batches sequentially, preserving individual spectrum customizations.
4.  **Preprocessing**: `spectrum.preprocess()` is called for all spectra (applying spectral ranges, evaluating and subtracting baselines). Spectra are then zero-padded into a uniform 2D matrix.
5.  **Evaluator Construction**: `TensorEvaluator.from_fit_model()` parses the model, determining which parameters are free vs. fixed, and mapping them to a flat vector space.
6.  **Data Extraction**: The \((N, M)\) data matrix \(\mathbf{Y}\) and initial parameter matrix \(\mathbf{p_0}\) are extracted. If it's a first fit, \(\mathbf{p_0}\) amplitudes are scaled to the actual data. If it's a re-fit, \(\mathbf{p_0}\) is extracted exactly from the existing fits (warm start) to allow continued optimization.
7.  **Optimization**: `batched_levenberg_marquardt()` runs the iterations until all spectra converge or `max_iter` is reached.
8.  **Result Writeback**: The optimized free parameters are reconstructed into full parameter sets by the evaluator, and written back to the `spectrum.result_fit` and `param_hints` of each `MSpectrum`.

---

## 5. Optimization Parameters and Adjustments

The engine behavior can be tuned via the `fit_params` dictionary passed to `fit_spectra()`.

### Key Parameters
*   **`max_ite` (default: 200)**: The maximum number of Levenberg-Marquardt iterations. Increasing this might help extremely difficult spectra converge but will increase total execution time.
*   **`xtol` (default: 1e-4)**: The relative tolerance for the parameter step size \(\delta p\). If the relative change in all parameters is less than `xtol`, the spectrum is considered converged.
*   **`ftol` (default: 1e-4)**: The relative tolerance for the cost function (sum of squared residuals). If the relative change in the cost is less than `ftol`, the spectrum is considered converged.

### Tuning for Performance vs. Accuracy
- **Fast Mapping**: For rapid previews, you can increase `xtol` and `ftol` to `1e-3` or `1e-2`. The optimizer will exit much earlier, providing a rough fit in a fraction of the time.
- **Precision Fitting**: For publication-quality results, decrease `xtol` and `ftol` to `1e-5` or `1e-6`.
- **Handling "Stuck" Spectra**: The optimizer tracks `consecutive_rejects`. If a spectrum's cost fails to improve for 15 consecutive iterations (despite damping adjustments), it is marked as converged (stuck) to prevent it from holding back the rest of the batch. This threshold (`MAX_REJECTS` in `optimizer.py`) can be adjusted if needed.

### Adding New Peak Models
To add a new peak shape to the fast tensor engine:
1.  Define the `batched_newshape(x, params)` function in `models.py`.
2.  (Crucial for speed) Derive and define the analytical Jacobian `batched_newshape_jac(x, params)`.
3.  Register the model in the `BATCHED_MODELS` dictionary at the bottom of `models.py`. If you skip the Jacobian, the engine will fall back to `numerical_jacobian`, drastically reducing performance.
