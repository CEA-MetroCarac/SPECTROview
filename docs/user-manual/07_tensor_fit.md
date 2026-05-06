## 7. Tensor Fit Engine

### What is the Tensor Fit Engine?
The Tensor Fit Engine is SPECTROview's proprietary, high-performance mathematical fitting backend. Instead of looping through and fitting spectra one at a time sequentially, it **fits the entire dataset simultaneously** by leveraging batched matrix and tensor operations.

### Why is it significantly faster?

- **Batched Optimization**: It processes all $N$ spectra in a single, parallelized mathematical pass.
- **Analytical Jacobians**: It relies on exact analytical Jacobian formulations rather than computationally expensive numerical approximations.
- **Vectorized Execution**: By executing core logic entirely through vectorized NumPy and LAPACK libraries, it completely eliminates slow Python loop and function call overhead.

**The Result**: The engine operates typically **10–15× faster** than traditional fitting libraries. For instance, a 1000-spectrum hyperspectral map that would traditionally take 30+ seconds to process can now be fitted in under 3 seconds.

### Tuning for Performance vs. Accuracy
You can dynamically adjust the strictness of the fitting algorithms by navigating to **Settings → Fit Parameters**.

| Parameter | Default Value | Recommended for Fast Preview | Recommended for High Precision |
|---------|---------|-------------|-------------------|
| `xtol` / `ftol` (Tolerance) | 1e-4 | 1e-2 | 1e-6 |
| `max_ite` (Maximum Iterations) | 200 | 50 | 500 |
