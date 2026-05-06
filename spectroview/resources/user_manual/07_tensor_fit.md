## 7. Tensor Fit Engine

### What is it?
The Tensor Fit Engine is SPECTROview's high-performance fitting backend. Instead of fitting spectra one at a time, it **fits all spectra simultaneously** using batched matrix operations.

### Why is it faster?
- Fits all N spectra at once (Batched tensor math).
- Uses Analytical Jacobians instead of slow Numerical Jacobians.
- Vectorized NumPy/LAPACK operations eliminate Python function call overhead.
**Result**: Typically **10–15× faster**. A 1000-spectrum map that would take 30+ seconds now fits in < 3 seconds.

### Tuning for Performance vs. Accuracy
Access fitting parameters in **Settings → Fit Parameters**:

| Setting | Default | Fast Preview | Precision Fitting |
|---------|---------|-------------|-------------------|
| `xtol` / `ftol` | 1e-4 | 1e-2 | 1e-6 |
| `max_ite` | 200 | 50 | 500 |
