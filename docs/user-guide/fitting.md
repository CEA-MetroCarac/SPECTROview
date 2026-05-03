# Tensor Fit Engine

## What Is It?

The Tensor Fit Engine is SPECTROview's high-performance fitting backend. Instead of fitting spectra one at a time, it **fits all spectra simultaneously** using batched matrix operations.

## Why Is It Faster?

| Traditional Approach | Tensor Engine |
|---------------------|---------------|
| Fits spectra one-by-one | Fits all N spectra at once |
| Numerical Jacobians (slow) | Analytical Jacobians (fast) |
| Python function call overhead | Vectorized NumPy/LAPACK operations |
| Sequential execution | Batched tensor math |

**Result**: Typically **10–15× faster**. A 1000-spectrum map that would take 30+ seconds now fits in < 3 seconds.

## How to Use It

The Tensor Engine is the **default** — no special action needed. When you click **Fit** in either workspace, SPECTROview uses it automatically.

## Tuning Parameters

Access via **Settings → Fit Parameters**:

| Setting | Default | Fast Preview | Precision |
|---------|---------|-------------|-----------|
| `xtol` | 1e-4 | 1e-2 | 1e-6 |
| `ftol` | 1e-4 | 1e-2 | 1e-6 |
| `max_ite` | 200 | 50 | 500 |

!!! tip "Quick Preview"
    For rapid previews during model building, increase `xtol` and `ftol` to `1e-2`. The optimizer will converge much faster with only a small accuracy trade-off.

!!! tip "Publication Quality"
    For final results, decrease `xtol` and `ftol` to `1e-6` and increase `max_ite` to 500.

## Supported Peak Models

The tensor engine includes optimized (analytical Jacobian) implementations for:

| Model | Speed |
|-------|-------|
| **Gaussian** | ⚡ Analytical |
| **Lorentzian** | ⚡ Analytical |
| **PseudoVoigt** | ⚡ Analytical |
| GaussianAsym | Numerical fallback |
| LorentzianAsym | Numerical fallback |
| Fano | Numerical fallback |
| DecaySingleExp | Numerical fallback |
| DecayBiExp | Numerical fallback |

!!! note
    Models without analytical Jacobians use a numerical finite-difference approximation, which is slower but still benefits from the batched architecture.
