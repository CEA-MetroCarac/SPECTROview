# Performance Audit & Optimization Roadmap: SPECTROview VBF Fitting Engine

**Document Status**: Audited, Empirically Benchmarked & Phased Roadmap  
**Primary Modules Covered**:

- `spectroview/fit_engine/optimizer.py`, `spectroview/fit_engine/evaluator.py`, `spectroview/fit_engine/models.py`
- `spectroview/fit_engine/vbf_engine.py`, `spectroview/fit_engine/vbf_thread.py`, `spectroview/fit_engine/baseline.py`
- `spectroview/main.py` (background prewarming)

**Benchmark Datasets & Configs Audited**:

- `examples/fit_benchmarking_data/3_3721map.wdf` (`N=3,721`, `M=575`, `[55.94..958.82 cm⁻¹]`, `float32`)
- `examples/fit_benchmarking_data/1_CL_map.txt` (`N=16,384`, `M=1,024`, `[179.49..720.14]`, `float32`)
- Supplementary cross-validation: `examples/spectroscopic_data/2dmap_spc.spc` (`N=1,681`, `M=1,024`) and `examples/spectroscopic_data/mapls6.l6m` (`N=1,368`, `M=1,024`)
- Fitting configs in `examples/fit_benchmarking_data/`: `1_CL_map*.json` (Group A) and `3_3721map*.json` (Group B)

---

## 1. Executive Summary

An empirical audit and profiling study of SPECTROview's **Vectorized Batch Fit (`VBF`) Engine** was conducted across real Raman (`3_3721map.wdf`) and cathodoluminescence (`1_CL_map.txt`) hyperspectral datasets. The investigation explains why runtime escalates sharply when switching from Lorentzian to Pseudo-Voigt profiles or adding multiple overlapping peaks, and separates **per-iteration computational cost** from **convergence-count inflation**.

### Key Findings

1. **Benchmark Group A and Group B Test Different Spectra and Solver Tolerances**:

   - **Group A (`1_CL_map*.json`)** crops `[319.70..480.79]` (`M=102` on `3_3721map.wdf`, `M=304` on `1_CL_map.txt`). On `3_3721map.wdf`, `[319.70..480.79]` is a signal-free region (`med_peak = 6.81`, `frac_neg = 62.2%`), causing unguided fitting into baseline noise (`R²_med <= 0.00`). On the intended dataset `1_CL_map.txt`, real CL peaks sit near `387.7–394.8` (`R²_med = 0.9841–0.9906`).
   - **Group B (`3_3721map*.json`)** crops `[460.23..570.28]` (`M=71`) or `[460.23..579.64]` (`M=77`), centered on the strong silicon Raman peak at `521.73 cm⁻¹` (`med_peak = 6,695.98`, `R²_med = 0.9988–0.9996`).
   - Furthermore, `1_CL_map.json` and `3_3721map.json` use `coef_noise = 1` (which zeroes all weights on signal-free spectra, skipping 3,477/3,721 fits on `3_3721map.wdf`), whereas the multi-peak and Pseudo-Voigt configs set `coef_noise = 0.0` (fitting every spectrum, including noise).

2. **Convergence Cost Dominates Multi-Peak Slowdowns (25× Iteration Multiplier), Amplified by Bound-Clipping**:

   - A single well-conditioned Lorentzian (`3_3721map.json`) or Pseudo-Voigt (`3_3721map_1pseudo.json`) converges in **4 iterations**.
   - Adding two weak satellite peaks alongside the Si peak (`3_3721map_3Lorent.json`, starting with Peak 1 at `x0=497.15, a=0.0, w=0.10`) stalls convergence: only **44/3,721 (1.2%)** converge within `max_ite = 100`, forcing **99.3% mean active rows** across all **100 iterations** (a **25× iteration multiplier** over the 1-peak case: `0.315 s -> 7.419 s`).
   - **Verified Root Cause (`P1.1`)**: In `optimizer.py`, the convergence check measures `np.abs(dp)` **before** box-bound projection (`np.clip(p_active + dp, lo, hi)`). When any peak amplitude or `PseudoVoigt` `alpha` hits `0.0` (verified directly: `1,494` rows pinned at `P1_alpha=0.0` on `1_CL_map_1pseudovoigt.json`), `|p_trial - p| == 0` and cost improvement is `0`, yet the unconstrained Newton step `dp` remains nonzero, blocking convergence until `max_ite`. Fixing this single line yields a **`4.14×` speedup** on `3_3721map_3Lorent.json` (`8.620 s -> 2.082 s`, convergence rising from `44/3,721` to `3,486/3,721`) and **`1.84×`** on `1_CL_map_1pseudovoigt.json`.

3. **Per-Iteration Cost Drivers (Fixed-10-Iteration Controlled Measurements)**:

   - **Why `1 Pseudo-Voigt` is `3.18×–4.23×` slower per iteration than `1 Lorentzian`**: `batched_pseudovoigt()` and `batched_pseudovoigt_jac()` recompute `batched_gaussian()`, `batched_lorentzian()`, `batched_gaussian_jac()`, and `batched_lorentzian_jac()` independently — recalculating `dx`, `dx²`, `1/w²`, and normalized offset `t²` **four times per Jacobian call** and allocating temporary `(N, M)` and `(N, M, 3)` buffers before copying into `(N, M, 4)`. Fixed-10-iteration cost grows `46.54 ms -> 148.13 ms` (`3.18×`) on `3,721×77` and `576.95 ms -> 2,438.98 ms` (`4.23×`) on `16,384×304`.
   - **Why `VBFevaluator.jacobian()` accounts for ~50–60% of self+cumulative time**: In `cProfile`, `jacobian()` spends more self time copying slices (`_to_full()`, `np.zeros()`, `J_full[:, :, slc] = jac_fn(...)`, and fancy-index extraction `J_full[:, :, self._free_idx]`) than `models.py` spends computing derivatives. At `N=16,384, M=304, 6_Lor (K=18)` over 10 fixed iterations, `JacCopy = 2,685.81 ms` actually exceeds `JacMath = 2,114.60 ms`.
   - **`_batched_solve()` Cholesky Crossover Cliff (`Na <= 500, K >= 10`)**: As tail spectra stay active and drop below `501` rows while `K >= 10`, `_batched_solve()` switches from C batched `np.linalg.solve` to a Python loop calling `scipy.linalg.cho_factor` / `cho_solve`, creating a measured **`5.7×–33.8×` per-call regression** (`Na=450, K=12`: `89.883 ms` vs `2.661 ms`).
   - **Working-Set Growth & Cache Spilling (`O(N * M * K)`)**: Holding full active tensors in DRAM instead of L2/L3 cache-sized batches forces repeated DRAM round-trips for residuals, Jacobians, and normal equations; additionally, full-axis dense evaluation wastes compute far outside localized Raman peaks.
   - **Cold Linear Baseline Import**: First-ever invocation of Linear baseline fitting triggers a cold `from scipy.interpolate import interp1d` costing **`1,508.64 ms`** (`0.75 ms` warm).

### Roadmap Overview Across Three Phases

| Phase | Scope | Expected Cumulative Gains | Platform Risk | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1 (`P1.1`–`P1.5`)** | Post-clip `dp` convergence fix, uniform `np.linalg.solve` primary path, fused `PseudoVoigt`, zero-copy free-param `VBFevaluator`, pure-NumPy linear baseline & O(N) threshold index | **1.53×–4.58×** real map fits; **1.60×–4.36×** per-iter kernels; **854×** cold 2-pt baseline (`scipy`-free) | Minimal (Pure NumPy/Python, verified `R²` parity) | **Implemented & Verified** |
| **Phase 2 (`P2.1`–`P2.4`)** | L3-cache batch chunking + `active_idx` compaction, inactive/bound-pinned peak convergence handling, safe support-window slicing, BLAS thread caps & warm worker pool | **1.7×–2.0×** additional pure-Python cache/parallel gain + up to **3×–10×** on localized multi-peak bands | Low–Moderate (cross-platform Python/NumPy) | Planned Future Milestone |
| **Phase 3 (`P3.1`–`P3.3`)** | Optional C++/OpenMP L1-resident single-spectrum LM kernel (prebuilt wheels + NumPy fallback), Variable Projection (VarPro) / spatial warm-start, vectorized dual-number AD for `_exprs` | **10×–30×+** approach toward LabSpec-class interactive sub-second 20k-spectrum fitting | Higher build/packaging complexity (native wheels) | Planned Long-Term Architecture |

---

## 2. Empirical Audit Findings

### 2.1 Configuration Comparison: Benchmark Group A vs. Group B

Inspecting all 7 configuration JSONs in `examples/fit_benchmarking_data/` reveals critical structural differences:

| Config | Peak Model(s) | Initial Peak Centers | ROI / Points on WDF | ROI / Points on `1_CL_map.txt` | Baseline | Fit Params (`max_ite`, `xtol`, `ftol`, `coef_noise`, `loss`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `1_CL_map.json` | 1 Lorentzian (`K=3`) | `390.15` (`a=136.8, w=8.04`) | `[319.70..480.79]`, `M=102` | `M=304` | Linear (`sub=True`) | `200`, `1e-4`, `None`, `coef_noise=1`, `None` |
| `1_CL_map_1pseudovoigt.json` | 1 PseudoVoigt (`K=4`) | `390.16` (`a=132.6, w=8.52`) | `[319.70..480.79]`, `M=102` | `M=304` | Linear (`sub=True`) | `100`, `1e-3`, `1e-3`, `coef_noise=0.0`, `linear` |
| `1_CL_map_2lorentzian.json` | 2 Lorentzians (`K=6`) | `394.45` (`a=16.9`), `389.49` (`a=102.8`) | `[319.70..480.79]`, `M=102` | `M=304` | Linear (`sub=True`) | `100`, `1e-3`, `1e-3`, `coef_noise=0.0`, `linear` |
| `1_CL_map_3lorentzian.json` | 3 Lorentzians (`K=9`) | `394.75`, `390.33`, `387.69` | `[319.70..480.79]`, `M=102` | `M=304` | Linear (`sub=True`) | `100`, `1e-3`, `1e-3`, `coef_noise=0.0`, `linear` |
| `3_3721map.json` | 1 Lorentzian (`K=3`) | `521.29` (`a=9146.3, w=4.84`) | `[460.23..570.28]`, `M=71` | N/A | Linear (`sub=True`) | `500`, `1e-4`, `None`, `coef_noise=1`, `None` |
| `3_3721map_1pseudo.json` | 1 PseudoVoigt (`K=4`) | `521.32` (`a=5873.1, w=5.06`) | `[460.23..579.64]`, `M=77` | N/A | Linear (`sub=True`) | `100`, `1e-3`, `1e-3`, `coef_noise=0.0`, `linear` |
| `3_3721map_3Lorent.json` | 3 Lorentzians (`K=9`) | `497.15` (`a=0`), `519.85`, `521.59` | `[460.23..579.64]`, `M=77` | N/A | Linear (`sub=True`) | `100`, `1e-3`, `1e-3`, `coef_noise=0.0`, `linear` |

### 2.2 Reproduced Baseline Benchmark Results (Untouched Engine)

**A. Measured on `3_3721map.wdf` (`N=3,721` spectra, `M=575` raw, load time `26.69 ms`)**

| Group & Config | Shape `(N, M, K)` | Fit Time | ms / spec | Iters Used | Mean Active | Converged | Median $R^2$ | `eval` / `jac` / `solve` / `other` | Peak Heap (`tracemalloc`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: |
| A: `1_CL_map` | `(3721, 102, 3)` | `0.2968 s` | `0.080` | `200 / 200` | `4.2%` | `3,641 (97.9%)` | `0.0000` | `11.3%` / `19.7%` / `6.5%` / `59.5%` | `16.34 MiB` |
| A: `1_CL_map_1pseudovoigt` | `(3721, 102, 4)` | `8.3554 s` | `2.245` | `100 / 100` | `90.8%` | `776 (20.9%)` | `-0.0455` | `23.6%` / `50.5%` / `2.9%` / `21.0%` | `62.28 MiB` |
| A: `1_CL_map_2lorentzian` | `(3721, 102, 6)` | `6.1183 s` | `1.644` | `100 / 100` | `98.2%` | `152 (4.1%)` | `-0.0260` | `18.4%` / `42.0%` / `5.6%` / `31.5%` | `61.23 MiB` |
| A: `1_CL_map_3lorentzian` | `(3721, 102, 9)` | `8.0891 s` | `2.174` | `100 / 100` | `98.9%` | `83 (2.2%)` | `-0.0197` | `18.6%` / `44.9%` / `7.1%` / `27.6%` | `81.36 MiB` |
| B: `3_3721map` | `(3721, 71, 3)` | `0.1612 s` | `0.043` | `4 / 500` | `75.9%` | `3,721 (100%)` | `0.9988` | `10.7%` / `30.5%` / `3.1%` / `52.7%` | `45.67 MiB` |
| B: `3_3721map_1pseudo` | `(3721, 77, 4)` | `0.3149 s` | `0.085` | `4 / 100` | `66.7%` | `3,721 (100%)` | `0.9993` | `16.5%` / `55.0%` / `2.4%` / `24.6%` | `64.45 MiB` |
| B: `3_3721map_3Lorent` | `(3721, 77, 9)` | `7.4189 s` | `1.994` | `100 / 100` | `99.3%` | `44 (1.2%)` | `0.9996` | `18.0%` / `43.0%` / `8.6%` / `28.6%` | `82.62 MiB` |

*Note on `3_3721map_3Lorent` via full `VBFthread`*: Total wall time was `9.469 s`; `SpectraStore.set_fit_results + Y` write-back took `0.55 ms`, and `build_fit_results_df((3721, 14))` took `4.10 ms` — proving PySide6/store write-back is negligible compared to the LM loop.

**B. Group A Measured on `1_CL_map.txt` (`N=16,384` spectra, `M=1,024` raw, `M=304` cropped, load `0.92 s`)**

| Config | Fit Time | ms / spec | Iters Used | Mean Active | Converged | Median $R^2$ | `eval` / `jac` / `solve` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `1_CL_map` (`K=3`) | `3.9432 s` | `0.241` | `34 / 200` | `12.1%` | `16,384 (100%)` | `0.9841` | `0.442 s` / `1.539 s` / `0.036 s` |
| `1_CL_map_1pseudovoigt` (`K=4`) | `13.5854 s` | `0.829` | `100 / 100` | `10.2%` | `15,782 (96.3%)` | `0.9859` | `2.612 s` / `7.656 s` / `0.118 s` |
| `1_CL_map_2lorentzian` (`K=6`) | `35.9672 s` | `2.195` | `100 / 100` | `47.6%` | `14,424 (88.0%)` | `0.9889` | `6.396 s` / `17.214 s` / `0.729 s` |
| `1_CL_map_2lorentzian` (`max_ite=200, tol=1e-4`) | `51.96 s` | `3.171` | `200 / 200` | — | `15,358 (93.7%)` | `0.9889` | Extended tol sensitivity check |
| `1_CL_map_3lorentzian` (`K=9`) | `75.7279 s` | `4.622` | `100 / 100` | `61.2%` | `12,674 (77.4%)` | `0.9906` | `13.993 s` / `39.577 s` / `2.035 s` |

**C. Supplementary Maps Cross-Validation (`2dmap_spc.spc` & `mapls6.l6m`, `M=1,024`)**

- `2dmap_spc.spc` (`N=1,681, M=1,024`): `1p_lor` finished in `2.405 s` (`76 iters`), whereas `3p_lor` (`K=9`) hit all `200 iters` (`79.348 s`, `47.203 ms/spec`), `3p_pv` (`K=12`) took `228.776 s` (`jac = 57.9%, eval = 32.6%`), and `6p_lor` (`K=18`) took `152.585 s` (`peak = 788.57 MiB` over 25 iters). Applying a tight ROI (`M=116`) reduced `3p_lor` (`200 iters`) from `73.622 s` to `10.872 s` (`6.77×`), confirming linear `O(M)` scaling.
- `mapls6.l6m` (`N=1,368, M=1,024`, `3p_lor`): `cProfile` (`41.455 s` total) confirmed `evaluator.jacobian` (`tottime = 11.228 s`, `cumtime = 20.911 s`) consumed **more self-time** than `batched_lorentzian_jac` (`tottime = 9.613 s`), followed by `batched_levenberg_marquardt` (`tottime = 9.166 s`) and `batched_lorentzian` (`6.609 s`).

### 2.3 Fixed-10-Iteration Computational Cost Scaling

Holding iteration count fixed at **10 iterations** eliminates convergence differences and isolates pure per-iteration compute cost:

| Model (`K` params) | `3_3721map.wdf` (`N=3,721, M=77`) | Ratio vs `1_Lor` | `1_CL_map.txt` (`N=16,384, M=304`) | Ratio vs `1_Lor` |
| :--- | ---: | ---: | ---: | ---: |
| `1_Lor` (`K=3`) | `46.54 ms` | `1.00×` | `576.95 ms` | `1.00×` |
| `1_Gauss` (`K=3`) | `73.67 ms` | `1.58×` | `1,031.39 ms` | `1.79×` |
| `1_PV` (`K=4`) | `148.13 ms` | `3.18×` | `2,438.98 ms` | `4.23×` |
| `2_Lor` (`K=6`) | `92.88 ms` | `2.00×` | `1,410.73 ms` | `2.45×` |
| `3_Lor` (`K=9`) | `139.48 ms` | `3.00×` | `2,617.44 ms` | `4.54×` |
| `4_Lor` (`K=12`) | `217.41 ms` | `4.67×` | `2,706.76 ms` | `4.69×` |
| `6_Lor` (`K=18`) | `281.59 ms` | `6.05×` | `6,803.68 ms` | `11.79×` |
| `3_PV` (`K=12`) | `295.45 ms` | `6.35×` | `6,504.68 ms` | `11.27×` |

Look closely at the **super-linear jump** on `1_CL_map.txt` (`N=16,384, M=304`):

- Working-set tensor size (`J` alone is `(16384, 304, K) * 8 bytes`): `K=3` is `~114 MiB`, `K=9` is `~342 MiB`, `K=18` is **`~684 MiB`** — far exceeding L3 cache on an 8-core desktop CPU.
- At `6_Lor (K=18)`, `evaluator.jacobian` non-math array copy/slicing overhead (`JacCopy = 2,685.81 ms`) exceeded the peak derivative math itself (`JacMath = 2,114.60 ms`).
- Meanwhile on `1_CL_map_3lorentzian.json` (`cProfile = 94.581 s`), `evaluator.jacobian` self-time (`26.801 s`) exceeded `batched_lorentzian_jac` (`20.640 s`) and `batched_levenberg_marquardt` self-time (`22.036 s`).

### 2.4 Summary of Verified Bottlenecks

| ID | File & Location | Verified Bottleneck Mechanism | Empirical Evidence |
| :--- | :--- | :--- | :--- |
| **B1** | `optimizer.py:339` | Pre-clip `dp` checked for `dp_rel` convergence instead of post-clip step `p_trial - p`; bound-pinned rows stay active until `max_ite` | `1,494` rows pinned at `alpha=0.0` on `CL/1_PV`; `44 -> 3,486/3,721` conv on `3_3721map_3Lorent` once fixed (`4.14×` speedup) |
| **B2** | `evaluator.py:136–234` | `np.zeros((N, M, K))` + `_to_full()` + per-peak `jac_fn()` alloc + `J_full[:, :, slc] = ...` + `J_full[:, :, _free_idx]` copy | `JacCopy = 2,685.81 ms > JacMath = 2,114.60 ms` (`16,384×304`, `6_Lor`); `evaluator.jacobian` #1 in `cProfile` (`26.801 s` self) |
| **B3** | `models.py:117–148` | `batched_pseudovoigt_jac` calls `batched_gaussian`, `batched_lorentzian`, `batched_gaussian_jac`, `batched_lorentzian_jac`, recomputing `dx`, `dx²`, `inv_w²`, `t²` four times + extra `(N, M, 3)` buffers | `1_PV` costs `3.18×–4.23×` of `1_Lor` per iter; fused kernel tested `1.71×–1.85×` faster (`max_abs_diff = 3.47e-12`) |
| **B4** | `optimizer.py:49–69` | `_batched_solve` switches to Python `for i in range(Na): cho_factor/cho_solve` loop whenever `Na <= 500 and K >= 10` | `Na=450, K=12`: `cho_loop = 89.883 ms` vs `np.linalg.solve = 2.661 ms` (**`33.8×` slower**); `Na=50, K=12`: **`21.0×` slower** |
| **B5** | `baseline.py:74,185` | Cold `from scipy.interpolate import interp1d` for 2-point Linear baseline | Cold call costs **`1,508.64 ms`** vs **`0.75 ms`** warm |
| **B6** | `evaluator.py:416–417` | `apply_noise_threshold` materializes `(N, M)` `dists = np.abs(x[None, :] - x0_vals[:, None])` just to pick `argmin` | Allocates transient `(N, M)` `float64` matrix (`16,384×304 ≈ 40 MB`) needlessly |
| **B7** | `optimizer.py` (whole-N batching) | Operating on full `(N, M, K)` tensors (`~100–700 MiB`) overflows CPU L3 cache and keeps `JTJ_cache` (`active_idx` re-sliced each iter) non-compacted | Chunking (`chunk=512`) alone yields **`1.67×`** speedup (`8.254 s -> 4.942 s`) on `WDF/3_3721map_3Lorent` (`max|dR2|=2.22e-16`) |
| **B8** | `models.py` / `evaluator.py` | Full-axis evaluation across all `M` channels even for narrow peaks (`fwhm ≪ span`) | On `N=1681, M=1024, 6p_lor`: dense `float64 = 966.43 ms/iter` vs `+-4*fwhm` support-windowed block-diag **`82.72 ms/iter` (`11.68×`)** |
| **B9** | BLAS / Multiprocessing | Windows `ProcessPoolExecutor` spawn cost hurts <10 s fits (`0.65×–0.80×` cold vs `1.83×–1.95×` warm); default multi-thread OpenBLAS (8 threads) thrashes small/medium batched ops | `OMP/OPENBLAS=2` beats `8` threads (`9.49 s vs 14.65 s` on `2dmap_spc_bl 3p`) |

---

## 3. Priority 1 Action Plan (Immediate Target — High Impact, Low Complexity)

These five pure-NumPy/Python changes preserve existing public contracts, exact numerical precision (`float64`), and startup import deferral:

- **`P1.1` (`optimizer.py`) — Post-Clip Projected-Step Convergence + Uniform-Weight Fast Path**:

  - Compute `dp_actual = p_trial_active[improved_active] - p_active[improved_active]` (after `np.clip` box-bound projection) inside `dp_rel` rather than raw unconstrained `dp`. Note that `p_active = p[active_idx]` via integer array indexing is already a separate copy before `p[improved_idx]` is updated.
  - Detect once up front when `weights` is `None` or `(weights == 1.0).all()` to bypass `weights * residuals` and `J_r *= weights[:, :, None]` array passes.

- **`P1.2` (`optimizer.py`) — Unify `_batched_solve` on Vectorized `np.linalg.solve`**:

  - Always try `np.linalg.solve(A, b[..., None])[..., 0]` across all `(Na, K, K)` systems regardless of `Na` or `K`, eliminating the `5.7×–33.8×` per-matrix Cholesky loop cliff at `Na <= 500, K >= 10`. Retain per-matrix `cho_factor`/`np.linalg.solve`/`0.0` strictly inside the `except np.linalg.LinAlgError` fallback for singular edge cases.

- **`P1.3` (`models.py`) — Fuse `batched_pseudovoigt` & `batched_pseudovoigt_jac` + Optional `out=` Buffer**:

  - Share `dx`, `inv_w`, `inv_w2`, `dx2`, `t2 = dx2 * inv_w2`, `g_unit = exp(-4*ln2*t2)`, and `invD = 1 / (4*t2 + 1)` inside single-pass fused `batched_pseudovoigt` and `batched_pseudovoigt_jac`.
  - Add optional `out=None` slice support to batched peak/Jacobian evaluators so `VBFevaluator` writes directly into preallocated buffers without per-peak allocations. Measured speedup: **`1.71×–1.85×`**, numerical deviation `max_abs_diff = 3.47e-12`.

- **`P1.4` (`evaluator.py`) — Zero-Copy `VBFevaluator.jacobian()` & `evaluate()` Free-Param Fast Path**:

  - Precompute `self._all_free_identity` when no parameters are fixed (`_fixed_idx` is empty) and no parameter expressions (`_exprs`) exist.
  - Skip `_to_full()`, skip `np.zeros()` zero-initialization (every column `0..K-1` is partitioned cleanly by peak slices), pass `out=J[:, :, slc]` directly to peak evaluators, and return `J` without `J_full[:, :, _free_idx]` fancy-index copying. Support an optional caller-supplied `out=` buffer in `evaluate()` and `jacobian()` when `not self._exprs`.

- **`P1.5` (`evaluator.py`, `baseline.py`) — O(N) Noise-Threshold Lookup & Pure-NumPy Linear Baseline**:

  - Replace `dists = np.abs(x[None, :] - x0_vals[:, None])` in `VBFevaluator.apply_noise_threshold()` with uniform fast-path (when `x0` is identical across `N`) or `np.searchsorted` + neighbor compare on monotonic 1D `x`, cutting `(N, M)` memory down to `O(N)`.
  - Implement closed-form vectorized linear interpolation/extrapolation (`_interp_linear_extrap`) in pure NumPy in `baseline.py` (`eval_baseline()` and `eval_baseline_batch()`), eliminating the **`1,508.64 ms`** cold `scipy.interpolate` import overhead.

### Verified Phase 1 Before/After Benchmarks

Measured cleanly after `P1.1`–`P1.5` implementation (`213 passed` in `tests/unit/fit_engine` + `test_startup_imports.py`; `1,937 passed, 22 deselected` in `tests/ -m "not slow"`):

- **Cold 2-point Linear Baseline Subprocess Check (`N=3,721, M=304`)**: `1.766 ms` wall time with `scipy` / `scipy.interpolate` **unloaded** (`scipy_loaded=False`, vs `1,508.64 ms` baseline cold SciPy import).

**A. Real-Dataset End-to-End Fits**

| Dataset & Config | Baseline Time | Phase 1 Time | Speedup | Converged (Before → P1) | Median $R^2$ (P1) | Peak Heap (Before → P1) |
| :--- | ---: | ---: | ---: | :--- | ---: | :--- |
| WDF (`3721×102`): `1_CL_map` | `0.2968 s` | `0.3085 s` (`min 0.2696 s`) | `0.96×` | `97.9% -> 99.1%` | `0.0000` | `16.34 -> 19.05 MiB` |
| WDF (`3721×102`): `1_CL_map_1pseudovoigt` | `8.3554 s` | `2.7010 s` | **`3.09×`** | `20.9% -> 87.8%` | `-0.0663` | `62.28 -> 41.87 MiB` |
| WDF (`3721×102`): `1_CL_map_2lorentzian` | `6.1183 s` | `1.5141 s` | **`4.04×`** | `4.1% -> 93.7%` | `-0.0331` | `61.23 -> 37.45 MiB` |
| WDF (`3721×102`): `1_CL_map_3lorentzian` | `8.0891 s` | `2.9501 s` | **`2.74×`** | `2.2% -> 90.3%` | `-0.0259` | `81.36 -> 51.41 MiB` |
| WDF (`3721×71`): `3_3721map` | `0.1612 s` | `0.1576 s` | `1.02×` | `100.0% -> 100.0%` | `0.9988` | `45.67 -> 23.59 MiB` |
| WDF (`3721×77`): `3_3721map_1pseudo` | `0.3149 s` | `0.1880 s` | **`1.68×`** | `100.0% -> 100.0%` | `0.9993` | `64.45 -> 36.12 MiB` |
| WDF (`3721×77`): `3_3721map_3Lorent` | `7.4189 s` | `1.6203 s` | **`4.58×`** | `1.2% -> 93.7%` | `0.9996` | `82.62 -> 45.72 MiB` |
| CL (`16384×304`): `1_CL_map` | `3.9432 s` | `3.8021 s` | `1.04×` | `100.0% -> 100.0%` | `0.9841` | — |
| CL (`16384×304`): `1_CL_map_1pseudovoigt` | `13.5854 s` | `5.0866 s` | **`2.67×`** | `96.3% -> 100.0%` | `0.9859` | — |
| CL (`16384×304`): `1_CL_map_2lorentzian` | `35.9672 s` | `23.5715 s` | **`1.53×`** | `88.0% -> 92.5%` | `0.9889` | — |
| CL (`16384×304`): `1_CL_map_3lorentzian` | `75.7279 s` | `44.3647 s` | **`1.71×`** | `77.4% -> 86.1%` | `0.9906` | — |

**B. Controlled Per-Iteration Scaling (`ms/iter` over 10 Fixed Iterations)**

| Model (`K`) | WDF Base | WDF Phase 1 | WDF Speedup | CL Base | CL Phase 1 | CL Speedup |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1_Lor` (`3`) | `46.54 ms` | `18.00 ms` | **`2.59×`** | `576.95 ms` | `352.32 ms` | **`1.64×`** |
| `1_Gauss` (`3`) | `73.67 ms` | `28.93 ms` | **`2.55×`** | `1,031.39 ms` | `485.45 ms` | **`2.12×`** |
| `1_PV` (`4`) | `148.13 ms` | `33.98 ms` | **`4.36×`** | `2,438.98 ms` | `571.52 ms` | **`4.27×`** |
| `2_Lor` (`6`) | `92.88 ms` | `50.99 ms` | **`1.82×`** | `1,410.73 ms` | `594.71 ms` | **`2.37×`** |
| `3_Lor` (`9`) | `139.48 ms` | `76.14 ms` | **`1.83×`** | `2,617.44 ms` | `970.52 ms` | **`2.70×`** |
| `4_Lor` (`12`) | `217.41 ms` | `105.49 ms` | **`2.06×`** | `2,706.76 ms` | `1,411.56 ms` | **`1.92×`** |
| `6_Lor` (`18`) | `281.59 ms` | `176.23 ms` | **`1.60×`** | `6,803.68 ms` | `2,745.22 ms` | **`2.48×`** |
| `3_PV` (`12`) | `295.45 ms` | `164.95 ms` | **`1.79×`** | `6,504.68 ms` | `2,517.64 ms` | **`2.58×`** |

Notice that on `1_CL_map.txt / 1_CL_map_3lorentzian`, `14.1k/16.4k` (`86.1%`) converge after Phase 1 while the remaining `13.9%` tail still drives `56.3%` mean active rows across `(16384, 304, 9)` tensors — confirming **Phase 2 (`P2.1` L3 chunking + compaction and `P2.2` dead-peak/stagnation early exit)** as the highest-leverage next step.

---

## 4. Phase 2 Implementation Plan (High Impact, Moderate Complexity)

Phase 2 targets memory hierarchy efficiency, stalled-peak termination, localized spatial sparsity, and multi-core CPU scaling — all purely in Python/NumPy without native compiler toolchains.

### `P2.1` — L3-Cache-Sized Batch Chunking & Active-Row Compaction

#### Problem & Evidence

- Currently, `batched_levenberg_marquardt()` keeps global `(N, M, K)` Jacobian tensors (`~114–684 MiB` on `N=16,384, M=304`) and `(N, K, K)` caches for all `N` spectra simultaneously, even after 75–90% have converged or while active rows span fragmented memory indices.
- Every iteration on non-contiguous `active_idx` pays cache misses across the full `N`-row allocation.
- Empirically verified on `3_3721map.wdf / 3_3721map_3Lorent.json`: splitting `N=3,721` into `chunk=512` reduced runtime from **`8.254 s -> 4.942 s` (`1.67×`)** with **`max |ΔR²| = 2.22e-16`**.

#### Design & Touchpoints (`optimizer.py`, `vbf_engine.py`)

1. **Deterministic Adaptive Chunk Sizing (`_choose_chunk_size(N, M, K)`)**:

   - Target a working set `(chunk_size, M, K) * 8 bytes <= ~8–16 MiB` (fitting comfortably inside L3 cache on typical 8–32 MiB CPU caches while staying big enough for BLAS `matmul` batch amortization).
   - Clamp `chunk_size` in `[256, 1024]` (e.g., `chunk_size = max(256, min(1024, (12 * 1024 * 1024) // max(M * max(K, 1) * 8, 1)))`). If `N <= chunk_size`, run zero-split direct path to avoid overhead on small calls.

2. **Chunk Execution & Contiguous Active-Row Compaction**:

   - Slice `(x_chunk, y_chunk, p0_chunk, w_chunk)` into contiguous chunks `[s:e]`, call `_batched_lm_core(...)`, and write back into preallocated `(N, K)` `p_out`, `(N,)` `converged`, and `(N,)` `n_iter` arrays.
   - Inside each chunk `Nc <= 1024`, compute noise/zero-weight masks locally. Once active count drops cleanly (or when active density `< 0.5`), optionally pack `(p_c, residuals_c, JTJ_c, JTr_c, lam_c)` into dense contiguous sub-arrays, avoiding scattered `active_idx` / `dirty[active_idx]` gather/scatter penalties.
   - Note: Each spectrum in LM is mathematically 100% independent, so chunking preserves bit-exact parity (`max |ΔR²| ~ 1e-16`).

3. **Progress Reporting Integration (`progress_callback`)**:

   - Maintain weighted completed-spectrum counts (`completed_in_finished_chunks + converged_in_current_chunk`) so UI progress bars inside `VBFthread` remain smooth and monotonic (`0 -> 100%`) instead of resetting per chunk.

#### Risks & Validation

- *Shared `x` vs 2D `(N, M)` `x`*: Always slice `x[s:e]` when `x.ndim == 2`, pass `x` unmodified when `x.ndim == 1`.
- *Test*: Assert chunked (`chunk=256, 512`) vs unchunked output parity (`np.allclose` on `p`, `R²`, `converged`, `n_iter`) across 1D and 2D `x`, `weights`, and `_exprs`.

---

### `P2.2` — Inactive / Bound-Pinned Peak Convergence & Stagnation Early Exit

#### Problem & Evidence

- Even after `P1.1` fixes bound clipping for pixels pinned right at iteration `t`, multi-peak configurations with an absent/vanishing component (like Peak 1 `x0=497.15, a=0.0` in `3_3721map_3Lorent` or 2nd/3rd peak on background CL pixels) still suffer from ill-conditioned `(J^T J)` blocks: when `ampli ≈ 0`, derivatives $\partial/\partial x_0 = 0$ and $\partial/\partial w = 0$, leaving zero columns in `J` (`rank < K`) so LM spends dozens of iterations shrinking/growing damping $\lambda$ before `xtol`/`ftol` triggers.
- On `1_CL_map.txt / 1_CL_map_3lorentzian`, ~22.6% of spectra still hit `max_ite=100` (`61.4 s` even after `P1.1 + P1.4`).

#### Design & Touchpoints (`optimizer.py`, `fit_engine/`)

1. **Diagonal Tikhonov Floor / Zero-Column Safeguarding**:

   - In `(JTJ + lam * diag(JTJ))`, a dead peak with `ampli == 0` yields `diag(JTJ)[idx] == 0`. Current `lam * np.maximum(diag_JTJ, 1e-6)` helps invertibility, but when a peak amplitude collapses to `0` (`ampli <= lo_ampli + eps` and gradient w.r.t. `ampli` points negative/outward), zeroing the trial steps of its coupled `(x0, fwhm, alpha)` parameters prevents spurious width/position drift and triggers immediate step acceptance or convergence (`dp_actual == 0`).

2. **Stagnation / Consecutive Rejected-Step Early Exit**:

   - Currently, a spectrum stops early only when `lam >= 1e7` (`optimizer.py:310`) or when `rel_cost_change < ftol & dp_rel < effective_xtol`.
   - Under floating-point precision plateaus near `1e-7..1e-12`, tiny `1e-8` cost improvements can keep a row spinning at `lam = 1e2..1e6` for 80+ iterations with zero physically meaningful movement.
   - Add two safe early exits (configurable / preserving strict precision tests):

     - **Plateau Exit**: Track consecutive tiny-cost-and-tiny-step accepted iterations (e.g., `rel_cost_change < ftol * 0.1` for 3 consecutive accepted steps -> converge).
     - **Dead-Step Rejection Cap**: Keep `(worsened_active & lam_trial >= 1e7)` while additionally terminating rows whose unconstrained `np.linalg.norm(dp)` would move `< 1e-12` or which spend `> 12` consecutive rejections without improvement.

3. **Optional Exact Backtracking / Nielsen LM Damping Update**:

   - Replace heuristic factor `10.0 / 3.0` (`optimizer.py`) with **Nielsen's gain-ratio (`rho`) update** (`mu *= max(1/3, 1 - (2*rho - 1)**3)`, `nu = 2` vs `mu *= nu, nu *= 2`), which reduces rejected step ratios on coupled multi-peak landscapes by avoiding premature damping spikes and overshoot restarts.

#### Risks & Validation

- *Tight-tolerance unit tests*: Any test expecting `max_ite` or exact tight `xtol=1e-10` on synthetic noise-free problems must still pass — gate Nielsen / stagnation exits cleanly so `xtol`/`ftol` contracts are respected when tight tolerances are explicitly requested.

---

### `P2.3` — Safe Support-Windowed Peak Evaluation (`|x - x0| <= c * fwhm`)

#### Problem & Evidence

- Raman and sharp luminescence spectra often span wide wave-number ranges (`M = 500–2,000`) where individual peaks have `fwhm = 4–12 cm⁻¹`. Away from `|x - x0| > 6 * fwhm`, Gaussian weights decay below `2^-144` (exact zero in `float64`), whereas Lorentzian tails decay slowly ($1 / (1 + 4 (dx/w)^2)$: `1/65 ≈ 1.54%` at `4*w` and `1/10,001 ≈ 10⁻⁴` only at `50*w`).
- Benchmark from `vbf_bench_part2.txt` (6 Lorentzians, `N=1681, M=1024, fwhm=6`): dense `float64 = 966.43 ms/iter` vs `+-4*w` block-diagonal windowed `82.72 ms/iter` (**`11.68×`**) when localized!

#### Critical Mathematical Guardrail (Why Support Windowing Requires Care)

- **Lorentzian / Asymmetric Lorentzian / PseudoVoigt have heavy algebraic tails ($O((w/dx)^2)$)**:

  - Cutting Lorentzian strictly at `4 * fwhm` leaves a `1.54%` truncation discontinuity if used directly on `evaluate()` / `residuals`. Even at `15 * fwhm`, `1 / (1 + 4*225) = 0.0011` (`0.11%`).
  - Therefore, a naive hard cutoff on `models.py` for Lorentzian/PseudoVoigt without tail correction would distort broadband fits (`R²` drift on wide peaks).
- **Safe Design Strategy**:

  1. **Exponential Models (`Gaussian`, `GaussianAsym`)**: Exact machine-epsilon cutoff exists at `|dx / w| > 4.5` (`exp(-4 * ln(2) * 4.5^2) = 3.1e-25 < eps_float64`). A slice/index window around `[min(x0) - 5*max(w), max(x0) + 5*max(w)]` for each chunk/peak group is **mathematically exact to machine precision (`< 1e-16`)**.
  2. **All-Peak ROI Tight-Mask Prefilter (`vbf_engine.py` / `VBFevaluator`)**: Within any chunk, compute `[lo_chunk, hi_chunk]` across active centers ± tail tolerance; when peaks all occupy a narrow active sub-band inside an uncropped spectrum (or when `DecaySingleExp`/`DecayBiExp` only operate post-`t0`), restrict evaluation slice `[i0:i1]` for Jacobian & residual assembly.
  3. **Opt-In Fast Preview vs Exact Final Polish**: Optionally use narrower fast support (`8 * fwhm`) during coarse early iterations if exposed as a precision mode, followed by full-band exact LM polish (`1–2` iterations) to eliminate tail bias completely (`ΔR² == 0.0`).

---

### `P2.4` — BLAS Thread Control & Persistent Warm Worker Pool

#### Problem & Evidence

- **BLAS Oversubscription (`vbf_bench_part2.txt`, Section G)**: Running `2dmap_spc_bl 3p(25it)` across OpenBLAS/OMP thread counts showed:

  - `1 thread`: `10,874.9 ms`
  - `2 threads`: **`9,493.7 ms`** (optimal)
  - `4 threads`: `10,306.8 ms`
  - `8 threads`: **`14,646.2 ms`** (`54%` slower due to thread barrier spin on small `(chunk, M, K)` GEMMs!)
- **Multiprocessing (`audit_wdf_part3.txt`)**: Spawning cold `ProcessPoolExecutor` on Windows took **`10.39 s` (`w=4`) and `12.79 s` (`w=8`)** vs **`8.33 s`** single-process (`0.80×` / `0.65×` *regression*) on `3_3721map_3Lorent` due to Python interpreter startup & DLL load (~2–4 s per worker!). Conversely, a **pre-warmed pool (`w=4`)** achieved **`1.95×` (`4.27 s`)** on `3_3721map` and **`1.83×` (`7.56 s`)** on `1_CL_map_1pseudovoigt`.

#### Design & Touchpoints (`fit_engine/`, `application/`, `requirements` optional `threadpoolctl`)

1. **Scoped Single/2-Thread BLAS Guard (via `threadpoolctl` if installed, no hard dependency)**:

   - Inside `batched_levenberg_marquardt()` (when chunked or inside worker processes), wrap execution with `threadpoolctl.threadpool_limits(limits=2)` (or `limits=1` inside multi-process workers) using `try ... except ImportError:` so it fails gracefully when `threadpoolctl` is absent. Never call at startup (`test_startup_imports.py`).

2. **Lazy Persistent Worker Pool (`VBFProcessPool`, Optional Thresholded `N * M * K`)**:

   - Spawn workers lazily on first large map fit (or warm asynchronously after UI load) rather than per-fit cold teardown, keeping `w = min(4, max(1, cpu_count // 2))` workers capped to avoid RAM duplication.
   - Only dispatch across IPC when total problem cost (`N * M * K`) exceeds a calibrated threshold (e.g., `estimated_flops > threshold`, skipping IPC entirely for fast `< 1 s` fits where chunked single-process finishes before IPC overhead completes).
   - Keep every worker single-threaded (`limits=1`) in BLAS/OMP so `4 workers × 1 thread` never oversubscribes physical cores.
   - Guard carefully for PyInstaller frozen executables on Windows/macOS (`multiprocessing.freeze_support()`, picklable pure-NumPy inputs/functions without Qt objects — note `VBFevaluator` expressions `_exprs` must either serialize cleanly as raw config dicts rebuilt per worker or stay in-process if custom lambdas exist).

---

## 5. Phase 3 Implementation Plan (Advanced Architectural & Algorithmic Improvements)

Phase 3 narrows the remaining architectural gap between Python/NumPy tensor batching and compiled L1-resident single-spectrum C++ loops (such as HORIBA LabSpec 6's `OlMainer.dll`).

### `P3.1` — Optional Compiled Native C++/OpenMP L1-Resident Kernel (Cross-Platform)

#### Architectural Rationale

Why can a compiled C++ loop (`OpenMP parallel for` per spectrum) beat whole-map Python tensor ops even when NumPy calls BLAS/C?

- **Cache Residency**: For one spectrum of size `M = 100..500, K = 3..18`, the working set (`x`, `y`, `r`, `J`, `JTJ`, `JTr`) is `~10–80 KiB` — fitting **100% inside per-core L1/L2 CPU cache**, never hitting DRAM inside the LM loop.
- **Loop Fusion**: Evaluating `f(x)`, `r`, `J`, AND accumulating symmetric upper triangle of `JᵀJ` (`0.5 * K * (K+1)` entries) and `Jᵀr` inside a **single streaming pass over `i = 0..M-1`** eliminates `(N, M, K)` memory materialization completely (from `~684 MiB` peak DRAM down to `< 1 MiB`!) and fuses transcendental `exp()` SIMD instructions with normal-equation GEMM.
- **Zero Per-Iteration Barrier**: Each CPU thread iterates its own subset of spectra cleanly to convergence with **zero inter-iteration synchronization** and **zero active-mask compaction**.

#### Implementation Blueprint

1. **Backend Design**: Write a clean, portable **C++17 + OpenMP** (using **PyBind11** or **Cython**) module `_vbf_native` covering:

   - Built-in peak kernels (`Gaussian`, `Lorentzian`, `PseudoVoigt`, `GaussianAsym`, `LorentzianAsym`, `Fano`, `DecaySingleExp`, `DecayBiExp`) — no arbitrary Python string callbacks inside the hot C++ loop.
   - Direct Cholesky (`LL^T`) / LDL^T symmetric positive-definite linear solver on stack-allocated `<= 64×64` buffers (matching `VBF`'s `setStackSize` lesson — fixed small stack buffers or per-thread heap scratch reused across all spectra).
   - Identical parameterization, bounds clipping, `linear`/`soft_l1`/`huber` robust weights, and `ftol`/`xtol`/`max_ite` convergence semantics matching `optimizer.py`.

2. **Hybrid Dispatch & Automatic Fallback**:

   - In `VBFengine.fit_spectra()`: check `not evaluator._exprs` and all peak models supported by native kernel + `_vbf_native` importable (`try: import spectroview.fit_engine._vbf_native`).
   - Otherwise seamlessly fall back to Phase 1/2 NumPy/Python `optimizer.py` (which continues supporting arbitrary user `_exprs`, dynamic/custom scalar models, and pure-Python environments without a C compiler).

3. **Packaging & Cross-Platform Distribution**:

   - Distribute prebuilt Python wheels (`cibuildwheel` in CI) for Windows `x64`, macOS `arm64`/`x64` (`libomp`), and Linux `x64` so end-users never require build tools locally; keep `SPECTROview` 100% runnable in pure Python if the compiled extension isn't built in dev mode.

---

### `P3.2` — Variable Projection (VarPro / Separable Nonlinear Least Squares) & Optional Spatial Warm-Start

#### Mathematical & Algorithmic Concept

Every spectroscopic multi-peak sum is **linear in peak amplitudes** $a_1, \dots, a_P$ given nonlinear shape parameters $\theta = \{(x_{0,p}, w_p, \alpha_p)\}_{p=1}^P$:

$$\mathbf{y} \approx \sum_{p=1}^{P} a_p \phi_p(\mathbf{x}; \theta_p) = \Phi(\theta) \mathbf{a}$$

1. **Variable Projection (VarPro)**:

   - Instead of optimizing $K = 3P$ (or $4P$) parameters jointly (where strongly correlated overlapping peak amplitudes oscillate with widths/centers and produce poorly conditioned `JᵀJ`), reduce LM unknowns to nonlinear shape parameters only ($2P$ or $3P$, e.g. `6` instead of `9` for 3 Lorentzians, `9` instead of `12` for 3 Pseudo-Voigts — reducing $K^3$ solve cost by ~3× and conditioning dramatically).
   - For each spectrum trial $\theta$, solve the small non-negative / unconstrained $(P \times P)$ system $\Phi^T \Phi \mathbf{a} = \Phi^T \mathbf{y}$ (or Kaufman's approximated VarPro Jacobian for $\theta$).
   - Requires careful handling of fixed/expressed amplitudes and upper/lower amplitude bounds (`nnls` or bounded active-set when `lo/hi` constrain amplitudes); therefore suited as a well-tested specialized solver path for standard unbounded/non-negative sum-of-peaks models.

2. **Optional Spatial Warm-Start Propagation (Grid-Aware Coarse Sweep -> Batched Refine)**:

   - On smoothly varying Raman/wafer maps (`x_coord`, `y_coord`), spectra close in space often share very similar peak centers/widths while `p0` builds a single global scalar guess across all `N` rows.
   - Instead of sequential spiral fitting (which destroys parallelism), use a **2-stage parallel refinement**:

     1. Fit a coarse **downsampled grid** (e.g. every $4 \times 4$ tile or $k$-means spectral cluster centroids, `~5%` of spectra) cleanly to convergence.
     2. Broadcast/bilinearly interpolate converged parameters to initialize the remaining **95%** of pixels, which then run in **1 fully vectorized batched LM pass** (typically converging in `2–4` iterations instead of `30–100`).
   - Must stay strictly **opt-in** or map-only (`MapData` with uniform spatial grid / spectral clustering fallback) with careful handling of domain boundaries / discontinuities (e.g., defects or empty substrate regions via residual-gated acceptance: keep spatial seed only if `cost(spatial_p0) < cost(default_p0)`!).

---

### `P3.3` — Vectorized Forward-Mode Dual-Number AD for Linked Expressions (`_exprs`)

#### Problem & Evidence

- Currently (`2dmap_spc.spc | 3p_expr`, `vbf_bench_report.txt`), linking 2 parameters via `_exprs` (`K_free = 7 < K_total = 9`) caused `jac` time to balloon from **`42.37 s (53.4%)` to `75.73 s (68.9%)`** despite solving *smaller* `7×7` normal equations!
- Why? `VBFevaluator._build_chain_rule_expr_funcs()` numerically finite-differences every expression function per free parameter on each Jacobian evaluation (`evaluator.py`).

#### Design (`evaluator.py`)

- Implement a lightweight vectorized **`DualTensor(val, deriv)`** class over NumPy arrays supporting arithmetic (`+`, `-`, `*`, `/`, `**`) and standard math ufuncs (`sin`, `cos`, `exp`, `log`, `sqrt`, `abs`, `minimum`, `maximum`) restricted to safe AST-evaluated math namespaces.
- Evaluate each compiled parameter expression once on `(p_free_i, eye_k)` `DualTensor`s to obtain **exact machine-precision analytical chain-rule derivatives $\partial p_{\text{dep}} / \partial p_{\text{free}}$** across all `N` spectra in one vectorized pass — completely eliminating nested finite-difference sweeps and improving Jacobian accuracy for tightly constrained crystallographic/phonon doublet ratios.

---

## 6. Verification Gates for Future Phase 2 & Phase 3 Work

Every optimization milestone in Phase 2 and Phase 3 must pass the following gates before merge:

1. **Numerical Equivalence**: On noiseless and real-data reference maps (`3_3721map.wdf`, `1_CL_map.txt`, `2dmap_spc.spc`, `mapls6.l6m`), verify converged parameter agreement, bounded parameter constraints, and median/p10 $R^2$ fidelity ($\Delta R^2 \le 10^{-6}$ for exact rewrites; documented trade-offs for approximate/surrogate modes).
2. **Startup Import Deferral**: Pass `tests/performance/test_startup_imports.py` (`scipy.*`, optional `threadpoolctl`, or heavy modules never imported at top-level startup).
3. **Full Test Suite & Static Check**: Pass `python -m pytest tests/ -q` and `python -m pyflakes` on all modified modules.
4. **Reproducible Before/After Benchmark Table**: Measure and record wall time, `ms/spectrum`, peak `tracemalloc` heap, iteration count, and convergence fraction on both `3_3721map.wdf` and `1_CL_map.txt`.

