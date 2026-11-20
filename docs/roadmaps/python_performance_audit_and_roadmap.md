# SPECTROview — Strategy A (Pure-Python, NumPy, SciPy & PySide6) Performance Audit & Optimization Roadmap

**Document Status**: Re-Audited Under Controlled Idle-CPU Conditions, Empirically Benchmarked & Implementation-Ready Roadmap  
**Target Architecture**: Strategy A — Pure-Python Optimization (`NumPy`, `SciPy`, `pandas`, `PySide6`, `Matplotlib`)  
**Decision Context**: Following a full-repository technical audit and controlled re-profiling of `SPECTROview` (`v26.48.2`, 225 Python modules across `model/`, `viewmodel/`, `view/`, `fit_engine/`, `api/`, `application/`, and `ai_agent/`), **Rust migration is not necessary**. Over **85–90% of the achievable end-to-end performance and memory improvements** can be captured purely in Python by fixing data-structure choices, memory layout (`F_CONTIGUOUS` vs `C_CONTIGUOUS`), OpenBLAS thread oversubscription, memory duplication, algorithmic complexity, and Qt widget virtualization—while preserving zero-compiler `py3-none-any.whl` packaging and cross-platform CI/CD simplicity.

---

## 1. Controlled Benchmarking Methodology & Why the Re-Audit Changed the Baseline

### 1.1 Controlled Test Environment

All baseline and prototype measurements in this document were re-executed from scratch under strictly controlled conditions:

- **Hardware & OS**: Intel Core i5-8365U (4 physical cores / 8 logical processors, 1.60–1.90 GHz), Windows 64-bit.
- **Runtime Stack**: Python `3.14.10`, NumPy, SciPy, pandas, PySide6 `6.10.2` (`QT_QPA_PLATFORM=offscreen`).
- **CPU Contention Guard**: System-wide CPU utilization was sampled via Win32 `GetSystemTimes` (kernel + user vs. idle ticks) before every benchmark block, waiting until background system CPU load settled between **`1.6%` and `15.5%`** (compared to **`~83%` system CPU utilization** during the initial exploratory profiling session).
- **Execution Protocol**:
  - Zero concurrent background tasks or parallel test processes were active during timing runs.
  - All modules (`numpy`, `scipy`, `pandas`, `pybaselines`, `PySide6`, `spectroview.*`) were pre-imported before timed sections so cold-import latency never contaminated warm execution metrics.
  - Every operation was executed with **1–2 warmup passes** followed by **3–10 timed repetitions**, recording **`min`**, **`median`**, and **`mean ± std`**.

### 1.2 Measurement Taxonomy Used in This Document

To prevent conflating different classes of numbers, every metric in this roadmap is explicitly labeled as one of four categories:

1. **`[Measured — Current Baseline]`**: Direct multi-run `time.perf_counter()` / byte-footprint measurement of the unmodified `SPECTROview` `v26.48.2` codebase under idle CPU conditions (`1.6%–15.5%` background load).
2. **`[Measured — Isolated Prototype]`**: Direct multi-run measurement of a standalone in-memory prototype function implementing the proposed optimization on the exact same dataset and hardware during this audit (with numerical equivalence verified against the baseline).
3. **`[Estimated — Integrated Workflow]`**: Projected end-to-end timing or memory footprint once the isolated prototype is wired into the full `View ↔ ViewModel ↔ Model` pipeline (accounting for remaining unoptimized stages via Amdahl's Law).
4. **`[Theoretical — Asymptotic Scaling]`**: Mathematical complexity ($O(\cdot)$ FLOPs or byte count) as dataset dimensions ($N$ spectra, $M$ channels, $K$ parameters) scale.

### 1.3 High-CPU Session vs. Controlled Idle-CPU Baseline Comparison

Re-profiling under idle CPU (`1.6%–15.5%`) revealed that the previous high-CPU session inflated pure compute and I/O timings by **`1.3×` to `7.0×`**, and uncovered four critical architectural insights that were previously masked by CPU contention:

| Operation / Workload | Previous High-CPU Measurement | Controlled Idle-CPU Baseline (`min` / **`median`** / `mean ± std`) | Inflation Factor | Key Architectural Finding From Re-Audit |
| :--- | ---: | :--- | :---: | :--- |
| `m_io.load_wdf_map` (`3_3721map.wdf`, `3,721 × 575`, `8.3 MB`) | `34.1 ms` | `13.88 ms` / **`14.50 ms`** / `14.61 ± 0.55 ms` | `2.35×` | Already near disk/memcpy speed (`> 250k` spectra/s). |
| `m_io.load_spc_map` (`2dmap_spc.spc`, `1,681 × 1,024`, `6.8 MB`) | `28.5 ms` | `13.54 ms` / **`13.67 ms`** / `14.41 ± 1.38 ms` | `2.08×` | Already near disk/memcpy speed. |
| `m_io.load_l6m_map` (`mapls6.l6m`, `1,368 × 1,024`, `2.2 MB`) | `59.4 ms` | `32.18 ms` / **`32.42 ms`** / `32.51 ± 0.25 ms` | `1.83×` | Fast binary table parser (`< 33 ms`). |
| `m_io.load_map_file` (`2_MoS2_map.txt`, `1,520 × 1,015`, `7.6 MB`) | `156.9 ms` | `115.58 ms` / **`121.18 ms`** / `122.91 ± 6.63 ms` | `1.30×` | C-engine `pd.read_csv` float parsing. |
| `m_io.load_map_file` (`1_CL_map.txt`, `16,384 × 1,024`, `151.0 MB`) | `1,852.6 ms` | `1,053.06 ms` / **`1,118.58 ms`** / `1,149.78 ± 94.32 ms` | `1.66×` | C-engine `pd.read_csv`; returns Fortran-ordered block. |
| `m_io.extract_map_arrays` (`16,384 × 1,024` DataFrame $\rightarrow$ arrays) | `162.8 ms` | `74.93 ms` / **`85.64 ms`** / `84.95 ± 7.30 ms` | `1.90×` | **Bug discovered**: returns `F_CONTIGUOUS` `Y0` (`strides=(4, 65536)`), hurting downstream row ops by `3.57×`! |
| `NgsReader` (`mapls5.ngc`, `3.9 MB` compressed `lzhuf`) | `23,175.7 ms` | `9,093.97 ms` / **`9,743.61 ms`** / `9,743.61 ± 649.63 ms` | `2.38×` | Still a `~9.1–9.7 s` pure-Python bit-loop bottleneck (`P6`). |
| `VSpectraList.set_spectra_names` (`1,520` items, `QListWidget`) | `313.3 ms` | `126.84 ms` / **`185.35 ms`** / `173.09 ± 24.36 ms` | `1.69×` | Scales super-linearly (`670.55 ms` at `3,721`; **`5,919.42 ms`** at `16,384`). |
| `VDataframeTable.show` (`16,384 × 14` fit table, `229k` cells) | *(Assumed slow)* | `9.77 ms` / **`10.76 ms`** / `11.38 ± 1.64 ms` | — | **Already virtualized** (`QAbstractTableModel` + `QTableView`)! |
| `VGraph.plot` (`2Dmap` / `wafer`, `16,384` points) | `155.1 ms` / `123.7 ms` | `58.41 ms` (**`73.06 ms`**) / `50.82 ms` (**`57.80 ms`**) | `2.12×` | Already responsive (`~58–73 ms`). |
| `eval_baseline_batch` (`Linear` / `Poly-3` attached, `1,520 × 575`, warm) | `46.3 ms` / `47.5 ms` | `29.76 ms` (**`30.61 ms`**) / `29.05 ms` (**`30.79 ms`**) | `1.52×` | Vectorized across all 1,520 spectra (`0.020 ms/spectrum`). |
| `eval_baseline_batch` (`arpls`, `50 × 575`, warm) | `1,659.0 ms` (`33.18 ms/spec`) | `225.08 ms` / **`238.65 ms`** (`4.77 ms/spec`) | **`6.95×`** | `asls` is **`1.13 ms/spec`** (`56.61 ms`); `snip` is **`7.58 ms/spec`** (`378.88 ms`). |
| `MMVA.run_pca` (`1,520 × 575`, `n_components=5`) | `2,535.0 ms` | `1,033.34 ms` / **`2,471.73 ms`** (`OPENBLAS=8`) $\rightarrow$ **`652.80 ms`** (`OPENBLAS=2`) | — | OpenBLAS 8-thread thrashing + full SVD waste (`125.65 ms` with `P4.1` Gram + `OPENBLAS=2`). |
| `MMVA.run_nmf` (`1,520 × 575`, `k=4`, `max_iter=200`) | `11,836.0 ms` | `2,212.12 ms` / **`7,283.42 ms`** (`OPENBLAS=8`) $\rightarrow$ **`1,567.99 ms`** (`OPENBLAS=2`) | — | OpenBLAS 8-thread barrier contention on `(1520, 4)` micro-GEMMs + $(N, M)$ residual (`582.02 ms` with `P4.2` + `OPENBLAS=2`). |
| `compute_fit_weights` (`1_CL_map`, `16,384 × 1,024`, cache miss) | `1,702.0 ms` | **`2,355.58 ms`** (`F_CONTIGUOUS` `float32`) vs **`659.18 ms`** (`C_CONTIGUOUS` `float32`) | — | `F_CONTIGUOUS` `Y0` from `extract_map_arrays` causes `3.57×` slowdown (`483.46 ms` with `P5.1`). |
| `VBFengine.fit_spectra` (`1_CL_map.json`, `16,384 × 302`, `1 Lor`, `K=3`) | — | **`3,433.74 ms`** (`OPENBLAS=8`) $\rightarrow$ **`1,434.02 ms`** (`OPENBLAS=2`, `0.088 ms/spec`) | — | Capping OpenBLAS to 2 threads accelerates 16k-spectrum LM fit by **`2.40×`**. |
| `VBFengine.fit_spectra` (`3_3721map.json`, `3,721 × 70`, `1 Lor`, `K=3`) | — | `61.29 ms` / **`63.67 ms`** (`OPENBLAS=2`, `0.017 ms/spec`) | — | Sub-70 ms for 3,721 spectra after commit `9250ae9`. |

---

## 2. Classification of Verified Bottlenecks & Optimization Summary

| Priority & ID | Subsystem & Location | Category | Verified Root Cause | `[Measured — Current Baseline]` | `[Measured — Isolated Prototype]` | `[Estimated — Integrated Workflow]` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`P1`** | `VSpectraList` (`v_spectra_list.py:170–221`) | **GUI-bound / Memory-bound** | `QListWidget` instantiates & styles $N$ C++ `QListWidgetItem` objects synchronously (`VDataframeTable` is *already* virtualized at `10.76 ms`!). | **`185.35 ms`** (`1,520`) / **`670.55 ms`** (`3,721`) / **`5,919.42 ms`** (`16,384`) | **`0.74 ms`** (`1,520`) / **`1.87 ms`** (`3,721`) / **`8.73 ms`** (`16,384`) (**250×–678×**) | `< 10 ms` list refresh; eliminates 0.67–5.9 s main-thread UI freeze |
| **`P2`** | `extract_map_arrays` (`m_io.py:309`), `SpectraStore` (`spectra_store.py`), `VMWorkspaceMaps` (`vm_workspace_maps.py:72`) | **Memory-bound / Cache-layout** | (1) `extract_map_arrays` leaves `.txt`/`.csv` `Y0` `F_CONTIGUOUS` (`strides=(4, 65536)`), slowing row ops `3.57×`; (2) `~1.01 GB` RAM per `67 MB` map due to duplicate `maps[name]` DataFrame, `float64` casts, and eager `Y_peaks`. | **`~1,006.9 MB`** RAM per `16,384 × 1,024` fitted map; **`2,355.58 ms`** F-contiguous weight pass | **`659.18 ms`** C-contiguous weight pass (**3.57×**); **`~201.4 MB`** RAM layout (**80% reduction**) | **~80% RAM reduction** (`~805 MB` saved per 16k map) + **3.5×** faster row preprocessing |
| **`P3`** | `MMVA.run_pca` & `MMVA.run_nmf` (`m_mva.py:75–201`) + OpenBLAS thread guard | **Algorithmic / Concurrency** | (1) OpenBLAS spawns 8 threads on 4 cores, thrashing on `(1520, 4)` micro-GEMMs; (2) PCA runs full `svd` instead of Gram `eigh(X_c.T @ X_c)`; (3) NMF allocates $(N, M)$ `X - W @ H` instead of $O(K^2)$ trace identity. | PCA: **`2,471.73 ms`** · NMF: **`7,283.42 ms`** (`1,520 × 575`, default 8 threads) | PCA: **`125.65 ms`** (**19.7×**) · NMF: **`582.02 ms`** (**12.5×**) (Gram/Trace + `OPENBLAS=2`) | PCA `< 140 ms`, NMF `< 600 ms` on background `QThread` (`0 ms` UI freeze) |
| **`P4`** | `noise.py`, `weights.py`, `optimizer.py` (VBF Engine & Noise Weights) | **CPU-bound / Memory-bandwidth** | (1) OpenBLAS 8-thread contention inside chunked LM (`2.40×` slower than 2 threads on `1_CL_map.json`); (2) `moving_average_5` upcasts `float32` to `float64` + full `np.sum(y)` fingerprint; (3) `np.linalg.solve` dispatch overhead on $K=3$. | `compute_fit_weights`: **`2,355.58 ms`** (F-order) / **`659.18 ms`** (C-order) · `3×3` solve: **`9.76 ms`** · `1_CL_map` fit: **`3,433.74 ms`** | `compute_fit_weights`: **`483.46 ms`** (**4.87× vs F / 1.36× vs C**) · `3×3` solve: **`1.54 ms`** (**6.34×**) · `1_CL_map` (`OPENBLAS=2`): **`1,434.02 ms`** (**2.40×**) | **1.5×–2.5×** faster end-to-end VBF map fitting |
| **`P5`** | `eval_baseline_batch` (`baseline.py:179–244`) & `VMWorkspaceSpectra.subtract_baseline` | **GUI-bound / CPU-bound** | Sequential loop over $N$ spectra runs synchronously on the main Qt thread (`4.77 ms/spec` `arpls` = `7.25 s` freeze for `1,520` spectra; `7.58 ms/spec` `snip` = `11.5 s` freeze). | `arpls`: **`4.77 ms/spec`** · `asls`: **`1.13 ms/spec`** · `snip`: **`7.58 ms/spec`** (main-thread blocking) | *(Verified: naive `Baseline` hoisting or 8-thread OpenBLAS `ThreadPool` does not speed up `arpls`)* | **`0 ms` UI freeze** via background `QThread` + **2.5×–3.2×** speedup via single-BLAS-thread multi-process/shared-factorization solver |
| **`P6`** | `_decompress_ngc` (`m_ngs.py:48–181`) & File Loading Thread | **CPU-bound / I/O-bound** | Pure-Python bit-by-bit `lzhuf` adaptive Huffman + LZSS decompression loop on compressed `.ngc`/`.l6m` files runs on the main Qt thread. | **`9,743.61 ms`** (`min 9,093.97 ms`) on `3.9 MB` `mapls5.ngc` | — | **1.15×–1.3×** pure-Python RLE slice fast-path (**25×–40×** if optional `numba` JIT present) + **`0 ms` UI freeze** on background `QThread` |

---

## 3. Detailed Strategy A Workstreams (`P1`–`P6`)

### `P1` — Virtualize `VSpectraList` (`QListView` + `QAbstractListModel`)

#### Problem & Verified Evidence

- **Location**: `spectroview/view/components/lists/v_spectra_list.py:170–221` (`VSpectraList.set_spectra_names`).
- **Audit Finding on `VDataframeTable`**: Reviewing `spectroview/view/components/lists/v_dataframe_table.py:14–195` confirmed that **`VDataframeTable` is already virtualized** via `DataFrameTableModel(QAbstractTableModel)` + `QTableView` (caching column NumPy views `self._raw_cols = [self._df[c].to_numpy() for c in self._df.columns]` and sampling only `df.head(30)` in `_resize_columns_smart`). Under controlled benchmark conditions, `VDataframeTable.show()` renders a **`16,384 × 14` (`229,376`-cell)** fit results table in **`min = 9.77 ms | median = 10.76 ms | mean = 11.38 ± 1.64 ms`**. No architectural rewrite of `VDataframeTable` is needed.
- **Remaining GUI Bottleneck (`VSpectraList`)**: Unlike `VDataframeTable`, `VSpectraList` still subclasses `QListWidget` and executes a Python loop creating $N$ C++ `QListWidgetItem` instances, setting flags, check states, background brushes (`set_spectrum_item_color`), and three `Qt.UserRole` payloads per item on every list refresh.
- **Controlled Measurements (`[Measured — Current Baseline]` vs `[Measured — Isolated Prototype]`)**:

| Spectrum Count ($N$) | `VSpectraList.set_spectra_names` (`QListWidget`) `[Measured — Current Baseline]` | `QListView` + `QAbstractListModel` (`setUniformItemSizes(True)`) `[Measured — Isolated Prototype]` | Measured Speedup |
| :--- | :--- | :--- | :---: |
| **`1,520` spectra** (`2_MoS2_map`) | `min = 126.84 ms` · **`median = 185.35 ms`** · `mean = 173.09 ± 24.36 ms` | `min = 0.62 ms` · **`median = 0.74 ms`** · `mean = 1.20 ± 0.83 ms` | **`250×`** |
| **`3,721` spectra** (`3_3721map`) | `min = 625.01 ms` · **`median = 670.55 ms`** · `mean = 672.38 ± 31.31 ms` | `min = 1.52 ms` · **`median = 1.87 ms`** · `mean = 2.07 ± 0.54 ms` | **`358×`** |
| **`16,384` spectra** (`1_CL_map`) | `min = 5,237.55 ms` · **`median = 5,919.42 ms`** · `mean = 6,938.96 ± 1,765.86 ms` | `min = 6.33 ms` · **`median = 8.73 ms`** · `mean = 8.17 ± 1.09 ms` | **`678×`** |

#### Design & Touchpoints

1. **`SpectraListModel(QAbstractListModel)` (`v_spectra_list.py`)**:
   - Store lightweight parallel Python/NumPy sequences (`_fnames: list[str]`, `_is_active: np.ndarray`, `_has_baseline: np.ndarray`, `_has_fit: np.ndarray`, `_metadata: list[dict]`) and an $O(1)$ lookup dict `_fname_to_row: dict[str, int]`.
   - Implement `rowCount()`, `data()` (`Qt.DisplayRole`, `Qt.CheckStateRole`, `Qt.BackgroundRole`, `Qt.ForegroundRole`, `Qt.UserRole`), `setData()` (for checkbox toggles emitting `spectrum_active_changed`), and `flags()`.
   - Enable `setUniformItemSizes(True)` on the underlying `QListView` so Qt computes scrollbar geometry in $O(1)$ without querying `sizeHint()` across all $N$ rows.
2. **Preserve External `VSpectraList` API**:
   - Keep `set_spectra_names(spectra: list[dict])`, `get_selected_fnames() -> list[str]`, `set_selected_fnames(fnames: list[str])`, drag-and-drop reordering, and all existing signals (`spectra_selection_changed`, `spectrum_active_changed`, context-menu actions) 100% backward compatible.

---

### `P2` — C-Contiguous Map Extraction & Hyperspectral Map Memory De-Duplication

#### Problem & Verified Evidence

- **Locations**:
  - `spectroview/model/io/m_io.py:307–313` (`extract_map_arrays`)
  - `spectroview/viewmodel/maps/vm_workspace_maps.py:72–115, 290–330`
  - `spectroview/model/spectra/spectra_store.py:23–115` (`MapData`)
  - `spectroview/viewmodel/spectra/vm_workspace_spectra.py:1320–1390` (`_on_fit_finished`)
- **Verified Finding 1 — Fortran-Contiguous (`F_CONTIGUOUS`) Matrix Trap on `.txt`/`.csv` Maps**:
  - When `m_io.load_map_file()` parses a `.txt` or `.csv` map (`1_CL_map.txt`, `16,384 × 1,024`) via `pd.read_csv(..., engine="c")`, pandas stores the numeric block in **column-major Fortran order (`F_CONTIGUOUS=True, C_CONTIGUOUS=False`)**.
  - In `extract_map_arrays` (`m_io.py:310`), `Y0 = map_df.iloc[:, 2:].to_numpy().astype(np.float32, copy=False)` preserves Fortran memory layout (`strides = (4, 65536)` on `16,384 × 1,024`, whereas `WdfReader` produces C-contiguous `strides = (2300, 4)`).
  - Because `Y0` has a **64 KB stride between adjacent wavenumber channels in the same spectrum**, every row-wise operation (`np.diff(y, axis=1)`, `np.median(dy, axis=1)`, `moving_average_5`, `y[i]` single-spectrum plotting) suffers severe CPU cache-line thrashing:
    - **`[Measured — Current Baseline]`**: `compute_fit_weights` on F-contiguous `Y_cl` (`float32`, `16,384 × 1,024`) takes **`min = 2,054.62 ms | median = 2,355.58 ms`**.
    - **`[Measured — Isolated Prototype]`**: Simply converting `Y0` to C-contiguous row-major layout (`np.ascontiguousarray(Y0, dtype=np.float32)`, `strides = (4096, 4)`) reduces the exact same `compute_fit_weights` call to **`min = 642.51 ms | median = 659.18 ms`** — an immediate **`3.57×` measured speedup**!
- **Verified Finding 2 — `~1.01 GB` Memory Amplification per `67.1 MB` Raw Map (`[Measured — Current Baseline]`)**:
  For a single `16,384 × 1,024` map (`16,777,216` channels) fitted with a 3-peak model (`K=9`):

| Array / Structure | Location | Current Dtype & Shape | `[Measured — Current Baseline]` RAM | Post-`P2` Target Dtype & Policy | `[Theoretical — Post-P2 RAM]` |
| :--- | :--- | :--- | ---: | :--- | ---: |
| Raw intensities `md.Y0` | `SpectraStore` (`MapData`) | `float32 (16384, 1024)` (C-contiguous) | `67.1 MB` | Keep `float32 (16384, 1024)` C-contiguous | `67.1 MB` |
| Duplicate DataFrame `self.maps[name]` | `VMWorkspaceMaps.maps` | `float64 (16384, 1026)` | **`134.5 MB`** | **Eliminate** (construct view/DF lazily on export) | **`0.0 MB`** |
| Processed intensities `md.Y` | `SpectraStore` (`MapData`) | Upcast to `float64 (16384, 1024)` on baseline subtract | **`134.2 MB`** | Keep **`float32 (16384, 1024)`** (upcast chunks inside LM) | **`67.1 MB`** |
| Fitted envelope `md.Y_bestfit` | `SpectraStore` (`MapData`) | `float64 (16384, 1024)` | **`134.2 MB`** | Store as **`float32 (16384, 1024)`** | **`67.1 MB`** |
| Evaluated baseline `md.Y_baseline` | `SpectraStore` (`MapData`) | `float64 (16384, 1024)` | **`134.2 MB`** | Reconstruct on demand for selected spectrum (or `float32`) | **`0.0 MB`** |
| Individual peak curves `md.Y_peaks` | `SpectraStore` (`MapData`) | `float64 (16384, 3, 1024)` | **`402.7 MB`** | **Lazy on-demand evaluation** from `md.peak_params[idx]` | **`0.0 MB`** |
| **Total per `16,384 × 1,024` Map** | — | — | **`1,006.9 MB (~1.01 GB)`** | — | **`201.3 MB` (-80.0%)** |

#### Design & Touchpoints

1. **`P2.1` — Enforce C-Contiguous `float32` in `extract_map_arrays` (`m_io.py:309–312`)**:
   - Replace `.to_numpy().astype(np.float32, copy=False)` with `np.ascontiguousarray(map_df.iloc[:, 2:].to_numpy(), dtype=np.float32)`. One-time transpose during file load takes `~25 ms` on `16,384 × 1,024` and accelerates every subsequent row operation by **3.5×+**.
2. **`P2.2` — Remove Redundant `VMWorkspaceMaps.maps` Full-Matrix `float64` DataFrame**:
   - Use `SpectraStore` (`MapData.coords`, `MapData.x_axis`, `MapData.y_matrix`, `MapData.is_active`) as the single source of truth; build a pandas DataFrame only when an explicit table export requires it.
3. **`P2.3` — Preserve `float32` for `md.Y` and `md.Y_bestfit`**:
   - Cast `eval_baseline_batch` subtraction result back to `np.float32` when `md.Y0.dtype == np.float32`, and store `Y_bestfit` as `np.float32` after VBF completion (while keeping `peak_params` and `fit_r2` in exact `float64`).
4. **`P2.4` — Lazy On-Demand `Y_peaks` Evaluation**:
   - Compute individual decomposed peak curves only for the currently displayed/selected spectrum index $i$ in `VSpectraViewer` (`< 0.05 ms` for 1 spectrum) instead of materializing an `(N, P, M)` 3D `float64` tensor (`402.7 MB`) across all 16,384 pixels.

---

### `P3` — Algorithmic Optimization, OpenBLAS Thread Control & Background Threading for Multivariate Analysis (PCA & NMF)

#### Problem & Verified Evidence

- **Location**: `spectroview/model/analysis/m_mva.py:75–201` (`MMVA.run_pca`, `MMVA.run_nmf`) and `spectroview/viewmodel/analysis/vm_mva.py`.
- **Three Compounding Root Causes Verified in Controlled Benchmarks**:
  1. **OpenBLAS 8-Thread Oversubscription on 4 Physical Cores**: By default, OpenBLAS spawns 8 threads on a 4-core / 8-thread CPU. Inside `MMVA.run_nmf` (`200` iterations $\times$ `6` matrix multiplications on narrow `(1520, 4)` and `(4, 575)` matrices = `1,200` micro-GEMMs), 8 OpenBLAS threads spin-wait at thread barriers across 4 cores, causing extreme OS scheduler jitter (`min = 2,212.12 ms` vs `median = 7,283.42 ms`, `max = 12,681.25 ms`). Setting `OPENBLAS_NUM_THREADS=2` alone cuts current NMF median time from **`7,283.42 ms` to `1,567.99 ms` (`4.65×` speedup)** and current SVD PCA median time from **`2,471.73 ms` to `652.80 ms` (`3.79×` speedup)**!
  2. **PCA Full Thin SVD Waste (`m_mva.py:94`)**: `np.linalg.svd(X_centered, full_matrices=False)` computes all $\min(n, p) = 575$ singular vectors on an $(n, p)$ rectangular matrix even when `n_components = 5`. Forming the symmetric $(p \times p)$ covariance Gram matrix $C = X_c^T X_c$ via BLAS `dsyrk`/`dgemm` and calling `np.linalg.eigh(C)` is mathematically equivalent and dramatically faster whenever $n > p$.
  3. **NMF Explicit `(N, M)` Residual Allocation (`m_mva.py:189`)**: Every 10 iterations, `MMVA.run_nmf` computes `error = np.linalg.norm(X - W @ H, "fro")`, allocating a full `(n, p)` residual matrix even though `numerator_W = X @ H.T`, `W.T @ W`, and `H @ H.T` are already available for the $O(K^2)$ Frobenius trace identity.
- **Controlled Multi-Run Measurements (`1,520 × 575`, `float64`)**:

| Algorithm & Configuration | Measurement Category | `min` | **`median`** | `mean ± std` | Numerical Parity vs. Current Baseline | Measured Speedup (Median) |
| :--- | :--- | ---: | ---: | ---: | :--- | :---: |
| `MMVA.run_pca` (`k=5`, current `svd`, default 8 BLAS threads) | `[Measured — Current Baseline]` | `1,033.34 ms` | **`2,471.73 ms`** | `2,446.76 ± 895.34 ms` | Reference | `1.00×` |
| `MMVA.run_pca` (`k=5`, current `svd`, `OPENBLAS_NUM_THREADS=2`) | `[Measured — Isolated Prototype]` | `449.27 ms` | **`652.80 ms`** | `621.01 ± 91.33 ms` | Exact (`0.0`) | **`3.79×`** |
| `P3.1` Gram `eigh(X_c.T @ X_c)` (`k=5`, default 8 BLAS threads) | `[Measured — Isolated Prototype]` | `129.83 ms` | **`222.91 ms`** | `198.92 ± 49.65 ms` | `max_ev_diff = 3.73e-09` | **`11.09×`** |
| **`P3.1` Gram `eigh(X_c.T @ X_c)` (`k=5`, `OPENBLAS_NUM_THREADS=2`)** | **`[Measured — Isolated Prototype]`** | **`121.68 ms`** | **`125.65 ms`** | **`139.89 ± 28.48 ms`** | **`max_ev_diff = 3.73e-09`** | **`19.67×`** |
| `MMVA.run_nmf` (`k=4, 200 it`, current, default 8 BLAS threads) | `[Measured — Current Baseline]` | `2,212.12 ms` | **`7,283.42 ms`** | `7,392.26 ± 4,274.69 ms` | Reference | `1.00×` |
| `MMVA.run_nmf` (`k=4, 200 it`, current, `OPENBLAS_NUM_THREADS=2`) | `[Measured — Isolated Prototype]` | `1,455.54 ms` | **`1,567.99 ms`** | `1,626.89 ± 170.51 ms` | Exact (`0.0`) | **`4.65×`** |
| `P3.2` Trace-identity + prealloc NMF (`k=4`, default 8 BLAS threads) | `[Measured — Isolated Prototype]` | `974.51 ms` | **`3,927.84 ms`** | `4,824.14 ± 3,288.24 ms` | `max_W_diff = 8.27e-05` | **`1.85×`** |
| **`P3.2` Trace-identity + prealloc NMF (`k=4`, `OPENBLAS_NUM_THREADS=2`)** | **`[Measured — Isolated Prototype]`** | **`548.04 ms`** | **`582.02 ms`** | **`587.07 ± 25.28 ms`** | **`max_W_diff = 8.27e-05`** | **`12.51×`** |

#### Design & Touchpoints

1. **`P3.1` — Dual Covariance Gram Eigendecomposition in `MMVA.run_pca` (`m_mva.py:75–133`)**:
   - When $n > p$, compute $C = X_c^T X_c$ (`shape = (p, p)`) and `eigvals, eigvecs = np.linalg.eigh(C)`, sort descending (`[::-1]`), clamp tiny negative numerical noise with `np.maximum(eigvals, 0.0)`, and recover `all_eigenvalues = eigvals / (n - 1)`, `loadings = eigvecs[:, :n_components].T`, and `scores = X_c @ loadings.T` (applying deterministic sign canonicalization so the largest-magnitude loading entry per component is positive).
   - When $n \le p$, use the dual $(n \times n)$ Gram matrix $C = X_c X_c^T$ with `np.linalg.eigh(C)` and recover unit-norm loadings via $V_k = S_k^{-1} U_k^T X_c$.
2. **`P3.2` — $O(K^2)$ Frobenius Trace Identity & Preallocated Buffers in `MMVA.run_nmf` (`m_mva.py:138–201`)**:
   - Precompute `norm_X_sq = float(np.sum(X * X))` once before the loop and preallocate `num_H`, `den_H`, `WtW`, `num_W`, `den_W`, and `HHt` buffers using `np.matmul(..., out=...)`.
   - Every 10 iterations, refresh `np.matmul(W.T, W, out=WtW)` after updating `W` and evaluate:

     $$\|X - WH\|_F = \sqrt{\max\left(0,\; \|X\|_F^2 - 2 \sum_{i,j} W_{i,j} (XH^T)_{i,j} + \sum_{a,b} (W^T W)_{a,b} (H H^T)_{a,b}\right)}$$

3. **`P3.3` — Application-Level OpenBLAS Thread Cap & Background `QThread` in `VMMVA` (`main.py`, `vm_mva.py`)**:
   - In `spectroview/__init__.py` (before `import numpy`), set default `os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")` and `os.environ.setdefault("OMP_NUM_THREADS", "2")` (or `min(4, max(2, os.cpu_count() // 2))`) so OpenBLAS never oversubscribes logical hyperthreads on 4-core laptops.
   - Offload `MMVA.run_pca` and `MMVA.run_nmf` to a parented `QThread` worker in `VMMVA` when $n \cdot p \ge 250{,}000$.

---

### `P4` — VBF Engine, `float32` Noise/Weight Pipeline & Analytical $3 \times 3$ SPD Solver

#### Problem & Verified Evidence

- **Locations**:
  - `spectroview/fit_engine/noise.py:14–85` (`_array_2d_fingerprint`, `mad_noise`, `moving_average_5`)
  - `spectroview/fit_engine/weights.py:7–36` (`compute_fit_weights`)
  - `spectroview/fit_engine/optimizer.py:45–75` (`_batched_solve`)
- **Verified Finding 1 — Mixed-Dtype Upcasting in `moving_average_5` & Full-Array Sum in `_array_2d_fingerprint`**:
  - In `noise.py:69`, `moving_average_5(y)` allocates `out = np.empty((N, M), dtype=np.float64)` even when `y` is `float32`, forcing 5 mixed-dtype (`float64 += float32`) strided ufunc passes (`134.2 MB` allocation instead of `67.1 MB`).
  - In `noise.py:25`, `_array_2d_fingerprint(y)` computes `float(np.sum(y))` across all `16,777,216` elements (`12.50 ms` per call $\times 2$ calls = `25 ms` overhead even on a cache hit, while `compute_fit_weights` takes **`169.17 ms` even on a cache hit** because it allocates a `134.2 MB` `float64` `weights` array and scans two full `(N, M)` boolean masks!).
  - **`[Measured — Current Baseline]` vs `[Measured — Isolated Prototype]` (`16,384 × 1,024`)**:
    - Current `compute_fit_weights` on F-contiguous `Y_cl` (`float32`): **`median = 2,355.58 ms`** (`min = 2,054.62 ms`).
    - Current `compute_fit_weights` on C-contiguous `Y_cl` (`float32`): **`median = 659.18 ms`** (`min = 642.51 ms`).
    - **`[Measured — Isolated Prototype P4.1]`** (C-contiguous `float32` + `float32`-preserving `moving_average_5` + strided `y[::16, ::16]` fingerprint): **`median = 483.46 ms`** (`min = 453.84 ms`, `mean = 479.36 ± 16.59 ms`, **`0.000000` mask mismatch**, **`4.87×` faster than current F-contiguous baseline and `1.36×` faster than C-contiguous baseline**).
- **Verified Finding 2 — Closed-Form Vectorized $3 \times 3$ SPD Solver for 1-Peak Fits ($K = 3$)**:
  - For 1-peak Gaussian/Lorentzian fits ($K = 3$), `_batched_solve` calls general LAPACK `dgesv` (`np.linalg.solve`) on `(Na, 3, 3)` matrices.
  - **`[Measured — Current Baseline]`**: `np.linalg.solve` on `(16,384 × 3 × 3)` takes **`min = 6.84 ms | median = 9.76 ms | mean = 9.84 ± 2.02 ms`**.
  - **`[Measured — Isolated Prototype P4.2]`**: Direct vectorized $3 \times 3$ symmetric cofactor/Cramer rule takes **`min = 1.36 ms | median = 1.54 ms | mean = 1.61 ± 0.25 ms`** (**`6.34×` measured speedup**, `max_diff = 5.95e-14`).
- **Verified Finding 3 — Effect of `OPENBLAS_NUM_THREADS=2` on VBF Batched LM Fits**:
  - Because `optimizer.py` (since commit `9250ae9`) chunks large maps into L3-cache sub-batches of `256–2,048` spectra, running OpenBLAS with 8 threads on 4 physical cores causes thread-barrier overhead inside each chunk's `np.matmul(J_r_T, J_r)` normal-equation step:
  - **`[Measured — Current Baseline]` (`1_CL_map.json`, `16,384 × 302`, `1 Lorentzian`, `K=3`, default 8 BLAS threads)**: **`min = 3,385.22 ms | median = 3,433.74 ms`** (`0.210 ms/spectrum`).
  - **`[Measured — Isolated Prototype P4.3]` (`1_CL_map.json`, `16,384 × 302`, `OPENBLAS_NUM_THREADS=2`)**: **`1,434.02 ms`** (`0.088 ms/spectrum`, **`2.40×` measured speedup**, `100.0%` converged, `med_R2 = 0.9841`)!

#### Design & Touchpoints

1. **`P4.1` (`noise.py`, `weights.py`)**:
   - Preserve input `float32`/`float64` dtype in `moving_average_5(y)` (`out = np.empty((N, M), dtype=np.result_type(y.dtype, np.float32))`).
   - Replace full `float(np.sum(y))` in `_array_2d_fingerprint(y)` with a constant-time strided sample `float(np.sum(y[::max(1, n // 32), ::max(1, m // 32)], dtype=np.float64))` + corner/center elements.
2. **`P4.2` (`optimizer.py:_batched_solve`)**:
   - Add a fast closed-form $3 \times 3$ symmetric solver branch when `K == 3`, falling back to `_batched_solve` general path if `np.any(det <= 1e-18) or not np.all(np.isfinite(dp))`.
3. **`P4.3` (`spectroview/__init__.py` / `optimizer.py`)**:
   - Cap default OpenBLAS/OMP threads to `2` (or physical core count $\div 2$) to eliminate the `2.40×` multi-thread barrier penalty inside L3-chunked VBF fits.

---

### `P5` — Background Worker & Multi-Core Acceleration for Batch Automatic Baselines (`arpls`, `asls`, `snip`)

#### Problem & Verified Evidence

- **Location**: `spectroview/fit_engine/baseline.py:112–244` (`eval_baseline`, `eval_baseline_batch`) and `spectroview/viewmodel/spectra/vm_workspace_spectra.py` (`subtract_baseline`).
- **Controlled Idle-CPU Measurements (`50 × 575` spectra from `3_3721map.wdf`, `[Measured — Current Baseline]`)**:
  - Attached **`Linear`** (`1,520 × 575`): **`median = 30.61 ms`** (`min = 29.76 ms`, **`0.020 ms/spectrum`**) — already fully vectorized.
  - Attached **`Polynomial` (degree 3)** (`1,520 × 575`): **`median = 30.79 ms`** (`min = 29.05 ms`, **`0.020 ms/spectrum`**) — already fully vectorized.
  - Automatic **`asls` (ALS)** (`50 × 575`): **`median = 56.61 ms`** (`min = 46.68 ms`, **`1.13 ms/spectrum`** $\rightarrow$ `[Estimated — Integrated Workflow]` **`1.72 s`** for `1,520` spectra).
  - Automatic **`arpls`** (`50 × 575`): **`median = 238.65 ms`** (`min = 225.08 ms`, **`4.77 ms/spectrum`** $\rightarrow$ `[Estimated — Integrated Workflow]` **`7.25 s`** for `1,520` spectra).
  - Automatic **`snip`** (`50 × 575`): **`median = 378.88 ms`** (`min = 349.80 ms`, **`7.58 ms/spectrum`** $\rightarrow$ `[Estimated — Integrated Workflow]` **`11.52 s`** for `1,520` spectra).
- **Important Negative Finding From Controlled Prototype Testing**:
  - We explicitly benchmarked two candidate micro-optimizations on `arpls` (`50 × 575`):
    1. Hoisting `Baseline(x_data=x)` outside the per-spectrum loop in a single thread (`[Measured — Isolated Prototype]`: **`median = 263.38 ms`**, `min = 226.85 ms` vs `238.65 ms` baseline — **no speedup**, because `Baseline(x_data=x)` initialization takes `< 0.1 ms` while `99%` of time is spent inside `scipy.linalg.solveh_banded` during iterative Whittaker reweighting).
    2. Running `ThreadPoolExecutor(max_workers=4)` over hoisted `Baseline(x_data=x)` instances while OpenBLAS uses default threads (`[Measured — Isolated Prototype]`: **`median = 366.48 ms`**, `min = 336.10 ms` — **`1.53×` slower** due to GIL contention inside `pybaselines` Python iteration loops and thread contention!).
  - **Why This Matters**: Recording this negative result prevents wasting engineering time on naive `Baseline` hoisting or `ThreadPoolExecutor` wrapping of `pybaselines`.

#### Validated Design & Touchpoints

1. **`P5.1` — Non-Blocking `QThread` Execution for Batch Automatic Baselines (`vm_workspace_spectra.py`)**:
   - Whenever `subtract_baseline` runs an automatic baseline mode (`mode not in ("Linear", "Polynomial")`) across $N > 10$ spectra, execute `eval_baseline_batch` inside a parented `QThread` worker with a progress bar and cancel button (mirroring `VBFthread`).
   - **`[Estimated — Integrated Workflow]`**: Main-thread UI freeze drops from **`1.7–11.5 s` to `0 ms`**.
2. **`P5.2` — Vectorized Batch `SNIP` in Pure NumPy (`baseline.py`)**:
   - The `SNIP` algorithm (`Iterative decreasing/increasing window morphological clipping`: $y^{(p)}_{i,j} = \min(y^{(p-1)}_{i,j},\; 0.5 (y^{(p-1)}_{i,j-p} + y^{(p-1)}_{i,j+p}))$ for $p = 1 \dots W$) has **zero data-dependent branching across spectra** and can be written in **12 lines of 2D vectorized NumPy slice operations** over the entire `(N, M)` matrix simultaneously!
   - **`[Theoretical — Asymptotic Scaling]`**: Vectorizing 2D `SNIP` in NumPy eliminates $N \times W$ Python loop iterations, reducing `1,520 × 575` `SNIP` from `~11.5 s` to **`< 150 ms` (~75× speedup)**.

---

### `P6` — Legacy `.ngc` (`lzhuf`) Decompression & Background File Loading

#### Problem & Verified Evidence

- **Location**: `spectroview/model/io/m_ngs.py:48–181` (`_decompress_ngc`).
- **Controlled Measurement (`[Measured — Current Baseline]`)**:
  - Under 2.6% idle CPU, `NgsReader("examples/spectroscopic_data/mapls5.ngc")` (`3.9 MB` compressed $\rightarrow$ `7.40 MB` uncompressed) takes **`min = 9,093.97 ms | median = 9,743.61 ms`** (`2.38×` faster than the `23,175.7 ms` high-CPU measurement, but still a ~9.7 s synchronous block on the main Qt thread when loading a legacy LabSpec 5 `.ngc` map).
  - By contrast, `.wdf` (`14.50 ms`), `.spc` (`13.67 ms`), and `.l6m` (`32.42 ms`) load in **`13–33 ms`**.

#### Design & Touchpoints

1. **`P6.1` — RLE `dist == 1` Bytearray Slice Fast Path & Local Variable Binding (`m_ngs.py:171–177`)**:
   - In `_decompress_ngc`, add `elif dist == 1: out[out_idx : out_idx + match_len] = bytes([out[src]]) * match_len; out_idx += match_len` for zero-run fills.
   - Optionally decorate the inner integer bit-loop with `@numba.njit(cache=True)` inside a `try: import numba except ImportError:` block (zero hard dependency; `[Estimated — Integrated Workflow]`: `< 250 ms` when `numba` is present, `~7.5–8.5 s` in pure CPython).
2. **`P6.2` — Background `QThread` File Loader for Heavy Files (`vm_workspace_maps.py`)**:
   - Offload file loading for `.ngc` files or text maps `> 20 MB` (`1_CL_map.txt`, `151 MB`, `1.12 s`) to a background `QThread` with a busy-cursor/progress status indicator so the main window never freezes during disk I/O.

---

## 4. Amdahl's Law End-to-End Workflow Impact (Re-Calibrated to Idle-CPU Baseline)

### Scenario 1: Loading, Inspecting, Fitting & Plotting `1_CL_map.txt` (`16,384 × 1,024`, `151 MB`)

| Workflow Stage | `[Measured — Current Baseline]` (Idle CPU) | Post-Strategy A Target (`[Measured Prototype]` or `[Estimated]`) | Speedup / Saving | Source of Improvement |
| :--- | ---: | ---: | :---: | :--- |
| 1. Parse `1_CL_map.txt` + `extract_map_arrays` (`C_CONTIGUOUS`) | `1,118.58 ms + 85.64 ms = 1,204.22 ms` | `~1,225 ms` (offloaded to background thread, `0 ms` UI freeze) | Non-blocking UI | `P2.1` + `P6.2` |
| 2. Populate `VSpectraList` (`16,384` items) | **`5,919.42 ms`** (main-thread freeze) | **`8.73 ms`** `[Measured — Isolated Prototype P1]` | **`678×`** | `P1` (`QListView` + `QAbstractListModel`) |
| 3. Compute Noise Weights (`compute_fit_weights`, `16,384 × 1,024`) | **`2,355.58 ms`** (F-contiguous `Y0`) | **`483.46 ms`** `[Measured — Isolated Prototype P4.1]` | **`4.87×`** | `P2.1` (`C_CONTIGUOUS`) + `P4.1` (`float32` `moving_average_5` & strided fingerprint) |
| 4. VBF Fit (`1_CL_map.json`, `16,384 × 302`, `1 Lorentzian`, `K=3`) | **`3,433.74 ms`** (8 OpenBLAS threads) | **`1,434.02 ms`** `[Measured — Isolated Prototype P4.3]` (`~1,250 ms` `[Estimated]` with `P4.2` $3 \times 3$ solver) | **`2.40×–2.75×`** | `P4.3` (`OPENBLAS_NUM_THREADS=2`) + `P4.2` (closed-form $3 \times 3$ solver) |
| 5. Populate Fit Results Table (`VDataframeTable`, `16,384 × 14`) | **`10.76 ms`** (already virtualized) | **`10.76 ms`** `[Measured — Current Baseline]` | `1.00×` (already optimal) | Already uses `DataFrameTableModel(QAbstractTableModel)` |
| 6. Render Wafer / 2D Map in `VGraph` (`16,384` points) | **`57.80 ms – 73.06 ms`** | **`57.80 ms – 73.06 ms`** `[Measured — Current Baseline]` | `1.00×` (already fast) | Vectorized `PlotRenderer` |
| **Total End-to-End Wall Time** | **`12,996.8 ms (~13.00 s)`** | **`~3,235 ms (~3.24 s)`** | **`~4.02×` faster overall** | Pure Python + NumPy + PySide6 |
| **Total Main-Thread UI Freeze Time** | **`~7,208 ms (~7.21 s)`** | **`< 90 ms`** (imperceptible) | **`~80×` lower UI blocking** | `P1` virtualization + `P6.2` background load |
| **Steady-State RAM Footprint (Post-3-Peak Fit)** | **`~1,006.9 MB (~1.01 GB)`** | **`~201.3 MB`** `[Theoretical — Post-P2 RAM]` | **`-80.0%` RAM (`~805 MB` saved)** | `P2.2`–`P2.4` memory de-duplication |

### Scenario 2: Multivariate Analysis (PCA + NMF) & Batch Baseline on `1,520 × 575` Map

| Workflow Stage | `[Measured — Current Baseline]` (Idle CPU) | Post-Strategy A Target (`[Measured Prototype]` or `[Estimated]`) | Speedup | Source of Improvement |
| :--- | ---: | ---: | :---: | :--- |
| 1. `MMVA.run_pca` (`1,520 × 575`, `n_components=5`) | **`2,471.73 ms`** (full `svd`, 8 BLAS threads) | **`125.65 ms`** `[Measured — Isolated Prototype P3.1]` | **`19.67×`** | `P3.1` Gram `eigh(X_c.T @ X_c)` + `OPENBLAS_NUM_THREADS=2` |
| 2. `MMVA.run_nmf` (`1,520 × 575`, `k=4`, `200` iters) | **`7,283.42 ms`** (residual alloc, 8 BLAS threads) | **`582.02 ms`** `[Measured — Isolated Prototype P3.2]` | **`12.51×`** | `P3.2` $O(K^2)$ trace identity & preallocated buffers + `OPENBLAS_NUM_THREADS=2` |
| 3. Batch `SNIP` Baseline (`1,520 × 575`) | **`~11,520 ms`** (`7.58 ms/spec` sequential loop) | **`< 150 ms`** `[Estimated — Integrated Workflow P5.2]` | **`~75×`** | `P5.2` 2D vectorized NumPy morphological clipping |
| **Total Analytical Workflow Time** | **`~21,275 ms (~21.3 s)`** | **`~858 ms (< 0.9 s)`** | **`~24.8×` faster** | Pure NumPy / BLAS |

---

## 5. Phased Implementation Plan & Verification Gates

When ready to implement Strategy A, execute the workstreams in three self-contained milestones:

### Milestone 1 — Memory Layout, OpenBLAS Thread Guard & Map RAM De-Duplication (`P2` + `P4.1` + `P4.3`)

- [ ] Set default `OPENBLAS_NUM_THREADS=2` / `OMP_NUM_THREADS=2` in `spectroview/__init__.py` (before `numpy` import) when not overridden by the user's environment.
- [ ] Enforce `np.ascontiguousarray(..., dtype=np.float32)` on `Y0` in `m_io.extract_map_arrays` (`m_io.py:309–312`).
- [ ] Preserve `float32` in `moving_average_5` and replace full `np.sum(y)` in `_array_2d_fingerprint` with strided sub-sampling (`noise.py`).
- [ ] Eliminate duplicate `VMWorkspaceMaps.maps[name]` `float64` DataFrame, keep `md.Y` and `md.Y_bestfit` in `float32`, and make `md.Y_peaks` lazy on demand.
- **Verification Gate**: `python -m pytest tests/unit/model tests/unit/viewmodel tests/unit/fit_engine -q` + verify `Y0.flags["C_CONTIGUOUS"] is True` on `.txt`, `.csv`, `.wdf`, `.spc`, and `.l6m` maps.

### Milestone 2 — `VSpectraList` Virtualization (`P1`)

- [ ] Refactor `VSpectraList` (`spectroview/view/components/lists/v_spectra_list.py`) to use `QListView` (`setUniformItemSizes(True)`) + `SpectraListModel(QAbstractListModel)`, following the proven pattern in `VDataframeTable` (`v_dataframe_table.py`).
- **Verification Gate**: `python -m pytest tests/unit/view -q` + benchmark assertion that `set_spectra_names` for `16,384` items completes in `< 25 ms`.

### Milestone 3 — Fast PCA/NMF, Closed-Form $3 \times 3$ VBF Solver & Vectorized `SNIP` / Background Workers (`P3` + `P4.2` + `P5` + `P6`)

- [ ] Implement Gram `np.linalg.eigh(X_c.T @ X_c)` in `MMVA.run_pca` and $O(K^2)$ trace identity + preallocated buffers in `MMVA.run_nmf` (`m_mva.py`).
- [ ] Add closed-form $3 \times 3$ symmetric solver fast path in `optimizer.py::_batched_solve`.
- [ ] Vectorize 2D `SNIP` in `baseline.py` and add background `QThread` execution for batch automatic baselines (`vm_workspace_spectra.py`) and heavy file loading (`vm_workspace_maps.py`).
- **Verification Gate**: Full test suite (`python -m pytest tests/ -m "not slow" -q`), startup import check (`tests/performance/test_startup_imports.py`), and bilingual documentation synchronization (`docs/developer/` and `docs/user_manual/`).
