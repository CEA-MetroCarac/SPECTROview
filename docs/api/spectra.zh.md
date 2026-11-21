# 光谱工作区 { #spectra-workspace }

本节介绍如何处理离散一维光谱：使用向量化批量拟合 (Vectorized Batch Fit, VBF) 引擎对其进行加载、预处理与拟合——既可使用有状态的 `SpectraWorkspace` 会话（推荐），也可直接调用其底层基于数组的函数。

---

## 有状态工作区 { #stateful-workspaces }

`spectroview.api.workspace.SpectraWorkspace` 与 GUI 中的光谱工作区完全对应：它用于存储您的光谱数据，为每条光谱跟踪预处理与拟合模型状态，并读写与 GUI 中 **保存工作区** (Save work) 按钮生成的完全相同的 `.spectra` 文件格式。完全由脚本构建的工作区可直接在 GUI 中打开，而在 GUI 中保存的 `.spectra` 文件也可直接在此加载。

```python
from spectroview.api import workspace, fitting

ws = workspace.SpectraWorkspace()
ws.load_files(["sample_a.wdf", "sample_b.wdf", "sample_c.txt"])
print(ws)  # SpectraWorkspace(3 spectra: ['sample_a', 'sample_b', 'sample_c'])

# Preprocess every spectrum in the workspace
ws.crop(range_min=400.0, range_max=800.0)
ws.set_baseline({"mode": "arpls", "smoothing_factor": 1e5})
ws.subtract_baseline()

# Attach a fit model and fit
fit_model = fitting.build_fit_model(
    peaks=[
        {"model": "Lorentzian",
         "x0": {"value": 520.0, "min": 515.0, "max": 525.0},
         "ampli": {"value": 1000.0, "min": 0.0, "max": 1e9},
         "fwhm": {"value": 3.0, "min": 1.0, "max": 10.0}},
    ]
)
ws.set_fit_model(fit_model)
ws.fit()

df = ws.collect_results()   # pandas DataFrame, one row per spectrum
print(df)

# Save — this file opens directly in the SPECTROview GUI
ws.save("session.spectra")
```

如需稍后重新加载会话（无论通过本 API 创建还是从 GUI 保存）：

```python
ws2 = workspace.SpectraWorkspace.load("session.spectra")
```

每个预处理和拟合方法均接受一个可选的 `names=[...]` 列表，用于指定仅对部分光谱子集（而非全部光谱）进行操作，例如 `ws.crop(range_min=400, range_max=800, names=["sample_a"])`。

!!! note "旧版文件"
    早期的 `.spectra` 文件（v1 格式，早于当前的 ZIP 保存格式）仅能在 GUI 中打开。`SpectraWorkspace.load()` 读取此类文件时会引发 `WorkspaceError`——请在 GUI 中打开该文件并重新保存，即可将其升级至当前格式。

---

## 加载数据 { #loading-data }

`spectroview.api.io` 模块提供了从多种格式（WDF、SPC、L6S、NGS、CSV、TXT、DAT）加载光谱的统一函数——这正是 `SpectraWorkspace.load_files()` 在内部调用的接口，同时也适合单独用于下文介绍的底层数组工作流。

```python
from spectroview.api import io
import numpy as np

spectra_dict = io.load_spectra("my_sample_spectra.wdf")

# Each item in the list is one spectrum (WDF series files can contain many)
items = spectra_dict["items"]
print(f"Number of spectra loaded: {len(items)}")

# Each item has: "name", "x0" (wavenumber axis), "y0" (intensities), "metadata"
first = items[0]
print("Wavenumber axis (x0):", first["x0"].shape)   # shape (M,)
print("Intensities (y0):    ", first["y0"].shape)   # shape (M,)
print("Laser Wavelength:", first["metadata"].get("Laser Wavelength (nm)"))
```

### 批量加载多个文件 { #loading-multiple-files-as-a-batch }

若希望将一组光谱作为单个二维矩阵（而非 `SpectraWorkspace`）进行统一处理，请使用 `io.load_spectra_to_matrix()`：

```python
import glob

file_paths = sorted(glob.glob("./measurements/*.wdf"))
data = io.load_spectra_to_matrix(file_paths)

x     = data["x"]      # float64[M]   — shared wavenumber axis
Y     = data["Y"]      # float64[N,M] — intensity matrix (N spectra)
names = data["names"]  # list[str]    — spectrum names
```

!!! note
    如果各光谱的横坐标轴略微不同，`load_spectra_to_matrix()` 会将所有光谱插值对齐到第一个文件的 X 轴上。

---

## 预处理 { #preprocessing }

如需在不使用 `SpectraWorkspace` 的情况下直接控制数组层级的操作，`spectroview.api.preprocessing` 以面向 `(x, Y)` 数组的无状态函数形式，提供了相同的裁剪、基线扣除和归一化功能。

```python
from spectroview.api import preprocessing

# Crop to the region of interest
x_crop, Y_crop = preprocessing.crop_spectra(x, Y, range_min=400.0, range_max=800.0)

# Baseline subtraction (same algorithms as the GUI, e.g. arpls, airpls, asls, Linear, Polynomial)
baseline_config = {"mode": "arpls", "smoothing_factor": 1e5}
Y_corrected, Y_baseline = preprocessing.subtract_baseline(x_crop, Y_crop, baseline_config)

# Normalize each spectrum to its own maximum
Y_norm = preprocessing.normalize_spectra(Y_corrected)
```

---

## 批量拟合 { #batch-fitting }

SPECTROview 的核心能力在于其向量化批量拟合 (VBF) 引擎，能够在极短时间内拟合数千条光谱。`spectroview.api.fitting` 将该引擎直接开放给数组级工作流（`SpectraWorkspace.fit()` 在底层也是逐条光谱调用它来完成拟合）。

### 第 1 步：定义或加载拟合模型 { #step-1-define-or-load-the-fit-model }

**方式 A：加载从 SPECTROview GUI 导出（或通过本 API 保存）的模板**

```python
from spectroview.api import fitting

fit_model = fitting.load_fit_model_template("my_gui_model.json")
```

**方式 B：使用 `fitting.build_fit_model()` 构建模型**

```python
fit_model = fitting.build_fit_model(
    peaks=[
        {"model": "Lorentzian",
         "x0": {"value": 520.0, "min": 515.0, "max": 525.0},
         "ampli": {"value": 1000.0, "min": 0.0, "max": 1e9},
         "fwhm": {"value": 3.0, "min": 1.0, "max": 10.0}},
        {"model": "Gaussian",
         "x0": {"value": 600.0, "min": 580.0, "max": 620.0},
         "ampli": {"value": 500.0, "min": 0.0, "max": 1e9},
         "fwhm": {"value": 15.0, "min": 5.0, "max": 50.0}},
    ]
)
```

每个峰参数字典均接受 `value`、`min`、`max`，并可选指定 `vary`（布尔值，默认为 `True`）或 `fix`（与 `vary` 相反）以及 `expr`（将该参数关联到其他参数的字符串表达式）。

### 第 2 步：执行拟合 { #step-2-execute-the-fit }

```python
fit_params = {"xtol": 1e-4, "ftol": 1e-4, "max_ite": 500}

results = fitting.fit_batch(x_crop, Y_corrected, fit_model, fit_params=fit_params)

print(f"Success rate: {results['success'].mean() * 100:.1f}%")
print(f"Average R-squared: {results['r_squared'].mean():.4f}")

fitted_params_matrix = results["params"]        # shape (N, K)
parameter_names      = results["param_names"]   # e.g. ['P1_x0', 'P1_fwhm', ...]
```

默认情况下，`fit_batch` 会在每次拟合前参照 GUI 的逻辑从 `fit_params` 自动推导拟合权重——负强度数据点将被排除（除非 `fit_params["fit_negative"]` 为 `True`），低于 `fit_params["coef_noise"]` 所设噪声阈值的数据点同样会被排除。传入 `auto_weights=False` 可对所有数据点以等权重进行拟合，或者也可传入自定义的 `weights` 数组。

对于仍需先应用拟合模型中裁剪和基线步骤的原始光谱矩阵，请使用 `fitting.apply_fit_model(x, Y, fit_model)`；该函数在单次调用中依次完成全部三个步骤（裁剪、基线、拟合），并同时返回实际参与拟合的 `x`/`Y`。

### 第 3 步：导出结果 { #step-3-export-results }

```python
import pandas as pd
from spectroview.api import io

df_results = pd.DataFrame(results["params"], columns=results["param_names"])
df_results["R_squared"]   = results["r_squared"]
df_results["Fit_Success"] = results["success"]

io.export_results(df_results, "my_fit_results.csv")
```

### 保存拟合模型以便复用 { #saving-a-fit-model-for-reuse }

```python
fitting.save_fit_model_template(fit_model, "my_model.json")

# List and reload templates from a folder later
names = fitting.list_fit_model_templates("./models")
fit_model = fitting.load_fit_model_template("./models/my_model.json")
```
