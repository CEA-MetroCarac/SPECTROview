# 二维图谱工作区与 2D 分析 { #maps-workspace-2d-analysis }

本节介绍如何处理高光谱二维图谱数据（例如拉曼晶圆图谱）：包括加载数据、对每个像素进行拟合、生成热图与线剖面，以及运行多元分析 (MVA)——既可通过有状态的 `MapsWorkspace` 会话（推荐方式）完成，对于 MVA 也可直接调用基于数组的底层函数。

---

## 有状态工作区 { #stateful-workspaces }

`spectroview.api.workspace.MapsWorkspace` 对应图形界面中的**二维图谱** (Maps) 工作区：每个加载的文件对应一幅二维图谱，图谱中的所有像素共享同一个拟合模型，各像素标识符完全按照图形界面的命名规范设为 `f"{map_name}_({x}, {y})"`（当时间分辨图谱在同一 `(x, y)` 坐标处包含重复扫描时，则为 `f"{map_name}_({x}, {y})_{k}"`）。它读写的 `.maps` 文件格式与图形界面中**保存工作区** (Save work) 生成的文件完全一致。

```python
from spectroview.api import workspace, fitting

ws = workspace.MapsWorkspace()          # or MapsWorkspace(map_type="wafer_300mm") for wafer maps
[map_name] = ws.load_files(["2Dmap_Si.txt"])

fit_model = fitting.load_fit_model_template("fit_model_Si.json")
ws.set_fit_model(fit_model, names=[map_name])
ws.fit(map_names=[map_name])            # one vectorized fit call over every pixel in the map

df = ws.collect_results()               # one row per pixel, includes X/Y coordinates
print(df.head())

# Build a heatmap of a fitted parameter
xi, yi, zi = ws.get_heatmap(map_name, "ampli_Si")

import matplotlib.pyplot as plt
plt.imshow(zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()], origin="lower", cmap="jet")
plt.colorbar(label="Amplitude")
plt.savefig("heatmap.png", dpi=200)

# Extract a line profile between two points on the heatmap
profile = ws.extract_profile(map_name, "ampli_Si", point1=(xi.min(), yi.min()), point2=(xi.max(), yi.max()))
print(profile)  # columns: X, Y, distance, values

# Save — this file opens directly in the SPECTROview GUI
ws.save("session.maps")
```

`get_heatmap(map_name, value_col, ...)` 支持将 `value_col` 设为 `"Intensity"` 或 `"Area"`（直接基于原始强度计算，无需拟合），或设为 `df_fit_results` 中的任意拟合参数列（请先调用 `collect_results()`）。对于晶圆图类型（`map_type != "2Dmap"`），网格会根据 `map_type` 对应的晶圆半径通过散点插值构建；而对于 `"2Dmap"`，则直接在采集数据自身的规则网格上执行快速精确的透视重排（pivot）。

后续重新加载先前保存的会话（无论是通过本 API 构建还是从图形界面保存）：

```python
ws2 = workspace.MapsWorkspace.load("session.maps")
```

`MapsWorkspace` 继承了 `SpectraWorkspace` 的全部预处理方法（`crop`、`set_baseline`、`subtract_baseline`、`normalize`、`reinit`）——每个方法都会一次性同时作用于目标图谱的所有像素。

---

## 直接加载图谱数据 { #loading-map-data-directly }

若需在不创建 `MapsWorkspace` 的情况下进行数组级控制，可调用 `spectroview.api.io.load_map()` 直接返回图谱的原始 DataFrame（这也是 `MapsWorkspace.load_files()` 在内部调用的函数）：

```python
from spectroview.api import io
import numpy as np

# WDF, SPC, L6M, and NGC files return (DataFrame, metadata_dict); TXT/CSV return a DataFrame only.
result = io.load_map("my_wafer_map.wdf")
df_map, metadata = result if isinstance(result, tuple) else (result, {})
```

返回的 DataFrame 中，前两列始终为空间坐标（`X`、`Y`）；其余各列均为波数数值（以字符串形式表示）：

```python
x_coords = df_map["X"].to_numpy(dtype=float)
y_coords = df_map["Y"].to_numpy(dtype=float)

wn_cols  = [c for c in df_map.columns if c not in ("X", "Y")]
x_axis   = np.array([float(c) for c in wn_cols])
Y_matrix = df_map[wn_cols].to_numpy(dtype=float)
```

`Y_matrix` 可直接传递给 `spectroview.api.fitting.fit_batch()`——参见[光谱工作区：批量拟合](spectra.md#batch-fitting)。

---

## 多元分析 (MVA) { #multivariate-analysis-mva }

多元分析无需预先获知峰位即可探索大型复杂数据集。`spectroview.api.analysis` 提供了 PCA 和 NMF 算法，适用于任意 `(n_spectra, n_wavenumbers)` 矩阵——例如上文中的 `Y_matrix`，或从 `MapsWorkspace` 底层数据存储中获取的 `Y0`。

### 主成分分析 (PCA) { #principal-component-analysis-pca }

```python
from spectroview.api import analysis, preprocessing

Y_norm = preprocessing.normalize_spectra(Y_matrix)
pca_result = analysis.pca(Y_norm, n_components=3, center=True)

print("Explained Variance Ratios:", pca_result.explained_variance_ratio)
scores   = pca_result.scores     # (n_spectra, n_components)
loadings = pca_result.loadings   # (n_components, n_wavenumbers)
```

### 非负矩阵分解 (NMF) { #non-negative-matrix-factorization-nmf }

NMF 强制约束组分和得分均严格为正，因此通常能获得比 PCA 更具物理可解释性的“纯组分”光谱。

```python
nmf_result = analysis.nmf(Y_norm, n_components=3, max_iter=1000)
scores_w   = nmf_result.W
loadings_h = nmf_result.H
```

### 重构误差 { #reconstruction-error }

```python
errors = analysis.reconstruction_error(Y_norm, pca_result.scores, pca_result.loadings, pca_result.mean_spectrum)
# errors: (n_spectra,) per-spectrum L2 residual -- useful for flagging outlier spectra
```

### 导出 MVA 结果 { #exporting-mva-results }

```python
import pandas as pd

df_pca = pd.DataFrame({"X": x_coords, "Y": y_coords})
for i in range(pca_result.n_components):
    df_pca[f"PCA_Score_PC{i+1}"] = scores[:, i]

df_pca.to_csv("map_with_pca_scores.csv", index=False)
```

若要以与 `MapsWorkspace.get_heatmap()` 相同的方式将 `df_pca` 可视化为热图，可调用 `spectroview.model.spectra.heatmap.build_heatmap_grid(x_coords, y_coords, scores[:, 0], map_type="2Dmap")`。
