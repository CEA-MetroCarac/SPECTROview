# API { #api }

SPECTROview Python API 可让您自动化分析工作流，并以编程方式调用 SPECTROview 的核心光谱处理引擎。它与 Qt 图形界面完全解耦（无需创建 `QApplication`），非常适合用于 Jupyter Notebook、批处理脚本以及 CI 流水线。

所有模块均通过 `spectroview.api` 包对外提供。

## API 模块 { #api-modules }

- **[`spectroview.api.workspace`](spectra.md#stateful-workspaces)** — `SpectraWorkspace` / `MapsWorkspace`：有状态会话，将加载、预处理、拟合、结果汇总以及保存/重新加载集中于一处。其读写的文件采用与图形界面自身的 **保存工作区** (Save work) 功能*完全相同的格式*——完全通过脚本构建的工作区可直接在图形界面中打开，反之亦然。
- **[`spectroview.api.io`](spectra.md#loading-data)** — 文件读取器（WDF、SPC、L6S、L6M、NGS、NGC、CSV、TXT、DAT）、数据集导入、结果导出，以及无图形界面（headless）的 Renishaw 二维图谱转换器。
- **[`spectroview.api.preprocessing`](spectra.md#preprocessing)** — 数组级别的无状态操作：基线扣除、波段裁剪与归一化。
- **[`spectroview.api.fitting`](spectra.md#batch-fitting)** — 向量化批量拟合 (VBF) 引擎接口：模型构建、批量拟合以及拟合模型模板的增删改查 (CRUD) 操作。
- **[`spectroview.api.analysis`](2dmap.md#multivariate-analysis-mva)** — 多元分析：主成分分析 (PCA) 与非负矩阵分解 (NMF)。
- **[`spectroview.api.graphs`](graphs.md)** — 发表级图表绘制，与图形界面自身的渲染效果逐像素完全一致，支持全部 9 种图表类型，并提供绘图模板的增删改查 (CRUD) 操作。
- **[`spectroview.api.calculators`](calculators.md)** — 用于光学参数计算与单位换算的物理计算器。
- **`spectroview.api.settings`** — 对持久化配置进行无图形界面读写访问（默认拟合参数、模型目录、上次使用目录，以及通过 `get_language()` / `set_language()` 控制界面语言）。
- **`spectroview.api.exceptions`** — 整个包中抛出的 `SpectroviewError` 异常层次结构（`LoadError`、`FitModelError`、`FitError`、`WorkspaceError`、`TemplateError`）。

## 探索 API { #exploring-the-api }

- [光谱工作区](spectra.md)：加载离散光谱、执行预处理并运行批量拟合引擎——同时支持有状态的 `SpectraWorkspace` 方式与基于底层数组的方式。
- [二维图谱工作区与二维分析](2dmap.md)：加载高光谱图谱或晶圆图、对每个像素进行拟合、生成热图与线扫描剖面，以及运行 PCA/NMF 分析。
- [数据可视化（图表）](graphs.md)：在脚本中复现 SPECTROview 原生图表（支持全部图表类型），并使用可复用的绘图模板。
- [快捷计算器](calculators.md)：以编程方式调用光学参数与光谱单位换算工具。
- [扩展 SPECTROview](extending.md)：实际可用的文件级扩展点（自定义基线方法、拟合模型/绘图模板文件夹）——本软件不提供插件框架，本页将说明实际可用的替代方案。

## 错误处理 { #error-handling }

发生错误时，`spectroview.api` 中的每个函数都会抛出 `spectroview.api.exceptions.SpectroviewError` 的子类异常——绝不会直接抛出底层模型层的原始 `KeyError`/`ValueError`，也不会弹出 Qt 对话框（否则会导致无图形界面脚本挂起）。您可以宽泛地捕获 `SpectroviewError`，也可以精准捕获具体的子类异常（`LoadError`、`FitModelError`、`FitError`、`WorkspaceError`、`TemplateError`）：

```python
from spectroview.api import io
from spectroview.api.exceptions import LoadError

try:
    data = io.load_spectra("missing_file.wdf")
except LoadError as e:
    print(f"Could not load spectra: {e}")
```
