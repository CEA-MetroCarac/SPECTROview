# **拉曼光谱数据的多元分析 (MVA)** { #multivariate-analysis-mva-of-raman-spectroscopic-data }

本文档提供 `SPECTROview` 中所实现多元分析 (MVA) 功能的技术概览。

## **架构** { #architecture }

MVA 功能遵循 MVVM (Model-View-ViewModel) 架构模式：

- **Model** (`spectroview/model/analysis/m_mva.py`)：包含 MVA 技术的核心数学实现，具体包括主成分分析 (PCA) 和非负矩阵分解 (NMF)。仅使用标准数值计算库（`numpy`、`scipy`），无需引入 `scikit-learn` 等额外依赖。
- **ViewModel** (`spectroview/viewmodel/analysis/vm_mva.py`)：负责编排数据流。它从激活光谱中提取经过预处理的数据，调用 Model 执行分解计算，缓存计算结果，并计算逐条光谱的重构误差。
- **View** (`spectroview/view/components/lower_tabs/`: `v_mva.py`, `v_mva_plots.py`)：提供用户界面，包含分页绘图面板（**摘要** (Summary)、**碎石图** (Scree Plot)、**载荷 (Loadings)**、**热图** (Heatmap)、**得分 (Scores)**、**残差** (Residuals)）、交互式参数控件，以及每个图表各自的 `matplotlib` 导航工具栏。

## **算法实现** { #algorithm-implementation }

`SPECTROview` 中的 PCA 实现基于对均值中心化数据的奇异值分解 (SVD)。

### **可视化增强** { #visualization-enhancements }

`SPECTROview` 提供了功能完善的分页图表界面来可视化分析结果：

- **摘要**：2×2 仪表盘，同屏展示碎石图（仅 PCA）、前 3 个载荷以及主要得分的二维散点投影
- **碎石图**（仅 PCA）：各主成分方差贡献率柱状图 + 累计方差曲线
- **载荷 (Loadings)**：叠加显示各组分的光谱载荷轮廓，并采用曲线下填充（fill-between）增强视觉清晰度
- **热图**：当光谱名称中带有 `(x, y)` 坐标时，绘制所选组分得分的二维空间分布图
- **得分 (Scores)**：支持交互式选择坐标轴的二维散点图
- **残差**：逐条光谱的重构误差，用于评估模型质量

## **已实现的方法** { #implemented-methods }

### **1. 主成分分析 (PCA)** { #1-principal-component-analysis-pca }

PCA 是一种降维技术，它通过正交变换将光谱数据映射到新的坐标系中，使数据在任意投影方向上的最大方差落在第一坐标轴（即第一主成分）上。

- **实现细节**：基于奇异值分解（`numpy.linalg.svd`）实现，支持对数据进行可选的均值中心化预处理。
- **参数**：
  - **主成分/组分数** (Components)：要保留的主成分数量（2–50）。
  - **均值中心化** (Mean centering)：是否在 SVD 前减去平均光谱（默认：开启）。这是光谱 PCA 分析的标准做法。
- **可视化图表**：
  - *摘要*：2×2 概览仪表盘，组合展示碎石图、前几个载荷及主要得分投影对。
  - *碎石图*：展示每个主成分所解释的方差百分比及累计方差贡献率。
  - *载荷 (Loadings)*：可视化各主成分的光谱特征贡献，并在图例中标注方差解释率百分比。
  - *热图*：当光谱名称中包含 `(x, y)` 坐标时，绘制可选主成分的二维空间得分分布图。
  - *得分 (Scores)*：展示光谱投影到主成分空间的二维散点图，支持交互式切换坐标轴。
  - *残差*：逐条光谱的 L2 重构误差，用于评估模型拟合质量。

### **2. 非负矩阵分解 (NMF)** { #2-non-negative-matrix-factorization-nmf }

NMF 将数据矩阵分解为两个非负矩阵（通常记作 W 和 H）。该方法尤其适用于光谱数据，因为它能够分解出可加的非负组分，这些组分通常对应于具有明确物理意义的纯物质组分（端元，endmembers）。

- **实现细节**：采用 Lee & Seung 乘法更新规则，通过迭代更新分解矩阵来最小化重构误差的 Frobenius 范数。
- **参数**：
  - **主成分/组分数**：从数据集中提取的非负组分（或端元）数量。
  - **最大迭代次数** (Max Iterations)：允许算法执行的最大乘法更新迭代次数。若算法在达到该上限前尚未收敛，将自动停止。
  - **容差** (Tolerance)：收敛判定阈值。若相邻两次迭代间重构误差的相对变化低于此数值，算法将提前停止。
  - **随机种子** (Random Seed)：用于初始化矩阵的随机数生成器种子。设置固定种子可保证不同次运行间的 NMF 结果完全可复现（默认：42）。
- **可视化图表**：
  - *摘要*：NMF 载荷及主要得分投影的概览仪表盘。
  - *载荷 (Loadings)*：展示非负光谱组分（端元光谱）。
  - *热图*：当光谱名称中包含 `(x, y)` 坐标时，绘制可选 NMF 组分的二维空间丰度分布图。
  - *得分 (Scores)*：展示各组分在整个数据集中的浓度/丰度分布二维散点图。
  - *残差*：逐条光谱的 L2 重构误差。

> **为什么 NMF 没有碎石图？**  
> 与严格按最大方差解释率对正交主成分进行排序的 PCA 不同，NMF 提取的是相互混合的非正交、可加组分（端元）。由于 NMF 组分既不正交，也不按方差大小依次排列，因此无法为单个组分计算孤立的“方差解释率”百分比。要评估 NMF 模型，请借助**残差**图来判断重构质量，并通过**载荷 (Loadings)** 图确认所提取的组分是否对应于具有物理意义的光谱。

## **用户界面** { #user-interface }

MVA 视图采用左右分栏布局：

### **左侧面板（控制区）** { #left-panel-controls }

- **方法选择**：用于在 `PCA` 与 `NMF` 之间切换的单选按钮。
- **参数分组**：分别为 PCA 和 NMF 参数设立的独立分组框，仅显示当前所选方法对应的相关参数。
- **运行按钮**：执行分析。
- **导出区域**：将得分发送至 `Graphs` 工作区。

### **右侧面板（分页图表区）** { #right-panel-tabbed-plots }

每种图表类型在 `QTabWidget` 中各自占据一个独立标签页：

- **摘要**：组合概览网格（碎石图 [PCA]、前 3 个载荷及主要得分的散点投影）。
- **碎石图**（仅 PCA）：各主成分解释的方差。
- **载荷 (Loadings)**：光谱载荷轮廓曲线。
- **热图**：所选组分得分的二维空间分布图（要求光谱名称中包含 `(x, y)` 坐标）。
- **得分 (Scores)**：支持坐标轴选择的交互式二维散点图。
- **残差**：逐条光谱的重构质量。

每个标签页均内置了 `matplotlib` 的 `NavigationToolbar`，支持缩放、平移和图像保存操作。

## **数据处理流程** { #data-pipeline }

1. **数据选择**：MVA 仅对 `SPECTROview` 工作区中处于*激活*（已勾选）状态的光谱进行计算。
2. **预处理**：`VMMVA` 直接从 `SpectraStore` 读取数据（存在预处理数据时读取 `md.Y`，否则回退至原始数据 `md.Y0`），这意味着它处理的是已经过用户配置的预处理步骤（`baseline` 扣除、裁剪、X 轴校正）后的数据。
3. **插值对齐**：如果所选光谱的波数轴不尽相同，系统会自动将它们插值到统一网格（即第一条激活光谱的波数网格）上，以构建维度一致的数据矩阵。
4. **结果导出**：分析结果（PCA 的得分或 NMF 的 W 矩阵）可直接导出至 `Graphs` 工作区，以便开展进一步可视化与对比分析。

## **引用与参考文献** { #citations-and-references }

如果您在科研论文中使用了上述功能生成分析结果，恳请引用所实现算法对应的相关文献：

[1] **PCA 算法**：Wold, S., Esbensen, K., and Geladi, P., "Principal component analysis." *Chemometrics and intelligent laboratory systems* 2.1-3 (1987): 37-52.

[2] **NMF 算法**：Lee, Daniel D., and H. Sebastian Seung, "Learning the parts of objects by non-negative matrix factorization." *Nature* 401.6755 (1999): 788-791.
