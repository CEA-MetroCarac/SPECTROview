# 数据可视化（图表） { #data-visualization-graphs }

`spectroview.api.graphs` 模块支持通过编程方式生成出版级高质量图表，为全部 9 种受支持的图表类型像素级复现 SPECTROview **图表** (Graphs) 工作区的原生视觉风格。该模块直接调用图形界面（GUI）中图表组件所使用的同一套底层渲染代码（无独立重实现，亦无近似简化），因此脚本绘制的图表与界面渲染结果完全一致——具有相同的颜色与标记循环序列、相同的 95% 置信区间（95% CI）计算方法，以及相同的箱线图、点图、趋势线、晶圆图和二维图谱样式。

---

## 准备数据 { #preparing-data }

绘图函数接收标准的 Pandas DataFrame 作为输入——通常为 `SpectraWorkspace.collect_results()` / `MapsWorkspace.collect_results()` 的输出结果，也可通过 `io.load_dataset()` 加载获得。

```python
import pandas as pd
from spectroview.api import io

df = io.load_dataset("fit_results.csv")["fit_results"]
print(df.columns)
```

每个 `plot_*` 函数均支持可选参数 `ax=`（直接在传入的坐标轴上绘图，而非创建新画布），并返回所绘制的 `Axes` 对象，便于您自行编排多子图布局。

---

## 分类与 XY 图表 { #categorical-xy-plots }

### 统计点图 { #statistical-point-plots }

按分类变量分组绘制均值与 95% 置信区间误差棒——非常适合对比不同实验条件下的峰参数。

```python
import matplotlib.pyplot as plt
from spectroview.api import graphs

fig, ax = plt.subplots(figsize=(8, 6))
graphs.plot_point(df, x="Sample", y="peak_1_x0", hue="Condition",
                   join=False, dodge=True, title="Peak Shift by Process Condition", ax=ax)
plt.tight_layout()
plt.savefig("point_plot.png", dpi=300)
```

### 折线图 { #line-plots }

与 `plot_point` 类似，但会用折线连接各 X 取值对应的均值，并附带阴影形式的 95% 置信区间带——适用于具有自然顺序的变量（如深度、时间、剂量等）。

```python
graphs.plot_line(df, x="Depth_nm", y="peak_1_ampli", hue="Condition", title="Amplitude vs Depth")
```

### 柱状图 { #bar-plots }

```python
graphs.plot_bar(df, x="Sample", y="peak_1_ampli", hue="Condition", show_error_bar=True)
```

### 箱线图 { #box-plots }

```python
graphs.plot_box(df, x="Sample", y="peak_1_fwhm", hue="Condition", title="FWHM Distribution Analysis")
```

### 直方图 { #histograms }

```python
graphs.plot_histogram(df, x="peak_1_fwhm", hue="Condition", bins=20, kde=True)
```

### 散点图与趋势线 { #scatter-plots-trendlines }

```python
graphs.plot_scatter(df, x="peak_1_ampli", y="peak_2_ampli", hue="Condition",
                     title="Peak 1 vs Peak 2 Amplitude Correlation")

graphs.plot_trendline(df, x="peak_1_x0", y="peak_1_fwhm", order=1, hue="Condition",
                       title="Correlation: FWHM vs Peak Position")
```

---

## 空间图表 { #spatial-plots }

此类函数接收规整（tidy）格式的 DataFrame：每一行对应一个 (x, y) 采样点，并包含一个数值列 `z`——这与 `MapsWorkspace.collect_results()` 输出的数据结构完全一致。

### 二维图谱 { #2d-maps }

基于采集数据自身的规则网格（光栅扫描 / 二维图谱采集），绘制快速、精确的热图。

```python
graphs.plot_2dmap(df, x="X", y="Y", z="ampli_Si", cmap="jet", title="Amplitude Map")
```

### 晶圆图 { #wafer-maps }

在圆形晶圆轮廓内绘制插值热图，并支持在图上标注汇总统计信息。

```python
graphs.plot_wafer(df, x="X", y="Y", z="ampli_Si", wafer_size=300.0, show_stats=True)
```

---

## 绘图配方 { #plot-recipes }

绘图配方（*plot recipe*）是一组已命名、可复用的图表配置集合（每项配置的结构均与图形界面中 `MGraph.save()` 生成的字典相同），并以 JSON 文件形式保存在文件夹中——这与图形界面中的 **保存绘图配方** (Save Plot Recipe) 功能采用完全相同的机制。（注意不要与*样式模板*混淆——样式模板仅保存外观样式，不包含数据绑定，详见 `spectroview/model/graphs/m_style_template_store.py`。）

```python
from spectroview.api import graphs

configs = [
    {"plot_style": "scatter", "x": "peak_1_ampli", "y": ["peak_2_ampli"], "z": "Condition"},
    {"plot_style": "box", "x": "Sample", "y": ["peak_1_fwhm"], "z": "Condition"},
]
recipe_id = graphs.save_plot_recipe("./recipes", "Standard QC Plots", configs)

# List and reload later
for summary in graphs.list_plot_recipes("./recipes"):
    print(summary["name"], summary["graph_count"])

configs = graphs.load_plot_recipe("./recipes", recipe_id)
graphs.delete_plot_recipe("./recipes", recipe_id)
```
