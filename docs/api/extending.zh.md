# 扩展 SPECTROview { #extending-spectroview }

SPECTROview 没有插件/入口点（entry-point）框架——应用中不存在用于挂载自定义代码的注册表。相反，扩展能力通过 API 直接提供的两种基于文件的机制实现。如果您想了解“如何向 SPECTROview 添加自定义的 X”，答案几乎总是这两种机制之一。

---

## 自定义基线方法 { #custom-baseline-methods }

基线计算（`spectroview.api.preprocessing.subtract_baseline` 以及各工作区的 `set_baseline`）按 `config["mode"]` 中的字符串名称进行分发。除了内置的 `None`、`"Linear"`、`"Polynomial"` 和 `"arpls"` 模式外，**来自 [pybaselines](https://pybaselines.readthedocs.io/) 的任意 `Baseline` 方法名称均可开箱即用**——无需任何注册步骤：

```python
from spectroview.api import preprocessing

# "asls", "airpls", "modpoly", "snip", ... any pybaselines.Baseline method name
Y_corrected, Y_baseline = preprocessing.subtract_baseline(x, Y, {"mode": "asls", "lam": 1e6})
```

如果 pybaselines 在未来版本中新增了方法，双方均无需修改代码，该方法即可立即在 SPECTROview 中使用。

---

## 拟合模型与图表模板 { #fit-model-plot-templates }

拟合模型和图表配置都是存放在由您管理的文件夹中的纯 JSON 文件——SPECTROview（GUI 或 API）只需扫描该文件夹即可读取。这意味着任何外部工具、脚本或受版本控制的模板仓库都可以向其中写入模板：

```python
from spectroview.api import fitting, graphs, settings

# Fit-model templates live under the SPECTROview Working Folder. Point it at a
# shared location — set_working_folder() creates fit_model/plot_recipe/plot_style/ai_chat_history/template
# subfolders and config.json under it:
settings.set_working_folder("./shared_spectroview")
# ...then any *.json dropped into the fit_model folder (by this API, the GUI, or
# hand-written) is immediately visible:
names = fitting.list_fit_model_templates(settings.get_fit_model_folder())

# Plot recipes work the same way, via a folder-backed store:
graphs.save_plot_recipe("./shared_spectroview/plot_recipe", "QC Dashboard", configs=[...])
```

拟合模型模板的 JSON 结构为 `{"0": {"peak_labels": [...], "peak_models": {...}, "baseline": {...}, "range_min": ..., "range_max": ..., "fit_params": {...}}}`——如果您从外部系统生成模板，可直接按此格式编写；也可以使用 `fitting.build_fit_model()` / `fitting.save_fit_model_template()` 构建模板并查看输出内容。

---

## 自定义峰型 { #custom-peak-shapes }

批量峰型注册表（`spectroview.fit_engine.models.BATCHED_MODELS`）是一个普通的 Python 字典，将峰型名称映射到 `(eval_fn, jacobian_fn, param_names)`。它不属于公共 API（跨版本间不提供稳定性保证），但如果您需要 SPECTROview 尚未内置的峰型，就必须将其添加到该注册表中——同时还需为非批量回退路径添加到 `spectroview.fit_engine.scalar_models.PEAK_MODEL_REGISTRY`。这需要修改源码，而非运行时扩展；目前尚无法在不修改已安装软件包的情况下通过脚本注册新的峰型。
