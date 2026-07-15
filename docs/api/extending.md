# Extending SPECTROview

SPECTROview does not have a plugin/entry-point framework — there is no registry to hook custom code into the application. Instead, extensibility works through two file-based mechanisms that the API exposes directly. If you're looking for "how do I add a custom X to SPECTROview," one of these is almost always the answer.

---

## Custom Baseline Methods

Baseline evaluation (`spectroview.api.preprocessing.subtract_baseline`, and every workspace's `set_baseline`) dispatches on `config["mode"]` by string name. Beyond the built-in `None`, `"Linear"`, `"Polynomial"`, and `"arpls"` modes, **any [pybaselines](https://pybaselines.readthedocs.io/) `Baseline` method name works out of the box** — no registration step:

```python
from spectroview.api import preprocessing

# "asls", "airpls", "modpoly", "snip", ... any pybaselines.Baseline method name
Y_corrected, Y_baseline = preprocessing.subtract_baseline(x, Y, {"mode": "asls", "lam": 1e6})
```

If pybaselines adds a new method in a future release, it becomes available to SPECTROview immediately with no code change on either side.

---

## Fit-Model & Plot Templates

Both fit models and plot configurations are plain JSON files in a folder you control — SPECTROview (GUI or API) just scans that folder. This means any external tool, script, or version-controlled repository of templates can populate them:

```python
from spectroview.api import fitting, graphs, settings

# Fit-model templates: point SPECTROview at a shared folder...
settings.set_model_folder("./shared_fit_models")
# ...then any *.json file dropped there (by this API, the GUI, or hand-written)
# is immediately visible:
names = fitting.list_fit_model_templates(settings.get_model_folder())

# Plot templates work the same way, via a folder-backed store:
graphs.save_plot_template("./shared_plot_templates", "QC Dashboard", configs=[...])
```

A fit-model template's JSON shape is `{"0": {"peak_labels": [...], "peak_models": {...}, "baseline": {...}, "range_min": ..., "range_max": ..., "fit_params": {...}}}` — write one directly if you're generating templates from an external system, or build one with `fitting.build_fit_model()` / `fitting.save_fit_model_template()` and inspect the output.

---

## Custom Peak Shapes

The batched peak-shape registry (`spectroview.fit_engine.models.BATCHED_MODELS`) is a plain Python dict mapping shape name to `(eval_fn, jacobian_fn, param_names)`. It is not part of the public API surface (no stability guarantee across releases), but if you need a shape SPECTROview doesn't ship, that registry — plus `spectroview.fit_engine.scalar_models.PEAK_MODEL_REGISTRY` for the non-batched fallback path — is where it would need to be added. This requires a source change, not a runtime extension; there is currently no way to register a new peak shape from a script without modifying the installed package.
