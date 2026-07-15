# SPECTROview Test Suite

Automated tests for the **Tensor Fit Engine** (`spectroview/fit_engine/`), the
**Spectra/Maps workspaces**, and the **Graphs workspace** (`spectroview/model/`,
`spectroview/viewmodel/`, and — for Graphs only — the plotting parts of
`spectroview/view/`), using pytest. See
[`GRAPHS_WORKSPACE_TESTING.md`](GRAPHS_WORKSPACE_TESTING.md) for a narrative
recap of how the Graphs workspace suite specifically was built against real
data, including the bugs it found. AI chat agent tests live under
`tests/unit/ai_agent/`; the update-checker and plot-template-store tests are
the remaining flat/adjacent files — both are out of scope for this document.

## Test Organization

```
tests/
├── conftest.py                        # Shared fixtures: QSettings isolation, qapp,
│                                       # example-data paths, synthetic data builders
├── unit/
│   ├── fit_engine/                    # Pure-numpy, no Qt: models, optimizer,
│   │   ├── test_models.py             #   evaluator, baseline, noise, engine
│   │   ├── test_scalar_models.py
│   │   ├── test_optimizer.py
│   │   ├── test_evaluator.py
│   │   ├── test_baseline.py
│   │   ├── test_noise.py
│   │   └── test_vbf_engine.py
│   ├── model/                         # SpectraStore, file I/O, settings, peak_model,
│   │   ├── test_spectra_store.py      #   MGraph, plot templates
│   │   ├── test_m_io.py
│   │   ├── test_peak_model.py
│   │   ├── test_m_settings.py
│   │   ├── test_workspace_io.py
│   │   ├── test_m_graph.py            #   Graphs workspace's plot-config model
│   │   └── test_m_plot_template.py
│   ├── viewmodel/                     # VMWorkspaceSpectra / VMWorkspaceMaps / VMWorkspaceGraphs
│   │   ├── test_vm_workspace_spectra.py
│   │   ├── test_vm_workspace_maps.py
│   │   └── test_vm_workspace_graphs.py
│   ├── view/                          # Deliberate, scoped exception to "View layer:
│   │   ├── test_v_graph_plotting.py   #   not tested" -- real headless matplotlib
│   │   └── test_customize_graph_dialog.py  # rendering for the Graphs workspace only
│   └── ai_agent/                      # AI chat agent (moved here from flat tests/*.py;
│       └── test_ai_agent_*.py         #   5 of 7 files require the optional `mcp` package)
├── integration/                       # End-to-end workflows across layers
│   ├── test_vbf_thread_integration.py #   VBFthread's 3 task-dict shapes + SpectraStore
│   ├── test_spectra_workflow.py       #   load -> ROI -> baseline -> peaks -> fit -> save/reload
│   ├── test_maps_workflow.py          #   same, for a real hyperspectral map
│   └── test_graphs_workflow.py        #   load Excel -> filter -> plot -> customize -> save/reload
└── performance/                       # Regression benchmarks on real datasets (@pytest.mark.slow)
    ├── conftest.py                    #   loads+fits each benchmark map twice per module
    ├── test_benchmark_cl_map.py       #   1_CL_map.txt/.json      (16384 spectra)
    ├── test_benchmark_mos2_map.py     #   2_MoS2_map.txt/.json    (1520 spectra, 3 peaks)
    └── test_memory_usage.py           #   tracemalloc peak-memory sanity checks
```

## Running Tests

```bash
# Everything except the slow performance/regression suite (recommended for local dev)
pytest tests/ -m "not slow"

# Everything, including the real-dataset performance benchmarks (~2 min)
pytest tests/

# Just one layer
pytest tests/unit/fit_engine/ -v
pytest tests/integration/ -v
pytest tests/performance/ -v

# A single test
pytest tests/unit/fit_engine/test_optimizer.py::TestBounds::test_solution_respects_tight_bounds -v

# With coverage
pytest tests/unit/ tests/integration/ --cov=spectroview --cov-report=html
```

On Windows, if you see `qt.qpa.plugin: Could not find the Qt platform plugin`, set:
```bash
set QT_QPA_PLATFORM=offscreen   # cmd
$env:QT_QPA_PLATFORM="offscreen" # PowerShell
```

## Design Principles

- **Deterministic**: `batched_levenberg_marquardt` has no randomness; every fit test
  either checks exact/near-exact reproducibility directly, or (for tests using
  synthetic noisy data) seeds `numpy.random.default_rng` explicitly.
- **No Qt dependency where none exists**: `spectra_store.py` and everything under
  `fit_engine/` (except `vbf_thread.py`, a `QThread`) are pure NumPy/pandas and are
  tested without a `QApplication`.
- **Real QThread, no event loop needed for assertions**: fitting goes through a real
  `VBFthread`. Tests call `.start()` then `thread.wait()` (a plain blocking join) to
  get deterministic completion without spinning a Qt event loop; `qapp.processEvents()`
  is used only when a test also needs the `finished`-signal side effects (e.g.
  `_on_fit_finished()` resetting `_is_fitting`). `tests/integration/test_vbf_thread_integration.py`
  instead calls `VBFthread.run()` directly (fully synchronous, no threading at all).
- **QSettings never touches the real user registry/plist**: `tests/conftest.py`'s
  autouse `_isolate_qsettings` fixture monkeypatches the `QSettings` name inside
  `spectroview.model.m_settings` to route every `QSettings(org, app)` construction
  through an explicit `QSettings(IniFormat, UserScope, org, app)` call pointed at a
  per-test temp directory. (Plain `QSettings.setDefaultFormat(IniFormat)` does **not**
  affect the 2-arg convenience constructor on this platform -- verified empirically;
  relying on it silently falls back to the native registry.) `tests/performance/conftest.py`
  has its own **module**-scoped variant of the same isolation, since the per-test
  one is function-scoped and would defeat the point of fitting each benchmark map
  only once per module.
- **Synthetic data with known ground truth**: `tests/conftest.py` provides
  `make_fit_model`, `make_synthetic_spectrum`, and `make_synthetic_map` factory
  fixtures that build a peak (or map of peaks) from explicit parameters, so fits
  can be checked against the exact values used to generate the data, not just
  "did it converge."
- **Real benchmark data for regression tests only**: `tests/performance/` is the only
  place that loads the full `1_CL_map.txt` (97MB, 16384 spectra) / `2_MoS2_map.txt`
  datasets and asserts against captured reference parameter values, convergence
  counts, R², timing budgets, and same-process reproducibility. Everything else in
  `tests/unit/` and `tests/integration/` uses small synthetic or lightweight example
  files so the non-`slow` suite stays fast — **except the Graphs workspace tests**
  (`tests/unit/view/`, `tests/unit/viewmodel/test_vm_workspace_graphs.py`,
  `tests/integration/test_graphs_workflow.py`), which deliberately load and plot the
  real `examples/datasets_for_plotting/dataset_Excel.xlsx` file throughout, since
  plot-style/customization correctness is best validated against real, messy,
  wafer-shaped data (duplicate coordinates, a `NaN` category, an uneven slot split)
  rather than a hand-crafted synthetic frame. See
  [`GRAPHS_WORKSPACE_TESTING.md`](GRAPHS_WORKSPACE_TESTING.md) for the full rationale
  and the three real bugs this approach found.

## Adding New Tests

- **New peak shape in `fit_engine/models.py`**: add it to `_SAMPLE_PARAMS` in
  `tests/unit/fit_engine/test_models.py` -- every existing parametrized test
  (scalar-parity, analytical-vs-numerical Jacobian, degenerate widths, 2D x-axis)
  picks it up automatically.
- **New ViewModel method**: add a test class to `tests/unit/viewmodel/`, using the
  `vm`/`settings`/`qapp` fixtures already in scope; prefer the `make_synthetic_*`
  factories over loading real files unless the behavior specifically depends on a
  real file format quirk.
- **New performance baseline**: after intentionally changing fit_engine numerics,
  re-run `tests/performance/` and update the `REFERENCE`/threshold constants at the
  top of the relevant `test_benchmark_*.py` -- don't just loosen tolerances blindly;
  confirm the new values are actually correct first.
- **New MGraph field / customize_graph_dialog.py control**: add it to `CUSTOM_VALUES`
  and `ALL_SAVE_KEYS` in `tests/unit/model/test_m_graph.py` (the round-trip test fails
  loudly if the two ever drift), then add an `_apply()`/read-back assertion to the
  matching `TestCustomize*` class in `tests/unit/view/test_customize_graph_dialog.py`.
  If the field is set on `VGraph.__init__` but has no `MGraph` counterpart, that's
  the same class of bug `scatter_edgecolor`/`axis_breaks` were — see
  [`GRAPHS_WORKSPACE_TESTING.md`](GRAPHS_WORKSPACE_TESTING.md).
- **New plot style**: add it to `spectroview.PLOT_STYLES` and the dispatch in
  `VGraph._plot_primary_axis`, then add a case to
  `TestPlotStyleCoverage` in `tests/unit/view/test_v_graph_plotting.py` — the
  `test_all_plot_styles_are_covered_by_this_test_class` test fails if the two lists
  diverge. Remember: `create_plot_widget(dpi)` before `plot(df)`, and never call
  `plot()` with an unhandled `plot_style` without first monkeypatching
  `QMessageBox.exec_`/`exec` (see that file's module docstring — it blocks forever
  otherwise, even headless).
