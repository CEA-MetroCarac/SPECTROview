# Graphs Workspace Testing — Recap

This is a narrative companion to [`README.md`](README.md), focused specifically
on how the Graphs workspace test suite was built and why it leans so heavily
on one real data file instead of synthetic fixtures. If you're looking for
"how do I run the tests" or "where does file X live," start with `README.md`
— this document explains the *reasoning* behind the Graphs-specific parts of
it, and records the three real bugs the approach found.

## Scope

"Graphs workspace" here means:

- `spectroview/model/m_graph.py` — `MGraph`, the serializable plot-config model.
- `spectroview/viewmodel/vm_workspace_graphs.py` — `VMWorkspaceGraphs`, DataFrame
  management, filtering, graph CRUD, workspace save/load.
- `spectroview/view/components/v_graph.py` — `VGraph`, the widget that actually
  renders a plot with matplotlib.
- `spectroview/view/components/v_plot_renderer.py` — `PlotRenderer`/`WaferPlot`,
  the per-style rendering logic `VGraph` delegates to.
- `spectroview/view/components/customize_graph/` — the "Customize Graph"
  dialog (`CustomizeGraphDialog`, `CustomizeLegend`, `CustomizeAxis`,
  `CustomizeMoreOptions`, `CustomizeAnnotations`), i.e. every control the
  user has for tweaking a plot after it's created.
- `spectroview/model/m_plot_template.py` / `m_plot_template_store.py` — reusable
  saved plot configs, adjacent to Graphs; its existing test file was migrated
  into the new directory layout unchanged (it was already solid).

## Why real data, not synthetic fixtures

Every other part of this test suite (fit_engine, Spectra, Maps) leans on small
synthetic data with known ground truth, because the thing under test is
*numerical correctness* — does the optimizer recover the parameters that
generated the data. The Graphs workspace is different: the thing under test is
*does this plot style, with this data shape, render the way a user's actual
data would render*. A hand-built 5-row DataFrame with clean, non-repeating
values can't exercise the failure modes that matter here — duplicate
coordinates that break a pivot, a `NaN` category that has to flow through a
`groupby`, an uneven number of points per group.

So the Graphs suite instead loads the real, checked-in
`examples/datasets_for_plotting/dataset_Excel.xlsx` — an actual wafer-mapping
Si-peak fit-results export — and plots real slices of it. Concretely, that
file is two sheets:

- **`sheet1`** — 588 rows: `X`, `Y` (wafer stage coordinates), `x0_Si`,
  `ampli_Si`, `area_Si`, `fwhm_Si` (Si-peak fit parameters — the same kind of
  output the Maps workspace's Tensor Fit Engine produces), `Quadrant`
  (`Q1`–`Q4`, plus one `NaN` for the center point that falls in no quadrant),
  `Zone` (`Center` / `Mid-Radius` / `Edge`), `Slot` (11 distinct wafer slots,
  `2`–`12`, 49 points each except slot 2 which has 98), `DeltaW`,
  `Strain (GPa)`, `NB pts`.
- **`sheet2`** — 588 rows, similar columns but **no `Slot` column** — used
  specifically to exercise the "this sheet doesn't support wafer-slot
  filtering" code paths (`has_slot_column`, `get_unique_slots`) against a
  sheet where that's genuinely true, not mocked.

One consequence of using the real file: **534 of `sheet1`'s 588 rows share an
`(X, Y)` pair with another row** (the same physical grid point, reused across
different slots). That's exactly the condition that makes a `2Dmap` pivot
raise `ValueError: Index contains duplicate entries, cannot reshape` — a
failure mode a synthetic non-repeating grid would never have surfaced. The
2Dmap happy-path test deduplicates on `(X, Y)` first (as a real workflow
would need to), and a dedicated test feeds the *un-deduplicated* data in on
purpose to confirm that failure is real, reproducible, and raised as a clear
`ValueError` rather than silently corrupting the plot.

## Methodology: headless rendering is a deliberate, scoped exception

The rest of this test suite follows a "View layer: not tested" policy (Qt
widgets are hard to test meaningfully and the payoff is low). Graphs breaks
that rule on purpose, because "test all customization ability... testing to
plot all features" is an explicit request that can't be satisfied by
inspecting config dicts alone — it requires actually calling matplotlib and
looking at what got drawn.

This works headlessly (`QT_QPA_PLATFORM=offscreen`, a real `QApplication` via
the shared `qapp` fixture) because `VGraph` never blocks on a real display for
the paths this suite drives. Two non-obvious rules had to be established
empirically before any of this was safe to automate:

1. **`create_plot_widget(dpi)` must be called before `plot(df)`.** `self.ax`
   is `None` until then, and `plot()` unconditionally calls `self.ax.clear()`
   — skip this and every test in the file would `AttributeError` identically,
   which is exactly what happened on the first draft.
2. **An unsupported `plot_style` hangs forever, even offscreen.** It routes
   through `show_alert()` → `QMessageBox.exec_()`, a blocking modal that waits
   for a click that will never come under `pytest`. The one test that
   exercises this path (`test_unsupported_style_shows_alert_without_hanging`)
   monkeypatches `QMessageBox.exec_`/`exec` *before* calling `plot()`. Every
   other test only ever uses real `PLOT_STYLES` values, so this never comes
   up by accident — but if you add a new negative-path test involving
   `v_workspace_graphs.py`'s many other `QMessageBox.critical/warning` calls
   (there are ~10+, e.g. on a plot-render exception), mock those first too.

Every `VGraph` test builds and plots exactly one widget per test function.
`create_plot_widget()` calls `plt.close('all')` internally (global pyplot
state), which would be a real cross-test hazard if two `VGraph` instances
were expected to stay alive and interact across a test boundary — this
suite's tests never do that, so it hasn't mattered in practice, but it's
worth knowing if you're tempted to keep two widgets around in one test.

## What's covered

- **`tests/unit/model/test_m_graph.py`** (39 tests) — every one of `MGraph`'s
  ~70 fields, individually and via one exhaustive "customize everything, then
  save→load, then assert every field survived" round-trip test
  (`test_save_load_roundtrip_every_customizable_field`). A companion test
  (`test_fresh_instance_save_never_raises`) exists specifically so
  `__init__`/`save()` can never silently drift apart again.
- **`tests/unit/viewmodel/test_vm_workspace_graphs.py`** (57 tests) — loading
  the real multi-sheet Excel file, per-sheet DataFrame naming, add/remove/
  select/refresh/save DataFrame, `apply_filters()` with real `Slot`/`Zone`
  expressions (including an intentionally invalid expression to confirm the
  error path doesn't crash), `has_slot_column`/`get_unique_slots` on both a
  sheet that has one and one that doesn't, `create_multi_wafer_graphs`
  (one graph per slot, correct filter merging, no duplicate `Slot ==` filters),
  graph CRUD, and full workspace save/reload (including the legacy JSON
  format) with real data and real customizations round-tripping.
- **`tests/unit/view/test_v_graph_plotting.py`** (36 tests) — one real-data
  render per entry in `PLOT_STYLES` (a test asserts the two lists can't
  diverge), hue-grouping/legend behavior, every axis customization
  (limits — including exactly `0.0`, log scale, breaks), scatter-specific
  customization (marker size, edge color), histogram customization (bin
  count, KDE overlay), trendline customization (polynomial order, anchor
  point), and eight distinct, real, uncaught-exception failure modes
  (missing column → `KeyError`, non-numeric trendline column → `ValueError`
  with a specific message, empty trendline data, wafer/2Dmap without a `z`
  column, duplicate 2Dmap coordinates, plotting before
  `create_plot_widget()`).
- **`tests/unit/view/test_customize_graph_dialog.py`** (35 tests) — the real
  `CustomizeLegend`/`CustomizeAxis`/`CustomizeMoreOptions`/
  `CustomizeAnnotations` widgets bound to a real, already-plotted `VGraph`:
  loading current settings into the controls, applying changes back onto the
  graph, the `properties_changed` signal payload, and the top-level
  `CustomizeGraphDialog`'s "Apply All"/"switch to a different graph" behavior.
- **`tests/integration/test_graphs_workflow.py`** (2 tests) — the full,
  realistic pipeline in one test: load the real Excel file → select a sheet →
  filter to one wafer slot → create and render a wafer plot → create a second
  scatter plot and customize it through the *real* dialog widgets (marker
  size, edge color, an axis break) → batch-create per-slot wafer plots →
  save the workspace → reload it into a fresh `VMWorkspaceGraphs` → confirm
  every DataFrame, filter, and customization survived, and that the reloaded
  graph still renders. A second test exercises `sheet2` (no `Slot` column) end
  to end with a `trendline` plot.

## Three real bugs found and fixed

Writing tests that assert "customize X, save, reload, X is still set" against
the *actual* `MGraph`/`VGraph` code (not a description of what they're
supposed to do) surfaced three genuine, pre-existing defects. All three were
minimal, additive, backward-compatible fixes, applied and re-verified against
the full suite (573 passed / 0 failed after each):

1. **`scatter_edgecolor` never persisted.** `VGraph.__init__` set
   `self.scatter_edgecolor = 'black'` and the Customize dialog read/wrote it,
   but `MGraph` — the thing that actually gets serialized to a `.graphs` file
   — never had this field at all. Every scatter/point/trendline marker
   edge-color customization was silently discarded the moment you saved and
   reloaded a workspace. This is also why `test_m_graph.py`'s
   `test_default_properties` and `test_save_load_roundtrip` were already
   failing before this session (`AttributeError: 'MGraph' object has no
   attribute 'scatter_edgecolor'`, tracked as a known pre-existing failure in
   `docs/log.md`) — that old test assumed the field existed. **Fix:** added
   `scatter_edgecolor` to `MGraph.__init__` (default `"black"`, matching
   `VGraph`) and to `save()`.
2. **`axis_breaks` (the "Broken axis (beta)" feature) never persisted.**
   Same shape of bug: `VGraph.axis_breaks` is a real, mutable feature — set
   via the Axis tab's "from:"/"to:" fields — that flows into the
   `properties_changed` signal payload, but `MGraph` never initialized or
   serialized it. Configuring an axis break, then saving and reloading,
   silently reset it to nothing. **Fix:** added `axis_breaks` to
   `MGraph.__init__` (default `{'x': None, 'y': None}`, matching `VGraph`)
   and to `save()`. `load()` needed no change — its generic
   `if hasattr(self, key): setattr(...)` loop already handles new dict-valued
   fields correctly once `__init__` declares them.
3. **Axis limits of exactly `0.0` were silently ignored.**
   `VGraph._set_limits()` guarded every limit pair with a *truthy* check
   (`if self.ymin and self.ymax:`), not an `is not None` check. `0.0` is
   falsy in Python, so setting a Y-axis minimum to exactly zero — a
   thoroughly ordinary thing to want, e.g. "start intensity at zero" — was
   silently dropped; the axis kept auto-scaling instead. Found by a test that
   simply happened to pick `ymin=0.0` as a normal test value and got the
   auto-scaled range back instead. **Fix:** all five limit-pair checks in
   `_set_limits()` (`x`, `y`, `y2`, `y3`, `x2`) now use explicit
   `is not None` checks.

None of the three required changing what any field *means* — only whether it
round-trips or applies correctly — so no existing `.graphs` file or workflow
should behave differently except that these customizations now actually work.

## Running just this part of the suite

```bash
# Everything Graphs-related
pytest tests/unit/model/test_m_graph.py tests/unit/model/test_m_plot_template.py \
       tests/unit/viewmodel/test_vm_workspace_graphs.py \
       tests/unit/view/ \
       tests/integration/test_graphs_workflow.py -v

# Just the real-data plotting/customization layer (the slowest part, ~60s,
# dominated by re-reading the Excel file and re-rendering per test)
pytest tests/unit/view/ -v
```

None of the Graphs tests are marked `slow` — they're all fast enough (~60s
total) to run in the default `pytest tests/ -m "not slow"` loop.

## Known limitations — what this suite does *not* cover

- **Pixel-level rendering correctness.** Tests assert *structural* facts
  (an axes has N collections/patches/lines, a legend has the right number of
  entries, a limit equals the value that was set) never pixel colors or exact
  layout — matplotlib's own test suite already covers rendering correctness;
  duplicating that here would be brittle and low-value.
  `test_scatter_edgecolor_applied` asserts an edge color array was set on the
  drawn path collection, not what color it visually is.
  `test_kde_overlay_adds_a_line` and `test_bin_count_changes_patch_count`
  are similarly structural, not pixel-exact.
- **`v_workspace_graphs.py` itself is not directly tested** — only the
  `VGraph`/`MGraph`/`VMWorkspaceGraphs`/`CustomizeGraphDialog` objects it
  wires together. It's a ~1800-line `QMainWindow`-style coordinator (menus,
  MDI subwindows, drag-and-drop, the wafer-slot checkbox list); the
  integration test reproduces its actual object-wiring pattern
  (`VGraph(...)` → copy `MGraph` fields onto it → `create_plot_widget()` →
  `plot(df)` → `properties_changed` → `vm.update_graph()`) directly, which
  exercises the same code paths without needing the full window.
- **`CustomizeAnnotations`' edit flow is only partially covered.** Add/delete
  are tested directly; editing an existing annotation opens a modal
  (`EditLineDialog`/`EditTextDialog` via `.exec()`), which this suite doesn't
  drive interactively — `EditLineDialog`/`EditTextDialog`'s own
  `get_properties()` logic is simple enough that a future test could
  instantiate them standalone and call `get_properties()` without `.exec()`
  if that coverage becomes worth adding.
- **`QColorDialog`/native color-picker interaction is not simulated** —
  tests set button text/state directly (`_set_color_button(...)`) rather than
  driving the OS color picker, which isn't meaningfully headless-testable
  anyway.
