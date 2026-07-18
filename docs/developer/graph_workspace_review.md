# Graph Workspace — Complete Review & Redesign Proposal

*Review date: 2026-07-18. Scope: `view/v_workspace_graphs.py` (1823 lines),
`viewmodel/vm_workspace_graphs.py` (480), `view/components/v_graph.py` (1191),
`view/components/v_plot_renderer.py` (732), `model/m_graph.py` (151), and the
`view/components/customize_graph/` package (~1400 lines across 5 tabs/dialogs).*

---

## 1. Architecture review

### 1.1 What is good

- The recent refactors show the right instincts: the customize dialog was split
  into a per-tab package, `_build_graph_widget()` deduplicated the
  create/replicate/load paths, `_apply_limit_pair()` centralised limit handling,
  and `_configure_graph_from_model()` documents and fixes a real `axis_breaks` bug.
- The workspace save format (ZIP + pickle via `WorkspaceIO`, with a legacy JSON
  fallback) is sound and versioned (`format_version: 3`).
- The template store (`MPlotTemplateStore`) and the multi-wafer slot workflow are
  genuinely useful domain features not found in generic plotting tools.
- The MDI area + singleton customize dialog that follows the active subwindow is
  a good UX pattern.

### 1.2 The central defect: three copies of every graph property

The same ~60 properties exist in **three places**:

1. `MGraph` (the model),
2. `VGraph` — the widget re-declares *every* model field as instance attributes
   in its `__init__` (lines 40–143 of `v_graph.py` mirror `m_graph.py` almost 1:1),
3. the GUI controls (side panel + customize dialog tabs).

Synchronisation between them is entirely manual and scattered:

- model → widget: `_configure_graph_from_model()` (generic `vars()` copy + ad-hoc
  deep-copy overrides for *some* mutable fields),
- widget → model: hand-maintained field lists in `_on_update_plot()` and
  `save_workspace()` (`xmin, xmax, ymin, …` enumerated by hand — forget one and
  it silently stops persisting; this is exactly how the `axis_breaks` bug happened),
- dialog → widget: every customize tab writes **directly onto the widget**
  (`gw.xmin = …`, `gw.legend_properties[idx][k] = v`) and then emits
  `properties_changed` so the View can forward it to the ViewModel.

Consequences: pervasive defensive `getattr(gw, 'x_as_numeric', None)` /
`hasattr` guards, "reset `legend_properties`" logic duplicated in four different
places, aliasing bugs (see §1.6), and any new property must be added in ~6 places
to fully work (model, widget, config collection, sync-back list, dialog load,
dialog apply). **This is the single biggest tax on adding features.**

### 1.3 The ViewModel is both anemic and view-contaminated

`VMWorkspaceGraphs` opens `QFileDialog` and `QMessageBox` directly
(`load_dataframes`, `save_dataframe`, `apply_filters`, `save_workspace`…). That
makes it untestable headless and inverts MVVM — the ViewModel should expose
signals/results and let the View do dialogs. Meanwhile the actual orchestration
(widget creation, state sync, legend bookkeeping, filter-capture ordering) lives
in the 1800-line View. The comment in `_create_and_display_plot` —
*"CRITICAL: … because `vm.selected_df_name` can be changed by window activation
events"* — is the symptom: `selected_df_name` is hidden mutable global state that
UI focus events mutate mid-operation. Note that `_on_update_plot()` still filters
by `vm.selected_df_name` (line 1095), so the same hazard the comment warns about
exists on the update path.

### 1.4 The renderer is not an engine

`PlotRenderer` takes the whole `VGraph` (`self.vg`) and reads ~30 attributes off
it — and writes back (`vg.trendline_equations`). It cannot run without a live Qt
widget and canvas. That blocks: headless high-resolution export, unit-testing the
plotting logic, and reuse by the scripting API (`api/graphs.py`) or the AI agent.

Rendering also depends on **pyplot global state**: `plt.figure(...)`,
`plt.colorbar(...)`, `plt.setp(...)` and — worst — `plt.close('all')` in
`create_plot_widget()`, which closes *every* pyplot figure in the process,
including any owned by the Spectra/Maps workspaces. The OO API
(`matplotlib.figure.Figure` + `FigureCanvasQTAgg`) removes this entire class of
cross-talk.

### 1.5 Update = destroy and rebuild

`_on_update_plot()` and every dialog Apply rebuild the whole canvas/toolbar
(`create_plot_widget`) or do a full `plot()` — even for a label edit. This is the
main performance cost (wafer plots re-run a 600×600 `griddata` interpolation each
time) and it loses toolbar zoom/pan history. There is no distinction between
"data changed → replot" and "style changed → restyle artists / redraw".

### 1.6 Legend state is inferred from the render — backwards

`get_legend_properties()` reads colors back out of rendered matplotlib handles
(with heuristic hex-matching against `DEFAULT_COLORS`) to reconstruct the style
state, which is then fed forward into the next render. State should flow one way:
spec → render. Reading it back from artists is why legend resets are needed in
four places (Z changed, filters changed, sort changed, category-count mismatch)
and why the code is full of try/except around handle inspection.

### 1.7 Bugs found during review

| # | Location | Bug |
|---|----------|-----|
| B1 | `v_graph.py:170` | `plt.close('all')` closes figures belonging to other workspaces (global side effect). |
| B2 | `v_plot_renderer.py:506,628` | `self.vg.zmin if self.vg.zmin else …` — a limit of exactly **0** is treated as unset (falsy). Same for `zmax`. Real for data crossing 0. |
| B3 | `v_workspace_graphs.py:1629` | `_on_grid_changed_toolbar` is **never connected** (dead code); grid/DPI/rotation toolbar controls only take effect via "Update plot", which is not discoverable. If connected as-is it would also plot the *unfiltered* dataframe. |
| B4 | `_configure_graph_from_model` | `annotations` and `filters` are **not** deep-copied (unlike `y`, `legend_properties`, `axis_breaks`) — widget and model share the same list objects; in-place edits (double-click edit dialog, drag) mutate the model silently, bypassing the documented "independent copies" policy. Works today by accident; will break the moment update/cancel semantics rely on the copy. |
| B5 | `m_graph.py:113` | `MGraph.save()` returns `dict(vars(self))` with **live references** to mutable lists — a saved template/workspace dict mutates when the graph is later edited, until serialisation happens. Also: any attribute assigned to the model becomes part of the save format implicitly (no schema, no versioning at graph level). |
| B6 | `v_graph.py:637` | `_set_axis_scale(df)` is called even when `df is None` → `df[self.x]` raises if a log-scale flag is set on an empty graph. |
| B7 | `m_graph.py:17` | `filters` typed `List[str]` but is actually `List[Dict]` everywhere. |
| B8 | `v_graph.py:47–48` vs `m_graph.py:21` | Default plot height disagrees: 400 (widget, toolbar label) vs 420 (model, `clear_workspace` label). |
| B9 | `spectroview/__init__.py:48` | `DEFAULT_MARKERS = ['o'] * len(MARKERS)` — marker cycling is silently a no-op; every series gets 'o'. If intentional, the indirection is misleading. |
| B10 | `v_graph.py:622` | Limit/scale skips report via `print()` to stdout, invisible to users. |
| B11 | `_on_update_plot` | Rebinds the active graph's `df_name` to whatever DataFrame is currently selected in the list — updating a plot while browsing another DataFrame silently re-targets it. |
| B12 | `customize_axis.py:302–318` | Y-break loading is split across two blocks with the minor-ticks loading interleaved; the `if not y_breaks:` block runs after checkboxes were already set — works, but only by accident of ordering. |

### 1.8 Performance notes

- `apply_filters` returns `df.copy()` for the no-filter case — every render of
  every graph copies the full DataFrame. A read-only view + copy-on-write
  discipline (or documenting pandas CoW mode) would avoid it.
- Full re-render on every property change (§1.5).
- Wafer `griddata` on a 600×600 mesh each render; cacheable keyed on
  (data hash, wafer_size, interp method).
- `_update_column_combos` rebuilds six combo boxes on every DataFrame selection,
  which fires on every MDI focus change through `_sync_gui_from_graph`.

---

## 2. Existing feature review

| Feature | Verdict | Notes |
|---|---|---|
| Plot styles (point/scatter/box/bar/line/trendline/histogram/wafer/2Dmap) | Good coverage | Point plot = mean ± 95 % CI is a good scientific default, but is **undocumented in the UI** — users may mistake it for raw data. Show "mean ± 95 % CI" in the default Y label or legend. |
| Categorical/numeric X handling with dodge | Good idea, opaque | Auto/Category/Numerical override buried in the Axis tab; dodge checkboxes in "More options". These belong together. |
| Trendline with anchor + equation table | Well designed | CI-band code duplicated in hue and no-hue branches (~40 lines each). Equations live only on the widget (`trendline_equations`), so they don't survive save/load. |
| Multi-axes Y2/Y3/X2 | Weak abstraction | Hardcoded colors (red/green/purple), hardcoded marker, single series each, no styling, X2 plots `y[0]` against a *different column* on the same rows (a correlation-axis, not a true secondary scale — scientifically ambiguous). Should become a general "additional axes" list with per-axis series and styles. |
| Legend editing (label/marker/color per series) | Works, fragile | Built on the state-inference problem (§1.6). Color choice limited to the 10 default colors — no free color picker for series (the edge color has one, series colors don't). |
| Annotations (vline/hline/text, draggable) | Good foundation | Type set is minimal (no arrows/spans/boxes). Edit via double-click works; model-sync relies on the aliasing bug B4. |
| Axis breaks | Ambitious, brittle | Post-hoc mutation of artist data (won't survive bars/boxplots, interacts with formatters). The standard robust approach is two sub-axes with `d`-marks (as brokenaxes does). Fine as "beta"; rewrite when the engine exists. |
| Filters (`df.query`) | Good | Power-user friendly; errors surface via modal box from the ViewModel (wrong layer). |
| Templates | Good feature | Whole-workspace only; no per-graph "apply style" (see §5). |
| Update plot flow | Confusing | One "Update plot" button applies *all* side-panel state to the active plot, including re-binding the DataFrame (B11). Users expect the side panel to live-edit the selected plot. |
| Copy to clipboard | Good | But it is the **only** export path — see §3 Export. |
| Wafer plots (interpolated map, stats, slot annotation) | Strong domain feature | Interpolation method fixed to 'linear'; no masking outside wafer edge for sparse data. |

---

## 3. Missing features (gap analysis)

**Figure/axes** — no font family/size controls (title, labels, ticks) per
figure; no subtitle; no tick direction (in/out), tick label number format
(decimal places, scientific), or custom tick locators; no symlog; no axis
inversion; no background color / frame (spine) visibility control; no margins /
physical figure size (everything is pixel-window-driven); no aspect control.

**Curves/series** — no per-series line width, line style, alpha, z-order, marker
size (only global `scatter_size`), marker face vs edge, or error-bar style
choice (CI vs SD vs SEM vs none — currently hardwired: point/line = 95 % CI,
bar = SD). No free color picker per series.

**Legend** — no font size, no columns, no frame on/off, no title, no
transparency slider, no location presets menu (only "best"/"outside"/drag).

**Annotations** — no arrows, shapes, shaded spans (axvspan/axhspan), callouts;
no annotation styling beyond color/width/style/fontsize; text supports mathtext
implicitly but there is no hint of it in the UI.

**Axes systems** — see §2 multi-axes; no true twin-axis with independent series
list, no inset axes, no multi-panel figures.

**Color management** — palette list is small and `jet` is the default (both a
scientific and accessibility problem); no colorblind-safe categorical palette
(Okabe-Ito, Tol); no reversed colormaps; no per-graph colormap normalization
(log norm, centered norm for diverging data).

**Export — the most important gap.** The matplotlib toolbar's Save action is
deliberately hidden, so the *only* output path is copy-to-clipboard. A
publication tool needs: PNG/TIFF/SVG/PDF/EPS, DPI choice, transparent
background, physical size (mm/in), font embedding (`pdf.fonttype 42`,
`ps.fonttype 42`, `svg.fonttype none`), and journal presets (e.g. single-column
90 mm / double-column 180 mm @ 300–600 dpi). Batch "export all graphs" would fit
the multi-wafer workflow perfectly.

**UX** — no undo/redo; no copy/paste style between graphs; no "reset to
default"; no keyboard shortcuts (except the ESC blocker); live preview is
inconsistent (legend edits are live, axis edits are Apply-only, side panel needs
"Update plot").

---

## 4. Publication-quality assessment

Current output is **screen-quality, not publication-quality**:

1. **No file export** (§3) — the blocking issue.
2. **Sizing is in pixels** of the MDI window; journals specify figures in mm at
   a given DPI. Add a physical-size mode (width mm × height mm, dpi for raster).
3. **`jet` default colormap** — perceptually non-uniform, misleading for wafer
   maps, colorblind-hostile; default should be `viridis` and categorical default
   Okabe-Ito.
4. **Font pipeline**: Arial via the style file is fine, but there is no UI
   control and no fallback declared (`font.sans-serif` list). Export must set
   `pdf.fonttype: 42` / `ps.fonttype: 42` so text stays editable text.
5. **Style sheet** is a reasonable base (12 pt labels, 9 pt ticks, thin frame)
   but hardcoded to one look; there is a `PLOT_POLICY_LIGHT` but styling is not
   user-selectable (no "theme" concept per figure).
6. Minor: error-bar cap sizes, marker edge width 0.5, box widths etc. are magic
   numbers spread through the renderer; they belong in a style object.

---

## 5. Proposed target architecture

### 5.1 Layered design

```
model/graph_spec.py          GraphSpec (dataclasses, versioned to_dict/from_dict)
   ├─ DataSourceSpec          df_name, filters
   ├─ AxisSpec (×n)           label, limits, scale, ticks, break, inverted
   ├─ SeriesSpec (×n)         y-column, target axis, SeriesStyle
   │    └─ SeriesStyle        color, marker, msize, lw, ls, alpha, zorder, errorbar
   ├─ LegendSpec              visible, loc/bbox, ncol, font, frame, title, alpha
   ├─ AnnotationSpec (×n)     typed: VLine | HLine | Text | Arrow | Span | Shape
   └─ FigureStyle             size (mm or px), dpi, fonts, grid, palette, theme

plot_engine/                 NO Qt imports; pure matplotlib OO API
   ├─ engine.py               render(fig: Figure, spec: GraphSpec, df) -> RenderResult
   ├─ styles/…                one renderer class per plot style, registry-based
   ├─ export.py               export(spec, df, path, format, dpi, transparent, preset)
   └─ palettes.py             Okabe-Ito, Tol, colormaps, cycling

viewmodel/vm_workspace_graphs.py
   owns {graph_id: GraphSpec} + dataframes; commands:
   create/update(patch)/delete/duplicate/apply_style; undo/redo stack of spec
   snapshots; emits graph_changed(id); NO QFileDialog/QMessageBox.

view/
   ├─ v_workspace_graphs.py   thin coordinator; owns dialogs and MDI
   ├─ v_graph.py              canvas host + interaction (drag → command to VM);
   │                          holds NO plot properties of its own
   └─ customize_graph/        tabs edit a *working copy* of GraphSpec;
                              Apply → vm.update_graph(id, patch)
```

**Why each piece:**
- *GraphSpec dataclasses*: kills the triple-state problem; one definition, typed,
  self-serialising, versioned (`schema: 4`) with a migration shim for old flat
  dicts (`MGraph.load` key mapping preserved for `.graphs` compatibility).
- *Headless engine*: enables export at arbitrary size/DPI, real unit tests
  (render into an Agg figure, assert artists), reuse by `api/graphs.py` and the
  AI agent, and removes all pyplot global state (fixes B1).
- *Renderer registry*: adding a plot style = one new class, no `if/elif` chain
  in `VGraph._plot_primary_axis`.
- *Command-style ViewModel*: single write path (`update_graph(id, patch)`)
  makes undo/redo a free feature and gives the customize dialog and the side
  panel identical semantics (ends the "which of my three copies is current?"
  bugs).
- *Series-first model*: `SeriesSpec` list replaces `y` + `y2` + `y3` + `x2` +
  `legend_properties` — multi-series, per-series styling, and real twin axes all
  fall out of one abstraction, and legend state is no longer inferred from
  rendered artists (§1.6).

### 5.2 Migration strategy (no big-bang rewrite)

1. Introduce `GraphSpec` alongside `MGraph`; `MGraph` becomes a thin adapter
   (`MGraph.spec`) so all existing call sites keep working.
2. Move rendering functions out of `PlotRenderer`/`VGraph` into `plot_engine`
   one plot style at a time, changing signatures from `self.vg.<attr>` to
   `(ax, spec, df)`.
3. Delete `VGraph`'s duplicated attributes last, once every reader goes through
   the spec.

Old workspaces/templates keep loading through the existing `load()` key-mapping;
add tests that round-trip a v3 `.graphs` file before touching serialisation.

---

## 6. Prioritised roadmap

Complexity: **S** ≤ ½ day, **M** = 1–3 days, **L** = 1–2 weeks.

### Phase 1 — Foundation & correctness
| Task | Size |
|---|---|
| Replace pyplot with OO `Figure` API; delete `plt.close('all')` (B1) | S |
| Fix falsy-zero limit bugs (B2), `_set_axis_scale` None-df crash (B6), type hints (B7), height default (B8) | S |
| Wire or remove toolbar grid/DPI/rotation handlers (B3); decide live-edit vs Update-button semantics | S |
| Deep-copy `annotations`/`filters` in `_configure_graph_from_model`; make `MGraph.save()` deep-copy (B4, B5) | S |
| `GraphSpec` dataclasses + versioned serialisation + legacy migration + round-trip tests | M |
| Extract headless `plot_engine` (per-style renderers, registry) with unit tests | L |
| Single source of truth: strip `VGraph` state, route all edits through `vm.update_graph(id, patch)` | L |
| Move `QFileDialog`/`QMessageBox` out of the ViewModel | S |

### Phase 2 — Core customisation
| Task | Size |
|---|---|
| Per-series styles (color picker, marker, size, line width/style, alpha, z-order) via `SeriesSpec` | M |
| Error-bar options (none/SD/SEM/CI, cap size) for point/line/bar | M |
| Axis: tick direction, label format, symlog, inverted axes, font sizes | M |
| Legend: ncol, frame, title, font size, alpha, location presets | S |
| Figure: title/subtitle fonts, background, spine visibility, margins | M |
| Generalise Y2/Y3/X2 into per-axis series lists with styling | L |

### Phase 3 — Publication output
| Task | Size |
|---|---|
| Export dialog: PNG/TIFF/SVG/PDF/EPS, DPI, transparent, font embedding (`fonttype 42`) | M |
| Physical figure sizing (mm/in) decoupled from window size + journal presets | M |
| Batch "export all graphs" | S |
| Default palette overhaul: Okabe-Ito categorical, viridis maps, keep jet opt-in | S |
| Style themes (mplstyle-backed) selectable per figure | M |

### Phase 4 — Advanced plotting
| Task | Size |
|---|---|
| Annotation types: arrows, spans (axvspan/axhspan), boxes, callouts; styling UI | M |
| Rewrite broken axis on two sub-axes (robust for all plot types) | L |
| Multi-panel figure composer (combine open graphs into one exported figure, shared axes) | L |
| Inset axes | M |
| Colormap normalisation options (log, centered) for wafer/2Dmap | S |

### Phase 5 — Templates, styles & UX
| Task | Size |
|---|---|
| Per-graph style templates: save/apply style (not data) to any graph | M |
| Copy/paste style between graphs; "reset to default" | S |
| Undo/redo (spec snapshot stack in ViewModel) | M |
| Keyboard shortcuts + consistent live preview (debounced re-render) | M |
| Restyle-without-replot fast path for pure style changes | L |

---

## 7. Coding guidelines for the implementation

- All new modules: full type hints, dataclasses for specs, docstrings on public
  API; `plot_engine` must not import PySide6 (enforce with a unit test).
- One write path for graph state (`vm.update_graph`); views never mutate specs
  they didn't copy.
- Serialisation is explicit (`to_dict`/`from_dict` + `SCHEMA_VERSION`), never
  `vars(self)`; every schema change ships with a migration + round-trip test.
- Backward compatibility: existing `.graphs` v3 files and templates must load
  unchanged throughout all phases (guard with a fixture-file test).
- Tests: engine renderers (headless Agg), spec round-trips, ViewModel commands
  (no Qt needed once dialogs are out), plus the existing integration test
  (`tests/integration/test_graphs_workflow.py`) kept green at every step.
