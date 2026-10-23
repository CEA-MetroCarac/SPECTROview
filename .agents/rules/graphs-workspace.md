---
name: graphs-workspace
description: Graphs workspace conventions, MGraph schema single source of truth, PlotRenderer, shared toolbar, MDI subwindows, live preview, and undo/redo.
trigger: model_decision
---

# Graphs Workspace — Conventions

Full narratives are in `docs/developer/graphs.md` and `docs/developer/graph_workspace_review.md`. This file captures the load-bearing patterns and architectural contracts to preserve when editing the Graphs workspace.

---

## 1. Single Source of Truth: `MGraph`

- **Canonical Schema**: `spectroview.model.m_graph.MGraph` is a dataclass defining the plot configuration model.
- **Seeded Defaults**: `VGraph.__init__` seeds every model field from `vars(MGraph())`. **Never hardcode default values or hand-copied field lists**. Adding a field to `MGraph` flows automatically through the widget layer.
- **Commit & Diff**: `spectroview/view/components/graph_commit.py` derives `COMMIT_FIELDS` programmatically from `MGraph`'s dataclass fields. Its `snapshot()` and `diff()` methods synchronize widget state to the model for persistence and undo/redo history.
- **Schema Lock**: `tests/unit/model/test_m_graph.py::TestModelSchemaSync` enforces that `VGraph` seeded defaults match `MGraph` fields exactly. Keep this test green.

---

## 2. Plot Rendering & Restyle Fast Path

- **Rendering Dispatch**: `view/components/v_plot_renderer.py::PlotRenderer` draws each plot style registered in `spectroview.PLOT_STYLES`. `VGraph` (`view/components/v_graph.py`) owns the matplotlib Figure, Canvas, navigation toolbar, annotations, and user interaction.
- **Restyle Fast Path**: `model/graph_style.py::RESTYLE_SAFE_FIELDS` contains properties whose modification only alters visual attributes (colors, line widths, alpha) without reshaping data or rebuilding artists. These changes take the fast path and avoid an expensive full replot. Fields modifying data shapes, axes bounds, or adding artists must **not** be included in `RESTYLE_SAFE_FIELDS`.

---

## 3. Shared Graph Toolbar Slot

- Each `VGraph` constructs a hidden `toolbar_container`.
- Only the currently active graph's container is reparented into the workspace's single `graph_toolbar_slot` via `_sync_active_graph_toolbar()`.
- Any code path that switches the active MDI subwindow or rebuilds a toolbar container must call `_sync_active_graph_toolbar()`.
- Navigation toolbar icons are mapped in `_CUSTOM_TOOLBAR_ICONS` to colorful PNGs. They are theme-independent and maintained by `ToolbarEventFilter`. `Ctrl+R` triggers `VGraph._rescale()` (matplotlib Home).

---

## 4. MDI Subwindows & UI Constraints

- `MdiSubWindow` (in `v_workspace_graphs.py`) wraps each `VGraph`.
- It configures a transparent window icon (a null icon falls back to Fusion's default window logo) and neutralizes the native Fusion title-bar shadow. Preserve both configurations.

---

## 5. Live Preview & Undo/Redo

- **Live Dialog Previews**: Customize dialogs (`CustomizeGraphDialog`, annotation dialogs) update the plot preview live with debounce. These previews are strictly visual and **must not commit to the ViewModel's undo stack** during interaction. If the user clicks **Cancel**, the snapshot baseline is restored; changes commit to the undo stack only on **OK**.
- **Annotation Edit Dialogs**: Subclass `LivePreviewAnnotationDialog` and register in the `_EDIT_DIALOGS` map in `customize_annotation_dialogs.py`. Maintain dictionary dispatch rather than long `if/elif` chains.
- **Atomic Undo/Redo**: Batch multi-step operations (e.g. deleting multiple graphs or annotations) into a single reversible command in `vm_workspace_graphs.py`.

---

## 6. Recipes vs Style Templates

Do not confuse these distinct persistence models:
- **Plot Recipe**: Complete plot configuration (data columns, axes, style, annotations): `model/m_plot_recipe.py`, `v_plot_recipe_dialog.py`.
- **Style Template**: Visual presentation settings only (fonts, spines, colors, markers): `model/m_style_template_store.py`, `v_style_template_dialog.py`.

---

## 7. Annotations & Pixel Transforms

- Annotations are rendered via `VGraph._render_*` and carry `_annotation_data`.
- Hit-testing and dragging must be calculated using pixel-based transforms:
  ```python
  self.ax.transData.inverted().transform((event.x, event.y))
  ```
  Do not rely on `event.xdata` / `event.ydata` when secondary `twinx()` axes are active (see [.agents/rules/pitfalls.md](pitfalls.md)).
