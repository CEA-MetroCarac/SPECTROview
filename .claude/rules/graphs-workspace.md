---
paths:
  - "spectroview/view/v_workspace_graphs.py"
  - "spectroview/view/components/v_graph.py"
  - "spectroview/view/components/v_plot_renderer.py"
  - "spectroview/view/components/v_multipanel_dialog.py"
  - "spectroview/view/components/graph_commit.py"
  - "spectroview/view/components/customize_graph/**"
  - "spectroview/viewmodel/vm_workspace_graphs.py"
  - "spectroview/model/m_graph.py"
  - "spectroview/model/graph_style.py"
  - "spectroview/model/m_plot_recipe*.py"
  - "spectroview/model/m_style_template_store.py"
---

# Graphs Workspace — Conventions

The Graphs workspace is the most actively developed area. Full narrative:
`docs/developer/graphs.md` and `docs/developer/graph_workspace_review.md`. This
file captures the load-bearing patterns to preserve when editing it.

## Single source of truth: `MGraph`

- `model/m_graph.py::MGraph` is a dataclass; its field set is the schema.
- `VGraph.__init__` **seeds every model field from `vars(MGraph())`** — never
  hand-copy the default list (a hand-written copy silently drifted once). Add a
  field to `MGraph` and it flows through automatically.
- `view/components/graph_commit.py` derives `COMMIT_FIELDS` from `MGraph`'s
  dataclass fields; `snapshot()`/`diff()` sync "whatever changed on the widget"
  back to the model for save and undo/redo. Don't reintroduce hand-picked field
  lists.
- `TestModelSchemaSync` locks the seeded-defaults invariant. Keep it green.

## Rendering

- `view/components/v_plot_renderer.py::PlotRenderer` draws each `PLOT_STYLES`
  entry. `VGraph` (`view/components/v_graph.py`) owns the figure/canvas, the
  matplotlib nav toolbar, annotations, and interaction.
- **Restyle fast path**: `model/graph_style.py::RESTYLE_SAFE_FIELDS` lists
  fields whose change only restyles existing artists — those skip a full
  replot. A field that changes what data/artists are drawn must NOT be added
  there.

## Shared toolbar

- Each `VGraph` builds a hidden, parentless `toolbar_container`. Only the active
  graph's container is reparented into the workspace's single
  `graph_toolbar_slot` by `_sync_active_graph_toolbar()`. Any code path that
  changes the active graph or rebuilds a `toolbar_container` must call
  `_sync_active_graph_toolbar()`.
- Nav-toolbar custom icons: `_CUSTOM_TOOLBAR_ICONS` (Home/Pan/Zoom → colorful
  PNGs), re-asserted on theme change by `ToolbarEventFilter`. `Ctrl+R` →
  `VGraph._rescale()` (matplotlib Home).

## MDI windows

- `MdiSubWindow` (in `v_workspace_graphs.py`) hosts each `VGraph`. It sets a
  transparent window icon (a null icon still shows Fusion's default logo) and
  neutralizes the Fusion title-bar shadow — preserve both when editing it.

## Live preview & undo/redo

- Customize dialogs preview live (debounced) and restore on Cancel; a preview
  must stay purely visual and must **not** commit to the ViewModel/undo stack —
  `cancel_all()` relies on the snapshot baseline. Commit happens on OK.
- The annotation edit dialogs (`customize_annotation_dialogs.py`) share a
  `LivePreviewAnnotationDialog` base; each type maps to its dialog via an
  `_EDIT_DIALOGS` dict (no long if/elif chains — keep the map).
- Undo/redo batches related updates into one step in the ViewModel
  (`vm_workspace_graphs.py`); multi-call operations (e.g. delete-all) capture
  ids first and are one reversible step.

## Recipes vs styles (don't confuse them)

- **Plot recipe** = a full plot-config set: `model/m_plot_recipe(_store).py`,
  `view/components/v_plot_recipe_dialog.py`, API `*_plot_recipe`.
- **Style template** = portable styling only: `model/m_style_template_store.py`,
  `view/components/v_style_template_dialog.py`.
  These are distinct features with distinct on-disk stores.

## Annotations

- Rendered by `VGraph._render_*`; each artist carries `_annotation_data`.
- Canvas interaction: click/drag moves; grabbing a border/endpoint resizes
  (`_annotation_handle_at` → `_resize_dragged_annotation`), with a hover cursor
  and pixel-based hit-testing via `self.ax.transData` (not `event.xdata` — see
  [pitfalls.md](pitfalls.md)).
