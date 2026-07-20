---
paths:
  - "spectroview/view/v_workspace_spectra.py"
  - "spectroview/view/v_workspace_maps.py"
  - "spectroview/viewmodel/vm_workspace_spectra.py"
  - "spectroview/viewmodel/vm_workspace_maps.py"
  - "spectroview/view/components/v_spectra_viewer.py"
  - "spectroview/view/components/v_fit_model_builder.py"
  - "spectroview/view/components/v_peak_table.py"
  - "spectroview/view/components/v_spectra_list.py"
  - "spectroview/view/components/v_map_viewer.py"
  - "spectroview/view/components/v_map_list.py"
  - "spectroview/view/components/v_map_viewer_dialog.py"
  - "spectroview/view/components/v_fit_results.py"
  - "spectroview/model/spectra_store.py"
  - "spectroview/model/m_io.py"
  - "spectroview/model/workspace_io.py"
  - "spectroview/viewmodel/vm_fit_model_builder.py"
---

# Spectra & Maps Workspaces

Deep narrative: `docs/developer/spectra.md`, `docs/developer/maps.md`,
`docs/developer/spectra_store.md`, `docs/developer/file_loading.md`. This file is
the load-bearing conventions to preserve when editing these files.

## The data backbone (`SpectraStore` / `MapData`)

- `model/spectra_store.py` is the **single source of truth** for all spectral
  arrays. Both workspaces read/write the same store. `get_map_data(name)` returns
  a **direct reference** (zero-copy), so **re-fetch `md` at the start of each
  operation — never cache it** as an instance attribute.
- One `MapData` block = one logical dataset of N rows sharing an x-axis. **Spectra
  workspace: N=1 per block; Maps workspace: N up to ~100k in one block.** The
  identical tensor layout is *why* one vectorized fit serves both.
- **Non-destructive dual arrays**: `x0`/`Y0` are raw, written once, **never
  mutated**. All preprocessing writes `x`/`Y`; when those are `None` the store
  falls back to `x0`/`Y0`. `reinit_spectra()` just sets `x=Y=None`.
- Update pattern is **pull-on-signal**: mutate `MapData` → `_emit_selected_spectra()`
  → optionally `_emit_list_update()`. Don't add per-field change callbacks.

## Base ↔ subclass (template-method pattern)

`VMWorkspaceSpectra` is the base; `VMWorkspaceMaps` **subclasses** it (same for the
Views). Read the parent before changing a Maps override. Customize bulk operations
**only** through the three hooks — never fork the algorithm:

- `_get_target_mds(apply_all)` — which `MapData`s a bulk op touches (Spectra:
  selected spectra; Maps: current map, or all maps on Ctrl).
- `_on_map_data_changed(md, action)` — per-block hook (Maps: re-`batch_preprocess`,
  invalidate heatmap cache).
- `_post_bulk_action(apply_all, action)` — after the loop (Maps: refresh fit
  results + map view).

**Ctrl-modifier convention**: bulk buttons use `_apply_with_ctrl(fn)` — plain click
= selected, Ctrl+click = all. Preserve it for any new bulk action.

**Maps View wiring**: `VWorkspaceMaps.__init__` sets `_skip_parent_setup=True` so the
parent's `setup_connections()` doesn't wire to the wrong ViewModel, swaps in
`VMWorkspaceMaps`, then re-runs connections. Don't "simplify" that guard away.

## Selection payloads

`_emit_selected_spectra()` emits a dict the viewer reads. Spectra emits
`type: 'tensor_list'` (independent arrays, possibly different x per spectrum); Maps
**overrides** to `type: 'tensor'` (one shared-x matrix). Match the existing payload
keys when adding rendered data (see `spectra.md` payload table).

## Maps specifics

- **Dual storage**: `self.maps[name]` is a `pd.DataFrame` (heatmap input + CSV
  export); `store.get_map_data(name)` is the tensor block (all analysis).
  Preprocessing changes only the tensor block, **not** the DataFrame.
- **fname convention**: `f"{map_name}_({float(x)}, {float(y)})"`. Coordinates are
  parsed back out of this string across the code — if you change the format, update
  `_extract_coords_from_fname` (View) and `_extract_coords_for_spectra` (VM), and
  keep `float()` formatting identical on both write and read sides.
- **Heatmap cache**: `VMapViewer._griddata_cache` is keyed on
  (map, parameter, x-range, map_type, mask, outliers). Invalidate by emitting
  `clear_map_cache_requested` after any op that changes what the heatmap shows.
- **Detachable viewers**: multiple `VMapViewerDialog`s share data by reference. On
  close, `plt.close(dialog.map_viewer.figure)` is required — the Figure is held by
  matplotlib's global `Gcf` and leaks the whole widget tree otherwise.

## Traps

- **Never call `_emit_selected_spectra()` from a slot connected to
  `spectra_selection_changed`** → signal loop. Defer with `QTimer.singleShot(0, …)`.
- Fitting runs on `VBFthread`; results write back into the store from the worker
  thread. See [fit-engine.md](fit-engine.md) for the thread-safety contract before
  touching that path.
- Loaders: add formats in `model/m_io.py` (+ `m_spc.py`) and route from the VM's
  loading path; `_extract_spectra_from_map` drops the **last** wavenumber column
  (legacy artifact). Tests + docs per [documentation.md](documentation.md).
