---
name: spectra-maps-workspace
description: Spectra and Maps workspaces, SpectraStore data backbone, dual array non-destructive preprocessing, template-method subclassing, and heatmap cache.
trigger: model_decision
---

# Spectra & Maps Workspaces

Deep architectural narrative is in `docs/developer/spectra.md`, `docs/developer/maps.md`, and `docs/developer/spectra_store.md`. This file captures the load-bearing conventions and architectural contracts for the Spectra and Maps workspaces.

---

## 1. Data Backbone: `SpectraStore` & `MapData`

- **Single Source of Truth**: `spectroview.model.spectra_store.SpectraStore` manages all spectral data for both workspaces.
- **Zero-Copy References**: `store.get_map_data(name)` returns a direct reference to the `MapData` instance. **Always re-fetch `md` at the start of an operation — never store `md` as a persistent instance attribute**.
- **Unified Tensor Layout**:
  - Spectra Workspace: $N = 1$ spectrum per block.
  - Maps Workspace: $N$ up to 100,000+ spectra sharing an x-axis in a single block.
  - The shared tensor layout allows the identical batched fit engine to process both workspaces.
- **Non-Destructive Dual Arrays**:
  - `x0` and `Y0` are raw data arrays written once on file import; **they must never be mutated**.
  - All preprocessing operations (crop, baseline, smoothing) write to `x` and `Y`.
  - When `x` and `Y` are `None`, the store falls back to `x0` and `Y0`.
  - Resetting preprocessing (`reinit_spectra()`) simply sets `x = Y = None`.
- **Pull-on-Signal Update Pattern**: Mutate `MapData` state, then emit `_emit_selected_spectra()` (and optionally `_emit_list_update()`). Do not introduce ad-hoc per-field observer loops.

---

## 2. Base $\leftrightarrow$ Subclass Inheritance (Template-Method Pattern)

`VMWorkspaceSpectra` is the base ViewModel; `VMWorkspaceMaps` **subclasses** it (matching the relationship between `VWorkspaceSpectra` and `VWorkspaceMaps`).

Always inspect the parent method before modifying an override. Bulk operations must be customized **only** via the three established template hooks:
1. `_get_target_mds(apply_all)`: Determines which `MapData` blocks to process (Spectra: selected spectra; Maps: current map, or all maps if Ctrl is pressed).
2. `_on_map_data_changed(md, action)`: Per-block hook (Maps: re-runs preprocessing and invalidates heatmap cache).
3. `_post_bulk_action(apply_all, action)`: Post-loop hook (Maps: refreshes fit results table and map canvas).

### View Wiring Guard
`VWorkspaceMaps.__init__` sets `_skip_parent_setup = True` so the parent constructor does not connect signals to the wrong ViewModel, injects `VMWorkspaceMaps`, and then invokes `setup_connections()`. **Do not remove or simplify this guard**.

### Ctrl-Modifier Convention
All bulk operation buttons follow `_apply_with_ctrl(fn)`:
- Click: applies operation to selected item.
- Ctrl + Click: applies operation across all items.

---

## 3. Selection Payloads

`_emit_selected_spectra()` emits a dictionary consumed by `v_spectra_viewer.py`:
- **Spectra Workspace**: emits `type: 'tensor_list'` (independent spectra, potentially with differing x-axes).
- **Maps Workspace**: overrides to emit `type: 'tensor'` (one shared x-axis matrix).
Always maintain established payload keys (`spectra_names`, `x`, `Y`, `peaks_params`).

---

## 4. Maps Specifics

- **Dual Storage**:
  - `self.maps[name]` is a `pd.DataFrame` containing spatial coordinates and metadata, used for heatmap interpolation and CSV table export.
  - `store.get_map_data(name)` is the tensor block containing raw/processed spectra, used for all mathematical operations and fitting.
  - Preprocessing modifies only the tensor block, not the DataFrame.
- **Spectrum Coordinate Parsing**:
  - Spectrum naming follows `f"{map_name}_({float(x)}, {float(y)})"`.
  - When changing coordinate parsing, update both `_extract_coords_from_fname` (View) and `_extract_coords_for_spectra` (ViewModel) to keep formatting consistent.
- **Heatmap Grid Cache**:
  - `VMapViewer._griddata_cache` caches interpolated grids keyed by `(map, parameter, x_range, map_type, mask, outliers)`.
  - Invalidate the cache by emitting `clear_map_cache_requested` after any operation that modifies fit parameters or data.
- **Detachable Map Viewers**:
  - Multiple `VMapViewerDialog` instances share map data.
  - On dialog close, `plt.close(dialog.map_viewer.figure)` must be called explicitly. Otherwise, the Figure is retained in matplotlib's global `Gcf` registry and leaks the entire widget tree.

---

## 5. Signal Loop Guard

Never trigger `_emit_selected_spectra()` synchronously from a slot connected to `spectra_selection_changed`. Defer the emission with `QTimer.singleShot(0, self._emit_selected_spectra)` to avoid signal recursion loops.
