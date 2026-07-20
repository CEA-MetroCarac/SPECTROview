# Common Workflows

Step-by-step touch-points for recurring tasks. Always read the named files
first; these lists are orientation, not a substitute for the code. Every task
below also ends with **tests + docs**: add/adjust tests, and update the
developer guide + user manual per [documentation.md](documentation.md).

## Add a keyboard shortcut (a workspace)
1. In the workspace View's `_setup_keyboard_shortcuts()`, add a `QShortcut` on
   the workspace container (e.g. `self.mdi_area` for Graphs) with
   `Qt.ShortcutContext.WidgetWithChildrenShortcut`.
2. Connect it to a small `_on_<action>_shortcut` handler that resolves the
   active target and delegates to the existing button/handler (don't duplicate
   the action's logic).
3. Add it to the shortcut-registration test (e.g. `TestKeyboardShortcuts`) and
   its dispatch test.

## Replace / add a matplotlib nav-toolbar icon (Graphs)
- Icons are mapped in `v_graph.py::_CUSTOM_TOOLBAR_ICONS` (action text →
  colorful PNG in `resources/icons/`). Custom icons are theme-independent and
  re-asserted on palette change by the toolbar's `ToolbarEventFilter`. Add an
  entry; don't add tinting.

## Add a plot style (Graphs)
1. Add the name to `PLOT_STYLES` in `spectroview/__init__.py`.
2. Implement its rendering in `view/components/v_plot_renderer.py`
   (`PlotRenderer`).
3. Add any new config field to `MGraph` (`model/m_graph.py`) — this
   auto-propagates through the seeded defaults and the commit/diff schema (see
   [graphs-workspace.md](graphs-workspace.md)); update the `test_m_graph.py`
   save-key/`ALL_SAVE_KEYS` lists and `TestPlotStyleCoverage`.
4. If it needs customization UI, extend the relevant `customize_graph/*` tab.

## Add a customization option (Customize Graph dialog)
1. Add the backing field to `MGraph` (default included).
2. Add the control to the correct tab widget under
   `view/components/customize_graph/` (`customize_axis`, `customize_legend`,
   `customize_more_options`, `customize_annotations`), wiring load + apply.
3. Read/apply it in `v_plot_renderer.py`. If the change only restyles existing
   artists (no data reshape), consider `graph_style.RESTYLE_SAFE_FIELDS` for the
   fast path; otherwise it triggers a full replot.
4. Cover it in `tests/unit/view/test_customize_graph_dialog.py`.

## Add a file loader (Spectra/Maps)
- Loaders live in `model/m_io.py` (+ `m_spc.py` for SPC, `renishawWiRE` for
  WDF). Add the format there and route it from the loading path in the Maps/
  Spectra ViewModel. See `docs/developer/file_loading.md`.

## Add a public API function
- Extend the matching module in `spectroview/api/` (e.g. `graphs.py`,
  `fitting.py`). Keep signatures backward compatible; positional callers in
  tests and the AI agent depend on them. Document in `docs/api/`.

## Wire two workspaces together
- Do it in `main.py::setup_connections()` (reference injection or a signal),
  never by reaching into another workspace's widget tree.

## Persist a new setting
- Add getter/setter on `MSettings` (`model/m_settings.py`) and the settings VM/
  View (`vm_settings.py`, `view/components/v_settings.py`). Remember the test
  QSettings backend is isolated per-test.
