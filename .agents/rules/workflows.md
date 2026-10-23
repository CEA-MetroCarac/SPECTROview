---
name: workflows
description: Step-by-step touchpoints for recurring tasks including adding shortcuts, plot styles, graph customizations, file loaders, API functions, and settings.
trigger: model_decision
---

# Common Development Workflows

Step-by-step implementation touchpoints for recurring tasks. Always read the referenced files first; these lists guide orientation, not replace code investigation. Every task must conclude with tests and documentation synchronization per [.agents/rules/documentation.md](documentation.md).

---

## 1. Add a Keyboard Shortcut (Workspace)

1. In the target workspace View's `_setup_keyboard_shortcuts()`, add a `QShortcut` attached to the workspace container (e.g. `self.mdi_area` for Graphs) with `Qt.ShortcutContext.WidgetWithChildrenShortcut`.
2. Connect it to a dedicated handler `_on_<action>_shortcut` that checks for the active target and delegates to the existing handler or button method.
3. Add tests in the matching test file (e.g. `TestKeyboardShortcuts` in `tests/unit/view/`).
4. Update `spectroview/resources/user_manual/shortcuts.md`.

---

## 2. Replace or Add a Matplotlib Nav-Toolbar Icon (Graphs)

1. Custom navigation icons are mapped in `v_graph.py::_CUSTOM_TOOLBAR_ICONS` (action text $\rightarrow$ colorful PNG in `spectroview/resources/icons/`).
2. Custom icons are theme-independent and maintained across palette changes by the toolbar's `ToolbarEventFilter`.
3. Add the PNG to `spectroview/resources/icons/` and map the action text. Do not add tinting.

---

## 3. Add a Plot Style (Graphs)

1. Register the plot style name in `PLOT_STYLES` in `spectroview/__init__.py`.
2. Implement rendering logic in `spectroview/view/components/v_plot_renderer.py` (`PlotRenderer`).
3. Add any necessary backing configuration fields to `MGraph` (`model/m_graph.py`).
4. Update `ALL_SAVE_KEYS` and `CUSTOM_VALUES` in `tests/unit/model/test_m_graph.py`, and add test coverage to `TestPlotStyleCoverage` in `tests/unit/view/test_v_graph_plotting.py`.
5. If UI controls are required, extend `spectroview/view/components/customize_graph/`.
6. Update documentation: `docs/developer/graphs.md`, `spectroview/resources/user_manual/graphs.md`, `docs/api/graphs.md`.

---

## 4. Add a Customization Option (Customize Graph Dialog)

1. Add the backing field and default value to `MGraph` in `spectroview/model/m_graph.py`.
2. Add the UI control to the appropriate tab in `spectroview/view/components/customize_graph/` (`customize_axis`, `customize_legend`, `customize_more_options`, `customize_annotations`), wiring data load and apply handlers.
3. Read and apply the field in `v_plot_renderer.py`. If the modification only restyles existing artists without data reshaping, consider adding it to `graph_style.RESTYLE_SAFE_FIELDS` for the fast path.
4. Cover the option in `tests/unit/view/test_customize_graph_dialog.py` and `test_m_graph.py`.
5. Update `spectroview/resources/user_manual/graphs.md`.

---

## 5. Add a File Loader (Spectra / Maps)

1. Implement format parsing in `spectroview/model/m_io.py` (or a dedicated format module such as `m_spc.py`).
2. Connect loader in the loading pipeline in `vm_workspace_spectra.py` and `vm_workspace_maps.py`.
3. Add sample data files in `examples/` if appropriate.
4. Write unit tests in `tests/unit/model/test_m_io.py`.
5. Update `docs/developer/file_loading.md` and `spectroview/resources/user_manual/supported_data.md`.

---

## 6. Add a Public API Function

1. Extend the appropriate module in `spectroview/api/` (`spectra.py`, `2dmap.py`, `graphs.py`, `fitting.py`, etc.).
2. Ensure signatures are backward compatible; positional arguments must remain stable.
3. Add unit tests in `tests/unit/api/` or matching unit test module.
4. Document the function, arguments, and return types in `docs/api/`.

---

## 7. Wire Two Workspaces Together

1. Add cross-workspace linkages in `main.py::setup_connections()` (via reference injection or Qt signals/slots).
2. Never couple workspaces by traversing widget trees across tabs.

---

## 8. Persist a New Application Setting

1. Add getter and setter methods to `MSettings` (`spectroview/model/m_settings.py`).
2. Add control to `spectroview/view/components/v_settings.py` and wire in `vm_settings.py`.
3. Add unit test in `tests/unit/model/test_m_settings.py`.
4. Update `spectroview/resources/user_manual/settings.md`.
