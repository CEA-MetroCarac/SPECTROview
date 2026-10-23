---
name: architecture
description: System architecture, MVVM boundaries, module map, workspaces, data backbone, and cross-workspace wiring for SPECTROview.
trigger: always_on
---

# Architecture & Design Principles

On-demand orientation. Detailed diagrams and narrative are in `docs/developer/index.md` (mkdocs). This file is the module map and wiring rules required before any structural change.

## 1. MVVM Architecture

`SPECTROview` enforces strict Model-View-ViewModel separation across all workspaces:
- **View $\leftrightarrow$ ViewModel communicate exclusively via Qt signals/slots**. A ViewModel that needs to reach a View emits a signal; it never calls View methods or imports View classes.
- **Model layer (`spectroview/model/`) holds no Qt widget references** and can be tested independently without a `QApplication`.
- **View layer (`spectroview/view/`) delegates all data operations and mutations to its ViewModel**.

### Layer & Naming Conventions

| Layer | Prefix | Location | May Import |
|---|---|---|---|
| **View** | `v_` | `view/`, `view/components/` | ViewModel, components, theme |
| **ViewModel** | `vm_` | `viewmodel/` | Model, `fit_engine`, `application` (never View) |
| **Model** | `m_`/unprefixed | `model/` | Standard library, numpy, pandas, scipy (never View or ViewModel) |

- Private members: `_leading_underscore`
- Qt event handlers: `_on_<event>`
- Signals: past-tense naming (e.g. `spectra_list_changed`, `fit_progress_updated`)
- Tests: `tests/**/test_*.py`, classes `Test*`, functions `test_*`

---

## 2. Package Map

```text
spectroview/
├── __init__.py        # VERSION, constants (PLOT_STYLES, MARKERS, PALETTE, ICON_DIR…), QT_API=pyside6
├── main.py            # QMainWindow, tab wiring, cross-workspace injection, launcher()
├── application/       # Domain services and main-thread dispatch boundaries
├── model/             # m_graph, m_settings, spectra_store, m_io, m_mva, graph_style,
│                      #   m_plot_recipe(_store), m_style_template_store, peak_model, workspace_io …
├── viewmodel/         # vm_workspace_{spectra,maps,graphs}, vm_fit_model_builder, vm_mva, vm_settings, utils
├── view/
│   ├── v_workspace_{spectra,maps,graphs}.py
│   ├── components/    # 27 widgets: v_spectra_viewer, v_graph, v_map_viewer, v_fit_model_builder,
│   │   │              #   v_plot_renderer, v_data_filter, v_export_dialog, v_multipanel_dialog …
│   │   └── customize_graph/   # CustomizeGraphDialog + per-tab widgets + edit dialogs
│   └── theme/         # ThemeManager, QSS templates, light/dark/soft-dark
├── api/               # Public programmatic API (analysis, fitting, graphs, io, preprocessing, settings…)
├── fit_engine/        # Vectorized Batch Fit: vbf_engine, optimizer, models, baseline, vbf_thread …
├── ai_agent/          # Optional LLM chat: agent/ (AppContext + commands), mcp/ (server,
│                      #   hub, config), prompts/rules/knowledge/examples/, config/, utils/
└── resources/         # icons/, *.mplstyle, user_manual/
```

---

## 3. Workspaces & Inheritance

- **Spectra Workspace** (`v_/vm_workspace_spectra`): Base workspace for discrete 1D spectra.
- **Maps Workspace** (`v_/vm_workspace_maps`): Subclasses both `VWorkspaceSpectra` and `VMWorkspaceSpectra` using the template-method pattern. Always inspect parent implementation before overriding methods.
- **Graphs Workspace** (`v_/vm_workspace_graphs`): Standalone plotting workspace based on MDI subwindows hosting `VGraph` widgets. See [.agents/rules/graphs-workspace.md](graphs-workspace.md).
- **Cross-Workspace Wiring**: Centralized in `main.py::setup_connections()` (e.g. Maps $\rightarrow$ Graphs reference injection, `send_spectra_to_workspace`, `fit_results_updated` $\rightarrow$ Graphs). Wire workspaces here, never by reaching across widget hierarchies.

---

## 4. Data Backbone

- **`model/spectra_store.py`**: Tensor-centric `SpectraStore` and `MapData` structures. Single source of truth for all spectral arrays. See [.agents/rules/spectra-maps-workspace.md](spectra-maps-workspace.md).
- **`model/m_graph.py`**: `MGraph` dataclass is the plot configuration model. Its fields define the schema from which graph settings, undo/redo diffs, and dialogs derive. See [.agents/rules/graphs-workspace.md](graphs-workspace.md).
- **`model/m_settings.py`**: `MSettings` wraps `QSettings("CEA-Leti", "SPECTROview")` for persistent app configuration.

---

## 5. Startup Import Deferral

To ensure fast application startup:
- Heavy scientific libraries (`scipy.*`) and LLM SDKs (`anthropic`, `openai`, `ollama`, `mcp`) must **never** be imported at top-level module scope.
- Import them lazily inside the functions that require them.
- `scipy` is prewarmed in a background thread by `Main._prewarm_heavy_imports()` after the UI is rendered.
- Check with `python -X importtime -c "import spectroview.main"`; enforced by `tests/performance/test_startup_imports.py`.

---

## 6. Public API Stability

Public APIs in `spectroview/api/` (`spectra.py`, `2dmap.py`, `graphs.py`, `fitting.py`, etc.) are stable contracts:
- Never rename, reorder, or remove parameters (even unused ones) without explicit instruction.
- Positional call sites in external scripts, tests, and the AI agent depend on these signatures.
