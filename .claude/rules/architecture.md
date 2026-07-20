# Architecture

On-demand orientation. The layer/import/naming rules are in `CLAUDE.md`; deep,
diagram-rich detail is in `docs/developer/index.md` (mkdocs). This file is the
module map + the wiring facts worth knowing before a structural change.

MVVM recap: View↔ViewModel talk **only via Qt signals/slots**; a ViewModel that
needs to reach a View should emit a signal instead. `model/` holds no Qt widget
references and is independently testable.

## Package map

```
spectroview/
├── __init__.py        # VERSION, constants (PLOT_STYLES, MARKERS, PALETTE, ICON_DIR…), QT_API=pyside6
├── main.py            # QMainWindow, tab wiring, cross-workspace injection, launcher()
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
├── ai_agent/          # Optional LLM chat (config/prompts/rules/knowledge/tools/mcp)
└── resources/         # icons/, *.mplstyle, user_manual/
```

## Workspaces

- **Spectra** (`v_/vm_workspace_spectra`) is the base. **Maps**
  (`v_/vm_workspace_maps`) subclasses both the View and the ViewModel and
  overrides hooks (template-method pattern) — read the parent before changing a
  Maps override.
- **Graphs** (`v_/vm_workspace_graphs`) is standalone plotting; see
  [graphs-workspace.md](graphs-workspace.md).
- Cross-workspace wiring is centralized in `main.py::setup_connections()`
  (Maps→Graphs reference injection, `send_spectra_to_workspace`,
  `fit_results_updated` → Graphs). Add cross-workspace links there, not by
  reaching across widget trees.

## Data backbone

- `model/spectra_store.py` — tensor-centric `SpectraStore` / `MapData`; the
  single source of truth for spectra/map arrays. See
  `docs/developer/spectra_store.md`.
- `model/m_graph.py` — `MGraph` dataclass: the plot configuration model. Its
  field set is the schema several Graphs helpers derive from (never hand-copy
  it — see [graphs-workspace.md](graphs-workspace.md)).
- `model/m_settings.py` — `MSettings` wraps `QSettings("CEA-Leti","SPECTROview")`
  for persistent app settings.

## Dependencies (see `pyproject.toml` / `requirements.txt`)

Core: `PySide6`, `matplotlib` (pinned `>=3.6.2,<3.10.9`), `pandas`,
`numpy<2.0`, `scipy`, `superqt` (range sliders), `pyarrow`, `openpyxl`,
`renishawWiRE` (WDF), `markdown`, `pyyaml`. Windows-only: `pywin32`. Optional AI:
`ollama`/`openai`/`anthropic`/`mcp` (commented out; `ai_agent` import is guarded).

## Build & run

- Entry point: console script `spectroview = spectroview.main:launcher`, or
  `python -m spectroview.main`. `main.py` sets `matplotlib.use('qtagg')`.
- Packaging: `main.spec` (PyInstaller); resource paths go through
  `spectroview.resource_path()` for frozen builds.
- Docs: `mkdocs` (`mkdocs.yml`, sources in `docs/`).
