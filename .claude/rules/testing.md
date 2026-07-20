---
paths: ["tests/**/*.py", "**/conftest.py"]
---

# Testing

## Running

```bash
python -m pytest tests/ -q                 # full suite (~8 min; ~1130+ tests)
python -m pytest tests/unit/view -q        # a subtree
python -m pytest "tests/unit/view/test_v_graph_plotting.py::TestToolbarIconsAndRescale" -q
python -m pytest -m "not slow" -q          # skip slow-marked tests
```

Config is in `pyproject.toml` (`[tool.pytest.ini_options]`): `testpaths=tests`,
`Test*`/`test_*` discovery, `slow` marker. Layout mirrors the package:
`tests/unit/{model,viewmodel,view,api,fit_engine,ai_agent}`, plus
`tests/integration` and `tests/performance`.

## Headless Qt (important)

`tests/conftest.py` sets `QT_QPA_PLATFORM=offscreen` **before** any
`QApplication` is built — this is what lets hundreds of real widgets/canvases be
created in one process without exhausting the platform's native surface budget
(otherwise: segfault). Don't undo it. When running an ad-hoc script that touches
Qt, set `QT_QPA_PLATFORM=offscreen` yourself.

Key fixtures (all in `conftest.py`):
- `qapp` — session `QApplication` for anything touching Qt.
- `_isolate_qsettings` (autouse) — redirects every `QSettings` to a throwaway
  per-test INI so tests never read/write the real user registry. See the
  QSettings note in [pitfalls.md](pitfalls.md).
- `_release_qt_widgets_between_tests` (autouse) — GC + pending `deleteLater()`
  after each test to bound live native widgets.
- Data/builders: `examples_dir`, `bench_dir`, `dataframe_excel_file`,
  `make_synthetic_spectrum`, `make_synthetic_map`, `make_fit_model`,
  `sample_dataframe`, and real sample files (`wdf_map_file`, `wafer_file`, …).

## What to test & how (house patterns)

- **Add/adjust tests alongside any logic change.** Mirror the existing style in
  the nearest `test_*.py`: construct a real Model/ViewModel/View against real
  sample data (from fixtures) rather than mocking the unit under test; only mock
  modal dialogs (`QColorDialog`, `QMessageBox`, `QFileDialog`) that would block
  headless.
- **Views**: build the widget from a real (already-plotted) graph/spectrum,
  drive the handler, assert on model state and emitted signals. For MDI graphs,
  headless doesn't auto-activate a subwindow — call
  `mdi_area.setActiveSubWindow(sub_window)` explicitly (see
  `TestSharedGraphToolbar`, `TestKeyboardShortcuts`).
- **Geometry/interaction** (e.g. annotation resize handles): size test
  annotations as a real fraction of the current axis view — hardcoded data
  coordinates can fall off-axis where distinct edges collapse to the same pixel.
- **Visual checks**: for something whose correctness is visual, render offscreen
  with `QWidget.grab().save(path)` and actually open the PNG to inspect it — do
  not assert only that it didn't crash.
- **Contracts**: schema/round-trip invariants have dedicated tests
  (`TestModelSchemaSync` keeps `VGraph` seeded defaults in lockstep with
  `MGraph`; save-key lists in `test_m_graph.py`). If you add an `MGraph` field or
  a persisted key, update those.

## Static check

No repo linter config is enforced. Run `python -m pyflakes <changed files>`
before finishing; `ruff`/`vulture` are not installed here.
