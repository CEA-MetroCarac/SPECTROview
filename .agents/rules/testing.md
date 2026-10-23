---
name: testing
description: Testing conventions, test execution commands, headless Qt setup, conftest fixtures, regression prevention, and validation gates.
trigger: model_decision
---

# Testing & Quality Assurance

## 1. Test Execution Commands

Always use the project's real test commands:

```powershell
# Fast test suite (skips slow-marked real-data benchmarks; ~1-2 min)
python -m pytest tests/ -m "not slow" -q

# Run a specific subtree or file
python -m pytest tests/unit/fit_engine/ -q
python -m pytest tests/unit/view/ -q
python -m pytest tests/integration/ -q

# Run a single test method
python -m pytest tests/unit/fit_engine/test_optimizer.py::TestBounds::test_solution_respects_tight_bounds -v

# Full suite (including real benchmark datasets; ~7-8 min)
python -m pytest tests/ -q

# Static syntax and import check
python -m pyflakes spectroview/<modified_file>.py
```

Configuration is defined in `pyproject.toml` (`[tool.pytest.ini_options]`): `testpaths=["tests"]`, `slow` marker registered.

---

## 2. Headless Qt & Test Isolation

### Headless Platform (`QT_QPA_PLATFORM=offscreen`)
`tests/conftest.py` configures `QT_QPA_PLATFORM=offscreen` **before** any `QApplication` instance is created. This allows hundreds of native Qt widgets and matplotlib canvases to be created in a single process without exhausting the operating system window handles (which would cause a hard segfault).
- **Ad-hoc Scripts**: When running ad-hoc Python scripts touching Qt, always set:
  ```powershell
  $env:QT_QPA_PLATFORM="offscreen"
  ```

### Key Shared Fixtures (`tests/conftest.py`)
- **`qapp`**: Shared session-scoped `QApplication` for all GUI tests.
- **`_isolate_qsettings` (autouse)**: Monkeypatches `QSettings` inside `spectroview.model.m_settings` to route every call through an isolated per-test INI file. Tests never read or modify the real Windows user registry.
- **`_release_qt_widgets_between_tests` (autouse)**: Flushes deferred Qt deletions (`deleteLater()`) and performs generational garbage collection between GUI tests to keep native handle consumption bounded.
- **Data & Model Builders**: `make_synthetic_spectrum`, `make_synthetic_map`, `make_fit_model`, `dataframe_excel_file`, `sample_dataframe`, `bench_dir`.

---

## 3. What to Test & House Patterns

- **Add Tests Alongside Code Changes**: Every new feature or bug fix must include corresponding tests in `tests/unit/` or `tests/integration/`.
- **Prefer Real Fixtures Over Mocking**: Construct real Model, ViewModel, and View objects against synthetic data builders or sample fixtures rather than mocking internal units. Only mock modal blocking dialogs (`QColorDialog`, `QMessageBox`, `QFileDialog`) that would block headless test execution.
- **Testing Views**:
  - Build widgets with plotted data, invoke handlers, and assert on emitted signals and Model state changes.
  - In MDI workspace testing, headless mode does not automatically activate subwindows: invoke `mdi_area.setActiveSubWindow(sub_window)` explicitly.
- **Geometry & Annotations**:
  - Size test annotations as realistic fractions of current axis data limits to prevent coordinates from collapsing off-axis.
- **Visual Checks**:
  - For visual rendering correctness, capture offscreen rendering via `QWidget.grab().save("temp.png")` and inspect the output.
- **Schema & Round-Trip Invariants**:
  - If adding fields to `MGraph` (`model/m_graph.py`), update `ALL_SAVE_KEYS` and `CUSTOM_VALUES` in `tests/unit/model/test_m_graph.py`. The `TestModelSchemaSync` test locks this invariant.
- **Fit Engine Math**:
  - Any new peak model in `spectroview/fit_engine/models.py` must be added to `_SAMPLE_PARAMS` in `tests/unit/fit_engine/test_models.py` (scalar parity, Jacobians, and degenerate widths are tested automatically).
