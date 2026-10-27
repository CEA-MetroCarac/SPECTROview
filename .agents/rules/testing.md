---
name: testing
description: Testing conventions, test execution commands, headless Qt setup, conftest fixtures, regression prevention, and validation gates.
trigger: model_decision
---

# Testing & Quality Assurance

## 1. Feature & Test Synchronization Mandate

**Strict Rule**: Every new feature, UI component, dialog, ViewModel method, peak model, or public API capability MUST be accompanied by new or updated unit/integration tests in the exact same task. Code without passing test coverage is incomplete and must not be committed.

### Layer-by-Layer Test Authoring Requirements

| Layer / Feature | Target Location | Mandatory Test Scenarios |
|---|---|---|
| **View / Widget / Dialog** | `tests/unit/view/` | • Widget instantiation without segfaults.<br>• Interaction triggers expected signals or ViewModel methods.<br>• Reacts correctly to incoming ViewModel signals.<br>• Mock modal dialogs (`QMessageBox`, `QFileDialog`, `QColorDialog`) with `monkeypatch`.<br>• Clean teardown without unparented timers or leaked event filters. |
| **ViewModel** | `tests/unit/viewmodel/` | • Method state transitions and data transformations.<br>• Signal emissions verified via `qtbot.waitSignal` or callback lists.<br>• Robust handling of edge cases (empty stores, NaNs, missing columns). |
| **Model & Schema** | `tests/unit/model/` | • Invariant and schema validation.<br>• Serialization & deserialization round-trip.<br>• If modifying `MGraph`, update `ALL_SAVE_KEYS` and `CUSTOM_VALUES` in `tests/unit/model/test_m_graph.py` (`TestModelSchemaSync`). |
| **Fit Engine** | `tests/unit/fit_engine/` | • Analytical Jacobians vs numerical finite differences.<br>• Batched `(N, M, K)` tensor evaluation vs scalar implementation parity.<br>• Any new peak model MUST be registered in `_SAMPLE_PARAMS` in `tests/unit/fit_engine/test_models.py`. |
| **File Loaders (`m_io`)** | `tests/unit/model/test_m_io.py` | • Correct extraction of spectral arrays, coordinates, and metadata.<br>• Graceful failure and error messages on corrupt or empty files. |
| **Public API** | `tests/unit/api/` | • Exact positional and keyword argument compatibility.<br>• Return types match API contracts. |

---

## 2. Test Execution Commands

Always use the project's real test commands:

```powershell
# Fast test suite (skips slow-marked real-data benchmarks; ~1 min)
python -m pytest tests/ -m "not slow" -q

# Run a specific subtree or file
python -m pytest tests/unit/fit_engine/ -q
python -m pytest tests/unit/view/ -q
python -m pytest tests/integration/ -q

# Run a single test method
python -m pytest tests/unit/fit_engine/test_optimizer.py::TestBounds::test_solution_respects_tight_bounds -v

# Full suite (all 1,600+ tests; ~1-1.5 min)
python -m pytest tests/ -q

# Static syntax and import check
python -m pyflakes spectroview/<modified_file>.py
```

Configuration is defined in `pyproject.toml` (`[tool.pytest.ini_options]`): `testpaths=["tests"]`, `slow` marker registered.

---

## 3. Headless Qt & Test Isolation

### Headless Platform (`QT_QPA_PLATFORM=offscreen`)
`tests/conftest.py` configures `QT_QPA_PLATFORM=offscreen` **before** any `QApplication` instance is created. This allows hundreds of native Qt widgets and matplotlib canvases to be created in a single process without exhausting operating system window handles.
- **Ad-hoc Scripts**: When running ad-hoc Python scripts touching Qt, always set:
  ```powershell
  $env:QT_QPA_PLATFORM="offscreen"
  ```

### Key Shared Fixtures (`tests/conftest.py`)
- **`qapp`**: Shared session-scoped `QApplication` for all GUI tests.
- **`_isolate_qsettings` (autouse)**: Monkeypatches `QSettings` inside `spectroview.model.m_settings` to route every call through an isolated per-test INI file. Tests never read or modify the real user registry.
- **`_release_qt_widgets_between_tests` (autouse)**: Flushes deferred Qt deletions (`deleteLater()`), closes leaked top-level widgets (`widget.close()`), closes matplotlib figures (`plt.close('all')`), and performs generational garbage collection between GUI tests.
- **Data & Model Builders**: `make_synthetic_spectrum`, `make_synthetic_map`, `make_fit_model`, `dataframe_excel_file`, `sample_dataframe`, `bench_dir`.

### PySide6 Lifecycle Safety in Tests
- **Context-Bound Timers**: Single-shot timers MUST always specify a context `QObject`: `QTimer.singleShot(msec, context, slot)`. Never use the 2-argument form with lambdas referencing widget state. Unparented timers fire on the global Qt event loop after widget destruction, causing `RuntimeError: Internal C++ object already deleted` during subsequent tests.
- **Mock Modal Dialogs**: Never allow blocking dialogs to execute in headless tests. Use `monkeypatch`:
  ```python
  monkeypatch.setattr(QMessageBox, "exec", lambda *args: QMessageBox.StandardButton.Ok)
  monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: ("/tmp/test.csv", "CSV Files (*.csv)"))
  ```
- **Event Filter Cleanup**: Any test or component that installs an event filter on `QApplication.instance()` must remove it upon teardown or verify `isValid()` to avoid $O(N \times M)$ cross-language traversal slowdowns.

---

## 4. What to Test & House Patterns

- **Prefer Real Fixtures Over Mocking**: Construct real Model, ViewModel, and View objects against synthetic data builders or sample fixtures rather than mocking internal units.
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
