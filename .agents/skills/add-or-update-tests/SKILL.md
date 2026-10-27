---
name: add-or-update-tests
description: >-
  Use this skill when authoring new unit, integration, or visual tests, expanding test coverage, or testing edge cases in SPECTROview.
---

# Test Authoring & Expansion Workflow

Follow this procedure to expand test coverage and build deterministic, maintainable test suites for SPECTROview.

---

## Phase 1: Identify Target Behaviors

1. Review the requirement, peak model, ViewModel method, or UI component to test.
2. Consult [.agents/rules/testing.md](../../rules/testing.md) and [tests/README.md](../../tests/README.md).

---

## Phase 2: Inspect Existing Test Patterns & Fixtures

Review test patterns in the relevant subdirectory:
- **`tests/unit/fit_engine/`**: Pure NumPy mathematical tests (`test_models.py`, `test_optimizer.py`, `test_evaluator.py`, `test_baseline.py`).
- **`tests/unit/model/`**: `SpectraStore`, settings, file I/O, `MGraph` schema sync (`test_spectra_store.py`, `test_m_graph.py`).
- **`tests/unit/viewmodel/`**: ViewModel signals and data transformations (`test_vm_workspace_spectra.py`, `test_vm_workspace_maps.py`, `test_vm_workspace_graphs.py`).
- **`tests/unit/view/`**: Headless matplotlib plotting and dialogs (`test_v_graph_plotting.py`, `test_customize_graph_dialog.py`).
- **`tests/integration/`**: Cross-layer workflows (`test_spectra_workflow.py`, `test_maps_workflow.py`, `test_graphs_workflow.py`).

### Key Fixtures Available (`tests/conftest.py`)
- `qapp`: Session-scoped `QApplication` for Qt tests.
- `_isolate_qsettings`: Autouse per-test INI redirection.
- `make_synthetic_spectrum`, `make_synthetic_map`, `make_fit_model`: Deterministic synthetic builders with known ground truth parameters.
- `dataframe_excel_file`, `sample_dataframe`: Realistic wafer/map test data.

---

## Phase 3: Author Tests

1. Create a new test file in `tests/unit/<layer>/test_<target>.py` or add to an existing file.
2. Structure each test with:
   - **Setup**: Use fixtures and synthetic builders rather than mocking the unit under test.
   - **Execution**: Call ViewModel methods or drive View handlers.
   - **Assertions**: Assert on emitted Qt signals, Model array values, or dataclass attributes.
3. **Headless Execution & Qt Lifetime Rules**:
   - Ensure `QT_QPA_PLATFORM=offscreen` is active.
   - In MDI tests, explicitly call `mdi_area.setActiveSubWindow(sub_window)`.
   - **Mock Modal Dialogs**: Never allow modal blocking dialogs to execute in headless tests. Mock them with `monkeypatch`:
     ```python
     monkeypatch.setattr(QMessageBox, "exec", lambda *args: QMessageBox.StandardButton.Ok)
     monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: ("/tmp/test.csv", "CSV Files (*.csv)"))
     monkeypatch.setattr(QColorDialog, "getColor", lambda *args: QColor(255, 0, 0))
     ```
   - **Clean Teardown**: Top-level widgets instantiated in tests should be closed or scheduled for deferred deletion (`widget.close()`, `widget.deleteLater()`).
   - **Signal Waiting**: Use `qtbot.waitSignal(signal, timeout=...)` or connect to a tracking list rather than arbitrary `time.sleep()`.

---

## Phase 4: Execute & Validate

Run the targeted test:
```powershell
python -m pytest tests/unit/<layer>/test_<target>.py -v
```
Run the fast test suite to guarantee zero side effects:
```powershell
python -m pytest tests/ -m "not slow" -q
```

---

## Phase 5: Commit & Report

Commit tests following [.agents/rules/git-workflow.md](../../rules/git-workflow.md):
```powershell
git commit -m "test(<scope>): cover <target behavior>" -m "- <Bullet 1: Test scenarios covered>" -m "- <Bullet 2: Fixtures and assertions added>"
```
