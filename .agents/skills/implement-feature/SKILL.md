---
name: implement-feature
description: >-
  Use this skill when implementing a new feature, plot style, file loader, UI component, or public API capability in SPECTROview.
---

# Feature Implementation Workflow

Follow this procedure to implement new features in SPECTROview while preserving MVVM architecture, vectorization, PySide6 safety, tests, and documentation.

---

## Phase 1: Understand & Inspect

1. **Review Architecture**: Consult [.agents/rules/architecture.md](../../rules/architecture.md) and [.agents/rules/coding-standards.md](../../rules/coding-standards.md).
2. **Check Git Status**: Run `git status` to verify a clean working tree and identify any uncommitted user work.
3. **Analyze Affected Modules**:
   - For Spectra/Maps: Inspect `spectroview/viewmodel/vm_workspace_spectra.py`, `spectroview/model/spectra_store.py`, and `spectroview/view/components/`.
   - For Graphs: Inspect `spectroview/model/m_graph.py`, `spectroview/view/components/v_plot_renderer.py`, and `spectroview/view/components/customize_graph/`.
   - For Fitting: Inspect `spectroview/fit_engine/models.py`, `evaluator.py`, and `optimizer.py`.
   - For File Loaders: Inspect `spectroview/model/m_io.py`.
   - For Public API: Inspect `spectroview/api/`.
4. **Identify Invariants**:
   - Verify MVVM boundaries (ViewModel never touches View; View communicates via signals/slots).
   - Ensure all array processing remains vectorized (no Python loops on hot paths).
   - Ensure `QTimer` and helper `QObject`s are parented to their host widget.

---

## Phase 2: Plan the Implementation

Outline:
- Backing models or schemas to update (e.g. `MGraph`, `SpectraStore`).
- ViewModel signals and methods.
- View components, dialogs, or keyboard shortcuts.
- Unit/integration tests to write in `tests/unit/` or `tests/integration/`.
- Documentation surfaces to update (Developer guide, Public API, User manual).

---

## Phase 3: Implement

1. **Write Clean, Idiomatic Code**:
   - Keep heavy packages (`scipy.*`, LLMs) off startup imports.
   - Use theme variables and ensure dark-theme compatibility.
   - Connect View $\leftrightarrow$ ViewModel strictly via Qt signals.
   - For bulk operations, implement the `_apply_with_ctrl(fn)` pattern.
   - **Qt Lifetime**: Always parent `QTimer`, event filters, and helpers. For single-shot timers, ALWAYS use `QTimer.singleShot(msec, context, slot)`; never use 2-arg lambdas referencing widgets.
2. **Handle Edge Cases**:
   - Check boundary conditions (empty datasets, single-spectrum vs map data, NaNs, missing columns).
   - Provide graceful non-blocking feedback (status bar, notifications).

---

## Phase 4: Add Unit & Integration Tests (Mandatory)

**Strict Policy**: Do not defer test writing. Every new feature, component, dialog, or method MUST be accompanied by new or updated tests authored in the same task before completing the milestone.

1. **Author Matching Tests**:
   - For Views / Dialogs: Test instantiation, interactions, signal emissions, and clean teardown in `tests/unit/view/`. Mock modal dialogs (`QMessageBox`, `QFileDialog`, `QColorDialog`) with `monkeypatch` so headless execution never blocks.
   - For ViewModels: Test methods, state transformations, and signal emission sequences in `tests/unit/viewmodel/`.
   - For Models: Test schema validation, serialization, and round-tripping in `tests/unit/model/`. If adding fields to `MGraph`, update `ALL_SAVE_KEYS` in `tests/unit/model/test_m_graph.py` to keep `TestModelSchemaSync` passing.
   - For Fit Engine: Test peak models in `tests/unit/fit_engine/test_models.py` (register in `_SAMPLE_PARAMS`).
   - For Public API: Test signatures and contracts in `tests/unit/api/`.
2. **Execute Targeted and Fast Suites**:
   ```powershell
   # Run the newly authored test
   python -m pytest tests/unit/view/test_<new_feature>.py -v

   # Run fast test suite to ensure zero regressions
   python -m pytest tests/ -m "not slow" -q
   ```

---

## Phase 5: Synchronize Documentation

Update the authoritative documents mapped in [.agents/rules/documentation.md](../../rules/documentation.md):
- Developer Guide: `docs/developer/<area>.md`
- User Manual: `spectroview/resources/user_manual/<area>.md`
- Public API: `docs/api/<area>.md` (if API surface expanded)
- Shortcuts: `spectroview/resources/user_manual/shortcuts.md` (if shortcut added)

---

## Phase 6: Validate Quality Gates

1. Run the fast test suite:
   ```powershell
   python -m pytest tests/ -m "not slow" -q
   ```
2. Run static check on changed files:
   ```powershell
   python -m pyflakes spectroview/<modified_file>.py
   ```
3. Inspect diff for unintended modifications:
   ```powershell
   git diff
   ```

---

## Phase 7: Commit & Report

Commit the completed feature according to [.agents/rules/git-workflow.md](../../rules/git-workflow.md):
```powershell
git add spectroview/ tests/ docs/
git commit -m "feat(<scope>): <concise summary>" -m "- <Bullet 1: Concrete change and rationale>" -m "- <Bullet 2: Tests added>" -m "- <Bullet 3: Documentation synchronized>"
```
Produce a clear report summarizing the delivered feature, tests executed, and documentation updated.
