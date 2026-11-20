---
name: review-code
description: >-
  Use this skill when conducting a comprehensive code review, quality audit, MVVM boundary check, or security check before committing in SPECTROview.
---

# Code Review & Quality Audit Workflow

Follow this procedure to conduct a rigorous review against SPECTROview architecture, performance, and stability standards.

---

## Phase 1: Diff & Scope Inspection

1. Run `git status` and `git diff` to view all uncommitted modifications.
2. Confirm changes are strictly scoped to the task at hand.
3. Check for unintended files (e.g. `__pycache__`, `.pytest_cache`, temp Excel files, `.cache`, local settings).

---

## Phase 2: Architectural & Quality Audit

Evaluate against project rules:

1. **MVVM Separation** ([.agents/rules/architecture.md](../../rules/architecture.md)):
   - Does any ViewModel import or reference View classes?
   - Do Views communicate with ViewModels exclusively via signals and slots?
   - Are Models completely free of Qt widget references?
2. **PySide6 Thread Safety & Lifetime** ([.agents/rules/coding-standards.md](../../rules/coding-standards.md) & [.agents/rules/pitfalls.md](../../rules/pitfalls.md)):
   - Are all GUI manipulations running strictly on the main Qt thread?
   - Are worker threads (`VBFthread`) communicating with UI exclusively via signals?
   - Are all `QTimer`, event filters, and helper `QObject`s parented to the widget they serve?
3. **Vectorization & Performance** ([.agents/rules/fit-engine.md](../../rules/fit-engine.md)):
   - Are calculations vectorized using NumPy/pandas arrays without per-spectrum Python loops?
   - Are heavy imports (`scipy.*`, LLMs) kept off top-level module scope?
4. **Public API Contracts**:
   - Are public API signatures in `spectroview/api/` completely preserved without renaming or dropping parameters?
5. **Testing Coverage** ([.agents/rules/testing.md](../../rules/testing.md)):
   - Are unit or integration tests present for all new behaviors or bug fixes?
   - Do tests use headless offscreen mode safely without native handle leaks?
   - Are schema sync tests (`TestModelSchemaSync`) green?
6. **Internationalization (i18n) Parity** ([.agents/rules/i18n.md](../../rules/i18n.md)):
   - Do all new/modified GUI and ViewModel user strings use `spectroview.i18n` (`tr`, `tr_plural`, `bind_*`) with matching keys in `spectroview/resources/i18n/fr.json`?
   - Are translated `QComboBox` items backed by canonical `Qt.UserRole` `itemData` (`combo_data_or_text` / `set_combo_by_data_or_text`)?
7. **Bilingual Documentation Synchronization** ([.agents/rules/documentation.md](../../rules/documentation.md)):
   - Have all affected documentation surfaces (Developer Guide, Public API, User Manual) been updated in **both English (`*.md`) and French (`*.fr.md`)** (plus `nav_translations` in `mkdocs.yml` for new pages)?
   - Do `test_markdown_syntax.py`, `test_docs_i18n.py`, and `mkdocs build --strict` pass without warnings or broken links?
8. **Release, CI & Packaging Integrity** ([.agents/rules/pitfalls.md](../../rules/pitfalls.md) & [.agents/rules/workflows.md](../../rules/workflows.md)):
   - Do `.github/workflows/release.yml`, `main.spec`, `packaging/*`, and `pyproject.toml` contain any stale import or asset paths after module moves/renames?
   - Does `spectroview/main.py` avoid PEP 562 lazy-facade entrypoint imports so PyInstaller `ModuleGraph` traces all runtime modules?

---

## Phase 3: Run Validation Gates

Run the verification suite:
```powershell
# 1. Fast test suite
python -m pytest tests/ -m "not slow" -q

# 2. Static check on modified files
python -m pyflakes spectroview/<modified_file>.py

# 3. Release, packaging & startup-import checks (when modules, imports, entry points, or packaging files changed)
python -m pytest tests/unit/test_release_workflow.py tests/unit/test_packaging_windows.py tests/unit/test_packaging_macos.py tests/unit/test_packaging_linux.py tests/performance/test_startup_imports.py -v
```

---

## Phase 4: Report Findings

Produce a structured review report:

- **Critical (Blockers)**: Segfault traps (unparented QTimers), MVVM violations, thread safety breaches, broken public APIs, stale CI/packaging imports, broken PyInstaller static reachability.
- **Warnings**: Unvectorized loops on large arrays, unparented helper objects, missing edge-case tests, stale docstrings.
- **Suggestions**: Code readability, dictionary dispatch refactoring, naming improvements.
- **Recommended Actions**: Clear steps required prior to final commit.
