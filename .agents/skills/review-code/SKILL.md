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
6. **Documentation Synchronization** ([.agents/rules/documentation.md](../../rules/documentation.md)):
   - Have all three documentation surfaces (Developer Guide, Public API, User Manual) been updated appropriately?

---

## Phase 3: Run Validation Gates

Run the verification suite:
```powershell
# 1. Fast test suite
python -m pytest tests/ -m "not slow" -q

# 2. Static check on modified files
python -m pyflakes spectroview/<modified_file>.py
```

---

## Phase 4: Report Findings

Produce a structured review report:
- **Critical (Blockers)**: Segfault traps (unparented QTimers), MVVM violations, thread safety breaches, broken public APIs.
- **Warnings**: Unvectorized loops on large arrays, unparented helper objects, missing edge-case tests, stale docstrings.
- **Suggestions**: Code readability, dictionary dispatch refactoring, naming improvements.
- **Recommended Actions**: Clear steps required prior to final commit.
