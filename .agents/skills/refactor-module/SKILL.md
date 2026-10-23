---
name: refactor-module
description: >-
  Use this skill when refactoring existing code, deduplicating logic, improving module structure, or optimizing performance in SPECTROview without altering external behavior.
---

# Code Refactoring Workflow

Follow this procedure to restructure and optimize code in SPECTROview while strictly preserving functionality, public API contracts, and test stability.

---

## Phase 1: Establish Test Baseline

1. Confirm a clean working tree: `git status`.
2. Run the relevant test suite **before modifying code** to establish a passing baseline:
   ```powershell
   python -m pytest tests/unit/<target_subsystem>/ -q
   ```
3. If existing tests are insufficient, author characterization tests first using [.agents/skills/add-or-update-tests/SKILL.md](../add-or-update-tests/SKILL.md).

---

## Phase 2: Plan the Refactoring

1. **Identify the Objective**:
   - Deduplicate logic (e.g. replacing long `if/elif` chains with dictionary dispatch maps).
   - Vectorize mathematical loops into batched NumPy/pandas matrix operations.
   - Separate concerns to restore clean MVVM boundaries.
   - Defer heavy top-level imports to improve startup time.
2. **Review Invariants**:
   - Confirm public APIs in `spectroview/api/` remain 100% backward-compatible.
   - Verify `MGraph` schema sync invariants (`TestModelSchemaSync`).
   - Verify zero widget references in `model/`.

---

## Phase 3: Execute Incremental Refactoring

1. Make small, logical refactoring steps rather than large all-at-once rewrites.
2. Keep methods focused on a single responsibility.
3. Remove unused variables, dead code, and redundant allocations.
4. Maintain surrounding code formatting and conventions.

---

## Phase 4: Validate Invariants & Tests

1. Run the target test suite after each step:
   ```powershell
   python -m pytest tests/unit/<target_subsystem>/ -v
   ```
2. Run the fast test suite:
   ```powershell
   python -m pytest tests/ -m "not slow" -q
   ```
3. Run static check:
   ```powershell
   python -m pyflakes spectroview/<modified_file>.py
   ```
4. If startup imports were affected, verify with:
   ```powershell
   python -m pytest tests/performance/test_startup_imports.py -v
   ```

---

## Phase 5: Synchronize Documentation

If classes, functions, or internal module boundaries moved:
- Update `docs/developer/index.md` or the corresponding subsystem developer guide.
- Check [.agents/rules/documentation.md](../../rules/documentation.md).

---

## Phase 6: Commit & Report

Commit the refactoring per [.agents/rules/git-workflow.md](../../rules/git-workflow.md):
```powershell
git commit -m "refactor(<scope>): <concise summary>" -m "- <Bullet 1: Structural change and motivation>" -m "- <Bullet 2: Verification across test suite>"
```
Produce a report detailing the improvements, preserved contracts, and executed validation checks.
