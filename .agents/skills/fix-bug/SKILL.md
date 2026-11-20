---
name: fix-bug
description: >-
  Use this skill when diagnosing, reproducing, and fixing bugs, crashes, segfaults, calculation errors, or UI glitches in SPECTROview.
---

# Bug Fixing Workflow

Follow this procedure to eliminate bugs at their root cause while safeguarding against regressions and segmentation faults.

---

## Phase 1: Understand & Reproduce

1. **Inspect Logs & Error Messages**: Check traceback, terminal output, and `git status`.
2. **Identify Root Cause**:
   - For hard crashes/segfaults: Check [.agents/rules/pitfalls.md](../../rules/pitfalls.md) (unparented `QTimer`/`QObject`, headless surface budget, PyQt5 import mixing, or macOS LAPACK stack limits).
   - For coordinate/picking bugs: Check twin-axis data transform via `transData.inverted().transform((event.x, event.y))`.
   - For fitting divergence: Check initial parameter estimation (`p0`), bounds, and analytical Jacobian formulas against `numerical_jacobian()`.
   - For UI freezes: Check if long-running operations are executing on the main GUI thread instead of worker threads (`VBFthread`).
3. **Write a Failing Regression Test**:
   - Author a test case in the appropriate `tests/unit/` or `tests/integration/` module reproducing the bug.
   - Run the test to confirm it fails on the unfixed codebase:
     ```powershell
     python -m pytest tests/unit/<path>/test_<module>.py -k <test_name> -v
     ```

---

## Phase 2: Implement Minimal Fix

1. Apply the smallest complete fix that resolves the issue at the source without unnecessary refactoring.
2. Ensure backward compatibility and preserve adjacent features.
3. Adhere to [.agents/rules/coding-standards.md](../../rules/coding-standards.md) and MVVM separation.

---

## Phase 3: Verify & Prevent Regressions

1. Run the regression test to verify resolution:
   ```powershell
   python -m pytest tests/unit/<path>/test_<module>.py -k <test_name> -v
   ```
2. Run the fast test suite to guarantee zero side effects:
   ```powershell
   python -m pytest tests/ -m "not slow" -q
   ```
3. Run static check on changed files:
   ```powershell
   python -m pyflakes spectroview/<modified_file>.py
   ```

---

## Phase 4: Synchronize Bilingual Documentation

If the bug fix alters documented behavior, parameters, or fixes an error in the manual:

- Update **both** English (`*.md`) and French (`*.fr.md`) counterparts in `docs/developer/`, `docs/api/`, or `docs/user_manual/` per [.agents/rules/documentation.md](../../rules/documentation.md) and verify with `tests/unit/test_docs_i18n.py` and `mkdocs build --strict`.

---

## Phase 5: Commit & Report

1. Review diff with `git diff`.
2. Commit using Conventional Commits according to [.agents/rules/git-workflow.md](../../rules/git-workflow.md):
   ```powershell
   git commit -m "fix(<scope>): <concise summary>" -m "- <Bullet 1: Root cause analysis and resolution>" -m "- <Bullet 2: Regression test added>" -m "- <Bullet 3: Documentation synchronized>"
   ```
3. Report the fix, verified tests, and root cause analysis.
