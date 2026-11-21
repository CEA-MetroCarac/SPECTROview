---
name: complete-mvp
description: >-
  Use this skill when declaring a development milestone or MVP complete, verifying all quality gates, and committing in SPECTROview.
---

# Complete MVP & Milestone Workflow

Follow this procedure to finalize, validate, and commit a development milestone or MVP in SPECTROview.

---

## Phase 1: Verify Scope & Acceptance Criteria

1. Check that all agreed requirements and acceptance criteria have been satisfied.
2. Confirm that no temporary debug prints, test hardcoded values, or unfinished `TODO` placeholders remain.
3. Review MVVM boundaries, PySide6 safety rules, and multilingual i18n coverage ([.agents/rules/i18n.md](../../rules/i18n.md)).

---

## Phase 2: Run Quality & Test Gates

Execute all verification checks:

```powershell
# 1. Fast test suite (includes test_i18n.py catalog and UI literal audits)
python -m pytest tests/ -m "not slow" -q

# 2. Static syntax and import check on modified files
python -m pyflakes spectroview/<modified_file>.py

# 3. Release, packaging & startup-import gate (when modules moved/renamed, entry points/dependencies changed, or preparing a release)
python -m pytest tests/unit/test_release_workflow.py tests/unit/test_packaging_windows.py tests/unit/test_packaging_macos.py tests/unit/test_packaging_linux.py tests/performance/test_startup_imports.py -v
```
All applicable checks must pass with zero failures, and `.github/workflows/release.yml`, `main.spec`, `packaging/*`, and `pyproject.toml` must contain zero stale module or asset paths.

---

## Phase 3: Confirm Multilingual Documentation Synchronization

Confirm that all relevant documentation surfaces have been updated across **all five languages (`*.md` and `*.{fr,de,ru,zh}.md`)** (with passing `test_markdown_syntax.py`, `test_docs_i18n.py`, `docs/i18n_check.py`, and `mkdocs build --strict`) per [.agents/rules/documentation.md](../../rules/documentation.md):

- Developer Guide: `docs/developer/*.md` & `*.{fr,de,ru,zh}.md`
- User Manual: `docs/user_manual/*.md` & `*.{fr,de,ru,zh}.md`
- Public API: `docs/api/*.md` & `*.{fr,de,ru,zh}.md`

---

## Phase 4: Stage & Commit

1. Inspect working tree status:
   ```powershell
   git status
   ```
2. Review complete diff:
   ```powershell
   git diff
   ```
3. Stage only relevant files:
   ```powershell
   git add <file1> <file2> ...
   ```
4. Commit following [.agents/rules/git-workflow.md](../../rules/git-workflow.md):
   ```powershell
   git commit -m "<type>(<scope>): <concise summary>" -m "- <Bullet 1: Concrete changes and technical rationale>" -m "- <Bullet 2: Interface or module adjustments>" -m "- <Bullet 3: Tests verified>" -m "- <Bullet 4: Documentation synchronized>"
   ```
5. Verify the commit was recorded:
   ```powershell
   git log -1 --stat
   ```

---

## Phase 5: Produce Completion Report

Provide a clear summary including:

- Summary of delivered capabilities.
- Test and static check verification results.
- Documentation updated.
- Commit hash and message.
- Any known limitations or next steps.
