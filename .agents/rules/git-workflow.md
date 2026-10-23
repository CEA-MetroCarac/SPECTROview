---
name: git-workflow
description: Git commit conventions, milestone-based commits, Conventional Commits format, atomic staging, and repository safety rules.
trigger: always_on
---

# Git Workflow & Repository Safety

## 1. Milestone-Based Commits

- **Commit Completed Milestones**: Commit every completed MVP or logically independent development milestone. Do not accumulate days of work into a single uncommitted working tree.
- **Completion Definition**: A milestone is ready to commit only when:
  1. The code changes are complete, cohesive, and follow project architecture.
  2. Appropriate new or updated unit tests pass cleanly.
  3. Relevant documentation across Developer Guide, Public API, and User Manual has been updated in the same task.
  4. The working tree has been reviewed (`git status` and `git diff`).

---

## 2. Conventional Commit Convention

Follow the repository's established Conventional Commits pattern:

```text
<type>(<scope>): <concise subject in present tense>

- <Bullet 1: Concrete change and rationale>
- <Bullet 2: Interface, architecture, or module adjustments>
- <Bullet 3: Tests added or updated>
- <Bullet 4: Documentation synchronized>
```

### Supported Types
- `feat`: New feature, plot style, loader, UI widget, or public API capability.
- `fix`: Bug fix, error resolution, edge-case fix, or regression fix.
- `perf`: Performance optimizations (e.g. vectorization, caching, indexing).
- `test`: Adding or updating unit, integration, or visual test suites.
- `docs`: Documentation updates across developer guide, API, or user manual.
- `refactor`: Structural changes preserving all existing behavior and public contracts.
- `chore`: Maintenance, version bumps, or build script updates.

### Recognized Scopes
- `(view)`: General GUI widgets, theme styling, dialogs, viewers.
- `(viewmodel)`: ViewModel orchestrators and signal wiring.
- `(model)`: Core data models, `SpectraStore`, `MGraph`, `MSettings`.
- `(fit_engine)`: Vectorized Batch Fit engine, optimizer, Jacobians, models.
- `(io)`: File loaders, parsers (SPC, WDF, TXT, CSV), export utilities.
- `(graphs)`: Graphs workspace, plot rendering, customization dialogs.
- `(spectra)`: Spectra workspace specific logic, baseline, peak builder.
- `(maps)`: Maps workspace, hyperspectral coordinates, heatmaps.
- `(api)`: Public programmatic API (`spectroview/api/`).
- `(ai_agent)`: In-app AI chat agent, MCP server, system prompts.

---

## 3. Atomic Staging & Clean Working Tree

- Before staging, run `git status` to verify the state of the workspace.
- Stage only files related to the completed milestone:
  ```powershell
  git add <file1> <file2> ...
  ```
- **Never stage**:
  - Cache directories (`.pytest_cache/`, `.ruff_cache/`, `__pycache__/`, `.cache/`).
  - Virtual environments (`.venv/`, `env/`).
  - Build or distribution artifacts (`build/`, `dist/`, `spectroview.egg-info/`, `site/`).
  - Temporary files, test outputs, or user logs.
  - Personal configuration or local files (`.claude/settings.local.json`).

---

## 4. Repository Safety Constraints

1. **No Destructive Operations**:
   - Never use `git reset --hard`, `git clean -f`, or history-altering commands.
   - Never amend commits that have already been pushed to a shared remote.
   - Never force-push (`git push --force`).
2. **No Remote Push Without Authorization**:
   - Committing locally to the working branch is required for completed milestones.
   - Pushing to remote (`git push origin ...`) requires explicit user confirmation.
3. **Preserve User Changes**:
   - Never overwrite, discard, stage, or commit unrelated user files or edits.
