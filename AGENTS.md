# SPECTROview — AI Development Guidelines (Google Antigravity)

Welcome to **SPECTROview**.
This repository is configured for pair programming with **Google Antigravity** and **Gemini**.
Follow these guidelines across all development sessions.

---

## 1. Project Purpose & Architecture Overview

`SPECTROview` is a high-performance desktop application for spectroscopic data processing, fitting, and visualization (discrete spectra, hyperspectral 2D/wafer maps, and general publication plotting). It includes a tensor-centric **Vectorized Batch Fit (VBF)** engine and an optional embedded AI data chat assistant.

### Architectural Core: Strict MVVM
The application strictly enforces Model-View-ViewModel separation across all workspaces:
- **`spectroview/model/`**: Pure data models and domain logic organized into subpackages (`spectra/`, `graphs/`, `io/`, `analysis/`, `system/`). Holds **no Qt widget references** and is independently testable.
  - `spectra/spectra_store.py`: Tensor-centric `SpectraStore` / `MapData` — the single source of truth for spectral arrays.
  - `graphs/m_graph.py`: `MGraph` dataclass — the canonical schema for plot configurations.
  - `m_settings.py`: `MSettings` wrapping `QSettings("CEA-Leti", "SPECTROview")`.
  - `io/m_io.py`: Multiformat loaders (TXT, CSV, WDF, SPC, L6S, NGS, TRPL, DAT).
- **`spectroview/viewmodel/`**: Business logic orchestrator organized into subpackages (`spectra/`, `maps/`, `graphs/`, `analysis/`, `common/`, plus `vm_settings.py` and `utils.py` compatibility facades).
  - Reads and mutates Models; communicates with Views **strictly via Qt signals/slots**.
  - **Never imports View classes.**
- **`spectroview/view/`**: PySide6/Qt widget layer organized into subpackages (`workspaces/` with `v_workspace_spectra.py`, `v_workspace_maps.py`, and `graphs/`; `components/` with `common/`, `customize_graph/`, `graphs/`, `lists/`, `lower_tabs/`, `map_viewer/`, `rgb_live_maps/`, `spectra_viewer/`, plus `v_settings.py` compat shim; and `theme/`).
  - Views connect to ViewModel signals and trigger ViewModel methods on user events.
- **`spectroview/fit_engine/`**: Tensor-based batched Levenberg-Marquardt optimizer (`vbf_engine.py`, `optimizer.py`, `models.py`, `vbf_thread.py`). Fits $N$ spectra simultaneously as `(N, M[, K])` tensors without per-spectrum Python loops.
- **`spectroview/api/`**: Public programmatic API (`spectra.py`, `2dmap.py`, `graphs.py`, `fitting.py`, `preprocessing.py`, `io.py`, `settings.py`). Backward compatibility is strictly maintained.
- **`spectroview/application/`**: Clean application service facade and thread-safe queued dispatch boundary.
- **`spectroview/ai_agent/`**: In-app LLM chat assistant feature with MCP server, tool calling, and local prompts.

### Fundamental Architectural Invariants
1. **MVVM Integrity**: View $\rightarrow$ ViewModel $\rightarrow$ Model one-direction dependency. ViewModel never imports View; Model never imports View or ViewModel.
2. **Vectorization Over Loops**: All data processing and fitting across spectra or maps must be vectorized via NumPy/pandas. Never introduce per-spectrum Python loops in hot paths.
3. **PySide6 Thread Safety & Object Lifetime**:
   - All GUI widget manipulations must execute on the main Qt thread.
   - Worker threads (e.g. `VBFthread`) communicate with UI strictly via Qt signals.
   - Always parent `QTimer`, event filters, and helper `QObject`s to their host widget to prevent segfaults when C++ objects are destroyed.
   - For single-shot timers, always use the 3-argument form with a context object: `QTimer.singleShot(msec, context_qobject, slot)`. Never fire unparented lambdas touching widget state.
   - Event filters installed on `QApplication` must cleanly auto-deregister when their target widget is destroyed to prevent $O(N \times M)$ event-loop slowdowns.
4. **Heavy Import Deferral**: Keep heavy third-party packages (`scipy.*`, LLM SDKs `anthropic`/`openai`/`ollama`/`mcp`) off startup import graph. Import inside functions or prewarm on background threads (`Main._prewarm_heavy_imports()`).
5. **Mandatory Synchronous Testing**: Every new feature, UI widget, ViewModel method, API endpoint, or data processing routine MUST include corresponding unit/integration tests authored in the exact same task. No feature code is complete or committable without passing test coverage.
6. **Release & Packaging Parity**:
   - Whenever moving, renaming, or reorganizing modules or data assets under `spectroview/`, audit and update every non-source release and packaging consumer: `.github/workflows/release.yml`, `main.spec`, `packaging/*`, and `pyproject.toml`.
   - Keep `spectroview/main.py` importing concrete workspace modules directly (never exclusively via a PEP 562 `__getattr__` lazy facade, which PyInstaller's static `ModuleGraph` cannot trace) and mirror lazy/compatibility shims in `main.spec`'s `hidden_imports`.
   - Verify with `python -m pytest tests/unit/test_release_workflow.py tests/unit/test_packaging_windows.py tests/unit/test_packaging_macos.py tests/unit/test_packaging_linux.py tests/performance/test_startup_imports.py -v`.

---

## 2. Framework Navigation: Rules & Skills

Detailed project rules and procedures are modularized in Antigravity workspace rules and skills:

### Workspace Rules (`.agents/rules/`)
- **[architecture.md](file:///.agents/rules/architecture.md)** `[always_on]`: Module map, MVVM boundaries, workspaces, and data backbone.
- **[git-workflow.md](file:///.agents/rules/git-workflow.md)** `[always_on]`: Conventional Commits, milestone commits, atomic staging, and repository safety.
- **[coding-standards.md](file:///.agents/rules/coding-standards.md)** `[model_decision]`: Qt idioms, PySide6 conventions, vectorization, docstrings, and style mechanics.
- **[i18n.md](file:///.agents/rules/i18n.md)** `[model_decision]`: Mandatory multilingual GUI localization (EN, FR, ZH, DE, RU catalogs, canonical `itemData`, catalog/UI literal audits).
- **[testing.md](file:///.agents/rules/testing.md)** `[model_decision]`: Test execution, headless Qt (`offscreen`), fixtures (`qapp`, `_isolate_qsettings`), and visual checks.
- **[documentation.md](file:///.agents/rules/documentation.md)** `[model_decision]`: Mandatory synchronization across Developer Guide, Public API, and User Manual.
- **[pitfalls.md](file:///.agents/rules/pitfalls.md)** `[model_decision]`: Hard-won gotchas (QTimer parentage, headless surface budgets, QSettings isolation, twinx coordinates).
- **[fit-engine.md](file:///.agents/rules/fit-engine.md)** `[model_decision]`: Tensor fitting invariants, analytical Jacobians, and thread write-back safety.
- **[graphs-workspace.md](file:///.agents/rules/graphs-workspace.md)** `[model_decision]`: `MGraph` single source of truth, restyle fast path, toolbar sharing, and undo/redo.
- **[spectra-maps-workspace.md](file:///.agents/rules/spectra-maps-workspace.md)** `[model_decision]`: `SpectraStore` dual arrays, base $\leftrightarrow$ subclass hooks, selection payloads, and heatmap cache.
- **[workflows.md](file:///.agents/rules/workflows.md)** `[model_decision]`: Step-by-step touchpoints (adding shortcuts, plot styles, loaders, customization options, API functions).
- **[release-notes.md](file:///.agents/rules/release-notes.md)** `[model_decision]`: Concise GitHub release note format, compact structure, style rules, and full commit-audit procedure.

### Workspace Skills (`.agents/skills/<name>/SKILL.md`)
- **[implement-feature](file:///.agents/skills/implement-feature/SKILL.md)**: End-to-end workflow for adding new features, plot styles, or UI components.
- **[fix-bug](file:///.agents/skills/fix-bug/SKILL.md)**: Bug reproduction, root-cause resolution, and regression test authoring.
- **[refactor-module](file:///.agents/skills/refactor-module/SKILL.md)**: Behavior-preserving architectural refactoring.
- **[add-or-update-tests](file:///.agents/skills/add-or-update-tests/SKILL.md)**: Authoring unit, integration, and visual tests.
- **[update-documentation](file:///.agents/skills/update-documentation/SKILL.md)**: Synchronizing Developer Guide, API docs, and User Manual.
- **[review-code](file:///.agents/skills/review-code/SKILL.md)**: Comprehensive quality, MVVM, Qt thread safety, and diff audit.
- **[complete-mvp](file:///.agents/skills/complete-mvp/SKILL.md)**: Verification gates, atomic staging, conventional commit, and completion reporting.

---

## 3. Mandatory Development Lifecycle

For every task, execute these steps proportionally:

1. **Understand**:
   - Inspect `git status` to verify a clean working tree and preserve user changes.
   - Read relevant source modules, existing tests, and related documentation.
   - Identify affected components and architectural invariants.
2. **Plan**:
   - Outline proposed changes, new/updated tests, and affected documentation surfaces.
   - Keep planning proportional: deep for architectural shifts, concise for small fixes.
3. **Implement**:
   - Make the smallest complete change that satisfies the requirements.
   - Follow MVVM boundaries, PySide6 idioms, vectorization, and multilingual GUI localization rules ([.agents/rules/i18n.md](file:///.agents/rules/i18n.md), updating `spectroview/resources/i18n/{fr,zh,de,ru}.json` alongside new UI strings).
   - Author corresponding unit or integration tests alongside the feature code in the same session.
   - Preserve existing public API signatures and backward compatibility.
4. **Test & Verify**:
   - Run newly authored tests, `python -m pytest tests/unit/test_i18n.py -v` (when UI/ViewModel strings change), and relevant unit/integration suites with headless Qt enabled.
   - Ensure tests execute cleanly without event loop exceptions (`RuntimeError: Internal C++ object already deleted`).
   - Run static check on changed files (`python -m pyflakes spectroview/<file>.py`).
   - Review git diff to ensure no unintended edits, leftover debug prints, or unwanted files.
5. **Document**:
   - Update relevant documentation across the three surfaces in **all five languages (`*.md` and `*.{fr,de,ru,zh}.md`)** in the same task (and add `nav_translations` for `fr`, `de`, `ru`, `zh` in `mkdocs.yml` for any new pages) per [.agents/rules/documentation.md](file:///.agents/rules/documentation.md):
     - Developer Guide (`docs/developer/*.md` & `*.{fr,de,ru,zh}.md`)
     - Public API (`docs/api/*.md` & `*.{fr,de,ru,zh}.md`)
     - User Manual (`docs/user_manual/*.md` & `*.{fr,de,ru,zh}.md`)
   - Verify documentation spacing, five-language parity, structural checks (`python docs/i18n_check.py check de|ru|zh`), links, and strict MkDocs build (`python -m pytest tests/unit/test_markdown_syntax.py tests/unit/test_docs_i18n.py -v` and `mkdocs build --strict`).
   - Synchronize Visual Assets: Whenever UI components, buttons, dialogs, or themes change, re-capture related screenshots and illustrations in `docs/user_manual_images/` in **Modern Dark mode** using native font rendering (no headless tofu `□□□` boxes) and high-contrast dark button tiles.
6. **Commit & Report**:
   - Stage only related files for the completed milestone.
   - Create a Conventional Commit (`feat(scope): ...`, `fix(scope): ...`).
   - Report changes, verified tests, and documentation updates.

---

## 4. Quality & Testing Commands

Always use the project's actual commands:

```powershell
# Fast test suite (skips slow real-data benchmarks; ~1 min)
python -m pytest tests/ -m "not slow" -q

# Run a focused test subtree or file
python -m pytest tests/unit/view -q
python -m pytest tests/unit/fit_engine/test_optimizer.py -v

# Run full test suite (~1-1.5 min across all 1,600+ tests)
python -m pytest tests/ -q

# Static syntax and import check on modified files
python -m pyflakes spectroview/<file>.py

# Multilingual documentation checks (markdown spacing, 5-language parity, strict build)
python -m pytest tests/unit/test_markdown_syntax.py tests/unit/test_docs_i18n.py tests/unit/test_docs_url.py -v
$env:SPECTROVIEW_DOCS_NO_STAMP="1"; python -m mkdocs build --strict -d .tmp_site_check

# Launch application
python -m spectroview.main
```

> [!IMPORTANT]
> Headless Qt: `tests/conftest.py` sets `QT_QPA_PLATFORM=offscreen` automatically for pytest.
> For ad-hoc scripts or one-off runs touching Qt, always set:
> `$env:QT_QPA_PLATFORM="offscreen"` (PowerShell) or `set QT_QPA_PLATFORM=offscreen` (cmd).
> Never claim a test or check passed without executing it and verifying the output.

---

## 5. Git Commit & Safety Rules

- **Commit Completed Milestones**: Commit every completed MVP or logically independent milestone.
- **Conventional Commits**: Format commit messages as `<type>(<scope>): <summary>` followed by a detailed body explaining changes, rationale, tests verified, and documentation updated.
  - Recognized types: `feat`, `fix`, `perf`, `test`, `refactor`, `docs`, `chore`.
  - Recognized scopes: `view`, `viewmodel`, `model`, `fit_engine`, `io`, `graphs`, `spectra`, `maps`, `api`, `ai_agent`.
- **Safety Constraints**:
  - Never run destructive Git commands (`git reset --hard`, `git push --force`, history rewrites).
  - Never push to remote without explicit user authorization.
  - Never discard or overwrite existing user changes.
  - Never commit local user settings, caches, temporary export files, or virtual environments.

---

## 6. Completion Criteria

A task is complete only when:
1. The implementation satisfies all acceptance criteria without breaking contracts or introducing regressions.
2. New or updated unit/integration tests cover the new feature and pass cleanly without event loop warnings or memory leaks (`test_i18n.py` catalog & UI literal audits pass when GUI/ViewModel strings change).
3. Relevant documentation across the three doc surfaces has been synchronized in all five supported languages (EN, FR, DE, RU, ZH) with passing docs build and parity checks in the same task.
4. The final diff has been inspected and contains no unrelated changes.
5. Completed milestones are committed with a meaningful Conventional Commit.
