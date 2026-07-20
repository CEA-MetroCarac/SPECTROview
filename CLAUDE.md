# CLAUDE.md — SPECTROview

Always-loaded context. Kept lean so most tasks need **no** extra reads; deeper
rules live in `.claude/rules/*.md` and load only when relevant (via `paths:`
frontmatter and the map below). Source of truth for deep architecture is
`docs/developer/*.md` + `docs/api/*.md` (mkdocs) — link to it, don't copy it here.

## Project

PySide6/Qt desktop app for spectroscopic data processing & visualization
(discrete spectra, 2D/hyperspectral maps, wafer data) + a plotting workspace.
Fitting = custom vectorized batch engine (`fit_engine/`). Strict **MVVM**.
Package `spectroview/`; `VERSION` in `spectroview/__init__.py`. Tabs: **Spectra**
(base), **Maps** (subclasses Spectra's View+ViewModel), **Graphs** (plotting),
optional **AI chat** (`ai_agent/`).

## Rules (always follow)

1. **Understand before changing** — read the specific code (and the matching
   rule file) first; don't re-scan the whole repo when a targeted read suffices.
2. **Reuse before adding** — extend/fix existing code; only add a
   class/method/module when nothing existing fits. Search first.
3. **Respect MVVM** — View → ViewModel → Model, one direction. ViewModel never
   imports View; Model imports stdlib/numpy/pandas/scipy only.
4. **Keep it simple & clean** — no over-engineering or duplicated logic; remove
   dead code, unused imports/vars; one responsibility per method; readability
   over cleverness; match surrounding style.
5. **Don't break contracts** — `spectroview/api/` signatures and user-facing
   behavior are backward-compatible unless you're told otherwise.
6. **Test what you change** — add/adjust tests next to the logic; keep the suite
   green; run `pyflakes` on changed files.
7. **Keep docs in sync** — in the *same* change, update the docstring and, for
   any new feature or structural change, the **developer guide**
   (`docs/developer/`) and the **user manual**
   (`spectroview/resources/user_manual/`). Never leave docs describing old
   behavior. Details: [rules/documentation.md](.claude/rules/documentation.md).
8. **Pause for the big calls** — explain major architectural changes before
   applying; ask when requirements are ambiguous instead of guessing. Proceed
   without asking on reversible, in-scope work.

## Layer & naming

| Layer | Prefix | Location | May import |
|---|---|---|---|
| View | `v_` | `view/`, `view/components/` | ViewModel, components |
| ViewModel | `vm_` | `viewmodel/` | Model, `fit_engine` (never View) |
| Model | `m_`/unprefixed | `model/` | stdlib, numpy/pandas/scipy |

Private members `_leading_underscore`; Qt handlers `_on_<event>`; signals are
past-tense (`spectra_list_changed`). Tests: `tests/**/test_*.py`, `Test*`,
`test_*`.

## Non-obvious traps (full list: rules/pitfalls.md)

- **Parent `QTimer`/event-filters to their widget** or they fire after the C++
  object is deleted → segfault.
- **Never `import PyQt5`** — binding is forced to PySide6 (`QT_API=pyside6` in
  `__init__.py`); mixing bindings crashes.
- **Ad-hoc Qt scripts need `QT_QPA_PLATFORM=offscreen`** (tests set it in
  `conftest.py`); otherwise the native surface budget exhausts → segfault.
- **Annotation/event coords via `ax.transData`, not `event.xdata`** — a
  `twinx()` axis makes `event.inaxes` resolve to the wrong Axes.

## Navigation map

| Task | Rule file |
|---|---|
| Layers, module map, cross-workspace wiring | [architecture.md](.claude/rules/architecture.md) |
| Naming, style, Qt idioms, docstrings, full coding rules | [coding-standards.md](.claude/rules/coding-standards.md) |
| Run/add tests, headless Qt, fixtures, visual checks | [testing.md](.claude/rules/testing.md) |
| Step-by-step common tasks (plot style, customize option, shortcut, loader…) | [workflows.md](.claude/rules/workflows.md) |
| Graphs workspace internals | [graphs-workspace.md](.claude/rules/graphs-workspace.md) |
| Spectra/Maps workspaces, SpectraStore, loaders | [spectra-maps-workspace.md](.claude/rules/spectra-maps-workspace.md) |
| Vectorized Batch Fit engine (`fit_engine/`) | [fit-engine.md](.claude/rules/fit-engine.md) |
| Full pitfall list + guards | [pitfalls.md](.claude/rules/pitfalls.md) |
| Which docs to update for a change | [documentation.md](.claude/rules/documentation.md) |

## Commands

```bash
python -m spectroview.main                # run (or `spectroview` console script)
python -m pytest tests/ -q                # full suite (~8 min; offscreen auto-set)
python -m pytest path::TestClass -q       # focused
python -m pyflakes spectroview/<file>.py  # only static check available here
pip install -e ".[dev]"                   # dev install
```

Docs: `mkdocs` (`mkdocs.yml`). Packaging: `main.spec` (PyInstaller) + `pyproject.toml`.

## Maintaining this knowledge base

```
CLAUDE.md                 # committed, always loaded, keep < ~140 lines
.claude/
├── settings.json         # committed: permissions/hooks/env
├── settings.local.json   # personal, gitignored
└── rules/                # architecture, coding-standards, testing, workflows,
                          #   graphs-workspace, spectra-maps-workspace, fit-engine,
                          #   pitfalls, documentation
```

- Files with `paths:` frontmatter auto-load only when a session touches matching
  files (`testing`→tests, `graphs-workspace`→graph modules,
  `spectra-maps-workspace`→Spectra/Maps modules, `fit-engine`→`fit_engine/`,
  `pitfalls`→the crash-prone rendering files, `documentation`→the `docs/` &
  user-manual trees). General references (`architecture`, `coding-standards`,
  `workflows`) have no `paths:` and are read on demand via the map above — this
  is what keeps per-session tokens low. The map guarantees discovery even if
  `paths:` isn't honored.
- Each fact lives in **one** place: terse actionable version here, depth in one
  rule file. Never duplicate `docs/` prose — link. Avoid contradictions between
  files (Claude picks arbitrarily).
- Update a rule in the same PR as the behavior it describes; treat as code.
  Verify any named file/function/flag against the code before relying on it.
- Enforcement (lint/test-before-commit) belongs in `.claude/settings.json`
  hooks, not here. Personal notes → `CLAUDE.local.md` (gitignored) or
  `~/.claude/CLAUDE.md`. `/context` and `/memory` show what actually loaded.
