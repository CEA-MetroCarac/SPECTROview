---
name: update-documentation
description: >-
  Use this skill when synchronizing the Developer Guide, Public API documentation, or User Manual after modifying code or architecture in SPECTROview.
---

# Documentation Synchronization Workflow

Follow this procedure to keep all three documentation surfaces synchronized with codebase reality.

---

## Phase 1: Identify Changed Capabilities & Scope

1. Run `git status` and `git diff` to identify all modified code, models, dialogs, shortcuts, or public functions.
2. Consult [.agents/rules/documentation.md](../../rules/documentation.md) for the surface mapping.

---

## Phase 2: Map Changes to Documentation Surfaces

Determine which surfaces require updates:

### 1. Developer Guide (`docs/developer/*.md`)
Touch whenever:
- Architectural contracts, threading models, or data flows change.
- New models, algorithms, or subsystems are introduced (`vbf_engine.md`, `spectra_store.md`, `graphs.md`, `maps.md`, `spectra.md`, `ai_agent.md`).
- File loading formats change (`file_loading.md`).
- Project directory structure is modified (`index.md`).

### 2. Public API Documentation (`docs/api/*.md`)
Touch whenever:
- Functions in `spectroview/api/` (`spectra.py`, `2dmap.py`, `graphs.py`, `fitting.py`, `preprocessing.py`, `io.py`, `settings.py`) are added or updated.
- Document signatures, parameter types, default values, and return structures.

### 3. User Manual (`spectroview/resources/user_manual/*.md`)
Touch whenever:
- User-facing features, plots, calculators, or options are added or modified.
- Keyboard shortcuts change (`shortcuts.md`).
- Settings dialog options change (`settings.md`).
- Supported file formats change (`supported_data.md`).
- Workspaces and menus change (`spectra_maps.md`, `graphs.md`, `menu_bar.md`).

---

## Phase 3: Author Updates

1. **Accuracy**: Document the code as implemented, including exact parameter names, defaults, and UI button texts.
2. **Style**:
   - Developer Guide: Use MkDocs Material markdown (Mermaid diagrams, callout admonitions `!!! note`).
   - User Manual: Clean, task-oriented Markdown compatible with `v_user_manual.py`.
3. **Avoid Duplication**: Link to authoritative source files or existing pages rather than duplicating large tables or code snippets.

---

## Phase 4: Verify Links & Syntax

1. Verify that all relative links resolve to existing files.
2. Ensure markdown tables are formatted properly.

---

## Phase 5: Commit & Report

Commit documentation updates per [.agents/rules/git-workflow.md](../../rules/git-workflow.md):
```powershell
git commit -m "docs(<scope>): synchronize <area> documentation" -m "- <Bullet 1: Summarize documentation updates>" -m "- <Bullet 2: Surfaces touched>"
```
