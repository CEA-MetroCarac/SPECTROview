---
name: documentation
description: Mandatory documentation synchronization policy, the three documentation surfaces, mapping table, and formatting standards.
trigger: model_decision
---

# Documentation Synchronization

## 1. Mandatory Synchronization Policy

Documentation is an essential part of every implementation task, not optional follow-up work. Whenever application code, features, public interfaces, configuration, keyboard shortcuts, or architecture change, Gemini **must** update the corresponding documentation in the same commit.

Never declare a task complete while documentation remains outdated.

---

## 2. The Three Documentation Surfaces

`SPECTROview` maintains three distinct documentation surfaces:

| Surface | Location | Audience | Rendered By |
|---|---|---|---|
| **Developer Guide** | `docs/developer/*.md` | Maintainers and contributors | MkDocs (`mkdocs.yml`) |
| **Public API** | `docs/api/*.md` | External Python API users | MkDocs (`mkdocs.yml`) |
| **User Manual** | `spectroview/resources/user_manual/*.md` | End users | In-app viewer (`v_user_manual.py`) and online MkDocs (via `docs_hooks.py`) |

> [!NOTE]
> `spectroview/resources/user_manual/` is the single source of truth for user-facing documentation. The build hook in `docs/docs_hooks.py` symlinks it into `docs/user_manual/` during MkDocs builds.

---

## 3. Which Documentation File to Touch by Area

| Area | Developer Guide (`docs/developer/`) | User Manual (`spectroview/resources/user_manual/`) | Public API (`docs/api/`) |
|---|---|---|---|
| **Spectra Workspace** | `spectra.md`, `spectra_store.md` | `spectra_maps.md` | `spectra.md` |
| **Maps Workspace** | `maps.md` | `spectra_maps.md` | `2dmap.md` |
| **Graphs Workspace** | `graphs.md` | `graphs.md` | `graphs.md` |
| **Fitting Engine (VBF)** | `vbf_engine.md` | `spectra_maps.md` | — |
| **File Loading & Formats** | `file_loading.md` | `supported_data.md` | — |
| **Multivariate Analysis (MVA)** | `mva.md` | `mva.md` | — |
| **AI Data Chat & MCP** | `ai_agent.md`, `mcp_server.md` | `ai_agent.md` | — |
| **Quick Calculators** | — | `calculators.md` | `calculators.md` |
| **Settings / Shortcuts / Menus** | — | `settings.md`, `shortcuts.md`, `menu_bar.md`, `save_load.md` | — |
| **Public API Additions** | — | — | `extending.md` + matching API file |

---

## 4. What to Update

- **New Feature or UI Control**: Describe it in the workspace's user-manual page and the developer guide; add any new keyboard shortcut to `shortcuts.md`.
- **Structural or Architecture Change**: Update `docs/developer/index.md` project structure and relevant subsystem page.
- **Changed Behavior or Bug Fix**: Fix existing prose immediately — never leave stale descriptions or parameters.
- **New Public API Function**: Add signature, docstring, and examples to `docs/api/`.

---

## 5. Style & Quality Standards

- **Developer Docs**: MkDocs Material format (Mermaid diagrams, admonitions, and syntax highlighting).
- **User Manual**: Clean, task-oriented Markdown compatible with PySide6 rich-text rendering (`v_user_manual.py`).
- **Conciseness**: Group long reference tables (e.g. `MGraph` fields) by logical category with a pointer to the source file rather than copying volatile lists.
