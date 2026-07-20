---
paths:
  - "docs/developer/**"
  - "docs/api/**"
  - "spectroview/resources/user_manual/**"
---

# Documentation

**Every new feature or structural change must update the docs in the same PR.**
Three doc surfaces exist — update each that applies:

| Surface | Location | Audience | Rendered by |
|---|---|---|---|
| Developer guide | `docs/developer/*.md` | maintainers/contributors | mkdocs (`mkdocs.yml`) |
| Public API | `docs/api/*.md` | API users | mkdocs |
| User manual | `spectroview/resources/user_manual/*.md` | end users | in-app viewer (`v_user_manual.py`) |

## Which file to touch (by area)

| Area | Developer | User manual | API |
|---|---|---|---|
| Spectra workspace | `spectra.md`, `spectra_store.md` | `spectra_maps.md` | `spectra.md` |
| Maps workspace | `maps.md` | `spectra_maps.md` | `2dmap.md` |
| Graphs workspace | `graphs.md` (+ `graph_workspace_review.md`) | `graphs.md` | `graphs.md` |
| Fitting engine | `vbf_engine.md` | `spectra_maps.md` | — |
| File loading / formats | `file_loading.md` | `supported_data.md` | — |
| MVA (PCA/NMF) | `mva.md` | `mva.md` | — |
| AI chat | `ai_agent.md` | `ai_agent.md` | — |
| Calculators | — | `calculators.md` | `calculators.md` |
| Settings / shortcuts / menus / save-load | — | `settings.md`, `shortcuts.md`, `menu_bar.md`, `save_load.md` | — |
| New public API surface | — | — | `extending.md` + the area file |

## What to update

- **New feature**: describe it where users/maintainers would look (usually the
  workspace's user-manual page + the developer page); add the keyboard shortcut
  to `user_manual/shortcuts.md` if you added one; document any new public
  function in `docs/api/`.
- **Structural change** (moved/renamed modules, new layer, changed data flow):
  update `docs/developer/index.md`'s project structure/architecture sections and
  the affected area page.
- **Changed behavior**: fix the existing prose in place — don't leave stale
  descriptions or screenshots that no longer match.

## Style

- Match the existing page's tone and heading structure. Developer docs use
  mkdocs (Mermaid diagrams, admonitions allowed); user-manual pages are plain,
  task-focused Markdown shown in-app.
- Keep it concise and accurate over exhaustive. Group long reference tables (e.g.
  `MGraph` fields) by category with a pointer to the source file rather than
  re-listing every field — it goes stale otherwise.
- These `docs/` files are the source of truth for deep detail; the
  `.claude/rules/` files summarize and link to them, never duplicate them.
