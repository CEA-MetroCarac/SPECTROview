---
paths:
  - "docs/developer/**"
  - "docs/api/**"
  - "docs/user_manual/**"
---

# Documentation

**Every new feature, API, config, or structural change must update docs in BOTH English (`*.md`) and French (`*.fr.md`) in the same PR.** Never silently leave one language outdated.
Three bilingual doc surfaces exist (`mkdocs-static-i18n`, `docs_structure: suffix`) — update each that applies:

| Surface | Location | Audience | Rendered by |
|---|---|---|---|
| Developer guide | `docs/developer/*.md` & `*.fr.md` | maintainers/contributors | mkdocs (`mkdocs.yml`) |
| Public API | `docs/api/*.md` & `*.fr.md` | API users | mkdocs |
| User manual | `docs/user_manual/*.md` & `*.fr.md` | end users | mkdocs (`mkdocs.yml`) |

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
- **New page**: create both `<page>.md` and `<page>.fr.md`, add `<page>.md` to `nav:` in `mkdocs.yml`, and add its French title to `plugins.i18n.languages[fr].nav_translations`.
- **Changed behavior**: fix the existing prose in **both** English and French in place — don't leave stale descriptions, `#anchor` links, or screenshots that no longer match.

## Bilingual quality & verification

- Write natural, accurate scientific/software French — never literal machine translation. Keep code blocks, API identifiers, CLI commands, config keys, file paths, shortcuts, LaTeX math, and exact English GUI widget labels unchanged.
- Verify spacing, parity, links, and strict build before finishing:
  `SPECTROVIEW_DOCS_NO_STAMP=1` + `python -m pytest tests/unit/test_markdown_syntax.py tests/unit/test_docs_i18n.py -v` + `python -m mkdocs build --strict -d .tmp_site_check`.

## Style

- Match the existing page's tone and heading structure. Developer docs use
  mkdocs (Mermaid diagrams, admonitions allowed); user-manual pages are plain,
  task-focused Markdown shown in-app.
- Keep it concise and accurate over exhaustive. Group long reference tables (e.g.
  `MGraph` fields) by category with a pointer to the source file rather than
  re-listing every field — it goes stale otherwise.
- These `docs/` files are the source of truth for deep detail; the
  `.claude/rules/` files summarize and link to them, never duplicate them.
