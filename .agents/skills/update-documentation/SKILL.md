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

### 1. Developer Guide (`docs/developer/*.md` & `*.{fr,de,ru,zh}.md`)
Touch whenever:

- Architectural contracts, threading models, or data flows change.
- New models, algorithms, or subsystems are introduced (`vbf_engine`, `spectra_store`, `graphs`, `maps`, `spectra`, `ai_agent`, `mcp_server`).
- File loading formats change (`file_loading`).
- Project directory structure is modified (`index`).

### 2. Public API Documentation (`docs/api/*.md` & `*.{fr,de,ru,zh}.md`)
Touch whenever:

- Functions in `spectroview/api/` (`spectra.py`, `2dmap.py`, `graphs.py`, `fitting.py`, `preprocessing.py`, `io.py`, `settings.py`) are added or updated.
- Document signatures, parameter types, default values, and return structures.

### 3. User Manual (`docs/user_manual/*.md` & `*.{fr,de,ru,zh}.md`)
Touch whenever:

- User-facing features, plots, calculators, or options are added or modified.
- Keyboard shortcuts change (`shortcuts`).
- Settings dialog options change (`settings`).
- Supported file formats change (`supported_data`).
- Workspaces and menus change (`spectra_maps`, `graphs`, `menu_bar`).

---

## Phase 3: Author Multilingual Updates (EN / FR / DE / RU / ZH)

1. **Mandatory Five-Language Sync**: Update `<page>.md` (English), `<page>.fr.md` (French), `<page>.de.md` (German), `<page>.ru.md` (Russian), and `<page>.zh.md` (Simplified Chinese) in the same task per [.agents/rules/documentation.md](../../rules/documentation.md) — never leave any language outdated. For any newly added page, create all five language files, add `<page>.md` to `nav:` in `mkdocs.yml`, and add its localized title to `nav_translations` for `fr`, `de`, `ru`, and `zh`.
2. **Accuracy**: Document the code as implemented, including exact parameter names, defaults, and UI button texts.
3. **Style & Translation Quality**:
   - Follow all heading-ID (`{ #english-slug }` on `.de.md`/`.ru.md`/`.zh.md`), GUI-catalog terminology, and non-translatable code/math rules in [.agents/rules/documentation.md](../../rules/documentation.md).
4. **Avoid Duplication**: Link to authoritative source files or existing pages rather than duplicating large tables or code snippets.

---

## Phase 4: Synchronize Visual Assets (Screenshots & Illustrations)

Whenever UI components, buttons, dialogs, layouts, or themes are altered:

1. **Re-capture Outdated Visuals**:
   - Inspect all static PNG images referenced in the affected manual pages (`docs/user_manual_images/`).
   - Re-capture screenshots using the live application in **Modern Dark mode** populated with realistic example data.
2. **Button Icon Contrast**:
   - Save standalone button icons on a solid dark tile background (`#28282D`, border `#3E3E46`) so they stay crisp and visible in both Light and Dark MkDocs documentation themes.
3. **Typography Verification**:
   - Capture widgets using native Windows platform font engine (`windows` QPA, DirectWrite / `Segoe UI`).
   - **Never** capture text with `QT_QPA_PLATFORM=offscreen`, which lacks font families and produces square tofu glyphs (`□□□□`).

---

## Phase 5: Verify Links, Five-Language Parity & Build

1. Verify markdown list/table spacing, five-language file and `nav_translations` parity, structural checks, and the strict MkDocs build:
   ```powershell
   $env:SPECTROVIEW_DOCS_NO_STAMP="1"
   python -m pytest tests/unit/test_markdown_syntax.py tests/unit/test_docs_i18n.py tests/unit/test_docs_url.py -v
   python docs/i18n_check.py check de
   python docs/i18n_check.py check ru
   python docs/i18n_check.py check zh
   python -m mkdocs build --strict -d .tmp_site_check
   ```
2. Delete temporary build output (`.tmp_site_check`) and confirm `git status` shows no unintended timestamp dirtiness.

---

## Phase 6: Commit & Report

Commit documentation updates per [.agents/rules/git-workflow.md](../../rules/git-workflow.md):
```powershell
git commit -m "docs(<scope>): synchronize <area> documentation" -m "- <Bullet 1: Summarize documentation updates>" -m "- <Bullet 2: Surfaces touched>"
```
