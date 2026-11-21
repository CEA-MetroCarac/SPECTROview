---
name: documentation
description: Mandatory documentation synchronization policy, the three documentation surfaces, mapping table, and formatting standards.
trigger: model_decision
---

# Documentation Synchronization

## 1. Mandatory Synchronization Policy

Documentation is an essential part of every implementation task, not optional follow-up work. Whenever application code, features, public interfaces, configuration, keyboard shortcuts, or architecture change, agents **must** update the corresponding documentation across **all five supported languages — English (`en`), French (`fr`), German (`de`), Russian (`ru`), and Simplified Chinese (`zh`)** — in the same task.

Never leave any language silently outdated when modifying another, never regress English or French content, and never declare a task complete while documentation or translations remain out of sync.

---

## 2. The Three Documentation Surfaces

`SPECTROview` maintains three multilingual documentation surfaces configured via `mkdocs-static-i18n` (`docs_structure: suffix`):

| Surface | Location | Audience | Rendered By |
|---|---|---|---|
| **Developer Guide** | `docs/developer/*.md` & `*.{fr,de,ru,zh}.md` | Maintainers and contributors | MkDocs (`mkdocs.yml`) |
| **Public API** | `docs/api/*.md` & `*.{fr,de,ru,zh}.md` | External Python API users | MkDocs (`mkdocs.yml`) |
| **User Manual** | `docs/user_manual/*.md` & `*.{fr,de,ru,zh}.md` | End users | Online MkDocs (`mkdocs.yml`, `docs_hooks.py`) |

> [!NOTE]
> `docs/user_manual/` (along with `docs/user_manual_images/`) is the single source of truth for user-facing documentation, opened directly via the User Manual toolbar button (`spectroview.docs_url("user_manual/", get_language())`, resolving to `/user_manual/` for English and `/<lang>/user_manual/` for `fr`, `de`, `ru`, and `zh`). All five languages share `docs/assets/` and `docs/user_manual_images/`.

---

## 3. Multilingual Documentation Policy (EN / FR / DE / RU / ZH)

- **Suffix File Convention**: Every navigation page has an English default file (`<page>.md`, served at `/`) and four side-by-side translations:
  - French: `<page>.fr.md` (served at `/fr/`)
  - German: `<page>.de.md` (served at `/de/`)
  - Russian: `<page>.ru.md` (served at `/ru/`)
  - Simplified Chinese: `<page>.zh.md` (served at `/zh/`)
  Internal non-nav documents (`docs/log.md`, `docs/roadmaps/*.md`, `docs/developer/theme_design_system.md`) remain English-only and fall back automatically via `fallback_to_default: true`.
- **Mandatory Five-Language Sync**: Updating or fixing any page requires updating `<page>.md`, `<page>.fr.md`, `<page>.de.md`, `<page>.ru.md`, and `<page>.zh.md` in the same task so all versions stay semantically aligned and technically accurate.
- **Adding a New Documentation Page**: Create all five files (`<page>.md`, `<page>.fr.md`, `<page>.de.md`, `<page>.ru.md`, `<page>.zh.md`), register `<page>.md` under `nav:` in `mkdocs.yml`, and add its localized title to `nav_translations` under each non-default locale (`fr`, `de`, `ru`, `zh`) in `mkdocs.yml`.
- **Stable Heading Anchors (`{ #english-slug }`)**:
  - In `.de.md`, `.ru.md`, and `.zh.md` files, every Markdown heading (`#`–`######`) **must** carry an explicit `attr_list` anchor ID matching the English heading slug, e.g. `## **Arbeitsbereich Diagramme** { #workspace-graphs }`. (Inspect English slugs with `python docs/i18n_check.py slugs docs/<page>.md`.)
  - Preserve existing heading IDs when editing `.de.md`, `.ru.md`, and `.zh.md` files, and keep `.fr.md` `#anchor` links aligned with French heading slugs so existing URLs and cross-page links never break.
- **GUI Terminology & Label Convention**:
  - Reuse terminology from the GUI localization catalogs (`spectroview/resources/i18n/{fr,de,ru,zh}.json`) so documentation matches the live interface.
  - In `.de.md`, `.ru.md`, and `.zh.md`, when an English UI label exists as an exact key in the GUI catalog `strings`, write the catalog translation in bold followed by the English original in parentheses on its **first occurrence on the page** — e.g. `**Plot hinzufügen** (Add plot)`, `**Добавить график** (Add plot)`, `**添加图表** (Add plot)` — and use the translated term thereafter. UI labels not present in the catalog remain verbatim in English. (French `.fr.md` preserves its established English widget-label convention.)
- **Non-Translatable Exclusions**:
  - Write natural, idiomatic scientific and software prose for native readers — never literal machine translations.
  - Keep fenced code blocks (except human-facing quoted labels inside `mermaid` diagrams, which should be translated inside quotes), inline code, API identifiers, class/function/parameter names, CLI commands, configuration keys, file paths, file extensions, keyboard shortcuts, physical units, and LaTeX equations unchanged.
- **Verification Gate**: Before declaring documentation work complete, install docs extras if needed (`pip install -e ".[docs]"`) and validate markdown spacing, five-language parity, structural checks, links, and strict MkDocs build:
  ```powershell
  $env:SPECTROVIEW_DOCS_NO_STAMP="1"
  python -m pytest tests/unit/test_markdown_syntax.py tests/unit/test_docs_i18n.py tests/unit/test_docs_url.py -v
  python docs/i18n_check.py check de
  python docs/i18n_check.py check ru
  python docs/i18n_check.py check zh
  python -m mkdocs build --strict -d .tmp_site_check
  ```
  (Remove `.tmp_site_check` after verifying; use `SPECTROVIEW_DOCS_NO_STAMP=1` during dry-run checks so tracked timestamps in `docs/user_manual/index*.md` are not dirtied.)

---

## 4. Which Documentation File to Touch by Area

| Area | Developer Guide (`docs/developer/`) | User Manual (`docs/user_manual/`) | Public API (`docs/api/`) |
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

*(For every file listed above, update `<page>.md` and `<page>.{fr,de,ru,zh}.md` together.)*

---

## 5. What to Update

- **New Feature or UI Control**: Describe it in the workspace's user-manual page and the developer guide across all five languages (`*.md` and `*.{fr,de,ru,zh}.md`); add any new keyboard shortcut to `shortcuts.md` and `shortcuts.{fr,de,ru,zh}.md`.
- **Structural or Architecture Change**: Update `docs/developer/index.md` and `index.{fr,de,ru,zh}.md` project structure and relevant subsystem pages.
- **Changed Behavior or Bug Fix**: Fix existing prose across all five languages immediately — never leave stale descriptions, parameters, or out-of-sync translations.
- **New Public API Function**: Add signature, docstring, and examples to `docs/api/*.md` and `*.{fr,de,ru,zh}.md`.

---

## 6. Style & Quality Standards

- **Developer Docs**: MkDocs Material format (Mermaid diagrams, admonitions, and syntax highlighting).
- **User Manual**: Clean, task-oriented Markdown rendered by MkDocs.
- **Conciseness**: Group long reference tables (e.g. `MGraph` fields) by logical category with a pointer to the source file rather than copying volatile lists.

---

## 7. Markdown Formatting Rules for Python-Markdown Compatibility

`SPECTROview` user manual pages and documentation are rendered by **Python-Markdown** (used by MkDocs Material). Python-Markdown enforces John Gruber's standard Markdown specification:

- **Mandatory Blank Line Before Lists**:
  Always insert an empty line between paragraph text and an unordered (`-`, `*`, `+`) or ordered (`1.`) list:
  ```markdown
  <!-- CORRECT -->
  Here is some introductory text:

  - Item 1
  - Item 2

  <!-- INCORRECT (Python-Markdown collapses all items into a single paragraph line) -->
  Here is some introductory text:
  - Item 1
  - Item 2
  ```
  > [!WARNING]
  > CommonMark / GFM parsers (e.g., VS Code or GitHub Markdown preview) allow lists to interrupt paragraphs without a preceding blank line. Python-Markdown does NOT. If the blank line is omitted, the list will render as an unformatted run-on paragraph in both MkDocs (`mkdocs gh-deploy`) and the in-app manual.

- **Mandatory Blank Line Before Tables**:
  Always separate tables from preceding text with a blank line.
- **Lists in Admonitions & Blockquotes**:
  Lists inside admonitions (`!!! note`) and blockquotes (`>`) also require an empty line (matching blockquote / indentation prefix) before the first list item.

---

## 8. Mandatory Screenshots & Visual Assets Synchronization

Whenever code, styling, or UI changes affect buttons, icons, colors, layouts, dialogs, viewers, tables, or themes, agents **must** update and re-capture the corresponding screenshots and illustrations in `docs/user_manual_images/` alongside Markdown prose:

1. **Modern Dark Mode**:
   - All screenshots and illustrations must consistently represent the application in **Modern Dark mode**.
   - Populate with realistic example datasets (`examples/graphs_NEW.graphs`, `examples/wafers_NEW.maps`, `examples/spectra_NEW.spectra`, `examples/spectroscopic_data/`).
2. **Button Icons & High-Contrast Tiles**:
   - Standalone toolbar/action button images (used in tables in `menu_bar.md`, `spectra_maps.md`, `save_load.md`) must be rendered on a solid dark button tile background (`#28282D`, 1px subtle border `#3E3E46`, rounded corners).
   - **Never** save white vector icon strokes with a transparent background for documentation tables: transparent white icons become completely invisible when users view the MkDocs site in Light theme.
3. **Native Typography (No Headless Tofu Squares)**:
   - **Never** capture text-bearing widgets under `QT_QPA_PLATFORM=offscreen` on Windows. Headless Qt offscreen QPA lacks DirectWrite font engine initialization (`len(QFontDatabase.families()) == 0`), causing text, labels, and table headers to render as missing-glyph square tofu boxes (`□□□□`).
   - Always render widgets with the native platform font engine (`Segoe UI` via standard Windows QPA) so all typography remains crisp and legible.
4. **Clean Presentation**:
   - Crop tightly to the relevant widget or panel; avoid desktop clutter, OS titlebars, or irrelevant windows.
   - Maintain existing filenames and image paths in `docs/user_manual_images/` whenever possible to preserve Markdown link integrity.
