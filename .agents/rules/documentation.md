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
| **User Manual** | `docs/user_manual/*.md` | End users | Online MkDocs (`mkdocs.yml`, `docs_hooks.py`) |

> [!NOTE]
> `docs/user_manual/` (along with `docs/user_manual_images/`) is the single source of truth for user-facing documentation, opened directly via the User Manual toolbar button (`https://cea-metrocarac.github.io/SPECTROview/user_manual/`).

---

## 3. Which Documentation File to Touch by Area

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

---

## 4. What to Update

- **New Feature or UI Control**: Describe it in the workspace's user-manual page and the developer guide; add any new keyboard shortcut to `shortcuts.md`.
- **Structural or Architecture Change**: Update `docs/developer/index.md` project structure and relevant subsystem page.
- **Changed Behavior or Bug Fix**: Fix existing prose immediately — never leave stale descriptions or parameters.
- **New Public API Function**: Add signature, docstring, and examples to `docs/api/`.

---

## 5. Style & Quality Standards

- **Developer Docs**: MkDocs Material format (Mermaid diagrams, admonitions, and syntax highlighting).
- **User Manual**: Clean, task-oriented Markdown rendered by MkDocs.
- **Conciseness**: Group long reference tables (e.g. `MGraph` fields) by logical category with a pointer to the source file rather than copying volatile lists.

---

## 6. Markdown Formatting Rules for Python-Markdown Compatibility

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

## 7. Mandatory Screenshots & Visual Assets Synchronization

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
