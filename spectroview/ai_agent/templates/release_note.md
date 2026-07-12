# Purpose

This is a reusable Markdown template for SPECTROview release notes.

---

# SPECTROview Release Notes — Version XX.YY.Z

**Release Date:** YYYY-MM-DD
**Minimum Python Version:** 3.11

---

## Highlights

Brief paragraph (2–3 sentences) describing the most significant changes in this release.

> **Example:** This release introduces the AI Agent's modular Prompt Engineering architecture, enabling customisation of all prompts, rules, and knowledge without modifying Python source code. Performance improvements in the VBF engine reduce batch fitting time by up to 40% for large hyperspectral maps.

---

## New Features

### [Feature Category]

- **[Feature Name]** — Brief description of the new feature and its user benefit.
- **[Feature Name]** — Brief description.

### AI Agent

- **[Feature Name]** — Description.

### Graphs Workspace

- **[Feature Name]** — Description.

---

## Improvements

- **[Component]** — Description of the improvement and why it matters.
- **[Component]** — Description.

---

## Bug Fixes

- **[Component]** — Fix description. ([Issue #NNN](https://github.com/))
- **[Component]** — Fix description.

---

## Breaking Changes

> **⚠️ Warning:** List any changes that may require user action or that break backward compatibility.

- **[Change]** — Description of the breaking change and migration path.
  - **Before:** `old_behavior`
  - **After:** `new_behavior`
  - **Migration:** Steps users need to take.

---

## Deprecations

- **[Feature]** — This feature is deprecated and will be removed in version XX.YY.Z. Use [alternative] instead.

---

## Known Issues

- **[Issue]** — Description of a known issue that was not fixed in this release, with any available workaround.

---

## Dependencies

| Package | Minimum Version | Change |
|---------|----------------|--------|
| PySide6 | 6.6.0 | Unchanged |
| numpy | 1.24.0 | Updated minimum |
| pandas | 2.0.0 | Unchanged |
| PyYAML | 6.0.0 | **New** |

---

## Contributors

Thanks to all contributors who helped with this release:

- [@username](https://github.com/) — Contribution description.

---

## Full Changelog

[View full changelog on GitHub](https://github.com/CEA-MetroCarac/SPECTROview/commits/main)
