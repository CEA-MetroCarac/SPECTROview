---
name: release-notes
description: GitHub release note format, structure, section template, writing style, and full commit-audit procedure inspired by v26.43.1.
trigger: model_decision
---

# GitHub Release Notes Standard

All GitHub Release notes for `SPECTROview` must follow the visual hierarchy, section structure, and technical clarity established in release **`v26.43.1`**.

---

## 1. Mandatory Pre-Drafting Audit

Before writing or rewriting release notes, perform a complete audit of the repository history between the previous release tag (`<prev_tag>`) and the target commit/tag (`HEAD`):

1. **Inspect every commit without exception**:
   - Count all commits: `git rev-list --count <prev_tag>..HEAD`
   - Read every commit subject and body: `git log <prev_tag>..HEAD --format="=== %h | %ad | %s ===%n%b" --date=short`
   - Inspect diff summaries across source, API, packaging, docs, and tests (`git diff --shortstat <prev_tag>..HEAD`, `git diff --name-status <prev_tag>..HEAD`).
   - Never rely on memory or summarize only recent commits when long release cycles span dozens of commits.

2. **Verify final shipped state**:
   - When a feature evolved, was refactored, or superseded by a later commit inside `<prev_tag>..HEAD` (e.g., an experimental theme superseded by a unified design system), document the **final shipped behavior** in the highlight sections rather than listing intermediate prototypes as still shipping.
   - Verify exact class names, module paths, CLI flags, asset filenames (`.github/workflows/release.yml`), version strings (`spectroview/__init__.py`), and verified test counts directly against the working codebase.

---

## 2. Canonical Structure & Template

Begin directly with `## Release Highlights` (GitHub renders the release title `v<VERSION>` above the body) and separate every major `###` block with a horizontal rule `---`:

```markdown
## Release Highlights

SPECTROview **v<VERSION>** is a <major feature / maintenance / security / multi-platform> release delivering <1 concise sentence summarizing the flagship themes and user impact>.

---

### 📦 Multi-Platform Distribution, Updater & Security

* **Standalone Installers & Packages (Windows, macOS, Linux + PyPI)**:
  - <Exact attached release artifacts, per-OS install notes, SHA256SUMS.txt, and Gatekeeper note if macOS builds are unsigned.>

---

### <Emoji> <Feature / Subsystem Category>

* **<Concise Feature Title (Optional Quantitative Metric)>**:
  - <Technical change referencing exact classes/modules/UI controls in backticks and architectural rationale.>
  - <User-facing benefit or measured before -> after benchmark in **bold**.>

---

### 🚀 Performance & Responsiveness Optimizations

* **<Optimized Operation (Speedup / Memory Metric)>**:
  - <Root optimization mechanism (e.g., NumPy vectorization, artist recycling, caching).>
  - <Benchmark: <operation> dropped from **<before>** down to **<after>**.>

---

### 🐛 Bug Fixes & Stability

* **<Bug Area or Edge Case>**:
  - <Root cause fixed and runtime crash/corruption prevented.>

---

### 🧪 Quality & Test Coverage

* Added **<N>+ lines across <M> new/updated test files** (`+<insertions>` / `-<deletions>` in `tests/`):
  - <Key new/updated test modules in backticks.>
* Verified **<X> passing tests** cleanly via `python -m pytest tests/ -m "not slow"`.

---

### 📦 Commit Log (<prev_tag>...<new_tag>)

* `<short_sha>` <commit subject verbatim from git log --oneline>

**Full Changelog**: https://github.com/CEA-MetroCarac/SPECTROview/compare/<prev_tag>...<new_tag>
```

---

## 3. Section Order & Approved Emoji Headings

Include only the sections that have real changes in `<prev_tag>..HEAD`, ordered roughly by user and architectural significance:

1. `## Release Highlights` (required executive summary)
2. `### 📦 Multi-Platform Distribution, Automatic Updates & TLS Security` (packaging installers/DMG/AppImage/deb, updater, `certifi` / `truststore` hardening)
3. `### 🎨 Modernized UI/UX & Token-Driven Theme System` (or `### 🎨 Theme & UI Refinements`)
4. `### 🔍 Spectra & Maps Workspaces` (interactive mapping, Live Maps, viewers, selection, notifications)
5. `### 📈 Graphs Workspace` (plot styles, customization dialogs, toolbar, filtering, resize/pan fluidity)
6. `### 🤖 AI Chat Assistant & Credential Security` (sidebar panel, keyring `SecretStore`, history, composer)
7. `### ⚙️ Settings, Preferences & Universal Working Folder` (settings dialog architecture, About build info, portable config)
8. `### 📁 Native File Format Support (I/O)` (new readers like `.wdf`, `.l6s`/`.l6m`, `.ngs`/`.ngc`, API updates)
9. `### 🚀 Massive Performance & Responsiveness Optimizations` (measured latency, memory, wheel size, startup deferrals)
10. `### 📚 Documentation, Dependencies & Platform Compatibility` (User Manual / MkDocs overhaul, Python / NumPy / Pandas compatibility, removed legacy tools)
11. `### 🐛 Bug Fixes & Stability` (always include if any `fix:` commits exist)
12. `### 🧪 Quality & Test Coverage` (always include if tests were added/updated)
13. `### 📦 Commit Log (<prev_tag>...<new_tag>)` + `**Full Changelog**` link (always include; if external PR contributors appear in the range, also add `## New Contributors` above the Full Changelog link)

---

## 4. Style & Markdown Formatting Rules

- **Two-Level Bullet Pattern**: Use `* **Bold Feature Title (Optional Impact)**:` at level 1, and two-space-indented `- ` sub-bullets for implementation specifics, affected symbols, and measured numbers.
- **Concrete Identifiers & Metrics**: Name exact UI labels (**Settings → General**), public API functions, modules, classes, or artifact filenames in backticks, and emphasize before → after metrics in **bold**.
- **GitHub-Ready & Public-Safe Markdown**:
  - Never mention internal/private project names or design inspirations (e.g., `PLUME`, `PLUME-inspired`) anywhere in release highlights or commit logs; describe features purely in `SPECTROview` public terminology and neutralize any matching commit subjects.
  - Never include local filesystem `file:///` links in release notes.
  - Always leave a blank line before every list and table for universal Markdown rendering.
  - Keep artifact filenames in Downloads descriptions identical to `.github/workflows/release.yml`.
- **Commit Log Completeness**: Every commit in `<prev_tag>..HEAD` should appear in reverse-chronological order (`* \`<short_sha>\` <commit subject>`), matching `v26.43.1`, with any private/internal reference phrases redacted cleanly. Do not omit commits or alter SHAs.

