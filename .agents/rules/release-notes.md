---
name: release-notes
description: Concise, user-focused GitHub release note format, compact section structure, style rules, and full commit-audit procedure.
trigger: model_decision
---

# GitHub Release Notes Standard

All GitHub Release notes for `SPECTROview` must be **concise, user-focused, and scannable in under 30 seconds** while remaining grounded in a complete commit audit.

---

## 1. Mandatory Pre-Drafting Audit

Before writing or rewriting release notes, audit the full commit range between the previous release tag (`<prev_tag>`) and the target commit/tag (`HEAD`):

1. **Inspect every commit without exception**:
   - Count all commits: `git rev-list --count <prev_tag>..HEAD`
   - Read every commit subject and body: `git log <prev_tag>..HEAD --format="=== %h | %ad | %s ===%n%b" --date=short`
   - Inspect diff summaries where needed (`git diff --shortstat <prev_tag>..HEAD`, `git diff --name-status <prev_tag>..HEAD`).
   - Never summarize only the most recent commits or rely on memory.

2. **Verify final shipped state**:
   - When a feature evolved, was refactored, or superseded within `<prev_tag>..HEAD`, document only the **final shipped behavior**.
   - Verify exact UI labels, public API symbols, artifact filenames (`.github/workflows/release.yml`), version strings (`spectroview/__init__.py`), and verified test counts directly against the working codebase.
   - Confirm the pre-tag CI/packaging verification suite passes cleanly before tagging: `python -m pytest tests/unit/test_release_workflow.py tests/unit/test_packaging_windows.py tests/unit/test_packaging_macos.py tests/unit/test_packaging_linux.py tests/performance/test_startup_imports.py -v`.

---

## 2. Canonical Structure & Template

Begin directly with `## Release Highlights` (GitHub renders the release title `v<VERSION>` above the body), keep the entire note around **25–55 lines**, and separate major sections with `---`:

```markdown
## Release Highlights

SPECTROview **v<VERSION>** delivers <1 crisp sentence summarizing the flagship user-facing themes>.

---

### ✨ New Features

* **<Short Feature Title>** — <1 concise sentence focused on user value and measurable impact>.
* **<Short Feature Title>** — <1 concise sentence focused on user value>.

---

### 🚀 Performance

* **<Optimized Workflow (**<before> → <after>**)>** — <1 concise sentence on what sped up or reduced memory>.

---

### 🐛 Bug Fixes

* **<Bug Area>** — <1 concise sentence describing what was fixed for users>.

---

### 📚 Documentation & API

* **<Docs / Public API Update>** — <1 concise sentence covering user manual, online docs, or public API additions>.

---

### 🧪 Verification

* Verified **<X> passing tests** via `python -m pytest tests/ -m "not slow"`.

**Full Changelog**: https://github.com/CEA-MetroCarac/SPECTROview/compare/<prev_tag>...<new_tag>
```

---

## 3. Approved Section Headings

Include **only** the sections that have meaningful user-facing changes in `<prev_tag>..HEAD`, in this order:

1. `## Release Highlights` (required 1-sentence executive summary)
2. `### 📦 Distribution & Security` (include only when installer, packaging, updater, or TLS behavior changed, or when authoring standalone notes outside `.github/workflows/release.yml` which already attaches `### Downloads & Verification`)
3. `### ✨ New Features` (group all new capabilities here; use focused sub-domain headings only when a single release has 3+ major highlights in one workspace)
4. `### 🚀 Performance` (measured speedups, startup time reductions, or RAM savings in **bold**)
5. `### 🐛 Bug Fixes` (user-visible fixes and crash/layout resolutions)
6. `### 📚 Documentation & API` (online documentation, User Manual, public API, or runtime dependency updates)
7. `### 🧪 Verification` (optional 1-line verified test count)
8. `**Full Changelog**: https://github.com/CEA-MetroCarac/SPECTROview/compare/<prev_tag>...<new_tag>` (always include; if external PR contributors appear in the range, add `## New Contributors` above this link)

---

## 4. Conciseness & Style Rules

- **Target Length (~25–55 lines)**: Keep notes crisp, scannable, and free of redundant filler.
- **Single-Level Bullets by Default**: Use `* **Short Title** — 1 concise sentence.` Avoid nested sub-bullets unless a flagship feature genuinely requires 2–3 short sub-items.
- **User-First, Low-Noise**: Focus on what users can see, click, measure, or call via the public API. Omit internal class/method names, source file paths, internal refactors, `chore:` version bumps, `mkdocs deploy`, and internal agent/CI plumbing unless they directly affect end users or contributors.
- **No Raw Commit Dump by Default**: Rely on the `**Full Changelog**` compare link instead of listing every commit. Only if a commit list is explicitly requested, place it at the bottom inside a collapsible `<details><summary>Commit Log (<N> commits)</summary>...</details>` block (one `* \`<short_sha>\` <subject>` per line, reverse-chronological).
- **Concrete Metrics**: Highlight measured before → after improvements and key counts in **bold** (e.g., **5,919 ms → 18 ms**).
- **Public-Safe & Valid Markdown**:
  - Never mention internal/private project names (e.g., `PLUME`) anywhere in release notes or commit logs.
  - Never include local filesystem `file:///` links.
  - Always leave a blank line before every list and table.
  - Keep any referenced artifact filenames identical to `.github/workflows/release.yml`.
