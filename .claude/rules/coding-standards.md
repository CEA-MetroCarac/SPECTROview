# Coding Standards

The must-follow rules are in `CLAUDE.md` (reuse-first, simple, clean, no broken
contracts). This file adds the project-specific depth. Match the file you're in
over any general preference here.

## Extra emphases
- Factor duplicated logic into a shared helper — see the dedup helpers in
  `v_workspace_graphs.py` and `customize_annotations.py` for the house style
  (a `{type: handler}` map beats a long if/elif chain).
- Prefer vectorized numpy/pandas over Python loops on large arrays; the whole
  `fit_engine` and `spectra_store` are built that way. Avoid needless
  allocations/copies and scale to large maps/wafers.
- Public API (`spectroview/api/`) and positional call sites (tests, `ai_agent`)
  are contracts: don't rename/reorder/drop params, even "unused" ones, without
  an explicit instruction.

## Qt idioms used here

- **Signals for View↔ViewModel**; never call across the boundary directly.
- Shortcuts are `QShortcut` with
  `Qt.ShortcutContext.WidgetWithChildrenShortcut`, scoped to the workspace's
  container (e.g. the Graphs `mdi_area`) so they fire only when that tab is
  active and don't shadow a focused `QLineEdit`'s native Ctrl+C/V/Z.
- Modifier-aware buttons read `QApplication.keyboardModifiers() &
  Qt.ControlModifier` at click time (e.g. Export = click one / Ctrl+click all).
- Widget lifetime matters: parent `QTimer`/`QObject` helpers to the widget they
  serve so Qt tears them down together (see the ToolbarEventFilter note in
  [pitfalls.md](pitfalls.md)); use `deleteLater()` for deferred disposal.
- Theme: icons that are **colorful/theme-independent** are set once and need no
  tinting; monochrome icons are tinted per theme via `get_tinted_icon(...)` in
  `apply_theme()`. Don't add tinting for a colorful icon.

## Docstrings & comments

- Write concise, informative docstrings on classes and non-trivial methods —
  say what and why, not a line-by-line paraphrase.
- Inline comments: short (≤2 lines) and only for a constraint or non-obvious
  reason the code can't show by itself. Never restate the code.
- Keep docs synchronized with the implementation; update the docstring in the
  same edit that changes the behavior.

## Style mechanics

- 4-space indent, PySide6 throughout. Dev extras include `black` and `flake8`
  (`pip install -e ".[dev]"`), but there is **no enforced repo linter config** —
  match existing formatting and run `pyflakes` on files you change.
- Prefer f-strings; prefer `pathlib`/`os.path.join` with `ICON_DIR` for resource
  paths (never hardcode separators).
