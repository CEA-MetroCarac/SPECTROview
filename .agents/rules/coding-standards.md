---
name: coding-standards
description: Python coding standards, PySide6 idioms, vectorization principles, Qt widget lifetime, docstring requirements, and style mechanics.
trigger: model_decision
---

# Coding Standards

The fundamental principles are in [AGENTS.md](../../AGENTS.md) (reuse-first, simple, clean, no broken contracts). This file adds the project-specific depth. Match the file you are editing over any general preference here.

---

## 1. Vectorization & Performance

- **Prefer vectorized NumPy/pandas operations over Python loops** on arrays. The entire `fit_engine` and `spectra_store` are designed around vectorized multi-dimensional array operations.
- Avoid unnecessary array allocations or memory copies.
- Scale algorithms to large hyperspectral maps (e.g. 100,000+ spectra) and large wafer datasets.

---

## 2. Heavy Imports & Startup Performance

- **Keep heavy scientific and AI libraries off the startup import path**:
  - `scipy.*` and LLM SDKs (`anthropic`, `openai`, `ollama`, `mcp`) must be imported inside the functions that use them, not at module top-level.
  - `scipy` is prewarmed on a background thread in `Main._prewarm_heavy_imports()` after startup.
- Verify startup impact with:
  ```powershell
  python -X importtime -c "import spectroview.main"
  ```
- Startup import constraints are enforced by `tests/performance/test_startup_imports.py`.

---

## 3. PySide6 & Qt Idioms

- **Signals for View $\leftrightarrow$ ViewModel**: Never call across the architectural boundary directly.
- **Widget Lifetime & Memory Safety**:
  - Parent all `QTimer`, event filters, and `QObject` helpers to the widget they serve so Qt destroys them together. Unparented timers firing on deleted C++ objects cause segfaults.
  - **Single-Shot Timers**: ALWAYS specify a context `QObject`: `QTimer.singleShot(delay_ms, context_qobject, slot)`. Never use the 2-argument form with lambdas referencing widget state (`QTimer.singleShot(delay, lambda: self.foo())`). Inside the callback, wrap accesses defensively with `try...except (RuntimeError, AttributeError)` to cleanly ignore invocations during teardown.
  - **Event Filters**: Event filters installed on `QApplication.instance()` or long-lived parent widgets must auto-remove themselves (`app.removeEventFilter(self)`) when their host or monitored widget is destroyed to prevent $O(N \times M)$ event-dispatch slowdowns and crashes.
  - Use `deleteLater()` for deferred cleanup of Qt widgets.
- **Keyboard Shortcuts**:
  - Implement shortcuts via `QShortcut` using `Qt.ShortcutContext.WidgetWithChildrenShortcut`, scoped to the workspace container (e.g. `self.mdi_area` for Graphs). This prevents shortcuts from firing when another tab is active or shadowing native `QLineEdit` clipboard shortcuts.
- **Modifier-Aware Actions**:
  - Bulk actions inspect `QApplication.keyboardModifiers() & Qt.ControlModifier` at click time (e.g. click = process selected item; Ctrl+click = process all items).
- **Theme & Icon Management**:
  - Icons that are colorful/theme-independent are set once and must **not** be tinted.
  - Monochrome icons are tinted per theme via `get_tinted_icon(...)` inside `apply_theme()`. Do not add tinting to colorful icons.

---

## 4. Public API & Signature Stability

- Modules in `spectroview/api/` (`spectra.py`, `2dmap.py`, `graphs.py`, `fitting.py`, `preprocessing.py`, `io.py`, `settings.py`) form the external contract.
- Never rename, reorder, or delete parameters, even if seemingly unused. Positional callers in scripts, tests, and the AI agent depend on exact signatures.

---

## 5. Docstrings & Comments

- Write concise, informative docstrings on all classes and non-trivial methods explaining **what** and **why**, not a line-by-line paraphrase of code.
- Inline comments: short ($\le 2$ lines) and reserved for non-obvious mathematical formulas, hardware constraints, or Qt edge cases.
- Update docstrings and inline comments in the same commit that modifies behavior.

---

## 6. Style Mechanics

- 4-space indentation throughout; PySide6 bindings (never import PyQt5).
- Prefer f-strings; use `pathlib` or `os.path.join` with `spectroview.ICON_DIR` / `resource_path()` for cross-platform and PyInstaller-frozen asset paths.
- Run static check before finishing:
  ```powershell
  python -m pyflakes spectroview/<modified_file>.py
  ```

---

## 7. Feature & Test Co-Development

- **Never write code without matching tests**: Every new View, ViewModel method, Model structure, peak model, or public API function must have corresponding unit or integration tests written and passing in the same session.
- **Update existing tests when modifying features**: Never disable, skip, or delete failing tests when altering existing behavior unless the underlying contract was intentionally replaced; update tests to assert the new behavior.
- **Run verification commands**: Validate changes locally with `python -m pytest tests/ -m "not slow" -q` before declaring work complete.
