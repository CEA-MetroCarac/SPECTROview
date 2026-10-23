---
name: pitfalls
description: Critical pitfalls, segfault traps, Qt lifetime hazards, coordinate transforms, and PySide6 compatibility rules for SPECTROview.
trigger: model_decision
---

# Common Pitfalls & Segfault Traps

Hard-won gotchas that have bitten this codebase. The guards described below are in place — never remove them, and apply the same patterns in all new code.

---

## 1. Qt Object Lifetime $\rightarrow$ Segfault

- **Trap**: An unparented `QTimer` or helper `QObject` (e.g. an event filter or single-shot timer) can fire after the underlying C++ widget has been destroyed, causing an immediate hard segfault.
- **Rule**: **Always parent `QTimer`, event filters, and worker helpers to the widget they serve** so Qt destroys and cancels them together.
- **Example**: `v_graph.py` parents `ToolbarEventFilter` to the toolbar, and its `_update_icons` method guards against `RuntimeError` ("wrapped C/C++ object has been deleted").

---

## 2. Native Surface Budget in Tests $\rightarrow$ Segfault

- **Trap**: Allocating hundreds of real native widgets or matplotlib canvases in a single process exhausts the platform's window handle budget, leading to an abrupt crash.
- **Rule**: `tests/conftest.py` sets `QT_QPA_PLATFORM=offscreen` before `QApplication` initialization, and the autouse `_release_qt_widgets_between_tests` fixture performs generational GC between tests. Keep both intact. When running ad-hoc Qt scripts, set `$env:QT_QPA_PLATFORM="offscreen"`.

---

## 3. QSettings Isolation in Tests

- **Trap**: `MSettings` constructs `QSettings("CEA-Leti", "SPECTROview")`. On Windows, the two-argument convenience constructor continues writing to the **native Windows Registry** even if `QSettings.setDefaultFormat(IniFormat)` is called.
- **Rule**: `tests/conftest.py` monkeypatches `QSettings` inside `spectroview.model.m_settings` using the explicit four-argument constructor `QSettings(IniFormat, UserScope, org, app)` pointed at an isolated per-test INI file. Any new `QSettings` instantiation must follow this pattern to avoid polluting the host registry.

---

## 4. Twin & Secondary-Axis Coordinate Transforms

- **Trap**: `event.xdata` and `event.ydata` derive from `event.inaxes`, which resolves to the **topmost** overlapping Axes. When a secondary `twinx()` axis is present, it sits on top of the primary Axes, breaking annotation picking, dragging, and coordinates.
- **Rule**: Compute data coordinates from the raw pixel location via:
  ```python
  self.ax.transData.inverted().transform((event.x, event.y))
  ```
  Perform hit-testing with pixel-based `artist.contains(event)`, not matplotlib pick events. See `VGraph._ax_data_coords`.

---

## 5. Qt Binding Must Be PySide6

- **Trap**: Third-party packages such as `superqt` resolve Qt bindings via `qtpy`, which prioritizes PyQt5 if installed. Mixing PyQt5 and PySide6 objects in memory causes undefined behavior and segmentation faults.
- **Rule**: `spectroview/__init__.py` explicitly sets `os.environ["QT_API"] = "pyside6"` before any submodule imports `qtpy`. **Never `import PyQt5`**.

---

## 6. `ax.clear()` Does Not Restore Dark Theme

- **Trap**: Calling `ax.clear()` on an Axes removes styled spines, labels, and ticks, reverting them to default matplotlib colors without adapting to the active application theme.
- **Rule**: Reapply colors from `plt.rcParams` inside the figure's style context (see `_set_figure_style`). Ensure dark theme colors are reapplied whenever clearing an Axes.

---

## 7. Matplotlib Version Pinning

- **Trap**: `matplotlib` is pinned to `>=3.6.2,<3.10.9`. Certain artist return types differ across versions (for instance, `axvspan` returns a `Rectangle` in this version range, but a polygon in newer releases).
- **Rule**: Verify artist types and methods empirically rather than assuming newer matplotlib signatures.
