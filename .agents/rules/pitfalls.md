---
name: pitfalls
description: Critical pitfalls, segfault traps, Qt lifetime hazards, coordinate transforms, and PySide6 compatibility rules for SPECTROview.
trigger: model_decision
---

# Common Pitfalls & Segfault Traps

Hard-won gotchas that have bitten this codebase. The guards described below are in place — never remove them, and apply the same patterns in all new code.

---

## 1. Qt Object Lifetime $\rightarrow$ Segfault & Test Failure

- **Trap**:
  1. An unparented `QTimer` or helper `QObject` can fire after the underlying C++ widget has been destroyed, causing an immediate hard segfault.
  2. Calling `QTimer.singleShot(msec, lambda: widget.method())` registers the lambda on the global Qt event loop without binding it to the widget's lifetime. When the widget is destroyed at test teardown, the timer fires during subsequent tests, raising:
     ```text
     RuntimeError: Internal C++ object (PySide6.QtWidgets.QScrollArea) already deleted
     ```
- **Rule**:
  1. **Always parent `QTimer`, event filters, and worker helpers to the widget they serve** so Qt destroys and cancels them together.
  2. **Always bind single-shot timers to a context QObject**:
     ```python
     # Correct: Qt automatically disarms the timer if `self` is deleted
     QTimer.singleShot(msec, self, self._on_timeout)
     ```
  3. Inside the callback method, wrap accesses defensively with `try...except (RuntimeError, AttributeError)`.

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

---

## 8. $O(N \times M)$ Event Filter & Leaked Top-Level Widget Trap $\rightarrow$ Test Hang

- **Trap**: Installing an event filter on `QApplication.instance()` (e.g. for global tooltips or drag-drop monitoring) without auto-deregistration leaks filter instances into the global event dispatcher. If GUI tests instantiate top-level widgets (`QWidget()`, `QMainWindow()`) without closing them, $N$ unclosed widgets and $M$ orphaned event filters accumulate. When any subsequent test or theme update calls `app.setStyleSheet(...)`, Qt dispatches `FontChange`/`PaletteChange`/`StyleChange` events to all $N$ widgets through all $M$ filters across the Shiboken C++/Python boundary, triggering millions of redundant Python calls ($N \times M \approx 35,000 \times 500 = 17,500,000$ calls) and stalling the test suite for 10+ minutes.
- **Rule**:
  1. Any event filter installed on `QApplication.instance()` must monitor the destruction of its target widget/viewport and call `app.removeEventFilter(self)`.
  2. Inside `eventFilter(watched, event)`, guard against deleted C++ objects with `isValid(watched)` and return early.
  3. `tests/conftest.py`'s `_release_qt_widgets_between_tests` fixture snapshots and closes newly created root widgets and matplotlib figures between tests. Test authors should still explicitly call `widget.close()` and `widget.deleteLater()`.
