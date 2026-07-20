---
paths:
  - "spectroview/view/components/v_graph.py"
  - "spectroview/view/components/v_plot_renderer.py"
  - "spectroview/model/m_settings.py"
  - "tests/**/*.py"
  - "**/conftest.py"
---

# Common Pitfalls

Hard-won gotchas (the top four are also summarized in `CLAUDE.md`). Each has
bitten this codebase; the guard is in place — don't remove it, and follow the
same pattern in new code.

## Qt object lifetime → segfault
An unparented `QTimer`/`QObject` (e.g. an event filter with a single-shot timer)
can fire after its target's C++ object is deleted and crash. **Parent helpers to
the widget they serve** so Qt cancels/destroys them together. Example:
`v_graph.py`'s `ToolbarEventFilter` is parented to the toolbar and its
`_update_icons` guards `RuntimeError` (C++ object already deleted).

## Native surface budget in tests → segfault
Hundreds of real widgets/canvases in one process exhaust the platform's window
handles. `conftest.py` sets `QT_QPA_PLATFORM=offscreen` before `QApplication`
and GC-releases widgets between tests. Keep both; set `offscreen` in any ad-hoc
Qt script too.

## QSettings isolation in tests
`MSettings` builds `QSettings("CEA-Leti","SPECTROview")`. On this platform the
`(org, app)` convenience constructor keeps using the **native registry** even
after `setDefaultFormat(IniFormat)`. The autouse `_isolate_qsettings` fixture
monkeypatches `QSettings` inside `m_settings` to the explicit
`QSettings(IniFormat, UserScope, org, app)` form. Don't rely on the default-
format switch; if you add another `QSettings(...)` call site, route it so tests
stay isolated.

## Twin/secondary-axis coordinates
`event.xdata`/`event.ydata` come from `event.inaxes`, which resolves to the
**topmost** overlapping Axes — a `twinx()` secondary axis covers the primary and
breaks annotation picking/dragging. Compute data coords from the raw pixel via
`self.ax.transData.inverted().transform((event.x, event.y))` (see
`VGraph._ax_data_coords`). Hit-test with `artist.contains(event)` (pixel-based),
not matplotlib pick events.

## Qt binding must be PySide6
`superqt` resolves its binding via `qtpy`, which defaults to **PyQt5** if both
are installed. Mixing PyQt5- and PySide6-wrapped Qt objects is undefined
behavior. `__init__.py` forces `QT_API=pyside6` before anything imports qtpy.
Never `import PyQt5`.

## `ax.clear()` doesn't repaint theme
Clearing an Axes built under one theme and re-plotting doesn't retroactively fix
label/tick/spine colors; they must be re-applied from `plt.rcParams` inside the
plot's style context (see `_set_figure_style`). Watch for this when adding dark-
theme-sensitive rendering.

## matplotlib version is pinned
`matplotlib>=3.6.2,<3.10.9`. Some artist return types differ across versions
(e.g. `axvspan` returns a `Rectangle` here). Verify artist APIs empirically
rather than assuming when touching rendering/interaction code.

(Public-API stability is covered in [coding-standards.md](coding-standards.md).)
