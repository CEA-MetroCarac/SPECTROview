# spectroview/view/components/customize_graph/customize_graph_dialog.py
"""Modern two-panel Customize Graph dialog with left sidebar navigation, real-time search, and modular tabs."""

from __future__ import annotations

import copy
from typing import Optional

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from spectroview.model.graph_style import can_restyle_without_replot
from spectroview.view.components.customize_graph.common import NavItemDelegate
from spectroview.view.components.customize_graph.customize_annotation_dialogs import (
    ColorDelegate,
    EditArrowDialog,
    EditBoxDialog,
    EditCalloutDialog,
    EditLineDialog,
    EditSpanDialog,
    EditTextDialog,
)
from spectroview.view.components.customize_graph.customize_annotations import CustomizeAnnotations
from spectroview.view.components.customize_graph.customize_axis import CustomizeAxis
from spectroview.view.components.customize_graph.customize_legend import CustomizeLegend
from spectroview.view.components.customize_graph.customize_more_options import CustomizeMoreOptions
from spectroview.view.components.customize_graph.tab_annotations import TabAnnotations
from spectroview.view.components.customize_graph.tab_axes_limits import TabAxesLimits
from spectroview.view.components.customize_graph.tab_general import TabGeneral
from spectroview.view.components.customize_graph.tab_legend_series import TabLegendSeries
from spectroview.view.components.customize_graph.tab_plot_styles import TabPlotStyles
from spectroview.view.components.customize_graph.tab_secondary_inset import TabSecondaryInset
from spectroview.view.components.customize_graph.tab_ticks_spines import TabTicksSpines
from spectroview.view.components.graph_commit import diff, snapshot
from spectroview.view.theme import Icons

__all__ = [
    "CustomizeGraphDialog",
    "TabGeneral",
    "TabAxesLimits",
    "TabTicksSpines",
    "TabSecondaryInset",
    "TabLegendSeries",
    "TabPlotStyles",
    "TabAnnotations",
    # Backward compatibility exports
    "CustomizeAxis",
    "CustomizeLegend",
    "CustomizeMoreOptions",
    "CustomizeAnnotations",
    "EditLineDialog",
    "EditTextDialog",
    "EditArrowDialog",
    "EditSpanDialog",
    "EditBoxDialog",
    "EditCalloutDialog",
    "ColorDelegate",
]



# ---------------------------------------------------------------------------
# Compatibility Proxy for Legacy QTabWidget Access
# ---------------------------------------------------------------------------

class _TabsCompatibilityProxy:
    """Provides backward-compatibility for callers or tests expecting QTabWidget methods on dialog.tabs."""

    def __init__(self, dialog: CustomizeGraphDialog) -> None:
        self._dialog = dialog

    def count(self) -> int:
        return self._dialog._stack.count()

    def widget(self, index: int) -> Optional[QWidget]:
        return self._dialog._stack.widget(index)

    def indexOf(self, widget: QWidget) -> int:
        return self._dialog._stack.indexOf(widget)

    def tabText(self, index: int) -> str:
        if 0 <= index < self._dialog._nav_list.count():
            item = self._dialog._nav_list.item(index)
            return item.text() if item else ""
        return ""

    def setCurrentWidget(self, widget: QWidget) -> None:
        idx = self._dialog._stack.indexOf(widget)
        if idx >= 0:
            self._dialog._nav_list.setCurrentRow(idx)

    def currentWidget(self) -> Optional[QWidget]:
        return self._dialog._stack.currentWidget()

    def currentIndex(self) -> int:
        return self._dialog._stack.currentIndex()

    def setCurrentIndex(self, index: int) -> None:
        if 0 <= index < self._dialog._nav_list.count():
            self._dialog._nav_list.setCurrentRow(index)


# ---------------------------------------------------------------------------
# Facades for Backward Compatibility with Sub-Widget References
# ---------------------------------------------------------------------------

class _AxisCompatibilityFacade:
    """Combines TabAxesLimits, TabTicksSpines, and TabSecondaryInset attributes."""

    def __init__(self, dialog: CustomizeGraphDialog) -> None:
        self._dialog = dialog

    @property
    def graph_widget(self):
        return self._dialog.graph_widget

    def __getattr__(self, name: str):
        for tab in (self._dialog.tab_axes_limits, self._dialog.tab_ticks_spines, self._dialog.tab_secondary_inset):
            if hasattr(tab, name):
                return getattr(tab, name)
        raise AttributeError(f"'_AxisCompatibilityFacade' has no attribute '{name}'")

    def _apply_axis_settings(self, silent: bool = False, replot: bool = True):
        self._dialog.tab_axes_limits.apply_settings(silent=silent, replot=False)
        self._dialog.tab_ticks_spines.apply_settings(replot=False)
        self._dialog.tab_secondary_inset.apply_settings(replot=replot)

    def load_axis_settings(self):
        self._dialog.tab_axes_limits.load_settings()
        self._dialog.tab_ticks_spines.load_settings()
        self._dialog.tab_secondary_inset.load_settings()

    def switch_graph(self, gw):
        self._dialog.tab_axes_limits.switch_graph(gw)
        self._dialog.tab_ticks_spines.switch_graph(gw)
        self._dialog.tab_secondary_inset.switch_graph(gw)


class _MoreOptionsCompatibilityFacade:
    """Combines TabGeneral and TabPlotStyles attributes."""

    def __init__(self, dialog: CustomizeGraphDialog) -> None:
        self._dialog = dialog

    @property
    def graph_widget(self):
        return self._dialog.graph_widget

    def __getattr__(self, name: str):
        for tab in (self._dialog.tab_general, self._dialog.tab_plot_styles):
            if hasattr(tab, name):
                return getattr(tab, name)
        raise AttributeError(f"'_MoreOptionsCompatibilityFacade' has no attribute '{name}'")

    def _apply(self, replot: bool = True):
        self._dialog.tab_general.apply_settings(replot=False)
        self._dialog.tab_plot_styles.apply_settings(replot=replot)

    def load_settings(self):
        self._dialog.tab_general.load_settings()
        self._dialog.tab_plot_styles.load_settings()

    def switch_graph(self, gw):
        self._dialog.tab_general.switch_graph(gw)
        self._dialog.tab_plot_styles.switch_graph(gw)


# ---------------------------------------------------------------------------
# Main CustomizeGraphDialog
# ---------------------------------------------------------------------------

class CustomizeGraphDialog(QDialog):
    """Modern two-panel Customize Graph dialog with left sidebar navigation and search."""

    _LIVE_PREVIEW_SIGNALS = ("valueChanged", "toggled", "currentIndexChanged", "textChanged")

    def __init__(self, graph_widget: QWidget, graph_id: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.graph_widget = graph_widget
        self.graph_id = graph_id

        self.setWindowTitle(f"Customize Graph #{graph_id}")
        self.setModal(False)
        self.setWindowFlags(
            Qt.Window | Qt.Tool | Qt.CustomizeWindowHint
            | Qt.WindowTitleHint | Qt.WindowCloseButtonHint
        )
        self.setAttribute(Qt.WA_MacAlwaysShowToolWindow)
        self.resize(840, 680)
        self.setMinimumSize(740, 560)

        self._setup_ui()

        self._original_snapshot = snapshot(self.graph_widget)

        # Debounced live preview timer (~400ms)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(400)
        self._preview_timer.timeout.connect(self._preview_apply)
        self._wire_live_preview()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Center Split: Left Sidebar + Right Content Area
        split_widget = QWidget()
        split_layout = QHBoxLayout(split_widget)
        split_layout.setContentsMargins(0, 0, 0, 0)
        split_layout.setSpacing(0)

        # -- Left Sidebar ---------------------------------------------------
        self._sidebar = QWidget()
        self._sidebar.setFixedWidth(230)
        sidebar_layout = QVBoxLayout(self._sidebar)
        sidebar_layout.setContentsMargins(10, 14, 10, 10)
        sidebar_layout.setSpacing(10)

        # Top Search Bar
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search options...")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.textChanged.connect(self._on_search_text_changed)
        self._search_edit.returnPressed.connect(self._on_search_return_pressed)
        self._search_edit.installEventFilter(self)
        sidebar_layout.addWidget(self._search_edit)

        # Navigation List
        self._nav_list = QListWidget()
        self._nav_list.setFrameShape(QFrame.NoFrame)
        self._nav_list.setItemDelegate(NavItemDelegate(self))
        self._nav_list.currentRowChanged.connect(self._on_nav_row_changed)
        sidebar_layout.addWidget(self._nav_list, 1)

        split_layout.addWidget(self._sidebar)

        # Vertical Divider
        self._v_divider = QFrame()
        self._v_divider.setFrameShape(QFrame.VLine)
        split_layout.addWidget(self._v_divider)

        # -- Right Content Area ----------------------------------------------
        self._content_container = QWidget()
        content_layout = QVBoxLayout(self._content_container)
        content_layout.setContentsMargins(10, 6, 10, 6)
        content_layout.setSpacing(4)

        # Section Title Header
        self._content_title = QLabel()
        self._content_title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 2px 0;")
        content_layout.addWidget(self._content_title)

        self._header_divider = QFrame()
        self._header_divider.setFrameShape(QFrame.HLine)
        content_layout.addWidget(self._header_divider)

        # Stacked Pages
        self._stack = QStackedWidget()
        content_layout.addWidget(self._stack, 1)

        split_layout.addWidget(self._content_container, 1)
        main_layout.addWidget(split_widget, 1)

        # -- Bottom Action Bar -----------------------------------------------
        self._bottom_divider = QFrame()
        self._bottom_divider.setFrameShape(QFrame.HLine)
        main_layout.addWidget(self._bottom_divider)

        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(16, 10, 16, 10)

        self._graph_info_label = QLabel()
        self._graph_info_label.setStyleSheet("color: gray; font-size: 11px;")
        bottom_bar.addWidget(self._graph_info_label)

        bottom_bar.addStretch(1)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setIcon(Icons.close())
        btn_cancel.setCursor(Qt.PointingHandCursor)
        btn_cancel.clicked.connect(self.cancel_all)
        bottom_bar.addWidget(btn_cancel)

        btn_apply = QPushButton("Apply")
        btn_apply.setIcon(Icons.check())
        btn_apply.setCursor(Qt.PointingHandCursor)
        btn_apply.clicked.connect(self.apply_all)
        bottom_bar.addWidget(btn_apply)

        bottom_widget = QWidget()
        bottom_widget.setLayout(bottom_bar)
        main_layout.addWidget(bottom_widget)

        # -- Initialize Tab Pages --------------------------------------------
        self.tab_general = TabGeneral(self.graph_widget, self)
        self.tab_axes_limits = TabAxesLimits(self.graph_widget, self)
        self.tab_ticks_spines = TabTicksSpines(self.graph_widget, self)
        self.tab_secondary_inset = TabSecondaryInset(self.graph_widget, self)
        self.tab_legend = TabLegendSeries(self.graph_widget, self)
        self.tab_plot_styles = TabPlotStyles(self.graph_widget, self)
        self.tab_annotations = TabAnnotations(self.graph_widget, self)

        # Backward compatibility properties
        self.tabs = _TabsCompatibilityProxy(self)
        self.legend_widget = self.tab_legend
        self.annotations_widget = self.tab_annotations
        self.axis_widget = _AxisCompatibilityFacade(self)
        self.more_options_widget = _MoreOptionsCompatibilityFacade(self)

        # Sections definition: (widget, title, icon, keywords)
        self._sections: list[tuple[QWidget, str, any, list[str]]] = [
            (self.tab_general, "General & Typography", Icons.settings(size=20), self.tab_general.KEYWORDS),
            (self.tab_axes_limits, "Axes & Limits", Icons.rescale(size=20), self.tab_axes_limits.KEYWORDS),
            (self.tab_ticks_spines, "Ticks & Spines", Icons.sliders(size=20), self.tab_ticks_spines.KEYWORDS),
            (self.tab_legend, "Legend & Series", Icons.legend(size=20), self.tab_legend.KEYWORDS),
            (self.tab_plot_styles, "Plot Style & Fitting", Icons.fit_curve(size=20), self.tab_plot_styles.KEYWORDS),
            (self.tab_annotations, "Annotations", Icons.edit(size=20), self.tab_annotations.KEYWORDS),
            (self.tab_secondary_inset, "Secondary & Inset Axes", Icons.duplicate(size=20), self.tab_secondary_inset.KEYWORDS),
        ]

        for page, title, icon, _kw in self._sections:
            self._stack.addWidget(page)
            item = QListWidgetItem(icon, title)
            self._nav_list.addItem(item)

        # Select first tab initially
        self._nav_list.setCurrentRow(0)
        self._update_graph_info()

    def _update_graph_info(self) -> None:
        style = getattr(self.graph_widget, "plot_style", "standard")
        self._graph_info_label.setText(f"Target: Graph #{self.graph_id} • {style.capitalize()} plot")

    def _vm(self):
        return getattr(self.parent(), "vm", None)

    # ──────────────────────────────────────────────
    # Live Preview & Debouncing
    # ──────────────────────────────────────────────
    def _wire_live_preview(self) -> None:
        seen = set()
        for w in self.findChildren(QWidget):
            if id(w) in seen:
                continue
            seen.add(id(w))
            for signal_name in self._LIVE_PREVIEW_SIGNALS:
                sig = getattr(w, signal_name, None)
                if sig is not None and hasattr(sig, "connect"):
                    try:
                        sig.connect(self._schedule_preview)
                    except (TypeError, RuntimeError):
                        pass

    def _schedule_preview(self, *_args) -> None:
        self._preview_timer.start()

    def _preview_apply(self) -> None:
        gw = self.graph_widget
        before = snapshot(gw)
        gw.blockSignals(True)
        try:
            self.tab_general.apply_settings(replot=False)
            self.tab_axes_limits.apply_settings(silent=True, replot=False)
            self.tab_ticks_spines.apply_settings(replot=False)
            self.tab_secondary_inset.apply_settings(replot=False)
            self.tab_legend.apply_settings(replot=False)
            self.tab_plot_styles.apply_settings(replot=False)
            self.tab_annotations.apply_settings(replot=False)
        finally:
            gw.blockSignals(False)

        changed = set(diff(gw, before).keys())
        if not changed:
            return

        if can_restyle_without_replot(changed) and gw.restyle():
            return

        if getattr(gw, "df", None) is not None:
            gw.plot(gw.df)
        elif hasattr(gw, "canvas") and gw.canvas is not None:
            gw.canvas.draw_idle()

        if getattr(gw, "plot_style", "") == "trendline":
            self.tab_plot_styles._refresh_equation_table()

    # ──────────────────────────────────────────────
    # Navigation & Real-Time Search Filtering
    # ──────────────────────────────────────────────
    def _on_nav_row_changed(self, row: int) -> None:
        if 0 <= row < self._stack.count():
            self._stack.setCurrentIndex(row)
            item = self._nav_list.item(row)
            if item:
                self._content_title.setText(item.text())
            curr_page = self._stack.currentWidget()
            if hasattr(curr_page, "highlight_search"):
                curr_page.highlight_search(self._search_edit.text())

    def _on_search_text_changed(self, query: str) -> None:
        q = query.strip().lower()
        first_visible = -1
        current_visible = False
        current_row = self._nav_list.currentRow()

        for idx, (_page, title, _icon, keywords) in enumerate(self._sections):
            item = self._nav_list.item(idx)
            if not item:
                continue
            matches = not q or q in title.lower() or any(q in kw.lower() for kw in keywords)
            item.setHidden(not matches)
            if matches:
                if first_visible < 0:
                    first_visible = idx
                if idx == current_row:
                    current_visible = True

        if not current_visible and first_visible >= 0:
            self._nav_list.setCurrentRow(first_visible)

        curr_page = self._stack.currentWidget()
        if hasattr(curr_page, "highlight_search"):
            curr_page.highlight_search(q)

    def _on_search_return_pressed(self) -> None:
        current_row = self._nav_list.currentRow()
        if 0 <= current_row < self._nav_list.count():
            item = self._nav_list.item(current_row)
            if item and not item.isHidden():
                self._nav_list.setFocus()
                return
        for idx in range(self._nav_list.count()):
            item = self._nav_list.item(idx)
            if item and not item.isHidden():
                self._nav_list.setCurrentRow(idx)
                self._nav_list.setFocus()
                break

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self._search_edit and event.type() == QEvent.KeyPress:
            key_event = event  # type: ignore[assignment]
            if key_event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._on_search_return_pressed()
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self._search_edit.hasFocus():
                self._on_search_return_pressed()
                event.accept()
                return
            focused = self.focusWidget()
            if not isinstance(focused, QPushButton):
                event.accept()
                return
        super().keyPressEvent(event)

    def select_section(self, name: str) -> None:
        """Select a section by name: 'general', 'axes', 'ticks', 'secondary', 'legend', 'plot_styles', 'annotations'."""
        name_lower = name.lower()
        if name_lower in ("general", "gen", "typography", "fonts"):
            idx = self._stack.indexOf(self.tab_general)
        elif name_lower in ("axes", "axis", "limits", "scale"):
            idx = self._stack.indexOf(self.tab_axes_limits)
        elif name_lower in ("ticks", "spines", "borders"):
            idx = self._stack.indexOf(self.tab_ticks_spines)
        elif name_lower in ("secondary", "inset", "twin", "multi"):
            idx = self._stack.indexOf(self.tab_secondary_inset)
        elif name_lower in ("legend", "series", "colors"):
            idx = self._stack.indexOf(self.tab_legend)
        elif name_lower in ("plot_styles", "style", "fitting", "trendline", "histogram"):
            idx = self._stack.indexOf(self.tab_plot_styles)
        elif name_lower in ("annotations", "shapes", "annotation"):
            idx = self._stack.indexOf(self.tab_annotations)
        else:
            idx = 0

        if idx >= 0:
            self._nav_list.setCurrentRow(idx)

    # ──────────────────────────────────────────────
    # Apply, Cancel, Switch Graph
    # ──────────────────────────────────────────────
    def apply_all(self) -> None:
        """Apply changes from all tabs and commit undo batch."""
        self._preview_timer.stop()
        vm = self._vm()
        if vm:
            vm.begin_undo_batch()
        try:
            self.tab_general.apply_settings(replot=False)
            self.tab_axes_limits.apply_settings(silent=False, replot=False)
            self.tab_ticks_spines.apply_settings(replot=False)
            self.tab_secondary_inset.apply_settings(replot=False)
            self.tab_legend.apply_settings(replot=False)
            self.tab_plot_styles.apply_settings(replot=False)
            self.tab_annotations.apply_settings(replot=False)
        finally:
            if vm:
                vm.end_undo_batch()

        gw = self.graph_widget
        if getattr(gw, "df", None) is not None:
            gw.plot(gw.df)
        elif hasattr(gw, "canvas") and gw.canvas is not None:
            gw.canvas.draw()

        if getattr(gw, "plot_style", "") == "trendline":
            self.tab_plot_styles._refresh_equation_table()

        self.tab_legend.load_settings()
        self._original_snapshot = snapshot(self.graph_widget)
        self._preview_timer.stop()

    def cancel_all(self) -> None:
        """Discard changes and restore baseline snapshot."""
        self._preview_timer.stop()
        gw = self.graph_widget
        for field, value in self._original_snapshot.items():
            setattr(gw, field, copy.deepcopy(value))

        if getattr(gw, "df", None) is not None:
            gw.plot(gw.df)
        elif hasattr(gw, "canvas") and gw.canvas is not None:
            gw.canvas.draw()

        self.tab_general.load_settings()
        self.tab_axes_limits.load_settings()
        self.tab_ticks_spines.load_settings()
        self.tab_secondary_inset.load_settings()
        self.tab_legend.load_settings()
        self.tab_plot_styles.load_settings()
        self.tab_annotations.load_settings()
        self._preview_timer.stop()

    def switch_graph(self, graph_widget: QWidget, graph_id: int) -> None:
        """Switch dialog to a different graph widget and reload all tabs."""
        if self.graph_id == graph_id and self.graph_widget is graph_widget:
            return

        self._preview_timer.stop()
        self.graph_widget = graph_widget
        self.graph_id = graph_id
        self.setWindowTitle(f"Customize Graph #{graph_id}")
        self._update_graph_info()

        self.tab_general.switch_graph(graph_widget)
        self.tab_axes_limits.switch_graph(graph_widget)
        self.tab_ticks_spines.switch_graph(graph_widget)
        self.tab_secondary_inset.switch_graph(graph_widget)
        self.tab_legend.switch_graph(graph_widget)
        self.tab_plot_styles.switch_graph(graph_widget)
        self.tab_annotations.switch_graph(graph_widget)

        self._original_snapshot = snapshot(self.graph_widget)
        self._preview_timer.stop()
