# spectroview/view/components/customize_graph/tab_general.py
"""General tab: figure theme, grid lines, label rotation, typography font sizes, and data sorting."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from spectroview.view.components.customize_graph.common import (
    create_scrollable_container,
    highlight_group_style,
)

_THEME_TEXT_MAP = {"Light": "light", "Dark": "dark"}
_THEME_VALUE_TEXT = {"light": "Light", "dark": "Dark", "soft_dark": "Dark"}

_DEFAULT_TITLE_FONTSIZE = 12
_DEFAULT_AXIS_LABEL_FONTSIZE = 12
_DEFAULT_TICK_FONTSIZE = 9
_DEFAULT_COLORBAR_FONTSIZE = 10


class TabGeneral(QWidget):
    """Category 1: General & Typography."""

    KEYWORDS = [
        "general", "theme", "dark", "light", "figure theme", "grid",
        "rotation", "x label rotation", "angle", "font", "fonts",
        "font size", "size", "typography", "title", "title font",
        "axis label", "tick label", "colorbar", "colorbar font",
        "sort", "sorting", "order", "intelligent sorting", "data sorting",
    ]

    def __init__(self, graph_widget: QWidget, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.graph_widget = graph_widget
        self._setup_ui()
        self.load_settings()

    def switch_graph(self, graph_widget: QWidget) -> None:
        """Switch to a new graph widget and reload settings."""
        self.graph_widget = graph_widget
        self.load_settings()

    def _setup_ui(self) -> None:
        self._scroll, _container, layout = create_scrollable_container(self)

        # ───── Display & Grid ─────
        self.grp_display = QGroupBox("Display & Grid:")
        display_layout = QVBoxLayout(self.grp_display)
        display_layout.setContentsMargins(8, 8, 8, 8)
        display_layout.setSpacing(8)

        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel("Figure theme:"))
        self._combo_figure_theme = QComboBox()
        self._combo_figure_theme.addItems(["Light", "Dark"])
        theme_row.addWidget(self._combo_figure_theme)

        theme_row.addSpacing(12)
        theme_row.addWidget(QLabel("X label rotation:"))
        self._spin_xlabel_rotation = QSpinBox()
        self._spin_xlabel_rotation.setRange(0, 90)
        self._spin_xlabel_rotation.setSingleStep(10)
        self._spin_xlabel_rotation.setMaximumWidth(60)
        theme_row.addWidget(self._spin_xlabel_rotation)

        theme_row.addSpacing(12)
        self._cb_grid = QCheckBox("Grid lines")
        theme_row.addWidget(self._cb_grid)

        theme_row.addStretch()
        display_layout.addLayout(theme_row)
        layout.addWidget(self.grp_display)

        # ───── Typography & Font Sizes ─────
        self.grp_fonts = QGroupBox("Typography & Font Sizes (pt):")
        fonts_layout = QHBoxLayout(self.grp_fonts)
        fonts_layout.setContentsMargins(8, 8, 8, 8)
        fonts_layout.setSpacing(8)

        for label_text, attr, default in [
            ("Title:", "_spin_title_fontsize", _DEFAULT_TITLE_FONTSIZE),
            ("Axis label:", "_spin_axis_label_fontsize", _DEFAULT_AXIS_LABEL_FONTSIZE),
            ("Tick label:", "_spin_tick_fontsize", _DEFAULT_TICK_FONTSIZE),
            ("Colorbar:", "_spin_colorbar_fontsize", _DEFAULT_COLORBAR_FONTSIZE),
        ]:
            fonts_layout.addWidget(QLabel(label_text))
            spin = QSpinBox()
            spin.setRange(4, 72)
            spin.setSingleStep(1)
            spin.setValue(default)
            spin.setMaximumWidth(60)
            setattr(self, attr, spin)
            fonts_layout.addWidget(spin)

        fonts_layout.addStretch()
        layout.addWidget(self.grp_fonts)

        # ───── Data Sorting ─────
        self._sorting_group = QGroupBox("Data Sorting:")
        sort_layout = QVBoxLayout(self._sorting_group)
        sort_layout.setContentsMargins(8, 8, 8, 8)
        sort_layout.setSpacing(8)

        sort_row = QHBoxLayout()
        self._cb_sort_enabled = QCheckBox("Enable intelligent data sorting")
        self._cb_sort_enabled.setChecked(True)
        self._cb_sort_enabled.toggled.connect(self._on_sort_enabled_toggled)
        sort_row.addWidget(self._cb_sort_enabled)

        self._lbl_sort_by = QLabel("Sort by:")
        sort_row.addWidget(self._lbl_sort_by)

        self._cbb_sort_by = QComboBox()
        self._cbb_sort_by.addItems(["Z (hue values)", "X values", "Y values"])
        self._cbb_sort_by.setCurrentIndex(0)
        self._cbb_sort_by.setMaximumWidth(160)
        sort_row.addWidget(self._cbb_sort_by)

        sort_row.addStretch()
        sort_layout.addLayout(sort_row)

        sort_info = QLabel("Sorts legend and series drawing order for consistent, deterministic plots.")
        sort_info.setStyleSheet("color: gray; font-style: italic; font-size: 10px;")
        sort_info.setWordWrap(True)
        sort_layout.addWidget(sort_info)

        layout.addWidget(self._sorting_group)
        layout.addStretch()

    def _on_sort_enabled_toggled(self, checked: bool) -> None:
        self._cbb_sort_by.setVisible(checked)
        if hasattr(self, "_lbl_sort_by"):
            self._lbl_sort_by.setVisible(checked)

    def load_settings(self) -> None:
        """Load settings from current graph widget into controls."""
        gw = self.graph_widget
        self._combo_figure_theme.setCurrentText(
            _THEME_VALUE_TEXT.get(getattr(gw, "figure_theme", "light"), "Light")
        )
        self._spin_xlabel_rotation.setValue(getattr(gw, "x_rot", 0))
        self._cb_grid.setChecked(getattr(gw, "grid", False))

        self._spin_title_fontsize.setValue(getattr(gw, "title_fontsize", None) or _DEFAULT_TITLE_FONTSIZE)
        self._spin_axis_label_fontsize.setValue(
            getattr(gw, "axis_label_fontsize", None) or _DEFAULT_AXIS_LABEL_FONTSIZE
        )
        self._spin_tick_fontsize.setValue(getattr(gw, "tick_label_fontsize", None) or _DEFAULT_TICK_FONTSIZE)
        self._spin_colorbar_fontsize.setValue(
            getattr(gw, "colorbar_fontsize", None) or _DEFAULT_COLORBAR_FONTSIZE
        )

        sort_enabled = getattr(gw, "sort_data_enabled", True)
        self._cb_sort_enabled.setChecked(sort_enabled)
        self._cbb_sort_by.setVisible(sort_enabled)
        if hasattr(self, "_lbl_sort_by"):
            self._lbl_sort_by.setVisible(sort_enabled)

        sort_by = getattr(gw, "sort_data_by", "Z")
        sort_map = {"Z": 0, "X": 1, "Y": 2}
        self._cbb_sort_by.setCurrentIndex(sort_map.get(sort_by, 0))

    def apply_settings(self, replot: bool = False) -> dict:
        """Write controls back to graph widget, returning changed property dict."""
        gw = self.graph_widget
        gw.figure_theme = _THEME_TEXT_MAP[self._combo_figure_theme.currentText()]
        gw.x_rot = self._spin_xlabel_rotation.value()
        gw.grid = self._cb_grid.isChecked()

        gw.title_fontsize = self._spin_title_fontsize.value()
        gw.axis_label_fontsize = self._spin_axis_label_fontsize.value()
        gw.tick_label_fontsize = self._spin_tick_fontsize.value()
        gw.colorbar_fontsize = self._spin_colorbar_fontsize.value()

        old_sort_enabled = getattr(gw, "sort_data_enabled", True)
        old_sort_by = getattr(gw, "sort_data_by", "Z")
        gw.sort_data_enabled = self._cb_sort_enabled.isChecked()
        sort_index = self._cbb_sort_by.currentIndex()
        gw.sort_data_by = ["Z", "X", "Y"][sort_index]

        if gw.sort_data_enabled != old_sort_enabled or gw.sort_data_by != old_sort_by:
            gw.legend_properties = []

        if replot and getattr(gw, "df", None) is not None:
            gw.plot(gw.df)

        return {
            "figure_theme": gw.figure_theme,
            "x_rot": gw.x_rot,
            "grid": gw.grid,
            "title_fontsize": gw.title_fontsize,
            "axis_label_fontsize": gw.axis_label_fontsize,
            "tick_label_fontsize": gw.tick_label_fontsize,
            "colorbar_fontsize": gw.colorbar_fontsize,
            "sort_data_enabled": gw.sort_data_enabled,
            "sort_data_by": gw.sort_data_by,
        }

    def highlight_search(self, query: str) -> None:
        """Highlight matching groupboxes based on search query."""
        q = query.strip().lower()
        self.grp_display.setStyleSheet("")
        self.grp_fonts.setStyleSheet("")
        self._sorting_group.setStyleSheet("")

        if not q:
            return

        display_terms = ["display", "theme", "dark", "light", "grid", "rotation", "angle"]
        font_terms = ["font", "size", "typography", "title", "label", "tick", "colorbar", "pt"]
        sort_terms = ["sort", "sorting", "order", "intelligent", "z", "hue"]

        first_match: Optional[QWidget] = None
        if any(t in q or q in t for t in display_terms):
            self.grp_display.setStyleSheet(highlight_group_style())
            if first_match is None:
                first_match = self.grp_display

        if any(t in q or q in t for t in font_terms):
            self.grp_fonts.setStyleSheet(highlight_group_style())
            if first_match is None:
                first_match = self.grp_fonts

        if any(t in q or q in t for t in sort_terms):
            self._sorting_group.setStyleSheet(highlight_group_style())
            if first_match is None:
                first_match = self._sorting_group

        if first_match:
            self._scroll.ensureWidgetVisible(first_match, 0, 30)
