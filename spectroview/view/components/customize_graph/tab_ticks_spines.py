# spectroview/view/components/customize_graph/tab_ticks_spines.py
"""Ticks & Spines tab: tick direction, tick label formatting, minor ticks, and spine frame borders."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from spectroview.view.components.customize_graph.common import (
    create_scrollable_container,
    highlight_group_style,
)

_TICK_DIRECTION_MAP = {"Default": None, "In": "in", "Out": "out", "In & Out": "inout"}
_TICK_DIRECTION_TEXT = {v: k for k, v in _TICK_DIRECTION_MAP.items()}

_TICK_FORMAT_PRESETS = [
    ("Auto (default)", None),
    ("Integer  (e.g. 1234)", "%.0f"),
    ("1 decimal  (e.g. 12.3)", "%.1f"),
    ("2 decimals  (e.g. 12.34)", "%.2f"),
    ("Scientific  (e.g. 1.2e+03)", "%.1e"),
]


class TabTicksSpines(QWidget):
    """Category 3: Ticks & Spines."""

    KEYWORDS = [
        "ticks", "tick", "direction", "tick direction", "in", "out",
        "inout", "format", "tick format", "label format", "decimals",
        "scientific", "minor ticks", "minor", "spines", "spine",
        "border", "borders", "frame", "top", "bottom", "left", "right",
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

        # ───── Tick Appearance ─────
        self.grp_ticks = QGroupBox("Tick Appearance:")
        ticks_layout = QVBoxLayout(self.grp_ticks)
        ticks_layout.setContentsMargins(8, 8, 8, 8)
        ticks_layout.setSpacing(6)

        tick_row = QHBoxLayout()
        lbl_td = QLabel("Tick direction:")
        lbl_td.setToolTip("Direction of tick marks relative to the axes spine: In (pointing inward), Out (outward), or In & Out.")
        tick_row.addWidget(lbl_td)
        self.combo_tick_direction = QComboBox()
        self.combo_tick_direction.addItems(list(_TICK_DIRECTION_MAP.keys()))
        self.combo_tick_direction.setMinimumWidth(120)
        self.combo_tick_direction.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        if self.combo_tick_direction.view():
            self.combo_tick_direction.view().setMinimumWidth(120)
        self.combo_tick_direction.setToolTip("Direction of tick marks relative to the axes spine: In (pointing inward), Out (outward), or In & Out.")
        tick_row.addWidget(self.combo_tick_direction)

        tick_row.addSpacing(16)
        lbl_tf = QLabel("Tick label format:")
        lbl_tf.setToolTip("Numeric formatting applied to major tick values (integer, decimal places, or scientific notation).")
        tick_row.addWidget(lbl_tf)
        self.combo_tick_format = QComboBox()
        for label, value in _TICK_FORMAT_PRESETS:
            self.combo_tick_format.addItem(label, value)
        self.combo_tick_format.setMinimumWidth(180)
        self.combo_tick_format.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        if self.combo_tick_format.view():
            self.combo_tick_format.view().setMinimumWidth(180)
        self.combo_tick_format.setToolTip("Numeric formatting applied to major tick values (integer, decimal places, or scientific notation).")
        tick_row.addWidget(self.combo_tick_format)

        tick_row.addStretch()
        ticks_layout.addLayout(tick_row)
        layout.addWidget(self.grp_ticks)

        # ───── Minor Ticks ─────
        self.grp_minor = QGroupBox("Show Minor Ticks:")
        minor_layout = QHBoxLayout(self.grp_minor)
        minor_layout.setContentsMargins(8, 8, 8, 8)
        minor_layout.setSpacing(16)

        self.cb_minor_bottom = QCheckBox("Bottom")
        self.cb_minor_bottom.setToolTip("Show minor subdivision tick marks along the bottom X axis.")
        self.cb_minor_left = QCheckBox("Left")
        self.cb_minor_left.setToolTip("Show minor subdivision tick marks along the left Y axis.")
        self.cb_minor_top = QCheckBox("Top")
        self.cb_minor_top.setToolTip("Show minor subdivision tick marks along the top X axis.")
        self.cb_minor_right = QCheckBox("Right")
        self.cb_minor_right.setToolTip("Show minor subdivision tick marks along the right Y axis.")

        minor_layout.addWidget(self.cb_minor_bottom)
        minor_layout.addWidget(self.cb_minor_left)
        minor_layout.addWidget(self.cb_minor_top)
        minor_layout.addWidget(self.cb_minor_right)
        minor_layout.addStretch()
        layout.addWidget(self.grp_minor)

        # ───── Spines (Frame Borders) ─────
        self.grp_spines = QGroupBox("Show Spines (Frame Borders):")
        spines_layout = QHBoxLayout(self.grp_spines)
        spines_layout.setContentsMargins(8, 8, 8, 8)
        spines_layout.setSpacing(16)

        self.cb_spine_bottom = QCheckBox("Bottom")
        self.cb_spine_bottom.setToolTip("Draw the bottom bounding spine line.")
        self.cb_spine_left = QCheckBox("Left")
        self.cb_spine_left.setToolTip("Draw the left bounding spine line.")
        self.cb_spine_top = QCheckBox("Top")
        self.cb_spine_top.setToolTip("Draw the top bounding spine line.")
        self.cb_spine_right = QCheckBox("Right")
        self.cb_spine_right.setToolTip("Draw the right bounding spine line.")

        spines_layout.addWidget(self.cb_spine_bottom)
        spines_layout.addWidget(self.cb_spine_left)
        spines_layout.addWidget(self.cb_spine_top)
        spines_layout.addWidget(self.cb_spine_right)
        spines_layout.addStretch()
        layout.addWidget(self.grp_spines)

        layout.addStretch()

    def _load_tick_format(self, value: Optional[str]) -> None:
        combo = self.combo_tick_format
        idx = combo.findData(value)
        if combo.itemData(combo.count() - 1) not in dict(_TICK_FORMAT_PRESETS).values():
            combo.removeItem(combo.count() - 1)
            idx = combo.findData(value)
        if idx < 0:
            combo.addItem(f"Custom  ({value})", value)
            idx = combo.count() - 1
        combo.setCurrentIndex(idx)

    def load_settings(self) -> None:
        """Load tick and spine settings from current graph widget."""
        gw = self.graph_widget
        self.combo_tick_direction.setCurrentText(
            _TICK_DIRECTION_TEXT.get(getattr(gw, "tick_direction", None), "Default")
        )
        self._load_tick_format(getattr(gw, "tick_label_format", None))

        self.cb_minor_bottom.setChecked(getattr(gw, "minor_ticks_bottom", True))
        self.cb_minor_left.setChecked(getattr(gw, "minor_ticks_left", True))
        self.cb_minor_top.setChecked(getattr(gw, "minor_ticks_top", False))
        self.cb_minor_right.setChecked(getattr(gw, "minor_ticks_right", False))

        spines = getattr(gw, "spines_visible", None) or {"top": True, "right": True, "bottom": True, "left": True}
        self.cb_spine_top.setChecked(spines.get("top", True))
        self.cb_spine_right.setChecked(spines.get("right", True))
        self.cb_spine_bottom.setChecked(spines.get("bottom", True))
        self.cb_spine_left.setChecked(spines.get("left", True))

    def apply_settings(self, replot: bool = False) -> dict:
        """Write controls back to graph widget, returning changed properties dict."""
        gw = self.graph_widget
        gw.tick_direction = _TICK_DIRECTION_MAP[self.combo_tick_direction.currentText()]
        gw.tick_label_format = self.combo_tick_format.currentData()

        gw.minor_ticks_bottom = self.cb_minor_bottom.isChecked()
        gw.minor_ticks_left = self.cb_minor_left.isChecked()
        gw.minor_ticks_top = self.cb_minor_top.isChecked()
        gw.minor_ticks_right = self.cb_minor_right.isChecked()

        gw.spines_visible = {
            "top": self.cb_spine_top.isChecked(),
            "right": self.cb_spine_right.isChecked(),
            "bottom": self.cb_spine_bottom.isChecked(),
            "left": self.cb_spine_left.isChecked(),
        }

        if replot and getattr(gw, "df", None) is not None:
            gw.plot(gw.df)

        return {
            "tick_direction": gw.tick_direction,
            "tick_label_format": gw.tick_label_format,
            "minor_ticks_bottom": gw.minor_ticks_bottom,
            "minor_ticks_left": gw.minor_ticks_left,
            "minor_ticks_top": gw.minor_ticks_top,
            "minor_ticks_right": gw.minor_ticks_right,
            "spines_visible": gw.spines_visible,
        }

    def highlight_search(self, query: str) -> None:
        """Highlight matching groupboxes based on search query."""
        q = query.strip().lower()
        self.grp_ticks.setStyleSheet("")
        self.grp_minor.setStyleSheet("")
        self.grp_spines.setStyleSheet("")

        if not q:
            return

        tick_terms = ["tick", "direction", "format", "scientific", "decimal", "integer", "in", "out"]
        minor_terms = ["minor", "minor ticks"]
        spine_terms = ["spine", "spines", "border", "borders", "frame"]

        first_match: Optional[QWidget] = None
        if any(t in q or q in t for t in tick_terms):
            self.grp_ticks.setStyleSheet(highlight_group_style())
            if first_match is None:
                first_match = self.grp_ticks

        if any(t in q or q in t for t in minor_terms):
            self.grp_minor.setStyleSheet(highlight_group_style())
            if first_match is None:
                first_match = self.grp_minor

        if any(t in q or q in t for t in spine_terms):
            self.grp_spines.setStyleSheet(highlight_group_style())
            if first_match is None:
                first_match = self.grp_spines

        if first_match:
            self._scroll.ensureWidgetVisible(first_match, 0, 30)

