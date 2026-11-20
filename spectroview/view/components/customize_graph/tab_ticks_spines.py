# spectroview/view/components/customize_graph/tab_ticks_spines.py
"""Ticks & Spines tab: tick direction, tick label formatting, minor ticks, and spine frame borders."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from spectroview.i18n import (
    bind_combo_items,
    bind_text,
    bind_title,
    bind_tooltip,
    combo_data_or_text,
    on_language_changed,
    set_combo_by_data_or_text,
    tr,
)
from spectroview.view.components.customize_graph.common import (
    create_scrollable_container,
    highlight_group_style,
)

_TICK_FORMAT_PRESET_VALUES = {None, "%.0f", "%.1f", "%.2f", "%.1e"}


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
        self.grp_ticks = QGroupBox()
        bind_title(self.grp_ticks, "Tick Appearance:")
        ticks_layout = QVBoxLayout(self.grp_ticks)
        ticks_layout.setContentsMargins(8, 8, 8, 8)
        ticks_layout.setSpacing(6)

        tick_row = QHBoxLayout()
        lbl_td = QLabel()
        bind_text(lbl_td, "Tick direction:")
        bind_tooltip(lbl_td, "Direction of tick marks relative to the axes spine: In (pointing inward), Out (outward), or In & Out.")
        tick_row.addWidget(lbl_td)
        self.combo_tick_direction = QComboBox()
        bind_combo_items(
            self.combo_tick_direction,
            [("Default", "default"), ("In", "in"), ("Out", "out"), ("In & Out", "inout")],
        )
        self.combo_tick_direction.setMinimumWidth(120)
        self.combo_tick_direction.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        if self.combo_tick_direction.view():
            self.combo_tick_direction.view().setMinimumWidth(120)
        bind_tooltip(self.combo_tick_direction, "Direction of tick marks relative to the axes spine: In (pointing inward), Out (outward), or In & Out.")
        tick_row.addWidget(self.combo_tick_direction)

        tick_row.addSpacing(16)
        lbl_tf = QLabel()
        bind_text(lbl_tf, "Tick label format:")
        bind_tooltip(lbl_tf, "Numeric formatting applied to major tick values (integer, decimal places, or scientific notation).")
        tick_row.addWidget(lbl_tf)
        self.combo_tick_format = QComboBox()
        self._repopulate_tick_formats()
        on_language_changed(self, self._retranslate_tick_formats)
        self.combo_tick_format.setMinimumWidth(180)
        self.combo_tick_format.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        if self.combo_tick_format.view():
            self.combo_tick_format.view().setMinimumWidth(180)
        bind_tooltip(self.combo_tick_format, "Numeric formatting applied to major tick values (integer, decimal places, or scientific notation).")
        tick_row.addWidget(self.combo_tick_format)

        tick_row.addStretch()
        ticks_layout.addLayout(tick_row)
        layout.addWidget(self.grp_ticks)

        # ───── Minor Ticks ─────
        self.grp_minor = QGroupBox()
        bind_title(self.grp_minor, "Show Minor Ticks:")
        minor_layout = QHBoxLayout(self.grp_minor)
        minor_layout.setContentsMargins(8, 8, 8, 8)
        minor_layout.setSpacing(16)

        self.cb_minor_bottom = QCheckBox()
        bind_text(self.cb_minor_bottom, "Bottom")
        bind_tooltip(self.cb_minor_bottom, "Show minor subdivision tick marks along the bottom X axis.")
        self.cb_minor_left = QCheckBox()
        bind_text(self.cb_minor_left, "Left")
        bind_tooltip(self.cb_minor_left, "Show minor subdivision tick marks along the left Y axis.")
        self.cb_minor_top = QCheckBox()
        bind_text(self.cb_minor_top, "Top")
        bind_tooltip(self.cb_minor_top, "Show minor subdivision tick marks along the top X axis.")
        self.cb_minor_right = QCheckBox()
        bind_text(self.cb_minor_right, "Right")
        bind_tooltip(self.cb_minor_right, "Show minor subdivision tick marks along the right Y axis.")

        minor_layout.addWidget(self.cb_minor_bottom)
        minor_layout.addWidget(self.cb_minor_left)
        minor_layout.addWidget(self.cb_minor_top)
        minor_layout.addWidget(self.cb_minor_right)
        minor_layout.addStretch()
        layout.addWidget(self.grp_minor)

        # ───── Spines (Frame Borders) ─────
        self.grp_spines = QGroupBox()
        bind_title(self.grp_spines, "Show Spines (Frame Borders):")
        spines_layout = QHBoxLayout(self.grp_spines)
        spines_layout.setContentsMargins(8, 8, 8, 8)
        spines_layout.setSpacing(16)

        self.cb_spine_bottom = QCheckBox()
        bind_text(self.cb_spine_bottom, "Bottom")
        bind_tooltip(self.cb_spine_bottom, "Draw the bottom bounding spine line.")
        self.cb_spine_left = QCheckBox()
        bind_text(self.cb_spine_left, "Left")
        bind_tooltip(self.cb_spine_left, "Draw the left bounding spine line.")
        self.cb_spine_top = QCheckBox()
        bind_text(self.cb_spine_top, "Top")
        bind_tooltip(self.cb_spine_top, "Draw the top bounding spine line.")
        self.cb_spine_right = QCheckBox()
        bind_text(self.cb_spine_right, "Right")
        bind_tooltip(self.cb_spine_right, "Draw the right bounding spine line.")

        spines_layout.addWidget(self.cb_spine_bottom)
        spines_layout.addWidget(self.cb_spine_left)
        spines_layout.addWidget(self.cb_spine_top)
        spines_layout.addWidget(self.cb_spine_right)
        spines_layout.addStretch()
        layout.addWidget(self.grp_spines)

        layout.addStretch()

    def _repopulate_tick_formats(self, custom_value: Optional[str] = None) -> None:
        presets = [
            (tr("Auto (default)"), None),
            (tr("Integer  (e.g. 1234)"), "%.0f"),
            (tr("1 decimal  (e.g. 12.3)"), "%.1f"),
            (tr("2 decimals  (e.g. 12.34)"), "%.2f"),
            (tr("Scientific  (e.g. 1.2e+03)"), "%.1e"),
        ]
        combo = self.combo_tick_format
        combo.blockSignals(True)
        try:
            combo.clear()
            for label, val in presets:
                combo.addItem(label, val)
            if custom_value is not None and custom_value not in _TICK_FORMAT_PRESET_VALUES:
                combo.addItem(tr("Custom  ({value})", value=custom_value), custom_value)
        finally:
            combo.blockSignals(False)

    def _retranslate_tick_formats(self) -> None:
        current_val = self.combo_tick_format.currentData() if self.combo_tick_format.count() else None
        self._repopulate_tick_formats(current_val)
        self.combo_tick_format.blockSignals(True)
        try:
            idx = self.combo_tick_format.findData(current_val)
            self.combo_tick_format.setCurrentIndex(idx if idx >= 0 else 0)
        finally:
            self.combo_tick_format.blockSignals(False)

    def _load_tick_format(self, value: Optional[str]) -> None:
        combo = self.combo_tick_format
        idx = combo.findData(value)
        if combo.itemData(combo.count() - 1) not in _TICK_FORMAT_PRESET_VALUES:
            combo.removeItem(combo.count() - 1)
            idx = combo.findData(value)
        if idx < 0:
            combo.addItem(tr("Custom  ({value})", value=value), value)
            idx = combo.count() - 1
        combo.setCurrentIndex(idx)

    def load_settings(self) -> None:
        """Load tick and spine settings from current graph widget."""
        gw = self.graph_widget
        set_combo_by_data_or_text(
            self.combo_tick_direction,
            getattr(gw, "tick_direction", None) or "default",
        )
        self._load_tick_format(getattr(gw, "tick_label_format", None))

        effective_bottom = (
            gw.minor_ticks_bottom
            if getattr(gw, "minor_ticks_bottom", None) is not None
            else (gw._resolve_minor_ticks("bottom") if hasattr(gw, "_resolve_minor_ticks") else True)
        )
        effective_left = (
            gw.minor_ticks_left
            if getattr(gw, "minor_ticks_left", None) is not None
            else (gw._resolve_minor_ticks("left") if hasattr(gw, "_resolve_minor_ticks") else True)
        )
        effective_top = (
            gw.minor_ticks_top
            if getattr(gw, "minor_ticks_top", None) is not None
            else (gw._resolve_minor_ticks("top") if hasattr(gw, "_resolve_minor_ticks") else False)
        )
        effective_right = (
            gw.minor_ticks_right
            if getattr(gw, "minor_ticks_right", None) is not None
            else (gw._resolve_minor_ticks("right") if hasattr(gw, "_resolve_minor_ticks") else False)
        )

        self.cb_minor_bottom.setChecked(bool(effective_bottom))
        self.cb_minor_left.setChecked(bool(effective_left))
        self.cb_minor_top.setChecked(bool(effective_top))
        self.cb_minor_right.setChecked(bool(effective_right))

        spines = getattr(gw, "spines_visible", None) or {"top": True, "right": True, "bottom": True, "left": True}
        self.cb_spine_top.setChecked(spines.get("top", True))
        self.cb_spine_right.setChecked(spines.get("right", True))
        self.cb_spine_bottom.setChecked(spines.get("bottom", True))
        self.cb_spine_left.setChecked(spines.get("left", True))

    def apply_settings(self, replot: bool = False) -> dict:
        """Write controls back to graph widget, returning changed properties dict."""
        gw = self.graph_widget
        td = combo_data_or_text(self.combo_tick_direction)
        gw.tick_direction = None if td in ("default", "Default", None) else td
        gw.tick_label_format = self.combo_tick_format.currentData()

        # Keep an "auto" (None) minor-tick setting as None while the checkbox
        # still matches the auto-resolved state -- writing the resolved bool
        # back would freeze the default as an explicit override (and make an
        # untouched dialog look like an edit, forcing a needless redraw).
        for side, cb in (("bottom", self.cb_minor_bottom), ("left", self.cb_minor_left),
                         ("top", self.cb_minor_top), ("right", self.cb_minor_right)):
            attr = f"minor_ticks_{side}"
            checked = cb.isChecked()
            if getattr(gw, attr, None) is None and hasattr(gw, "_resolve_minor_ticks"):
                if bool(gw._resolve_minor_ticks(side)) == checked:
                    continue
            setattr(gw, attr, checked)

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

