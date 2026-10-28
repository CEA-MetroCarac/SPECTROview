# spectroview/view/components/customize_graph/tab_legend_series.py
"""Legend and Series tab: legend box styling, marker unification, and per-series style overrides."""

from __future__ import annotations

import copy
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from spectroview import DEFAULT_COLORS, MARKERS
from spectroview.view.components.customize_graph.common import (
    create_scrollable_container,
    highlight_group_style,
)
from spectroview.view.components.customize_graph.customize_annotation_dialogs import ColorDelegate
from spectroview.view.components.customize_graph.spin_widgets import (
    PlaceholderDoubleSpinBox,
    PlaceholderSpinBox,
)

_MARKER_STYLES = ("scatter", "trendline", "point")
_UNSET = -999999.0


class TabLegendSeries(QWidget):
    """Category 5: Legend & Series."""

    KEYWORDS = [
        "legend", "legend box", "title", "columns", "frame", "border",
        "position", "location", "alpha", "transparency", "marker",
        "marker size", "scatter", "scatter size", "edge color", "series",
        "label", "color", "palette", "line width", "unify", "style",
    ]

    _UNSET = _UNSET

    def __init__(self, graph_widget: QWidget, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.graph_widget = graph_widget
        self.original_legend_properties = None
        self.original_legend_outside = False
        self.original_legend_bbox = None
        self._setup_ui()
        self.load_settings()

    def switch_graph(self, graph_widget: QWidget) -> None:
        """Switch to a new graph widget and reload settings."""
        self.graph_widget = graph_widget
        self.original_legend_properties = None
        self.load_settings()

    def _make_optional_spinbox(
        self,
        is_int: bool = False,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        decimals: int = 2,
        step: Optional[float] = None,
        start_value: float = 0.0,
    ) -> PlaceholderSpinBox | PlaceholderDoubleSpinBox:
        cls = PlaceholderSpinBox if is_int else PlaceholderDoubleSpinBox
        spin = cls(self._UNSET, start_value, "default")
        lo = self._UNSET if minimum is None else min(minimum, self._UNSET)
        hi = 999999 if maximum is None else maximum
        spin.setRange(lo, hi)
        if not is_int:
            spin.setDecimals(decimals)
        if step is not None:
            spin.setSingleStep(step)
        return spin

    def _setup_ui(self) -> None:
        self._scroll, _container, layout = create_scrollable_container(self)

        # ───── Legend Box Configuration ─────
        self.grp_legend_box = QGroupBox("Legend Box Configuration:")
        box_layout = QVBoxLayout(self.grp_legend_box)
        box_layout.setContentsMargins(8, 8, 8, 8)
        box_layout.setSpacing(8)

        top_row = QHBoxLayout()
        self.cb_legend_outside = QCheckBox("Put legend box outside plot")
        self.cb_legend_outside.stateChanged.connect(self._on_legend_outside_changed)
        top_row.addWidget(self.cb_legend_outside)

        top_row.addSpacing(12)
        self.cb_legend_frame = QCheckBox("Frame border")
        top_row.addWidget(self.cb_legend_frame)

        top_row.addSpacing(12)
        top_row.addWidget(QLabel("Columns:"))
        self.spin_legend_ncol = QSpinBox()
        self.spin_legend_ncol.setRange(1, 10)
        top_row.addWidget(self.spin_legend_ncol)

        top_row.addSpacing(12)
        top_row.addWidget(QLabel("Title:"))
        self.edit_legend_title = QLineEdit()
        self.edit_legend_title.setPlaceholderText("Optional title")
        top_row.addWidget(self.edit_legend_title)

        top_row.addStretch()
        box_layout.addLayout(top_row)

        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Font size:"))
        self.spin_legend_fontsize = QSpinBox()
        self.spin_legend_fontsize.setRange(4, 72)
        self.spin_legend_fontsize.setValue(9)
        style_row.addWidget(self.spin_legend_fontsize)

        style_row.addSpacing(12)
        style_row.addWidget(QLabel("Alpha:"))
        self.spin_legend_alpha = QDoubleSpinBox()
        self.spin_legend_alpha.setRange(0.0, 1.0)
        self.spin_legend_alpha.setDecimals(2)
        self.spin_legend_alpha.setSingleStep(0.1)
        style_row.addWidget(self.spin_legend_alpha)

        style_row.addSpacing(12)
        style_row.addWidget(QLabel("Position:"))
        self.combo_legend_loc = QComboBox()
        self.combo_legend_loc.addItems([
            "best", "upper right", "upper left", "lower left", "lower right",
            "center left", "center right", "lower center", "upper center", "center",
        ])
        style_row.addWidget(self.combo_legend_loc)

        style_row.addStretch()
        box_layout.addLayout(style_row)
        layout.addWidget(self.grp_legend_box)

        # ───── Global Marker Styling ─────
        self.grp_markers = QGroupBox("Global Marker Styling:")
        marker_box_layout = QVBoxLayout(self.grp_markers)
        marker_box_layout.setContentsMargins(8, 8, 8, 8)
        marker_box_layout.setSpacing(8)

        unify_row = QHBoxLayout()
        self.cb_unify_marker_style = QCheckBox("Unify marker size / edge color across all series")
        self.cb_unify_marker_style.setToolTip(
            "Checked: every series uses the same marker size and edge color below.\n"
            "Unchecked: each series can override them individually in the table below."
        )
        self.cb_unify_marker_style.toggled.connect(self._on_unify_marker_style_toggled)
        unify_row.addWidget(self.cb_unify_marker_style)

        self.unified_marker_widget = QWidget()
        unified_layout = QHBoxLayout(self.unified_marker_widget)
        unified_layout.setContentsMargins(0, 0, 0, 0)
        unified_layout.setSpacing(8)
        unified_layout.addWidget(QLabel("Marker size:"))
        self.spin_scatter_size = QSpinBox()
        self.spin_scatter_size.setRange(5, 500)
        self.spin_scatter_size.setSingleStep(10)
        self.spin_scatter_size.setValue(70)
        unified_layout.addWidget(self.spin_scatter_size)

        unified_layout.addSpacing(10)
        unified_layout.addWidget(QLabel("Edge color:"))
        self.btn_scatter_edgecolor = QPushButton()
        self.btn_scatter_edgecolor.setFixedWidth(80)
        self._set_color_button(self.btn_scatter_edgecolor, "black")
        self.btn_scatter_edgecolor.clicked.connect(self._pick_scatter_edgecolor)
        unified_layout.addWidget(self.btn_scatter_edgecolor)

        unify_row.addWidget(self.unified_marker_widget)
        unify_row.addStretch()
        marker_box_layout.addLayout(unify_row)
        layout.addWidget(self.grp_markers)

        # ───── Series Customization Table ─────
        self.grp_series = QGroupBox("Series Customization:")
        series_box_layout = QVBoxLayout(self.grp_series)
        series_box_layout.setContentsMargins(8, 8, 8, 8)
        series_box_layout.setSpacing(8)

        self.legend_layout = QHBoxLayout()
        series_box_layout.addLayout(self.legend_layout)
        layout.addWidget(self.grp_series)

        layout.addStretch()

    def _set_color_button(self, button: QPushButton, color_str: str) -> None:
        button.setText(color_str)
        try:
            c = QColor(color_str)
            if c.isValid():
                text_col = "#000000" if c.lightness() > 140 else "#ffffff"
                button.setStyleSheet(f"background-color: {c.name()}; color: {text_col}; font-weight: bold;")
        except Exception:
            pass

    def _pick_scatter_edgecolor(self) -> None:
        current = QColor(self.btn_scatter_edgecolor.text())
        color = QColorDialog.getColor(current, self, "Select Marker Edge Color")
        if color.isValid():
            self._set_color_button(self.btn_scatter_edgecolor, color.name())

    def _pick_series_edge_color(self, idx: int, btn: QPushButton) -> None:
        current = QColor(btn.text())
        color = QColorDialog.getColor(current, self, "Select Series Edge Color")
        if color.isValid():
            self._set_color_button(btn, color.name())
            btn.setToolTip("")
            self._update_legend_property(idx, "edge_color", color.name())

    def _on_legend_outside_changed(self, state: int) -> None:
        is_outside = bool(state)
        self.combo_legend_loc.setEnabled(not is_outside)

    def _on_unify_marker_style_toggled(self, checked: bool) -> None:
        is_marker_style = getattr(self.graph_widget, "plot_style", "") in _MARKER_STYLES
        self.unified_marker_widget.setVisible(checked and is_marker_style)
        self._rebuild_legend_widgets()

    def _update_combobox_color(self, combo: QComboBox) -> None:
        color = combo.currentText()
        try:
            c = QColor(color)
            if c.isValid():
                text_col = "#000000" if c.lightness() > 140 else "#ffffff"
                combo.setStyleSheet(f"background-color: {c.name()}; color: {text_col};")
        except Exception:
            pass

    def _update_legend_property(self, index: int, prop_name: str, value: str) -> None:
        legend_props = getattr(self.graph_widget, "legend_properties", [])
        if 0 <= index < len(legend_props):
            legend_props[index][prop_name] = value

    def _update_legend_property_numeric(self, index: int, prop_name: str, value: float) -> None:
        legend_props = getattr(self.graph_widget, "legend_properties", [])
        if 0 <= index < len(legend_props):
            if value == self._UNSET:
                legend_props[index].pop(prop_name, None)
            else:
                legend_props[index][prop_name] = value

    def _rebuild_legend_widgets(self) -> None:
        legend_properties = self.graph_widget.get_legend_properties()
        while self.legend_layout.count():
            item = self.legend_layout.takeAt(0)
            if item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()
            if item.widget():
                item.widget().deleteLater()

        if not legend_properties:
            self.legend_layout.addWidget(QLabel("No legend series available for this plot."))
        else:
            self._build_legend_widgets(legend_properties)

    def _build_legend_widgets(self, legend_properties: list[dict]) -> None:
        label_layout = QVBoxLayout()
        marker_layout = QVBoxLayout()
        color_layout = QVBoxLayout()
        alpha_layout = QVBoxLayout()

        show_lw = (
            self.graph_widget.plot_style in ("line", "trendline")
            or (self.graph_widget.plot_style == "point" and getattr(self.graph_widget, "join_for_point_plot", False))
        )
        linewidth_layout = QVBoxLayout() if show_lw else None

        show_marker_col = (
            self.graph_widget.plot_style in _MARKER_STYLES
            and not self.cb_unify_marker_style.isChecked()
        )
        marker_size_layout = QVBoxLayout() if show_marker_col else None
        edge_color_layout = QVBoxLayout() if show_marker_col else None

        # Headers
        headers = [("Label", label_layout)]
        if self.graph_widget.plot_style == "point":
            headers.append(("Marker", marker_layout))
        headers.append(("Color", color_layout))
        if show_lw:
            headers.append(("Line width", linewidth_layout))
        headers.append(("Alpha", alpha_layout))
        if show_marker_col:
            headers.append(("Size", marker_size_layout))
            headers.append(("Edge Color", edge_color_layout))

        for header_text, h_layout in headers:
            lbl = QLabel(f"<b>{header_text}</b>")
            lbl.setAlignment(Qt.AlignCenter)
            h_layout.addWidget(lbl)

        for idx, prop in enumerate(legend_properties):
            # Label
            label = QLineEdit(prop.get("label", ""))
            label.setFixedWidth(120)
            label.textChanged.connect(lambda text, i=idx: self._update_legend_property(i, "label", text))
            label_layout.addWidget(label)

            # Marker (point plot)
            if self.graph_widget.plot_style == "point":
                marker = QComboBox()
                marker.setFixedWidth(55)
                marker.addItems(MARKERS)
                marker.setCurrentText(prop.get("marker", "o"))
                marker.currentTextChanged.connect(lambda text, i=idx: self._update_legend_property(i, "marker", text))
                marker_layout.addWidget(marker)

            # Color
            color = QComboBox()
            color.setFixedWidth(90)
            color.setItemDelegate(ColorDelegate(color))
            for color_code in list(dict.fromkeys(DEFAULT_COLORS)):
                color.addItem(color_code)
                item = color.model().item(color.count() - 1)
                item.setBackground(QColor(color_code))
            color.setCurrentText(prop.get("color", "#1f77b4"))
            color.currentIndexChanged.connect(lambda *_a, cb=color: self._update_combobox_color(cb))
            color.currentTextChanged.connect(lambda text, i=idx: self._update_legend_property(i, "color", text))
            color_layout.addWidget(color)
            self._update_combobox_color(color)

            # Linewidth
            if show_lw:
                lw_spin = self._make_optional_spinbox(minimum=0.0, maximum=20.0, decimals=2, step=0.5, start_value=1.5)
                lw_spin.setFixedWidth(60)
                lw_spin.setValue(prop.get("linewidth", self._UNSET))
                lw_spin.set_placeholder_value(1.5)
                lw_spin.valueChanged.connect(lambda v, i=idx: self._update_legend_property_numeric(i, "linewidth", v))
                linewidth_layout.addWidget(lw_spin)

            # Alpha
            alpha_spin = self._make_optional_spinbox(minimum=0.0, maximum=1.0, decimals=2, step=0.1, start_value=1.0)
            alpha_spin.setFixedWidth(60)
            alpha_spin.setValue(prop.get("alpha", self._UNSET))
            alpha_spin.set_placeholder_value(1.0)
            alpha_spin.valueChanged.connect(lambda v, i=idx: self._update_legend_property_numeric(i, "alpha", v))
            alpha_layout.addWidget(alpha_spin)

            # Marker size & Edge color
            if show_marker_col:
                def_size = getattr(self.graph_widget, "scatter_size", 70)
                ms_spin = self._make_optional_spinbox(is_int=True, minimum=1, maximum=500, step=10, start_value=def_size)
                ms_spin.setFixedWidth(60)
                ms_spin.setValue(int(prop.get("marker_size", self._UNSET)))
                ms_spin.set_placeholder_value(def_size)
                ms_spin.valueChanged.connect(lambda v, i=idx: self._update_legend_property_numeric(i, "marker_size", v))
                marker_size_layout.addWidget(ms_spin)

                edge_btn = QPushButton()
                edge_btn.setFixedWidth(70)
                edge_val = prop.get("edge_color")
                self._set_color_button(edge_btn, edge_val or getattr(self.graph_widget, "scatter_edgecolor", "black"))
                edge_btn.clicked.connect(lambda *_a, i=idx, btn=edge_btn: self._pick_series_edge_color(i, btn))
                edge_color_layout.addWidget(edge_btn)

        for _name, l in headers:
            l.addStretch()
            self.legend_layout.addLayout(l)

    def load_settings(self) -> None:
        """Load legend and series settings from graph widget into controls."""
        gw = self.graph_widget
        legend_properties = gw.get_legend_properties()
        self.original_legend_properties = copy.deepcopy(legend_properties)
        self.original_legend_outside = getattr(gw, "legend_outside", False)
        self.original_legend_bbox = getattr(gw, "legend_bbox", None)

        self.cb_legend_outside.blockSignals(True)
        self.cb_legend_outside.setChecked(self.original_legend_outside)
        self.cb_legend_outside.blockSignals(False)

        self.cb_unify_marker_style.blockSignals(True)
        self.cb_unify_marker_style.setChecked(getattr(gw, "unify_marker_style", True))
        self.cb_unify_marker_style.blockSignals(False)

        self.spin_legend_ncol.setValue(getattr(gw, "legend_ncol", 1))
        self.cb_legend_frame.setChecked(getattr(gw, "legend_frame", True))
        self.edit_legend_title.setText(getattr(gw, "legend_title", None) or "")
        self.spin_legend_fontsize.setValue(getattr(gw, "legend_fontsize", None) or 9)
        self.spin_legend_alpha.setValue(getattr(gw, "legend_alpha", 0.7))
        self.combo_legend_loc.setCurrentText(getattr(gw, "legend_loc", "best"))
        self.combo_legend_loc.setEnabled(not self.original_legend_outside)

        is_marker_style = getattr(gw, "plot_style", "") in _MARKER_STYLES
        self.grp_markers.setVisible(is_marker_style)
        self.spin_scatter_size.setValue(getattr(gw, "scatter_size", 70))
        edge_c = getattr(gw, "scatter_edgecolor", "black")
        if not edge_c or not isinstance(edge_c, str) or edge_c.strip() in ("", "None", "none", "null"):
            edge_c = "black"
        self._set_color_button(self.btn_scatter_edgecolor, edge_c)
        self.unified_marker_widget.setVisible(self.cb_unify_marker_style.isChecked() and is_marker_style)

        self._rebuild_legend_widgets()

    # Alias for existing test calls
    load_legend_properties = load_settings

    def apply_settings(self, replot: bool = False) -> dict:
        """Apply legend and series properties to graph widget."""
        gw = self.graph_widget
        gw.legend_outside = self.cb_legend_outside.isChecked()
        gw.legend_frame = self.cb_legend_frame.isChecked()
        gw.legend_ncol = self.spin_legend_ncol.value()
        gw.legend_title = self.edit_legend_title.text().strip() or None
        gw.legend_fontsize = self.spin_legend_fontsize.value()
        gw.legend_alpha = self.spin_legend_alpha.value()
        gw.legend_loc = self.combo_legend_loc.currentText()

        gw.unify_marker_style = self.cb_unify_marker_style.isChecked()
        gw.scatter_size = self.spin_scatter_size.value()
        gw.scatter_edgecolor = self.btn_scatter_edgecolor.text()

        props = {
            "legend_outside": gw.legend_outside,
            "legend_frame": gw.legend_frame,
            "legend_ncol": gw.legend_ncol,
            "legend_title": gw.legend_title,
            "legend_fontsize": gw.legend_fontsize,
            "legend_alpha": gw.legend_alpha,
            "legend_loc": gw.legend_loc,
            "legend_properties": gw.legend_properties,
            "unify_marker_style": gw.unify_marker_style,
            "scatter_size": gw.scatter_size,
            "scatter_edgecolor": gw.scatter_edgecolor,
        }

        if replot and getattr(gw, "df", None) is not None:
            gw.plot(gw.df)

        return props

    # Alias for existing test calls
    apply_changes = apply_settings

    def highlight_search(self, query: str) -> None:
        """Highlight matching groupboxes based on search query."""
        q = query.strip().lower()
        self.grp_legend_box.setStyleSheet("")
        self.grp_markers.setStyleSheet("")
        self.grp_series.setStyleSheet("")

        if not q:
            return

        box_terms = ["legend", "box", "outside", "frame", "columns", "title", "position", "alpha", "location"]
        marker_terms = ["marker", "scatter", "size", "edge color", "unify"]
        series_terms = ["series", "label", "color", "palette", "line width"]

        first_match: Optional[QWidget] = None
        if any(t in q or q in t for t in box_terms):
            self.grp_legend_box.setStyleSheet(highlight_group_style())
            if first_match is None:
                first_match = self.grp_legend_box

        if any(t in q or q in t for t in marker_terms):
            self.grp_markers.setStyleSheet(highlight_group_style())
            if first_match is None:
                first_match = self.grp_markers

        if any(t in q or q in t for t in series_terms):
            self.grp_series.setStyleSheet(highlight_group_style())
            if first_match is None:
                first_match = self.grp_series

        if first_match:
            self._scroll.ensureWidgetVisible(first_match, 0, 30)
