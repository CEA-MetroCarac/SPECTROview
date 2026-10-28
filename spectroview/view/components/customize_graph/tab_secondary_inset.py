# spectroview/view/components/customize_graph/tab_secondary_inset.py
"""Secondary and Inset axes tab: twin axes (Y2/Y3/X2) and inset zoom axes."""

from __future__ import annotations

from typing import Optional

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from spectroview import MARKERS
from spectroview.view.components.customize_graph.common import (
    create_scrollable_container,
    highlight_group_style,
)
from spectroview.view.components.customize_graph.spin_widgets import PlaceholderDoubleSpinBox

_PLACEHOLDER_SPIN_WIDTH = 105
_UNSET_LIMIT = -999999.0


class TabSecondaryInset(QWidget):
    """Category 4: Secondary & Inset Axes."""

    KEYWORDS = [
        "secondary", "twin", "dual", "multi-axis", "y2", "y3", "x2",
        "inset", "zoom", "zoom axes", "sub-plot", "connector",
        "zoom indicator", "bounds", "position",
    ]

    _UNSET_LIMIT = _UNSET_LIMIT

    def __init__(self, graph_widget: QWidget, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.graph_widget = graph_widget
        self._setup_ui()
        self.load_settings()

    def switch_graph(self, graph_widget: QWidget) -> None:
        """Switch to a new graph widget and reload settings."""
        self.graph_widget = graph_widget
        self.load_settings()

    def _make_limit_spinbox(self, start_value: float = 0.0) -> PlaceholderDoubleSpinBox:
        spin = PlaceholderDoubleSpinBox(self._UNSET_LIMIT, start_value, "default")
        spin.setRange(self._UNSET_LIMIT, 999999)
        return spin

    def _setup_ui(self) -> None:
        self._scroll, _container, layout = create_scrollable_container(self)

        # ───── Secondary Axes (Y2, Y3, X2) ─────
        self.grp_secondary = QGroupBox("Secondary Twin Axes:")
        secondary_layout = QGridLayout(self.grp_secondary)
        secondary_layout.setContentsMargins(8, 8, 8, 8)
        secondary_layout.setHorizontalSpacing(8)
        secondary_layout.setVerticalSpacing(6)

        headers = [
            ("Axis", "Axis identifier (Y2/Y3: twin vertical axes, X2: twin horizontal axis)"),
            ("Label", "Custom axis label title"),
            ("Min", "Minimum axis limit (blank for auto-scale)"),
            ("Max", "Maximum axis limit (blank for auto-scale)"),
            ("Log", "Logarithmic scale toggle"),
            ("Color", "Axis ticks, label, and trace color"),
            ("Marker", "Marker shape symbol for this secondary trace"),
        ]
        for col, (text, tip) in enumerate(headers):
            lbl = QLabel(f"<b>{text}</b>")
            lbl.setToolTip(tip)
            secondary_layout.addWidget(lbl, 0, col)

        self._secondary_axis_rows: dict[str, dict[str, QWidget]] = {}
        for row, (axis_key, axis_display) in enumerate(
            [("y2", "Y2"), ("y3", "Y3"), ("x2", "X2")], start=1
        ):
            lbl_ax = QLabel(axis_display)
            secondary_layout.addWidget(lbl_ax, row, 0)

            edit_label = QLineEdit()
            edit_label.setMaximumWidth(90)
            edit_label.setToolTip(f"{axis_display} axis label text")
            secondary_layout.addWidget(edit_label, row, 1)

            spin_min = self._make_limit_spinbox()
            spin_min.setMaximumWidth(_PLACEHOLDER_SPIN_WIDTH)
            spin_min.setToolTip(f"{axis_display} minimum limit")
            secondary_layout.addWidget(spin_min, row, 2)

            spin_max = self._make_limit_spinbox()
            spin_max.setMaximumWidth(_PLACEHOLDER_SPIN_WIDTH)
            spin_max.setToolTip(f"{axis_display} maximum limit")
            secondary_layout.addWidget(spin_max, row, 3)

            cb_log = QCheckBox()
            cb_log.setToolTip(f"Enable logarithmic scale on {axis_display}")
            secondary_layout.addWidget(cb_log, row, 4)

            btn_color = QPushButton()
            btn_color.setFixedWidth(70)
            btn_color.setToolTip(f"Choose color for {axis_display} axis and trace")
            btn_color.clicked.connect(lambda *_a, k=axis_key: self._pick_secondary_axis_color(k))
            secondary_layout.addWidget(btn_color, row, 5)

            combo_marker = QComboBox()
            combo_marker.addItems(MARKERS)
            combo_marker.setMaximumWidth(60)
            combo_marker.setToolTip(f"Marker shape for {axis_display} series")
            secondary_layout.addWidget(combo_marker, row, 6)

            self._secondary_axis_rows[axis_key] = {
                "label": edit_label, "min": spin_min, "max": spin_max,
                "log": cb_log, "color": btn_color, "marker": combo_marker,
            }

        layout.addWidget(self.grp_secondary)

        # ───── Inset (Zoom) Axes ─────
        self.inset_group = QGroupBox("Inset (Zoom) Axes:")
        self.inset_group.setCheckable(True)
        self.inset_group.setToolTip("Enable an inset zoom sub-axis on top of the main figure.")
        inset_layout = QVBoxLayout(self.inset_group)
        inset_layout.setContentsMargins(8, 8, 8, 8)
        inset_layout.setSpacing(6)

        bounds_row = QHBoxLayout()
        lbl_pos = QLabel("Position (x₀, y₀):")
        lbl_pos.setToolTip("Normalized bottom-left corner of the inset axes (0.0 to 1.0).")
        bounds_row.addWidget(lbl_pos)
        self.spin_inset_x0 = QDoubleSpinBox()
        self.spin_inset_x0.setRange(0.0, 1.0)
        self.spin_inset_x0.setDecimals(2)
        self.spin_inset_x0.setSingleStep(0.05)
        self.spin_inset_x0.setToolTip("Normalized X coordinate of the bottom-left corner (0.0 = left, 1.0 = right).")
        bounds_row.addWidget(self.spin_inset_x0)

        self.spin_inset_y0 = QDoubleSpinBox()
        self.spin_inset_y0.setRange(0.0, 1.0)
        self.spin_inset_y0.setDecimals(2)
        self.spin_inset_y0.setSingleStep(0.05)
        self.spin_inset_y0.setToolTip("Normalized Y coordinate of the bottom-left corner (0.0 = bottom, 1.0 = top).")
        bounds_row.addWidget(self.spin_inset_y0)

        bounds_row.addSpacing(16)
        lbl_sz = QLabel("Size (w, h):")
        lbl_sz.setToolTip("Normalized width and height of the inset axes (0.05 to 1.0).")
        bounds_row.addWidget(lbl_sz)
        self.spin_inset_w = QDoubleSpinBox()
        self.spin_inset_w.setRange(0.05, 1.0)
        self.spin_inset_w.setDecimals(2)
        self.spin_inset_w.setSingleStep(0.05)
        self.spin_inset_w.setToolTip("Normalized width of the inset axes.")
        bounds_row.addWidget(self.spin_inset_w)

        self.spin_inset_h = QDoubleSpinBox()
        self.spin_inset_h.setRange(0.05, 1.0)
        self.spin_inset_h.setDecimals(2)
        self.spin_inset_h.setSingleStep(0.05)
        self.spin_inset_h.setToolTip("Normalized height of the inset axes.")
        bounds_row.addWidget(self.spin_inset_h)

        bounds_row.addStretch()
        inset_layout.addLayout(bounds_row)

        xlim_row = QHBoxLayout()
        lbl_xlim = QLabel("X limits:")
        lbl_xlim.setToolTip("Data X limits displayed inside the zoomed inset axes.")
        xlim_row.addWidget(lbl_xlim)
        self.spin_inset_xmin = self._make_limit_spinbox()
        self.spin_inset_xmin.setMaximumWidth(_PLACEHOLDER_SPIN_WIDTH)
        self.spin_inset_xmin.setToolTip("Inset data X minimum limit (blank for auto).")
        xlim_row.addWidget(self.spin_inset_xmin)
        self.spin_inset_xmax = self._make_limit_spinbox()
        self.spin_inset_xmax.setMaximumWidth(_PLACEHOLDER_SPIN_WIDTH)
        self.spin_inset_xmax.setToolTip("Inset data X maximum limit (blank for auto).")
        xlim_row.addWidget(self.spin_inset_xmax)
        xlim_row.addStretch()
        inset_layout.addLayout(xlim_row)

        ylim_row = QHBoxLayout()
        lbl_ylim = QLabel("Y limits:")
        lbl_ylim.setToolTip("Data Y limits displayed inside the zoomed inset axes.")
        ylim_row.addWidget(lbl_ylim)
        self.spin_inset_ymin = self._make_limit_spinbox()
        self.spin_inset_ymin.setMaximumWidth(_PLACEHOLDER_SPIN_WIDTH)
        self.spin_inset_ymin.setToolTip("Inset data Y minimum limit (blank for auto).")
        ylim_row.addWidget(self.spin_inset_ymin)
        self.spin_inset_ymax = self._make_limit_spinbox()
        self.spin_inset_ymax.setMaximumWidth(_PLACEHOLDER_SPIN_WIDTH)
        self.spin_inset_ymax.setToolTip("Inset data Y maximum limit (blank for auto).")
        ylim_row.addWidget(self.spin_inset_ymax)
        ylim_row.addStretch()
        inset_layout.addLayout(ylim_row)

        self.cb_inset_zoom_indicator = QCheckBox("Show zoom indicator (connector lines on main plot)")
        self.cb_inset_zoom_indicator.setChecked(True)
        self.cb_inset_zoom_indicator.setToolTip("Draw connector lines linking the inset box to the zoomed region on the main plot.")
        inset_layout.addWidget(self.cb_inset_zoom_indicator)

        layout.addWidget(self.inset_group)
        layout.addStretch()

    def _pick_secondary_axis_color(self, axis_key: str) -> None:
        row = self._secondary_axis_rows[axis_key]
        current = QColor(row["color"].text())
        color = QColorDialog.getColor(current, self, "Select Axis Color")
        if color.isValid():
            row["color"].setText(color.name())
            row["color"].setStyleSheet(f"background-color: {color.name()};")

    def load_settings(self) -> None:
        """Load secondary and inset axis settings from current graph widget."""
        gw = self.graph_widget
        u = self._UNSET_LIMIT

        # Load secondary axes
        for axis_key, row in self._secondary_axis_rows.items():
            active = bool(getattr(gw, axis_key, None))
            for widget in row.values():
                widget.setEnabled(active)

            row["label"].setText(getattr(gw, f"{axis_key}label", None) or "")
            row["min"].setValue(getattr(gw, f"{axis_key}min", None) if getattr(gw, f"{axis_key}min", None) is not None else u)
            row["max"].setValue(getattr(gw, f"{axis_key}max", None) if getattr(gw, f"{axis_key}max", None) is not None else u)
            row["log"].setChecked(getattr(gw, f"{axis_key}logscale", False))

            color = getattr(gw, f"{axis_key}color", "red")
            row["color"].setText(color)
            row["color"].setStyleSheet(f"background-color: {QColor(color).name()};")
            row["marker"].setCurrentText(getattr(gw, f"{axis_key}marker", "o"))

        # Load inset axes
        self.inset_group.setChecked(getattr(gw, "inset_enabled", False))
        bounds = getattr(gw, "inset_bounds", None) or [0.55, 0.55, 0.35, 0.35]
        self.spin_inset_x0.setValue(bounds[0])
        self.spin_inset_y0.setValue(bounds[1])
        self.spin_inset_w.setValue(bounds[2])
        self.spin_inset_h.setValue(bounds[3])

        self.spin_inset_xmin.setValue(gw.inset_xmin if getattr(gw, "inset_xmin", None) is not None else u)
        self.spin_inset_xmax.setValue(gw.inset_xmax if getattr(gw, "inset_xmax", None) is not None else u)
        self.spin_inset_ymin.setValue(gw.inset_ymin if getattr(gw, "inset_ymin", None) is not None else u)
        self.spin_inset_ymax.setValue(gw.inset_ymax if getattr(gw, "inset_ymax", None) is not None else u)
        self.cb_inset_zoom_indicator.setChecked(getattr(gw, "inset_show_zoom_indicator", True))

        # Placeholder bounds from live axes or main axes
        x_min, x_max, y_min, y_max = 0.0, 100.0, 0.0, 100.0
        inset_ax = getattr(gw, "inset_ax", None)
        if inset_ax is not None:
            try:
                x_min, x_max = inset_ax.get_xlim()
                y_min, y_max = inset_ax.get_ylim()
            except Exception:
                pass
        elif hasattr(gw, "ax") and gw.ax is not None:
            try:
                x_min, x_max = gw.ax.get_xlim()
                y_min, y_max = gw.ax.get_ylim()
            except Exception:
                pass
        self.spin_inset_xmin.set_placeholder_value(x_min)
        self.spin_inset_xmax.set_placeholder_value(x_max)
        self.spin_inset_ymin.set_placeholder_value(y_min)
        self.spin_inset_ymax.set_placeholder_value(y_max)

    def apply_settings(self, replot: bool = False) -> dict:
        """Write controls back to graph widget, returning changed properties dict."""
        gw = self.graph_widget
        u = self._UNSET_LIMIT
        props = {}

        # Apply secondary axes
        for axis_key, row in self._secondary_axis_rows.items():
            label = row["label"].text().strip() or None
            vmin = row["min"].value() if row["min"].value() != u else None
            vmax = row["max"].value() if row["max"].value() != u else None
            log = row["log"].isChecked()
            color = row["color"].text()
            marker = row["marker"].currentText()

            setattr(gw, f"{axis_key}label", label)
            setattr(gw, f"{axis_key}min", vmin)
            setattr(gw, f"{axis_key}max", vmax)
            setattr(gw, f"{axis_key}logscale", log)
            setattr(gw, f"{axis_key}color", color)
            setattr(gw, f"{axis_key}marker", marker)

            props.update({
                f"{axis_key}label": label, f"{axis_key}min": vmin, f"{axis_key}max": vmax,
                f"{axis_key}logscale": log, f"{axis_key}color": color, f"{axis_key}marker": marker,
            })

        # Apply inset axes
        gw.inset_enabled = self.inset_group.isChecked()
        gw.inset_bounds = [
            self.spin_inset_x0.value(), self.spin_inset_y0.value(),
            self.spin_inset_w.value(), self.spin_inset_h.value(),
        ]
        gw.inset_xmin = self.spin_inset_xmin.value() if self.spin_inset_xmin.value() != u else None
        gw.inset_xmax = self.spin_inset_xmax.value() if self.spin_inset_xmax.value() != u else None
        gw.inset_ymin = self.spin_inset_ymin.value() if self.spin_inset_ymin.value() != u else None
        gw.inset_ymax = self.spin_inset_ymax.value() if self.spin_inset_ymax.value() != u else None
        gw.inset_show_zoom_indicator = self.cb_inset_zoom_indicator.isChecked()

        props.update({
            "inset_enabled": gw.inset_enabled,
            "inset_bounds": gw.inset_bounds,
            "inset_xmin": gw.inset_xmin, "inset_xmax": gw.inset_xmax,
            "inset_ymin": gw.inset_ymin, "inset_ymax": gw.inset_ymax,
            "inset_show_zoom_indicator": gw.inset_show_zoom_indicator,
        })

        if replot and getattr(gw, "df", None) is not None:
            gw.plot(gw.df)

        return props

    def highlight_search(self, query: str) -> None:
        """Highlight matching groupboxes based on search query."""
        q = query.strip().lower()
        self.grp_secondary.setStyleSheet("")
        self.inset_group.setStyleSheet("")

        if not q:
            return

        secondary_terms = ["secondary", "twin", "y2", "y3", "x2", "dual", "multi-axis"]
        inset_terms = ["inset", "zoom", "sub-plot", "connector", "bounds", "indicator"]

        first_match: Optional[QWidget] = None
        if any(t in q or q in t for t in secondary_terms):
            self.grp_secondary.setStyleSheet(highlight_group_style())
            if first_match is None:
                first_match = self.grp_secondary

        if any(t in q or q in t for t in inset_terms):
            self.inset_group.setStyleSheet(highlight_group_style())
            if first_match is None:
                first_match = self.inset_group

        if first_match:
            self._scroll.ensureWidgetVisible(first_match, 0, 30)

