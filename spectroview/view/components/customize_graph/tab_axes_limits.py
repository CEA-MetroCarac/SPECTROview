# spectroview/view/components/customize_graph/tab_axes_limits.py
"""Axes & Limits tab: primary X/Y/Z scales, data types, inversion, limits with sliders, and broken axis."""

from __future__ import annotations

from typing import Optional

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from spectroview.view.components.customize_graph.common import (
    create_scrollable_container,
    highlight_group_style,
)
from spectroview.view.components.customize_graph.spin_widgets import PlaceholderDoubleSpinBox

try:
    from superqt import QLabeledDoubleRangeSlider
except ImportError:  # pragma: no cover
    QLabeledDoubleRangeSlider = None

_PLACEHOLDER_SPIN_WIDTH = 105
_UNSET_LIMIT = -999999.0


class TabAxesLimits(QWidget):
    """Category 2: Axes & Limits."""

    KEYWORDS = [
        "axis", "axes", "scale", "scales", "linear", "log", "logarithmic",
        "symlog", "centered", "data type", "numerical", "category",
        "invert", "inverted", "limits", "range", "slider", "min", "max",
        "xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "colormap",
        "colormap center", "broken", "break", "broken axis", "x-axis break",
        "y-axis break",
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

    def _add_limit_row_with_slider(self, parent_layout: QVBoxLayout, axis_label: str, prefix: str) -> None:
        row = QHBoxLayout()
        row.addWidget(QLabel(f"{axis_label} axis limits:"))

        spin_min = self._make_limit_spinbox()
        spin_min.setMaximumWidth(_PLACEHOLDER_SPIN_WIDTH)
        setattr(self, f"spin_{prefix}min", spin_min)
        row.addWidget(spin_min)

        slider = None
        if QLabeledDoubleRangeSlider is not None:
            slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
            slider.setEdgeLabelMode(QLabeledDoubleRangeSlider.EdgeLabelMode.NoLabel)
            slider.setHandleLabelPosition(QLabeledDoubleRangeSlider.LabelPosition.NoLabel)
            slider.setRange(0, 100)
            slider.setValue((0, 100))
            setattr(self, f"{prefix}_range_slider", slider)
            row.addWidget(slider, stretch=1)

        spin_max = self._make_limit_spinbox()
        spin_max.setMaximumWidth(_PLACEHOLDER_SPIN_WIDTH)
        setattr(self, f"spin_{prefix}max", spin_max)
        row.addWidget(spin_max)

        if slider is not None:
            slider.valueChanged.connect(lambda values, p=prefix: self._on_range_slider_changed(p, values))
            spin_min.valueChanged.connect(lambda *_a, p=prefix: self._update_range_slider_from_spins(p))
            spin_max.valueChanged.connect(lambda *_a, p=prefix: self._update_range_slider_from_spins(p))

        parent_layout.addLayout(row)

    def _on_range_slider_changed(self, prefix: str, values: tuple[float, float]) -> None:
        spin_min = getattr(self, f"spin_{prefix}min")
        spin_max = getattr(self, f"spin_{prefix}max")
        spin_min.blockSignals(True)
        spin_max.blockSignals(True)
        spin_min.setValue(values[0])
        spin_max.setValue(values[1])
        spin_min._update_placeholder_style()
        spin_max._update_placeholder_style()
        spin_min.blockSignals(False)
        spin_max.blockSignals(False)

    def _update_range_slider_from_spins(self, prefix: str) -> None:
        slider = getattr(self, f"{prefix}_range_slider", None)
        if slider is None:
            return
        spin_min = getattr(self, f"spin_{prefix}min")
        spin_max = getattr(self, f"spin_{prefix}max")
        u = self._UNSET_LIMIT
        lo, hi = slider.minimum(), slider.maximum()
        vmin = lo if spin_min.value() == u else spin_min.value()
        vmax = hi if spin_max.value() == u else spin_max.value()
        vmin = max(lo, min(vmin, hi))
        vmax = max(lo, min(vmax, hi))
        if vmin <= vmax:
            slider.blockSignals(True)
            slider.setValue((vmin, vmax))
            slider.blockSignals(False)

    def _update_range_slider_bounds(self) -> None:
        gw = self.graph_widget
        df = getattr(gw, "df", None)
        columns = {"x": getattr(gw, "x", None), "y": (gw.y[0] if getattr(gw, "y", None) else None), "z": getattr(gw, "z", None)}
        for prefix, col in columns.items():
            slider = getattr(self, f"{prefix}_range_slider", None)
            if slider is None:
                continue
            lo, hi = 0.0, 100.0
            if df is not None and col is not None and col in getattr(df, "columns", []):
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if not series.empty:
                    data_min, data_max = float(series.min()), float(series.max())
                    pad = (data_max - data_min) * 0.1 or 1.0
                    lo, hi = data_min - pad, data_max + pad
            slider.blockSignals(True)
            slider.setRange(lo, hi)
            slider.blockSignals(False)
            getattr(self, f"spin_{prefix}min").set_start_value(lo)
            getattr(self, f"spin_{prefix}max").set_start_value(hi)
            self._update_range_slider_from_spins(prefix)

    def _setup_ui(self) -> None:
        self._scroll, _container, layout = create_scrollable_container(self)

        # ───── Axis Limits (Placed on top per user request) ─────
        self.grp_limits = QGroupBox("Axis Limits:")
        limits_layout = QVBoxLayout(self.grp_limits)
        limits_layout.setContentsMargins(8, 8, 8, 8)
        limits_layout.setSpacing(6)

        self.xy_limits_widget = QWidget()
        xy_limits_layout = QVBoxLayout(self.xy_limits_widget)
        xy_limits_layout.setContentsMargins(0, 0, 0, 0)
        xy_limits_layout.setSpacing(6)
        self._add_limit_row_with_slider(xy_limits_layout, "X", "x")
        self._add_limit_row_with_slider(xy_limits_layout, "Y", "y")
        limits_layout.addWidget(self.xy_limits_widget)

        self._add_limit_row_with_slider(limits_layout, "Z", "z")

        limits_btn_row = QHBoxLayout()
        self.btn_set_limits = QPushButton("Get current limits from plot")
        self.btn_set_limits.setToolTip("Query current rendered axis limits from the active plot canvas.")
        self.btn_set_limits.clicked.connect(self._on_get_current_limits)

        self.btn_clear_limits = QPushButton("Reinitialize limits")
        self.btn_clear_limits.setToolTip("Reset all axis limits back to automatic auto-scaling.")
        self.btn_clear_limits.clicked.connect(self._on_clear_limits)

        limits_btn_row.addWidget(self.btn_set_limits)
        limits_btn_row.addWidget(self.btn_clear_limits)
        limits_btn_row.addStretch()
        limits_layout.addLayout(limits_btn_row)

        layout.addWidget(self.grp_limits)

        # ───── Primary Scales & Data Types ─────
        self.grp_scales = QGroupBox("Axis Scales & Data Types:")
        scales_layout = QVBoxLayout(self.grp_scales)
        scales_layout.setContentsMargins(8, 8, 8, 8)
        scales_layout.setSpacing(6)

        x_row = QHBoxLayout()
        lbl_x = QLabel("X axis:")
        lbl_x.setFixedWidth(50)
        x_row.addWidget(lbl_x)
        x_row.addWidget(QLabel("Scale:"))
        self.combo_x_scale = QComboBox()
        self.combo_x_scale.addItems(["Linear", "Logarithmic", "Symlog"])
        self.combo_x_scale.setMinimumWidth(125)
        self.combo_x_scale.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        if self.combo_x_scale.view():
            self.combo_x_scale.view().setMinimumWidth(125)
        self.combo_x_scale.setToolTip("Scaling mode for the X axis: Linear, Logarithmic, or Symlog (symmetrical log).")
        x_row.addWidget(self.combo_x_scale)

        x_row.addSpacing(12)
        x_row.addWidget(QLabel("Data type:"))
        self.combo_x_type = QComboBox()
        self.combo_x_type.addItems(["Auto", "Category", "Numerical"])
        self.combo_x_type.setMinimumWidth(110)
        self.combo_x_type.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        if self.combo_x_type.view():
            self.combo_x_type.view().setMinimumWidth(110)
        self.combo_x_type.setToolTip("Interpretation for X values: Auto (detected), Category (discrete levels), or Numerical (continuous).")
        x_row.addWidget(self.combo_x_type)

        x_row.addSpacing(12)
        self.cb_invert_x = QCheckBox("Invert")
        self.cb_invert_x.setToolTip("Invert X axis so values decrease from left to right.")
        x_row.addWidget(self.cb_invert_x)
        x_row.addStretch()
        scales_layout.addLayout(x_row)

        y_row = QHBoxLayout()
        lbl_y = QLabel("Y axis:")
        lbl_y.setFixedWidth(50)
        y_row.addWidget(lbl_y)
        y_row.addWidget(QLabel("Scale:"))
        self.combo_y_scale = QComboBox()
        self.combo_y_scale.addItems(["Linear", "Logarithmic", "Symlog"])
        self.combo_y_scale.setMinimumWidth(125)
        self.combo_y_scale.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        if self.combo_y_scale.view():
            self.combo_y_scale.view().setMinimumWidth(125)
        self.combo_y_scale.setToolTip("Scaling mode for the Y axis: Linear, Logarithmic, or Symlog.")
        y_row.addWidget(self.combo_y_scale)

        y_row.addSpacing(12)
        y_row.addWidget(QLabel("Data type:"))
        self.combo_y_type = QComboBox()
        self.combo_y_type.addItems(["Auto", "Category", "Numerical"])
        self.combo_y_type.setMinimumWidth(110)
        self.combo_y_type.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        if self.combo_y_type.view():
            self.combo_y_type.view().setMinimumWidth(110)
        self.combo_y_type.setToolTip("Interpretation for Y values: Auto, Category, or Numerical.")
        y_row.addWidget(self.combo_y_type)

        y_row.addSpacing(12)
        self.cb_invert_y = QCheckBox("Invert")
        self.cb_invert_y.setToolTip("Invert Y axis so values decrease from bottom to top.")
        y_row.addWidget(self.cb_invert_y)
        y_row.addStretch()
        scales_layout.addLayout(y_row)

        # Z axis color scale (wafer/2Dmap only)
        self.z_scale_widget = QWidget()
        z_row = QHBoxLayout(self.z_scale_widget)
        z_row.setContentsMargins(0, 0, 0, 0)
        lbl_z = QLabel("Z axis:")
        lbl_z.setFixedWidth(50)
        z_row.addWidget(lbl_z)
        z_row.addWidget(QLabel("Scale:"))
        self.combo_z_scale = QComboBox()
        self.combo_z_scale.addItem("Linear", "linear")
        self.combo_z_scale.addItem("Logarithmic", "log")
        self.combo_z_scale.addItem("Centered", "centered")
        self.combo_z_scale.setMinimumWidth(125)
        self.combo_z_scale.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        if self.combo_z_scale.view():
            self.combo_z_scale.view().setMinimumWidth(125)
        self.combo_z_scale.setToolTip("Colormap normalization for wafer/2Dmap plots: Linear, Logarithmic, or Centered.")
        self.combo_z_scale.currentIndexChanged.connect(self._on_z_scale_changed)
        z_row.addWidget(self.combo_z_scale)

        z_row.addSpacing(12)
        self.lbl_colormap_center = QLabel("Center value:")
        z_row.addWidget(self.lbl_colormap_center)
        self.spin_colormap_center = QDoubleSpinBox()
        self.spin_colormap_center.setRange(-1e6, 1e6)
        self.spin_colormap_center.setDecimals(2)
        self.spin_colormap_center.setSingleStep(1.0)
        self.spin_colormap_center.setMaximumWidth(90)
        self.spin_colormap_center.setToolTip("Center midpoint value for Centered colormaps (e.g. 0.0 for delta/strain).")
        z_row.addWidget(self.spin_colormap_center)
        z_row.addStretch()
        scales_layout.addWidget(self.z_scale_widget)

        layout.addWidget(self.grp_scales)

        # ───── Broken Axis ─────
        self.grp_break = QGroupBox("Broken Axis:")
        break_layout = QVBoxLayout(self.grp_break)
        break_layout.setContentsMargins(8, 8, 8, 8)
        break_layout.setSpacing(6)

        x_break_row = QHBoxLayout()
        self.x_break_enabled = QCheckBox("X-axis break")
        self.x_break_enabled.setToolTip("Enable break on X axis. X and Y breaks cannot both be active simultaneously.")
        self.x_break_start = QDoubleSpinBox()
        self.x_break_start.setRange(-999999, 999999)
        self.x_break_start.setDecimals(2)
        self.x_break_start.setToolTip("Start coordinate of the omitted X range.")
        self.x_break_end = QDoubleSpinBox()
        self.x_break_end.setRange(-999999, 999999)
        self.x_break_end.setDecimals(2)
        self.x_break_end.setToolTip("End coordinate of the omitted X range.")
        x_break_row.addWidget(self.x_break_enabled)
        x_break_row.addWidget(QLabel("from:"))
        x_break_row.addWidget(self.x_break_start)
        x_break_row.addWidget(QLabel("to:"))
        x_break_row.addWidget(self.x_break_end)
        x_break_row.addStretch()
        break_layout.addLayout(x_break_row)

        y_break_row = QHBoxLayout()
        self.y_break_enabled = QCheckBox("Y-axis break")
        self.y_break_enabled.setToolTip("Enable break on Y axis. X and Y breaks cannot both be active simultaneously.")
        self.y_break_start = QDoubleSpinBox()
        self.y_break_start.setRange(-999999, 999999)
        self.y_break_start.setDecimals(2)
        self.y_break_start.setToolTip("Start coordinate of the omitted Y range.")
        self.y_break_end = QDoubleSpinBox()
        self.y_break_end.setRange(-999999, 999999)
        self.y_break_end.setDecimals(2)
        self.y_break_end.setToolTip("End coordinate of the omitted Y range.")
        y_break_row.addWidget(self.y_break_enabled)
        y_break_row.addWidget(QLabel("from:"))
        y_break_row.addWidget(self.y_break_start)
        y_break_row.addWidget(QLabel("to:"))
        y_break_row.addWidget(self.y_break_end)
        y_break_row.addStretch()
        break_layout.addLayout(y_break_row)

        self.x_break_enabled.toggled.connect(self._on_x_break_toggled)
        self.y_break_enabled.toggled.connect(self._on_y_break_toggled)

        layout.addWidget(self.grp_break)
        layout.addStretch()

    def _on_x_break_toggled(self, checked: bool) -> None:
        if checked and self.y_break_enabled.isChecked():
            self.y_break_enabled.setChecked(False)

    def _on_y_break_toggled(self, checked: bool) -> None:
        if checked and self.x_break_enabled.isChecked():
            self.x_break_enabled.setChecked(False)

    def _on_z_scale_changed(self) -> None:
        is_centered = self.combo_z_scale.currentData() == "centered"
        self.lbl_colormap_center.setEnabled(is_centered)
        self.spin_colormap_center.setEnabled(is_centered)

    def _get_current_ax_limits(self) -> tuple[float, float, float, float]:
        x_min, x_max, y_min, y_max = 0.0, 100.0, 0.0, 100.0
        if hasattr(self.graph_widget, "ax") and self.graph_widget.ax is not None:
            try:
                x_min, x_max = self.graph_widget.ax.get_xlim()
                y_min, y_max = self.graph_widget.ax.get_ylim()
            except Exception:
                pass
        return x_min, x_max, y_min, y_max

    def _on_get_current_limits(self) -> None:
        try:
            if hasattr(self.graph_widget, "ax") and self.graph_widget.ax is not None:
                xmin, xmax = self.graph_widget.ax.get_xlim()
                ymin, ymax = self.graph_widget.ax.get_ylim()
                self.spin_xmin.setValue(round(xmin, 3))
                self.spin_xmax.setValue(round(xmax, 3))
                self.spin_ymin.setValue(round(ymin, 3))
                self.spin_ymax.setValue(round(ymax, 3))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error getting limits: {str(e)}")

    def _on_clear_limits(self) -> None:
        u = self._UNSET_LIMIT
        self.spin_xmin.setValue(u)
        self.spin_xmax.setValue(u)
        self.spin_ymin.setValue(u)
        self.spin_ymax.setValue(u)
        self.spin_zmin.setValue(u)
        self.spin_zmax.setValue(u)

    def load_settings(self) -> None:
        """Load axis settings and limits from current graph widget."""
        gw = self.graph_widget
        is_map = getattr(gw, "plot_style", "") in ("wafer", "2Dmap")
        self.xy_limits_widget.setVisible(not is_map)
        self.z_scale_widget.setVisible(is_map)

        if is_map:
            norm_kind = getattr(gw, "colormap_norm", "linear")
            idx = self.combo_z_scale.findData(norm_kind)
            self.combo_z_scale.setCurrentIndex(idx if idx >= 0 else 0)
            self.spin_colormap_center.setValue(getattr(gw, "colormap_center", 0.0))
            self._on_z_scale_changed()

        u = self._UNSET_LIMIT
        self.spin_xmin.setValue(gw.xmin if getattr(gw, "xmin", None) is not None else u)
        self.spin_xmax.setValue(gw.xmax if getattr(gw, "xmax", None) is not None else u)
        self.spin_ymin.setValue(gw.ymin if getattr(gw, "ymin", None) is not None else u)
        self.spin_ymax.setValue(gw.ymax if getattr(gw, "ymax", None) is not None else u)
        self.spin_zmin.setValue(gw.zmin if getattr(gw, "zmin", None) is not None else u)
        self.spin_zmax.setValue(gw.zmax if getattr(gw, "zmax", None) is not None else u)
        self._update_range_slider_bounds()

        x_min, x_max, y_min, y_max = self._get_current_ax_limits()
        self.spin_xmin.set_placeholder_value(x_min)
        self.spin_xmax.set_placeholder_value(x_max)
        self.spin_ymin.set_placeholder_value(y_min)
        self.spin_ymax.set_placeholder_value(y_max)
        self.spin_zmin.set_placeholder_value(self.spin_zmin._start_value)
        self.spin_zmax.set_placeholder_value(self.spin_zmax._start_value)

        # Scales
        if getattr(gw, "xlogscale", False):
            self.combo_x_scale.setCurrentText(
                "Symlog" if getattr(gw, "xscale_mode", "log") == "symlog" else "Logarithmic"
            )
        else:
            self.combo_x_scale.setCurrentText("Linear")

        if getattr(gw, "ylogscale", False):
            self.combo_y_scale.setCurrentText(
                "Symlog" if getattr(gw, "yscale_mode", "log") == "symlog" else "Logarithmic"
            )
        else:
            self.combo_y_scale.setCurrentText("Linear")

        # Data types
        x_num = getattr(gw, "x_as_numeric", None)
        self.combo_x_type.setCurrentText("Auto" if x_num is None else ("Numerical" if x_num else "Category"))

        y_num = getattr(gw, "y_as_numeric", None)
        self.combo_y_type.setCurrentText("Auto" if y_num is None else ("Numerical" if y_num else "Category"))

        self.cb_invert_x.setChecked(getattr(gw, "x_inverted", False))
        self.cb_invert_y.setChecked(getattr(gw, "y_inverted", False))

        # Axis breaks
        if not hasattr(gw, "axis_breaks"):
            gw.axis_breaks = {"x": None, "y": None}
        breaks = gw.axis_breaks or {}

        if breaks.get("x"):
            self.x_break_enabled.setChecked(True)
            self.x_break_start.setValue(breaks["x"]["start"])
            self.x_break_end.setValue(breaks["x"]["end"])
        else:
            self.x_break_enabled.setChecked(False)
            mid_x = (x_min + x_max) / 2
            range_x = (x_max - x_min) / 4
            self.x_break_start.setValue(mid_x - range_x / 2)
            self.x_break_end.setValue(mid_x + range_x / 2)

        y_breaks = breaks.get("y", {})
        if y_breaks:
            self.y_break_enabled.setChecked(True)
            self.y_break_start.setValue(y_breaks.get("start", 0.0))
            self.y_break_end.setValue(y_breaks.get("end", 0.0))
        else:
            self.y_break_enabled.setChecked(False)
            mid_y = (y_min + y_max) / 2
            range_y = (y_max - y_min) / 4
            self.y_break_start.setValue(mid_y - range_y / 2)
            self.y_break_end.setValue(mid_y + range_y / 2)

    def apply_settings(self, silent: bool = False, replot: bool = False) -> dict:
        """Write controls back to graph widget, returning changed properties dict."""
        gw = self.graph_widget
        u = self._UNSET_LIMIT

        gw.xmin = self.spin_xmin.value() if self.spin_xmin.value() != u else None
        gw.xmax = self.spin_xmax.value() if self.spin_xmax.value() != u else None
        gw.ymin = self.spin_ymin.value() if self.spin_ymin.value() != u else None
        gw.ymax = self.spin_ymax.value() if self.spin_ymax.value() != u else None
        gw.zmin = self.spin_zmin.value() if self.spin_zmin.value() != u else None
        gw.zmax = self.spin_zmax.value() if self.spin_zmax.value() != u else None

        if not hasattr(gw, "axis_breaks"):
            gw.axis_breaks = {"x": None, "y": None}

        if self.x_break_enabled.isChecked():
            start, end = self.x_break_start.value(), self.x_break_end.value()
            if start >= end:
                if not silent:
                    QMessageBox.warning(self, "Invalid Range", "X-axis break start must be less than end.")
                    return {}
            else:
                gw.axis_breaks["x"] = {"start": start, "end": end}
        else:
            gw.axis_breaks["x"] = None

        if self.y_break_enabled.isChecked():
            start, end = self.y_break_start.value(), self.y_break_end.value()
            if start >= end:
                if not silent:
                    QMessageBox.warning(self, "Invalid Range", "Y-axis break start must be less than end.")
                    return {}
            else:
                gw.axis_breaks["y"] = {"start": start, "end": end}
        else:
            gw.axis_breaks["y"] = None

        x_scale_text = self.combo_x_scale.currentText()
        gw.xlogscale = x_scale_text in ("Logarithmic", "Symlog")
        gw.xscale_mode = "symlog" if x_scale_text == "Symlog" else "log"

        y_scale_text = self.combo_y_scale.currentText()
        gw.ylogscale = y_scale_text in ("Logarithmic", "Symlog")
        gw.yscale_mode = "symlog" if y_scale_text == "Symlog" else "log"

        x_text = self.combo_x_type.currentText()
        gw.x_as_numeric = None if x_text == "Auto" else (x_text == "Numerical")

        y_text = self.combo_y_type.currentText()
        gw.y_as_numeric = None if y_text == "Auto" else (y_text == "Numerical")

        gw.x_inverted = self.cb_invert_x.isChecked()
        gw.y_inverted = self.cb_invert_y.isChecked()

        props = {
            "xmin": gw.xmin, "xmax": gw.xmax,
            "ymin": gw.ymin, "ymax": gw.ymax,
            "zmin": gw.zmin, "zmax": gw.zmax,
            "axis_breaks": gw.axis_breaks,
            "xlogscale": gw.xlogscale, "ylogscale": gw.ylogscale,
            "xscale_mode": gw.xscale_mode, "yscale_mode": gw.yscale_mode,
            "x_as_numeric": gw.x_as_numeric, "y_as_numeric": gw.y_as_numeric,
            "x_inverted": gw.x_inverted, "y_inverted": gw.y_inverted,
        }

        if getattr(gw, "plot_style", "") in ("wafer", "2Dmap"):
            gw.colormap_norm = self.combo_z_scale.currentData()
            gw.colormap_center = self.spin_colormap_center.value()
            props["colormap_norm"] = gw.colormap_norm
            props["colormap_center"] = gw.colormap_center

        if replot and getattr(gw, "df", None) is not None:
            gw.plot(gw.df)

        return props

    def highlight_search(self, query: str) -> None:
        """Highlight matching groupboxes based on search query."""
        q = query.strip().lower()
        self.grp_scales.setStyleSheet("")
        self.grp_limits.setStyleSheet("")
        self.grp_break.setStyleSheet("")

        if not q:
            return

        scale_terms = ["scale", "scales", "linear", "log", "logarithmic", "symlog", "centered", "type", "invert", "inverted", "colormap"]
        limit_terms = ["limit", "limits", "range", "slider", "min", "max", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
        break_terms = ["break", "broken", "x-axis break", "y-axis break", "gap"]

        first_match: Optional[QWidget] = None
        if any(t in q or q in t for t in scale_terms):
            self.grp_scales.setStyleSheet(highlight_group_style())
            if first_match is None:
                first_match = self.grp_scales

        if any(t in q or q in t for t in limit_terms):
            self.grp_limits.setStyleSheet(highlight_group_style())
            if first_match is None:
                first_match = self.grp_limits

        if any(t in q or q in t for t in break_terms):
            self.grp_break.setStyleSheet(highlight_group_style())
            if first_match is None:
                first_match = self.grp_break

        if first_match:
            self._scroll.ensureWidgetVisible(first_match, 0, 30)

