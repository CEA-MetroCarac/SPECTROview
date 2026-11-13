"""Live Maps Tab View component for SPECTROview.

Displays 3 live-updating panels corresponding to Red, Blue, and Green
spectral range selectors:
- In Maps workspace ('map' mode): renders 2D spatial heatmaps with colormap
  selection for 2D maps, or line plots along the profile for 1D maps (profiles).
- In Spectra workspace ('spectra' mode): renders 1D parameter variation graphs across selected spectra.

The Green panel can alternatively display the Red/Blue intensity ratio (R/B).
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, AutoLocator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QFrame, QSizePolicy, QScrollArea, QToolButton,
    QMenu, QWidgetAction, QDoubleSpinBox, QSpinBox,
)

from spectroview import PLOT_POLICY_LIGHT, PLOT_POLICY_DARK, DEFAULT_COLORS
from spectroview.view.theme import Icons, current_tokens
from spectroview.viewmodel.utils import copy_fig_to_clb
from spectroview.model.spectral_parameters import (
    SPECTRAL_PARAMETERS,
    PARAM_INTEGRATED_INTENSITY,
    PARAM_RB_RATIO,
)
from spectroview.view.components.v_range_selector import (
    SELECTOR_RED,
    SELECTOR_BLUE,
    SELECTOR_GREEN,
    SELECTOR_KEYS,
    SELECTOR_COLORS,
    SELECTOR_NAMES,
)
from spectroview.view.components.v_live_maps_legend_editor import (
    LEGEND_LOCATIONS,
    LiveMapsLegendEditorDialog,
)


DEFAULT_PALETTES = {
    SELECTOR_RED: "Reds",
    SELECTOR_BLUE: "Blues",
    SELECTOR_GREEN: "Greens",
}

AVAILABLE_PALETTES = [
    "viridis", "plasma", "inferno", "magma", "cividis", "turbo",
    "Reds", "Blues", "Greens", "coolwarm", "bwr", "jet", "rainbow"
]

_DEFAULT_VIEW_OPTIONS = {
    "xscale": "linear",
    "yscale": "linear",
    "zscale": "linear",
    "plot_style": "line+dots",
    "line_width": 1.8,
    "dot_size": 4.0,
    "show_legend": False,
    "legend_loc": "best",
    "legend_ncol": 1,
    "legend_fontsize": 8,
    "show_grid": True,
    "share_y": False,
    "heatmap_smoothing": False,
    "show_colorbar": True,
    "custom_labels": {},
    "custom_colors": {},
}


def _copy_view_options(src: Optional[dict] = None) -> dict:
    """Return an isolated copy of Live Maps view options with cloned dicts."""
    base = dict(_DEFAULT_VIEW_OPTIONS)
    if src:
        base.update(src)
    base["custom_labels"] = dict(base.get("custom_labels") or {})
    base["custom_colors"] = dict(base.get("custom_colors") or {})
    return base


class LiveMapSubpanel(QFrame):
    """Subpanel rendering a 2D heatmap or a 1D line plot for one spectral range.

    The plot type follows the data received, not a fixed mode:
    ``update_map`` draws a 2D heatmap (2D maps), ``update_series`` draws a
    line plot — either vs. spectrum index (Spectra workspace) or vs. spatial
    position (1D profile maps, when ``x_values`` is given).
    """

    parameter_changed = Signal(str, str)  # (color_id, parameter_name)
    legend_edit_requested = Signal()

    def __init__(self, color_id: str, parent=None, mode: str = "map"):
        super().__init__(parent)
        self.color_id = color_id
        self.color_hex = SELECTOR_COLORS[color_id]
        self.color_name = SELECTOR_NAMES[color_id]
        self.mode = mode  # "map" or "spectra"

        self.setObjectName(f"liveMapPanel_{color_id}")
        self.setFrameShape(QFrame.StyledPanel)

        # Cached 2D grid data (Maps mode)
        self._grid = None
        self._extent = [0.0, 1.0, 0.0, 1.0]
        self._vmin = 0.0
        self._vmax = 1.0

        # Cached 1D series data (Spectra mode / 1D profile maps)
        self._series_values: Optional[np.ndarray] = None
        self._series_labels: list[str] = []
        self._series_x: Optional[np.ndarray] = None  # None → spectrum index axis
        self._series_xlabel: str = "Spectrum #"

        # Cached multi-1D profile entries
        self._multi_entries: List[dict] = []
        self._multi_x_label: str = "X"

        # What is currently drawn: None, "map", "series", or "multi_profile"
        self._display_kind: Optional[str] = None

        self._status_msg = ""
        self._x_range = (0.0, 100.0)
        self._current_theme = "Light Mode"

        # Artist references (multiple Line2Ds supported for multi-1D overlay)
        self.img = None
        self.cbar = None
        self.lines: List = []
        self._legend = None
        self._view_options = _copy_view_options()
        self._valid_y_bounds: Optional[Tuple[float, float]] = None
        self._valid_pos_y_bounds: Optional[Tuple[float, float]] = None

        self._init_ui()
        self.set_mode(mode)

    @property
    def line(self):
        """Primary single Line2D artist for backward-compatible callers and tests."""
        return self.lines[0] if self.lines else None

    @line.setter
    def line(self, value):
        self.lines = [] if value is None else [value]

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Header Bar: 2 compact rows to prevent squeezing sidebar ──
        # Row 1: Badge + Range + Stretch + Copy
        row1_layout = QHBoxLayout()
        row1_layout.setContentsMargins(0, 0, 0, 0)
        row1_layout.setSpacing(4)

        self.lbl_badge = QLabel(self)
        self.lbl_badge.setText(f" {self.color_name} ")
        self.lbl_badge.setStyleSheet(f"""
            QLabel {{
                background-color: {self.color_hex};
                color: white;
                font-weight: bold;
                border-radius: 3px;
                padding: 1px 4px;
            }}
        """)
        row1_layout.addWidget(self.lbl_badge)

        self.lbl_range = QLabel("[0.0, 0.0]", self)
        self.lbl_range.setFixedWidth(115)
        self.lbl_range.setAlignment(Qt.AlignCenter)
        self.lbl_range.setStyleSheet("font-size: 11px; font-weight: 500;")
        row1_layout.addWidget(self.lbl_range)

        row1_layout.addStretch(1)

        self.btn_copy = QPushButton(self)
        self.btn_copy.setIcon(Icons.copy())
        self.btn_copy.setIconSize(QSize(14, 14))
        self.btn_copy.setFixedSize(22, 22)
        self.btn_copy.setToolTip("Copy map to clipboard")
        self.btn_copy.clicked.connect(self._copy_to_clipboard)
        row1_layout.addWidget(self.btn_copy)

        layout.addLayout(row1_layout)

        # Row 2: Parameter combo + Colormap combo
        row2_layout = QHBoxLayout()
        row2_layout.setContentsMargins(0, 0, 0, 0)
        row2_layout.setSpacing(4)

        self.cbb_param = QComboBox(self)
        for param in SPECTRAL_PARAMETERS:
            self.cbb_param.addItem(param)
        if self.color_id == SELECTOR_GREEN:
            # Derived option: Red/Blue integrated-intensity ratio
            self.cbb_param.insertSeparator(self.cbb_param.count())
            self.cbb_param.addItem(PARAM_RB_RATIO)
            self.cbb_param.setItemData(
                self.cbb_param.count() - 1,
                "Ratio of the Red and Blue integrated intensities (I_Red / I_Blue).\n"
                "Requires the Red and Blue range selectors to be active.",
                Qt.ToolTipRole,
            )
        self.cbb_param.setCurrentText(PARAM_INTEGRATED_INTENSITY)
        self.cbb_param.setToolTip("Select spectral parameter for this map")
        self.cbb_param.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cbb_param.currentTextChanged.connect(self._on_param_changed)
        row2_layout.addWidget(self.cbb_param, stretch=1)

        self.cbb_palette = QComboBox(self)
        self.cbb_palette.addItems(AVAILABLE_PALETTES)
        default_pal = DEFAULT_PALETTES.get(self.color_id, "viridis")
        self.cbb_palette.setCurrentText(default_pal)
        self.cbb_palette.setFixedWidth(85)
        self.cbb_palette.setToolTip("Select colormap")
        self.cbb_palette.currentTextChanged.connect(self._on_palette_changed)
        row2_layout.addWidget(self.cbb_palette)

        layout.addLayout(row2_layout)

        # ── Matplotlib Figure & Canvas ──
        with plt.style.context(PLOT_POLICY_LIGHT):
            self.figure = Figure(dpi=70, layout="compressed")
            self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_button_press)
        layout.addWidget(self.canvas, stretch=1)

    def _on_canvas_button_press(self, event):
        """Open legend label editor when double-clicking inside the Matplotlib legend box."""
        if not getattr(event, "dblclick", False) or self._legend is None:
            return
        if event.x is None or event.y is None:
            return
        try:
            renderer = self.canvas.get_renderer()
            if renderer is None:
                return
            if self._legend.get_window_extent(renderer=renderer).contains(event.x, event.y):
                self.legend_edit_requested.emit()
        except Exception:
            pass

    def set_view_options(self, options: dict, redraw: bool = False):
        """Apply view options dictionary and optionally re-render cached artists."""
        self._view_options = _copy_view_options(options)
        if redraw and self._display_kind is not None:
            self.rerender_cached(theme=self._current_theme)

    def rerender_cached(self, theme: Optional[str] = None, force_rebuild: bool = False):
        """Re-render the currently displayed plot from cached arrays without recomputing."""
        active_theme = theme or self._current_theme
        kind = self._display_kind or ("series" if self.mode == "spectra" else "map")
        if force_rebuild:
            self.img = None
            self.lines.clear()
        if kind == "multi_profile":
            self.update_multi_profile(
                self._multi_entries,
                shared_x_label=self._multi_x_label,
                status_msg=self._status_msg,
                theme=active_theme,
            )
        elif kind == "series":
            self.update_series(
                self._series_values,
                self._series_labels,
                self._status_msg,
                theme=active_theme,
                x_values=self._series_x,
                x_label=self._series_xlabel,
            )
        else:
            self.update_map(
                self._grid,
                self._extent,
                self._vmin,
                self._vmax,
                self._status_msg,
                theme=active_theme,
            )

    def set_mode(self, mode: str):
        """Configure subpanel for 'map' (2D heatmap) or 'spectra' (1D variation graph) mode."""
        self.mode = mode
        if mode == "map":
            self.btn_copy.setToolTip("Copy map to clipboard")
            self.cbb_param.setToolTip("Select spectral parameter for this map")
        else:
            self.btn_copy.setToolTip("Copy graph to clipboard")
            self.cbb_param.setToolTip("Select spectral parameter for this graph")
        self._sync_palette_visibility()

    def _sync_palette_visibility(self):
        """Colormap selector only applies to heatmaps (Maps mode, not 1D line plots)."""
        self.cbb_palette.setVisible(
            self.mode == "map" and self._display_kind not in ("series", "multi_profile")
        )

    def _reset_axes(self):
        """Remove any colorbar, legend, clear lines, clear axes, and restore aspect scaling.

        Matplotlib's ``Axes.clear()`` does not reset the aspect ratio set by
        ``imshow(..., aspect="equal")``; without resetting to ``"auto"``, a
        subsequent 1D line plot inherits a 1:1 data aspect and collapses into a
        thin vertical or horizontal strip when X and Y spans differ.
        """
        if self._legend is not None:
            try:
                self._legend.remove()
            except Exception:
                pass
            self._legend = None
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
        self.img = None
        for ln in self.lines:
            try:
                ln.remove()
            except Exception:
                pass
        self.lines.clear()
        self.ax.clear()
        self.ax.set_aspect("auto", adjustable="box")

    def clear_plot(self):
        """Clear all plotted artists and cached data."""
        self._grid = None
        self._series_values = None
        self._series_labels = []
        self._series_x = None
        self._multi_entries = []
        self._display_kind = None
        self._valid_y_bounds = None
        self._valid_pos_y_bounds = None
        self._reset_axes()
        if self.canvas is not None:
            self.canvas.draw_idle()

    def is_ratio_display(self) -> bool:
        """True if this panel currently displays the Red/Blue ratio."""
        return self.cbb_param.currentText() == PARAM_RB_RATIO

    def _on_param_changed(self, param: str):
        self._refresh_range_label()
        self.parameter_changed.emit(self.color_id, param)

    def _copy_to_clipboard(self):
        """Copy figure to system clipboard."""
        copy_fig_to_clb(self.canvas)

    def _on_palette_changed(self, palette_name: str):
        """Update colormap without recomputing grid data."""
        if self.img is not None and self.canvas is not None:
            try:
                cmap = plt.get_cmap(palette_name).copy()
                cmap.set_bad(color="none")
                self.img.set_cmap(cmap)
                self.canvas.draw_idle()
            except Exception:
                pass

    def set_range_text(self, x_min: float, x_max: float):
        """Update range label in header."""
        self._x_range = (x_min, x_max)
        self._refresh_range_label()

    def _refresh_range_label(self):
        if self.is_ratio_display():
            self.lbl_range.setText("I(Red) / I(Blue)")
        else:
            x_min, x_max = self._x_range
            self.lbl_range.setText(f"[{x_min:.1f}, {x_max:.1f}]")

    def get_selected_parameter(self) -> str:
        """Return currently selected parameter name."""
        return self.cbb_param.currentText()

    def set_selected_parameter(self, parameter: str):
        """Select a parameter in the dropdown."""
        idx = self.cbb_param.findText(parameter)
        if idx >= 0:
            self.cbb_param.setCurrentIndex(idx)

    @staticmethod
    def _padded_ylim(val_arr: np.ndarray) -> Tuple[float, float]:
        ymin = float(np.nanmin(val_arr))
        ymax = float(np.nanmax(val_arr))
        span = ymax - ymin
        pad = span * 0.12 if span > 1e-12 else (abs(ymin) * 0.1 if abs(ymin) > 1e-12 else 1.0)
        return ymin - pad, ymax + pad

    def _curve_style_kwargs(self, is_primary: bool = True, multi: bool = False) -> dict:
        """Build Matplotlib linestyle, linewidth, marker, and markersize from view options."""
        opts = self._view_options
        style = opts.get("plot_style", "line+dots")
        lw = float(opts.get("line_width", 1.8))
        ds = float(opts.get("dot_size", 4.0))
        if multi:
            lw = lw + 0.4 if is_primary else max(0.5, lw - 0.4)
        if style == "dot":
            return {"linestyle": "none", "linewidth": 0.0, "marker": "o", "markersize": ds}
        if style == "line":
            return {"linestyle": "-", "linewidth": lw, "marker": "", "markersize": 0.0}
        return {"linestyle": "-", "linewidth": lw, "marker": "o", "markersize": ds}

    def _apply_x_limits(self, x: np.ndarray, is_index_axis: bool):
        """Set x-limits/ticks for an index axis (Spectrum #) or a spatial profile axis."""
        if len(x) == 0:
            return
        n = len(x)
        if is_index_axis:
            if n == 1:
                self.ax.set_xlim(0.5, 1.5)
                self.ax.set_xticks([1])
            else:
                margin = max(0.25, 0.04 * (n - 1))
                self.ax.set_xlim(1.0 - margin, float(n) + margin)
                self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            return
        x_lo = float(np.nanmin(x))
        x_hi = float(np.nanmax(x))
        span = x_hi - x_lo
        margin = span * 0.03 if span > 1e-12 else 0.5
        self.ax.set_xlim(x_lo - margin, x_hi + margin)
        self.ax.xaxis.set_major_locator(AutoLocator())

    def _apply_1d_scales_and_limits(
        self,
        all_x: np.ndarray,
        all_y: np.ndarray,
        is_index_axis: bool,
    ):
        """Apply linear/log X and Y axis scaling with guards for non-positive domains."""
        opts = self._view_options
        req_xscale = opts.get("xscale", "linear")
        req_yscale = opts.get("yscale", "linear")

        finite_y = all_y[np.isfinite(all_y)]
        if finite_y.size > 0:
            self._valid_y_bounds = self._padded_ylim(finite_y)
            pos_y = finite_y[finite_y > 0]
            if pos_y.size > 0:
                p_min = float(np.min(pos_y))
                p_max = float(np.max(pos_y))
                if p_max / max(p_min, 1e-30) < 1.05:
                    self._valid_pos_y_bounds = (max(p_min * 0.7, 1e-300), p_max * 1.4)
                else:
                    self._valid_pos_y_bounds = (max(p_min * 0.85, 1e-300), p_max * 1.15)
            else:
                self._valid_pos_y_bounds = None
        else:
            self._valid_y_bounds = None
            self._valid_pos_y_bounds = None

        if req_yscale == "log" and self._valid_pos_y_bounds is not None:
            self.ax.set_yscale("log")
            self.ax.set_ylim(*self._valid_pos_y_bounds)
        else:
            self.ax.set_yscale("linear")
            if self._valid_y_bounds is not None:
                self.ax.set_ylim(*self._valid_y_bounds)

        finite_x = all_x[np.isfinite(all_x)]
        if req_xscale == "log":
            pos_x = finite_x[finite_x > 0]
            if pos_x.size > 0:
                self.ax.set_xscale("log")
                x_min = float(np.min(pos_x))
                x_max = float(np.max(pos_x))
                if x_max / max(x_min, 1e-30) < 1.05:
                    self.ax.set_xlim(max(x_min * 0.8, 1e-300), x_max * 1.25)
                else:
                    self.ax.set_xlim(max(x_min * 0.92, 1e-300), x_max * 1.08)
                return
        self.ax.set_xscale("linear")
        self._apply_x_limits(finite_x if not is_index_axis else all_x, is_index_axis)

    def _sync_legend(self):
        """Create, update, or remove the Matplotlib legend box per current view options."""
        if self._legend is not None:
            try:
                self._legend.remove()
            except Exception:
                pass
            self._legend = None

        if not self._view_options.get("show_legend", False):
            return
        if self._display_kind not in ("series", "multi_profile"):
            return

        handles = []
        labels = []
        for ln in self.lines:
            lbl = ln.get_label()
            if lbl and not str(lbl).startswith("_"):
                handles.append(ln)
                labels.append(str(lbl))
        if not handles:
            return

        loc = self._view_options.get("legend_loc", "best")
        ncol = max(1, int(self._view_options.get("legend_ncol", 1)))
        fontsize = max(6, int(self._view_options.get("legend_fontsize", 8)))
        if loc == "outside right":
            self._legend = self.ax.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                ncol=ncol,
                fontsize=fontsize,
                frameon=True,
            )
        elif loc == "outside top":
            self._legend = self.ax.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=ncol,
                fontsize=fontsize,
                frameon=True,
            )
        else:
            if loc not in LEGEND_LOCATIONS:
                loc = "best"
            self._legend = self.ax.legend(
                handles,
                labels,
                loc=loc,
                ncol=ncol,
                fontsize=fontsize,
                frameon=True,
            )
            self._legend.set_draggable(True)

    def update_series(
        self,
        values: Optional[np.ndarray],
        labels: Optional[list[str]] = None,
        status_msg: str = "",
        theme: str = "Light Mode",
        x_values: Optional[np.ndarray] = None,
        x_label: str = "Spectrum #",
    ):
        """Render or update a 1D line plot.

        Without ``x_values`` the values are plotted against the spectrum index
        (Spectra workspace). With ``x_values`` they are plotted against those
        positions, e.g. the spatial coordinate of a 1D profile map.
        """
        val_arr = np.asarray(values, dtype=np.float64) if values is not None else np.array([], dtype=np.float64)
        is_index_axis = x_values is None
        if is_index_axis:
            x_arr = np.arange(1, len(val_arr) + 1, dtype=np.float64)
        else:
            x_arr = np.asarray(x_values, dtype=np.float64)
            if len(x_arr) != len(val_arr):
                val_arr = np.array([], dtype=np.float64)
        self._series_values = val_arr
        self._series_labels = list(labels) if labels else []
        self._series_x = None if is_index_axis else x_arr
        self._series_xlabel = x_label
        self._multi_entries = []
        self._status_msg = status_msg
        self._current_theme = theme

        param_name = self.cbb_param.currentText()
        title_text = f"{self.color_name}: {param_name}"
        custom_labels = self._view_options.get("custom_labels", {})
        custom_colors = self._view_options.get("custom_colors", {})
        curve_label = custom_labels.get(self.color_id) or title_text
        curve_color = custom_colors.get(self.color_id) or self.color_hex
        style_kw = self._curve_style_kwargs(is_primary=True, multi=False)
        show_grid = bool(self._view_options.get("show_grid", True))

        has_valid = len(val_arr) > 0 and bool(np.any(np.isfinite(val_arr)))

        # Fast path: in-place single-line update during continuous bracket dragging
        if (
            self._display_kind == "series"
            and len(self.lines) == 1
            and self.line is not None
            and has_valid
            and len(self.line.get_xdata()) == len(val_arr)
        ):
            self.line.set_data(x_arr, val_arr)
            self.line.set_color(curve_color)
            self.line.set_linestyle(style_kw["linestyle"])
            self.line.set_linewidth(style_kw["linewidth"])
            self.line.set_marker(style_kw["marker"])
            self.line.set_markersize(style_kw["markersize"])
            self.line.set_label(curve_label)
            finite_mask = np.isfinite(x_arr) & np.isfinite(val_arr)
            self._apply_1d_scales_and_limits(
                x_arr[finite_mask] if np.any(finite_mask) else x_arr,
                val_arr[finite_mask] if np.any(finite_mask) else val_arr,
                is_index_axis=is_index_axis,
            )
            self.ax.grid(show_grid, alpha=0.25, linestyle="--")
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel(param_name)
            self.ax.set_title(title_text, fontsize=10, pad=4)
            self._sync_legend()
            self.canvas.draw_idle()
            return

        is_dark = theme in ("Dark Mode", "dark", "classic_dark")
        style_path = PLOT_POLICY_DARK if is_dark else PLOT_POLICY_LIGHT

        self._display_kind = "series"
        self._sync_palette_visibility()

        with plt.style.context(style_path):
            self._reset_axes()

            if not has_valid:
                self._valid_y_bounds = None
                self._valid_pos_y_bounds = None
                msg = status_msg or ("No spectra selected" if is_index_axis else "No profile data available")
                self.ax.text(
                    0.5, 0.5, msg,
                    ha="center", va="center", transform=self.ax.transAxes,
                    fontsize=10, color="gray", wrap=True
                )
                self.ax.set_xticks([])
                self.ax.set_yticks([])
                self.canvas.draw_idle()
                return

            self.lines = list(
                self.ax.plot(
                    x_arr,
                    val_arr,
                    color=curve_color,
                    alpha=0.95,
                    label=curve_label,
                    **style_kw,
                )
            )

            finite_mask = np.isfinite(x_arr) & np.isfinite(val_arr)
            self._apply_1d_scales_and_limits(
                x_arr[finite_mask] if np.any(finite_mask) else x_arr,
                val_arr[finite_mask] if np.any(finite_mask) else val_arr,
                is_index_axis=is_index_axis,
            )
            self.ax.grid(show_grid, alpha=0.25, linestyle="--")
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel(param_name)
            self.ax.set_title(title_text, fontsize=10, pad=4)
            self._sync_legend()

            self.canvas.draw_idle()

    def update_multi_profile(
        self,
        entries: List[dict],
        shared_x_label: str = "X",
        status_msg: str = "",
        theme: str = "Light Mode",
    ):
        """Render or update multiple stacked 1D spatial profiles in one subpanel."""
        self._multi_entries = list(entries) if entries else []
        self._multi_x_label = shared_x_label
        self._series_values = None
        self._series_labels = []
        self._series_x = None
        self._status_msg = status_msg
        self._current_theme = theme

        param_name = self.cbb_param.currentText()
        title_text = f"{self.color_name}: {param_name}"
        x_label = (
            shared_x_label
            if shared_x_label.endswith("(mm)")
            else f"{shared_x_label or 'Position'} (mm)"
        )

        # Compute manual finite bounds across all valid curves (guards NaN/empty curves)
        finite_x_parts = []
        finite_y_parts = []
        for entry in self._multi_entries:
            px = np.asarray(entry.get("positions", []), dtype=np.float64)
            vy = np.asarray(entry.get("values", []), dtype=np.float64)
            if len(px) == 0 or len(px) != len(vy):
                continue
            mask = np.isfinite(px) & np.isfinite(vy)
            if np.any(mask):
                finite_x_parts.append(px[mask])
                finite_y_parts.append(vy[mask])

        has_valid = len(finite_y_parts) > 0
        if has_valid:
            all_x = np.concatenate(finite_x_parts)
            all_y = np.concatenate(finite_y_parts)
        else:
            all_x = np.array([], dtype=np.float64)
            all_y = np.array([], dtype=np.float64)

        custom_labels = self._view_options.get("custom_labels", {})
        custom_colors = self._view_options.get("custom_colors", {})
        show_grid = bool(self._view_options.get("show_grid", True))

        # Fast path: reuse existing Line2D pool when entry count matches
        if (
            self._display_kind == "multi_profile"
            and has_valid
            and self.img is None
            and len(self.lines) == len(self._multi_entries)
            and len(self.lines) > 0
        ):
            for idx, (ln, entry) in enumerate(zip(self.lines, self._multi_entries)):
                px = np.asarray(entry.get("positions", []), dtype=np.float64)
                vy = np.asarray(entry.get("values", []), dtype=np.float64)
                key = entry.get("name", f"Map {idx + 1}")
                auto_color = (
                    self.color_hex
                    if len(self._multi_entries) == 1
                    else DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
                )
                color = custom_colors.get(key) or auto_color
                is_primary = bool(entry.get("is_primary", False))
                style_kw = self._curve_style_kwargs(is_primary=is_primary, multi=True)
                ln.set_data(px, vy)
                ln.set_color(color)
                ln.set_linestyle(style_kw["linestyle"])
                ln.set_linewidth(style_kw["linewidth"])
                ln.set_marker(style_kw["marker"])
                ln.set_markersize(style_kw["markersize"])
                ln.set_alpha(1.0 if is_primary else 0.78)
                ln.set_zorder(5 if is_primary else 3)
                ln.set_label(custom_labels.get(key) or key)
            self._apply_1d_scales_and_limits(all_x, all_y, is_index_axis=False)
            self.ax.grid(show_grid, alpha=0.25, linestyle="--")
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel(param_name)
            self.ax.set_title(title_text, fontsize=10, pad=4)
            self._sync_legend()
            self.canvas.draw_idle()
            return

        is_dark = theme in ("Dark Mode", "dark", "classic_dark")
        style_path = PLOT_POLICY_DARK if is_dark else PLOT_POLICY_LIGHT

        self._display_kind = "multi_profile"
        self._sync_palette_visibility()

        with plt.style.context(style_path):
            self._reset_axes()

            if not has_valid:
                self._valid_y_bounds = None
                self._valid_pos_y_bounds = None
                msg = status_msg or "No valid profile data within selected range"
                self.ax.text(
                    0.5, 0.5, msg,
                    ha="center", va="center", transform=self.ax.transAxes,
                    fontsize=10, color="gray", wrap=True,
                )
                self.ax.set_xticks([])
                self.ax.set_yticks([])
                self.canvas.draw_idle()
                return

            new_lines = []
            for idx, entry in enumerate(self._multi_entries):
                px = np.asarray(entry.get("positions", []), dtype=np.float64)
                vy = np.asarray(entry.get("values", []), dtype=np.float64)
                key = entry.get("name", f"Map {idx + 1}")
                auto_color = (
                    self.color_hex
                    if len(self._multi_entries) == 1
                    else DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
                )
                color = custom_colors.get(key) or auto_color
                is_primary = bool(entry.get("is_primary", False))
                style_kw = self._curve_style_kwargs(is_primary=is_primary, multi=True)
                created = self.ax.plot(
                    px,
                    vy,
                    color=color,
                    alpha=1.0 if is_primary else 0.78,
                    zorder=5 if is_primary else 3,
                    label=custom_labels.get(key) or key,
                    **style_kw,
                )
                new_lines.extend(created)
            self.lines = new_lines

            self._apply_1d_scales_and_limits(all_x, all_y, is_index_axis=False)
            self.ax.grid(show_grid, alpha=0.25, linestyle="--")
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel(param_name)
            self.ax.set_title(title_text, fontsize=10, pad=4)
            self._sync_legend()

            self.canvas.draw_idle()

    def _resolve_2d_render_params(self, grid: np.ndarray, extent: list[float], vmin: float, vmax: float):
        """Compute normalized grid, norm, interpolation, axis scales, and aspect for 2D heatmaps."""
        opts = self._view_options
        interp = "bilinear" if opts.get("heatmap_smoothing", False) else "none"
        use_log_z = opts.get("zscale", "linear") == "log"
        norm = None
        display_grid = grid

        if use_log_z and vmax > 0:
            pos_vals = grid[np.isfinite(grid) & (grid > 0)]
            if pos_vals.size > 0:
                pos_min = float(np.min(pos_vals))
                safe_vmin = max(vmin, pos_min) if vmin > 0 else pos_min
                if safe_vmin >= vmax:
                    safe_vmin = pos_min * 0.9 if pos_min > 0 else 1e-12
                display_grid = np.where(np.isfinite(grid) & (grid > 0), grid, np.nan)
                norm = LogNorm(vmin=max(safe_vmin, 1e-300), vmax=max(vmax, safe_vmin * 1.01))

        if norm is None:
            norm = Normalize(vmin=vmin, vmax=vmax)

        req_xscale = "log" if (opts.get("xscale", "linear") == "log" and len(extent) >= 2 and extent[1] > 0) else "linear"
        req_yscale = "log" if (opts.get("yscale", "linear") == "log" and len(extent) >= 4 and extent[3] > 0) else "linear"
        aspect = "equal" if (req_xscale == "linear" and req_yscale == "linear") else "auto"
        return display_grid, norm, interp, req_xscale, req_yscale, aspect

    def _apply_2d_axis_scales(self, extent: list[float], xscale: str, yscale: str, show_grid: bool):
        """Set linear/log scales and valid domain bounds on 2D heatmap axes."""
        if xscale == "log":
            self.ax.set_xscale("log")
            x_hi = float(extent[1])
            x_lo = float(extent[0]) if float(extent[0]) > 0 else max(x_hi * 1e-3, 1e-6)
            self.ax.set_xlim(x_lo, x_hi)
        else:
            self.ax.set_xscale("linear")
            self.ax.set_xlim(extent[0], extent[1])

        if yscale == "log":
            self.ax.set_yscale("log")
            y_hi = float(extent[3])
            y_lo = float(extent[2]) if float(extent[2]) > 0 else max(y_hi * 1e-3, 1e-6)
            self.ax.set_ylim(y_lo, y_hi)
        else:
            self.ax.set_yscale("linear")
            self.ax.set_ylim(extent[2], extent[3])

        self.ax.grid(show_grid, alpha=0.25, linestyle="--")

    def update_map(
        self,
        grid: Optional[np.ndarray],
        extent: list[float],
        vmin: float,
        vmax: float,
        status_msg: str = "",
        theme: str = "Light Mode",
    ):
        """Render or update the 2D heatmap on the canvas."""
        self._grid = grid
        self._extent = extent
        self._vmin = vmin
        self._vmax = vmax
        self._series_values = None
        self._multi_entries = []
        self._valid_y_bounds = None
        self._valid_pos_y_bounds = None
        self._status_msg = status_msg
        self._current_theme = theme

        param_name = self.cbb_param.currentText()
        title_text = f"{self.color_name}: {param_name}"
        want_colorbar = bool(self._view_options.get("show_colorbar", True))
        show_grid = bool(self._view_options.get("show_grid", True))
        has_valid_grid = grid is not None and not np.all(np.isnan(grid))

        # Fast path: in-place update when image and colorbar presence match expectations
        if (
            self._display_kind == "map"
            and self.img is not None
            and (self.cbar is not None) == want_colorbar
            and has_valid_grid
        ):
            display_grid, norm, interp, xscale, yscale, aspect = self._resolve_2d_render_params(
                grid, extent, vmin, vmax
            )
            self.img.set_data(display_grid)
            self.img.set_extent(extent)
            self.img.set_norm(norm)
            self.img.set_interpolation(interp)
            self.ax.set_aspect(aspect, adjustable="box")
            self._apply_2d_axis_scales(extent, xscale, yscale, show_grid)
            self.ax.set_title(title_text, fontsize=10, pad=4)
            if self.cbar is not None:
                self.cbar.update_normal(self.img)
            self.canvas.draw_idle()
            return

        # Re-render path for status messages, colorbar toggle, or first draw
        is_dark = theme in ("Dark Mode", "dark", "classic_dark")
        style_path = PLOT_POLICY_DARK if is_dark else PLOT_POLICY_LIGHT

        self._display_kind = "map"
        self._sync_palette_visibility()

        with plt.style.context(style_path):
            self._reset_axes()

            if not has_valid_grid:
                msg = status_msg or "No map data available"
                self.ax.text(
                    0.5, 0.5, msg,
                    ha="center", va="center", transform=self.ax.transAxes,
                    fontsize=10, color="gray", wrap=True
                )
                self.ax.set_xticks([])
                self.ax.set_yticks([])
                self.canvas.draw_idle()
                return

            cmap_name = self.cbb_palette.currentText()
            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad(color="none")

            display_grid, norm, interp, xscale, yscale, aspect = self._resolve_2d_render_params(
                grid, extent, vmin, vmax
            )
            self.img = self.ax.imshow(
                display_grid,
                extent=extent,
                origin="lower",
                aspect=aspect,
                cmap=cmap,
                norm=norm,
                interpolation=interp,
            )

            if want_colorbar:
                self.cbar = self.figure.colorbar(self.img, ax=self.ax, fraction=0.046, pad=0.04)

            self._apply_2d_axis_scales(extent, xscale, yscale, show_grid)
            self.ax.set_xlabel("X (mm)")
            self.ax.set_ylabel("Y (mm)")
            self.ax.set_title(title_text, fontsize=10, pad=4)

            self.canvas.draw_idle()


class VLiveMaps(QWidget):
    """Live Maps Tab View coordinating the 3 spectral range subpanels."""

    parameter_change_requested = Signal(str, str)  # (color_id, parameter)
    active_2d_map_requested = Signal(str)          # active 2D map name from switcher

    def __init__(self, parent=None, mode: str = "map"):
        super().__init__(parent)
        self.setObjectName("LiveMapsTab")
        self.mode = mode

        self.panels: Dict[str, LiveMapSubpanel] = {}
        self._active_states: Dict[str, bool] = {
            SELECTOR_RED: False,
            SELECTOR_BLUE: False,
            SELECTOR_GREEN: False,
        }
        self._ranges: Dict[str, Tuple[float, float]] = {
            SELECTOR_RED: (200.0, 400.0),
            SELECTOR_BLUE: (500.0, 700.0),
            SELECTOR_GREEN: (800.0, 1000.0),
        }
        self._2d_candidates: List[str] = []
        self._legend_entries: List[Tuple[str, bool]] = []
        self._last_raw_entries: List[dict] = []
        self._context_status: str = ""
        self._view_options = _copy_view_options()

        self._current_theme = "Light Mode"
        self._init_ui()
        self.set_mode(mode)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # ── Top bar: slim context header (left) + spacer + view options buttons (right) ──
        self.top_bar = QWidget(self)
        self.top_bar.setObjectName("liveMapsTopBar")
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)

        # ── Slim shared context header (multi-1D legend, 2D switcher, status) ──
        self.header_bar = QFrame(self.top_bar)
        self.header_bar.setObjectName("liveMapsContextHeader")
        self.header_bar.setFixedHeight(30)
        header_layout = QHBoxLayout(self.header_bar)
        header_layout.setContentsMargins(6, 2, 6, 2)
        header_layout.setSpacing(8)

        # Multi-1D colored chip legend strip (scrollable horizontally if many profiles)
        self.legend_scroll = QScrollArea(self.header_bar)
        self.legend_scroll.setWidgetResizable(True)
        self.legend_scroll.setFrameShape(QFrame.NoFrame)
        self.legend_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.legend_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.legend_scroll.setFixedHeight(26)
        self.legend_scroll.setStyleSheet("background: transparent;")

        self.legend_container = QWidget(self.legend_scroll)
        self.legend_layout = QHBoxLayout(self.legend_container)
        self.legend_layout.setContentsMargins(0, 0, 0, 0)
        self.legend_layout.setSpacing(6)
        self.legend_layout.addStretch(1)
        self.legend_scroll.setWidget(self.legend_container)
        self.legend_scroll.setVisible(False)
        header_layout.addWidget(self.legend_scroll, stretch=1)

        # Multi-2D candidate switcher: ◀ [ComboBox] ▶
        self.switcher_container = QWidget(self.header_bar)
        switcher_layout = QHBoxLayout(self.switcher_container)
        switcher_layout.setContentsMargins(0, 0, 0, 0)
        switcher_layout.setSpacing(4)

        lbl_2d = QLabel("2D Map:", self.switcher_container)
        lbl_2d.setStyleSheet("font-size: 11px; font-weight: bold;")
        switcher_layout.addWidget(lbl_2d)

        self.btn_prev_2d = QPushButton("◀", self.switcher_container)
        self.btn_prev_2d.setFixedSize(22, 22)
        self.btn_prev_2d.setToolTip("Previous selected 2D map")
        self.btn_prev_2d.clicked.connect(lambda: self._step_2d_candidate(-1))
        switcher_layout.addWidget(self.btn_prev_2d)

        self.cbb_2d_map = QComboBox(self.switcher_container)
        self.cbb_2d_map.setMinimumWidth(140)
        self.cbb_2d_map.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.cbb_2d_map.currentTextChanged.connect(self._on_2d_combo_changed)
        switcher_layout.addWidget(self.cbb_2d_map)

        self.btn_next_2d = QPushButton("▶", self.switcher_container)
        self.btn_next_2d.setFixedSize(22, 22)
        self.btn_next_2d.setToolTip("Next selected 2D map")
        self.btn_next_2d.clicked.connect(lambda: self._step_2d_candidate(1))
        switcher_layout.addWidget(self.btn_next_2d)

        self.switcher_container.setVisible(False)
        header_layout.addWidget(self.switcher_container, stretch=0)

        # Exclusion / diagnostic status badge
        self.lbl_context_status = QLabel("", self.header_bar)
        self.lbl_context_status.setStyleSheet("font-size: 11px; color: #b08840;")
        self.lbl_context_status.setVisible(False)
        header_layout.addWidget(self.lbl_context_status, stretch=0)

        self.header_bar.setVisible(False)
        top_layout.addWidget(self.header_bar, stretch=1)

        self._toolbar_spacer = QWidget(self.top_bar)
        self._toolbar_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        top_layout.addWidget(self._toolbar_spacer, stretch=1)

        # ── Persistent View Options toolbar ([Show Legend | More view options ⋯]) ──
        self.options_toolbar = QWidget(self.top_bar)
        opts_layout = QHBoxLayout(self.options_toolbar)
        opts_layout.setContentsMargins(0, 0, 0, 0)
        opts_layout.setSpacing(4)

        tok = current_tokens()
        self.btn_legend = QToolButton(self.options_toolbar)
        self.btn_legend.setObjectName("liveMapsBtnLegend")
        self.btn_legend.setIcon(Icons.legend(color=tok.text_primary))
        self.btn_legend.setIconSize(QSize(16, 16))
        self.btn_legend.setFixedSize(26, 26)
        self.btn_legend.setCheckable(True)
        self.btn_legend.setChecked(self._view_options["show_legend"])
        self.btn_legend.setToolTip(
            "Show / Hide plot legend\n(Double-click a canvas legend box to edit labels)"
        )
        self.btn_legend.toggled.connect(self._on_legend_toggled)
        opts_layout.addWidget(self.btn_legend)

        self.btn_options = QToolButton(self.options_toolbar)
        self.btn_options.setObjectName("liveMapsBtnOptions")
        self.btn_options.setIcon(Icons.settings(color=tok.text_primary))
        self.btn_options.setIconSize(QSize(16, 16))
        self.btn_options.setFixedSize(26, 26)
        self.btn_options.setPopupMode(QToolButton.InstantPopup)
        self.btn_options.setToolTip("More view options")
        self.options_menu = self._create_options_menu()
        self.btn_options.setMenu(self.options_menu)
        opts_layout.addWidget(self.btn_options)

        top_layout.addWidget(self.options_toolbar, stretch=0)
        self.top_bar.setVisible(False)
        main_layout.addWidget(self.top_bar)

        # ── Placeholder when all selectors are off ──
        self.lbl_placeholder = QLabel(self)
        self.lbl_placeholder.setAlignment(Qt.AlignCenter)
        self.lbl_placeholder.setStyleSheet("font-size: 13px; color: #888; padding: 30px;")
        main_layout.addWidget(self.lbl_placeholder)

        # ── Horizontal container for the 3 subpanels ──
        self.maps_container = QWidget(self)
        maps_layout = QHBoxLayout(self.maps_container)
        maps_layout.setContentsMargins(0, 0, 0, 0)
        maps_layout.setSpacing(6)

        for color_id in SELECTOR_KEYS:
            panel = LiveMapSubpanel(color_id, self.maps_container, mode=self.mode)
            panel.parameter_changed.connect(self.parameter_change_requested.emit)
            panel.legend_edit_requested.connect(self.open_legend_editor)
            panel.set_view_options(self._view_options, redraw=False)
            panel.setVisible(False)  # Hidden by default until activated
            self.panels[color_id] = panel
            maps_layout.addWidget(panel, stretch=1)

        main_layout.addWidget(self.maps_container, stretch=1)
        self._update_visibility()

    def _wrap_menu_row(self, parent_menu: QMenu, label: str, widget: QWidget) -> QWidgetAction:
        """Embed a labeled control widget cleanly inside a QMenu row."""
        row_widget = QWidget(parent_menu)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(10, 4, 10, 4)
        row_layout.setSpacing(8)
        row_layout.addWidget(QLabel(label, row_widget))
        row_layout.addWidget(widget, stretch=1)
        action = QWidgetAction(parent_menu)
        action.setDefaultWidget(row_widget)
        return action

    def _create_options_menu(self) -> QMenu:
        """Build 'More view options' menu mirroring VSpectraViewer idioms."""
        menu = QMenu(self)

        # ── Axis scales ──
        self.cbb_xscale = QComboBox(menu)
        self.cbb_xscale.addItems(["Linear", "Log"])
        self.cbb_xscale.currentTextChanged.connect(
            lambda txt: self._update_option("xscale", txt.lower())
        )
        menu.addAction(self._wrap_menu_row(menu, "X-scale:", self.cbb_xscale))

        self.cbb_yscale = QComboBox(menu)
        self.cbb_yscale.addItems(["Linear", "Log"])
        self.cbb_yscale.currentTextChanged.connect(
            lambda txt: self._update_option("yscale", txt.lower())
        )
        menu.addAction(self._wrap_menu_row(menu, "Y-scale:", self.cbb_yscale))

        self.cbb_zscale = QComboBox(menu)
        self.cbb_zscale.addItems(["Linear", "Log"])
        self.cbb_zscale.currentTextChanged.connect(
            lambda txt: self._update_option("zscale", txt.lower())
        )
        menu.addAction(self._wrap_menu_row(menu, "Color (Z) scale:", self.cbb_zscale))

        menu.addSeparator()

        # ── Plot style & sizes ──
        self._style_labels_to_keys = {
            "Line + Dots": "line+dots",
            "Line": "line",
            "Dot": "dot",
        }
        self._style_keys_to_labels = {v: k for k, v in self._style_labels_to_keys.items()}
        self.cbb_plot_style = QComboBox(menu)
        self.cbb_plot_style.addItems(list(self._style_labels_to_keys.keys()))
        self.cbb_plot_style.currentTextChanged.connect(
            lambda txt: self._update_option(
                "plot_style", self._style_labels_to_keys.get(txt, "line+dots")
            )
        )
        menu.addAction(self._wrap_menu_row(menu, "Plot style:", self.cbb_plot_style))

        self.spb_linewidth = QDoubleSpinBox(menu)
        self.spb_linewidth.setRange(0.5, 6.0)
        self.spb_linewidth.setSingleStep(0.2)
        self.spb_linewidth.setDecimals(1)
        self.spb_linewidth.setValue(self._view_options["line_width"])
        self.spb_linewidth.valueChanged.connect(
            lambda val: self._update_option("line_width", float(val))
        )
        menu.addAction(self._wrap_menu_row(menu, "Line width:", self.spb_linewidth))

        self.spb_dotsize = QDoubleSpinBox(menu)
        self.spb_dotsize.setRange(1.0, 14.0)
        self.spb_dotsize.setSingleStep(0.5)
        self.spb_dotsize.setDecimals(1)
        self.spb_dotsize.setValue(self._view_options["dot_size"])
        self.spb_dotsize.valueChanged.connect(
            lambda val: self._update_option("dot_size", float(val))
        )
        menu.addAction(self._wrap_menu_row(menu, "Dot size:", self.spb_dotsize))

        menu.addSeparator()

        # ── Legend options ──
        self.act_show_legend = QAction("Show legend", menu)
        self.act_show_legend.setCheckable(True)
        self.act_show_legend.setChecked(self._view_options["show_legend"])
        self.act_show_legend.toggled.connect(self._on_legend_toggled)
        menu.addAction(self.act_show_legend)

        self.cbb_legend_loc = QComboBox(menu)
        self.cbb_legend_loc.addItems(LEGEND_LOCATIONS)
        self.cbb_legend_loc.setCurrentText(self._view_options["legend_loc"])
        self.cbb_legend_loc.currentTextChanged.connect(
            lambda txt: self._update_option("legend_loc", txt)
        )
        menu.addAction(self._wrap_menu_row(menu, "Legend location:", self.cbb_legend_loc))

        self.spb_legend_ncol = QSpinBox(menu)
        self.spb_legend_ncol.setRange(1, 4)
        self.spb_legend_ncol.setValue(self._view_options["legend_ncol"])
        self.spb_legend_ncol.valueChanged.connect(
            lambda val: self._update_option("legend_ncol", int(val))
        )
        menu.addAction(self._wrap_menu_row(menu, "Legend columns:", self.spb_legend_ncol))

        self.spb_legend_fontsize = QSpinBox(menu)
        self.spb_legend_fontsize.setRange(6, 16)
        self.spb_legend_fontsize.setValue(self._view_options["legend_fontsize"])
        self.spb_legend_fontsize.valueChanged.connect(
            lambda val: self._update_option("legend_fontsize", int(val))
        )
        menu.addAction(self._wrap_menu_row(menu, "Legend font size:", self.spb_legend_fontsize))

        self.act_edit_legend = QAction("Edit legend labels...", menu)
        self.act_edit_legend.triggered.connect(self.open_legend_editor)
        menu.addAction(self.act_edit_legend)

        menu.addSeparator()

        # ── Additional Live Maps aids ──
        self.act_show_grid = QAction("Show grid", menu)
        self.act_show_grid.setCheckable(True)
        self.act_show_grid.setChecked(self._view_options["show_grid"])
        self.act_show_grid.toggled.connect(
            lambda chk: self._update_option("show_grid", bool(chk))
        )
        menu.addAction(self.act_show_grid)

        self.act_share_y = QAction("Share Y-axis (1D)", menu)
        self.act_share_y.setCheckable(True)
        self.act_share_y.setChecked(self._view_options["share_y"])
        self.act_share_y.toggled.connect(
            lambda chk: self._update_option("share_y", bool(chk))
        )
        menu.addAction(self.act_share_y)

        self.act_smoothing = QAction("Heatmap smoothing", menu)
        self.act_smoothing.setCheckable(True)
        self.act_smoothing.setChecked(self._view_options["heatmap_smoothing"])
        self.act_smoothing.toggled.connect(
            lambda chk: self._update_option("heatmap_smoothing", bool(chk))
        )
        menu.addAction(self.act_smoothing)

        self.act_show_colorbar = QAction("Show colorbars", menu)
        self.act_show_colorbar.setCheckable(True)
        self.act_show_colorbar.setChecked(self._view_options["show_colorbar"])
        self.act_show_colorbar.toggled.connect(
            lambda chk: self._update_option("show_colorbar", bool(chk))
        )
        menu.addAction(self.act_show_colorbar)

        menu.addSeparator()

        self.act_reset_options = QAction("Reset view options", menu)
        self.act_reset_options.triggered.connect(self.reset_view_options)
        menu.addAction(self.act_reset_options)

        return menu

    def _on_legend_toggled(self, checked: bool):
        """Keep top toggle button and menu action synchronized on legend toggle."""
        checked = bool(checked)
        self.btn_legend.blockSignals(True)
        self.act_show_legend.blockSignals(True)
        self.btn_legend.setChecked(checked)
        self.act_show_legend.setChecked(checked)
        self.btn_legend.blockSignals(False)
        self.act_show_legend.blockSignals(False)
        self._update_option("show_legend", checked)

    def _update_option(self, key: str, value):
        """Update one view option and apply cleanly to visible panels."""
        if self._view_options.get(key) == value:
            return
        self._view_options[key] = value
        self._apply_view_options_to_panels()

    def _sync_options_controls(self):
        """Synchronize menu widgets and toolbar toggles to self._view_options."""
        opts = self._view_options
        widgets = (
            self.cbb_xscale, self.cbb_yscale, self.cbb_zscale,
            self.cbb_plot_style, self.spb_linewidth, self.spb_dotsize,
            self.btn_legend, self.act_show_legend,
            self.cbb_legend_loc, self.spb_legend_ncol, self.spb_legend_fontsize,
            self.act_show_grid, self.act_share_y, self.act_smoothing, self.act_show_colorbar,
        )
        for w in widgets:
            w.blockSignals(True)
        try:
            self.cbb_xscale.setCurrentText("Log" if opts["xscale"] == "log" else "Linear")
            self.cbb_yscale.setCurrentText("Log" if opts["yscale"] == "log" else "Linear")
            self.cbb_zscale.setCurrentText("Log" if opts["zscale"] == "log" else "Linear")
            self.cbb_plot_style.setCurrentText(
                self._style_keys_to_labels.get(opts["plot_style"], "Line + Dots")
            )
            self.spb_linewidth.setValue(float(opts["line_width"]))
            self.spb_dotsize.setValue(float(opts["dot_size"]))
            self.btn_legend.setChecked(bool(opts["show_legend"]))
            self.act_show_legend.setChecked(bool(opts["show_legend"]))
            loc = opts["legend_loc"]
            if self.cbb_legend_loc.findText(loc) >= 0:
                self.cbb_legend_loc.setCurrentText(loc)
            self.spb_legend_ncol.setValue(int(opts["legend_ncol"]))
            self.spb_legend_fontsize.setValue(int(opts["legend_fontsize"]))
            self.act_show_grid.setChecked(bool(opts["show_grid"]))
            self.act_share_y.setChecked(bool(opts["share_y"]))
            self.act_smoothing.setChecked(bool(opts["heatmap_smoothing"]))
            self.act_show_colorbar.setChecked(bool(opts["show_colorbar"]))
        finally:
            for w in widgets:
                w.blockSignals(False)

    def get_view_options(self) -> dict:
        """Return a deep copy of current Live Maps view options."""
        return _copy_view_options(self._view_options)

    def set_view_options(self, options: dict):
        """Apply a partial or full view options dictionary and refresh visible panels."""
        self._view_options = _copy_view_options({**self._view_options, **(options or {})})
        self._sync_options_controls()
        if self._last_raw_entries:
            self._rebuild_legend_chips(self._last_raw_entries, force=True)
        self._apply_view_options_to_panels()

    def reset_view_options(self):
        """Reset all view options and custom legend labels/colors to defaults."""
        self._view_options = _copy_view_options()
        self._sync_options_controls()
        if self._last_raw_entries:
            self._rebuild_legend_chips(self._last_raw_entries, force=True)
        self._apply_view_options_to_panels()

    def _apply_view_options_to_panels(self):
        """Push view options into all subpanels and re-render visible populated ones."""
        for panel in self.panels.values():
            panel.set_view_options(self._view_options, redraw=False)
            if panel.isVisible() and panel._display_kind is not None:
                panel.rerender_cached(theme=self._current_theme)
        self._sync_shared_y()

    def _legend_editor_entries(self) -> List[dict]:
        """Collect editable curve/series entries from active 1D subpanel contents."""
        if self._last_raw_entries and len(self._last_raw_entries) >= 2:
            return [
                {
                    "key": str(e.get("name", f"Map {idx + 1}")),
                    "default_label": str(e.get("name", f"Map {idx + 1}")),
                    "auto_color": DEFAULT_COLORS[idx % len(DEFAULT_COLORS)],
                }
                for idx, e in enumerate(self._last_raw_entries)
            ]
        entries = []
        for cid in SELECTOR_KEYS:
            panel = self.panels.get(cid)
            if panel is not None and self._active_states.get(cid, False):
                param = panel.get_selected_parameter()
                entries.append({
                    "key": cid,
                    "default_label": f"{panel.color_name}: {param}",
                    "auto_color": panel.color_hex,
                })
        if not entries:
            for cid in SELECTOR_KEYS:
                panel = self.panels[cid]
                entries.append({
                    "key": cid,
                    "default_label": f"{panel.color_name}: {panel.get_selected_parameter()}",
                    "auto_color": panel.color_hex,
                })
        return entries

    def open_legend_editor(self):
        """Open LiveMapsLegendEditorDialog to customize legend placement, labels, and colors."""
        dlg = LiveMapsLegendEditorDialog(
            entries=self._legend_editor_entries(),
            custom_labels=self._view_options.get("custom_labels"),
            custom_colors=self._view_options.get("custom_colors"),
            legend_loc=self._view_options.get("legend_loc", "best"),
            legend_ncol=self._view_options.get("legend_ncol", 1),
            legend_fontsize=self._view_options.get("legend_fontsize", 8),
            parent=self,
        )
        if dlg.exec():
            res = dlg.result_state()
            self.set_view_options({
                **res,
                "show_legend": True,
            })

    def _sync_shared_y(self):
        """Synchronize Y-axis limits across visible 1D subpanels when Share Y-axis is enabled."""
        if not self._view_options.get("share_y", False):
            return
        use_log = self._view_options.get("yscale", "linear") == "log"
        target_panels = []
        bounds = []
        for cid in SELECTOR_KEYS:
            panel = self.panels.get(cid)
            if (
                panel is not None
                and panel.isVisible()
                and panel._display_kind in ("series", "multi_profile")
            ):
                b = panel._valid_pos_y_bounds if use_log else panel._valid_y_bounds
                if b is not None and panel.ax.get_yscale() == ("log" if use_log else "linear"):
                    target_panels.append(panel)
                    bounds.append(b)
        if len(target_panels) < 2:
            return
        g_lo = min(b[0] for b in bounds)
        g_hi = max(b[1] for b in bounds)
        if not (np.isfinite(g_lo) and np.isfinite(g_hi)) or g_hi <= g_lo:
            return
        for panel in target_panels:
            panel.ax.set_ylim(g_lo, g_hi)
            panel.canvas.draw_idle()

    def set_mode(self, mode: str):
        """Configure tab for 'map' (2D spatial heatmaps) or 'spectra' (1D variation graphs) mode."""
        self.mode = mode
        if mode == "spectra":
            self.lbl_placeholder.setText(
                "Live Spectral Mapping\n\n"
                "Activate one or more spectral range selectors (Red, Blue, Green)\n"
                "in the Spectra viewer right panel to display live parameter variation graphs."
            )
        else:
            self.lbl_placeholder.setText(
                "Live Spectral Mapping\n\n"
                "Activate one or more spectral range selectors (Red, Blue, Green)\n"
                "in the Spectra viewer right panel to display live 2D heatmaps\n"
                "(or line plots for 1D profile maps)."
            )
        for panel in self.panels.values():
            panel.set_mode(mode)

    def set_selector_active(self, color_id: str, is_active: bool):
        """Update visibility of a subpanel based on selector state."""
        self._active_states[color_id] = is_active
        panel = self.panels.get(color_id)
        if panel:
            panel.setVisible(is_active)
        self._update_visibility()

    def set_selector_range(self, color_id: str, x_min: float, x_max: float):
        """Update range label on a subpanel."""
        self._ranges[color_id] = (x_min, x_max)
        panel = self.panels.get(color_id)
        if panel:
            panel.set_range_text(x_min, x_max)

    def _update_visibility(self):
        """Show or hide top bar, container, and placeholder based on active selectors."""
        any_active = any(self._active_states.values())
        self.lbl_placeholder.setVisible(not any_active)
        self.maps_container.setVisible(any_active)
        self._update_header_visibility(any_active)

    def _update_header_visibility(self, any_active: Optional[bool] = None):
        """Show top options toolbar whenever selectors are active; show context header only when needed."""
        if any_active is None:
            any_active = any(self._active_states.values())
        has_legend = len(self._legend_entries) >= 2
        has_switcher = len(self._2d_candidates) >= 2
        has_status = bool(self._context_status)
        show_context_header = any_active and (has_legend or has_switcher or has_status)
        self.legend_scroll.setVisible(has_legend)
        self.switcher_container.setVisible(has_switcher)
        self.lbl_context_status.setVisible(has_status)
        self.header_bar.setVisible(show_context_header)
        self._toolbar_spacer.setVisible(any_active and not show_context_header)
        self.top_bar.setVisible(any_active)

    def _rebuild_legend_chips(self, entries: List[dict], force: bool = False):
        """Populate compact colored legend chips matching multi-1D profile colors."""
        self._last_raw_entries = list(entries) if entries else []
        custom_labels = self._view_options.get("custom_labels", {})
        custom_colors = self._view_options.get("custom_colors", {})
        pairs = [
            (
                custom_labels.get(str(e.get("name", f"Map {i + 1}")))
                or str(e.get("name", f"Map {i + 1}")),
                bool(e.get("is_primary", False)),
                custom_colors.get(str(e.get("name", f"Map {i + 1}")))
                or DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
            )
            for i, e in enumerate(self._last_raw_entries)
        ]
        if not force and pairs == self._legend_entries:
            return
        self._legend_entries = pairs

        while self.legend_layout.count() > 0:
            item = self.legend_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if len(pairs) >= 2:
            for display_name, is_primary, color in pairs:
                chip = QLabel(self.legend_container)
                suffix = " (Primary)" if is_primary else ""
                chip.setText(f"<span style='color:{color}; font-size:13px;'>●</span> {display_name}{suffix}")
                weight = "bold" if is_primary else "normal"
                chip.setStyleSheet(
                    f"font-size: 11px; font-weight: {weight}; "
                    f"padding: 1px 6px; border-radius: 3px; "
                    f"background: rgba(128, 128, 128, 0.10);"
                )
                self.legend_layout.addWidget(chip)
        self.legend_layout.addStretch(1)

    def _on_2d_combo_changed(self, name: str):
        if name:
            self.active_2d_map_requested.emit(name)
            self._sync_2d_nav_buttons()

    def _step_2d_candidate(self, delta: int):
        idx = self.cbb_2d_map.currentIndex() + delta
        if 0 <= idx < self.cbb_2d_map.count():
            self.cbb_2d_map.setCurrentIndex(idx)

    def _sync_2d_nav_buttons(self):
        count = self.cbb_2d_map.count()
        idx = self.cbb_2d_map.currentIndex()
        self.btn_prev_2d.setEnabled(count > 1 and idx > 0)
        self.btn_next_2d.setEnabled(count > 1 and 0 <= idx < count - 1)

    def update_2d_candidates(
        self,
        candidates: List[str],
        active_name: Optional[str],
        status_msg: str = "",
    ):
        """Update the 2D switcher combo box and exclusion/diagnostic badge."""
        self._2d_candidates = list(candidates) if candidates else []
        self._context_status = status_msg or ""
        self.lbl_context_status.setText(self._context_status)

        self.cbb_2d_map.blockSignals(True)
        try:
            self.cbb_2d_map.clear()
            for name in self._2d_candidates:
                self.cbb_2d_map.addItem(name)
            if active_name:
                idx = self.cbb_2d_map.findText(active_name)
                if idx >= 0:
                    self.cbb_2d_map.setCurrentIndex(idx)
        finally:
            self.cbb_2d_map.blockSignals(False)

        self._sync_2d_nav_buttons()
        self._update_header_visibility()

    def update_map_data(
        self,
        color_id: str,
        grid: Optional[np.ndarray],
        extent: list[float],
        vmin: float,
        vmax: float,
        status_msg: str = "",
    ):
        """Forward calculated 2D grid data to matching subpanel."""
        if self._legend_entries:
            self._rebuild_legend_chips([])
            self._update_header_visibility()
        panel = self.panels.get(color_id)
        if panel:
            panel.update_map(grid, extent, vmin, vmax, status_msg, theme=self._current_theme)

    def update_series_data(
        self,
        color_id: str,
        values: Optional[np.ndarray],
        labels: list[str],
        status_msg: str = "",
    ):
        """Forward calculated 1D parameter series data to matching subpanel."""
        if self._legend_entries:
            self._rebuild_legend_chips([])
            self._update_header_visibility()
        panel = self.panels.get(color_id)
        if panel:
            panel.update_series(values, labels, status_msg, theme=self._current_theme)
        self._sync_shared_y()

    def update_profile_data(
        self,
        color_id: str,
        positions: Optional[np.ndarray],
        values: Optional[np.ndarray],
        axis_label: str = "X",
        status_msg: str = "",
    ):
        """Forward a single 1D profile (values along the profile axis) to the matching subpanel."""
        if self._legend_entries:
            self._rebuild_legend_chips([])
            self._update_header_visibility()
        panel = self.panels.get(color_id)
        if panel:
            panel.update_series(
                values, None, status_msg, theme=self._current_theme,
                x_values=positions if positions is not None else np.array([]),
                x_label=f"{axis_label or 'Position'} (mm)",
            )
        self._sync_shared_y()

    def update_multi_profile_data(
        self,
        color_id: str,
        entries: List[dict],
        shared_x_label: str = "X",
        status_msg: str = "",
    ):
        """Forward stacked multi-1D profile entries and refresh shared color legend."""
        self._rebuild_legend_chips(entries)
        if status_msg and status_msg != "OK":
            self._context_status = status_msg
            self.lbl_context_status.setText(self._context_status)
        self._update_header_visibility()
        panel = self.panels.get(color_id)
        if panel:
            panel.update_multi_profile(
                entries, shared_x_label=shared_x_label, status_msg=status_msg,
                theme=self._current_theme,
            )
        self._sync_shared_y()

    def clear_all(self):
        """Clear all subpanel artists, legend chips, 2D candidate switcher, and status."""
        self._rebuild_legend_chips([])
        self._last_raw_entries = []
        self._2d_candidates = []
        self._context_status = ""
        self.lbl_context_status.setText("")
        self.cbb_2d_map.blockSignals(True)
        self.cbb_2d_map.clear()
        self.cbb_2d_map.blockSignals(False)
        self._sync_2d_nav_buttons()
        self._update_header_visibility()
        for panel in self.panels.values():
            panel.clear_plot()

    def apply_theme(self, theme_name: str):
        """Update theme for toolbar buttons and all subpanels."""
        self._current_theme = theme_name
        tok = current_tokens()
        icon_color = tok.text_primary
        self.btn_legend.setIcon(Icons.legend(color=icon_color))
        self.btn_options.setIcon(Icons.settings(color=icon_color))
        for panel in self.panels.values():
            panel.btn_copy.setIcon(Icons.copy(color=icon_color))
            panel.set_view_options(self._view_options, redraw=False)
            panel.img = None
            panel.lines.clear()
            if panel.isVisible():
                panel.rerender_cached(theme=theme_name)
        self._sync_shared_y()
