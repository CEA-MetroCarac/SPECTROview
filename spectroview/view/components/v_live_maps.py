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
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, AutoLocator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QFrame, QSizePolicy, QScrollArea,
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


DEFAULT_PALETTES = {
    SELECTOR_RED: "Reds",
    SELECTOR_BLUE: "Blues",
    SELECTOR_GREEN: "Greens",
}

AVAILABLE_PALETTES = [
    "viridis", "plasma", "inferno", "magma", "cividis", "turbo",
    "Reds", "Blues", "Greens", "coolwarm", "bwr", "jet", "rainbow"
]


class LiveMapSubpanel(QFrame):
    """Subpanel rendering a 2D heatmap or a 1D line plot for one spectral range.

    The plot type follows the data received, not a fixed mode:
    ``update_map`` draws a 2D heatmap (2D maps), ``update_series`` draws a
    line plot — either vs. spectrum index (Spectra workspace) or vs. spatial
    position (1D profile maps, when ``x_values`` is given).
    """

    parameter_changed = Signal(str, str)  # (color_id, parameter_name)

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
        layout.addWidget(self.canvas, stretch=1)

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
        """Colormap selector only applies to heatmaps (Maps mode, not a 1D line plot)."""
        self.cbb_palette.setVisible(self.mode == "map" and self._display_kind != "series")

    def _reset_axes(self):
        """Remove any colorbar, clear lines, clear axes, and restore automatic aspect scaling.

        Matplotlib's ``Axes.clear()`` does not reset the aspect ratio set by
        ``imshow(..., aspect="equal")``; without resetting to ``"auto"``, a
        subsequent 1D line plot inherits a 1:1 data aspect and collapses into a
        thin vertical or horizontal strip when X and Y spans differ.
        """
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

    def _apply_x_limits(self, x: np.ndarray, is_index_axis: bool):
        """Set x-limits/ticks for an index axis (Spectrum #) or a spatial profile axis."""
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
            if not is_index_axis:
                self._apply_x_limits(x_arr, is_index_axis=False)
            self.ax.set_ylim(*self._padded_ylim(val_arr))
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel(param_name)
            self.ax.set_title(title_text, fontsize=10, pad=4)
            self.canvas.draw_idle()
            return

        is_dark = theme in ("Dark Mode", "dark", "classic_dark")
        style_path = PLOT_POLICY_DARK if is_dark else PLOT_POLICY_LIGHT

        self._display_kind = "series"
        self._sync_palette_visibility()

        with plt.style.context(style_path):
            self._reset_axes()

            if not has_valid:
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

            n = len(val_arr)
            self.lines = list(
                self.ax.plot(
                    x_arr,
                    val_arr,
                    color=self.color_hex,
                    marker="o",
                    markersize=5 if n <= 30 else 3,
                    linewidth=1.8,
                    alpha=0.95,
                )
            )

            self._apply_x_limits(x_arr, is_index_axis)
            self.ax.set_ylim(*self._padded_ylim(val_arr))

            self.ax.grid(True, alpha=0.25, linestyle="--")
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel(param_name)
            self.ax.set_title(title_text, fontsize=10, pad=4)

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
            y_lo, y_hi = self._padded_ylim(all_y)
            x_min = float(np.min(all_x))
            x_max = float(np.max(all_x))
            x_span = x_max - x_min
            x_margin = x_span * 0.03 if x_span > 1e-12 else 0.5
        else:
            y_lo, y_hi = 0.0, 1.0
            x_min, x_max, x_margin = 0.0, 1.0, 0.0

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
                color = (
                    self.color_hex
                    if len(self._multi_entries) == 1
                    else DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
                )
                is_primary = bool(entry.get("is_primary", False))
                ln.set_data(px, vy)
                ln.set_color(color)
                ln.set_linewidth(2.2 if is_primary else 1.4)
                ln.set_alpha(1.0 if is_primary else 0.78)
                ln.set_zorder(5 if is_primary else 3)
            self.ax.set_xlim(x_min - x_margin, x_max + x_margin)
            self.ax.set_ylim(y_lo, y_hi)
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel(param_name)
            self.ax.set_title(title_text, fontsize=10, pad=4)
            self.canvas.draw_idle()
            return

        is_dark = theme in ("Dark Mode", "dark", "classic_dark")
        style_path = PLOT_POLICY_DARK if is_dark else PLOT_POLICY_LIGHT

        self._display_kind = "multi_profile"
        self._sync_palette_visibility()

        with plt.style.context(style_path):
            self._reset_axes()

            if not has_valid:
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
                color = (
                    self.color_hex
                    if len(self._multi_entries) == 1
                    else DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
                )
                is_primary = bool(entry.get("is_primary", False))
                n_pts = len(px)
                marker = "o" if n_pts <= 30 else ""
                created = self.ax.plot(
                    px,
                    vy,
                    color=color,
                    marker=marker,
                    markersize=4 if n_pts <= 30 else 0,
                    linewidth=2.2 if is_primary else 1.4,
                    alpha=1.0 if is_primary else 0.78,
                    zorder=5 if is_primary else 3,
                    label=entry.get("name", f"Map {idx + 1}"),
                )
                new_lines.extend(created)
            self.lines = new_lines

            self.ax.set_xlim(x_min - x_margin, x_max + x_margin)
            self.ax.xaxis.set_major_locator(AutoLocator())
            self.ax.set_ylim(y_lo, y_hi)
            self.ax.grid(True, alpha=0.25, linestyle="--")
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel(param_name)
            self.ax.set_title(title_text, fontsize=10, pad=4)

            self.canvas.draw_idle()

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
        self._status_msg = status_msg
        self._current_theme = theme

        param_name = self.cbb_param.currentText()
        title_text = f"{self.color_name}: {param_name}"

        # Fast path: in-place update if image and colorbar already exist and grid is valid
        if (
            self._display_kind == "map"
            and self.img is not None
            and self.cbar is not None
            and grid is not None
            and not np.all(np.isnan(grid))
        ):
            self.img.set_data(grid)
            self.img.set_extent(extent)
            self.img.set_clim(vmin, vmax)
            self.ax.set_title(title_text, fontsize=10, pad=4)
            self.cbar.update_normal(self.img)
            self.canvas.draw_idle()
            return

        # Re-render path for status messages or first draw
        is_dark = theme in ("Dark Mode", "dark", "classic_dark")
        style_path = PLOT_POLICY_DARK if is_dark else PLOT_POLICY_LIGHT

        self._display_kind = "map"
        self._sync_palette_visibility()

        with plt.style.context(style_path):
            self._reset_axes()

            if grid is None or np.all(np.isnan(grid)):
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

            self.img = self.ax.imshow(
                grid,
                extent=extent,
                origin="lower",
                aspect="equal",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="none",
            )

            self.cbar = self.figure.colorbar(self.img, ax=self.ax, fraction=0.046, pad=0.04)

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
        self._context_status: str = ""

        self._current_theme = "Light Mode"
        self._init_ui()
        self.set_mode(mode)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # ── Slim shared context header (multi-1D legend, 2D switcher, status) ──
        self.header_bar = QFrame(self)
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
        main_layout.addWidget(self.header_bar)

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
            panel.setVisible(False)  # Hidden by default until activated
            self.panels[color_id] = panel
            maps_layout.addWidget(panel, stretch=1)

        main_layout.addWidget(self.maps_container, stretch=1)
        self._update_visibility()

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
        """Show or hide header, container, and placeholder based on active selectors."""
        any_active = any(self._active_states.values())
        self.lbl_placeholder.setVisible(not any_active)
        self.maps_container.setVisible(any_active)
        self._update_header_visibility(any_active)

    def _update_header_visibility(self, any_active: Optional[bool] = None):
        """Show header bar only when selectors are active AND contextual controls/notes exist."""
        if any_active is None:
            any_active = any(self._active_states.values())
        has_legend = len(self._legend_entries) >= 2
        has_switcher = len(self._2d_candidates) >= 2
        has_status = bool(self._context_status)
        self.legend_scroll.setVisible(has_legend)
        self.switcher_container.setVisible(has_switcher)
        self.lbl_context_status.setVisible(has_status)
        self.header_bar.setVisible(any_active and (has_legend or has_switcher or has_status))

    def _rebuild_legend_chips(self, entries: List[dict]):
        """Populate compact colored legend chips matching multi-1D profile colors."""
        pairs = [(e.get("name", f"Map {i + 1}"), bool(e.get("is_primary", False))) for i, e in enumerate(entries)]
        if pairs == self._legend_entries:
            return
        self._legend_entries = pairs

        while self.legend_layout.count() > 0:
            item = self.legend_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if len(pairs) >= 2:
            for idx, (name, is_primary) in enumerate(pairs):
                color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
                chip = QLabel(self.legend_container)
                suffix = " (Primary)" if is_primary else ""
                chip.setText(f"<span style='color:{color}; font-size:13px;'>●</span> {name}{suffix}")
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

    def clear_all(self):
        """Clear all subpanel artists, legend chips, 2D candidate switcher, and status."""
        self._rebuild_legend_chips([])
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
        """Update theme for all subpanels."""
        self._current_theme = theme_name
        tok = current_tokens()
        icon_color = tok.text_primary
        for panel in self.panels.values():
            panel.btn_copy.setIcon(Icons.copy(color=icon_color))
            panel.img = None
            panel.lines.clear()
            if panel.isVisible():
                kind = panel._display_kind or ("series" if self.mode == "spectra" else "map")
                if kind == "multi_profile":
                    panel.update_multi_profile(
                        panel._multi_entries,
                        shared_x_label=panel._multi_x_label,
                        status_msg=panel._status_msg,
                        theme=theme_name,
                    )
                elif kind == "series":
                    panel.update_series(
                        panel._series_values,
                        panel._series_labels,
                        panel._status_msg,
                        theme=theme_name,
                        x_values=panel._series_x,
                        x_label=panel._series_xlabel,
                    )
                else:
                    panel.update_map(
                        panel._grid, panel._extent, panel._vmin, panel._vmax,
                        panel._status_msg, theme=theme_name
                    )
