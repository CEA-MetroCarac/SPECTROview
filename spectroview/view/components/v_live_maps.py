"""Live Maps Tab View component for SPECTROview.

Displays 3 live-updating panels corresponding to Red, Blue, and Green
spectral range selectors:
- In Maps workspace ('map' mode): renders 2D spatial heatmaps with colormap
  selection for 2D maps, or line plots along the profile for 1D maps (profiles).
- In Spectra workspace ('spectra' mode): renders 1D parameter variation graphs across selected spectra.

The Green panel can alternatively display the Red/Blue intensity ratio (R/B).
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, AutoLocator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QFrame, QSizePolicy
)

from spectroview import PLOT_POLICY_LIGHT, PLOT_POLICY_DARK
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

        # What is currently drawn: None, "map" (heatmap) or "series" (line plot)
        self._display_kind: Optional[str] = None

        self._status_msg = ""
        self._x_range = (0.0, 100.0)
        self._current_theme = "Light Mode"

        # Artist references
        self.img = None
        self.cbar = None
        self.line = None

        self._init_ui()
        self.set_mode(mode)

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
        """Remove any colorbar, clear the axes, and restore automatic aspect scaling.

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
        self.line = None
        self.ax.clear()
        self.ax.set_aspect("auto", adjustable="box")

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
        self._status_msg = status_msg
        self._current_theme = theme

        param_name = self.cbb_param.currentText()
        title_text = f"{self.color_name}: {param_name}"

        has_valid = len(val_arr) > 0 and bool(np.any(np.isfinite(val_arr)))

        # Fast path: in-place line update during continuous bracket dragging
        if (
            self.line is not None
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
            (self.line,) = self.ax.plot(
                x_arr,
                val_arr,
                color=self.color_hex,
                marker="o",
                markersize=5 if n <= 30 else 3,
                linewidth=1.8,
                alpha=0.95,
            )

            self._apply_x_limits(x_arr, is_index_axis)
            self.ax.set_ylim(*self._padded_ylim(val_arr))

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
        self._status_msg = status_msg
        self._current_theme = theme

        param_name = self.cbb_param.currentText()
        title_text = f"{self.color_name}: {param_name}"

        # Fast path: in-place update if image and colorbar already exist and grid is valid
        if (
            self.img is not None
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

        self._current_theme = "Light Mode"
        self._init_ui()
        self.set_mode(mode)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

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
        """Show or hide container and placeholder based on active count."""
        any_active = any(self._active_states.values())
        self.lbl_placeholder.setVisible(not any_active)
        self.maps_container.setVisible(any_active)

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
        """Forward a 1D profile (values along the profile axis) to the matching subpanel."""
        panel = self.panels.get(color_id)
        if panel:
            panel.update_series(
                values, None, status_msg, theme=self._current_theme,
                x_values=positions if positions is not None else np.array([]),
                x_label=f"{axis_label or 'Position'} (mm)",
            )

    def apply_theme(self, theme_name: str):
        """Update theme for all subpanels."""
        self._current_theme = theme_name
        tok = current_tokens()
        icon_color = tok.text_primary
        for panel in self.panels.values():
            panel.btn_copy.setIcon(Icons.copy(color=icon_color))
            panel.img = None
            panel.line = None
            if panel.isVisible():
                kind = panel._display_kind or ("series" if self.mode == "spectra" else "map")
                if kind == "series":
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
