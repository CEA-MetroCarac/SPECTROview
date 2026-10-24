"""Live Maps Tab View component for SPECTROview.

Displays 3 live-updating 2D heatmaps corresponding to Red, Blue, and Green
spectral range selectors, with parameter selection and colormap configuration.
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QFrame, QSizePolicy
)

from spectroview import ICON_DIR, PLOT_POLICY_LIGHT, PLOT_POLICY_DARK
from spectroview.viewmodel.utils import copy_fig_to_clb
from spectroview.model.spectral_parameters import (
    SPECTRAL_PARAMETERS,
    PARAM_INTEGRATED_INTENSITY,
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
    """Subpanel rendering a single 2D heatmap for one spectral range selector."""

    parameter_changed = Signal(str, str)  # (color_id, parameter_name)

    def __init__(self, color_id: str, parent=None):
        super().__init__(parent)
        self.color_id = color_id
        self.color_hex = SELECTOR_COLORS[color_id]
        self.color_name = SELECTOR_NAMES[color_id]

        self.setObjectName(f"liveMapPanel_{color_id}")
        self.setFrameShape(QFrame.StyledPanel)

        # Cached grid data
        self._grid = None
        self._extent = [0.0, 1.0, 0.0, 1.0]
        self._vmin = 0.0
        self._vmax = 1.0
        self._status_msg = ""
        self._x_range = (0.0, 100.0)
        self._current_theme = "Light Mode"

        # Image and colorbar references
        self.img = None
        self.cbar = None

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Header Bar ──
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(2, 2, 2, 2)
        header_layout.setSpacing(6)

        # Color badge
        self.lbl_badge = QLabel(self)
        self.lbl_badge.setText(f" {self.color_name} ")
        self.lbl_badge.setStyleSheet(f"""
            QLabel {{
                background-color: {self.color_hex};
                color: white;
                font-weight: bold;
                border-radius: 3px;
                padding: 2px 4px;
            }}
        """)
        header_layout.addWidget(self.lbl_badge)

        # Range label
        self.lbl_range = QLabel("[0.0, 0.0]", self)
        self.lbl_range.setStyleSheet("font-size: 11px; color: #666;")
        header_layout.addWidget(self.lbl_range)

        header_layout.addStretch()

        # Parameter selector
        self.cbb_param = QComboBox(self)
        for param in SPECTRAL_PARAMETERS:
            self.cbb_param.addItem(param)
        self.cbb_param.setCurrentText(PARAM_INTEGRATED_INTENSITY)
        self.cbb_param.setToolTip("Select spectral parameter for this map")
        self.cbb_param.currentTextChanged.connect(
            lambda param: self.parameter_changed.emit(self.color_id, param)
        )
        header_layout.addWidget(self.cbb_param)

        # Colormap selector
        self.cbb_palette = QComboBox(self)
        self.cbb_palette.addItems(AVAILABLE_PALETTES)
        default_pal = DEFAULT_PALETTES.get(self.color_id, "viridis")
        self.cbb_palette.setCurrentText(default_pal)
        self.cbb_palette.setToolTip("Select colormap")
        self.cbb_palette.currentTextChanged.connect(self._on_palette_changed)
        header_layout.addWidget(self.cbb_palette)

        # Copy button
        self.btn_copy = QPushButton(self)
        self.btn_copy.setIcon(QIcon(f"{ICON_DIR}/copy.png"))
        self.btn_copy.setIconSize(QSize(16, 16))
        self.btn_copy.setFixedSize(24, 24)
        self.btn_copy.setToolTip("Copy map to clipboard")
        self.btn_copy.clicked.connect(self._copy_to_clipboard)
        header_layout.addWidget(self.btn_copy)

        layout.addLayout(header_layout)

        # ── Matplotlib Figure & Canvas ──
        with plt.style.context(PLOT_POLICY_LIGHT):
            self.figure = Figure(dpi=70, layout="compressed")
            self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas, stretch=1)

    def _copy_to_clipboard(self):
        """Copy heatmap figure to system clipboard."""
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
        self.lbl_range.setText(f"[{x_min:.1f}, {x_max:.1f}]")

    def get_selected_parameter(self) -> str:
        """Return currently selected parameter name."""
        return self.cbb_param.currentText()

    def set_selected_parameter(self, parameter: str):
        """Select a parameter in the dropdown."""
        idx = self.cbb_param.findText(parameter)
        if idx >= 0:
            self.cbb_param.setCurrentIndex(idx)

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

        style_path = PLOT_POLICY_DARK if theme == "Dark Mode" else PLOT_POLICY_LIGHT

        with plt.style.context(style_path):
            self.ax.clear()

            if self.cbar is not None:
                try:
                    self.cbar.remove()
                except Exception:
                    pass
                self.cbar = None

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
            param_name = self.cbb_param.currentText()
            self.ax.set_title(f"{self.color_name}: {param_name}", fontsize=10, pad=4)

            self.canvas.draw_idle()


class VLiveMaps(QWidget):
    """Live Maps Tab View coordinating the 3 spectral range heatmaps."""

    parameter_change_requested = Signal(str, str)  # (color_id, parameter)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LiveMapsTab")

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

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # ── Placeholder when all selectors are off ──
        self.lbl_placeholder = QLabel(
            "🗺️ Live Spectral Mapping\n\n"
            "Activate one or more spectral range selectors (Red, Blue, Green)\n"
            "in the Spectra viewer right panel to display live 2D heatmaps.",
            self
        )
        self.lbl_placeholder.setAlignment(Qt.AlignCenter)
        self.lbl_placeholder.setStyleSheet("font-size: 13px; color: #888; padding: 30px;")
        main_layout.addWidget(self.lbl_placeholder)

        # ── Horizontal container for the 3 subpanels ──
        self.maps_container = QWidget(self)
        maps_layout = QHBoxLayout(self.maps_container)
        maps_layout.setContentsMargins(0, 0, 0, 0)
        maps_layout.setSpacing(6)

        for color_id in SELECTOR_KEYS:
            panel = LiveMapSubpanel(color_id, self.maps_container)
            panel.parameter_changed.connect(self.parameter_change_requested.emit)
            panel.setVisible(False)  # Hidden by default until activated
            self.panels[color_id] = panel
            maps_layout.addWidget(panel, stretch=1)

        main_layout.addWidget(self.maps_container, stretch=1)
        self._update_visibility()

    def set_selector_active(self, color_id: str, is_active: bool):
        """Update visibility of a map panel based on selector state."""
        self._active_states[color_id] = is_active
        panel = self.panels.get(color_id)
        if panel:
            panel.setVisible(is_active)
        self._update_visibility()

    def set_selector_range(self, color_id: str, x_min: float, x_max: float):
        """Update range label on a map panel."""
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
        """Forward calculated grid data to matching subpanel."""
        panel = self.panels.get(color_id)
        if panel:
            panel.update_map(grid, extent, vmin, vmax, status_msg, theme=self._current_theme)

    def apply_theme(self, theme_name: str):
        """Update theme for all subpanels."""
        self._current_theme = theme_name
        for panel in self.panels.values():
            if panel.isVisible():
                panel.update_map(
                    panel._grid, panel._extent, panel._vmin, panel._vmax,
                    panel._status_msg, theme=theme_name
                )
