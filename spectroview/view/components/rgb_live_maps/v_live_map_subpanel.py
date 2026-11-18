"""Subpanel widget rendering a 2D heatmap or 1D line plot for one RGB Live Maps channel."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import AutoLocator, MaxNLocator
import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
)

from spectroview import DEFAULT_COLORS
from spectroview.model.spectral_parameters import (
    PARAM_INTEGRATED_INTENSITY,
    PARAM_RB_RATIO,
    SELECTOR_COLORS,
    SELECTOR_GREEN,
    SELECTOR_NAMES,
    SPECTRAL_PARAMETERS,
)
from spectroview.view.components.customized_widgets import NoDoubleClickZoomToolbar
from spectroview.view.components.map_viewer.map_selection import (
    MapCanvasSelector,
    compute_2d_spatial_geometry,
    get_drag_box_edgecolor,
    is_ctrl_event,
    resolve_1d_profile_selection,
    resolve_xy_selection,
)
from spectroview.view.components.rgb_live_maps.common import (
    AVAILABLE_PALETTES,
    DEFAULT_PALETTES,
    _copy_view_options,
    _get_cached_style_dict,
    _normalize_1d_values,
    _normalize_plot_theme,
    _parse_norm_bound,
    _resolve_style_path,
)
from spectroview.view.components.rgb_live_maps.v_live_maps_legend_editor import (
    LEGEND_LOCATIONS,
)
from spectroview.view.theme import Icons, current_tokens
from spectroview.viewmodel.utils import copy_df_to_clb, copy_fig_to_clb


class LiveMapSubpanel(QFrame):
    """Subpanel rendering a 2D heatmap or a 1D line plot for one spectral range.

    The plot type follows the data received, not a fixed mode:
    ``update_map`` draws a 2D heatmap (2D maps), ``update_series`` draws a
    line plot — either vs. spectrum index (Spectra workspace) or vs. spatial
    position (1D profile maps, when ``x_values`` is given).
    """

    parameter_changed = Signal(str, str)  # (color_id, parameter_name)
    legend_edit_requested = Signal()
    view_reset_requested = Signal()
    normalization_changed = Signal()      # 1D Y normalization toggled or range edited
    spectra_selected = Signal(list)       # List of (x, y) tuples selected interactively
    copy_data_requested = Signal(str)     # color_id when user Ctrl+Clicks Copy button

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
        self._legend_signature = None
        self._view_options = _copy_view_options()
        self._valid_y_bounds: Optional[Tuple[float, float]] = None
        self._valid_pos_y_bounds: Optional[Tuple[float, float]] = None
        self._relayout_needed: bool = True
        self._spatial_sites: Optional[dict] = None
        self._2d_data_provider: Optional[Callable[[str], Optional[tuple]]] = None
        self._selected_points: list[tuple[float, float]] = []
        self._selector: Optional[MapCanvasSelector] = None
        self._last_norm_params: tuple[bool, Optional[float], Optional[float]] = (False, None, None)
        self._styled_theme_sig = None
        self._styled_text_fs_sig = None
        self._styled_grid_sig = None
        self._styled_x_domain_sig = None
        self._vis_controls_sig = None
        self._last_range_label_text: Optional[str] = None

        # Parented coalesce timer for option-change redraws (prevents UI lag during bursts)
        self._draw_timer = QTimer(self)
        self._draw_timer.setSingleShot(True)
        self._draw_timer.setInterval(35)
        self._draw_timer.timeout.connect(self._on_draw_timer)

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
        # Row 1: Badge + Range + Stretch + Rescale / Pan / Zoom / Copy
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

        tok = current_tokens()
        icon_color = tok.text_primary
        self._icon_color = icon_color

        self.btn_home = QPushButton(self)
        self.btn_home.setIcon(Icons.rescale(color=icon_color))
        self.btn_home.setIconSize(QSize(14, 14))
        self.btn_home.setFixedSize(22, 22)
        self.btn_home.setToolTip("Rescale / Reset original view")
        row1_layout.addWidget(self.btn_home)
        self.btn_rescale = self.btn_home

        self.btn_pan = QToolButton(self)
        self.btn_pan.setCheckable(True)
        self.btn_pan.setIcon(Icons.pan(color=icon_color))
        self.btn_pan.setIconSize(QSize(14, 14))
        self.btn_pan.setFixedSize(22, 22)
        self.btn_pan.setToolTip("Pan / Drag axes with left mouse, zoom with right")
        row1_layout.addWidget(self.btn_pan)

        self.btn_zoom = QToolButton(self)
        self.btn_zoom.setCheckable(True)
        self.btn_zoom.setIcon(Icons.zoom(color=icon_color))
        self.btn_zoom.setIconSize(QSize(14, 14))
        self.btn_zoom.setFixedSize(22, 22)
        self.btn_zoom.setToolTip("Zoom to rectangle")
        row1_layout.addWidget(self.btn_zoom)

        self.btn_copy = QPushButton(self)
        self.btn_copy.setIcon(Icons.copy(color=icon_color))
        self.btn_copy.setIconSize(QSize(14, 14))
        self.btn_copy.setFixedSize(22, 22)
        self.btn_copy.setToolTip("Copy map to clipboard. Hold Ctrl + Click to copy data to clipboard")
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

        # 1D normalization controls (shown for 1D line plots, hidden on 2D heatmaps)
        self.btn_norm = QToolButton(self)
        self.btn_norm.setObjectName("btn_norm")
        self.btn_norm.setCheckable(True)
        self._update_norm_button_style()
        self.btn_norm.setIcon(Icons.norm(color=icon_color))
        self.btn_norm.setIconSize(QSize(14, 14))
        self.btn_norm.setFixedSize(22, 22)
        self.btn_norm.setToolTip(
            "Intensity normalization to maximum of curve or selected [Xmin, Xmax] region.\n"
            "Note: Visual change only; underlying data is not modified."
        )
        self.btn_norm.toggled.connect(self._on_norm_toggled)
        self.btn_norm.setVisible(False)
        row2_layout.addWidget(self.btn_norm)

        self.norm_xmin = QLineEdit(self)
        self.norm_xmin.setFixedSize(48, 22)
        self.norm_xmin.setPlaceholderText("Xmin")
        self.norm_xmin.setToolTip(
            "Minimum X bound for local 1D normalization peak search"
        )
        self.norm_xmin.editingFinished.connect(self._on_norm_range_edited)
        self.norm_xmin.setVisible(False)
        row2_layout.addWidget(self.norm_xmin)

        self.norm_xmax = QLineEdit(self)
        self.norm_xmax.setFixedSize(48, 22)
        self.norm_xmax.setPlaceholderText("Xmax")
        self.norm_xmax.setToolTip(
            "Maximum X bound for local 1D normalization peak search"
        )
        self.norm_xmax.editingFinished.connect(self._on_norm_range_edited)
        self.norm_xmax.setVisible(False)
        row2_layout.addWidget(self.norm_xmax)

        layout.addLayout(row2_layout)

        # ── Matplotlib Figure, Hidden Navigation Toolbar & Canvas ──
        self.figure = Figure(dpi=96, layout="none")
        self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._selector = MapCanvasSelector(self.ax, self.canvas)

        self.toolbar = NoDoubleClickZoomToolbar(self.canvas, self)
        self.toolbar.hide()
        self.toolbar.set_message = lambda _s: None
        self.btn_home.clicked.connect(self.reset_view)
        self.btn_pan.clicked.connect(self.toolbar.pan)
        self.btn_zoom.clicked.connect(self.toolbar.zoom)
        action_dict = {a.text(): a for a in self.toolbar.actions() if a.text()}
        if "Pan" in action_dict:
            action_dict["Pan"].toggled.connect(self.btn_pan.setChecked)
        if "Zoom" in action_dict:
            action_dict["Zoom"].toggled.connect(self.btn_zoom.setChecked)

        self.canvas.mpl_connect("button_press_event", self._on_canvas_button_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self.canvas.mpl_connect("draw_event", self._on_canvas_draw)
        self.canvas.mpl_connect("resize_event", self._on_canvas_resize)
        self._apply_figure_theme(self._current_theme)
        layout.addWidget(self.canvas, stretch=1)

    def _on_draw_timer(self):
        """Execute coalesced canvas redraw from parented QTimer."""
        try:
            if self.canvas is not None and self.isVisible():
                self.canvas.draw_idle()
        except (RuntimeError, AttributeError):
            pass

    def _schedule_draw(self):
        """Coalesce rapid consecutive UI option changes into one redraw."""
        if not self.isVisible():
            return
        self._draw_timer.start()

    def _has_layout_headroom(self) -> bool:
        """Return True when figure dimensions are large enough for compressed layout."""
        if self.canvas is None or self.figure is None:
            return False
        fig_w_px = float(self.figure.get_figwidth()) * float(self.figure.dpi)
        fig_h_px = float(self.figure.get_figheight()) * float(self.figure.dpi)
        return fig_w_px >= 180.0 and fig_h_px >= 120.0

    def _request_relayout(self):
        """Enable compressed layout when the canvas has headroom; skip on tiny surfaces."""
        self._relayout_needed = True
        try:
            target = "compressed" if self._has_layout_headroom() else "none"
            engine = self.figure.get_layout_engine()
            cur_name = type(engine).__name__ if engine is not None else "None"
            already_ok = (
                (target == "compressed" and cur_name in ("ConstrainedLayoutEngine", "CompressedLayoutEngine"))
                or (target == "none" and cur_name == "PlaceHolderLayoutEngine")
            )
            if not already_ok:
                self.figure.set_layout_engine(target)
        except Exception:
            pass

    def _on_canvas_resize(self, _event):
        """Re-evaluate compressed layout whenever canvas dimensions change."""
        self._request_relayout()

    def _on_canvas_draw(self, _event):
        """Freeze settled layout engine so pure restyles and pan/zoom skip constraint solving."""
        try:
            if self._relayout_needed and self._has_layout_headroom():
                self._relayout_needed = False
                self.figure.set_layout_engine("none")
        except Exception:
            pass

    def _update_norm_button_style(self):
        """Update normalization toggle button styling based on current theme tokens."""
        tok = current_tokens()
        self.btn_norm.setStyleSheet(
            f"QToolButton#btn_norm:checked {{ background-color: {tok.warning_tint}; border: 1px solid {tok.warning}; }}"
        )

    def apply_icon_theme(self, icon_color: str):
        """Update subpanel Qt toolbar button icons independently of Matplotlib plot theme."""
        self._icon_color = icon_color
        self.btn_home.setIcon(Icons.rescale(color=icon_color))
        self.btn_pan.setIcon(Icons.pan(color=icon_color))
        self.btn_zoom.setIcon(Icons.zoom(color=icon_color))
        self.btn_copy.setIcon(Icons.copy(color=icon_color))
        self.btn_norm.setIcon(Icons.norm(color=icon_color))
        self._update_norm_button_style()

    def _current_norm_params(self) -> tuple[bool, Optional[float], Optional[float]]:
        """Return ``(enabled, xmin, xmax)`` from this subpanel's normalization toolbar."""
        enabled = bool(getattr(self, "btn_norm", None) and self.btn_norm.isChecked())
        if not enabled:
            return False, None, None
        return (
            True,
            _parse_norm_bound(self.norm_xmin.text()),
            _parse_norm_bound(self.norm_xmax.text()),
        )

    def _apply_normalization(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Apply view-only 1D curve normalization when enabled on this subpanel."""
        y_arr = np.asarray(y, dtype=np.float64)
        enabled, xmin, xmax = self._current_norm_params()
        if not enabled:
            return y_arr
        return _normalize_1d_values(x, y_arr, xmin=xmin, xmax=xmax)

    def _on_norm_toggled(self, _checked: bool):
        """Re-render 1D curve(s) with view-only normalization and rescaled bounds."""
        self._last_norm_params = self._current_norm_params()
        if self._display_kind in ("series", "multi_profile"):
            self._request_relayout()
            self.rerender_cached(theme=self._current_theme)
            self.normalization_changed.emit()

    def _on_norm_range_edited(self):
        """Re-render and rescale when normalization bounds change while active."""
        if not self.btn_norm.isChecked():
            return
        params = self._current_norm_params()
        if params == self._last_norm_params:
            return
        self._last_norm_params = params
        if self._display_kind in ("series", "multi_profile"):
            self._request_relayout()
            self.rerender_cached(theme=self._current_theme)
            self.normalization_changed.emit()

    def reset_view(self):
        """Restore natural autoscale bounds for the current plot and clear zoom/pan stack."""
        self._styled_x_domain_sig = None
        if self._display_kind is None:
            self.toolbar.home()
            return
        try:
            self.toolbar.update()
        except Exception:
            pass
        self._request_relayout()
        if self._display_kind == "map" and self._grid is not None:
            _dg, _norm, _interp, xscale, yscale, _aspect = self._resolve_2d_render_params(
                self._grid, self._extent, self._vmin, self._vmax
            )
            self._apply_2d_axis_scales(
                self._extent, xscale, yscale, bool(self._view_options.get("show_grid", False))
            )
        elif self._display_kind == "multi_profile" and self._multi_entries:
            x_parts, y_parts = [], []
            for entry in self._multi_entries:
                px = np.asarray(entry.get("positions", []), dtype=np.float64)
                raw_vy = np.asarray(entry.get("values", []), dtype=np.float64)
                vy = self._apply_normalization(px, raw_vy) if len(px) == len(raw_vy) else raw_vy
                if len(px) > 0 and len(px) == len(vy):
                    m = np.isfinite(px) & np.isfinite(vy)
                    if np.any(m):
                        x_parts.append(px[m])
                        y_parts.append(vy[m])
            if x_parts:
                self._apply_1d_scales_and_limits(
                    np.concatenate(x_parts), np.concatenate(y_parts), is_index_axis=False
                )
        elif self._display_kind == "series" and self._series_values is not None and len(self._series_values) > 0:
            is_index = self._series_x is None
            x_arr = (
                np.arange(1, len(self._series_values) + 1, dtype=np.float64)
                if is_index
                else np.asarray(self._series_x, dtype=np.float64)
            )
            val_arr = self._apply_normalization(x_arr, self._series_values)
            m = np.isfinite(x_arr) & np.isfinite(val_arr)
            self._apply_1d_scales_and_limits(
                x_arr[m] if np.any(m) else x_arr,
                val_arr[m] if np.any(m) else val_arr,
                is_index_axis=is_index,
            )
        self.view_reset_requested.emit()
        if self.canvas is not None:
            self.canvas.draw_idle()

    def set_spatial_sites(self, sites: Optional[dict]):
        """Store current map spatial site metadata and refresh selection overlay."""
        if not sites:
            self._spatial_sites = None
            if self._selector is not None:
                self._selector.clear_drag_box()
                self._selector.clear_overlay()
                if self.canvas is not None:
                    self.canvas.draw_idle()
            return

        coords = np.asarray(sites.get("coords", []), dtype=np.float64)
        if coords.ndim != 2 or len(coords) == 0 or coords.shape[1] < 2:
            self._spatial_sites = None
            return

        is_profile = bool(sites.get("is_profile", False))
        raw_active = sites.get("active_mask")
        if raw_active is not None and len(raw_active) == len(coords):
            active_mask = np.asarray(raw_active, dtype=bool)
        else:
            active_mask = np.ones(len(coords), dtype=bool)

        positions = sites.get("positions")
        pos_arr = (
            np.asarray(positions, dtype=np.float64)
            if (is_profile and positions is not None and len(positions) == len(coords))
            else None
        )

        active_site_keys: set[tuple[float, float]] = set()
        site_index: dict[tuple[float, float], int] = {}
        for idx in range(len(coords)):
            key = (round(float(coords[idx, 0]), 4), round(float(coords[idx, 1]), 4))
            if bool(active_mask[idx]):
                active_site_keys.add(key)
                site_index.setdefault(key, idx)

        cell_dx, cell_dy = (1.0, 1.0)
        if not is_profile:
            geom_2d = compute_2d_spatial_geometry(coords[:, 0], coords[:, 1])
            cell_dx, cell_dy = float(geom_2d["dx"]), float(geom_2d["dy"])

        self._spatial_sites = {
            "map_name": sites.get("map_name"),
            "is_profile": is_profile,
            "coords": coords[:, :2],
            "positions": pos_arr,
            "active_mask": active_mask,
            "active_site_keys": active_site_keys,
            "site_index": site_index,
            "cell_dx": cell_dx,
            "cell_dy": cell_dy,
        }
        if self._display_kind is not None:
            self._update_selection_overlay(draw=True)

    def set_selected_points(self, points: list):
        """Update highlighted (X, Y) spectrum selection without emitting feedback signal."""
        self._selected_points = list(
            dict.fromkeys((float(pt[0]), float(pt[1])) for pt in (points or []))
        )
        self._update_selection_overlay(draw=True)

    def _is_event_on_legend(self, event) -> bool:
        """Return True if mouse event falls inside the visible legend box."""
        if self._legend is None or not self._legend.get_visible():
            return False
        if event.x is None or event.y is None or self.canvas is None:
            return False
        try:
            renderer = self.canvas.get_renderer()
            if renderer is None:
                return False
            return bool(self._legend.get_window_extent(renderer=renderer).contains(event.x, event.y))
        except Exception:
            return False

    def _is_selection_interactive(self) -> bool:
        """Allow interactive XY click/drag selection only in Map mode when nav tools are idle."""
        if self.mode != "map" or self._spatial_sites is None or self._selector is None:
            return False
        if bool(getattr(self.toolbar, "mode", "")) or self.btn_pan.isChecked() or self.btn_zoom.isChecked():
            return False
        if self._display_kind == "map":
            return self.img is not None and not self._spatial_sites.get("is_profile", False)
        if self._display_kind in ("series", "multi_profile"):
            return bool(self.lines) and bool(self._spatial_sites.get("is_profile", False))
        return False

    def _get_primary_profile_values(self) -> Optional[np.ndarray]:
        """Return plotted Y values corresponding to the primary 1D spatial curve."""
        if self._display_kind == "series":
            vals = self._series_values
            if vals is None:
                return None
            x_arr = (
                self._series_x
                if self._series_x is not None
                else np.arange(1, len(vals) + 1, dtype=np.float64)
            )
            return self._apply_normalization(x_arr, vals)
        if self._display_kind == "multi_profile" and self._multi_entries:
            target = next(
                (entry for entry in self._multi_entries if entry.get("is_primary")),
                self._multi_entries[0],
            )
            px = np.asarray(target.get("positions", []), dtype=np.float64)
            vy = np.asarray(target.get("values", []), dtype=np.float64)
            return self._apply_normalization(px, vy)
        return None

    def _filter_active_selected_points(self) -> list[tuple[float, float]]:
        """Filter stored selected points to active sites on the current spatial map."""
        if not self._spatial_sites:
            return list(self._selected_points)
        active_keys = self._spatial_sites.get("active_site_keys")
        if active_keys is None:
            return list(self._selected_points)
        return [
            pt for pt in self._selected_points
            if (round(float(pt[0]), 4), round(float(pt[1]), 4)) in active_keys
        ]

    def _update_selection_overlay(self, draw: bool = False):
        """Draw red selection boxes (2D) or markers (1D profile) matching active map sites."""
        if self._selector is None:
            return
        self._selector.ax = self.ax
        if (
            self.mode != "map"
            or not self._selected_points
            or self._spatial_sites is None
            or self._display_kind not in ("map", "series", "multi_profile")
        ):
            self._selector.clear_overlay()
            if draw and self.canvas is not None:
                self.canvas.draw_idle()
            return

        if self._display_kind == "map" and self.img is not None and not self._spatial_sites.get("is_profile", False):
            pts = self._filter_active_selected_points()
            dx = self._spatial_sites.get("cell_dx", 1.0)
            dy = self._spatial_sites.get("cell_dy", 1.0)
            self._selector.draw_2d_overlay(pts, dx, dy, draw=draw)
            return

        if self._display_kind in ("series", "multi_profile") and self.lines and self._spatial_sites.get("is_profile", False):
            positions = self._spatial_sites.get("positions")
            site_index = self._spatial_sites.get("site_index")
            active_mask = self._spatial_sites.get("active_mask")
            vals = self._get_primary_profile_values()
            if positions is None or site_index is None or vals is None or len(vals) != len(positions):
                self._selector.clear_overlay()
                if draw and self.canvas is not None:
                    self.canvas.draw_idle()
                return
            xs, ys = [], []
            for x, y in self._selected_points:
                idx = site_index.get((round(float(x), 4), round(float(y), 4)))
                if idx is None:
                    continue
                if active_mask is not None and not bool(active_mask[idx]):
                    continue
                px = float(positions[idx])
                py = float(vals[idx])
                if np.isfinite(px) and np.isfinite(py):
                    xs.append(px)
                    ys.append(py)
            markersize = max(5.5, float(self._view_options.get("dot_size", 1.5)) + 4.5)
            self._selector.draw_1d_overlay(xs, ys, markersize=markersize, draw=draw)
            return

        self._selector.clear_overlay()
        if draw and self.canvas is not None:
            self.canvas.draw_idle()

    def _on_canvas_button_press(self, event):
        """Handle legend double-click editing and interactive map spectrum click start."""
        if getattr(event, "dblclick", False) and self._is_event_on_legend(event):
            self.legend_edit_requested.emit()
            return
        if not self._is_selection_interactive():
            return
        if self._is_event_on_legend(event):
            return
        self._selector.ax = self.ax
        self._selector.begin_press(event)

    def _on_canvas_motion(self, event):
        """Update rectangular selection marquee while dragging on the plot."""
        if not self._is_selection_interactive():
            return
        edgecolor = get_drag_box_edgecolor(self._current_theme == "Dark Mode")
        self._selector.ax = self.ax
        self._selector.update_move(event, edgecolor=edgecolor)

    def _on_canvas_release(self, event):
        """Finish click/drag selection and emit chosen (X, Y) spectrum coordinates."""
        if self._selector is None:
            return
        self._selector.ax = self.ax
        gesture = self._selector.finish_release(event)
        if gesture is None or not self._is_selection_interactive():
            return

        x0, y0, x1, y1, is_rect = gesture
        ctrl_held = is_ctrl_event(event)
        sites = self._spatial_sites
        coords = sites["coords"]
        active = sites["active_mask"]

        if self._display_kind == "map":
            act_coords = coords[active]
            new_points = resolve_xy_selection(
                act_coords[:, 0],
                act_coords[:, 1],
                self._selected_points,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                is_rect=is_rect,
                ctrl_held=ctrl_held,
            )
        else:
            positions = sites.get("positions")
            if positions is None or len(positions) != len(coords):
                return
            primary_vals = self._get_primary_profile_values()
            act_pos = positions[active]
            act_coords = coords[active]
            act_vals = (
                primary_vals[active]
                if (primary_vals is not None and len(primary_vals) == len(coords))
                else None
            )
            new_points = resolve_1d_profile_selection(
                act_pos,
                act_vals,
                act_coords,
                self._selected_points,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                is_rect=is_rect,
                ctrl_held=ctrl_held,
            )

        self._selected_points = new_points
        self._update_selection_overlay(draw=True)
        self.spectra_selected.emit(list(self._selected_points))

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
            self.btn_copy.setToolTip("Copy map to clipboard. Hold Ctrl + Click to copy data to clipboard")
            self.cbb_param.setToolTip("Select spectral parameter for this map")
        else:
            self.btn_copy.setToolTip("Copy graph to clipboard. Hold Ctrl + Click to copy data to clipboard")
            self.cbb_param.setToolTip("Select spectral parameter for this graph")
        self._sync_palette_visibility()

    def _invalidate_style_caches(self):
        """Reset cached axes styling signatures so the next render reapplies them."""
        self._styled_theme_sig = None
        self._styled_text_fs_sig = None
        self._styled_grid_sig = None
        self._styled_x_domain_sig = None

    def _sync_1d_controls_visibility(self):
        """Show colormap picker on 2D heatmaps and normalization controls on 1D line plots."""
        is_1d_plot = self._display_kind in ("series", "multi_profile")
        sig = (self.mode, is_1d_plot)
        if self._vis_controls_sig == sig:
            return
        self._vis_controls_sig = sig
        self.cbb_palette.setVisible(self.mode == "map" and not is_1d_plot)
        self.btn_norm.setVisible(is_1d_plot)
        self.norm_xmin.setVisible(is_1d_plot)
        self.norm_xmax.setVisible(is_1d_plot)

    def _sync_palette_visibility(self):
        """Synchronize Row-2 palette and 1D normalization control visibility."""
        self._sync_1d_controls_visibility()

    def _reset_axes(self):
        """Remove any colorbar, legend, clear lines, clear axes, and restore aspect scaling.

        Matplotlib's ``Axes.clear()`` does not reset the aspect ratio set by
        ``imshow(..., aspect="equal")``; without resetting to ``"auto"``, a
        subsequent 1D line plot inherits a 1:1 data aspect and collapses into a
        thin vertical or horizontal strip when X and Y spans differ.
        """
        self._request_relayout()
        self._invalidate_style_caches()
        if self._legend is not None:
            try:
                self._legend.remove()
            except Exception:
                pass
            self._legend = None
        self._legend_signature = None
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
        if self._selector is not None:
            self._selector.ax = self.ax
            self._selector.reset()
        self.ax.clear()
        self.ax.set_aspect("auto", adjustable="box")
        try:
            self.toolbar.update()
        except Exception:
            pass

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
        self._sync_1d_controls_visibility()
        self._reset_axes()
        self._apply_figure_theme(self._current_theme)
        if self.canvas is not None:
            self.canvas.draw_idle()

    def is_ratio_display(self) -> bool:
        """True if this panel currently displays the Red/Blue ratio."""
        return self.cbb_param.currentText() == PARAM_RB_RATIO

    def _on_param_changed(self, param: str):
        self._refresh_range_label()
        self.parameter_changed.emit(self.color_id, param)

    def _parse_copied_figure_size(self) -> Tuple[float, float]:
        """Return validated copied figure (width, height) in inches from view options."""
        def _parse_one(raw, fallback: float) -> float:
            try:
                val = float(str(raw or "").strip().replace(",", "."))
                if 0.5 <= val <= 50.0 and np.isfinite(val):
                    return val
            except (TypeError, ValueError):
                pass
            return fallback

        return (
            _parse_one(self._view_options.get("copy_fig_width"), 5.5),
            _parse_one(self._view_options.get("copy_fig_height"), 4.0),
        )

    def set_2d_data_provider(self, provider: Optional[Callable[[str], Optional[tuple]]]):
        """Register optional callable supplying raw ``(x, y, values)`` points for 2D map copy."""
        self._2d_data_provider = provider

    @staticmethod
    def _unique_column_name(base: str, used: set[str]) -> str:
        """Return a unique DataFrame column name avoiding key collisions."""
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base} ({suffix})"
            suffix += 1
        used.add(candidate)
        return candidate

    def build_clipboard_dataframe(self):
        """Build a pandas DataFrame containing the current subpanel's underlying numerical data."""
        import pandas as pd

        param_name = self.cbb_param.currentText() or "Value"
        kind = self._display_kind

        if kind == "multi_profile":
            custom_labels = self._view_options.get("custom_labels", {})
            valid_entries = []
            for idx, entry in enumerate(self._multi_entries):
                px = np.asarray(entry.get("positions", []), dtype=np.float64)
                vy = np.asarray(entry.get("values", []), dtype=np.float64)
                if len(px) == 0 or len(px) != len(vy):
                    continue
                if not np.any(np.isfinite(px) & np.isfinite(vy)):
                    continue
                raw_name = str(entry.get("name") or f"Map {idx + 1}")
                display_name = str(custom_labels.get(raw_name) or raw_name)
                valid_entries.append((display_name, px, vy))
            if not valid_entries:
                return None

            x_label = (
                self._multi_x_label
                if self._multi_x_label.endswith("(mm)")
                else f"{self._multi_x_label or 'Position'} (mm)"
            )
            ref_x = valid_entries[0][1]
            shared_x = all(
                len(px) == len(ref_x) and np.allclose(px, ref_x, rtol=1e-7, atol=1e-9, equal_nan=True)
                for _, px, _ in valid_entries[1:]
            )
            used: set[str] = set()
            data = {}
            if shared_x:
                data[self._unique_column_name(x_label, used)] = pd.Series(ref_x)
                for label, _px, vy in valid_entries:
                    data[self._unique_column_name(label, used)] = pd.Series(vy)
            else:
                for label, px, vy in valid_entries:
                    data[self._unique_column_name(f"{label} ({x_label})", used)] = pd.Series(px)
                    data[self._unique_column_name(f"{label} ({param_name})", used)] = pd.Series(vy)
            return pd.DataFrame(data)

        if kind == "series":
            if self._series_values is None or len(self._series_values) == 0:
                return None
            val_arr = np.asarray(self._series_values, dtype=np.float64)
            if not np.any(np.isfinite(val_arr)):
                return None
            if self._series_x is not None and len(self._series_x) == len(val_arr):
                x_label = self._series_xlabel or "Position (mm)"
                return pd.DataFrame({
                    x_label: np.asarray(self._series_x, dtype=np.float64),
                    param_name: val_arr,
                })
            data = {"Spectrum #": np.arange(1, len(val_arr) + 1)}
            if self._series_labels and len(self._series_labels) == len(val_arr):
                data["Spectrum"] = [str(lbl) for lbl in self._series_labels]
            data[param_name] = val_arr
            return pd.DataFrame(data)

        if kind == "map":
            if self._2d_data_provider is not None:
                raw_pts = self._2d_data_provider(self.color_id)
                if raw_pts is not None and len(raw_pts) == 3:
                    x_raw, y_raw, v_raw = (np.asarray(arr, dtype=np.float64) for arr in raw_pts)
                    if len(x_raw) > 0 and len(x_raw) == len(y_raw) == len(v_raw):
                        return pd.DataFrame({
                            "X (mm)": x_raw,
                            "Y (mm)": y_raw,
                            param_name: v_raw,
                        })
            if self._grid is None:
                return None
            grid = np.asarray(self._grid, dtype=np.float64)
            if grid.ndim != 2 or grid.shape[0] == 0 or grid.shape[1] == 0 or not np.any(np.isfinite(grid)):
                return None
            ny, nx = grid.shape
            if self._extent and len(self._extent) >= 4:
                x0, x1, y0, y1 = (float(self._extent[i]) for i in range(4))
                dx = (x1 - x0) / nx
                dy = (y1 - y0) / ny
                xs = (
                    x0 + (np.arange(nx, dtype=np.float64) + 0.5) * dx
                    if (np.isfinite(dx) and abs(dx) > 0)
                    else np.arange(nx, dtype=np.float64)
                )
                ys = (
                    y0 + (np.arange(ny, dtype=np.float64) + 0.5) * dy
                    if (np.isfinite(dy) and abs(dy) > 0)
                    else np.arange(ny, dtype=np.float64)
                )
            else:
                xs = np.arange(nx, dtype=np.float64)
                ys = np.arange(ny, dtype=np.float64)
            gx, gy = np.meshgrid(xs, ys)
            return pd.DataFrame({
                "X (mm)": gx.ravel(),
                "Y (mm)": gy.ravel(),
                param_name: grid.ravel(),
            })

        return None

    def copy_data_to_clipboard(self) -> bool:
        """Copy current subpanel numerical data as TSV to the Qt clipboard."""
        df = self.build_clipboard_dataframe()
        return copy_df_to_clb(df)

    def _copy_to_clipboard(self):
        """Copy figure (click) or raw plot data (Ctrl+Click) to the system clipboard."""
        if is_ctrl_event(None):
            self.copy_data_requested.emit(self.color_id)
            if self.copy_data_to_clipboard():
                self._flash_copy_feedback()
            return

        size_ratio = self._parse_copied_figure_size()
        target_theme = _normalize_plot_theme(
            self._view_options.get("copy_fig_theme", "Light Mode")
        )
        original_theme = _normalize_plot_theme(self._current_theme)
        try:
            if target_theme != original_theme:
                self._apply_figure_theme(target_theme)
            if (
                size_ratio[0] * float(self.figure.dpi) >= 180.0
                and size_ratio[1] * float(self.figure.dpi) >= 120.0
            ):
                self._relayout_needed = True
                self.figure.set_layout_engine("compressed")
            copy_fig_to_clb(self.canvas, size_ratio=size_ratio)
        finally:
            if target_theme != original_theme:
                self._apply_figure_theme(original_theme)
            self._request_relayout()
            if self.canvas is not None:
                self.canvas.draw_idle()
        self._flash_copy_feedback()

    def _flash_copy_feedback(self) -> None:
        """Show 1-second checkmark feedback on the Copy button."""
        self.btn_copy.setIcon(Icons.check())
        QTimer.singleShot(1000, self, self._reset_copy_feedback)

    def _reset_copy_feedback(self) -> None:
        try:
            self.btn_copy.setIcon(Icons.copy(color=getattr(self, "_icon_color", None)))
        except (RuntimeError, AttributeError, TypeError):
            pass

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
            target = "I(Red) / I(Blue)"
        else:
            x_min, x_max = self._x_range
            target = f"[{x_min:.1f}, {x_max:.1f}]"
        if self._last_range_label_text != target:
            self._last_range_label_text = target
            self.lbl_range.setText(target)

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
        """Build uniform Matplotlib linestyle, linewidth, marker, and markersize from view options."""
        opts = self._view_options
        style = opts.get("plot_style", "line")
        lw = float(opts.get("line_width", 1.5))
        ds = float(opts.get("dot_size", 1.5))
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
            if self.ax.get_yscale() != "log":
                self.ax.set_yscale("log")
            self.ax.set_ylim(*self._valid_pos_y_bounds)
        else:
            if self.ax.get_yscale() != "linear":
                self.ax.set_yscale("linear")
            if self._valid_y_bounds is not None:
                self.ax.set_ylim(*self._valid_y_bounds)

        finite_x = all_x[np.isfinite(all_x)]
        if req_xscale == "log":
            pos_x = finite_x[finite_x > 0]
            if pos_x.size > 0:
                x_min = float(np.min(pos_x))
                x_max = float(np.max(pos_x))
                if x_max / max(x_min, 1e-30) < 1.05:
                    x_bounds = (max(x_min * 0.8, 1e-300), x_max * 1.25)
                else:
                    x_bounds = (max(x_min * 0.92, 1e-300), x_max * 1.08)
                x_sig = ("log", x_bounds[0], x_bounds[1])
                if self.ax.get_xscale() != "log" or self._styled_x_domain_sig != x_sig:
                    if self.ax.get_xscale() != "log":
                        self.ax.set_xscale("log")
                    self.ax.set_xlim(*x_bounds)
                    self._styled_x_domain_sig = x_sig
                return
        if self.ax.get_xscale() != "linear":
            self.ax.set_xscale("linear")
            self._styled_x_domain_sig = None
        x_target = finite_x if not is_index_axis else all_x
        if is_index_axis:
            x_sig = ("index", len(x_target))
        elif finite_x.size > 0:
            x_sig = ("linear", len(finite_x), float(np.min(finite_x)), float(np.max(finite_x)))
        else:
            x_sig = ("empty", 0)
        if self._styled_x_domain_sig != x_sig:
            self._apply_x_limits(x_target, is_index_axis)
            self._styled_x_domain_sig = x_sig

    def _apply_grid_state(self, show_grid: bool, theme: Optional[str] = None, force: bool = True):
        """Apply Matplotlib grid state cleanly without raising false+kwargs warnings."""
        active_theme = _normalize_plot_theme(theme if theme is not None else self._current_theme)
        sig = (bool(show_grid), active_theme if show_grid else None)
        if not force and self._styled_grid_sig == sig:
            return
        self._styled_grid_sig = sig
        if show_grid:
            _, style_path = _resolve_style_path(active_theme)
            grid_col = _get_cached_style_dict(style_path).get("grid.color", "#888888")
            self.ax.grid(True, alpha=0.25, linestyle="--", color=grid_col)
        else:
            self.ax.grid(False)

    def _apply_legend_theme_colors(self):
        """Ensure explicit high-contrast legend frame and label colors for the active plot theme."""
        if self._legend is None:
            return
        is_dark, _ = _resolve_style_path(self._current_theme)
        leg_fg = "#FFFFFF" if is_dark else "#111111"
        leg_bg = "#2A2A2A" if is_dark else "#FFFFFF"
        leg_edge = "#555555" if is_dark else "#CCCCCC"
        frame = self._legend.get_frame()
        if frame is not None:
            frame.set_facecolor(leg_bg)
            frame.set_edgecolor(leg_edge)
            frame.set_alpha(0.88)
        for text in self._legend.get_texts():
            text.set_color(leg_fg)

    def _current_legend_signature(self):
        opts = self._view_options
        labels_tuple = tuple(
            str(ln.get_label()) for ln in self.lines
            if ln.get_label() and not str(ln.get_label()).startswith("_")
        )
        colors_tuple = tuple(
            ln.get_color() for ln in self.lines
            if ln.get_label() and not str(ln.get_label()).startswith("_")
        )
        return (
            bool(opts.get("show_legend", False)),
            str(opts.get("legend_loc", "best")),
            max(1, int(opts.get("legend_ncol", 1))),
            max(6, int(opts.get("legend_fontsize", 8))),
            labels_tuple,
            colors_tuple,
        )

    def _sync_legend(self, force_rebuild: bool = False):
        """Create, update, or remove the Matplotlib legend box per current view options."""
        want_show = bool(self._view_options.get("show_legend", False))
        if not want_show or self._display_kind not in ("series", "multi_profile"):
            if self._legend is not None:
                try:
                    self._legend.remove()
                except Exception:
                    pass
                self._legend = None
            self._legend_signature = None
            return

        handles = []
        labels = []
        for ln in self.lines:
            lbl = ln.get_label()
            if lbl and not str(lbl).startswith("_"):
                handles.append(ln)
                labels.append(str(lbl))
        if not handles:
            if self._legend is not None:
                try:
                    self._legend.remove()
                except Exception:
                    pass
                self._legend = None
            self._legend_signature = None
            return

        sig = self._current_legend_signature()
        if not force_rebuild and self._legend is not None and sig == self._legend_signature:
            # Reuse existing legend and only refresh theme/handle styling in place
            self._apply_legend_theme_colors()
            return

        if self._legend is not None:
            try:
                self._legend.remove()
            except Exception:
                pass
            self._legend = None

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

        self._legend_signature = sig
        self._apply_legend_theme_colors()

    def _apply_figure_theme(self, theme: Optional[str] = None, force: bool = True):
        """Explicitly paint Figure, Axes, ticks, spines, grid, legend, and colorbar for the active theme."""
        active_theme = _normalize_plot_theme(theme if theme is not None else self._current_theme)
        self._current_theme = active_theme
        grid_on = bool(self._display_kind is not None and self._view_options.get("show_grid", False))
        sig = (active_theme, self.cbar is not None, grid_on)
        if not force and self._styled_theme_sig == sig:
            self._apply_legend_theme_colors()
            return
        self._styled_theme_sig = sig
        _is_dark, style_path = _resolve_style_path(active_theme)
        s = _get_cached_style_dict(style_path)
        self.figure.patch.set_facecolor(s.get("figure.facecolor", "#FFFFFF"))
        self.figure.patch.set_edgecolor(s.get("figure.edgecolor", "white"))
        self.ax.set_facecolor(s.get("axes.facecolor", "#FFFFFF"))
        label_col = s.get("axes.labelcolor", "#000000")
        self.ax.xaxis.label.set_color(label_col)
        self.ax.yaxis.label.set_color(label_col)
        self.ax.title.set_color(s.get("text.color", "black"))
        tick_col = s.get("xtick.color", "#000000")
        self.ax.tick_params(axis="both", colors=tick_col)
        edge_col = s.get("axes.edgecolor", "#000000")
        for spine in self.ax.spines.values():
            spine.set_edgecolor(edge_col)
        if self.cbar is not None:
            try:
                self.cbar.ax.tick_params(colors=s.get("ytick.color", tick_col))
                self.cbar.outline.set_edgecolor(edge_col)
            except Exception:
                pass
        if grid_on:
            self._apply_grid_state(True, active_theme, force=force)
        self._apply_legend_theme_colors()

    def _apply_text_sizes(self, force: bool = True):
        """Apply user-configured text size to title, axis labels, ticks, and colorbar ticks."""
        text_fs = max(6, int(self._view_options.get("text_fontsize", 10)))
        sig = (text_fs, self.cbar is not None)
        if not force and self._styled_text_fs_sig == sig:
            return
        self._styled_text_fs_sig = sig
        tick_fs = max(6, text_fs - 1)
        self.ax.title.set_fontsize(text_fs)
        self.ax.xaxis.label.set_fontsize(text_fs)
        self.ax.yaxis.label.set_fontsize(text_fs)
        self.ax.tick_params(axis="both", labelsize=tick_fs)
        if self.cbar is not None:
            try:
                self.cbar.ax.tick_params(labelsize=tick_fs)
            except Exception:
                pass

    def _sync_title(self, has_content: Optional[bool] = None):
        """Show or clear the Matplotlib axes title label per ``show_figure_label``."""
        if has_content is None:
            has_content = bool(self.img is not None or self.lines)
        want_label = bool(self._view_options.get("show_figure_label", False)) and bool(has_content)
        target = f"{self.color_name}: {self.cbb_param.currentText()}" if want_label else ""
        if self.ax.get_title() != target:
            text_fs = max(6, int(self._view_options.get("text_fontsize", 10)))
            self.ax.set_title(target, fontsize=text_fs, pad=4)

    def _sync_1d_line_styles(self):
        """Update curve linestyle, linewidth, marker, markersize, and legend handles in place."""
        if not self.lines:
            return
        multi = self._display_kind == "multi_profile"
        entries = self._multi_entries if multi else []
        for idx, ln in enumerate(self.lines):
            is_primary = bool(entries[idx].get("is_primary", False)) if idx < len(entries) else True
            kw = self._curve_style_kwargs(is_primary=is_primary, multi=multi)
            ln.set_linestyle(kw["linestyle"])
            ln.set_linewidth(kw["linewidth"])
            ln.set_marker(kw["marker"])
            ln.set_markersize(kw["markersize"])
        if self._legend is not None:
            leg_lines = self._legend.get_lines()
            for idx, leg_ln in enumerate(leg_lines):
                if idx < len(self.lines):
                    src = self.lines[idx]
                    leg_ln.set_linestyle(src.get_linestyle())
                    leg_ln.set_linewidth(src.get_linewidth())
                    leg_ln.set_marker(src.get_marker())
                    leg_ln.set_markersize(src.get_markersize())
                    leg_ln.set_color(src.get_color())

    def apply_view_options_fast(
        self,
        options: dict,
        changed_keys: Optional[set] = None,
        redraw: bool = True,
    ):
        """Apply view option changes directly to existing Matplotlib artists when possible."""
        prev_loc = self._view_options.get("legend_loc", "best")
        self._view_options = _copy_view_options(options)
        self._current_theme = self._view_options.get("plot_theme", "Light Mode")

        if changed_keys and changed_keys <= {"copy_fig_width", "copy_fig_height", "copy_fig_theme"}:
            return

        if self._display_kind is None:
            if changed_keys is None or "plot_theme" in changed_keys:
                self._apply_figure_theme(self._current_theme)
                if redraw:
                    self._schedule_draw()
            return

        if not changed_keys:
            self._request_relayout()
            self.rerender_cached(theme=self._current_theme)
            return

        # Options that require full axis/colorbar/scale recomputation
        if changed_keys & {"xscale", "yscale", "show_colorbar"}:
            if changed_keys == {"show_colorbar"} and self._display_kind != "map":
                return
            self._request_relayout()
            self.rerender_cached(theme=self._current_theme)
            return

        if "plot_theme" in changed_keys:
            self._apply_figure_theme(self._current_theme)

        if "show_grid" in changed_keys:
            self._apply_grid_state(bool(self._view_options.get("show_grid", False)), self._current_theme)

        if "show_figure_label" in changed_keys:
            self._sync_title()
            self._apply_figure_theme(self._current_theme)
            self._apply_text_sizes()
            self._request_relayout()

        if "text_fontsize" in changed_keys:
            self._apply_text_sizes()
            self._request_relayout()

        if self._display_kind in ("series", "multi_profile"):
            if changed_keys & {"plot_style", "line_width", "dot_size"}:
                self._sync_1d_line_styles()
                if "dot_size" in changed_keys and self._selected_points:
                    self._update_selection_overlay(draw=False)
            if changed_keys & {"custom_labels", "custom_colors"}:
                custom_labels = self._view_options.get("custom_labels", {})
                custom_colors = self._view_options.get("custom_colors", {})
                if self._display_kind == "series" and self.line is not None:
                    title_text = f"{self.color_name}: {self.cbb_param.currentText()}"
                    self.line.set_label(custom_labels.get(self.color_id) or title_text)
                    self.line.set_color(custom_colors.get(self.color_id) or self.color_hex)
                elif self._display_kind == "multi_profile":
                    for idx, (ln, entry) in enumerate(zip(self.lines, self._multi_entries)):
                        key = entry.get("name", f"Map {idx + 1}")
                        auto_col = (
                            self.color_hex
                            if len(self._multi_entries) == 1
                            else DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
                        )
                        ln.set_label(custom_labels.get(key) or key)
                        ln.set_color(custom_colors.get(key) or auto_col)
                self._sync_legend(force_rebuild=True)
            elif changed_keys & {"show_legend", "legend_loc", "legend_ncol", "legend_fontsize"}:
                new_loc = self._view_options.get("legend_loc", "best")
                if "outside" in str(prev_loc) or "outside" in str(new_loc):
                    self._request_relayout()
                self._sync_legend(force_rebuild=True)
        elif self._display_kind == "map" and self.img is not None:
            if "heatmap_smoothing" in changed_keys:
                self.img.set_interpolation(
                    "bilinear" if self._view_options.get("heatmap_smoothing", False) else "none"
                )
            if "zscale" in changed_keys and self._grid is not None:
                display_grid, norm, _interp, _xs, _ys, _asp = self._resolve_2d_render_params(
                    self._grid, self._extent, self._vmin, self._vmax
                )
                self.img.set_data(display_grid)
                self.img.set_norm(norm)
                if self.cbar is not None:
                    self.cbar.update_normal(self.img)
                    self._apply_text_sizes()
                    self._request_relayout()
            # Ignore 1D-only option bursts on 2D heatmaps
            if changed_keys <= {
                "plot_style", "line_width", "dot_size",
                "show_legend", "legend_loc", "legend_ncol", "legend_fontsize",
                "custom_labels", "custom_colors", "share_y",
            }:
                return

        if redraw:
            self._schedule_draw()

    def update_series(
        self,
        values: Optional[np.ndarray],
        labels: Optional[list[str]] = None,
        status_msg: str = "",
        theme: Optional[str] = None,
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
        active_theme = _normalize_plot_theme(
            theme if theme is not None else self._view_options.get("plot_theme", self._current_theme)
        )
        self._current_theme = active_theme

        param_name = self.cbb_param.currentText()
        title_text = f"{self.color_name}: {param_name}"
        custom_labels = self._view_options.get("custom_labels", {})
        custom_colors = self._view_options.get("custom_colors", {})
        curve_label = custom_labels.get(self.color_id) or title_text
        curve_color = custom_colors.get(self.color_id) or self.color_hex
        style_kw = self._curve_style_kwargs(is_primary=True, multi=False)
        show_grid = bool(self._view_options.get("show_grid", False))

        has_valid = len(val_arr) > 0 and bool(np.any(np.isfinite(val_arr)))
        plot_vals = self._apply_normalization(x_arr, val_arr) if has_valid else val_arr
        text_fs = max(6, int(self._view_options.get("text_fontsize", 10)))

        # Fast path: in-place single-line update during continuous bracket dragging
        if (
            self._display_kind == "series"
            and len(self.lines) == 1
            and self.line is not None
            and has_valid
            and len(self.line.get_xdata()) == len(plot_vals)
        ):
            self._sync_1d_controls_visibility()
            self.line.set_data(x_arr, plot_vals)
            self.line.set_color(curve_color)
            self.line.set_linestyle(style_kw["linestyle"])
            self.line.set_linewidth(style_kw["linewidth"])
            self.line.set_marker(style_kw["marker"])
            self.line.set_markersize(style_kw["markersize"])
            self.line.set_label(curve_label)
            finite_mask = np.isfinite(x_arr) & np.isfinite(plot_vals)
            self._apply_1d_scales_and_limits(
                x_arr[finite_mask] if np.any(finite_mask) else x_arr,
                plot_vals[finite_mask] if np.any(finite_mask) else plot_vals,
                is_index_axis=is_index_axis,
            )
            self._apply_grid_state(show_grid, active_theme, force=False)
            if self.ax.get_xlabel() != x_label:
                self.ax.set_xlabel(x_label)
            if self.ax.get_ylabel() != param_name:
                self.ax.set_ylabel(param_name)
            self._sync_title(has_content=True)
            self._apply_figure_theme(active_theme, force=False)
            self._apply_text_sizes(force=False)
            self._sync_legend()
            self._update_selection_overlay(draw=False)
            self.canvas.draw_idle()
            return

        self._display_kind = "series"
        self._sync_palette_visibility()
        self._reset_axes()
        self._apply_figure_theme(active_theme)

        if not has_valid:
            self._valid_y_bounds = None
            self._valid_pos_y_bounds = None
            msg = status_msg or ("No spectra selected" if is_index_axis else "No profile data available")
            self.ax.text(
                0.5, 0.5, msg,
                ha="center", va="center", transform=self.ax.transAxes,
                fontsize=text_fs, color="gray", wrap=True
            )
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw_idle()
            return

        self.lines = list(
            self.ax.plot(
                x_arr,
                plot_vals,
                color=curve_color,
                alpha=0.95,
                label=curve_label,
                **style_kw,
            )
        )

        finite_mask = np.isfinite(x_arr) & np.isfinite(plot_vals)
        self._apply_1d_scales_and_limits(
            x_arr[finite_mask] if np.any(finite_mask) else x_arr,
            plot_vals[finite_mask] if np.any(finite_mask) else plot_vals,
            is_index_axis=is_index_axis,
        )
        self._apply_grid_state(show_grid, active_theme)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(param_name)
        self._sync_title(has_content=True)
        self._apply_figure_theme(active_theme)
        self._apply_text_sizes()
        self._sync_legend()
        self._update_selection_overlay(draw=False)
        try:
            self.toolbar.update()
        except Exception:
            pass
        self.canvas.draw_idle()

    def update_multi_profile(
        self,
        entries: List[dict],
        shared_x_label: str = "X",
        status_msg: str = "",
        theme: Optional[str] = None,
    ):
        """Render or update multiple stacked 1D spatial profiles in one subpanel."""
        self._multi_entries = list(entries) if entries else []
        self._multi_x_label = shared_x_label
        self._series_values = None
        self._series_labels = []
        self._series_x = None
        self._status_msg = status_msg
        active_theme = _normalize_plot_theme(
            theme if theme is not None else self._view_options.get("plot_theme", self._current_theme)
        )
        self._current_theme = active_theme

        param_name = self.cbb_param.currentText()
        x_label = (
            shared_x_label
            if shared_x_label.endswith("(mm)")
            else f"{shared_x_label or 'Position'} (mm)"
        )

        # Compute finite bounds across normalized curves without mutating cached raw entries
        plotted_curves = []
        finite_x_parts = []
        finite_y_parts = []
        for entry in self._multi_entries:
            px = np.asarray(entry.get("positions", []), dtype=np.float64)
            raw_vy = np.asarray(entry.get("values", []), dtype=np.float64)
            vy = self._apply_normalization(px, raw_vy) if len(px) == len(raw_vy) else raw_vy
            plotted_curves.append((entry, px, vy))
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
        show_grid = bool(self._view_options.get("show_grid", False))
        text_fs = max(6, int(self._view_options.get("text_fontsize", 10)))

        # Fast path: reuse existing Line2D pool when entry count matches
        if (
            self._display_kind == "multi_profile"
            and has_valid
            and self.img is None
            and len(self.lines) == len(plotted_curves)
            and len(self.lines) > 0
        ):
            self._sync_1d_controls_visibility()
            for idx, (ln, (entry, px, vy)) in enumerate(zip(self.lines, plotted_curves)):
                key = entry.get("name", f"Map {idx + 1}")
                auto_color = (
                    self.color_hex
                    if len(plotted_curves) == 1
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
            self._apply_grid_state(show_grid, active_theme, force=False)
            if self.ax.get_xlabel() != x_label:
                self.ax.set_xlabel(x_label)
            if self.ax.get_ylabel() != param_name:
                self.ax.set_ylabel(param_name)
            self._sync_title(has_content=True)
            self._apply_figure_theme(active_theme, force=False)
            self._apply_text_sizes(force=False)
            self._sync_legend()
            self._update_selection_overlay(draw=False)
            self.canvas.draw_idle()
            return

        self._display_kind = "multi_profile"
        self._sync_palette_visibility()
        self._reset_axes()
        self._apply_figure_theme(active_theme)

        if not has_valid:
            self._valid_y_bounds = None
            self._valid_pos_y_bounds = None
            msg = status_msg or "No valid profile data within selected range"
            self.ax.text(
                0.5, 0.5, msg,
                ha="center", va="center", transform=self.ax.transAxes,
                fontsize=text_fs, color="gray", wrap=True,
            )
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw_idle()
            return

        new_lines = []
        for idx, (entry, px, vy) in enumerate(plotted_curves):
            key = entry.get("name", f"Map {idx + 1}")
            auto_color = (
                self.color_hex
                if len(plotted_curves) == 1
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
        self._apply_grid_state(show_grid, active_theme)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(param_name)
        self._sync_title(has_content=True)
        self._apply_figure_theme(active_theme)
        self._apply_text_sizes()
        self._sync_legend()
        self._update_selection_overlay(draw=False)
        try:
            self.toolbar.update()
        except Exception:
            pass
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

        self._apply_grid_state(show_grid)

    def update_map(
        self,
        grid: Optional[np.ndarray],
        extent: list[float],
        vmin: float,
        vmax: float,
        status_msg: str = "",
        theme: Optional[str] = None,
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
        active_theme = _normalize_plot_theme(
            theme if theme is not None else self._view_options.get("plot_theme", self._current_theme)
        )
        self._current_theme = active_theme

        want_colorbar = bool(self._view_options.get("show_colorbar", True))
        show_grid = bool(self._view_options.get("show_grid", False))
        has_valid_grid = grid is not None and not np.all(np.isnan(grid))
        text_fs = max(6, int(self._view_options.get("text_fontsize", 10)))

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
            self._sync_title(has_content=True)
            if self.cbar is not None:
                self.cbar.update_normal(self.img)
            self._apply_figure_theme(active_theme)
            self._apply_text_sizes()
            self._update_selection_overlay(draw=False)
            self.canvas.draw_idle()
            return

        # Re-render path for status messages, colorbar toggle, or first draw
        self._display_kind = "map"
        self._sync_palette_visibility()
        self._reset_axes()
        self._apply_figure_theme(active_theme)

        if not has_valid_grid:
            msg = status_msg or "No map data available"
            self.ax.text(
                0.5, 0.5, msg,
                ha="center", va="center", transform=self.ax.transAxes,
                fontsize=text_fs, color="gray", wrap=True
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
        self._sync_title(has_content=True)
        self._apply_figure_theme(active_theme)
        self._apply_text_sizes()
        self._update_selection_overlay(draw=False)
        try:
            self.toolbar.update()
        except Exception:
            pass
        self.canvas.draw_idle()
