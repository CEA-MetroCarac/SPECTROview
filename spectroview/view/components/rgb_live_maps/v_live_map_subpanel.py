"""Subpanel widget rendering a 2D heatmap or 1D line plot for one RGB Live Maps channel."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
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
from spectroview.model.spectra.spectral_parameters import (
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
    is_ctrl_event,
)
from spectroview.view.components.rgb_live_maps.common import (
    AVAILABLE_PALETTES,
    DEFAULT_PALETTES,
    _copy_view_options,
    _normalize_1d_values,
    _normalize_plot_theme,
    _parse_norm_bound,
)
from spectroview.view.components.rgb_live_maps.v_live_map_1d import LiveMap1DMixin
from spectroview.view.components.rgb_live_maps.v_live_map_2d import LiveMap2DMixin
from spectroview.view.components.rgb_live_maps.v_live_map_selection import (
    LiveMapSelectionMixin,
)
from spectroview.view.theme import Icons, current_tokens
from spectroview.viewmodel.common.export_utils import copy_df_to_clb, copy_fig_to_clb


class LiveMapSubpanel(LiveMap1DMixin, LiveMap2DMixin, LiveMapSelectionMixin, QFrame):
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
