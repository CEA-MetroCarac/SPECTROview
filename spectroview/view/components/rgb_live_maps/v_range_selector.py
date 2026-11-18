"""Spectral Range Selector component and VSpectraViewer bracket mixin.

Provides draggable vertical range markers (brackets) on Matplotlib axes,
independent toggle buttons for Red, Blue, Green selectors, hit testing,
and Agg background blitting helpers for VSpectraViewer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import numpy as np
from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QToolButton, QVBoxLayout, QWidget

from spectroview.model.spectra.spectral_parameters import (
    PARAM_INTEGRATED_INTENSITY,
    PARAM_RB_RATIO,
    SELECTOR_BLUE,
    SELECTOR_COLORS,
    SELECTOR_GREEN,
    SELECTOR_KEYS,
    SELECTOR_NAMES,
    SELECTOR_RED,
)


@dataclass
class RangeSelectorState:
    color_id: str
    color_hex: str
    is_active: bool = False
    x_min: float = 0.0
    x_max: float = 100.0
    parameter: str = PARAM_INTEGRATED_INTENSITY
    palette: str = "viridis"


def create_color_icon(color_hex: str, size: int = 16, checked: bool = False) -> QIcon:
    """Generate a colored indicator icon for toggle buttons."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    color = QColor(color_hex)
    painter.setBrush(color)
    painter.setPen(Qt.NoPen)
    # Rounded badge
    painter.drawRoundedRect(1, 1, size - 2, size - 2, 3, 3)

    if checked:
        # Draw small inner highlight
        painter.setPen(QColor(255, 255, 255, 200))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(2, 2, size - 4, size - 4, 2, 2)

    painter.end()
    return QIcon(pixmap)


class VRangeSelectorPanel(QWidget):
    """Vertical panel hosting the 3 spectral range toggle buttons."""

    selector_toggled = Signal(str, bool)  # (color_id, is_active)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("rangeSelectorPanel")
        self.setFixedWidth(22)
        self.setStyleSheet("""
            #rangeSelectorPanel {
                background: transparent;
                border: none;
                margin: 0px;
                padding: 0px;
            }
        """)

        self.buttons: Dict[str, QToolButton] = {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignHCenter)

        for color_id in SELECTOR_KEYS:
            color_hex = SELECTOR_COLORS[color_id]
            btn = QToolButton(self)
            btn.setObjectName(f"btn_range_{color_id}")
            btn.setCheckable(True)
            btn.setFixedSize(22, 22)
            btn.setIconSize(QSize(16, 16))
            btn.setIcon(create_color_icon(color_hex, size=16))
            btn.setToolTip(f"Toggle {SELECTOR_NAMES[color_id]} Spectral Range Selector")

            # Style button with color accent when checked
            btn.setStyleSheet(f"""
                QToolButton#btn_range_{color_id} {{
                    border: 1px solid transparent;
                    border-radius: 4px;
                    padding: 0px;
                    margin: 0px;
                    background: transparent;
                }}
                QToolButton#btn_range_{color_id}:hover {{
                    border: 1px solid {color_hex};
                    background: rgba(128, 128, 128, 0.2);
                }}
                QToolButton#btn_range_{color_id}:checked {{
                    border: 1.5px solid {color_hex};
                    background: rgba(128, 128, 128, 0.25);
                }}
            """)

            btn.toggled.connect(lambda checked, cid=color_id: self.selector_toggled.emit(cid, checked))
            self.buttons[color_id] = btn
            layout.addWidget(btn)

    def set_selector_active(self, color_id: str, active: bool):
        """Set button checked state without re-emitting if already set."""
        btn = self.buttons.get(color_id)
        if btn and btn.isChecked() != active:
            btn.blockSignals(True)
            btn.setChecked(active)
            btn.blockSignals(False)


class RangeSelectorManager(QObject):
    """Manages spectral range selectors, Matplotlib artists, and dragging."""

    range_changed = Signal(str, float, float)  # (color_id, x_min, x_max)
    range_toggled = Signal(str, bool)          # (color_id, is_active)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selectors: Dict[str, RangeSelectorState] = {
            SELECTOR_RED: RangeSelectorState(
                color_id=SELECTOR_RED,
                color_hex=SELECTOR_COLORS[SELECTOR_RED],
                is_active=False,
                x_min=200.0,
                x_max=400.0,
            ),
            SELECTOR_BLUE: RangeSelectorState(
                color_id=SELECTOR_BLUE,
                color_hex=SELECTOR_COLORS[SELECTOR_BLUE],
                is_active=False,
                x_min=500.0,
                x_max=700.0,
            ),
            SELECTOR_GREEN: RangeSelectorState(
                color_id=SELECTOR_GREEN,
                color_hex=SELECTOR_COLORS[SELECTOR_GREEN],
                is_active=False,
                x_min=800.0,
                x_max=1000.0,
            ),
        }

        # Drag state
        self._drag_state: Optional[dict] = None
        self._animated: bool = False
        # Track artists created on axes so they can be cleaned up
        self._artists: Dict[str, List[Any]] = {
            SELECTOR_RED: [],
            SELECTOR_BLUE: [],
            SELECTOR_GREEN: [],
        }

    def set_active(self, color_id: str, is_active: bool):
        """Set active state of a selector."""
        if color_id in self.selectors:
            sel = self.selectors[color_id]
            if sel.is_active != is_active:
                sel.is_active = is_active
                self.range_toggled.emit(color_id, is_active)

    def is_active(self, color_id: str) -> bool:
        """Check if selector is active."""
        return self.selectors.get(color_id, RangeSelectorState(color_id, "#fff")).is_active

    def set_parameter(self, color_id: str, parameter: str) -> None:
        """Record the parameter displayed for *color_id* and cancel drag if bracket hides."""
        sel = self.selectors.get(color_id)
        if sel is None:
            return
        sel.parameter = str(parameter)
        if (
            self._drag_state is not None
            and self._drag_state.get("color_id") == color_id
            and not self.is_bracket_visible(color_id)
        ):
            self._drag_state = None

    def get_parameter(self, color_id: str) -> str:
        """Return the configured Live Maps parameter for *color_id*."""
        sel = self.selectors.get(color_id)
        return sel.parameter if sel is not None else PARAM_INTEGRATED_INTENSITY

    def is_bracket_visible(self, color_id: str) -> bool:
        """Return True when the bracket for *color_id* should be drawn on the spectrum axes.

        The Green bracket is hidden when its Live Maps panel displays ``R/B ratio``,
        since that plot depends solely on the Red and Blue brackets.
        """
        sel = self.selectors.get(color_id)
        if sel is None or not sel.is_active:
            return False
        if color_id == SELECTOR_GREEN and sel.parameter == PARAM_RB_RATIO:
            return False
        return True

    def has_visible_brackets(self) -> bool:
        """Return True when at least one bracket is currently visible on the spectrum axes."""
        return any(self.is_bracket_visible(cid) for cid in SELECTOR_KEYS)

    def get_range(self, color_id: str) -> Tuple[float, float]:
        """Return (x_min, x_max) for a selector."""
        sel = self.selectors.get(color_id)
        if sel:
            return (sel.x_min, sel.x_max)
        return (0.0, 100.0)

    def set_range(self, color_id: str, x_min: float, x_max: float, emit: bool = True):
        """Update range coordinates for a selector."""
        sel = self.selectors.get(color_id)
        if sel:
            sel.x_min = min(float(x_min), float(x_max))
            sel.x_max = max(float(x_min), float(x_max))
            if emit:
                self.range_changed.emit(color_id, sel.x_min, sel.x_max)

    def initialize_ranges_from_axis(self, x_axis: np.ndarray):
        """Set default non-overlapping positions partitioned across the x axis."""
        if x_axis is None or len(x_axis) < 2:
            return
        x_start = float(np.min(x_axis))
        x_end = float(np.max(x_axis))
        span = x_end - x_start
        if span <= 0:
            return

        # Red: [15% to 35%]
        self.selectors[SELECTOR_RED].x_min = x_start + 0.15 * span
        self.selectors[SELECTOR_RED].x_max = x_start + 0.35 * span

        # Blue: [40% to 60%]
        self.selectors[SELECTOR_BLUE].x_min = x_start + 0.40 * span
        self.selectors[SELECTOR_BLUE].x_max = x_start + 0.60 * span

        # Green: [65% to 85%]
        self.selectors[SELECTOR_GREEN].x_min = x_start + 0.65 * span
        self.selectors[SELECTOR_GREEN].x_max = x_start + 0.85 * span

    def refresh_or_clamp_ranges(
        self,
        curr_xmin: float,
        curr_xmax: float,
        target_color_id: Optional[str] = None,
    ):
        """Ensure range selectors are positioned within the valid spectrum range [curr_xmin, curr_xmax].

        If a selector lies entirely outside [curr_xmin, curr_xmax] (e.g. after data cropping or
        file loading), it is repositioned to default proportional sub-intervals. If it partially
        exceeds bounds, its edges are clamped.
        """
        if curr_xmin >= curr_xmax:
            return

        span = curr_xmax - curr_xmin
        proportions = {
            SELECTOR_RED: (0.15, 0.35),
            SELECTOR_BLUE: (0.40, 0.60),
            SELECTOR_GREEN: (0.65, 0.85),
        }

        color_keys = [target_color_id] if target_color_id in self.selectors else SELECTOR_KEYS
        for color_id in color_keys:
            sel = self.selectors[color_id]
            p_start, p_end = proportions.get(color_id, (0.2, 0.4))
            default_min = curr_xmin + p_start * span
            default_max = curr_xmin + p_end * span

            # If completely outside or inverted
            if sel.x_max <= curr_xmin or sel.x_min >= curr_xmax or sel.x_min >= sel.x_max:
                sel.x_min = default_min
                sel.x_max = default_max
                self.range_changed.emit(color_id, sel.x_min, sel.x_max)
            else:
                # Partially overlapping: clamp to bounds
                clamped_min = max(curr_xmin, sel.x_min)
                clamped_max = min(curr_xmax, sel.x_max)
                if clamped_min >= clamped_max:
                    sel.x_min = default_min
                    sel.x_max = default_max
                    self.range_changed.emit(color_id, sel.x_min, sel.x_max)
                elif clamped_min != sel.x_min or clamped_max != sel.x_max:
                    sel.x_min = clamped_min
                    sel.x_max = clamped_max
                    self.range_changed.emit(color_id, sel.x_min, sel.x_max)

    def _clear_color_artists(self, color_id: str) -> None:
        """Remove tracked selector artists for a single color channel."""
        artist_list = self._artists.get(color_id)
        if not artist_list:
            return
        for art in artist_list:
            try:
                art.remove()
            except Exception:
                pass
        artist_list.clear()

    def clear_artists(self, ax=None):
        """Remove all selector artists from the axes."""
        for color_id in list(self._artists.keys()):
            self._clear_color_artists(color_id)

    def active_artists(self) -> List[Any]:
        """Return rendered artists for currently visible range selectors."""
        out: List[Any] = []
        for color_id in SELECTOR_KEYS:
            if self.is_bracket_visible(color_id):
                out.extend(self._artists[color_id])
        return out

    def set_animated(self, animated: bool) -> None:
        """Set Matplotlib animated flag on all selector artists for blitting."""
        self._animated = bool(animated)
        for artist_list in self._artists.values():
            for art in artist_list:
                art.set_animated(self._animated)

    def render(self, ax):
        """Render all visible range selectors on the given Matplotlib axes.

        Reuses existing artists in-place when already attached to *ax* so
        interactive bracket dragging does not thrash Matplotlib's artist list.
        Uses ax.add_artist for all patches and lines so that ax.dataLim is never
        polluted, ensuring rescale/autoscale operates exclusively on data lines.
        """
        # Estimate tick width in data coordinates
        xlim = ax.get_xlim()
        tick_w = abs(xlim[1] - xlim[0]) * 0.015
        trans = None

        for color_id in SELECTOR_KEYS:
            sel = self.selectors[color_id]
            if not self.is_bracket_visible(color_id):
                self._clear_color_artists(color_id)
                continue

            x_min = sel.x_min
            x_max = sel.x_max
            artists = self._artists[color_id]

            if len(artists) == 7 and all(getattr(art, "axes", None) is ax for art in artists):
                rect, line_left, line_right, t_l_top, t_l_bot, t_r_top, t_r_bot = artists
                rect.set_x(x_min)
                rect.set_width(x_max - x_min)
                line_left.set_xdata([x_min, x_min])
                line_right.set_xdata([x_max, x_max])
                t_l_top.set_xdata([x_min, x_min + tick_w])
                t_l_bot.set_xdata([x_min, x_min + tick_w])
                t_r_top.set_xdata([x_max, x_max - tick_w])
                t_r_bot.set_xdata([x_max, x_max - tick_w])
                continue

            self._clear_color_artists(color_id)
            if trans is None:
                # Matplotlib blended transform: x in data coordinates, y in axes coordinates (0 to 1)
                trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

            color = sel.color_hex

            # 1. Shaded interval span (mpatches.Rectangle -> does NOT mutate dataLim)
            rect = mpatches.Rectangle(
                (x_min, 0.0),
                x_max - x_min,
                1.0,
                transform=trans,
                facecolor=color,
                edgecolor="none",
                alpha=0.18,
                zorder=3,
            )

            # 2. Left boundary line (Line2D -> does NOT mutate dataLim)
            line_left = mlines.Line2D(
                [x_min, x_min],
                [0.0, 1.0],
                transform=trans,
                color=color,
                linestyle="--",
                linewidth=1.5,
                zorder=4,
            )

            # 3. Right boundary line
            line_right = mlines.Line2D(
                [x_max, x_max],
                [0.0, 1.0],
                transform=trans,
                color=color,
                linestyle="--",
                linewidth=1.5,
                zorder=4,
            )

            # 4. Bracket tick heads (top and bottom)
            t_left_top = mlines.Line2D(
                [x_min, x_min + tick_w], [0.97, 0.97],
                transform=trans, color=color, linewidth=2.0, zorder=5
            )
            t_left_bot = mlines.Line2D(
                [x_min, x_min + tick_w], [0.03, 0.03],
                transform=trans, color=color, linewidth=2.0, zorder=5
            )
            t_right_top = mlines.Line2D(
                [x_max, x_max - tick_w], [0.97, 0.97],
                transform=trans, color=color, linewidth=2.0, zorder=5
            )
            t_right_bot = mlines.Line2D(
                [x_max, x_max - tick_w], [0.03, 0.03],
                transform=trans, color=color, linewidth=2.0, zorder=5
            )

            created = (rect, line_left, line_right, t_left_top, t_left_bot, t_right_top, t_right_bot)
            for art in created:
                if self._animated:
                    art.set_animated(True)
                ax.add_artist(art)
            artists.extend(created)

    def hit_test(self, ax, event, zoom_active: bool = False) -> Optional[Tuple[str, str]]:
        """Determine if mouse event is near a visible selector boundary or body.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The active plot axes.
        event : matplotlib.backend_bases.MouseEvent
            The mouse event.
        zoom_active : bool, default False
            If True, only boundary lines are testable so that Zoom tool
            can operate freely across the spectrum body without conflict.

        Returns
        -------
        tuple of (color_id, 'min' | 'max' | 'body') or None.
        """
        if event.inaxes != ax or event.x is None or event.xdata is None:
            return None

        tol_px = 10.0  # Pixel tolerance for grabbing boundary line

        # First check boundary lines (higher priority than span body)
        for color_id in SELECTOR_KEYS:
            if not self.is_bracket_visible(color_id):
                continue
            sel = self.selectors[color_id]

            try:
                px_min, _ = ax.transData.transform((sel.x_min, 0))
                px_max, _ = ax.transData.transform((sel.x_max, 0))
            except Exception:
                continue

            if abs(event.x - px_min) <= tol_px:
                return (color_id, "min")
            if abs(event.x - px_max) <= tol_px:
                return (color_id, "max")

        # Second check if clicked inside span body (only when zoom is NOT active)
        if not zoom_active:
            for color_id in SELECTOR_KEYS:
                if not self.is_bracket_visible(color_id):
                    continue
                sel = self.selectors[color_id]

                x0 = min(sel.x_min, sel.x_max)
                x1 = max(sel.x_min, sel.x_max)
                if x0 <= event.xdata <= x1:
                    return (color_id, "body")

        return None

    def start_drag(self, color_id: str, target: str, event_xdata: float):
        """Begin dragging a selector boundary or body."""
        sel = self.selectors.get(color_id)
        if not sel:
            return
        self._drag_state = {
            "color_id": color_id,
            "target": target,
            "start_x": event_xdata,
            "orig_min": sel.x_min,
            "orig_max": sel.x_max,
        }

    def is_dragging(self) -> bool:
        """Return True if currently dragging a selector."""
        return self._drag_state is not None

    def update_drag(self, event_xdata: float) -> Optional[str]:
        """Update range coordinates during drag.

        Returns
        -------
        color_id if updated, None otherwise.
        """
        if not self._drag_state or event_xdata is None:
            return None

        color_id = self._drag_state["color_id"]
        target = self._drag_state["target"]
        sel = self.selectors[color_id]

        if target == "min":
            sel.x_min = event_xdata
        elif target == "max":
            sel.x_max = event_xdata
        elif target == "body":
            dx = event_xdata - self._drag_state["start_x"]
            sel.x_min = self._drag_state["orig_min"] + dx
            sel.x_max = self._drag_state["orig_max"] + dx

        # Ensure min <= max
        if sel.x_min > sel.x_max and target in ("min", "max"):
            # Swap if user drags past the opposite boundary
            sel.x_min, sel.x_max = sel.x_max, sel.x_min
            self._drag_state["target"] = "max" if target == "min" else "min"

        self.range_changed.emit(color_id, sel.x_min, sel.x_max)
        return color_id

    def finish_drag(self) -> Optional[str]:
        """Finish active drag gesture."""
        if not self._drag_state:
            return None
        color_id = self._drag_state["color_id"]
        sel = self.selectors[color_id]
        # Normalize bounds
        if sel.x_min > sel.x_max:
            sel.x_min, sel.x_max = sel.x_max, sel.x_min
        self._drag_state = None
        self.range_changed.emit(color_id, sel.x_min, sel.x_max)
        return color_id


class SpectraRangeBracketMixin:
    """Mixin encapsulating RGB range-bracket UI wiring and blitted dragging for VSpectraViewer."""

    def _init_range_brackets(self) -> None:
        """Initialize the RangeSelectorManager and drag/blitting state on the host viewer."""
        self.range_manager = RangeSelectorManager(self)
        self.range_manager.range_toggled.connect(self.range_toggled.emit)
        self.range_manager.range_changed.connect(self.range_changed.emit)
        self._range_drag_cids: List[int] = []
        self._range_hover_active: bool = False
        self._ranges_initialized: bool = False
        self._range_drag_bg = None
        self._range_drag_bg_size = None
        self._suspended_range_layout = None

    def _create_range_selector_panel(self, parent_panel: QWidget) -> VRangeSelectorPanel:
        """Create the vertical R/B/G toggle panel and wire shortcut button references."""
        self.range_panel = VRangeSelectorPanel(parent_panel)
        self.range_panel.selector_toggled.connect(self._toggle_range_selector)
        self.btn_range_red = self.range_panel.buttons[SELECTOR_RED]
        self.btn_range_blue = self.range_panel.buttons[SELECTOR_BLUE]
        self.btn_range_green = self.range_panel.buttons[SELECTOR_GREEN]
        return self.range_panel

    def _sync_ranges_with_plot_data(self, x_arr) -> None:
        """Initialize or clamp range selector bounds when spectral plot data updates."""
        if x_arr is None:
            return
        first_x = x_arr[0] if isinstance(x_arr, list) and len(x_arr) > 0 else x_arr
        if hasattr(first_x, "__len__") and len(first_x) > 1:
            if not getattr(self, "_ranges_initialized", False):
                self.range_manager.initialize_ranges_from_axis(first_x)
                self._ranges_initialized = True
            else:
                self.range_manager.refresh_or_clamp_ranges(
                    float(np.min(first_x)), float(np.max(first_x))
                )

    def _handle_range_mouse_press(self, event) -> bool:
        """Start dragging a visible range bracket if clicked; return True when handled."""
        if not hasattr(self, "range_manager") or not self.range_manager.has_visible_brackets():
            return False
        hit = self.range_manager.hit_test(self.ax, event, zoom_active=self.zoom_pan_active)
        if not hit or event.button != 1:
            return False

        # Suppress concurrent Matplotlib zoom interaction if active
        if hasattr(self.toolbar, "_zoom_info") and self.toolbar._zoom_info is not None:
            try:
                if hasattr(self.toolbar._zoom_info, "cid"):
                    self.canvas.mpl_disconnect(self.toolbar._zoom_info.cid)
                self.toolbar.remove_rubberband()
                self.toolbar._zoom_info = None
            except Exception:
                pass

        color_id, target = hit
        self.range_manager.start_drag(color_id, target, event.xdata)
        self._prepare_range_drag_blit()
        self.canvas.setCursor(Qt.ClosedHandCursor if target == "body" else Qt.SizeHorCursor)
        cid_m = self.canvas.mpl_connect("motion_notify_event", self._on_range_drag_motion)
        cid_r = self.canvas.mpl_connect("button_release_event", self._on_range_drag_release)
        self._range_drag_cids = [cid_m, cid_r]
        return True

    def _handle_range_hover(self, event) -> bool:
        """Update cursor feedback when hovering over a visible range bracket."""
        if not hasattr(self, "range_manager"):
            return False
        if self.range_manager.has_visible_brackets():
            hit = self.range_manager.hit_test(self.ax, event, zoom_active=self.zoom_pan_active)
            if hit:
                if hit[1] in ("min", "max"):
                    self.canvas.setCursor(Qt.SizeHorCursor)
                    self._range_hover_active = True
                    return True
                if hit[1] == "body":
                    self.canvas.setCursor(Qt.OpenHandCursor)
                    self._range_hover_active = True
                    return True
        if getattr(self, "_range_hover_active", False):
            self.canvas.unsetCursor()
            self._range_hover_active = False
        return False

    def set_range_selectors_visible(self, visible: bool):
        """Show or hide the spectral range selector panel on the right toolbar.

        When hidden, any active selectors are cleanly deactivated.
        """
        self.range_panel.setVisible(visible)
        if not visible:
            self.deactivate_all_range_selectors()

    def deactivate_all_range_selectors(self):
        """Deactivate all range selectors and clear their visual markers."""
        if self._range_drag_bg is not None or self._suspended_range_layout is not None:
            self._end_range_drag_blit()
        for color_id, btn in self.range_panel.buttons.items():
            if btn.isChecked():
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
                self.range_manager.set_active(color_id, False)
        if self.ax is not None:
            self.range_manager.render(self.ax)
            self.canvas.draw_idle()

    def set_range_selector_parameter(self, color_id: str, parameter: str) -> None:
        """Update the Live Maps parameter for a selector and refresh bracket visibility."""
        if not hasattr(self, "range_manager"):
            return
        if self._range_drag_bg is not None or self._suspended_range_layout is not None:
            self._end_range_drag_blit()
        was_dragging = self.range_manager.is_dragging()
        self.range_manager.set_parameter(color_id, parameter)
        if was_dragging and not self.range_manager.is_dragging():
            for cid in getattr(self, "_range_drag_cids", []):
                try:
                    self.canvas.mpl_disconnect(cid)
                except Exception:
                    pass
            if hasattr(self, "_range_drag_cids"):
                self._range_drag_cids.clear()
            if self.canvas is not None:
                self.canvas.unsetCursor()
            self._range_hover_active = False
        elif getattr(self, "_range_hover_active", False) and not self.range_manager.has_visible_brackets():
            if self.canvas is not None:
                self.canvas.unsetCursor()
            self._range_hover_active = False
        if self.ax is not None:
            self.range_manager.render(self.ax)
            if self.canvas is not None:
                self.canvas.draw_idle()

    def _get_current_data_xrange(self) -> Tuple[Optional[float], Optional[float]]:
        """Return (min_x, max_x) of currently loaded spectra data."""
        if self._tensor_data:
            x_arr = self._tensor_data.get("x")
            if x_arr is not None:
                if isinstance(x_arr, list) and len(x_arr) > 0:
                    mins = [float(np.min(xa)) for xa in x_arr if hasattr(xa, "__len__") and len(xa) > 1]
                    maxs = [float(np.max(xa)) for xa in x_arr if hasattr(xa, "__len__") and len(xa) > 1]
                    if mins and maxs:
                        return min(mins), max(maxs)
                elif hasattr(x_arr, "__len__") and len(x_arr) > 1:
                    return float(np.min(x_arr)), float(np.max(x_arr))
        if self._current_spectra:
            x_vals = []
            for s in self._current_spectra:
                if hasattr(s, "x") and s.x is not None and len(s.x) > 0:
                    x_vals.extend([float(np.min(s.x)), float(np.max(s.x))])
            if x_vals:
                return min(x_vals), max(x_vals)
        if self.ax is not None:
            xlim = self.ax.get_xlim()
            if xlim[0] < xlim[1]:
                return float(xlim[0]), float(xlim[1])
        return None, None

    def _toggle_range_selector(self, color_id: str, checked: bool):
        """Activate or deactivate one of the spectral range selectors."""
        if self._range_drag_bg is not None or self._suspended_range_layout is not None:
            self._end_range_drag_blit()
        if checked and self.btn_zoom.isChecked():
            self.btn_zoom.blockSignals(True)
            self.btn_zoom.setChecked(False)
            self.btn_zoom.blockSignals(False)
            self._set_zoom_tool_active(False)

        if checked:
            curr_xmin, curr_xmax = self._get_current_data_xrange()
            if curr_xmin is not None and curr_xmax is not None:
                self.range_manager.refresh_or_clamp_ranges(curr_xmin, curr_xmax, target_color_id=color_id)

        self.range_manager.set_active(color_id, checked)
        if self.ax is not None:
            self.range_manager.render(self.ax)
            self.canvas.draw_idle()

    def _prepare_range_drag_blit(self) -> bool:
        """Capture clean static canvas background once before blitted bracket dragging."""
        self._end_range_drag_blit()
        if self.ax is None or self.canvas is None:
            return False
        if not (
            hasattr(self.canvas, "copy_from_bbox")
            and hasattr(self.canvas, "restore_region")
            and hasattr(self.canvas, "blit")
        ):
            return False
        try:
            w = int(self.canvas.width())
            h = int(self.canvas.height())
            if w <= 0 or h <= 0:
                return False

            # Ensure selector artists exist, then mark them animated before capturing
            # the background so the initial bracket position isn't baked into the buffer.
            if not self.range_manager.active_artists():
                self.range_manager.render(self.ax)
            self.range_manager.set_animated(True)
            self.canvas.draw()

            # Suspend constrained/compressed layout during drag in case any fallback redraw runs.
            layout_engine = self.figure.get_layout_engine()
            if layout_engine is not None:
                self._suspended_range_layout = layout_engine
                self.figure.set_layout_engine(None)

            self._range_drag_bg = self.canvas.copy_from_bbox(self.figure.bbox)
            self._range_drag_bg_size = (w, h)
            # Re-paint selector artists over the freshly captured clean buffer.
            self._blit_range_selectors()
            return True
        except Exception:
            self._end_range_drag_blit()
            return False

    def _blit_range_selectors(self) -> bool:
        """Redraw only range-selector artists on top of the cached background."""
        if self.ax is None or self.canvas is None or self._range_drag_bg is None:
            return False
        try:
            cur_size = (int(self.canvas.width()), int(self.canvas.height()))
            if cur_size != self._range_drag_bg_size:
                if not self._prepare_range_drag_blit():
                    return False
            self.canvas.restore_region(self._range_drag_bg)
            for art in self.range_manager.active_artists():
                self.ax.draw_artist(art)
            self.canvas.blit(self.figure.bbox)
            return True
        except Exception:
            self._end_range_drag_blit()
            return False

    def _end_range_drag_blit(self) -> None:
        """Restore layout engine and normal non-animated drawing after range drag."""
        if getattr(self, "_suspended_range_layout", None) is not None:
            try:
                if self.figure.get_layout_engine() is None:
                    self.figure.set_layout_engine(self._suspended_range_layout)
            except Exception:
                pass
            self._suspended_range_layout = None
        self._range_drag_bg = None
        self._range_drag_bg_size = None
        if hasattr(self, "range_manager"):
            self.range_manager.set_animated(False)

    def _event_to_clamped_xdata(self, event) -> Optional[float]:
        """Extract or invert event X coordinate clamped to current axes X limits."""
        if self.ax is None:
            return None
        xdata = event.xdata if event.inaxes == self.ax and event.xdata is not None else None
        if xdata is None:
            if event.x is None or event.y is None:
                return None
            try:
                xdata = float(self.ax.transData.inverted().transform((event.x, event.y))[0])
            except Exception:
                return None
        xlim = self.ax.get_xlim()
        lo, hi = min(float(xlim[0]), float(xlim[1])), max(float(xlim[0]), float(xlim[1]))
        return float(np.clip(xdata, lo, hi))

    def _on_range_drag_motion(self, event):
        """Update range selector boundaries while user drags."""
        if not hasattr(self, "range_manager") or not self.range_manager.is_dragging():
            return
        xdata = self._event_to_clamped_xdata(event)
        if xdata is None:
            return
        color_id = self.range_manager.update_drag(xdata)
        if color_id and self.ax is not None:
            self.range_manager.render(self.ax)
            if not self._blit_range_selectors():
                self.canvas.draw_idle()

    def _on_range_drag_release(self, event):
        """Finalize range selector dragging on mouse button release."""
        if hasattr(self, "_range_drag_cids"):
            for cid in self._range_drag_cids:
                try:
                    self.canvas.mpl_disconnect(cid)
                except Exception:
                    pass
            self._range_drag_cids.clear()

        # Suppress any lingering Matplotlib zoom interaction
        if hasattr(self.toolbar, "_zoom_info") and self.toolbar._zoom_info is not None:
            try:
                if hasattr(self.toolbar._zoom_info, "cid"):
                    self.canvas.mpl_disconnect(self.toolbar._zoom_info.cid)
                self.toolbar.remove_rubberband()
                self.toolbar._zoom_info = None
            except Exception:
                pass

        if hasattr(self, "range_manager") and self.range_manager.is_dragging():
            color_id = self.range_manager.finish_drag()
            self._end_range_drag_blit()
            if self.ax is not None:
                self.range_manager.render(self.ax)
                self.canvas.draw_idle()
            self.canvas.unsetCursor()
            self._range_hover_active = False
            if color_id:
                self.range_drag_finished.emit(color_id)
        elif self._range_drag_bg is not None or self._suspended_range_layout is not None:
            self._end_range_drag_blit()

