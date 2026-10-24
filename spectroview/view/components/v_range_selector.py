"""Spectral Range Selector component for SPECTROview.

Provides draggable vertical range markers (brackets) on Matplotlib axes,
independent toggle buttons for Red, Blue, Green selectors, and hit testing.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any

from PySide6.QtCore import QObject, Signal, Qt, QSize
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QVBoxLayout, QToolButton, QWidget

import numpy as np
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from spectroview.model.spectral_parameters import (
    SELECTOR_RED,
    SELECTOR_BLUE,
    SELECTOR_GREEN,
    SELECTOR_KEYS,
    SELECTOR_COLORS,
    SELECTOR_NAMES,
)


@dataclass
class RangeSelectorState:
    color_id: str
    color_hex: str
    is_active: bool = False
    x_min: float = 0.0
    x_max: float = 100.0
    parameter: str = "Integrated intensity"
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

    def clear_artists(self, ax=None):
        """Remove all selector artists from the axes."""
        for color_id, artist_list in self._artists.items():
            for art in artist_list:
                try:
                    art.remove()
                except Exception:
                    pass
            artist_list.clear()

    def render(self, ax):
        """Render all active range selectors on the given Matplotlib axes.

        Uses ax.add_artist for all patches and lines so that ax.dataLim is never
        polluted, ensuring rescale/autoscale operates exclusively on data lines.
        """
        self.clear_artists(ax)

        # Matplotlib blended transform: x in data coordinates, y in axes coordinates (0 to 1)
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

        # Estimate tick width in data coordinates
        xlim = ax.get_xlim()
        tick_w = abs(xlim[1] - xlim[0]) * 0.015

        for color_id in SELECTOR_KEYS:
            sel = self.selectors[color_id]
            if not sel.is_active:
                continue

            color = sel.color_hex
            x_min = sel.x_min
            x_max = sel.x_max

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
            ax.add_artist(rect)
            self._artists[color_id].append(rect)

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
            ax.add_artist(line_left)
            self._artists[color_id].append(line_left)

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
            ax.add_artist(line_right)
            self._artists[color_id].append(line_right)

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
            for t in (t_left_top, t_left_bot, t_right_top, t_right_bot):
                ax.add_artist(t)
                self._artists[color_id].append(t)

    def hit_test(self, ax, event, zoom_active: bool = False) -> Optional[Tuple[str, str]]:
        """Determine if mouse event is near a selector boundary or body.

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
            sel = self.selectors[color_id]
            if not sel.is_active:
                continue

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
                sel = self.selectors[color_id]
                if not sel.is_active:
                    continue

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
