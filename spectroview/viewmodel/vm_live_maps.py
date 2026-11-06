"""ViewModel for Live Maps feature in SPECTROview.

Orchestrates range state, parameter selection, debounced background calculation,
and communication with SpectraStore and MapData.
"""

from __future__ import annotations
from typing import Optional, Dict, Set

import numpy as np

from PySide6.QtCore import QObject, Signal, QTimer

from spectroview.model.m_settings import MSettings
from spectroview.model.spectra_store import MapData
from spectroview.model.spectral_parameters import (
    PARAM_INTEGRATED_INTENSITY,
    SELECTOR_RED,
    SELECTOR_BLUE,
    SELECTOR_GREEN,
    compute_spectral_parameter,
    compute_series_spectral_parameter,
    compute_map_grid,
)


class VMLiveMaps(QObject):
    """ViewModel orchestrating Live Maps (2D heatmaps) and Live Series (1D trend graphs)."""

    # (color_id, grid_2d, extent, vmin, vmax, status_msg)
    map_data_ready = Signal(str, object, list, float, float, str)
    # (color_id, values_1d, labels, status_msg)
    series_data_ready = Signal(str, object, list, str)
    # (color_id, is_active, x_min, x_max)
    selector_state_changed = Signal(str, bool, float, float)

    def __init__(self, settings: Optional[MSettings] = None, parent=None, mode: str = "map"):
        super().__init__(parent)
        self.settings = settings
        self.mode = mode  # "map" (Maps workspace) or "spectra" (Spectra workspace)

        # Selector states: {color_id: {'active': bool, 'range': (min, max), 'param': str, 'user_set': bool}}
        self._states: Dict[str, dict] = {
            SELECTOR_RED: {
                "active": False,
                "range": (200.0, 400.0),
                "param": PARAM_INTEGRATED_INTENSITY,
                "user_set": False,
            },
            SELECTOR_BLUE: {
                "active": False,
                "range": (500.0, 700.0),
                "param": PARAM_INTEGRATED_INTENSITY,
                "user_set": False,
            },
            SELECTOR_GREEN: {
                "active": False,
                "range": (800.0, 1000.0),
                "param": PARAM_INTEGRATED_INTENSITY,
                "user_set": False,
            },
        }

        # Current MapData reference (Maps workspace) or list of MapData (Spectra workspace)
        self._current_md: Optional[MapData] = None
        self._current_series_mds: list[MapData] = []

        # Debounced calculation during interactive dragging
        self._pending_calc: Set[str] = set()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)  # 60 ms debounce for fluid 60fps UI
        self._timer.timeout.connect(self._recalculate_pending)

    def set_mode(self, mode: str):
        """Switch between 'map' (2D spatial heatmap) and 'spectra' (1D variation series) modes."""
        self.mode = mode

    def _sync_or_init_ranges_from_axis(self, x_axis: np.ndarray):
        """Initialize or clamp selector intervals against a spectral axis.

        Preserves user-adjusted intervals if they fall within the axis domain.
        """
        if x_axis is None or len(x_axis) < 2:
            return
        x_min = float(np.min(x_axis))
        x_max = float(np.max(x_axis))
        span = x_max - x_min
        if span <= 0:
            return

        proportions = {
            SELECTOR_RED: (0.15, 0.35),
            SELECTOR_BLUE: (0.40, 0.60),
            SELECTOR_GREEN: (0.65, 0.85),
        }
        for cid, (p0, p1) in proportions.items():
            state = self._states[cid]
            r0, r1 = state["range"]
            if not state.get("user_set", False) or r1 <= x_min or r0 >= x_max or r0 >= r1:
                state["range"] = (x_min + p0 * span, x_min + p1 * span)
            else:
                c0 = max(x_min, r0)
                c1 = min(x_max, r1)
                if c0 < c1:
                    state["range"] = (c0, c1)
                else:
                    state["range"] = (x_min + p0 * span, x_min + p1 * span)

    def set_map_data(self, md: Optional[MapData]):
        """Update active MapData and trigger recalculation for all active selectors."""
        self._current_md = md
        if md is not None and md.x_axis is not None and len(md.x_axis) >= 2:
            self._sync_or_init_ranges_from_axis(md.x_axis)

        # Recalculate all currently active selectors
        for cid, state in self._states.items():
            if state["active"]:
                self.recalculate(cid)

    def set_series_data(self, mds: Optional[list[MapData]]):
        """Update selected spectra series and trigger recalculation for all active selectors."""
        self._current_series_mds = [md for md in (mds or []) if md is not None]
        if self._current_series_mds:
            first_md = self._current_series_mds[0]
            if first_md.x_axis is not None and len(first_md.x_axis) >= 2:
                self._sync_or_init_ranges_from_axis(first_md.x_axis)

        # Recalculate all currently active selectors
        for cid, state in self._states.items():
            if state["active"]:
                self.recalculate(cid)

    def set_selector_active(self, color_id: str, is_active: bool):
        """Toggle active state of a selector."""
        if color_id not in self._states:
            return
        state = self._states[color_id]
        state["active"] = is_active
        x_min, x_max = state["range"]
        self.selector_state_changed.emit(color_id, is_active, x_min, x_max)

        if is_active:
            self.recalculate(color_id)

    def set_selector_range(self, color_id: str, x_min: float, x_max: float, immediate: bool = False):
        """Update spectral interval for a selector."""
        if color_id not in self._states:
            return
        state = self._states[color_id]
        state["range"] = (min(float(x_min), float(x_max)), max(float(x_min), float(x_max)))
        state["user_set"] = True
        self.selector_state_changed.emit(color_id, state["active"], state["range"][0], state["range"][1])

        if state["active"]:
            if immediate:
                self._pending_calc.discard(color_id)
                self.recalculate(color_id)
            else:
                self._pending_calc.add(color_id)
                self._timer.start()

    def set_selector_parameter(self, color_id: str, parameter: str):
        """Update spectral parameter choice and immediately recalculate."""
        if color_id not in self._states:
            return
        state = self._states[color_id]
        state["param"] = parameter
        if state["active"]:
            self.recalculate(color_id)

    def _recalculate_pending(self):
        """Execute debounced calculation for queued selectors."""
        to_calc = list(self._pending_calc)
        self._pending_calc.clear()
        for cid in to_calc:
            if self._states[cid]["active"]:
                self.recalculate(cid)

    def recalculate(self, color_id: str):
        """Compute spectral parameter for a single selector in either 'map' or 'spectra' mode."""
        if color_id not in self._states:
            return

        state = self._states[color_id]
        x_range = state["range"]
        param = state["param"]

        if self.mode == "spectra":
            mds = self._current_series_mds
            if not mds:
                self.series_data_ready.emit(
                    color_id, np.array([]), [], "No spectra selected"
                )
                return

            vals, labels, is_valid, msg = compute_series_spectral_parameter(
                mds=mds,
                x_range=x_range,
                parameter=param,
            )
            if not is_valid:
                self.series_data_ready.emit(
                    color_id, np.array([]), labels, msg or "Unable to calculate parameter"
                )
                return

            self.series_data_ready.emit(color_id, vals, labels, msg)
            return

        md = self._current_md
        if md is None:
            self.map_data_ready.emit(
                color_id, None, [0.0, 1.0, 0.0, 1.0], 0.0, 1.0,
                "No hyperspectral map loaded"
            )
            return

        x_axis = md.x_axis
        Y = md.y_matrix
        coords = md.coords

        if x_axis is None or Y is None or coords is None or len(coords) == 0:
            self.map_data_ready.emit(
                color_id, None, [0.0, 1.0, 0.0, 1.0], 0.0, 1.0,
                "Dataset does not contain spatial (X, Y) coordinates"
            )
            return

        # Compute parameter values across all spectra
        vals, is_valid, msg = compute_spectral_parameter(
            x_axis=x_axis,
            Y=Y,
            x_range=x_range,
            parameter=param,
            baseline_subtracted=getattr(md, "is_baseline_subtracted", False),
            peak_params=md.peak_params if md.has_fit_results() else None,
            param_names=md.param_names if md.has_fit_results() else None,
            fit_model=md.fit_model,
        )

        if not is_valid:
            self.map_data_ready.emit(
                color_id, None, [0.0, 1.0, 0.0, 1.0], 0.0, 1.0,
                msg or "Unable to calculate parameter"
            )
            return

        # Compute 2D map grid
        grid, extent, vmin, vmax = compute_map_grid(coords, vals)

        self.map_data_ready.emit(
            color_id, grid, extent, vmin, vmax, msg
        )
