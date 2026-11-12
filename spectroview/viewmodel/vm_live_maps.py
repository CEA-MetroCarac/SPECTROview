"""ViewModel for Live Maps feature in SPECTROview.

Orchestrates range state, parameter selection, debounced background calculation,
and communication with SpectraStore and MapData.

Output adapts to the data being visualized:
- ``"spectra"`` mode → 1D trend across the selected spectra (``series_data_ready``).
- ``"map"`` mode, 2D map → 2D spatial heatmap (``map_data_ready``).
- ``"map"`` mode, 1D profile (collinear sites) → line plot along the profile
  (``profile_data_ready``). Dimensionality is derived from ``MapData.coords``.

The Green channel can alternatively display the Red/Blue integrated-intensity
ratio (``PARAM_RB_RATIO``), recomputed whenever the Red or Blue range changes.
"""

from __future__ import annotations
from typing import Optional, Dict, Set, Iterable, Tuple

import numpy as np

from PySide6.QtCore import QObject, Signal, QTimer

from spectroview.model.m_settings import MSettings
from spectroview.model.spectra_store import MapData
from spectroview.model.spectral_parameters import (
    PARAM_INTEGRATED_INTENSITY,
    PARAM_RB_RATIO,
    SELECTOR_RED,
    SELECTOR_BLUE,
    SELECTOR_GREEN,
    SELECTOR_KEYS,
    MapGeometry,
    analyze_map_geometry,
    compute_intensity_ratio,
    compute_spectral_parameter,
    compute_series_spectral_parameter,
    compute_map_grid,
)

_EMPTY_EXTENT = [0.0, 1.0, 0.0, 1.0]
MSG_RATIO_NEEDS_RB = "Activate the Red and Blue range selectors to display the R/B ratio"

# (values, labels, is_valid, message)
_Result = Tuple[np.ndarray, list, bool, str]


class VMLiveMaps(QObject):
    """ViewModel orchestrating Live Maps (2D heatmaps / 1D profiles) and Live Series (1D trend graphs)."""

    # (color_id, grid_2d, extent, vmin, vmax, status_msg)
    map_data_ready = Signal(str, object, list, float, float, str)
    # (color_id, values_1d, labels, status_msg)
    series_data_ready = Signal(str, object, list, str)
    # (color_id, positions_1d, values_1d, axis_label, status_msg) — 1D profile maps
    profile_data_ready = Signal(str, object, object, str, str)
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

        # Map dimensionality cache: (coords array reference, MapGeometry)
        self._geometry_cache: Optional[Tuple[np.ndarray, MapGeometry]] = None

        # Integrated intensities computed during the current recalculation pass,
        # keyed by color_id → (range, result). Pass-scoped (cleared before and
        # after each batch) so in-place data edits can never be served stale.
        self._intensity_cache: Dict[str, Tuple[Tuple[float, float], _Result]] = {}

        # Debounced calculation during interactive dragging
        self._pending_calc: Set[str] = set()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)  # 60 ms debounce for fluid 60fps UI
        self._timer.timeout.connect(self._recalculate_pending)

    def set_mode(self, mode: str):
        """Switch between 'map' (2D spatial heatmap) and 'spectra' (1D variation series) modes."""
        self.mode = mode

    # ── Dependency helpers ────────────────────────────────────────────

    def _is_ratio_display(self, color_id: str = SELECTOR_GREEN) -> bool:
        """True if *color_id* (only Green supports it) currently displays the R/B ratio."""
        return color_id == SELECTOR_GREEN and self._states[SELECTOR_GREEN]["param"] == PARAM_RB_RATIO

    def _dependents(self, color_id: str) -> Set[str]:
        """Selectors whose display depends on *color_id*'s range/state (besides itself)."""
        if (
            color_id in (SELECTOR_RED, SELECTOR_BLUE)
            and self._states[SELECTOR_GREEN]["active"]
            and self._is_ratio_display()
        ):
            return {SELECTOR_GREEN}
        return set()

    def get_map_geometry(self) -> MapGeometry:
        """Return the (cached) spatial dimensionality of the current map."""
        md = self._current_md
        coords = getattr(md, "coords", None) if md is not None else None
        cached = self._geometry_cache
        if cached is not None and coords is not None and cached[0] is coords:
            return cached[1]
        geom = analyze_map_geometry(coords)
        self._geometry_cache = (coords, geom) if coords is not None else None
        return geom

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

    # ── Public inputs ─────────────────────────────────────────────────

    def set_map_data(self, md: Optional[MapData]):
        """Update active MapData and trigger recalculation for all active selectors."""
        self._current_md = md
        if md is not None and md.x_axis is not None and len(md.x_axis) >= 2:
            self._sync_or_init_ranges_from_axis(md.x_axis)
        self._recalculate_batch(cid for cid, s in self._states.items() if s["active"])

    def set_series_data(self, mds: Optional[list[MapData]]):
        """Update selected spectra series and trigger recalculation for all active selectors."""
        self._current_series_mds = [md for md in (mds or []) if md is not None]
        if self._current_series_mds:
            first_md = self._current_series_mds[0]
            if first_md.x_axis is not None and len(first_md.x_axis) >= 2:
                self._sync_or_init_ranges_from_axis(first_md.x_axis)
        self._recalculate_batch(cid for cid, s in self._states.items() if s["active"])

    def set_selector_active(self, color_id: str, is_active: bool):
        """Toggle active state of a selector."""
        if color_id not in self._states:
            return
        state = self._states[color_id]
        state["active"] = is_active
        x_min, x_max = state["range"]
        self.selector_state_changed.emit(color_id, is_active, x_min, x_max)

        to_calc = set(self._dependents(color_id))
        if is_active:
            to_calc.add(color_id)
        self._recalculate_batch(to_calc)

    def set_selector_range(self, color_id: str, x_min: float, x_max: float, immediate: bool = False):
        """Update spectral interval for a selector."""
        if color_id not in self._states:
            return
        state = self._states[color_id]
        state["range"] = (min(float(x_min), float(x_max)), max(float(x_min), float(x_max)))
        state["user_set"] = True
        self.selector_state_changed.emit(color_id, state["active"], state["range"][0], state["range"][1])

        if not state["active"]:
            return
        to_calc = {color_id} | self._dependents(color_id)
        if immediate:
            self._pending_calc.difference_update(to_calc)
            self._recalculate_batch(to_calc)
        else:
            self._pending_calc.update(to_calc)
            self._timer.start()

    def set_selector_parameter(self, color_id: str, parameter: str):
        """Update spectral parameter choice and immediately recalculate."""
        if color_id not in self._states:
            return
        state = self._states[color_id]
        state["param"] = parameter
        if state["active"]:
            self.recalculate(color_id)

    # ── Calculation ───────────────────────────────────────────────────

    def _recalculate_pending(self):
        """Execute debounced calculation for queued selectors."""
        to_calc = list(self._pending_calc)
        self._pending_calc.clear()
        self._recalculate_batch(cid for cid in to_calc if self._states[cid]["active"])

    def recalculate(self, color_id: str):
        """Compute and emit the display data for a single selector."""
        if color_id not in self._states:
            return
        self._recalculate_batch([color_id])

    def _recalculate_batch(self, color_ids: Iterable[str]):
        """Recalculate several selectors in one pass, sharing integrated intensities.

        Selectors are processed in Red → Blue → Green order so that an R/B
        ratio displayed in Green reuses the Red/Blue intensities just computed.
        """
        wanted = set(color_ids)
        if not wanted:
            return
        self._intensity_cache.clear()
        try:
            for cid in SELECTOR_KEYS:
                if cid in wanted:
                    self._recalculate_one(cid)
        finally:
            self._intensity_cache.clear()

    def _recalculate_one(self, color_id: str):
        state = self._states[color_id]

        if self.mode == "spectra":
            if not self._current_series_mds:
                self.series_data_ready.emit(color_id, np.array([]), [], "No spectra selected")
                return
        else:
            md = self._current_md
            if md is None:
                self.map_data_ready.emit(
                    color_id, None, list(_EMPTY_EXTENT), 0.0, 1.0, "No hyperspectral map loaded"
                )
                return
            if md.x_axis is None or md.y_matrix is None or md.coords is None or len(md.coords) == 0:
                self.map_data_ready.emit(
                    color_id, None, list(_EMPTY_EXTENT), 0.0, 1.0,
                    "Dataset does not contain spatial (X, Y) coordinates"
                )
                return

        if self._is_ratio_display(color_id):
            result = self._compute_rb_ratio()
        elif state["param"] == PARAM_INTEGRATED_INTENSITY:
            result = self._integrated_intensity(color_id)
        else:
            result = self._compute_values(state["range"], state["param"])
        self._emit_result(color_id, *result)

    def _compute_values(self, x_range: Tuple[float, float], param: str) -> _Result:
        """Evaluate *param* over *x_range* on the current data (both modes)."""
        if self.mode == "spectra":
            vals, labels, ok, msg = compute_series_spectral_parameter(
                mds=self._current_series_mds, x_range=x_range, parameter=param,
            )
            return vals, labels, ok, msg

        md = self._current_md
        has_fit = md.has_fit_results()
        vals, ok, msg = compute_spectral_parameter(
            x_axis=md.x_axis,
            Y=md.y_matrix,
            x_range=x_range,
            parameter=param,
            baseline_subtracted=getattr(md, "is_baseline_subtracted", False),
            peak_params=md.peak_params if has_fit else None,
            param_names=md.param_names if has_fit else None,
            fit_model=md.fit_model,
        )
        return vals, [], ok, msg

    def _integrated_intensity(self, color_id: str) -> _Result:
        """Integrated intensity over *color_id*'s range, memoized within the current pass."""
        rng = self._states[color_id]["range"]
        cached = self._intensity_cache.get(color_id)
        if cached is not None and cached[0] == rng:
            return cached[1]
        result = self._compute_values(rng, PARAM_INTEGRATED_INTENSITY)
        self._intensity_cache[color_id] = (rng, result)
        return result

    def _compute_rb_ratio(self) -> _Result:
        """Point-wise ratio I_Red / I_Blue of the integrated intensities."""
        if not (self._states[SELECTOR_RED]["active"] and self._states[SELECTOR_BLUE]["active"]):
            return np.array([], dtype=np.float64), [], False, MSG_RATIO_NEEDS_RB

        r_vals, labels, r_ok, r_msg = self._integrated_intensity(SELECTOR_RED)
        if not r_ok:
            return r_vals, labels, False, f"Red range: {r_msg}"
        b_vals, _, b_ok, b_msg = self._integrated_intensity(SELECTOR_BLUE)
        if not b_ok:
            return b_vals, labels, False, f"Blue range: {b_msg}"
        if len(r_vals) != len(b_vals):
            return np.array([], dtype=np.float64), labels, False, "Red/Blue size mismatch"

        ratio, n_masked = compute_intensity_ratio(r_vals, b_vals)
        if n_masked == ratio.size:
            return ratio, labels, False, "R/B ratio undefined: Blue intensity is zero"
        msg = f"{n_masked} point(s) with ~zero Blue intensity masked" if n_masked else "OK"
        return ratio, labels, True, msg

    def _emit_result(self, color_id: str, vals: np.ndarray, labels: list, ok: bool, msg: str):
        """Route computed values to the signal matching the data dimensionality."""
        if self.mode == "spectra":
            if not ok:
                self.series_data_ready.emit(
                    color_id, np.array([]), labels, msg or "Unable to calculate parameter"
                )
            else:
                self.series_data_ready.emit(color_id, vals, labels, msg)
            return

        geom = self.get_map_geometry()
        if geom.is_profile:
            if not ok or len(vals) != len(geom.positions):
                self.profile_data_ready.emit(
                    color_id, np.array([]), np.array([]), geom.axis_label,
                    msg or "Unable to calculate parameter",
                )
                return
            order = geom.order
            self.profile_data_ready.emit(
                color_id, geom.positions[order], np.asarray(vals)[order], geom.axis_label, msg
            )
            return

        if not ok:
            self.map_data_ready.emit(
                color_id, None, list(_EMPTY_EXTENT), 0.0, 1.0, msg or "Unable to calculate parameter"
            )
            return
        grid, extent, vmin, vmax = compute_map_grid(self._current_md.coords, vals)
        self.map_data_ready.emit(color_id, grid, extent, vmin, vmax, msg)
