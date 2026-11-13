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
from collections import OrderedDict
import time
from typing import Optional, Dict, Set, Iterable, Tuple, List

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
    MapGridder,
    analyze_map_geometry,
    build_map_gridder,
    compute_intensity_ratio,
    compute_spectral_parameter,
    compute_series_spectral_parameter,
    compute_map_grid,
)

_EMPTY_EXTENT = [0.0, 1.0, 0.0, 1.0]
_VERSIONED_CACHE_MAX = 64
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
    # (color_id, entries, shared_x_label, status_msg) — multi-1D profile comparison
    multi_profile_data_ready = Signal(str, list, str, str)
    # (candidate_names, active_name_or_none, status_msg)
    map_2d_candidates_changed = Signal(list, object, str)
    # Emitted when selection becomes empty so the view clears all plots and timers cleanly
    live_maps_cleared = Signal()
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

        # Current MapData snapshot (Maps workspace) or list of MapData (Spectra workspace)
        self._current_md: Optional[MapData] = None
        self._current_mds: List[MapData] = []
        self._profile_mds: List[MapData] = []
        self._current_series_mds: List[MapData] = []
        self._primary_map_name: Optional[str] = None
        self._active_2d_map_name: Optional[str] = None
        self._2d_candidates: List[str] = []
        self._excluded_names: List[str] = []
        self._display_mode: str = "empty"
        self._last_data_sig: Optional[tuple] = None

        # Map caches keyed by MapData id / version
        self._geometry_cache: Optional[Tuple[np.ndarray, MapGeometry]] = None
        self._geom_cache: Dict[int, Tuple[np.ndarray, MapGeometry]] = {}
        self._gridder_cache: Dict[int, Tuple[np.ndarray, Optional[bytes], MapGridder]] = {}
        self._versioned_cache: OrderedDict = OrderedDict()

        # Integrated intensities computed during the current recalculation pass,
        # keyed by (id(md), color_id) → (range, result). Pass-scoped (cleared before
        # and after each batch) so in-place data edits can never be served stale.
        self._intensity_cache: Dict[tuple, Tuple[Tuple[float, float], _Result]] = {}

        # Debounced calculation during interactive dragging (adapts to heavy maps)
        self._pending_calc: Set[str] = set()
        self._last_calc_ms: float = 0.0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)  # 60 ms debounce baseline; scales smoothly on heavy maps
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
        """Return the (cached) spatial dimensionality of the current active map."""
        return self._get_geometry_for(self._current_md)

    def _get_geometry_for(self, md: Optional[MapData]) -> MapGeometry:
        """Return cached MapGeometry for *md* using coords object identity."""
        if md is None:
            return analyze_map_geometry(None)
        coords = getattr(md, "coords", None)
        if coords is None:
            return analyze_map_geometry(None)
        cached = self._geom_cache.get(id(md))
        if cached is not None and cached[0] is coords:
            return cached[1]
        geom = analyze_map_geometry(coords)
        self._geom_cache[id(md)] = (coords, geom)
        if md is self._current_md:
            self._geometry_cache = (coords, geom)
        return geom

    def _get_gridder_for(self, md: MapData) -> MapGridder:
        """Return cached MapGridder for *md*, invalidating when coords or is_active changes."""
        coords = md.coords
        active_arr = getattr(md, "is_active", None)
        active_sig = active_arr.tobytes() if active_arr is not None else None
        cached = self._gridder_cache.get(id(md))
        if cached is not None and cached[0] is coords and cached[1] == active_sig:
            return cached[2]
        gridder = build_map_gridder(coords, grid_res=80, active_mask=active_arr)
        self._gridder_cache[id(md)] = (coords, active_sig, gridder)
        return gridder

    def _format_excluded_status(self) -> str:
        """Format a short status notice for excluded mixed-dimensionality maps."""
        if not self._excluded_names:
            return ""
        shown = ", ".join(self._excluded_names[:3])
        extra = f" (+{len(self._excluded_names) - 3})" if len(self._excluded_names) > 3 else ""
        if self._display_mode == "multi_profile":
            return f"Excluded 2D map(s): {shown}{extra}"
        return f"Excluded 1D profile(s): {shown}{extra}"

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
                new_rng = (x_min + p0 * span, x_min + p1 * span)
            else:
                c0 = max(x_min, r0)
                c1 = min(x_max, r1)
                if c0 < c1:
                    new_rng = (c0, c1)
                else:
                    new_rng = (x_min + p0 * span, x_min + p1 * span)
            if new_rng != (r0, r1):
                state["range"] = new_rng
                self.selector_state_changed.emit(cid, state["active"], new_rng[0], new_rng[1])

    # ── Public inputs ─────────────────────────────────────────────────

    def set_map_data(self, md: Optional[MapData]):
        """Update active single MapData and trigger recalculation for all active selectors."""
        self.set_map_datas(
            [md] if md is not None else [],
            primary_name=md.name if md is not None else None,
        )

    def set_map_datas(
        self,
        mds: Optional[Iterable[MapData]],
        primary_name: Optional[str] = None,
        active_2d_name: Optional[str] = None,
    ):
        """Update selected maps using Primary-precedence dimensionality rules.

        - If Primary is a 1D profile, overlay all selected 1D profiles and exclude 2D maps.
        - If Primary is a 2D map, populate 2D candidates and evaluate only the active 2D map.
        - Identical data+selection signatures short-circuit without recomputation.
        """
        clean_mds = [md for md in (mds or []) if md is not None]
        if not clean_mds:
            empty_sig = ("empty",)
            if self._last_data_sig == empty_sig:
                return False
            self._timer.stop()
            self._pending_calc.clear()
            self._current_md = None
            self._current_mds = []
            self._profile_mds = []
            self._primary_map_name = None
            self._active_2d_map_name = None
            self._2d_candidates = []
            self._excluded_names = []
            self._display_mode = "empty"
            self._geometry_cache = None
            self._geom_cache.clear()
            self._gridder_cache.clear()
            self._intensity_cache.clear()
            self._versioned_cache.clear()
            self._last_data_sig = empty_sig
            self.map_2d_candidates_changed.emit([], None, "")
            self.live_maps_cleared.emit()
            return True

        by_name = {m.name: m for m in clean_mds}
        primary_md = by_name.get(primary_name) if primary_name else None
        if primary_md is None:
            primary_md = clean_mds[0]
        resolved_primary = primary_md.name
        primary_changed = resolved_primary != self._primary_map_name

        # Prune per-map caches for deselected/deleted maps
        valid_ids = {id(m) for m in clean_mds}
        for mid in list(self._geom_cache.keys()):
            if mid not in valid_ids:
                del self._geom_cache[mid]
        for mid in list(self._gridder_cache.keys()):
            if mid not in valid_ids:
                del self._gridder_cache[mid]

        primary_geom = self._get_geometry_for(primary_md)
        if primary_geom.is_profile:
            profile_mds = [m for m in clean_mds if self._get_geometry_for(m).is_profile]
            excluded = [m.name for m in clean_mds if not self._get_geometry_for(m).is_profile]
            sig = (
                "multi_profile",
                tuple((id(m), m.name, getattr(m, "data_version", 0)) for m in profile_mds),
                resolved_primary,
                tuple(excluded),
            )
            if sig == self._last_data_sig:
                self._current_mds = clean_mds
                return False
            self._display_mode = "multi_profile"
            self._current_mds = clean_mds
            self._profile_mds = profile_mds
            self._current_md = primary_md
            self._primary_map_name = resolved_primary
            self._active_2d_map_name = None
            self._2d_candidates = []
            self._excluded_names = excluded
            self._last_data_sig = sig
        else:
            candidates_2d = [m for m in clean_mds if not self._get_geometry_for(m).is_profile]
            candidate_names = [m.name for m in candidates_2d]
            excluded = [m.name for m in clean_mds if self._get_geometry_for(m).is_profile]
            if active_2d_name and active_2d_name in candidate_names:
                target_2d = active_2d_name
            elif primary_changed or self._active_2d_map_name not in candidate_names:
                target_2d = resolved_primary if resolved_primary in candidate_names else candidate_names[0]
            else:
                target_2d = self._active_2d_map_name
            target_md = next(m for m in candidates_2d if m.name == target_2d)
            sig = (
                "map_2d",
                (id(target_md), target_md.name, getattr(target_md, "data_version", 0)),
                tuple(candidate_names),
                target_2d,
                tuple(excluded),
            )
            if sig == self._last_data_sig:
                self._current_mds = clean_mds
                self._primary_map_name = resolved_primary
                return False
            self._display_mode = "map_2d"
            self._current_mds = clean_mds
            self._profile_mds = []
            self._current_md = target_md
            self._primary_map_name = resolved_primary
            self._active_2d_map_name = target_2d
            self._2d_candidates = candidate_names
            self._excluded_names = excluded
            self._last_data_sig = sig

        self._geometry_cache = (
            (self._current_md.coords, self._get_geometry_for(self._current_md))
            if self._current_md is not None and self._current_md.coords is not None
            else None
        )
        self.map_2d_candidates_changed.emit(
            list(self._2d_candidates), self._active_2d_map_name, self._format_excluded_status()
        )
        if self._current_md is not None and self._current_md.x_axis is not None and len(self._current_md.x_axis) >= 2:
            self._sync_or_init_ranges_from_axis(self._current_md.x_axis)
        self._recalculate_batch(cid for cid, s in self._states.items() if s["active"])
        return True

    def set_active_2d_map(self, map_name: str):
        """Switch which 2D candidate map is displayed when multiple 2D maps are selected."""
        if not map_name or map_name not in self._2d_candidates:
            return
        if map_name == self._active_2d_map_name:
            return
        self.set_map_datas(
            self._current_mds,
            primary_name=self._primary_map_name,
            active_2d_name=map_name,
        )

    def recalculate_active(self):
        """Recalculate all currently active selectors."""
        self._recalculate_batch(cid for cid, s in self._states.items() if s["active"])

    def set_series_data(self, mds: Optional[list[MapData]]):
        """Update selected spectra series and trigger recalculation for all active selectors."""
        self._current_series_mds = [md for md in (mds or []) if md is not None]
        if self._current_series_mds:
            first_md = self._current_series_mds[0]
            if first_md.x_axis is not None and len(first_md.x_axis) >= 2:
                self._sync_or_init_ranges_from_axis(first_md.x_axis)
        self._recalculate_batch(cid for cid, s in self._states.items() if s["active"])

    def set_selector_active(self, color_id: str, is_active: bool, calculate: bool = True):
        """Toggle active state of a selector."""
        if color_id not in self._states:
            return
        state = self._states[color_id]
        state["active"] = is_active
        x_min, x_max = state["range"]
        self.selector_state_changed.emit(color_id, is_active, x_min, x_max)

        if not calculate:
            return
        to_calc = set(self._dependents(color_id))
        if is_active:
            to_calc.add(color_id)
        self._recalculate_batch(to_calc)

    def set_selector_range(
        self,
        color_id: str,
        x_min: float,
        x_max: float,
        immediate: bool = False,
        calculate: bool = True,
    ):
        """Update spectral interval for a selector."""
        if color_id not in self._states:
            return
        state = self._states[color_id]
        state["range"] = (min(float(x_min), float(x_max)), max(float(x_min), float(x_max)))
        state["user_set"] = True
        self.selector_state_changed.emit(color_id, state["active"], state["range"][0], state["range"][1])

        if not calculate or not state["active"]:
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
        t0 = time.perf_counter()
        try:
            for cid in SELECTOR_KEYS:
                if cid in wanted:
                    self._recalculate_one(cid)
        finally:
            self._intensity_cache.clear()
            if self.mode == "map":
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self._last_calc_ms = elapsed_ms
                self._timer.setInterval(max(60, min(250, int(elapsed_ms * 1.1))))

    def _recalculate_one(self, color_id: str):
        state = self._states[color_id]

        if self.mode == "spectra":
            if not self._current_series_mds:
                self.series_data_ready.emit(color_id, np.array([]), [], "No spectra selected")
                return
            if self._is_ratio_display(color_id):
                result = self._compute_rb_ratio()
            elif state["param"] == PARAM_INTEGRATED_INTENSITY:
                result = self._integrated_intensity(color_id)
            else:
                result = self._compute_values(state["range"], state["param"])
            self._emit_result(color_id, *result)
            return

        if self._display_mode == "multi_profile" and len(self._profile_mds) >= 2:
            self._recalculate_multi_profile(color_id)
            return

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

        result = self._evaluate_selector_for_md(md, color_id)
        self._emit_result(color_id, *result)

    def _evaluate_selector_for_md(self, md: MapData, color_id: str) -> _Result:
        """Evaluate the selector *color_id* on *md* (R/B ratio, cached intensity, or parameter)."""
        state = self._states[color_id]
        if self._is_ratio_display(color_id):
            return self._compute_rb_ratio_for_md(md)
        if state["param"] == PARAM_INTEGRATED_INTENSITY:
            return self._integrated_intensity_for_md(md, color_id)
        return self._compute_values_for_md(md, state["range"], state["param"])

    def _recalculate_multi_profile(self, color_id: str):
        """Compute and emit stacked 1D profile entries for all selected 1D maps."""
        entries = []
        out_of_range_names = []
        primary_msg = ""

        for md in self._profile_mds:
            is_primary = md.name == self._primary_map_name
            if md.x_axis is None or md.y_matrix is None or md.coords is None or len(md.coords) == 0:
                out_of_range_names.append(md.name)
                entries.append({
                    "name": md.name,
                    "positions": np.array([], dtype=np.float64),
                    "values": np.array([], dtype=np.float64),
                    "is_primary": is_primary,
                    "axis_label": "Position",
                    "out_of_range": True,
                    "msg": "No coordinates",
                })
                continue

            geom = self._get_geometry_for(md)
            vals, _, ok, msg = self._evaluate_selector_for_md(md, color_id)
            order = geom.order
            pos_sorted = np.asarray(geom.positions[order], dtype=np.float64)
            if ok and len(vals) == len(geom.positions):
                val_sorted = np.asarray(vals, dtype=np.float64)[order]
                out_of_range = False
                if np.all(np.isnan(val_sorted)):
                    out_of_range = True
                    msg = "All spectra unchecked or undefined"
            else:
                val_sorted = np.full(len(pos_sorted), np.nan, dtype=np.float64)
                out_of_range = True
                out_of_range_names.append(md.name)

            if is_primary:
                primary_msg = msg
            entries.append({
                "name": md.name,
                "positions": pos_sorted,
                "values": val_sorted,
                "is_primary": is_primary,
                "axis_label": geom.axis_label,
                "out_of_range": out_of_range,
                "msg": msg,
            })

        axis_labels = {e["axis_label"] for e in entries if e["positions"].size > 0}
        shared_x_label = next(iter(axis_labels)) if len(axis_labels) == 1 else "Spatial Position"
        notes = []
        excl = self._format_excluded_status()
        if excl:
            notes.append(excl)
        if out_of_range_names:
            notes.append(f"Out of range / undefined: {', '.join(out_of_range_names[:3])}")
        elif primary_msg and primary_msg != "OK":
            notes.append(primary_msg)
        combined_msg = " · ".join(notes) if notes else "OK"
        self.multi_profile_data_ready.emit(color_id, entries, shared_x_label, combined_msg)

    def _compute_values(self, x_range: Tuple[float, float], param: str) -> _Result:
        """Evaluate *param* over *x_range* on the current primary data (both modes)."""
        if self.mode == "spectra":
            vals, labels, ok, msg = compute_series_spectral_parameter(
                mds=self._current_series_mds, x_range=x_range, parameter=param,
            )
            return vals, labels, ok, msg
        return self._compute_raw_for_md(self._current_md, x_range, param)

    def _compute_values_for_md(self, md: MapData, x_range: Tuple[float, float], param: str) -> _Result:
        """Evaluate *param* on *md* with LRU versioned caching keyed by MapData.data_version."""
        if md is None:
            return np.array([], dtype=np.float64), [], False, "No hyperspectral map loaded"
        cache_key = (id(md), md.name, getattr(md, "data_version", 0), param, x_range)
        cached = self._versioned_cache.get(cache_key)
        if cached is not None:
            self._versioned_cache.move_to_end(cache_key)
            return cached

        if md is self._current_md:
            result = self._compute_values(x_range, param)
        else:
            result = self._compute_raw_for_md(md, x_range, param)

        self._versioned_cache[cache_key] = result
        while len(self._versioned_cache) > _VERSIONED_CACHE_MAX:
            self._versioned_cache.popitem(last=False)
        return result

    def _compute_raw_for_md(self, md: Optional[MapData], x_range: Tuple[float, float], param: str) -> _Result:
        """Direct spectral parameter computation for a single MapData (no cache)."""
        if md is None:
            return np.array([], dtype=np.float64), [], False, "No hyperspectral map loaded"
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
            active_mask=getattr(md, "is_active", None),
        )
        return vals, [], ok, msg

    def _integrated_intensity(self, color_id: str) -> _Result:
        """Integrated intensity over *color_id*'s range on current primary data."""
        if self.mode == "spectra":
            rng = self._states[color_id]["range"]
            cache_key = ("series", color_id)
            cached = self._intensity_cache.get(cache_key)
            if cached is not None and cached[0] == rng:
                return cached[1]
            result = self._compute_values(rng, PARAM_INTEGRATED_INTENSITY)
            self._intensity_cache[cache_key] = (rng, result)
            return result
        return self._integrated_intensity_for_md(self._current_md, color_id)

    def _integrated_intensity_for_md(self, md: MapData, color_id: str) -> _Result:
        """Integrated intensity for *md*, memoized within the current pass and LRU-cached."""
        rng = self._states[color_id]["range"]
        cache_key = (id(md), getattr(md, "data_version", 0), color_id)
        cached = self._intensity_cache.get(cache_key)
        if cached is not None and cached[0] == rng:
            return cached[1]
        result = self._compute_values_for_md(md, rng, PARAM_INTEGRATED_INTENSITY)
        self._intensity_cache[cache_key] = (rng, result)
        return result

    def _compute_rb_ratio(self) -> _Result:
        """Point-wise ratio I_Red / I_Blue on current primary data (Spectra or single Map)."""
        if self.mode == "spectra":
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
        return self._compute_rb_ratio_for_md(self._current_md)

    def _compute_rb_ratio_for_md(self, md: MapData) -> _Result:
        """Point-wise ratio I_Red / I_Blue of integrated intensities for *md*."""
        if not (self._states[SELECTOR_RED]["active"] and self._states[SELECTOR_BLUE]["active"]):
            return np.array([], dtype=np.float64), [], False, MSG_RATIO_NEEDS_RB

        r_vals, labels, r_ok, r_msg = self._integrated_intensity_for_md(md, SELECTOR_RED)
        if not r_ok:
            return r_vals, labels, False, f"Red range: {r_msg}"
        b_vals, _, b_ok, b_msg = self._integrated_intensity_for_md(md, SELECTOR_BLUE)
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
            vals_arr = np.asarray(vals, dtype=np.float64)
            if np.all(np.isnan(vals_arr)):
                self.profile_data_ready.emit(
                    color_id, np.array([]), np.array([]), geom.axis_label,
                    "All spectra unchecked or undefined",
                )
                return
            order = geom.order
            self.profile_data_ready.emit(
                color_id, geom.positions[order], vals_arr[order], geom.axis_label, msg
            )
            return

        if not ok:
            self.map_data_ready.emit(
                color_id, None, list(_EMPTY_EXTENT), 0.0, 1.0, msg or "Unable to calculate parameter"
            )
            return
        gridder = self._get_gridder_for(self._current_md)
        grid, extent, vmin, vmax = compute_map_grid(
            self._current_md.coords, vals, gridder=gridder
        )
        if grid is None:
            self.map_data_ready.emit(
                color_id, None, list(_EMPTY_EXTENT), 0.0, 1.0,
                "All spectra unchecked or undefined",
            )
            return
        self.map_data_ready.emit(color_id, grid, extent, vmin, vmax, msg)
