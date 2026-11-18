"""Background QRunnable worker and async offload mixin for Live Maps."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

from spectroview.model.spectra.spectral_parameters import (
    PARAM_INTEGRATED_INTENSITY,
    PARAM_RB_RATIO,
    SELECTOR_BLUE,
    SELECTOR_GREEN,
    SELECTOR_KEYS,
    SELECTOR_RED,
    MapGeometry,
    MapGridder,
    compute_intensity_ratio,
    compute_map_grid,
    compute_spectral_parameter,
)
from spectroview.model.spectra.spectra_store import MapData

_EMPTY_EXTENT = [0.0, 1.0, 0.0, 1.0]
_VERSIONED_CACHE_MAX = 64
ASYNC_SPECTRUM_THRESHOLD = 25_000
MSG_RATIO_NEEDS_RB = "Activate the Red and Blue range selectors to display the R/B ratio"

# (values, labels, is_valid, message)
_Result = Tuple[np.ndarray, list, bool, str]


class _LiveMapWorkerBridge(QObject):
    """Main-thread Qt signal bridge for background Live Maps QRunnable workers."""

    finished = Signal(int, object)


class _LiveMapWorker(QRunnable):
    """Pure-NumPy background worker that evaluates heavy Live Maps batches off the GUI thread."""

    def __init__(
        self, generation_id: int, snapshot: dict, bridge: _LiveMapWorkerBridge
    ):
        super().__init__()
        self.setAutoDelete(True)
        self._gen_id = generation_id
        self._snap = snapshot
        self._bridge = bridge

    def run(self):
        t0 = time.perf_counter()
        snap = self._snap
        pre_cached = snap["pre_cached"]
        states = snap["states"]
        selector_order = snap["selector_order"]
        excluded_status = snap["excluded_status"]
        display_mode = snap["display_mode"]

        pass_intensity: Dict[tuple, Tuple[Tuple[float, float], _Result]] = {}
        new_cache_entries: Dict[tuple, _Result] = {}

        def eval_param(
            md_snap: dict, x_range: Tuple[float, float], param: str
        ) -> _Result:
            cache_key = (
                md_snap["id"],
                md_snap["name"],
                md_snap["data_version"],
                param,
                x_range,
            )
            hit = pre_cached.get(cache_key)
            if hit is not None:
                return hit
            hit = new_cache_entries.get(cache_key)
            if hit is not None:
                return hit
            vals, ok, msg = compute_spectral_parameter(
                x_axis=md_snap["x_axis"],
                Y=md_snap["y_matrix"],
                x_range=x_range,
                parameter=param,
                baseline_subtracted=md_snap["is_baseline_subtracted"],
                peak_params=md_snap["peak_params"],
                param_names=md_snap["param_names"],
                fit_model=md_snap["fit_model"],
                active_mask=md_snap["is_active"],
            )
            res: _Result = (vals, [], ok, msg)
            new_cache_entries[cache_key] = res
            return res

        def integrated_for(md_snap: dict, color_id: str) -> _Result:
            rng = states[color_id]["range"]
            ikey = (md_snap["id"], md_snap["data_version"], color_id)
            cached = pass_intensity.get(ikey)
            if cached is not None and cached[0] == rng:
                return cached[1]
            res = eval_param(md_snap, rng, PARAM_INTEGRATED_INTENSITY)
            pass_intensity[ikey] = (rng, res)
            return res

        def ratio_for(md_snap: dict) -> _Result:
            if not (states[SELECTOR_RED]["active"] and states[SELECTOR_BLUE]["active"]):
                return np.array([], dtype=np.float64), [], False, MSG_RATIO_NEEDS_RB
            r_vals, labels, r_ok, r_msg = integrated_for(md_snap, SELECTOR_RED)
            if not r_ok:
                return r_vals, labels, False, f"Red range: {r_msg}"
            b_vals, _, b_ok, b_msg = integrated_for(md_snap, SELECTOR_BLUE)
            if not b_ok:
                return b_vals, labels, False, f"Blue range: {b_msg}"
            if len(r_vals) != len(b_vals):
                return (
                    np.array([], dtype=np.float64),
                    labels,
                    False,
                    "Red/Blue size mismatch",
                )
            ratio, n_masked = compute_intensity_ratio(r_vals, b_vals)
            if n_masked == ratio.size:
                return (
                    ratio,
                    labels,
                    False,
                    "R/B ratio undefined: Blue intensity is zero",
                )
            msg = (
                f"{n_masked} point(s) with ~zero Blue intensity masked"
                if n_masked
                else "OK"
            )
            return ratio, labels, True, msg

        def eval_selector(md_snap: dict, color_id: str) -> _Result:
            st = states[color_id]
            if color_id == SELECTOR_GREEN and st["param"] == PARAM_RB_RATIO:
                return ratio_for(md_snap)
            if st["param"] == PARAM_INTEGRATED_INTENSITY:
                return integrated_for(md_snap, color_id)
            return eval_param(md_snap, st["range"], st["param"])

        outputs: list = []
        try:
            if display_mode == "multi_profile":
                profile_snaps = snap["profile_mds"]
                primary_name = snap["primary_map_name"]
                for cid in selector_order:
                    entries = []
                    out_of_range_names = []
                    primary_msg = ""
                    for md_snap in profile_snaps:
                        is_primary = md_snap["name"] == primary_name
                        if not md_snap["has_coords"]:
                            out_of_range_names.append(md_snap["name"])
                            entries.append({
                                "name": md_snap["name"],
                                "positions": np.array([], dtype=np.float64),
                                "values": np.array([], dtype=np.float64),
                                "is_primary": is_primary,
                                "axis_label": "Position",
                                "out_of_range": True,
                                "msg": "No coordinates",
                            })
                            continue
                        geom = md_snap["geom"]
                        vals, _, ok, msg = eval_selector(md_snap, cid)
                        order = geom.order
                        pos_sorted = np.asarray(
                            geom.positions[order], dtype=np.float64
                        )
                        if ok and len(vals) == len(geom.positions):
                            val_sorted = np.asarray(vals, dtype=np.float64)[order]
                            out_of_range = False
                            if np.all(np.isnan(val_sorted)):
                                out_of_range = True
                                msg = "All spectra unchecked or undefined"
                        else:
                            val_sorted = np.full(
                                len(pos_sorted), np.nan, dtype=np.float64
                            )
                            out_of_range = True
                            out_of_range_names.append(md_snap["name"])
                        if is_primary:
                            primary_msg = msg
                        entries.append({
                            "name": md_snap["name"],
                            "positions": pos_sorted,
                            "values": val_sorted,
                            "is_primary": is_primary,
                            "axis_label": geom.axis_label,
                            "out_of_range": out_of_range,
                            "msg": msg,
                        })
                    axis_labels = {
                        e["axis_label"] for e in entries if e["positions"].size > 0
                    }
                    shared_x_label = (
                        next(iter(axis_labels))
                        if len(axis_labels) == 1
                        else "Spatial Position"
                    )
                    notes = []
                    if excluded_status:
                        notes.append(excluded_status)
                    if out_of_range_names:
                        notes.append(
                            f"Out of range / undefined: {', '.join(out_of_range_names[:3])}"
                        )
                    elif primary_msg and primary_msg != "OK":
                        notes.append(primary_msg)
                    combined_msg = " · ".join(notes) if notes else "OK"
                    outputs.append((
                        "multi_profile",
                        cid,
                        entries,
                        shared_x_label,
                        combined_msg,
                    ))
            else:
                md_snap = snap["current_md"]
                geom = md_snap["geom"]
                for cid in selector_order:
                    vals, _, ok, msg = eval_selector(md_snap, cid)
                    if geom.is_profile:
                        if not ok or len(vals) != len(geom.positions):
                            outputs.append((
                                "profile",
                                cid,
                                np.array([]),
                                np.array([]),
                                geom.axis_label,
                                msg or "Unable to calculate parameter",
                            ))
                        else:
                            vals_arr = np.asarray(vals, dtype=np.float64)
                            if np.all(np.isnan(vals_arr)):
                                outputs.append((
                                    "profile",
                                    cid,
                                    np.array([]),
                                    np.array([]),
                                    geom.axis_label,
                                    "All spectra unchecked or undefined",
                                ))
                            else:
                                order = geom.order
                                outputs.append((
                                    "profile",
                                    cid,
                                    geom.positions[order],
                                    vals_arr[order],
                                    geom.axis_label,
                                    msg,
                                ))
                    else:
                        if not ok:
                            outputs.append((
                                "map_2d",
                                cid,
                                None,
                                list(_EMPTY_EXTENT),
                                0.0,
                                1.0,
                                msg or "Unable to calculate parameter",
                            ))
                        else:
                            grid, extent, vmin, vmax = compute_map_grid(
                                md_snap["coords"], vals, gridder=md_snap["gridder"]
                            )
                            if grid is None:
                                outputs.append((
                                    "map_2d",
                                    cid,
                                    None,
                                    list(_EMPTY_EXTENT),
                                    0.0,
                                    1.0,
                                    "All spectra unchecked or undefined",
                                ))
                            else:
                                outputs.append((
                                    "map_2d",
                                    cid,
                                    grid,
                                    extent,
                                    vmin,
                                    vmax,
                                    msg,
                                ))
            error_msg = None
        except Exception as exc:
            outputs = []
            error_msg = str(exc)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        payload = {
            "elapsed_ms": elapsed_ms,
            "outputs": outputs,
            "cache_entries": new_cache_entries,
            "md_versions": snap["md_versions"],
            "error": error_msg,
        }
        try:
            self._bridge.finished.emit(self._gen_id, payload)
        except RuntimeError:
            pass


class LiveMapsAsyncMixin:
    """Async snapshot construction and worker delivery methods for VMLiveMaps."""

    def _total_active_spectra(self) -> int:
        """Return total number of spectra evaluated in the current map view."""
        if self.mode != "map":
            return 0
        if self._display_mode == "multi_profile" and len(self._profile_mds) >= 2:
            return sum(
                m.n_spectra
                for m in self._profile_mds
                if m is not None and m.y_matrix is not None
            )
        md = self._current_md
        if (
            md is None
            or md.y_matrix is None
            or md.coords is None
            or len(md.coords) == 0
        ):
            return 0
        return md.n_spectra

    def _should_run_async(self, ordered_cids: List[str]) -> bool:
        """Return True when the batch exceeds the spectrum threshold and has valid work."""
        if self._shutting_down or not ordered_cids or self.mode != "map":
            return False
        if self._display_mode != "multi_profile":
            md = self._current_md
            if (
                md is None
                or md.x_axis is None
                or md.y_matrix is None
                or md.coords is None
                or len(md.coords) == 0
            ):
                return False
        return self._total_active_spectra() >= self._async_threshold

    @staticmethod
    def _snapshot_md(
        md: MapData, geom: MapGeometry, gridder: Optional[MapGridder]
    ) -> dict:
        """Build an immutable-metadata snapshot of *md* on the main thread before async work."""
        has_fit = md.has_fit_results()
        has_coords = (
            md.x_axis is not None
            and md.y_matrix is not None
            and md.coords is not None
            and len(md.coords) > 0
        )
        return {
            "id": id(md),
            "name": md.name,
            "data_version": getattr(md, "data_version", 0),
            "has_coords": has_coords,
            "x_axis": md.x_axis,
            "y_matrix": md.y_matrix,
            "coords": md.coords,
            "is_active": getattr(md, "is_active", None),
            "is_baseline_subtracted": getattr(md, "is_baseline_subtracted", False),
            "peak_params": md.peak_params if has_fit else None,
            "param_names": md.param_names if has_fit else None,
            "fit_model": md.fit_model,
            "geom": geom,
            "gridder": gridder,
        }

    def _build_worker_snapshot(self, ordered_cids: List[str]) -> dict:
        """Capture pure-NumPy inputs, pre-cached results, and version checks for _LiveMapWorker."""
        states_copy = {
            cid: {
                "active": st["active"],
                "range": tuple(st["range"]),
                "param": st["param"],
            }
            for cid, st in self._states.items()
        }
        pre_cached: Dict[tuple, _Result] = {}
        md_versions: Dict[int, Tuple[str, int]] = {}

        if self._display_mode == "multi_profile" and len(self._profile_mds) >= 2:
            profile_snaps = []
            for md in self._profile_mds:
                geom = self._get_geometry_for(md)
                profile_snaps.append(self._snapshot_md(md, geom, None))
                md_versions[id(md)] = (md.name, getattr(md, "data_version", 0))
            mds_for_cache = self._profile_mds
            current_snap = None
        else:
            md = self._current_md
            geom = self._get_geometry_for(md)
            gridder = None if geom.is_profile else self._get_gridder_for(md)
            current_snap = self._snapshot_md(md, geom, gridder)
            md_versions[id(md)] = (md.name, getattr(md, "data_version", 0))
            profile_snaps = []
            mds_for_cache = [md]

        for md in mds_for_cache:
            ver = getattr(md, "data_version", 0)
            for cid in SELECTOR_KEYS:
                st = states_copy[cid]
                for param in (st["param"], PARAM_INTEGRATED_INTENSITY):
                    ck = (id(md), md.name, ver, param, st["range"])
                    hit = self._versioned_cache.get(ck)
                    if hit is not None:
                        pre_cached[ck] = hit

        return {
            "display_mode": (
                self._display_mode
                if (
                    self._display_mode == "multi_profile"
                    and len(self._profile_mds) >= 2
                )
                else "single"
            ),
            "selector_order": ordered_cids,
            "states": states_copy,
            "pre_cached": pre_cached,
            "profile_mds": profile_snaps,
            "current_md": current_snap,
            "primary_map_name": self._primary_map_name,
            "excluded_status": self._format_excluded_status(),
            "md_versions": md_versions,
        }

    def _on_async_batch_finished(self, gen_id: int, payload: dict):
        """Deliver background calculation results on the main Qt thread if still current."""
        if self._shutting_down or gen_id != self._generation_id:
            return
        self._async_busy = False

        expected_versions = payload.get("md_versions", {})
        if self._display_mode == "multi_profile":
            current_map = {
                id(m): (m.name, getattr(m, "data_version", 0))
                for m in self._profile_mds
            }
        elif self._current_md is not None:
            current_map = {
                id(self._current_md): (
                    self._current_md.name,
                    getattr(self._current_md, "data_version", 0),
                )
            }
        else:
            current_map = {}

        for mid, ver_info in expected_versions.items():
            if current_map.get(mid) != ver_info:
                return

        for ck, res in payload.get("cache_entries", {}).items():
            self._versioned_cache[ck] = res
            while len(self._versioned_cache) > _VERSIONED_CACHE_MAX:
                self._versioned_cache.popitem(last=False)

        elapsed_ms = float(payload.get("elapsed_ms", 0.0))
        self._last_calc_ms = elapsed_ms
        self._timer.setInterval(max(60, min(250, int(elapsed_ms * 1.1))))

        for item in payload.get("outputs", []):
            kind = item[0]
            if kind == "multi_profile":
                _, cid, entries, shared_x_label, combined_msg = item
                self.multi_profile_data_ready.emit(
                    cid, entries, shared_x_label, combined_msg
                )
            elif kind == "profile":
                _, cid, pos, vals, axis_label, msg = item
                self.profile_data_ready.emit(cid, pos, vals, axis_label, msg)
            elif kind == "map_2d":
                _, cid, grid, extent, vmin, vmax, msg = item
                self.map_data_ready.emit(cid, grid, extent, vmin, vmax, msg)
