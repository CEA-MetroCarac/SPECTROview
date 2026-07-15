"""Stateful, GUI-file-compatible workspace sessions.

`SpectraWorkspace` and `MapsWorkspace` wrap a `SpectraStore` and replicate
the GUI's Spectra/Maps Workspace operations (load, preprocess, fit, collect
results, save/load) synchronously and without any Qt dependency. Files
written by `.save()` use the exact same ZIP/npy/parquet format (format
version 2) the GUI's own "Save work" produces, so they open directly in the
GUI, and files saved by the GUI load directly here.

Legacy format-v1 workspace files (pre-ZIP, raw JSON+zlib/base64) are not
supported by `.load()` — open them once in the GUI and re-save to upgrade.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from spectroview.api import io as api_io
from spectroview.api import fitting as api_fitting
from spectroview.api.exceptions import FitModelError, WorkspaceError
from spectroview.fit_engine.baseline import eval_baseline_batch
from spectroview.fit_engine.evaluator import eval_peak_initial
from spectroview.model.heatmap import build_heatmap_grid
from spectroview.model.spectra_store import SpectraStore
from spectroview.model.workspace_io import WorkspaceIO


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable equivalents."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _restore_preprocessed_state(md) -> None:
    """Rebuild md.x/md.Y (crop) and md.Y_baseline from md.x0/md.Y0 + the saved
    baseline/range config. Processed arrays are never saved to disk directly
    (only the raw x0/Y0 are), so this must run once per map after loading —
    mirrors VMWorkspaceSpectra._restore_preprocessed_state.
    """
    if not md.baseline_config and md.fit_model and isinstance(md.fit_model, dict) and "baseline" in md.fit_model:
        md.baseline_config = md.fit_model["baseline"]

    if md.range_min is not None or md.range_max is not None:
        mask = np.logical_and(
            md.x0 >= (md.range_min if md.range_min is not None else -np.inf),
            md.x0 <= (md.range_max if md.range_max is not None else np.inf),
        )
        md.x = md.x0[mask].copy()
        md.Y = md.Y0[:, mask].copy()
    else:
        md.x = md.x0.copy()
        md.Y = md.Y0.copy()

    if md.baseline_config:
        md.Y_baseline = eval_baseline_batch(md.x, md.Y, md.baseline_config)

    is_sub = md.is_baseline_subtracted
    is_sub_any = is_sub.any() if isinstance(is_sub, np.ndarray) else bool(is_sub)
    if is_sub_any and md.Y_baseline is not None:
        if isinstance(is_sub, np.ndarray):
            sub_indices = np.where(is_sub)[0]
            if len(sub_indices) > 0:
                md.Y[sub_indices] = md.Y[sub_indices] - md.Y_baseline[sub_indices]
                md.Y_baseline[sub_indices] = 0.0
        else:
            md.Y = md.Y - md.Y_baseline
            md.Y_baseline = None


def _reconstruct_y_peaks(md) -> None:
    """Rebuild md.Y_peaks preview curves from md.fit_model (before fitting)."""
    if not md.fit_model or not md.fit_model.get("peak_models"):
        md.Y_peaks = None
        return
    x_arr = md.x if md.x is not None else md.x0
    N = md.Y.shape[0] if md.Y is not None else md.Y0.shape[0]
    peaks = []
    for p_model in md.fit_model["peak_models"].values():
        y_curve = eval_peak_initial(x_arr, p_model)
        peaks.append(np.tile(y_curve, (N, 1)).astype(np.float32))
    md.Y_peaks = peaks


class SpectraWorkspace:
    """A stateful, file-compatible equivalent of the GUI's Spectra Workspace.

    Holds one `MapData` block per loaded discrete spectrum (each with a
    single row). Every mutating method targets all spectra by default, or a
    subset via the `names` parameter.

    Example::

        ws = SpectraWorkspace()
        ws.load_files(["a.txt", "b.txt"])
        ws.crop(range_min=400, range_max=600)
        ws.set_baseline({"mode": "arpls", "coef": 5})
        ws.subtract_baseline()
        ws.set_fit_model(fitting.build_fit_model([...]))
        ws.fit()
        df = ws.collect_results()
        ws.save("session.spectra")
    """

    def __init__(self) -> None:
        self.store: SpectraStore = SpectraStore()
        self.df_fit_results: Optional[pd.DataFrame] = None

    # ── ergonomics ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.store.map_names)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self)} spectra: {self.store.map_names!r})"

    @property
    def names(self) -> List[str]:
        """Names of all spectra currently in the workspace."""
        return self.store.map_names

    def _target_mds(self, names: Optional[List[str]]):
        target_names = names if names is not None else self.store.map_names
        mds = []
        for n in target_names:
            md = self.store.get_map_data(n)
            if md is None:
                raise WorkspaceError(f"No spectrum named '{n}' in this workspace.")
            mds.append(md)
        return mds

    # ── loading ─────────────────────────────────────────────────────────

    def load_files(self, paths: Union[str, Path, List[Union[str, Path]]]) -> List[str]:
        """Load one or more spectrum files (txt/csv/wdf/spc/dat).

        Returns:
            Names of spectra actually added (files whose name already
            exists in the workspace are skipped).
        """
        if isinstance(paths, (str, Path)):
            paths = [paths]

        existing = set(self.store.map_names)
        added: List[str] = []

        for p in paths:
            path = Path(p)
            data_dict = api_io.load_spectra(path)
            for item in data_dict.get("items", []):
                fname = item["name"]
                if fname in existing:
                    continue
                x0 = item["x0"]
                y0 = item["y0"]
                self.store.add_map(
                    name=fname,
                    x0=x0.copy(),
                    Y0=np.asarray(y0, dtype=np.float32).reshape(1, -1),
                    coords=np.array([[0.0, 0.0]], dtype=np.float64),
                    fnames=[fname],
                    is_active=np.array([True], dtype=bool),
                )
                md = self.store.get_map_data(fname)
                md.map_metadata = dict(item.get("metadata", {}))
                md.map_metadata["source_path"] = str(path.resolve())
                existing.add(fname)
                added.append(fname)

        return added

    # ── preprocessing ───────────────────────────────────────────────────

    def crop(
        self,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
        names: Optional[List[str]] = None,
    ) -> None:
        """Crop spectra to a wavenumber range (operates on current x/Y)."""
        for md in self._target_mds(names):
            x_src = md.x if md.x is not None else md.x0
            Y_src = md.Y if md.Y is not None else md.Y0
            mask = np.ones_like(x_src, dtype=bool)
            if range_min is not None:
                mask &= x_src >= range_min
            if range_max is not None:
                mask &= x_src <= range_max
            if not mask.any():
                raise WorkspaceError(f"Crop range excludes all points for '{md.name}'.")
            md.x = x_src[mask].copy()
            md.Y = Y_src[:, mask].copy().astype(np.float32)
            md.range_min = float(md.x[0])
            md.range_max = float(md.x[-1])

    def set_baseline(self, config: Dict[str, Any], names: Optional[List[str]] = None) -> None:
        """Configure and evaluate a baseline (does not subtract it yet)."""
        for md in self._target_mds(names):
            md.baseline_config = dict(config)
            x = md.x if md.x is not None else md.x0
            Y = md.Y if md.Y is not None else md.Y0
            md.Y_baseline = eval_baseline_batch(x, Y, md.baseline_config)

    def subtract_baseline(self, names: Optional[List[str]] = None) -> None:
        """Subtract the currently configured baseline into Y."""
        for md in self._target_mds(names):
            if md.Y_baseline is None:
                raise WorkspaceError(f"No baseline configured for '{md.name}' — call set_baseline() first.")
            Y = md.Y if md.Y is not None else md.Y0.copy()
            md.Y = (Y - md.Y_baseline).astype(np.float32)
            md.is_baseline_subtracted = np.ones(md.n_spectra, dtype=bool)
            md.Y_baseline = None

    def normalize(self, factor: Optional[float] = None, names: Optional[List[str]] = None) -> None:
        """Normalize intensities.

        Args:
            factor: None (default) normalizes each spectrum to its own max;
                a float divides every spectrum by the same shared value.
        """
        for md in self._target_mds(names):
            Y = md.Y if md.Y is not None else md.Y0.copy()
            if factor is None:
                max_vals = np.max(Y, axis=1, keepdims=True)
                max_vals[max_vals == 0] = 1.0
                md.Y = (Y / max_vals).astype(np.float32)
            else:
                md.Y = (Y / factor).astype(np.float32)
                md.intensity_norm_factor = float(factor)

    def reinit(self, names: Optional[List[str]] = None) -> None:
        """Revert spectra to raw x0/Y0, clearing crop/baseline/fit state."""
        for md in self._target_mds(names):
            md.x = md.x0.copy()
            md.Y = md.Y0.copy()
            md.range_min = None
            md.range_max = None
            md.baseline_config = None
            md.Y_baseline = None
            md.is_baseline_subtracted = False
            md.fit_model = None
            md.Y_bestfit = None
            md.Y_peaks = None
            md.peak_params = None
            md.fit_success = None
            md.fit_r2 = None
            md.param_names = []

    # ── fit models ──────────────────────────────────────────────────────

    def set_fit_model(self, fit_model: Dict[str, Any], names: Optional[List[str]] = None) -> None:
        """Apply a fit_model dict (range crop + baseline + peaks) to the
        given spectra, without fitting. Always applies against a clean reset
        (raw x0/Y0), mirroring VMWorkspaceSpectra._apply_fit_model_to_mapdata
        with indices=None.
        """
        for md in self._target_mds(names):
            md.x = md.x0.copy()
            md.Y = md.Y0.copy()
            md.range_min = None
            md.range_max = None
            md.Y_baseline = None
            md.Y_bestfit = None
            md.peak_params = None
            md.fit_success = None
            md.fit_r2 = None
            md.Y_peaks = None
            md.is_baseline_subtracted = False

            xmin = fit_model.get("range_min")
            xmax = fit_model.get("range_max")
            if xmin is not None or xmax is not None:
                mask = np.ones_like(md.x, dtype=bool)
                if xmin is not None:
                    mask &= md.x >= xmin
                if xmax is not None:
                    mask &= md.x <= xmax
                if not mask.any():
                    raise FitModelError(f"range_min/range_max crop excludes all points for '{md.name}'.")
                md.x = md.x[mask]
                md.Y = md.Y[:, mask]
                md.range_min = float(md.x[0])
                md.range_max = float(md.x[-1])

            bl_info = fit_model.get("baseline")
            if bl_info and bl_info.get("mode"):
                md.baseline_config = deepcopy(bl_info)
                md.Y_baseline = eval_baseline_batch(md.x, md.Y, md.baseline_config)
                if bl_info.get("is_subtracted", False):
                    md.Y = md.Y - md.Y_baseline
                    md.Y_baseline = None
                    md.is_baseline_subtracted = True

            md.fit_model = deepcopy(fit_model)
            _reconstruct_y_peaks(md)

    # ── fitting ─────────────────────────────────────────────────────────

    def fit(self, names: Optional[List[str]] = None, fit_params: Optional[Dict[str, Any]] = None) -> None:
        """Fit spectra that currently have a fit_model attached.

        One VBF engine call per spectrum (mirrors VMWorkspaceSpectra.fit()),
        using each spectrum's own currently-attached fit_model.

        Raises:
            FitModelError: a targeted spectrum has no fit_model attached.
            FitError: the fitting engine failed for a spectrum.
        """
        targets = names if names is not None else [
            n for n in self.store.map_names if self.store.get_map_data(n).fit_model
        ]
        for name in targets:
            md = self.store.get_map_data(name)
            if md is None:
                raise WorkspaceError(f"No spectrum named '{name}' in this workspace.")
            if not md.fit_model:
                raise FitModelError(f"'{name}' has no fit_model attached — call set_fit_model() first.")

            indices = np.array([0])
            x, Y = self.store.get_xy_batch(name, indices)
            fp = fit_params if fit_params is not None else md.fit_model.get("fit_params", {})
            result = api_fitting.fit_batch(x, Y, md.fit_model, fit_params=fp)

            self.store.set_fit_results(
                map_name=name, indices=indices,
                peak_params=result["params"], success=result["success"], r2=result["r_squared"],
                param_names=result["param_names"], fit_model=md.fit_model,
            )
            md.Y_bestfit = result["best_fits"].astype(np.float32)
            if result["peaks"]:
                md.Y_peaks = [p.astype(np.float32) for p in result["peaks"]]

    # ── results ─────────────────────────────────────────────────────────

    def collect_results(self, only_converged: bool = False) -> Optional[pd.DataFrame]:
        """Build/refresh `self.df_fit_results` from all fitted spectra/maps.

        Returns:
            The results DataFrame, or None if nothing has been fitted yet.
        """
        map_type = getattr(self, "map_type", "2Dmap")
        dfs = []
        for name in self.store.map_names:
            md = self.store.get_map_data(name)
            if md is None or not md.has_fit_results():
                continue
            peak_labels = md.fit_model.get("peak_labels") if md.fit_model else None
            df = self.store.build_fit_results_df(
                name=name, map_type=map_type, peak_labels=peak_labels, only_converged=only_converged,
            )
            if df is not None and not df.empty:
                dfs.append(df)

        if not dfs:
            self.df_fit_results = None
            return None

        df_all = pd.concat(dfs, ignore_index=True)
        if type(self).__name__ == "SpectraWorkspace":
            df_all = df_all.drop(columns=[c for c in ("X", "Y") if c in df_all.columns])

        self.df_fit_results = df_all
        return df_all

    def get_results_dataframe(self) -> Optional[pd.DataFrame]:
        """Return the last-collected results without recomputing."""
        return self.df_fit_results

    # ── misc ────────────────────────────────────────────────────────────

    def remove(self, names: List[str]) -> None:
        """Remove spectra from the workspace."""
        for n in names:
            self.store.remove_map(n)

    def get_xy(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (x, Y) for one spectrum — processed if available, else raw."""
        if self.store.get_map_data(name) is None:
            raise WorkspaceError(f"No spectrum named '{name}' in this workspace.")
        x, Y = self.store.get_xy_batch(name, np.array([0]))
        return x, Y[0]

    # ── persistence (GUI-compatible) ───────────────────────────────────

    def _add_save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Hook for subclasses to add extra metadata keys before saving."""

    def _load_extra_metadata(self, metadata: Dict[str, Any]) -> None:
        """Hook for subclasses to restore extra metadata keys after loading."""

    def save(self, file_path: Union[str, Path]) -> None:
        """Write a workspace file with the same structure the GUI's own
        "Save work" produces (format_version=2), openable directly in the GUI.
        """
        arrays: Dict[str, Any] = {}
        store_meta: Dict[str, Any] = {}
        for map_name in self.store.map_names:
            arrays.update(self.store.to_npz_dict(map_name))
            store_meta[map_name] = self.store.to_metadata_dict(map_name)

        metadata: Dict[str, Any] = {
            "format_version": 2,
            "store_meta": _sanitize_for_json(store_meta),
        }
        self._add_save_metadata(metadata)

        dataframes = {}
        if self.df_fit_results is not None and not self.df_fit_results.empty:
            dataframes["df_fit_results"] = self.df_fit_results

        try:
            WorkspaceIO.save_workspace(str(file_path), metadata, arrays, dataframes)
        except Exception as e:
            raise WorkspaceError(f"Failed to save workspace to {file_path}: {e}") from e

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "SpectraWorkspace":
        """Load a workspace file written by either this API or the GUI.

        Raises:
            WorkspaceError: file not found, or a legacy format-v1 file (open
                once in the GUI and re-save to upgrade).
        """
        try:
            metadata, arrays, dataframes, is_legacy = WorkspaceIO.load_workspace(str(file_path))
        except FileNotFoundError as e:
            raise WorkspaceError(str(e)) from e

        if is_legacy or metadata.get("format_version", 1) < 2:
            raise WorkspaceError(
                f"{file_path} is a legacy format-v1 workspace file, which this API does not "
                "read directly. Open it once in the SPECTROview GUI and re-save it to upgrade "
                "it to format v2."
            )

        ws = cls()
        ws.store = SpectraStore()
        store_meta = metadata.get("store_meta", {})
        for map_name, meta in store_meta.items():
            SpectraStore.load_map_from_npz(arrays, meta, map_name, store=ws.store)
            md = ws.store.get_map_data(map_name)
            if md:
                _restore_preprocessed_state(md)

        ws.df_fit_results = dataframes.get("df_fit_results") if dataframes else None
        ws._load_extra_metadata(metadata)
        return ws


class MapsWorkspace(SpectraWorkspace):
    """A stateful, file-compatible equivalent of the GUI's Maps Workspace.

    One `MapData` block per loaded hyperspectral map (all pixel rows share
    that map's fit_model). A pixel's identifier is
    ``f"{map_name}_({x}, {y})"``, matching the GUI's convention.

    Example::

        ws = MapsWorkspace()
        ws.load_files(["2Dmap_Si.txt"])
        ws.set_fit_model(fitting.load_fit_model_template("fit_model_Si.json"))
        ws.fit()
        df = ws.collect_results()
        xi, yi, zi = ws.get_heatmap("2Dmap_Si", "ampli_Si")
        ws.save("session.maps")
    """

    def __init__(self, map_type: str = "2Dmap") -> None:
        super().__init__()
        self.maps: Dict[str, pd.DataFrame] = {}
        self.maps_metadata: Dict[str, dict] = {}
        self.map_type = map_type

    # ── loading ─────────────────────────────────────────────────────────

    def load_files(self, paths: Union[str, Path, List[Union[str, Path]]]) -> List[str]:
        """Load one or more hyperspectral map files (txt/csv/wdf/spc)."""
        if isinstance(paths, (str, Path)):
            paths = [paths]

        added: List[str] = []
        for p in paths:
            path = Path(p)
            map_name = path.stem
            if map_name in self.maps:
                continue

            result = api_io.load_map(path)
            if isinstance(result, tuple):
                map_df, metadata = result
            else:
                map_df, metadata = result, {}

            self.maps[map_name] = map_df
            self.maps_metadata[map_name] = metadata
            self._extract_spectra_from_map(map_name, map_df)
            added.append(map_name)

        return added

    def _extract_spectra_from_map(self, map_name: str, map_df: pd.DataFrame) -> None:
        wavenumber_cols = [c for c in map_df.columns if c not in ("X", "Y")]
        x_values = pd.to_numeric(pd.Index(wavenumber_cols), errors="coerce").tolist()
        if len(x_values) > 1:
            x_values = x_values[:-1]  # skip last value, matches GUI behavior
            wavenumber_cols = wavenumber_cols[:-1]
        x_data = np.asarray(x_values, dtype=np.float64)

        x_positions = map_df["X"].values
        y_positions = map_df["Y"].values
        intensity_data = map_df[wavenumber_cols].values

        N = len(map_df)
        coords = np.column_stack([x_positions, y_positions]).astype(np.float64)
        Y0 = intensity_data.astype(np.float32)
        fnames = [f"{map_name}_({float(x_positions[i])}, {float(y_positions[i])})" for i in range(N)]

        self.store.add_map(
            name=map_name, x0=x_data, Y0=Y0, coords=coords, fnames=fnames,
            is_active=np.ones(N, dtype=bool),
        )
        md = self.store.get_map_data(map_name)
        if md:
            md.map_metadata = self.maps_metadata.get(map_name, {})

    # ── fitting ─────────────────────────────────────────────────────────

    def fit(self, map_names: Optional[List[str]] = None, fit_params: Optional[Dict[str, Any]] = None) -> None:
        """Fit all pixels of the given maps (default: all maps with a
        fit_model attached). One VBF engine call per map, covering every
        pixel row in that map — already fully vectorized at map scale.

        Raises:
            FitModelError: a targeted map has no fit_model attached.
            FitError: the fitting engine failed for a map.
        """
        targets = map_names if map_names is not None else [
            n for n in self.store.map_names if self.store.get_map_data(n).fit_model
        ]
        for name in targets:
            md = self.store.get_map_data(name)
            if md is None:
                raise WorkspaceError(f"No map named '{name}' in this workspace.")
            if not md.fit_model:
                raise FitModelError(f"'{name}' has no fit_model attached — call set_fit_model() first.")

            indices = np.arange(md.n_spectra)
            x, Y = self.store.get_xy_batch(name, indices)
            fp = fit_params if fit_params is not None else md.fit_model.get("fit_params", {})
            result = api_fitting.fit_batch(x, Y, md.fit_model, fit_params=fp)

            self.store.set_fit_results(
                map_name=name, indices=indices,
                peak_params=result["params"], success=result["success"], r2=result["r_squared"],
                param_names=result["param_names"], fit_model=md.fit_model,
            )
            md.Y_bestfit = result["best_fits"].astype(np.float32)
            if result["peaks"]:
                md.Y_peaks = [p.astype(np.float32) for p in result["peaks"]]

    # ── map analysis ────────────────────────────────────────────────────

    def get_heatmap(
        self,
        map_name: str,
        value_col: str,
        x_range: Optional[Tuple[float, float]] = None,
        method: str = "linear",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a 2D grid for `value_col` on `map_name`.

        Args:
            map_name: a loaded map's name.
            value_col: 'Intensity' or 'Area' (computed from raw intensities
                within `x_range`), or any fit-parameter column present in
                `self.df_fit_results` (call `collect_results()` first).
            x_range: (min, max) wavenumber range used for 'Intensity'/'Area'.
            method: interpolation method for wafer maps (`self.map_type != '2Dmap'`).

        Returns:
            (xi, yi, zi) — see `spectroview.model.heatmap.build_heatmap_grid`.

        Raises:
            WorkspaceError: unknown map_name, or value_col not available.
        """
        md = self.store.get_map_data(map_name)
        map_df = self.maps.get(map_name)
        if md is None or map_df is None:
            raise WorkspaceError(f"No map named '{map_name}' in this workspace.")

        wavenumber_cols = [c for c in map_df.columns if c not in ("X", "Y")]
        if x_range is not None:
            lo, hi = x_range
            wavenumber_cols = [c for c in wavenumber_cols if lo <= float(c) <= hi]
            if not wavenumber_cols:
                raise WorkspaceError(f"x_range {x_range} excludes every wavenumber column.")

        x_col = map_df["X"].to_numpy(dtype=np.float64)
        y_col = map_df["Y"].to_numpy(dtype=np.float64)

        if value_col in ("Intensity", "Area"):
            sub = map_df[wavenumber_cols].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
            z = sub.max(axis=1).to_numpy() if value_col == "Intensity" else sub.sum(axis=1).to_numpy()
        else:
            if self.df_fit_results is None or value_col not in self.df_fit_results.columns:
                raise WorkspaceError(
                    f"'{value_col}' is not 'Intensity'/'Area' and not a column in results — "
                    "call collect_results() first."
                )
            results = self.df_fit_results[self.df_fit_results["Filename"] == map_name]
            coords_df = pd.DataFrame({"X": x_col.round(6), "Y": y_col.round(6)})
            res = results[["X", "Y", value_col]].copy()
            res["X"] = res["X"].round(6)
            res["Y"] = res["Y"].round(6)
            merged = coords_df.merge(res, on=["X", "Y"], how="left")
            z = merged[value_col].to_numpy(dtype=np.float64)

        return build_heatmap_grid(x_col, y_col, z, map_type=self.map_type, method=method)

    def extract_profile(
        self,
        map_name: str,
        value_col: str,
        point1: Tuple[float, float],
        point2: Tuple[float, float],
        n_samples: int = 100,
        x_range: Optional[Tuple[float, float]] = None,
        method: str = "linear",
    ) -> pd.DataFrame:
        """Sample `value_col`'s heatmap along the line from point1 to point2.

        Returns:
            DataFrame with columns X, Y, distance, values.
        """
        from scipy.interpolate import RegularGridInterpolator

        xi, yi, zi = self.get_heatmap(map_name, value_col, x_range=x_range, method=method)
        interpolator = RegularGridInterpolator((yi, xi), zi, bounds_error=False, fill_value=np.nan)

        x1, y1 = point1
        x2, y2 = point2
        xs = np.linspace(x1, x2, n_samples)
        ys = np.linspace(y1, y2, n_samples)
        zs = interpolator(np.column_stack([ys, xs]))
        dist = np.sqrt((xs - x1) ** 2 + (ys - y1) ** 2)

        return pd.DataFrame({"X": xs, "Y": ys, "distance": dist, "values": zs})

    # ── misc ────────────────────────────────────────────────────────────

    def remove(self, names: List[str]) -> None:
        for n in names:
            self.store.remove_map(n)
            self.maps.pop(n, None)
            self.maps_metadata.pop(n, None)

    # ── persistence ─────────────────────────────────────────────────────

    def _add_save_metadata(self, metadata: Dict[str, Any]) -> None:
        metadata["maps_metadata"] = _sanitize_for_json(self.maps_metadata)
        metadata["map_type"] = self.map_type

    def _load_extra_metadata(self, metadata: Dict[str, Any]) -> None:
        self.maps_metadata = metadata.get("maps_metadata", {})
        self.map_type = metadata.get("map_type", "2Dmap")
        self.maps = {}
        for map_name in self.store.map_names:
            md = self.store.get_map_data(map_name)
            col_names = list(map(str, md.x0))
            df = pd.DataFrame(md.Y0, columns=col_names)
            df.insert(0, "Y", md.coords[:, 1])
            df.insert(0, "X", md.coords[:, 0])
            self.maps[map_name] = df
