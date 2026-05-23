"""SpectraStore — tensor-centric data backbone for SPECTROview.

Owns all heavy numerical data for a collection of spectra (typically one or
more hyperspectral maps) as contiguous NumPy arrays per map. Replaces the
per-spectrum Python-object iteration pattern inherited from Fitspy.

Architecture
------------
Data is organized per map in a `MapData` structure:

    ┌────────────────────────────────────────────────────────┐
    │  MapData (per-map tensor block)                        │
    │  ─────────────────────────                             │
    │  x0        : float64[M]      wavenumber axis           │
    │  Y0        : float32[N, M]   raw intensities           │
    │  coords    : float64[N, 2]   spatial (X, Y)            │
    │  is_active : bool[N]         checkbox state            │
    │  fnames    : list[str]       unique identifiers        │
    │  colors    : list[str|None]                            │
    │  labels    : list[str|None]                            │
    │  peak_params: float64[N, K]  fitted parameters         │
    │  fit_success: bool[N]                                  │
    │  fit_r2    : float64[N]                                │
    │  param_names: list[str]      K param names             │
    │  fit_model  : dict           shared peak model def     │
    └────────────────────────────────────────────────────────┘

Multiple maps can coexist in the same SpectraStore (each with independent
x-axis length, enabling the Maps workspace to hold heterogeneous datasets).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# MapData — tensor block for a single hyperspectral map
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MapData:
    """All numerical data for a single hyperspectral map."""
    name: str

    # Spectral arrays — shapes (N, M) / (M,)
    x0: np.ndarray            # float64[M]  wavenumber axis
    Y0: np.ndarray            # float32[N, M]  raw intensities
    coords: np.ndarray        # float64[N, 2]  spatial positions

    # Per-spectrum flags / metadata (lightweight)
    is_active: np.ndarray     # bool[N]
    fnames: list              # list[str][N]
    colors: list              # list[str|None][N]
    labels: list              # list[str|None][N]

    Y: Optional[np.ndarray] = None  # float32[N, M_proc] processed intensities
    x: Optional[np.ndarray] = None  # float64[M_proc] processed wavenumber axis

    # Fit results (filled after TensorEngine.fit())
    peak_params: Optional[np.ndarray] = None   # float64[N, K]
    fit_success: Optional[np.ndarray] = None   # bool[N]
    fit_r2: Optional[np.ndarray] = None        # float64[N]
    param_names: list = field(default_factory=list)
    fit_model: Optional[dict] = None

    @property
    def n_spectra(self) -> int:
        return len(self.fnames)

    @property
    def n_wavenumbers(self) -> int:
        return len(self.x0)

    def has_fit_results(self) -> bool:
        return self.peak_params is not None and len(self.param_names) > 0


# ═══════════════════════════════════════════════════════════════════════════
# MapInfo — lightweight map descriptor (for API compatibility)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MapInfo:
    """Lightweight descriptor returned by get_map_info()."""
    name: str
    row_start: int   # always 0 for per-map storage (kept for API compat)
    row_end: int
    n_spectra: int
    n_wavenumbers: int


# ═══════════════════════════════════════════════════════════════════════════
# SpectraStore
# ═══════════════════════════════════════════════════════════════════════════

class SpectraStore:
    """Tensor-centric container for hyperspectral map data.

    Each map is stored independently as a `MapData` block containing all its
    numerical data in contiguous NumPy arrays. This supports:
    - Multiple maps with different x-axis lengths (heterogeneous datasets).
    - O(1) per-map access (no global index arithmetic needed).
    - Efficient save/load via per-map NPZ blocks.

    Phase-1 usage: SpectraStore coexists with self.spectra (the legacy
    MSpectrum list) in VMWorkspaceMaps. Phase-4 will remove self.spectra.
    """

    # ── Construction ──────────────────────────────────────────────────────

    def __init__(self):
        self.clear()

    def clear(self):
        """Reset store to empty state."""
        # Per-map data blocks
        self._maps: dict[str, MapData] = {}

    @property
    def n_spectra(self) -> int:
        """Total number of spectra across all maps."""
        return sum(m.n_spectra for m in self._maps.values())

    @property
    def map_names(self) -> list[str]:
        return list(self._maps.keys())

    # ── Map registration ─────────────────────────────────────────────────

    def add_map(
        self,
        name: str,
        x0: np.ndarray,
        Y0: np.ndarray,
        coords: np.ndarray,
        fnames: list,
        is_active: Optional[np.ndarray] = None,
        colors: Optional[list] = None,
        labels: Optional[list] = None,
    ):
        """Register a new map into the store.

        Args:
            name:    Unique map name (typically the file stem).
            x0:      Wavenumber axis               float64[M].
            Y0:      Raw intensity matrix           float32[N, M].
            coords:  Spatial coordinates            float64[N, 2].
            fnames:  Unique fname strings           list[N].
            is_active: Initial checkbox state       bool[N].  Defaults to all True.
            colors:  Per-spectrum color             list[N].   Defaults to None.
            labels:  Per-spectrum label             list[N].   Defaults to None.
        """
        N, M = Y0.shape
        if is_active is None:
            is_active = np.ones(N, dtype=bool)
        if colors is None:
            colors = [None] * N
        if labels is None:
            labels = [None] * N

        self._maps[name] = MapData(
            name=name,
            x0=x0.astype(np.float64, copy=False),
            Y0=Y0.astype(np.float32, copy=False),
            coords=coords.astype(np.float64, copy=False),
            is_active=is_active.copy(),
            fnames=list(fnames),
            colors=list(colors),
            labels=list(labels),
        )

    def remove_map(self, name: str):
        """Remove a map and its data from the store."""
        self._maps.pop(name, None)

    # ── Map-level helpers ─────────────────────────────────────────────────

    def get_map_data(self, name: str) -> Optional[MapData]:
        """Return the raw MapData block for `name`."""
        return self._maps.get(name)

    def get_map_info(self, name: str) -> Optional[MapInfo]:
        """Return a lightweight MapInfo descriptor (for API compatibility)."""
        md = self._maps.get(name)
        if md is None:
            return None
        return MapInfo(
            name=name,
            row_start=0,
            row_end=md.n_spectra,
            n_spectra=md.n_spectra,
            n_wavenumbers=md.n_wavenumbers,
        )

    def get_map_slice(self, name: str) -> slice:
        """Return slice(0, N) for the named map (compat shim)."""
        md = self._maps[name]
        return slice(0, md.n_spectra)

    def get_map_indices(self, name: str) -> np.ndarray:
        md = self._maps[name]
        return np.arange(md.n_spectra)

    def get_active_mask_for_map(self, name: str) -> np.ndarray:
        return self._maps[name].is_active.copy()

    def get_fnames_for_map(self, name: str) -> list:
        return self._maps[name].fnames

    # Thin wrappers to _is_active for _writeback_fit_results_to_store
    @property
    def _is_active(self):
        """Flat array of all is_active flags (across all maps in insertion order).
        
        Provided for backward compatibility with code that uses
        store._is_active[info.row_start + i].
        Each map's slice starts at its cumulative offset.
        """
        if not self._maps:
            return np.array([], dtype=bool)
        parts = [md.is_active for md in self._maps.values()]
        return np.concatenate(parts)

    # ── Per-spectrum attribute access ─────────────────────────────────────

    def get_fname(self, map_name: str, local_idx: int) -> str:
        return self._maps[map_name].fnames[local_idx]

    def get_is_active(self, map_name: str, local_idx: int) -> bool:
        return bool(self._maps[map_name].is_active[local_idx])

    def set_is_active(self, map_name: str, local_idx: int, value: bool):
        self._maps[map_name].is_active[local_idx] = value

    def get_color(self, map_name: str, local_idx: int) -> Optional[str]:
        return self._maps[map_name].colors[local_idx]

    def set_color(self, map_name: str, local_idx: int, value: Optional[str]):
        self._maps[map_name].colors[local_idx] = value

    # ── x0 accessor (per-map) ─────────────────────────────────────────────

    @property
    def x0(self) -> Optional[np.ndarray]:
        """Wavenumber axis of the first map (convenience property).

        For multi-map scenarios, use get_x0_for_map(name) instead.
        """
        if not self._maps:
            return None
        return next(iter(self._maps.values())).x0

    def get_x0_for_map(self, name: str) -> Optional[np.ndarray]:
        md = self._maps.get(name)
        return md.x0 if md else None

    # ── Raw data accessors (per-map) ──────────────────────────────────────

    def get_xy_batch(self, map_name: str, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return X, Y arrays for a subset of spectra in a map.
        Returns processed (x, Y) if available, otherwise raw (x0, Y0).
        """
        md = self._maps[map_name]
        x = md.x if md.x is not None else md.x0
        Y = md.Y if md.Y is not None else md.Y0
        return x, Y[indices].astype(np.float64)

    def get_processed_batch(self, map_name: str, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (x0, Y_proc) for given indices. Falls back to raw if not set."""
        md = self._maps[map_name]
        return md.x0, md.Y0[indices].astype(np.float64)

    # ── Preprocessing (Phase 5 Vectorized) ────────────────────────────────

    def clear_preprocess(self, map_name: str):
        """Clear preprocessed data (Y and x) for a map, reverting to raw Y0 and x0."""
        md = self._maps.get(map_name)
        if md:
            md.Y = None
            md.x = None

    def batch_preprocess(self, map_name: str, baseline_config: dict, range_min: float = None, range_max: float = None):
        """Apply range cropping and baseline subtraction to the entire map using matrix operations."""
        md = self._maps.get(map_name)
        if not md: return

        x = md.x0
        # 1. Apply range cropping
        if range_min is not None or range_max is not None:
            mask = np.logical_and(
                x >= (range_min if range_min is not None else -np.inf),
                x <= (range_max if range_max is not None else np.inf)
            )
            x_proc = x[mask]
            Y_proc = md.Y0[:, mask].copy()
        else:
            x_proc = x.copy()
            Y_proc = md.Y0.copy()

        if len(x_proc) == 0:
            return # Empty range, abort preprocessing

        # 2. Vectorized Baseline Subtraction
        bl_mode = baseline_config.get("mode")
        bl_attached = baseline_config.get("attached", False)
        
        if bl_mode is not None:
            if not bl_attached:
                # Static mode: compute on first spectrum, subtract from all
                try:
                    from fitspy.core.baseline import BaseLine
                    bl = BaseLine()
                    for k, v in baseline_config.items():
                        setattr(bl, k, v)
                    y_static = bl.eval(x_proc, Y_proc[0], attached=False)
                    Y_proc -= y_static
                except Exception:
                    pass
            elif bl_attached and bl_mode == 'Linear':
                # Fast matrix-based linear interpolation between attached points
                points = baseline_config.get("points", [[], []])
                if points and len(points[0]) >= 1:
                    pts_x = np.array(points[0])
                    # Find indices in x_proc
                    pt_indices = np.array([np.argmin(np.abs(x_proc - px)) for px in pts_x])
                    
                    if len(pt_indices) == 1:
                        # Single point constant subtraction
                        Y_proc -= Y_proc[:, pt_indices[0]][:, None]
                    else:
                        # Multi-point linear interpolation for each row (vectorized)
                        # We use scipy interp1d over the axis
                        try:
                            from scipy.interpolate import interp1d
                            y_pts = Y_proc[:, pt_indices]
                            # interpolate for each row. 
                            func = interp1d(x_proc[pt_indices], y_pts, axis=1, fill_value="extrapolate")
                            y_baseline = func(x_proc)
                            Y_proc -= y_baseline
                        except Exception:
                            pass
            else:
                # For more complex baseline algorithms (like polynomial or arPLS), 
                # we could apply them per-row, but for performance we leave it as is or fallback.
                # In Phase 5, the TensorFitThread will handle complex modes via its own loop, 
                # but for instant GUI preview, linear/static are fast enough.
                pass
                
        md.x = x_proc
        md.Y = Y_proc

    # ── Fit result accessors ──────────────────────────────────────────────

    def set_fit_results(
        self,
        map_name: str,
        indices: np.ndarray,
        peak_params: np.ndarray,
        success: np.ndarray,
        r2: np.ndarray,
        param_names: list,
        fit_model: dict,
    ):
        """Write batch fit results into a specific map's data block.

        Called by VMWorkspaceMaps._writeback_fit_results_to_store() after
        TensorFitThread finishes.

        Args:
            map_name:    The target map name.
            indices:     Local row indices within the map (0..N-1).
            peak_params: (len(indices), K) fitted parameter values.
            success:     (len(indices),) convergence flags.
            r2:          (len(indices),) R² values.
            param_names: List of K parameter names.
            fit_model:   Fit model dict (for serialization).
        """
        md = self._maps[map_name]
        N = md.n_spectra
        K = len(param_names)

        if md.peak_params is None or md.peak_params.shape != (N, K):
            md.peak_params = np.zeros((N, K), dtype=np.float64)
            md.fit_success = np.zeros(N, dtype=bool)
            md.fit_r2 = np.zeros(N, dtype=np.float64)

        md.peak_params[indices] = peak_params
        md.fit_success[indices] = success
        md.fit_r2[indices] = r2
        md.param_names = param_names
        md.fit_model = fit_model

    def has_fit_results(self, map_name: Optional[str] = None) -> bool:
        """Check if any fit results exist.

        If map_name is given, check only that map.
        """
        if map_name:
            md = self._maps.get(map_name)
            return md is not None and md.has_fit_results()
        return any(md.has_fit_results() for md in self._maps.values())

    def get_fit_r2_for_map(self, name: str) -> Optional[np.ndarray]:
        md = self._maps.get(name)
        return md.fit_r2 if md else None

    def get_peak_param_for_map(self, name: str, param_name: str) -> Optional[np.ndarray]:
        """Return a 1D array of one parameter's values across all spectra in a map."""
        md = self._maps.get(name)
        if md is None or md.peak_params is None:
            return None
        if param_name not in md.param_names:
            return None
        k = md.param_names.index(param_name)
        return md.peak_params[:, k]

    # ── DataFrame construction ────────────────────────────────────────────

    def build_map_dataframe(self, name: str) -> pd.DataFrame:
        """Reconstruct the hyperspectral map DataFrame for a given map.

        Columns: [X, Y, w0, w1, w2, ...] where wi are wavenumber values.
        """
        md = self._maps[name]
        col_names = ['X', 'Y'] + [str(float(w)) for w in md.x0]
        data = np.hstack([md.coords, md.Y0.astype(np.float64)])
        return pd.DataFrame(data, columns=col_names)

    def build_fit_results_df(
        self,
        name: str,
        map_type: str = '2Dmap',
        peak_labels: Optional[list] = None,
    ) -> Optional[pd.DataFrame]:
        """Build the full fit results DataFrame for a map (vectorized).

        Columns: Filename, X, Y, [Zone, Quadrant for wafers], param0, param1, ...

        This replaces the per-object Python loop in collect_fit_results().
        Returns None if no fit results are available.
        """
        md = self._maps.get(name)
        if md is None or not md.has_fit_results():
            return None

        # Filter to active spectra that converged
        mask = md.is_active & md.fit_success
        if not mask.any():
            return None

        coords = md.coords[mask]
        params = md.peak_params[mask]

        # Build column names (apply peak labels if available)
        col_names = list(md.param_names)
        if peak_labels:
            col_names = self._apply_peak_labels(col_names, peak_labels)

        # Add _area columns
        area_cols = self._compute_area_columns(params, md.param_names, col_names)

        # Sort columns by parameter type
        param_priority = {
            'x0': 0, 'fwhm': 1, 'ampli': 2, 'area': 3,
            'sigma': 4, 'gamma': 5, 'fraction': 6, 'height': 7,
        }

        def sort_key(cname):
            if '_' in cname:
                ptype = cname.split('_', 1)[0]
                return (param_priority.get(ptype, 999), cname)
            return (999, cname)

        all_param_data = {c: params[:, i].round(4) for i, c in enumerate(col_names)}
        all_param_data.update(area_cols)
        sorted_param_cols = sorted(all_param_data.keys(), key=sort_key)

        data = {
            'Filename': [name] * int(mask.sum()),
            'X': coords[:, 0],
            'Y': coords[:, 1],
        }

        if map_type != '2Dmap':
            radius = 150
            if '300' in map_type:
                radius = 150
            elif '200' in map_type:
                radius = 100
            elif '100' in map_type:
                radius = 50

            x, y = coords[:, 0], coords[:, 1]
            dist = np.sqrt(x**2 + y**2)

            zone_arr = np.full(len(x), np.nan, dtype=object)
            zone_arr[dist <= radius * 0.35] = 'Center'
            zone_arr[(dist > radius * 0.35) & (dist < radius * 0.8)] = 'Mid-Radius'
            zone_arr[dist >= radius * 0.8] = 'Edge'
            data['Zone'] = zone_arr

            quadrant_arr = np.full(len(x), np.nan, dtype=object)
            quadrant_arr[(x < 0) & (y < 0)] = 'Q1'
            quadrant_arr[(x < 0) & (y > 0)] = 'Q2'
            quadrant_arr[(x > 0) & (y > 0)] = 'Q3'
            quadrant_arr[(x > 0) & (y < 0)] = 'Q4'
            data['Quadrant'] = quadrant_arr

        for col in sorted_param_cols:
            data[col] = all_param_data[col]

        return pd.DataFrame(data)

    # ── Serialization (format v4) ─────────────────────────────────────────

    def to_npz_dict(self, map_name: str) -> dict:
        """Export per-map arrays as a dict for NPZ storage.

        Keys produced (prefix = 'store_{name}'):
            _coords      float64[N, 2]
            _x0          float64[M]
            _y0          float32[N, M]
            _is_active   bool[N]
            _peak_params float64[N, K]   (if fit results exist)
            _fit_success bool[N]         (if fit results exist)
            _fit_r2      float64[N]      (if fit results exist)
        """
        md = self._maps[map_name]
        pfx = f'store_{map_name}'

        result = {
            f'{pfx}_coords': md.coords,
            f'{pfx}_x0': md.x0,
            f'{pfx}_y0': md.Y0,
            f'{pfx}_is_active': md.is_active,
        }

        if md.peak_params is not None:
            result[f'{pfx}_peak_params'] = md.peak_params
            result[f'{pfx}_fit_success'] = md.fit_success
            result[f'{pfx}_fit_r2'] = md.fit_r2

        return result

    def to_metadata_dict(self, map_name: str) -> dict:
        """Export lightweight per-map metadata for JSON storage.

        Keys produced:
            fnames:      list[str]
            colors:      list[str|None]
            labels:      list[str|None]
            param_names: list[str]
            fit_model:   dict
        """
        md = self._maps[map_name]
        return {
            'fnames': md.fnames,
            'colors': md.colors,
            'labels': md.labels,
            'param_names': md.param_names,
            'fit_model': md.fit_model or {},
        }

    @classmethod
    def load_map_from_npz(
        cls,
        arrays: dict,
        meta: dict,
        map_name: str,
        store: Optional['SpectraStore'] = None,
    ) -> 'SpectraStore':
        """Restore a single map from NPZ arrays + metadata dict (format v4).

        If `store` is provided, the map is appended to it.
        Otherwise a new SpectraStore is created.
        """
        if store is None:
            store = cls()

        pfx = f'store_{map_name}'
        coords = arrays[f'{pfx}_coords']
        x0 = arrays[f'{pfx}_x0']
        Y0 = arrays[f'{pfx}_y0']
        N = len(coords)
        is_active = arrays.get(f'{pfx}_is_active', np.ones(N, dtype=bool))

        fnames = meta.get('fnames', [])
        colors = meta.get('colors', [None] * len(fnames))
        labels = meta.get('labels', [None] * len(fnames))

        store.add_map(
            name=map_name,
            x0=x0,
            Y0=Y0,
            coords=coords,
            fnames=fnames,
            is_active=is_active,
            colors=colors,
            labels=labels,
        )

        # Restore fit results if present
        pk = f'{pfx}_peak_params'
        if pk in arrays:
            param_names = meta.get('param_names', [])
            K = arrays[pk].shape[1]
            indices = np.arange(N)

            store.set_fit_results(
                map_name=map_name,
                indices=indices,
                peak_params=arrays[pk],
                success=arrays.get(f'{pfx}_fit_success', np.zeros(N, dtype=bool)),
                r2=arrays.get(f'{pfx}_fit_r2', np.zeros(N, dtype=np.float64)),
                param_names=param_names,
                fit_model=meta.get('fit_model', {}),
            )

        return store

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _apply_peak_labels(param_names: list, peak_labels: list) -> list:
        """Replace numeric peak prefixes with user labels using legacy column format.

        Legacy format (matching replace_peak_labels in utils.py):
            m01_x0 → x0_Si    (param first, then label)
            m01_fwhm → fwhm_Si

        This keeps column names consistent between the fast tensor path and
        the legacy per-object path.
        """
        result = []
        for name in param_names:
            if '_' in name:
                prefix, param_part = name.split('_', 1)
                # prefix is like 'm01', 'm02', ...
                try:
                    peak_index = int(prefix[1:]) - 1
                    if 0 <= peak_index < len(peak_labels):
                        label = peak_labels[peak_index]
                        result.append(f"{param_part}_{label}")
                        continue
                except (ValueError, IndexError):
                    pass
            result.append(name)
        return result

    @staticmethod
    def _compute_area_columns(
        params: np.ndarray,
        param_names: list,
        col_names: list,
    ) -> dict:
        """Compute peak area columns (ampli * fwhm * factor) for each peak.

        Returns a dict of {area_col_name: array}. Fully vectorized.
        """
        area_cols = {}
        SQRT_PI_LN2 = np.sqrt(np.pi / np.log(2))

        # Group parameter indices by original prefix (e.g. 'm01')
        orig_prefixes: dict[str, dict[str, int]] = {}
        for i, name in enumerate(param_names):
            if '_' in name:
                prefix, short = name.split('_', 1)
                orig_prefixes.setdefault(prefix, {})[short] = i

        for prefix, orig_idx in orig_prefixes.items():
            if 'fwhm' not in orig_idx or 'ampli' not in orig_idx:
                continue

            fwhm = params[:, orig_idx['fwhm']]
            ampli = params[:, orig_idx['ampli']]

            if 'alpha' in orig_idx:
                alpha = params[:, orig_idx['alpha']]
                area = ampli * fwhm * (alpha * np.pi / 2 + (1 - alpha) * SQRT_PI_LN2 / 2)
            else:
                area = ampli * fwhm * np.pi / 2  # Lorentzian default

            # Derive the column name using the mapped column name for 'ampli'
            # If original name was 'm01_ampli', display name is col_names[orig_idx['ampli']] (e.g. 'ampli_1')
            # We want to extract '1' to name the column 'area_1'.
            disp_ampli = col_names[orig_idx['ampli']]
            if '_' in disp_ampli:
                _, label = disp_ampli.split('_', 1)
                area_name = f'area_{label}'
            else:
                area_name = f'area_{prefix}'

            area_cols[area_name] = area.round(4)

        return area_cols
