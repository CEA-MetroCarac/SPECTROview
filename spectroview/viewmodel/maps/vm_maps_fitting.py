"""Maps-specific batch fitting and wafer fit-results post-processing mixin."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from spectroview.i18n import tr


class MapsFittingMixin:
    """Overrides fitting and result collection for 2D/wafer hyperspectral maps."""

    def _apply_fit_model_and_run(self, fit_model: dict, apply_all: bool):
        """Apply ``fit_model`` to the target map(s), refresh the UI, and launch fitting.

        Shared tail of apply_fit_model()/paste_fit_model() — the only
        difference between the two is how they obtain ``fit_model`` (load
        from a JSON file vs. read from the clipboard).
        """
        maps_to_apply = (
            self.store.map_names
            if apply_all
            else ([self.current_map_name] if self.current_map_name else [])
        )
        for name in maps_to_apply:
            md = self.store.get_map_data(name)
            if md:
                self._apply_fit_model_to_mapdata(md, fit_model)
                self.clear_map_cache_requested.emit(name)

        if self.current_map_name:
            self._emit_selected_spectra()
            self._emit_list_update()

        if apply_all:
            fnames = []
            for name in self.store.map_names:
                md = self.store.get_map_data(name)
                if md:
                    fnames.extend(
                        [f for i, f in enumerate(md.fnames) if md.is_active[i]]
                    )
        else:
            fnames = self._get_active_spectra()

        self._run_fit_thread(fit_model, fnames, apply_all=apply_all)

    def apply_fit_model(self, apply_all: bool = False):
        """Override parent to apply loaded fit model and sync preprocessing/fit thread."""
        fit_model = self._load_selected_fit_model_dict()
        if fit_model is None:
            return

        self._apply_fit_model_and_run(fit_model, apply_all)

    def paste_fit_model(self, apply_all: bool = False) -> bool:
        """Override parent to apply clipboard fit model and sync preprocessing/fit thread."""
        if not hasattr(self, "_fitmodel_clipboard") or self._fitmodel_clipboard is None:
            self.notify.emit(tr("No fit model copied."))
            return False

        if not apply_all and not self.selected_fnames and not self.current_map_name:
            self.notify.emit(tr("No spectrum selected."))
            return False

        self._fitmodel_clipboard["fit_params"] = self.settings.load_fit_settings()

        self._apply_fit_model_and_run(deepcopy(self._fitmodel_clipboard), apply_all)
        self.clipboard_action_succeeded.emit("fitmodel_paste")
        return True

    def fit(self, apply_all: bool = False):
        """Override: Fitting action for the Maps workspace."""
        if self._is_fitting:
            self.notify.emit(tr("Fit already in progress. Please wait..."))
            return

        tasks = []
        if apply_all:
            for map_name in self.store.map_names:
                md = self.store.get_map_data(map_name)
                if md and md.fit_model and md.fit_model.get("peak_models"):
                    self._reindex_fit_model(md)
                    tasks.append({
                        "map_name": map_name,
                        "indices": np.arange(md.n_spectra),
                        "fit_model": md.fit_model,
                    })
        else:
            md = self.store.get_map_data(self.current_map_name)
            if not md or not md.fit_model or not md.fit_model.get("peak_models"):
                self.notify.emit(tr("No peaks to fit."))
                return
            self._reindex_fit_model(md)
            fnames = self._get_selected_spectra()
            fname_set = set(fnames)
            indices = np.array([i for i, f in enumerate(md.fnames) if f in fname_set])
            if len(indices) == 0:
                return
            tasks = [{
                "map_name": self.current_map_name,
                "indices": indices,
                "fit_model": md.fit_model,
            }]

        if not tasks:
            return

        self._launch_fit_thread(tasks)

    def _run_fit_thread(
        self, fit_model: dict, fnames: list[str], apply_all: bool = False
    ):
        """Override: Run fit thread for Maps workspace loaded fit models.

        When apply_all=True, all maps' spectra are merged into a single batch
        (grouped by x-axis length) so the engine fits everything in one vectorized
        call for maximum speed.
        """
        if self._is_fitting:
            self.notify.emit(tr("Fit already in progress. Please wait..."))
            return

        tasks = []

        if "fit_params" not in fit_model:
            fit_model["fit_params"] = self.settings.load_fit_settings()

        if apply_all:
            groups = {}
            for map_name in self.store.map_names:
                md = self.store.get_map_data(map_name)
                if md:
                    x_len = len(md.x_axis)
                    if x_len not in groups:
                        groups[x_len] = []
                    groups[x_len].append((map_name, md))

            for _x_len, group in groups.items():
                map_boundaries = []
                offset = 0
                for map_name, md in group:
                    n = md.n_spectra
                    map_boundaries.append((map_name, offset, offset + n))
                    offset += n

                tasks.append({
                    "grouped_2d_maps": [item[0] for item in group],
                    "map_boundaries": map_boundaries,
                    "indices": np.arange(offset),
                    "fit_model": fit_model,
                })
        else:
            md = self.store.get_map_data(self.current_map_name)
            if not md:
                return
            fname_set = set(fnames)
            indices = np.array([i for i, f in enumerate(md.fnames) if f in fname_set])
            if len(indices) == 0:
                return
            tasks = [{
                "map_name": self.current_map_name,
                "indices": indices,
                "fit_model": fit_model,
            }]

        if not tasks:
            return

        self._launch_fit_thread(tasks)

    def get_fit_results_dataframe(self):
        """Get fit results DataFrame for the current map.

        Reuses self.df_fit_results (created by collect_fit_results) to avoid redundancy.
        """
        if self.df_fit_results is not None and not self.df_fit_results.empty:
            return self.df_fit_results
        return pd.DataFrame()

    def collect_fit_results(self):
        """Collect fit results, delegating to SpectraStore via parent class and post-processing wafer coords."""
        map_names = self.store.map_names
        if not map_names:
            self.notify.emit(tr("No maps loaded to collect results from."))
            return

        super().collect_fit_results(map_names=map_names)

        if self.df_fit_results is None or self.df_fit_results.empty:
            return

        # Wafer post-processing (only if not already computed in build_fit_results_df)
        if self.map_type != "2Dmap" and (
            "Quadrant" not in self.df_fit_results.columns
            or "Zone" not in self.df_fit_results.columns
        ):
            radius = 150
            if "300" in self.map_type:
                radius = 150
            elif "200" in self.map_type:
                radius = 100
            elif "100" in self.map_type:
                radius = 50

            x = self.df_fit_results["X"]
            y = self.df_fit_results["Y"]

            quadrant_arr = np.full(len(self.df_fit_results), np.nan, dtype=object)
            quadrant_arr[(x < 0) & (y < 0)] = "Q1"
            quadrant_arr[(x < 0) & (y > 0)] = "Q2"
            quadrant_arr[(x > 0) & (y > 0)] = "Q3"
            quadrant_arr[(x > 0) & (y < 0)] = "Q4"
            self.df_fit_results["Quadrant"] = quadrant_arr

            dist = np.sqrt(x ** 2 + y ** 2)
            zone_arr = np.full(len(self.df_fit_results), np.nan, dtype=object)
            zone_arr[dist <= radius * 0.35] = "Center"
            zone_arr[(dist > radius * 0.35) & (dist < radius * 0.8)] = "Mid-Radius"
            zone_arr[dist >= radius * 0.8] = "Edge"
            self.df_fit_results["Zone"] = zone_arr

            self.fit_results_updated.emit(self.df_fit_results)

        if self.current_map_name:
            self.clear_map_cache_requested.emit(self.current_map_name)
            self.map_data_updated.emit(self.current_map_df)
