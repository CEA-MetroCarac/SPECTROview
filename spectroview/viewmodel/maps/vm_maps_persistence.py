"""Legacy JSON v1 .maps persistence mixin for VMWorkspaceMaps."""

from __future__ import annotations

import gzip
import io
import json
import re

import numpy as np
import pandas as pd

from spectroview.i18n import tr
from spectroview.model.spectra.spectra_store import SpectraStore


class MapsPersistenceMixin:
    """Legacy JSON-based ``.maps`` workspace loader."""

    def _load_legacy_maps(self, file_path: str):
        """Load legacy JSON-based .maps workspace (OLD version)."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.store = SpectraStore()
        self.maps = {}

        spectrums_data = data.get("spectrums_data", {})

        # Build coordinate lookup dictionary from individual spectrum data
        lookup = {}
        for _skey, sdata in spectrums_data.items():
            fname = sdata.get("fname", "")
            match = re.match(r"(.+?)_\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)", fname)
            if match:
                mname = match.group(1)
                xc = float(match.group(2))
                yc = float(match.group(3))
                lookup[(mname, round(xc, 3), round(yc, 3))] = sdata

        # Process each 2D map stored in the file
        for map_name, hex_string in data.get("maps", {}).items():
            hex_bytes = bytes.fromhex(hex_string)
            decompressed = gzip.decompress(hex_bytes)
            df = pd.read_csv(io.BytesIO(decompressed))
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

            coords = df[["X", "Y"]].values.astype(np.float64)
            x0 = np.array([float(col) for col in df.columns[2:]], dtype=np.float64)
            y0_mat = df.iloc[:, 2:].values.astype(np.float32)

            n_rows = len(df)
            fnames = []
            is_active_list = []
            colors = []
            labels = []
            map_spectra_sdata = []

            for i in range(n_rows):
                row = df.iloc[i]
                xc, yc = row["X"], row["Y"]
                sdata = lookup.get((map_name, round(xc, 3), round(yc, 3)))

                if sdata:
                    fnames.append(sdata.get("fname", f"{map_name}_({xc}, {yc})"))
                    is_active_list.append(sdata.get("is_active", True))
                    colors.append(sdata.get("color"))
                    labels.append(sdata.get("label"))
                    map_spectra_sdata.append(sdata)
                else:
                    fnames.append(f"{map_name}_({xc}, {yc})")
                    is_active_list.append(True)
                    colors.append(None)
                    labels.append(None)
                    map_spectra_sdata.append({})

            self.store.add_map(
                name=map_name,
                x0=x0,
                Y0=y0_mat,
                coords=coords,
                fnames=fnames,
                is_active=np.array(is_active_list, dtype=bool),
                colors=colors,
                labels=labels,
            )

            # Reconstruct parameter names and peak_params matrix
            param_names_set = set()
            for sdata in map_spectra_sdata:
                pmodels = sdata.get("peak_models", {}) or {}
                for k, pdict in pmodels.items():
                    shape = list(pdict.keys())[0]
                    for pname in pdict[shape].keys():
                        param_names_set.add(f"{pname}_{int(k) + 1}")

            def param_key(p):
                parts = p.split("_")
                idx = int(parts[-1])
                name = "_".join(parts[:-1])
                return (idx, name)

            param_names = sorted(list(param_names_set), key=param_key)

            peak_params_matrix = np.zeros((n_rows, len(param_names)), dtype=np.float64)
            success_list = []
            for i, sdata in enumerate(map_spectra_sdata):
                success_list.append(sdata.get("result_fit_success", False))
                pmodels = sdata.get("peak_models", {}) or {}
                for k, pdict in pmodels.items():
                    shape = list(pdict.keys())[0]
                    for pname, pvals in pdict[shape].items():
                        p_fullname = f"{pname}_{int(k) + 1}"
                        if p_fullname in param_names:
                            col_idx = param_names.index(p_fullname)
                            peak_params_matrix[i, col_idx] = pvals.get("value", 0.0)

            fit_model_dict = {}
            for sdata in map_spectra_sdata:
                pmodels = sdata.get("peak_models", {})
                plabels = sdata.get("peak_labels", [])
                if pmodels:
                    fit_model_dict = {"peak_labels": plabels, "peak_models": pmodels}
                    break

            if param_names:
                self.store.set_fit_results(
                    map_name=map_name,
                    indices=np.arange(n_rows),
                    peak_params=peak_params_matrix,
                    success=np.array(success_list, dtype=bool),
                    r2=np.zeros(n_rows, dtype=np.float64),
                    param_names=param_names,
                    fit_model=fit_model_dict,
                )

            md = self.store.get_map_data(map_name)
            if md:
                for sdata in map_spectra_sdata:
                    legacy_bl = sdata.get("baseline")
                    if legacy_bl:
                        md.baseline_config = {
                            "mode": legacy_bl.get("mode", "Linear"),
                            "points": legacy_bl.get("points", [[], []]),
                            "attached": legacy_bl.get("attached", False),
                            "coef": legacy_bl.get("coef", 5),
                        }
                        break

                if map_spectra_sdata:
                    first_s = map_spectra_sdata[0]
                    md.range_min = first_s.get("range_min")
                    md.range_max = first_s.get("range_max")
                    md.xcorrection_value = float(
                        first_s.get("xcorrection_value", 0.0)
                    )
                    md.intensity_norm_factor = float(
                        first_s.get("intensity_norm_factor", 1.0)
                    )

                    is_sub_list = []
                    for sdata in map_spectra_sdata:
                        if "is_baseline_subtracted" in sdata:
                            is_sub_list.append(sdata["is_baseline_subtracted"])
                        elif "baseline" in sdata and isinstance(
                            sdata["baseline"], dict
                        ):
                            is_sub_list.append(
                                sdata["baseline"].get("is_subtracted", False)
                            )
                        else:
                            is_sub_list.append(False)
                    md.is_baseline_subtracted = np.array(is_sub_list, dtype=bool)

                self._restore_preprocessed_state(md)

            if md is not None:
                self.maps[map_name] = self._build_map_df_from_md(md)

        legacy_results = data.get("df_fit_results")
        if legacy_results:
            self.df_fit_results = pd.DataFrame(legacy_results)
            self.fit_results_updated.emit(self.df_fit_results)
        else:
            self.df_fit_results = None

        self.maps_metadata = data.get("maps_metadata", {})
        self.map_type = "2Dmap"

        map_names = self.store.map_names
        if map_names:
            self.select_map(map_names[0])
        self.maps_list_changed.emit(map_names)
        self.count_changed.emit(sum(md.n_spectra for md in self.store._maps.values()))
        self.notify.emit(tr("Legacy maps workspace loaded successfully."))
