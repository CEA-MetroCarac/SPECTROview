"""End-to-end integration tests for the spectroview.api public package.

Exercises full load -> preprocess -> fit -> results -> plot -> save/load
workflows for both a discrete-spectra session and a hyperspectral-map
session, and verifies the on-disk structure is exactly what the GUI itself
reads (format v2, same metadata keys), independent of the API's own
loader -- i.e. this checks GUI/API file compatibility, not just
"the API can read back what it wrote".
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from spectroview.api import fitting, graphs, workspace
from spectroview.model.workspace_io import WorkspaceIO


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestDiscreteSpectraWorkflow:
    def test_full_workflow_round_trips(self, multiple_spectra_files, tmp_path):
        ws = workspace.SpectraWorkspace()
        ws.load_files(multiple_spectra_files)
        assert len(ws) == 3

        ws.crop(range_min=505.0, range_max=535.0)
        ws.set_baseline({"mode": "Polynomial", "order_max": 1})
        ws.subtract_baseline()

        for name in ws.names:
            x, y = ws.get_xy(name)
            peak_idx = int(y.argmax())
            fm = fitting.build_fit_model([
                {"model": "Lorentzian",
                 "x0": {"value": float(x[peak_idx]), "min": float(x.min()), "max": float(x.max())},
                 "ampli": {"value": float(y[peak_idx]), "min": 0.0, "max": float(y[peak_idx]) * 10},
                 "fwhm": {"value": 4.0, "min": 0.5, "max": 20.0}},
            ])
            ws.set_fit_model(fm, names=[name])

        ws.fit()
        df = ws.collect_results()
        assert df is not None
        assert all(md.has_fit_results() for md in (ws.store.get_map_data(n) for n in ws.names))

        ax = graphs.plot_scatter(df.reset_index(), x=df.reset_index().columns[0], y="P1_ampli")
        assert ax is not None

        out_path = tmp_path / "workflow.spectra"
        ws.save(out_path)
        state_before = {n: ws.store.get_map_data(n).peak_params.tolist() for n in ws.names}

        ws_reloaded = workspace.SpectraWorkspace.load(out_path)
        state_after = {n: ws_reloaded.store.get_map_data(n).peak_params.tolist() for n in ws_reloaded.names}
        assert state_before == state_after

        pd.testing.assert_frame_equal(
            df.sort_values("Filename").reset_index(drop=True),
            ws_reloaded.get_results_dataframe().sort_values("Filename").reset_index(drop=True),
        )


class TestHyperspectralMapWorkflow:
    def test_full_workflow_round_trips(self, map_2d_file, fit_model_si_file, tmp_path):
        if not fit_model_si_file.exists():
            pytest.skip("fit_model_Si_.json fixture not present")

        ws = workspace.MapsWorkspace()
        [map_name] = ws.load_files([map_2d_file])

        fm = fitting.load_fit_model_template(fit_model_si_file)
        ws.set_fit_model(fm, names=[map_name])
        ws.fit(map_names=[map_name])

        df = ws.collect_results()
        assert df is not None and not df.empty

        value_col = [c for c in df.columns if "ampli" in c][0]
        xi, yi, zi = ws.get_heatmap(map_name, value_col)
        assert zi.shape == (len(yi), len(xi))

        profile = ws.extract_profile(map_name, value_col, (xi.min(), yi.min()), (xi.max(), yi.max()))
        assert set(profile.columns) == {"X", "Y", "distance", "values"}

        out_path = tmp_path / "workflow.maps"
        ws.save(out_path)
        ws_reloaded = workspace.MapsWorkspace.load(out_path)
        assert ws_reloaded.map_type == ws.map_type
        pd.testing.assert_frame_equal(
            df.reset_index(drop=True), ws_reloaded.get_results_dataframe().reset_index(drop=True),
        )

    def test_saved_file_matches_gui_expected_structure(self, map_2d_file, tmp_path):
        """Structural check that .maps files this API writes carry the exact
        metadata keys VMWorkspaceMaps.load_work() reads (format_version,
        store_meta, maps_metadata, map_type) -- independent of this API's
        own loader, so a real drift in the GUI's expected shape would be
        caught even if SpectraWorkspace.load() happened to be lenient."""
        ws = workspace.MapsWorkspace(map_type="wafer_300mm")
        [map_name] = ws.load_files([map_2d_file])
        out_path = tmp_path / "structural.maps"
        ws.save(out_path)

        metadata, arrays, dataframes, is_legacy = WorkspaceIO.load_workspace(str(out_path))
        assert is_legacy is False
        assert metadata["format_version"] == 2
        assert map_name in metadata["store_meta"]
        assert metadata["map_type"] == "wafer_300mm"
        assert "maps_metadata" in metadata
        assert f"store_{map_name}_x0" in arrays
        assert f"store_{map_name}_y0" in arrays
