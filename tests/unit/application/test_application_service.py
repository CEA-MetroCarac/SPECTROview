"""Tests for the domain facade over the running desktop state."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from spectroview.application.dispatch import DirectDispatcher
from spectroview.application.errors import ApplicationAPIError
from spectroview.application.service import SpectroviewApplicationAPI
from spectroview.viewmodel.vm_workspace_graphs import VMWorkspaceGraphs
from spectroview.viewmodel.vm_workspace_maps import VMWorkspaceMaps
from spectroview.viewmodel.vm_workspace_spectra import VMWorkspaceSpectra


class _Tabs:
    def __init__(self, workspaces):
        self.workspaces = workspaces
        self.current = workspaces[0]

    def currentIndex(self):
        return self.workspaces.index(self.current)

    def tabText(self, index):
        return ("Spectra", "Maps", "Graphs")[index]

    def setCurrentWidget(self, widget):
        self.current = widget


class _Subwindow:
    def __init__(self, owner, graph_id):
        self.owner = owner
        self.graph_id = graph_id

    def close(self):
        self.owner.vm.delete_graph(self.graph_id)
        self.owner.graph_widgets.pop(self.graph_id, None)


class _GraphWorkspace:
    def __init__(self, settings):
        self.vm = VMWorkspaceGraphs(settings)
        self.graph_widgets = {}
        self.active_graph_id = None

    def _get_active_graph_id(self):
        return self.active_graph_id

    def create_plot_from_config(self, dataframe_name, config):
        graph = self.vm.create_graph(config)
        self.active_graph_id = graph.graph_id
        self.graph_widgets[graph.graph_id] = (
            object(), object(), _Subwindow(self, graph.graph_id)
        )
        return True

    def update_graphs_from_config(self, graph_id, properties):
        ids = self.vm.get_graph_ids() if str(graph_id).lower() == "all" else [int(graph_id)]
        updated = []
        for gid in ids:
            if self.vm.update_graph(gid, properties):
                updated.append(gid)
        return updated


@pytest.fixture
def live_api(qapp, settings):
    spectra_vm = VMWorkspaceSpectra(settings)
    maps_vm = VMWorkspaceMaps(settings)
    graphs_workspace = _GraphWorkspace(settings)

    x = np.linspace(100.0, 200.0, 11)
    spectra_vm.store.add_map(
        "sample", x, np.arange(11, dtype=float)[None, :],
        np.array([[0.0, 0.0]]), ["sample"],
    )
    spectra_vm.selected_fnames = ["sample"]
    maps_vm.store.add_map(
        "wafer", x, np.vstack([np.arange(11), np.arange(11) * 2.0]),
        np.array([[0.0, 0.0], [1.0, 0.0]]), ["pixel-0", "pixel-1"],
    )
    maps_vm.maps["wafer"] = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 0.0]})
    maps_vm.current_map_name = "wafer"
    maps_vm.selected_fnames = ["pixel-1"]
    graphs_workspace.vm.add_dataframe(
        "results", pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    )
    graphs_workspace.vm.select_dataframe("results")

    spectra_ws = SimpleNamespace(vm=spectra_vm)
    maps_ws = SimpleNamespace(vm=maps_vm)
    app = SimpleNamespace(
        v_spectra_workspace=spectra_ws,
        v_maps_workspace=maps_ws,
        v_graphs_workspace=graphs_workspace,
    )
    app.tabWidget = _Tabs([spectra_ws, maps_ws, graphs_workspace])
    return SpectroviewApplicationAPI(app, DirectDispatcher())


class TestStateAndData:
    def test_application_state_reports_ready_and_mcp_status(self, live_api):
        state = live_api.get_application_state()
        assert state["ready"] is True
        assert state["active_workspace"] == "spectra"
        assert state["mcp"] == {"running": False, "endpoint": None}

    def test_lists_all_workspace_dataset_kinds(self, live_api):
        datasets = live_api.list_datasets()
        assert {item["dataset_id"] for item in datasets} == {
            "spectra:sample", "map:wafer", "dataframe:results"
        }

    def test_current_spectrum_uses_selection_and_downsamples(self, live_api):
        result = live_api.get_current_spectrum(max_points=5)
        assert result["dataset_id"] == "spectra:sample"
        assert result["total_points"] == 11
        assert result["returned_points"] == 5
        assert result["downsampled"] is True

    def test_named_map_spectrum_is_retrievable(self, live_api):
        result = live_api.get_spectrum("map:wafer", "pixel-1", max_points=20)
        assert result["spectrum_index"] == 1
        assert result["y"][-1] == 20.0

    def test_invalid_dataset_id_is_actionable(self, live_api):
        with pytest.raises(ApplicationAPIError) as exc:
            live_api.get_dataset_info("sample")
        assert exc.value.code == "INVALID_DATASET_ID"

    def test_no_active_spectrum_is_reported(self, live_api):
        live_api._spectra_vm.selected_fnames = []
        with pytest.raises(ApplicationAPIError) as exc:
            live_api.get_current_spectrum()
        assert exc.value.code == "NO_ACTIVE_SPECTRUM"

    def test_application_not_ready_after_close(self, live_api):
        live_api.close()
        with pytest.raises(ApplicationAPIError) as exc:
            live_api.get_application_state()
        assert exc.value.code == "APPLICATION_NOT_READY"


class TestProcessingAndFittingFailures:
    def test_crop_reuses_workspace_processing(self, live_api):
        result = live_api.crop_spectrum("spectra", 120.0, 180.0)
        assert result["targets"] == ["sample"]
        info = live_api.get_dataset_info("spectra:sample")
        assert info["point_count"] == 7

    def test_normalization_rejects_zero(self, live_api):
        with pytest.raises(ApplicationAPIError) as exc:
            live_api.normalize_spectrum("spectra", 0.0)
        assert exc.value.code == "INVALID_NORMALIZATION"

    def test_baseline_requires_configuration(self, live_api):
        with pytest.raises(ApplicationAPIError) as exc:
            live_api.subtract_baseline("spectra")
        assert exc.value.code == "BASELINE_NOT_CONFIGURED"

    def test_fit_requires_model(self, live_api):
        with pytest.raises(ApplicationAPIError) as exc:
            live_api.fit_spectrum("spectra")
        assert exc.value.code == "FIT_NOT_CONFIGURED"

    def test_processing_is_rejected_while_fit_is_running(self, live_api):
        live_api._spectra_vm._is_fitting = True
        with pytest.raises(ApplicationAPIError) as exc:
            live_api.crop_spectrum("spectra", 120.0, 180.0)
        assert exc.value.code == "FIT_IN_PROGRESS"

    def test_fit_results_absence_is_reported(self, live_api):
        with pytest.raises(ApplicationAPIError) as exc:
            live_api.get_fit_results("spectra")
        assert exc.value.code == "FIT_RESULTS_NOT_AVAILABLE"


class TestGraphsAndExport:
    def test_graph_create_update_delete_round_trip(self, live_api):
        created = live_api.create_graph({
            "df_name": "results", "x": "x", "y": ["y"], "plot_style": "line"
        })
        graph_id = created["graph_id"]
        assert live_api.get_active_graph()["graph_id"] == graph_id
        assert live_api.update_graph(graph_id, {"plot_title": "Updated"})["graph_ids"] == [graph_id]
        assert live_api.list_graph_configurations()[0]["plot_title"] == "Updated"
        assert live_api.delete_graphs(graph_ids=[graph_id])["deleted_graph_ids"] == [graph_id]

    def test_no_active_graph_is_reported(self, live_api):
        with pytest.raises(ApplicationAPIError) as exc:
            live_api.get_active_graph()
        assert exc.value.code == "NO_ACTIVE_GRAPH"

    def test_export_requires_collected_results(self, live_api, tmp_path):
        with pytest.raises(ApplicationAPIError) as exc:
            live_api.export_results("spectra", str(tmp_path / "fit.csv"))
        assert exc.value.code == "FIT_RESULTS_NOT_AVAILABLE"

    def test_export_refuses_overwrite_without_explicit_permission(self, live_api, tmp_path):
        live_api._spectra_vm.df_fit_results = pd.DataFrame({"R2": [0.99]})
        path = tmp_path / "fit.csv"
        path.write_text("existing", encoding="utf-8")
        with pytest.raises(ApplicationAPIError) as exc:
            live_api.export_results("spectra", str(path))
        assert exc.value.code == "OVERWRITE_REQUIRED"

    def test_export_rejects_unsupported_extension(self, live_api, tmp_path):
        live_api._spectra_vm.df_fit_results = pd.DataFrame({"R2": [0.99]})
        with pytest.raises(ApplicationAPIError) as exc:
            live_api.export_results("spectra", str(tmp_path / "fit.json"))
        assert exc.value.code == "INVALID_EXPORT_PATH"
