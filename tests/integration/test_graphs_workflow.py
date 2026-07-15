"""End-to-end integration tests for the Graphs workspace, driven entirely on
the real examples/datasets_for_plotting/dataset_Excel.xlsx dataset.

Chains together the full realistic pipeline: load Excel -> select a sheet ->
apply a filter -> create + render a graph for several plot styles -> drive
the real CustomizeGraphDialog sub-widgets to customize it (including the two
previously-broken scatter_edgecolor/axis_breaks properties) -> multi-wafer
batch creation -> save workspace -> reload into a fresh ViewModel -> verify
every property survived and the reloaded graph still renders.

This intentionally reproduces the same object wiring
v_workspace_graphs.py's `_create_and_display_plot`/`_on_graph_properties_changed`
use (VGraph -> properties_changed signal -> VMWorkspaceGraphs.update_graph),
without needing the full QMainWindow-based workspace widget.
"""
import pandas as pd
import pytest

from spectroview.model.m_graph import MGraph
from spectroview.viewmodel.vm_workspace_graphs import VMWorkspaceGraphs
from spectroview.view.components.v_graph import VGraph
from spectroview.view.components.customize_graph_dialog import CustomizeLegend, CustomizeAxis


@pytest.fixture
def vm(settings):
    return VMWorkspaceGraphs(settings)


def _render_graph_from_model(vm, graph: MGraph, df: pd.DataFrame) -> VGraph:
    """Minimal, faithful reproduction of v_workspace_graphs.py's
    VGraph(...) -> configure-from-model -> create_plot_widget -> plot(df)
    pipeline, without needing the workspace QMainWindow."""
    vg = VGraph(graph_id=graph.graph_id)
    vg.properties_changed.connect(
        lambda gid, props: vm.update_graph(gid, props)
    )
    for field in (
        "df_name", "x", "y", "z", "y2", "y3", "x2",
        "xmin", "xmax", "ymin", "ymax", "zmin", "zmax",
        "xlogscale", "ylogscale", "plot_title", "xlabel", "ylabel",
        "grid", "legend_visible", "legend_outside", "legend_properties",
        "color_palette", "wafer_size", "wafer_stats",
        "trendline_order", "trendline_anchor_enabled", "trendline_anchor_origin",
        "trendline_anchor_x", "trendline_anchor_y",
        "show_bar_plot_error_bar", "join_for_point_plot", "dodge_point_plot",
        "dodge_scatter_plot", "scatter_size", "scatter_edgecolor",
        "x_as_numeric", "y_as_numeric", "hist_bins", "hist_kde", "hist_step",
        "sort_data_enabled", "sort_data_by", "annotations", "axis_breaks",
        "minor_ticks_bottom", "minor_ticks_left", "minor_ticks_top", "minor_ticks_right",
        "plot_style",
    ):
        setattr(vg, field, getattr(graph, field))
    vg.create_plot_widget(dpi=graph.dpi)
    vg.plot(df)
    return vg


class TestFullWaferAnalysisWorkflow:
    def test_load_filter_plot_customize_save_reload(self, vm, dataframe_excel_file, tmp_path, monkeypatch, qapp):
        from PySide6.QtWidgets import QFileDialog

        # 1. Load the real multi-sheet Excel file.
        vm.load_dataframes([str(dataframe_excel_file)])
        assert "dataset_Excel_sheet1" in vm.dataframes
        vm.select_dataframe("dataset_Excel_sheet1")

        # 2. Filter down to one wafer slot.
        filters = [{"expression": "Slot == 4", "state": True}]
        filtered_df = vm.apply_filters("dataset_Excel_sheet1", filters)
        assert len(filtered_df) == 49

        # 3. Create and render a wafer plot of that slot's Si-peak amplitude.
        graph = vm.create_graph({
            "df_name": "dataset_Excel_sheet1", "plot_style": "wafer",
            "x": "X", "y": ["Y"], "z": "ampli_Si", "filters": filters,
            "wafer_size": 300.0,
        })
        vg = _render_graph_from_model(vm, graph, filtered_df)
        assert len(vg.ax.collections) > 0 or len(vg.ax.images) > 0

        # 4. Customize via the real dialog sub-widgets: scatter-plot follow-up
        #    (switch to scatter to exercise the marker customization tab) and
        #    an axis break -- the two properties whose MGraph persistence was
        #    broken before this test suite (see tests/README.md).
        scatter_graph = vm.create_graph({
            "df_name": "dataset_Excel_sheet1", "plot_style": "scatter",
            "x": "x0_Si", "y": ["ampli_Si"], "z": "Quadrant", "filters": filters,
        })
        vg2 = _render_graph_from_model(vm, scatter_graph, filtered_df)

        legend_widget = CustomizeLegend(vg2)
        legend_widget.spin_scatter_size.setValue(150)
        legend_widget._set_color_button(legend_widget.btn_scatter_edgecolor, "#ff00ff")
        legend_widget.apply_changes()

        axis_widget = CustomizeAxis(vg2)
        axis_widget.x_break_enabled.setChecked(True)
        axis_widget.x_break_start.setValue(514.0)
        axis_widget.x_break_end.setValue(515.0)
        axis_widget._apply_axis_settings()

        # Confirm the customization reached the MGraph via properties_changed.
        assert scatter_graph.scatter_edgecolor == "#ff00ff"
        assert scatter_graph.scatter_size == 150
        assert scatter_graph.axis_breaks["x"] == {"start": 514.0, "end": 515.0}

        # 5. Multi-wafer batch: one graph per slot, all built off real data.
        multi_graphs = vm.create_multi_wafer_graphs(
            "dataset_Excel_sheet1", [2, 3],
            {"df_name": "dataset_Excel_sheet1", "plot_style": "wafer",
             "x": "X", "y": ["Y"], "z": "fwhm_Si"},
            base_filters=[],
        )
        for g in multi_graphs:
            df_for_slot = vm.apply_filters("dataset_Excel_sheet1", g.filters)
            _render_graph_from_model(vm, g, df_for_slot)

        # 6. Collect fit-results-style dataframe stats before persisting, to
        #    cross-check against the reloaded workspace later.
        n_graphs_before = len(vm.get_graph_ids())
        n_dataframes_before = len(vm.dataframes)

        # 7. Save the whole workspace and reload it into a fresh ViewModel.
        save_path = tmp_path / "wafer_analysis.graphs"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(save_path), ""))
        vm.save_workspace()
        assert save_path.exists()

        vm2 = VMWorkspaceGraphs(vm.settings)
        vm2.load_workspace(str(save_path))

        # 8. Verify everything survived the round trip.
        assert len(vm2.dataframes) == n_dataframes_before
        assert len(vm2.get_graph_ids()) == n_graphs_before
        pd.testing.assert_frame_equal(
            vm2.dataframes["dataset_Excel_sheet1"].reset_index(drop=True),
            vm.dataframes["dataset_Excel_sheet1"].reset_index(drop=True),
        )

        reloaded_scatter = vm2.get_graph(scatter_graph.graph_id)
        assert reloaded_scatter.scatter_edgecolor == "#ff00ff"
        assert reloaded_scatter.scatter_size == 150
        assert reloaded_scatter.axis_breaks["x"] == {"start": 514.0, "end": 515.0}
        assert reloaded_scatter.filters == filters

        reloaded_wafer = vm2.get_graph(graph.graph_id)
        assert reloaded_wafer.plot_style == "wafer"
        assert reloaded_wafer.z == "ampli_Si"

        # 9. The reloaded graph's config must still actually render.
        refiltered = vm2.apply_filters("dataset_Excel_sheet1", reloaded_scatter.filters)
        vg3 = _render_graph_from_model(vm2, reloaded_scatter, refiltered)
        assert vg3.scatter_edgecolor == "#ff00ff"
        assert len(vg3.ax.collections) > 0


class TestCrossSheetWorkflow:
    """sheet2 has no Slot column -- exercises the has_slot_column()/
    get_unique_slots() guards on a real second sheet, and a plot style
    (line/trendline) driven from that sheet's own numeric columns."""

    def test_sheet_without_slot_column_supports_non_wafer_plots(self, vm, dataframe_excel_file, qapp):
        vm.load_dataframes([str(dataframe_excel_file)])
        assert vm.has_slot_column("dataset_Excel_sheet2") is False
        assert vm.get_unique_slots("dataset_Excel_sheet2") == []

        df = vm.dataframes["dataset_Excel_sheet2"]
        graph = vm.create_graph({
            "df_name": "dataset_Excel_sheet2", "plot_style": "trendline",
            "x": "x0_Si", "y": ["area_Si"],
        })
        vg = _render_graph_from_model(vm, graph, df)
        assert len(vg.ax.get_lines()) > 0
        assert len(vg.trendline_equations) >= 1
