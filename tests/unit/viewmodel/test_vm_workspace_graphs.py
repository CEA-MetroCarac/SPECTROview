"""Unit tests for viewmodel/vm_workspace_graphs.py - VMWorkspaceGraphs.

Driven against the real examples/datasets_for_plotting/dataset_Excel.xlsx
file wherever the test exercises DataFrame content (loading, filtering,
wafer slots) -- see tests/README.md for the data's real shape:
sheet1 (588 rows: X, Y, x0_Si, ampli_Si, area_Si, fwhm_Si, Quadrant
[Q1-Q4 + NaN], Zone [Center/Mid-Radius/Edge], Slot [2..12], DeltaW,
'Strain (GPa)', 'NB pts') and sheet2 (588 rows, no Slot column).
"""
import numpy as np
import pandas as pd
import pytest

from spectroview.model.m_graph import MGraph
from spectroview.viewmodel.vm_workspace_graphs import VMWorkspaceGraphs


@pytest.fixture
def vm(settings):
    return VMWorkspaceGraphs(settings)


@pytest.fixture(scope="module")
def excel_sheets(dataframe_excel_file):
    if not dataframe_excel_file.exists():
        pytest.skip("dataset_Excel.xlsx not present")
    xl = pd.ExcelFile(dataframe_excel_file)
    return {name: pd.read_excel(xl, sheet_name=name) for name in xl.sheet_names}


@pytest.fixture
def loaded_vm(vm, dataframe_excel_file):
    """VM with both real sheets already loaded, matching how the app loads
    a multi-sheet Excel file: one DataFrame per sheet, named
    '{stem}_{sheet_name}'."""
    vm.load_dataframes([str(dataframe_excel_file)])
    return vm


class TestLoadDataframes:
    def test_load_real_excel_creates_one_df_per_sheet(self, loaded_vm, excel_sheets):
        assert len(loaded_vm.dataframes) == len(excel_sheets)
        for sheet_name in excel_sheets:
            assert f"dataset_Excel_{sheet_name}" in loaded_vm.dataframes

    def test_loaded_dataframe_matches_real_content(self, loaded_vm, excel_sheets):
        df = loaded_vm.dataframes["dataset_Excel_sheet1"]
        assert len(df) == len(excel_sheets["sheet1"])
        assert set(df.columns) == set(excel_sheets["sheet1"].columns)

    def test_load_records_source_path_for_refresh(self, loaded_vm, dataframe_excel_file):
        assert loaded_vm.dataframe_sources["dataset_Excel_sheet1"] == str(dataframe_excel_file)

    def test_load_updates_last_directory_setting(self, loaded_vm, dataframe_excel_file):
        assert loaded_vm.settings.get_last_directory() == str(dataframe_excel_file.parent)

    def test_duplicate_load_is_skipped_and_notified(self, loaded_vm, dataframe_excel_file, qapp):
        notifications = []
        loaded_vm.notify.connect(notifications.append)
        loaded_vm.load_dataframes([str(dataframe_excel_file)])
        assert any("skipped" in n.lower() or "already loaded" in n.lower() for n in notifications)

    def test_load_nonexistent_file_notifies_error(self, vm, tmp_path, qapp):
        notifications = []
        vm.notify.connect(notifications.append)
        vm.load_dataframes([str(tmp_path / "does_not_exist.xlsx")])
        assert len(notifications) >= 1
        assert vm.dataframes == {}

    def test_load_opens_file_dialog_when_no_paths_given(self, vm, dataframe_excel_file, monkeypatch, qapp):
        from PySide6.QtWidgets import QFileDialog
        monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                             lambda *a, **k: ([str(dataframe_excel_file)], ""))
        vm.load_dataframes(None)
        assert "dataset_Excel_sheet1" in vm.dataframes

    def test_load_no_paths_and_dialog_cancelled_is_noop(self, vm, monkeypatch, qapp):
        from PySide6.QtWidgets import QFileDialog
        monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *a, **k: ([], ""))
        vm.load_dataframes(None)
        assert vm.dataframes == {}


class TestDataframeManagement:
    def test_add_dataframe(self, vm, qapp):
        df = pd.DataFrame({"A": [1, 2, 3]})
        vm.add_dataframe("mydf", df)
        assert "mydf" in vm.dataframes
        pd.testing.assert_frame_equal(vm.dataframes["mydf"], df)

    def test_add_dataframe_existing_without_force_replace_notifies_and_keeps_original(self, vm, qapp):
        original = pd.DataFrame({"A": [1]})
        replacement = pd.DataFrame({"A": [999]})
        vm.add_dataframe("mydf", original)
        notifications = []
        vm.notify.connect(notifications.append)
        vm.add_dataframe("mydf", replacement, force_replace=False)
        assert any("already exists" in n for n in notifications)
        pd.testing.assert_frame_equal(vm.dataframes["mydf"], original)

    def test_add_dataframe_force_replace_overwrites(self, vm, qapp):
        vm.add_dataframe("mydf", pd.DataFrame({"A": [1]}))
        vm.add_dataframe("mydf", pd.DataFrame({"A": [999]}), force_replace=True)
        assert vm.dataframes["mydf"]["A"].tolist() == [999]

    def test_remove_dataframe_clears_selection_if_selected(self, loaded_vm, qapp):
        loaded_vm.select_dataframe("dataset_Excel_sheet1")
        loaded_vm.remove_dataframe("dataset_Excel_sheet1")
        assert "dataset_Excel_sheet1" not in loaded_vm.dataframes
        assert "dataset_Excel_sheet1" not in loaded_vm.dataframe_sources
        assert loaded_vm.selected_df_name is None

    def test_remove_unknown_dataframe_is_noop(self, vm):
        vm.remove_dataframe("nope")  # must not raise

    def test_select_dataframe_emits_real_columns(self, loaded_vm, excel_sheets, qapp):
        received = []
        loaded_vm.dataframe_columns_changed.connect(received.append)
        loaded_vm.select_dataframe("dataset_Excel_sheet1")
        assert loaded_vm.selected_df_name == "dataset_Excel_sheet1"
        assert received[-1] == list(excel_sheets["sheet1"].columns)

    def test_select_unknown_dataframe_is_noop(self, vm):
        vm.select_dataframe("nope")
        assert vm.selected_df_name is None

    def test_get_dataframe(self, loaded_vm):
        df = loaded_vm.get_dataframe("dataset_Excel_sheet1")
        assert df is not None and len(df) == 588

    def test_get_unknown_dataframe_returns_none(self, vm):
        assert vm.get_dataframe("nope") is None

    def test_refresh_dataframe_reloads_from_real_source(self, loaded_vm):
        assert loaded_vm.refresh_dataframe("dataset_Excel_sheet1") is True
        assert len(loaded_vm.dataframes["dataset_Excel_sheet1"]) == 588

    def test_refresh_unknown_dataframe_returns_false(self, vm):
        assert vm.refresh_dataframe("nope") is False

    def test_refresh_dataframe_missing_source_returns_false(self, vm):
        vm.add_dataframe("nosrc", pd.DataFrame({"A": [1]}))  # no source path recorded
        assert vm.refresh_dataframe("nosrc") is False

    def test_refresh_dataframe_deleted_source_file_returns_false(self, vm, tmp_path):
        src = tmp_path / "temp.csv"
        pd.DataFrame({"A": [1, 2]}).to_csv(src, index=False, sep=";")
        vm.dataframes["temp"] = pd.read_csv(src, sep=";")
        vm.dataframe_sources["temp"] = str(src)
        src.unlink()
        assert vm.refresh_dataframe("temp") is False

    def test_save_dataframe_excel(self, loaded_vm, tmp_path, monkeypatch, qapp):
        from PySide6.QtWidgets import QFileDialog
        out_path = tmp_path / "exported.xlsx"
        monkeypatch.setattr(QFileDialog, "getSaveFileName",
                             lambda *a, **k: (str(out_path), "Excel Files (*.xlsx)"))
        loaded_vm.save_dataframe("dataset_Excel_sheet1")
        assert out_path.exists()
        reloaded = pd.read_excel(out_path)
        assert len(reloaded) == 588

    def test_save_dataframe_csv(self, loaded_vm, tmp_path, monkeypatch, qapp):
        from PySide6.QtWidgets import QFileDialog
        out_path = tmp_path / "exported.csv"
        monkeypatch.setattr(QFileDialog, "getSaveFileName",
                             lambda *a, **k: (str(out_path), "CSV Files (*.csv)"))
        loaded_vm.save_dataframe("dataset_Excel_sheet1")
        assert out_path.exists()
        reloaded = pd.read_csv(out_path, sep=';')
        assert len(reloaded) == 588

    def test_save_unknown_dataframe_is_noop(self, vm, monkeypatch, qapp):
        from PySide6.QtWidgets import QFileDialog
        called = []
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: called.append(1))
        vm.save_dataframe("nope")
        assert called == []


class TestApplyFilters:
    def test_no_filters_returns_full_copy(self, loaded_vm):
        result = loaded_vm.apply_filters("dataset_Excel_sheet1", [])
        assert len(result) == 588
        assert result is not loaded_vm.dataframes["dataset_Excel_sheet1"]

    def test_single_checked_filter_narrows_by_slot(self, loaded_vm):
        result = loaded_vm.apply_filters("dataset_Excel_sheet1",
                                           [{"expression": "Slot == 3", "state": True}])
        assert len(result) == 49
        assert (result["Slot"] == 3).all()

    def test_unchecked_filter_is_ignored(self, loaded_vm):
        result = loaded_vm.apply_filters("dataset_Excel_sheet1",
                                           [{"expression": "Slot == 3", "state": False}])
        assert len(result) == 588

    def test_multiple_checked_filters_combine_with_and(self, loaded_vm):
        result = loaded_vm.apply_filters("dataset_Excel_sheet1", [
            {"expression": "Slot == 2", "state": True},
            {"expression": "Zone == 'Center'", "state": True},
        ])
        assert len(result) > 0
        assert (result["Slot"] == 2).all()
        assert (result["Zone"] == "Center").all()

    def test_zone_filter_matches_real_category(self, loaded_vm, excel_sheets):
        result = loaded_vm.apply_filters("dataset_Excel_sheet1",
                                           [{"expression": "Zone == 'Edge'", "state": True}])
        expected = (excel_sheets["sheet1"]["Zone"] == "Edge").sum()
        assert len(result) == expected

    def test_invalid_expression_shows_error_and_returns_none(self, loaded_vm, monkeypatch, qapp):
        from PySide6.QtWidgets import QMessageBox
        errors = []
        monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a))
        result = loaded_vm.apply_filters("dataset_Excel_sheet1",
                                           [{"expression": "this is not valid python (((", "state": True}])
        assert result is None
        assert len(errors) == 1

    def test_apply_filters_unknown_dataframe_returns_none(self, vm):
        assert vm.apply_filters("nope", []) is None


class TestSlotHelpers:
    def test_has_slot_column_true_for_sheet1(self, loaded_vm):
        assert loaded_vm.has_slot_column("dataset_Excel_sheet1") is True

    def test_has_slot_column_false_for_sheet2(self, loaded_vm):
        assert loaded_vm.has_slot_column("dataset_Excel_sheet2") is False

    def test_has_slot_column_unknown_df_is_false(self, vm):
        assert vm.has_slot_column("nope") is False

    def test_get_unique_slots_matches_real_data(self, loaded_vm):
        assert loaded_vm.get_unique_slots("dataset_Excel_sheet1") == list(range(2, 13))

    def test_get_unique_slots_empty_when_no_slot_column(self, loaded_vm):
        assert loaded_vm.get_unique_slots("dataset_Excel_sheet2") == []


class TestMultiWaferGraphs:
    def test_creates_one_graph_per_slot(self, loaded_vm):
        graphs = loaded_vm.create_multi_wafer_graphs(
            "dataset_Excel_sheet1", [2, 3, 4],
            {"plot_style": "wafer", "x": "X", "y": ["Y"], "z": "ampli_Si", "df_name": "dataset_Excel_sheet1"},
            base_filters=[],
        )
        assert len(graphs) == 3
        assert all(isinstance(g, MGraph) for g in graphs)
        assert all(g.plot_style == "wafer" for g in graphs)

    def test_each_graph_gets_its_own_slot_filter(self, loaded_vm):
        graphs = loaded_vm.create_multi_wafer_graphs(
            "dataset_Excel_sheet1", [5, 6],
            {"plot_style": "wafer", "df_name": "dataset_Excel_sheet1"}, base_filters=[],
        )
        slot_exprs = [g.filters[-1]["expression"] for g in graphs]
        assert slot_exprs == ["Slot == 5", "Slot == 6"]
        assert all(g.filters[-1]["state"] is True for g in graphs)

    def test_base_filters_are_preserved_alongside_slot_filter(self, loaded_vm):
        base = [{"expression": "Zone == 'Center'", "state": True}]
        graphs = loaded_vm.create_multi_wafer_graphs(
            "dataset_Excel_sheet1", [7], {"df_name": "dataset_Excel_sheet1"}, base_filters=base,
        )
        exprs = [f["expression"] for f in graphs[0].filters]
        assert "Zone == 'Center'" in exprs
        assert "Slot == 7" in exprs

    def test_existing_slot_filter_is_replaced_not_duplicated(self, loaded_vm):
        base = [{"expression": "Slot == 999", "state": False}]
        graphs = loaded_vm.create_multi_wafer_graphs(
            "dataset_Excel_sheet1", [8], {"df_name": "dataset_Excel_sheet1"}, base_filters=base,
        )
        slot_filters = [f for f in graphs[0].filters if "Slot ==" in f["expression"]]
        assert len(slot_filters) == 1
        assert slot_filters[0]["expression"] == "Slot == 8"
        assert slot_filters[0]["state"] is True

    def test_multi_wafer_graphs_are_registered_and_resolvable_by_id(self, loaded_vm):
        graphs = loaded_vm.create_multi_wafer_graphs(
            "dataset_Excel_sheet1", [9], {"df_name": "dataset_Excel_sheet1"}, base_filters=[],
        )
        assert loaded_vm.get_graph(graphs[0].graph_id) is graphs[0]

    def test_resulting_filters_actually_isolate_correct_rows(self, loaded_vm):
        """End-to-end: the filter list a multi-wafer graph ends up with must
        actually select only that slot's rows when run through apply_filters."""
        graphs = loaded_vm.create_multi_wafer_graphs(
            "dataset_Excel_sheet1", [4], {"df_name": "dataset_Excel_sheet1"}, base_filters=[],
        )
        filtered = loaded_vm.apply_filters("dataset_Excel_sheet1", graphs[0].filters)
        assert len(filtered) == 49
        assert (filtered["Slot"] == 4).all()


class TestGraphManagement:
    def test_create_graph_default(self, vm):
        graph = vm.create_graph()
        assert graph.graph_id == 1
        assert graph.plot_style == "point"

    def test_create_graph_with_config_applies_known_fields(self, vm):
        graph = vm.create_graph({"plot_style": "scatter", "x": "X", "y": ["Y"], "df_name": "d"})
        assert graph.plot_style == "scatter"
        assert graph.x == "X"
        assert graph.y == ["Y"]

    def test_create_graph_ignores_unknown_config_keys(self, vm):
        graph = vm.create_graph({"totally_bogus_field": 123})
        assert not hasattr(graph, "totally_bogus_field")

    def test_graph_ids_increase_monotonically(self, vm):
        g1 = vm.create_graph()
        g2 = vm.create_graph()
        g3 = vm.create_graph()
        assert [g1.graph_id, g2.graph_id, g3.graph_id] == [1, 2, 3]
        assert vm.get_graph_ids() == [1, 2, 3]

    def test_get_graph_unknown_returns_none(self, vm):
        assert vm.get_graph(999) is None

    def test_update_graph_sets_properties(self, vm):
        graph = vm.create_graph()
        vm.update_graph(graph.graph_id, {"grid": True, "color_palette": "viridis"})
        assert graph.grid is True
        assert graph.color_palette == "viridis"

    def test_update_graph_can_set_the_two_previously_missing_fields(self, vm):
        """scatter_edgecolor/axis_breaks -- confirms the MGraph fix (see
        test_m_graph.py) is reachable through the ViewModel's own API, the
        same path CustomizeGraphDialog's properties_changed handler uses."""
        graph = vm.create_graph()
        vm.update_graph(graph.graph_id, {
            "scatter_edgecolor": "#00FF00",
            "axis_breaks": {"x": {"start": 1.0, "end": 2.0}, "y": None},
        })
        assert graph.scatter_edgecolor == "#00FF00"
        assert graph.axis_breaks == {"x": {"start": 1.0, "end": 2.0}, "y": None}

    def test_update_unknown_graph_is_noop(self, vm):
        vm.update_graph(999, {"grid": True})  # must not raise

    def test_delete_graph(self, vm):
        graph = vm.create_graph()
        vm.delete_graph(graph.graph_id)
        assert vm.get_graph(graph.graph_id) is None
        assert graph.graph_id not in vm.get_graph_ids()

    def test_delete_unknown_graph_is_noop(self, vm):
        vm.delete_graph(999)  # must not raise


class TestSaveLoadWorkspace:
    def test_save_and_load_round_trip_with_real_data_and_customizations(
        self, loaded_vm, tmp_path, monkeypatch, qapp,
    ):
        from PySide6.QtWidgets import QFileDialog

        loaded_vm.select_dataframe("dataset_Excel_sheet1")
        graph = loaded_vm.create_graph({
            "df_name": "dataset_Excel_sheet1", "plot_style": "scatter",
            "x": "x0_Si", "y": ["ampli_Si"], "z": "Quadrant",
            "filters": [{"expression": "Slot == 3", "state": True}],
        })
        loaded_vm.update_graph(graph.graph_id, {
            "scatter_edgecolor": "#123ABC",
            "axis_breaks": {"x": {"start": 515.0, "end": 516.0}, "y": None},
            "grid": True,
        })

        save_path = tmp_path / "roundtrip.graphs"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(save_path), ""))
        loaded_vm.save_workspace()
        assert save_path.exists()

        vm2 = VMWorkspaceGraphs(loaded_vm.settings)
        vm2.load_workspace(str(save_path))

        assert set(vm2.dataframes.keys()) == set(loaded_vm.dataframes.keys())
        pd.testing.assert_frame_equal(
            vm2.dataframes["dataset_Excel_sheet1"].reset_index(drop=True),
            loaded_vm.dataframes["dataset_Excel_sheet1"].reset_index(drop=True),
        )

        reloaded_graph = vm2.get_graph(graph.graph_id)
        assert reloaded_graph is not None
        assert reloaded_graph.plot_style == "scatter"
        assert reloaded_graph.x == "x0_Si"
        assert reloaded_graph.z == "Quadrant"
        assert reloaded_graph.scatter_edgecolor == "#123ABC"
        assert reloaded_graph.axis_breaks == {"x": {"start": 515.0, "end": 516.0}, "y": None}
        assert reloaded_graph.grid is True
        assert reloaded_graph.filters == [{"expression": "Slot == 3", "state": True}]

    def test_load_workspace_legacy_json_format(self, vm, tmp_path):
        import gzip
        import json

        csv_bytes = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}).to_csv(index=False).encode("utf-8")
        legacy_data = {
            "original_dfs": {"legacy_df": gzip.compress(csv_bytes).hex()},
            "dataframe_sources": {},
            "plots": {"1": MGraph(graph_id=1).save()},
        }
        legacy_path = tmp_path / "legacy.graphs"
        legacy_path.write_text(json.dumps(legacy_data))

        vm.load_workspace(str(legacy_path))
        assert "legacy_df" in vm.dataframes
        assert len(vm.dataframes["legacy_df"]) == 3
        assert 1 in vm.get_graph_ids()

    def test_load_nonexistent_workspace_shows_error(self, vm, tmp_path, monkeypatch, qapp):
        from PySide6.QtWidgets import QMessageBox
        errors = []
        monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a))
        vm.load_workspace(str(tmp_path / "does_not_exist.graphs"))
        assert len(errors) == 1

    def test_next_graph_id_continues_after_reload(self, loaded_vm, tmp_path, monkeypatch, qapp):
        from PySide6.QtWidgets import QFileDialog
        loaded_vm.create_graph()
        loaded_vm.create_graph()
        save_path = tmp_path / "ids.graphs"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(save_path), ""))
        loaded_vm.save_workspace()

        vm2 = VMWorkspaceGraphs(loaded_vm.settings)
        vm2.load_workspace(str(save_path))
        new_graph = vm2.create_graph()
        assert new_graph.graph_id == 3


class TestClearWorkspace:
    def test_clear_resets_everything(self, loaded_vm):
        loaded_vm.create_graph()
        loaded_vm.select_dataframe("dataset_Excel_sheet1")
        loaded_vm.clear_workspace()
        assert loaded_vm.dataframes == {}
        assert loaded_vm.dataframe_sources == {}
        assert loaded_vm.graphs == {}
        assert loaded_vm.selected_df_name is None
        assert loaded_vm._next_graph_id == 1


class TestUndoRedo:
    """Workspace-level undo/redo -- one stack of whole-workspace graph
    snapshots (not per-graph), since create/delete affect the graph *set*,
    not just one graph's fields. See begin_undo_batch()/undo()/redo()
    docstrings for the design.

    _restore_snapshot() always builds fresh MGraph instances via load()
    rather than reusing existing objects, so assertions after undo()/redo()
    must re-fetch via vm.graphs[graph_id] / vm.get_graph(...) -- a
    previously-held graph reference goes stale."""

    def test_no_history_initially(self, vm):
        assert vm.can_undo is False
        assert vm.can_redo is False
        assert vm.undo() is False
        assert vm.redo() is False

    def test_create_graph_is_undoable(self, vm):
        vm.create_graph()
        assert vm.can_undo is True

        assert vm.undo() is True
        assert vm.get_graph_ids() == []
        assert vm.can_undo is False
        assert vm.can_redo is True

    def test_redo_reapplies_create(self, vm):
        graph = vm.create_graph()
        vm.undo()

        assert vm.redo() is True
        assert vm.get_graph_ids() == [graph.graph_id]
        assert vm.can_redo is False

    def test_update_graph_is_undoable_without_undoing_the_create(self, vm):
        graph = vm.create_graph({"plot_style": "scatter", "x": "A", "y": ["B"]})
        vm.update_graph(graph.graph_id, {"x": "changed"})
        assert graph.x == "changed"

        vm.undo()

        assert graph.graph_id in vm.get_graph_ids()  # graph itself survives
        assert vm.graphs[graph.graph_id].x == "A"  # only the update reverted

    def test_delete_graph_is_undoable(self, vm):
        graph = vm.create_graph()
        vm.delete_graph(graph.graph_id)
        assert vm.get_graph_ids() == []

        vm.undo()

        assert vm.get_graph_ids() == [graph.graph_id]

    def test_new_action_after_undo_clears_redo_history(self, vm):
        vm.create_graph()
        vm.undo()
        assert vm.can_redo is True

        vm.create_graph()

        assert vm.can_redo is False
        assert vm.redo() is False

    def test_batch_collapses_several_mutations_into_one_undo_step(self, vm):
        vm.begin_undo_batch()
        try:
            g1 = vm.create_graph()
            vm.create_graph()
            vm.update_graph(g1.graph_id, {"grid": True})
        finally:
            vm.end_undo_batch()

        assert len(vm._undo_stack) == 1

        vm.undo()

        assert vm.get_graph_ids() == []  # both creates + the update undone together

    def test_nested_batches_compose_into_one_outer_step(self, vm):
        vm.begin_undo_batch()
        try:
            for _ in range(3):
                vm.begin_undo_batch()
                try:
                    vm.create_graph()
                finally:
                    vm.end_undo_batch()
        finally:
            vm.end_undo_batch()

        assert len(vm._undo_stack) == 1

        vm.undo()

        assert vm.get_graph_ids() == []

    def test_max_undo_depth_caps_stack_and_drops_oldest(self, vm):
        vm._max_undo_depth = 3
        for _ in range(5):
            vm.create_graph()
        assert len(vm._undo_stack) == 3

    def test_load_workspace_resets_undo_history(self, loaded_vm, tmp_path, monkeypatch, qapp):
        from PySide6.QtWidgets import QFileDialog

        loaded_vm.select_dataframe("dataset_Excel_sheet1")
        loaded_vm.create_graph({"df_name": "dataset_Excel_sheet1"})
        assert loaded_vm.can_undo is True

        save_path = tmp_path / "undo_reset.graphs"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(save_path), ""))
        loaded_vm.save_workspace()

        loaded_vm.load_workspace(str(save_path))

        assert loaded_vm.can_undo is False
        assert loaded_vm.can_redo is False

    def test_clear_workspace_resets_undo_history(self, vm):
        vm.create_graph()
        assert vm.can_undo is True

        vm.clear_workspace()

        assert vm.can_undo is False
        assert vm.can_redo is False

    def test_undo_state_changed_signal_emitted_on_push_undo_and_redo(self, vm):
        received = []
        vm.undo_state_changed.connect(lambda: received.append(None))

        vm.create_graph()
        assert len(received) == 1

        vm.undo()
        assert len(received) == 2

        vm.redo()
        assert len(received) == 3
