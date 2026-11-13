"""Unit tests for VMapsList and its integration with VWorkspaceMaps."""
import pytest

from spectroview.view.components.v_map_list import VMapsList
from spectroview.view.v_workspace_maps import VWorkspaceMaps


@pytest.fixture
def maps_list_widget(qapp):
    widget = VMapsList()
    widget.show()
    yield widget
    widget.close()


class TestVMapsListUnit:
    """Unit tests for VMapsList widget without workspace dependencies."""

    def test_init_ui_structure(self, maps_list_widget):
        """VMapsList contains maps list and preserved spectra action buttons."""
        w = maps_list_widget
        assert hasattr(w, "maps_list")
        assert hasattr(w, "btn_view")
        assert hasattr(w, "btn_delete")
        assert hasattr(w, "btn_save")
        assert hasattr(w, "btn_select_all")
        assert hasattr(w, "btn_reinit")
        assert not hasattr(w, "btn_stats")
        assert not hasattr(w, "stats_requested")
        assert hasattr(w, "btn_send_to_spectra")
        # Ensure spectra_list is no longer present
        assert not hasattr(w, "spectra_list")
        assert not hasattr(w, "cb_check_all")

    def test_map_actions_signals(self, maps_list_widget, qtbot):
        """Map action buttons emit their respective signals."""
        w = maps_list_widget

        with qtbot.waitSignal(w.view_map_requested, timeout=1000):
            w.btn_view.click()

        with qtbot.waitSignal(w.delete_map_requested, timeout=1000):
            w.btn_delete.click()

        with qtbot.waitSignal(w.save_requested, timeout=1000):
            w.btn_save.click()

    def test_spectra_actions_signals(self, maps_list_widget, qtbot):
        """Spectra action buttons emit their respective signals."""
        w = maps_list_widget

        with qtbot.waitSignal(w.select_all_requested, timeout=1000):
            w.btn_select_all.click()

        with qtbot.waitSignal(w.reinitialize_requested, timeout=1000):
            w.btn_reinit.click()

        with qtbot.waitSignal(w.send_to_spectra_requested, timeout=1000):
            w.btn_send_to_spectra.click()

    def test_set_maps_names(self, maps_list_widget):
        """set_maps_names populates list and auto-selects first item."""
        w = maps_list_widget
        w.set_maps_names(["Map_A", "Map_B"])
        assert w.maps_list.count() == 2
        assert w.get_selected_map_index() == 0

    def test_set_selected_map_name(self, maps_list_widget):
        """set_selected_map_name changes selection without re-triggering unnecessary signals."""
        w = maps_list_widget
        w.set_maps_names(["Map_A", "Map_B"])
        w.set_selected_map_name("Map_B")
        assert w.get_selected_map_index() == 1

    def test_apply_theme_switches_icons(self, maps_list_widget):
        """apply_theme updates all 7 button icons without error."""
        w = maps_list_widget
        w.apply_theme("light")
        assert not w.btn_view.icon().isNull()
        assert not w.btn_select_all.icon().isNull()

        w.apply_theme("dark")
        assert not w.btn_view.icon().isNull()
        assert not w.btn_select_all.icon().isNull()


class TestVWorkspaceMapsHeatmapSelection:
    """Integration tests verifying spectrum actions operate on heatmap selection."""

    def test_map_open_auto_selects_first_spectrum_and_heatmap_point(self, qapp):
        workspace = VWorkspaceMaps()
        workspace.show()

        import pandas as pd

        map_df = pd.DataFrame({
            'X': [12.0, 14.0],
            'Y': [25.0, 30.0],
            100.0: [5.0, 10.0],
            200.0: [15.0, 20.0],
        })
        workspace.vm.maps["map_test"] = map_df
        workspace.vm._extract_spectra_from_map("map_test", map_df)

        # Trigger map selection as when opening a map
        workspace.vm.select_map("map_test")
        qapp.processEvents()

        # Verify ViewModel selected the first spectrum
        assert workspace.vm.selected_fnames == ["map_test_(12.0, 25.0)"]
        # Verify heatmap viewer auto-selected the first coordinate
        assert workspace.v_map_viewer.selected_points == [(12.0, 25.0)]

        workspace.close()

    def test_map_viewer_selection_updates_viewmodel(self, qapp, monkeypatch):
        workspace = VWorkspaceMaps()
        workspace.show()

        # Mock current map in viewmodel
        import numpy as np
        coords = np.array([[10.0, 20.0], [30.0, 40.0]])
        fnames = ["map1_(10.0, 20.0)", "map1_(30.0, 40.0)"]
        workspace.vm.store.add_map(
            name="map1",
            x0=np.array([100.0, 200.0]),
            Y0=np.ones((2, 2)),
            coords=coords,
            fnames=fnames,
            is_active=np.ones(2, dtype=bool),
        )
        workspace.vm.maps["map1"] = None
        workspace.vm.current_map_name = "map1"

        # Simulate selection of point (10.0, 20.0) in map viewer
        workspace.v_map_viewer.spectra_selected.emit([(10.0, 20.0)])
        qapp.processEvents()

        assert workspace.vm.selected_fnames == ["map1_(10.0, 20.0)"]

        # Simulate empty selection
        workspace.v_map_viewer.spectra_selected.emit([])
        qapp.processEvents()
        assert workspace.vm.selected_fnames == []

        workspace.close()

    def test_select_all_button_selects_all_coords(self, qapp):
        workspace = VWorkspaceMaps()
        workspace.show()

        import numpy as np
        coords = np.array([[1.0, 2.0], [3.0, 4.0]])
        fnames = ["map1_(1.0, 2.0)", "map1_(3.0, 4.0)"]
        workspace.vm.store.add_map(
            name="map1",
            x0=np.array([100.0, 200.0]),
            Y0=np.ones((2, 2)),
            coords=coords,
            fnames=fnames,
            is_active=np.ones(2, dtype=bool),
        )
        workspace.vm.maps["map1"] = None
        workspace.vm.current_map_name = "map1"

        # Click Select All button
        workspace.v_maps_list.btn_select_all.click()
        qapp.processEvents()

        assert set(workspace.vm.selected_fnames) == set(fnames)
        assert workspace.v_map_viewer.selected_points == [(1.0, 2.0), (3.0, 4.0)]

        workspace.close()

    def test_map_viewer_renders_time_resolved_duplicate_coords_without_crash(
        self, qapp, examples_dir
    ):
        import pytest

        tr_path = examples_dir / "spectroscopic_data" / "time_resolved_wafer_maps.csv"
        assert tr_path.exists()

        workspace = VWorkspaceMaps()
        try:
            workspace.vm.load_map_files([str(tr_path)])
            qapp.processEvents()

            md = workspace.vm.store.get_map_data("time_resolved_wafer_maps")
            assert md is not None and md.n_spectra == 50
            # X-range spinboxes initialize to actual spectral axis (~-79.28..3411 cm-1), not fallback (0..100)
            assert workspace.v_map_viewer.spin_xmin.value() == pytest.approx(float(md.x0[0]), abs=0.5)
            assert workspace.v_map_viewer.spin_xmax.value() == pytest.approx(float(md.x0[-1]), abs=0.5)

            # Selecting one (X, Y) wafer site selects all 10 time-resolved spectra at that site
            first_site = (float(md.coords[0, 0]), float(md.coords[0, 1]))
            workspace._on_map_viewer_selection([first_site])
            assert len(workspace.vm.selected_fnames) == 10
        finally:
            workspace.close()
            workspace.deleteLater()

    def test_multi_selection_placeholders_and_preservation(self, qapp):
        from PySide6.QtWidgets import QAbstractItemView
        from PySide6.QtCore import Qt
        from spectroview.view.components.v_map_list import VMapsList

        v = VMapsList()
        assert v.maps_list.selectionMode() == QAbstractItemView.ExtendedSelection

        # Default placeholders have NoItemFlags and emit empty selection cleanly
        events = []
        v.maps_selection_changed.connect(lambda names, prim: events.append((list(names), prim)))
        v.maps_list.setCurrentRow(0)
        qapp.processEvents()
        assert v.get_selected_map_names() == []
        assert v.get_primary_map_name() is None

        # Populate real maps and test multi-selection + Primary preservation
        v.set_maps_names(["map_A", "map_B", "map_C"])
        events.clear()
        v.set_selected_maps(["map_A", "map_C"], "map_A")
        assert v.get_selected_map_names() == ["map_A", "map_C"]
        assert v.get_primary_map_name() == "map_A"

        # Simulate Ctrl-Click deselecting current item (map_A) -> promotes remaining survivor (map_C)
        item_a = v._items_by_name["map_A"]
        item_a.setSelected(False)
        qapp.processEvents()
        assert v.get_selected_map_names() == ["map_C"]
        assert v.get_primary_map_name() == "map_C"
        assert events[-1] == (["map_C"], "map_C")

        # set_maps_names preserves multi-selection across additions
        v.set_selected_maps(["map_A", "map_C"], "map_C")
        v.set_maps_names(["map_A", "map_B", "map_C", "map_D"])
        assert v.get_selected_map_names() == ["map_A", "map_C"]
        assert v.get_primary_map_name() == "map_C"
        assert v.maps_list.item(0).flags() & Qt.ItemIsSelectable
