"""Integration test for Interactive Spectral Mapping in SPECTROview."""

import numpy as np
import pytest

from spectroview.model.spectra_store import MapData
from spectroview.model.spectral_parameters import (
    SELECTOR_RED,
    PARAM_PEAK_POSITION,
)
from spectroview.view.v_workspace_spectra import VWorkspaceSpectra
from spectroview.view.v_workspace_maps import VWorkspaceMaps


@pytest.fixture
def test_map_data():
    """Create synthetic hyperspectral dataset (4x5 = 20 points)."""
    x = np.linspace(500, 1500, 101)
    N = 20
    Y = np.zeros((N, len(x)), dtype=np.float32)
    coords = np.zeros((N, 2), dtype=np.float64)

    for i in range(N):
        gx = float(i % 5)
        gy = float(i // 5)
        coords[i] = [gx, gy]
        # Peak at 900
        p1 = (10.0 + gx * 2.0) * np.exp(-((x - 900) ** 2) / (2 * 25 ** 2))
        Y[i] = p1 + 5.0

    return MapData(
        name="test_hyperspectral_map",
        x0=x,
        Y0=Y,
        coords=coords,
        is_active=np.ones(N, dtype=bool),
        fnames=[f"test_hyperspectral_map_({coords[i, 0]}, {coords[i, 1]})" for i in range(N)],
        colors=[None] * N,
        labels=[None] * N,
    )


def test_workspace_modes_and_live_maps_tab_position(qapp):
    """Verify Live Maps tab and range selectors are present in both Spectra and Maps workspaces with appropriate modes."""
    # 1. Spectra Workspace: Live Maps in 'spectra' (1D variation graph) mode at index 2, range selectors visible
    ws_spectra = VWorkspaceSpectra()
    spectra_tab_texts = [ws_spectra.bottom_tabs.tabText(i) for i in range(ws_spectra.bottom_tabs.count())]
    assert "Live Maps" in spectra_tab_texts, "Live Maps must be present in Spectra workspace"
    assert spectra_tab_texts.index("Live Maps") == 2, "Live Maps must be at index 2 (after Fit Results)"
    assert spectra_tab_texts[-1] == "More", "More tab must be at the end"
    assert ws_spectra.v_spectra_viewer.range_panel.isHidden() is False
    assert ws_spectra.v_live_maps.mode == "spectra"
    assert ws_spectra.vm_live_maps.mode == "spectra"

    # 2. Maps Workspace: Live Maps in 'map' (2D spatial heatmap) mode at index 2, range selectors visible
    ws_maps = VWorkspaceMaps()
    maps_tab_texts = [ws_maps.bottom_tabs.tabText(i) for i in range(ws_maps.bottom_tabs.count())]
    assert "Live Maps" in maps_tab_texts, "Live Maps must be present in Maps workspace"
    assert maps_tab_texts.index("Live Maps") == 2, "Live Maps must be at index 2 (after Fit Results)"
    assert maps_tab_texts[-1] == "More", "More tab must be at the end"
    assert ws_maps.v_spectra_viewer.range_panel.isHidden() is False
    assert ws_maps.v_live_maps.mode == "map"
    assert ws_maps.vm_live_maps.mode == "map"


def test_spectra_workspace_range_selector_to_live_series_synchronization(qapp):
    """Test full cycle in Spectra workspace: select spectra -> toggle RGB brackets -> drag bracket -> verify variation graph."""
    ws = VWorkspaceSpectra()
    ws.show()

    x = np.linspace(500, 1500, 101)
    fnames = []
    for i in range(5):
        fname = f"spectrum_{i + 1}"
        fnames.append(fname)
        # Peak at 900 + i*5 with amplitude 10 + i*4
        center = 900.0 + i * 5.0
        amp = 10.0 + i * 4.0
        y = (amp * np.exp(-((x - center) ** 2) / (2 * 20.0 ** 2)) + 3.0).reshape(1, -1)
        ws.vm.store.add_map(
            name=fname,
            x0=x.copy(),
            Y0=y.astype(np.float32),
            coords=np.array([[0.0, 0.0]], dtype=np.float64),
            fnames=[fname],
            is_active=np.array([True], dtype=bool),
        )

    ws.vm._emit_list_update()
    # Select all 5 spectra
    ws.v_spectra_list.select_all()
    assert len(ws.vm._get_selected_spectra()) == 5

    ws.bottom_tabs.setCurrentWidget(ws.v_live_maps)

    # 1. Toggle Red selector in spectra viewer
    ws.v_spectra_viewer.btn_range_red.setChecked(True)
    red_panel = ws.v_live_maps.panels[SELECTOR_RED]
    assert not red_panel.isHidden()
    assert ws.vm_live_maps._states[SELECTOR_RED]["active"] is True

    # 2. Set Red interval over the peak [850, 960]
    ws.v_spectra_viewer.range_manager.set_range(SELECTOR_RED, 850.0, 960.0)
    ws.vm_live_maps.recalculate(SELECTOR_RED)

    # 3. Verify Red subpanel rendered a 5-point variation curve with increasing intensity
    assert red_panel.line is not None
    assert len(red_panel._series_values) == 5
    assert np.all(np.diff(red_panel._series_values) > 0)

    # 4. Switch parameter to Peak position and verify increasing peak center trend
    ws.v_live_maps.parameter_change_requested.emit(SELECTOR_RED, PARAM_PEAK_POSITION)
    assert red_panel.line is not None
    expected_centers = np.array([900.0 + i * 5.0 for i in range(5)])
    assert np.allclose(red_panel._series_values, expected_centers, atol=6.0)

    # 5. Select only first 3 spectra -> graph updates to 3 points
    ws.vm.set_selected_fnames(fnames[:3])
    assert len(red_panel._series_values) == 3
    assert len(red_panel.line.get_xdata()) == 3

    # 6. Deactivate Red selector -> panel hidden
    ws.v_spectra_viewer.btn_range_red.setChecked(False)
    assert red_panel.isVisible() is False


def test_range_selector_to_live_maps_synchronization(qapp, test_map_data):
    """Test full cycle: button toggle -> bracket drag -> live map computation -> canvas update."""
    workspace = VWorkspaceMaps()
    workspace.show()

    # Inject synthetic map data directly into store and select it
    workspace.vm.store._maps[test_map_data.name] = test_map_data
    workspace.vm.current_map_name = test_map_data.name
    workspace.vm_live_maps.set_map_data(test_map_data)

    # Switch to Live Maps tab so its widgets are active
    workspace.bottom_tabs.setCurrentWidget(workspace.v_live_maps)

    # 1. Toggle Red selector in spectra viewer
    workspace.v_spectra_viewer.btn_range_red.setChecked(True)
    assert not workspace.v_live_maps.panels[SELECTOR_RED].isHidden()
    assert workspace.vm_live_maps._states[SELECTOR_RED]["active"] is True

    # 2. Adjust Red interval over the peak at 900
    workspace.v_spectra_viewer.range_manager.set_range(SELECTOR_RED, 850.0, 950.0)
    # Immediate recalculation for test
    workspace.vm_live_maps.recalculate(SELECTOR_RED)

    # 3. Verify Live Maps received valid grid
    red_panel = workspace.v_live_maps.panels[SELECTOR_RED]
    assert red_panel._grid is not None
    assert red_panel._grid.shape == (4, 5)
    assert red_panel.img is not None

    # 4. Change parameter to Peak position
    workspace.v_live_maps.parameter_change_requested.emit(SELECTOR_RED, PARAM_PEAK_POSITION)
    assert red_panel._grid is not None
    valid_grid = red_panel._grid[np.isfinite(red_panel._grid)]
    assert np.all(np.isclose(valid_grid, 900.0, atol=10.0))

    # 5. Deactivate Red selector -> panel hidden
    workspace.v_spectra_viewer.btn_range_red.setChecked(False)
    assert workspace.v_live_maps.panels[SELECTOR_RED].isVisible() is False
    assert workspace.vm_live_maps._states[SELECTOR_RED]["active"] is False


def _make_line_scan(name="line_scan", n=12):
    """1D line scan along X with a Red peak (700) growing and a constant Blue peak (1300)."""
    x = np.linspace(500, 1500, 101)
    xs = np.linspace(0.0, 11.0, n)
    Y = np.zeros((n, len(x)), dtype=np.float32)
    for i, px in enumerate(xs):
        Y[i] = ((2.0 + px) * np.exp(-((x - 700) ** 2) / (2 * 25 ** 2))
                + 4.0 * np.exp(-((x - 1300) ** 2) / (2 * 25 ** 2)) + 1.0)
    coords = np.column_stack([xs, np.full(n, 3.0)])
    return MapData(
        name=name, x0=x, Y0=Y, coords=coords, is_active=np.ones(n, dtype=bool),
        fnames=[f"{name}_({c[0]}, {c[1]})" for c in coords],
        colors=[None] * n, labels=[None] * n,
    )


def test_maps_workspace_live_maps_adapt_to_map_dimensionality_and_rb_ratio(qapp, test_map_data):
    """2D map → heatmaps, 1D profile → line plots, switching without refresh, plus R/B ratio in Green."""
    from spectroview.model.spectral_parameters import SELECTOR_BLUE, SELECTOR_GREEN, PARAM_RB_RATIO

    ws = VWorkspaceMaps()
    ws.show()
    line_md = _make_line_scan()
    ws.vm.store._maps[test_map_data.name] = test_map_data
    ws.vm.store._maps[line_md.name] = line_md
    ws.bottom_tabs.setCurrentWidget(ws.v_live_maps)

    ws.vm_live_maps.set_map_data(test_map_data)
    viewer = ws.v_spectra_viewer
    viewer.btn_range_red.setChecked(True)
    viewer.btn_range_blue.setChecked(True)
    viewer.btn_range_green.setChecked(True)
    viewer.range_manager.set_range(SELECTOR_RED, 650.0, 750.0)
    viewer.range_manager.set_range(SELECTOR_BLUE, 1250.0, 1350.0)
    viewer.range_manager.set_range(SELECTOR_GREEN, 850.0, 950.0)
    ws.vm_live_maps._recalculate_batch([SELECTOR_RED, SELECTOR_BLUE, SELECTOR_GREEN])

    panels = ws.v_live_maps.panels
    # 1. 2D map → three heatmaps
    for cid in (SELECTOR_RED, SELECTOR_BLUE, SELECTOR_GREEN):
        assert panels[cid].img is not None and panels[cid].line is None

    # 2. Switch to 1D profile → line plots along X, no manual refresh, auto aspect
    ws.vm_live_maps.set_map_data(line_md)
    for cid in (SELECTOR_RED, SELECTOR_BLUE, SELECTOR_GREEN):
        assert panels[cid].line is not None and panels[cid].img is None
        assert panels[cid].cbb_palette.isHidden() is True
        assert panels[cid].ax.get_aspect() == "auto"
    np.testing.assert_allclose(panels[SELECTOR_RED].line.get_xdata(), line_md.coords[:, 0])
    assert np.all(np.diff(panels[SELECTOR_RED]._series_values) > 0)

    # 3. Green → R/B ratio from the panel combo, rendered as a profile line
    panels[SELECTOR_GREEN].set_selected_parameter(PARAM_RB_RATIO)
    ratio = panels[SELECTOR_GREEN]._series_values
    red_vals = panels[SELECTOR_RED]._series_values
    blue_vals = panels[SELECTOR_BLUE]._series_values
    np.testing.assert_allclose(ratio, red_vals / blue_vals, rtol=1e-9)

    # 4. Moving the Red bracket updates the ratio (debounced path flushed)
    viewer.range_manager.set_range(SELECTOR_RED, 600.0, 800.0)
    ws.vm_live_maps.set_selector_range(SELECTOR_RED, 600.0, 800.0)
    ws.vm_live_maps._recalculate_pending()
    assert not np.allclose(panels[SELECTOR_GREEN]._series_values, ratio)

    # 5. Back to the 2D map → heatmaps again, Green ratio as heatmap
    ws.vm_live_maps.set_map_data(test_map_data)
    for cid in (SELECTOR_RED, SELECTOR_BLUE, SELECTOR_GREEN):
        assert panels[cid].img is not None and panels[cid].line is None
    assert panels[SELECTOR_GREEN]._grid.shape == (4, 5)


def test_maps_workspace_multi_selection_gating_and_batch_delete(qapp, test_map_data, monkeypatch):
    """End-to-end multi-map Live Maps, tab visibility gating, point-click immunity, and batch delete."""
    from PySide6.QtWidgets import QMessageBox

    ws = VWorkspaceMaps()
    ws.show()
    try:
        p1 = _make_line_scan("prof_1", n=10)
        p2 = _make_line_scan("prof_2", n=10)
        p2.Y0 = p2.Y0 * 1.8

        for md in (p1, p2, test_map_data):
            ws.vm.store._maps[md.name] = md
            ws.vm.maps[md.name] = None
            ws.vm.maps_metadata[md.name] = {}
        ws.v_maps_list.set_maps_names(["prof_1", "prof_2", test_map_data.name])

        # Activate Red selector while on Live Maps tab
        ws.bottom_tabs.setCurrentWidget(ws.v_live_maps)
        ws.v_spectra_viewer.btn_range_red.setChecked(True)
        ws.v_spectra_viewer.range_manager.set_range(SELECTOR_RED, 650.0, 750.0)
        ws.vm_live_maps.set_selector_range(SELECTOR_RED, 650.0, 750.0, immediate=True)

        # 1. Select prof_1 (Primary), then Ctrl-select prof_2 + 2D map -> 2 curves + excluded 2D notice
        ws.v_maps_list.maps_list.setCurrentItem(ws.v_maps_list._items_by_name["prof_1"])
        ws.v_maps_list._items_by_name["prof_2"].setSelected(True)
        ws.v_maps_list._items_by_name[test_map_data.name].setSelected(True)
        qapp.processEvents()
        red_panel = ws.v_live_maps.panels[SELECTOR_RED]
        assert len(red_panel.lines) == 2
        assert red_panel.img is None
        assert ws.v_live_maps.header_bar.isVisible() is True
        assert test_map_data.name in ws.v_live_maps.lbl_context_status.text()

        # 2. Clicking a point in VMapViewer causes ZERO Live Maps recomputation
        calls = []
        orig_set = ws.vm_live_maps.set_map_datas
        monkeypatch.setattr(
            ws.vm_live_maps, "set_map_datas",
            lambda *a, **k: calls.append(True) or orig_set(*a, **k),
        )
        ws._on_map_viewer_selection([(float(p1.coords[0, 0]), float(p1.coords[0, 1]))])
        assert calls == []

        # 3. Switching away from Live Maps tab gates updates until tab becomes active again
        ws.bottom_tabs.setCurrentIndex(0)  # Fit Model Builder (Live Maps hidden)
        ws.vm.select_maps(["prof_1", test_map_data.name], test_map_data.name)
        assert calls == []
        assert ws._live_maps_dirty is True

        ws.bottom_tabs.setCurrentWidget(ws.v_live_maps)
        assert len(calls) == 1
        assert ws._live_maps_dirty is False
        assert red_panel.img is not None and red_panel.lines == []

        # 4. Batch delete selected maps with confirmation clears Live Maps cleanly when empty
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
        ws.vm.select_maps(["prof_1", "prof_2", test_map_data.name], "prof_1")
        ws._on_delete_maps_requested()
        qapp.processEvents()
        assert ws.vm.maps == {}
        assert ws.vm.current_map_name is None
        assert red_panel.img is None and red_panel.lines == []
    finally:
        ws.close()
        ws.deleteLater()
