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


def test_workspace_isolation_and_live_maps_tab_position(qapp):
    """Verify Live Maps tab and range selectors are in Maps workspace, but NOT Spectra workspace."""
    # 1. Spectra Workspace: no Live Maps, range selectors hidden
    ws_spectra = VWorkspaceSpectra()
    spectra_tab_texts = [ws_spectra.bottom_tabs.tabText(i) for i in range(ws_spectra.bottom_tabs.count())]
    assert "Live Maps" not in spectra_tab_texts, "Live Maps must NOT be present in Spectra workspace"
    assert not hasattr(ws_spectra, "v_live_maps")
    assert ws_spectra.v_spectra_viewer.range_panel.isHidden() is True
    assert spectra_tab_texts[-1] == "More", "More tab must be at the end"

    # 2. Maps Workspace: Live Maps present at index 2, range selectors visible
    ws_maps = VWorkspaceMaps()
    maps_tab_texts = [ws_maps.bottom_tabs.tabText(i) for i in range(ws_maps.bottom_tabs.count())]
    assert "Live Maps" in maps_tab_texts, "Live Maps must be present in Maps workspace"
    assert maps_tab_texts.index("Live Maps") == 2, "Live Maps must be at index 2 (after Fit Results)"
    assert maps_tab_texts[-1] == "More", "More tab must be at the end"
    assert ws_maps.v_spectra_viewer.range_panel.isHidden() is False


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
