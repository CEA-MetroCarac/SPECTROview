"""Unit tests for VLiveMaps and LiveMapSubpanel."""

import numpy as np

from spectroview.model.spectral_parameters import (
    PARAM_INTEGRATED_INTENSITY,
    PARAM_PEAK_POSITION,
    SELECTOR_RED,
    SELECTOR_BLUE,
    SELECTOR_GREEN,
)
from spectroview.view.components.v_live_maps import (
    VLiveMaps,
    LiveMapSubpanel,
)


def test_live_map_subpanel_creation(qapp):
    panel = LiveMapSubpanel(SELECTOR_RED)
    assert panel.color_id == SELECTOR_RED
    assert panel.cbb_param.currentText() == PARAM_INTEGRATED_INTENSITY

    # Signal on parameter change
    emitted = []
    panel.parameter_changed.connect(lambda cid, p: emitted.append((cid, p)))
    panel.set_selected_parameter(PARAM_PEAK_POSITION)
    assert len(emitted) == 1
    assert emitted[0] == (SELECTOR_RED, PARAM_PEAK_POSITION)

    # Range label update
    panel.set_range_text(100.5, 250.8)
    assert "[100.5, 250.8]" in panel.lbl_range.text()


def test_live_map_subpanel_render(qapp):
    panel = LiveMapSubpanel(SELECTOR_BLUE)
    grid = np.arange(12, dtype=float).reshape(3, 4)
    extent = [0.0, 4.0, 0.0, 3.0]

    panel.update_map(grid, extent, 0.0, 11.0, status_msg="OK", theme="Light Mode")
    assert panel.img is not None
    assert panel.cbar is not None
    orig_img = panel.img
    orig_cbar = panel.cbar

    # In-place update with new grid preserves existing image and colorbar (no redraw bounce)
    grid2 = np.arange(12, dtype=float).reshape(3, 4) * 2.0
    panel.update_map(grid2, extent, 0.0, 22.0, status_msg="OK", theme="Light Mode")
    assert panel.img is orig_img
    assert panel.cbar is orig_cbar

    # Render None / empty grid gracefully
    panel.update_map(None, extent, 0.0, 1.0, status_msg="No data", theme="Dark Mode")
    assert panel.cbar is None


def test_v_live_maps_visibility_toggling(qapp):
    live_maps = VLiveMaps()
    live_maps.show()

    # Initially placeholder visible, maps container hidden
    assert live_maps.lbl_placeholder.isVisible() is True
    assert live_maps.maps_container.isVisible() is False
    assert live_maps.panels[SELECTOR_RED].isVisible() is False

    # Activate Red
    live_maps.set_selector_active(SELECTOR_RED, True)
    assert live_maps.lbl_placeholder.isVisible() is False
    assert live_maps.maps_container.isVisible() is True
    assert live_maps.panels[SELECTOR_RED].isVisible() is True
    assert live_maps.panels[SELECTOR_BLUE].isVisible() is False
    assert live_maps.panels[SELECTOR_GREEN].isVisible() is False

    # Activate Blue
    live_maps.set_selector_active(SELECTOR_BLUE, True)
    assert live_maps.panels[SELECTOR_RED].isVisible() is True
    assert live_maps.panels[SELECTOR_BLUE].isVisible() is True
    assert live_maps.panels[SELECTOR_GREEN].isVisible() is False

    # Deactivate both
    live_maps.set_selector_active(SELECTOR_RED, False)
    live_maps.set_selector_active(SELECTOR_BLUE, False)
    assert live_maps.lbl_placeholder.isVisible() is True
    assert live_maps.maps_container.isVisible() is False


def test_v_live_maps_update_data_and_theme(qapp):
    live_maps = VLiveMaps()
    live_maps.set_selector_active(SELECTOR_GREEN, True)

    grid = np.ones((5, 5))
    live_maps.update_map_data(SELECTOR_GREEN, grid, [0, 5, 0, 5], 0.0, 1.0, "OK")
    assert live_maps.panels[SELECTOR_GREEN].img is not None

    # Apply dark theme
    live_maps.apply_theme("dark")
    assert live_maps._current_theme == "dark"
