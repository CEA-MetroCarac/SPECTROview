"""Unit tests for VRangeSelectorPanel and RangeSelectorManager."""

import numpy as np
import matplotlib.pyplot as plt

from spectroview.model.spectral_parameters import (
    SELECTOR_RED,
    SELECTOR_BLUE,
    SELECTOR_GREEN,
    SELECTOR_KEYS,
)
from spectroview.view.components.v_range_selector import (
    VRangeSelectorPanel,
    RangeSelectorManager,
)


def test_range_selector_panel_buttons(qapp):
    panel = VRangeSelectorPanel()
    assert len(panel.buttons) == 3
    for cid in SELECTOR_KEYS:
        assert cid in panel.buttons
        btn = panel.buttons[cid]
        assert btn.isCheckable() is True
        assert btn.isChecked() is False

    signals_received = []
    panel.selector_toggled.connect(lambda cid, act: signals_received.append((cid, act)))

    # Toggle Red
    panel.buttons[SELECTOR_RED].setChecked(True)
    assert len(signals_received) == 1
    assert signals_received[0] == (SELECTOR_RED, True)
    assert panel.buttons[SELECTOR_BLUE].isChecked() is False
    assert panel.buttons[SELECTOR_GREEN].isChecked() is False

    # Toggle Blue
    panel.buttons[SELECTOR_BLUE].setChecked(True)
    assert len(signals_received) == 2
    assert signals_received[1] == (SELECTOR_BLUE, True)

    # Untoggle Red
    panel.buttons[SELECTOR_RED].setChecked(False)
    assert len(signals_received) == 3
    assert signals_received[2] == (SELECTOR_RED, False)
    assert panel.buttons[SELECTOR_BLUE].isChecked() is True


def test_range_selector_manager_initialization(qapp):
    mgr = RangeSelectorManager()
    assert mgr.is_active(SELECTOR_RED) is False
    assert mgr.is_active(SELECTOR_BLUE) is False
    assert mgr.is_active(SELECTOR_GREEN) is False

    # Initialize from axis
    x_axis = np.linspace(100, 1100, 500)
    mgr.initialize_ranges_from_axis(x_axis)

    r_min, r_max = mgr.get_range(SELECTOR_RED)
    b_min, b_max = mgr.get_range(SELECTOR_BLUE)
    g_min, g_max = mgr.get_range(SELECTOR_GREEN)

    assert 100 < r_min < r_max < b_min < b_max < g_min < g_max < 1100


def test_range_selector_manager_render_and_clear(qapp):
    mgr = RangeSelectorManager()
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 100)

    # Initially none active -> no artists
    mgr.render(ax)
    assert len(mgr._artists[SELECTOR_RED]) == 0

    # Activate Red
    mgr.set_active(SELECTOR_RED, True)
    mgr.set_range(SELECTOR_RED, 200, 400)
    mgr.render(ax)
    # Should have span, left line, right line, and 4 bracket ticks = 7 artists
    assert len(mgr._artists[SELECTOR_RED]) >= 3
    assert len(mgr._artists[SELECTOR_BLUE]) == 0

    # Clear artists
    mgr.clear_artists(ax)
    assert len(mgr._artists[SELECTOR_RED]) == 0
    plt.close(fig)


def test_range_selector_manager_drag_lifecycle(qapp):
    mgr = RangeSelectorManager()
    mgr.set_active(SELECTOR_RED, True)
    mgr.set_range(SELECTOR_RED, 200.0, 400.0)

    # Start drag on 'min' boundary
    mgr.start_drag(SELECTOR_RED, "min", 200.0)
    assert mgr.is_dragging() is True

    # Move to 250
    mgr.update_drag(250.0)
    r_min, r_max = mgr.get_range(SELECTOR_RED)
    assert np.isclose(r_min, 250.0)
    assert np.isclose(r_max, 400.0)

    # Finish drag
    mgr.finish_drag()
    assert mgr.is_dragging() is False
