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


def test_range_selector_panel_alignment_and_dimensions(qapp):
    panel = VRangeSelectorPanel()
    assert panel.width() == 22
    for cid in SELECTOR_KEYS:
        btn = panel.buttons[cid]
        assert btn.width() == 22
        assert btn.height() == 22


def test_range_selector_hit_test_zoom_active(qapp):
    mgr = RangeSelectorManager()
    mgr.set_active(SELECTOR_RED, True)
    mgr.set_range(SELECTOR_RED, 200.0, 400.0)

    fig, ax = plt.subplots()
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 100)
    fig.canvas.draw()

    class FakeEvent:
        def __init__(self, x_data, y_val=50):
            self.inaxes = ax
            self.xdata = x_data
            self.ydata = y_val
            px, py = ax.transData.transform((x_data, y_val))
            self.x = px
            self.y = py
            self.button = 1

    # Click on min boundary line (200.0)
    ev_boundary = FakeEvent(200.0)
    assert mgr.hit_test(ax, ev_boundary, zoom_active=False) == (SELECTOR_RED, "min")
    assert mgr.hit_test(ax, ev_boundary, zoom_active=True) == (SELECTOR_RED, "min")

    # Click on body center (300.0)
    ev_body = FakeEvent(300.0)
    # When zoom is NOT active, body is draggable
    assert mgr.hit_test(ax, ev_body, zoom_active=False) == (SELECTOR_RED, "body")
    # When zoom IS active, body is ignored so Zoom can operate freely
    assert mgr.hit_test(ax, ev_body, zoom_active=True) is None

    plt.close(fig)


def test_render_does_not_mutate_datalim(qapp):
    """Ensure range selectors do not alter ax.dataLim or interfere with autoscale / rescale."""
    mgr = RangeSelectorManager()
    fig, ax = plt.subplots()
    ax.plot([500.0, 600.0], [10.0, 20.0])

    initial_datalim_x = list(ax.dataLim.intervalx)
    assert np.isclose(initial_datalim_x[0], 500.0)
    assert np.isclose(initial_datalim_x[1], 600.0)

    # Render selector at coordinates far outside [500, 600]
    mgr.set_active(SELECTOR_RED, True)
    mgr.set_range(SELECTOR_RED, 100.0, 300.0)
    mgr.render(ax)

    # dataLim must be completely unaffected
    assert np.isclose(ax.dataLim.intervalx[0], 500.0)
    assert np.isclose(ax.dataLim.intervalx[1], 600.0)

    # autoscale must scale purely to the [500, 600] data
    ax.autoscale()
    xlim = ax.get_xlim()
    assert 480.0 <= xlim[0] <= 500.0
    assert 600.0 <= xlim[1] <= 620.0
    plt.close(fig)


def test_refresh_or_clamp_ranges(qapp):
    """Ensure selectors are clamped or repositioned when data range changes (e.g. crop)."""
    mgr = RangeSelectorManager()
    mgr.set_range(SELECTOR_RED, 100.0, 200.0)
    mgr.set_range(SELECTOR_BLUE, 550.0, 750.0)

    # Case 1: Red is completely outside [500, 800] -> repositioned to default proportional sub-interval
    mgr.refresh_or_clamp_ranges(500.0, 800.0, target_color_id=SELECTOR_RED)
    r_min, r_max = mgr.get_range(SELECTOR_RED)
    assert 500.0 <= r_min < r_max <= 800.0

    # Case 2: Blue partially overlaps [500, 700] -> clamped to 700.0
    mgr.refresh_or_clamp_ranges(500.0, 700.0, target_color_id=SELECTOR_BLUE)
    b_min, b_max = mgr.get_range(SELECTOR_BLUE)
    assert np.isclose(b_min, 550.0)
    assert np.isclose(b_max, 700.0)

