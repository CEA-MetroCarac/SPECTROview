"""Unit tests for spectroview.viewmodel.vm_live_maps."""

import numpy as np
import pytest

from spectroview.model.spectra_store import MapData
from spectroview.model.spectral_parameters import (
    PARAM_INTEGRATED_INTENSITY,
    PARAM_PEAK_POSITION,
    SELECTOR_RED,
    SELECTOR_BLUE,
    SELECTOR_GREEN,
)
from spectroview.viewmodel.vm_live_maps import VMLiveMaps


@pytest.fixture
def sample_map_data():
    """Create a synthetic MapData instance with 12 spectra (3x4 grid)."""
    x = np.linspace(400, 1600, 121)
    N = 12
    Y = np.zeros((N, len(x)), dtype=np.float32)
    coords = np.zeros((N, 2), dtype=np.float64)

    for i in range(N):
        gx = float(i % 4)
        gy = float(i // 4)
        coords[i] = [gx, gy]
        # Peak at 700 with amplitude proportional to gx, peak at 1300 proportional to gy
        p1 = (5.0 + gx * 2.0) * np.exp(-((x - 700) ** 2) / (2 * 25 ** 2))
        p2 = (10.0 + gy * 3.0) * np.exp(-((x - 1300) ** 2) / (2 * 30 ** 2))
        Y[i] = p1 + p2 + 2.0

    return MapData(
        name="test_map",
        x0=x,
        Y0=Y,
        coords=coords,
        is_active=np.ones(N, dtype=bool),
        fnames=[f"s_{i}" for i in range(N)],
        colors=[None] * N,
        labels=[None] * N,
    )


def test_vm_live_maps_initial_state(qapp):
    vm = VMLiveMaps()
    for cid in (SELECTOR_RED, SELECTOR_BLUE, SELECTOR_GREEN):
        assert vm._states[cid]["active"] is False
        assert vm._states[cid]["param"] == PARAM_INTEGRATED_INTENSITY
        assert vm._states[cid]["range"][0] < vm._states[cid]["range"][1]


def test_vm_live_maps_toggle_active(qapp):
    vm = VMLiveMaps()
    states_received = []
    vm.selector_state_changed.connect(lambda cid, act, x0, x1: states_received.append((cid, act, x0, x1)))

    vm.set_selector_active(SELECTOR_RED, True)
    assert vm._states[SELECTOR_RED]["active"] is True
    assert vm._states[SELECTOR_BLUE]["active"] is False
    assert len(states_received) == 1
    assert states_received[0][0] == SELECTOR_RED
    assert states_received[0][1] is True

    vm.set_selector_active(SELECTOR_RED, False)
    assert vm._states[SELECTOR_RED]["active"] is False
    assert len(states_received) == 2
    assert states_received[1][1] is False


def test_vm_live_maps_set_range_independence(qapp):
    vm = VMLiveMaps()
    vm.set_selector_range(SELECTOR_RED, 600.0, 800.0)
    assert vm._states[SELECTOR_RED]["range"] == (600.0, 800.0)
    # Blue and Green unaffected
    assert vm._states[SELECTOR_BLUE]["range"] != (600.0, 800.0)
    assert vm._states[SELECTOR_GREEN]["range"] != (600.0, 800.0)


def test_vm_live_maps_calculation_with_map_data(qapp, sample_map_data):
    vm = VMLiveMaps()
    results = {}

    def on_ready(cid, grid, extent, vmin, vmax, msg):
        results[cid] = {
            "grid": grid,
            "extent": extent,
            "vmin": vmin,
            "vmax": vmax,
            "msg": msg,
        }

    vm.map_data_ready.connect(on_ready)

    # Activate Red and set range over Peak 1 (700)
    vm.set_selector_active(SELECTOR_RED, True)
    vm.set_selector_range(SELECTOR_RED, 650.0, 750.0, immediate=True)
    vm.set_map_data(sample_map_data)

    assert SELECTOR_RED in results
    res = results[SELECTOR_RED]
    assert res["grid"] is not None
    assert res["grid"].shape == (3, 4)  # 3 rows, 4 cols
    assert res["vmin"] < res["vmax"]


def test_vm_live_maps_parameter_change(qapp, sample_map_data):
    vm = VMLiveMaps()
    results = {}
    vm.map_data_ready.connect(lambda cid, g, e, v0, v1, m: results.update({cid: (g, m)}))

    vm.set_map_data(sample_map_data)
    vm.set_selector_active(SELECTOR_BLUE, True)
    vm.set_selector_range(SELECTOR_BLUE, 1200.0, 1400.0, immediate=True)

    # Change to Peak Position
    vm.set_selector_parameter(SELECTOR_BLUE, PARAM_PEAK_POSITION)
    assert SELECTOR_BLUE in results
    grid, msg = results[SELECTOR_BLUE]
    assert grid is not None
    # Peak 2 is at 1300
    assert np.all(np.isclose(grid[~np.isnan(grid)], 1300.0, atol=15.0))


def test_vm_live_maps_no_data_graceful(qapp):
    vm = VMLiveMaps()
    results = {}
    vm.map_data_ready.connect(lambda cid, g, e, v0, v1, m: results.update({cid: (g, m)}))

    vm.set_selector_active(SELECTOR_GREEN, True)
    assert SELECTOR_GREEN in results
    grid, msg = results[SELECTOR_GREEN]
    assert grid is None
    assert "No hyperspectral map loaded" in msg


def test_vm_live_maps_spectra_series_mode(qapp):
    """Test VMLiveMaps in 'spectra' mode computing parameter trends across selected spectra."""
    vm = VMLiveMaps(mode="spectra")
    series_results = {}
    vm.series_data_ready.connect(
        lambda cid, vals, lbls, msg: series_results.update({cid: (vals, lbls, msg)})
    )

    # 1. No spectra selected -> emits empty array with message
    vm.set_selector_active(SELECTOR_RED, True)
    assert SELECTOR_RED in series_results
    vals0, lbls0, msg0 = series_results[SELECTOR_RED]
    assert len(vals0) == 0
    assert "No spectra selected" in msg0

    # 2. Load 4 discrete spectra with linearly increasing peak height at 900
    x = np.linspace(500, 1500, 101)
    mds = []
    for i in range(4):
        amp = 10.0 + i * 5.0
        y = (amp * np.exp(-((x - 900.0) ** 2) / (2 * 25.0 ** 2)) + 2.0).reshape(1, -1)
        mds.append(
            MapData(
                name=f"spec_{i + 1}",
                x0=x,
                Y0=y.astype(np.float32),
                coords=np.array([[0.0, 0.0]]),
                is_active=np.array([True]),
                fnames=[f"spec_{i + 1}"],
                colors=[None],
                labels=[f"S{i + 1}"],
            )
        )

    vm.set_selector_range(SELECTOR_RED, 850.0, 950.0, immediate=True)
    vm.set_series_data(mds)

    vals, lbls, msg = series_results[SELECTOR_RED]
    assert len(vals) == 4
    assert lbls == ["S1", "S2", "S3", "S4"]
    # Integrated intensity increases monotonically across S1..S4
    assert np.all(np.diff(vals) > 0)

    # 3. User-adjusted range [850, 950] is preserved when selection subset changes
    vm.set_series_data(mds[:2])
    assert vm._states[SELECTOR_RED]["range"] == (850.0, 950.0)
    vals_sub, lbls_sub, _ = series_results[SELECTOR_RED]
    assert len(vals_sub) == 2
    assert lbls_sub == ["S1", "S2"]
