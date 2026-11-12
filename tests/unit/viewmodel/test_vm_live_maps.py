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
from spectroview.viewmodel.vm_live_maps import VMLiveMaps, MSG_RATIO_NEEDS_RB
from spectroview.model.spectral_parameters import PARAM_RB_RATIO


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


# ── 1D profile maps & R/B ratio ──────────────────────────────────────────────


def _make_profile_map(n=8, along="x", blue_zero_at=None):
    """1D line scan with Red peak (700) growing and Blue peak (1300) constant."""
    x = np.linspace(400, 1600, 121)
    pos = np.arange(n, dtype=np.float64)[::-1]  # reversed acquisition order
    Y = np.zeros((n, len(x)), dtype=np.float32)
    for i, p in enumerate(pos):
        red = (2.0 + p) * np.exp(-((x - 700) ** 2) / (2 * 25 ** 2))
        blue = 4.0 * np.exp(-((x - 1300) ** 2) / (2 * 25 ** 2))
        if blue_zero_at is not None and i == blue_zero_at:
            blue = 0.0 * blue
        Y[i] = red + blue
    coords = np.column_stack([pos, np.zeros(n)]) if along == "x" else np.column_stack([np.zeros(n), pos])
    return MapData(
        name=f"profile_{along}", x0=x, Y0=Y, coords=coords,
        is_active=np.ones(n, dtype=bool), fnames=[f"p_{i}" for i in range(n)],
        colors=[None] * n, labels=[None] * n,
    )


def _collect(vm):
    out = {"map": {}, "profile": {}}
    vm.map_data_ready.connect(lambda cid, g, e, v0, v1, m: out["map"].update({cid: (g, m)}))
    vm.profile_data_ready.connect(
        lambda cid, pos, vals, lbl, m: out["profile"].update({cid: (pos, vals, lbl, m)})
    )
    return out


def test_vm_profile_map_emits_line_data_sorted_by_position(qapp):
    vm = VMLiveMaps()
    out = _collect(vm)
    vm.set_selector_active(SELECTOR_RED, True)
    vm.set_selector_range(SELECTOR_RED, 650.0, 750.0, immediate=True)
    out["map"].clear()
    vm.set_map_data(_make_profile_map(along="y"))

    assert SELECTOR_RED not in out["map"]  # no heatmap for a 1D profile
    pos, vals, label, msg = out["profile"][SELECTOR_RED]
    assert label == "Y"
    np.testing.assert_allclose(pos, np.arange(8.0))
    assert np.all(np.diff(vals) > 0)  # red grows with position


def test_vm_switching_between_2d_and_profile_maps(qapp, sample_map_data):
    vm = VMLiveMaps()
    out = _collect(vm)
    vm.set_selector_active(SELECTOR_RED, True)

    vm.set_map_data(sample_map_data)
    assert out["map"][SELECTOR_RED][0].shape == (3, 4)

    out["map"].clear()
    vm.set_map_data(_make_profile_map())
    assert SELECTOR_RED in out["profile"] and SELECTOR_RED not in out["map"]
    assert vm.get_map_geometry().is_profile

    out["profile"].clear()
    vm.set_map_data(sample_map_data)
    assert SELECTOR_RED in out["map"] and SELECTOR_RED not in out["profile"]
    assert not vm.get_map_geometry().is_profile


def test_vm_geometry_cached_per_coords_array(qapp, monkeypatch):
    import spectroview.viewmodel.vm_live_maps as mod

    calls = []
    real = mod.analyze_map_geometry
    monkeypatch.setattr(mod, "analyze_map_geometry", lambda c: calls.append(1) or real(c))
    vm = VMLiveMaps()
    vm.set_selector_active(SELECTOR_RED, True)
    md = _make_profile_map()
    vm.set_map_data(md)
    for lo in (600.0, 620.0, 640.0):
        vm.set_selector_range(SELECTOR_RED, lo, lo + 100.0, immediate=True)
    assert len(calls) == 1


def test_vm_rb_ratio_requires_red_and_blue_active(qapp, sample_map_data):
    vm = VMLiveMaps()
    out = _collect(vm)
    vm.set_map_data(sample_map_data)
    vm.set_selector_active(SELECTOR_GREEN, True)
    vm.set_selector_parameter(SELECTOR_GREEN, PARAM_RB_RATIO)
    grid, msg = out["map"][SELECTOR_GREEN]
    assert grid is None
    assert msg == MSG_RATIO_NEEDS_RB

    # Activating Red and Blue refreshes the dependent Green ratio automatically
    vm.set_selector_range(SELECTOR_RED, 650.0, 750.0)
    vm.set_selector_range(SELECTOR_BLUE, 1250.0, 1350.0)
    vm.set_selector_active(SELECTOR_RED, True)
    vm.set_selector_active(SELECTOR_BLUE, True)
    grid, msg = out["map"][SELECTOR_GREEN]
    assert grid is not None and grid.shape == (3, 4)
    assert msg == "OK"


def test_vm_rb_ratio_2d_values_match_integrated_intensities(qapp, sample_map_data):
    from spectroview.model.spectral_parameters import compute_spectral_parameter, compute_map_grid

    vm = VMLiveMaps()
    out = _collect(vm)
    vm.set_map_data(sample_map_data)
    for cid, rng in ((SELECTOR_RED, (650.0, 750.0)), (SELECTOR_BLUE, (1250.0, 1350.0))):
        vm.set_selector_range(cid, *rng)
        vm.set_selector_active(cid, True)
    vm.set_selector_active(SELECTOR_GREEN, True)
    vm.set_selector_parameter(SELECTOR_GREEN, PARAM_RB_RATIO)

    i_r, _, _ = compute_spectral_parameter(sample_map_data.x0, sample_map_data.Y0, (650.0, 750.0))
    i_b, _, _ = compute_spectral_parameter(sample_map_data.x0, sample_map_data.Y0, (1250.0, 1350.0))
    expected, *_ = compute_map_grid(sample_map_data.coords, i_r / i_b)
    np.testing.assert_allclose(out["map"][SELECTOR_GREEN][0], expected, rtol=1e-9)


def test_vm_rb_ratio_follows_red_bracket_moves(qapp, sample_map_data):
    vm = VMLiveMaps()
    out = _collect(vm)
    vm.set_map_data(sample_map_data)
    for cid, rng in ((SELECTOR_RED, (650.0, 750.0)), (SELECTOR_BLUE, (1250.0, 1350.0))):
        vm.set_selector_range(cid, *rng)
        vm.set_selector_active(cid, True)
    vm.set_selector_active(SELECTOR_GREEN, True)
    vm.set_selector_parameter(SELECTOR_GREEN, PARAM_RB_RATIO)
    before = out["map"][SELECTOR_GREEN][0].copy()

    # Debounced path: moving Red queues Green as a dependent
    vm.set_selector_range(SELECTOR_RED, 600.0, 800.0)
    assert vm._pending_calc == {SELECTOR_RED, SELECTOR_GREEN}
    vm._recalculate_pending()
    after = out["map"][SELECTOR_GREEN][0]
    assert not np.allclose(before, after)


def test_vm_rb_ratio_reuses_red_intensity_within_pass(qapp, sample_map_data, monkeypatch):
    vm = VMLiveMaps()
    vm.set_map_data(sample_map_data)
    for cid, rng in ((SELECTOR_RED, (650.0, 750.0)), (SELECTOR_BLUE, (1250.0, 1350.0))):
        vm.set_selector_range(cid, *rng)
        vm.set_selector_active(cid, True)
    vm.set_selector_active(SELECTOR_GREEN, True)
    vm.set_selector_parameter(SELECTOR_GREEN, PARAM_RB_RATIO)

    calls = []
    real = vm._compute_values
    monkeypatch.setattr(vm, "_compute_values", lambda rng, p: calls.append(rng) or real(rng, p))
    vm.set_selector_range(SELECTOR_RED, 600.0, 800.0, immediate=True)
    # Red computed once (shared with ratio) + Blue once — not three times
    assert sorted(calls) == [(600.0, 800.0), (1250.0, 1350.0)]
    assert vm._intensity_cache == {}  # pass-scoped cache is released


def test_vm_rb_ratio_profile_masks_zero_blue(qapp):
    vm = VMLiveMaps()
    out = _collect(vm)
    md = _make_profile_map(n=6, blue_zero_at=2)
    vm.set_map_data(md)
    for cid, rng in ((SELECTOR_RED, (650.0, 750.0)), (SELECTOR_BLUE, (1250.0, 1350.0))):
        vm.set_selector_range(cid, *rng)
        vm.set_selector_active(cid, True)
    vm.set_selector_active(SELECTOR_GREEN, True)
    vm.set_selector_parameter(SELECTOR_GREEN, PARAM_RB_RATIO)

    pos, vals, label, msg = out["profile"][SELECTOR_GREEN]
    assert label == "X"
    assert len(vals) == 6
    # acquisition index 2 has position 3 (reversed order) → sorted index 3
    assert np.isnan(vals[3])
    assert np.all(np.isfinite(np.delete(vals, 3)))
    assert "masked" in msg


def test_vm_rb_ratio_switch_back_to_green_intensity(qapp, sample_map_data):
    vm = VMLiveMaps()
    out = _collect(vm)
    vm.set_map_data(sample_map_data)
    vm.set_selector_active(SELECTOR_GREEN, True)
    vm.set_selector_parameter(SELECTOR_GREEN, PARAM_RB_RATIO)
    vm.set_selector_parameter(SELECTOR_GREEN, PARAM_INTEGRATED_INTENSITY)
    grid, msg = out["map"][SELECTOR_GREEN]
    assert grid is not None
    assert msg == "OK"


def test_vm_rb_ratio_spectra_series_mode(qapp):
    vm = VMLiveMaps(mode="spectra")
    series = {}
    vm.series_data_ready.connect(lambda cid, v, l, m: series.update({cid: (v, m)}))
    x = np.linspace(400, 1600, 121)
    mds = []
    for i in range(3):
        y = ((2.0 + i) * np.exp(-((x - 700) ** 2) / 1250) + 4.0 * np.exp(-((x - 1300) ** 2) / 1250))
        mds.append(MapData(
            name=f"s{i}", x0=x, Y0=y.reshape(1, -1).astype(np.float32),
            coords=np.zeros((1, 2)), is_active=np.array([True]), fnames=[f"s{i}"],
            colors=[None], labels=[f"S{i}"],
        ))
    vm.set_series_data(mds)
    for cid, rng in ((SELECTOR_RED, (650.0, 750.0)), (SELECTOR_BLUE, (1250.0, 1350.0))):
        vm.set_selector_range(cid, *rng)
        vm.set_selector_active(cid, True)
    vm.set_selector_active(SELECTOR_GREEN, True)
    vm.set_selector_parameter(SELECTOR_GREEN, PARAM_RB_RATIO)
    vals, msg = series[SELECTOR_GREEN]
    assert len(vals) == 3
    assert np.all(np.diff(vals) > 0)
