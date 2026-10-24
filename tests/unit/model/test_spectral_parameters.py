"""Unit tests for spectroview.model.spectral_parameters."""

import numpy as np
import pytest

from spectroview.model.spectral_parameters import (
    compute_spectral_parameter,
    compute_map_grid,
    is_fit_parameter,
    PARAM_INTEGRATED_INTENSITY,
    PARAM_MEAN_INTENSITY,
    PARAM_MAX_INTENSITY,
    PARAM_PEAK_POSITION,
    PARAM_PEAK_HEIGHT,
    PARAM_PEAK_POSITION_FIT,
    PARAM_PEAK_HEIGHT_FIT,
    PARAM_PEAK_AREA_FIT,
)


@pytest.fixture
def synthetic_data():
    """Create synthetic spectral dataset with 10 spectra and known peaks."""
    x = np.linspace(500, 1500, 201)  # step = 5.0
    N = 10
    # Two peaks: Peak 1 at 800, Peak 2 at 1200
    Y = np.zeros((N, len(x)), dtype=np.float64)
    for i in range(N):
        amp1 = 10.0 + i * 2.0
        amp2 = 25.0 - i * 1.5
        # Gaussians
        p1 = amp1 * np.exp(-((x - 800) ** 2) / (2 * 20 ** 2))
        p2 = amp2 * np.exp(-((x - 1200) ** 2) / (2 * 30 ** 2))
        baseline = 5.0 + 0.005 * x
        Y[i] = p1 + p2 + baseline

    # Spatial coords (2x5 grid)
    coords = np.zeros((N, 2))
    for i in range(N):
        coords[i] = [float(i % 5), float(i // 5)]

    # Peak fit results for p1 and p2
    param_names = [
        "m01_x0", "m01_ampli", "m01_fwhm",
        "m02_x0", "m02_ampli", "m02_fwhm",
    ]
    peak_params = np.zeros((N, 6))
    for i in range(N):
        amp1 = 10.0 + i * 2.0
        amp2 = 25.0 - i * 1.5
        peak_params[i] = [800.0, amp1, 20.0, 1200.0, amp2, 30.0]

    return {
        "x": x,
        "Y": Y,
        "coords": coords,
        "param_names": param_names,
        "peak_params": peak_params,
    }


def test_is_fit_parameter():
    assert is_fit_parameter(PARAM_PEAK_POSITION_FIT) is True
    assert is_fit_parameter(PARAM_PEAK_HEIGHT_FIT) is True
    assert is_fit_parameter(PARAM_PEAK_AREA_FIT) is True
    assert is_fit_parameter(PARAM_INTEGRATED_INTENSITY) is False
    assert is_fit_parameter(PARAM_MEAN_INTENSITY) is False
    assert is_fit_parameter(PARAM_MAX_INTENSITY) is False
    assert is_fit_parameter(PARAM_PEAK_POSITION) is False
    assert is_fit_parameter(PARAM_PEAK_HEIGHT) is False


def test_compute_integrated_intensity(synthetic_data):
    x = synthetic_data["x"]
    Y = synthetic_data["Y"]
    vals, ok, msg = compute_spectral_parameter(x, Y, (750, 850), PARAM_INTEGRATED_INTENSITY)
    assert ok is True
    assert len(vals) == len(Y)
    assert np.all(vals > 0)
    # vals should increase with index i since amp1 increases
    assert vals[0] < vals[-1]


def test_compute_mean_and_max_intensity(synthetic_data):
    x = synthetic_data["x"]
    Y = synthetic_data["Y"]
    mean_vals, ok_m, _ = compute_spectral_parameter(x, Y, (750, 850), PARAM_MEAN_INTENSITY)
    max_vals, ok_x, _ = compute_spectral_parameter(x, Y, (750, 850), PARAM_MAX_INTENSITY)
    assert ok_m is True and ok_x is True
    assert np.all(max_vals > mean_vals)


def test_compute_peak_position(synthetic_data):
    x = synthetic_data["x"]
    Y = synthetic_data["Y"]
    # Peak 1 is at 800
    pos_p1, ok1, _ = compute_spectral_parameter(x, Y, (750, 850), PARAM_PEAK_POSITION)
    assert ok1 is True
    assert np.all(np.isclose(pos_p1, 800.0, atol=6.0))

    # Peak 2 is at 1200
    pos_p2, ok2, _ = compute_spectral_parameter(x, Y, (1150, 1250), PARAM_PEAK_POSITION)
    assert ok2 is True
    assert np.all(np.isclose(pos_p2, 1200.0, atol=6.0))


def test_compute_peak_height_baseline_subtracted(synthetic_data):
    x = synthetic_data["x"]
    Y = synthetic_data["Y"]
    # Local baseline subtracted
    h_vals, ok, _ = compute_spectral_parameter(x, Y, (700, 900), PARAM_PEAK_HEIGHT, baseline_subtracted=False)
    assert ok is True
    # Should be close to amp1 (10 + 2*i)
    expected_amps = 10.0 + 2.0 * np.arange(len(Y))
    assert np.all(np.isclose(h_vals, expected_amps, atol=1.5))


def test_compute_fit_parameters(synthetic_data):
    x = synthetic_data["x"]
    Y = synthetic_data["Y"]
    p_names = synthetic_data["param_names"]
    p_params = synthetic_data["peak_params"]

    # Peak 1 fit position
    fit_pos, ok_pos, msg_pos = compute_spectral_parameter(
        x, Y, (750, 850), PARAM_PEAK_POSITION_FIT,
        peak_params=p_params, param_names=p_names
    )
    assert ok_pos is True
    assert np.allclose(fit_pos, 800.0)

    # Peak 1 fit height
    fit_h, ok_h, msg_h = compute_spectral_parameter(
        x, Y, (750, 850), PARAM_PEAK_HEIGHT_FIT,
        peak_params=p_params, param_names=p_names
    )
    assert ok_h is True
    expected_amps = 10.0 + 2.0 * np.arange(len(Y))
    assert np.allclose(fit_h, expected_amps)

    # Peak 1 fit area
    fit_area, ok_a, msg_a = compute_spectral_parameter(
        x, Y, (750, 850), PARAM_PEAK_AREA_FIT,
        peak_params=p_params, param_names=p_names
    )
    assert ok_a is True
    assert np.all(fit_area > 0)


def test_fit_parameters_unavailable(synthetic_data):
    x = synthetic_data["x"]
    Y = synthetic_data["Y"]
    # Without fit params
    vals, ok, msg = compute_spectral_parameter(
        x, Y, (750, 850), PARAM_PEAK_AREA_FIT,
        peak_params=None, param_names=None
    )
    assert ok is False
    assert np.all(np.isnan(vals))
    assert "No fitting results" in msg


def test_edge_cases(synthetic_data):
    x = synthetic_data["x"]
    Y = synthetic_data["Y"]

    # Inverted range (850, 750)
    vals_inv, ok_inv, _ = compute_spectral_parameter(x, Y, (850, 750), PARAM_INTEGRATED_INTENSITY)
    vals_norm, ok_norm, _ = compute_spectral_parameter(x, Y, (750, 850), PARAM_INTEGRATED_INTENSITY)
    assert ok_inv is True and ok_norm is True
    assert np.allclose(vals_inv, vals_norm)

    # Out of bounds range
    vals_out, ok_out, msg_out = compute_spectral_parameter(x, Y, (2000, 3000), PARAM_MEAN_INTENSITY)
    assert ok_out is False
    assert np.all(np.isnan(vals_out))

    # Single spectrum 1D array
    vals_1d, ok_1d, _ = compute_spectral_parameter(x, Y[0], (750, 850), PARAM_MAX_INTENSITY)
    assert ok_1d is True
    assert len(vals_1d) == 1

    # Empty axis
    vals_empty, ok_empty, _ = compute_spectral_parameter(np.array([]), Y, (100, 200), PARAM_MEAN_INTENSITY)
    assert ok_empty is False


def test_compute_map_grid(synthetic_data):
    coords = synthetic_data["coords"]
    # 10 values
    values = np.arange(10, dtype=float)

    grid, extent, vmin, vmax = compute_map_grid(coords, values)
    assert grid is not None
    assert grid.shape == (2, 5)
    assert vmin < vmax
    assert len(extent) == 4
    # Check values mapped correctly
    assert grid[0, 0] == 0.0
    assert grid[0, 4] == 4.0
    assert grid[1, 0] == 5.0
    assert grid[1, 4] == 9.0
