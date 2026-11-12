"""Unit tests for map dimensionality detection and R/B intensity ratio
(spectroview.model.spectral_parameters)."""

import numpy as np
import pytest

from spectroview.model.spectral_parameters import (
    analyze_map_geometry,
    compute_intensity_ratio,
)


class TestAnalyzeMapGeometry:
    def test_regular_2d_grid_is_2d(self):
        gx, gy = np.meshgrid(np.arange(5.0), np.arange(4.0))
        coords = np.column_stack([gx.ravel(), gy.ravel()])
        geom = analyze_map_geometry(coords)
        assert geom.ndim == 2
        assert not geom.is_profile
        assert geom.positions is None

    def test_elongated_two_row_map_is_still_2d(self):
        gx, gy = np.meshgrid(np.arange(200.0), np.arange(2.0))
        geom = analyze_map_geometry(np.column_stack([gx.ravel(), gy.ravel()]))
        assert geom.ndim == 2

    def test_horizontal_line_scan_is_profile_along_x(self):
        x = np.array([3.0, 1.0, 2.0, 0.0])  # unordered acquisition
        coords = np.column_stack([x, np.full_like(x, 7.5)])
        geom = analyze_map_geometry(coords)
        assert geom.is_profile
        assert geom.axis_label == "X"
        np.testing.assert_allclose(geom.positions, x)
        np.testing.assert_allclose(geom.positions[geom.order], [0.0, 1.0, 2.0, 3.0])

    def test_vertical_line_scan_is_profile_along_y(self):
        y = np.linspace(-5.0, 5.0, 11)
        geom = analyze_map_geometry(np.column_stack([np.zeros_like(y), y]))
        assert geom.is_profile
        assert geom.axis_label == "Y"
        np.testing.assert_allclose(geom.positions, y)

    def test_diagonal_line_scan_is_profile_with_distance(self):
        t = np.linspace(0.0, 10.0, 6)
        coords = np.column_stack([1.0 + 3.0 * t, 2.0 + 4.0 * t])  # step length 5*dt
        geom = analyze_map_geometry(coords)
        assert geom.is_profile
        assert geom.axis_label == "Distance"
        np.testing.assert_allclose(geom.positions[geom.order], 5.0 * t, atol=1e-9)

    def test_nearly_axis_aligned_jitter_keeps_native_axis(self):
        x = np.linspace(0.0, 100.0, 51)
        y = 2.0 + 1e-6 * np.sin(x)  # rounding jitter in Y
        geom = analyze_map_geometry(np.column_stack([x, y]))
        assert geom.is_profile
        assert geom.axis_label == "X"

    @pytest.mark.parametrize(
        "coords",
        [None, np.empty((0, 2)), np.array([[1.0, 1.0]]), np.array([[1.0, 1.0], [1.0, 1.0]]),
         np.array([[0.0, np.nan], [1.0, 0.0]]), np.zeros((3, 3))],
    )
    def test_degenerate_inputs_fall_back_to_2d(self, coords):
        assert analyze_map_geometry(coords).ndim == 2


class TestComputeIntensityRatio:
    def test_basic_ratio(self):
        ratio, n_masked = compute_intensity_ratio(np.array([2.0, 6.0, 9.0]), np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(ratio, [2.0, 3.0, 3.0])
        assert n_masked == 0

    def test_zero_and_near_zero_denominator_masked(self):
        num = np.array([1.0, 1.0, 1.0, 1.0])
        den = np.array([10.0, 0.0, 1e-9, 5.0])
        with np.errstate(all="raise"):  # must never divide by zero
            ratio, n_masked = compute_intensity_ratio(num, den)
        assert n_masked == 2
        assert np.isnan(ratio[1]) and np.isnan(ratio[2])
        np.testing.assert_allclose(ratio[[0, 3]], [0.1, 0.2])
        assert not np.any(np.isinf(ratio))

    def test_non_finite_operands_masked(self):
        ratio, n_masked = compute_intensity_ratio(
            np.array([np.nan, 1.0, 2.0]), np.array([1.0, np.inf, 2.0])
        )
        assert n_masked == 2
        assert np.isnan(ratio[0]) and np.isnan(ratio[1])
        assert ratio[2] == pytest.approx(1.0)

    def test_all_zero_denominator_fully_masked(self):
        ratio, n_masked = compute_intensity_ratio(np.ones(4), np.zeros(4))
        assert n_masked == 4
        assert np.all(np.isnan(ratio))

    def test_relative_threshold_configurable(self):
        ratio, n_masked = compute_intensity_ratio(np.ones(2), np.array([100.0, 5.0]), rel_eps=0.1)
        assert n_masked == 1
        assert np.isnan(ratio[1])

