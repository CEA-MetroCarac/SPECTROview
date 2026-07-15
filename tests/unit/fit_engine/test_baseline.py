"""Unit tests for fit_engine/baseline.py - single and batched baseline evaluation."""
import numpy as np
import pytest

from spectroview.fit_engine.baseline import (
    eval_baseline, eval_baseline_batch, get_baseline_method_meta,
)


@pytest.fixture
def x():
    return np.linspace(0.0, 100.0, 101)


@pytest.fixture
def linear_y(x):
    """y = 2x + 3, so a Linear baseline through two points recovers it exactly."""
    return 2.0 * x + 3.0


class TestGetBaselineMethodMeta:
    def test_known_internal_methods(self):
        assert get_baseline_method_meta("Linear")["use_points"] is True
        assert get_baseline_method_meta("Polynomial")["use_points"] is True
        assert get_baseline_method_meta(None)["use_points"] is False

    def test_pybaselines_whitelist_methods(self):
        meta = get_baseline_method_meta("airpls")
        assert meta["coef_kwarg"] == "lam"

    def test_unknown_method_returns_empty_dict(self):
        assert get_baseline_method_meta("not_a_real_method") == {}


class TestEvalBaselineNoMode:
    def test_no_mode_returns_zeros(self, x, linear_y):
        assert np.array_equal(eval_baseline(x, linear_y, {}), np.zeros_like(x))
        assert np.array_equal(eval_baseline(x, linear_y, {"mode": None}), np.zeros_like(x))


class TestEvalBaselineLinear:
    def test_unattached_two_points_recovers_exact_line(self, x, linear_y):
        config = {
            "mode": "Linear",
            "attached": False,
            "points": [[0.0, 100.0], [3.0, 203.0]],  # y=2x+3 at both ends
        }
        baseline = eval_baseline(x, linear_y, config)
        np.testing.assert_allclose(baseline, linear_y, atol=1e-8)

    def test_single_point_is_constant(self, x, linear_y):
        config = {"mode": "Linear", "attached": False, "points": [[50.0], [7.5]]}
        baseline = eval_baseline(x, linear_y, config)
        assert np.all(baseline == 7.5)

    def test_no_points_returns_zeros(self, x, linear_y):
        config = {"mode": "Linear", "attached": False, "points": [[], []]}
        assert np.array_equal(eval_baseline(x, linear_y, config), np.zeros_like(x))

    def test_attached_reads_from_data_not_stored_points(self, x, linear_y):
        """attached=True ignores points[1] and instead samples (smoothed) y
        at the point x-locations."""
        config = {
            "mode": "Linear", "attached": True, "sigma": 0,
            "points": [[0.0, 100.0], [-999.0, -999.0]],  # bogus y, must be ignored
        }
        baseline = eval_baseline(x, linear_y, config)
        np.testing.assert_allclose(baseline, linear_y, atol=1e-8)


class TestEvalBaselinePolynomial:
    def test_quadratic_recovered_with_order_2(self, x):
        y = 0.5 * x ** 2 - 3 * x + 1
        pts_x = [0.0, 50.0, 100.0]
        pts_y = [0.5 * px ** 2 - 3 * px + 1 for px in pts_x]
        config = {"mode": "Polynomial", "attached": False, "order_max": 2,
                   "points": [pts_x, pts_y]}
        baseline = eval_baseline(x, y, config)
        np.testing.assert_allclose(baseline, y, atol=1e-6)

    def test_order_capped_by_number_of_points(self, x, linear_y):
        # order_max=5 but only 2 points -> falls back to order=1 (a line)
        config = {"mode": "Polynomial", "attached": False, "order_max": 5,
                   "points": [[0.0, 100.0], [3.0, 203.0]]}
        baseline = eval_baseline(x, linear_y, config)
        np.testing.assert_allclose(baseline, linear_y, atol=1e-6)


class TestEvalBaselineBatchMatchesPerSpectrumLoop:
    """eval_baseline_batch has a fully-vectorized fast path for Linear and
    Polynomial; it must match calling eval_baseline() row-by-row."""

    @pytest.mark.parametrize("mode", ["Linear", "Polynomial"])
    @pytest.mark.parametrize("attached", [True, False])
    def test_batch_matches_loop(self, x, mode, attached):
        rng = np.random.default_rng(0)
        N = 6
        Y = rng.normal(0, 1, size=(N, len(x))) + (2.0 * x + 5.0)[None, :]

        if attached:
            config = {"mode": mode, "attached": True, "sigma": 3,
                       "points": [[10.0, 50.0, 90.0], [0, 0, 0]], "order_max": 2}
        else:
            config = {"mode": mode, "attached": False,
                       "points": [[10.0, 50.0, 90.0], [25.0, 105.0, 185.0]],
                       "order_max": 2}

        batch_result = eval_baseline_batch(x, Y, config)
        loop_result = np.stack([eval_baseline(x, Y[i], config) for i in range(N)])

        np.testing.assert_allclose(batch_result, loop_result, rtol=1e-4, atol=1e-4)

    def test_no_mode_returns_zeros_like_Y(self, x):
        Y = np.ones((4, len(x)))
        result = eval_baseline_batch(x, Y, {})
        assert result.shape == Y.shape
        assert np.array_equal(result, np.zeros_like(Y))

    def test_no_points_returns_zeros(self, x):
        Y = np.ones((4, len(x)))
        config = {"mode": "Linear", "attached": False, "points": [[], []]}
        result = eval_baseline_batch(x, Y, config)
        assert np.array_equal(result, np.zeros_like(Y))
