"""Unit tests for fit_engine/optimizer.py - batched_levenberg_marquardt.

Uses small synthetic evaluate_fn/jacobian_fn closures (not real peak models)
so the optimizer's convergence/bounds/caching behavior can be tested in
isolation from the rest of the fit_engine stack.
"""
import numpy as np
import pytest

from spectroview.fit_engine.optimizer import batched_levenberg_marquardt


def _linear_problem():
    """y = m*x + b, perfectly linear -> LM should converge in a handful of
    iterations to the exact least-squares solution."""
    x = np.linspace(0, 10, 50)

    def evaluate_fn(x, p):
        m = p[:, 0:1]
        b = p[:, 1:2]
        return m * x[None, :] + b

    def jacobian_fn(x, p):
        N, M = p.shape[0], x.shape[-1]
        J = np.empty((N, M, 2))
        J[:, :, 0] = x[None, :]
        J[:, :, 1] = 1.0
        return J

    return x, evaluate_fn, jacobian_fn


class TestConvergenceOnExactData:
    def test_recovers_exact_linear_params(self):
        x, evaluate_fn, jacobian_fn = _linear_problem()
        true_p = np.array([[3.0, -2.0]])
        Y = evaluate_fn(x, true_p)

        p0 = np.array([[1.0, 1.0]])
        p_opt, success, cost = batched_levenberg_marquardt(
            x=x, Y_data=Y, evaluate_fn=evaluate_fn, jacobian_fn=jacobian_fn,
            p0=p0, lower_bounds=np.array([-100.0, -100.0]),
            upper_bounds=np.array([100.0, 100.0]),
        )
        assert success[0]
        np.testing.assert_allclose(p_opt[0], true_p[0], atol=1e-4)
        assert cost[0] < 1e-8

    def test_batched_n_spectra_all_converge(self):
        x, evaluate_fn, jacobian_fn = _linear_problem()
        rng = np.random.default_rng(0)
        N = 20
        true_p = np.column_stack([
            rng.uniform(1, 5, N), rng.uniform(-5, 5, N),
        ])
        Y = evaluate_fn(x, true_p)
        p0 = np.tile([1.0, 1.0], (N, 1))

        p_opt, success, cost = batched_levenberg_marquardt(
            x=x, Y_data=Y, evaluate_fn=evaluate_fn, jacobian_fn=jacobian_fn,
            p0=p0, lower_bounds=np.array([-100.0, -100.0]),
            upper_bounds=np.array([100.0, 100.0]),
        )
        assert success.all()
        np.testing.assert_allclose(p_opt, true_p, atol=1e-3)

    def test_deterministic_across_repeated_runs(self):
        x, evaluate_fn, jacobian_fn = _linear_problem()
        true_p = np.array([[3.0, -2.0]])
        Y = evaluate_fn(x, true_p)
        p0 = np.array([[0.5, 0.5]])
        kwargs = dict(x=x, Y_data=Y, evaluate_fn=evaluate_fn, jacobian_fn=jacobian_fn,
                       p0=p0, lower_bounds=np.array([-100.0, -100.0]),
                       upper_bounds=np.array([100.0, 100.0]))

        p1, s1, c1 = batched_levenberg_marquardt(**kwargs)
        p2, s2, c2 = batched_levenberg_marquardt(**kwargs)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(c1, c2)


class TestBounds:
    def test_solution_respects_tight_bounds(self):
        x, evaluate_fn, jacobian_fn = _linear_problem()
        true_p = np.array([[3.0, -2.0]])
        Y = evaluate_fn(x, true_p)
        p0 = np.array([[1.0, 1.0]])

        # Force slope to stay below the true optimum.
        p_opt, success, cost = batched_levenberg_marquardt(
            x=x, Y_data=Y, evaluate_fn=evaluate_fn, jacobian_fn=jacobian_fn,
            p0=p0, lower_bounds=np.array([-100.0, -100.0]),
            upper_bounds=np.array([2.0, 100.0]),
        )
        assert p_opt[0, 0] <= 2.0 + 1e-9

    def test_initial_guess_outside_bounds_is_clipped(self):
        x, evaluate_fn, jacobian_fn = _linear_problem()
        true_p = np.array([[3.0, -2.0]])
        Y = evaluate_fn(x, true_p)
        # p0 starts far outside bounds; optimizer must clip before evaluating.
        p0 = np.array([[1000.0, 1000.0]])
        p_opt, success, cost = batched_levenberg_marquardt(
            x=x, Y_data=Y, evaluate_fn=evaluate_fn, jacobian_fn=jacobian_fn,
            p0=p0, lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
        )
        assert np.isfinite(p_opt).all()
        assert (p_opt >= -5.0).all() and (p_opt <= 5.0).all()


class TestWeights:
    def test_zero_weight_row_marked_converged_without_moving(self):
        x, evaluate_fn, jacobian_fn = _linear_problem()
        Y = np.zeros((2, len(x)))
        Y[0] = evaluate_fn(x, np.array([[3.0, -2.0]]))[0]
        weights = np.ones((2, len(x)))
        weights[1] = 0.0  # second spectrum entirely masked out
        p0 = np.tile([0.0, 0.0], (2, 1))

        p_opt, success, cost = batched_levenberg_marquardt(
            x=x, Y_data=Y, evaluate_fn=evaluate_fn, jacobian_fn=jacobian_fn,
            p0=p0, lower_bounds=np.array([-100.0, -100.0]),
            upper_bounds=np.array([100.0, 100.0]), weights=weights,
        )
        assert success[1]  # zero-weight rows are marked converged immediately
        np.testing.assert_array_equal(p_opt[1], p0[1])  # never moved

    def test_weighted_points_excluded_from_fit(self):
        x, evaluate_fn, jacobian_fn = _linear_problem()
        true_p = np.array([[3.0, -2.0]])
        Y = evaluate_fn(x, true_p)
        Y_corrupted = Y.copy()
        Y_corrupted[0, :5] += 1000.0  # corrupt first 5 points badly

        weights = np.ones((1, len(x)))
        weights[0, :5] = 0.0  # mask the corrupted points out
        p0 = np.array([[0.0, 0.0]])

        p_opt, success, cost = batched_levenberg_marquardt(
            x=x, Y_data=Y_corrupted, evaluate_fn=evaluate_fn, jacobian_fn=jacobian_fn,
            p0=p0, lower_bounds=np.array([-100.0, -100.0]),
            upper_bounds=np.array([100.0, 100.0]), weights=weights,
        )
        np.testing.assert_allclose(p_opt[0], true_p[0], atol=1e-3)

    def test_wrong_shape_weights_raises(self):
        x, evaluate_fn, jacobian_fn = _linear_problem()
        Y = evaluate_fn(x, np.array([[1.0, 1.0]]))
        bad_weights = np.ones((3, len(x) + 1))
        with pytest.raises(ValueError):
            batched_levenberg_marquardt(
                x=x, Y_data=Y, evaluate_fn=evaluate_fn, jacobian_fn=jacobian_fn,
                p0=np.array([[1.0, 1.0]]), lower_bounds=np.array([-1.0, -1.0]),
                upper_bounds=np.array([1.0, 1.0]), weights=bad_weights,
            )


class TestStuckSpectraGiveUp:
    def test_non_improving_function_still_terminates_and_returns_finite(self):
        """A function whose Jacobian is always zero can never improve; the
        optimizer must still terminate (via MAX_REJECTS) rather than loop
        forever or crash, and must return the untouched initial guess."""
        x = np.linspace(0, 10, 20)

        def evaluate_fn(x, p):
            return np.tile(p[:, 0:1], (1, len(x)))

        def jacobian_fn(x, p):
            return np.zeros((p.shape[0], len(x), 1))

        Y = np.ones((1, len(x))) * 999.0  # unreachable target given zero Jacobian
        p0 = np.array([[0.0]])

        p_opt, success, cost = batched_levenberg_marquardt(
            x=x, Y_data=Y, evaluate_fn=evaluate_fn, jacobian_fn=jacobian_fn,
            p0=p0, lower_bounds=np.array([-10.0]), upper_bounds=np.array([10.0]),
            max_iter=200,
        )
        assert success[0]  # gives up after MAX_REJECTS, marked converged (stuck)
        assert np.isfinite(p_opt).all()

    def test_nan_jacobian_is_sanitized_not_propagated(self):
        x = np.linspace(0, 10, 20)

        def evaluate_fn(x, p):
            m = p[:, 0:1]
            return m * x[None, :]

        def jacobian_fn(x, p):
            J = np.empty((p.shape[0], len(x), 1))
            J[:, :, 0] = np.nan
            return J

        Y = 2.0 * x[None, :]
        p0 = np.array([[1.0]])
        p_opt, success, cost = batched_levenberg_marquardt(
            x=x, Y_data=Y, evaluate_fn=evaluate_fn, jacobian_fn=jacobian_fn,
            p0=p0, lower_bounds=np.array([-10.0]), upper_bounds=np.array([10.0]),
            max_iter=50,
        )
        assert np.isfinite(p_opt).all()
        assert np.isfinite(cost).all()


class TestCancellation:
    def test_cancel_check_stops_iteration_early(self):
        x, evaluate_fn, jacobian_fn = _linear_problem()
        true_p = np.array([[3.0, -2.0]])
        Y = evaluate_fn(x, true_p)
        p0 = np.array([[0.0, 0.0]])

        calls = {"n": 0}

        def cancel_after_one():
            calls["n"] += 1
            return calls["n"] > 1

        p_opt, success, cost = batched_levenberg_marquardt(
            x=x, Y_data=Y, evaluate_fn=evaluate_fn, jacobian_fn=jacobian_fn,
            p0=p0, lower_bounds=np.array([-100.0, -100.0]),
            upper_bounds=np.array([100.0, 100.0]), max_iter=200,
            cancel_check=cancel_after_one,
        )
        # Cancelled almost immediately: should not have fully converged to
        # the exact optimum (still far from true_p) yet must return finite.
        assert np.isfinite(p_opt).all()
        assert calls["n"] == 2

    def test_progress_callback_invoked(self):
        x, evaluate_fn, jacobian_fn = _linear_problem()
        true_p = np.array([[3.0, -2.0]])
        Y = evaluate_fn(x, true_p)
        p0 = np.array([[0.0, 0.0]])

        progress_calls = []
        batched_levenberg_marquardt(
            x=x, Y_data=Y, evaluate_fn=evaluate_fn, jacobian_fn=jacobian_fn,
            p0=p0, lower_bounds=np.array([-100.0, -100.0]),
            upper_bounds=np.array([100.0, 100.0]),
            progress_callback=lambda cur, tot: progress_calls.append((cur, tot)),
        )
        assert len(progress_calls) > 0
        assert all(tot == 1 for _, tot in progress_calls)
