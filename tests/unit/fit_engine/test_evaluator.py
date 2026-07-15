"""Unit tests for fit_engine/evaluator.py - VBFevaluator.

VBFevaluator maps a fit_model dict (peak_models with bounds/vary/expr hints)
to the batched tensor API. This covers parsing, bounds/vary/expr handling,
evaluate/jacobian, p0 building, the noise-threshold mechanism, and the R²
formula in build_results_batch.
"""
import numpy as np
import pytest

from spectroview.fit_engine.evaluator import VBFevaluator, eval_peak_initial


def _lorentzian_peak(value=100.0, x0=500.0, fwhm=6.0, vary=True, expr=None,
                      x0_expr=None):
    return {
        "Lorentzian": {
            "ampli": {"value": value, "min": 0.0, "max": 1e6, "vary": vary, "expr": expr},
            "fwhm": {"value": fwhm, "min": 1e-3, "max": 100.0, "vary": True, "expr": None},
            "x0": {"value": x0, "min": x0 - 20, "max": x0 + 20, "vary": True, "expr": x0_expr},
        }
    }


class TestFromFitModel:
    def test_single_peak_param_count_and_names(self):
        fit_model = {"peak_models": {"0": _lorentzian_peak()}}
        ev = VBFevaluator.from_fit_model(fit_model)
        assert ev.n_params_total == 3
        assert ev.n_params_free == 3
        assert ev._param_names == ["P1_ampli", "P1_fwhm", "P1_x0"]

    def test_multi_peak_naming_is_peak_major(self):
        fit_model = {"peak_models": {
            "0": _lorentzian_peak(x0=400.0),
            "1": _lorentzian_peak(x0=450.0),
        }}
        ev = VBFevaluator.from_fit_model(fit_model)
        assert ev._param_names == [
            "P1_ampli", "P1_fwhm", "P1_x0",
            "P2_ampli", "P2_fwhm", "P2_x0",
        ]

    def test_peak_keys_sorted_numerically_not_lexically(self):
        # "10" must sort after "2" numerically, not before it lexically.
        fit_model = {"peak_models": {
            "10": _lorentzian_peak(x0=100.0),
            "2": _lorentzian_peak(x0=200.0),
        }}
        ev = VBFevaluator.from_fit_model(fit_model)
        assert ev.initial_params[2] == 200.0  # P1 (was key "2") x0
        assert ev.initial_params[5] == 100.0  # P2 (was key "10") x0

    def test_missing_peak_models_gives_zero_params(self):
        ev = VBFevaluator.from_fit_model({})
        assert ev.n_params_total == 0
        assert ev.n_params_free == 0

    def test_unknown_model_name_raises(self):
        fit_model = {"peak_models": {"0": {"NotAShape": {"ampli": {"value": 1.0}}}}}
        with pytest.raises(ValueError, match="Unknown peak model"):
            VBFevaluator.from_fit_model(fit_model)

    def test_none_min_max_normalized_to_inf(self):
        peak = _lorentzian_peak()
        peak["Lorentzian"]["ampli"]["min"] = None
        peak["Lorentzian"]["ampli"]["max"] = None
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak}})
        assert ev.lower_bounds[0] == -np.inf
        assert ev.upper_bounds[0] == np.inf

    def test_initial_value_clamped_to_bounds(self):
        peak = _lorentzian_peak(value=-5.0)  # below min=0.0
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak}})
        assert ev.initial_params[0] == 0.0


class TestVaryAndFixedParams:
    def test_vary_false_removed_from_free_params(self):
        peak = _lorentzian_peak(vary=False)
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak}})
        assert ev.n_params_free == 2  # fwhm, x0 still free; ampli fixed
        assert "P1_ampli" not in [ev._param_names[i] for i in ev._free_idx]

    def test_fixed_param_value_preserved_through_to_full(self):
        peak = _lorentzian_peak(value=42.0, vary=False)
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak}})
        p_free = ev.initial_params
        p_full = ev._to_full(p_free)
        assert p_full[0] == 42.0  # P1_ampli, fixed at 42


class TestExpressions:
    def test_expr_forces_vary_false(self):
        peak = _lorentzian_peak(vary=True, x0_expr="500.0")
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak}})
        assert ev.n_params_free == 2  # x0 excluded despite vary=True

    def test_expr_referencing_another_param(self):
        """P2's x0 always equals P1's x0 + 50 (a common 'linked peaks' pattern)."""
        peak1 = _lorentzian_peak(x0=400.0)
        peak2 = _lorentzian_peak(x0=999.0, x0_expr="P1_x0 + 50")
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak1, "1": peak2}})

        p_free = ev.initial_params
        p_full = ev._to_full(p_free)
        idx_p1_x0 = ev._param_names.index("P1_x0")
        idx_p2_x0 = ev._param_names.index("P2_x0")
        assert p_full[idx_p2_x0] == pytest.approx(p_full[idx_p1_x0] + 50)

    def test_expr_updates_when_free_param_changes(self):
        peak1 = _lorentzian_peak(x0=400.0)
        peak2 = _lorentzian_peak(x0=999.0, x0_expr="P1_x0 + 50")
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak1, "1": peak2}})

        p_free = ev.initial_params.copy()
        idx_free_p1_x0 = np.searchsorted(ev._free_idx, ev._param_names.index("P1_x0"))
        p_free[idx_free_p1_x0] = 350.0
        p_full = ev._to_full(p_free)
        idx_p2_x0 = ev._param_names.index("P2_x0")
        assert p_full[idx_p2_x0] == pytest.approx(400.0)

    def test_unresolvable_expr_does_not_raise(self):
        peak = _lorentzian_peak(x0_expr="totally_unknown_name * 2")
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak}})
        p_full = ev._to_full(ev.initial_params)
        assert np.isfinite(p_full).all()  # silently keeps last value, no crash

    def test_null_string_expr_is_treated_as_no_expr(self):
        peak = _lorentzian_peak(x0_expr="None")
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak}})
        assert ev.n_params_free == 3  # x0 still free: "none" (any case) == no expr


class TestEvaluateAndJacobian:
    def test_evaluate_matches_sum_of_peaks(self):
        x = np.linspace(300, 700, 100)
        peak1 = _lorentzian_peak(value=100.0, x0=450.0)
        peak2 = _lorentzian_peak(value=50.0, x0=550.0)
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak1, "1": peak2}})

        Y = ev.evaluate(x, ev.initial_params[None, :])
        assert Y.shape == (1, len(x))
        assert np.max(Y) > 0

    def test_jacobian_shape_matches_free_params(self):
        x = np.linspace(300, 700, 50)
        peak = _lorentzian_peak()
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak}})
        J = ev.jacobian(x, ev.initial_params[None, :])
        assert J.shape == (1, len(x), ev.n_params_free)

    def test_jacobian_excludes_fixed_columns(self):
        x = np.linspace(300, 700, 50)
        peak = _lorentzian_peak(vary=False)
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak}})
        J = ev.jacobian(x, ev.initial_params[None, :])
        assert J.shape[-1] == 2  # ampli fixed -> only fwhm, x0 columns


class TestBuildP0Matrix:
    def test_amplitude_rescaled_toward_data_peak_height(self):
        x = np.linspace(300, 700, 400)
        true_ampli = 250.0
        y_true = true_ampli / (1 + 4 * ((x - 500.0) / 6.0) ** 2)
        Y = np.tile(y_true, (3, 1))

        # Seed amplitude within the rescale guard's 0.01-100x ratio window
        # (data_amp/model_amp); a too-wrong seed is deliberately left alone.
        peak = _lorentzian_peak(value=50.0, x0=500.0)
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak}})
        p0 = ev.build_p0_matrix(x, Y)

        ampli_free_idx = np.searchsorted(ev._free_idx, ev._param_names.index("P1_ampli"))
        np.testing.assert_allclose(p0[:, ampli_free_idx], true_ampli, rtol=0.05)

    def test_amplitude_not_rescaled_when_ratio_out_of_range(self):
        x = np.linspace(300, 700, 400)
        Y = np.tile(np.full_like(x, 0.001), (2, 1))  # essentially flat/near-zero data

        peak = _lorentzian_peak(value=100.0, x0=500.0)
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak}})
        p0 = ev.build_p0_matrix(x, Y)

        ampli_free_idx = np.searchsorted(ev._free_idx, ev._param_names.index("P1_ampli"))
        # data_amp/model_amp ~ 0.00001/100 << 0.01 -> guard skips rescaling
        np.testing.assert_allclose(p0[:, ampli_free_idx], 100.0)

    def test_x0_outside_axis_range_is_skipped(self):
        x = np.linspace(300, 700, 100)
        Y = np.ones((2, len(x))) * 50.0
        peak = _lorentzian_peak(value=10.0, x0=900.0)  # x0 outside [300,700]
        peak["Lorentzian"]["x0"]["min"] = 0.0
        peak["Lorentzian"]["x0"]["max"] = 2000.0
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak}})
        p0 = ev.build_p0_matrix(x, Y)
        ampli_free_idx = np.searchsorted(ev._free_idx, ev._param_names.index("P1_ampli"))
        np.testing.assert_allclose(p0[:, ampli_free_idx], 10.0)  # unchanged


class TestNoiseThreshold:
    def test_noop_when_coef_noise_zero(self):
        x = np.linspace(300, 700, 100)
        peak = _lorentzian_peak()
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak}})
        p = np.tile(ev.initial_params, (2, 1))
        p_before = p.copy()
        ev.apply_noise_threshold(x, np.zeros((2, len(x))), p, {"coef_noise": 0})
        np.testing.assert_array_equal(p, p_before)

    def test_noop_when_fit_params_none(self):
        x = np.linspace(300, 700, 100)
        peak = _lorentzian_peak()
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak}})
        p = np.tile(ev.initial_params, (2, 1))
        p_before = p.copy()
        ev.apply_noise_threshold(x, np.zeros((2, len(x))), p, None)
        np.testing.assert_array_equal(p, p_before)

    def test_suppresses_peak_in_flat_noisy_region(self):
        x = np.linspace(300, 700, 200)
        # A peak whose x0 sits in a pure-noise region with no real signal.
        # (A literally-flat/all-zero Y gives noise_level==0 too -- since the
        # threshold is derived FROM the data's own point-to-point scatter --
        # so suppression needs actual noise, not just a flat baseline.)
        peak = _lorentzian_peak(value=100.0, x0=500.0)
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak}})
        rng = np.random.default_rng(7)
        Y = rng.normal(0.0, 1.0, size=(1, len(x)))

        p = np.tile(ev.initial_params, (1, 1))
        ev.apply_noise_threshold(x, Y, p, {"coef_noise": 1.0})

        ampli_free_idx = np.searchsorted(ev._free_idx, ev._param_names.index("P1_ampli"))
        fwhm_free_idx = np.searchsorted(ev._free_idx, ev._param_names.index("P1_fwhm"))
        assert p[0, ampli_free_idx] == 0.0
        assert p[0, fwhm_free_idx] == 0.0

    def test_does_not_suppress_peak_above_noise_floor(self):
        x = np.linspace(300, 700, 200)
        peak = _lorentzian_peak(value=1000.0, x0=500.0)
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak}})
        # Strong real signal at x0, tiny noise elsewhere.
        Y = ev.evaluate(x, ev.initial_params[None, :]) + 1e-6

        p = np.tile(ev.initial_params, (1, 1))
        ev.apply_noise_threshold(x, Y, p, {"coef_noise": 1.0})
        ampli_free_idx = np.searchsorted(ev._free_idx, ev._param_names.index("P1_ampli"))
        assert p[0, ampli_free_idx] != 0.0

    def test_shared_noise_stats_matches_recomputed(self):
        x = np.linspace(300, 700, 150)
        rng = np.random.default_rng(1)
        Y = rng.normal(0, 1.0, size=(3, len(x)))
        peak = _lorentzian_peak(value=5.0, x0=500.0)
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak}})

        p_a = np.tile(ev.initial_params, (3, 1))
        p_b = p_a.copy()
        stats = ev.compute_noise_stats(Y, 1.0)

        ev.apply_noise_threshold(x, Y, p_a, {"coef_noise": 1.0})  # recomputes internally
        ev.apply_noise_threshold(x, Y, p_b, {"coef_noise": 1.0}, noise_stats=stats)
        np.testing.assert_array_equal(p_a, p_b)


class TestComputeNoiseStats:
    def test_2d_matches_1d_per_row_in_the_interior(self):
        """The 2D (batched) path edge-pads with 'edge' replication while the
        1D path's np.convolve(..., mode='same') implicitly zero-pads, so the
        two only need to agree away from the first/last 2 samples."""
        rng = np.random.default_rng(2)
        Y = rng.normal(0, 2.0, size=(4, 100))
        ymean_2d, noise_2d = VBFevaluator.compute_noise_stats(Y, coef_noise=1.5)
        for i in range(4):
            ymean_1d, noise_1d = VBFevaluator.compute_noise_stats(Y[i], coef_noise=1.5)
            np.testing.assert_allclose(ymean_2d[i][2:-2], ymean_1d[2:-2])
            np.testing.assert_allclose(noise_2d[i], noise_1d)


class TestBuildResultsBatch:
    def test_r_squared_perfect_fit_is_one(self):
        x = np.linspace(300, 700, 100)
        peak = _lorentzian_peak(value=80.0, x0=500.0)
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak}})
        p_opt = ev.initial_params[None, :]
        Y_data = ev.evaluate(x, p_opt)  # exact match

        p_full, success, r2, best_fits, y_peaks = ev.build_results_batch(
            p_opt, x, Y_data, np.array([True]), None, shared_x=True,
        )
        assert r2[0] == pytest.approx(1.0, abs=1e-9)
        assert len(y_peaks) == 1

    def test_r_squared_manual_formula_matches(self):
        x = np.linspace(300, 700, 50)
        peak = _lorentzian_peak(value=80.0, x0=500.0)
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak}})
        p_opt = ev.initial_params[None, :]

        rng = np.random.default_rng(3)
        Y_data = ev.evaluate(x, p_opt) + rng.normal(0, 5.0, size=(1, len(x)))

        p_full, success, r2, best_fits, y_peaks = ev.build_results_batch(
            p_opt, x, Y_data, np.array([True]), None, shared_x=True,
        )
        ss_res = np.sum((Y_data[0] - best_fits[0]) ** 2)
        ss_tot = np.sum((Y_data[0] - Y_data[0].mean()) ** 2)
        expected_r2 = 1.0 - ss_res / ss_tot
        assert r2[0] == pytest.approx(expected_r2, rel=1e-10)

    def test_r_squared_zero_variance_data_is_zero(self):
        x = np.linspace(300, 700, 50)
        peak = _lorentzian_peak(value=0.0, vary=False, x0=500.0)
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak}})
        p_opt = ev.initial_params[None, :]
        Y_data = np.zeros((1, len(x)))  # constant -> ss_tot == 0

        p_full, success, r2, best_fits, y_peaks = ev.build_results_batch(
            p_opt, x, Y_data, np.array([True]), None, shared_x=True,
        )
        assert r2[0] == 0.0

    def test_weighted_r_squared_ignores_zero_weight_points(self):
        x = np.linspace(300, 700, 50)
        peak = _lorentzian_peak(value=80.0, x0=500.0)
        ev = VBFevaluator.from_fit_model({"peak_models": {"0": peak}})
        p_opt = ev.initial_params[None, :]
        Y_data = ev.evaluate(x, p_opt).copy()
        Y_data[0, :5] += 1e6  # corrupt a few points

        weights = np.ones((1, len(x)))
        weights[0, :5] = 0.0

        p_full, success, r2, best_fits, y_peaks = ev.build_results_batch(
            p_opt, x, Y_data, np.array([True]), weights, shared_x=True,
        )
        assert r2[0] == pytest.approx(1.0, abs=1e-6)  # corrupted points masked out


class TestEvalPeakInitial:
    def test_matches_scalar_registry(self):
        x = np.linspace(300, 700, 401)  # step=1.0, so x0=500.0 lands exactly on-grid
        p_model = {"Lorentzian": {"ampli": {"value": 50.0}, "fwhm": {"value": 8.0},
                                    "x0": {"value": 500.0}}}
        y = eval_peak_initial(x, p_model)
        assert y.shape == x.shape
        assert y.max() == pytest.approx(50.0, rel=1e-3)

    def test_unknown_shape_returns_zeros(self):
        x = np.linspace(300, 700, 20)
        y = eval_peak_initial(x, {"NotAShape": {}})
        assert np.array_equal(y, np.zeros_like(x))
