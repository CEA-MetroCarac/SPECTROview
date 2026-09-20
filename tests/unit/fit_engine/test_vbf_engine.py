"""Unit tests for fit_engine/vbf_engine.py - VBFengine.fit_spectra().

Exercises the full evaluator + optimizer pipeline end-to-end on synthetic
(known ground-truth) spectra: single-peak, multi-peak, several shapes,
bounds/expr constraints, and the all-fixed-params early-exit branch.
"""
import numpy as np
import pytest

from spectroview.fit_engine.vbf_engine import VBFengine


class TestSinglePeakRecovery:
    def test_recovers_lorentzian_params_from_noise_free_data(self, make_fit_model, make_synthetic_spectrum):
        true_peak = ("Lorentzian", {"x0": 500.0, "ampli": 150.0, "fwhm": 6.0})
        fit_model = make_fit_model([true_peak])
        x, y = make_synthetic_spectrum([(true_peak[0], fit_model["peak_models"]["0"]["Lorentzian"])])

        # Perturb the initial guess so the optimizer has real work to do.
        seed_model = make_fit_model([("Lorentzian", {"x0": 495.0, "ampli": 100.0, "fwhm": 4.0})])

        engine = VBFengine()
        p_full, success, r2, best_fits, y_peaks, names = engine.fit_spectra(
            x=x, Y=y[None, :], fit_model=seed_model,
        )
        assert success[0]
        assert r2[0] > 0.999
        result = dict(zip(names, p_full[0]))
        assert result["P1_x0"] == pytest.approx(500.0, abs=0.05)
        assert result["P1_fwhm"] == pytest.approx(6.0, abs=0.1)
        assert result["P1_ampli"] == pytest.approx(150.0, rel=0.02)

    @pytest.mark.parametrize("shape", ["Lorentzian", "Gaussian", "PseudoVoigt", "Fano"])
    def test_recovers_params_across_shapes(self, shape, make_fit_model, make_synthetic_spectrum):
        true_kwargs = {"x0": 500.0, "ampli": 100.0, "fwhm": 8.0}
        true_peak = (shape, true_kwargs)
        fit_model = make_fit_model([true_peak])
        x, y = make_synthetic_spectrum([(shape, fit_model["peak_models"]["0"][shape])])

        seed_kwargs = {"x0": 494.0, "ampli": 70.0, "fwhm": 6.0}
        seed_model = make_fit_model([(shape, seed_kwargs)])

        engine = VBFengine()
        p_full, success, r2, best_fits, y_peaks, names = engine.fit_spectra(x=x, Y=y[None, :], fit_model=seed_model)
        assert success[0]
        assert r2[0] > 0.999
        result = dict(zip(names, p_full[0]))
        assert result["P1_x0"] == pytest.approx(500.0, abs=0.1)


class TestMultiPeakRecovery:
    def test_three_overlapping_lorentzians(self, make_fit_model, make_synthetic_spectrum):
        true_peaks = [
            ("Lorentzian", {"x0": 480.0, "ampli": 100.0, "fwhm": 6.0}),
            ("Lorentzian", {"x0": 500.0, "ampli": 80.0, "fwhm": 5.0}),
            ("Lorentzian", {"x0": 525.0, "ampli": 60.0, "fwhm": 7.0}),
        ]
        fit_model = make_fit_model(true_peaks)
        peaks_for_data = [("Lorentzian", fit_model["peak_models"][str(i)]["Lorentzian"])
                           for i in range(3)]
        x, y = make_synthetic_spectrum(peaks_for_data)

        seed_peaks = [
            ("Lorentzian", {"x0": 478.0, "ampli": 90.0, "fwhm": 5.0}),
            ("Lorentzian", {"x0": 502.0, "ampli": 70.0, "fwhm": 4.0}),
            ("Lorentzian", {"x0": 523.0, "ampli": 50.0, "fwhm": 6.0}),
        ]
        seed_model = make_fit_model(seed_peaks)

        engine = VBFengine()
        p_full, success, r2, best_fits, y_peaks, names = engine.fit_spectra(x=x, Y=y[None, :], fit_model=seed_model)
        assert success[0]
        assert r2[0] > 0.999
        assert len(y_peaks) == 3
        result = dict(zip(names, p_full[0]))
        for i, (_, kwargs) in enumerate(true_peaks, start=1):
            assert result[f"P{i}_x0"] == pytest.approx(kwargs["x0"], abs=0.15)


class TestBatchFitting:
    def test_batch_matches_individual_fits(self, make_synthetic_map):
        x, Y, coords, fnames, true_params, canonical = make_synthetic_map(
            shape="Lorentzian", n_spectra=8, noise_std=0.0, seed=1,
        )
        fit_model = {
            "fit_params": {"max_ite": 200, "xtol": 1e-5, "ftol": 1e-5},
            "peak_models": {"0": {"Lorentzian": {
                "ampli": {"value": 90.0, "min": 0.0, "max": 1e5, "vary": True, "expr": None},
                "fwhm": {"value": 7.0, "min": 1e-3, "max": 100.0, "vary": True, "expr": None},
                "x0": {"value": 499.0, "min": 450.0, "max": 550.0, "vary": True, "expr": None},
            }}},
        }
        engine = VBFengine()
        p_batch, success_batch, r2_batch, _, _, names = engine.fit_spectra(x=x, Y=Y, fit_model=fit_model)
        assert success_batch.all()

        for row in range(Y.shape[0]):
            p_row, success_row, r2_row, _, _, _ = engine.fit_spectra(
                x=x, Y=Y[row:row + 1], fit_model=fit_model,
            )
            np.testing.assert_allclose(p_batch[row], p_row[0], atol=1e-4)
            assert r2_row[0] == pytest.approx(r2_batch[row], abs=1e-6)

    def test_batch_recovers_per_row_ground_truth(self, make_synthetic_map):
        x, Y, coords, fnames, true_params, canonical = make_synthetic_map(
            shape="Lorentzian", n_spectra=15, noise_std=0.0, seed=2,
        )
        fit_model = {
            "fit_params": {"max_ite": 200, "xtol": 1e-5, "ftol": 1e-5},
            "peak_models": {"0": {"Lorentzian": {
                "ampli": {"value": 90.0, "min": 0.0, "max": 1e5, "vary": True, "expr": None},
                "fwhm": {"value": 7.0, "min": 1e-3, "max": 100.0, "vary": True, "expr": None},
                "x0": {"value": 499.0, "min": 450.0, "max": 550.0, "vary": True, "expr": None},
            }}},
        }
        engine = VBFengine()
        p_full, success, r2, best_fits, y_peaks, names = engine.fit_spectra(x=x, Y=Y, fit_model=fit_model)
        assert success.all()
        idx_ampli = names.index("P1_ampli")
        idx_x0 = names.index("P1_x0")
        idx_fwhm = names.index("P1_fwhm")
        col = {"ampli": 0, "fwhm": 1, "x0": 2}
        np.testing.assert_allclose(p_full[:, idx_ampli], true_params[:, col["ampli"]], rtol=0.02)
        np.testing.assert_allclose(p_full[:, idx_x0], true_params[:, col["x0"]], atol=0.1)
        np.testing.assert_allclose(p_full[:, idx_fwhm], true_params[:, col["fwhm"]], rtol=0.05)


class TestBoundsAndConstraints:
    def test_fixed_parameter_stays_at_seed_value(self, make_fit_model, make_synthetic_spectrum):
        true_peak = ("Lorentzian", {"x0": 500.0, "ampli": 100.0, "fwhm": 6.0})
        fit_model = make_fit_model([true_peak])
        x, y = make_synthetic_spectrum([(true_peak[0], fit_model["peak_models"]["0"]["Lorentzian"])])

        seed_model = make_fit_model([("Lorentzian", {"x0": 500.0, "ampli": 100.0, "fwhm": 6.0})])
        seed_model["peak_models"]["0"]["Lorentzian"]["fwhm"]["vary"] = False
        seed_model["peak_models"]["0"]["Lorentzian"]["fwhm"]["value"] = 20.0  # deliberately wrong, fixed

        engine = VBFengine()
        p_full, success, r2, best_fits, y_peaks, names = engine.fit_spectra(x=x, Y=y[None, :], fit_model=seed_model)
        result = dict(zip(names, p_full[0]))
        assert result["P1_fwhm"] == 20.0  # never varied
        assert r2[0] < 0.99  # forced-wrong fwhm prevents a perfect fit

    def test_bounds_clamp_result(self, make_fit_model, make_synthetic_spectrum):
        true_peak = ("Lorentzian", {"x0": 500.0, "ampli": 100.0, "fwhm": 6.0})
        fit_model = make_fit_model([true_peak])
        x, y = make_synthetic_spectrum([(true_peak[0], fit_model["peak_models"]["0"]["Lorentzian"])])

        seed_model = make_fit_model([("Lorentzian", {"x0": 500.0, "ampli": 100.0, "fwhm": 6.0})])
        seed_model["peak_models"]["0"]["Lorentzian"]["x0"]["min"] = 480.0
        seed_model["peak_models"]["0"]["Lorentzian"]["x0"]["max"] = 495.0  # excludes true x0=500

        engine = VBFengine()
        p_full, success, r2, best_fits, y_peaks, names = engine.fit_spectra(x=x, Y=y[None, :], fit_model=seed_model)
        result = dict(zip(names, p_full[0]))
        assert result["P1_x0"] <= 495.0 + 1e-9

    def test_expr_linked_peaks_stay_linked_after_fit(self, make_fit_model, make_synthetic_spectrum):
        p1_kwargs = {"x0": 480.0, "ampli": 100.0, "fwhm": 5.0}
        p2_kwargs = {"x0": 520.0, "ampli": 60.0, "fwhm": 5.0}
        fit_model = make_fit_model([("Lorentzian", p1_kwargs), ("Lorentzian", p2_kwargs)])
        peaks_for_data = [("Lorentzian", fit_model["peak_models"]["0"]["Lorentzian"]),
                           ("Lorentzian", fit_model["peak_models"]["1"]["Lorentzian"])]
        x, y = make_synthetic_spectrum(peaks_for_data)

        seed_model = make_fit_model([("Lorentzian", {"x0": 480.0, "ampli": 90.0, "fwhm": 5.0}),
                                       ("Lorentzian", {"x0": 520.0, "ampli": 55.0, "fwhm": 5.0})])
        # Force P2's fwhm to always equal P1's fwhm.
        seed_model["peak_models"]["1"]["Lorentzian"]["fwhm"]["expr"] = "P1_fwhm"

        engine = VBFengine()
        p_full, success, r2, best_fits, y_peaks, names = engine.fit_spectra(x=x, Y=y[None, :], fit_model=seed_model)
        result = dict(zip(names, p_full[0]))
        assert result["P1_fwhm"] == pytest.approx(result["P2_fwhm"])


class TestEarlyExitAllFixed:
    def test_all_params_fixed_skips_optimizer_and_reports_success(self, make_fit_model, make_synthetic_spectrum):
        true_peak = ("Lorentzian", {"x0": 500.0, "ampli": 100.0, "fwhm": 6.0})
        fit_model = make_fit_model([true_peak])
        x, y = make_synthetic_spectrum([(true_peak[0], fit_model["peak_models"]["0"]["Lorentzian"])])

        for pname in ("ampli", "fwhm", "x0"):
            fit_model["peak_models"]["0"]["Lorentzian"][pname]["vary"] = False

        engine = VBFengine()
        p_full, success, r2, best_fits, y_peaks, names = engine.fit_spectra(
            x=x, Y=y[None, :], fit_model=fit_model, progress_callback=lambda c, t: None,
        )
        assert success.all()
        assert y_peaks == []  # documented early-exit contract
        np.testing.assert_array_equal(p_full, np.zeros_like(p_full))
        np.testing.assert_array_equal(r2, np.zeros(1))
        np.testing.assert_array_equal(best_fits, np.zeros_like(y[None, :]))


class TestDeterminism:
    def test_repeated_fit_is_bit_identical(self, make_synthetic_map):
        x, Y, coords, fnames, true_params, canonical = make_synthetic_map(
            shape="Lorentzian", n_spectra=10, noise_std=1.0, seed=5,
        )
        fit_model = {
            "fit_params": {"max_ite": 200, "xtol": 1e-4, "ftol": 1e-4, "coef_noise": 1.0},
            "peak_models": {"0": {"Lorentzian": {
                "ampli": {"value": 90.0, "min": 0.0, "max": 1e5, "vary": True, "expr": None},
                "fwhm": {"value": 7.0, "min": 1e-3, "max": 100.0, "vary": True, "expr": None},
                "x0": {"value": 499.0, "min": 450.0, "max": 550.0, "vary": True, "expr": None},
            }}},
        }
        engine1, engine2 = VBFengine(), VBFengine()
        p1, s1, r1, b1, _, _ = engine1.fit_spectra(x=x.copy(), Y=Y.copy(), fit_model=fit_model)
        p2, s2, r2_, b2, _, _ = engine2.fit_spectra(x=x.copy(), Y=Y.copy(), fit_model=fit_model)

        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(r1, r2_)
        np.testing.assert_array_equal(b1, b2)


class TestTimings:
    def test_timings_dict_has_expected_keys(self, make_fit_model, make_synthetic_spectrum):
        true_peak = ("Lorentzian", {"x0": 500.0, "ampli": 100.0, "fwhm": 6.0})
        fit_model = make_fit_model([true_peak])
        x, y = make_synthetic_spectrum([(true_peak[0], fit_model["peak_models"]["0"]["Lorentzian"])])

        engine = VBFengine()
        engine.fit_spectra(x=x, Y=y[None, :], fit_model=fit_model)
        assert set(engine.timings.keys()) == {
            "Step 3 - build p0", "Step 4 - batch fit", "Step 5 - write_back",
        }


class TestCancellation:
    def test_cancel_check_stops_without_crashing(self, make_synthetic_map):
        x, Y, coords, fnames, true_params, canonical = make_synthetic_map(
            shape="Lorentzian", n_spectra=5, noise_std=0.0, seed=9,
        )
        fit_model = {
            "fit_params": {"max_ite": 200},
            "peak_models": {"0": {"Lorentzian": {
                "ampli": {"value": 90.0, "min": 0.0, "max": 1e5, "vary": True, "expr": None},
                "fwhm": {"value": 7.0, "min": 1e-3, "max": 100.0, "vary": True, "expr": None},
                "x0": {"value": 499.0, "min": 450.0, "max": 550.0, "vary": True, "expr": None},
            }}},
        }
        engine = VBFengine()
        p_full, success, r2, best_fits, y_peaks, names = engine.fit_spectra(
            x=x, Y=Y, fit_model=fit_model, cancel_check=lambda: True,
        )
        assert np.isfinite(p_full).all()


class TestRobustLoss:
    def test_spikes_recovered_better_under_soft_l1_than_linear(self, make_fit_model, make_synthetic_spectrum):
        true_peak = ("Lorentzian", {"x0": 500.0, "ampli": 100.0, "fwhm": 6.0})
        fit_model = make_fit_model([true_peak])
        x, y = make_synthetic_spectrum([(true_peak[0], fit_model["peak_models"]["0"]["Lorentzian"])])

        # Inject spike outliers into the spectrum
        y_spikes = y.copy()
        y_spikes[10] += 300.0
        y_spikes[80] += 500.0
        y_spikes[len(x) - 15] += 400.0

        seed_model_lin = make_fit_model(
            [("Lorentzian", {"x0": 496.0, "ampli": 80.0, "fwhm": 5.0})],
            fit_params={"loss": "linear", "max_ite": 200, "xtol": 1e-4, "ftol": 1e-4},
        )
        seed_model_soft = make_fit_model(
            [("Lorentzian", {"x0": 496.0, "ampli": 80.0, "fwhm": 5.0})],
            fit_params={"loss": "soft_l1", "f_scale": 5.0, "max_ite": 200, "xtol": 1e-4, "ftol": 1e-4},
        )

        engine = VBFengine()
        p_lin, s_lin, r2_lin, _, _, names = engine.fit_spectra(x=x, Y=y_spikes[None, :], fit_model=seed_model_lin)
        p_soft, s_soft, r2_soft, _, _, _ = engine.fit_spectra(x=x, Y=y_spikes[None, :], fit_model=seed_model_soft)

        assert s_soft[0]
        res_lin = dict(zip(names, p_lin[0]))
        res_soft = dict(zip(names, p_soft[0]))

        # Under soft_l1, the recovered peak position should be closer to truth (500.0)
        err_x0_lin = abs(res_lin["P1_x0"] - 500.0)
        err_x0_soft = abs(res_soft["P1_x0"] - 500.0)
        assert err_x0_soft < err_x0_lin

    def test_spikes_recovered_better_under_huber_than_linear(self, make_fit_model, make_synthetic_spectrum):
        true_peak = ("Lorentzian", {"x0": 500.0, "ampli": 100.0, "fwhm": 6.0})
        fit_model = make_fit_model([true_peak])
        x, y = make_synthetic_spectrum([(true_peak[0], fit_model["peak_models"]["0"]["Lorentzian"])])

        y_spikes = y.copy()
        y_spikes[15] += 400.0
        y_spikes[85] += 600.0

        seed_model_lin = make_fit_model(
            [("Lorentzian", {"x0": 495.0, "ampli": 80.0, "fwhm": 5.0})],
            fit_params={"loss": "linear", "max_ite": 200, "xtol": 1e-4, "ftol": 1e-4},
        )
        seed_model_huber = make_fit_model(
            [("Lorentzian", {"x0": 495.0, "ampli": 80.0, "fwhm": 5.0})],
            fit_params={"loss": "huber", "f_scale": 5.0, "max_ite": 200, "xtol": 1e-4, "ftol": 1e-4},
        )

        engine = VBFengine()
        p_lin, _, _, _, _, names = engine.fit_spectra(x=x, Y=y_spikes[None, :], fit_model=seed_model_lin)
        p_huber, s_huber, _, _, _, _ = engine.fit_spectra(x=x, Y=y_spikes[None, :], fit_model=seed_model_huber)

        assert s_huber[0]
        res_lin = dict(zip(names, p_lin[0]))
        res_huber = dict(zip(names, p_huber[0]))

        err_x0_lin = abs(res_lin["P1_x0"] - 500.0)
        err_x0_huber = abs(res_huber["P1_x0"] - 500.0)
        assert err_x0_huber < err_x0_lin

    def test_coef_noise_and_robust_loss_composition(self, make_fit_model, make_synthetic_spectrum):
        true_peak = ("Lorentzian", {"x0": 500.0, "ampli": 100.0, "fwhm": 6.0})
        fit_model = make_fit_model([true_peak])
        x, y = make_synthetic_spectrum([(true_peak[0], fit_model["peak_models"]["0"]["Lorentzian"])])

        # Add baseline noise + spikes
        rng = np.random.default_rng(42)
        y_noisy = y + rng.normal(0, 1.0, len(x))
        y_noisy[20] += 300.0  # spike

        seed_model = make_fit_model(
            [("Lorentzian", {"x0": 498.0, "ampli": 90.0, "fwhm": 5.5})],
            fit_params={"loss": "soft_l1", "f_scale": 5.0, "coef_noise": 1.0, "max_ite": 200},
        )
        engine = VBFengine()
        p_full, success, r2, _, _, names = engine.fit_spectra(x=x, Y=y_noisy[None, :], fit_model=seed_model)
        assert success[0]
        res = dict(zip(names, p_full[0]))
        assert res["P1_x0"] == pytest.approx(500.0, abs=0.5)

    def test_spike_at_peak_maximum_recovery(self, make_fit_model, make_synthetic_spectrum):
        true_peak = ("Lorentzian", {"x0": 500.0, "ampli": 100.0, "fwhm": 6.0})
        fit_model = make_fit_model([true_peak])
        x, y = make_synthetic_spectrum([(true_peak[0], fit_model["peak_models"]["0"]["Lorentzian"])])

        # Inject spike directly at peak center
        idx_center = np.argmin(np.abs(x - 500.0))
        y_spike = y.copy()
        y_spike[idx_center] += 400.0

        seed_model_lin = make_fit_model(
            [("Lorentzian", {"x0": 496.0, "ampli": 80.0, "fwhm": 5.0})],
            fit_params={"loss": "linear", "max_ite": 200},
        )
        seed_model_soft = make_fit_model(
            [("Lorentzian", {"x0": 496.0, "ampli": 80.0, "fwhm": 5.0})],
            fit_params={"loss": "soft_l1", "f_scale": 5.0, "max_ite": 200},
        )
        seed_model_huber = make_fit_model(
            [("Lorentzian", {"x0": 496.0, "ampli": 80.0, "fwhm": 5.0})],
            fit_params={"loss": "huber", "f_scale": 5.0, "max_ite": 200},
        )

        engine = VBFengine()
        p_lin, _, _, _, _, names = engine.fit_spectra(x=x, Y=y_spike[None, :], fit_model=seed_model_lin)
        p_soft, s_soft, _, _, _, _ = engine.fit_spectra(x=x, Y=y_spike[None, :], fit_model=seed_model_soft)
        p_huber, s_huber, _, _, _, _ = engine.fit_spectra(x=x, Y=y_spike[None, :], fit_model=seed_model_huber)

        assert s_soft[0]
        assert s_huber[0]
        res_lin = dict(zip(names, p_lin[0]))
        res_soft = dict(zip(names, p_soft[0]))
        res_huber = dict(zip(names, p_huber[0]))

        # Linear fit gets severely corrupted by spike on the peak
        assert abs(res_lin["P1_ampli"] - 100.0) > 300.0
        # Robust losses recover the amplitude and FWHM accurately
        assert abs(res_soft["P1_ampli"] - 100.0) < 10.0
        assert abs(res_huber["P1_ampli"] - 100.0) < 10.0
        assert res_soft["P1_x0"] == pytest.approx(500.0, abs=0.1)
        assert res_huber["P1_x0"] == pytest.approx(500.0, abs=0.1)
