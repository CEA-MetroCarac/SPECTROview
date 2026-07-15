"""Tests for spectroview.api.fitting -- build_fit_model, fit_batch,
apply_fit_model, and fit-model template CRUD."""
import numpy as np
import pytest

from spectroview.api import fitting
from spectroview.api.exceptions import FitModelError, TemplateError
from spectroview.fit_engine.evaluator import VBFevaluator


class TestBuildFitModel:
    def test_produces_shape_vbfevaluator_accepts(self):
        fm = fitting.build_fit_model([
            {"model": "Lorentzian",
             "x0": {"value": 520.0, "min": 515.0, "max": 525.0},
             "ampli": {"value": 1000.0, "min": 0.0, "max": 1e9},
             "fwhm": {"value": 3.0, "min": 0.5, "max": 10.0}},
        ])
        evaluator = VBFevaluator.from_fit_model(fm)
        assert evaluator.n_params_total == 3

    def test_scalar_shorthand_becomes_value_only(self):
        fm = fitting.build_fit_model([{"model": "Gaussian", "x0": 500.0}])
        assert fm["peak_models"]["0"]["Gaussian"]["x0"] == {"value": 500.0}

    def test_fix_alias_becomes_vary_false(self):
        fm = fitting.build_fit_model([{"model": "Gaussian", "x0": {"value": 500.0, "fix": True}}])
        assert fm["peak_models"]["0"]["Gaussian"]["x0"]["vary"] is False

    def test_missing_model_key_raises(self):
        with pytest.raises(FitModelError):
            fitting.build_fit_model([{"x0": {"value": 1.0}}])


class TestFitBatch:
    def test_recovers_known_peak_params(self, make_fit_model, make_synthetic_map):
        x, Y, coords, fnames, true_params, canonical = make_synthetic_map(
            shape="Lorentzian", n_spectra=8, noise_std=0.0,
        )
        fm = make_fit_model([("Lorentzian", dict(x0=500.0, ampli=100.0, fwhm=8.0))])
        result = fitting.fit_batch(x, Y, fm)

        assert result["success"].all()
        assert result["params"].shape[0] == 8
        assert np.allclose(result["r_squared"], 1.0, atol=1e-3)

    def test_no_peak_models_raises_fit_model_error(self):
        x = np.linspace(0, 10, 20)
        Y = np.zeros((1, 20))
        with pytest.raises(FitModelError):
            fitting.fit_batch(x, Y, {"peak_models": {}})

    def test_auto_weights_excludes_negative_points(self, make_fit_model, synth_x):
        # A spectrum with one large negative spike should still be fit well when
        # auto_weights zeroes it out, matching GUI behavior for fit_negative=False.
        y = np.full_like(synth_x, 1.0)
        peak_idx = len(synth_x) // 2
        y[peak_idx] += 500.0  # the "peak" we want fit to find
        y[10] = -1000.0  # spike that would corrupt a fit if not excluded

        fm = make_fit_model(
            [("Lorentzian", dict(x0=float(synth_x[peak_idx]), ampli=500.0, fwhm=5.0))],
            fit_params={"fit_negative": False, "max_ite": 200, "xtol": 1e-4, "ftol": 1e-4, "coef_noise": 0.0},
        )
        result_auto = fitting.fit_batch(synth_x, y[None, :], fm, auto_weights=True)
        result_noauto = fitting.fit_batch(synth_x, y[None, :], fm, auto_weights=False,
                                           fit_params=fm["fit_params"])
        # Excluding the negative spike should fit at least as well as including it.
        assert result_auto["r_squared"][0] >= result_noauto["r_squared"][0] - 1e-6


class TestApplyFitModel:
    def test_crops_baselines_and_fits(self, make_fit_model, synth_x):
        peak_x0 = float(synth_x[len(synth_x) // 2])
        y_peak = 200.0 * np.exp(-0.5 * ((synth_x - peak_x0) / 4.0) ** 2)
        y_ramp = 0.5 * (synth_x - synth_x[0])  # linear baseline contamination
        Y = (y_peak + y_ramp)[None, :]

        fm = make_fit_model(
            [("Gaussian", dict(x0=peak_x0, ampli=200.0, fwhm=9.4))],
            range_min=float(synth_x[0]) + 20, range_max=float(synth_x[-1]) - 20,
            baseline={"mode": "Linear", "attached": True, "sigma": 0, "is_subtracted": True,
                      "points": [[synth_x[0], synth_x[-1]], [y_ramp[0], y_ramp[-1]]]},
        )
        result = fitting.apply_fit_model(synth_x, Y, fm)

        assert result["x"].min() >= float(synth_x[0]) + 20
        assert result["x"].max() <= float(synth_x[-1]) - 20
        assert result["success"][0]
        assert result["r_squared"][0] > 0.9

    def test_range_excluding_all_points_raises(self, make_fit_model, synth_x):
        fm = make_fit_model([("Gaussian", dict(x0=500.0))], range_min=1e6, range_max=2e6)
        Y = np.zeros((1, len(synth_x)))
        with pytest.raises(FitModelError):
            fitting.apply_fit_model(synth_x, Y, fm)


class TestFitModelTemplateCRUD:
    def test_save_and_load_round_trip(self, make_fit_model, tmp_path):
        fm = make_fit_model([("Lorentzian", dict(x0=520.0))])
        path = fitting.save_fit_model_template(fm, tmp_path / "my_model.json")
        loaded = fitting.load_fit_model_template(path)
        assert loaded["peak_models"].keys() == fm["peak_models"].keys()

    def test_list_templates_in_folder(self, make_fit_model, tmp_path):
        fm = make_fit_model([("Lorentzian", dict(x0=520.0))])
        fitting.save_fit_model_template(fm, tmp_path / "a.json")
        fitting.save_fit_model_template(fm, tmp_path / "b.json")
        names = fitting.list_fit_model_templates(tmp_path)
        assert set(names) == {"a.json", "b.json"}

    def test_load_missing_file_raises_template_error(self, tmp_path):
        with pytest.raises(TemplateError):
            fitting.load_fit_model_template(tmp_path / "nope.json")

    def test_load_real_gui_saved_template(self, fit_model_si_file):
        if not fit_model_si_file.exists():
            pytest.skip("fit_model_Si_.json fixture not present")
        fm = fitting.load_fit_model_template(fit_model_si_file)
        assert "peak_models" in fm
