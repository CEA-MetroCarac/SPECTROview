"""Unit tests for model/peak_model.py - initialize_peak_params()."""
import numpy as np
import pytest

from spectroview.model.peak_model import initialize_peak_params

STANDARD_SHAPES = ["Lorentzian", "Gaussian", "PseudoVoigt", "GaussianAsym", "LorentzianAsym", "Fano"]


class TestStandardShapes:
    @pytest.mark.parametrize("shape", STANDARD_SHAPES)
    def test_produces_x0_bounds_from_maxshift(self, shape):
        peak_model = {}
        initialize_peak_params(peak_model, shape, x0=500.0, ampli=100.0,
                                minfwhm=1.0, maxfwhm=50.0, maxshift=10.0)
        assert peak_model["x0"]["value"] == 500.0
        assert peak_model["x0"]["min"] == 490.0
        assert peak_model["x0"]["max"] == 510.0

    @pytest.mark.parametrize("shape", [s for s in STANDARD_SHAPES if s != "Fano"])
    def test_ampli_value_passed_through_unscaled(self, shape):
        # Fano is the one exception (rescaled by q^2+1) -- covered separately.
        peak_model = {}
        initialize_peak_params(peak_model, shape, x0=500.0, ampli=100.0,
                                minfwhm=1.0, maxfwhm=50.0, maxshift=10.0)
        assert peak_model["ampli"]["value"] == 100.0

    def test_asym_shapes_get_two_width_params(self):
        for shape in ("GaussianAsym", "LorentzianAsym"):
            peak_model = {}
            initialize_peak_params(peak_model, shape, x0=500.0, ampli=100.0,
                                    minfwhm=1.0, maxfwhm=50.0, maxshift=10.0)
            assert "fwhm_l" in peak_model and "fwhm_r" in peak_model
            assert "fwhm" not in peak_model
            assert peak_model["fwhm_l"]["min"] == 1.0
            assert peak_model["fwhm_l"]["max"] == 50.0

    def test_symmetric_shapes_get_single_width_param(self):
        for shape in ("Lorentzian", "Gaussian", "PseudoVoigt", "Fano"):
            peak_model = {}
            initialize_peak_params(peak_model, shape, x0=500.0, ampli=100.0,
                                    minfwhm=1.0, maxfwhm=50.0, maxshift=10.0)
            assert "fwhm" in peak_model
            assert "fwhm_l" not in peak_model

    def test_pseudovoigt_adds_alpha_bounded_zero_one(self):
        peak_model = {}
        initialize_peak_params(peak_model, "PseudoVoigt", x0=500.0, ampli=100.0,
                                minfwhm=1.0, maxfwhm=50.0, maxshift=10.0)
        assert peak_model["alpha"] == {"value": 0.5, "min": 0.0, "max": 1.0, "vary": True, "expr": None}

    def test_fano_adds_q_and_rescales_ampli(self):
        peak_model = {}
        initialize_peak_params(peak_model, "Fano", x0=500.0, ampli=100.0,
                                minfwhm=1.0, maxfwhm=50.0, maxshift=10.0)
        assert "q" in peak_model
        q = peak_model["q"]["value"]
        # ampli is pre-divided by (q^2+1) so the *displayed* peak height (via
        # fano_display_amplitude) still matches the requested `ampli`.
        assert peak_model["ampli"]["value"] == pytest.approx(100.0 / (q ** 2 + 1))

    def test_ampli_bounds_are_nonnegative(self):
        peak_model = {}
        initialize_peak_params(peak_model, "Lorentzian", x0=500.0, ampli=100.0,
                                minfwhm=1.0, maxfwhm=50.0, maxshift=10.0)
        assert peak_model["ampli"]["min"] == 0.0
        assert peak_model["ampli"]["max"] == 100.0 * 1e6

    def test_clears_previous_content(self):
        peak_model = {"leftover_key": "should be gone"}
        initialize_peak_params(peak_model, "Lorentzian", x0=500.0, ampli=100.0,
                                minfwhm=1.0, maxfwhm=50.0, maxshift=10.0)
        assert "leftover_key" not in peak_model


class TestDecayShapes:
    def test_decay_single_exp_uses_data_extrema(self):
        peak_model = {}
        y_arr = np.array([1.0, 5.0, 100.0, 20.0, 2.0])
        initialize_peak_params(peak_model, "DecaySingleExp", x0=0.0, ampli=1.0,
                                minfwhm=0, maxfwhm=0, maxshift=0, y_arr=y_arr)
        assert peak_model["A"]["value"] == 100.0
        assert peak_model["B"]["value"] == 1.0
        assert set(peak_model.keys()) == {"A", "tau", "B"}

    def test_decay_bi_exp_splits_amplitude_70_30(self):
        peak_model = {}
        y_arr = np.array([0.0, 50.0, 100.0])
        initialize_peak_params(peak_model, "DecayBiExp", x0=0.0, ampli=1.0,
                                minfwhm=0, maxfwhm=0, maxshift=0, y_arr=y_arr)
        assert peak_model["A1"]["value"] == pytest.approx(70.0)
        assert peak_model["A2"]["value"] == pytest.approx(30.0)
        assert set(peak_model.keys()) == {"A1", "tau1", "A2", "tau2", "B"}

    def test_decay_without_y_arr_falls_back_to_ampli(self):
        peak_model = {}
        initialize_peak_params(peak_model, "DecaySingleExp", x0=0.0, ampli=42.0,
                                minfwhm=0, maxfwhm=0, maxshift=0)
        assert peak_model["A"]["value"] == 42.0
