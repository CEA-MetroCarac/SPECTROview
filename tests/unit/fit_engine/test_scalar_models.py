"""Unit tests for fit_engine/scalar_models.py - PEAK_MODEL_REGISTRY.

These are the non-batched (single-spectrum) reference implementations used
by evaluator.eval_peak_initial() for instant UI peak-curve previews before a
fit has run, and as the numerical fallback for any peak shape without a
BATCHED_MODELS entry.
"""
import numpy as np
import pytest

from spectroview.fit_engine.scalar_models import (
    PEAK_MODEL_REGISTRY,
    gaussian, lorentzian, pseudovoigt, fano,
    decay_single_exp, decay_bi_exp,
)


class TestRegistryShape:
    def test_all_entries_are_function_paramlist_pairs(self):
        for shape, (fn, params) in PEAK_MODEL_REGISTRY.items():
            assert callable(fn), shape
            assert isinstance(params, list) and len(params) > 0, shape

    def test_registered_functions_accept_their_own_param_count(self):
        x = np.linspace(0, 100, 50)
        for shape, (fn, params) in PEAK_MODEL_REGISTRY.items():
            args = [1.0] * len(params)
            y = fn(x, *args)
            assert y.shape == x.shape, shape
            assert np.isfinite(y).all(), shape


class TestGaussian:
    def test_peak_at_x0_equals_amplitude(self):
        x = np.array([490.0, 500.0, 510.0])
        y = gaussian(x, ampli=10.0, fwhm=4.0, x0=500.0)
        assert y[1] == pytest.approx(10.0)

    def test_symmetric_around_x0(self):
        x0 = 500.0
        x = np.array([x0 - 5.0, x0, x0 + 5.0])
        y = gaussian(x, ampli=10.0, fwhm=4.0, x0=x0)
        assert y[0] == pytest.approx(y[2])

    def test_fwhm_definition(self):
        """At x0 +/- fwhm/2 the value should be half the amplitude."""
        x0, fwhm, ampli = 500.0, 8.0, 20.0
        x = np.array([x0 - fwhm / 2, x0, x0 + fwhm / 2])
        y = gaussian(x, ampli=ampli, fwhm=fwhm, x0=x0)
        assert y[1] == pytest.approx(ampli)
        assert y[0] == pytest.approx(ampli / 2, rel=1e-6)
        assert y[2] == pytest.approx(ampli / 2, rel=1e-6)


class TestLorentzian:
    def test_fwhm_definition(self):
        x0, fwhm, ampli = 500.0, 8.0, 20.0
        x = np.array([x0 - fwhm / 2, x0, x0 + fwhm / 2])
        y = lorentzian(x, ampli=ampli, fwhm=fwhm, x0=x0)
        assert y[1] == pytest.approx(ampli, rel=1e-4)
        assert y[0] == pytest.approx(ampli / 2, rel=1e-3)
        assert y[2] == pytest.approx(ampli / 2, rel=1e-3)


class TestPseudoVoigt:
    def test_alpha_one_equals_gaussian(self):
        x = np.linspace(480, 520, 50)
        y_pv = pseudovoigt(x, ampli=10, fwhm=5, x0=500, alpha=1.0)
        y_g = gaussian(x, ampli=10, fwhm=5, x0=500)
        np.testing.assert_allclose(y_pv, y_g)

    def test_alpha_zero_equals_lorentzian(self):
        x = np.linspace(480, 520, 50)
        y_pv = pseudovoigt(x, ampli=10, fwhm=5, x0=500, alpha=0.0)
        y_l = lorentzian(x, ampli=10, fwhm=5, x0=500)
        np.testing.assert_allclose(y_pv, y_l)


class TestFano:
    def test_large_q_approaches_lorentzian_shape(self):
        """As q -> inf, the Fano profile's peak position approaches x0
        (symmetric limit), matching a Lorentzian's peak location."""
        x = np.linspace(480, 520, 2001)
        y = fano(x, ampli=1.0, fwhm=5.0, x0=500.0, q=1000.0)
        peak_x = x[np.argmax(y)]
        assert peak_x == pytest.approx(500.0, abs=0.5)


class TestDecayModels:
    def test_single_exp_decays_toward_baseline(self):
        x = np.linspace(0, 100, 200)
        y = decay_single_exp(x, A=100.0, tau=5.0, B=2.0)
        assert y[0] == pytest.approx(102.0, rel=1e-3)
        assert y[-1] == pytest.approx(2.0, abs=1e-3)
        assert np.all(np.diff(y) <= 1e-9)  # monotonically non-increasing

    def test_bi_exp_reduces_to_single_when_second_amplitude_zero(self):
        x = np.linspace(0, 100, 200)
        y_bi = decay_bi_exp(x, A1=100.0, tau1=5.0, A2=0.0, tau2=20.0, B=1.0)
        y_single = decay_single_exp(x, A=100.0, tau=5.0, B=1.0)
        np.testing.assert_allclose(y_bi, y_single)

    def test_scalar_matches_batched_to_machine_precision_even_for_narrow_widths(self):
        from spectroview.fit_engine.models import BATCHED_MODELS

        x_spec = np.linspace(495.0, 505.0, 201)
        x_dec = np.linspace(0.0, 10.0, 201)
        narrow_cases = {
            "Gaussian": (x_spec, [100.0, 0.01, 500.0]),
            "Lorentzian": (x_spec, [100.0, 0.01, 500.0]),
            "PseudoVoigt": (x_spec, [100.0, 0.01, 500.0, 0.35]),
            "GaussianAsym": (x_spec, [100.0, 0.01, 0.02, 500.0]),
            "LorentzianAsym": (x_spec, [100.0, 0.01, 0.02, 500.0]),
            "Fano": (x_spec, [40.0, 0.01, 500.0, 2.5]),
            "DecaySingleExp": (x_dec, [100.0, 0.05, 1.5]),
            "DecayBiExp": (x_dec, [80.0, 0.05, 40.0, 0.5, 1.5]),
        }
        for name, (x_in, p_vals) in narrow_cases.items():
            scalar_fn, _ = PEAK_MODEL_REGISTRY[name]
            batched_fn, _, _ = BATCHED_MODELS[name]
            y_s = scalar_fn(x_in, *p_vals)
            y_b = batched_fn(x_in, np.array([p_vals], dtype=np.float64))[0]
            np.testing.assert_allclose(y_s, y_b, rtol=1e-12, atol=1e-11)

        # At x == x0 with fwhm == 0, Lorentzian peak maximum equals ampli (not 0)
        assert lorentzian(np.array([500.0]), 123.0, 0.0, 500.0)[0] == pytest.approx(123.0)

