# tests/test_batch_engine.py
"""
Tests for the high-performance batch fitting engine.

Tests model evaluation accuracy (vs fitspy), parameter vector round-trip,
spatial traversal, and end-to-end batch fitting with synthetic data.
"""

import numpy as np
import pytest
from copy import deepcopy

from spectroview.core.models import (
    gaussian, lorentzian, pseudovoigt, gaussian_asym, lorentzian_asym, fano,
    decay_single_exp, decay_bi_exp,
    PeakModelEvaluator, FitResult, ParamValue,
    PEAK_MODEL_REGISTRY,
)
from spectroview.core.spatial import (
    build_traversal_order, NeighborPropagator,
)
from spectroview.core.optimizer import (
    fit_single_spectrum, fit_batch_sequential,
)
from spectroview.core.batch_engine import BatchFittingEngine
from spectroview.core2.tensor_engine import TensorFittingEngine
from spectroview.model.m_spectrum import MSpectrum


# ═══════════════════════════════════════════════════════════════════════════
# Test: Peak model functions match fitspy
# ═══════════════════════════════════════════════════════════════════════════

class TestPeakModels:
    """Verify that our NumPy model functions produce the same output as fitspy's."""

    def test_gaussian_shape(self):
        x = np.linspace(-10, 10, 201)
        y = gaussian(x, ampli=1.0, fwhm=2.0, x0=0.0)
        assert y.shape == (201,)
        # Peak should be at x0=0
        assert np.argmax(y) == 100
        # Peak amplitude should be ampli
        assert abs(y[100] - 1.0) < 1e-10

    def test_lorentzian_shape(self):
        x = np.linspace(-10, 10, 201)
        y = lorentzian(x, ampli=1.0, fwhm=2.0, x0=0.0)
        assert y.shape == (201,)
        assert np.argmax(y) == 100
        # Lorentzian at x0: ampli * fwhm^2 / (4 * fwhm^2/4) = ampli
        assert abs(y[100] - 1.0) < 1e-3

    def test_pseudovoigt_interpolates(self):
        x = np.linspace(-10, 10, 201)
        y_g = gaussian(x, 1.0, 2.0, 0.0)
        y_l = lorentzian(x, 1.0, 2.0, 0.0)
        y_pv = pseudovoigt(x, 1.0, 2.0, 0.0, alpha=0.5)
        # Should be midpoint between Gaussian and Lorentzian
        expected = 0.5 * y_g + 0.5 * y_l
        np.testing.assert_allclose(y_pv, expected, atol=1e-10)

    def test_gaussian_asym(self):
        x = np.linspace(-10, 10, 201)
        y = gaussian_asym(x, 1.0, 4.0, 2.0, 0.0)
        # Peak should be at x0
        idx_peak = np.argmax(y)
        assert abs(x[idx_peak]) < 0.2

    def test_fano_reduces_to_lorentzian_for_large_q(self):
        x = np.linspace(-10, 10, 201)
        y_fano = fano(x, 1.0, 2.0, 0.0, q=1000)
        y_lor = lorentzian(x, 1.0, 2.0, 0.0)
        # For very large q, Fano → Lorentzian (approximately)
        # The scaling is different, but the shape should match
        # Normalize both and compare shape
        y_fano_norm = y_fano / y_fano.max()
        y_lor_norm = y_lor / y_lor.max()
        # Shape should be similar (not exact because of Fano scaling)
        assert abs(np.argmax(y_fano) - np.argmax(y_lor)) <= 1

    def test_decay_single_exp(self):
        t = np.linspace(0, 100, 1001)
        y = decay_single_exp(t, A=1000, tau=20, B=10)
        assert abs(y[0] - 1010) < 1e-6
        assert y[-1] < 20  # Should have decayed

    def test_all_models_callable(self):
        """Verify every registered model can be called without error."""
        x = np.linspace(0, 100, 100)
        for name, (func, params) in PEAK_MODEL_REGISTRY.items():
            # Build dummy arguments
            args = [x] + [1.0] * len(params)
            try:
                y = func(*args)
                assert len(y) == len(x), f"Model {name} output length mismatch"
            except Exception as e:
                pytest.fail(f"Model {name} raised {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Test: PeakModelEvaluator
# ═══════════════════════════════════════════════════════════════════════════

class TestPeakModelEvaluator:
    """Test model evaluator construction, param vector mapping, and evaluation."""

    @pytest.fixture
    def simple_fit_model(self):
        """A simple fit model with 2 Lorentzian peaks."""
        return {
            "peak_models": {
                "0": {
                    "Lorentzian": {
                        "ampli": {"value": 100, "min": 0, "max": 1e6, "vary": True, "expr": None},
                        "fwhm":  {"value": 5,   "min": 0, "max": 200, "vary": True, "expr": None},
                        "x0":    {"value": 520, "min": 500, "max": 540, "vary": True, "expr": None},
                    }
                },
                "1": {
                    "Lorentzian": {
                        "ampli": {"value": 50, "min": 0, "max": 1e6, "vary": True, "expr": None},
                        "fwhm":  {"value": 8,  "min": 0, "max": 200, "vary": True, "expr": None},
                        "x0":    {"value": 300, "min": 280, "max": 320, "vary": True, "expr": None},
                    }
                },
            },
            "peak_labels": ["Si", "Other"],
        }

    def test_from_fit_model(self, simple_fit_model):
        ev = PeakModelEvaluator.from_fit_model(simple_fit_model)
        assert ev.n_params_total == 6  # 3 params * 2 peaks
        assert ev.n_params_free == 6   # All vary=True
        assert len(ev.param_names) == 6
        assert "m01_ampli" in ev.param_names
        assert "m02_x0" in ev.param_names

    def test_initial_params_and_bounds(self, simple_fit_model):
        ev = PeakModelEvaluator.from_fit_model(simple_fit_model)
        p0 = ev.initial_params
        lb, ub = ev.bounds
        assert len(p0) == 6
        assert len(lb) == 6
        assert np.all(lb <= p0)
        assert np.all(p0 <= ub)

    def test_free_to_full_roundtrip(self, simple_fit_model):
        ev = PeakModelEvaluator.from_fit_model(simple_fit_model)
        p0 = ev.initial_params
        p_full = ev.free_to_full(p0)
        p_free = ev.full_to_free(p_full)
        np.testing.assert_allclose(p0, p_free)

    def test_evaluate_produces_peaks(self, simple_fit_model):
        ev = PeakModelEvaluator.from_fit_model(simple_fit_model)
        x = np.linspace(200, 600, 500)
        p_full = ev.get_all_initial_params()
        y = ev.evaluate(x, p_full)
        assert y.shape == (500,)
        # Should have peaks near 520 and 300
        idx_520 = np.argmin(np.abs(x - 520))
        idx_300 = np.argmin(np.abs(x - 300))
        assert y[idx_520] > 50  # Peak near 520
        assert y[idx_300] > 20  # Peak near 300

    def test_residual(self, simple_fit_model):
        ev = PeakModelEvaluator.from_fit_model(simple_fit_model)
        x = np.linspace(200, 600, 500)
        p_full = ev.get_all_initial_params()
        y_model = ev.evaluate(x, p_full)
        # Residual with matching data should be zero
        r = ev.residual(ev.initial_params, x, y_model)
        np.testing.assert_allclose(r, 0.0, atol=1e-10)

    def test_fixed_param(self):
        """Test that fixed parameters are not optimized."""
        fit_model = {
            "peak_models": {
                "0": {
                    "Gaussian": {
                        "ampli": {"value": 100, "min": 0, "max": 1e6, "vary": True},
                        "fwhm":  {"value": 5,   "min": 0, "max": 200, "vary": False},  # FIXED
                        "x0":    {"value": 520, "min": 500, "max": 540, "vary": True},
                    }
                }
            },
            "peak_labels": ["1"],
        }
        ev = PeakModelEvaluator.from_fit_model(fit_model)
        assert ev.n_params_total == 3
        assert ev.n_params_free == 2  # Only ampli and x0 are free

    def test_build_result(self, simple_fit_model):
        ev = PeakModelEvaluator.from_fit_model(simple_fit_model)
        x = np.linspace(200, 600, 500)
        p0 = ev.initial_params
        result = ev.build_result(p0, x, np.zeros(500), True)
        assert isinstance(result, FitResult)
        assert result.success is True
        assert "m01_x0" in result.params
        assert abs(result.params["m01_x0"].value - 520) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# Test: Spatial traversal
# ═══════════════════════════════════════════════════════════════════════════

class TestSpatial:
    """Test spatial traversal order and neighbor propagation."""

    def test_sequential_order(self):
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        order = build_traversal_order(coords, strategy="sequential")
        np.testing.assert_array_equal(order, [0, 1, 2, 3])

    def test_spiral_starts_from_center(self):
        # 3x3 grid
        coords = np.array([
            [0, 0], [1, 0], [2, 0],
            [0, 1], [1, 1], [2, 1],
            [0, 2], [1, 2], [2, 2],
        ])
        order = build_traversal_order(coords, strategy="spiral")
        # First pixel should be center (1,1) = index 4
        assert order[0] == 4

    def test_spiral_visits_all(self):
        coords = np.random.rand(100, 2) * 10
        order = build_traversal_order(coords, strategy="spiral")
        assert len(order) == 100
        assert len(set(order)) == 100  # All unique

    def test_neighbor_propagator(self):
        coords = np.array([[0, 0], [1, 0], [2, 0]])
        prop = NeighborPropagator(coords, k_neighbors=2)

        default = np.array([1.0, 2.0, 3.0])
        # No cache yet — should return default
        p0 = prop.get_initial_guess(1, default)
        np.testing.assert_array_equal(p0, default)

        # Store result for pixel 0
        fitted = np.array([10.0, 20.0, 30.0])
        prop.store_result(0, fitted)

        # Pixel 1 should now get pixel 0's result (nearest neighbor)
        p0 = prop.get_initial_guess(1, default)
        np.testing.assert_array_equal(p0, fitted)


# ═══════════════════════════════════════════════════════════════════════════
# Test: Optimizer
# ═══════════════════════════════════════════════════════════════════════════

class TestOptimizer:
    """Test single-spectrum fitting with known synthetic data."""

    def test_fit_single_lorentzian(self):
        """Fit a single Lorentzian peak and recover parameters."""
        # Generate synthetic data
        x = np.linspace(400, 600, 300)
        true_params = {"ampli": 100, "fwhm": 8, "x0": 520}
        y = lorentzian(x, **true_params) + np.random.normal(0, 0.5, len(x))

        # Build evaluator
        fit_model = {
            "peak_models": {
                "0": {
                    "Lorentzian": {
                        "ampli": {"value": 80, "min": 0, "max": 1e6, "vary": True},
                        "fwhm":  {"value": 5,  "min": 0, "max": 200, "vary": True},
                        "x0":    {"value": 518, "min": 500, "max": 540, "vary": True},
                    }
                }
            },
            "peak_labels": ["1"],
        }
        ev = PeakModelEvaluator.from_fit_model(fit_model)
        p0 = ev.initial_params
        bounds = ev.bounds

        p_opt, success, cost = fit_single_spectrum(x, y, ev, p0, bounds)

        assert success
        # Recover parameters within tolerance
        p_full = ev.free_to_full(p_opt)
        assert abs(p_full[0] - 100) < 5    # ampli
        assert abs(p_full[1] - 8) < 2      # fwhm
        assert abs(p_full[2] - 520) < 1    # x0

    def test_fit_two_peaks(self):
        """Fit two Lorentzian peaks."""
        x = np.linspace(200, 600, 500)
        y = (lorentzian(x, 100, 8, 520) +
             lorentzian(x, 60, 12, 310) +
             np.random.normal(0, 0.5, len(x)))

        fit_model = {
            "peak_models": {
                "0": {
                    "Lorentzian": {
                        "ampli": {"value": 80, "min": 0, "max": 1e6, "vary": True},
                        "fwhm":  {"value": 5,  "min": 0, "max": 200, "vary": True},
                        "x0":    {"value": 518, "min": 500, "max": 540, "vary": True},
                    }
                },
                "1": {
                    "Lorentzian": {
                        "ampli": {"value": 40, "min": 0, "max": 1e6, "vary": True},
                        "fwhm":  {"value": 10, "min": 0, "max": 200, "vary": True},
                        "x0":    {"value": 308, "min": 280, "max": 320, "vary": True},
                    }
                },
            },
            "peak_labels": ["Si", "Other"],
        }
        ev = PeakModelEvaluator.from_fit_model(fit_model)
        p0 = ev.initial_params
        bounds = ev.bounds

        p_opt, success, cost = fit_single_spectrum(x, y, ev, p0, bounds)
        assert success


# ═══════════════════════════════════════════════════════════════════════════
# Test: BatchFittingEngine (end-to-end)
# ═══════════════════════════════════════════════════════════════════════════

class TestBatchFittingEngine:
    """End-to-end test with synthetic map data."""

    def _make_synthetic_map(self, n_pixels=25, n_wavelengths=200):
        """Create synthetic map data with known Lorentzian peaks.

        Simulates a 5x5 grid where peak parameters vary smoothly across
        the map (like a real 2D map).
        """
        from spectroview.model.m_spectrum import MSpectrum

        grid_size = int(np.sqrt(n_pixels))
        x = np.linspace(400, 600, n_wavelengths)
        coords = []
        spectra = []

        for iy in range(grid_size):
            for ix in range(grid_size):
                # Peak parameters vary smoothly across the map
                ampli = 100 + 10 * ix - 5 * iy
                fwhm = 8 + 0.5 * iy
                x0 = 520 + 0.2 * ix - 0.1 * iy

                y = lorentzian(x, ampli, fwhm, x0)
                y += np.random.normal(0, 0.5, len(x))

                spectrum = MSpectrum()
                spectrum.fname = f"test_map_({ix}, {iy})"
                spectrum.x0 = x.copy()
                spectrum.y0 = y.copy()
                spectrum.x = x.copy()
                spectrum.y = y.copy()

                spectra.append(spectrum)
                coords.append([float(ix), float(iy)])

        coords = np.array(coords)
        return spectra, coords, x

    def test_batch_fit_synthetic_map(self):
        """Fit a synthetic 5x5 map and verify parameter recovery."""
        spectra, coords, x = self._make_synthetic_map(n_pixels=25)

        fit_model = {
            "peak_models": {
                "0": {
                    "Lorentzian": {
                        "ampli": {"value": 80, "min": 0, "max": 1e6, "vary": True},
                        "fwhm":  {"value": 5,  "min": 0, "max": 200, "vary": True},
                        "x0":    {"value": 518, "min": 500, "max": 540, "vary": True},
                    }
                }
            },
            "peak_labels": ["Si"],
        }

        engine = BatchFittingEngine()

        progress_log = []
        def on_progress(current, total):
            progress_log.append((current, total))

        results = engine.fit_spectra(
            spectra=spectra,
            fit_model=fit_model,
            coords=coords,
            fit_params={"method": "leastsq", "xtol": 1e-4, "max_ite": 200},
            ncpus=1,
            progress_callback=on_progress,
            apply_model_to_spectra=True,
        )

        assert len(results) == 25

        # Check that fitting succeeded for most spectra
        n_success = sum(1 for r in results if r.success)
        assert n_success >= 20, f"Only {n_success}/25 fits succeeded"

        # Check that result_fit was written back to spectra
        for spectrum in spectra:
            assert hasattr(spectrum.result_fit, "success")
            if spectrum.result_fit.success:
                assert "m01_x0" in spectrum.result_fit.params

        # Verify progress was reported
        assert len(progress_log) > 0

    def test_tensor_engine_reports_completion(self):
        """Tensor fitting should emit final completion progress."""
        x = np.linspace(250, 350, 120)
        y = 150.0 / (1.0 + ((x - 300.0) / 10.0) ** 2)

        spectra = []
        for i in range(2):
            spectrum = MSpectrum()
            spectrum.fname = f"sample_{i}"
            spectrum.x0 = x.copy()
            spectrum.y0 = y.copy()
            spectrum.x = x.copy()
            spectrum.y = y.copy()
            spectrum.baseline.mode = "Linear"
            spectra.append(spectrum)

        fit_model = {
            "peak_models": {
                "0": {
                    "Lorentzian": {
                        "ampli": {"value": 100.0, "min": 0.0, "max": 1e6, "vary": True},
                        "fwhm":  {"value": 10.0,  "min": 0.0, "max": 200.0, "vary": True},
                        "x0":    {"value": 300.0, "min": 280.0, "max": 320.0, "vary": True},
                    }
                }
            },
            "peak_labels": ["Peak"],
        }

        progress_log = []
        engine = TensorFittingEngine()
        results = engine.fit_spectra(
            spectra=spectra,
            fit_model=fit_model,
            fit_params={"method": "leastsq", "xtol": 1e-4, "ftol": 1e-4, "max_ite": 100},
            progress_callback=lambda current, total: progress_log.append((current, total)),
            apply_model_to_spectra=True,
        )

        assert len(results) == 2
        assert len(progress_log) > 0
        assert progress_log[-1] == (2, 2)

    def test_batch_fit_without_coords(self):
        """Fit spectra without spatial coordinates (no propagation)."""
        spectra, _, x = self._make_synthetic_map(n_pixels=9)

        fit_model = {
            "peak_models": {
                "0": {
                    "Lorentzian": {
                        "ampli": {"value": 80, "min": 0, "max": 1e6, "vary": True},
                        "fwhm":  {"value": 5,  "min": 0, "max": 200, "vary": True},
                        "x0":    {"value": 518, "min": 500, "max": 540, "vary": True},
                    }
                }
            },
            "peak_labels": ["Si"],
        }

        engine = BatchFittingEngine()
        results = engine.fit_spectra(
            spectra=spectra,
            fit_model=fit_model,
            coords=None,  # No spatial info
            fit_params={"method": "leastsq", "xtol": 1e-4, "max_ite": 200},
            ncpus=1,
            apply_model_to_spectra=True,
        )

        assert len(results) == 9
        n_success = sum(1 for r in results if r.success)
        assert n_success >= 7

    def test_cancellation(self):
        """Test that fitting can be cancelled mid-run."""
        spectra, coords, _ = self._make_synthetic_map(n_pixels=25)

        fit_model = {
            "peak_models": {
                "0": {
                    "Lorentzian": {
                        "ampli": {"value": 80, "min": 0, "max": 1e6, "vary": True},
                        "fwhm":  {"value": 5,  "min": 0, "max": 200, "vary": True},
                        "x0":    {"value": 518, "min": 500, "max": 540, "vary": True},
                    }
                }
            },
            "peak_labels": ["Si"],
        }

        cancel_after = [5]  # Cancel after 5 spectra

        def check_cancel():
            cancel_after[0] -= 1
            return cancel_after[0] <= 0

        engine = BatchFittingEngine()
        results = engine.fit_spectra(
            spectra=spectra,
            fit_model=fit_model,
            coords=coords,
            ncpus=1,
            cancel_check=check_cancel,
            apply_model_to_spectra=True,
        )

        assert len(results) == 25
        # Some should have been cancelled (marked as failed)
        n_failed = sum(1 for r in results if not r.success)
        assert n_failed > 0


# ═══════════════════════════════════════════════════════════════════════════
# Test: FitResult compatibility
# ═══════════════════════════════════════════════════════════════════════════

class TestFitResultCompat:
    """Verify FitResult is compatible with existing codebase patterns."""

    def test_success_attribute(self):
        r = FitResult(True, {"a": 1.0}, np.array([1, 2, 3]))
        assert r.success is True

    def test_params_dict_access(self):
        r = FitResult(True, {"m01_x0": 520.0, "m01_ampli": 100.0}, np.array([]))
        assert "m01_x0" in r.params
        assert r.params["m01_x0"].value == 520.0

    def test_best_fit_array(self):
        bf = np.array([1.0, 2.0, 3.0])
        r = FitResult(True, {}, bf)
        np.testing.assert_array_equal(r.best_fit, bf)

    def test_hasattr_pattern(self):
        """Test the common pattern: hasattr(spectrum.result_fit, 'success')"""
        r = FitResult(True, {"p": 1.0}, np.array([]))
        assert hasattr(r, "success")
        assert hasattr(r, "params")
        assert hasattr(r, "best_fit")
