"""Tests for spectroview.api.preprocessing -- thin array-level wrapper
contract (baseline math itself is covered by tests/unit/fit_engine/test_baseline.py)."""
import numpy as np

from spectroview.api import preprocessing


class TestCropSpectra:
    def test_no_bounds_returns_copies(self, synth_x):
        Y = np.random.rand(3, len(synth_x))
        x2, Y2 = preprocessing.crop_spectra(synth_x, Y)
        assert np.array_equal(x2, synth_x)
        assert np.array_equal(Y2, Y)
        assert x2 is not synth_x and Y2 is not Y

    def test_crops_to_range(self, synth_x):
        Y = np.random.rand(2, len(synth_x))
        x2, Y2 = preprocessing.crop_spectra(synth_x, Y, range_min=400.0, range_max=500.0)
        assert x2.min() >= 400.0
        assert x2.max() <= 500.0
        assert Y2.shape == (2, len(x2))

    def test_one_sided_bound(self, synth_x):
        x2, Y2 = preprocessing.crop_spectra(synth_x, np.zeros((1, len(synth_x))), range_min=600.0)
        assert x2.min() >= 600.0
        assert x2.max() == synth_x.max()


class TestNormalizeSpectra:
    def test_each_row_normalized_to_its_own_max(self):
        Y = np.array([[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]])
        Yn = preprocessing.normalize_spectra(Y)
        assert np.allclose(Yn.max(axis=1), 1.0)
        assert np.allclose(Yn[0], Y[0] / 4.0)
        assert np.allclose(Yn[1], Y[1] / 40.0)

    def test_zero_row_avoids_division_by_zero(self):
        Y = np.zeros((1, 5))
        Yn = preprocessing.normalize_spectra(Y)
        assert np.all(np.isfinite(Yn))
        assert np.array_equal(Yn, Y)


class TestSubtractBaseline:
    def test_linear_baseline_delegates_to_eval_baseline_batch(self, synth_x):
        Y = np.tile(synth_x * 0.1 + 5.0, (2, 1))  # pure linear ramp
        config = {"mode": "Linear", "attached": True, "sigma": 0,
                   "points": [[synth_x[0], synth_x[-1]], [Y[0, 0], Y[0, -1]]]}
        Y_corrected, Y_baseline = preprocessing.subtract_baseline(synth_x, Y, config)
        assert Y_corrected.shape == Y.shape
        assert Y_baseline.shape == Y.shape
        # A pure linear ramp fully explained by a linear baseline should leave ~0 residual.
        assert np.allclose(Y_corrected, 0.0, atol=1e-4)
