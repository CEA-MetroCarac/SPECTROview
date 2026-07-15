"""Unit tests for fit_engine/noise.py."""
import numpy as np

from spectroview.fit_engine.noise import detect_noise_level


class TestDetectNoiseLevel:
    def test_zero_for_constant_signal(self):
        y = np.full(200, 5.0)
        assert detect_noise_level(y) == 0.0

    def test_scales_with_gaussian_noise_amplitude(self):
        # For iid N(0, sigma^2) samples, first differences have std sigma*sqrt(2),
        # and median(|dy|)/0.6745 recovers that std, so the extra *sqrt(2) factor
        # in detect_noise_level makes it converge to ~2*sigma (verified empirically
        # below), not sigma itself -- it's a relative noise-floor heuristic, not
        # an unbiased sigma estimator.
        rng = np.random.default_rng(42)
        y_small = rng.normal(0, 1.0, size=5000)
        y_large = rng.normal(0, 10.0, size=5000)
        small = detect_noise_level(y_small)
        large = detect_noise_level(y_large)
        assert large > small
        np.testing.assert_allclose(small, 2.0, atol=0.3)
        np.testing.assert_allclose(large, 20.0, rtol=0.15)
        # Linear scaling with sigma:
        np.testing.assert_allclose(large / small, 10.0, rtol=0.2)

    def test_robust_to_outliers(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1.0, size=2000)
        y_with_outliers = y.copy()
        y_with_outliers[::100] += 1000.0  # 1% gross outliers
        clean_level = detect_noise_level(y)
        outlier_level = detect_noise_level(y_with_outliers)
        # Median-based estimator should barely move despite large outliers
        assert outlier_level < clean_level * 3

    def test_returns_python_float(self):
        y = np.linspace(0, 1, 50)
        assert isinstance(detect_noise_level(y), float)
