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

    def test_empty_or_single_element_array_returns_zero(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert detect_noise_level(np.array([])) == 0.0
            assert detect_noise_level(np.array([42.0])) == 0.0
            assert detect_noise_level(None) == 0.0


class TestBatchedNoiseAndMovingAverage:
    def test_moving_average_5_matches_edge_padded_reference_and_invalidates_on_mutation(self):
        from spectroview.fit_engine.noise import mad_noise, moving_average_5

        rng = np.random.default_rng(123)
        Y = rng.normal(10.0, 2.0, size=(64, 100))
        yp = np.pad(Y, ((0, 0), (2, 2)), mode="edge")
        ref_ma = (yp[:, 0:-4] + yp[:, 1:-3] + yp[:, 2:-2] + yp[:, 3:-1] + yp[:, 4:]) / 5.0
        ref_mad = np.median(np.abs(np.diff(Y, axis=1)), axis=1) / 0.6745 * np.sqrt(2)

        ma1 = moving_average_5(Y)
        mad1 = mad_noise(Y, axis=1)
        np.testing.assert_allclose(ma1, ref_ma, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(mad1, ref_mad, rtol=1e-14, atol=1e-14)

        # Repeated call on unmutated 2D matrix hits cache and returns identical result
        assert moving_average_5(Y) is ma1
        assert mad_noise(Y, axis=1) is mad1

        # In-place mutation of a single interior element must invalidate cache
        Y[17, 43] += 50.0
        yp2 = np.pad(Y, ((0, 0), (2, 2)), mode="edge")
        ref_ma2 = (yp2[:, 0:-4] + yp2[:, 1:-3] + yp2[:, 2:-2] + yp2[:, 3:-1] + yp2[:, 4:]) / 5.0
        ref_mad2 = np.median(np.abs(np.diff(Y, axis=1)), axis=1) / 0.6745 * np.sqrt(2)

        np.testing.assert_allclose(moving_average_5(Y), ref_ma2, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(mad_noise(Y, axis=1), ref_mad2, rtol=1e-14, atol=1e-14)

        # Short width M < 5 fallback works cleanly
        Y_short = rng.normal(0.0, 1.0, size=(4, 3))
        yp_s = np.pad(Y_short, ((0, 0), (2, 2)), mode="edge")
        ref_s = (yp_s[:, 0:-4] + yp_s[:, 1:-3] + yp_s[:, 2:-2] + yp_s[:, 3:-1] + yp_s[:, 4:]) / 5.0
        np.testing.assert_allclose(moving_average_5(Y_short), ref_s, rtol=1e-14, atol=1e-14)

