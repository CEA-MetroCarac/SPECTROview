"""Performance/regression benchmark: 2_MoS2_map.txt + 2_fit_MoS2map_NEW.json.

1520 spectra x 575 points, three overlapping Lorentzian peaks, a pathological
dataset where a large fraction of pixels are background/low-signal and get
their amplitude/fwhm suppressed to zero by the noise-threshold mechanism
(coef_noise=1 in this model) -- mean R^2 across ALL pixels is therefore low
by design, so accuracy is checked on a specific real-peak pixel (index 739)
instead, matching the pre-refactor integration test's reference values
(re-verified against the current engine, see docs/log.md's einsum->matmul
note for why exact equality isn't asserted).

Marked slow: fits 1520 spectra twice (~1-2s total, small compared to the CL
map but still real I/O + fitting work).
"""
import numpy as np
import pytest

pytestmark = pytest.mark.slow

N_SPECTRA = 1520
N_POINTS = 575
TARGET_ROW = 739  # spectrum at (1764.6, 72.9) in the original GUI benchmarking

REFERENCE_ROW_739 = {
    "P1_x0": 384.241823, "P1_fwhm": 2.173862, "P1_ampli": 404.980100,
    "P2_x0": 403.847484, "P2_fwhm": 5.067702, "P2_ampli": 393.862359,
    "P3_x0": 446.330726, "P3_fwhm": 18.955638, "P3_ampli": 34.617652,
    "r2": 0.98487,
}


def _params_dict(md, row):
    return dict(zip(md.param_names, md.peak_params[row]))


class TestShapeAndConvergence:
    def test_dataset_shape_matches_expected(self, mos2_map_benchmark):
        run = mos2_map_benchmark["run1"]
        assert run["n_spectra"] == N_SPECTRA
        assert run["n_points"] == N_POINTS

    def test_convergence_count_stays_above_floor(self, mos2_map_benchmark):
        md = mos2_map_benchmark["run1"]["md"]
        n_converged = int(np.sum(md.fit_success))
        assert n_converged >= 1400, (
            f"Too few fits converged ({n_converged}/{N_SPECTRA}); this dataset "
            f"is pathological (many pixels hit max_ite) but should stay near 1435/1520"
        )

    def test_param_count_matches_three_peak_model(self, mos2_map_benchmark):
        md = mos2_map_benchmark["run1"]["md"]
        assert len(md.param_names) == 9  # 3 peaks x (ampli, fwhm, x0)


class TestAccuracyAgainstReference:
    def test_row_739_matches_gui_reference_fit(self, mos2_map_benchmark):
        md = mos2_map_benchmark["run1"]["md"]
        assert md.fit_success[TARGET_ROW]
        result = _params_dict(md, TARGET_ROW)

        for peak in ("P1", "P2", "P3"):
            assert result[f"{peak}_x0"] == pytest.approx(REFERENCE_ROW_739[f"{peak}_x0"], abs=0.5)
            assert result[f"{peak}_fwhm"] == pytest.approx(REFERENCE_ROW_739[f"{peak}_fwhm"], abs=2.0)
            assert result[f"{peak}_ampli"] == pytest.approx(REFERENCE_ROW_739[f"{peak}_ampli"], rel=0.05)

        assert md.fit_r2[TARGET_ROW] == pytest.approx(REFERENCE_ROW_739["r2"], abs=0.02)

    def test_noise_threshold_suppresses_background_pixels(self, mos2_map_benchmark):
        """Sanity-check the noise-suppression mechanism actually engaged on
        this dataset (coef_noise=1 in the model): background/low-signal rows
        should have ampli forced to exactly 0."""
        md = mos2_map_benchmark["run1"]["md"]
        idx_ampli = md.param_names.index("P1_ampli")
        n_suppressed = int(np.sum(md.peak_params[:, idx_ampli] == 0.0))
        assert n_suppressed > 0

    def test_median_r2_of_real_peaks_is_high(self, mos2_map_benchmark):
        """Restricting to rows where P1 wasn't noise-suppressed isolates the
        'real signal' pixels from the background ones for a meaningful
        aggregate accuracy metric."""
        md = mos2_map_benchmark["run1"]["md"]
        idx_ampli = md.param_names.index("P1_ampli")
        has_signal = md.peak_params[:, idx_ampli] > 0
        real_r2 = md.fit_r2[has_signal & md.fit_success]
        assert len(real_r2) > 50
        assert np.median(real_r2) > 0.8


class TestReproducibility:
    def test_two_independent_runs_are_bit_identical(self, mos2_map_benchmark):
        md1 = mos2_map_benchmark["run1"]["md"]
        md2 = mos2_map_benchmark["run2"]["md"]
        np.testing.assert_array_equal(md1.peak_params, md2.peak_params)
        np.testing.assert_array_equal(md1.fit_success, md2.fit_success)
        np.testing.assert_array_equal(md1.fit_r2, md2.fit_r2)


class TestSpeed:
    def test_total_fit_time_within_budget(self, mos2_map_benchmark):
        elapsed = mos2_map_benchmark["run2"]["elapsed"]
        assert elapsed < 15.0, f"Full-map fit took {elapsed:.1f}s (budget: 15s)"

    def test_per_spectrum_time_within_budget(self, mos2_map_benchmark):
        elapsed = mos2_map_benchmark["run2"]["elapsed"]
        ms_per_spectrum = elapsed / N_SPECTRA * 1000
        assert ms_per_spectrum < 5.0, f"{ms_per_spectrum:.3f} ms/spectrum (budget: 5.0)"
