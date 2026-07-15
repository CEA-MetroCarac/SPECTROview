"""Performance/regression benchmark: 1_CL_map.txt + 1_CL_map.json.

16384 spectra x 1024 points, a single Lorentzian peak, well-converging
dataset (100% convergence at time of writing). Reference values below were
captured directly from the current VBF engine (see docs/log.md for the
matmul/einsum optimization history that can shift values at the sub-percent
level) -- tolerances are set to catch real regressions in accuracy, speed,
or convergence stability, not to lock in bit-exact numerics forever.

Marked slow: loads a 97MB file and fits 16384 spectra twice (~10-30s total).
Run with `pytest -m "not slow"` to skip during normal development.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.slow

N_SPECTRA = 16384
N_POINTS = 1024

# Captured reference values (see tests/performance/conftest.py for how these
# runs are produced). Index 0 and a mid-map index (8192), chosen for
# reasonable spatial spread across the map.
REFERENCE = {
    0: {"P1_ampli": 135.136547, "P1_fwhm": 8.101075, "P1_x0": 390.144938, "r2": 0.99091},
    8192: {"P1_ampli": 64.279202, "P1_fwhm": 7.625874, "P1_x0": 390.448433, "r2": 0.97298},
}


def _params_dict(md, row):
    return dict(zip(md.param_names, md.peak_params[row]))


class TestShapeAndConvergence:
    def test_dataset_shape_matches_expected(self, cl_map_benchmark):
        run = cl_map_benchmark["run1"]
        assert run["n_spectra"] == N_SPECTRA
        assert run["n_points"] == N_POINTS

    def test_convergence_rate_stays_near_total(self, cl_map_benchmark):
        md = cl_map_benchmark["run1"]["md"]
        n_converged = int(np.sum(md.fit_success))
        assert n_converged >= int(0.999 * N_SPECTRA), (
            f"Convergence regressed: {n_converged}/{N_SPECTRA} converged "
            f"(expected effectively all, dataset is well-conditioned)"
        )

    def test_mean_r2_of_converged_spectra_stays_high(self, cl_map_benchmark):
        md = cl_map_benchmark["run1"]["md"]
        good_r2 = md.fit_r2[md.fit_success]
        assert len(good_r2) > 0
        assert np.mean(good_r2) > 0.95, f"Mean R^2 regressed: {np.mean(good_r2):.4f}"


class TestAccuracyAgainstReference:
    @pytest.mark.parametrize("row", sorted(REFERENCE.keys()))
    def test_fitted_params_match_reference(self, cl_map_benchmark, row):
        md = cl_map_benchmark["run1"]["md"]
        assert md.fit_success[row]
        result = _params_dict(md, row)
        ref = REFERENCE[row]

        assert result["P1_x0"] == pytest.approx(ref["P1_x0"], abs=0.5)
        assert result["P1_fwhm"] == pytest.approx(ref["P1_fwhm"], abs=1.0)
        assert result["P1_ampli"] == pytest.approx(ref["P1_ampli"], rel=0.05)
        assert md.fit_r2[row] == pytest.approx(ref["r2"], abs=0.02)

    def test_all_fitted_x0_stay_within_the_model_bounds(self, cl_map_benchmark):
        """x0 was constrained to seed +/- 20 in the JSON model; a fit that
        violates its own bounds would indicate a broken optimizer."""
        md = cl_map_benchmark["run1"]["md"]
        idx_x0 = md.param_names.index("P1_x0")
        x0_seed = 390.153554604542  # from 1_CL_map.json
        x0_vals = md.peak_params[:, idx_x0]
        assert x0_vals.min() >= x0_seed - 20 - 1e-6
        assert x0_vals.max() <= x0_seed + 20 + 1e-6

    def test_fwhm_never_negative(self, cl_map_benchmark):
        md = cl_map_benchmark["run1"]["md"]
        idx_fwhm = md.param_names.index("P1_fwhm")
        assert (md.peak_params[:, idx_fwhm] >= 0).all()


class TestReproducibility:
    def test_two_independent_runs_are_bit_identical(self, cl_map_benchmark):
        """batched_levenberg_marquardt has no randomness anywhere, so two
        fits of the same data with the same model should be exactly equal,
        not just close -- any drift here indicates nondeterminism (e.g. an
        uninitialized buffer, dict-ordering dependency, or a race)."""
        md1 = cl_map_benchmark["run1"]["md"]
        md2 = cl_map_benchmark["run2"]["md"]
        np.testing.assert_array_equal(md1.peak_params, md2.peak_params)
        np.testing.assert_array_equal(md1.fit_success, md2.fit_success)
        np.testing.assert_array_equal(md1.fit_r2, md2.fit_r2)


class TestSpeed:
    def test_total_fit_time_within_budget(self, cl_map_benchmark):
        # Generous margin (~5-10x) over observed times (~7-18s) to absorb
        # slower CI hardware while still catching a real regression (e.g.
        # reverting the JᵀJ einsum->matmul optimization was a 13-20x slowdown).
        elapsed = cl_map_benchmark["run2"]["elapsed"]
        assert elapsed < 90.0, f"Full-map fit took {elapsed:.1f}s (budget: 90s)"

    def test_per_spectrum_time_within_budget(self, cl_map_benchmark):
        elapsed = cl_map_benchmark["run2"]["elapsed"]
        ms_per_spectrum = elapsed / N_SPECTRA * 1000
        assert ms_per_spectrum < 5.0, f"{ms_per_spectrum:.3f} ms/spectrum (budget: 5.0)"
