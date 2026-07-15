"""Unit tests for fit_engine/models.py - batched peak-shape functions.

Covers every shape registered in BATCHED_MODELS: value parity against the
scalar reference implementation (scalar_models.py), analytical Jacobians
against finite differences, registry consistency with spectroview.PEAK_MODELS,
and degenerate-parameter edge cases (zero/negative width, 2D per-spectrum x).
"""
import numpy as np
import pytest

from spectroview import PEAK_MODELS
from spectroview.fit_engine.models import BATCHED_MODELS, numerical_jacobian
from spectroview.fit_engine.scalar_models import PEAK_MODEL_REGISTRY

ALL_SHAPES = list(BATCHED_MODELS.keys())

# Representative, well-conditioned parameter values (away from singularities)
# for each shape, keyed by canonical parameter name.
_SAMPLE_PARAMS = {
    "Lorentzian": {"ampli": 120.0, "fwhm": 6.0, "x0": 500.0},
    "Gaussian": {"ampli": 120.0, "fwhm": 6.0, "x0": 500.0},
    "PseudoVoigt": {"ampli": 120.0, "fwhm": 6.0, "x0": 500.0, "alpha": 0.4},
    "GaussianAsym": {"ampli": 120.0, "fwhm_l": 5.0, "fwhm_r": 9.0, "x0": 500.0},
    "LorentzianAsym": {"ampli": 120.0, "fwhm_l": 5.0, "fwhm_r": 9.0, "x0": 500.0},
    "Fano": {"ampli": 120.0, "fwhm": 6.0, "x0": 500.0, "q": 3.0},
    "DecaySingleExp": {"A": 100.0, "tau": 5.0, "B": 2.0},
    "DecayBiExp": {"A1": 60.0, "tau1": 2.0, "A2": 40.0, "tau2": 10.0, "B": 1.0},
}


def _params_row(shape, overrides=None):
    canonical = BATCHED_MODELS[shape][2]
    vals = dict(_SAMPLE_PARAMS[shape])
    if overrides:
        vals.update(overrides)
    return np.array([[vals[name] for name in canonical]])


@pytest.fixture
def x_axis():
    return np.linspace(300.0, 700.0, 200)


@pytest.fixture
def decay_x_axis():
    # Decay shapes are defined for x >= 0 (time axis), not a Raman shift axis.
    return np.linspace(0.0, 50.0, 200)


def _axis_for(shape, decay_x_axis, x_axis):
    return decay_x_axis if shape.startswith("Decay") else x_axis


class TestRegistry:
    def test_all_peak_models_constant_are_registered(self):
        assert set(PEAK_MODELS) == set(BATCHED_MODELS.keys())

    def test_scalar_registry_matches_batched_registry_shapes(self):
        assert set(PEAK_MODEL_REGISTRY.keys()) == set(BATCHED_MODELS.keys())

    @pytest.mark.parametrize("shape", ALL_SHAPES)
    def test_registry_entry_shape(self, shape):
        eval_fn, jac_fn, canonical = BATCHED_MODELS[shape]
        assert callable(eval_fn)
        assert callable(jac_fn)
        assert canonical == list(PEAK_MODEL_REGISTRY[shape][1])

    @pytest.mark.parametrize("shape", ALL_SHAPES)
    def test_sample_params_cover_all_canonical_names(self, shape):
        canonical = BATCHED_MODELS[shape][2]
        assert set(canonical) == set(_SAMPLE_PARAMS[shape].keys())


class TestEvalMatchesScalarReference:
    """The batched (tensor) and scalar (lmfit-parity) implementations should
    agree numerically for well-conditioned parameters, since both docstrings
    claim to implement 'the exact functional forms used by the lmfit engine'.
    """

    @pytest.mark.parametrize("shape", ALL_SHAPES)
    def test_single_spectrum_matches_scalar(self, shape, x_axis, decay_x_axis):
        x = _axis_for(shape, decay_x_axis, x_axis)
        params = _params_row(shape)
        eval_fn, _, canonical = BATCHED_MODELS[shape]
        y_batched = eval_fn(x, params)[0]

        scalar_fn, _ = PEAK_MODEL_REGISTRY[shape]
        args = [_SAMPLE_PARAMS[shape][name] for name in canonical]
        y_scalar = scalar_fn(x, *args)

        np.testing.assert_allclose(y_batched, y_scalar, rtol=1e-3, atol=1e-6)

    @pytest.mark.parametrize("shape", ALL_SHAPES)
    def test_batched_multi_spectrum_matches_row_by_row(self, shape, x_axis, decay_x_axis):
        x = _axis_for(shape, decay_x_axis, x_axis)
        row1 = _params_row(shape)[0]
        scale_key = next(k for k in ("ampli", "A", "A1") if k in _SAMPLE_PARAMS[shape])
        row2 = _params_row(shape, {scale_key: _SAMPLE_PARAMS[shape][scale_key] * 1.5})[0]
        params = np.vstack([row1, row2])
        eval_fn = BATCHED_MODELS[shape][0]

        Y = eval_fn(x, params)
        assert Y.shape == (2, len(x))
        np.testing.assert_allclose(Y[0], eval_fn(x, row1[None, :])[0])
        np.testing.assert_allclose(Y[1], eval_fn(x, row2[None, :])[0])


class TestAnalyticalJacobian:
    @pytest.mark.parametrize("shape", ALL_SHAPES)
    def test_matches_finite_difference(self, shape, x_axis, decay_x_axis):
        x = _axis_for(shape, decay_x_axis, x_axis)
        eval_fn, jac_fn, canonical = BATCHED_MODELS[shape]
        params = _params_row(shape)

        J_analytical = jac_fn(x, params)
        J_numerical = numerical_jacobian(eval_fn, x, params, eps=1e-6)

        assert J_analytical.shape == (1, len(x), len(canonical))
        np.testing.assert_allclose(J_analytical, J_numerical, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("shape", ALL_SHAPES)
    def test_matches_finite_difference_multi_spectrum(self, shape, x_axis, decay_x_axis):
        x = _axis_for(shape, decay_x_axis, x_axis)
        eval_fn, jac_fn, canonical = BATCHED_MODELS[shape]
        base = _params_row(shape)[0]
        scaled = base * 1.3
        params = np.vstack([base, scaled])

        J_analytical = jac_fn(x, params)
        J_numerical = numerical_jacobian(eval_fn, x, params, eps=1e-6)
        np.testing.assert_allclose(J_analytical, J_numerical, rtol=1e-4, atol=1e-5)


class TestDegenerateParameters:
    """Zero/negative widths must not produce NaN/Inf: all width params are
    clamped via max(abs(w), eps) before use."""

    @pytest.mark.parametrize("shape", ALL_SHAPES)
    @pytest.mark.parametrize("width_value", [0.0, -5.0])
    def test_zero_or_negative_width_is_finite(self, shape, width_value, x_axis, decay_x_axis):
        x = _axis_for(shape, decay_x_axis, x_axis)
        width_keys = [k for k in _SAMPLE_PARAMS[shape] if k in ("fwhm", "fwhm_l", "fwhm_r", "tau", "tau1", "tau2")]
        if not width_keys:
            pytest.skip(f"{shape} has no width-like parameter")
        overrides = {k: width_value for k in width_keys}
        params = _params_row(shape, overrides)

        eval_fn, jac_fn, _ = BATCHED_MODELS[shape]
        Y = eval_fn(x, params)
        J = jac_fn(x, params)
        assert np.isfinite(Y).all()
        assert np.isfinite(J).all()

    @pytest.mark.parametrize("shape", ALL_SHAPES)
    def test_negative_width_matches_positive_abs_value(self, shape, x_axis, decay_x_axis):
        x = _axis_for(shape, decay_x_axis, x_axis)
        width_keys = [k for k in _SAMPLE_PARAMS[shape] if k in ("fwhm", "fwhm_l", "fwhm_r", "tau", "tau1", "tau2")]
        if not width_keys:
            pytest.skip(f"{shape} has no width-like parameter")

        eval_fn = BATCHED_MODELS[shape][0]
        pos = _params_row(shape)
        neg_overrides = {k: -_SAMPLE_PARAMS[shape][k] for k in width_keys}
        neg = _params_row(shape, neg_overrides)

        np.testing.assert_allclose(eval_fn(x, pos), eval_fn(x, neg))


class TestPerSpectrumXAxis:
    """Some maps have uncalibrated, per-spectrum x-axes (x is 2D, shape (N, M))."""

    @pytest.mark.parametrize("shape", ALL_SHAPES)
    def test_2d_x_matches_1d_when_rows_identical(self, shape, x_axis, decay_x_axis):
        x1d = _axis_for(shape, decay_x_axis, x_axis)
        params = np.vstack([_params_row(shape)[0], _params_row(shape)[0]])
        x2d = np.vstack([x1d, x1d])

        eval_fn, jac_fn, _ = BATCHED_MODELS[shape]
        Y_1d = eval_fn(x1d, params)
        Y_2d = eval_fn(x2d, params)
        np.testing.assert_allclose(Y_1d, Y_2d)

        J_1d = jac_fn(x1d, params)
        J_2d = jac_fn(x2d, params)
        np.testing.assert_allclose(J_1d, J_2d)


class TestNumericalJacobianHelper:
    def test_relative_perturbation_scales_with_parameter_magnitude(self, x_axis):
        eval_fn = BATCHED_MODELS["Lorentzian"][0]
        small = np.array([[1.0, 6.0, 500.0]])
        large = np.array([[1e6, 6.0, 500.0]])

        J_small = numerical_jacobian(eval_fn, x_axis, small)
        J_large = numerical_jacobian(eval_fn, x_axis, large)
        assert np.isfinite(J_small).all()
        assert np.isfinite(J_large).all()
        # dY/d(ampli) is exactly linear in ampli for Lorentzian -> ratio ~1 regardless of scale
        np.testing.assert_allclose(J_small[0, :, 0], J_large[0, :, 0], rtol=1e-4)
