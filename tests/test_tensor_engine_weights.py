import numpy as np

from spectroview.fit_engine.tensor_engine import TensorFittingEngine
from spectroview.model.m_spectrum import MSpectrum
from spectroview.viewmodel.utils import apply_custom_fit_model


def _build_sample_spectrum(name, x, y):
    spectrum = MSpectrum()
    spectrum.fname = name
    spectrum.x0 = x.copy()
    spectrum.y0 = y.copy()
    spectrum.x = x.copy()
    spectrum.y = y.copy()
    spectrum.baseline.mode = "Linear"
    return spectrum


def _extract_param_values(result):
    return {k: getattr(v, "value", v) for k, v in result.params.items()}


def _legacy_fit_spectrum(spectrum, fit_model, fit_kwargs):
    apply_custom_fit_model(spectrum, fit_model, spectrum.fname)
    spectrum.preprocess()
    spectrum.fit(**fit_kwargs)
    return spectrum.result_fit


def _tensor_fit_spectrum(spectrum, fit_model, fit_params):
    engine = TensorFittingEngine()
    [result] = engine.fit_spectra(
        spectra=[spectrum],
        fit_model=fit_model,
        fit_params=fit_params,
        apply_model_to_spectra=True,
    )
    return result


def test_tensor_engine_masks_negative_points_with_weights():
    """Ensure tensor fitting masks negative points when fit_negative=False."""
    x = np.linspace(250.0, 350.0, 80)
    y = 120.0 / (1.0 + ((x - 300.0) / 12.0) ** 2)
    y[10:20] -= 200.0

    spectrum = _build_sample_spectrum("masked_negative", x, y)

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

    result = _tensor_fit_spectrum(
        spectrum,
        fit_model,
        {
            "method": "leastsq",
            "xtol": 1e-4,
            "ftol": 1e-4,
            "max_ite": 100,
            "fit_negative": False,
        },
    )

    assert result.success is True
    masked_indices = spectrum.y < 0
    assert masked_indices.sum() > 0
    assert np.all(result.best_fit[masked_indices] == 0.0)
    assert np.all(spectrum.result_fit.best_fit[masked_indices] == 0.0)


def test_tensor_engine_matches_legacy_fitspy_single_spectrum():
    """Compare TensorFittingEngine to legacy Spectrum.fit() on a single spectrum."""
    rng = np.random.default_rng(0)
    x = np.linspace(400.0, 600.0, 250)
    y_true = 120.0 / (1.0 + ((x - 510.0) / 8.0) ** 2)
    y = y_true + rng.normal(0.0, 0.7, size=x.shape)

    fit_model = {
        "peak_models": {
            "0": {
                "Lorentzian": {
                    "ampli": {"value": 100.0, "min": 0.0, "max": 1e6, "vary": True},
                    "fwhm":  {"value": 12.0,  "min": 0.0, "max": 200.0, "vary": True},
                    "x0":    {"value": 505.0, "min": 480.0, "max": 540.0, "vary": True},
                }
            }
        },
        "peak_labels": ["Peak"],
    }

    legacy_spectrum = _build_sample_spectrum("legacy_single", x, y)
    tensor_spectrum = _build_sample_spectrum("tensor_single", x, y)

    legacy_result = _legacy_fit_spectrum(
        legacy_spectrum,
        fit_model,
        {
            "fit_method": "leastsq",
            "fit_negative": False,
            "fit_outliers": False,
            "coef_noise": 0,
            "xtol": 1e-4,
            "max_ite": 200,
            "reinit_guess": False,
        },
    )

    tensor_result = _tensor_fit_spectrum(
        tensor_spectrum,
        fit_model,
        {
            "method": "leastsq",
            "xtol": 1e-4,
            "ftol": 1e-4,
            "max_ite": 200,
            "fit_negative": False,
            "fit_outliers": False,
            "coef_noise": 0,
        },
    )

    assert legacy_result.success is True
    assert tensor_result.success is True

    legacy_values = _extract_param_values(legacy_result)
    tensor_values = _extract_param_values(tensor_result)

    for key, legacy_value in legacy_values.items():
        assert key in tensor_values
        tensor_value = tensor_values[key]
        assert np.isfinite(tensor_value), f"Tensor fit returned non-finite value for {key}"
        assert np.isfinite(legacy_value), f"Legacy fit returned non-finite value for {key}"
        assert abs(tensor_value - legacy_value) <= max(1e-2, 0.1 * abs(legacy_value))

    legacy_cost = np.linalg.norm(legacy_spectrum.y - legacy_result.best_fit)
    tensor_cost = np.linalg.norm(tensor_spectrum.y - tensor_result.best_fit)
    assert tensor_cost <= legacy_cost * 1.1 + 1e-6

    np.testing.assert_allclose(
        tensor_result.best_fit,
        legacy_result.best_fit,
        rtol=0.1,
        atol=0.5,
    )


def test_tensor_engine_matches_legacy_fitspy_batch_quality():
    """Compare TensorFittingEngine to legacy Spectrum.fit() across several spectra."""
    rng = np.random.default_rng(1)
    x = np.linspace(420.0, 520.0, 200)

    fit_model = {
        "peak_models": {
            "0": {
                "Lorentzian": {
                    "ampli": {"value": 80.0, "min": 0.0, "max": 1e6, "vary": True},
                    "fwhm":  {"value": 10.0,  "min": 0.0, "max": 200.0, "vary": True},
                    "x0":    {"value": 470.0, "min": 440.0, "max": 500.0, "vary": True},
                }
            }
        },
        "peak_labels": ["Peak"],
    }

    legacy_results = []
    tensor_spectra = []
    for idx, x0_shift in enumerate([0.0, -5.0, 5.0, -3.0]):
        y_true = 120.0 / (1.0 + ((x - (470.0 + x0_shift)) / 10.0) ** 2)
        y = y_true + rng.normal(0.0, 0.8, size=x.shape)

        legacy_spectrum = _build_sample_spectrum(f"legacy_batch_{idx}", x, y)
        tensor_spectrum = _build_sample_spectrum(f"tensor_batch_{idx}", x, y)
        tensor_spectra.append(tensor_spectrum)

        legacy_results.append(
            _legacy_fit_spectrum(
                legacy_spectrum,
                fit_model,
                {
                    "fit_method": "leastsq",
                    "fit_negative": False,
                    "fit_outliers": False,
                    "coef_noise": 0,
                    "xtol": 1e-4,
                    "max_ite": 200,
                    "reinit_guess": False,
                },
            )
        )

    tensor_results = TensorFittingEngine().fit_spectra(
        spectra=tensor_spectra,
        fit_model=fit_model,
        fit_params={
            "method": "leastsq",
            "xtol": 1e-4,
            "ftol": 1e-4,
            "max_ite": 200,
            "fit_negative": False,
            "fit_outliers": False,
            "coef_noise": 0,
        },
        apply_model_to_spectra=True,
    )

    assert len(tensor_results) == len(legacy_results)

    total_legacy_cost = 0.0
    total_tensor_cost = 0.0
    for legacy_result, tensor_result, spectrum in zip(legacy_results, tensor_results, tensor_spectra):
        assert legacy_result.success is True
        assert tensor_result.success is True
        total_legacy_cost += np.linalg.norm(spectrum.y - legacy_result.best_fit)
        total_tensor_cost += np.linalg.norm(spectrum.y - tensor_result.best_fit)

    assert total_tensor_cost <= total_legacy_cost * 1.1 + 1e-6
