import numpy as np

from spectroview.fit_engine.tensor_engine import TensorFittingEngine
from spectroview.model.m_spectrum import MSpectrum


def test_tensor_engine_masks_negative_points_with_weights():
    """Ensure tensor fitting masks negative points when fit_negative=False."""
    x = np.linspace(250.0, 350.0, 80)
    y = 120.0 / (1.0 + ((x - 300.0) / 12.0) ** 2)
    y[10:20] -= 200.0  # simulate negative outliers that should be excluded

    spectrum = MSpectrum()
    spectrum.fname = "masked_negative"
    spectrum.x0 = x.copy()
    spectrum.y0 = y.copy()
    spectrum.x = x.copy()
    spectrum.y = y.copy()
    spectrum.weights = np.ones_like(x)

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

    engine = TensorFittingEngine()
    results = engine.fit_spectra(
        spectra=[spectrum],
        fit_model=fit_model,
        fit_params={
            "method": "leastsq",
            "xtol": 1e-4,
            "ftol": 1e-4,
            "max_ite": 100,
            "fit_negative": False,
        },
        apply_model_to_spectra=True,
    )

    assert len(results) == 1
    assert results[0].success is True

    masked_indices = spectrum.y < 0
    assert masked_indices.sum() > 0
    assert np.all(results[0].best_fit[masked_indices] == 0.0)
    assert np.all(spectrum.result_fit.best_fit[masked_indices] == 0.0)
