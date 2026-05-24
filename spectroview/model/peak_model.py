import numpy as np
from typing import Optional

def initialize_peak_params(peak_model: dict, peak_shape: str, x0: float, ampli: float, minfwhm: float, maxfwhm: float, maxshift: float, y_arr=None):
    """Build canonical parameters dictionary based on peak shape."""
    peak_model.clear()
    
    # Check for decay single exp or decay bi exp
    if peak_shape in ["DecaySingleExp", "DecayBiExp"]:
        if y_arr is None:
            y_arr = np.array([ampli])
        y_max = float(np.max(y_arr))
        y_min = float(np.min(y_arr))
        
        if peak_shape == "DecaySingleExp":
            peak_model["A"] = {"value": y_max, "min": 0, "max": y_max * 100, "vary": True}
            peak_model["tau"] = {"value": 5.0, "min": 0.1, "max": 100, "vary": True}
            peak_model["B"] = {"value": y_min, "min": 0, "max": y_min * 10, "vary": True}
        elif peak_shape == "DecayBiExp":
            peak_model["A1"] = {"value": y_max * 0.7, "min": 0, "max": y_max * 100, "vary": True}
            peak_model["tau1"] = {"value": 2.0, "min": 0.1, "max": 50, "vary": True}
            peak_model["A2"] = {"value": y_max * 0.3, "min": 0, "max": y_max * 100, "vary": True}
            peak_model["tau2"] = {"value": 10.0, "min": 0.1, "max": 100, "vary": True}
            peak_model["B"] = {"value": y_min, "min": 0, "max": y_min * 10, "vary": True}
        return

    peak_model["ampli"] = {"value": ampli, "min": 0.0, "max": ampli * 1e6, "vary": True}
    peak_model["x0"] = {"value": x0, "min": x0 - maxshift, "max": x0 + maxshift, "vary": True}
    
    if peak_shape in ["GaussianAsym", "LorentzianAsym"]:
        peak_model["fwhm_l"] = {"value": 5.0, "min": minfwhm, "max": maxfwhm, "vary": True}
        peak_model["fwhm_r"] = {"value": 5.0, "min": minfwhm, "max": maxfwhm, "vary": True}
    else:
        peak_model["fwhm"] = {"value": 5.0, "min": minfwhm, "max": maxfwhm, "vary": True}

    if peak_shape == "PseudoVoigt":
        peak_model["alpha"] = {"value": 0.5, "min": 0.0, "max": 1.0, "vary": True}
    elif peak_shape == "Fano":
        q_val = 50.0
        peak_model["q"] = {"value": q_val, "min": -200, "max": 200, "vary": True}
        peak_model["ampli"]["value"] = ampli / (q_val**2 + 1)

def make_peak_hint(shape: str, x0: float, ampli: float = 100.0,
                   fwhm: float = 10.0, **kwargs) -> dict:
    """Create a peak model hint dict."""
    hint = {
        "shape": shape,
        "x0": {"value": x0, "min": x0 - 20, "max": x0 + 20, "vary": True},
        "ampli": {"value": ampli, "min": 0, "max": 1e6, "vary": True},
        "fwhm": {"value": fwhm, "min": 0.01, "max": 200, "vary": True},
    }
    # Add any extra shape-specific parameters
    for k, v in kwargs.items():
        hint[k] = {"value": v["value"], "min": v.get("min", 0), "max": v.get("max", 1e6), "vary": v.get("vary", True)}
        
    return hint

def fit_model_to_dict(peak_hints: list[dict], baseline_config: dict,
                      bkg_model: Optional[dict] = None,
                      range_min=None, range_max=None,
                      peak_labels=None) -> dict:
    """Build the standard fit_model dict from components."""
    peak_models = {}
    for i, hint in enumerate(peak_hints):
        shape = hint.pop("shape")
        peak_models[str(i)] = {shape: hint}
        
    return {
        "peak_models": peak_models,
        "bkg_model": bkg_model,
        "baseline": baseline_config,
        "range_min": range_min,
        "range_max": range_max,
        "peak_labels": peak_labels or [str(i + 1) for i in range(len(peak_hints))],
    }
