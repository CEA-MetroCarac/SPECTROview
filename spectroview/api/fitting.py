"""Vectorized batch fitting capabilities."""
import numpy as np
from typing import Dict, Any, List, Optional
from spectroview.fit_engine.vbf_engine import VBFengine

def fit_batch(x: np.ndarray, Y: np.ndarray, fit_model: Dict[str, Any], 
              weights: Optional[np.ndarray] = None, 
              fit_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Fit multiple spectra simultaneously using the VBF engine.
    
    Args:
        x: Wavenumber axis, shape (M,) or (N, M).
        Y: Intensity matrix, shape (N, M).
        fit_model: Dictionary defining the peak model and baseline.
            Use `build_fit_model()` to construct this from a simple peak list,
            or load a JSON file exported from the SPECTROview GUI.
        weights: Optional fit weights.
        fit_params: Optimizer parameters (xtol, ftol, max_ite).
        
    Returns:
        Dict containing 'params', 'success', 'r_squared', 'best_fits', 'peaks', 'param_names'.
    """
    engine = VBFengine()
    p_full, success, rsquared, best_fits, Y_peaks, param_names = engine.fit_spectra(
        x, Y, fit_model, weights=weights, fit_params=fit_params
    )
    
    return {
        "params": p_full,
        "success": success,
        "r_squared": rsquared,
        "best_fits": best_fits,
        "peaks": Y_peaks,
        "param_names": param_names
    }


def build_fit_model(peaks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a fit model dict in the format expected by `fit_batch()` and the VBF engine.

    This is the recommended way to define fit models in scripts, as it
    abstracts away the internal key naming conventions.

    Args:
        peaks: List of peak definition dicts. Each dict must have a ``"model"`` key
            (e.g. ``"Lorentzian"``, ``"Gaussian"``, ``"PseudoVoigt"``) and one entry
            per parameter (``x0``, ``ampli``, ``fwhm``, etc.). Each parameter entry
            is itself a dict with optional keys:

            - ``value`` (float) — initial guess (default ``1.0``)
            - ``min``   (float) — lower bound (default ``-inf``)
            - ``max``   (float) — upper bound (default ``+inf``)
            - ``vary``  (bool)  — whether to optimize this parameter (default ``True``)
            - ``expr``  (str)   — expression linking this parameter to another

    Returns:
        A ``fit_model`` dict ready to pass to `fit_batch()`.

    Example::

        fit_model = fitting.build_fit_model(
            peaks=[
                {
                    "model": "Lorentzian",
                    "x0":    {"value": 520.0, "min": 515.0, "max": 525.0},
                    "ampli": {"value": 1000.0, "min": 0.0, "max": 1e9},
                    "fwhm":  {"value": 3.0, "min": 0.5, "max": 10.0},
                },
                {
                    "model": "Gaussian",
                    "x0":    {"value": 600.0, "min": 580.0, "max": 620.0},
                    "ampli": {"value": 500.0, "min": 0.0, "max": 1e9},
                    "fwhm":  {"value": 15.0, "min": 5.0, "max": 50.0},
                },
            ]
        )
    """
    peak_models: Dict[str, Any] = {}

    for i, peak_def in enumerate(peaks):
        model_name = peak_def.get("model")
        if not model_name:
            raise ValueError(f"Peak at index {i} is missing the required 'model' key.")

        # Build parameter hints dict (all keys except 'model')
        param_hints: Dict[str, Any] = {}
        for pname, pinfo in peak_def.items():
            if pname == "model":
                continue
            if isinstance(pinfo, dict):
                hints: Dict[str, Any] = {}
                if "value" in pinfo:
                    hints["value"] = float(pinfo["value"])
                if "min" in pinfo:
                    hints["min"] = float(pinfo["min"])
                if "max" in pinfo:
                    hints["max"] = float(pinfo["max"])
                # Support both "vary" (internal) and "fix" (user-friendly alias)
                if "vary" in pinfo:
                    hints["vary"] = bool(pinfo["vary"])
                elif "fix" in pinfo:
                    hints["vary"] = not bool(pinfo["fix"])
                if "expr" in pinfo:
                    hints["expr"] = str(pinfo["expr"])
                param_hints[pname] = hints
            else:
                # Scalar shorthand: {"x0": 520.0} → {"x0": {"value": 520.0}}
                param_hints[pname] = {"value": float(pinfo)}

        # VBFevaluator expects: peak_models[str(i)] = {model_name: param_hints}
        peak_models[str(i)] = {model_name: param_hints}

    return {"peak_models": peak_models}

