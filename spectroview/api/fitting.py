"""Peak fitting: building fit models, running the vectorized batch fit
engine, and reading/writing fit-model JSON templates.

For a stateful session that tracks fit models per-spectrum and writes
results back into a persistent store, see `spectroview.api.workspace`.
"""
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from spectroview.api.exceptions import FitError, FitModelError, TemplateError
from spectroview.fit_engine.baseline import eval_baseline_batch
from spectroview.fit_engine.vbf_engine import VBFengine
from spectroview.fit_engine.weights import compute_fit_weights


def build_fit_model(peaks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a fit_model dict in the format expected by `fit_batch()` and the VBF engine.

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

    Raises:
        FitModelError: a peak entry is missing the required 'model' key.

    Example::

        fit_model = fitting.build_fit_model(
            peaks=[
                {
                    "model": "Lorentzian",
                    "x0":    {"value": 520.0, "min": 515.0, "max": 525.0},
                    "ampli": {"value": 1000.0, "min": 0.0, "max": 1e9},
                    "fwhm":  {"value": 3.0, "min": 0.5, "max": 10.0},
                },
            ]
        )
    """
    peak_models: Dict[str, Any] = {}

    for i, peak_def in enumerate(peaks):
        model_name = peak_def.get("model")
        if not model_name:
            raise FitModelError(f"Peak at index {i} is missing the required 'model' key.")

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
                if "vary" in pinfo:
                    hints["vary"] = bool(pinfo["vary"])
                elif "fix" in pinfo:
                    hints["vary"] = not bool(pinfo["fix"])
                if "expr" in pinfo:
                    hints["expr"] = str(pinfo["expr"])
                param_hints[pname] = hints
            else:
                param_hints[pname] = {"value": float(pinfo)}

        peak_models[str(i)] = {model_name: param_hints}

    return {"peak_models": peak_models}


def fit_batch(
    x: np.ndarray,
    Y: np.ndarray,
    fit_model: Dict[str, Any],
    weights: Optional[np.ndarray] = None,
    fit_params: Optional[Dict[str, Any]] = None,
    auto_weights: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Fit multiple spectra simultaneously using the vectorized batch-fit engine.

    Args:
        x: Wavenumber axis, shape (M,) or (N, M).
        Y: Intensity matrix, shape (N, M).
        fit_model: Dictionary defining the peak model and baseline.
            Use `build_fit_model()` to construct this from a simple peak list,
            or `load_fit_model_template()` to load a JSON file saved by the GUI.
        weights: Optional explicit fit weights, shape (N, M). If given, `auto_weights` is ignored.
        fit_params: Optimizer parameters (fit_negative, coef_noise, xtol, ftol, max_ite, ...).
        auto_weights: If True (default) and `weights` is None, weights are derived
            from `fit_params` exactly as the GUI does before every fit: points with
            negative intensity are excluded unless fit_params['fit_negative'] is
            True, and points below a noise floor (fit_params['coef_noise']) are
            excluded. Set False to fit every point with equal weight.
        progress_callback: optional callable(current, total) called during fitting.
        cancel_check: optional callable() -> bool; return True to abort mid-fit.

    Returns:
        dict with keys: 'params' (N,K), 'success' (N,), 'r_squared' (N,),
        'best_fits' (N,M), 'peaks' (list of (N,M) arrays, one per peak),
        'param_names' (list[str]).

    Raises:
        FitModelError: fit_model has no 'peak_models' entry.
        FitError: the fitting engine raised an exception.
    """
    if not fit_model.get("peak_models"):
        raise FitModelError("fit_model has no 'peak_models' entry — nothing to fit.")

    if weights is None and auto_weights:
        weights = compute_fit_weights(Y, fit_params or {})

    try:
        engine = VBFengine()
        p_full, success, rsquared, best_fits, Y_peaks, param_names = engine.fit_spectra(
            x, Y, fit_model, weights=weights, fit_params=fit_params,
            progress_callback=progress_callback, cancel_check=cancel_check,
        )
    except Exception as e:
        raise FitError(f"Batch fit failed: {e}") from e

    return {
        "params": p_full,
        "success": success,
        "r_squared": rsquared,
        "best_fits": best_fits,
        "peaks": Y_peaks,
        "param_names": param_names,
    }


def apply_fit_model(
    x: np.ndarray,
    Y: np.ndarray,
    fit_model: Dict[str, Any],
    fit_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Apply a full fit_model dict (range crop + baseline + peaks) and fit it.

    Replicates what the GUI does when you click "Apply Fit Model" or run a
    batch fit from a JSON template: crop to `fit_model['range_min']`/
    `['range_max']` if present, subtract `fit_model['baseline']` if it is set
    and marked `is_subtracted`, then fit.

    Args:
        x: Wavenumber axis, shape (M,).
        Y: Intensity matrix, shape (N, M).
        fit_model: A fit_model dict, typically loaded via `load_fit_model_template()`.
        fit_params: Optimizer parameters; if None, uses `fit_model.get('fit_params')`.

    Returns:
        Same dict as `fit_batch()`, plus 'x' (the cropped axis actually used)
        and 'Y' (the preprocessed matrix actually fitted).

    Raises:
        FitModelError: fit_model has no 'peak_models' entry, or cropping empties the range.
        FitError: the fitting engine raised an exception.
    """
    x_proc, Y_proc = x, Y

    range_min = fit_model.get("range_min")
    range_max = fit_model.get("range_max")
    if range_min is not None or range_max is not None:
        mask = np.ones_like(x_proc, dtype=bool)
        if range_min is not None:
            mask &= x_proc >= range_min
        if range_max is not None:
            mask &= x_proc <= range_max
        if not mask.any():
            raise FitModelError("range_min/range_max crop excludes all wavenumber points.")
        x_proc = x_proc[mask]
        Y_proc = Y_proc[:, mask]

    baseline_cfg = fit_model.get("baseline")
    if baseline_cfg and baseline_cfg.get("mode") and baseline_cfg.get("is_subtracted", True):
        Y_baseline = eval_baseline_batch(x_proc, Y_proc, baseline_cfg)
        Y_proc = Y_proc - Y_baseline

    result = fit_batch(x_proc, Y_proc, fit_model, fit_params=(fit_params or fit_model.get("fit_params")))
    result["x"] = x_proc
    result["Y"] = Y_proc
    return result


def list_fit_model_templates(folder: Union[str, Path]) -> List[str]:
    """List available *.json fit-model template filenames in `folder`."""
    from spectroview.model.m_fit_model_manager import MFitModelManager

    return MFitModelManager().scan_folder(str(folder))


def load_fit_model_template(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a fit_model dict from a template JSON file saved by the GUI or by
    `save_fit_model_template()`.

    Raises:
        TemplateError: the file is missing or not a valid fit-model template.
    """
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise TemplateError(f"Could not read fit-model template {path}: {e}") from e

    if "0" not in data:
        raise TemplateError(f"{path} is not a valid SPECTROview fit-model template (missing key '0').")
    return data["0"]


def save_fit_model_template(fit_model: Dict[str, Any], path: Union[str, Path]) -> Path:
    """Save `fit_model` to `path` in the same {"0": {...}} JSON shape the
    GUI's "Save Fit Model" button produces, so it round-trips through the
    GUI's fit-model manager.
    """
    path = Path(path)

    def _default(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        raise TypeError(f"{obj.__class__.__name__} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"0": fit_model}, f, indent=2, default=_default)
    return path
