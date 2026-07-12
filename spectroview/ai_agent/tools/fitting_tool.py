"""tools/fitting_tool.py

Reusable peak fitting utilities for the SPECTROview AI Agent.

These helpers provide parameter validation, model metadata, and
fit-result analysis utilities. The AI Agent itself cannot trigger
fitting — it only queries and visualises fit results stored in
DataFrames. These utilities support interpreting and validating
those results.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Peak model registry
# ---------------------------------------------------------------------------

# Each entry: model_name → {parameters, description}
PEAK_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "Gaussian": {
        "parameters": ["ampli", "x0", "fwhm"],
        "description": "Symmetric Gaussian peak. G = ampli · exp(−4·ln2·((x−x0)/fwhm)²)",
        "shape": "symmetric",
    },
    "Lorentzian": {
        "parameters": ["ampli", "x0", "fwhm"],
        "description": "Symmetric Lorentzian (Cauchy) peak. L = ampli / (1 + 4·((x−x0)/fwhm)²)",
        "shape": "symmetric",
    },
    "PseudoVoigt": {
        "parameters": ["ampli", "x0", "fwhm", "alpha"],
        "description": (
            "Mixed Gaussian/Lorentzian. PV = alpha·G + (1−alpha)·L. "
            "alpha=1 → pure Gaussian, alpha=0 → pure Lorentzian."
        ),
        "shape": "symmetric",
    },
    "GaussianAsym": {
        "parameters": ["ampli", "x0", "fwhm_l", "fwhm_r"],
        "description": "Piecewise asymmetric Gaussian with independent left (fwhm_l) and right (fwhm_r) widths.",
        "shape": "asymmetric",
    },
    "LorentzianAsym": {
        "parameters": ["ampli", "x0", "fwhm_l", "fwhm_r"],
        "description": "Piecewise asymmetric Lorentzian with independent left (fwhm_l) and right (fwhm_r) widths.",
        "shape": "asymmetric",
    },
    "Fano": {
        "parameters": ["ampli", "x0", "fwhm", "q"],
        "description": (
            "Fano resonance profile. f = ampli·(q + ε)²/(1 + ε²), ε = 2·(x−x0)/fwhm. "
            "q is the Fano asymmetry parameter."
        ),
        "shape": "asymmetric",
    },
    "DecaySingleExp": {
        "parameters": ["A", "tau", "B"],
        "description": "Single exponential decay. y = A·exp(−t/tau) + B.",
        "shape": "decay",
    },
    "DecayBiExp": {
        "parameters": ["A1", "tau1", "A2", "tau2", "B"],
        "description": "Bi-exponential decay. y = A1·exp(−t/tau1) + A2·exp(−t/tau2) + B.",
        "shape": "decay",
    },
}


# ---------------------------------------------------------------------------
# Model queries
# ---------------------------------------------------------------------------

def list_peak_models() -> list[str]:
    """Return the list of all supported peak model names.

    Returns
    -------
    list[str]
        Model names in registration order.
    """
    return list(PEAK_MODEL_REGISTRY.keys())


def get_model_info(model_name: str) -> Optional[dict[str, Any]]:
    """Return metadata for a peak model, or None if not found.

    Parameters
    ----------
    model_name:
        Exact model name (case-sensitive), e.g. ``"PseudoVoigt"``.

    Returns
    -------
    dict or None
        Dictionary with keys ``parameters``, ``description``, ``shape``.
    """
    return PEAK_MODEL_REGISTRY.get(model_name)


def get_model_parameters(model_name: str) -> list[str]:
    """Return the parameter names for a given peak model.

    Parameters
    ----------
    model_name:
        Exact model name.

    Returns
    -------
    list[str]
        Parameter names, or an empty list if the model is not found.
    """
    info = PEAK_MODEL_REGISTRY.get(model_name)
    if info is None:
        logger.warning("Unknown peak model: %r", model_name)
        return []
    return list(info["parameters"])


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def validate_peak_params(params: dict[str, Any], model_name: str) -> tuple[bool, list[str]]:
    """Validate that a parameter dictionary contains the required keys
    for the specified peak model.

    Parameters
    ----------
    params:
        Dictionary of parameter names → values (or constraint dicts).
    model_name:
        The peak model name (must be in PEAK_MODEL_REGISTRY).

    Returns
    -------
    tuple[bool, list[str]]
        ``(is_valid, list_of_errors)``.
        ``is_valid`` is True when no required parameters are missing.
    """
    expected = get_model_parameters(model_name)
    if not expected:
        return False, [f"Unknown peak model: '{model_name}'"]

    errors: list[str] = []
    for param in expected:
        if param not in params:
            errors.append(f"Missing required parameter '{param}' for model '{model_name}'")

    # Check that min < max for any bounded parameters
    for key, val in params.items():
        if isinstance(val, dict):
            lo = val.get("min")
            hi = val.get("max")
            if lo is not None and hi is not None:
                try:
                    if float(lo) >= float(hi):
                        errors.append(
                            f"Parameter '{key}': min ({lo}) must be less than max ({hi})"
                        )
                except (ValueError, TypeError):
                    pass

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Fit quality interpretation
# ---------------------------------------------------------------------------

def interpret_r_squared(r_squared: float) -> str:
    """Return a human-readable interpretation of an R² value.

    Parameters
    ----------
    r_squared:
        Coefficient of determination in [0, 1].

    Returns
    -------
    str
        Quality label: "excellent", "good", "acceptable", or "poor".
    """
    if r_squared >= 0.99:
        return "excellent"
    elif r_squared >= 0.95:
        return "good"
    elif r_squared >= 0.90:
        return "acceptable"
    else:
        return "poor"


def suggest_filter_for_quality(
    threshold: float = 0.95,
    column: str = "R_squared",
) -> str:
    """Return a pandas `.query()` expression to filter poor fits.

    Parameters
    ----------
    threshold:
        Minimum R² value to consider acceptable.
    column:
        Column name containing R² values in the DataFrame.

    Returns
    -------
    str
        A pandas `.query()` expression string.
    """
    return f"{column} < {threshold}"
