"""Pure spectral, coordinate, fit-model, and metadata helpers for ViewModels."""

from __future__ import annotations

from copy import deepcopy
import numpy as np


def closest_index(array, value):
    """Return index of the element in *array* closest to *value* (NaN-safe)."""
    diff = np.abs(array - value)
    if np.isnan(diff).any():
        try:
            return int(np.nanargmin(diff))
        except ValueError:
            return 0
    return int(diff.argmin())


def parse_coords_from_fname(fname: str) -> tuple[float, float] | None:
    """Parse (x, y) spatial coordinates from a map spectrum fname.

    Spectra are named ``"{map_name}_({x}, {y})"`` (see
    ``VMWorkspaceMaps._extract_spectra_from_map``). Returns the coordinates as a
    float tuple, or None if the fname carries no parseable ``(x, y)`` suffix.
    """
    if "(" not in fname or ")" not in fname:
        return None
    coords_str = fname[fname.rfind("(") + 1 : fname.rfind(")")]
    try:
        x_str, y_str = coords_str.split(",")
        return (float(x_str.strip()), float(y_str.strip()))
    except (ValueError, AttributeError):
        return None


def fano_display_amplitude(internal_ampli, q):
    """Convert a Fano peak's internal fit amplitude to its displayed peak height.

    The Fano lineshape's 'ampli' fit parameter scales the whole curve rather
    than the peak height directly (see fit_engine/models.py's batched_fano);
    the actual maximum intensity is ampli * (q**2 + 1). Used wherever the UI
    shows or edits a Fano peak's amplitude as an intuitive peak height
    instead of the raw fit parameter.
    """
    return internal_ampli * (q ** 2 + 1)


def fano_internal_amplitude(display_ampli, q):
    """Inverse of fano_display_amplitude(): displayed peak height -> internal fit amplitude."""
    return display_ampli / (q ** 2 + 1)


def build_clean_fit_model(fit_model):
    """Build a clean, consistently ordered fit model dict for serialization.

    Key order: fit_params, range_min, range_max, baseline, peak_labels, peak_models.
    Also rounds float values in fit_params to avoid QSettings float32 precision artifacts
    (e.g. 9.999999747378752e-06 -> 1e-05).
    """
    fit_params = fit_model.get("fit_params", {})
    if fit_params:
        cleaned_params = {}
        for k, v in fit_params.items():
            cleaned_params[k] = float(f"{v:.6g}") if isinstance(v, float) else v
        fit_params = cleaned_params

    baseline = fit_model.get("baseline")
    if baseline and isinstance(baseline, dict):
        baseline = deepcopy(baseline)
        baseline.pop("y_eval", None)
        if baseline.get("mode") == "Linear":
            baseline.pop("coef", None)
            baseline.pop("order_max", None)

    return {
        "fit_params": fit_params,
        "range_min": fit_model.get("range_min"),
        "range_max": fit_model.get("range_max"),
        "baseline": baseline,
        "peak_labels": fit_model.get("peak_labels", []),
        "peak_models": fit_model.get("peak_models", {}),
    }


def _clean_metadata_value(val):
    if val is None:
        return None
    s = str(val).strip()
    if s.lower() in ("", "unknown", "none", "nan", "n/a", "null"):
        return None
    return s


def extract_primary_metadata(metadata: dict | object | None) -> dict[str, str]:
    """Extract standard acquisition metadata fields from a dictionary or spectrum object.

    Returns a dict with keys:
    - 'Laser Wavelength'
    - 'Laser Power'
    - 'Exposure Time'
    - 'Accumulation'
    - 'Objective'

    Missing or unavailable values are gracefully represented as 'N/A'.
    """
    if metadata is None:
        raw_meta = {}
    elif isinstance(metadata, dict):
        raw_meta = metadata
    elif hasattr(metadata, "metadata") and isinstance(metadata.metadata, dict):
        raw_meta = metadata.metadata
    elif hasattr(metadata, "map_metadata") and isinstance(metadata.map_metadata, dict):
        raw_meta = metadata.map_metadata
    else:
        raw_meta = {}

    def get_first_match(candidate_keys):
        for k in candidate_keys:
            if k in raw_meta:
                v = _clean_metadata_value(raw_meta[k])
                if v is not None:
                    return k, v
        lower_map = {k.lower(): (k, v) for k, v in raw_meta.items()}
        for k in candidate_keys:
            lk = k.lower()
            if lk in lower_map:
                orig_k, orig_v = lower_map[lk]
                v = _clean_metadata_value(orig_v)
                if v is not None:
                    return orig_k, v
        return None, None

    # 1. Laser Wavelength
    k, v = get_first_match([
        "Laser Wavelength (nm)", "Laser Wavelength", "laser_wavelength",
        "EXCIT_LINE", "EXCIT LINE", "Wavelength (nm)", "Wavelength",
    ])
    if v:
        wavelength = v if v.lower().endswith("nm") else f"{v} nm"
    else:
        wavelength = "N/A"

    # 2. Laser Power
    k, v = get_first_match([
        "Laser Power (%)", "Laser Power (mW)", "Laser Power", "laser_power",
        "ND FILTER", "Power (%)", "Power (mW)", "Power", "POWER",
    ])
    if v:
        has_unit = "%" in v or "mw" in v.lower() or "w" in v.lower()
        if ("(%)" in (k or "") or "%" in (k or "")) and not has_unit:
            power = f"{v} %"
        elif "(mw)" in (k or "").lower() and not has_unit:
            power = f"{v} mW"
        else:
            power = v
    else:
        power = "N/A"

    # 3. Exposure Time
    k, v = get_first_match([
        "Exposure Time (s)", "Exposure Time", "exposure_time",
        "ACQ. TIME (S)", "Acquisition Time (s)", "Acquisition Time",
        "TIME", "TIME (S)", "ACQ. TIME", "ACQ TIME", "EXP TIME", "EXPOSURE TIME",
    ])
    if v:
        exposure = v if (v.lower().endswith("s") or v.lower().endswith("sec")) else f"{v} s"
    else:
        exposure = "N/A"

    # 4. Accumulation
    k, v = get_first_match([
        "Accumulations", "Accumulation", "accumulation_count", "ACCUMULATIONS",
        "ACCUM", "ACC", "NB ACCUM",
    ])
    accum = v if v else "N/A"

    # 5. Objective
    k, v = get_first_match([
        "Objective Used", "Objective", "objective_name", "OBJECTIV", "OBJECTIVE",
    ])
    obj = v if v else "N/A"

    return {
        "Laser Wavelength": wavelength,
        "Laser Power": power,
        "Exposure Time": exposure,
        "Accumulation": accum,
        "Objective": obj,
    }


def format_metadata_tooltip(metadata: dict | object | None) -> str:
    """Format primary metadata into a compact HTML tooltip."""
    fields = extract_primary_metadata(metadata)
    lines = [
        "<b>Metadata</b>",
        f"• Laser Wavelength: {fields['Laser Wavelength']}",
        f"• Laser Power: {fields['Laser Power']}",
        f"• Exposure Time: {fields['Exposure Time']}",
        f"• Accumulation: {fields['Accumulation']}",
        f"• Objective: {fields['Objective']}",
    ]
    return "<br>".join(lines)
