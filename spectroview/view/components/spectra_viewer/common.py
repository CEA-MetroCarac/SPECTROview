"""Shared constants, style cache, palette helpers, and token utilities for SpectraViewer."""

from __future__ import annotations

from functools import lru_cache
import re
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np

from spectroview import DEFAULT_COLORS
from spectroview.i18n import tr

if TYPE_CHECKING:
    from PySide6.QtWidgets import QComboBox


@lru_cache(maxsize=8)
def get_cached_style_dict(style_path: str) -> dict:
    """Cache parsed mplstyle files to prevent disk re-reading on every plot/theme switch."""
    return dict(mpl.rc_params_from_file(style_path))


# Precompiled regex for numeric tokens and file extensions in filenames
_RE_NUMBERS = re.compile(r"\d+\.?\d*")
_RE_FILE_EXT = re.compile(r"\.[A-Za-z][A-Za-z0-9]*$")

SPECTRA_DISCRETE_PALETTES = (
    "DEFAULT_COLORS",
    "tab20", "tab20b", "tab20c", "Dark2", "Paired", "Accent",
)
SPECTRA_GRADIENT_PALETTES = (
    "viridis", "plasma", "jet", "cividis", "magma",
)
SPECTRA_COLOR_PALETTES = (
    SPECTRA_DISCRETE_PALETTES + SPECTRA_GRADIENT_PALETTES
)
SPECTRA_CUSTOM_PALETTES = {
    "DEFAULT_COLORS": DEFAULT_COLORS,
}

# A selection can contain thousands of spectra. The plot itself is batched,
# but constructing thousands of per-row Qt editors would be costly in either
# the Spectra or Maps workspace.
LEGEND_EDITOR_MAX_ROWS = 100


class _MockPeakModelObj:
    """Lightweight stand-in for tensor-mode plotting to carry a shape name."""

    __slots__ = ("name2",)

    def __init__(self, name):
        self.name2 = name


def get_colors_for_palette(palette_name: str, count: int | None = None) -> list[str]:
    """Return ``count`` colors sampled from a named spectrum palette.

    Discrete palettes repeat only after all of their distinct colors have
    been used. Gradient palettes are sampled evenly from end to end, which
    makes the spectrum order visible for time/temperature series.
    """
    requested_count = (
        len(DEFAULT_COLORS) if count is None else max(0, int(count))
    )

    if palette_name in SPECTRA_CUSTOM_PALETTES:
        base_colors = list(SPECTRA_CUSTOM_PALETTES[palette_name])
    else:
        try:
            cmap = mpl.colormaps[palette_name]
        except (KeyError, AttributeError):
            try:
                cmap = mpl.cm.get_cmap(palette_name)
            except (ValueError, KeyError):
                base_colors = list(DEFAULT_COLORS)
                cmap = None

        if palette_name in SPECTRA_DISCRETE_PALETTES and cmap is not None:
            cmap_colors = getattr(cmap, "colors", None)
            if cmap_colors is None:
                cmap_colors = cmap(np.linspace(0.0, 1.0, cmap.N))
            base_colors = [mpl.colors.to_hex(color) for color in cmap_colors]
        elif cmap is not None:
            if requested_count == 0:
                return []
            positions = (
                np.array([0.5]) if requested_count == 1
                else np.linspace(0.0, 1.0, requested_count)
            )
            return [mpl.colors.to_hex(cmap(position)) for position in positions]

    if not base_colors:
        base_colors = list(DEFAULT_COLORS) or ["#1f77b4"]
    return [base_colors[i % len(base_colors)] for i in range(requested_count)]


def get_palette_color_choices(palette_name: str, spectrum_count: int = 0) -> list[str]:
    """Return the selectable swatches for a palette in the legend editor."""
    if palette_name in SPECTRA_CUSTOM_PALETTES:
        return list(SPECTRA_CUSTOM_PALETTES[palette_name])

    try:
        cmap = mpl.colormaps[palette_name]
    except (KeyError, AttributeError):
        return list(DEFAULT_COLORS)

    if palette_name in SPECTRA_DISCRETE_PALETTES:
        cmap_colors = getattr(cmap, "colors", None)
        if cmap_colors is None:
            cmap_colors = cmap(np.linspace(0.0, 1.0, cmap.N))
        return [mpl.colors.to_hex(color) for color in cmap_colors]

    # A continuous gradient has infinitely many colors. Offer a useful,
    # bounded set of evenly spaced swatches without making every table
    # combobox excessively large for maps containing hundreds of spectra.
    swatch_count = max(20, min(32, max(0, int(spectrum_count))))
    return [
        mpl.colors.to_hex(cmap(position))
        for position in np.linspace(0.0, 1.0, swatch_count)
    ]


def normalized_optional_color(color) -> str | None:
    """Normalize an optional Matplotlib color specification to lowercase hex."""
    if not color:
        return None
    try:
        return mpl.colors.to_hex(color)
    except (TypeError, ValueError):
        return None


def extract_filename_tokens(fname: str) -> list[tuple[int, str, float]]:
    """Split *fname* on underscores and return tokens that contain digits.

    Returns a list of ``(position, token_text, numeric_value)`` tuples.
    *position* is the 0-based index inside the underscore-split list so
    the same slot can be looked up across different filenames.
    """
    stem = _RE_FILE_EXT.sub("", fname)
    tokens = stem.split("_")
    result = []
    for pos, tok in enumerate(tokens):
        numbers = _RE_NUMBERS.findall(tok)
        if numbers:
            result.append((pos, tok, float(numbers[0])))
    return result


def token_pattern(token_text: str) -> str:
    """Normalize numeric parts of a filename token to '#' (e.g. '25°C' -> '#°C')."""
    if not token_text or token_text in ("Index", tr("Index")):
        return "Index"
    return _RE_NUMBERS.sub("#", token_text)


def record_token_selection(combo: QComboBox, state_dict: dict) -> None:
    """Record the active parameter selection (text, token slot position, unit pattern)."""
    text = combo.currentText()
    pos = combo.currentData()
    if pos is None or not text or text in ("Index", tr("Index")):
        state_dict.update({"text": "Index", "pos": None, "pattern": "Index"})
    else:
        state_dict.update({
            "text": text,
            "pos": pos if isinstance(pos, int) else state_dict.get("pos"),
            "pattern": token_pattern(text),
        })


def refresh_token_combobox(
    combo: QComboBox,
    state_dict: dict,
    tokens: list[tuple[int, str]],
) -> None:
    """Rebuild a parameter combobox from filenames while preserving session choice."""
    target_text = state_dict.get("text", "Index")
    target_pos = state_dict.get("pos")
    target_pattern = state_dict.get("pattern", "Index")

    combo.blockSignals(True)
    combo.clear()
    combo.addItem(tr("Index"), None)
    for pos, token_text in tokens:
        combo.addItem(token_text, pos)

    match_idx = None
    if target_text and target_text != "Index" and tokens:
        # 1. Exact text & position match
        for i, (pos, token_text) in enumerate(tokens, start=1):
            if token_text == target_text and (target_pos is None or pos == target_pos):
                match_idx = i
                break
        # 2. Same underscore position & same unit pattern (e.g. '500°C' -> '25°C')
        if match_idx is None and target_pos is not None:
            for i, (pos, token_text) in enumerate(tokens, start=1):
                if pos == target_pos and token_pattern(token_text) == target_pattern:
                    match_idx = i
                    break
        # 3. Exact text match anywhere
        if match_idx is None:
            for i, (_pos, token_text) in enumerate(tokens, start=1):
                if token_text == target_text:
                    match_idx = i
                    break
        # 4. Same underscore position
        if match_idx is None and target_pos is not None:
            for i, (pos, _token_text) in enumerate(tokens, start=1):
                if pos == target_pos:
                    match_idx = i
                    break
        # 5. Same non-generic unit pattern
        if match_idx is None and target_pattern not in ("Index", "#"):
            for i, (_pos, token_text) in enumerate(tokens, start=1):
                if token_pattern(token_text) == target_pattern:
                    match_idx = i
                    break

    if match_idx is not None:
        combo.setCurrentIndex(match_idx)
        state_dict["pos"] = combo.itemData(match_idx)
        state_dict["text"] = combo.currentText()
    combo.blockSignals(False)


def extract_series_parameter_values(
    fnames: list[str],
    n_spectra: int,
    selected_token: str,
    custom_label: str,
    token_pos: int | None = None,
) -> tuple[np.ndarray, str]:
    """Return ``(values, label)`` parsed from filenames or spectrum order.

    * **Index** → simple 0 … N-1.
    * Otherwise *selected_token* is matched to a token position (via
      *token_pos* or from the *first* filename), and the numeric part of
      that same position is extracted from every filename.
    """
    custom = custom_label.strip() if custom_label else ""

    if (token_pos is None and selected_token in ("Index", tr("Index"))) or not fnames:
        return np.arange(n_spectra, dtype=float), (custom or tr("Spectrum index"))

    resolved_pos = token_pos if isinstance(token_pos, int) else None
    if resolved_pos is None:
        ref_tokens = extract_filename_tokens(fnames[0])
        for pos, tok, _val in ref_tokens:
            if tok == selected_token:
                resolved_pos = pos
                break
        if resolved_pos is None:
            target_pattern = token_pattern(selected_token)
            if target_pattern not in ("Index", "#"):
                for pos, tok, _val in ref_tokens:
                    if token_pattern(tok) == target_pattern:
                        resolved_pos = pos
                        break

    if resolved_pos is None:
        return np.arange(n_spectra, dtype=float), (custom or selected_token)

    values = []
    for fname in fnames:
        stem = _RE_FILE_EXT.sub("", fname)
        parts = stem.split("_")
        if resolved_pos < len(parts):
            nums = _RE_NUMBERS.findall(parts[resolved_pos])
            values.append(float(nums[0]) if nums else float("nan"))
        else:
            values.append(float("nan"))

    return np.array(values, dtype=float), (custom or selected_token)


def non_singular_bounds(vmin: float, vmax: float, pad: float = 0.5) -> tuple[float, float]:
    """Ensure axis bounds are finite and non-singular."""
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        return 0.0, 1.0
    if vmin == vmax:
        return vmin - pad, vmax + pad
    return float(vmin), float(vmax)


def compute_normalized_y(x, y, xmin=None, xmax=None):
    """Apply view-only normalization to a single 1D spectrum."""
    if xmin is not None and xmax is not None:
        if xmin > xmax:
            xmin, xmax = xmax, xmin

        imin = np.abs(x - xmin).argmin()
        imax = np.abs(x - xmax).argmin()
        idx_start = min(imin, imax)
        idx_end = max(imin, imax) + 1
        norm_val = np.max(y[idx_start:idx_end]) if idx_end > idx_start else np.max(y)
    else:
        norm_val = np.max(y)

    if norm_val != 0:
        return y / norm_val

    return y


def compute_normalized_y_tensor(x, Y, xmin=None, xmax=None):
    """Apply view-only normalization across a batch (list or 2D tensor) of spectra."""
    if isinstance(Y, list):
        y_norm = []
        for i in range(len(Y)):
            xi = x[i] if isinstance(x, list) else x
            y_norm.append(compute_normalized_y(xi, Y[i], xmin=xmin, xmax=xmax))
        return y_norm

    if xmin is not None and xmax is not None:
        if xmin > xmax:
            xmin, xmax = xmax, xmin

        imin = np.abs(x - xmin).argmin()
        imax = np.abs(x - xmax).argmin()

        idx_start = min(imin, imax)
        idx_end = max(imin, imax) + 1
        if idx_end > idx_start:
            sub_y = Y[:, idx_start:idx_end]
            norm_vals = np.max(sub_y, axis=1, keepdims=True)
        else:
            norm_vals = np.max(Y, axis=1, keepdims=True)
    else:
        norm_vals = np.max(Y, axis=1, keepdims=True)

    norm_vals = np.where(norm_vals == 0, 1.0, norm_vals)
    return Y / norm_vals
