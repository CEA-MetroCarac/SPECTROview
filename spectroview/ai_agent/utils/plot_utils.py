"""tools/plot_tool.py

Reusable plot configuration utilities for the SPECTROview AI Agent.

These helpers normalise the LLM's raw JSON output into the typed format
expected by the MGraph model and the Graphs workspace, and expand
comma-separated multi-style plot configs into individual entries.
"""
from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants mirroring spectroview/__init__.py values
# (duplicated here to avoid a circular import — kept in sync manually)
# ---------------------------------------------------------------------------

VALID_PLOT_STYLES: frozenset[str] = frozenset({
    "point", "scatter", "box", "bar",
    "line", "trendline", "histogram", "wafer", "2Dmap",
})

VALID_PALETTES: frozenset[str] = frozenset({
    "jet", "viridis", "plasma", "magma",
    "cividis", "cool", "hot", "YlGnBu", "YlOrRd",
    "seismic", "bwr", "Spectral",
    "tab10", "Set2", "Set3", "coolwarm", "RdBu", "inferno",
})

# Fields that must be floats or None
_FLOAT_KEYS: tuple[str, ...] = (
    "xmin", "xmax", "ymin", "ymax",
    "zmin", "zmax", "y2min", "y2max",
    "y3min", "y3max", "x2min", "x2max",
)

# Fields that must be ints
_INT_KEYS: tuple[str, ...] = (
    "x_rot", "scatter_size", "plot_width", "plot_height",
    "dpi", "trendline_order", "hist_bins",
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_plot_style(style: str) -> bool:
    """Return True if *style* is a recognised SPECTROview plot style.

    Parameters
    ----------
    style:
        Plot style string to validate.
    """
    return style.strip().lower() in VALID_PLOT_STYLES


def validate_palette(palette: str) -> bool:
    """Return True if *palette* is a supported color palette name.

    Parameters
    ----------
    palette:
        Color palette name to validate.
    """
    return palette in VALID_PALETTES


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_plot_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Coerce a raw LLM plot config dictionary into the typed format
    expected by :class:`spectroview.model.m_graph.MGraph`.

    This function is the single source of truth for type coercion and
    replaces the inline normalisation previously scattered across
    ``main.py`` and ``vm_chat.py``.

    Parameters
    ----------
    cfg:
        Raw plot configuration dictionary from the LLM JSON response.
        Modified **in-place** for performance; a deep copy is the
        caller's responsibility if the original must be preserved.

    Returns
    -------
    dict[str, Any]
        The same dictionary with types corrected.
    """
    # ── y must be a list of strings ──────────────────────────────────────
    if "y" in cfg:
        y = cfg["y"]
        if isinstance(y, str):
            cfg["y"] = [y] if y else []
        elif y is None:
            cfg["y"] = []
        elif not isinstance(y, list):
            cfg["y"] = []

    # ── Float limits ─────────────────────────────────────────────────────
    for key in _FLOAT_KEYS:
        if key in cfg:
            val = cfg[key]
            if val is None or val == "" or val == "null":
                cfg[key] = None
            else:
                try:
                    cfg[key] = float(val)
                except (ValueError, TypeError):
                    logger.debug("Cannot convert %r=%r to float; setting None", key, val)
                    cfg[key] = None

    # ── Integer fields ───────────────────────────────────────────────────
    for key in _INT_KEYS:
        if key in cfg:
            try:
                cfg[key] = int(cfg[key])
            except (ValueError, TypeError):
                logger.debug("Cannot convert %r=%r to int; removing key", key, cfg[key])
                del cfg[key]

    # ── Filters — normalise to list[dict] ────────────────────────────────
    if "filters" in cfg:
        raw_filters = cfg["filters"]
        if isinstance(raw_filters, list):
            parsed: list[dict] = []
            for f in raw_filters:
                if isinstance(f, str) and f.strip():
                    parsed.append({"expression": f, "state": True})
                elif isinstance(f, dict) and "expression" in f:
                    f.setdefault("state", True)
                    parsed.append(f)
            cfg["filters"] = parsed
        else:
            cfg["filters"] = []

    # ── Validate plot_style ───────────────────────────────────────────────
    if "plot_style" in cfg:
        # Comma-separated styles are valid at this stage (expanded later)
        styles = [s.strip() for s in str(cfg["plot_style"]).split(",")]
        invalid = [s for s in styles if s and not validate_plot_style(s)]
        if invalid:
            logger.warning("Unrecognised plot style(s): %s", invalid)

    # ── Validate / sanitise color_palette ────────────────────────────────
    if "color_palette" in cfg and cfg["color_palette"]:
        if not validate_palette(str(cfg["color_palette"])):
            logger.warning(
                "Unrecognised palette %r; falling back to 'jet'",
                cfg["color_palette"],
            )
            cfg["color_palette"] = "jet"

    return cfg


# ---------------------------------------------------------------------------
# Multi-style expansion
# ---------------------------------------------------------------------------

def expand_comma_styles(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a single plot config with comma-separated ``plot_style`` into
    multiple individual configs.

    The LLM may return ``"plot_style": "box, scatter"`` as a compact
    shorthand for two graphs.  This function splits them into separate
    configs so each can be sent to the workspace independently.

    If ``plot_style`` contains only a single style (no comma), the
    original config is returned wrapped in a single-element list.

    Parameters
    ----------
    cfg:
        A single normalised plot configuration dictionary.

    Returns
    -------
    list[dict[str, Any]]
        One entry per style.
    """
    raw_style = str(cfg.get("plot_style", "")).strip()
    if "," not in raw_style:
        return [cfg]

    styles = [s.strip() for s in raw_style.split(",") if s.strip()]
    result: list[dict[str, Any]] = []
    for style in styles:
        entry = copy.deepcopy(cfg)
        entry["plot_style"] = style
        result.append(entry)
    return result


def expand_all_plot_configs(
    raw_configs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalise and expand a list of raw plot configs from the LLM.

    Applies :func:`normalize_plot_config` and :func:`expand_comma_styles`
    to every entry.

    Parameters
    ----------
    raw_configs:
        List of raw plot config dicts from the parsed LLM JSON.

    Returns
    -------
    list[dict[str, Any]]
        Fully processed, type-safe, expanded list ready for the workspace.
    """
    result: list[dict[str, Any]] = []
    for raw in raw_configs:
        cfg = copy.deepcopy(raw)
        normalize_plot_config(cfg)
        result.extend(expand_comma_styles(cfg))
    return result
