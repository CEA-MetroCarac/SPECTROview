"""tools/plot_tool.py

Reusable plot configuration utilities for the SPECTROview AI Agent.

These helpers normalise the LLM's raw JSON output into the typed format
expected by the MGraph model and the Graphs workspace, and expand
comma-separated multi-style plot configs into individual entries.
"""
from __future__ import annotations

import copy
from typing import Any

from spectroview import PALETTE, PLOT_STYLES
from spectroview.model.graph_control import normalize_graph_patch

VALID_PLOT_STYLES: frozenset[str] = frozenset(PLOT_STYLES)
VALID_PALETTES: frozenset[str] = frozenset(PALETTE) | frozenset({
    "tab10", "Set2", "Set3", "coolwarm", "RdBu", "inferno",
})

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

    Compatibility wrapper around the application-level
    :func:`spectroview.model.graph_control.normalize_graph_patch`.  Keeping
    this function avoids breaking recipe/agent callers while ensuring GUI,
    AI, and MCP no longer maintain different coercion rules.

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
    normalized = normalize_graph_patch(cfg)
    cfg.clear()
    cfg.update(normalized)
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
        # Split first: GraphPatch correctly constrains plot_style to one real
        # renderer style, while this legacy compact shorthand can still be
        # accepted at the boundary and expanded into valid individual calls.
        for cfg in expand_comma_styles(copy.deepcopy(raw)):
            normalize_plot_config(cfg)
            result.append(cfg)
    return result
