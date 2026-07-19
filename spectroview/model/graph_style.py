# model/graph_style.py
"""Splits MGraph's flat field set into "style" (how a graph looks) vs
everything else (what data it plots, and per-graph state that isn't a
portable "look"), for Phase 5's per-graph style templates / copy-paste
style / reset-to-default features.

Derived by EXCLUSION from `dataclasses.fields(MGraph)` (mirroring
`graph_commit.py`'s `COMMIT_FIELDS` pattern) rather than an explicit
inclusion whitelist: a newly-added MGraph field is almost always a visual
setting, so defaulting new fields to "style" and only excluding the
specific fields that are clearly data/identity/layout-state keeps this
list from silently going stale the way an inclusion list would.
"""
import copy
import dataclasses
from typing import Any, Dict

from spectroview.model.m_graph import MGraph

_NON_STYLE_FIELDS = frozenset({
    # Identity / which data this graph plots -- copying a "style" must
    # never repoint another graph at different data or columns.
    'graph_id', 'df_name', 'filters', 'plot_style',
    'x', 'y', 'z', 'y2', 'y3', 'x2',
    'x_as_numeric', 'y_as_numeric',

    # Text content that describes *this* data, not a reusable visual
    # style (font sizes for these are still style fields, kept below).
    'plot_title', 'plot_subtitle', 'xlabel', 'ylabel', 'zlabel',
    'y2label', 'y3label', 'x2label', 'legend_title',

    # Data-coordinate-specific: meaningless (or actively wrong) on a graph
    # with a different data range.
    'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax',
    'y2min', 'y2max', 'y3min', 'y3max', 'x2min', 'x2max',
    'inset_xmin', 'inset_xmax', 'inset_ymin', 'inset_ymax',
    'annotations', 'axis_breaks', 'trendline_anchor_x', 'trendline_anchor_y',

    # Per-graph derived/layout state (dragged legend position; per-series
    # colors tied to *this* graph's actual hue-category count) -- applying
    # a style resets these so they re-derive fresh under the new settings
    # rather than copying stale entries that may not fit the target.
    'legend_bbox', 'legend_properties',

    # Window/output geometry, not a visual "look".
    'plot_width', 'plot_height', 'dpi', 'export_width_mm', 'export_height_mm',
})

STYLE_FIELD_NAMES: tuple = tuple(
    f.name for f in dataclasses.fields(MGraph) if f.name not in _NON_STYLE_FIELDS
)


def extract_style(source: Dict[str, Any]) -> Dict[str, Any]:
    """Pull just the style-field subset out of a full graph dict (e.g. from
    `MGraph.save()` or `vars(graph_widget)`). Deep-copied so the result is
    independent of the source's own mutable fields (mirrors `MGraph.save()`'s
    own deep-copy discipline)."""
    return {k: copy.deepcopy(source[k]) for k in STYLE_FIELD_NAMES if k in source}


def default_style() -> Dict[str, Any]:
    """The style-field subset of a fresh MGraph()'s own dataclass defaults
    -- used for "reset to default"."""
    return extract_style(vars(MGraph()))


def apply_style_dict(graph_widget, style_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Write a style dict's fields onto a graph widget (generic setattr,
    mirroring `_configure_graph_from_model`'s pattern), and reset
    `legend_properties` so per-series colors/markers re-derive fresh under
    the new style on the next render rather than keeping stale entries
    from whatever the target graph had before.

    Returns the dict of fields actually applied (deep-copied), for the
    caller to fold into a `properties_changed` emission.
    """
    applied = {}
    for key, value in style_dict.items():
        if key not in STYLE_FIELD_NAMES:
            continue  # defensive: ignore anything outside the style whitelist
        value = copy.deepcopy(value)
        setattr(graph_widget, key, value)
        applied[key] = value

    graph_widget.legend_properties = []
    applied['legend_properties'] = []
    return applied


# A narrower cut than STYLE_FIELD_NAMES: only fields VGraph.restyle() can
# repaint on existing artists without a full replot (chrome, not what an
# artist looks like or which artists exist). legend_properties is excluded
# even though _set_legend() reads it, since it never recolors the actual
# series artists.
RESTYLE_SAFE_FIELDS = frozenset({
    # Titles / labels
    'plot_title', 'plot_subtitle', 'xlabel', 'ylabel',
    'title_fontsize', 'axis_label_fontsize', 'subtitle_fontsize',
    # Grid / ticks
    'grid', 'minor_ticks_bottom', 'minor_ticks_left', 'minor_ticks_top', 'minor_ticks_right',
    'tick_direction', 'tick_label_fontsize', 'tick_label_format',
    # Tick-label rotation
    'x_rot',
    # Primary-axes limits (not y2/y3/x2min/max -- those live on twin axes,
    # which restyle() doesn't touch; see _plot_secondary_axis() etc.)
    'xmin', 'xmax', 'ymin', 'ymax',
    # Scale / direction
    'xlogscale', 'ylogscale', 'xscale_mode', 'yscale_mode',
    'x_inverted', 'y_inverted',
    # Figure/axes chrome
    'figure_facecolor', 'spines_visible', 'figure_margins',
    # Legend box styling (position/frame/columns/title/font -- not the
    # per-series legend_properties list, see module note above)
    'legend_visible', 'legend_ncol', 'legend_frame', 'legend_alpha',
    'legend_title', 'legend_fontsize', 'legend_outside', 'legend_loc', 'legend_bbox',
})


def can_restyle_without_replot(changed_fields) -> bool:
    """True if every field in `changed_fields` is purely cosmetic -- safe
    to repaint via VGraph.restyle() instead of a full plot()/replot.

    False (needs a full replot) for no changes at all (nothing to do, but
    that's the caller's call to skip -- this only answers "is the fast
    path safe"), for any data-derived field (series color/marker/size,
    colormap, secondary axes, annotations, insets), or for the identity
    fields (x/y/z/plot_style/df_name/filters) themselves.
    """
    changed = set(changed_fields)
    return bool(changed) and changed <= RESTYLE_SAFE_FIELDS
