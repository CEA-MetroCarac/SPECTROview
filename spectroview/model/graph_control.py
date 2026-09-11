"""Validated command layer for :class:`~spectroview.model.m_graph.MGraph`.

The Graphs GUI, the embedded AI agent, and MCP clients all submit partial
updates to the same flat ``MGraph`` state.  This module owns the rules for
normalising, validating, and atomically applying those updates.  The Pydantic
``GraphPatch`` model is generated from the dataclass itself so a newly-added
graph property cannot silently be omitted from the AI/MCP JSON schema.

Rendering remains a View responsibility.  This layer deliberately knows
nothing about Qt, Matplotlib figures, the chat ViewModel, or MCP transports.
"""
from __future__ import annotations

import copy
import re
import dataclasses
import types
from typing import Any, Dict, List, Literal, Mapping, Optional, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from spectroview import PLOT_STYLES
from spectroview.model.m_graph import MGraph


_UNION_ORIGINS = (Union,) + ((types.UnionType,) if hasattr(types, "UnionType") else ())


class GraphValidationError(ValueError):
    """A graph patch is unknown, malformed, or inconsistent with its data."""


GRAPH_PROPERTY_NAMES = tuple(
    field.name for field in dataclasses.fields(MGraph) if field.name != "graph_id"
)
GRAPH_PROPERTY_SET = frozenset(GRAPH_PROPERTY_NAMES)
VALID_PLOT_STYLES = frozenset(PLOT_STYLES)


class _StrictGraphObject(BaseModel):
    """Base for nested tool inputs; typos must fail instead of being ignored."""

    model_config = ConfigDict(extra="forbid")


class FilterSpec(_StrictGraphObject):
    expression: str = Field(description="Non-empty pandas DataFrame.query expression.")
    state: bool = Field(default=True, description="Whether this filter is active.")


class SeriesStyle(_StrictGraphObject):
    """Every per-series customization currently consumed by VGraph."""

    label: Optional[str] = None
    marker: Optional[str] = None
    color: Optional[str] = None
    rgba: Optional[List[float]] = Field(default=None, min_length=4, max_length=4)
    linewidth: Optional[float] = Field(default=None, ge=0)
    alpha: Optional[float] = Field(default=None, ge=0, le=1)
    zorder: Optional[float] = None
    marker_size: Optional[float] = Field(default=None, gt=0)
    edge_color: Optional[str] = None


class AxisRange(_StrictGraphObject):
    start: float
    end: float


class AxisBreaks(_StrictGraphObject):
    # Both keys are required because this field replaces, rather than deep
    # merges, the existing break definition.
    x: Optional[AxisRange]
    y: Optional[AxisRange]


class SpineVisibility(_StrictGraphObject):
    top: bool
    right: bool
    bottom: bool
    left: bool


class GraphAnnotation(_StrictGraphObject):
    """Union-shaped annotation record used by all eight GUI annotation tools."""

    type: Literal["vline", "hline", "text", "arrow", "vspan", "hspan", "box", "callout"]
    id: Optional[str] = None
    label: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    x1: Optional[float] = None
    y1: Optional[float] = None
    x2: Optional[float] = None
    y2: Optional[float] = None
    tx: Optional[float] = None
    ty: Optional[float] = None
    width: Optional[float] = Field(default=None, gt=0)
    height: Optional[float] = Field(default=None, gt=0)
    text: Optional[str] = None
    fontsize: Optional[float] = Field(default=None, gt=0)
    color: Optional[str] = None
    arrowcolor: Optional[str] = None
    facecolor: Optional[str] = None
    edgecolor: Optional[str] = None
    linestyle: Optional[str] = None
    linewidth: Optional[float] = Field(default=None, ge=0)
    alpha: Optional[float] = Field(default=None, ge=0, le=1)
    ha: Optional[Literal["left", "center", "right"]] = None
    va: Optional[Literal["top", "center", "bottom", "baseline", "center_baseline"]] = None
    bbox: Optional[Dict[str, Any]] = None


# Static enum types materially improve constrained tool calling.  Everything
# else keeps MGraph's own annotation, which is already the source of truth for
# scalar/list/dict shapes.
_TYPE_OVERRIDES = {
    "plot_style": Literal[
        "point", "scatter", "box", "bar", "line",
        "trendline", "histogram", "wafer", "2Dmap",
    ],
    "xscale_mode": Literal["log", "symlog"],
    "yscale_mode": Literal["log", "symlog"],
    "tick_direction": Literal["in", "out", "inout"],
    "figure_theme": Literal["light", "dark", "soft_dark"],
    "colormap_norm": Literal["linear", "log", "centered"],
    "error_bar_type": Literal["none", "sd", "sem", "ci95"],
    "bar_error_bar_type": Literal["none", "sd", "sem", "ci95"],
    "sort_data_by": Literal["X", "Y", "Z"],
    "legend_loc": Literal[
        "best", "upper right", "upper left", "lower left", "lower right",
        "right", "center left", "center right", "lower center",
        "upper center", "center",
    ],
    "filters": List[FilterSpec],
    "legend_properties": List[SeriesStyle],
    "axis_breaks": AxisBreaks,
    "spines_visible": SpineVisibility,
    "annotations": List[GraphAnnotation],
}


_FIELD_DESCRIPTIONS = {
    "df_name": "Loaded DataFrame supplying this graph. Change only when intentionally repointing the graph.",
    "filters": "Complete filter list. Each item is {'expression': pandas query, 'state': boolean}.",
    "plot_style": "Graph style: point, scatter, box, bar, line, trendline, histogram, wafer, or 2Dmap.",
    "plot_width": "Graph subwindow width in screen pixels.",
    "plot_height": "Graph subwindow height in screen pixels.",
    "dpi": "On-screen Matplotlib figure DPI.",
    "x": "Primary X column name.",
    "y": "Primary Y column names; multiple entries create multiple primary-axis series.",
    "z": "Hue/grouping column, or the measured-value column for wafer/2Dmap.",
    "y2": "Secondary Y-axis column, or null to remove that axis.",
    "y3": "Tertiary Y-axis column, or null to remove that axis.",
    "x2": "Secondary X-axis column, or null to remove that axis.",
    "xmin": "Primary X-axis lower limit; null restores automatic limits.",
    "xmax": "Primary X-axis upper limit; null restores automatic limits.",
    "ymin": "Primary Y-axis lower limit; null restores automatic limits.",
    "ymax": "Primary Y-axis upper limit; null restores automatic limits.",
    "zmin": "Color scale lower limit; null restores automatic limits.",
    "zmax": "Color scale upper limit; null restores automatic limits.",
    "xlogscale": "Whether the primary X axis uses xscale_mode instead of linear scale.",
    "ylogscale": "Whether the primary Y axis uses yscale_mode instead of linear scale.",
    "xscale_mode": "Scale selected when xlogscale is true: log or symlog.",
    "yscale_mode": "Scale selected when ylogscale is true: log or symlog.",
    "plot_title": "Figure title text; null removes the custom title.",
    "plot_subtitle": "Figure subtitle text; null removes it.",
    "xlabel": "Primary X-axis label; null restores the data-column label.",
    "ylabel": "Primary Y-axis label; null restores the data-column label.",
    "zlabel": "Colorbar/Z label; null restores the data-column label.",
    "grid": "Show or hide primary-axis grid lines.",
    "tick_direction": "Major tick direction; null restores Matplotlib's theme default.",
    "tick_label_format": "Printf-style numeric tick format such as %.2f; null restores automatic formatting.",
    "figure_facecolor": "Matplotlib color for figure and axes background; null uses the selected theme.",
    "figure_margins": "Two non-negative margins [x_margin, y_margin].",
    "spines_visible": "Visibility mapping for top, right, bottom, and left axes spines.",
    "figure_theme": "Per-graph rendering theme: light, dark, or soft_dark.",
    "legend_visible": "Show or hide the legend.",
    "legend_outside": "Place the legend outside the axes.",
    "legend_properties": (
        "Complete per-series style list. Entries may set label, color, marker, linewidth, "
        "alpha, zorder, marker_size, and edge_color. Preserve entries not being changed."
    ),
    "legend_bbox": "Dragged legend position [x, y] in axes coordinates; null restores automatic placement.",
    "legend_loc": "Named inside-axes legend position.",
    "color_palette": "Matplotlib colormap/palette name used by grouped and spatial plots.",
    "colormap_norm": "Spatial color normalization: linear, log, or centered.",
    "colormap_center": "Center value for centered spatial color normalization.",
    "axis_breaks": (
        "Broken-axis mapping {'x': range-or-null, 'y': range-or-null}; each range has numeric start/end. "
        "Only one axis may be broken at once."
    ),
    "annotations": (
        "Complete annotation list. Supported types are vline, hline, text, arrow, vspan, hspan, box, and callout."
    ),
    "inset_bounds": "Inset placement [x0, y0, width, height] in axes-fraction coordinates.",
    "export_width_mm": "Preferred export width in millimetres; null uses current figure size.",
    "export_height_mm": "Preferred export height in millimetres; null uses current figure size.",
}


def _human_description(name: str) -> str:
    return _FIELD_DESCRIPTIONS.get(
        name,
        "Graph customization property: " + name.replace("_", " ") + ".",
    )


def _optional(annotation):
    origin = get_origin(annotation)
    if origin in _UNION_ORIGINS and type(None) in get_args(annotation):
        return annotation
    return Optional[annotation]


def _declared_nullable(annotation) -> bool:
    origin = get_origin(annotation)
    return origin in _UNION_ORIGINS and type(None) in get_args(annotation)


def _build_graph_patch_model():
    definitions = {}
    for model_field in dataclasses.fields(MGraph):
        if model_field.name == "graph_id":
            continue
        annotation = _TYPE_OVERRIDES.get(model_field.name, model_field.type)
        definitions[model_field.name] = (
            _optional(annotation),
            Field(default=None, description=_human_description(model_field.name)),
        )
    return create_model(
        "GraphPatch",
        __config__=ConfigDict(
            extra="forbid",
            title="Graph customization patch",
            json_schema_extra={
                "description": (
                    "A partial MGraph update. Supply only requested properties; omitted values are preserved. "
                    "Explicit null clears nullable labels, limits, optional axes, positions, and export sizes."
                )
            },
        ),
        **definitions,
    )


GraphPatch = _build_graph_patch_model()


def graph_patch_to_dict(value: Any) -> Dict[str, Any]:
    """Return a deep-copied, explicitly-set patch from a dict/Pydantic value."""
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return copy.deepcopy(value.model_dump(exclude_unset=True))
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    raise GraphValidationError("Graph properties must be a JSON object.")


_EQ_FILTER_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*|`[^`]+`)\s*==\s*(.+?)\s*$")


def _merge_equality_filters(filters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Automatically merge multiple active equality filters on the same column.

    For example, if a model or user provides:
        ["Slot == 2", "Slot == 6", "Slot == 8", "Slot == 10", "Quadrant != 'Q4'"]
    which would evaluate as AND and yield an empty DataFrame (0 rows),
    merge them into:
        ["Slot in [2, 6, 8, 10]", "Quadrant != 'Q4'"].
    """
    col_counts: Dict[str, List[str]] = {}
    for f in filters:
        if not f.get("state", True):
            continue
        m = _EQ_FILTER_RE.match(f["expression"])
        if m:
            col, val = m.group(1), m.group(2).strip()
            col_counts.setdefault(col, []).append(val)

    multi_cols = {col: vals for col, vals in col_counts.items() if len(set(vals)) > 1}
    if not multi_cols:
        return filters

    seen_cols = set()
    result = []
    for f in filters:
        if not f.get("state", True):
            result.append(f)
            continue
        m = _EQ_FILTER_RE.match(f["expression"])
        if m and m.group(1) in multi_cols:
            col = m.group(1)
            if col not in seen_cols:
                seen_cols.add(col)
                vals = []
                for v in multi_cols[col]:
                    if v not in vals:
                        vals.append(v)
                result.append({
                    "expression": f"{col} in [{', '.join(vals)}]",
                    "state": True,
                })
        else:
            result.append(f)
    return result


def _normalize_filters(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise GraphValidationError("filters must be a list of query strings or filter objects.")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, str) and item.strip():
            result.append({"expression": item.strip(), "state": True})
        elif isinstance(item, Mapping) and str(item.get("expression", "")).strip():
            result.append({
                "expression": str(item["expression"]).strip(),
                "state": bool(item.get("state", True)),
            })
        else:
            raise GraphValidationError(
                f"filters[{index}] must be a non-empty query string or an object with expression/state."
            )
    return _merge_equality_filters(result)


def _validate_limit_pair(state: Mapping[str, Any], low: str, high: str) -> None:
    lo, hi = state.get(low), state.get(high)
    if lo is not None and hi is not None and lo >= hi:
        raise GraphValidationError(f"{low} must be less than {high}.")


def _validate_axis_breaks(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) - {"x", "y"}:
        raise GraphValidationError("axis_breaks must contain only the keys 'x' and 'y'.")
    enabled = 0
    for axis in ("x", "y"):
        axis_range = value.get(axis)
        if axis_range is None:
            continue
        enabled += 1
        if not isinstance(axis_range, Mapping) or "start" not in axis_range or "end" not in axis_range:
            raise GraphValidationError(f"axis_breaks.{axis} must contain numeric start and end values.")
        try:
            start, end = float(axis_range["start"]), float(axis_range["end"])
        except (TypeError, ValueError) as exc:
            raise GraphValidationError(f"axis_breaks.{axis} start/end must be numeric.") from exc
        if start >= end:
            raise GraphValidationError(f"axis_breaks.{axis}.start must be less than end.")
    if enabled > 1:
        raise GraphValidationError("Only one of the X or Y axes can be broken at a time.")


def _validate_structures(state: Mapping[str, Any]) -> None:
    margins = state.get("figure_margins")
    if not isinstance(margins, list) or len(margins) != 2 or any(v < 0 for v in margins):
        raise GraphValidationError("figure_margins must be [x_margin, y_margin] with non-negative values.")

    spines = state.get("spines_visible")
    if not isinstance(spines, Mapping) or set(spines) != {"top", "right", "bottom", "left"}:
        raise GraphValidationError("spines_visible must specify top, right, bottom, and left.")

    bbox = state.get("legend_bbox")
    if bbox is not None and (not isinstance(bbox, list) or len(bbox) != 2):
        raise GraphValidationError("legend_bbox must be [x, y] or null.")

    inset = state.get("inset_bounds")
    if not isinstance(inset, list) or len(inset) != 4:
        raise GraphValidationError("inset_bounds must be [x0, y0, width, height].")
    if any(v < 0 or v > 1 for v in inset) or inset[2] <= 0 or inset[3] <= 0:
        raise GraphValidationError("inset_bounds values must be within 0..1 and width/height must be positive.")
    if inset[0] + inset[2] > 1 or inset[1] + inset[3] > 1:
        raise GraphValidationError("inset_bounds must fit within the parent axes.")

    if not isinstance(state.get("legend_properties"), list) or not all(
        isinstance(item, Mapping) for item in state["legend_properties"]
    ):
        raise GraphValidationError("legend_properties must be a list of per-series objects.")

    annotations = state.get("annotations")
    if not isinstance(annotations, list) or not all(isinstance(item, Mapping) for item in annotations):
        raise GraphValidationError("annotations must be a list of annotation objects.")
    valid_annotations = {"vline", "hline", "text", "arrow", "vspan", "hspan", "box", "callout"}
    invalid_types = sorted({str(item.get("type")) for item in annotations if item.get("type") not in valid_annotations})
    if invalid_types:
        raise GraphValidationError(
            "Unsupported annotation type(s): " + ", ".join(invalid_types) + "."
        )
    required_by_type = {
        "vline": {"x"},
        "hline": {"y"},
        "text": {"x", "y", "text"},
        "arrow": {"x1", "y1", "x2", "y2"},
        "vspan": {"x1", "x2"},
        "hspan": {"y1", "y2"},
        "box": {"x", "y", "width", "height"},
        "callout": {"x", "y", "tx", "ty", "text"},
    }
    for index, item in enumerate(annotations):
        required = required_by_type.get(item.get("type"), set())
        missing = sorted(name for name in required if item.get(name) is None)
        if missing:
            raise GraphValidationError(
                f"annotations[{index}] ({item.get('type')}) is missing: {', '.join(missing)}."
            )

    _validate_axis_breaks(state.get("axis_breaks"))


def _validate_ranges(state: Mapping[str, Any]) -> None:
    for low, high in (
        ("xmin", "xmax"), ("ymin", "ymax"), ("zmin", "zmax"),
        ("y2min", "y2max"), ("y3min", "y3max"), ("x2min", "x2max"),
        ("inset_xmin", "inset_xmax"), ("inset_ymin", "inset_ymax"),
    ):
        _validate_limit_pair(state, low, high)

    positive = {
        "plot_width", "plot_height", "dpi", "title_fontsize", "axis_label_fontsize",
        "tick_label_fontsize", "colorbar_fontsize", "legend_fontsize", "legend_ncol",
        "scatter_size", "hist_bins", "trendline_order", "wafer_size",
    }
    for name in positive:
        if state.get(name) is None or state[name] <= 0:
            raise GraphValidationError(f"{name} must be greater than zero.")

    for name in ("legend_alpha",):
        if not 0.0 <= state[name] <= 1.0:
            raise GraphValidationError(f"{name} must be between 0 and 1.")
    if state["error_bar_capsize"] < 0:
        raise GraphValidationError("error_bar_capsize cannot be negative.")
    if not -360 <= state["x_rot"] <= 360:
        raise GraphValidationError("x_rot must be between -360 and 360 degrees.")


def _validate_columns(state: Mapping[str, Any], dataframe: Any) -> None:
    if dataframe is None:
        return
    available = set(getattr(dataframe, "columns", []))
    missing = []
    for name in ("x", "z", "y2", "y3", "x2"):
        value = state.get(name)
        if value and value not in available:
            missing.append(value)
    missing.extend(value for value in state.get("y", []) if value and value not in available)
    if missing:
        raise GraphValidationError(
            "Unknown DataFrame column(s): " + ", ".join(sorted(set(map(str, missing)))) + "."
        )


def normalize_graph_patch(
    properties: Any,
    *,
    current: Optional[Union[MGraph, Mapping[str, Any]]] = None,
    dataframe: Any = None,
) -> Dict[str, Any]:
    """Normalize and validate a partial graph update without mutating state.

    Validation is performed against the merged current+patch state, so range
    checks and column checks also work for small updates such as ``{"xmax": 5}``.
    The returned dict contains only explicitly supplied properties, plus a
    deterministic ``legend_properties=[]`` reset when a data/series-defining
    field changed and no explicit per-series styles were supplied.
    """
    raw = graph_patch_to_dict(properties)
    unknown = set(raw) - GRAPH_PROPERTY_SET
    if unknown:
        raise GraphValidationError(
            "Unknown or read-only graph property/properties: " + ", ".join(sorted(unknown)) + "."
        )

    if "y" in raw:
        if isinstance(raw["y"], str):
            raw["y"] = [raw["y"]] if raw["y"] else []
        elif raw["y"] is None:
            raw["y"] = []
    if "filters" in raw:
        raw["filters"] = _normalize_filters(raw["filters"])

    try:
        parsed = GraphPatch(**raw)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors()
        )
        raise GraphValidationError(details) from exc
    normalized = parsed.model_dump(exclude_unset=True)

    declared = {field.name: field.type for field in dataclasses.fields(MGraph)}
    for name, value in normalized.items():
        if value is None and not _declared_nullable(declared[name]):
            raise GraphValidationError(f"{name} cannot be null.")

    if isinstance(current, MGraph):
        base = current.save()
    elif current is None:
        base = MGraph().save()
    else:
        base = MGraph().save()
        base.update(copy.deepcopy(dict(current)))
    base.pop("graph_id", None)
    state = {**base, **normalized}

    if state["plot_style"] not in VALID_PLOT_STYLES:
        raise GraphValidationError(
            f"Invalid plot_style {state['plot_style']!r}; expected one of {sorted(VALID_PLOT_STYLES)}."
        )
    if not isinstance(state["y"], list) or not all(isinstance(item, str) for item in state["y"]):
        raise GraphValidationError("y must be a string or a list of column-name strings.")

    _validate_ranges(state)
    _validate_structures(state)
    _validate_columns(state, dataframe)

    series_context = {"df_name", "filters", "plot_style", "x", "y", "z", "sort_data_enabled", "sort_data_by"}
    if current is not None and series_context.intersection(normalized) and "legend_properties" not in normalized:
        normalized["legend_properties"] = []
    return copy.deepcopy(normalized)


def apply_graph_patch(
    graph: MGraph,
    properties: Any,
    *,
    dataframe: Any = None,
) -> Dict[str, Any]:
    """Validate then atomically apply a partial update to ``graph``."""
    patch = normalize_graph_patch(properties, current=graph, dataframe=dataframe)
    for name, value in patch.items():
        setattr(graph, name, copy.deepcopy(value))
    return patch


def graph_capability_schema() -> Dict[str, Any]:
    """JSON Schema for every mutable Graph workspace property."""
    return GraphPatch.model_json_schema()
