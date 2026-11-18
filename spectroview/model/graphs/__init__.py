"""Model graphs subpackage: graph schema, patch validation, style partitioning, recipes, and templates."""

from spectroview.model.graphs.graph_control import (
    GRAPH_PROPERTY_NAMES,
    GRAPH_PROPERTY_SET,
    VALID_PLOT_STYLES,
    AxisBreaks,
    AxisRange,
    FilterSpec,
    GraphAnnotation,
    GraphPatch,
    GraphValidationError,
    SeriesStyle,
    SpineVisibility,
    apply_graph_patch,
    graph_capability_schema,
    graph_patch_to_dict,
    normalize_graph_patch,
)
from spectroview.model.graphs.graph_style import (
    RESTYLE_SAFE_FIELDS,
    STYLE_FIELD_NAMES,
    apply_style_dict,
    can_restyle_without_replot,
    default_style,
    extract_style,
)
from spectroview.model.graphs.m_graph import MGraph
from spectroview.model.graphs.m_plot_recipe import MPlotRecipe
from spectroview.model.graphs.m_plot_recipe_store import (
    MPlotRecipeStore,
    PlotRecipeSummary,
)
from spectroview.model.graphs.m_style_template_store import (
    MStyleTemplateStore,
    StyleTemplateSummary,
)

__all__ = [
    "AxisBreaks",
    "AxisRange",
    "FilterSpec",
    "GRAPH_PROPERTY_NAMES",
    "GRAPH_PROPERTY_SET",
    "GraphAnnotation",
    "GraphPatch",
    "GraphValidationError",
    "MGraph",
    "MPlotRecipe",
    "MPlotRecipeStore",
    "MStyleTemplateStore",
    "PlotRecipeSummary",
    "RESTYLE_SAFE_FIELDS",
    "STYLE_FIELD_NAMES",
    "SeriesStyle",
    "SpineVisibility",
    "StyleTemplateSummary",
    "VALID_PLOT_STYLES",
    "apply_graph_patch",
    "apply_style_dict",
    "can_restyle_without_replot",
    "default_style",
    "extract_style",
    "graph_capability_schema",
    "graph_patch_to_dict",
    "normalize_graph_patch",
]
