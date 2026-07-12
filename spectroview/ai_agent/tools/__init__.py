"""tools/__init__.py — SPECTROview AI Agent tools package."""
from spectroview.ai_agent.tools.dataframe_tool import (
    build_schema_info,
    build_graphs_info,
    safe_query,
    safe_describe,
    format_dataframe_as_markdown,
)
from spectroview.ai_agent.tools.plot_tool import (
    validate_plot_style,
    validate_palette,
    normalize_plot_config,
    expand_comma_styles,
    expand_all_plot_configs,
    VALID_PLOT_STYLES,
    VALID_PALETTES,
)
from spectroview.ai_agent.tools.fitting_tool import (
    list_peak_models,
    get_model_info,
    get_model_parameters,
    validate_peak_params,
    interpret_r_squared,
    suggest_filter_for_quality,
    PEAK_MODEL_REGISTRY,
)

__all__ = [
    # dataframe_tool
    "build_schema_info",
    "build_graphs_info",
    "safe_query",
    "safe_describe",
    "format_dataframe_as_markdown",
    # plot_tool
    "validate_plot_style",
    "validate_palette",
    "normalize_plot_config",
    "expand_comma_styles",
    "expand_all_plot_configs",
    "VALID_PLOT_STYLES",
    "VALID_PALETTES",
    # fitting_tool
    "list_peak_models",
    "get_model_info",
    "get_model_parameters",
    "validate_peak_params",
    "interpret_r_squared",
    "suggest_filter_for_quality",
    "PEAK_MODEL_REGISTRY",
]
