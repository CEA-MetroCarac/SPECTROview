"""spectroview/ai_agent/mcp/server.py

Model Context Protocol (MCP) Server for SPECTROview.

Exposes the operations the AI Agent may perform on the user's data as MCP
tools. The server reads and writes the application only through an
:class:`~spectroview.ai_agent.agent.ports.AppContext`, so it knows nothing about
Qt or the ViewModel and can be unit-tested against a fake context.

Graph tools do not draw anything themselves: they submit a typed
:mod:`~spectroview.ai_agent.agent.commands` object, which the context queues
until the agent turn ends.
"""
import json
from typing import Annotated, Any, List, Literal, Optional, Union

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from spectroview.ai_agent.agent.commands import CreatePlot, DeletePlots, UpdatePlot
from spectroview.ai_agent.agent.ports import AppContext
from spectroview.ai_agent.utils.df_summary import summarize_dataframe_columns
from spectroview.ai_agent.utils.safe_eval import evaluate_pandas_expression, format_query_result

# Spelled out rather than derived from spectroview.PLOT_STYLES: MCP builds each
# tool's JSON Schema from these annotations, so the literal must be statically
# analysable. test_ai_agent_schema.py fails if the two ever drift apart.
PlotStyle = Literal[
    "point", "scatter", "box", "bar", "line",
    "trendline", "histogram", "wafer", "2Dmap",
]
VALID_PLOT_STYLES = frozenset(PlotStyle.__args__)


def create_mcp_server(context: AppContext) -> FastMCP:
    """Create and configure the FastMCP server with SPECTROview tools.

    Parameters
    ----------
    context:
        The application the tools operate on. Anything satisfying
        :class:`AppContext` works, including
        :class:`~spectroview.ai_agent.agent.ports.RecordingContext` in tests.
    """
    mcp = FastMCP("SPECTROview")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _named_props(**values) -> dict:
        """Keep only the plot properties the model actually supplied.

        Omitted arguments arrive as None and must not reach the config, or
        they would overwrite the graph's existing values with nulls.
        """
        return {k: v for k, v in values.items() if v is not None}

    def _validate_filters(filters: Optional[List[str]], df: Optional[Any]) -> Optional[str]:
        """Dry-run each filter against *df*. Returns an error message, or None if all valid."""
        if not filters or df is None:
            return None
        for f in filters:
            _, error = evaluate_pandas_expression(df, f)
            if error is not None:
                return (
                    f"Error: filter {f!r} is invalid ({error}). Common cause: string values must be "
                    f"quoted, e.g. \"Zone == 'Edge'\" not \"Zone == Edge\"."
                )
        return None

    def _invalid_style_message(plot_style: str) -> str:
        return (f"Error: {plot_style!r} is not a valid plot_style. Valid values: "
                f"{', '.join(sorted(VALID_PLOT_STYLES))}.")

    # -------------------------------------------------------------------------
    # Resources
    # -------------------------------------------------------------------------
    #
    # Context too bulky to push into every system prompt, fetched on demand
    # instead. The prompt always carries the cheap half — DataFrame names and
    # their column names/dtypes — so the model can never invent a column; what
    # lives here is the detail it only occasionally needs. The client reaches
    # these through the hub's `get_context` tool, since no LLM API has a native
    # notion of an MCP resource.

    @mcp.resource("spectroview://dataframes/detail")
    def dataframes_detail() -> str:
        """Sample values and a row preview for every loaded DataFrame."""
        names = context.list_dataframes()
        if not names:
            return "No DataFrames are currently loaded."

        parts: List[str] = []
        for name in names:
            df = context.get_dataframe(name)
            if df is None:
                continue
            try:
                preview = df.head(3).to_string(max_cols=8)
            except Exception:                   # noqa: BLE001
                preview = "(preview unavailable)"
            parts.append(
                f"DATAFRAME: {name!r} ({len(df)} rows, {len(df.columns)} columns)\n"
                f"  Columns:\n{summarize_dataframe_columns(df)}\n"
                f"  Preview:\n{preview}"
            )
        return "\n\n".join(parts)

    @mcp.resource("spectroview://graphs/detail")
    def graphs_detail() -> str:
        """Full configuration of every currently open graph."""
        graphs = context.list_graphs()
        if not graphs:
            return "No graphs are currently open."
        return json.dumps(
            [{"id": gid, **info} for gid, info in sorted(graphs.items())],
            indent=2, default=str,
        )

    # -------------------------------------------------------------------------
    # Tools
    # -------------------------------------------------------------------------

    @mcp.tool()
    def query_dataframe(query: str, df_name: str = "") -> str:
        """Filter or query data from the dataframe and return a summary of the result.

        Args:
            query: A pandas expression string. Simple filters use bare column names
                (e.g., "age > 30 and city == 'NY'"). Aggregations/groupby expressions
                use `df` (e.g., "df.groupby('Slot')['x'].mean().idxmax()").
            df_name: The name of the dataframe to query. If empty, uses the active one.
        """
        df = context.get_dataframe(df_name)
        if df is None:
            return f"Error: DataFrame '{df_name or context.active_dataframe_name()}' not found."

        result, error = evaluate_pandas_expression(df, query)
        if error is not None:
            return f"Error evaluating query: {error}"
        return format_query_result(result)

    @mcp.tool()
    def plot_graph(
        x: str,
        y: Union[str, List[str]],
        plot_style: PlotStyle,
        z: Annotated[Optional[str], Field(description=(
            "Grouping / colour column. For 'wafer' and '2Dmap' this MUST be the metric "
            "value to visualise. For every OTHER style it is the hue: the data is split "
            "into one coloured series per distinct value of z, while x and y stay "
            "unchanged. Use z — never x — when the user says 'group by', 'colour by', "
            "'split by', 'per', or 'for each' a categorical column."))] = None,
        filters: Optional[List[str]] = None,
        df_name: str = "",
        grid: Annotated[Optional[bool], Field(description="Show grid lines. Default false — omit unless the user explicitly asks for grid lines.")] = None,
        plot_title: Annotated[Optional[str], Field(description="Custom plot title. Omit unless the user explicitly provides one.")] = None,
        xlabel: Annotated[Optional[str], Field(description="Custom X-axis label. Omit unless explicitly provided.")] = None,
        ylabel: Annotated[Optional[str], Field(description="Custom Y-axis label. Omit unless explicitly provided.")] = None,
        zlabel: Annotated[Optional[str], Field(description="Custom Z-axis/colorbar label. Omit unless explicitly provided.")] = None,
        xmin: Annotated[Optional[float], Field(description="X-axis lower limit. Omit unless the user provides a numeric value.")] = None,
        xmax: Annotated[Optional[float], Field(description="X-axis upper limit. Omit unless the user provides a numeric value.")] = None,
        ymin: Annotated[Optional[float], Field(description="Y-axis lower limit. Omit unless the user provides a numeric value.")] = None,
        ymax: Annotated[Optional[float], Field(description="Y-axis upper limit. Omit unless the user provides a numeric value.")] = None,
        zmin: Annotated[Optional[float], Field(description="Z-axis lower limit. Omit unless the user provides a numeric value.")] = None,
        zmax: Annotated[Optional[float], Field(description="Z-axis upper limit. Omit unless the user provides a numeric value.")] = None,
        color_palette: Annotated[Optional[str], Field(description="Color palette name. Default 'jet'. Only set if the user requests a specific palette (e.g. 'viridis', 'plasma').")] = None,
        xlogscale: Annotated[Optional[bool], Field(description="Log scale on the X axis. Default false.")] = None,
        ylogscale: Annotated[Optional[bool], Field(description="Log scale on the Y axis. Default false.")] = None,
        scatter_size: Annotated[Optional[int], Field(description="Marker size for scatter/point plots.")] = None,
        hist_bins: Annotated[Optional[int], Field(description="Number of histogram bins.")] = None,
        trendline_order: Annotated[Optional[int], Field(description="Polynomial order for trendline plots.")] = None,
        other_properties: Annotated[Optional[dict], Field(description="Catch-all for properties without a dedicated parameter above (e.g. 'x_rot', 'plot_width', 'plot_height', 'dpi', 'hist_kde'). Prefer the named parameters above when available.")] = None,
    ) -> str:
        """Create a new graph from a loaded DataFrame. One tool call = one graph window.

        Args:
            x: Column name for X-axis. For 'wafer' and '2Dmap', this MUST be the X-coordinate column.
            y: Column name(s) for Y-axis (can be a string or a list of strings). For 'wafer' and '2Dmap', this MUST be the Y-coordinate column, NOT the metric value.
            plot_style: The visual style.
            z: Grouping / colour (hue) column — see the parameter description. "Group by Zone" means z='Zone', NOT x='Zone'.
            filters: Optional list of pandas query strings to filter data. String values MUST be quoted (e.g., ["Zone == 'Edge'", "Yield > 90"]).
            df_name: Optional target DataFrame name. If empty, uses the active one.
        """
        target_df_name = df_name or context.active_dataframe_name()

        if plot_style not in VALID_PLOT_STYLES:
            return _invalid_style_message(plot_style) + " This plot was NOT created; please retry."

        filter_error = _validate_filters(filters, context.get_dataframe(target_df_name))
        if filter_error is not None:
            return filter_error + " This plot was NOT created; please fix the filter and retry."

        config = {
            "x": x,
            "y": y if isinstance(y, list) else [y],
            "plot_style": plot_style,
            "z": z,
            "filters": filters or [],
            "df_name": target_df_name,
        }
        if other_properties:
            config.update(other_properties)
        config.update(_named_props(
            grid=grid, plot_title=plot_title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
            color_palette=color_palette, xlogscale=xlogscale, ylogscale=ylogscale,
            scatter_size=scatter_size, hist_bins=hist_bins, trendline_order=trendline_order,
        ))

        context.submit(CreatePlot(config))
        return "Plot command sent to UI successfully."

    @mcp.tool()
    def get_statistics(columns: List[str], df_name: str = "") -> str:
        """Compute descriptive statistics for specified columns.

        Args:
            columns: List of column names to compute statistics for.
            df_name: Optional target DataFrame name. If empty, uses the active one.
        """
        df = context.get_dataframe(df_name)
        if df is None:
            return f"Error: DataFrame '{df_name or context.active_dataframe_name()}' not found."

        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return "Error: None of the requested columns exist in the DataFrame."

        try:
            return f"Statistics:\n{df[valid_cols].describe().to_string()}"
        except Exception as e:
            return f"Error computing statistics: {e}"

    def _resolve_graph_df(graph_id: str) -> Optional[Any]:
        """Best-effort lookup of the DataFrame backing an open graph, for filter validation."""
        if str(graph_id).strip().lower() == "all":
            return None  # could span multiple DataFrames — validating against one would mislead
        try:
            gid_int = int(graph_id)
        except (TypeError, ValueError):
            return None
        info = context.list_graphs().get(gid_int)
        if not info:
            return None
        return context.get_dataframe(info.get("df", ""))

    @mcp.tool()
    def update_graph(
        graph_id: str,
        x: Annotated[Optional[str], Field(description=(
            "New X-axis column. Omit to keep the graph's current X axis — only pass "
            "this when the user explicitly asks to change what is on the X axis."))] = None,
        y: Optional[Union[str, List[str]]] = None,
        plot_style: Optional[PlotStyle] = None,
        z: Annotated[Optional[str], Field(description=(
            "Grouping / colour (hue) column. Setting z splits the EXISTING plot into one "
            "coloured series per distinct value, leaving x and y as they are. 'Group the "
            "data by Zone' means z='Zone' and x untouched — it does NOT mean x='Zone'."))] = None,
        filters: Optional[List[str]] = None,
        grid: Annotated[Optional[bool], Field(description="Show grid lines.")] = None,
        plot_title: Annotated[Optional[str], Field(description="Custom plot title.")] = None,
        xlabel: Annotated[Optional[str], Field(description="Custom X-axis label.")] = None,
        ylabel: Annotated[Optional[str], Field(description="Custom Y-axis label.")] = None,
        zlabel: Annotated[Optional[str], Field(description="Custom Z-axis/colorbar label.")] = None,
        xmin: Annotated[Optional[float], Field(description="X-axis lower limit.")] = None,
        xmax: Annotated[Optional[float], Field(description="X-axis upper limit.")] = None,
        ymin: Annotated[Optional[float], Field(description="Y-axis lower limit.")] = None,
        ymax: Annotated[Optional[float], Field(description="Y-axis upper limit.")] = None,
        zmin: Annotated[Optional[float], Field(description="Z-axis lower limit.")] = None,
        zmax: Annotated[Optional[float], Field(description="Z-axis upper limit.")] = None,
        color_palette: Annotated[Optional[str], Field(description="Color palette name, e.g. 'jet', 'viridis'.")] = None,
        xlogscale: Annotated[Optional[bool], Field(description="Log scale on the X axis.")] = None,
        ylogscale: Annotated[Optional[bool], Field(description="Log scale on the Y axis.")] = None,
        scatter_size: Annotated[Optional[int], Field(description="Marker size for scatter/point plots.")] = None,
        hist_bins: Annotated[Optional[int], Field(description="Number of histogram bins.")] = None,
        trendline_order: Annotated[Optional[int], Field(description="Polynomial order for trendline plots.")] = None,
        other_properties: Annotated[Optional[dict], Field(description="Catch-all for properties without a dedicated parameter above. Prefer the named parameters above when available.")] = None,
    ) -> str:
        """Update an existing graph by ID.

        Pass ONLY the properties the user asked to change. Every argument you omit
        keeps its current value; every argument you pass overwrites it. Re-sending an
        axis the user never mentioned is the most common way this tool goes wrong.

        Args:
            graph_id: The ID of the graph to update (e.g., '1', '2') or 'all' to update all open graphs.
            x: New X-axis column — omit unless the user asked to change the X axis.
            y: Optional new column name(s) for Y-axis (string or list of strings).
            plot_style: Optional new style of the plot.
            z: Grouping / colour (hue) column — see the parameter description. "Group by Zone" means z='Zone' with x left alone.
            filters: Optional new list of pandas query strings to filter data. To keep existing filters while adding new ones, you MUST include the existing filters in this list. String values MUST be quoted (e.g., ["Zone == 'Edge'"]).
        """
        if plot_style is not None and plot_style not in VALID_PLOT_STYLES:
            return _invalid_style_message(plot_style) + " This update was NOT applied; please retry."

        filter_error = _validate_filters(filters, _resolve_graph_df(graph_id))
        if filter_error is not None:
            return filter_error + " This update was NOT applied; please fix the filter and retry."

        update_props: dict = {}
        if x is not None: update_props["x"] = x
        if y is not None: update_props["y"] = y if isinstance(y, list) else [y]
        if plot_style is not None: update_props["plot_style"] = plot_style
        if z is not None: update_props["z"] = z
        if filters is not None: update_props["filters"] = filters
        if other_properties:
            update_props.update(other_properties)
        update_props.update(_named_props(
            grid=grid, plot_title=plot_title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
            color_palette=color_palette, xlogscale=xlogscale, ylogscale=ylogscale,
            scatter_size=scatter_size, hist_bins=hist_bins, trendline_order=trendline_order,
        ))

        context.submit(UpdatePlot(graph_id=graph_id, properties=update_props))
        return f"Update command for graph {graph_id} sent to UI successfully."

    @mcp.tool()
    def delete_graph(delete_all: bool = False, graph_ids: Optional[List[int]] = None) -> str:
        """Delete/close specific graphs or all graphs.

        Args:
            delete_all: If true, closes all graphs.
            graph_ids: List of specific graph IDs to close (e.g., [1, 2]). Ignored if delete_all is true.
        """
        context.submit(DeletePlots(delete_all=delete_all, graph_ids=graph_ids or []))
        return "Delete command sent to UI successfully."

    return mcp
