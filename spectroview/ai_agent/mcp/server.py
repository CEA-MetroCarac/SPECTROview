"""spectroview/ai_agent/mcp/server.py

Model Context Protocol (MCP) Server for SPECTROview.
"""
import json
from typing import Annotated, Any, List, Literal, Optional, Union

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from spectroview.ai_agent.utils.safe_eval import evaluate_pandas_expression, format_query_result

PlotStyle = Literal[
    "point", "scatter", "box", "bar", "line",
    "trendline", "histogram", "wafer", "2Dmap",
]
VALID_PLOT_STYLES = frozenset(PlotStyle.__args__)

# To communicate plot configurations back to the UI, we can use a callback or just return JSON string.
# We will define a factory function to create the server, injecting a reference to vm_chat.

def create_mcp_server(vm_chat) -> FastMCP:
    """Create and configure the FastMCP server with SPECTROview resources and tools."""
    mcp = FastMCP("SPECTROview")

    # -------------------------------------------------------------------------
    # Resources
    # -------------------------------------------------------------------------
    
    @mcp.resource("spectroview://dataframes/active")
    def get_active_dataframe_schema() -> str:
        """Get the schema of the currently active DataFrame."""
        df_name = vm_chat._active_df_name
        if not df_name or df_name not in vm_chat._dfs:
            return "No active DataFrame."
        
        df = vm_chat._dfs[df_name]
        col_info = ", ".join(f"{col} ({df[col].dtype})" for col in df.columns)
        
        # Include a compact sample of unique values for small-cardinality columns
        sample_parts: list[str] = []
        for col in df.columns:
            n_unique = df[col].nunique()
            if n_unique <= 10:
                vals = df[col].dropna().unique().tolist()
                sample_parts.append(f"  - {col}: {vals[:10]}")
        sample_block = "\n".join(sample_parts) if sample_parts else ""
        
        return (
            f"DataFrame '{df_name}': {len(df)} rows × {len(df.columns)} columns\n"
            f"Columns: {col_info}\n"
            + (f"Sample categories:\n{sample_block}" if sample_block else "")
        )

    @mcp.resource("spectroview://dataframes/list")
    def list_dataframes() -> str:
        """List all loaded DataFrames in the workspace."""
        if not vm_chat._dfs:
            return "No DataFrames loaded."
        return json.dumps(list(vm_chat._dfs.keys()))

    @mcp.resource("spectroview://graphs/open")
    def list_open_graphs() -> str:
        """List all currently open graphs and their IDs."""
        if not vm_chat._graphs:
            return "No open graphs."
        
        graphs_info = []
        for gid, info in vm_chat._graphs.items():
            graphs_info.append({
                "id": gid,
                "style": info.get("style"),
                "x": info.get("x"),
                "y": info.get("y"),
                "z": info.get("z"),
                "df": info.get("df"),
                "filters": info.get("filters", [])
            })
        return json.dumps(graphs_info, indent=2)

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
        target_df = df_name or vm_chat._active_df_name
        if not target_df or target_df not in vm_chat._dfs:
            return f"Error: DataFrame '{target_df}' not found."

        df = vm_chat._dfs[target_df]
        result, error = evaluate_pandas_expression(df, query)
        if error is not None:
            return f"Error evaluating query: {error}"
        return format_query_result(result)

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

    @mcp.tool()
    def plot_graph(
        x: str,
        y: Union[str, List[str]],
        plot_style: PlotStyle,
        z: Optional[str] = None,
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
            z: Optional column name for Z-axis or color encoding (hue). For 'wafer' and '2Dmap', this MUST be the metric value to visualize. For 'point'/'scatter', used as hue.
            filters: Optional list of pandas query strings to filter data. String values MUST be quoted (e.g., ["Zone == 'Edge'", "Yield > 90"]).
            df_name: Optional target DataFrame name. If empty, uses the active one.
        """
        target_df_name = df_name or vm_chat._active_df_name

        if plot_style not in VALID_PLOT_STYLES:
            return (f"Error: {plot_style!r} is not a valid plot_style. Valid values: "
                    f"{', '.join(sorted(VALID_PLOT_STYLES))}. This plot was NOT created; please retry.")

        filter_error = _validate_filters(filters, vm_chat._dfs.get(target_df_name))
        if filter_error is not None:
            return filter_error + " This plot was NOT created; please fix the filter and retry."

        named_props = {
            "grid": grid, "plot_title": plot_title, "xlabel": xlabel, "ylabel": ylabel, "zlabel": zlabel,
            "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "zmin": zmin, "zmax": zmax,
            "color_palette": color_palette, "xlogscale": xlogscale, "ylogscale": ylogscale,
            "scatter_size": scatter_size, "hist_bins": hist_bins, "trendline_order": trendline_order,
        }

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
        config.update({k: v for k, v in named_props.items() if v is not None})

        if not hasattr(vm_chat, '_pending_plots'):
            vm_chat._pending_plots = []
        vm_chat._pending_plots.append(config)

        return "Plot command sent to UI successfully."

    @mcp.tool()
    def get_statistics(columns: List[str], df_name: str = "") -> str:
        """Compute descriptive statistics for specified columns.
        
        Args:
            columns: List of column names to compute statistics for.
            df_name: Optional target DataFrame name. If empty, uses the active one.
        """
        target_df = df_name or vm_chat._active_df_name
        if not target_df or target_df not in vm_chat._dfs:
            return f"Error: DataFrame '{target_df}' not found."
            
        df = vm_chat._dfs[target_df]
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return "Error: None of the requested columns exist in the DataFrame."
            
        try:
            stats = df[valid_cols].describe()
            return f"Statistics:\n{stats.to_string()}"
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
        info = vm_chat._graphs.get(gid_int)
        if not info:
            return None
        return vm_chat._dfs.get(info.get("df"))

    @mcp.tool()
    def update_graph(
        graph_id: str,
        x: Optional[str] = None,
        y: Optional[Union[str, List[str]]] = None,
        plot_style: Optional[PlotStyle] = None,
        z: Optional[str] = None,
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

        Args:
            graph_id: The ID of the graph to update (e.g., '1', '2') or 'all' to update all open graphs.
            x: Optional new column name for X-axis.
            y: Optional new column name(s) for Y-axis (string or list of strings).
            plot_style: Optional new style of the plot.
            z: Optional new column name for Z-axis or color encoding (hue).
            filters: Optional new list of pandas query strings to filter data. To keep existing filters while adding new ones, you MUST include the existing filters in this list. String values MUST be quoted (e.g., ["Zone == 'Edge'"]).
        """
        if plot_style is not None and plot_style not in VALID_PLOT_STYLES:
            return (f"Error: {plot_style!r} is not a valid plot_style. Valid values: "
                    f"{', '.join(sorted(VALID_PLOT_STYLES))}. This update was NOT applied; please retry.")

        filter_error = _validate_filters(filters, _resolve_graph_df(graph_id))
        if filter_error is not None:
            return filter_error + " This update was NOT applied; please fix the filter and retry."

        named_props = {
            "grid": grid, "plot_title": plot_title, "xlabel": xlabel, "ylabel": ylabel, "zlabel": zlabel,
            "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "zmin": zmin, "zmax": zmax,
            "color_palette": color_palette, "xlogscale": xlogscale, "ylogscale": ylogscale,
            "scatter_size": scatter_size, "hist_bins": hist_bins, "trendline_order": trendline_order,
        }

        update_props = {}
        if x is not None: update_props["x"] = x
        if y is not None: update_props["y"] = y if isinstance(y, list) else [y]
        if plot_style is not None: update_props["plot_style"] = plot_style
        if z is not None: update_props["z"] = z
        if filters is not None: update_props["filters"] = filters
        if other_properties:
            update_props.update(other_properties)
        update_props.update({k: v for k, v in named_props.items() if v is not None})

        config = {
            "_graph_update": {
                "graph_id": graph_id,
                "properties": update_props
            }
        }

        if not hasattr(vm_chat, '_pending_plots'):
            vm_chat._pending_plots = []
        vm_chat._pending_plots.append(config)

        return f"Update command for graph {graph_id} sent to UI successfully."

    @mcp.tool()
    def delete_graph(delete_all: bool = False, graph_ids: Optional[List[int]] = None) -> str:
        """Delete/close specific graphs or all graphs.
        
        Args:
            delete_all: If true, closes all graphs.
            graph_ids: List of specific graph IDs to close (e.g., [1, 2]). Ignored if delete_all is true.
        """
        config = {
            "_graph_delete": {
                "delete_all": delete_all,
                "graph_ids": graph_ids or []
            }
        }
        
        if not hasattr(vm_chat, '_pending_plots'):
            vm_chat._pending_plots = []
        vm_chat._pending_plots.append(config)
        
        return "Delete command sent to UI successfully."

    return mcp
