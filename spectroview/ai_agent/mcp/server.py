"""spectroview/ai_agent/mcp/server.py

Model Context Protocol (MCP) Server for SPECTROview.
"""
import json
from typing import Any, List, Optional

from mcp.server.fastmcp import FastMCP

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
            query: A valid pandas `.query()` expression string (e.g., "age > 30 and city == 'NY'").
            df_name: The name of the dataframe to query. If empty, uses the active one.
        """
        target_df = df_name or vm_chat._active_df_name
        if not target_df or target_df not in vm_chat._dfs:
            return f"Error: DataFrame '{target_df}' not found."
            
        df = vm_chat._dfs[target_df]
        try:
            import pandas as pd
            import numpy as np
            local_vars = {"df": df, "pd": pd, "np": np}
            res_obj = eval(query, {"__builtins__": {}}, local_vars)
            
            if isinstance(res_obj, pd.DataFrame):
                return f"Query returned a DataFrame with {len(res_obj)} rows. First few rows:\n{res_obj.head().to_string()}"
            elif isinstance(res_obj, pd.Series):
                return f"Query returned a Series. First few elements:\n{res_obj.head().to_string()}"
            else:
                return f"Query result: {res_obj}"
        except Exception as e:
            return f"Error evaluating query: {e}"

    @mcp.tool()
    def plot_graph(x: str, y: Any, plot_style: str, z: Optional[str] = None, filters: Optional[List[str]] = None, other_properties: Optional[dict] = None, df_name: str = "") -> str:
        """Plot a graph using loaded dataframes.
        
        Args:
            x: Column name for X-axis. For 'wafer' and '2Dmap', this MUST be the X-coordinate column.
            y: Column name(s) for Y-axis (can be a string or a list of strings). For 'wafer' and '2Dmap', this MUST be the Y-coordinate column, NOT the metric value.
            plot_style: The visual style (e.g., 'point', 'line', 'bar', 'wafer', '2Dmap', 'trendline', 'histogram', 'scatter', 'box').
            z: Optional column name for Z-axis or color encoding (hue). For 'wafer' and '2Dmap', this MUST be the metric value to visualize. For 'point'/'scatter', used as hue.
            filters: Optional list of pandas query strings to filter data (e.g. ['Zone == 1', 'Yield > 90']).
            other_properties: A dictionary of other properties to set (e.g. {'grid': True, 'x_rot': 45, 'ylabel': 'AAA', 'xlabel': 'BBB', 'plot_title': 'CCC'}).
            df_name: Optional target DataFrame name. If empty, uses the active one.
        """
        config = {
            "x": x,
            "y": y if isinstance(y, list) else [y],
            "plot_style": plot_style,
            "z": z,
            "filters": filters or [],
            "df_name": df_name or vm_chat._active_df_name
        }
        if other_properties:
            config.update(other_properties)
        
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

    @mcp.tool()
    def update_graph(graph_id: str, x: Optional[str] = None, y: Optional[Any] = None, plot_style: Optional[str] = None, z: Optional[str] = None, filters: Optional[List[str]] = None, other_properties: Optional[dict] = None) -> str:
        """Update an existing graph by ID.
        
        Args:
            graph_id: The ID of the graph to update (e.g., '1', '2') or 'all' to update all open graphs.
            x: Optional new column name for X-axis.
            y: Optional new column name(s) for Y-axis (string or list of strings).
            plot_style: Optional new style of the plot.
            z: Optional new column name for Z-axis or color encoding (hue).
            filters: Optional new list of pandas query strings to filter data. To keep existing filters while adding new ones, you MUST include the existing filters in this list.
            other_properties: A dictionary of other properties to update (e.g. {'grid': True, 'x_rot': 45, 'ylabel': 'AAA', 'xlabel': 'BBB', 'plot_title': 'CCC'}).
        """
        update_props = {}
        if x is not None: update_props["x"] = x
        if y is not None: update_props["y"] = y if isinstance(y, list) else [y]
        if plot_style is not None: update_props["plot_style"] = plot_style
        if z is not None: update_props["z"] = z
        if filters is not None: update_props["filters"] = filters
        if other_properties is not None:
            update_props.update(other_properties)
        
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
    def delete_graph(delete_all: bool = False, graph_ids: List[int] = None) -> str:
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
