"""spectroview/ai_agent/mcp/server.py

Model Context Protocol (MCP) Server for SPECTROview.

Exposes the operations the AI Agent may perform on the user's data as MCP
tools. The server reads and writes the application only through an
:class:`~spectroview.ai_agent.agent.ports.AppContext`, so it knows nothing about
Qt or the ViewModel and can be unit-tested against a fake context.

Graph tools do not draw anything themselves: they submit a typed
:mod:`~spectroview.ai_agent.agent.commands` object. The chat context queues it
until the agent turn ends; the external desktop context executes it through the
Qt-safe running-application facade.
"""
import json
from typing import Annotated, Any, List, Literal, Optional, Union

try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from spectroview.ai_agent.agent.commands import CreatePlot, DeletePlots, UpdatePlot
from spectroview.ai_agent.agent.ports import AppContext
from spectroview.ai_agent.utils.df_summary import summarize_dataframe_columns
from spectroview.ai_agent.utils.safe_eval import evaluate_pandas_expression, format_query_result
from spectroview.application.errors import ApplicationAPIError
from spectroview.model.graph_control import (
    GraphPatch,
    GraphValidationError,
    graph_patch_to_dict,
    normalize_graph_patch,
)

# Spelled out rather than derived from spectroview.PLOT_STYLES: MCP builds each
# tool's JSON Schema from these annotations, so the literal must be statically
# analysable. test_ai_agent_schema.py fails if the two ever drift apart.
PlotStyle = Literal[
    "point", "scatter", "box", "bar", "line",
    "trendline", "histogram", "wafer", "2Dmap",
]
VALID_PLOT_STYLES = frozenset(PlotStyle.__args__)

READ_ONLY = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
)
STATE_CHANGE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
)
STATE_REPLACE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False
)
FILESYSTEM_WRITE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True
)


WorkspaceName = Literal["spectra", "maps", "graphs"]


def create_mcp_server(
    context: AppContext,
    *,
    include_application_tools: bool = False,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> FastMCP:
    """Create and configure the FastMCP server with SPECTROview tools.

    Parameters
    ----------
    context:
        The application the tools operate on. Anything satisfying
        :class:`AppContext` works, including
        :class:`~spectroview.ai_agent.agent.ports.RecordingContext` in tests.
    """
    try:
        mcp = FastMCP(
            "SPECTROview",
            host=host,
            port=port,
            streamable_http_path="/mcp",
        )
    except TypeError:
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

    def _merge_properties(advanced: Optional[Any], **named: Any) -> dict:
        """Merge the schema-complete advanced patch with common arguments.

        Named arguments win when both forms contain a field.  This keeps the
        original public MCP API backward compatible while replacing its old
        unstructured catch-all dict with the MGraph-derived ``GraphPatch``.
        """
        merged = graph_patch_to_dict(advanced)
        merged.update(_named_props(**named))
        return merged

    def _validate_filters(filters: Optional[List[Any]], df: Optional[Any]) -> Optional[str]:
        """Dry-run each filter against *df*. Returns an error message, or None if all valid."""
        if not filters or df is None:
            return None
        for f in filters:
            expression = f.get("expression", "") if isinstance(f, dict) else f
            if isinstance(f, dict) and not f.get("state", True):
                continue
            _, error = evaluate_pandas_expression(df, expression)
            if error is not None:
                return (
                    f"Error: filter {expression!r} is invalid ({error}). Common cause: string values must be "
                    f"quoted, e.g. \"Zone == 'Edge'\" not \"Zone == Edge\"."
                )
        return None

    def _invalid_style_message(plot_style: str) -> str:
        return (f"Error: {plot_style!r} is not a valid plot_style. Valid values: "
                f"{', '.join(sorted(VALID_PLOT_STYLES))}.")

    def _application_result(method_name: str, *args: Any, **kwargs: Any) -> dict:
        """Return one predictable structured result from the application API."""
        method = getattr(context, method_name, None)
        if method is None:
            return {
                "ok": False,
                "error": {
                    "code": "APPLICATION_NOT_READY",
                    "message": "This MCP session is not attached to the running desktop application.",
                },
            }
        try:
            return {"ok": True, "result": method(*args, **kwargs)}
        except ApplicationAPIError as exc:
            return {"ok": False, "error": exc.as_dict()}
        except Exception as exc:  # keep client-visible failures actionable
            return {
                "ok": False,
                "error": {"code": "INTERNAL_ERROR", "message": str(exc)},
            }

    def _submit_command(command: Any, queued_message: str) -> str:
        """Keep chat responses stable; give desktop clients an outcome envelope."""
        try:
            outcome = context.submit(command)
        except ApplicationAPIError as exc:
            if include_application_tools:
                err = exc.as_dict()
                # Tell the LLM not to retry render failures
                if err.get("code") == "GRAPH_RENDER_FAILED":
                    err["retry"] = False
                return json.dumps({"ok": False, "error": err})
            return f"Error: {exc.message}"
        except Exception as exc:
            if include_application_tools:
                return json.dumps({
                    "ok": False,
                    "error": {"code": "INTERNAL_ERROR", "message": str(exc)},
                })
            return f"Error: {exc}"
        if include_application_tools:
            if isinstance(outcome, dict) and outcome.get("image_path"):
                gid = outcome.get("graph_id")
                img_path = outcome.get("image_path")
                from pathlib import Path
                url_path = Path(img_path).as_uri()
                title = (command.config.get("plot_title") if hasattr(command, "config") else None) or f"Graph #{gid}"
                msg = (
                    f"Plot #{gid} successfully created in SPECTROview.\n\n"
                    f"![{title}]({url_path})\n\n"
                    f"[Open in SPECTROview](plume-spectroview://graph/{gid})"
                )
                outcome["display_markdown"] = msg
                return f"{msg}\n\nDetails: " + json.dumps({"ok": True, "result": outcome}, default=str)
            elif isinstance(outcome, dict) and outcome.get("graph_id") is not None:
                gid = outcome.get("graph_id")
                msg = (
                    f"Plot #{gid} successfully created in SPECTROview.\n\n"
                    f"[Open in SPECTROview](plume-spectroview://graph/{gid})"
                )
                outcome["display_markdown"] = msg
                return f"{msg}\n\nDetails: " + json.dumps({"ok": True, "result": outcome}, default=str)
            return json.dumps({"ok": True, "result": outcome}, default=str)
        return queued_message

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

    if include_application_tools:
        @mcp.resource("spectroview://application/state")
        def application_state_resource() -> str:
            """Current workspace, selection, dataset counts, and fitting state."""
            return json.dumps(_application_result("get_application_state"), indent=2)

        @mcp.resource("spectroview://workspace/current")
        def current_workspace_resource() -> str:
            """The active workspace and its current domain selection."""
            state = {
                "workspace": _application_result("get_active_workspace"),
                "selection": _application_result("get_current_selection"),
            }
            return json.dumps(state, indent=2)

        @mcp.resource("spectroview://datasets")
        def datasets_resource() -> str:
            """Compact catalog of datasets loaded in all workspaces."""
            return json.dumps(_application_result("list_datasets"), indent=2)

        @mcp.resource("spectroview://graphs/current")
        def current_graph_resource() -> str:
            """Full configuration of the active graph."""
            return json.dumps(_application_result("get_active_graph"), indent=2)

        @mcp.resource("spectroview://fit/current")
        def current_fit_resource() -> str:
            """Fit configuration for the active spectrum or map."""
            return json.dumps(_application_result("get_fit_configuration"), indent=2)

        # Application/workspace -------------------------------------------------

        @mcp.tool(annotations=READ_ONLY)
        def get_application_state() -> dict:
            """Inspect the running SPECTROview session.

            Returns the active workspace, current selection, loaded-object
            counts, and whether a Spectra or Maps fit is in progress. Read-only.
            """
            return _application_result("get_application_state")

        @mcp.tool(annotations=READ_ONLY)
        def get_active_workspace() -> dict:
            """Return the workspace currently visible in SPECTROview. Read-only."""
            return _application_result("get_active_workspace")

        @mcp.tool(annotations=READ_ONLY)
        def get_current_selection() -> dict:
            """Return the domain selection in the active workspace. Read-only."""
            return _application_result("get_current_selection")

        # Datasets/spectra ------------------------------------------------------

        @mcp.tool(annotations=READ_ONLY)
        def list_datasets(workspace: Optional[WorkspaceName] = None) -> dict:
            """List loaded spectra, maps, and DataFrames with stable dataset IDs.

            Args:
                workspace: Optional workspace filter. Omit to list everything.
            """
            return _application_result("list_datasets", workspace)

        @mcp.tool(annotations=READ_ONLY)
        def get_dataset_info(dataset_id: str) -> dict:
            """Describe one loaded dataset without returning its full arrays.

            Args:
                dataset_id: ID from list_datasets, such as ``spectra:sample``.
            """
            return _application_result("get_dataset_info", dataset_id)

        @mcp.tool(annotations=READ_ONLY)
        def get_spectrum(
            dataset_id: str,
            spectrum: Annotated[
                Optional[Union[int, str]],
                Field(description="Zero-based row index or exact spectrum name; defaults to row 0."),
            ] = None,
            processed: bool = True,
            max_points: Annotated[int, Field(ge=2, le=10000)] = 1000,
        ) -> dict:
            """Return X/Y values for one spectrum, downsampled to a safe size.

            Args:
                dataset_id: A spectra or map dataset ID from list_datasets.
                spectrum: Row index or name within a map; omit for the first row.
                processed: True for current processed arrays, false for raw arrays.
                max_points: Maximum returned X/Y samples (2 to 10000).
            """
            return _application_result(
                "get_spectrum", dataset_id, spectrum, processed, max_points
            )

        @mcp.tool(annotations=READ_ONLY)
        def get_current_spectrum(
            processed: bool = True,
            max_points: Annotated[int, Field(ge=2, le=10000)] = 1000,
        ) -> dict:
            """Return the selected spectrum in the active Spectra/Maps workspace."""
            return _application_result("get_current_spectrum", processed, max_points)

        # Processing ------------------------------------------------------------

        @mcp.tool(annotations=STATE_REPLACE)
        def crop_spectrum(
            workspace: Literal["spectra", "maps"],
            xmin: float,
            xmax: float,
            apply_all: bool = False,
        ) -> dict:
            """Crop selected spectral data using SPECTROview's existing processing path.

            This changes application state. In Spectra, apply_all targets every
            checked spectrum; in Maps it targets every loaded map. Otherwise it
            targets the current selection/current map.
            """
            return _application_result(
                "crop_spectrum", workspace, xmin, xmax, apply_all
            )

        @mcp.tool(annotations=STATE_CHANGE)
        def normalize_spectrum(
            workspace: Literal["spectra", "maps"],
            factor: Annotated[float, Field(description="Finite non-zero divisor.")],
            apply_all: bool = False,
        ) -> dict:
            """Divide selected intensities by a factor using existing processing logic.

            This changes application state. ``apply_all`` means all checked
            spectra; for Maps those spectra are within the current map.
            """
            return _application_result(
                "normalize_spectrum", workspace, factor, apply_all
            )

        @mcp.tool(annotations=STATE_REPLACE)
        def subtract_baseline(
            workspace: Literal["spectra", "maps"], apply_all: bool = False
        ) -> dict:
            """Subtract an already-configured baseline from selected spectral data.

            This changes application state and fails if a target has no baseline
            configuration. It does not invent or replace baseline parameters.
            """
            return _application_result("subtract_baseline", workspace, apply_all)

        # Fitting ---------------------------------------------------------------

        @mcp.tool(annotations=READ_ONLY)
        def get_fit_configuration(dataset_id: str = "") -> dict:
            """Return the current fit model for a spectra/map dataset. Read-only.

            Omit dataset_id to use the active selected spectrum or map.
            """
            return _application_result("get_fit_configuration", dataset_id)

        @mcp.tool(annotations=STATE_REPLACE)
        def fit_spectrum(
            workspace: Literal["spectra", "maps"], apply_all: bool = False
        ) -> dict:
            """Start the existing vectorized fit engine for configured targets.

            This changes fit state and returns immediately with ``status=started``;
            poll get_application_state and then call get_fit_results. It never
            replaces the fit model and never starts a second concurrent fit.
            """
            return _application_result("fit_spectrum", workspace, apply_all)

        @mcp.tool(annotations=STATE_CHANGE)
        def get_fit_results(
            workspace: Literal["spectra", "maps"],
            collect: bool = False,
            limit: Annotated[int, Field(ge=1, le=1000)] = 100,
        ) -> dict:
            """Return collected fit results as structured rows.

            Args:
                workspace: Spectra or Maps.
                collect: If true, rebuild the results table from current fit arrays.
                limit: Maximum returned rows; total_rows reports the full size.
            """
            return _application_result("get_fit_results", workspace, collect, limit)

        # Graph/map inspection --------------------------------------------------

        @mcp.tool(annotations=READ_ONLY)
        def list_graphs() -> dict:
            """List every graph with its complete, typed MGraph configuration."""
            return _application_result("list_graph_configurations")

        @mcp.tool(annotations=READ_ONLY)
        def get_active_graph() -> dict:
            """Return the complete configuration of the active graph. Read-only."""
            return _application_result("get_active_graph")

        @mcp.tool(annotations=READ_ONLY)
        def get_active_map() -> dict:
            """Return configuration and selection metadata for the active map."""
            return _application_result("get_active_map")

        # Explicit filesystem write --------------------------------------------

        @mcp.tool(annotations=FILESYSTEM_WRITE)
        def export_results(
            workspace: Literal["spectra", "maps"],
            output_path: str,
            overwrite: bool = False,
        ) -> dict:
            """Export collected fit results to CSV or Excel.

            This writes to the local filesystem. Existing files are rejected
            unless ``overwrite=true`` is explicitly supplied; parent folders are
            never created implicitly.
            """
            return _application_result(
                "export_results", workspace, output_path, overwrite
            )

    # -------------------------------------------------------------------------
    # Tools
    # -------------------------------------------------------------------------

    @mcp.tool(annotations=STATE_CHANGE)
    def load_dataframe(file_path: str) -> str:
        """Load an Excel (.xlsx, .xls) or CSV (.csv, .tsv) file into the SPECTROview workspace.

        Args:
            file_path: Absolute local path to the data file on disk.
        """
        from pathlib import Path
        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            return f"Error: File not found: {file_path}"
        loader = getattr(context, "load_dataframes", None)
        if callable(loader):
            try:
                loaded = loader([str(path)])
                names = ", ".join(loaded) if loaded else path.stem
                return f"Successfully loaded dataframe(s) into SPECTROview: {names}"
            except Exception as exc:
                return f"Error loading dataframe from {path.name}: {exc}"
        return "Error: DataFrame loading is not supported in this context."

    @mcp.tool(annotations=READ_ONLY)
    def show_graph(graph_id: Optional[int] = None) -> str:
        """Bring the SPECTROview application window to the front and display the specified graph.

        Args:
            graph_id: Optional ID of the graph to display and activate.
        """
        shower = getattr(context, "show_graph", None)
        if callable(shower):
            try:
                shower(graph_id)
                target = f"Graph #{graph_id}" if graph_id is not None else "Graphs workspace"
                return f"SPECTROview window activated and focused on {target}."
            except Exception as exc:
                return f"Error showing graph: {exc}"
        return "Error: Window activation is not supported in this context."


    @mcp.tool(annotations=READ_ONLY)
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

    @mcp.tool(annotations=STATE_CHANGE)
    def plot_graph(
        x: str,
        y: Union[str, List[str]],
        plot_style: PlotStyle,
        file_path: Annotated[Optional[str], Field(description=(
            "Optional local path to an Excel or CSV file. If specified and the dataset "
            "is not yet loaded in SPECTROview, it will be loaded automatically before plotting."
        ))] = None,
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
        other_properties: Annotated[Optional[GraphPatch], Field(description=(
            "Typed advanced graph patch. It exposes every mutable MGraph customization, including "
            "secondary axes, ticks/spines, font and figure settings, per-series legend_properties, "
            "error bars, histogram/wafer/trendline options, annotations, axis breaks, insets, and "
            "export geometry. Supply only requested fields; explicit null clears nullable fields. "
            "Prefer the common named parameters above when one exists."
        ))] = None,
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
        if file_path:
            from pathlib import Path
            fpath = Path(file_path).expanduser().resolve()
            if fpath.is_file():
                loader = getattr(context, "load_dataframes", None)
                if callable(loader):
                    try:
                        loaded = loader([str(fpath)])
                        if not df_name and loaded:
                            df_name = loaded[0]
                    except Exception:
                        pass

        target_df_name = df_name or context.active_dataframe_name()

        target_df = context.get_dataframe(target_df_name)
        if target_df is None:
            return f"Error: DataFrame {target_df_name!r} not found. This plot was NOT created."

        if plot_style not in VALID_PLOT_STYLES:
            return _invalid_style_message(plot_style) + " This plot was NOT created; please retry."

        filter_error = _validate_filters(filters, target_df)
        if filter_error is not None:
            return filter_error + " This plot was NOT created; please fix the filter and retry."

        config = _merge_properties(
            other_properties,
            grid=grid, plot_title=plot_title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
            color_palette=color_palette, xlogscale=xlogscale, ylogscale=ylogscale,
            scatter_size=scatter_size, hist_bins=hist_bins, trendline_order=trendline_order,
        )
        config.update({
            "x": x,
            "y": y if isinstance(y, list) else [y],
            "plot_style": plot_style,
            "z": z,
            "filters": filters or [],
            "df_name": target_df_name,
        })

        try:
            config = normalize_graph_patch(config, dataframe=target_df)
        except GraphValidationError as exc:
            return f"Error: invalid graph configuration ({exc}). This plot was NOT created; please retry."

        return _submit_command(
            CreatePlot(config),
            "Plot configuration successfully validated and queued for the Graphs workspace.",
        )

    @mcp.tool(annotations=READ_ONLY)
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

    @mcp.tool(annotations=STATE_CHANGE)
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
        other_properties: Annotated[Optional[GraphPatch], Field(description=(
            "Typed partial graph patch exposing every mutable Graph workspace property. Use it for "
            "advanced or multi-property changes; omitted fields are preserved and explicit null clears "
            "nullable values. Prefer common named parameters when available."
        ))] = None,
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

        update_props = _merge_properties(
            other_properties,
            grid=grid, plot_title=plot_title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
            color_palette=color_palette, xlogscale=xlogscale, ylogscale=ylogscale,
            scatter_size=scatter_size, hist_bins=hist_bins, trendline_order=trendline_order,
        )
        if x is not None:
            update_props["x"] = x
        if y is not None:
            update_props["y"] = y if isinstance(y, list) else [y]
        if plot_style is not None:
            update_props["plot_style"] = plot_style
        if z is not None:
            update_props["z"] = z
        if filters is not None:
            update_props["filters"] = filters

        if not update_props:
            return "Error: no graph properties were supplied; nothing was queued."

        graph_key = str(graph_id).strip().lower()
        graphs = context.list_graphs()
        if graph_key == "all":
            targets = sorted(graphs.items())
            if not targets:
                return "Error: no graphs are currently open; nothing was queued."
        else:
            try:
                graph_number = int(graph_id)
            except (TypeError, ValueError):
                return "Error: graph_id must be a numeric graph ID or 'all'; nothing was queued."
            if graph_number not in graphs:
                return f"Error: graph {graph_number} is not open; nothing was queued."
            targets = [(graph_number, graphs[graph_number])]

        normalized_by_target = []
        for target_id, current in targets:
            df_key = current.get("df_name", current.get("df", ""))
            target_df = context.get_dataframe(df_key)
            filter_error = _validate_filters(update_props.get("filters"), target_df)
            if filter_error is not None:
                return f"{filter_error} This update was NOT applied; please fix the filter and retry."
            try:
                normalized_by_target.append(normalize_graph_patch(
                    update_props,
                    current=current,
                    dataframe=target_df,
                ))
            except GraphValidationError as exc:
                return (
                    f"Error: update is invalid for graph {target_id} ({exc}). "
                    "No graphs were updated; please correct the properties and retry."
                )

        # One queued patch is sufficient when every target normalized to the
        # same values (the usual case).  Context-specific legend resets are
        # deterministic and have the same shape across targets.
        update_props = normalized_by_target[0]

        return _submit_command(
            UpdatePlot(graph_id=graph_id, properties=update_props),
            f"Update for graph {graph_id} successfully validated and queued for the Graphs workspace.",
        )

    @mcp.tool(annotations=STATE_REPLACE)
    def delete_graph(delete_all: bool = False, graph_ids: Optional[List[int]] = None) -> str:
        """Delete/close specific graphs or all graphs.

        Args:
            delete_all: If true, closes all graphs except any IDs listed in graph_ids.
            graph_ids: IDs to close, or IDs to preserve when delete_all is true.
        """
        return _submit_command(
            DeletePlots(delete_all=delete_all, graph_ids=graph_ids or []),
            "Delete command sent to UI successfully.",
        )

    return mcp


def create_desktop_mcp_server(
    context: AppContext, *, host: str = "127.0.0.1", port: int = 8765
) -> FastMCP:
    """Create the full server used by trusted local external clients.

    The existing AI Chat intentionally keeps the compact five-tool profile for
    small-model reliability. Both profiles are built from this same module and
    operate through the same application context.
    """
    return create_mcp_server(
        context,
        include_application_tools=True,
        host=host,
        port=port,
    )
