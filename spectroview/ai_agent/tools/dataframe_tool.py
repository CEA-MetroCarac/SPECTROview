"""tools/dataframe_tool.py

Reusable DataFrame utility functions for the SPECTROview AI Agent.

These helpers encapsulate the data operations that the AI Agent performs
on the user's pandas DataFrames: building schema summaries, executing
safe query filters, computing statistics, and building the dynamic
context sections injected into the system prompt.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema / context builders
# ---------------------------------------------------------------------------

def build_schema_info(
    dataframes: dict[str, pd.DataFrame],
    active_df_name: str,
) -> tuple[str, str]:
    """Build the DataFrames schema section and active-df info string.

    These strings are injected into the ``{dataframes_section}`` and
    ``{active_df_info}`` placeholders in ``prompts/system.md``.

    Parameters
    ----------
    dataframes:
        Mapping of DataFrame name → DataFrame.
    active_df_name:
        The currently active/selected DataFrame name.

    Returns
    -------
    tuple[str, str]
        ``(dataframes_section, active_df_info)`` as formatted strings.
    """
    if not dataframes:
        return "No DataFrames are currently loaded.", ""

    lines: list[str] = []
    for name, df in dataframes.items():
        col_info = ", ".join(
            f"{col} ({df[col].dtype})" for col in df.columns
        )
        # Include a compact sample of unique values for small-cardinality columns
        sample_parts: list[str] = []
        for col in df.columns:
            n_unique = df[col].nunique()
            if n_unique <= 10:
                vals = df[col].dropna().unique().tolist()
                sample_parts.append(f"  - {col}: {vals[:10]}")
        sample_block = "\n".join(sample_parts) if sample_parts else ""

        lines.append(
            f"DataFrame '{name}': {len(df)} rows × {len(df.columns)} columns\n"
            f"  Columns: {col_info}"
            + (f"\n  Sample categories:\n{sample_block}" if sample_block else "")
        )

    dataframes_section = "\n\n".join(lines)

    # Active-df info
    if active_df_name and active_df_name in dataframes:
        active_df = dataframes[active_df_name]
        active_df_info = (
            f"\nCurrently active DataFrame: '{active_df_name}' "
            f"({len(active_df)} rows × {len(active_df.columns)} columns)"
        )
    else:
        active_df_info = ""

    return dataframes_section, active_df_info


def build_graphs_info(graphs: list) -> str:
    """Build a summary of currently open graphs.

    This string is injected into the ``{graphs_info}`` placeholder in
    ``prompts/system.md``.

    Parameters
    ----------
    graphs:
        List of ``MGraph`` model objects (from the Graphs workspace ViewModel).

    Returns
    -------
    str
        Formatted multi-line string describing each open graph.
    """
    if not graphs:
        return "\nNo graphs are currently open."

    lines: list[str] = ["\nCurrently open graphs:"]
    for g in graphs:
        gid = getattr(g, "graph_id", "?")
        style = getattr(g, "plot_style", "?")
        x = getattr(g, "x", None) or "?"
        y = getattr(g, "y", None)
        y_str = ", ".join(y) if isinstance(y, list) else (y or "?")
        z = getattr(g, "z", None)
        df_name = getattr(g, "df_name", None) or "?"
        filters = getattr(g, "filters", []) or []

        line = f"  - Graph {gid}: {style} | x={x}, y={y_str}"
        if z:
            line += f", z={z}"
        line += f" | DataFrame='{df_name}'"
        if filters:
            filter_str = "; ".join(str(f) for f in filters)
            line += f" | filters=[{filter_str}]"
        lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Safe data operations
# ---------------------------------------------------------------------------

def safe_query(df: pd.DataFrame, expression: str) -> tuple[pd.DataFrame, str]:
    """Apply a pandas `.query()` expression safely.

    Parameters
    ----------
    df:
        The DataFrame to filter.
    expression:
        A valid pandas `.query()` string.

    Returns
    -------
    tuple[pd.DataFrame, str]
        ``(filtered_df, error_message)``.
        On success, ``error_message`` is an empty string.
        On failure, ``filtered_df`` is the original (unfiltered) DataFrame
        and ``error_message`` contains the exception description.
    """
    if not expression or not expression.strip():
        return df, "Empty query expression."

    try:
        result = df.query(expression)
        return result, ""
    except Exception as exc:
        logger.warning("Query failed: %s | expression=%r", exc, expression)
        return df, str(exc)


def safe_describe(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, str]:
    """Compute descriptive statistics for the specified columns.

    Parameters
    ----------
    df:
        The DataFrame to describe.
    columns:
        List of column names to include.  Columns not present in ``df``
        are silently skipped; a warning is logged.

    Returns
    -------
    tuple[pd.DataFrame, str]
        ``(description_df, error_message)``.
        On success, ``error_message`` is an empty string.
    """
    valid_cols = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]

    if missing:
        logger.warning(
            "Columns not found in DataFrame and skipped: %s", missing
        )

    if not valid_cols:
        msg = f"None of the requested columns exist: {columns}"
        return pd.DataFrame(), msg

    try:
        result = df[valid_cols].describe()
        return result, ""
    except Exception as exc:
        logger.warning("describe() failed: %s", exc)
        return pd.DataFrame(), str(exc)


def format_dataframe_as_markdown(
    df: pd.DataFrame,
    max_rows: int = 20,
    float_fmt: str = ".4f",
) -> str:
    """Format a DataFrame as a Markdown table string.

    Parameters
    ----------
    df:
        DataFrame to format.
    max_rows:
        Maximum number of rows to include. Adds a note if truncated.
    float_fmt:
        Format specifier for floating-point values.

    Returns
    -------
    str
        Markdown table string.
    """
    if df.empty:
        return "*(empty result)*"

    truncated = len(df) > max_rows
    display_df = df.head(max_rows)

    # Build header
    cols = list(display_df.columns)
    index_name = display_df.index.name or "index"
    header = f"| {index_name} | " + " | ".join(str(c) for c in cols) + " |"
    separator = "| --- | " + " | ".join("---" for _ in cols) + " |"

    rows: list[str] = [header, separator]
    for idx, row in display_df.iterrows():
        values = []
        for v in row:
            if isinstance(v, float):
                values.append(f"{v:{float_fmt}}")
            else:
                values.append(str(v))
        rows.append(f"| {idx} | " + " | ".join(values) + " |")

    table = "\n".join(rows)
    if truncated:
        table += f"\n\n*…showing {max_rows} of {len(df)} rows.*"
    return table
