"""spectroview/ai_agent/utils/df_summary.py

Token-efficient DataFrame column summary for the AI Agent system prompt.

Below ``max_ungrouped_columns``, output is identical to a plain per-column
listing. Above it, columns sharing a name prefix (the text before the first
``_``) are grouped into one compact line — e.g. a dozen ``center_*`` columns
become a single line naming all of them, with one representative sample.
Every column name always appears somewhere in the output; only per-column
sample-value detail is compacted. Grouping never hides a column name, since
the AI Agent must never reference a column it was never shown.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, List

import pandas as pd


def compact_dataframe_schema(df: pd.DataFrame) -> str:
    """Return every column name with its dtype, and nothing else.

    This is the block pushed into *every* system prompt. Column names must
    always be present — a model that has never seen a name will invent one —
    but sample values and row previews are bulky and only occasionally needed,
    so they live behind the ``spectroview://dataframes/detail`` resource
    instead. See :func:`summarize_dataframe_columns` for that fuller view.
    """
    return ", ".join(f"{col} ({df[col].dtype})" for col in df.columns)


def _column_line(df: pd.DataFrame, col: Any, samples_per_column: int) -> str:
    dtype = str(df[col].dtype)
    sample_vals = df[col].dropna().unique()[:samples_per_column].tolist()
    return f"    - {col!r} ({dtype}): sample values {sample_vals}"


def summarize_dataframe_columns(
    df: pd.DataFrame,
    *,
    max_ungrouped_columns: int = 30,
    min_group_size: int = 3,
    samples_per_column: int = 3,
) -> str:
    """Return a Markdown-friendly column listing for *df*.

    Parameters
    ----------
    df:
        The DataFrame to summarize.
    max_ungrouped_columns:
        Below this column count, every column gets its own detailed line
        (identical to the original unconditional per-column loop). At or
        above it, columns are grouped by shared name prefix.
    min_group_size:
        Minimum number of same-prefix columns required to collapse them
        into one grouped line; smaller groups stay ungrouped.
    samples_per_column:
        Number of sample values shown per (ungrouped) column.
    """
    columns = list(df.columns)
    if len(columns) < max_ungrouped_columns:
        return "\n".join(_column_line(df, col, samples_per_column) for col in columns)

    groups: "OrderedDict[str, List[Any]]" = OrderedDict()
    for col in columns:
        prefix = str(col).split("_", 1)[0] if "_" in str(col) else str(col)
        groups.setdefault(prefix, []).append(col)

    lines: List[str] = []
    for prefix, cols in groups.items():
        if len(cols) >= min_group_size:
            sample_col = cols[0]
            sample_vals = df[sample_col].dropna().unique()[:samples_per_column].tolist()
            names = ", ".join(repr(c) for c in cols)
            lines.append(
                f"    - {len(cols)} columns starting with {prefix!r}: {names} "
                f"(dtype {df[sample_col].dtype}; e.g. {sample_col!r} sample values {sample_vals})"
            )
        else:
            for col in cols:
                lines.append(_column_line(df, col, samples_per_column))

    return "\n".join(lines)
