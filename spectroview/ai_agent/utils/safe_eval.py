"""spectroview/ai_agent/utils/safe_eval.py

Shared, sandboxed pandas expression evaluator for the AI Agent.

Every prompt/example that teaches the AI Agent how to write a filter or
query expression (e.g. ``"FWHM_Si > 5"``, ``"Zone == 'center'"``) teaches
bare-column syntax compatible with ``DataFrame.query()``. This module tries
that first (safe, restricted grammar, native bare-column support) and only
falls back to a namespace-restricted ``eval()`` for expressions ``.query()``
cannot express (aggregations, ``.groupby()``, tuples, etc.) — the same
fallback the AI Agent has always used, just with individual columns also
bound by name so the documented bare-column syntax actually works.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd


def evaluate_pandas_expression(df: pd.DataFrame, expr: str) -> Tuple[Optional[Any], Optional[str]]:
    """Evaluate *expr* against *df*, trying the safest option first.

    Parameters
    ----------
    df:
        The target DataFrame.
    expr:
        A pandas expression string. Simple filter expressions
        (``"FWHM_Si > 5"``, ``"Zone == 'center'"``) are evaluated via
        :meth:`pandas.DataFrame.query`. Expressions ``.query()`` cannot
        express (e.g. ``"df.groupby('Slot')['x'].mean().idxmax()"``) fall
        back to a namespace-restricted ``eval()``.

    Returns
    -------
    tuple
        ``(result, None)`` on success, or ``(None, error_message)`` on
        failure. Never raises.
    """
    if not isinstance(expr, str) or not expr.strip():
        return None, "empty or non-string expression"

    try:
        return df.query(expr), None
    except Exception:
        pass  # fall back to eval() below — .query() can't express everything

    safe_locals = {"df": df, "pd": pd, "np": np}
    for col in df.columns:
        col_str = str(col)
        if col_str.isidentifier() and col_str not in safe_locals:
            safe_locals[col_str] = df[col]

    try:
        return eval(expr, {"__builtins__": {}}, safe_locals), None  # noqa: S307
    except Exception as exc:
        return None, str(exc)


def format_query_result(result: Any) -> str:
    """Render an :func:`evaluate_pandas_expression` result as a human-readable string."""
    if isinstance(result, pd.DataFrame):
        return f"Query returned a DataFrame with {len(result)} rows. First few rows:\n{result.head().to_string()}"
    if isinstance(result, pd.Series):
        return f"Query returned a Series. First few elements:\n{result.head().to_string()}"
    return f"Query result: {result}"
