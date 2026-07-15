"""
Tests for spectroview/ai_agent/utils/safe_eval.py

Covers the exact repro from the investigation: `query_dataframe`'s docstring
and every prompt example teach bare-column syntax (e.g. "FWHM_Si > 5"), but
the original implementation used raw eval() with only df/pd/np bound, so
that syntax raised NameError. evaluate_pandas_expression() fixes this while
keeping the same sandboxed eval() fallback for aggregation expressions.
"""
import pandas as pd
import pytest

from spectroview.ai_agent.utils.safe_eval import evaluate_pandas_expression, format_query_result


@pytest.fixture
def df():
    return pd.DataFrame({
        "Slot": [1, 2, 3, 5, 6, 7, 8, 10],
        "Zone": ["Edge", "Center", "Edge", "Center", "Edge", "Center", "Edge", "Center"],
        "FWHM_Si": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8],
    })


class TestBareColumnSyntax:
    def test_bare_column_numeric_filter_succeeds(self, df):
        """This exact expression raised NameError before the fix."""
        result, error = evaluate_pandas_expression(df, "FWHM_Si > 5")
        assert error is None
        assert isinstance(result, pd.DataFrame)
        assert list(result["FWHM_Si"]) == [5.5, 6.6, 7.7, 8.8]

    def test_bare_column_string_filter_succeeds(self, df):
        result, error = evaluate_pandas_expression(df, "Zone == 'Edge'")
        assert error is None
        assert (result["Zone"] == "Edge").all()

    def test_compound_condition_succeeds(self, df):
        result, error = evaluate_pandas_expression(df, "FWHM_Si > 2 and Zone == 'Center'")
        assert error is None
        # Center rows: FWHM_Si = [2.2, 4.4, 6.6, 8.8] — all four are > 2.
        assert len(result) == 4


class TestAggregationFallback:
    def test_groupby_idxmax_expression_succeeds(self, df):
        """.query() cannot express this — must fall back to eval()."""
        result, error = evaluate_pandas_expression(
            df, "df.groupby('Zone')['FWHM_Si'].mean().idxmax()"
        )
        assert error is None
        assert result == "Center"

    def test_tuple_expression_succeeds(self, df):
        result, error = evaluate_pandas_expression(df, "(df['FWHM_Si'].max(), df['FWHM_Si'].min())")
        assert error is None
        assert result == (8.8, 1.1)


class TestErrorHandling:
    def test_unquoted_string_value_returns_clear_error(self, df):
        result, error = evaluate_pandas_expression(df, "Zone == Edge")
        assert result is None
        assert error is not None

    def test_nonexistent_column_returns_error(self, df):
        result, error = evaluate_pandas_expression(df, "Nonexistent > 5")
        assert result is None
        assert error is not None

    def test_empty_expression_returns_error(self, df):
        result, error = evaluate_pandas_expression(df, "")
        assert result is None
        assert error is not None


class TestSandboxSafety:
    def test_builtins_are_blocked(self, df):
        result, error = evaluate_pandas_expression(df, "__import__('os').system('echo pwned')")
        assert result is None
        assert error is not None

    def test_open_is_blocked(self, df):
        result, error = evaluate_pandas_expression(df, "open('/etc/passwd').read()")
        assert result is None
        assert error is not None


class TestFormatQueryResult:
    def test_dataframe_result_shows_row_count(self, df):
        text = format_query_result(df)
        assert f"{len(df)} rows" in text

    def test_series_result_labeled(self, df):
        text = format_query_result(df["FWHM_Si"])
        assert "Series" in text

    def test_scalar_result(self):
        text = format_query_result(6)
        assert "6" in text
