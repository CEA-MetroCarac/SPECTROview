"""Tests for spectroview/api/graphs.py -- the public scripted/notebook
plotting API.

Previously this module wrapped seaborn (`import seaborn as sns`), but
`seaborn` was removed from pyproject.toml's dependencies at some point
without this module being updated -- so `import spectroview.api.graphs`
raised `ModuleNotFoundError` unconditionally, and was untested. It's been
rewritten to delegate to the same PlotRenderer the native Graphs workspace
uses (v_plot_renderer.py), which has no Qt dependency, so these tests don't
need a QApplication -- only real data (dataset_Excel.xlsx, matching the
Graphs workspace's own test fixtures) and matplotlib.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from spectroview.api import graphs


@pytest.fixture(scope="module")
def excel_df(dataframe_excel_file):
    if not dataframe_excel_file.exists():
        pytest.skip("dataset_Excel.xlsx not present")
    return pd.read_excel(dataframe_excel_file, sheet_name="sheet1")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


class TestNoSeabornDependency:
    def test_module_does_not_import_seaborn(self):
        """`seaborn` was removed from pyproject.toml's dependencies at some
        point without this module being updated, so `import
        spectroview.api.graphs` used to raise ModuleNotFoundError
        unconditionally regardless of whether seaborn happened to be
        installed elsewhere on a given machine. Parse the module's actual
        import statements (not whether seaborn is importable in this
        environment, which is a fact about the machine, not the code; and
        not a raw substring search, which would false-positive on this
        module's own docstring explaining the history)."""
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(graphs))
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module.split('.')[0])

        assert 'seaborn' not in imported_names
        assert not hasattr(graphs, 'sns')


class TestPlotFunctionsRenderRealData:
    def test_plot_scatter_creates_own_figure_and_draws_points(self, excel_df):
        ax = graphs.plot_scatter(excel_df, x="x0_Si", y="ampli_Si", title="Scatter")
        assert ax is not None
        assert len(ax.collections) > 0
        assert ax.get_title() == "Scatter"

    def test_plot_scatter_with_hue_draws_a_legend(self, excel_df):
        ax = graphs.plot_scatter(excel_df, x="x0_Si", y="ampli_Si", hue="Quadrant")
        assert ax.get_legend() is not None
        assert len(ax.get_legend().get_texts()) == excel_df["Quadrant"].nunique()

    def test_plot_scatter_without_hue_draws_no_legend(self, excel_df):
        ax = graphs.plot_scatter(excel_df, x="x0_Si", y="ampli_Si")
        assert ax.get_legend() is None

    def test_plot_point_draws_error_bars(self, excel_df):
        ax = graphs.plot_point(excel_df, x="Zone", y="ampli_Si", hue="Quadrant")
        assert len(ax.get_lines()) > 0 or len(ax.collections) > 0
        assert ax.get_legend() is not None

    def test_plot_point_join_flag_connects_points_with_a_line(self, excel_df):
        ax_joined = graphs.plot_point(excel_df, x="Zone", y="ampli_Si", join=True)
        ax_unjoined = graphs.plot_point(excel_df, x="Zone", y="ampli_Si", join=False)
        # errorbar with linestyle='-' produces Line2D objects with real segments;
        # 'none' produces a degenerate (invisible) connecting line.
        joined_has_visible_line = any(
            line.get_linestyle() not in ('None', 'none') for line in ax_joined.get_lines()
        )
        unjoined_has_visible_line = any(
            line.get_linestyle() not in ('None', 'none') for line in ax_unjoined.get_lines()
        )
        assert joined_has_visible_line
        assert not unjoined_has_visible_line

    def test_plot_box_draws_boxes(self, excel_df):
        ax = graphs.plot_box(excel_df, x="Zone", y="fwhm_Si")
        assert len(ax.patches) > 0 or len(ax.lines) > 0

    def test_plot_box_with_hue_draws_a_legend(self, excel_df):
        ax = graphs.plot_box(excel_df, x="Zone", y="fwhm_Si", hue="Quadrant")
        assert ax.get_legend() is not None

    def test_plot_trendline_draws_fit_line_and_scatter(self, excel_df):
        ax = graphs.plot_trendline(excel_df, x="x0_Si", y="area_Si", order=1)
        assert len(ax.get_lines()) > 0
        assert len(ax.collections) > 0


class TestPlotTrendlineHueUsesGivenAxes:
    """Regression test for a real bug in the old seaborn-based implementation:
    plot_trendline(..., hue=..., ax=ax) called sns.lmplot, which always
    creates its own new figure and silently ignores the passed-in `ax` --
    contradicting the function's own documented contract. The PlotRenderer-
    based rewrite must actually draw onto the given ax regardless of hue."""

    def test_trendline_with_hue_plots_onto_the_given_ax(self, excel_df):
        fig, ax = plt.subplots()
        n_figures_before = len(plt.get_fignums())

        returned_ax = graphs.plot_trendline(excel_df, x="x0_Si", y="area_Si", hue="Quadrant", ax=ax)

        assert returned_ax is ax
        assert len(plt.get_fignums()) == n_figures_before  # no extra figure was created
        assert len(ax.get_lines()) > 0
        assert len(ax.collections) > 0
        assert ax.get_legend() is not None
        assert len(ax.get_legend().get_texts()) == excel_df["Quadrant"].dropna().nunique()


class TestAxParameterIsHonored:
    """Every function must draw onto a caller-supplied ax rather than
    silently creating a new figure, and must create its own figure only
    when ax=None (the documented contract). Trendline needs a numeric x;
    box/point use a categorical x so they group into a handful of boxes/
    points rather than ~588 near-unique ones."""

    _CASES = [
        (graphs.plot_scatter, dict(x="x0_Si", y="ampli_Si")),
        (graphs.plot_point, dict(x="Zone", y="ampli_Si")),
        (graphs.plot_box, dict(x="Zone", y="ampli_Si")),
        (graphs.plot_trendline, dict(x="x0_Si", y="area_Si")),
    ]

    @pytest.mark.parametrize("fn,kwargs", _CASES)
    def test_given_ax_is_used_directly(self, excel_df, fn, kwargs):
        fig, ax = plt.subplots()
        n_figures_before = len(plt.get_fignums())
        returned_ax = fn(excel_df, ax=ax, **kwargs)
        assert returned_ax is ax
        assert len(plt.get_fignums()) == n_figures_before

    @pytest.mark.parametrize("fn,kwargs", _CASES)
    def test_no_ax_creates_a_new_figure(self, excel_df, fn, kwargs):
        n_figures_before = len(plt.get_fignums())
        ax = fn(excel_df, **kwargs)
        assert len(plt.get_fignums()) == n_figures_before + 1
        assert ax.figure is not None
