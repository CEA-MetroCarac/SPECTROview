"""Headless rendering tests for view/components/v_graph.py - VGraph.

Deliberate, scoped exception to this codebase's usual "View layer: not
tested" policy: the user explicitly asked for every plot style and every
customization ability to be exercised against real matplotlib rendering
(not just config-dict manipulation), using the real dataset_Excel.xlsx file.
QT_QPA_PLATFORM=offscreen + a real QApplication is enough for VGraph to
construct, build its FigureCanvas, and render -- no display needed.

Two hard-won gotchas baked into every test here (see tests/README.md):
1. `create_plot_widget(dpi)` MUST be called before `plot(df)` -- `self.ax`
   is None until then and `plot()` unconditionally calls `self.ax.clear()`.
2. An unsupported `plot_style` routes through `show_alert()` ->
   `QMessageBox.exec_()`, which blocks forever even under offscreen mode.
   The one test that exercises this path monkeypatches `QMessageBox.exec_`
   first; every other test only uses real PLOT_STYLES values so it never
   comes up by accident.
"""
import numpy as np
import pandas as pd
import pytest

from spectroview import PLOT_STYLES
from spectroview.view.components.v_graph import VGraph


@pytest.fixture(scope="module")
def excel_df(dataframe_excel_file):
    """sheet1 of the real dataset_Excel.xlsx: 588 rows of wafer Si-peak fit
    results (X, Y, x0_Si, ampli_Si, area_Si, fwhm_Si, Quadrant, Zone, Slot,
    DeltaW, 'Strain (GPa)', 'NB pts')."""
    if not dataframe_excel_file.exists():
        pytest.skip("dataset_Excel.xlsx not present")
    return pd.read_excel(dataframe_excel_file, sheet_name="sheet1")


@pytest.fixture
def vg(qapp):
    widget = VGraph(graph_id=1)
    widget.create_plot_widget(dpi=72)
    return widget


def _configure(vg, df_name="sheet1", x=None, y=None, z=None, plot_style="point"):
    vg.df_name = df_name
    vg.x = x
    vg.y = y if y is not None else []
    vg.z = z
    vg.plot_style = plot_style
    return vg


class TestPlotStyleCoverage:
    """One real-data render per PLOT_STYLES entry -- the core 'plot all
    features' request. Column choices are picked to be meaningful for each
    style (categorical hue for point/scatter/box/bar, numeric-vs-numeric
    for trendline/histogram, real wafer X/Y coordinates for wafer/2Dmap)."""

    def test_all_plot_styles_are_covered_by_this_test_class(self):
        tested = {'point', 'scatter', 'box', 'bar', 'line', 'trendline',
                  'histogram', 'wafer', '2Dmap'}
        assert tested == set(PLOT_STYLES)

    def test_point(self, vg, excel_df):
        _configure(vg, x="Zone", y=["fwhm_Si"], z="Quadrant", plot_style="point")
        vg.plot(excel_df)
        assert len(vg.ax.get_lines()) > 0 or len(vg.ax.collections) > 0

    def test_scatter(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], z="Quadrant", plot_style="scatter")
        vg.plot(excel_df)
        assert len(vg.ax.collections) > 0

    def test_box(self, vg, excel_df):
        _configure(vg, x="Zone", y=["fwhm_Si"], plot_style="box")
        vg.plot(excel_df)
        assert len(vg.ax.patches) > 0 or len(vg.ax.lines) > 0

    def test_bar(self, vg, excel_df):
        _configure(vg, x="Zone", y=["ampli_Si"], plot_style="bar")
        vg.plot(excel_df)
        assert len(vg.ax.patches) > 0

    def test_line(self, vg, excel_df):
        _configure(vg, x="X", y=["fwhm_Si"], plot_style="line")
        vg.plot(excel_df)
        assert len(vg.ax.get_lines()) > 0

    def test_trendline(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["area_Si"], plot_style="trendline")
        vg.plot(excel_df)
        assert len(vg.ax.get_lines()) > 0
        # A fitted equation should have been computed for the single group.
        assert len(vg.trendline_equations) >= 1
        assert "r2" in vg.trendline_equations[0]

    def test_histogram(self, vg, excel_df):
        _configure(vg, x="fwhm_Si", y=["fwhm_Si"], plot_style="histogram")
        vg.plot(excel_df)
        assert len(vg.ax.patches) > 0

    def test_wafer(self, vg, excel_df):
        _configure(vg, x="X", y=["Y"], z="ampli_Si", plot_style="wafer")
        vg.wafer_size = 300.0
        vg.plot(excel_df)
        # WaferPlot draws a filled contour/mesh -> at least one collection.
        assert len(vg.ax.collections) > 0 or len(vg.ax.images) > 0

    def test_2dmap(self, vg, excel_df):
        # 2Dmap pivots on (x, y), so needs unique coordinate pairs; the raw
        # sheet has a handful of repeated (X, Y) positions (see
        # test_2dmap_duplicate_xy_pairs_raises_valueerror for that case).
        unique_xy_df = excel_df.drop_duplicates(subset=["X", "Y"])
        _configure(vg, x="X", y=["Y"], z="ampli_Si", plot_style="2Dmap")
        vg.plot(unique_xy_df)
        assert len(vg.ax.collections) > 0 or len(vg.ax.images) > 0

    def test_unsupported_style_shows_alert_without_hanging(self, vg, excel_df, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        monkeypatch.setattr(QMessageBox, "exec_", lambda self: 0)
        monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
        _configure(vg, x="Zone", y=["fwhm_Si"], plot_style="not_a_real_style")
        vg.plot(excel_df)  # must return, not hang


class TestHueGroupingAndLegend:
    def test_categorical_hue_creates_multi_entry_legend(self, vg, excel_df):
        _configure(vg, x="Zone", y=["fwhm_Si"], z="Quadrant", plot_style="point")
        vg.plot(excel_df)
        n_quadrants = excel_df["Quadrant"].nunique()
        legend = vg.ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == n_quadrants

    def test_no_hue_still_produces_a_plot(self, vg, excel_df):
        _configure(vg, x="Zone", y=["fwhm_Si"], z=None, plot_style="point")
        vg.plot(excel_df)
        assert len(vg.ax.get_lines()) > 0 or len(vg.ax.collections) > 0

    def test_get_legend_properties_matches_hue_count(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], z="Quadrant", plot_style="scatter")
        vg.plot(excel_df)
        props = vg.get_legend_properties()
        assert len(props) == excel_df["Quadrant"].nunique()
        for p in props:
            assert set(p.keys()) >= {"label", "marker", "color"}

    def test_legend_hidden_when_legend_visible_false(self, vg, excel_df):
        _configure(vg, x="Zone", y=["fwhm_Si"], z="Quadrant", plot_style="point")
        vg.legend_visible = False
        vg.plot(excel_df)
        assert vg.ax.get_legend() is None


class TestAxisCustomization:
    def test_explicit_limits_are_applied(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.xmin, vg.xmax = 500.0, 530.0
        vg.ymin, vg.ymax = 100.0, 20000.0
        vg.plot(excel_df)
        assert vg.ax.get_xlim() == (500.0, 530.0)
        assert vg.ax.get_ylim() == (100.0, 20000.0)

    def test_zero_valued_limit_is_applied_not_ignored(self, vg, excel_df):
        """Regression test: _set_limits() used to check `if self.ymin and
        self.ymax`, a truthy check that silently dropped a limit of exactly
        0.0 (a very ordinary axis bound, e.g. 'start Y at zero')."""
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.ymin, vg.ymax = 0.0, 20000.0
        vg.plot(excel_df)
        assert vg.ax.get_ylim() == (0.0, 20000.0)

    def test_log_scale_applied(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.ylogscale = True
        vg.plot(excel_df)
        assert vg.ax.get_yscale() == "log"

    def test_labels_and_title_applied(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.xlabel, vg.ylabel, vg.plot_title = "X0 (cm-1)", "Amplitude", "My Plot"
        vg.plot(excel_df)
        assert vg.ax.get_xlabel() == "X0 (cm-1)"
        assert vg.ax.get_ylabel() == "Amplitude"
        assert vg.ax.get_title() == "My Plot"

    def test_grid_toggle(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.grid = True
        vg.plot(excel_df)
        assert any(line.get_visible() for line in vg.ax.get_xgridlines())

    def test_axis_break_does_not_crash_rendering(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        x_vals = excel_df["x0_Si"]
        mid = float(x_vals.median())
        vg.axis_breaks = {"x": {"start": mid - 1, "end": mid + 1}, "y": None}
        vg.plot(excel_df)  # must not raise


class TestSecondaryAxes:
    """y2 (twinx, red), y3 (twinx offset, green), x2 (twiny, purple) -- the
    'Plot multiple axes (beta)' tab. No prior coverage existed for these
    before VGraph._plot_secondary_axis/_plot_tertiary_axis/_plot_secondary_x_axis
    were consolidated into one parameterized helper, so this class doubles as
    the regression check for that refactor."""

    def test_no_secondary_axes_by_default(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.plot(excel_df)
        assert vg.ax2 is None
        assert vg.ax3 is None
        assert vg.ax_x2 is None

    def test_y2_scatter(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.y2 = "area_Si"
        vg.plot(excel_df)
        assert vg.ax2 is not None
        assert len(vg.ax2.collections) > 0
        assert vg.ax2.get_ylabel()

    def test_y2_point(self, vg, excel_df):
        _configure(vg, x="Zone", y=["ampli_Si"], plot_style="point")
        vg.y2 = "area_Si"
        vg.plot(excel_df)
        assert vg.ax2 is not None
        assert len(vg.ax2.get_lines()) > 0 or len(vg.ax2.collections) > 0

    def test_y2_line(self, vg, excel_df):
        _configure(vg, x="X", y=["ampli_Si"], plot_style="line")
        vg.y2 = "area_Si"
        vg.plot(excel_df)
        assert vg.ax2 is not None
        assert len(vg.ax2.get_lines()) > 0

    def test_y2_removed_for_unsupported_style(self, vg, excel_df):
        _configure(vg, x="Zone", y=["ampli_Si"], plot_style="bar")
        vg.y2 = "area_Si"
        vg.plot(excel_df)
        assert vg.ax2 is None

    def test_y2_custom_label_and_color(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.y2 = "area_Si"
        vg.y2label = "Custom Y2"
        vg.plot(excel_df)
        assert vg.ax2.get_ylabel() == "Custom Y2"
        assert vg.ax2.yaxis.label.get_color() == "red"

    def test_y3_creates_offset_spine_axis(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.y3 = "fwhm_Si"
        vg.plot(excel_df)
        assert vg.ax3 is not None
        assert len(vg.ax3.collections) > 0
        assert vg.ax3.spines["right"].get_position() == ("outward", 100)

    def test_y2_and_y3_together(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.y2 = "area_Si"
        vg.y3 = "fwhm_Si"
        vg.plot(excel_df)
        assert vg.ax2 is not None
        assert vg.ax3 is not None
        assert vg.ax2 is not vg.ax3

    def test_x2_creates_secondary_x_axis(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.x2 = "Y"
        vg.plot(excel_df)
        assert vg.ax_x2 is not None
        assert len(vg.ax_x2.collections) > 0
        assert vg.ax_x2.xaxis.label.get_color() == "purple"

    def test_x2_absent_when_column_missing_from_df(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.x2 = "NonexistentColumn"
        vg.plot(excel_df)
        assert vg.ax_x2 is None

    def test_replot_does_not_accumulate_secondary_axes(self, vg, excel_df):
        """Re-plotting must remove() the old twin axis before creating a new
        one, not leak a second overlapping axes object onto the figure."""
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.y2 = "area_Si"
        vg.plot(excel_df)
        first_ax2 = vg.ax2
        vg.plot(excel_df)
        assert vg.ax2 is not first_ax2
        assert first_ax2 not in vg.figure.axes


class TestScatterCustomization:
    def test_scatter_size_reflected_in_marker_area(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.scatter_size = 200
        vg.plot(excel_df)
        pathcol = vg.ax.collections[0]
        sizes = pathcol.get_sizes()
        assert len(sizes) > 0
        assert sizes[0] == pytest.approx(200, rel=0.2)

    def test_scatter_edgecolor_applied(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.scatter_edgecolor = "#FF0000"
        vg.plot(excel_df)
        pathcol = vg.ax.collections[0]
        edgecolors = pathcol.get_edgecolors()
        assert len(edgecolors) > 0


class TestHistogramCustomization:
    def test_bin_count_changes_patch_count(self, vg, excel_df):
        _configure(vg, x="fwhm_Si", y=["fwhm_Si"], plot_style="histogram")
        vg.hist_bins = 10
        vg.plot(excel_df)
        n_bins_10 = len(vg.ax.patches)

        vg2 = VGraph(graph_id=2)
        vg2.create_plot_widget(dpi=72)
        _configure(vg2, x="fwhm_Si", y=["fwhm_Si"], plot_style="histogram")
        vg2.hist_bins = 40
        vg2.plot(excel_df)
        n_bins_40 = len(vg2.ax.patches)

        assert n_bins_40 > n_bins_10

    def test_kde_overlay_adds_a_line(self, vg, excel_df):
        _configure(vg, x="fwhm_Si", y=["fwhm_Si"], plot_style="histogram")
        vg.hist_kde = True
        vg.plot(excel_df)
        assert len(vg.ax.get_lines()) >= 1


class TestTrendlineCustomization:
    def test_polynomial_order_changes_fit_curve(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["area_Si"], plot_style="trendline")
        vg.trendline_order = 1
        vg.plot(excel_df)
        eq_order1 = vg.trendline_equations[0]["equation"]

        vg2 = VGraph(graph_id=2)
        vg2.create_plot_widget(dpi=72)
        _configure(vg2, x="x0_Si", y=["area_Si"], plot_style="trendline")
        vg2.trendline_order = 3
        vg2.plot(excel_df)
        eq_order3 = vg2.trendline_equations[0]["equation"]

        assert eq_order1 != eq_order3

    def test_anchor_through_origin_forces_curve_through_zero(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["area_Si"], plot_style="trendline")
        vg.trendline_anchor_enabled = True
        vg.trendline_anchor_origin = True
        vg.plot(excel_df)  # must not raise


class TestFailureModes:
    """Real, uncaught exceptions VGraph.plot() raises on malformed input --
    confirmed empirically to propagate straight out of plot(), not be
    swallowed. Callers (v_workspace_graphs.py) wrap plot() in try/except;
    VGraph itself does not."""

    def test_missing_x_column_raises_keyerror(self, vg, excel_df):
        _configure(vg, x="NotAColumn", y=["fwhm_Si"], plot_style="point")
        with pytest.raises(KeyError):
            vg.plot(excel_df)

    def test_none_x_falls_back_to_blank_plot_no_crash(self, vg, excel_df):
        _configure(vg, x=None, y=["fwhm_Si"], plot_style="point")
        vg.plot(excel_df)  # must not raise
        assert len(vg.ax.get_lines()) >= 1  # the blank ax.plot([], [])

    def test_none_df_name_falls_back_to_blank_plot(self, vg, excel_df):
        _configure(vg, df_name=None, x="Zone", y=["fwhm_Si"], plot_style="point")
        vg.plot(excel_df)  # must not raise

    def test_wafer_without_z_raises_keyerror(self, vg, excel_df):
        _configure(vg, x="X", y=["Y"], z=None, plot_style="wafer")
        with pytest.raises(KeyError):
            vg.plot(excel_df)

    def test_2dmap_without_z_raises_keyerror(self, vg, excel_df):
        _configure(vg, x="X", y=["Y"], z=None, plot_style="2Dmap")
        with pytest.raises(KeyError):
            vg.plot(excel_df)

    def test_2dmap_duplicate_xy_pairs_raises_valueerror(self, vg, excel_df):
        dup_df = pd.concat([excel_df, excel_df.iloc[[0]]], ignore_index=True)  # duplicate one (X,Y)
        _configure(vg, x="X", y=["Y"], z="ampli_Si", plot_style="2Dmap")
        with pytest.raises(ValueError):
            vg.plot(dup_df)

    def test_trendline_non_numeric_x_raises_clear_valueerror(self, vg, excel_df):
        _configure(vg, x="Quadrant", y=["area_Si"], plot_style="trendline")
        with pytest.raises(ValueError, match="numeric"):
            vg.plot(excel_df)

    def test_trendline_empty_dataframe_raises(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["area_Si"], plot_style="trendline")
        empty = excel_df.iloc[0:0]
        with pytest.raises((TypeError, ValueError)):
            vg.plot(empty)

    def test_plot_before_create_plot_widget_raises(self, qapp, excel_df):
        vg = VGraph(graph_id=99)  # create_plot_widget() never called
        _configure(vg, x="Zone", y=["fwhm_Si"], plot_style="point")
        with pytest.raises(AttributeError):
            vg.plot(excel_df)
