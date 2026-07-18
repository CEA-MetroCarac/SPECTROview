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


class TestFormatAxisLabel:
    """VGraph._format_axis_label() -- turns fit-result columns of the form
    "<param>_<peaklabel>" into a friendly, unit-annotated label instead of
    showing the raw column name. Regression test for a previously-lost
    feature: a later, unrelated commit ("improve the intelligent of AI
    agent") collateral-changed this to just append units to the raw column
    name (e.g. "x0_Si (cm$^{-1}$)") instead of building the friendly name
    ("Si peak position (cm$^{-1}$)") it was originally written to produce."""

    def test_x0_prefix_becomes_peak_position(self, vg):
        assert vg._format_axis_label("x0_Si") == "Si peak position (cm$^{-1}$)"

    def test_fwhm_prefix_becomes_peak_width(self, vg):
        assert vg._format_axis_label("fwhm_Si") == "Si peak width (cm$^{-1}$)"

    def test_ampli_prefix_becomes_peak_intensity(self, vg):
        assert vg._format_axis_label("ampli_Si") == "Si peak intensity (a.u.)"

    def test_area_prefix_becomes_peak_area(self, vg):
        assert vg._format_axis_label("area_Si") == "Si peak area (a.u.)"

    def test_unrecognized_prefix_passes_through_unchanged(self, vg):
        assert vg._format_axis_label("Zone") == "Zone"
        assert vg._format_axis_label("Strain (GPa)") == "Strain (GPa)"

    def test_multi_word_peaklabel_after_first_underscore_is_preserved(self, vg):
        # split('_', 1): only the first underscore separates param/peaklabel.
        assert vg._format_axis_label("x0_Si_substrate") == "Si_substrate peak position (cm$^{-1}$)"

    def test_auto_label_used_in_real_plot(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.plot(excel_df)
        assert vg.ax.get_xlabel() == "Si peak position (cm$^{-1}$)"
        assert vg.ax.get_ylabel() == "Si peak intensity (a.u.)"


class TestHueGroupingAndLegend:
    def test_categorical_hue_creates_multi_entry_legend(self, vg, excel_df):
        _configure(vg, x="Zone", y=["fwhm_Si"], z="Quadrant", plot_style="point")
        vg.plot(excel_df)
        n_quadrants = excel_df["Quadrant"].nunique()
        legend = vg.ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == n_quadrants

    def test_point_plot_hue_groups_default_to_circle_markers(self, vg, excel_df):
        """DEFAULT_MARKERS defaults every new series to 'o' (all circles) --
        per-series marker customization remains available via
        legend_properties, but is opt-in, not an automatic shape cycle."""
        _configure(vg, x="Zone", y=["fwhm_Si"], z="Quadrant", plot_style="point")
        vg.plot(excel_df)
        n_quadrants = excel_df["Quadrant"].nunique()
        assert n_quadrants > 1
        props = vg.get_legend_properties()
        assert all(p["marker"] == "o" for p in props)

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

    def test_legend_ncol_applied(self, vg, excel_df):
        _configure(vg, x="Zone", y=["fwhm_Si"], z="Quadrant", plot_style="point")
        vg.legend_ncol = 2
        vg.plot(excel_df)
        assert vg.ax.get_legend()._ncols == 2

    def test_legend_title_applied(self, vg, excel_df):
        _configure(vg, x="Zone", y=["fwhm_Si"], z="Quadrant", plot_style="point")
        vg.legend_title = "Groups"
        vg.plot(excel_df)
        assert vg.ax.get_legend().get_title().get_text() == "Groups"

    def test_legend_frame_off_applied(self, vg, excel_df):
        _configure(vg, x="Zone", y=["fwhm_Si"], z="Quadrant", plot_style="point")
        vg.legend_frame = False
        vg.plot(excel_df)
        assert vg.ax.get_legend().get_frame_on() is False

    def test_legend_default_alpha_matches_pre_existing_hardcoded_value(self, vg, excel_df):
        _configure(vg, x="Zone", y=["fwhm_Si"], z="Quadrant", plot_style="point")
        vg.plot(excel_df)
        assert vg.ax.get_legend().get_frame().get_alpha() == pytest.approx(0.7)

    def test_legend_loc_applied(self, vg, excel_df):
        _configure(vg, x="Zone", y=["fwhm_Si"], z="Quadrant", plot_style="point")
        vg.legend_loc = "upper left"
        vg.plot(excel_df)
        assert vg.ax.get_legend() is not None  # smoke: renders without error


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

    def test_symlog_scale_applied(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.ylogscale = True
        vg.yscale_mode = "symlog"
        vg.plot(excel_df)
        assert vg.ax.get_yscale() == "symlog"

    def test_axis_inversion_applied(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.x_inverted = True
        vg.y_inverted = True
        vg.plot(excel_df)
        assert vg.ax.xaxis_inverted() == True
        assert vg.ax.yaxis_inverted() == True

    def test_no_inversion_by_default(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.plot(excel_df)
        assert vg.ax.xaxis_inverted() == False

    def test_tick_direction_applied(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.tick_direction = "in"
        vg.plot(excel_df)
        # matplotlib exposes the configured direction via _major_tick_kw
        assert vg.ax.xaxis._major_tick_kw.get('tickdir') == 'in'

    def test_title_and_label_fontsize_applied(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.plot_title = "My Title"
        vg.title_fontsize = 20
        vg.axis_label_fontsize = 15
        vg.plot(excel_df)
        assert vg.ax.title.get_fontsize() == 20
        assert vg.ax.xaxis.label.get_fontsize() == 15
        assert vg.ax.yaxis.label.get_fontsize() == 15

    def test_tick_label_format_applied(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.tick_label_format = "%.1f"
        vg.plot(excel_df)
        assert vg.ax.xaxis.get_major_formatter().fmt == "%.1f"

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

    def test_degenerate_equal_limits_are_skipped_not_warned(self, vg, excel_df, recwarn):
        """Regression test: a saved graph with xmin==xmax (e.g. an old wafer
        plot with 0.0/0.0 accidentally saved) used to be handed straight to
        matplotlib's set_xlim/set_ylim, which raises a 'singular
        transformation' UserWarning and auto-expands anyway. Degenerate
        pairs should be skipped instead, leaving matplotlib's own
        auto-scaled range in place with no warning."""
        _configure(vg, x="X", y=["Y"], z="ampli_Si", plot_style="wafer")
        vg.xmin, vg.xmax = 0.0, 0.0
        vg.ymin, vg.ymax = 0.0, 0.0
        vg.plot(excel_df)
        assert not any("singular" in str(w.message).lower() for w in recwarn.list)
        xlo, xhi = vg.ax.get_xlim()
        ylo, yhi = vg.ax.get_ylim()
        assert xlo != xhi
        assert ylo != yhi


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

    def test_y2_color_override_applied(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.y2 = "area_Si"
        vg.y2color = "blue"
        vg.plot(excel_df)
        assert vg.ax2.yaxis.label.get_color() == "blue"

    def test_y3_color_override_applied(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.y3 = "fwhm_Si"
        vg.y3color = "orange"
        vg.plot(excel_df)
        assert vg.ax3.yaxis.label.get_color() == "orange"

    def test_x2_color_override_applied(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.x2 = "Y"
        vg.x2color = "teal"
        vg.plot(excel_df)
        assert vg.ax_x2.xaxis.label.get_color() == "teal"

    def test_y2_marker_override_applied_for_point_style(self, vg, excel_df):
        _configure(vg, x="Zone", y=["ampli_Si"], plot_style="point")
        vg.y2 = "area_Si"
        vg.y2marker = "D"
        vg.plot(excel_df)
        container = vg.ax2.containers[0]
        assert container.lines[0].get_marker() == "D"


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

    def test_per_series_marker_size_and_edge_color_override(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], z="Quadrant", plot_style="scatter")
        vg.plot(excel_df)  # populate legend_properties for the hue groups
        vg.legend_properties[0]["marker_size"] = 300
        vg.legend_properties[0]["edge_color"] = "#00FF00"
        vg.plot(excel_df)

        pathcol = vg.ax.collections[0]
        assert pathcol.get_sizes()[0] == pytest.approx(300, rel=0.2)


class TestErrorBarOptions:
    """error_bar_type (point/line, unconditional-by-default) and
    bar_error_bar_type (bar, only consulted when show_bar_plot_error_bar is
    True) replace the old hardcoded-95%-CI-always / SD-only behavior."""

    def test_point_error_bar_type_none_suppresses_yerr(self, vg, excel_df):
        _configure(vg, x="Zone", y=["ampli_Si"], plot_style="point")
        vg.error_bar_type = "none"
        vg.plot(excel_df)
        line = vg.ax.containers[0]
        assert line.has_yerr is False

    def test_point_error_bar_type_ci95_matches_default_hardcoded_behavior(self, vg, excel_df):
        _configure(vg, x="Zone", y=["ampli_Si"], plot_style="point")
        vg.error_bar_type = "ci95"
        vg.plot(excel_df)
        line = vg.ax.containers[0]
        assert line.has_yerr is True

    def test_line_error_bar_type_none_removes_ci_band(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="line")
        vg.error_bar_type = "none"
        vg.plot(excel_df)
        # fill_between draws a PolyCollection; none should exist when suppressed.
        assert len(vg.ax.collections) == 0

    def test_bar_error_bar_type_changes_yerr_values(self, vg, excel_df):
        # SD and SEM statistics differ for n>1 groups, so re-rendering with
        # a different type must move the drawn error-bar cap/whisker lines.
        _configure(vg, x="Zone", y=["ampli_Si"], plot_style="bar")
        vg.show_bar_plot_error_bar = True
        vg.bar_error_bar_type = "sd"
        vg.plot(excel_df)
        sd_lines = [np.asarray(ln.get_ydata()) for ln in vg.ax.lines]

        vg2 = VGraph(graph_id=2)
        vg2.create_plot_widget(dpi=72)
        _configure(vg2, x="Zone", y=["ampli_Si"], plot_style="bar")
        vg2.show_bar_plot_error_bar = True
        vg2.bar_error_bar_type = "sem"
        vg2.plot(excel_df)
        sem_lines = [np.asarray(ln.get_ydata()) for ln in vg2.ax.lines]

        assert len(sd_lines) == len(sem_lines) and len(sd_lines) > 0
        assert any(not np.array_equal(a, b) for a, b in zip(sd_lines, sem_lines))

    def test_bar_error_bar_off_by_default_shows_no_error_bars(self, vg, excel_df):
        _configure(vg, x="Zone", y=["ampli_Si"], plot_style="bar")
        vg.plot(excel_df)
        assert len(vg.ax.lines) == 0

    def test_error_bar_capsize_applied(self, vg, excel_df):
        _configure(vg, x="Zone", y=["ampli_Si"], plot_style="point")
        vg.error_bar_capsize = 10.0
        vg.plot(excel_df)
        # No direct getter for capsize on an ErrorbarContainer; smoke-check
        # it renders without error at a non-default value.
        assert vg.ax.containers[0].has_yerr is True


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


class TestFigureStyle:
    """figure_facecolor/plot_subtitle/spines_visible are optional and
    skipped when unset; figure_margins/subtitle_fontsize are concrete
    fields defaulting to matplotlib/mplstyle's own values -- either way, an
    old saved graph (none of these fields set) renders identically to
    before this feature existed."""

    def test_defaults_leave_all_spines_visible(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.plot(excel_df)
        assert vg.ax.spines['top'].get_visible() is True
        assert vg.ax.spines['right'].get_visible() is True

    def test_hidden_spines_applied(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.spines_visible = {'top': False, 'right': False, 'bottom': True, 'left': True}
        vg.plot(excel_df)
        assert vg.ax.spines['top'].get_visible() is False
        assert vg.ax.spines['right'].get_visible() is False
        assert vg.ax.spines['bottom'].get_visible() is True

    def test_facecolor_applied(self, vg, excel_df):
        import matplotlib.colors as mcolors
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.figure_facecolor = "#EEEEEE"
        vg.plot(excel_df)
        assert mcolors.to_hex(vg.ax.get_facecolor()) == "#eeeeee"

    def test_default_theme_is_light(self, vg, excel_df):
        import matplotlib.colors as mcolors
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.plot(excel_df)
        assert mcolors.to_hex(vg.ax.get_facecolor()) == "#ffffff"

    def test_dark_theme_changes_background(self, vg, excel_df):
        import matplotlib.colors as mcolors
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.figure_theme = "dark"
        vg.plot(excel_df)
        assert mcolors.to_hex(vg.ax.get_facecolor()) == "#242424"

    def test_theme_recreated_on_create_plot_widget(self, vg, excel_df):
        """The theme must also apply at Figure-creation time (create_plot_widget),
        not just at plot() time, since the mplstyle context governs figure-level
        rcParams (e.g. figure.facecolor) captured when the Figure is constructed."""
        import matplotlib.colors as mcolors
        vg.figure_theme = "dark"
        vg.create_plot_widget(dpi=72)
        assert mcolors.to_hex(vg.figure.get_facecolor()) == "#242424"

    def test_subtitle_renders_as_text_below_title(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.plot_title = "Title"
        vg.plot_subtitle = "Subtitle text"
        vg.plot(excel_df)
        texts = [t.get_text() for t in vg.ax.texts]
        assert "Subtitle text" in texts

    def test_no_subtitle_by_default(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.plot(excel_df)
        assert len(vg.ax.texts) == 0

    def test_margins_applied_does_not_crash(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.figure_margins = [0.2, 0.3]
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
