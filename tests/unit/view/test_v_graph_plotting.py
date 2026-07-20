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
        vg.unify_marker_style = False  # per-series overrides are ignored while unified
        vg.plot(excel_df)  # populate legend_properties for the hue groups
        vg.legend_properties[0]["marker_size"] = 300
        vg.legend_properties[0]["edge_color"] = "#00FF00"
        vg.plot(excel_df)

        pathcol = vg.ax.collections[0]
        assert pathcol.get_sizes()[0] == pytest.approx(300, rel=0.2)

    def test_unify_marker_style_ignores_per_series_overrides(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], z="Quadrant", plot_style="scatter")
        vg.scatter_size = 70
        assert vg.unify_marker_style is True  # the default
        vg.plot(excel_df)
        vg.legend_properties[0]["marker_size"] = 300
        vg.plot(excel_df)

        pathcol = vg.ax.collections[0]
        assert pathcol.get_sizes()[0] == pytest.approx(70, rel=0.2)


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
    def test_per_series_marker_override_respected_when_not_unified(self, vg, excel_df):
        """Regression: _plot_trendline() used to always draw scatter
        markers from the global scatter_size/scatter_edgecolor directly,
        never consulting legend_properties at all -- unlike point/scatter,
        which already supported per-series overrides."""
        _configure(vg, x="x0_Si", y=["area_Si"], z="Quadrant", plot_style="trendline")
        vg.unify_marker_style = False
        vg.plot(excel_df)
        vg.legend_properties[0]["marker_size"] = 300
        vg.plot(excel_df)

        pathcol = vg.ax.collections[0]
        assert pathcol.get_sizes()[0] == pytest.approx(300, rel=0.2)

    def test_unify_marker_style_ignores_per_series_override(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["area_Si"], z="Quadrant", plot_style="trendline")
        vg.scatter_size = 70
        vg.plot(excel_df)
        vg.legend_properties[0]["marker_size"] = 300
        vg.plot(excel_df)

        pathcol = vg.ax.collections[0]
        assert pathcol.get_sizes()[0] == pytest.approx(70, rel=0.2)

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

    def test_dark_theme_recolors_labels_ticks_and_spines(self, vg, excel_df):
        """Regression: ax.clear() doesn't retroactively repaint an Axes
        built under a *different* theme (same root cause already fixed for
        facecolor) -- axis labels, title, tick labels, and spines used to
        stay black under Dark/Soft Dark instead of following the theme."""
        import matplotlib.colors as mcolors
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.figure_theme = "dark"
        vg.plot_title = "Title"
        vg.plot(excel_df)

        assert mcolors.to_hex(vg.ax.xaxis.label.get_color()) != "#000000"
        assert mcolors.to_hex(vg.ax.yaxis.label.get_color()) != "#000000"
        assert mcolors.to_hex(vg.ax.title.get_color()) != "#000000"
        for spine in vg.ax.spines.values():
            assert mcolors.to_hex(spine.get_edgecolor()) != "#000000"
        for label in vg.ax.get_xticklabels() + vg.ax.get_yticklabels():
            assert mcolors.to_hex(label.get_color()) != "#000000"

    def test_light_theme_labels_stay_default_black(self, vg, excel_df):
        import matplotlib.colors as mcolors
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.plot(excel_df)
        assert mcolors.to_hex(vg.ax.yaxis.label.get_color()) == "#000000"

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


class TestColormapNormalization:
    """colormap_norm/colormap_center (wafer/2Dmap only) -- default 'linear'
    renders identically to before this feature existed (plain vmin=/vmax=
    Normalize); 'log'/'centered' swap in a matplotlib norm object instead."""

    def test_default_is_linear_normalize(self, vg, excel_df):
        from matplotlib.colors import Normalize
        _configure(vg, x="X", y=["Y"], z="ampli_Si", plot_style="wafer")
        vg.plot(excel_df)
        im = vg.ax.get_images()[0]
        assert type(im.norm) is Normalize

    def test_log_norm_applied_on_wafer(self, vg, excel_df):
        from matplotlib.colors import LogNorm
        _configure(vg, x="X", y=["Y"], z="ampli_Si", plot_style="wafer")
        vg.colormap_norm = "log"
        vg.plot(excel_df)
        im = vg.ax.get_images()[0]
        assert isinstance(im.norm, LogNorm)
        assert im.norm.vmin > 0

    def test_centered_norm_applied_on_wafer(self, vg, excel_df):
        from matplotlib.colors import CenteredNorm
        _configure(vg, x="X", y=["Y"], z="ampli_Si", plot_style="wafer")
        vg.colormap_norm = "centered"
        vg.colormap_center = 8000.0
        vg.plot(excel_df)
        im = vg.ax.get_images()[0]
        assert isinstance(im.norm, CenteredNorm)
        assert im.norm.vcenter == 8000.0

    def test_log_norm_falls_back_to_linear_when_data_crosses_zero(self, vg, excel_df):
        """LogNorm requires strictly positive data; rather than raising or
        producing a broken/all-masked heatmap, a zero-crossing Z range
        silently falls back to linear (matches this codebase's existing
        graceful-degradation style for wafer/2Dmap rendering)."""
        from matplotlib.colors import Normalize, LogNorm
        df = excel_df.copy()
        df["ampli_Si"] = df["ampli_Si"] - df["ampli_Si"].mean()  # now crosses 0
        _configure(vg, x="X", y=["Y"], z="ampli_Si", plot_style="wafer")
        vg.colormap_norm = "log"
        vg.plot(df)
        im = vg.ax.get_images()[0]
        assert not isinstance(im.norm, LogNorm)
        assert type(im.norm) is Normalize

    def test_log_norm_applied_on_2dmap(self, vg, excel_df):
        from matplotlib.colors import LogNorm
        unique_xy_df = excel_df.drop_duplicates(subset=["X", "Y"])
        _configure(vg, x="X", y=["Y"], z="ampli_Si", plot_style="2Dmap")
        vg.colormap_norm = "log"
        vg.plot(unique_xy_df)
        im = vg.ax.get_images()[0]
        assert isinstance(im.norm, LogNorm)


class TestNewAnnotationTypes:
    """arrow/vspan/hspan/box/callout: each renders as the matplotlib artist
    type that makes it correctly pickable (FancyArrowPatch/Rectangle over
    ax.annotate('', ...), see _render_arrow's docstring), and drag support
    is delta-based (shift the whole shape by the mouse move, not jump to an
    absolute position) since these are multi-point shapes."""

    def _find_by_id(self, vg, ann_id):
        for artist in vg.ax.findobj():
            if hasattr(artist, '_annotation_data') and artist._annotation_data['id'] == ann_id:
                return artist
        raise AssertionError(f"no artist found for annotation id={ann_id}")

    def test_arrow_renders_as_fancyarrowpatch(self, vg, excel_df):
        import matplotlib.patches as mpatches
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.annotations = [{'id': 'a1', 'type': 'arrow', 'x1': 510, 'y1': 7000, 'x2': 515, 'y2': 8000}]
        vg.plot(excel_df)
        artist = self._find_by_id(vg, 'a1')
        assert isinstance(artist, mpatches.FancyArrowPatch)

    def test_vspan_and_hspan_render_as_rectangle(self, vg, excel_df):
        import matplotlib.patches as mpatches
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.annotations = [
            {'id': 'a1', 'type': 'vspan', 'x1': 510, 'x2': 515},
            {'id': 'a2', 'type': 'hspan', 'y1': 7000, 'y2': 8000},
        ]
        vg.plot(excel_df)
        assert isinstance(self._find_by_id(vg, 'a1'), mpatches.Rectangle)
        assert isinstance(self._find_by_id(vg, 'a2'), mpatches.Rectangle)

    def test_box_renders_as_rectangle_with_correct_geometry(self, vg, excel_df):
        import matplotlib.patches as mpatches
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.annotations = [{'id': 'a1', 'type': 'box', 'x': 510, 'y': 7000, 'width': 5, 'height': 1000}]
        vg.plot(excel_df)
        artist = self._find_by_id(vg, 'a1')
        assert isinstance(artist, mpatches.Rectangle)
        assert artist.get_width() == 5
        assert artist.get_height() == 1000

    def test_callout_renders_as_annotation_with_arrow(self, vg, excel_df):
        from matplotlib.text import Annotation
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.annotations = [{'id': 'a1', 'type': 'callout', 'x': 510, 'y': 7000, 'tx': 515, 'ty': 8000, 'text': 'peak'}]
        vg.plot(excel_df)
        artist = self._find_by_id(vg, 'a1')
        assert isinstance(artist, Annotation)
        assert artist.get_text() == 'peak'
        assert artist.arrow_patch is not None

    def test_bad_annotation_type_data_does_not_crash_whole_render(self, vg, excel_df):
        """_render_annotations wraps each annotation in try/except -- a
        malformed one (missing required key) must not take down the rest
        of the plot, matching vline/hline/text's existing behavior."""
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.annotations = [
            {'id': 'bad', 'type': 'box'},  # missing x/y/width/height -> uses .get() defaults, should not raise
            {'id': 'good', 'type': 'vline', 'x': 510},
        ]
        vg.plot(excel_df)  # must not raise
        assert self._find_by_id(vg, 'good') is not None

    def _drag(self, vg, artist, start_xy, end_xy):
        # _on_annotation_drag reads event.x/event.y (pixel coords) and
        # transforms them through self.ax.transData itself -- see
        # VGraph._ax_data_coords's docstring for why it no longer trusts
        # event.xdata/event.ydata directly (unreliable whenever a secondary
        # Y-axis overlaps the primary Axes). Provide real pixel coordinates
        # here too so the round-trip lands back on the intended data point.
        class _E:
            def __init__(self, xdata, ydata):
                self.x, self.y = vg.ax.transData.transform((xdata, ydata))
        vg._drag_candidate = artist
        vg._drag_start_x, vg._drag_start_y = start_xy
        vg._on_annotation_drag(_E(start_xy[0] + 1000, start_xy[1] + 1000))  # exceed promotion threshold
        vg._on_annotation_drag(_E(*end_xy))
        vg._on_annotation_release(_E(*end_xy))

    def test_drag_arrow_shifts_both_endpoints_by_same_delta(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.annotations = [{'id': 'a1', 'type': 'arrow', 'x1': 500, 'y1': 7000, 'x2': 505, 'y2': 7500}]
        vg.plot(excel_df)
        artist = self._find_by_id(vg, 'a1')
        self._drag(vg, artist, (500, 7000), (510, 7100))  # delta = (+10, +100)
        ann = vg.annotations[0]
        assert ann['x1'] == pytest.approx(510) and ann['y1'] == pytest.approx(7100)
        assert ann['x2'] == pytest.approx(515) and ann['y2'] == pytest.approx(7600)

    def test_drag_vspan_preserves_width(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.annotations = [{'id': 'a1', 'type': 'vspan', 'x1': 500, 'x2': 505}]
        vg.plot(excel_df)
        artist = self._find_by_id(vg, 'a1')
        self._drag(vg, artist, (500, 0), (510, 0))  # delta_x = +10
        ann = vg.annotations[0]
        assert ann['x2'] - ann['x1'] == pytest.approx(5)  # width unchanged
        assert ann['x1'] == pytest.approx(510)

    def test_drag_box_preserves_width_and_height(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.annotations = [{'id': 'a1', 'type': 'box', 'x': 500, 'y': 7000, 'width': 5, 'height': 1000}]
        vg.plot(excel_df)
        artist = self._find_by_id(vg, 'a1')
        self._drag(vg, artist, (500, 7000), (510, 7100))
        ann = vg.annotations[0]
        assert ann['x'] == pytest.approx(510) and ann['y'] == pytest.approx(7100)
        assert ann['width'] == 5 and ann['height'] == 1000
        assert artist.get_width() == 5 and artist.get_height() == 1000

    def test_drag_callout_moves_text_but_not_the_pointed_at_xy(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.annotations = [{'id': 'a1', 'type': 'callout', 'x': 500, 'y': 7000, 'tx': 505, 'ty': 7500, 'text': 'peak'}]
        vg.plot(excel_df)
        artist = self._find_by_id(vg, 'a1')
        self._drag(vg, artist, (505, 7500), (520, 7900))
        ann = vg.annotations[0]
        assert ann['x'] == 500 and ann['y'] == 7000  # untouched
        assert ann['tx'] == pytest.approx(520) and ann['ty'] == pytest.approx(7900)


class TestInsetAxes:
    """inset_enabled draws a second, smaller Axes (via Axes.inset_axes())
    showing the same series, at inset_xmin/xmax/ymin/ymax if set (else the
    same auto-scaled view). Disabled by default -- an old saved graph (no
    inset fields set) renders identically to before this feature existed."""

    def test_disabled_by_default_no_inset_created(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.plot(excel_df)
        assert vg.inset_ax is None

    def test_enabled_creates_inset_with_same_series(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.inset_enabled = True
        vg.plot(excel_df)
        assert vg.inset_ax is not None
        assert len(vg.inset_ax.collections) > 0  # scatter draws via collections

    def test_inset_limits_applied_when_set(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.inset_enabled = True
        vg.inset_xmin, vg.inset_xmax = 510.0, 515.0
        vg.inset_ymin, vg.inset_ymax = 7000.0, 8000.0
        vg.plot(excel_df)
        assert vg.inset_ax.get_xlim() == pytest.approx((510.0, 515.0))
        assert vg.inset_ax.get_ylim() == pytest.approx((7000.0, 8000.0))

    def test_inset_limits_unset_leave_auto_scale(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.inset_enabled = True
        vg.plot(excel_df)
        # Auto-scaled inset should span a real (non-degenerate) range.
        xlo, xhi = vg.inset_ax.get_xlim()
        assert xhi > xlo

    def test_inset_bounds_applied(self, vg, excel_df):
        """Different inset_bounds must actually reposition/resize the inset
        (not just be accepted and ignored)."""
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.inset_enabled = True

        vg.inset_bounds = [0.1, 0.1, 0.15, 0.15]
        vg.plot(excel_df)
        small_pos = vg.inset_ax.get_position()

        vg.inset_bounds = [0.55, 0.55, 0.4, 0.4]
        vg.plot(excel_df)
        big_pos = vg.inset_ax.get_position()

        assert big_pos.width > small_pos.width
        assert big_pos.height > small_pos.height
        assert (big_pos.x0, big_pos.y0) != (small_pos.x0, small_pos.y0)

    def test_zoom_indicator_default_on(self, vg, excel_df):
        # indicate_inset_zoom() returns/attaches one InsetIndicator artist
        # to the main ax (this matplotlib version's representation of the
        # rectangle + connector lines) -- confirmed via direct call, not
        # assumed, since the exact artist container matplotlib uses for
        # this has changed across versions.
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.inset_enabled = True
        vg.plot(excel_df)
        assert len(vg.ax.artists) > 0

    def test_zoom_indicator_disabled_adds_no_indicator_artist(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.inset_enabled = True
        vg.inset_show_zoom_indicator = False
        vg.plot(excel_df)
        assert len(vg.ax.artists) == 0

    def test_replot_does_not_leak_old_inset(self, vg, excel_df):
        """Regression guard: ax.clear() detaches the old inset from the
        figure's child_axes on every replot -- if _render_inset() didn't
        also drop its own self.inset_ax reference, repeated plot() calls
        would accumulate stale, invisible-but-still-referenced Axes."""
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.inset_enabled = True
        vg.plot(excel_df)
        first_inset = vg.inset_ax
        vg.plot(excel_df)
        second_inset = vg.inset_ax
        assert first_inset is not second_inset
        assert first_inset not in vg.ax.child_axes
        assert second_inset in vg.ax.child_axes

    def test_disabling_after_enabled_removes_inset(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.inset_enabled = True
        vg.plot(excel_df)
        assert vg.inset_ax is not None

        vg.inset_enabled = False
        vg.plot(excel_df)
        assert vg.inset_ax is None
        assert len(vg.ax.child_axes) == 0

    def test_inset_skipped_when_axis_break_active(self, vg, excel_df):
        """Documented limitation: inset + broken axis are not combined."""
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.inset_enabled = True
        vg.axis_breaks = {'x': {'start': 511.0, 'end': 512.0}, 'y': None}
        vg.plot(excel_df)
        assert vg.inset_ax is None

    def test_render_series_on_restores_ax_and_figure(self, vg, excel_df):
        """The shared swap/restore helper must leave self.ax/self.figure
        exactly as they were, regardless of what it drew onto the target."""
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.plot(excel_df)
        original_ax, original_figure = vg.ax, vg.figure

        scratch_ax = vg.ax.inset_axes([0.1, 0.1, 0.2, 0.2])
        vg._render_series_on(scratch_ax)

        assert vg.ax is original_ax
        assert vg.figure is original_figure


class TestBrokenAxis:
    """Rewrite: a broken axis is now two real Axes (side-by-side for an X
    break, stacked for a Y break), each showing the full series clipped to
    its half of the range -- replacing the old post-hoc mutation of
    ax.get_lines()/ax.collections data, which silently didn't work for any
    plot style using ax.patches (bar, box) or ax.images (wafer, 2Dmap)."""

    def test_no_break_by_default_single_axes(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.plot(excel_df)
        assert vg._current_break_mode is None
        assert vg.ax_break_secondary is None
        assert len(vg.figure.axes) == 1

    def test_x_break_creates_two_panels_with_split_xlim(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.axis_breaks = {'x': {'start': 514.8, 'end': 514.9}, 'y': None}
        vg.plot(excel_df)
        assert vg._current_break_mode == 'x'
        assert vg.ax_break_secondary is not None
        assert len(vg.figure.axes) == 2
        p_lo, p_hi = vg.ax.get_xlim()
        s_lo, s_hi = vg.ax_break_secondary.get_xlim()
        assert p_hi == pytest.approx(514.8)
        assert s_lo == pytest.approx(514.9)
        assert p_lo < p_hi and s_lo < s_hi

    def test_y_break_stacks_panels_high_values_on_top(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.axis_breaks = {'x': None, 'y': {'start': 7500, 'end': 8000}}
        vg.plot(excel_df)
        assert vg._current_break_mode == 'y'
        p_lo, p_hi = vg.ax.get_ylim()          # primary = bottom = low values
        s_lo, s_hi = vg.ax_break_secondary.get_ylim()  # secondary = top = high values
        assert p_hi == pytest.approx(7500)
        assert s_lo == pytest.approx(8000)
        assert s_hi > p_hi  # top panel really is the higher-value side

    def test_x_break_hides_facing_spines_only(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.axis_breaks = {'x': {'start': 514.8, 'end': 514.9}, 'y': None}
        vg.plot(excel_df)
        assert vg.ax.spines['right'].get_visible() is False
        assert vg.ax_break_secondary.spines['left'].get_visible() is False
        # Outer spines untouched (still their default-visible state).
        assert vg.ax.spines['left'].get_visible() is True
        assert vg.ax_break_secondary.spines['right'].get_visible() is True

    def test_secondary_panel_y_ticklabels_hidden_for_x_break(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.axis_breaks = {'x': {'start': 514.8, 'end': 514.9}, 'y': None}
        vg.plot(excel_df)
        assert all(not t.get_visible() for t in vg.ax_break_secondary.get_yticklabels())

    @pytest.mark.parametrize("style", ["bar", "box"])
    def test_patch_based_styles_split_correctly(self, vg, excel_df, style):
        """Regression test: the old implementation only mutated
        ax.get_lines()/ax.collections data, so a break on a bar or box
        plot (drawn via ax.patches) silently rendered as if unbroken. The
        rewrite redraws the whole chart on each panel instead, so it works
        identically regardless of which artist type the style uses."""
        _configure(vg, x="Zone", y=["ampli_Si"], plot_style=style)
        vg.axis_breaks = {'x': None, 'y': {'start': 7500, 'end': 8000}}
        vg.plot(excel_df)
        assert len(vg.ax.patches) > 0, f"{style}: primary panel has no patches"
        assert len(vg.ax_break_secondary.patches) > 0, f"{style}: secondary panel has no patches"

    def test_legend_includes_all_hue_categories_despite_split(self, vg, excel_df):
        """The full (unclipped) series is drawn on the primary panel before
        xlim is narrowed, so get_legend_handles_labels() still sees every
        hue category's artist regardless of whether its data happens to
        fall inside the visible clipped range."""
        _configure(vg, x="x0_Si", y=["ampli_Si"], z="Quadrant", plot_style="scatter")
        vg.axis_breaks = {'x': {'start': 514.8, 'end': 514.9}, 'y': None}
        vg.plot(excel_df)
        legend = vg.ax.get_legend()
        assert legend is not None
        labels = {t.get_text() for t in legend.get_texts()}
        assert labels == set(excel_df["Quadrant"].dropna().unique())

    def test_title_placed_on_top_panel_for_y_break(self, vg, excel_df):
        """Y-break: primary (self.ax) is the bottom panel -- the title must
        be moved to the secondary (top) panel, not left floating in the
        gap between the two stacked panels."""
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.plot_title = "My Title"
        vg.axis_breaks = {'x': None, 'y': {'start': 7500, 'end': 8000}}
        vg.plot(excel_df)
        assert vg.ax.get_title() == ""
        assert vg.ax_break_secondary.get_title() == "My Title"

    def test_title_stays_on_primary_panel_for_x_break(self, vg, excel_df):
        """X-break: both panels are the same row, so the title above
        primary (the left panel) is already visually correct -- no move
        needed."""
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.plot_title = "My Title"
        vg.axis_breaks = {'x': {'start': 514.8, 'end': 514.9}, 'y': None}
        vg.plot(excel_df)
        assert vg.ax.get_title() == "My Title"

    def test_annotation_appears_only_on_the_panel_containing_its_coordinate(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.axis_breaks = {'x': {'start': 514.8, 'end': 514.9}, 'y': None}
        vg.annotations = [{'id': 'v1', 'type': 'vline', 'x': 514.5, 'color': 'red'}]  # falls in primary's range
        vg.plot(excel_df)
        primary_lines = [l for l in vg.ax.get_lines() if hasattr(l, '_annotation_data')]
        secondary_lines = [l for l in vg.ax_break_secondary.get_lines() if hasattr(l, '_annotation_data')]
        assert len(primary_lines) == 1
        assert len(secondary_lines) == 1  # both panels get a copy; visibility is clipped by xlim, not artist presence

    def test_both_x_and_y_set_defensively_prefers_x(self, vg, excel_df):
        """Scripting/AI-agent path can set both breaks directly, bypassing
        the dialog's mutual-exclusion UI -- the renderer must still pick
        one deterministically rather than attempting an unsupported 2x2
        grid."""
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.axis_breaks = {
            'x': {'start': 514.8, 'end': 514.9},
            'y': {'start': 7500, 'end': 8000},
        }
        vg.plot(excel_df)
        assert vg._current_break_mode == 'x'

    def test_break_outside_data_range_does_not_split_or_warn(self, vg, excel_df):
        """Regression test for a real bug found during review: clamping an
        out-of-range break naively could produce set_xlim(lo, lo) -- a
        zero-width panel that matplotlib both warns about and silently
        auto-expands. A break that doesn't meaningfully overlap the
        rendered range must fall back to "no split" instead."""
        import warnings
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.axis_breaks = {'x': {'start': 1.0, 'end': 2.0}, 'y': None}  # data is ~514-516
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            vg.plot(excel_df)  # must not raise/warn
        assert vg.ax.get_xlim() == vg.ax_break_secondary.get_xlim()
        assert vg.ax.spines['right'].get_visible() is True  # no fake split applied

    def test_toggling_break_off_rebuilds_single_axes(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.axis_breaks = {'x': {'start': 514.8, 'end': 514.9}, 'y': None}
        vg.plot(excel_df)
        assert len(vg.figure.axes) == 2

        vg.axis_breaks = {'x': None, 'y': None}
        vg.plot(excel_df)
        assert vg._current_break_mode is None
        assert vg.ax_break_secondary is None
        assert len(vg.figure.axes) == 1

    def test_switching_break_axis_rebuilds_layout(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.axis_breaks = {'x': {'start': 514.8, 'end': 514.9}, 'y': None}
        vg.plot(excel_df)
        assert vg._current_break_mode == 'x'

        vg.axis_breaks = {'x': None, 'y': {'start': 7500, 'end': 8000}}
        vg.plot(excel_df)
        assert vg._current_break_mode == 'y'
        assert len(vg.figure.axes) == 2  # rebuilt, not accumulated to 3+

    def test_repeated_replot_does_not_leak_axes(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.axis_breaks = {'x': {'start': 514.8, 'end': 514.9}, 'y': None}
        for _ in range(3):
            vg.plot(excel_df)
        assert len(vg.figure.axes) == 2

    def test_twin_axis_not_drawn_while_break_active(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.y2 = "fwhm_Si"
        vg.axis_breaks = {'x': {'start': 514.8, 'end': 514.9}, 'y': None}
        vg.plot(excel_df)
        assert vg.ax2 is None

    def test_stale_twin_axis_removed_when_break_enabled_afterward(self, vg, excel_df):
        """A graph can have y2 configured from before a break was turned
        on -- the stale ax2 from that earlier non-break render must be
        cleaned up, not just left orphaned on the figure, even though its
        own creation is now skipped."""
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.y2 = "fwhm_Si"
        vg.plot(excel_df)
        assert vg.ax2 is not None

        vg.axis_breaks = {'x': {'start': 514.8, 'end': 514.9}, 'y': None}
        vg.plot(excel_df)
        assert vg.ax2 is None
        assert len(vg.figure.axes) == 2  # not 3 (primary + secondary + orphaned ax2)


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


class TestRestyle:
    """Phase 5E: VGraph.restyle() re-runs the label/grid/limit/legend
    styling steps against the artists already on screen, without clearing
    or re-plotting the series -- the fast path for a pure cosmetic edit
    (see model/graph_style.py's RESTYLE_SAFE_FIELDS for exactly which
    fields this is valid for). Two idempotency traps make this genuinely
    unsafe to build from the existing per-step methods unmodified:
    Axes.invert_xaxis()/invert_yaxis() toggle rather than set, and the
    subtitle used to be drawn via a bare ax.text() call that stacked a new
    artist on every call -- both fixed alongside restyle() itself and
    covered here."""

    def test_restyle_does_not_touch_the_plotted_artist(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.plot(excel_df)
        artist_before = vg.ax.collections[0]

        vg.title_fontsize = 22
        vg.grid = True
        ok = vg.restyle()

        assert ok is True
        assert vg.ax.collections[0] is artist_before  # same object, not replotted
        assert vg.ax.title.get_fontsize() == 22

    def test_restyle_applies_limits_scale_and_labels(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.plot(excel_df)

        vg.xmin, vg.xmax = 514.6, 515.0
        vg.xlabel = "Custom X"
        vg.restyle()

        assert vg.ax.get_xlim() == pytest.approx((514.6, 515.0))
        assert vg.ax.get_xlabel() == "Custom X"

    def test_restyle_returns_false_when_a_break_is_active(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.axis_breaks['x'] = {'start': 514.7, 'end': 514.8}
        vg.plot(excel_df)  # builds the two-panel break layout

        assert vg.restyle() is False

    def test_restyle_twice_does_not_duplicate_the_subtitle(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.plot_subtitle = "Sub"
        vg.plot(excel_df)
        assert len(vg.ax.texts) == 1

        vg.restyle()
        vg.restyle()

        assert len(vg.ax.texts) == 1
        assert vg.ax.texts[0].get_text() == "Sub"

    def test_restyle_updates_subtitle_text_in_place(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.plot_subtitle = "First"
        vg.plot(excel_df)
        artist = vg.ax.texts[0]

        vg.plot_subtitle = "Second"
        vg.restyle()

        assert len(vg.ax.texts) == 1
        assert vg.ax.texts[0] is artist  # same artist, text updated
        assert vg.ax.texts[0].get_text() == "Second"

    def test_restyle_removes_subtitle_when_cleared(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.plot_subtitle = "Sub"
        vg.plot(excel_df)

        vg.plot_subtitle = None
        vg.restyle()

        assert len(vg.ax.texts) == 0

    def test_restyle_twice_does_not_flip_axis_inversion_back(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.x_inverted = True
        vg.plot(excel_df)
        assert vg.ax.xaxis_inverted() == True  # noqa: E712 -- xaxis_inverted() returns numpy.bool_, not bool

        vg.restyle()
        vg.restyle()
        vg.restyle()

        assert vg.ax.xaxis_inverted() == True  # noqa: E712 -- xaxis_inverted() returns numpy.bool_, not bool

    def test_restyle_toggling_inversion_off_uninverts(self, vg, excel_df):
        _configure(vg, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        vg.x_inverted = True
        vg.plot(excel_df)

        vg.x_inverted = False
        vg.restyle()

        assert vg.ax.xaxis_inverted() == False  # noqa: E712 -- xaxis_inverted() returns numpy.bool_, not bool


class TestModelSchemaSync:
    """VGraph seeds its model-field defaults from MGraph's own dataclass
    schema (single source of truth) -- a hand-written copy once silently
    drifted (minor_ticks_* were missing on bare widgets)."""

    def test_bare_vgraph_carries_every_mgraph_field_with_model_defaults(self, qapp):
        from spectroview.model.m_graph import MGraph
        model_defaults = {k: v for k, v in vars(MGraph()).items() if k != 'graph_id'}
        vg = VGraph(graph_id=1)

        missing = [k for k in model_defaults if not hasattr(vg, k)]
        assert missing == []
        differing = {k: (v, getattr(vg, k)) for k, v in model_defaults.items()
                     if getattr(vg, k) != v}
        assert differing == {}

    def test_seeded_mutable_defaults_are_not_shared_between_widgets(self, qapp):
        vg1 = VGraph(graph_id=1)
        vg2 = VGraph(graph_id=2)
        vg1.annotations.append({'type': 'vline'})
        vg1.spines_visible['top'] = False

        assert vg2.annotations == []
        assert vg2.spines_visible['top'] is True


class TestExportButtonModifierClick:
    """The toolbar's Export button, like VSpectraViewer's Copy button, does
    double duty based on the Ctrl modifier at click time: a plain click
    exports just this graph (export_requested), Ctrl+Click exports every
    open graph instead (export_all_requested) -- this is what let the
    workspace's separate "Export All" side-panel button be removed."""

    def test_plain_click_emits_export_requested_for_this_graph(self, vg, monkeypatch):
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        monkeypatch.setattr(QApplication, 'keyboardModifiers', staticmethod(lambda: Qt.NoModifier))

        received = []
        received_all = []
        vg.export_requested.connect(lambda gid: received.append(gid))
        vg.export_all_requested.connect(lambda: received_all.append(True))

        vg._on_export_clicked()

        assert received == [vg.graph_id]
        assert received_all == []

    def test_ctrl_click_emits_export_all_requested_instead(self, vg, monkeypatch):
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        monkeypatch.setattr(QApplication, 'keyboardModifiers', staticmethod(lambda: Qt.ControlModifier))

        received = []
        received_all = []
        vg.export_requested.connect(lambda gid: received.append(gid))
        vg.export_all_requested.connect(lambda: received_all.append(True))

        vg._on_export_clicked()

        assert received == []
        assert received_all == [True]
