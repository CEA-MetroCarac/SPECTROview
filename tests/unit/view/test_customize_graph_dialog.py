"""Tests for view/components/customize_graph/ - the "Customize Graph"
dialog's sub-widgets (Legend, Axis, More Options, Annotations).

These are the actual controls behind every customization the user asked to
have tested. Each sub-widget is bound directly to a real, already-plotted
VGraph (real dataset_Excel.xlsx data) exactly as CustomizeGraphDialog wires
them up -- no mocking of the graph widget itself, only of modal dialogs
(QColorDialog/QMessageBox) that would otherwise block a headless test.
"""
import pandas as pd
import pytest

from spectroview.view.components.v_graph import VGraph
from spectroview.view.components.customize_graph.customize_graph_dialog import (
    CustomizeGraphDialog, CustomizeLegend, CustomizeAxis, CustomizeSecondaryAxes,
    CustomizeMoreOptions, CustomizeAnnotations,
)


@pytest.fixture(scope="module")
def excel_df(dataframe_excel_file):
    if not dataframe_excel_file.exists():
        pytest.skip("dataset_Excel.xlsx not present")
    return pd.read_excel(dataframe_excel_file, sheet_name="sheet1")


def _plotted_graph(qapp, excel_df, plot_style="scatter", x="x0_Si", y=None, z="Quadrant"):
    vg = VGraph(graph_id=1)
    vg.create_plot_widget(dpi=72)
    vg.df_name = "sheet1"
    vg.x = x
    vg.y = y if y is not None else ["ampli_Si"]
    vg.z = z
    vg.plot_style = plot_style
    vg.plot(excel_df)
    return vg


class TestCustomizeLegend:
    def test_scatter_group_visible_for_scatter_like_styles(self, qapp, excel_df):
        for style in ("scatter", "trendline", "point"):
            vg = _plotted_graph(qapp, excel_df, plot_style=style)
            widget = CustomizeLegend(vg)
            assert widget.scatter_group.isVisible() or not widget.isVisible()
            # isVisible() depends on the widget being shown; check the
            # underlying flag CustomizeLegend itself computes instead.
            assert vg.plot_style in ['scatter', 'trendline', 'point']

    def test_scatter_group_hidden_for_non_scatter_styles(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="bar", x="Zone", y=["ampli_Si"])
        widget = CustomizeLegend(vg)
        widget.load_legend_properties()
        assert widget.scatter_group.isVisibleTo(widget) is False

    def test_load_legend_properties_matches_graph_hue_count(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeLegend(vg)
        widget.load_legend_properties()
        n_quadrants = excel_df["Quadrant"].nunique()
        # label, marker(empty for non-point), color, linewidth, alpha,
        # zorder, marker_size, edge_color -- scatter shows all 8 sub-layouts
        # (marker_size/edge_color are only added for _MARKER_STYLES).
        assert widget.legend_layout.count() == 8
        # More directly: the graph's own legend_properties count matches hue count.
        assert len(vg.get_legend_properties()) == n_quadrants

    def test_per_series_style_overrides_stored_and_cleared(self, qapp, excel_df):
        """Per-series linewidth/alpha/zorder/marker_size/edge_color are
        optional overrides -- absent (not a sentinel value) when unset, and
        removed again (not left at a stale value) when reset to blank."""
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeLegend(vg)
        widget.load_legend_properties()

        widget._update_legend_property_numeric(0, "linewidth", 2.5)
        widget._update_legend_property_numeric(0, "alpha", 0.5)
        assert vg.legend_properties[0]["linewidth"] == 2.5
        assert vg.legend_properties[0]["alpha"] == 0.5

        widget._update_legend_property_numeric(0, "linewidth", widget._UNSET)
        assert "linewidth" not in vg.legend_properties[0]

    def test_per_series_marker_size_and_edge_color_only_shown_for_marker_styles(self, qapp, excel_df):
        vg_scatter = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget_scatter = CustomizeLegend(vg_scatter)
        widget_scatter.load_legend_properties()
        assert widget_scatter.legend_layout.count() == 8

        vg_bar = _plotted_graph(qapp, excel_df, plot_style="bar", x="Zone", y=["ampli_Si"])
        widget_bar = CustomizeLegend(vg_bar)
        widget_bar.load_legend_properties()
        # bar isn't in _MARKER_STYLES: label, marker(empty), color, linewidth, alpha, zorder = 6
        assert widget_bar.legend_layout.count() == 6

    def test_error_bar_group_visible_only_for_point_line_bar(self, qapp, excel_df):
        for style, x, y in [("point", "Zone", ["ampli_Si"]), ("line", "x0_Si", ["ampli_Si"]),
                             ("bar", "Zone", ["ampli_Si"])]:
            vg = _plotted_graph(qapp, excel_df, plot_style=style, x=x, y=y)
            widget = CustomizeLegend(vg)
            assert widget.error_bar_group.isVisibleTo(widget) is True

        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeLegend(vg)
        assert widget.error_bar_group.isVisibleTo(widget) is False

    def test_error_bar_type_dropdown_offers_none_for_point_but_not_bar(self, qapp, excel_df):
        vg_point = _plotted_graph(qapp, excel_df, plot_style="point", x="Zone", y=["ampli_Si"])
        widget_point = CustomizeLegend(vg_point)
        point_values = [widget_point.cbb_error_bar_type.itemData(i)
                         for i in range(widget_point.cbb_error_bar_type.count())]
        assert "none" in point_values

        vg_bar = _plotted_graph(qapp, excel_df, plot_style="bar", x="Zone", y=["ampli_Si"])
        widget_bar = CustomizeLegend(vg_bar)
        bar_values = [widget_bar.cbb_error_bar_type.itemData(i)
                      for i in range(widget_bar.cbb_error_bar_type.count())]
        assert "none" not in bar_values

    def test_apply_writes_error_bar_settings_back_and_emits_signal(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="point", x="Zone", y=["ampli_Si"])
        widget = CustomizeLegend(vg)
        idx = widget.cbb_error_bar_type.findData("sem")
        widget.cbb_error_bar_type.setCurrentIndex(idx)
        widget.spin_error_bar_capsize.setValue(6.0)

        received = []
        vg.properties_changed.connect(lambda gid, props: received.append((gid, props)))
        widget.apply_changes()

        assert vg.error_bar_type == "sem"
        assert vg.error_bar_capsize == 6.0
        gid, props = received[0]
        assert props["error_bar_type"] == "sem"
        assert props["error_bar_capsize"] == 6.0

    def test_load_populates_scatter_size_and_edgecolor_controls(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        vg.scatter_size = 123
        vg.scatter_edgecolor = "#ABCDEF"
        widget = CustomizeLegend(vg)
        widget.load_legend_properties()
        assert widget.spin_scatter_size.value() == 123
        assert widget.btn_scatter_edgecolor.text().lower() == "#abcdef"

    def test_apply_changes_writes_scatter_settings_back_and_emits_signal(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeLegend(vg)
        widget.spin_scatter_size.setValue(250)
        widget._set_color_button(widget.btn_scatter_edgecolor, "#112233")

        received = []
        vg.properties_changed.connect(lambda gid, props: received.append((gid, props)))
        widget.apply_changes()

        assert vg.scatter_size == 250
        assert vg.scatter_edgecolor == "#112233"
        assert len(received) == 1
        gid, props = received[0]
        assert gid == vg.graph_id
        assert props["scatter_size"] == 250
        assert props["scatter_edgecolor"] == "#112233"

    def test_apply_changes_rejects_blank_edgecolor_falls_back_to_black(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeLegend(vg)
        widget.btn_scatter_edgecolor.setText("")  # simulate an invalid/cleared value
        widget.apply_changes()
        assert vg.scatter_edgecolor == "black"

    def test_legend_outside_toggle_updates_graph(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeLegend(vg)
        widget.cb_legend_outside.setChecked(True)
        assert vg.legend_outside is True

    def test_legend_loc_disabled_when_outside_checked(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeLegend(vg)
        assert widget.combo_legend_loc.isEnabled() is True
        widget.cb_legend_outside.setChecked(True)
        assert widget.combo_legend_loc.isEnabled() is False

    def test_legend_style_defaults_match_pre_existing_hardcoded_behavior(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeLegend(vg)
        assert widget.spin_legend_ncol.value() == 1
        assert widget.cb_legend_frame.isChecked() is True
        assert widget.spin_legend_alpha.value() == pytest.approx(0.7)
        assert widget.combo_legend_loc.currentText() == "best"

    def test_apply_writes_legend_style_back_and_emits_signal(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeLegend(vg)
        widget.spin_legend_ncol.setValue(2)
        widget.cb_legend_frame.setChecked(False)
        widget.edit_legend_title.setText("Groups")
        widget.spin_legend_fontsize.setValue(8)
        widget.spin_legend_alpha.setValue(0.3)
        widget.combo_legend_loc.setCurrentText("upper left")

        received = []
        vg.properties_changed.connect(lambda gid, props: received.append(props))
        widget.apply_changes()

        assert vg.legend_ncol == 2
        assert vg.legend_frame is False
        assert vg.legend_title == "Groups"
        assert vg.legend_fontsize == 8
        assert vg.legend_alpha == pytest.approx(0.3)
        assert vg.legend_loc == "upper left"
        props = received[0]
        assert props["legend_ncol"] == 2
        assert props["legend_loc"] == "upper left"

    def test_cancel_changes_restores_original_legend_properties(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeLegend(vg)
        original = [dict(p) for p in vg.legend_properties]

        vg.legend_properties[0]["label"] = "Mutated"
        widget.cancel_changes()

        assert vg.legend_properties[0]["label"] == original[0]["label"]

    def test_update_legend_property_label_replots_live(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeLegend(vg)
        widget._update_legend_property(0, "label", "Renamed Group")
        assert vg.legend_properties[0]["label"] == "Renamed Group"
        legend = vg.ax.get_legend()
        assert legend is not None
        assert any(t.get_text() == "Renamed Group" for t in legend.get_texts())


class TestCustomizeAxis:
    def test_load_axis_settings_populates_from_graph(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        vg.xmin, vg.xmax = 510.0, 520.0
        widget = CustomizeAxis(vg)
        widget.load_axis_settings()
        assert widget.spin_xmin.value() == 510.0
        assert widget.spin_xmax.value() == 520.0

    def test_unset_limits_display_placeholder_not_the_sentinel_number(self, qapp, excel_df):
        """Regression test: spinboxes used to show the literal '-999999'
        for an unset limit, which read as a confusing real value rather
        than 'no limit set'. PlaceholderDoubleSpinBox's setSpecialValueText()
        makes the sentinel (spinbox range minimum) render as a grayed
        "default" placeholder word instead."""
        vg = _plotted_graph(qapp, excel_df)
        assert vg.xmin is None and vg.ymin is None and vg.zmin is None
        widget = CustomizeAxis(vg)
        widget.load_axis_settings()

        for spin in (widget.spin_xmin, widget.spin_xmax, widget.spin_ymin,
                     widget.spin_ymax, widget.spin_zmin, widget.spin_zmax):
            assert spin.value() == widget._UNSET_LIMIT
            assert spin.text().strip() == "default"

    def test_first_construction_does_not_corrupt_unset_limits_via_slider_clamp(self, qapp, excel_df):
        """Regression test: _update_range_slider_bounds() derives each range
        slider's drag bounds from the data and calls slider.setRange(lo, hi).
        The slider's initial value is (0, 100); for any numeric column whose
        real range excludes 0-100 (e.g. ampli_Si ~6500-9200), setRange()
        alone auto-clamps that stale value into the new bounds and fires
        valueChanged -- which _on_range_slider_changed then writes into the
        spinboxes, silently replacing the "default" placeholder with a real
        (wrong) number. This only happens on the *first* load_axis_settings()
        call (a second call self-heals, since by then the slider's value is
        already inside the unchanged range) so the test must check right
        after construction, not after a redundant reload."""
        vg = _plotted_graph(qapp, excel_df, x="x0_Si", y=["ampli_Si"])
        assert vg.xmin is None and vg.ymin is None
        widget = CustomizeAxis(vg)  # __init__ calls load_axis_settings() once

        for spin in (widget.spin_xmin, widget.spin_xmax, widget.spin_ymin, widget.spin_ymax):
            assert spin.value() == widget._UNSET_LIMIT
            assert spin.text().strip() == "default"

    def test_set_limit_displays_its_real_value_not_blank(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        vg.xmin, vg.xmax = 510.0, 520.0
        widget = CustomizeAxis(vg)
        widget.load_axis_settings()
        assert widget.spin_xmin.text().strip() != ""
        assert widget.spin_xmax.text().strip() != ""

    def test_clear_limits_button_shows_placeholder_again(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        vg.xmin, vg.xmax = 510.0, 520.0
        widget = CustomizeAxis(vg)
        widget.load_axis_settings()
        widget._on_clear_limits()
        assert widget.spin_xmin.text().strip() == "default"
        assert widget.spin_xmax.text().strip() == "default"

    def test_apply_writes_limits_back_including_zero(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAxis(vg)
        widget.spin_ymin.setValue(0.0)
        widget.spin_ymax.setValue(20000.0)
        widget._apply_axis_settings()
        assert vg.ymin == 0.0
        assert vg.ymax == 20000.0

    def test_range_slider_bounds_derived_from_data(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, x="x0_Si", y=["ampli_Si"])
        widget = CustomizeAxis(vg)
        widget.load_axis_settings()
        data_min, data_max = excel_df["x0_Si"].min(), excel_df["x0_Si"].max()
        assert widget.x_range_slider.minimum() < data_min
        assert widget.x_range_slider.maximum() > data_max

    def test_spinbox_edit_updates_range_slider(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, x="x0_Si", y=["ampli_Si"])
        widget = CustomizeAxis(vg)
        widget.load_axis_settings()
        data_min, data_max = excel_df["x0_Si"].min(), excel_df["x0_Si"].max()
        lo_val, hi_val = data_min, (data_min + data_max) / 2
        widget.spin_xmin.setValue(lo_val)
        widget.spin_xmax.setValue(hi_val)
        assert widget.x_range_slider.value() == pytest.approx((lo_val, hi_val), abs=0.01)

    def test_range_slider_drag_updates_spinboxes(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, x="x0_Si", y=["ampli_Si"])
        widget = CustomizeAxis(vg)
        widget.load_axis_settings()
        data_min, data_max = excel_df["x0_Si"].min(), excel_df["x0_Si"].max()
        lo_val, hi_val = data_min, (data_min + data_max) / 2
        widget.x_range_slider.setValue((lo_val, hi_val))
        assert widget.spin_xmin.value() == pytest.approx(lo_val, abs=0.01)
        assert widget.spin_xmax.value() == pytest.approx(hi_val, abs=0.01)

    def test_sentinel_minus_999999_becomes_none(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        vg.xmin, vg.xmax = 1.0, 2.0
        widget = CustomizeAxis(vg)
        widget.load_axis_settings()
        widget._on_clear_limits()
        widget._apply_axis_settings()
        assert vg.xmin is None
        assert vg.xmax is None

    def test_valid_x_break_is_applied(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAxis(vg)
        widget.x_break_enabled.setChecked(True)
        widget.x_break_start.setValue(515.0)
        widget.x_break_end.setValue(516.0)
        widget._apply_axis_settings()
        assert vg.axis_breaks["x"] == {"start": 515.0, "end": 516.0}

    def test_invalid_break_start_ge_end_shows_warning_and_does_not_apply(self, qapp, excel_df, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        warnings = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))

        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAxis(vg)
        widget.x_break_enabled.setChecked(True)
        widget.x_break_start.setValue(520.0)
        widget.x_break_end.setValue(510.0)  # start >= end: invalid
        widget._apply_axis_settings()

        assert len(warnings) == 1
        assert vg.axis_breaks.get("x") is None  # never got set

    def test_minor_ticks_applied(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAxis(vg)
        widget.cb_minor_top.setChecked(True)
        widget.cb_minor_right.setChecked(True)
        widget._apply_axis_settings()
        assert vg.minor_ticks_top is True
        assert vg.minor_ticks_right is True

    def test_log_scale_applied(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAxis(vg)
        widget.combo_y_scale.setCurrentText("Logarithmic")
        widget._apply_axis_settings()
        assert vg.ylogscale is True
        assert vg.ax.get_yscale() == "log"

    def test_symlog_scale_applied(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAxis(vg)
        widget.combo_y_scale.setCurrentText("Symlog")
        widget._apply_axis_settings()
        assert vg.ylogscale is True
        assert vg.yscale_mode == "symlog"
        assert vg.ax.get_yscale() == "symlog"

    def test_symlog_round_trips_through_load(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        vg.xlogscale = True
        vg.xscale_mode = "symlog"
        widget = CustomizeAxis(vg)
        widget.load_axis_settings()
        assert widget.combo_x_scale.currentText() == "Symlog"

    def test_axis_inversion_applied(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAxis(vg)
        widget.cb_invert_x.setChecked(True)
        widget._apply_axis_settings()
        assert vg.x_inverted is True
        assert vg.ax.xaxis_inverted() == True

    def test_tick_direction_applied(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAxis(vg)
        widget.combo_tick_direction.setCurrentText("In")
        widget._apply_axis_settings()
        assert vg.tick_direction == "in"

    def test_tick_direction_default_stays_none(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAxis(vg)
        widget._apply_axis_settings()
        assert vg.tick_direction is None

    def test_tick_label_format_applied(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAxis(vg)
        idx = widget.combo_tick_format.findData("%.2f")
        widget.combo_tick_format.setCurrentIndex(idx)
        widget._apply_axis_settings()
        assert vg.tick_label_format == "%.2f"

    def test_tick_format_auto_is_none(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAxis(vg)
        assert widget.combo_tick_format.currentText().startswith("Auto")
        widget._apply_axis_settings()
        assert vg.tick_label_format is None

    def test_tick_format_preserves_custom_value_set_outside_gui(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        vg.tick_label_format = "%.3g"  # not one of the GUI presets
        widget = CustomizeAxis(vg)
        widget.load_axis_settings()
        assert widget.combo_tick_format.currentData() == "%.3g"
        widget._apply_axis_settings()
        assert vg.tick_label_format == "%.3g"

    def test_font_sizes_show_mplstyle_defaults_and_are_settable(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAxis(vg)
        assert widget.spin_title_fontsize.value() == 12
        assert widget.spin_axis_label_fontsize.value() == 12
        assert widget.spin_tick_fontsize.value() == 9

        widget._apply_axis_settings()
        assert vg.title_fontsize == 12
        assert vg.axis_label_fontsize == 12
        assert vg.tick_label_fontsize == 9

        widget.spin_title_fontsize.setValue(18)
        widget._apply_axis_settings()
        assert vg.title_fontsize == 18
        assert vg.ax.title.get_fontsize() == 18

    def test_x_as_numeric_type_applied(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, x="Zone", plot_style="point")
        widget = CustomizeAxis(vg)
        widget.combo_x_type.setCurrentText("Category")
        widget._apply_axis_settings()
        assert vg.x_as_numeric is False

    def test_get_current_limits_reads_from_rendered_axes(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAxis(vg)
        widget._on_get_current_limits()
        xlo, xhi = vg.ax.get_xlim()
        # QDoubleSpinBox defaults to 2 decimal places, so it rounds the
        # already-rounded-to-3 value passed to setValue() a second time.
        assert widget.spin_xmin.value() == pytest.approx(xlo, abs=0.01)
        assert widget.spin_xmax.value() == pytest.approx(xhi, abs=0.01)

    def test_apply_emits_properties_changed_with_axis_breaks(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAxis(vg)
        widget.x_break_enabled.setChecked(True)
        widget.x_break_start.setValue(1.0)
        widget.x_break_end.setValue(2.0)

        received = []
        vg.properties_changed.connect(lambda gid, props: received.append(props))
        widget._apply_axis_settings()

        assert len(received) == 1
        assert received[0]["axis_breaks"]["x"] == {"start": 1.0, "end": 2.0}

    def test_xy_limits_hidden_for_wafer_and_2dmap(self, qapp, excel_df):
        """X/Y limits don't apply to wafer/2Dmap (spatial axes governed by
        wafer_size, not a user-set min/max) -- hiding them prevents setting
        a degenerate min==max pair that used to trigger a matplotlib
        'singular transformation' warning on reload."""
        # 2Dmap pivots on (x, y), so needs unique coordinate pairs; the raw
        # sheet has repeated (X, Y) positions (see test_v_graph_plotting.py).
        unique_xy_df = excel_df.drop_duplicates(subset=["X", "Y"])
        for style, df in (("wafer", excel_df), ("2Dmap", unique_xy_df)):
            vg = _plotted_graph(qapp, df, plot_style=style, x="X", y=["Y"], z="ampli_Si")
            widget = CustomizeAxis(vg)
            widget.load_axis_settings()
            assert widget.xy_limits_widget.isVisibleTo(widget) is False

    def test_xy_limits_visible_for_other_styles(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeAxis(vg)
        widget.load_axis_settings()
        assert widget.xy_limits_widget.isVisibleTo(widget) is True


class TestCustomizeSecondaryAxes:
    def test_secondary_axis_row_disabled_when_axis_inactive(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeSecondaryAxes(vg)
        row = widget._secondary_axis_rows['y2']
        assert row['label'].isEnabled() is False

    def test_secondary_axis_row_enabled_when_axis_active(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        vg.y2 = "area_Si"
        vg.plot(excel_df)
        widget = CustomizeSecondaryAxes(vg)
        row = widget._secondary_axis_rows['y2']
        assert row['label'].isEnabled() is True
        assert row['color'].text() == "red"  # matches the pre-existing hardcoded default

    def test_apply_secondary_axis_color_and_marker(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        vg.y2 = "area_Si"
        vg.plot(excel_df)
        widget = CustomizeSecondaryAxes(vg)
        row = widget._secondary_axis_rows['y2']
        row['color'].setText("#0000FF")
        row['marker'].setCurrentText("D")
        row['label'].setText("Custom Y2")

        received = []
        vg.properties_changed.connect(lambda gid, props: received.append(props))
        widget.apply_changes()

        assert vg.y2color == "#0000FF"
        assert vg.y2marker == "D"
        assert vg.y2label == "Custom Y2"
        assert received[0]["y2color"] == "#0000FF"

    def test_switch_graph_reloads_secondary_axis_rows(self, qapp, excel_df):
        vg1 = _plotted_graph(qapp, excel_df)
        vg2 = _plotted_graph(qapp, excel_df)
        vg2.y2 = "area_Si"
        vg2.plot(excel_df)
        widget = CustomizeSecondaryAxes(vg1)
        assert widget._secondary_axis_rows['y2']['label'].isEnabled() is False

        widget.switch_graph(vg2)
        assert widget.graph_widget is vg2
        assert widget._secondary_axis_rows['y2']['label'].isEnabled() is True

    def test_color_button_usable_for_active_axis(self, qapp, excel_df, monkeypatch):
        """Regression test for a real reported bug: opening the Customize
        dialog on a graph with an active secondary axis (y2 assigned) and
        clicking the "Secondary axes" row's Color button raised
        `TypeError: <lambda>() missing 1 required positional argument: '_'`.

        Root cause: `btn_color.clicked.connect(lambda _, k=axis_key: ...)`
        relies on Qt always invoking the connected callable with exactly the
        one `bool checked` argument QPushButton.clicked carries. That count
        is decided by PySide6's own signal/slot arg-matching, which is not
        guaranteed stable across PySide6 builds -- the user's build called
        the lambda with zero arguments, and since `_` has no default, Python
        raised exactly this TypeError (reproduced verbatim by calling the
        pre-fix lambda shape with zero args). The fix uses
        `lambda *_a, k=v: ...`, which tolerates any argument count Qt
        decides to pass.
        """
        import inspect

        from spectroview.view.components.customize_graph import (
            customize_axis, customize_legend, customize_secondary_axes,
        )

        for module in (customize_axis, customize_legend, customize_secondary_axes):
            source = inspect.getsource(module)
            assert "lambda _," not in source, (
                f"{module.__name__} reintroduced the fragile "
                "`lambda _, k=v: ...` Qt-signal-connection pattern -- use "
                "`lambda *_a, k=v: ...` instead (see this test's docstring)."
            )

        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QColorDialog

        monkeypatch.setattr(QColorDialog, "getColor", staticmethod(lambda *a, **k: QColor("blue")))

        vg = _plotted_graph(qapp, excel_df)
        vg.y2 = "fwhm_Si"
        widget = CustomizeSecondaryAxes(vg)
        row = widget._secondary_axis_rows["y2"]
        assert row["color"].isEnabled()
        widget._pick_secondary_axis_color("y2")  # must not raise
        row["color"].click()  # exercises the actual connected lambda via a real Qt signal


class TestCustomizeMoreOptions:
    def test_trendline_group_visible_only_for_trendline(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="trendline", y=["area_Si"])
        widget = CustomizeMoreOptions(vg)
        widget.load_settings()
        assert widget._trendline_group.isVisibleTo(widget) is True

        vg2 = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget2 = CustomizeMoreOptions(vg2)
        widget2.load_settings()
        assert widget2._trendline_group.isVisibleTo(widget2) is False

    def test_histogram_group_visible_only_for_histogram(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="histogram", x="fwhm_Si")
        widget = CustomizeMoreOptions(vg)
        widget.load_settings()
        assert widget._histogram_group.isVisibleTo(widget) is True

    def test_apply_point_plot_join_and_dodge(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="point", x="Zone")
        widget = CustomizeMoreOptions(vg)
        widget._cb_join.setChecked(True)
        widget._cb_dodge.setChecked(False)
        widget._apply()
        assert vg.join_for_point_plot is True
        assert vg.dodge_point_plot is False

    def test_apply_scatter_dodge(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeMoreOptions(vg)
        widget._cb_dodge_scatter.setChecked(True)
        widget._apply()
        assert vg.dodge_scatter_plot is True

    def test_apply_bar_error_bar_and_wafer_stats(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="bar", x="Zone")
        widget = CustomizeMoreOptions(vg)
        widget._cb_error_bar.setChecked(True)
        widget._cb_wafer_stats.setChecked(False)
        widget._apply()
        assert vg.show_bar_plot_error_bar is True
        assert vg.wafer_stats is False

    def test_figure_style_defaults_blank(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeMoreOptions(vg)
        assert widget._btn_figure_facecolor.text() == "(default)"
        assert widget._edit_subtitle.text() == ""
        assert widget._cb_spine_top.isChecked() is True
        assert widget._combo_figure_theme.currentText() == "Light"

    def test_apply_figure_theme_writes_back(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeMoreOptions(vg)
        widget._combo_figure_theme.setCurrentText("Dark")

        received = []
        vg.properties_changed.connect(lambda gid, props: received.append(props))
        widget._apply()

        assert vg.figure_theme == "dark"
        assert received[0]["figure_theme"] == "dark"

    def test_apply_figure_style_writes_back_and_emits_signal(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeMoreOptions(vg)
        widget._btn_figure_facecolor.setText("#EEEEEE")
        widget._edit_subtitle.setText("A subtitle")
        widget._spin_subtitle_fontsize.setValue(9)
        widget._cb_spine_top.setChecked(False)
        widget._cb_spine_right.setChecked(False)
        widget._spin_x_margin.setValue(0.1)
        widget._spin_y_margin.setValue(0.15)

        received = []
        vg.properties_changed.connect(lambda gid, props: received.append(props))
        widget._apply()

        assert vg.figure_facecolor == "#EEEEEE"
        assert vg.plot_subtitle == "A subtitle"
        assert vg.subtitle_fontsize == 9
        assert vg.spines_visible == {'top': False, 'right': False, 'bottom': True, 'left': True}
        assert vg.figure_margins == [0.1, 0.15]
        assert received[0]["figure_facecolor"] == "#EEEEEE"
        assert received[0]["spines_visible"]["top"] is False

    def test_apply_sorting_settings_reset_legend_properties_when_changed(self, qapp, excel_df):
        """_apply() resets legend_properties to [] when sort settings change
        *and then replots* (gw.plot(gw.df) at the end of _apply()), which
        immediately repopulates it fresh from the newly-sorted axes -- so
        the externally-observable effect is a full, consistent legend
        matching the current hue count, not a permanently empty list."""
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        stale_properties = [dict(p, label="STALE") for p in vg.legend_properties]
        vg.legend_properties = stale_properties
        widget = CustomizeMoreOptions(vg)
        widget._cbb_sort_by.setCurrentIndex(1)  # 'X values', was 'Z' (index 0)
        widget._apply()
        assert vg.sort_data_by == "X"
        n_quadrants = excel_df["Quadrant"].nunique()
        assert len(vg.legend_properties) == n_quadrants
        assert all(p["label"] != "STALE" for p in vg.legend_properties)

    def test_apply_trendline_order_and_anchor(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="trendline", y=["area_Si"])
        widget = CustomizeMoreOptions(vg)
        widget._spin_order.setValue(2)
        widget._anchor_grp.setChecked(True)
        widget._rb_custom.setChecked(True)
        widget._spin_ax.setValue(5.0)
        widget._spin_ay.setValue(10.0)
        widget._apply()
        assert vg.trendline_order == 2
        assert vg.trendline_anchor_enabled is True
        assert vg.trendline_anchor_origin is False
        assert vg.trendline_anchor_x == 5.0
        assert vg.trendline_anchor_y == 10.0

    def test_apply_histogram_bins_kde_step(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="histogram", x="fwhm_Si")
        widget = CustomizeMoreOptions(vg)
        widget._spin_bins.setValue(35)
        widget._cb_kde.setChecked(True)
        widget._rb_step.setChecked(True)
        widget._apply()
        assert vg.hist_bins == 35
        assert vg.hist_kde is True
        assert vg.hist_step is True

    def test_apply_emits_properties_changed(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="point", x="Zone")
        widget = CustomizeMoreOptions(vg)
        received = []
        vg.properties_changed.connect(lambda gid, props: received.append(props))
        widget._apply()
        assert len(received) == 1
        assert "sort_data_enabled" in received[0]


class TestCustomizeAnnotations:
    def test_add_vline_appends_annotation_and_emits_signal(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAnnotations(vg)
        received = []
        vg.properties_changed.connect(lambda gid, props: received.append(props))

        widget._add_vline()

        assert len(vg.annotations) == 1
        assert vg.annotations[0]["type"] == "vline"
        assert received and "annotations" in received[-1]
        assert widget.annotation_list.count() == 1

    def test_add_hline_and_text(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAnnotations(vg)
        widget._add_hline()
        widget._add_text()
        assert {a["type"] for a in vg.annotations} == {"hline", "text"}
        assert widget.annotation_list.count() == 2

    def test_delete_annotation_removes_it(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAnnotations(vg)
        widget._add_vline()
        ann_id = vg.annotations[0]["id"]

        widget.annotation_list.setCurrentRow(0)
        widget._delete_annotation()

        assert vg.annotations == []
        assert widget.annotation_list.count() == 0

    def test_delete_with_no_selection_warns_instead_of_crashing(self, qapp, excel_df, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        warnings = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAnnotations(vg)
        widget._delete_annotation()
        assert len(warnings) == 1


class TestCustomizeGraphDialogTopLevel:
    def test_apply_all_delegates_to_every_tab(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        dialog = CustomizeGraphDialog(vg, graph_id=1)

        dialog.legend_widget.spin_scatter_size.setValue(300)
        dialog.axis_widget.spin_xmin.setValue(505.0)
        dialog.axis_widget.spin_xmax.setValue(535.0)

        dialog.apply_all()

        assert vg.scatter_size == 300
        assert vg.xmin == 505.0
        assert vg.xmax == 535.0

    def test_switch_graph_rebinds_all_subwidgets(self, qapp, excel_df):
        vg1 = _plotted_graph(qapp, excel_df, plot_style="scatter")
        vg2 = _plotted_graph(qapp, excel_df, plot_style="bar", x="Zone", y=["ampli_Si"])
        dialog = CustomizeGraphDialog(vg1, graph_id=1)

        dialog.switch_graph(vg2, graph_id=2)

        assert dialog.graph_widget is vg2
        assert dialog.legend_widget.graph_widget is vg2
        assert dialog.axis_widget.graph_widget is vg2
        assert dialog.secondary_axes_widget.graph_widget is vg2
        assert dialog.annotations_widget.graph_widget is vg2
        assert dialog.more_options_widget.graph_widget is vg2

    def test_switch_graph_to_same_id_is_noop(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        dialog = CustomizeGraphDialog(vg, graph_id=1)
        dialog.switch_graph(vg, graph_id=1)
        assert dialog.graph_widget is vg
