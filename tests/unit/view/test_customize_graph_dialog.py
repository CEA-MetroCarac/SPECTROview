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
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette

from spectroview.model.m_graph import MGraph
from spectroview.view.components.v_graph import VGraph
from spectroview.view.components.customize_graph.customize_graph_dialog import (
    CustomizeGraphDialog, CustomizeLegend, CustomizeAxis,
    CustomizeMoreOptions, CustomizeAnnotations,
)


def _is_grayed(spin) -> bool:
    """True if `spin` (a PlaceholderDoubleSpinBox/PlaceholderSpinBox) is
    currently showing its grayed "unset" placeholder styling.

    _update_placeholder_style() uses QPalette (not setStyleSheet()) to gray
    out the sentinel state -- setStyleSheet() was found to break native
    widget margins/height on macOS, clipping the text -- so this checks the
    line edit's own Text color role rather than a stylesheet string.
    """
    return spin.lineEdit().palette().color(QPalette.ColorRole.Text) == Qt.GlobalColor.gray


@pytest.fixture(scope="module")
def excel_df(dataframe_excel_file):
    if not dataframe_excel_file.exists():
        pytest.skip("dataset_Excel.xlsx not present")
    return pd.read_excel(dataframe_excel_file, sheet_name="sheet1")


def _plotted_graph(qapp, excel_df, plot_style="scatter", x="x0_Si", y=None, z="Quadrant"):
    # Copy every MGraph field onto the widget first (mirroring
    # v_workspace_graphs.py's _configure_graph_from_model()), not just the
    # handful this helper cares about -- a bare VGraph() leaves fields like
    # minor_ticks_bottom simply absent as attributes, which every real
    # widget never is (the app always routes through a real MGraph's
    # dataclass defaults before a dialog can be opened on it). Phase 5D's
    # cancel_all()/switch_graph() snapshot the widget's full field set and
    # write it back wholesale, which surfaced the gap.
    model = MGraph(graph_id=1, df_name="sheet1", x=x, y=y if y is not None else ["ampli_Si"],
                    z=z, plot_style=plot_style)
    vg = VGraph(graph_id=1)
    for key, value in vars(model).items():
        if key == 'graph_id':
            continue
        setattr(vg, key, value)
    vg.create_plot_widget(dpi=72)
    vg.plot(excel_df)
    return vg


class TestCustomizeLegend:
    def test_unified_marker_widget_visible_for_scatter_like_styles_by_default(self, qapp, excel_df):
        """unify_marker_style defaults to True, so the shared marker size/
        edge color row shows for every marker-drawing style out of the box."""
        for style in ("scatter", "trendline", "point"):
            vg = _plotted_graph(qapp, excel_df, plot_style=style)
            assert vg.unify_marker_style is True
            widget = CustomizeLegend(vg)
            widget.load_legend_properties()
            assert widget.unified_marker_widget.isVisibleTo(widget) is True

    def test_unified_marker_widget_hidden_for_non_scatter_styles(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="bar", x="Zone", y=["ampli_Si"])
        widget = CustomizeLegend(vg)
        widget.load_legend_properties()
        assert widget.unified_marker_widget.isVisibleTo(widget) is False

    def test_unified_marker_widget_hidden_when_unify_unchecked(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        vg.unify_marker_style = False
        widget = CustomizeLegend(vg)
        widget.load_legend_properties()
        assert widget.unified_marker_widget.isVisibleTo(widget) is False

    def test_load_legend_properties_matches_graph_hue_count(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeLegend(vg)
        widget.load_legend_properties()
        n_quadrants = excel_df["Quadrant"].nunique()
        # label, color, alpha, + trailing stretch -- 4 items by default for
        # scatter (unify_marker_style=True hides the per-series
        # marker_size/edge_color columns; Marker is point-only; Line width
        # only shows for styles that actually draw a line -- see the "not
        # unified" test below for the 6-item case).
        assert widget.legend_layout.count() == 4
        # More directly: the graph's own legend_properties count matches hue count.
        assert len(vg.get_legend_properties()) == n_quadrants

    def test_load_legend_properties_shows_per_series_columns_when_not_unified(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        vg.unify_marker_style = False
        widget = CustomizeLegend(vg)
        widget.load_legend_properties()
        assert widget.legend_layout.count() == 6

    def test_per_series_style_overrides_stored_and_cleared(self, qapp, excel_df):
        """Per-series linewidth/alpha/marker_size/edge_color are optional
        overrides -- absent (not a sentinel value) when unset, and removed
        again (not left at a stale value) when reset to blank."""
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeLegend(vg)
        widget.load_legend_properties()

        widget._update_legend_property_numeric(0, "linewidth", 2.5)
        widget._update_legend_property_numeric(0, "alpha", 0.5)
        assert vg.legend_properties[0]["linewidth"] == 2.5
        assert vg.legend_properties[0]["alpha"] == 0.5

        widget._update_legend_property_numeric(0, "linewidth", widget._UNSET)
        assert "linewidth" not in vg.legend_properties[0]

    def test_per_series_unset_spinboxes_show_the_real_effective_value(self, qapp, excel_df):
        """Alpha/Line width unset placeholders show the value that will
        actually be used (1.00 / 1.50) instead of a generic "default" word;
        Marker size shows the graph's current global scatter_size.

        Uses 'trendline' (not 'scatter') so the Line width column is
        actually present -- it only shows for styles that draw a line
        (line/trendline, or point with join enabled); trendline is also in
        _MARKER_STYLES so the marker_size/edge_color columns show too once
        unify_marker_style is off, letting one test cover all three.
        """
        vg = _plotted_graph(qapp, excel_df, plot_style="trendline")
        vg.scatter_size = 123
        vg.unify_marker_style = False  # per-series marker_size/edge_color columns only exist when not unified
        widget = CustomizeLegend(vg)
        widget.load_legend_properties()

        # Column order in legend_layout: label(0), color(1), linewidth(2),
        # alpha(3), marker_size(4), edge_color(5) -- Marker is point-only,
        # so it's absent here; item 0 within each column is its header,
        # item 1 is row 0's widget.
        lw_spin = widget.legend_layout.itemAt(2).layout().itemAt(1).widget()
        alpha_spin = widget.legend_layout.itemAt(3).layout().itemAt(1).widget()
        ms_spin = widget.legend_layout.itemAt(4).layout().itemAt(1).widget()
        edge_btn = widget.legend_layout.itemAt(5).layout().itemAt(1).widget()

        assert lw_spin.specialValueText() == "1.50"
        assert alpha_spin.specialValueText() == "1.00"
        assert ms_spin.specialValueText() == "123"
        assert edge_btn.text().lower() == "#000000"  # global scatter_edgecolor default

    def test_per_series_marker_size_and_edge_color_only_shown_for_marker_styles(self, qapp, excel_df):
        vg_scatter = _plotted_graph(qapp, excel_df, plot_style="scatter")
        vg_scatter.unify_marker_style = False
        widget_scatter = CustomizeLegend(vg_scatter)
        widget_scatter.load_legend_properties()
        assert widget_scatter.legend_layout.count() == 6

        vg_bar = _plotted_graph(qapp, excel_df, plot_style="bar", x="Zone", y=["ampli_Si"])
        widget_bar = CustomizeLegend(vg_bar)
        widget_bar.load_legend_properties()
        # bar isn't in _MARKER_STYLES and doesn't draw a line: label, color,
        # alpha, + trailing stretch = 4
        assert widget_bar.legend_layout.count() == 4

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

    def test_unchecking_unify_reveals_per_series_columns_live(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeLegend(vg)
        assert widget.unified_marker_widget.isVisibleTo(widget) is True
        assert widget.legend_layout.count() == 4

        widget.cb_unify_marker_style.setChecked(False)

        assert vg.unify_marker_style is False
        assert widget.unified_marker_widget.isVisibleTo(widget) is False
        assert widget.legend_layout.count() == 6

    def test_apply_persists_unify_marker_style(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeLegend(vg)
        widget.cb_unify_marker_style.setChecked(False)

        received = []
        vg.properties_changed.connect(lambda gid, props: received.append(props))
        widget.apply_changes()

        assert vg.unify_marker_style is False
        assert received[-1]["unify_marker_style"] is False

    def test_unify_toggle_schedules_a_window_resize(self, qapp, excel_df):
        """Regression: adding/removing the per-series marker_size/edge_color
        columns changes legend_widget's sizeHint, but the dialog doesn't
        auto-grow to fit until Qt has processed the old columns' pending
        deleteLater() cleanup -- _resize_window_to_fit_content() (deferred
        via QTimer.singleShot so it runs after that settles) is what
        actually calls adjustSize() on the top-level window."""
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeLegend(vg)
        adjusted = []
        widget.window().adjustSize = lambda: adjusted.append(True)

        widget._resize_window_to_fit_content()

        assert adjusted == [True]

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
        makes the sentinel (spinbox range minimum) render grayed instead --
        as the real current/effective value (e.g. the actually-rendered
        axis limit), not a generic "default" word that doesn't tell the
        user what will actually be used."""
        vg = _plotted_graph(qapp, excel_df)
        assert vg.xmin is None and vg.ymin is None and vg.zmin is None
        widget = CustomizeAxis(vg)
        widget.load_axis_settings()

        for spin in (widget.spin_xmin, widget.spin_xmax, widget.spin_ymin,
                     widget.spin_ymax, widget.spin_zmin, widget.spin_zmax):
            assert spin.value() == widget._UNSET_LIMIT
            text = spin.text().strip()
            assert text not in ("", "default", str(int(widget._UNSET_LIMIT)))
            assert _is_grayed(spin)

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
            assert _is_grayed(spin)

    def test_unset_limit_placeholder_shows_the_actual_rendered_axis_limit(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, x="x0_Si", y=["ampli_Si"], plot_style="scatter")
        assert vg.xmin is None
        widget = CustomizeAxis(vg)
        widget.load_axis_settings()

        rendered_xmin, rendered_xmax = vg.ax.get_xlim()
        assert widget.spin_xmin.specialValueText() == f"{rendered_xmin:.2f}"
        assert widget.spin_xmax.specialValueText() == f"{rendered_xmax:.2f}"

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
        assert widget.spin_xmin.value() == widget._UNSET_LIMIT
        assert widget.spin_xmax.value() == widget._UNSET_LIMIT
        assert _is_grayed(widget.spin_xmin)
        assert _is_grayed(widget.spin_xmax)

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

    def test_spine_visibility_defaults_all_checked(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAxis(vg)
        assert widget.cb_spine_top.isChecked() is True
        assert widget.cb_spine_right.isChecked() is True
        assert widget.cb_spine_bottom.isChecked() is True
        assert widget.cb_spine_left.isChecked() is True

    def test_spine_visibility_applied_and_emitted(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAxis(vg)
        widget.cb_spine_top.setChecked(False)
        widget.cb_spine_right.setChecked(False)

        received = []
        vg.properties_changed.connect(lambda gid, props: received.append(props))
        widget._apply_axis_settings()

        assert vg.spines_visible == {'top': False, 'right': False, 'bottom': True, 'left': True}
        assert received[0]["spines_visible"]["top"] is False

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

    # ---- Inset (zoom) axes -- merged into this tab from the old
    # standalone Inset Axes tab. ----

    def test_inset_disabled_by_default(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeAxis(vg)
        assert widget.inset_group.isChecked() is False
        # Checkable-groupbox unchecked -> Qt disables its children for free.
        assert widget.spin_inset_x0.isEnabled() is False

    def test_inset_load_populates_bounds_and_limits_from_graph(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        vg.inset_enabled = True
        vg.inset_bounds = [0.1, 0.2, 0.3, 0.4]
        vg.inset_xmin, vg.inset_xmax = 505.0, 520.0
        vg.inset_show_zoom_indicator = False
        widget = CustomizeAxis(vg)
        widget.load_axis_settings()

        assert widget.inset_group.isChecked() is True
        assert widget.spin_inset_x0.value() == pytest.approx(0.1)
        assert widget.spin_inset_y0.value() == pytest.approx(0.2)
        assert widget.spin_inset_w.value() == pytest.approx(0.3)
        assert widget.spin_inset_h.value() == pytest.approx(0.4)
        assert widget.spin_inset_xmin.value() == pytest.approx(505.0)
        assert widget.spin_inset_xmax.value() == pytest.approx(520.0)
        assert widget.spin_inset_ymin.value() == widget._UNSET_LIMIT
        assert _is_grayed(widget.spin_inset_ymin)
        assert widget.cb_inset_zoom_indicator.isChecked() is False

    def test_inset_apply_writes_back_all_inset_fields(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeAxis(vg)
        widget.inset_group.setChecked(True)
        widget.spin_inset_x0.setValue(0.15)
        widget.spin_inset_y0.setValue(0.15)
        widget.spin_inset_w.setValue(0.25)
        widget.spin_inset_h.setValue(0.25)
        widget.spin_inset_xmin.setValue(508.0)
        widget.spin_inset_ymax.setValue(9000.0)
        widget.cb_inset_zoom_indicator.setChecked(False)

        received = []
        vg.properties_changed.connect(lambda gid, props: received.append(props))
        widget._apply_axis_settings()

        assert vg.inset_enabled is True
        assert vg.inset_bounds == [0.15, 0.15, 0.25, 0.25]
        assert vg.inset_xmin == 508.0
        assert vg.inset_xmax is None  # left at placeholder -> None
        assert vg.inset_ymax == 9000.0
        assert vg.inset_show_zoom_indicator is False
        assert received[0]["inset_enabled"] is True
        assert received[0]["inset_bounds"] == [0.15, 0.15, 0.25, 0.25]

    def test_inset_apply_clears_limits_back_to_none_when_left_at_placeholder(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        vg.inset_xmin, vg.inset_xmax = 100.0, 200.0
        widget = CustomizeAxis(vg)
        widget.load_axis_settings()

        u = widget._UNSET_LIMIT
        widget.spin_inset_xmin.setValue(u)
        widget.spin_inset_xmax.setValue(u)
        widget._apply_axis_settings()

        assert vg.inset_xmin is None
        assert vg.inset_xmax is None

    # ---- Secondary axes (Y2/Y3/X2) -- merged into this tab from the old
    # standalone Secondary axes tab. ----

    def test_secondary_axis_row_disabled_when_axis_inactive(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAxis(vg)
        row = widget._secondary_axis_rows['y2']
        assert row['label'].isEnabled() is False

    def test_secondary_axis_row_enabled_when_axis_active(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        vg.y2 = "area_Si"
        vg.plot(excel_df)
        widget = CustomizeAxis(vg)
        row = widget._secondary_axis_rows['y2']
        assert row['label'].isEnabled() is True
        assert row['color'].text() == "red"  # matches the pre-existing hardcoded default

    def test_apply_secondary_axis_color_and_marker(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        vg.y2 = "area_Si"
        vg.plot(excel_df)
        widget = CustomizeAxis(vg)
        row = widget._secondary_axis_rows['y2']
        row['color'].setText("#0000FF")
        row['marker'].setCurrentText("D")
        row['label'].setText("Custom Y2")

        received = []
        vg.properties_changed.connect(lambda gid, props: received.append(props))
        widget._apply_axis_settings()

        assert vg.y2color == "#0000FF"
        assert vg.y2marker == "D"
        assert vg.y2label == "Custom Y2"
        assert received[0]["y2color"] == "#0000FF"

    def test_switch_graph_reloads_secondary_axis_rows(self, qapp, excel_df):
        vg1 = _plotted_graph(qapp, excel_df)
        vg2 = _plotted_graph(qapp, excel_df)
        vg2.y2 = "area_Si"
        vg2.plot(excel_df)
        widget = CustomizeAxis(vg1)
        assert widget._secondary_axis_rows['y2']['label'].isEnabled() is False

        widget.switch_graph(vg2)
        assert widget.graph_widget is vg2
        assert widget._secondary_axis_rows['y2']['label'].isEnabled() is True

    def test_secondary_axis_color_button_usable_for_active_axis(self, qapp, excel_df, monkeypatch):
        """Regression test for a real reported bug: opening the Customize
        dialog on a graph with an active secondary axis (y2 assigned) and
        clicking the "Secondary axes" row's Color button raised
        `TypeError: <lambda>() missing 1 required positional argument: '_'`.
        Root cause: `lambda _, k=v: ...` assumes Qt always calls it with
        exactly one arg, which isn't guaranteed stable across PySide6
        builds. Fix: `lambda *_a, k=v: ...`, tolerant of any arg count.
        """
        import inspect

        from spectroview.view.components.customize_graph import customize_axis, customize_legend

        for module in (customize_axis, customize_legend):
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
        widget = CustomizeAxis(vg)
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

    def test_theme_defaults_to_light(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeMoreOptions(vg)
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

    def test_apply_xlabel_rotation_and_grid_writes_back(self, qapp, excel_df):
        """X label rotation and Grid -- migrated here from the workspace's
        bottom toolbar -- share the Theme row and write straight to
        VGraph.x_rot/grid."""
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeMoreOptions(vg)
        assert widget._spin_xlabel_rotation.value() == 0
        assert widget._cb_grid.isChecked() is False

        widget._spin_xlabel_rotation.setValue(45)
        widget._cb_grid.setChecked(True)
        received = []
        vg.properties_changed.connect(lambda gid, props: received.append(props))
        widget._apply()

        assert vg.x_rot == 45
        assert vg.grid is True
        assert received[0]["x_rot"] == 45
        assert received[0]["grid"] is True

    def test_font_sizes_show_mplstyle_defaults_and_are_settable(self, qapp, excel_df):
        """Every font-size control (Title/Subtitle/Axis label/Tick label)
        lives here now, in one row -- consolidated from the old standalone
        Text Size tab."""
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget = CustomizeMoreOptions(vg)
        assert widget._spin_title_fontsize.value() == 12
        assert widget._spin_subtitle_fontsize.value() == 10
        assert widget._spin_axis_label_fontsize.value() == 12
        assert widget._spin_tick_fontsize.value() == 9

        widget._spin_title_fontsize.setValue(18)
        received = []
        vg.properties_changed.connect(lambda gid, props: received.append(props))
        widget._apply()

        assert vg.title_fontsize == 18
        assert vg.ax.title.get_fontsize() == 18
        assert received[0]["title_fontsize"] == 18

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

    def test_colormap_group_visible_only_for_wafer_and_2dmap(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="wafer", x="X", y=["Y"], z="ampli_Si")
        widget = CustomizeMoreOptions(vg)
        widget.load_settings()
        assert widget._colormap_group.isVisibleTo(widget) is True

        vg2 = _plotted_graph(qapp, excel_df, plot_style="scatter")
        widget2 = CustomizeMoreOptions(vg2)
        widget2.load_settings()
        assert widget2._colormap_group.isVisibleTo(widget2) is False

    def test_colormap_load_populates_from_graph(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="wafer", x="X", y=["Y"], z="ampli_Si")
        vg.colormap_norm = "centered"
        vg.colormap_center = 12.5
        widget = CustomizeMoreOptions(vg)
        widget.load_settings()
        assert widget._combo_colormap_norm.currentData() == "centered"
        assert widget._spin_colormap_center.value() == 12.5
        assert widget._spin_colormap_center.isEnabled() is True

    def test_colormap_center_spinbox_disabled_unless_centered(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="wafer", x="X", y=["Y"], z="ampli_Si")
        widget = CustomizeMoreOptions(vg)
        widget.load_settings()
        assert widget._combo_colormap_norm.currentData() == "linear"
        assert widget._spin_colormap_center.isEnabled() is False

        idx = widget._combo_colormap_norm.findData("centered")
        widget._combo_colormap_norm.setCurrentIndex(idx)
        assert widget._spin_colormap_center.isEnabled() is True

    def test_apply_writes_colormap_settings_and_emits_them(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="wafer", x="X", y=["Y"], z="ampli_Si")
        widget = CustomizeMoreOptions(vg)
        idx = widget._combo_colormap_norm.findData("log")
        widget._combo_colormap_norm.setCurrentIndex(idx)

        received = []
        vg.properties_changed.connect(lambda gid, props: received.append(props))
        widget._apply()

        assert vg.colormap_norm == "log"
        assert received[0]["colormap_norm"] == "log"

    def test_apply_does_not_touch_colormap_settings_for_non_map_styles(self, qapp, excel_df):
        """Guards the plot_style in ('wafer','2Dmap') gate in _apply(): a
        non-map style must not overwrite colormap_norm/colormap_center from
        whatever the (hidden, stale) combo/spinbox happen to show."""
        vg = _plotted_graph(qapp, excel_df, plot_style="point", x="Zone")
        vg.colormap_norm = "centered"
        widget = CustomizeMoreOptions(vg)
        widget._apply()
        assert vg.colormap_norm == "centered"


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

    @pytest.mark.parametrize("method,expected_type", [
        ("_add_arrow", "arrow"),
        ("_add_vspan", "vspan"),
        ("_add_hspan", "hspan"),
        ("_add_box", "box"),
        ("_add_callout", "callout"),
    ])
    def test_add_new_annotation_types_appends_and_emits_signal(self, qapp, excel_df, method, expected_type):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAnnotations(vg)
        received = []
        vg.properties_changed.connect(lambda gid, props: received.append(props))

        getattr(widget, method)()

        assert len(vg.annotations) == 1
        assert vg.annotations[0]["type"] == expected_type
        assert received and "annotations" in received[-1]
        assert widget.annotation_list.count() == 1
        # List text must not fall through to the "Unknown type" branch.
        assert "Unknown type" not in widget.annotation_list.item(0).text()

    def test_add_all_new_types_together_produces_distinct_list_rows(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAnnotations(vg)
        for method in ("_add_arrow", "_add_vspan", "_add_hspan", "_add_box", "_add_callout"):
            getattr(widget, method)()

        assert {a["type"] for a in vg.annotations} == {"arrow", "vspan", "hspan", "box", "callout"}
        assert widget.annotation_list.count() == 5
        row_texts = [widget.annotation_list.item(i).text() for i in range(5)]
        assert len(set(row_texts)) == 5  # every row textually distinct

    @pytest.mark.parametrize("add_method,dialog_name,new_props", [
        ("_add_arrow", "EditArrowDialog", {"x1": 42.0, "y1": 1.0, "x2": 43.0, "y2": 2.0, "color": "#123456", "linestyle": "--", "linewidth": 2.0}),
        ("_add_vspan", "EditSpanDialog", {"x1": 10.0, "x2": 20.0, "color": "#654321", "alpha": 0.5}),
        ("_add_box", "EditBoxDialog", {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0, "facecolor": "#ABCDEF", "edgecolor": "#000000", "linewidth": 2.0, "alpha": 0.5}),
        ("_add_callout", "EditCalloutDialog", {"text": "edited", "x": 1.0, "y": 2.0, "tx": 3.0, "ty": 4.0, "fontsize": 14, "color": "#111111", "arrowcolor": "#222222"}),
    ])
    def test_edit_new_annotation_types_applies_dialog_properties(
        self, qapp, excel_df, monkeypatch, add_method, dialog_name, new_props
    ):
        import spectroview.view.components.customize_graph.customize_annotations as mod
        from PySide6.QtWidgets import QDialog

        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAnnotations(vg)
        getattr(widget, add_method)()

        dialog_cls = getattr(mod, dialog_name)
        monkeypatch.setattr(dialog_cls, "exec", lambda self: QDialog.Accepted)
        monkeypatch.setattr(dialog_cls, "get_properties", lambda self: dict(new_props))

        widget.annotation_list.setCurrentRow(0)
        widget._edit_annotation()

        for key, value in new_props.items():
            assert vg.annotations[0][key] == value

    def test_delete_new_annotation_type_removes_it(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAnnotations(vg)
        widget._add_box()

        widget.annotation_list.setCurrentRow(0)
        widget._delete_annotation()

        assert vg.annotations == []
        assert widget.annotation_list.count() == 0

    def test_double_click_list_row_opens_edit_dialog(self, qapp, excel_df, monkeypatch):
        """Double-clicking a list item routes to _edit_annotation for that row."""
        vg = _plotted_graph(qapp, excel_df)
        widget = CustomizeAnnotations(vg)
        widget._add_box()

        edited = []
        monkeypatch.setattr(widget, "_edit_annotation", lambda: edited.append(True))
        widget._on_item_double_clicked(widget.annotation_list.item(0))

        assert widget.annotation_list.currentItem() is widget.annotation_list.item(0)
        assert edited == [True]


def _event_at(px, py):
    """A minimal stand-in for a matplotlib MouseEvent: the handle/resize
    logic only reads .x/.y (display pixels)."""
    import types
    return types.SimpleNamespace(x=px, y=py)


def _artist_for(vg, ann_id):
    for a in vg.ax.findobj():
        data = getattr(a, "_annotation_data", None)
        if data and data.get("id") == ann_id:
            return a
    return None


def _view_box(vg):
    """(cx, cy, wx, wy) from the graph's current axis view -- annotations
    must be sized as a real fraction of the visible data so their edges land
    well apart in pixels (the real dataset is only ~1 x-unit wide)."""
    x0, x1 = vg.ax.get_xlim()
    y0, y1 = vg.ax.get_ylim()
    return (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)


class TestAnnotationCanvasResize:
    """Part A: border/endpoint grabbing so spans/boxes resize, arrow
    endpoints and the callout anchor move, instead of only whole-shape drags."""

    def test_box_border_handles_and_resize(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        cx, cy, wx, wy = _view_box(vg)
        x0, y0, w, h = cx - 0.2 * wx, cy - 0.2 * wy, 0.4 * wx, 0.4 * wy
        box = {"id": "b", "type": "box", "x": x0, "y": y0, "width": w, "height": h,
               "facecolor": "yellow", "edgecolor": "black", "linewidth": 1.5, "alpha": 0.3}
        vg.annotations = [box]
        vg.ax.clear(); vg.plot(vg.df)

        t = vg.ax.transData.transform
        left, bottom = t((x0, y0)); right, top = t((x0 + w, y0 + h))
        artist = _artist_for(vg, "b")
        assert vg._annotation_handle_at(artist, box, _event_at(right, (bottom + top) / 2)) == "resize-r"
        assert vg._annotation_handle_at(artist, box, _event_at(left, top)) == "resize-tl"
        assert vg._annotation_handle_at(artist, box, _event_at((left + right) / 2, (bottom + top) / 2)) == "move"

        vg._drag_handle = "resize-r"
        vg._resize_dragged_annotation(artist, box, x0 + 2 * w, 0.0)
        assert box["x"] == x0 and abs(box["width"] - 2 * w) < 1e-6

    def test_box_resize_cannot_collapse_below_min_extent(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        cx, cy, wx, wy = _view_box(vg)
        box = {"id": "b", "type": "box", "x": cx - 0.2 * wx, "y": cy - 0.2 * wy,
               "width": 0.4 * wx, "height": 0.4 * wy, "facecolor": "yellow",
               "edgecolor": "black", "linewidth": 1.5, "alpha": 0.3}
        vg.annotations = [box]
        vg.ax.clear(); vg.plot(vg.df)
        # Drag the right edge far to the left, past the left edge.
        vg._drag_handle = "resize-r"
        vg._resize_dragged_annotation(_artist_for(vg, "b"), box, cx - 100 * wx, 0.0)
        assert box["width"] > 0  # clamped, not inverted/collapsed

    def test_vspan_edge_handles_and_resize(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        cx, cy, wx, wy = _view_box(vg)
        x1v, x2v = cx - 0.2 * wx, cx + 0.2 * wx
        vs = {"id": "v", "type": "vspan", "x1": x1v, "x2": x2v, "color": "orange", "alpha": 0.3}
        vg.annotations = [vs]
        vg.ax.clear(); vg.plot(vg.df)
        t = vg.ax.transData.transform
        py = t((0, vg.ax.get_ylim()[0]))[1]
        artist = _artist_for(vg, "v")
        assert vg._annotation_handle_at(artist, vs, _event_at(t((x1v, 0))[0], py)) == "resize-x1"
        assert vg._annotation_handle_at(artist, vs, _event_at(t((x2v, 0))[0], py)) == "resize-x2"

        vg._drag_handle = "resize-x2"
        new_x2 = cx + 0.35 * wx
        vg._resize_dragged_annotation(artist, vs, new_x2, 0.0)
        assert abs(vs["x2"] - new_x2) < 1e-6 and vs["x1"] == x1v

    def test_arrow_endpoint_and_callout_anchor_move(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        cx, cy, wx, wy = _view_box(vg)
        p1 = (cx - 0.2 * wx, cy - 0.2 * wy)
        p2 = (cx + 0.2 * wx, cy + 0.2 * wy)
        anchor = (cx, cy)
        text = (cx + 0.3 * wx, cy + 0.3 * wy)
        ar = {"id": "a", "type": "arrow", "x1": p1[0], "y1": p1[1], "x2": p2[0], "y2": p2[1],
              "color": "black", "linewidth": 1.5, "linestyle": "-"}
        co = {"id": "c", "type": "callout", "x": anchor[0], "y": anchor[1],
              "tx": text[0], "ty": text[1], "text": "hi", "fontsize": 11,
              "color": "black", "arrowcolor": "black"}
        vg.annotations = [ar, co]
        vg.ax.clear(); vg.plot(vg.df)
        t = vg.ax.transData.transform

        assert vg._annotation_handle_at(_artist_for(vg, "a"), ar, _event_at(*t(p2))) == "p2"
        vg._drag_handle = "p2"
        new_p2 = (cx + 0.4 * wx, cy + 0.4 * wy)
        vg._resize_dragged_annotation(_artist_for(vg, "a"), ar, new_p2[0], new_p2[1])
        assert (ar["x2"], ar["y2"]) == new_p2 and (ar["x1"], ar["y1"]) == p1

        assert vg._annotation_handle_at(_artist_for(vg, "c"), co, _event_at(*t(anchor))) == "anchor"
        vg._drag_handle = "anchor"
        new_anchor = (cx + 0.1 * wx, cy + 0.1 * wy)
        vg._resize_dragged_annotation(_artist_for(vg, "c"), co, new_anchor[0], new_anchor[1])
        assert (co["x"], co["y"]) == new_anchor and (co["tx"], co["ty"]) == text


class TestAnnotationLivePreview:
    """Part C: edit dialogs preview live and restore on Cancel, coordinate
    spinboxes take an axis-derived range, spans get a synced range slider."""

    def _span(self):
        return {"id": "v", "type": "vspan", "x1": 3.0, "x2": 6.0, "color": "orange", "alpha": 0.3}

    def test_span_dialog_has_axis_range_and_synced_slider(self, qapp, excel_df):
        from spectroview.view.components.customize_graph.customize_annotation_dialogs import EditSpanDialog
        vg = _plotted_graph(qapp, excel_df)
        span = self._span()
        vg.annotations = [span]; vg.ax.clear(); vg.plot(vg.df)
        d = EditSpanDialog(span, vg, None)

        assert d.start_spin.minimum() > -999999  # data-relative, not the raw wide range
        assert d.range_slider is not None
        # Slider drag mirrors into the spinboxes.
        d._on_slider_changed((4.0, 5.5))
        assert d.start_spin.value() == 4.0 and d.end_spin.value() == 5.5

    def test_preview_mutates_then_cancel_restores(self, qapp, excel_df):
        from spectroview.view.components.customize_graph.customize_annotation_dialogs import EditSpanDialog
        vg = _plotted_graph(qapp, excel_df)
        span = self._span()
        vg.annotations = [span]; vg.ax.clear(); vg.plot(vg.df)
        d = EditSpanDialog(span, vg, None)

        d.start_spin.setValue(3.5)
        d._apply_preview()
        assert span["x1"] == 3.5  # previewed onto the live annotation dict

        d.reject()
        assert span["x1"] == 3.0  # original restored on Cancel

    def test_accept_keeps_edited_values(self, qapp, excel_df, monkeypatch):
        from PySide6.QtWidgets import QDialog
        from spectroview.view.components.customize_graph.customize_annotation_dialogs import EditSpanDialog
        vg = _plotted_graph(qapp, excel_df)
        span = self._span()
        vg.annotations = [span]; vg.ax.clear(); vg.plot(vg.df)
        widget = CustomizeAnnotations(vg)
        widget.load_annotations()

        monkeypatch.setattr(EditSpanDialog, "exec", lambda self: (self.start_spin.setValue(4.2), QDialog.Accepted)[1])
        widget.annotation_list.setCurrentRow(0)
        widget._edit_annotation()
        assert span["x1"] == 4.2

    def test_dialog_without_graph_widget_is_apply_only(self, qapp):
        """No graph attached -> no live preview, no crash (backward compat)."""
        from spectroview.view.components.customize_graph.customize_annotation_dialogs import EditBoxDialog
        box = {"id": "b", "type": "box", "x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0,
               "facecolor": "yellow", "edgecolor": "black", "linewidth": 1.5, "alpha": 0.3}
        d = EditBoxDialog(box, None, None)
        d.width_spin.setValue(3.0)
        d._schedule_preview()  # must be a no-op without a graph widget
        assert box["width"] == 1.0  # unchanged until explicit get_properties()
        assert d.get_properties()["width"] == 3.0


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
        assert dialog.annotations_widget.graph_widget is vg2
        assert dialog.more_options_widget.graph_widget is vg2

    def test_switch_graph_to_same_id_is_noop(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        dialog = CustomizeGraphDialog(vg, graph_id=1)
        dialog.switch_graph(vg, graph_id=1)
        assert dialog.graph_widget is vg


class TestLivePreview:
    """Phase 5D: every tab now previews live (debounced) instead of only
    the legend tab's per-series edits being live and everything else
    Apply-only. The debounce timer itself is real Qt-timer plumbing not
    worth testing with a real wait -- these tests call _preview_apply()
    directly (what the timer's timeout ultimately calls) and check the
    wiring that schedules it.

    A preview must stay purely visual: it must not commit to the
    ViewModel/undo stack (see graph_commit.py's snapshot() used for
    cancel_all()'s baseline) -- otherwise cancel_all() reverting only the
    widget would desync it from a ViewModel that still thinks the preview
    committed."""

    def test_changing_a_control_schedules_the_preview_timer(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        dialog = CustomizeGraphDialog(vg, graph_id=1)
        assert dialog._preview_timer.isActive() is False

        dialog.more_options_widget._spin_title_fontsize.setValue(20)

        assert dialog._preview_timer.isActive() is True

    def test_preview_apply_mutates_widget_without_committing(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        dialog = CustomizeGraphDialog(vg, graph_id=1)
        received = []
        vg.properties_changed.connect(lambda gid, props: received.append(props))

        dialog.more_options_widget._spin_title_fontsize.setValue(20)
        dialog._preview_apply()

        assert vg.title_fontsize == 20  # visual mutation happened
        assert received == []  # but nothing was committed

    def test_preview_apply_does_not_pop_a_warning_for_an_invalid_axis_break(
        self, qapp, excel_df, monkeypatch
    ):
        from PySide6.QtWidgets import QMessageBox
        warnings = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))

        vg = _plotted_graph(qapp, excel_df)
        dialog = CustomizeGraphDialog(vg, graph_id=1)
        dialog.axis_widget.x_break_enabled.setChecked(True)
        dialog.axis_widget.x_break_start.setValue(100)
        dialog.axis_widget.x_break_end.setValue(50)  # start >= end: invalid

        dialog._preview_apply()  # must not raise/block, must not warn

        assert warnings == []
        assert vg.axis_breaks['x'] is None  # left unset, not silently saved invalid

    def test_cancel_all_reverts_a_previewed_change(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        dialog = CustomizeGraphDialog(vg, graph_id=1)
        original_fontsize = vg.title_fontsize

        dialog.more_options_widget._spin_title_fontsize.setValue(20)
        dialog._preview_apply()
        assert vg.title_fontsize == 20

        dialog.cancel_all()

        assert vg.title_fontsize == original_fontsize
        assert dialog._preview_timer.isActive() is False

    def test_apply_all_refreshes_the_cancel_baseline(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        dialog = CustomizeGraphDialog(vg, graph_id=1)

        dialog.more_options_widget._spin_title_fontsize.setValue(20)
        dialog.apply_all()  # committed -- this is the new baseline

        dialog.more_options_widget._spin_title_fontsize.setValue(28)
        dialog._preview_apply()
        assert vg.title_fontsize == 28

        dialog.cancel_all()

        assert vg.title_fontsize == 20  # reverts to the Apply, not the original

    def test_switch_graph_stops_a_pending_preview_and_resets_the_baseline(
        self, qapp, excel_df
    ):
        vg1 = _plotted_graph(qapp, excel_df, plot_style="scatter")
        vg2 = _plotted_graph(qapp, excel_df, plot_style="bar", x="Zone", y=["ampli_Si"])
        dialog = CustomizeGraphDialog(vg1, graph_id=1)

        dialog.more_options_widget._spin_title_fontsize.setValue(20)
        assert dialog._preview_timer.isActive() is True

        dialog.switch_graph(vg2, graph_id=2)

        assert dialog._preview_timer.isActive() is False
        assert dialog._original_snapshot['title_fontsize'] == vg2.title_fontsize


class TestRestyleFastPath:
    """Phase 5E: _preview_apply() decides, per tick, between doing nothing
    (no field actually changed), VGraph.restyle() (every changed field is
    purely cosmetic -- see graph_style.RESTYLE_SAFE_FIELDS), or a single
    full replot (anything data-derived changed) -- instead of each of the
    four tabs replotting on its own. The artist-identity check (is the
    plotted collection/line the *same object* afterward) is how these
    tests tell "restyle()'s in-place repaint ran" apart from "a full
    replot recreated everything", since both leave the visible values
    looking correct either way."""

    def test_pure_cosmetic_change_uses_restyle_not_replot(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        dialog = CustomizeGraphDialog(vg, graph_id=1)
        artist_before = vg.ax.collections[0]

        dialog.more_options_widget._spin_title_fontsize.setValue(22)
        dialog._preview_apply()

        assert vg.title_fontsize == 22
        assert vg.ax.collections[0] is artist_before  # same artist: no replot

    def test_data_relevant_change_falls_back_to_full_replot(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        dialog = CustomizeGraphDialog(vg, graph_id=1)
        artist_before = vg.ax.collections[0]

        dialog.legend_widget.spin_scatter_size.setValue(333)
        dialog._preview_apply()

        assert vg.scatter_size == 333
        assert vg.ax.collections[0] is not artist_before  # replotted

    def test_mixed_cosmetic_and_data_change_falls_back_to_full_replot(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        dialog = CustomizeGraphDialog(vg, graph_id=1)
        artist_before = vg.ax.collections[0]

        dialog.more_options_widget._spin_title_fontsize.setValue(22)  # cosmetic
        dialog.legend_widget.spin_scatter_size.setValue(333)  # data-relevant
        dialog._preview_apply()

        assert vg.title_fontsize == 22
        assert vg.scatter_size == 333
        assert vg.ax.collections[0] is not artist_before  # one changed field forces a full replot

    def test_no_change_does_not_touch_the_canvas(self, qapp, excel_df, monkeypatch):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        dialog = CustomizeGraphDialog(vg, graph_id=1)
        draws = []
        monkeypatch.setattr(vg.canvas, "draw_idle", lambda: draws.append(1))

        dialog._preview_apply()  # nothing was edited since the dialog opened

        assert draws == []

    def test_preview_apply_does_not_emit_properties_changed_either_path(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        dialog = CustomizeGraphDialog(vg, graph_id=1)
        received = []
        vg.properties_changed.connect(lambda gid, props: received.append(props))

        dialog.more_options_widget._spin_title_fontsize.setValue(22)  # restyle() path
        dialog._preview_apply()
        dialog.legend_widget.spin_scatter_size.setValue(333)  # full-replot path
        dialog._preview_apply()

        assert received == []

    def test_trendline_equation_table_refreshes_after_a_replot_triggering_preview(
        self, qapp, excel_df
    ):
        vg = _plotted_graph(qapp, excel_df, plot_style="trendline")
        dialog = CustomizeGraphDialog(vg, graph_id=1)

        dialog.more_options_widget._spin_order.setValue(2)  # data-relevant: forces a replot
        dialog._preview_apply()

        equations = getattr(vg, 'trendline_equations', [])
        assert dialog.more_options_widget._eq_table.rowCount() == len(equations)

    def test_restyle_used_even_when_only_more_options_figure_fields_change(
        self, qapp, excel_df
    ):
        """title_fontsize is written by the More Options tab but is
        restyle-safe -- confirms the fast path isn't limited to the
        Axis/Legend tabs."""
        vg = _plotted_graph(qapp, excel_df, plot_style="scatter")
        dialog = CustomizeGraphDialog(vg, graph_id=1)
        artist_before = vg.ax.collections[0]

        dialog.more_options_widget._spin_title_fontsize.setValue(20)
        dialog._preview_apply()

        assert vg.title_fontsize == 20
        assert vg.ax.collections[0] is artist_before  # restyled, not replotted
