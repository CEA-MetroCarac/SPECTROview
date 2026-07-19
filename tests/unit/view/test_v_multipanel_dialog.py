"""Tests for view/components/v_multipanel_dialog.py - the multi-panel figure
composer (combine several open graphs into one exported figure).

Pure-function tests (panel_label/suggest_grid) need no Qt at all. The
dialog/composition tests use real, already-plotted VGraph widgets exactly
as VWorkspaceGraphs wires them up (see test_v_graph_plotting.py's own
docstring for why this codebase makes an exception to its usual
"View layer: not tested" policy for the Graph Workspace).
"""
import pandas as pd
import pytest

from spectroview.view.components.v_graph import VGraph
from spectroview.view.components.v_multipanel_dialog import (
    VMultiPanelDialog, panel_label, suggest_grid,
)


class TestPanelLabel:
    def test_lowercase_letters(self):
        assert [panel_label("a, b, c, ...", i) for i in range(4)] == ["a", "b", "c", "d"]

    def test_uppercase_letters(self):
        assert [panel_label("A, B, C, ...", i) for i in range(4)] == ["A", "B", "C", "D"]

    def test_roman_numerals(self):
        assert [panel_label("i, ii, iii, ...", i) for i in range(4)] == ["i", "ii", "iii", "iv"]

    def test_arabic_numerals(self):
        assert [panel_label("1, 2, 3, ...", i) for i in range(4)] == ["1", "2", "3", "4"]

    def test_unknown_style_falls_back_to_lowercase(self):
        assert panel_label("not a real style", 0) == "a"

    def test_wraps_past_26_letters_without_raising(self):
        assert panel_label("a, b, c, ...", 26) == "a1"
        assert panel_label("A, B, C, ...", 27) == "A2"


class TestSuggestGrid:
    @pytest.mark.parametrize("n,expected", [
        (0, (1, 1)), (1, (1, 1)), (2, (1, 2)), (3, (2, 2)),
        (4, (2, 2)), (5, (2, 3)), (6, (2, 3)), (9, (3, 3)),
    ])
    def test_grid_fits_and_is_roughly_square(self, n, expected):
        assert suggest_grid(n) == expected

    @pytest.mark.parametrize("n", range(1, 20))
    def test_grid_always_has_enough_cells(self, n):
        rows, cols = suggest_grid(n)
        assert rows * cols >= n


@pytest.fixture(scope="module")
def excel_df(dataframe_excel_file):
    if not dataframe_excel_file.exists():
        pytest.skip("dataset_Excel.xlsx not present")
    return pd.read_excel(dataframe_excel_file, sheet_name="sheet1")


def _plotted_graph(graph_id, excel_df, plot_style="scatter", x="x0_Si", y=None, z=None):
    vg = VGraph(graph_id=graph_id)
    vg.create_plot_widget(dpi=72)
    vg.df_name = "sheet1"
    vg.x = x
    vg.y = y if y is not None else ["ampli_Si"]
    vg.z = z
    vg.plot_style = plot_style
    vg.plot(excel_df)
    return vg


class TestVMultiPanelDialogUI:
    def test_populate_graph_list_shows_all_checked_by_default(self, qapp, excel_df):
        widgets = {1: _plotted_graph(1, excel_df), 2: _plotted_graph(2, excel_df, x="Zone")}
        dialog = VMultiPanelDialog(widgets)
        assert dialog.graph_list.count() == 2
        assert dialog._checked_graph_ids_in_order() == [1, 2]

    def test_unchecking_excludes_from_order(self, qapp, excel_df):
        widgets = {1: _plotted_graph(1, excel_df), 2: _plotted_graph(2, excel_df, x="Zone")}
        dialog = VMultiPanelDialog(widgets)
        from PySide6.QtCore import Qt
        dialog.graph_list.item(0).setCheckState(Qt.Unchecked)
        assert dialog._checked_graph_ids_in_order() == [2]

    def test_move_down_then_up_reorders_list(self, qapp, excel_df):
        widgets = {1: _plotted_graph(1, excel_df), 2: _plotted_graph(2, excel_df, x="Zone")}
        dialog = VMultiPanelDialog(widgets)
        dialog.graph_list.setCurrentRow(0)
        dialog._move_selected_down()
        assert dialog._checked_graph_ids_in_order() == [2, 1]
        dialog._move_selected_up()
        assert dialog._checked_graph_ids_in_order() == [1, 2]

    def test_auto_grid_matches_checked_count(self, qapp, excel_df):
        widgets = {i: _plotted_graph(i, excel_df, x="Zone") for i in range(1, 5)}
        dialog = VMultiPanelDialog(widgets)
        assert (dialog.spin_rows.value(), dialog.spin_cols.value()) == suggest_grid(4)

        from PySide6.QtCore import Qt
        dialog.graph_list.item(0).setCheckState(Qt.Unchecked)
        dialog._auto_grid()
        assert (dialog.spin_rows.value(), dialog.spin_cols.value()) == suggest_grid(3)


class TestComposeFigure:
    def test_creates_one_axes_per_graph(self, qapp, excel_df):
        widgets = {
            1: _plotted_graph(1, excel_df, plot_style="scatter"),
            2: _plotted_graph(2, excel_df, plot_style="bar", x="Zone"),
            3: _plotted_graph(3, excel_df, plot_style="line"),
        }
        dialog = VMultiPanelDialog(widgets)
        composed = dialog._compose_figure()
        # No colorbars in this mix, so axes count == graph count exactly.
        assert len(composed.axes) == 3

    def test_raises_when_nothing_checked(self, qapp, excel_df):
        widgets = {1: _plotted_graph(1, excel_df)}
        dialog = VMultiPanelDialog(widgets)
        from PySide6.QtCore import Qt
        dialog.graph_list.item(0).setCheckState(Qt.Unchecked)
        with pytest.raises(ValueError):
            dialog._compose_figure()

    def test_grid_smaller_than_selection_silently_drops_overflow(self, qapp, excel_df):
        widgets = {i: _plotted_graph(i, excel_df, x="Zone") for i in range(1, 5)}
        dialog = VMultiPanelDialog(widgets)
        dialog.spin_rows.setValue(1)
        dialog.spin_cols.setValue(2)  # only 2 cells for 4 checked graphs
        composed = dialog._compose_figure()
        assert len(composed.axes) == 2

    def test_source_widgets_fully_restored_after_compose(self, qapp, excel_df):
        """The core safety property: composing must never leave a live
        graph widget pointed at the throwaway composed figure."""
        widgets = {
            1: _plotted_graph(1, excel_df, plot_style="scatter", z="Quadrant"),
            2: _plotted_graph(2, excel_df, plot_style="bar", x="Zone"),
        }
        original_ax = {gid: gw.ax for gid, gw in widgets.items()}
        original_figure = {gid: gw.figure for gid, gw in widgets.items()}
        original_canvas = {gid: gw.canvas for gid, gw in widgets.items()}

        dialog = VMultiPanelDialog(widgets)
        composed = dialog._compose_figure()

        for gid, gw in widgets.items():
            assert gw.ax is original_ax[gid]
            assert gw.figure is original_figure[gid]
            assert gw.canvas is original_canvas[gid]
            # And the restored ax must still belong to the ORIGINAL figure,
            # not the composed one.
            assert gw.ax not in composed.axes
            assert gw.ax.figure is gw.figure

    def test_shared_labels_suppress_interior_panel_labels(self, qapp, excel_df):
        widgets = {i: _plotted_graph(i, excel_df, x="Zone", plot_style="bar") for i in range(1, 5)}
        dialog = VMultiPanelDialog(widgets)
        dialog.spin_rows.setValue(2)
        dialog.spin_cols.setValue(2)
        dialog.cb_shared_labels.setChecked(True)
        composed = dialog._compose_figure()

        top_left, top_right, bottom_left, bottom_right = composed.axes[:4]
        # Top row (not last in its column) -> x label suppressed.
        assert top_left.get_xlabel() == ""
        assert top_right.get_xlabel() == ""
        # Bottom row -> x label kept.
        assert bottom_left.get_xlabel() != ""
        # Right column -> y label suppressed; left column kept.
        assert top_right.get_ylabel() == ""
        assert top_left.get_ylabel() != ""

    def test_shared_labels_off_keeps_every_panel_labeled(self, qapp, excel_df):
        widgets = {i: _plotted_graph(i, excel_df, x="Zone", plot_style="bar") for i in range(1, 5)}
        dialog = VMultiPanelDialog(widgets)
        dialog.spin_rows.setValue(2)
        dialog.spin_cols.setValue(2)
        dialog.cb_shared_labels.setChecked(False)
        composed = dialog._compose_figure()
        assert all(ax.get_xlabel() != "" for ax in composed.axes[:4])

    def test_panel_labels_added_when_enabled(self, qapp, excel_df):
        widgets = {1: _plotted_graph(1, excel_df), 2: _plotted_graph(2, excel_df, x="Zone")}
        dialog = VMultiPanelDialog(widgets)
        dialog.cb_panel_labels.setChecked(True)
        composed = dialog._compose_figure()
        all_texts = [t.get_text() for ax in composed.axes for t in ax.texts]
        assert "a" in all_texts and "b" in all_texts

    def test_panel_labels_omitted_when_disabled(self, qapp, excel_df):
        widgets = {1: _plotted_graph(1, excel_df), 2: _plotted_graph(2, excel_df, x="Zone")}
        dialog = VMultiPanelDialog(widgets)
        dialog.cb_panel_labels.setChecked(False)
        composed = dialog._compose_figure()
        all_texts = [t.get_text() for ax in composed.axes for t in ax.texts]
        assert "a" not in all_texts

    def test_graph_with_active_broken_axis_composes_without_corrupting_live_panel(self, qapp, excel_df):
        """Regression guard for the trickiest edge case in this feature: a
        source graph with its own active broken axis has TWO live Axes
        (self.ax + self.ax_break_secondary). Composing it into a single
        grid cell must not remove/repurpose the live secondary panel --
        _render_graph_onto() takes a deliberately simplified path
        (_render_series_on(), not the full two-panel plot()) specifically
        to avoid that."""
        vg = _plotted_graph(1, excel_df, plot_style="scatter")
        vg.axis_breaks = {"x": {"start": 514.8, "end": 514.9}, "y": None}
        vg.plot(excel_df)
        assert vg._current_break_mode == "x"
        live_primary, live_secondary = vg.ax, vg.ax_break_secondary

        widgets = {1: vg, 2: _plotted_graph(2, excel_df, x="Zone", plot_style="bar")}
        dialog = VMultiPanelDialog(widgets)
        composed = dialog._compose_figure()  # must not raise

        # The live broken-axis graph's own two panels are completely
        # untouched -- same objects, still both attached to the real figure.
        assert vg.ax is live_primary
        assert vg.ax_break_secondary is live_secondary
        assert vg._current_break_mode == "x"
        assert live_primary in vg.figure.axes
        assert live_secondary in vg.figure.axes
        assert live_primary not in composed.axes
        assert live_secondary not in composed.axes

    def test_export_uses_configured_physical_size(self, qapp, excel_df):
        widgets = {1: _plotted_graph(1, excel_df)}
        dialog = VMultiPanelDialog(widgets)
        dialog.spin_fig_width.setValue(100.0)
        dialog.spin_fig_height.setValue(80.0)
        composed = dialog._compose_figure()
        from spectroview.view.components.v_multipanel_dialog import _MM_PER_INCH
        w_in, h_in = composed.get_size_inches()
        assert w_in == pytest.approx(100.0 / _MM_PER_INCH, abs=0.01)
        assert h_in == pytest.approx(80.0 / _MM_PER_INCH, abs=0.01)
