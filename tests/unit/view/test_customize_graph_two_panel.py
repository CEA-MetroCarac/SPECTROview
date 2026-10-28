# tests/unit/view/test_customize_graph_two_panel.py
"""Unit tests for the redesigned two-panel CustomizeGraphDialog with search and 7 modular tabs."""

import pandas as pd
import pytest
from PySide6.QtCore import Qt

from spectroview.model.m_graph import MGraph
from spectroview.view.components.customize_graph.customize_graph_dialog import (
    CustomizeGraphDialog,
)
from spectroview.view.components.v_graph import VGraph

_active_test_widgets = []


@pytest.fixture(autouse=True)
def _cleanup_test_widgets():
    yield
    while _active_test_widgets:
        w = _active_test_widgets.pop()
        try:
            if hasattr(w, "figure") and w.figure is not None:
                w.figure.clear()
            w.close()
            w.deleteLater()
        except Exception:
            pass


@pytest.fixture(scope="module")
def excel_df(dataframe_excel_file):
    if not dataframe_excel_file.exists():
        pytest.skip("dataset_Excel.xlsx not present")
    return pd.read_excel(dataframe_excel_file, sheet_name="sheet1")


def _make_graph(qapp, excel_df, plot_style="scatter"):
    model = MGraph(graph_id=1, df_name="sheet1", x="x0_Si", y=["ampli_Si"], z="Quadrant", plot_style=plot_style)
    vg = VGraph(graph_id=1)
    for key, value in vars(model).items():
        if key != "graph_id":
            setattr(vg, key, value)
    vg.create_plot_widget(dpi=72)
    vg.plot(excel_df)
    _active_test_widgets.append(vg)
    return vg


class TestCustomizeGraphTwoPanelUI:
    """Tests covering the new Settings-like two-panel layout, navigation, and proxy."""

    def test_sidebar_initializes_with_seven_categories(self, qapp, excel_df):
        vg = _make_graph(qapp, excel_df)
        dialog = CustomizeGraphDialog(vg, graph_id=1)
        assert dialog._nav_list.count() == 7
        assert dialog._stack.count() == 7

        expected_titles = [
            "General & Typography",
            "Axes & Limits",
            "Ticks & Spines",
            "Legend & Series",
            "Plot Style & Fitting",
            "Annotations",
            "Secondary & Inset Axes",
        ]
        actual_titles = [dialog._nav_list.item(i).text() for i in range(7)]
        assert actual_titles == expected_titles

    def test_navigation_row_change_switches_stacked_widget(self, qapp, excel_df):
        vg = _make_graph(qapp, excel_df)
        dialog = CustomizeGraphDialog(vg, graph_id=1)

        # Switch to Axes & Limits (index 1)
        dialog._nav_list.setCurrentRow(1)
        assert dialog._stack.currentIndex() == 1
        assert dialog._content_title.text() == "Axes & Limits"

        # Switch to Annotations (index 5)
        dialog._nav_list.setCurrentRow(5)
        assert dialog._stack.currentIndex() == 5
        assert dialog._content_title.text() == "Annotations"

        # Switch to Secondary & Inset Axes (index 6)
        dialog._nav_list.setCurrentRow(6)
        assert dialog._stack.currentIndex() == 6
        assert dialog._content_title.text() == "Secondary & Inset Axes"

    def test_select_section_shortcuts(self, qapp, excel_df):
        vg = _make_graph(qapp, excel_df)
        dialog = CustomizeGraphDialog(vg, graph_id=1)

        dialog.select_section("ticks")
        assert dialog._stack.currentWidget() is dialog.tab_ticks_spines

        dialog.select_section("secondary")
        assert dialog._stack.currentWidget() is dialog.tab_secondary_inset

        dialog.select_section("legend")
        assert dialog._stack.currentWidget() is dialog.tab_legend

        dialog.select_section("fitting")
        assert dialog._stack.currentWidget() is dialog.tab_plot_styles

        dialog.select_section("annotations")
        assert dialog._stack.currentWidget() is dialog.tab_annotations

        dialog.select_section("general")
        assert dialog._stack.currentWidget() is dialog.tab_general

    def test_tabs_compatibility_proxy(self, qapp, excel_df):
        vg = _make_graph(qapp, excel_df)
        dialog = CustomizeGraphDialog(vg, graph_id=1)

        assert dialog.tabs.count() == 7
        assert dialog.tabs.tabText(0) == "General & Typography"
        assert dialog.tabs.widget(0) is dialog.tab_general

        dialog.tabs.setCurrentIndex(2)
        assert dialog._stack.currentIndex() == 2
        assert dialog.tabs.currentIndex() == 2
        assert dialog.tabs.currentWidget() is dialog.tab_ticks_spines


class TestGlobalSearchAndFiltering:
    """Tests covering real-time search filtering across all categories and in-page highlighting."""

    def test_search_filters_navigation_list(self, qapp, excel_df):
        vg = _make_graph(qapp, excel_df)
        dialog = CustomizeGraphDialog(vg, graph_id=1)

        # Search for "broken" -> should only match Axes & Limits (index 1)
        dialog._search_edit.setText("broken")
        assert not dialog._nav_list.item(1).isHidden()
        assert dialog._nav_list.item(0).isHidden()
        assert dialog._nav_list.item(2).isHidden()
        assert dialog._nav_list.item(5).isHidden()
        assert dialog._stack.currentIndex() == 1

        # Search for "arrow" -> matches Annotations (index 5)
        dialog._search_edit.setText("arrow")
        assert not dialog._nav_list.item(5).isHidden()
        assert dialog._nav_list.item(1).isHidden()
        assert dialog._stack.currentIndex() == 5

        # Clear search -> all visible
        dialog._search_edit.setText("")
        for i in range(7):
            assert not dialog._nav_list.item(i).isHidden()

    def test_search_keywords_cover_all_categories(self, qapp, excel_df):
        vg = _make_graph(qapp, excel_df)
        dialog = CustomizeGraphDialog(vg, graph_id=1)

        queries_to_expected_index = [
            ("typography", 0),  # General & Typography
            ("symlog", 1),      # Axes & Limits
            ("spines", 2),      # Ticks & Spines
            ("palette", 3),     # Legend & Series
            ("trendline", 4),   # Plot Style & Fitting
            ("callout", 5),     # Annotations
            ("twin", 6),        # Secondary & Inset Axes
        ]

        for query, expected_idx in queries_to_expected_index:
            dialog._search_edit.setText(query)
            assert not dialog._nav_list.item(expected_idx).isHidden(), f"Query '{query}' did not match index {expected_idx}"

    def test_in_page_groupbox_highlighting(self, qapp, excel_df):
        vg = _make_graph(qapp, excel_df)
        dialog = CustomizeGraphDialog(vg, graph_id=1)

        # Tab General
        dialog.tab_general.highlight_search("theme")
        assert "border: 1.5px solid #3B82F6" in dialog.tab_general.grp_display.styleSheet()
        dialog.tab_general.highlight_search("")
        assert dialog.tab_general.grp_display.styleSheet() == ""

        # Tab Axes Limits
        dialog.tab_axes_limits.highlight_search("broken")
        assert "border: 1.5px solid #3B82F6" in dialog.tab_axes_limits.grp_break.styleSheet()
        dialog.tab_axes_limits.highlight_search("")
        assert dialog.tab_axes_limits.grp_break.styleSheet() == ""

        # Tab Ticks Spines
        dialog.tab_ticks_spines.highlight_search("minor")
        assert "border: 1.5px solid #3B82F6" in dialog.tab_ticks_spines.grp_minor.styleSheet()
        dialog.tab_ticks_spines.highlight_search("")
        assert dialog.tab_ticks_spines.grp_minor.styleSheet() == ""

        # Tab Secondary Inset
        dialog.tab_secondary_inset.highlight_search("inset")
        assert "border: 1.5px solid #3B82F6" in dialog.tab_secondary_inset.inset_group.styleSheet()
        dialog.tab_secondary_inset.highlight_search("")
        assert dialog.tab_secondary_inset.inset_group.styleSheet() == ""

        # Tab Legend
        dialog.tab_legend.highlight_search("marker")
        assert "border: 1.5px solid #3B82F6" in dialog.tab_legend.grp_markers.styleSheet()
        dialog.tab_legend.highlight_search("")
        assert dialog.tab_legend.grp_markers.styleSheet() == ""

        # Tab Plot Styles
        dialog.tab_plot_styles.highlight_search("error")
        assert "border: 1.5px solid #3B82F6" in dialog.tab_plot_styles.grp_error_bars.styleSheet()
        dialog.tab_plot_styles.highlight_search("")
        assert dialog.tab_plot_styles.grp_error_bars.styleSheet() == ""

        # Tab Annotations
        dialog.tab_annotations.highlight_search("arrow")
        assert "border: 1.5px solid #3B82F6" in dialog.tab_annotations.grp_add.styleSheet()
        dialog.tab_annotations.highlight_search("")
        assert dialog.tab_annotations.grp_add.styleSheet() == ""


class TestUnifiedErrorBarsAndOptions:
    """Tests covering unified error bar options and style settings."""

    def test_error_bars_in_plot_styles_tab(self, qapp, excel_df):
        vg = _make_graph(qapp, excel_df, plot_style="bar")
        dialog = CustomizeGraphDialog(vg, graph_id=1)

        tab = dialog.tab_plot_styles
        assert tab.grp_error_bars.isVisibleTo(tab) is True
        assert tab._cb_error_bar.isVisibleTo(tab) is True

        tab._cb_error_bar.setChecked(True)
        idx = tab.cbb_error_bar_type.findData("sem")
        tab.cbb_error_bar_type.setCurrentIndex(idx)
        tab.spin_error_bar_capsize.setValue(5.5)

        dialog.apply_all()

        assert vg.show_bar_plot_error_bar is True
        assert vg.bar_error_bar_type == "sem"
        assert vg.error_bar_capsize == 5.5

