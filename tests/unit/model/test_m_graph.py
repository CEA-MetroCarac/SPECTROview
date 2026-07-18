"""Unit tests for model/m_graph.py - MGraph, the Graphs workspace's plot-config model.

Every field here mirrors a control somewhere in
spectroview/view/components/customize_graph/ (Axis / Legend-Color /
Annotations / More-options tabs). The guiding invariant under test is:
save() -> load() must round-trip every single customization property
unchanged, since that's exactly what happens on workspace save/reload and
on "replicate graph". Two real gaps were found and fixed while writing
these tests -- scatter_edgecolor and axis_breaks were set on the View
widget (VGraph) but never made it into MGraph.__init__/save()/load(), so
those two customizations silently didn't survive a save/reload. See
docs/log.md and tests/README.md for details.
"""
import pytest

from spectroview.model.m_graph import MGraph

# Every field MGraph.save() is expected to serialize. Used to make sure the
# round-trip test can never silently miss a newly-added field.
ALL_SAVE_KEYS = {
    'graph_id', 'grid', 'minor_ticks_bottom', 'minor_ticks_left', 'minor_ticks_top',
    'minor_ticks_right', 'filters', 'plot_style', 'plot_width', 'plot_height', 'dpi',
    'df_name', 'x', 'y', 'z', 'y2', 'y3', 'x2',
    'y2color', 'y2marker', 'y3color', 'y3marker', 'x2color', 'x2marker',
    'xmin', 'xmax', 'ymin', 'ymax', 'zmin',
    'zmax', 'y2min', 'y2max', 'y3min', 'y3max', 'x2min', 'x2max', 'xlogscale',
    'ylogscale', 'y2logscale', 'y3logscale', 'x2logscale', 'xscale_mode', 'yscale_mode',
    'plot_title', 'xlabel',
    'ylabel', 'zlabel', 'y2label', 'y3label', 'x2label', 'x_rot', 'legend_visible',
    'tick_direction', 'tick_label_format', 'x_inverted', 'y_inverted',
    'title_fontsize', 'axis_label_fontsize', 'tick_label_fontsize',
    'plot_subtitle', 'subtitle_fontsize', 'figure_facecolor', 'figure_margins', 'spines_visible',
    'figure_theme', 'export_width_mm', 'export_height_mm',
    'legend_ncol', 'legend_frame', 'legend_title', 'legend_fontsize', 'legend_alpha', 'legend_loc',
    'legend_outside', 'legend_properties', 'legend_bbox', 'color_palette', 'wafer_size',
    'wafer_stats', 'trendline_order', 'show_trendline_eq', 'trendline_anchor_enabled',
    'trendline_anchor_origin', 'trendline_anchor_x', 'trendline_anchor_y',
    'show_bar_plot_error_bar', 'error_bar_type', 'bar_error_bar_type', 'error_bar_capsize',
    'join_for_point_plot', 'dodge_point_plot',
    'dodge_scatter_plot', 'scatter_size', 'scatter_edgecolor', 'x_as_numeric',
    'y_as_numeric', 'hist_bins', 'hist_kde', 'hist_step', 'sort_data_enabled',
    'sort_data_by', 'annotations', 'axis_breaks',
}

# A representative, non-default value for every customizable field, keyed by
# name -- used to build a "fully customized" graph for the round-trip test
# without hand-writing 70 assignments in every test that needs one.
CUSTOM_VALUES = {
    'df_name': 'fit_results', 'filters': [{'expression': 'Slot == 2', 'state': True}],
    'plot_style': 'scatter', 'plot_width': 800, 'plot_height': 600, 'dpi': 150,
    'x': 'Slot', 'y': ['fwhm_Si', 'x0_Si'], 'z': 'Zone',
    'y2': 'ampli_Si', 'y3': 'area_Si', 'x2': 'DeltaW',
    'y2color': '#1F78B4', 'y2marker': 'D', 'y3color': '#FF7F00', 'y3marker': '^',
    'x2color': '#33A02C', 'x2marker': 'x',
    'xmin': -10.0, 'xmax': 100.0, 'ymin': 0.0, 'ymax': 50.0, 'zmin': 1.0, 'zmax': 9.0,
    'y2min': 2.0, 'y2max': 20.0, 'y3min': 3.0, 'y3max': 30.0, 'x2min': 4.0, 'x2max': 40.0,
    'xlogscale': True, 'ylogscale': True, 'y2logscale': True, 'y3logscale': True,
    'x2logscale': True, 'xscale_mode': 'symlog', 'yscale_mode': 'symlog',
    'plot_title': 'My Title', 'xlabel': 'X Label', 'ylabel': 'Y Label',
    'zlabel': 'Z Label', 'y2label': 'Y2 Label', 'y3label': 'Y3 Label', 'x2label': 'X2 Label',
    'x_rot': 45, 'grid': True,
    'tick_direction': 'in', 'tick_label_format': '%.2f',
    'x_inverted': True, 'y_inverted': True,
    'title_fontsize': 16, 'axis_label_fontsize': 14, 'tick_label_fontsize': 11,
    'plot_subtitle': 'A subtitle', 'subtitle_fontsize': 9,
    'figure_facecolor': '#EEEEEE', 'figure_margins': [0.1, 0.15],
    'spines_visible': {'top': False, 'right': False, 'bottom': True, 'left': True},
    'figure_theme': 'dark', 'export_width_mm': 89.0, 'export_height_mm': 65.0,
    'legend_visible': False, 'legend_outside': True,
    'legend_ncol': 2, 'legend_frame': False, 'legend_title': 'Groups',
    'legend_fontsize': 8, 'legend_alpha': 0.3, 'legend_loc': 'upper left',
    'legend_properties': [{
        'label': 'A', 'marker': 'o', 'color': '#E31A1C',
        'linewidth': 2.5, 'alpha': 0.8, 'zorder': 5.0,
        'marker_size': 90, 'edge_color': '#00FF00',
    }],
    'legend_bbox': [0.5, 0.5],
    'color_palette': 'viridis', 'wafer_size': 200.0, 'wafer_stats': False,
    'trendline_order': 3, 'show_trendline_eq': False, 'trendline_anchor_enabled': True,
    'trendline_anchor_origin': False, 'trendline_anchor_x': 1.5, 'trendline_anchor_y': 2.5,
    'show_bar_plot_error_bar': True, 'error_bar_type': 'sem', 'bar_error_bar_type': 'ci95',
    'error_bar_capsize': 5.0,
    'join_for_point_plot': True, 'dodge_point_plot': False,
    'dodge_scatter_plot': True, 'scatter_size': 120, 'scatter_edgecolor': '#00FF00',
    'x_as_numeric': True, 'y_as_numeric': False,
    'hist_bins': 50, 'hist_kde': True, 'hist_step': True,
    'sort_data_enabled': False, 'sort_data_by': 'X',
    'annotations': [{'id': 'vline_1', 'type': 'vline', 'x': 5.0, 'color': 'red'}],
    'minor_ticks_bottom': False, 'minor_ticks_left': False,
    'minor_ticks_top': True, 'minor_ticks_right': True,
    'axis_breaks': {'x': {'start': 10.0, 'end': 20.0}, 'y': {'start': 1.0, 'end': 2.0}},
}


def _fully_customized_graph(graph_id=42):
    graph = MGraph(graph_id=graph_id)
    for key, value in CUSTOM_VALUES.items():
        setattr(graph, key, value)
    return graph


class TestMGraphInitialization:
    def test_create_graph_with_id(self):
        assert MGraph(graph_id=5).graph_id == 5

    def test_create_graph_without_id(self):
        assert MGraph().graph_id is None

    def test_default_properties(self):
        graph = MGraph()

        assert graph.df_name is None
        assert graph.filters == []
        assert graph.plot_style == "point"
        assert graph.plot_width == 480
        assert graph.plot_height == 420
        assert graph.dpi == 100

        assert graph.x is None
        assert graph.y == []
        assert graph.z is None
        assert graph.y2 is None
        assert graph.y3 is None

        assert graph.xmin is None
        assert graph.xmax is None
        assert graph.ymin is None
        assert graph.ymax is None

        assert graph.xlogscale is False
        assert graph.ylogscale is False

        assert graph.plot_title is None
        assert graph.xlabel is None
        assert graph.ylabel is None

        assert graph.legend_visible is True

        assert graph.color_palette == "jet"
        assert graph.wafer_size == 300.0
        assert graph.scatter_size == 70
        assert graph.scatter_edgecolor == "black"
        assert graph.axis_breaks == {'x': None, 'y': None}

        assert graph.minor_ticks_bottom is True
        assert graph.minor_ticks_left is True
        assert graph.minor_ticks_top is False
        assert graph.minor_ticks_right is False

        assert graph.sort_data_enabled is True
        assert graph.sort_data_by == "Z"
        assert graph.annotations == []

    def test_fresh_instance_save_never_raises(self):
        """Every field save() reads must exist on a bare MGraph() -- this
        would AttributeError if __init__ and save() ever drift apart."""
        MGraph().save()


class TestMGraphSaveLoad:
    def test_save_returns_dict_with_every_documented_key(self):
        data = MGraph(graph_id=1).save()
        assert isinstance(data, dict)
        assert set(data.keys()) == ALL_SAVE_KEYS

    def test_save_includes_all_properties(self):
        graph = MGraph(graph_id=10)
        graph.df_name = "test_df"
        graph.plot_style = "line"
        graph.x = "X_column"
        graph.y = ["Y1", "Y2"]
        graph.xlabel = "X Label"
        graph.ylabel = "Y Label"

        data = graph.save()
        assert data['df_name'] == "test_df"
        assert data['plot_style'] == "line"
        assert data['x'] == "X_column"
        assert data['y'] == ["Y1", "Y2"]
        assert data['xlabel'] == "X Label"
        assert data['ylabel'] == "Y Label"

    def test_load_from_dict(self):
        graph = MGraph()
        data = {
            'graph_id': 99, 'df_name': 'loaded_df', 'plot_style': 'bar',
            'x': 'Category', 'y': ['Value1', 'Value2'],
            'xlabel': 'Categories', 'ylabel': 'Values',
            'xmin': 0.0, 'xmax': 10.0, 'grid': True,
        }
        graph.load(data)

        assert graph.graph_id == 99
        assert graph.df_name == 'loaded_df'
        assert graph.plot_style == 'bar'
        assert graph.x == 'Category'
        assert graph.y == ['Value1', 'Value2']
        assert graph.xlabel == 'Categories'
        assert graph.ylabel == 'Values'
        assert graph.xmin == 0.0
        assert graph.xmax == 10.0
        assert graph.grid is True

    def test_save_load_roundtrip_every_customizable_field(self):
        """The exhaustive version of the round-trip test: every single field
        in CUSTOM_VALUES (mirroring every tab of CustomizeGraphDialog) must
        come back unchanged after save() -> load() into a fresh MGraph."""
        graph1 = _fully_customized_graph()
        data = graph1.save()

        graph2 = MGraph()
        graph2.load(data)

        assert graph2.graph_id == graph1.graph_id
        for key, expected in CUSTOM_VALUES.items():
            assert getattr(graph2, key) == expected, f"field '{key}' did not round-trip"

    def test_scatter_edgecolor_round_trips(self):
        """Regression test for the scatter_edgecolor persistence gap."""
        graph1 = MGraph(graph_id=1)
        graph1.scatter_edgecolor = "#123456"
        graph2 = MGraph()
        graph2.load(graph1.save())
        assert graph2.scatter_edgecolor == "#123456"

    def test_axis_breaks_round_trip(self):
        """Regression test for the axis_breaks ('Broken axis (beta)') persistence gap."""
        graph1 = MGraph(graph_id=1)
        graph1.axis_breaks = {'x': {'start': 5.0, 'end': 15.0}, 'y': None}
        graph2 = MGraph()
        graph2.load(graph1.save())
        assert graph2.axis_breaks == {'x': {'start': 5.0, 'end': 15.0}, 'y': None}

    def test_load_handles_none_limits(self):
        graph = MGraph()
        graph.load({'xmin': None, 'xmax': '', 'ymin': '5.5', 'ymax': None})
        assert graph.xmin is None
        assert graph.xmax is None
        assert graph.ymin == 5.5
        assert graph.ymax is None

    def test_load_converts_string_limits_to_float(self):
        graph = MGraph()
        graph.load({'xmin': '0.5', 'xmax': '10.5', 'ymax': '5.0'})
        assert graph.xmin == 0.5
        assert graph.xmax == 10.5
        assert graph.ymax == 5.0

    def test_load_ignores_unknown_keys(self):
        graph = MGraph()
        graph.load({'not_a_real_field': 123, 'x': 'X'})
        assert not hasattr(graph, 'not_a_real_field')
        assert graph.x == 'X'

    def test_load_x_as_numeric_false_becomes_none_for_backward_compat(self):
        graph = MGraph()
        graph.load({'x_as_numeric': False})
        assert graph.x_as_numeric is None

    def test_load_missing_annotations_still_ends_up_a_list(self):
        """Old .graphs files predating the annotations feature must not
        leave graph.annotations as None."""
        graph = MGraph()
        graph.annotations = None
        graph.load({'x': 'X'})
        assert graph.annotations == []

    def test_load_does_not_touch_fields_absent_from_data(self):
        graph = MGraph(graph_id=1)
        graph.color_palette = 'viridis'
        graph.load({'x': 'onlyX'})
        assert graph.color_palette == 'viridis'  # untouched, not reset to default


class TestMGraphDisplayName:
    def test_display_name_with_all_axes(self):
        graph = MGraph(graph_id=1)
        graph.plot_style = "line"
        graph.x = "Time"
        graph.y = ["Temperature"]
        graph.z = "Pressure"
        assert graph.get_display_name() == "1-line_plot: [Time] - [Temperature] - [Pressure]"

    def test_display_name_with_missing_axes(self):
        graph = MGraph(graph_id=5)
        graph.plot_style = "scatter"
        graph.x = "X"
        graph.y = []
        graph.z = None
        assert graph.get_display_name() == "5-scatter_plot: [X] - [None] - [None]"

    def test_display_name_with_default_values(self):
        assert MGraph(graph_id=0).get_display_name() == "0-point_plot: [None] - [None] - [None]"

    def test_display_name_with_multiple_y_values_shows_only_first(self):
        graph = MGraph(graph_id=3)
        graph.x = "X"
        graph.y = ["Y1", "Y2", "Y3"]
        assert graph.get_display_name() == "3-point_plot: [X] - [Y1] - [None]"


class TestMGraphProperties:
    def test_set_plot_configuration(self):
        graph = MGraph()
        graph.plot_width, graph.plot_height, graph.dpi = 800, 600, 150
        assert (graph.plot_width, graph.plot_height, graph.dpi) == (800, 600, 150)

    def test_set_axis_limits(self):
        graph = MGraph()
        graph.xmin, graph.xmax, graph.ymin, graph.ymax = -10.0, 10.0, 0.0, 100.0
        assert (graph.xmin, graph.xmax, graph.ymin, graph.ymax) == (-10.0, 10.0, 0.0, 100.0)

    def test_set_filters(self):
        graph = MGraph()
        graph.filters = [{"expression": "X > 5", "state": True}, {"expression": "Y < 100", "state": False}]
        assert len(graph.filters) == 2
