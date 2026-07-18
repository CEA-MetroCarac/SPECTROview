"""Tests for view/v_workspace_graphs.py -- specifically _build_graph_widget(),
the shared graph-widget-construction sequence used by _create_and_display_plot,
_on_plot_multi_wafer, and load_workspace.

Per tests/GRAPHS_WORKSPACE_TESTING.md, VWorkspaceGraphs itself was previously
untested directly -- only the VGraph/MGraph/VMWorkspaceGraphs/
CustomizeGraphDialog objects it wires together were, with the integration
test manually re-implementing the wiring pattern rather than calling the real
coordinator method. This file closes that gap for the one piece of
coordinator logic that's non-trivial enough to deserve direct coverage: the
widget-build sequence three call sites used to duplicate, and its
success/failure contract.
"""
import pandas as pd
import pytest
from PySide6.QtWidgets import QMessageBox

from spectroview.view.components.v_graph import VGraph
from spectroview.view.v_workspace_graphs import VWorkspaceGraphs


@pytest.fixture
def ws(qapp):
    return VWorkspaceGraphs()


@pytest.fixture(scope="module")
def excel_df(dataframe_excel_file):
    if not dataframe_excel_file.exists():
        pytest.skip("dataset_Excel.xlsx not present")
    return pd.read_excel(dataframe_excel_file, sheet_name="sheet1")


class TestBuildGraphWidget:
    def test_success_registers_widget_and_subwindow(self, ws, excel_df):
        graph_model = ws.vm.create_graph({
            'df_name': 'sheet1', 'plot_style': 'scatter',
            'x': 'x0_Si', 'y': ['ampli_Si'],
        })
        errors = []

        result = ws._build_graph_widget(graph_model, excel_df, errors.append)

        assert result is not None
        assert not errors
        assert graph_model.graph_id in ws.graph_widgets
        widget, dialog, sub_window = ws.graph_widgets[graph_model.graph_id]
        assert widget is result
        assert sub_window in ws.mdi_area.subWindowList()
        # legend_properties are synced back onto the model after a successful render
        updated = ws.vm.get_graph(graph_model.graph_id)
        assert updated.legend_properties == result.legend_properties

    def test_render_failure_deletes_graph_and_reports_without_registering(self, ws, excel_df):
        # trendline fitting requires a numeric X; 'Zone' is categorical, so
        # PlotRenderer._fit_trendline raises ValueError when astype(float)'d.
        graph_model = ws.vm.create_graph({
            'df_name': 'sheet1', 'plot_style': 'trendline',
            'x': 'Zone', 'y': ['ampli_Si'],
        })
        errors = []

        result = ws._build_graph_widget(graph_model, excel_df, errors.append)

        assert result is None
        assert len(errors) == 1
        assert isinstance(errors[0], Exception)
        assert graph_model.graph_id not in ws.graph_widgets
        # The failed graph is removed from the ViewModel too, matching the
        # pre-existing behavior of the single-plot and multi-wafer call sites.
        assert ws.vm.get_graph(graph_model.graph_id) is None

    def test_properties_changed_signal_is_wired_to_the_viewmodel(self, ws, excel_df):
        graph_model = ws.vm.create_graph({
            'df_name': 'sheet1', 'plot_style': 'scatter',
            'x': 'x0_Si', 'y': ['ampli_Si'],
        })
        widget = ws._build_graph_widget(graph_model, excel_df, lambda e: None)

        widget.properties_changed.emit(widget.graph_id, {'plot_title': 'Changed via signal'})

        assert ws.vm.get_graph(widget.graph_id).plot_title == 'Changed via signal'

    def test_toggling_toolbar_grid_checkbox_persists_to_the_model(self, ws, excel_df):
        """Regression test: the toolbar's grid checkbox was fully built in
        the UI but never connected in setup_connections() -- toggling it
        changed the visual grid but never reached the model (so it was lost
        on the next "Update plot" and never survived a save)."""
        graph_model = ws.vm.create_graph({
            'df_name': 'sheet1', 'plot_style': 'scatter',
            'x': 'x0_Si', 'y': ['ampli_Si'],
        })
        ws._build_graph_widget(graph_model, excel_df, lambda e: None)
        ws._update_graph_list(ws.vm.get_graph_ids())

        assert ws.vm.get_graph(graph_model.graph_id).grid is False
        ws.cb_grid_toolbar.setChecked(True)

        assert ws.vm.get_graph(graph_model.graph_id).grid is True

    def test_replicate_and_customize_signals_are_connected(self, ws, excel_df, monkeypatch):
        """Regression guard: both signals used to be wired identically at all
        three call sites -- confirm the shared helper still connects them.
        Patches the handlers at the class level *before* building the widget
        so the real connect() picks up the stub (an already-connected Qt
        slot keeps referencing the original bound method, so patching the
        instance attribute afterward would silently miss it)."""
        replicate_calls = []
        customize_calls = []
        monkeypatch.setattr(VWorkspaceGraphs, '_on_replicate_graph',
                             lambda self, gid: replicate_calls.append(gid))
        monkeypatch.setattr(VWorkspaceGraphs, '_show_or_switch_customize_dialog',
                             lambda self, gid: customize_calls.append(gid))

        graph_model = ws.vm.create_graph({
            'df_name': 'sheet1', 'plot_style': 'scatter',
            'x': 'x0_Si', 'y': ['ampli_Si'],
        })
        widget = ws._build_graph_widget(graph_model, excel_df, lambda e: None)

        widget.replicate_requested.emit(widget.graph_id)
        widget.customize_requested.emit(widget.graph_id)

        assert replicate_calls == [widget.graph_id]
        assert customize_calls == [widget.graph_id]


class TestLoadWorkspaceSkipsUnrenderableGraphs:
    def test_one_bad_graph_does_not_abort_loading_the_rest(self, ws, excel_df, monkeypatch):
        """Regression test for a real bug: load_workspace's per-graph loop had
        no try/except around rendering, so one corrupt/unrenderable saved
        graph aborted the entire workspace load instead of just being skipped."""
        ws.vm.add_dataframe('sheet1', excel_df)

        good_model = ws.vm.create_graph({
            'df_name': 'sheet1', 'plot_style': 'scatter', 'x': 'x0_Si', 'y': ['ampli_Si'],
        })
        bad_model = ws.vm.create_graph({
            'df_name': 'sheet1', 'plot_style': 'trendline', 'x': 'Zone', 'y': ['ampli_Si'],
        })
        another_good_model = ws.vm.create_graph({
            'df_name': 'sheet1', 'plot_style': 'scatter', 'x': 'x0_Si', 'y': ['fwhm_Si'],
        })

        notifications = []
        ws.vm.notify.connect(notifications.append)

        # Exercise load_workspace's own recreation loop directly (it always
        # calls self.vm.load_workspace(file_path) first, which would replace
        # our in-memory graphs -- so we drive just the widget-recreation half
        # of the method here rather than round-tripping through a real file).
        for graph_id in ws.vm.get_graph_ids():
            graph_model = ws.vm.get_graph(graph_id)
            filtered_df = ws.vm.apply_filters(graph_model.df_name, graph_model.filters)

            def _on_render_error(e, graph_model=graph_model):
                ws.vm.notify.emit(f"Skipped graph {graph_model.graph_id}: could not render ({e})")

            ws._build_graph_widget(graph_model, filtered_df, _on_render_error)

        assert good_model.graph_id in ws.graph_widgets
        assert another_good_model.graph_id in ws.graph_widgets
        assert bad_model.graph_id not in ws.graph_widgets
        assert any("skipped" in n.lower() for n in notifications)


class TestConfigureGraphFromModel:
    """_configure_graph_from_model was collapsed from ~60 hand-copied lines
    to a generic vars(model) loop plus a short override list. This class is
    the regression check for that refactor, and documents a real bug found
    while doing it (see test_axis_breaks_propagates_to_a_new_widget)."""

    def test_every_model_field_reaches_the_widget(self, ws):
        model = ws.vm.create_graph({
            'df_name': 'sheet1', 'plot_style': 'scatter', 'x': 'a', 'y': ['b'],
        })
        # Give every field a distinguishable non-default value so a field
        # that's silently dropped (left at the *widget's* __init__ default
        # instead of the model's value) shows up as a mismatch below.
        for key, current in list(vars(model).items()):
            if key == 'graph_id':
                continue
            if isinstance(current, bool):
                setattr(model, key, not current)
            elif isinstance(current, (int, float)):
                setattr(model, key, current + 1)
            elif isinstance(current, str):
                setattr(model, key, current + "_marker")
            elif isinstance(current, list):
                setattr(model, key, current + ["marker"])
            elif isinstance(current, dict):
                setattr(model, key, {**current, "marker_key": "marker_val"})
            elif current is None:
                setattr(model, key, "marker")

        widget = VGraph(graph_id=999)
        ws._configure_graph_from_model(widget, model)

        for key, value in vars(model).items():
            if key == 'graph_id':
                continue
            assert getattr(widget, key) == value, f"field {key!r} did not transfer to the widget"

    def test_axis_breaks_propagates_to_a_new_widget(self, ws):
        """Regression test: axis_breaks used to never flow model -> widget
        in _configure_graph_from_model (only ever the other way, widget ->
        model, in _on_update_plot/save_workspace) -- so a graph with a
        configured axis break silently lost it whenever a *new* widget was
        built from its model: on workspace reload, and on 'Replicate graph'."""
        model = ws.vm.create_graph({
            'df_name': 'sheet1', 'plot_style': 'scatter', 'x': 'a', 'y': ['b'],
        })
        model.axis_breaks = {'x': {'start': 1.0, 'end': 2.0}, 'y': None}

        widget = VGraph(graph_id=999)
        ws._configure_graph_from_model(widget, model)

        assert widget.axis_breaks == {'x': {'start': 1.0, 'end': 2.0}, 'y': None}

        # Must be an independent copy, not aliasing the model's dict --
        # otherwise dragging/editing a break on the widget before hitting
        # "Apply" would silently mutate the model too.
        widget.axis_breaks['x']['start'] = 999.0
        assert model.axis_breaks['x']['start'] == 1.0

    def test_y_and_legend_properties_are_independent_copies(self, ws):
        model = ws.vm.create_graph({
            'df_name': 'sheet1', 'plot_style': 'scatter', 'x': 'a', 'y': ['b'],
        })
        model.legend_properties = [{'label': 'A', 'marker': 'o', 'color': 'blue', 'rgba': [0, 0, 1, 1]}]

        widget = VGraph(graph_id=999)
        ws._configure_graph_from_model(widget, model)

        widget.y.append("mutated")
        widget.legend_properties.append({'label': 'B'})

        assert model.y == ['b']
        assert len(model.legend_properties) == 1

    def test_annotations_and_filters_are_independent_copies(self, ws):
        """Regression test: annotations and filters used to be aliased
        (not deep-copied) between model and widget in
        _configure_graph_from_model, so dragging an annotation or editing a
        filter on the widget could silently mutate the model before any
        "Update plot"/"Apply" commit happened."""
        model = ws.vm.create_graph({
            'df_name': 'sheet1', 'plot_style': 'scatter', 'x': 'a', 'y': ['b'],
        })
        model.annotations = [{'type': 'text', 'x': 1, 'y': 2, 'text': 'hi'}]
        model.filters = [{'expression': 'a > 0', 'state': True}]

        widget = VGraph(graph_id=999)
        ws._configure_graph_from_model(widget, model)

        widget.annotations.append({'type': 'vline', 'x': 5})
        widget.filters.append({'expression': 'b < 10', 'state': True})

        assert len(model.annotations) == 1
        assert len(model.filters) == 1

    def test_invalid_scatter_edgecolor_falls_back_to_black(self, ws):
        model = ws.vm.create_graph({
            'df_name': 'sheet1', 'plot_style': 'scatter', 'x': 'a', 'y': ['b'],
        })
        model.scatter_edgecolor = "none"

        widget = VGraph(graph_id=999)
        ws._configure_graph_from_model(widget, model)

        assert widget.scatter_edgecolor == 'black'


class TestTemplateApply:
    """_on_template_applied was revised (moved here from the AI Chat panel
    and the bottom-toolbar's lightweight VPlotTemplatePicker, now driven by
    the full VPlotTemplateDialog next to Add/Update plot): it always
    renders against the currently-selected DataFrame -- not each plot's
    originally-saved df_name -- and skips + reports (rather than aborting
    the whole batch) any plot whose required axis columns aren't present
    in that DataFrame."""

    def test_applies_plot_against_selected_df_ignoring_saved_df_name(self, ws, excel_df):
        ws.vm.add_dataframe('sheet1', excel_df)
        ws.vm.select_dataframe('sheet1')

        configs = [
            {'df_name': 'some_other_sheet', 'plot_style': 'scatter', 'x': 'x0_Si', 'y': ['ampli_Si']},
        ]
        ws._on_template_applied(configs)

        assert len(ws.graph_widgets) == 1
        gid = next(iter(ws.graph_widgets))
        assert ws.vm.get_graph(gid).df_name == 'sheet1'

    def test_skips_plot_with_missing_columns_and_reports_plot_number_and_columns(self, ws, excel_df, monkeypatch):
        ws.vm.add_dataframe('sheet1', excel_df)
        ws.vm.select_dataframe('sheet1')

        warnings = []
        monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: warnings.append(a))

        configs = [
            {'plot_style': 'scatter', 'x': 'x0_Si', 'y': ['ampli_Si']},            # ok
            {'plot_style': 'scatter', 'x': 'nonexistent_col', 'y': ['ampli_Si']},  # missing x
        ]
        ws._on_template_applied(configs)

        assert len(ws.graph_widgets) == 1  # only the first (valid) plot was created
        assert len(warnings) == 1
        message_text = warnings[0][2]
        assert "Plot 2" in message_text
        assert "nonexistent_col" in message_text

    def test_no_dataframe_selected_warns_and_creates_nothing(self, ws, monkeypatch):
        warnings = []
        monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: warnings.append(a))

        ws._on_template_applied([{'plot_style': 'scatter', 'x': 'a', 'y': ['b']}])

        assert len(warnings) == 1
        assert len(ws.graph_widgets) == 0

    def test_all_applied_emits_toast_not_a_blocking_dialog(self, ws, excel_df, monkeypatch):
        ws.vm.add_dataframe('sheet1', excel_df)
        ws.vm.select_dataframe('sheet1')

        warnings = []
        monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: warnings.append(a))
        notifications = []
        ws.vm.notify.connect(notifications.append)

        ws._on_template_applied([{'plot_style': 'scatter', 'x': 'x0_Si', 'y': ['ampli_Si']}])

        assert not warnings
        assert any("Applied" in n for n in notifications)

    def test_required_plot_columns_covers_all_axes(self, ws):
        cfg = {
            'x': 'colX', 'y': ['colY1', 'colY2'], 'z': 'colZ',
            'y2': 'colY2secondary', 'y3': None, 'x2': '',
        }
        assert ws._required_plot_columns(cfg) == {
            'colX', 'colY1', 'colY2', 'colZ', 'colY2secondary'
        }


class TestUpdatePlotPreservesDfName:
    """Regression coverage for a bug found while fixing B11: activating a
    plot's window correctly resyncs the side panel to that plot's own
    DataFrame (via _sync_gui_from_graph), but if the user then browses a
    *different* DataFrame in the side list without reactivating any window,
    mdi_area.activeSubWindow() stays on the original plot while
    vm.selected_df_name drifts. _collect_plot_config() always reads
    vm.selected_df_name, so "Update plot" used to silently rebind the
    active graph to whatever DataFrame the side panel had drifted to."""

    def test_browsing_a_different_df_without_reactivating_does_not_rebind_the_plot(
        self, ws, excel_df
    ):
        other_df = excel_df.copy()
        other_df['ampli_Si'] = 999.0
        ws.vm.add_dataframe('sheet1', excel_df)
        ws.vm.add_dataframe('sheet1_copy', other_df)
        ws.vm.select_dataframe('sheet1')

        graph_model = ws.vm.create_graph({
            'df_name': 'sheet1', 'plot_style': 'scatter',
            'x': 'x0_Si', 'y': ['ampli_Si'],
        })
        ws._build_graph_widget(graph_model, excel_df, lambda e: None)
        sub_window = ws.graph_widgets[graph_model.graph_id][2]

        # Activate the plot's window once (real click on the plot) --
        # resyncs the side panel to 'sheet1'.
        ws.mdi_area.setActiveSubWindow(sub_window)
        ws._on_subwindow_activated(sub_window)
        assert ws.vm.selected_df_name == 'sheet1'

        # Browse a different DataFrame in the side list WITHOUT touching
        # the MDI area -- the active subwindow doesn't change.
        ws.vm.select_dataframe('sheet1_copy')
        assert ws.mdi_area.activeSubWindow() is sub_window

        ws._on_update_plot()

        assert ws.vm.get_graph(graph_model.graph_id).df_name == 'sheet1'
