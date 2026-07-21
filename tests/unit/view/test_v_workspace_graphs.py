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
from PySide6.QtCore import Qt
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import QMessageBox, QInputDialog

from spectroview.model.graph_style import default_style
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


class TestSharedGraphToolbar:
    """Every VGraph builds its own matplotlib nav toolbar + action-button
    row (toolbar_container) but never shows it itself -- only the currently
    active graph's row is reparented into the workspace's one shared
    toolbar slot (_sync_active_graph_toolbar), so all MDI windows visually
    share a single toolbar instead of each carrying an identical copy."""

    def _make_graph(self, ws, excel_df, **overrides):
        """Build a graph and explicitly activate its subwindow: the `ws`
        fixture is never .show()'n, so QMdiArea doesn't auto-activate a
        newly added subwindow the way it does in the real, on-screen app
        (matches the pattern already used by _on_update_plot()'s tests)."""
        cfg = {'df_name': 'sheet1', 'plot_style': 'scatter', 'x': 'x0_Si', 'y': ['ampli_Si']}
        cfg.update(overrides)
        graph_model = ws.vm.create_graph(cfg)
        widget = ws._build_graph_widget(graph_model, excel_df, lambda e: None)
        sub_window = ws.graph_widgets[widget.graph_id][2]
        ws.mdi_area.setActiveSubWindow(sub_window)
        return widget

    def test_new_graph_toolbar_is_not_shown_inside_its_own_window(self, ws, excel_df):
        widget = self._make_graph(ws, excel_df)
        assert widget.toolbar_container.parent() is not widget
        assert widget.toolbar_container.isVisible() is False

    def test_subwindow_has_no_default_qt_window_icon(self, ws, excel_df):
        """A transparent icon overrides Fusion's default app-logo fallback
        (a null icon alone still paints the generic Qt logo)."""
        widget = self._make_graph(ws, excel_df)
        sub_window = ws.graph_widgets[widget.graph_id][2]
        icon = sub_window.windowIcon()
        assert not icon.isNull()
        assert icon.pixmap(16, 16).toImage().pixelColor(8, 8).alpha() == 0

    def test_newly_added_graph_becomes_the_one_shown_in_the_shared_slot(self, ws, excel_df):
        widget = self._make_graph(ws, excel_df)
        assert widget.toolbar_container.parent() is ws.graph_toolbar_slot
        assert ws._graph_toolbar_slot_layout.count() == 1

    def test_switching_active_subwindow_moves_the_toolbar_to_the_new_graph(self, ws, excel_df):
        first = self._make_graph(ws, excel_df)
        second = self._make_graph(ws, excel_df, y=['fwhm_Si'])
        _, _, sw_first = ws.graph_widgets[first.graph_id]
        _, _, sw_second = ws.graph_widgets[second.graph_id]

        # Second graph was added last, so it's the active one by default.
        assert second.toolbar_container.parent() is ws.graph_toolbar_slot
        assert first.toolbar_container.parent() is not ws.graph_toolbar_slot

        ws.mdi_area.setActiveSubWindow(sw_first)

        assert first.toolbar_container.parent() is ws.graph_toolbar_slot
        assert second.toolbar_container.parent() is not ws.graph_toolbar_slot
        assert ws._graph_toolbar_slot_layout.count() == 1

    def test_deactivated_toolbar_is_explicitly_hidden_not_a_stray_window(self, ws, excel_df):
        """Regression: a deactivated toolbar_container must be *explicitly*
        hidden when it leaves the shared slot. Reparenting a visible widget
        to None only implicitly hides it, and on Windows that implicit hide
        does not survive the reparent -- the widget becomes a visible
        top-level window, so a multi-graph load spawned one stray floating
        toolbar window per inactive graph. WA_WState_ExplicitShowHide being
        set is what guarantees it stays hidden on every platform (the
        offscreen test platform hides it either way, so this attribute is the
        only cross-platform witness of the bug)."""
        first = self._make_graph(ws, excel_df)
        self._make_graph(ws, excel_df, y=['fwhm_Si'])  # deactivates `first`

        assert first.toolbar_container.parent() is None
        assert first.toolbar_container.isVisible() is False
        assert first.toolbar_container.testAttribute(Qt.WA_WState_ExplicitShowHide) is True

    def test_closing_the_active_graph_clears_the_slot(self, ws, excel_df):
        widget = self._make_graph(ws, excel_df)
        graph_id = widget.graph_id
        assert ws._graph_toolbar_slot_layout.count() == 1

        ws._on_graph_closed(graph_id)

        assert ws._graph_toolbar_slot_layout.count() == 0

    def test_update_plot_rebuilds_and_resyncs_the_active_graphs_toolbar(self, ws, excel_df):
        """create_plot_widget() (called again by Update Plot) builds a brand
        new toolbar_container -- the shared slot must pick up the new one,
        not keep displaying the stale, about-to-be-deleted instance."""
        widget = self._make_graph(ws, excel_df)
        old_container = widget.toolbar_container
        ws.vm.add_dataframe('sheet1', excel_df)

        ws.cbb_x.setCurrentText('x0_Si')
        ws.cbb_y.setCurrentText('fwhm_Si')
        ws._on_update_plot()

        assert widget.toolbar_container is not old_container
        assert widget.toolbar_container.parent() is ws.graph_toolbar_slot
        assert ws._graph_toolbar_slot_layout.count() == 1


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


class TestRecipeApply:
    """_on_recipe_applied was revised (moved here from the AI Chat panel
    and the bottom-toolbar's lightweight picker, now driven by the full
    VPlotRecipeDialog next to Add/Update plot): it always renders against
    the currently-selected DataFrame -- not each plot's originally-saved
    df_name -- and skips + reports (rather than aborting the whole batch)
    any plot whose required axis columns aren't present in that
    DataFrame."""

    def test_applies_plot_against_selected_df_ignoring_saved_df_name(self, ws, excel_df):
        ws.vm.add_dataframe('sheet1', excel_df)
        ws.vm.select_dataframe('sheet1')

        configs = [
            {'df_name': 'some_other_sheet', 'plot_style': 'scatter', 'x': 'x0_Si', 'y': ['ampli_Si']},
        ]
        ws._on_recipe_applied(configs)

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
        ws._on_recipe_applied(configs)

        assert len(ws.graph_widgets) == 1  # only the first (valid) plot was created
        assert len(warnings) == 1
        message_text = warnings[0][2]
        assert "Plot 2" in message_text
        assert "nonexistent_col" in message_text

    def test_no_dataframe_selected_warns_and_creates_nothing(self, ws, monkeypatch):
        warnings = []
        monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: warnings.append(a))

        ws._on_recipe_applied([{'plot_style': 'scatter', 'x': 'a', 'y': ['b']}])

        assert len(warnings) == 1
        assert len(ws.graph_widgets) == 0

    def test_all_applied_emits_toast_not_a_blocking_dialog(self, ws, excel_df, monkeypatch):
        ws.vm.add_dataframe('sheet1', excel_df)
        ws.vm.select_dataframe('sheet1')

        warnings = []
        monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: warnings.append(a))
        notifications = []
        ws.vm.notify.connect(notifications.append)

        ws._on_recipe_applied([{'plot_style': 'scatter', 'x': 'x0_Si', 'y': ['ampli_Si']}])

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


class TestStyleActions:
    """Dispatch for VGraph's per-graph "🎨 Style" menu (save/apply
    template, copy/paste, reset-to-default, set-as-default) -- see
    model/graph_style.py for the style/data field partition every branch
    here shares."""

    def _graph(self, ws, excel_df, **overrides):
        cfg = {
            'df_name': 'sheet1', 'plot_style': 'scatter',
            'x': 'x0_Si', 'y': ['ampli_Si'],
        }
        cfg.update(overrides)
        graph_model = ws.vm.create_graph(cfg)
        return ws._build_graph_widget(graph_model, excel_df, lambda e: None)

    def test_unknown_graph_id_is_noop(self, ws):
        ws._on_style_action_requested(999999, "copy")  # must not raise

    def test_copy_then_paste_applies_style(self, ws, excel_df):
        source = self._graph(ws, excel_df)
        source.color_palette = "viridis"
        source.grid = True
        target = self._graph(ws, excel_df)

        ws._on_style_action_requested(source.graph_id, "copy")
        ws._on_style_action_requested(target.graph_id, "paste")

        assert target.color_palette == "viridis"
        assert target.grid is True

    def test_paste_without_copy_shows_message(self, ws, excel_df, monkeypatch):
        infos = []
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: infos.append(a))
        gw = self._graph(ws, excel_df)

        ws._on_style_action_requested(gw.graph_id, "paste")

        assert len(infos) == 1

    def test_reset_to_default_restores_defaults(self, ws, excel_df):
        gw = self._graph(ws, excel_df)
        gw.color_palette = "viridis"
        gw.grid = True

        ws._on_style_action_requested(gw.graph_id, "reset")

        factory = default_style()
        assert gw.color_palette == factory["color_palette"]
        assert gw.grid == factory["grid"]

    def test_style_action_emits_properties_changed(self, ws, excel_df):
        source = self._graph(ws, excel_df)
        source.color_palette = "viridis"
        target = self._graph(ws, excel_df)
        ws._on_style_action_requested(source.graph_id, "copy")

        received = []
        target.properties_changed.connect(lambda gid, props: received.append(props))
        ws._on_style_action_requested(target.graph_id, "paste")

        assert len(received) == 1
        assert received[0]["color_palette"] == "viridis"

    def test_set_default_saves_the_graphs_style_to_settings(self, ws, excel_df):
        gw = self._graph(ws, excel_df)
        gw.color_palette = "magma"
        gw.grid = True

        ws._on_style_action_requested(gw.graph_id, "set_default")

        saved = ws.m_settings.get_default_graph_style()
        assert saved["color_palette"] == "magma"
        assert saved["grid"] is True

    def test_set_default_does_not_touch_reset_to_default_behavior(self, ws, excel_df):
        """Set as Default Style and Reset to Default are two independent
        concepts: Reset to Default always restores the hardcoded factory
        style regardless of what's been set as the default for *new*
        graphs."""
        gw = self._graph(ws, excel_df)
        gw.color_palette = "magma"
        ws._on_style_action_requested(gw.graph_id, "set_default")

        gw.color_palette = "viridis"
        ws._on_style_action_requested(gw.graph_id, "reset")

        assert gw.color_palette == default_style()["color_palette"]
        assert gw.color_palette != "magma"

    def test_save_style_template_without_folder_configured_warns(self, ws, excel_df, monkeypatch):
        gw = self._graph(ws, excel_df)
        monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("My Style", True)))
        warnings = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))

        ws._on_style_action_requested(gw.graph_id, "save_template")

        assert len(warnings) == 1

    def test_save_style_template_with_folder_configured_saves(self, ws, excel_df, tmp_path, monkeypatch):
        # Working Folder is the source of truth now, not the store attribute
        # directly -- _save_style_template() rebuilds self.style_template_store
        # from it on every call (that rebuild is the fix for the reported
        # "configured folder doesn't take effect" bug), so a test must go
        # through m_settings for the injected folder to actually stick.
        ws.m_settings.set_working_folder(str(tmp_path))
        gw = self._graph(ws, excel_df)
        gw.color_palette = "cool"
        monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("My Style", True)))

        ws._on_style_action_requested(gw.graph_id, "save_template")

        summaries = ws.style_template_store.list_templates()
        assert len(summaries) == 1
        assert summaries[0].name == "My Style"
        saved_style = ws.style_template_store.load_style(summaries[0].id)
        assert saved_style["color_palette"] == "cool"

    def test_save_style_template_cancelled_dialog_saves_nothing(self, ws, excel_df, tmp_path, monkeypatch):
        ws.m_settings.set_working_folder(str(tmp_path))
        gw = self._graph(ws, excel_df)
        monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("", False)))

        ws._on_style_action_requested(gw.graph_id, "save_template")

        assert ws.style_template_store.list_templates() == []

    def test_apply_style_template_dialog_applies_selected_style(self, ws, excel_df, monkeypatch):
        gw = self._graph(ws, excel_df)

        # VStyleTemplateDialog's real style_applied Signal must fire before
        # exec() blocks (a modal dialog in the real app) -- stub exec() to
        # simulate the user picking a style and clicking Apply.
        from spectroview.view.components.v_style_template_dialog import VStyleTemplateDialog

        def _fake_exec(self):
            self.style_applied.emit({"color_palette": "Spectral", "grid": True})

        monkeypatch.setattr(VStyleTemplateDialog, "exec", _fake_exec)

        ws._on_style_action_requested(gw.graph_id, "apply_template")

        assert gw.color_palette == "Spectral"
        assert gw.grid is True


class TestApplyDefaultStyleToConfig:
    """_apply_default_style_to_config() merges the user's "Set as Default
    Style" baseline under a freshly-collected plot_config for graphs being
    built from scratch (Add Plot / Add Multi-Wafer) -- separate from
    Reset to Default, which is always the hardcoded factory style."""

    def test_noop_when_no_default_configured(self, ws):
        ws.m_settings.clear_default_graph_style()
        cfg = {'df_name': 'sheet1', 'plot_style': 'scatter', 'x': 'x0_Si', 'y': ['ampli_Si']}
        original = dict(cfg)

        ws._apply_default_style_to_config(cfg)

        # Unrelated to the "Set as Default Style" merge this method mainly
        # exists for: it also always fills in a default spines_visible when
        # the config doesn't already have one (left-only for wafer plots,
        # all four otherwise) -- so "no default style configured" isn't a
        # full no-op, just a no-op on every *other* field.
        assert {k: v for k, v in cfg.items() if k != 'spines_visible'} == original
        assert cfg['spines_visible'] == {'top': True, 'right': True, 'bottom': True, 'left': True}

    def test_merges_the_configured_default_style(self, ws):
        ws.m_settings.set_default_graph_style({'color_palette': 'magma', 'grid': True})
        cfg = {'df_name': 'sheet1', 'plot_style': 'scatter', 'x': 'x0_Si', 'y': ['ampli_Si']}

        ws._apply_default_style_to_config(cfg)

        assert cfg['color_palette'] == 'magma'
        assert cfg['grid'] is True
        # Identity fields untouched.
        assert cfg['df_name'] == 'sheet1'
        assert cfg['x'] == 'x0_Si'

    def test_configs_own_fields_always_win_over_the_default(self, ws):
        ws.m_settings.set_default_graph_style({'color_palette': 'magma'})
        cfg = {
            'df_name': 'sheet1', 'plot_style': 'scatter', 'x': 'x0_Si', 'y': ['ampli_Si'],
            'color_palette': 'viridis',  # collected value must win over the default
        }

        ws._apply_default_style_to_config(cfg)

        assert cfg['color_palette'] == 'viridis'


class TestEnsureWaferSpines:
    """The AI-agent / Maps-profile plot path (create_plot_from_config) bypasses
    the GUI 'Add Plot' default-style step, so it must apply the wafer left-only
    spine convention itself -- otherwise a wafer config with no spines_visible
    falls back to MGraph's all-four default and renders with four borders."""

    WAFER_LEFT_ONLY = {'top': False, 'right': False, 'bottom': False, 'left': True}

    def test_wafer_without_spines_gets_left_only(self, ws):
        cfg = {'plot_style': 'wafer', 'x': 'X', 'y': ['Y'], 'z': 'Strain (GPa)'}
        ws._ensure_wafer_spines(cfg)
        assert cfg['spines_visible'] == self.WAFER_LEFT_ONLY

    def test_non_wafer_is_untouched(self, ws):
        cfg = {'plot_style': 'scatter', 'x': 'X', 'y': ['Y']}
        ws._ensure_wafer_spines(cfg)
        assert 'spines_visible' not in cfg

    def test_explicit_spines_are_preserved(self, ws):
        # A saved recipe carries its own spines_visible -- don't override it.
        explicit = {'top': True, 'right': True, 'bottom': True, 'left': True}
        cfg = {'plot_style': 'wafer', 'x': 'X', 'y': ['Y'], 'spines_visible': explicit}
        ws._ensure_wafer_spines(cfg)
        assert cfg['spines_visible'] == explicit

    def test_create_plot_from_config_applies_wafer_default(self, ws, monkeypatch):
        # The AI/Maps entry point must apply the wafer default before building
        # the plot. Stub the render step so this stays focused and headless-safe
        # (a real wafer render needs coordinate columns and would pop a dialog
        # on failure); _ensure_wafer_spines mutates the config in place.
        monkeypatch.setattr(ws.vm, 'select_dataframe', lambda name: None)
        monkeypatch.setattr(ws, '_create_and_display_plot', lambda cfg, **kw: None)
        cfg = {
            'df_name': 'sheet1', 'plot_style': 'wafer',
            'x': 'X', 'y': ['Y'], 'z': 'Strain (GPa)',
        }
        ws.create_plot_from_config('sheet1', cfg)
        assert cfg['spines_visible'] == self.WAFER_LEFT_ONLY


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


class TestSubtitleSidePanelSync:
    """The side panel's 'Title and labels' group and the Customize dialog's
    More Options > Figure style tab both edit MGraph.plot_subtitle -- they
    must stay consistent (one shared field, not two independent copies)."""

    def test_sync_gui_from_graph_loads_subtitle(self, ws, excel_df):
        ws.vm.add_dataframe('sheet1', excel_df)
        ws.vm.select_dataframe('sheet1')
        graph_model = ws.vm.create_graph({
            'df_name': 'sheet1', 'plot_style': 'scatter',
            'x': 'x0_Si', 'y': ['ampli_Si'], 'plot_subtitle': 'From dialog',
        })
        ws._sync_gui_from_graph(graph_model)
        assert ws.edit_plot_subtitle.text() == 'From dialog'

    def test_update_plot_writes_subtitle_back(self, ws, excel_df):
        ws.vm.add_dataframe('sheet1', excel_df)
        ws.vm.select_dataframe('sheet1')
        graph_model = ws.vm.create_graph({
            'df_name': 'sheet1', 'plot_style': 'scatter',
            'x': 'x0_Si', 'y': ['ampli_Si'],
        })
        ws._build_graph_widget(graph_model, excel_df, lambda e: None)
        sub_window = ws.graph_widgets[graph_model.graph_id][2]
        ws.mdi_area.setActiveSubWindow(sub_window)
        ws._on_subwindow_activated(sub_window)

        ws.edit_plot_subtitle.setText("From side panel")
        ws._on_update_plot()

        assert ws.vm.get_graph(graph_model.graph_id).plot_subtitle == "From side panel"


class TestExportAll:
    def test_empty_workspace_shows_info_message_and_opens_no_dialog(self, ws, monkeypatch):
        infos = []
        monkeypatch.setattr(QMessageBox, 'information', lambda *a, **k: infos.append(a))
        opened = []
        class _StubDialog:
            def __init__(self, widgets, parent=None):
                opened.append(widgets)
            def exec(self):
                pass
        monkeypatch.setattr("spectroview.view.v_workspace_graphs.VBatchExportDialog", _StubDialog)

        ws._on_export_all_clicked()

        assert len(infos) == 1
        assert opened == []

    def test_open_workspace_opens_batch_dialog_with_every_graph_widget(self, ws, excel_df, monkeypatch):
        graph_model = ws.vm.create_graph({
            'df_name': 'sheet1', 'plot_style': 'scatter',
            'x': 'x0_Si', 'y': ['ampli_Si'],
        })
        widget = ws._build_graph_widget(graph_model, excel_df, lambda e: None)

        received = {}
        class _StubDialog:
            def __init__(self, widgets, parent=None):
                received['widgets'] = widgets
            def exec(self):
                received['exec_called'] = True
        monkeypatch.setattr("spectroview.view.v_workspace_graphs.VBatchExportDialog", _StubDialog)

        ws._on_export_all_clicked()

        assert received['widgets'] == {graph_model.graph_id: widget}
        assert received['exec_called'] is True


class TestUndoRedo:
    """View-level integration of VMWorkspaceGraphs' undo/redo stacks (see
    tests/unit/viewmodel/test_vm_workspace_graphs.py::TestUndoRedo for the
    pure ViewModel-level stack semantics): the Undo/Redo toolbar buttons,
    _rebuild_all_graph_widgets()'s interaction with the stack, and a couple
    of regression scenarios found while wiring it up."""

    def _graph(self, ws, excel_df, **overrides):
        cfg = {
            'df_name': 'sheet1', 'plot_style': 'scatter',
            'x': 'x0_Si', 'y': ['ampli_Si'],
        }
        cfg.update(overrides)
        graph_model = ws.vm.create_graph(cfg)
        return ws._build_graph_widget(graph_model, excel_df, lambda e: None)

    def test_undo_clicked_with_empty_history_is_a_noop(self, ws):
        assert ws.vm.can_undo is False
        ws._on_undo_clicked()  # must not raise
        assert ws.graph_widgets == {}

    def test_undo_redo_buttons_track_stack_state(self, ws, excel_df):
        assert ws.btn_undo.isEnabled() is False
        assert ws.btn_redo.isEnabled() is False

        gw = self._graph(ws, excel_df)
        ws._on_undo_state_changed()
        assert ws.btn_undo.isEnabled() is True
        assert ws.btn_redo.isEnabled() is False

        ws._on_undo_clicked()
        assert ws.btn_undo.isEnabled() is False
        assert ws.btn_redo.isEnabled() is True

        ws._on_redo_clicked()
        assert ws.btn_undo.isEnabled() is True
        assert ws.btn_redo.isEnabled() is False

    def test_undo_to_before_creation_empties_the_workspace_and_redo_restores_it(self, ws, excel_df):
        gw = self._graph(ws, excel_df)
        gid = gw.graph_id

        ws._on_undo_clicked()
        assert ws.vm.get_graph_ids() == []
        assert ws.graph_widgets == {}

        ws._on_redo_clicked()
        assert ws.vm.get_graph_ids() == [gid]
        assert gid in ws.graph_widgets

    def test_undo_then_redo_round_trips_a_style_change(self, ws, excel_df):
        gw = self._graph(ws, excel_df)
        gid = gw.graph_id
        assert ws.vm.get_graph(gid).grid is False

        ws.vm.update_graph(gid, {'grid': True})
        assert ws.vm.get_graph(gid).grid is True

        ws._on_undo_clicked()
        assert ws.vm.get_graph(gid).grid is False

        ws._on_redo_clicked()
        assert ws.vm.get_graph(gid).grid is True

    def test_on_delete_all_is_one_undo_step_and_fully_reversible(self, ws, excel_df, monkeypatch):
        monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.Yes)
        gw1 = self._graph(ws, excel_df)
        gw2 = self._graph(ws, excel_df, x='fwhm_Si')
        ids_before = {gw1.graph_id, gw2.graph_id}

        ws._on_delete_all()

        assert ws.vm.get_graph_ids() == []
        assert ws.graph_widgets == {}

        ws._on_undo_clicked()

        assert set(ws.vm.get_graph_ids()) == ids_before
        assert set(ws.graph_widgets.keys()) == ids_before

    def test_on_update_plot_collapses_its_several_update_graph_calls_into_one_undo_step(
        self, ws, excel_df
    ):
        """_on_update_plot() calls vm.update_graph() up to three times
        (plot_config, legend sync before reconfigure, legend sync after
        render) inside one begin_undo_batch()/end_undo_batch() pair -- a
        single undo must revert all of it at once, not just the last call."""
        ws.vm.add_dataframe('sheet1', excel_df)
        ws.vm.select_dataframe('sheet1')
        gw = self._graph(ws, excel_df)
        sub_window = ws.graph_widgets[gw.graph_id][2]
        ws.mdi_area.setActiveSubWindow(sub_window)
        ws._on_subwindow_activated(sub_window)

        original_title = ws.vm.get_graph(gw.graph_id).plot_title
        ws.edit_plot_title.setText("Updated via panel")

        ws._on_update_plot()
        assert ws.vm.get_graph(gw.graph_id).plot_title == "Updated via panel"

        ws._on_undo_clicked()
        assert ws.vm.get_graph(gw.graph_id).plot_title == original_title

    def test_undo_restores_a_graph_deleted_by_the_old_rebuild_cascade_bug(self, ws, excel_df):
        """Regression test: _rebuild_all_graph_widgets() (used by both
        undo() and redo()) must disconnect each subwindow's `closed` signal
        before tearing it down -- otherwise the programmatic teardown
        cascades into _on_graph_closed() -> vm.delete_graph(), corrupting
        the very undo stack undo()/redo() is in the middle of using."""
        gw1 = self._graph(ws, excel_df)
        gw2 = self._graph(ws, excel_df, x='fwhm_Si')
        ids = {gw1.graph_id, gw2.graph_id}

        sub_window = ws.graph_widgets[gw1.graph_id][2]
        ws.mdi_area.setActiveSubWindow(sub_window)
        sub_window.close()  # user-closed -> vm.delete_graph(gw1.graph_id), one undo step

        assert set(ws.vm.get_graph_ids()) == {gw2.graph_id}

        ws._on_undo_clicked()

        assert set(ws.vm.get_graph_ids()) == ids
        assert set(ws.graph_widgets.keys()) == ids
        # The rebuild itself must not have pushed a spurious extra undo
        # step -- redo should still be available after this one undo.
        assert ws.vm.can_redo is True


class TestKeyboardShortcuts:
    """Ctrl+Z/Ctrl+Shift+Z/Ctrl+C/Ctrl+V accelerators, scoped to
    self.mdi_area (not the whole workspace widget) so they don't shadow a
    side-panel QLineEdit's own native shortcuts while typing."""

    def _graph(self, ws, excel_df, **overrides):
        cfg = {
            'df_name': 'sheet1', 'plot_style': 'scatter',
            'x': 'x0_Si', 'y': ['ampli_Si'],
        }
        cfg.update(overrides)
        graph_model = ws.vm.create_graph(cfg)
        return ws._build_graph_widget(graph_model, excel_df, lambda e: None)

    def test_all_shortcuts_registered_with_expected_context(self, ws):
        shortcuts = ws.mdi_area.findChildren(QShortcut)
        by_key = {sc.key().toString(): sc for sc in shortcuts}

        for key in ("Ctrl+Z", "Ctrl+Shift+Z", "Ctrl+C", "Ctrl+V", "Ctrl+E", "Ctrl+R"):
            assert key in by_key, f"{key} not registered on mdi_area"
            assert by_key[key].context() == Qt.ShortcutContext.WidgetWithChildrenShortcut

    def test_get_active_graph_id_none_when_nothing_active(self, ws):
        assert ws._get_active_graph_id() is None

    def test_get_active_graph_id_returns_the_active_subwindow_graph(self, ws, excel_df):
        gw = self._graph(ws, excel_df)
        sub_window = ws.graph_widgets[gw.graph_id][2]
        ws.mdi_area.setActiveSubWindow(sub_window)

        assert ws._get_active_graph_id() == gw.graph_id

    def test_copy_figure_and_paste_style_shortcuts_are_noop_with_no_active_graph(self, ws):
        ws._on_copy_figure_shortcut()  # must not raise
        ws._on_paste_style_shortcut()  # must not raise
        assert getattr(ws, '_copied_style', None) is None

    def test_copy_figure_shortcut_copies_the_active_graphs_figure(self, ws, excel_df, monkeypatch):
        """Ctrl+C copies the active graph's *figure* (matching the toolbar's
        Copy button), not its style -- style copy/paste is still reachable
        via the Style menu, just without a Ctrl+C shortcut of its own."""
        gw = self._graph(ws, excel_df)
        sub_window = ws.graph_widgets[gw.graph_id][2]
        ws.mdi_area.setActiveSubWindow(sub_window)
        calls = []
        monkeypatch.setattr(gw, 'copy_to_clipboard', lambda: calls.append(gw.graph_id))

        ws._on_copy_figure_shortcut()

        assert calls == [gw.graph_id]

    def test_paste_style_shortcut_applies_a_style_copied_via_the_style_menu(self, ws, excel_df):
        source = self._graph(ws, excel_df)
        source.color_palette = "viridis"
        ws._on_style_action_requested(source.graph_id, "copy")

        target = self._graph(ws, excel_df)
        target_window = ws.graph_widgets[target.graph_id][2]
        ws.mdi_area.setActiveSubWindow(target_window)
        ws._on_paste_style_shortcut()

        assert target.color_palette == "viridis"

    def test_customize_shortcut_is_noop_with_no_active_graph(self, ws):
        ws._on_customize_shortcut()  # must not raise

    def test_customize_shortcut_opens_the_dialog_for_the_active_graph(self, ws, excel_df, monkeypatch):
        """Ctrl+E mirrors clicking the toolbar's Customize button."""
        gw = self._graph(ws, excel_df)
        sub_window = ws.graph_widgets[gw.graph_id][2]
        ws.mdi_area.setActiveSubWindow(sub_window)
        calls = []
        monkeypatch.setattr(ws, '_show_or_switch_customize_dialog', lambda gid: calls.append(gid))

        ws._on_customize_shortcut()

        assert calls == [gw.graph_id]

    def test_rescale_shortcut_is_noop_with_no_active_graph(self, ws):
        ws._on_rescale_shortcut()  # must not raise

    def test_rescale_shortcut_rescales_the_active_graph(self, ws, excel_df, monkeypatch):
        """Ctrl+R rescales the active graph (matplotlib Home), like the
        Spectra/Maps workspaces' rescale shortcut."""
        gw = self._graph(ws, excel_df)
        sub_window = ws.graph_widgets[gw.graph_id][2]
        ws.mdi_area.setActiveSubWindow(sub_window)
        calls = []
        monkeypatch.setattr(gw, '_rescale', lambda: calls.append(gw.graph_id))

        ws._on_rescale_shortcut()

        assert calls == [gw.graph_id]

    def test_undo_redo_shortcuts_drive_the_same_handlers_as_the_buttons(self, ws, excel_df, monkeypatch):
        calls = []
        monkeypatch.setattr(ws, '_on_undo_clicked', lambda: calls.append('undo'))
        monkeypatch.setattr(ws, '_on_redo_clicked', lambda: calls.append('redo'))

        shortcuts = ws.mdi_area.findChildren(QShortcut)
        by_key = {sc.key().toString(): sc for sc in shortcuts}
        by_key["Ctrl+Z"].activated.emit()
        by_key["Ctrl+Shift+Z"].activated.emit()

        assert calls == ['undo', 'redo']
