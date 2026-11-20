"""Recipes, style templates, export, customize dialog, undo/redo, and persistence mixin for VWorkspaceGraphs."""

import copy

from PySide6.QtWidgets import QInputDialog, QMessageBox

from spectroview.i18n import tr, tr_plural
from spectroview.model.graphs.graph_style import apply_style_dict, default_style, extract_style
from spectroview.model.graphs.m_plot_recipe_store import MPlotRecipeStore
from spectroview.model.graphs.m_style_template_store import MStyleTemplateStore
from spectroview.view.components.customize_graph.customize_graph_dialog import CustomizeGraphDialog
from spectroview.view.components.graphs.graph_commit import diff, snapshot
from spectroview.view.components.graphs.v_export_dialog import VBatchExportDialog, VExportDialog
from spectroview.view.components.graphs.v_graph import VGraph
from spectroview.view.components.graphs.v_multipanel_dialog import VMultiPanelDialog
from spectroview.view.components.graphs.v_plot_recipe_dialog import VPlotRecipeDialog
from spectroview.view.components.graphs.v_style_template_dialog import VStyleTemplateDialog


class GraphsActionsMixin:
    """Mixin managing recipes, style templates, exports, customize dialog, undo/redo, and workspace persistence."""

    def _refresh_recipe_and_style_stores(self) -> None:
        """Rebuild both stores from the *current* Working Folder setting."""
        self.recipe_store = MPlotRecipeStore(self.m_settings.get_plot_recipe_folder())
        self.style_template_store = MStyleTemplateStore(self.m_settings.get_plot_style_folder())

    def _on_apply_recipe_clicked(self) -> None:
        self._refresh_recipe_and_style_stores()
        dialog = VPlotRecipeDialog(self.recipe_store, self)
        dialog.recipe_applied.connect(self._on_recipe_applied)
        dialog.exec()

    def _on_save_as_recipe_clicked(self) -> None:
        """Save every currently open graph's live configuration as a new Plot Recipe."""
        self._refresh_recipe_and_style_stores()
        configs = []
        for graph_model in self.vm.graphs.values():
            cfg = graph_model.save()
            cfg.pop('graph_id', None)
            configs.append(cfg)

        if not configs:
            self._show_notification(
                tr("No open graphs to save as a recipe."), notification_type="warning"
            )
            return

        name, ok = QInputDialog.getText(
            self, tr("Save Plot Recipe"), tr("Recipe name:")
        )
        if ok and name:
            self.recipe_store.save_recipe(name, configs)
            self._show_notification(
                tr_plural(
                    "Saved plot recipe '{name}' with {count} plot.",
                    "Saved plot recipe '{name}' with {count} plots.",
                    len(configs),
                    name=name,
                ),
                notification_type="success",
            )

    def _on_export_all_clicked(self) -> None:
        """Export every currently open graph to a folder in one pass."""
        if not self.graph_widgets:
            self._show_notification(
                tr("No open graphs to export."), notification_type="warning"
            )
            return

        widgets = {gid: gw for gid, (gw, _, _) in self.graph_widgets.items()}
        dialog = VBatchExportDialog(widgets, parent=self)
        dialog.exec()

    def _on_compose_figure_clicked(self) -> None:
        """Open the multi-panel composer to combine several open graphs into one exported figure."""
        if not self.graph_widgets:
            self._show_notification(
                tr("No open graphs to compose."), notification_type="warning"
            )
            return

        widgets = {gid: gw for gid, (gw, _, _) in self.graph_widgets.items()}
        dialog = VMultiPanelDialog(widgets, parent=self)
        dialog.exec()

    def _on_style_action_requested(self, graph_id: int, action: str) -> None:
        """Central dispatch for VGraph's per-graph Style menu."""
        entry = self.graph_widgets.get(graph_id)
        if not entry:
            return
        gw = entry[0]

        if action == "save_template":
            self._save_style_template(gw)
        elif action == "apply_template":
            self._apply_style_template_dialog(gw)
        elif action == "copy":
            self._copied_style = extract_style(vars(gw))
            self._show_notification(tr("Style copied from graph {gid}.", gid=graph_id))
        elif action == "paste":
            if not self._copied_style:
                self._show_notification(
                    tr("Nothing to paste. Copy a graph's style first."),
                    notification_type="warning",
                )
                return
            self._apply_style_to_graph(gw, self._copied_style)
        elif action == "reset":
            self._apply_style_to_graph(gw, default_style())
        elif action == "set_default":
            self.m_settings.set_default_graph_style(extract_style(vars(gw)))
            self._show_notification(
                tr(
                    "Graph {gid}'s style is now the default for new plots.",
                    gid=graph_id,
                )
            )

    def _save_style_template(self, gw: VGraph) -> None:
        self._refresh_recipe_and_style_stores()
        name, ok = QInputDialog.getText(self, tr("Save Style"), tr("Style name:"))
        if not (ok and name):
            return
        style = extract_style(vars(gw))
        tpl_id = self.style_template_store.save_template(name, style)
        if tpl_id:
            self._show_notification(tr("Saved style '{name}'.", name=name))
        else:
            QMessageBox.warning(
                self,
                tr("Could Not Save"),
                tr("No working folder is configured yet (set one in Settings)."),
            )

    def _apply_style_template_dialog(self, gw: VGraph) -> None:
        self._refresh_recipe_and_style_stores()
        dialog = VStyleTemplateDialog(self.style_template_store, self)
        dialog.style_applied.connect(lambda style, target=gw: self._apply_style_to_graph(target, style))
        dialog.exec()

    def _apply_style_to_graph(self, gw: VGraph, style: dict) -> None:
        """Write `style` onto `gw`, replot, and persist via properties_changed."""
        applied = apply_style_dict(gw, style)
        if gw.df is not None:
            gw.plot(gw.df)
        gw.properties_changed.emit(gw.graph_id, applied)

    def _on_recipe_applied(self, configs: list) -> None:
        """Apply every plot config in a saved recipe against the currently selected DataFrame."""
        from spectroview.ai_agent.utils.plot_utils import normalize_plot_config

        df_name = self.vm.selected_df_name
        if not df_name:
            QMessageBox.warning(
                self,
                tr("No DataFrame Selected"),
                tr("Please select a DataFrame before applying a recipe."),
            )
            return

        df = self.vm.get_dataframe(df_name)
        available_columns = set(df.columns) if df is not None else set()

        applied = 0
        skipped = []
        self.vm.begin_undo_batch()
        try:
            for i, raw_cfg in enumerate(configs, start=1):
                cfg = copy.deepcopy(raw_cfg)
                normalize_plot_config(cfg)

                missing = sorted(self._required_plot_columns(cfg) - available_columns)
                if missing:
                    skipped.append((i, missing))
                    continue

                cfg['df_name'] = df_name
                self.create_plot_from_config(df_name, cfg)
                applied += 1
        finally:
            self.vm.end_undo_batch()

        if skipped:
            lines = [
                tr(
                    "Plot {index} can not be plotted since the selected dataframe does not have columns: {cols}.",
                    index=i,
                    cols=", ".join(cols),
                )
                for i, cols in skipped
            ]
            QMessageBox.warning(
                self,
                tr("Some Plots Skipped"),
                tr(
                    "{applied} of {total} plot(s) applied.\n\n{details}",
                    applied=applied,
                    total=len(configs),
                    details="\n".join(lines),
                ),
            )
        elif applied:
            self.vm.notify.emit(
                tr_plural(
                    "Applied {count} plot from recipe.",
                    "Applied {count} plots from recipe.",
                    applied,
                )
            )

    @staticmethod
    def _required_plot_columns(cfg: dict) -> set:
        """Column names a plot config's axes reference (x/y/z/y2/y3/x2)."""
        cols = set()
        for key in ('x', 'z', 'y2', 'y3', 'x2'):
            val = cfg.get(key)
            if val:
                cols.add(val)
        y = cfg.get('y')
        if isinstance(y, list):
            cols.update(v for v in y if v)
        elif y:
            cols.add(y)
        return cols

    def _refresh_open_customize_dialog(self, graph_widget: VGraph):
        """Reload open CustomizeGraphDialog tabs and baseline snapshot after plot update."""
        dialog = self._customize_dialog
        if hasattr(dialog, '_update_graph_info'):
            dialog._update_graph_info()
        for tab_attr in (
            'tab_general',
            'tab_axes_limits',
            'tab_ticks_spines',
            'tab_secondary_inset',
            'tab_legend',
            'tab_plot_styles',
            'tab_annotations',
        ):
            tab = getattr(dialog, tab_attr, None)
            if tab is not None and hasattr(tab, 'load_settings'):
                tab.load_settings()
        if hasattr(dialog, '_original_snapshot'):
            dialog._original_snapshot = snapshot(graph_widget)
        if hasattr(dialog, '_preview_timer'):
            dialog._preview_timer.stop()

    def _show_or_switch_customize_dialog(self, graph_id: int):
        """Show, create, or switch the singleton CustomizeGraphDialog."""
        if graph_id not in self.graph_widgets:
            return

        graph_widget, _, _ = self.graph_widgets[graph_id]

        if self._customize_dialog is None:
            self._customize_dialog = CustomizeGraphDialog(graph_widget, graph_id, parent=self)
            self._customize_dialog.show()
        elif self._customize_dialog.isVisible():
            self._customize_dialog.switch_graph(graph_widget, graph_id)
            self._customize_dialog.raise_()
            self._customize_dialog.activateWindow()
        else:
            self._customize_dialog.switch_graph(graph_widget, graph_id)
            self._customize_dialog.show()
            self._customize_dialog.raise_()
            self._customize_dialog.activateWindow()

    def _on_export_graph_requested(self, graph_id: int):
        """Open the (modal, one-shot) export dialog for a single graph."""
        if graph_id not in self.graph_widgets:
            return
        graph_widget, _, _ = self.graph_widgets[graph_id]
        dialog = VExportDialog(graph_widget, parent=self)
        dialog.exec()

    def save_workspace(self):
        """Save workspace."""
        for gid, (gw, gd, sw) in self.graph_widgets.items():
            before = snapshot(gw)
            size = sw.size()
            gw.plot_width, gw.plot_height = size.width(), size.height()
            gw._save_legend_position()
            patch = diff(gw, before)
            if patch:
                self.vm.update_graph(gid, patch)

        self.vm.save_workspace()

    def load_workspace(self, file_path: str):
        """Load workspace."""
        self.clear_workspace()

        self.vm.load_workspace(file_path)

        for graph_id in self.vm.get_graph_ids():
            graph_model = self.vm.get_graph(graph_id)
            if graph_model and graph_model.filters:
                for f in graph_model.filters:
                    if isinstance(f, dict):
                        expr = f.get('expression', '').strip()
                        if expr:
                            self.v_data_filter.add_filter_definition(expr)

        if self.df_listbox.count() > 0:
            self.df_listbox.setCurrentRow(0)

        self._rebuild_all_graph_widgets()

    def _rebuild_all_graph_widgets(self):
        """Tear down every VGraph widget/MDI subwindow and rebuild them
        from the ViewModel's current self.vm.graphs."""
        self._teardown_all_graph_windows()

        for graph_id in self.vm.get_graph_ids():
            graph_model = self.vm.get_graph(graph_id)
            if not graph_model:
                continue

            filtered_df = self.vm.apply_filters(graph_model.df_name, graph_model.filters)

            def _on_render_error(e, graph_model=graph_model):
                self.vm.notify.emit(
                    tr(
                        "Skipped graph {gid} ({style}): could not render ({error})",
                        gid=graph_model.graph_id,
                        style=graph_model.plot_style,
                        error=e,
                    )
                )

            self._build_graph_widget(graph_model, filtered_df, _on_render_error,
                                     sync_toolbar=False)

        self._update_graph_list(self.vm.get_graph_ids())
        self._sync_active_graph_toolbar()

    def _on_undo_clicked(self):
        """Revert the last undo-tracked action."""
        if self.vm.undo():
            self._rebuild_all_graph_widgets()
            self.vm.notify.emit(tr("Undid last action."))

    def _on_redo_clicked(self):
        """Re-apply the last undone action."""
        if self.vm.redo():
            self._rebuild_all_graph_widgets()
            self.vm.notify.emit(tr("Redid last action."))

    def _on_undo_state_changed(self):
        """Keep the Undo/Redo buttons' enabled state in sync with the ViewModel's undo/redo stacks."""
        self.btn_undo.setEnabled(self.vm.can_undo)
        self.btn_redo.setEnabled(self.vm.can_redo)

    def clear_workspace(self):
        """Clear workspace."""
        self._teardown_all_graph_windows()
        self._sync_active_graph_toolbar()
        self.df_listbox.clear()
        self.cbb_x.clear()
        self.cbb_y.clear()
        self.cbb_z.clear()
        self.cbb_y2.clear()
        self.cbb_y3.clear()
        self.cbb_x2.clear()
        self.cbb_graph_list.clear()
        self.edit_plot_title.clear()
        self.edit_plot_subtitle.clear()
        self.edit_xlabel.clear()
        self.edit_ylabel.clear()
        self.edit_zlabel.clear()
        self.edit_y2label.clear()
        self.edit_y3label.clear()
        self.edit_x2label.clear()
        self.v_data_filter.clear_filters()

        self.vm.clear_workspace()
