"""MDI subwindow management, shared toolbar sync, and plot build/update lifecycle mixin."""

import copy
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QDialog, QMdiSubWindow, QMessageBox, QVBoxLayout

from spectroview.model.graphs.graph_control import GraphValidationError, normalize_graph_patch
from spectroview.view.components.graphs.graph_commit import snapshot
from spectroview.view.components.graphs.v_graph import VGraph
from spectroview.view.workspaces.graphs.v_mdi_subwindow import MdiSubWindow


class GraphsMdiMixin:
    """Mixin coordinating graph widgets, MDI subwindows, shared toolbar sync, and GUI sync."""

    def _graph_id_for_subwindow(self, sub_window) -> Optional[int]:
        """graph_id owning `sub_window`, or None if it's not a graph window."""
        for gid, (_gw, _gd, sw) in self.graph_widgets.items():
            if sw == sub_window:
                return gid
        return None

    def _get_active_graph_id(self) -> Optional[int]:
        """graph_id of the currently active MDI subwindow, or None if no
        graph window is active."""
        active_subwindow = self.mdi_area.activeSubWindow()
        if not active_subwindow:
            return None
        return self._graph_id_for_subwindow(active_subwindow)

    def _on_copy_figure_shortcut(self):
        gid = self._get_active_graph_id()
        if gid is not None and gid in self.graph_widgets:
            graph_widget, _, _ = self.graph_widgets[gid]
            graph_widget.copy_to_clipboard()

    def _on_paste_style_shortcut(self):
        gid = self._get_active_graph_id()
        if gid is not None:
            self._on_style_action_requested(gid, "paste")

    def _on_customize_shortcut(self):
        gid = self._get_active_graph_id()
        if gid is not None:
            self._show_or_switch_customize_dialog(gid)

    def _on_rescale_shortcut(self):
        gid = self._get_active_graph_id()
        if gid is not None and gid in self.graph_widgets:
            graph_widget, _, _ = self.graph_widgets[gid]
            graph_widget._rescale()

    def _on_add_plot(self):
        """Add new plot."""
        current_filters = self.v_data_filter.get_filters()

        if not self._validate_plot_request():
            return

        plot_config = self._collect_plot_config(include_labels=False)
        if not plot_config['x'] or not plot_config['y']:
            self._show_notification("Please select X and Y axes.", notification_type="warning")
            return
        self._apply_default_style_to_config(plot_config)

        self._create_and_display_plot(plot_config, filters=current_filters)

    def _apply_default_style_to_config(self, plot_config: dict) -> None:
        """Merge the "Set as Default Style" baseline (if any) under a
        freshly-collected plot_config, for graphs built from scratch (Add
        Plot / Add Multi-Wafer) only -- never for replicate/recipe-apply,
        whose configs already carry an intentional, fully-specified style.
        plot_config's own keys always win over the default style."""
        default_style = self.m_settings.get_default_graph_style()
        if default_style:
            if plot_config.get('plot_style') == 'wafer':
                default_style.pop('spines_visible', None)

            merged = dict(default_style)
            merged.update(plot_config)
            plot_config.clear()
            plot_config.update(merged)

        if 'spines_visible' not in plot_config:
            if plot_config.get('plot_style') == 'wafer':
                self._ensure_wafer_spines(plot_config)
            else:
                plot_config['spines_visible'] = {'top': True, 'right': True, 'bottom': True, 'left': True}

    @staticmethod
    def _ensure_wafer_spines(plot_config: dict) -> None:
        """Force the wafer convention of showing only the left spine."""
        if (plot_config.get('plot_style') == 'wafer'
                and 'spines_visible' not in plot_config):
            plot_config['spines_visible'] = {
                'top': False, 'right': False, 'bottom': False, 'left': True
            }

    def _validate_plot_request(self) -> bool:
        """Validate DataFrame selection."""
        if not self.vm.selected_df_name:
            self._show_notification("Please select a DataFrame first.", notification_type="warning")
            return False

        df = self.vm.get_dataframe(self.vm.selected_df_name)
        if df is None or df.empty:
            self._show_notification("Selected DataFrame is empty.", notification_type="warning")
            return False
        return True

    def _build_graph_widget(self, graph_model, filtered_df, on_render_error,
                            sync_toolbar: bool = True) -> Optional[VGraph]:
        """Instantiate a VGraph for `graph_model`, wire its signals, render
        it, and register it into `graph_widgets` + the MDI area."""
        graph_widget = VGraph(graph_id=graph_model.graph_id)
        graph_widget.replicate_requested.connect(self._on_replicate_graph)
        graph_widget.customize_requested.connect(self._show_or_switch_customize_dialog)
        graph_widget.export_requested.connect(self._on_export_graph_requested)
        graph_widget.export_all_requested.connect(self._on_export_all_clicked)
        graph_widget.style_action_requested.connect(self._on_style_action_requested)
        graph_widget.properties_changed.connect(self._on_graph_properties_changed)
        graph_widget.notify.connect(self._show_notification)

        self._configure_graph_from_model(graph_widget, graph_model)
        graph_widget.create_plot_widget(graph_model.dpi)

        try:
            self._render_plot(graph_widget, filtered_df, graph_model)
        except Exception as e:
            self.vm.delete_graph(graph_model.graph_id)
            on_render_error(e)
            return None

        self.vm.sync_derived_graph_properties(graph_model.graph_id, {
            'legend_properties': graph_widget.legend_properties,
        })

        sub_window = self._create_mdi_subwindow(graph_model)
        graph_dialog = self._wrap_graph_in_dialog(graph_widget)
        sub_window.setWidget(graph_dialog)

        self.graph_widgets[graph_model.graph_id] = (graph_widget, graph_dialog, sub_window)
        self.mdi_area.addSubWindow(sub_window)
        sub_window.show()
        if sync_toolbar:
            self._sync_active_graph_toolbar()

        return graph_widget

    def _create_and_display_plot(self, plot_config: dict, select_in_list: bool = True, filters: list = None):
        """Create and display a plot from configuration."""
        self.vm.begin_undo_batch()
        try:
            if filters is None:
                filters = self.v_data_filter.get_filters()

            graph_model = self.vm.create_graph(plot_config)
            filtered_df = self.vm.apply_filters(plot_config['df_name'], filters)

            def _on_render_error(e):
                self._show_notification(f"Error rendering plot: {str(e)}", notification_type="error")

            if self._build_graph_widget(graph_model, filtered_df, _on_render_error) is None:
                return

            self._update_graph_list(self.vm.get_graph_ids())

            if select_in_list:
                for i in range(self.cbb_graph_list.count()):
                    if self.cbb_graph_list.itemData(i) == graph_model.graph_id:
                        self.cbb_graph_list.setCurrentIndex(i)
                        break
        finally:
            self.vm.end_undo_batch()

    def _wrap_graph_in_dialog(self, graph_widget: VGraph) -> QDialog:
        """Wrap graph widget in dialog."""
        graph_dialog = QDialog(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(graph_widget)
        graph_dialog.setLayout(layout)
        return graph_dialog

    def create_plot_from_config(self, df_name: str, plot_config: dict) -> bool:
        """API to create a plot directly from a configuration dict."""
        self.vm.select_dataframe(df_name)
        self._ensure_wafer_spines(plot_config)
        filters = plot_config.get('filters', [])
        if filters is None:
            filters = []
        self._create_and_display_plot(plot_config, select_in_list=False, filters=filters)
        return True

    def update_graphs_from_config(self, graph_id, properties: dict) -> list:
        """Apply one shared Graph patch to one graph or ``"all"``."""
        key = str(graph_id).strip().lower()
        if key == 'all':
            target_ids = self.vm.get_graph_ids()
        else:
            try:
                target_ids = [int(graph_id)]
            except (TypeError, ValueError):
                self.vm.notify.emit("Graph update rejected: graph ID must be numeric or 'all'.")
                return []
        target_ids = [gid for gid in target_ids if self.vm.get_graph(gid) is not None]
        if not target_ids:
            self.vm.notify.emit("Graph update rejected: no matching open graph.")
            return []

        prepared = {}
        snapshots = {}
        try:
            for gid in target_ids:
                model = self.vm.get_graph(gid)
                target_df_name = properties.get('df_name', model.df_name)
                prepared[gid] = normalize_graph_patch(
                    properties,
                    current=model,
                    dataframe=self.vm.get_dataframe(target_df_name),
                )
                snapshots[gid] = model.save()
        except GraphValidationError as exc:
            self.vm.notify.emit(f"Graph update rejected: {exc}")
            return []

        self.vm.begin_undo_batch()
        try:
            for gid in target_ids:
                self.vm.update_graph(gid, prepared[gid])
            for gid in target_ids:
                self._refresh_graph_from_model(gid, set(prepared[gid]))
        except Exception as exc:
            for gid, state in snapshots.items():
                self.vm.restore_graph_state(gid, state)
            for gid in target_ids:
                try:
                    self._refresh_graph_from_model(gid, set())
                except Exception:
                    pass
            self._show_notification(
                f"Graph update rolled back: {exc}",
                notification_type="warning",
            )
            return []
        finally:
            self.vm.end_undo_batch()

        self._update_graph_list(self.vm.get_graph_ids())
        self._sync_active_graph_toolbar()
        return target_ids

    def _refresh_graph_from_model(self, graph_id: int, changed_fields: set) -> None:
        """Re-render one existing graph and synchronize its live controls."""
        entry = self.graph_widgets.get(graph_id)
        model = self.vm.get_graph(graph_id)
        if entry is None or model is None:
            return
        graph_widget, _, sub_window = entry
        filtered_df = self.vm.apply_filters(model.df_name, model.filters)
        self._configure_graph_from_model(graph_widget, model)
        graph_widget.create_plot_widget(model.dpi)
        self._render_plot(graph_widget, filtered_df, model)

        self.vm.sync_derived_graph_properties(graph_id, {
            'legend_properties': graph_widget.legend_properties,
            'legend_bbox': graph_widget.legend_bbox,
        })
        sub_window.setWindowTitle(self._graph_window_title(model))
        if {'plot_width', 'plot_height'} & changed_fields:
            sub_window.resize(model.plot_width, model.plot_height)

        if (self._customize_dialog is not None
                and self._customize_dialog.isVisible()
                and self._customize_dialog.graph_id == graph_id):
            self._customize_dialog.switch_graph(graph_widget, graph_id)

    def _on_replicate_graph(self, graph_id: int):
        """Replicate a graph."""
        graph_model = self.vm.get_graph(graph_id)
        if not graph_model:
            return

        plot_config = graph_model.save()
        del plot_config['graph_id']

        if graph_id in self.graph_widgets:
            _, _, sub_window = self.graph_widgets[graph_id]
            size = sub_window.size()
            plot_config['plot_width'] = size.width()
            plot_config['plot_height'] = size.height()

        if not plot_config.get('df_name'):
            self._show_notification("Cannot replicate a plot without a valid DataFrame.", notification_type="warning")
            return

        self._create_and_display_plot(plot_config, select_in_list=True, filters=plot_config.get('filters', []))

    def _on_graph_properties_changed(self, graph_id: int, properties: dict):
        """Update graph model when properties change directly from graph widget."""
        self.vm.update_graph(graph_id, properties)
        active_subwindow = self.mdi_area.activeSubWindow()
        if active_subwindow and self._graph_id_for_subwindow(active_subwindow) == graph_id:
            shared_keys = self._SIDE_PANEL_SHARED_FIELDS.intersection(properties)
            if shared_keys:
                graph_model = self.vm.get_graph(graph_id)
                if graph_model:
                    self._sync_side_panel_shared_fields(graph_model, keys=shared_keys)

    def _on_update_plot(self):
        """Update selected plot."""
        self.vm.begin_undo_batch()
        try:
            active_subwindow = self.mdi_area.activeSubWindow()
            if not active_subwindow:
                self._show_notification("Please select a plot to update.", notification_type="warning")
                return

            gid = self._graph_id_for_subwindow(active_subwindow)
            graph_model = self.vm.get_graph(gid) if gid is not None else None
            if not graph_model:
                return
            graph_widget = self.graph_widgets[gid][0]

            customize_open_for_graph = (
                self._customize_dialog is not None
                and self._customize_dialog.isVisible()
                and self._customize_dialog.graph_id == gid
            )
            if customize_open_for_graph:
                if self._customize_dialog._preview_timer.isActive():
                    self._customize_dialog._preview_timer.stop()
                    self._customize_dialog._preview_apply()
                changed_secondary = {
                    key for key in self._SECONDARY_SHARED_FIELDS
                    if getattr(graph_widget, key, None) != getattr(graph_model, key, None)
                }
                if changed_secondary:
                    self._sync_side_panel_shared_fields(graph_widget, keys=changed_secondary)

            plot_config = snapshot(graph_widget)
            panel_config = self._collect_plot_config()
            panel_config.pop('xlogscale', None)
            panel_config.pop('ylogscale', None)
            panel_config.pop('legend_visible', None)
            use_palette = (
                panel_config['plot_style'] in ['wafer', '2Dmap']
                or panel_config['z'] is not None
            )
            if not use_palette:
                panel_config.pop('color_palette', None)

            plot_config.update(panel_config)
            plot_config['df_name'] = graph_model.df_name

            if plot_config['plot_style'] != graph_model.plot_style:
                if plot_config['plot_style'] == 'wafer':
                    plot_config['spines_visible'] = {'top': False, 'right': False, 'bottom': False, 'left': True}
                elif graph_model.plot_style == 'wafer':
                    plot_config['spines_visible'] = {'top': True, 'right': True, 'bottom': True, 'left': True}
            else:
                if plot_config['plot_style'] == 'wafer' and getattr(graph_model, 'spines_visible', None) == {'top': True, 'right': True, 'bottom': True, 'left': True}:
                    plot_config['spines_visible'] = {'top': False, 'right': False, 'bottom': False, 'left': True}

            z_changed = plot_config['z'] != graph_model.z
            current_filters = plot_config['filters']
            if z_changed or current_filters != graph_model.filters:
                graph_widget.legend_properties = []

            graph_widget._save_legend_position()

            plot_config['legend_properties'] = graph_widget.legend_properties
            plot_config['legend_bbox'] = graph_widget.legend_bbox

            self.vm.update_graph(graph_model.graph_id, plot_config)
            graph_model = self.vm.get_graph(graph_model.graph_id)

            filtered_df = self.vm.apply_filters(graph_model.df_name, current_filters)

            self._configure_graph_from_model(graph_widget, graph_model)
            graph_widget.create_plot_widget(graph_model.dpi)

            try:
                self._render_plot(graph_widget, filtered_df, graph_model)
            except Exception as e:
                self._show_notification(f"Error updating plot: {str(e)}", notification_type="error")
                return

            self.vm.sync_derived_graph_properties(graph_model.graph_id, {
                'legend_properties': graph_widget.legend_properties,
                'legend_bbox': graph_widget.legend_bbox,
            })

            if customize_open_for_graph:
                self._refresh_open_customize_dialog(graph_widget)

            active_subwindow.setWindowTitle(self._graph_window_title(graph_model))
            self._sync_active_graph_toolbar()
        finally:
            self.vm.end_undo_batch()

    def _update_graph_list(self, graph_ids: list):
        """Update graph list."""
        current_id = self.cbb_graph_list.currentData()

        self.cbb_graph_list.blockSignals(True)

        self.cbb_graph_list.clear()
        for gid in graph_ids:
            graph = self.vm.get_graph(gid)
            if graph:
                display_name = graph.get_display_name()
                self.cbb_graph_list.addItem(display_name, gid)

        if current_id is not None:
            for i in range(self.cbb_graph_list.count()):
                if self.cbb_graph_list.itemData(i) == current_id:
                    self.cbb_graph_list.setCurrentIndex(i)
                    break

        self.cbb_graph_list.blockSignals(False)

    def _on_graph_selected_toolbar(self):
        """Handle graph selection."""
        index = self.cbb_graph_list.currentIndex()
        if index >= 0:
            graph_id = self.cbb_graph_list.currentData()
            if graph_id and graph_id in self.graph_widgets:
                _, _, sub_window = self.graph_widgets[graph_id]
                self.mdi_area.setActiveSubWindow(sub_window)
                sub_window.showNormal()

    def _on_minimize_all(self):
        """Minimize all windows."""
        for window in self.mdi_area.subWindowList():
            window.showMinimized()

    def _on_delete_all(self):
        """Delete all graphs."""
        reply = QMessageBox.question(
            self,
            "Delete All Graphs",
            "Are you sure you want to delete all graphs?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            graph_ids = list(self.graph_widgets.keys())
            self._teardown_all_graph_windows()

            self.vm.begin_undo_batch()
            try:
                for graph_id in graph_ids:
                    self.vm.delete_graph(graph_id)
            finally:
                self.vm.end_undo_batch()

            self._update_graph_list([])
            self._sync_active_graph_toolbar()

    def _clear_graph_toolbar_slot(self):
        """Unparent and hide all widgets currently inside the shared toolbar slot."""
        while self._graph_toolbar_slot_layout.count():
            item = self._graph_toolbar_slot_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)

    def _sync_active_graph_toolbar(self):
        """Reparent the active MDI subwindow's graph toolbar/action-button
        row into the shared toolbar slot."""
        gid = self._get_active_graph_id()
        if gid is None:
            self._clear_graph_toolbar_slot()
            return

        graph_widget = self.graph_widgets.get(gid, (None,))[0]
        if graph_widget is None:
            self._clear_graph_toolbar_slot()
            return

        container = getattr(graph_widget, 'toolbar_container', None)
        if container is None:
            self._clear_graph_toolbar_slot()
            return

        if (self._graph_toolbar_slot_layout.count() == 1 and
                self._graph_toolbar_slot_layout.itemAt(0).widget() is container):
            if not container.isVisible():
                container.show()
            return

        self._clear_graph_toolbar_slot()
        try:
            self._graph_toolbar_slot_layout.addWidget(container)
            container.show()
        except RuntimeError:
            pass

    def _on_subwindow_activated(self, sub_window):
        """Handle subwindow activation."""
        if not sub_window:
            self._sync_active_graph_toolbar()
            return

        gid = self._graph_id_for_subwindow(sub_window)
        graph_model = self.vm.get_graph(gid) if gid is not None else None
        if not graph_model:
            return
        graph_widget = self.graph_widgets[gid][0]

        self._sync_active_graph_toolbar()

        if getattr(self, '_last_synced_graph_id', None) == gid:
            return
        self._last_synced_graph_id = gid

        self._sync_gui_from_graph(graph_model)

        if self._customize_dialog is not None and self._customize_dialog.isVisible():
            self._customize_dialog.switch_graph(graph_widget, gid)

        self.cbb_graph_list.blockSignals(True)
        for i in range(self.cbb_graph_list.count()):
            if self.cbb_graph_list.itemData(i) == gid:
                self.cbb_graph_list.setCurrentIndex(i)
                break
        self.cbb_graph_list.blockSignals(False)

    def _sync_gui_from_graph(self, model):
        """Sync GUI from graph model."""
        self.cbb_plot_style.blockSignals(True)
        self.cbb_x.blockSignals(True)
        self.cbb_y.blockSignals(True)
        self.cbb_z.blockSignals(True)
        self.df_listbox.blockSignals(True)

        try:
            if model.df_name:
                if self.vm.selected_df_name != model.df_name:
                    self.vm.select_dataframe(model.df_name)

                for i in range(self.df_listbox.count()):
                    if self.df_listbox.item(i).text() == model.df_name:
                        self.df_listbox.setCurrentRow(i)
                        break

            for cbb_name, value in [('plot_style', model.plot_style), ('x', model.x), ('z', model.z or "None")]:
                cbb = getattr(self, f'cbb_{cbb_name}')
                idx = cbb.findText(value)
                if idx >= 0:
                    cbb.setCurrentIndex(idx)

            if model.y:
                idx = self.cbb_y.findText(model.y[0])
                if idx >= 0:
                    self.cbb_y.setCurrentIndex(idx)

            self.v_data_filter.set_filters(model.filters)

            y2_val = model.y2 or "None"
            idx = self.cbb_y2.findText(y2_val)
            if idx >= 0:
                self.cbb_y2.setCurrentIndex(idx)

            y3_val = model.y3 or "None"
            idx = self.cbb_y3.findText(y3_val)
            if idx >= 0:
                self.cbb_y3.setCurrentIndex(idx)

            x2_val = getattr(model, 'x2', None) or "None"
            idx = self.cbb_x2.findText(x2_val)
            if idx >= 0:
                self.cbb_x2.setCurrentIndex(idx)

            self._sync_side_panel_shared_fields(model)
        finally:
            self.cbb_plot_style.blockSignals(False)
            self.cbb_x.blockSignals(False)
            self.cbb_y.blockSignals(False)
            self.cbb_z.blockSignals(False)
            self.df_listbox.blockSignals(False)

    def _sync_side_panel_shared_fields(self, source, keys=None):
        """Sync side-panel controls whose MGraph fields are also editable via CustomizeGraphDialog."""
        fields = self._SIDE_PANEL_SHARED_FIELDS if keys is None else set(keys)
        if 'color_palette' in fields:
            palette = getattr(source, 'color_palette', None)
            if palette:
                idx = self.cbb_colormap.findText(palette)
                if idx >= 0:
                    self.cbb_colormap.setCurrentIndex(idx)

        text_controls = (
            ('plot_title', self.edit_plot_title),
            ('plot_subtitle', self.edit_plot_subtitle),
            ('xlabel', self.edit_xlabel),
            ('ylabel', self.edit_ylabel),
            ('zlabel', self.edit_zlabel),
            ('y2label', self.edit_y2label),
            ('y3label', self.edit_y3label),
            ('x2label', self.edit_x2label),
        )
        for field_name, line_edit in text_controls:
            if field_name in fields:
                line_edit.setText(getattr(source, field_name, None) or "")

        checkbox_controls = (
            ('y2logscale', self.cb_y2log),
            ('y3logscale', self.cb_y3log),
            ('x2logscale', self.cb_x2log),
        )
        for field_name, checkbox in checkbox_controls:
            if field_name in fields:
                checkbox.setChecked(bool(getattr(source, field_name, False)))

    def _collect_plot_config(self, include_labels: bool = True) -> dict:
        """Collect plot configuration."""
        plot_style = self.cbb_plot_style.currentText()
        z_value = self.cbb_z.currentText() if self.cbb_z.currentText() != "None" else None

        use_palette = plot_style in ['wafer', '2Dmap'] or z_value is not None

        return {
            'df_name': self.vm.selected_df_name,
            'plot_style': plot_style,
            'x': self.cbb_x.currentText(),
            'y': [self.cbb_y.currentText()],
            'z': z_value,
            'xlogscale': False,
            'ylogscale': False,
            'plot_title': (self.edit_plot_title.text() or None) if include_labels else None,
            'plot_subtitle': (self.edit_plot_subtitle.text() or None) if include_labels else None,
            'xlabel': (self.edit_xlabel.text() or None) if include_labels else None,
            'ylabel': (self.edit_ylabel.text() or None) if include_labels else None,
            'zlabel': (self.edit_zlabel.text() or None) if include_labels else None,
            'color_palette': self.cbb_colormap.currentText() if use_palette else 'jet',
            'wafer_size': float(self.cbb_wafer_size.currentText()),
            'legend_visible': True,
            'filters': self.v_data_filter.get_filters(),
            'y2': self.cbb_y2.currentText() if self.cbb_y2.currentText() != "None" else None,
            'y3': self.cbb_y3.currentText() if self.cbb_y3.currentText() != "None" else None,
            'x2': self.cbb_x2.currentText() if self.cbb_x2.currentText() != "None" else None,
            'y2label': (self.edit_y2label.text() or None) if include_labels else None,
            'y3label': (self.edit_y3label.text() or None) if include_labels else None,
            'x2label': (self.edit_x2label.text() or None) if include_labels else None,
            'y2logscale': self.cb_y2log.isChecked(),
            'y3logscale': self.cb_y3log.isChecked(),
            'x2logscale': self.cb_x2log.isChecked(),
        }

    def _configure_graph_from_model(self, graph_widget: VGraph, model):
        """Copy every MGraph field onto the widget generically."""
        for key, value in vars(model).items():
            if key == 'graph_id':
                continue
            setattr(graph_widget, key, value)

        graph_widget.y = model.y.copy() if model.y else []
        graph_widget.legend_properties = model.legend_properties.copy() if model.legend_properties else []
        graph_widget.axis_breaks = copy.deepcopy(model.axis_breaks) if model.axis_breaks else {'x': None, 'y': None}
        graph_widget.annotations = copy.deepcopy(model.annotations) if model.annotations else []
        graph_widget.filters = copy.deepcopy(model.filters) if model.filters else []
        graph_widget.spines_visible = (
            copy.deepcopy(model.spines_visible) if model.spines_visible
            else {'top': True, 'right': True, 'bottom': True, 'left': True}
        )

        edge_c = model.scatter_edgecolor
        if not edge_c or not isinstance(edge_c, str) or edge_c.strip() in ("", "None", "none", "null"):
            edge_c = 'black'
        graph_widget.scatter_edgecolor = edge_c

    def _render_plot(self, graph_widget: VGraph, filtered_df, model):
        """Render plot."""
        graph_widget.plot(filtered_df)

    @staticmethod
    def _graph_window_title(model) -> str:
        """Window title for a graph: 'id-style: [x] vs [y] - [z]'."""
        title = f"{model.graph_id}-{model.plot_style}: [{model.x}] vs [{model.y[0] if model.y else 'None'}]"
        if model.z:
            title += f" - [{model.z}]"
        return title

    def _create_mdi_subwindow(self, model) -> QMdiSubWindow:
        """Create MDI subwindow."""
        sub_window = MdiSubWindow(
            graph_id=model.graph_id,
            mdi_area=self.mdi_area
        )

        sub_window.setWindowTitle(self._graph_window_title(model))
        sub_window.resize(model.plot_width, model.plot_height)

        shortcut = QShortcut(QKeySequence(Qt.Key_Escape), sub_window)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(lambda: None)

        sub_window.closed.connect(lambda gid=model.graph_id: self._on_graph_closed(gid))

        return sub_window

    def _on_graph_closed(self, graph_id: int):
        """Handle graph closing."""
        if self._customize_dialog is not None and self._customize_dialog.graph_id == graph_id:
            self._customize_dialog.close()
            self._customize_dialog = None

        if graph_id in self.graph_widgets:
            del self.graph_widgets[graph_id]

        self.vm.delete_graph(graph_id)

        self._update_graph_list(self.vm.get_graph_ids())
        self._sync_active_graph_toolbar()

    def _teardown_all_graph_windows(self):
        """Close the customize dialog and every graph subwindow, then clear `graph_widgets`."""
        if self._customize_dialog is not None:
            self._customize_dialog.close()
            self._customize_dialog = None

        for sub_window in self.mdi_area.subWindowList():
            sub_window.closed.disconnect()
            self.mdi_area.removeSubWindow(sub_window)
            sub_window.close()

        self.graph_widgets.clear()
