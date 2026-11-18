"""DataFrame management, column/slot selectors, and multi-wafer batch plotting mixin for VWorkspaceGraphs."""

import os

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QCheckBox, QDialog, QVBoxLayout

from spectroview.view.components.lists.v_dataframe_table import VDataframeTable


class GraphsDataFramesMixin:
    """Mixin managing DataFrame selection, filtering, inspection, and wafer slot controls."""

    def _on_plot_style_changed(self, plot_style: str):
        """Handle plot style change."""
        if plot_style == 'wafer':
            x_index = self.cbb_x.findText('X')
            y_index = self.cbb_y.findText('Y')

            if x_index >= 0:
                self.cbb_x.setCurrentIndex(x_index)
            if y_index >= 0:
                self.cbb_y.setCurrentIndex(y_index)

    def _on_df_selected(self):
        """Handle DataFrame selection."""
        current_item = self.df_listbox.currentItem()
        if current_item:
            df_name = current_item.text()
            self.vm.select_dataframe(df_name)

    def _on_view_df(self):
        """View DataFrame in table, or open source file if Ctrl is held."""
        current_item = self.df_listbox.currentItem()
        if not current_item:
            self._show_notification("Please select a DataFrame to view.", notification_type="warning")
            return

        df_name = current_item.text()
        modifiers = QApplication.keyboardModifiers()

        if modifiers & Qt.ControlModifier:
            if df_name in self.vm.dataframe_sources:
                source_path = self.vm.dataframe_sources[df_name]
                try:
                    file_url = QUrl.fromLocalFile(source_path)
                    if QDesktopServices.openUrl(file_url):
                        self.vm.notify.emit(f"Opening file: {os.path.basename(source_path)}")
                    else:
                        self._show_notification(f"Could not open file: {source_path}", notification_type="error")
                except Exception as e:
                    self._show_notification(f"Could not open file: {str(e)}", notification_type="error")
            else:
                self._show_notification(f"No source file found for '{df_name}'.", notification_type="warning")
            return

        df = self.vm.get_dataframe(df_name)

        if df is None or df.empty:
            self._show_notification("Selected DataFrame is empty.", notification_type="warning")
            return

        filters = self.v_data_filter.get_filters()
        active_filters = [f for f in filters if f.get("state", False)]

        if active_filters:
            df = self.vm.apply_filters(df_name, filters)
            if df is None or df.empty:
                self._show_notification("No data after applying filters.", notification_type="warning")
                return

        dialog = QDialog(self)
        title = f"DataFrame: {df_name}"
        if active_filters:
            title += f" (with {len(active_filters)} filter(s))"
        dialog.setWindowTitle(title)
        dialog.resize(800, 600)

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)

        table_view = VDataframeTable(layout)
        table_view.show(df, fill_colors=False)

        dialog.setLayout(layout)
        dialog.exec()

    def _on_remove_df(self):
        """Remove DataFrame."""
        current_item = self.df_listbox.currentItem()
        if current_item:
            df_name = current_item.text()
            self.vm.remove_dataframe(df_name)

    def _on_save_df(self):
        """Save DataFrame to Excel or CSV."""
        current_item = self.df_listbox.currentItem()
        if current_item:
            df_name = current_item.text()
            self.vm.save_dataframe(df_name)

    def _on_refresh_df(self):
        """Refresh DataFrame from source file."""
        current_item = self.df_listbox.currentItem()
        if not current_item:
            self._show_notification("Please select a DataFrame to refresh.", notification_type="warning")
            return

        df_name = current_item.text()

        current_graph_id = None
        current_graph_index = self.cbb_graph_list.currentIndex()
        if current_graph_index >= 0:
            current_graph_id = self.cbb_graph_list.currentData()

        success = self.vm.refresh_dataframe(df_name)

        if success:
            self.vm.notify.emit(f"DataFrame '{df_name}' refreshed successfully")

            if current_graph_id:
                graph_model = self.vm.get_graph(current_graph_id)
                if graph_model and graph_model.df_name == df_name:
                    self._sync_gui_from_graph(graph_model)
        else:
            self._show_notification(
                f"Could not refresh DataFrame '{df_name}'. Source file may not exist or be accessible.",
                notification_type="warning",
            )

    def _on_apply_filters(self):
        """Apply filters."""
        if not self.vm.selected_df_name:
            return

        filters = self.v_data_filter.get_filters()
        self.vm.apply_filters(self.vm.selected_df_name, filters)

    def _update_df_list(self, df_names: list):
        """Update DataFrame list from ViewModel."""
        self.df_listbox.blockSignals(True)

        current_selection = self.df_listbox.currentItem()
        selected_name = current_selection.text() if current_selection else None

        self.df_listbox.clear()
        self._has_df_placeholder = False

        for name in df_names:
            self.df_listbox.addItem(name)

        if selected_name:
            items = self.df_listbox.findItems(selected_name, Qt.MatchExactly)
            if items:
                self.df_listbox.setCurrentItem(items[0])
        elif df_names:
            self.df_listbox.setCurrentRow(0)

        self.df_listbox.blockSignals(False)

        self._on_df_selected()

        if len(df_names) == 0:
            self._update_df_placeholder()

    def _update_column_combos(self, columns: list):
        """Update column comboboxes."""
        self.cbb_x.clear()
        self.cbb_y.clear()
        self.cbb_z.clear()
        self.cbb_y2.clear()
        self.cbb_y3.clear()
        self.cbb_x2.clear()

        self.cbb_x.addItems(columns)
        self.cbb_y.addItems(columns)
        self.cbb_z.addItem("None")
        self.cbb_z.addItems(columns)

        self.cbb_y2.addItem("None")
        self.cbb_y2.addItems(columns)
        self.cbb_y3.addItem("None")
        self.cbb_y3.addItems(columns)
        self.cbb_x2.addItem("None")
        self.cbb_x2.addItems(columns)

        self.v_data_filter.update_column_list(columns)

    def _update_slot_selector(self, columns: list):
        """Update slot selector."""
        saved_states = {}
        for cb in self.slot_checkboxes:
            saved_states[cb.text()] = cb.isChecked()

        for cb in self.slot_checkboxes:
            cb.deleteLater()
        self.slot_checkboxes.clear()

        while self.slot_grid_layout.count():
            item = self.slot_grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self.vm.selected_df_name:
            self.slot_selector_widget.setVisible(False)
            self.btn_add_multi_wafer.setVisible(False)
            return

        if not self.vm.has_slot_column(self.vm.selected_df_name):
            self.slot_selector_widget.setVisible(False)
            self.btn_add_multi_wafer.setVisible(False)
            return

        unique_slots = self.vm.get_unique_slots(self.vm.selected_df_name)

        if not unique_slots:
            self.slot_selector_widget.setVisible(False)
            self.btn_add_multi_wafer.setVisible(False)
            return

        self.slot_selector_widget.setVisible(True)
        self.select_all_checkbox.setVisible(True)
        self.btn_add_multi_wafer.setVisible(True)

        row, col = 0, 0
        for slot in unique_slots:
            cb = QCheckBox(str(slot))
            slot_str = str(slot)
            cb.setChecked(saved_states.get(slot_str, False))
            cb.stateChanged.connect(self._on_slot_checkbox_changed)
            self.slot_grid_layout.addWidget(cb, row, col)
            self.slot_checkboxes.append(cb)

            col += 1
            if col >= 8:
                col = 0
                row += 1

    def _on_select_all_slots(self, state):
        """Toggle all slots."""
        checked = bool(state)
        for cb in self.slot_checkboxes:
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)

    def _on_slot_checkbox_changed(self):
        """Update select all checkbox."""
        all_checked = all(cb.isChecked() for cb in self.slot_checkboxes)
        self.select_all_checkbox.blockSignals(True)
        self.select_all_checkbox.setChecked(all_checked)
        self.select_all_checkbox.blockSignals(False)

    def _on_plot_multi_wafer(self):
        """Create multi-wafer plots."""
        current_filters = self.v_data_filter.get_filters()

        checked_slots = [int(float(cb.text())) for cb in self.slot_checkboxes if cb.isChecked()]

        if not checked_slots:
            self._show_notification("Please select at least one slot.", notification_type="warning")
            return

        if not self._validate_plot_request():
            return

        self.vm.begin_undo_batch()
        try:
            wafer_index = self.cbb_plot_style.findText('wafer')
            if wafer_index >= 0:
                self.cbb_plot_style.setCurrentIndex(wafer_index)

            plot_config = self._collect_plot_config(include_labels=False)
            plot_config['plot_style'] = 'wafer'
            self._apply_default_style_to_config(plot_config)

            created_graphs = self.vm.create_multi_wafer_graphs(
                self.vm.selected_df_name,
                checked_slots,
                plot_config,
                current_filters
            )

            successfully_created = 0
            for graph_model in created_graphs:
                filtered_df = self.vm.apply_filters(self.vm.selected_df_name, graph_model.filters)

                def _on_render_error(e, graph_model=graph_model):
                    slot_expr = graph_model.filters[-1].get('expression', 'unknown') if graph_model.filters else 'unknown'
                    self._show_notification(
                        f"Error rendering plot for {slot_expr}: {str(e)}",
                        notification_type="warning",
                    )

                if self._build_graph_widget(graph_model, filtered_df, _on_render_error) is None:
                    continue
                successfully_created += 1

            self._update_graph_list(self.vm.get_graph_ids())

            if successfully_created > 0:
                self.vm.notify.emit(f"Created {successfully_created} wafer plot(s)")
        finally:
            self.vm.end_undo_batch()
