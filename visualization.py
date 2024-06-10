import os
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
import dill
import json

from common import view_df, show_alert
from common import PLOT_STYLES, PALETTE
from common import Graph, Filter

from PySide6.QtWidgets import QApplication, QFileDialog, QDialog, QVBoxLayout, \
    QLineEdit, QListWidgetItem, QMdiSubWindow, QCheckBox, QMdiArea, \
    QSizePolicy, \
    QMessageBox
from PySide6.QtCore import Qt, QFileInfo, QTimer, QObject, Signal


class Visualization(QDialog):
    """Class to GUI and callbacks"""

    def __init__(self, settings, ui, common):
        super().__init__()
        self.ui = ui
        self.settings = settings
        self.common = common
        self.setWindowTitle("Graph Plot")

        self.ui.btn_clear_env.clicked.connect(self.clear_env)

        # DATAFRAME
        self.original_dfs = {}
        self.sel_df = None
        self.ui.btn_open_dfs.clicked.connect(self.open_dfs)
        self.ui.btn_view_df_3.clicked.connect(self.show_df)
        self.ui.dfs_listbox.itemSelectionChanged.connect(self.update_gui)
        self.ui.btn_remove_df_2.clicked.connect(self.remove_df)
        self.ui.btn_save_df_2.clicked.connect(self.save_df_to_excel)

        # FILTER
        self.filter = Filter(self.ui.filter_query, self.ui.listbox_filters,
                             self.sel_df)
        self.ui.filter_query.returnPressed.connect(self.filter.add_filter)
        self.ui.btn_add_filter_4.clicked.connect(self.filter.add_filter)
        self.ui.btn_remove_filters_4.clicked.connect(self.filter.remove_filter)
        self.ui.btn_apply_filters_4.clicked.connect(self.apply_filters)
        self.filtered_df = None

        # GRAPH
        self.sub_windows = []
        self.plots = {}
        self.graph_id = 0  # Initialize graph number
        self.ui.btn_add_graph.clicked.connect(self.add_graph)
        self.ui.btn_upd_graph.clicked.connect(self.update_graph)
        # GRAPH: add 2nd and 3rd lines for the current ax
        self.ui.btn_add_y12.clicked.connect(self.add_y12)
        self.ui.btn_add_y13.clicked.connect(self.add_y13)
        # GRAPH: add twin axis (second and third y axis)
        self.ui.btn_add_y2.clicked.connect(self.add_y2)
        self.ui.btn_remove_y2.clicked.connect(self.remove_y2)
        self.ui.btn_add_y3.clicked.connect(self.add_y3)
        self.ui.btn_remove_y3.clicked.connect(self.remove_y3)

        self.ui.btn_copy_graph.clicked.connect(self.copy_fig_to_clb)
        self.ui.cbb_palette.addItems(PALETTE)
        self.ui.cbb_plotstyle.addItems(PLOT_STYLES)

        self.ui.btn_adjust_dpi.clicked.connect(self.adjust_dpi)
        # self.ui.spb_dpi.valueChanged.connect(self.adjust_dpi)

        # Track selected sub-window
        self.ui.mdiArea.subWindowActivated.connect(self.on_selected_graph)
        self.ui.cbb_graph_list.currentIndexChanged.connect(
            self.select_sub_window_from_combo_box)

        # SAVE / LOAD
        self.ui.btn_save_work.clicked.connect(self.save)
        self.ui.btn_load_work.clicked.connect(self.load)
        self.ui.btn_minimize_all.clicked.connect(self.minimize_all_graph)
        # self.ui.btn_maximize_all.clicked.connect(self.restore_all)

    def add_graph(self, df_name=None, filters=None):
        """Plot new graph"""
        # Get the current dataframe (filtered or not)
        if filters:
            current_filters = filters
        else:
            current_filters = self.filter.get_current_filters()
        if df_name:
            df = self.original_dfs[df_name]
            filtered_df = self.apply_filters(df, current_filters)
        else:
            df_name = self.ui.dfs_listbox.currentItem().text()
            filtered_df = self.apply_filters(self.sel_df, current_filters)

        x = self.ui.cbb_x_2.currentText()
        y = self.ui.cbb_y_2.currentText()
        z = self.ui.cbb_z_2.currentText()

        # Get available graph IDs considering any vacancies in the list
        available_ids = [i for i in range(1, len(self.plots) + 2) if
                         i not in self.plots]
        # Use the smallest available ID or assign a new one
        if available_ids:
            graph_id = min(available_ids)
        else:
            graph_id = len(self.plots) + 1

        # Create an instance of the Graph class
        graph = Graph(graph_id=graph_id)

        graph.plot_style = self.ui.cbb_plotstyle.currentText()
        title = self.ui.lbl_plot_title.text()
        graph.plot_title = title if title != "None" else None
        graph.color_palette = self.ui.cbb_palette.currentText()
        graph.df_name = df_name
        graph.filters = current_filters
        graph.x = x
        if len(graph.y) == 0:
            graph.y.append(y)
        else:
            graph.y[0] = y
        graph.z = z if z != "None" else None
        graph.dpi = float(self.ui.spb_dpi.text())
        graph.wafer_size = float(self.ui.lbl_wafersize.text())
        graph.wafer_stats = self.ui.cb_wafer_stats.isChecked()
        # Pass legend & grid settings
        graph.legend_visible = self.ui.cb_legend_visible.isChecked()
        graph.legend_outside = self.ui.cb_legend_outside.isChecked()
        graph.grid = self.ui.cb_grid.isChecked()
        graph.trendline_order = float(self.ui.spb_trendline_oder.text())
        graph.show_trendline_eq = self.ui.cb_trendline_eq.isChecked()
        graph.show_bar_plot_error_bar = self.ui.cb_show_err_bar_plot.isChecked()
        graph.join_for_point_plot = self.ui.cb_join_for_point_plot.isChecked()

        # Plot the graph
        graph.create_plot_widget(graph.dpi)
        graph.plot(filtered_df)

        # Store the plot in the dictionary with graph_id as key
        self.plots[graph.graph_id] = graph

        # Create a QDialog to hold the Graph instance
        graph_dialog = QDialog(self)
        graph_dialog.setWindowTitle(
            f"{graph.graph_id}-{graph.plot_style}_plot: [{x}] - [{y}] - "
            f"[{z}]")
        layout = QVBoxLayout()
        layout.addWidget(graph)
        graph_dialog.setLayout(layout)

        # Add the QDialog to a QMdiSubWindow
        sub_window = MdiSubWindow(graph_id)
        sub_window.setWidget(graph_dialog)
        # delete graph when sub windows is closed
        sub_window.closed.connect(self.delete_graph)
        # Set initial size of the QMdiSubWindow
        sub_window.resize(graph.plot_width, graph.plot_height)

        self.ui.mdiArea.addSubWindow(sub_window)
        sub_window.show()
        QTimer.singleShot(100, self.update_graph)
        self.populate_graph_combo_box()
        # Add sub-window to the list
        self.sub_windows.append(sub_window)

    def update_graph(self):
        """ Update the selected graph with new properties"""
        graph, graph_dialog, sub_window = self.get_sel_graph()
        if graph:
            # Retrieve the current size of the sub_window
            sub_window_size = sub_window.size()
            graph.plot_width = sub_window_size.width()
            graph.plot_height = sub_window_size.height()

            plot_style = self.ui.cbb_plotstyle.currentText()
            plot_title = self.ui.lbl_plot_title.text()
            x = self.ui.cbb_x_2.currentText()
            y = self.ui.cbb_y_2.currentText()
            z = self.ui.cbb_z_2.currentText()
            xlabel = self.ui.lbl_xlabel.text()
            ylabel = self.ui.lbl_ylabel.text()
            y2label = self.ui.lbl_y2label.text()
            y3label = self.ui.lbl_y3label.text()
            zlabel = self.ui.lbl_zlabel.text()
            xmin = self.ui.xmin_2.text()
            ymin = self.ui.ymin_2.text()
            xmax = self.ui.xmax_2.text()
            ymax = self.ui.ymax_2.text()
            zmin = self.ui.zmin_2.text()
            zmax = self.ui.zmax_2.text()

            palette = self.ui.cbb_palette.currentText()
            x_rot = float(self.ui.x_rot.text())
            current_filters = self.filter.get_current_filters()

            # Apply values for "graph" object
            graph.df_name = self.ui.dfs_listbox.currentItem().text()
            graph.filters = current_filters
            graph.x = x
            if len(graph.y) == 0:
                graph.y.append(y)
            else:
                graph.y[0] = y
            graph.z = z if z != "None" else None

            graph.xmin = xmin
            graph.xmax = xmax
            graph.ymin = ymin
            graph.ymax = ymax
            graph.zmin = zmin
            graph.zmax = zmax
            graph.xlabel = xlabel
            graph.ylabel = ylabel
            graph.y2label = y2label
            graph.y3label = y3label
            graph.zlabel = zlabel
            graph.x_rot = x_rot
            graph.plot_style = plot_style
            graph.plot_title = plot_title
            graph.color_palette = palette
            graph.wafer_size = float(self.ui.lbl_wafersize.text())
            graph.wafer_stats = self.ui.cb_wafer_stats.isChecked()
            graph.legend_visible = self.ui.cb_legend_visible.isChecked()
            graph.legend_outside = self.ui.cb_legend_outside.isChecked()
            graph.grid = self.ui.cb_grid.isChecked()
            graph.dpi = float(self.ui.spb_dpi.text())
            graph.trendline_order = float(self.ui.spb_trendline_oder.text())
            graph.show_trendline_eq = self.ui.cb_trendline_eq.isChecked()
            graph.show_bar_plot_error_bar = \
                self.ui.cb_show_err_bar_plot.isChecked()
            graph.join_for_point_plot = \
                self.ui.cb_join_for_point_plot.isChecked()

            graph_dialog.setWindowTitle(
                f"{graph.graph_id}-{graph.plot_style}_plot: [{x}] - [{y}] - ["
                f"{z}]")

            if plot_style == 'wafer':
                graph.create_plot_widget(graph.dpi,
                                         graph.graph_layout)
                graph.plot(self.filtered_df)
            else:
                graph.plot(self.filtered_df)

    def add_y12(self):
        """Add a second line in the current plot ax"""
        graph, graph_dialog, sub_window = self.get_sel_graph()
        y12 = self.ui.cbb_y12.currentText()
        if len(graph.y) == 1:
            graph.y.append(y12)
        else:
            graph.y[1] = y12
        self.update_graph()

    def add_y13(self):
        """Add a 3rd line in the current plot ax"""
        graph, graph_dialog, sub_window = self.get_sel_graph()
        y13 = self.ui.cbb_y13.currentText()
        if len(graph.y) == 2:
            graph.y.append(y13)
        else:
            graph.y[2] = y13
        self.update_graph()

    def add_y2(self):
        """Add 2nd Y axis for the selected plot"""
        graph, graph_dialog, sub_window = self.get_sel_graph()
        if graph.plot_style == 'line' or graph.plot_style == 'point' or \
                graph.plot_style == 'scatter':
            y2 = self.ui.cbb_y2_2.currentText()
            y2min = self.ui.y2min_2.text()
            y2max = self.ui.y2max_2.text()
            graph.y2 = y2
            graph.y2label = y2
            graph.y2min = y2min
            graph.y2max = y2max
            self.update_graph()
        else:
            pass

    def add_y3(self):
        """Add 2nd Y axis for the selected plot"""
        graph, graph_dialog, sub_window = self.get_sel_graph()
        if graph.plot_style == 'line' or graph.plot_style == 'point' or \
                graph.plot_style == 'scatter':
            y3 = self.ui.cbb_y3_2.currentText()
            y3min = self.ui.y3min_2.text()
            y3max = self.ui.y3max_2.text()
            graph.y3 = y3
            graph.y3label = y3
            graph.y3min = y3min
            graph.y3max = y3max
            self.update_graph()
        else:
            pass

    def remove_y2(self):
        """Remove the 2nd Y axis from the selected plot"""
        graph, graph_dialog, sub_window = self.get_sel_graph()
        if graph.ax2:
            graph.ax2.remove()  # Remove the ax2 instance
            graph.ax2 = None

        # Clear y2-related attributes
        graph.y2 = None
        graph.y2label = None
        graph.y2min = None
        graph.y2max = None

        self.update_graph()

    def remove_y3(self):
        """Remove the 2nd Y axis from the selected plot"""
        graph, graph_dialog, sub_window = self.get_sel_graph()
        if graph.ax3:
            graph.ax3.remove()  # Remove the ax2 instance
            graph.ax3 = None

        # Clear y2-related attributes
        graph.y3 = None
        graph.y3label = None
        graph.y3min = None
        graph.y3max = None

        self.update_graph()

    def adjust_dpi(self):
        graph, graph_dialog, sub_window = self.get_sel_graph()
        dpi = float(self.ui.spb_dpi.text())
        if graph:
            # Recreate the plot widget with new DPI
            graph.create_plot_widget(dpi, graph.graph_layout)
            QTimer.singleShot(100, self.update_graph)

    def on_selected_graph(self, sub_window):
        """Reflect all properties of selected graph object to GUI"""
        graph, graph_dialog, sub_window = self.get_sel_graph()

        if graph:
            # Plot style
            plot_style = graph.plot_style
            items = [self.ui.cbb_plotstyle.itemText(i) for i in
                     range(self.ui.cbb_plotstyle.count())]
            if plot_style in items:
                self.ui.cbb_plotstyle.setCurrentText(plot_style)

            # Reflect df_name in the listbox
            current_items = [self.ui.dfs_listbox.item(i).text() for i in
                             range(self.ui.dfs_listbox.count())]
            if graph.df_name not in current_items:
                self.ui.dfs_listbox.addItem(graph.df_name)
            else:
                index = current_items.index(graph.df_name)
                self.ui.dfs_listbox.setCurrentRow(index)

            # Reflect filter's states in the listbox
            self.reflect_filters_to_gui(graph)

            # Update combobox selections
            x = self.ui.cbb_x_2.findText(graph.x)
            y = self.ui.cbb_y_2.findText(graph.y[0])
            y2 = self.ui.cbb_y_2.findText(graph.y2)
            y3 = self.ui.cbb_y_2.findText(graph.y3)
            z = self.ui.cbb_z_2.findText(graph.z)
            self.ui.cbb_x_2.setCurrentIndex(x if x != -1 else 0)
            self.ui.cbb_y_2.setCurrentIndex(y if y != -1 else 0)
            self.ui.cbb_y2_2.setCurrentIndex(y2 if y2 != -1 else 0)
            self.ui.cbb_y3_2.setCurrentIndex(y3 if y3 != -1 else 0)
            self.ui.cbb_z_2.setCurrentIndex(z if z != -1 else 0)

            # WAFER
            self.ui.lbl_wafersize.setText(str(graph.wafer_size))
            self.ui.cb_wafer_stats.setChecked(graph.wafer_stats)

            # Rotation x label:
            self.ui.x_rot.setValue(graph.x_rot)
            # Reflect Titles:
            self.ui.lbl_plot_title.setText(graph.plot_title)
            self.ui.lbl_xlabel.setText(graph.xlabel)
            self.ui.lbl_ylabel.setText(graph.ylabel)
            self.ui.lbl_y2label.setText(graph.y2label)
            self.ui.lbl_y3label.setText(graph.y3label)
            self.ui.lbl_zlabel.setText(graph.zlabel)

            # Reflect limits:
            self.ui.xmin_2.setText(graph.xmin)
            self.ui.xmax_2.setText(graph.xmax)
            self.ui.ymin_2.setText(graph.ymin)
            self.ui.ymax_2.setText(graph.ymax)
            self.ui.y2min_2.setText(graph.y2min)
            self.ui.y2max_2.setText(graph.y2max)
            self.ui.y3min_2.setText(graph.y3min)
            self.ui.y3max_2.setText(graph.y3max)
            self.ui.zmax_2.setText(graph.zmax)
            self.ui.zmin_2.setText(graph.zmin)

            # Reflect legend status
            self.ui.cb_legend_visible.setChecked(graph.legend_visible)
            self.ui.cb_legend_outside.setChecked(graph.legend_outside)

            # Grid
            self.ui.cb_grid.setChecked(graph.grid)

            # Reflect Color palette
            color_palette = graph.color_palette
            combo_items = [self.ui.cbb_palette.itemText(i) for i in
                           range(self.ui.cbb_palette.count())]
            if color_palette in combo_items:
                self.ui.cbb_palette.setCurrentText(color_palette)

            # Reflect DPI
            self.ui.spb_dpi.setValue(graph.dpi)

            # Trendline
            self.ui.spb_trendline_oder.setValue(graph.trendline_order)
            self.ui.cb_trendline_eq.setChecked(graph.show_trendline_eq)

            # Show error bar for bar_plot
            self.ui.cb_show_err_bar_plot.setChecked(
                graph.show_bar_plot_error_bar)
            self.ui.cb_join_for_point_plot.setChecked(
                graph.join_for_point_plot)

            # self.apply_filters()

    def reflect_filters_to_gui(self, sel_graph):
        """Reflect the filters of a graph object and their states to the df
        listbox"""
        # Clear the existing items and uncheck them
        for index in range(self.ui.listbox_filters.count()):
            item = self.ui.listbox_filters.item(index)
            if isinstance(item, QListWidgetItem):
                widget = self.ui.listbox_filters.itemWidget(item)
                if isinstance(widget, QCheckBox):
                    widget.setChecked(False)

        for filter_info in sel_graph.filters:
            filter_expression = filter_info["expression"]
            filter_state = filter_info["state"]
            # Check if the filter expression already exists in the listbox
            existing_item = None
            for index in range(self.ui.listbox_filters.count()):
                item = self.ui.listbox_filters.item(index)
                if isinstance(item, QListWidgetItem):
                    widget = self.ui.listbox_filters.itemWidget(item)
                    if isinstance(widget,
                                  QCheckBox) and widget.text() == \
                            filter_expression:
                        existing_item = item
                        break
            # Update the state if the filter expression already exists,
            # otherwise add a new item
            if existing_item:
                checkbox = self.ui.listbox_filters.itemWidget(existing_item)
                checkbox.setChecked(filter_state)
            else:
                item = QListWidgetItem()
                checkbox = QCheckBox(filter_expression)
                checkbox.setChecked(filter_state)
                item.setSizeHint(checkbox.sizeHint())
                self.ui.listbox_filters.addItem(item)
                self.ui.listbox_filters.setItemWidget(item, checkbox)

    def open_dfs(self, dfs=None, fnames=None):
        if self.original_dfs is None:
            self.original_dfs = {}
        if dfs:
            self.original_dfs = dfs
        else:
            if fnames is None:
                # Initialize the last used directory from QSettings
                last_dir = self.settings.value("last_directory", "/")
                options = QFileDialog.Options()
                options |= QFileDialog.ReadOnly
                fnames, _ = QFileDialog.getOpenFileNames(
                    self.ui.tabWidget, "Open dataframe(s)", last_dir,
                    "Excel Files (*.xlsx)", options=options)
                # Load RAW spectra data from CSV files
            if fnames:
                last_dir = QFileInfo(fnames[0]).absolutePath()
                self.settings.setValue("last_directory", last_dir)
                for fnames in fnames:
                    fnames = Path(fnames)
                    fname = fnames.stem  # get fname w/o extension
                    extension = fnames.suffix.lower()
                    if extension == '.xlsx':
                        excel_file = pd.ExcelFile(fnames)
                        sheet_names = excel_file.sheet_names
                        for sheet_name in sheet_names:
                            # Remove spaces within sheet_names
                            sheet_name_cleaned = sheet_name.replace(" ", "")
                            df_name = f"{fname}_{sheet_name_cleaned}"
                            self.original_dfs[df_name] = pd.read_excel(
                                excel_file, sheet_name=sheet_name)
                    else:
                        pass
        self.update_dfs_list()

    def update_dfs_list(self):
        """ To update the dataframe listbox"""
        current_row = self.ui.dfs_listbox.currentRow()
        self.ui.dfs_listbox.clear()
        df_names = list(self.original_dfs.keys())
        for df_name in df_names:
            item = QListWidgetItem(df_name)
            self.ui.dfs_listbox.addItem(item)
        item_count = self.ui.dfs_listbox.count()
        # Management of selecting item of listbox
        if current_row >= item_count:
            current_row = item_count - 1
        if current_row >= 0:
            self.ui.dfs_listbox.setCurrentRow(current_row)
        else:
            if item_count > 0:
                self.ui.dfs_listbox.setCurrentRow(0)

    def update_gui(self):
        """To update GUI """
        self.show_df_in_gui()
        self.update_cbb()
        self.sel_df = self.get_sel_df()

    def update_cbb(self):
        """Populate columns of selected data to comboboxes"""
        sel_df = self.get_sel_df()
        if sel_df is not None:
            columns = sel_df.columns.tolist()
            self.ui.cbb_x_2.clear()
            self.ui.cbb_y_2.clear()
            self.ui.cbb_y12.clear()
            self.ui.cbb_y13.clear()
            self.ui.cbb_y2_2.clear()
            self.ui.cbb_y3_2.clear()
            self.ui.cbb_z_2.clear()
            self.ui.cbb_x_2.addItem("None")
            self.ui.cbb_y_2.addItem("None")
            self.ui.cbb_y12.addItem("None")
            self.ui.cbb_y13.addItem("None")
            self.ui.cbb_y2_2.addItem("None")
            self.ui.cbb_y3_2.addItem("None")
            self.ui.cbb_z_2.addItem("None")
            for column in columns:
                self.ui.cbb_x_2.addItem(column)
                self.ui.cbb_y_2.addItem(column)
                self.ui.cbb_y12.addItem(column)
                self.ui.cbb_y13.addItem(column)
                self.ui.cbb_y2_2.addItem(column)
                self.ui.cbb_y3_2.addItem(column)
                self.ui.cbb_z_2.addItem(column)

    def copy_fig_to_clb(self):
        """Copy the selected figure to clipboard"""
        sel_graph, graph_dialog, sub_window = self.get_sel_graph()
        self.common.copy_fig_to_clb(canvas=sel_graph.canvas)

    def get_sel_graph(self):
        """Get the canvas of the selected sub window"""
        try:
            sel_graph = None
            graph_dialog = None
            sub_window = self.ui.mdiArea.activeSubWindow()
            if sub_window:
                graph_dialog = sub_window.widget()
                if graph_dialog:
                    graph = graph_dialog.layout().itemAt(0).widget()
                    if graph:
                        sel_graph = graph
        except Exception as e:
            print("An error occurred:", e)
        return sel_graph, graph_dialog, sub_window

    def get_sel_df(self):
        """Get the current selected df among the df within 'dfr' dict"""
        sel_item = self.ui.dfs_listbox.currentItem()
        if sel_item is not None:
            sel_df_name = sel_item.text()
            if sel_df_name in self.original_dfs:
                self.sel_df = self.original_dfs[sel_df_name]
            else:
                self.sel_df = None  # Return None if the dataframe doesn't exist
        else:
            self.sel_df = None  # Return None if no item is selected
        return self.sel_df

    def remove_df(self):
        """Remove a dataframe from the listbox and self.original_dfs"""
        sel_item = self.ui.dfs_listbox.currentItem()
        sel_df_name = sel_item.text()
        if sel_df_name in self.original_dfs:
            del self.original_dfs[sel_df_name]

        # Remove from listbox
        items = self.ui.dfs_listbox.findItems(sel_df_name, Qt.MatchExactly)
        if items:
            for item in items:
                row = self.ui.dfs_listbox.row(item)
                self.ui.dfs_listbox.takeItem(row)

    def save_df_to_excel(self):
        """Functon to save fitted results in an excel file"""
        last_dir = self.settings.value("last_directory", "/")
        save_path, _ = QFileDialog.getSaveFileName(
            self.ui.tabWidget, "Save DF fit results", last_dir,
            "Excel Files (*.xlsx)")
        if save_path:
            sel_df = self.get_sel_df()
            if not sel_df.empty:
                sel_df.to_excel(save_path, index=False)
                QMessageBox.information(
                    self.ui.tabWidget, "Success",
                    "DataFrame saved successfully.")
            else:
                QMessageBox.warning(
                    self.ui.tabWidget, "Warning",
                    "DataFrame is empty. Nothing to save.")

    def show_df(self):
        """To view selected dataframe"""
        current_filters = self.filter.get_current_filters()
        current_df = self.apply_filters(self.sel_df, current_filters)
        if current_df is not None:
            view_df(self.ui.tabWidget, current_df)
        else:
            show_alert("No fit dataframe to display")

    def show_df_in_gui(self):
        current_filters = self.filter.get_current_filters()
        current_df = self.apply_filters(self.sel_df, current_filters)
        if self.filtered_df is not None:  # Check if filtered_df is not None
            self.common.display_df_in_table(self.ui.tableWidget,
                                            self.filtered_df)

    def apply_filters(self, df=None, filters=None):
        """Apply filters to the specified dataframe or the current dataframe"""
        if df is None:
            sel_df = self.get_sel_df()
        else:
            sel_df = df
        if filters is None:
            current_filters = self.filter.get_current_filters()
        else:
            current_filters = filters

        self.filter.df = sel_df
        self.filtered_df = self.filter.apply_filters(current_filters)

        if self.filtered_df is not None:  # Check if filtered_df is not None
            self.common.display_df_in_table(self.ui.tableWidget,
                                            self.filtered_df)

        return self.filtered_df

    def populate_graph_combo_box(self):
        """Populate graph titles into a combobox"""
        self.ui.cbb_graph_list.clear()
        for graph_id, graph in self.plots.items():
            self.ui.cbb_graph_list.addItem(
                f"{graph.graph_id}-{graph.plot_style}_plot: [{graph.x}] - ["
                f"{graph.y[0]}] - ["
                f"{graph.z}]")
        # Set the current selection to the last item added
        if self.ui.cbb_graph_list.count() > 0:
            self.ui.cbb_graph_list.setCurrentIndex(
                self.ui.cbb_graph_list.count() - 1)

    def select_sub_window_from_combo_box(self):
        """Select and show a graph on top via a combobox"""
        graph_title = self.ui.cbb_graph_list.currentText()
        for sub_window in self.ui.mdiArea.subWindowList():
            graph_dialog = sub_window.widget()
            if isinstance(graph_dialog, QDialog):
                graph = graph_dialog.layout().itemAt(0).widget()
                if graph and graph_title == f"{graph.graph_id}-" \
                                            f"{graph.plot_style}_plot: [" \
                                            f"{graph.x}] - [{graph.y[0]}] - [" \
                                            f"{graph.z}]":
                    if sub_window.isMinimized():
                        sub_window.showNormal()
                    self.ui.mdiArea.setActiveSubWindow(sub_window)
                    return

    def delete_graph(self, graph_id):
        """Delete a graph from the self.plots dictionary"""
        graph, graph_dialog, sub_window = self.get_sel_graph()
        graph_id = graph.graph_id
        # Remove the graph from the dictionary
        self.plots.pop(graph_id)
        if sub_window:
            self.ui.mdiArea.removeSubWindow(sub_window)
            sub_window.close()
            self.populate_graph_combo_box()
        print(f"Plot {graph_id} is deleted")

    def minimize_all_graph(self):
        """Minimize all graph sub-windows"""
        for sub_window in self.ui.mdiArea.subWindowList():
            sub_window.showMinimized()

    def clear_env(self):
        # Clear original dataframes
        self.original_dfs = {}
        self.sel_df = None
        self.filtered_df = None
        self.filter.filters = []

        # Close and delete all sub-windows
        for sub_window in self.ui.mdiArea.subWindowList():
            self.ui.mdiArea.removeSubWindow(sub_window)
            sub_window.close()
        self.plots.clear()

        # Clear GUI elements
        self.ui.dfs_listbox.clear()
        self.ui.cbb_x_2.clear()
        self.ui.cbb_y_2.clear()
        self.ui.cbb_y2_2.clear()
        self.ui.cbb_y3_2.clear()
        self.ui.cbb_z_2.clear()
        self.ui.listbox_filters.clear()
        self.ui.tableWidget.clearContents()
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.setColumnCount(0)
        self.ui.cbb_graph_list.clear()

    def save(self):
        """Save current work"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(None,
                                                       "Save work",
                                                       "",
                                                       "SPECTROview Files ("
                                                       "*.json)")
            if file_path:
                # Convert Graph objects to serializable format
                plots_data = {}
                for graph_id, graph in self.plots.items():
                    graph_data = {
                        'plot_width': graph.plot_width,
                        'plot_height': graph.plot_height,
                        'df_name': graph.df_name,
                        'filters': graph.filters,
                        'graph_id': graph.graph_id,
                        'plot_style': graph.plot_style,
                        'x': graph.x,
                        'y': graph.y,
                        'y2': graph.y2,
                        'y3': graph.y3,
                        'z': graph.z,
                        'xmin': graph.xmin,
                        'xmax': graph.xmax,
                        'ymin': graph.ymin,
                        'ymax': graph.ymax,
                        'y2min': graph.y2min,
                        'y2max': graph.y2max,
                        'y3min': graph.y3min,
                        'y3max': graph.y3max,
                        'zmin': graph.zmin,
                        'zmax': graph.zmax,
                        'plot_title': graph.plot_title,
                        'xlabel': graph.xlabel,
                        'ylabel': graph.ylabel,
                        'y2label': graph.y2label,
                        'y3label': graph.y3label,
                        'zlabel': graph.zlabel,
                        'x_rot': graph.x_rot,
                        'grid': graph.grid,
                        'legend_visible': graph.legend_visible,
                        'legend_outside': graph.legend_outside,
                        'color_palette': graph.color_palette,
                        'dpi': graph.dpi,
                        'wafer_size': graph.wafer_size,
                        'wafer_stats': graph.wafer_stats,
                        'trendline_order': graph.trendline_order,
                        'show_trendline_eq': graph.show_trendline_eq,
                        'show_bar_plot_error_bar':
                            graph.show_bar_plot_error_bar,
                        'join_for_point_plot': graph.join_for_point_plot
                    }
                    plots_data[graph_id] = graph_data

                # Prepare data to save
                data_to_save = {
                    'plots': plots_data,
                    'original_dfs': {key: df.to_dict() for key, df in
                                     self.original_dfs.items()},
                }

                # Save to JSON file
                with open(file_path, 'w') as f:
                    json.dump(data_to_save, f, indent=4)
                show_alert("Work saved successfully.")
        except Exception as e:
            show_alert(f"Error saving work: {e}")

    def load(self):
        """Open saved work"""
        self.clear_env()
        try:
            file_path, _ = QFileDialog.getOpenFileName(None,
                                                       "Load work",
                                                       "",
                                                       "SPECTROview Files ("
                                                       "*.json)")
            if file_path:
                with open(file_path, 'r') as f:
                    load = json.load(f)
                    self.original_dfs = {key: pd.DataFrame(value) for key, value
                                         in
                                         load.get('original_dfs', {}).items()}
                    self.update_dfs_list()
                    self.sel_df = pd.DataFrame(
                        load.get('sel_df', {})) if load.get(
                        'sel_df') is not None else None
                    self.filter.filters = load.get('filters', [])
                    self.filtered_df = pd.DataFrame(
                        load.get('filtered_df', {})) if load.get(
                        'filtered_df') is not None else None

                    # Load plots
                    plots_data = load.get('plots', {})
                    for graph_id, graph_data in plots_data.items():
                        # Recreate graph instance
                        graph = Graph(graph_id=graph_data['graph_id'])

                        # Get plot size
                        try:
                            graph.plot_width = graph_data['plot_width']
                            graph.plot_height = graph_data['plot_height']

                        except KeyError:
                            graph.plot_width = 600
                            graph.plot_width = 450
                        try:
                            graph.y2 = graph_data['y2']
                            graph.y3 = graph_data['y3']
                            graph.y2min = graph_data['y2min']
                            graph.y2max = graph_data['y2max']
                            graph.y3min = graph_data['y3min']
                            graph.y3max = graph_data['y3max']
                            graph.y2label = graph_data['y2label']
                            graph.y3label = graph_data['y3label']
                        except KeyError:
                            pass

                        graph.df_name = graph_data['df_name']
                        graph.filters = graph_data['filters']
                        graph.plot_style = graph_data['plot_style']
                        graph.x = graph_data['x']
                        graph.y = graph_data['y']
                        graph.z = graph_data['z']
                        graph.xmin = graph_data['xmin']
                        graph.xmax = graph_data['xmax']
                        graph.ymin = graph_data['ymin']
                        graph.ymax = graph_data['ymax']

                        graph.zmin = graph_data['zmin']
                        graph.zmax = graph_data['zmax']
                        graph.plot_title = graph_data['plot_title']
                        graph.xlabel = graph_data['xlabel']
                        graph.ylabel = graph_data['ylabel']
                        graph.zlabel = graph_data['zlabel']
                        graph.x_rot = graph_data['x_rot']
                        graph.grid = graph_data['grid']
                        graph.legend_visible = graph_data['legend_visible']
                        graph.legend_outside = graph_data['legend_outside']
                        graph.color_palette = graph_data['color_palette']
                        graph.dpi = graph_data['dpi']
                        graph.wafer_size = graph_data['wafer_size']
                        graph.wafer_stats = graph_data['wafer_stats']
                        graph.trendline_order = graph_data['trendline_order']
                        graph.show_trendline_eq = graph_data[
                            'show_trendline_eq']

                        try:
                            graph.show_bar_plot_error_bar = graph_data[
                                'show_bar_plot_error_bar']
                        except KeyError:
                            graph.show_bar_plot_error_bar = False

                        try:
                            graph.join_for_point_plot = graph_data[
                                'join_for_point_plot']
                        except KeyError:
                            graph.join_for_point_plot = False

                        # Plot the graph
                        graph.create_plot_widget(graph.dpi)
                        filtered_df = self.apply_filters(
                            self.original_dfs[graph.df_name], graph.filters)
                        graph.plot(filtered_df)

                        # Store the plot in the dictionary with graph_id as key
                        self.plots[graph.graph_id] = graph

                        # Create a QDialog to hold the Graph instance
                        graph_dialog = QDialog(self)
                        graph_dialog.setWindowTitle(
                            f"{graph.graph_id}-{graph.plot_style}_plot: ["
                            f"{graph.x}] - "
                            f"[{graph.y[0]}] - [{graph.z}]")

                        layout = QVBoxLayout()
                        layout.addWidget(graph)
                        graph_dialog.setLayout(layout)

                        # Add the QDialog to the mdiArea
                        sub_window = MdiSubWindow(graph.graph_id)
                        sub_window.setWidget(graph_dialog)
                        sub_window.closed.connect(self.delete_graph)
                        self.ui.mdiArea.addSubWindow(sub_window)
                        sub_window.resize(graph.plot_width, graph.plot_height)
                        sub_window.show()
                        self.update_graph()

                    self.filter.upd_filter_listbox()
                    self.populate_graph_combo_box()

        except Exception as e:
            show_alert(f"Error loading work: {e}")


class MdiSubWindow(QMdiSubWindow):
    """Custom class of QMdiSubWindow to get signal when closing sub window"""
    closed = Signal(int)

    def __init__(self, graph_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_id = graph_id

    def closeEvent(self, event):
        """Override closeEvent to emit a signal when the subwindow is closing"""
        self.closed.emit(self.graph_id)
        super().closeEvent(event)

    def resizeEvent(self, event):
        """Override resizeEvent to handle window resizing"""
        new_size = self.size()
        width, height = new_size.width(), new_size.height()
        print(f"Subwindow resized: Width = {width}, Height = {height}")
        super().resizeEvent(event)
