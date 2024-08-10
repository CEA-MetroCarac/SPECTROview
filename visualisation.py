import os
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
import dill
import json

from common import view_df, show_alert
from common import PLOT_STYLES, PALETTE, LEGEND_LOCATION
from common import Graph, Filter, DataframeTable

from PySide6.QtWidgets import QApplication, QFileDialog, QDialog, QVBoxLayout, \
    QLineEdit, QListWidgetItem, QMdiSubWindow, QCheckBox, QMdiArea, QLabel, \
    QSizePolicy, QMessageBox
from PySide6.QtCore import Qt, QFileInfo, QTimer, QObject, Signal


class Visualization(QDialog):
    """
    This class provides a GUI for plotting graphs based on selected dataframes,
    applying filters, customizing graph properties, and managing graph
    instances.

    Attributes:
        settings (QSettings): Object for managing application settings.
        ui (Ui_MainWindow): User interface object.
        common (Common): Object providing common functionalities.

        original_dfs (dict): Dictionary holding original dataframes loaded
        from files.
        sel_df (pd.DataFrame or None): Currently selected dataframe for
        visualization.
        filtered_df (pd.DataFrame or None): Dataframe after applying current
        filters.
        plots (dict): Dictionary storing Graph instances.
        graph_id (int): Identifier for the next graph to be created.

        filter (Filter): Instance of Filter class managing filter operations.
    """

    def __init__(self, settings, ui, common):
        super().__init__()
        self.ui = ui
        self.settings = settings
        self.common = common
        self.setWindowTitle("Graph Plot")

        # DATAFRAME
        self.original_dfs = {}
        self.sel_df = None
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
        self.plots = {}
        self.graph_id = 0  # Initialize graph number
        # Add a graph
        self.ui.btn_add_graph.clicked.connect(self.plotting)
        # Update an existing graph
        self.ui.btn_upd_graph.clicked.connect(
            lambda: self.plotting(update_graph=True))

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
        self.ui.cbb_legend_loc.addItems(LEGEND_LOCATION)

        # Track selected sub-window
        self.ui.mdiArea.subWindowActivated.connect(self.on_selected_graph)
        self.ui.cbb_graph_list.currentIndexChanged.connect(
            self.select_sub_window_from_combo_box)

        self.ui.btn_minimize_all.clicked.connect(self.minimize_all_graph)

    def open_dfs(self, dfs=None, file_paths=None):
        """Open and load dataframes from Excel files."""

        if self.original_dfs is None:
            self.original_dfs = {}
        if dfs:
            self.original_dfs = dfs  # If dataframes are passed directly
        else:
            if file_paths:
                for file_path in file_paths:
                    file_path = Path(file_path)
                    fname = file_path.stem  # get fname w/o extension
                    extension = file_path.suffix.lower()
                    if extension == '.xlsx':
                        excel_file = pd.ExcelFile(file_path)
                        sheet_names = excel_file.sheet_names
                        for sheet_name in sheet_names:
                            sheet_name_cleaned = sheet_name.replace(" ", "")
                            df_name = f"{fname}_{sheet_name_cleaned}"
                            self.original_dfs[df_name] = pd.read_excel(
                                excel_file, sheet_name=sheet_name)
                    else:
                        show_alert(f"Unsupported file format: {extension}")

        self.update_dfs_list()

    def update_dfs_list(self):
        """
        This method updates the dataframe listbox with current dataframes.
        """
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

    def plotting(self, update_graph=False):
        """Plot a new graph or update an existing graph."""
        if update_graph:
            # Update the selected graph
            graph, graph_dialog, sub_window = self.get_sel_graph()
            sub_window_size = sub_window.size()
            graph.plot_width = sub_window_size.width()
            graph.plot_height = sub_window_size.height()
        else:
            # Create new graph
            # Get available graph IDs considering vacancies in the list
            available_ids = [i for i in range(1, len(self.plots) + 2) if
                             i not in self.plots]
            graph_id = min(available_ids) if available_ids else len(
                self.plots) + 1
            # Create new graph
            graph = Graph(graph_id=graph_id)
            self.plots[graph.graph_id] = graph

        # Collecting properties of graph from GUI
        graph.plot_style = self.ui.cbb_plotstyle.currentText()

        title = self.ui.lbl_plot_title.text()
        graph.plot_title = title if title != "None" else None

        current_filters = self.filter.get_current_filters()
        if current_filters != graph.filters:
            graph.legend_properties = []
        else:
            pass
        current_df_name = self.ui.dfs_listbox.currentItem().text()
        graph.df_name = current_df_name
        graph.filters = current_filters

        x = self.ui.cbb_x_2.currentText()
        y = self.ui.cbb_y_2.currentText()
        z = self.ui.cbb_z_2.currentText()

        # Check if z has changed and reset legend_properties if needed
        self.is_z_changed(graph)

        graph.x = x
        if len(graph.y) == 0:
            graph.y.append(y)
        else:
            graph.y[0] = y
        graph.z = z if z != "None" else None

        graph.color_palette = self.ui.cbb_palette.currentText()
        graph.wafer_size = float(self.ui.lbl_wafersize.text())
        graph.x_rot = float(self.ui.x_rot.text())
        graph.wafer_size = float(self.ui.lbl_wafersize.text())
        graph.wafer_stats = self.ui.cb_wafer_stats.isChecked()

        graph.dpi = float(self.ui.spb_dpi.text())

        graph.legend_visible = self.ui.cb_legend_visible.isChecked()
        graph.legend_location = self.ui.cbb_legend_loc.currentText()
        graph.legend_outside = self.ui.cb_legend_outside.isChecked()
        graph.grid = self.ui.cb_grid.isChecked()
        graph.trendline_order = float(self.ui.spb_trendline_oder.text())
        graph.show_trendline_eq = self.ui.cb_trendline_eq.isChecked()
        graph.show_bar_plot_error_bar = self.ui.cb_show_err_bar_plot.isChecked()
        graph.join_for_point_plot = self.ui.cb_join_for_point_plot.isChecked()

        # PLOTTING
        graph.create_plot_widget(graph.dpi)

        if not update_graph:
            # Create new graph widget
            graph_dialog = QDialog(self)
            layout = QVBoxLayout()
            layout.addWidget(graph)
            graph_dialog.setLayout(layout)

            # Add the QDialog to a QMdiSubWindow
            sub_window = MdiSubWindow(graph_id, self.ui.lbl_figsize)
            sub_window.setWidget(graph_dialog)
            sub_window.closed.connect(self.delete_graph)
            sub_window.resize(graph.plot_width, graph.plot_height)
            self.ui.mdiArea.addSubWindow(sub_window)
            sub_window.show()
            self.add_graph_list_to_combobox()
        else:
            # Update existing graph
            graph, graph_dialog, sub_window = self.get_sel_graph()
            sub_window_size = sub_window.size()
            graph.plot_width = sub_window_size.width()
            graph.plot_height = sub_window_size.height()

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

        text = f"{graph.graph_id}-{graph.plot_style}_plot: [{x}] - [{y}] - [" \
               f"{z}]"
        graph_dialog.setWindowTitle(text)

        # Plot action
        QTimer.singleShot(100, self.plot_action)
        QTimer.singleShot(200, self.customize_legend)

    def plot_action(self):
        """
        Perform the plot action for the selected graph.
        This method fetches the selected graph, applies filters, and triggers
        the actual plotting.
        """
        graph, graph_dialog, sub_window = self.get_sel_graph()
        self.filtered_df = self.apply_filters(self.sel_df, graph.filters)
        # print(f"self.sel_df {self.sel_df}")
        if graph:
            if graph.plot_style == 'wafer':
                graph.create_plot_widget(graph.dpi, graph.graph_layout)
                graph.plot(self.filtered_df)
            else:
                graph.plot(self.filtered_df)

    def is_z_changed(self, graph):
        """Check if z-axis value has changed from the current graph settings"""
        current_z = self.ui.cbb_z_2.currentText()
        if current_z != graph.z:
            graph.legend_properties = []
            print("'z' values are changed, resets legends to default")
            return True
        return False

    def customize_legend(self):
        """ Show all legend's properties in GUI for customization"""
        graph, graph_dialog, sub_window = self.get_sel_graph()
        main_layout = self.ui.main_layout
        graph.customize_legend_via_gui(main_layout)

    def on_selected_graph(self, sub_window):
        """Update GUI elements based on the properties of the selected graph"""
        graph, graph_dialog, sub_window = self.get_sel_graph()

        if graph:
            # Display figure size in GUI
            sub_window_size = sub_window.size()
            width = sub_window_size.width()
            height = sub_window_size.height()
            self.ui.lbl_figsize.setText(f"({width}x{height})")

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
            # Reflect legend location:
            legend_loc = graph.legend_location
            items = [self.ui.cbb_legend_loc.itemText(i) for i in
                     range(self.ui.cbb_legend_loc.count())]
            if legend_loc in items:
                self.ui.cbb_legend_loc.setCurrentText(legend_loc)

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

            # Show legends on GUI for customization
            self.customize_legend()

    def reflect_filters_to_gui(self, sel_graph):
        """Reflect the state of filters associated with a graph to the GUI"""

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

    def update_gui(self):
        """Update the GUI elements based on the selected dataframe"""
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
        """Copy the selected graph figure to the clipboard"""
        sel_graph, graph_dialog, sub_window = self.get_sel_graph()
        self.common.copy_fig_to_clb(canvas=sel_graph.canvas)

    def get_sel_graph(self):
        """Retrieve the currently selected graph object"""
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
        """Retrieve the currently selected dataframe"""
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
        """
        Remove the selected dataframe from the listbox and original_dfs
        dictionary.
        """
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
        """This method saves the currently selected dataframe to an Excel
        file."""
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
        """This method displays the selected dataframe in a new window"""
        current_filters = self.filter.get_current_filters()
        current_df = self.apply_filters(self.sel_df, current_filters)
        if current_df is not None:
            view_df(self.ui.tabWidget, current_df)
        else:
            show_alert("No fit dataframe to display")

    def apply_filters(self, df=None, filters=None):
        """
        Apply filters to the specified dataframe or the currently selected
        dataframe.

        Args:
            df (pd.DataFrame, optional): Dataframe to apply filters to.
            filters (list, optional): List of filters to apply.

        Returns:
            pd.DataFrame: Filtered dataframe.

        """
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

        return self.filtered_df

    def add_graph_list_to_combobox(self):
        """
        Populate graph titles into the combobox for graph selection.

        This method updates the combobox with current graph titles.

        """
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
        """
        This method selects and displays a graph based on the user selection
        in the combobox.
        """
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
        """Delete the specified graph from the plots dictionary"""
        graph, graph_dialog, sub_window = self.get_sel_graph()
        if graph is None:
            return
        graph_id = graph.graph_id
        if graph_id:
            self.plots.pop(graph_id, None)
            if sub_window:
                self.ui.mdiArea.removeSubWindow(sub_window)
                sub_window.close()
                self.add_graph_list_to_combobox()
            print(f"Plot {graph_id} is deleted")

    def minimize_all_graph(self):
        """
        This method minimizes all open graph sub-windows.
        """
        for sub_window in self.ui.mdiArea.subWindowList():
            sub_window.showMinimized()

    def clear_env(self):
        """
        This method reinit all attributes.
        """
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
        self.ui.cbb_graph_list.clear()

    def add_y12(self):
        """Add a second line in the current plot ax"""
        graph, graph_dialog, sub_window = self.get_sel_graph()
        y12 = self.ui.cbb_y12.currentText()
        if len(graph.y) == 1:
            graph.y.append(y12)
        else:
            graph.y[1] = y12
        self.plotting(update_graph=True)

    def add_y13(self):
        """Add a 3rd line in the current plot ax"""
        graph, graph_dialog, sub_window = self.get_sel_graph()
        y13 = self.ui.cbb_y13.currentText()
        if len(graph.y) == 2:
            graph.y.append(y13)
        else:
            graph.y[2] = y13
        self.plotting(update_graph=True)

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
            self.plotting(update_graph=True)
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
            self.plotting(update_graph=True)
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

        self.plotting(update_graph=True)

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

        self.plotting(update_graph=True)

    def save(self):
        """Save current work"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(None,
                                                       "Save work",
                                                       "",
                                                       "SPECTROview Files ("
                                                       "*.graphs)")
            if file_path:
                # Convert Graph objects to serializable format
                plots_data = {}
                for graph_id, graph in self.plots.items():
                    graph_data = graph.save(fname=None)
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

    def load(self, file_path):
        """Reload saved works"""
        try:
            self.clear_env()
            with open(file_path, 'r') as f:
                load = json.load(f)
                self.original_dfs = {key: pd.DataFrame(value) for key, value in load.get('original_dfs', {}).items()}
                self.update_dfs_list()

                # Load plots
                plots_data = load.get('plots', {})
                for graph_id, graph_data in plots_data.items():
                    # Recreate graph instance
                    graph = Graph(graph_id=graph_id)
                    graph.set_attributes(graph_data)

                    # Plot the graph
                    graph.create_plot_widget(graph.dpi)
                    self.plots[graph.graph_id] = graph

                    # Create a QDialog to hold the Graph instance
                    graph_dialog = QDialog(self)
                    graph_dialog.setWindowTitle(
                        f"{graph.graph_id}-{graph.plot_style}_plot: [{graph.x}] - [{graph.y[0]}] - [{graph.z}]"
                    )
                    layout = QVBoxLayout()
                    layout.addWidget(graph)
                    graph_dialog.setLayout(layout)

                    # Add the QDialog to the mdiArea
                    sub_window = MdiSubWindow(graph.graph_id, self.ui.lbl_figsize)
                    sub_window.setWidget(graph_dialog)
                    sub_window.closed.connect(self.delete_graph)
                    self.ui.mdiArea.addSubWindow(sub_window)
                    sub_window.resize(graph.plot_width, graph.plot_height)
                    sub_window.show()

                    self.plot_action()

                self.filter.upd_filter_listbox()
                self.add_graph_list_to_combobox()

        except Exception as e:
            show_alert(f"Error loading work: {e}")


class MdiSubWindow(QMdiSubWindow):
    """
    Custom class of QMdiSubWindow to get signal when closing subwindow.

    Attributes:
    closed (Signal): Signal emitted when the subwindow is closing, carrying
    the graph ID.
    graph_id (int): ID associated with the graph in the subwindow.
    figsize_label (QLabel): QLabel used to display the size of the subwindow.
    """
    closed = Signal(int)

    def __init__(self, graph_id, figsize_label, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_id = graph_id
        self.figsize_label = figsize_label

    def closeEvent(self, event):
        """Override closeEvent to emit a signal when the subwindow is closing"""
        self.closed.emit(self.graph_id)
        super().closeEvent(event)

    def resizeEvent(self, event):
        """Override resizeEvent to handle window resizing"""
        new_size = self.size()
        width, height = new_size.width(), new_size.height()
        # Update QLabel with the new size
        self.figsize_label.setText(f"({width}x{height})")
        super().resizeEvent(event)
