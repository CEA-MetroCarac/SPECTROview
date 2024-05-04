import os
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
import dill

from common import view_df, show_alert
from common import PLOT_STYLES, PALETTE
from common import  Graph, Filter

from PySide6.QtWidgets import QApplication, QFileDialog, QDialog, QVBoxLayout, \
    QLineEdit, QListWidgetItem, QMdiSubWindow,QCheckBox
from PySide6.QtCore import Qt, QFileInfo, QTimer, QObject, Signal


class Visu(QDialog):
    """Class to GUI and callbacks"""

    def __init__(self, settings, ui, common):
        super().__init__()
        self.ui = ui
        self.settings = settings
        self.common = common
        self.setWindowTitle("Graph Plot")

        # DATAFRAME
        self.original_dfs = {}
        self.sel_df = None
        self.ui.btn_open_dfs.clicked.connect(self.open_dfs)
        self.ui.btn_view_df_3.clicked.connect(self.show_df)
        self.ui.dfs_listbox.itemSelectionChanged.connect(self.update_gui)

        # FILTER
        self.filter = Filter(self.ui.ent_filter_query_4,
                             self.ui.filter_listbox_3,
                             self.sel_df)
        self.filtered_df = None
        self.ui.btn_add_filter_4.clicked.connect(self.filter.add_filter)
        self.ui.ent_filter_query_4.returnPressed.connect(self.filter.add_filter)
        self.ui.btn_remove_filters_4.clicked.connect(self.filter.remove_filter)
        self.ui.btn_apply_filters_4.clicked.connect(self.apply_filters)

        # GRAPH
        self.plots = {}  # Dictionary to store plots
        self.graph_id = 0  # Initialize graph number
        self.ui.btn_add_graph.clicked.connect(self.add_graph)
        self.ui.btn_upd_graph.clicked.connect(self.upd_graph)
        self.ui.btn_copy_graph.clicked.connect(self.copy_fig_to_clb)
        self.ui.cbb_palette.addItems(PALETTE)
        self.ui.cbb_plotstyle.addItems(PLOT_STYLES)
        self.ui.btn_adjust_dpi.clicked.connect(self.adjust_dpi)
        # Track selected sub-window
        self.ui.mdiArea.subWindowActivated.connect(self.on_selected_graph)
        #self.ui.spb_dpi.valueChanged.connect(self.adjust_dpi)
    def add_graph(self, df_name = None, filters = None):
        """Plot new graph"""
        # Increment plot number
        self.graph_id += 1

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

        # Create an instance of the Graph class
        graph = Graph(graph_id=self.graph_id)

        graph.plot_style = self.ui.cbb_plotstyle.currentText()
        graph.plot_title = self.ui.lbl_plot_title.text()
        graph.color_palette = self.ui.cbb_palette.currentText()
        graph.df_name = df_name
        graph.filters = current_filters
        graph.x = x
        graph.y = y
        graph.z = z if z != "None" else None
        graph.xlabel = x
        graph.ylabel = y
        graph.zlabel = z
        graph.dpi = float(self.ui.spb_dpi.text())
        graph.wafer_size = float(self.ui.lbl_wafersize.text())
        graph.wafer_stats = self.ui.cb_wafer_stats.isChecked()
        # Pass legend & grid settings
        graph.legend_visible = self.ui.cb_legend_visible.isChecked()
        graph.legend_outside = self.ui.cb_legend_outside.isChecked()
        graph.grid = self.ui.cb_grid.isChecked()

        # Plot the graph
        graph.create_plot_widget(graph.dpi)
        graph.plot(filtered_df)

        # Store the plot in the dictionary with graph_id as key
        self.plots[graph.graph_id] = graph

        # Create a QDialog to hold the Graph instance
        graph_dialog = QDialog(self)
        graph_dialog.setWindowTitle(f"Graph_{graph.graph_id}: ({x} vs. {y} vs. {z})")
        layout = QVBoxLayout()
        layout.addWidget(graph)
        graph_dialog.setLayout(layout)

        # Add the QDialog to the mdiArea
        sub_window = QMdiSubWindow()
        sub_window.setWidget(graph_dialog)
        self.ui.mdiArea.addSubWindow(sub_window)
        sub_window.show()
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
        return sel_graph, graph_dialog
    def upd_graph(self):
        """ Update the existing graph with new properties"""
        sel_graph, graph_dialog = self.get_sel_graph()
        if sel_graph:
            plot_style = self.ui.cbb_plotstyle.currentText()
            plot_title = self.ui.lbl_plot_title.text()
            x = self.ui.cbb_x_2.currentText()
            y = self.ui.cbb_y_2.currentText()
            z = self.ui.cbb_z_2.currentText()
            xlabel = self.ui.lbl_xlabel.text()
            ylabel = self.ui.lbl_ylabel.text()
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
            sel_graph.df_name = self.ui.dfs_listbox.currentItem().text()
            sel_graph.filters = current_filters
            sel_graph.x = x
            sel_graph.y = y
            sel_graph.z = z if z != "None" else None

            sel_graph.xmin = xmin
            sel_graph.xmax = xmax
            sel_graph.ymin = ymin
            sel_graph.ymax = ymax
            sel_graph.zmin = zmin
            sel_graph.zmax = zmax
            sel_graph.xlabel = xlabel
            sel_graph.ylabel = ylabel
            sel_graph.zlabel = zlabel
            sel_graph.x_rot = x_rot
            sel_graph.plot_style = plot_style
            sel_graph.plot_title = plot_title
            sel_graph.color_palette = palette
            sel_graph.wafer_size = float(self.ui.lbl_wafersize.text())
            sel_graph.wafer_stats = self.ui.cb_wafer_stats.isChecked()
            sel_graph.legend_visible = self.ui.cb_legend_visible.isChecked()
            sel_graph.legend_outside = self.ui.cb_legend_outside.isChecked()
            sel_graph.grid = self.ui.cb_grid.isChecked()
            sel_graph.dpi = float(self.ui.spb_dpi.text())
            graph_dialog.setWindowTitle(f"Graph_{sel_graph.graph_id}: ({x} vs. {y} vs.{z})")
            if plot_style =='wafer':
                sel_graph.create_plot_widget(sel_graph.dpi, sel_graph.graph_layout)
                sel_graph.plot(self.filtered_df)
            else:
                sel_graph.plot(self.filtered_df)

    def adjust_dpi(self):
        sel_graph, graph_dialog = self.get_sel_graph()
        dpi = float(self.ui.spb_dpi.text())
        if sel_graph:
            sel_graph.create_plot_widget(dpi, sel_graph.graph_layout )
            QTimer.singleShot(100, self.upd_graph)

    def on_selected_graph(self, sub_window):
        """Reflect all properties of selected graph object to GUI"""
        sel_graph, graph_dialog = self.get_sel_graph()

        if sel_graph:
            # Plot style
            plot_style = sel_graph.plot_style
            items = [self.ui.cbb_plotstyle.itemText(i) for i in
                     range(self.ui.cbb_plotstyle.count())]
            if plot_style in items:
                self.ui.cbb_plotstyle.setCurrentText(plot_style)

            # Reflect df_name in the listbox
            current_items = [self.ui.dfs_listbox.item(i).text() for i in
                             range(self.ui.dfs_listbox.count())]
            if sel_graph.df_name not in current_items:
                self.ui.dfs_listbox.addItem(sel_graph.df_name)
            else:
                index = current_items.index(sel_graph.df_name)
                self.ui.dfs_listbox.setCurrentRow(index)

            # Reflect filter's states in the listbox
            self.reflect_filters_to_gui(sel_graph)

            # Update combobox selections
            x = self.ui.cbb_x_2.findText(sel_graph.x)
            y = self.ui.cbb_y_2.findText(sel_graph.y)
            z = self.ui.cbb_z_2.findText(sel_graph.z)
            self.ui.cbb_x_2.setCurrentIndex(
                x if x != -1 else 0)
            self.ui.cbb_y_2.setCurrentIndex(
                y if y != -1 else 0)
            self.ui.cbb_z_2.setCurrentIndex(
                z if z != -1 else 0)

            # WAFER
            self.ui.lbl_wafersize.setText(str(sel_graph.wafer_size))
            self.ui.cb_wafer_stats.setChecked(sel_graph.wafer_stats)


            # Rotation x label:
            self.ui.x_rot.setValue(sel_graph.x_rot)
            # Reflect Titles:
            self.ui.lbl_plot_title.setText(sel_graph.plot_title)
            self.ui.lbl_xlabel.setText(sel_graph.xlabel)
            self.ui.lbl_ylabel.setText(sel_graph.ylabel)
            self.ui.lbl_zlabel.setText(sel_graph.zlabel)
            # Reflect limits:
            self.ui.xmin_2.setText(sel_graph.xmin)
            self.ui.ymin_2.setText(sel_graph.ymin)
            self.ui.zmin_2.setText(sel_graph.zmin)
            self.ui.xmax_2.setText(sel_graph.xmax)
            self.ui.ymax_2.setText(sel_graph.ymax)
            self.ui.zmax_2.setText(sel_graph.zmax)

            # Reflect legend status
            self.ui.cb_legend_visible.setChecked(sel_graph.legend_visible)
            self.ui.cb_legend_outside.setChecked(sel_graph.legend_outside)

            # Grid
            self.ui.cb_grid.setChecked(sel_graph.grid)

            # Reflect Color palette
            color_palette = sel_graph.color_palette
            combo_items = [self.ui.cbb_palette.itemText(i) for i in
                           range(self.ui.cbb_palette.count())]
            if color_palette in combo_items:
                self.ui.cbb_palette.setCurrentText(color_palette)

            # Reflect DPI
            self.ui.spb_dpi.setValue(sel_graph.dpi)
    def reflect_filters_to_gui(self, sel_graph):
        """Reflect the filters of a graph object and their states to the df listbox"""
        # Clear the existing items and uncheck them
        for index in range(self.ui.filter_listbox_3.count()):
            item = self.ui.filter_listbox_3.item(index)
            if isinstance(item, QListWidgetItem):
                widget = self.ui.filter_listbox_3.itemWidget(item)
                if isinstance(widget, QCheckBox):
                    widget.setChecked(False)

        for filter_info in sel_graph.filters:
            filter_expression = filter_info["expression"]
            filter_state = filter_info["state"]

            # Check if the filter expression already exists in the listbox
            existing_item = None
            for index in range(self.ui.filter_listbox_3.count()):
                item = self.ui.filter_listbox_3.item(index)
                if isinstance(item, QListWidgetItem):
                    widget = self.ui.filter_listbox_3.itemWidget(item)
                    if isinstance(widget, QCheckBox) and widget.text() == filter_expression:
                        existing_item = item
                        break

            # Update the state if the filter expression already exists, otherwise add a new item
            if existing_item:
                checkbox = self.ui.filter_listbox_3.itemWidget(existing_item)
                checkbox.setChecked(filter_state)
            else:
                item = QListWidgetItem()
                checkbox = QCheckBox(filter_expression)
                checkbox.setChecked(filter_state)
                item.setSizeHint(checkbox.sizeHint())
                self.ui.filter_listbox_3.addItem(item)
                self.ui.filter_listbox_3.setItemWidget(item, checkbox)

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
        self.upd_dfs_list()

    def upd_dfs_list(self):
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
        current_filters = self.filter.get_current_filters()
        self.filtered_df = self.apply_filters(self.sel_df, current_filters)

    def update_cbb(self):
        """Populate columns of selected data to comboboxes"""
        sel_df = self.get_sel_df()
        if sel_df is not None:
            columns = sel_df.columns.tolist()
            self.ui.cbb_x_2.clear()
            self.ui.cbb_y_2.clear()
            self.ui.cbb_z_2.clear()
            self.ui.cbb_x_2.addItem("None")
            self.ui.cbb_y_2.addItem("None")
            self.ui.cbb_z_2.addItem("None")
            for column in columns:
                self.ui.cbb_x_2.addItem(column)
                self.ui.cbb_y_2.addItem(column)
                self.ui.cbb_z_2.addItem(column)

    def copy_fig_to_clb(self):
        """Copy the selected figure to clipboard"""
        sel_graph = self.get_sel_graph()
        self.common.copy_fig_to_clb(canvas=sel_graph.canvas)



    def get_sel_df(self):
        """Get the current selected df among the df within 'dfr' dict"""
        sel_item = self.ui.dfs_listbox.currentItem()
        sel_df_name = sel_item.text()
        self.sel_df = self.original_dfs[sel_df_name]
        return self.sel_df

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
        self.common.display_df_in_table(self.ui.tableWidget, current_df)

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
        self.common.display_df_in_table(self.ui.tableWidget, self.filtered_df)
        return self.filtered_df
