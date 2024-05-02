import os
import numpy as np
import pandas as pd
import json
from copy import deepcopy
from pathlib import Path
import dill

from common import view_df, show_alert, CommonUtilities, Graph, Filter, PALETTE

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtWidgets import QApplication, QFileDialog, QDialog, QVBoxLayout, \
    QLineEdit, QListWidgetItem, QMdiSubWindow
from PySide6.QtGui import QResizeEvent
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
        self.dfs = {}
        self.sel_df = None
        self.ui.btn_open_dfs.clicked.connect(self.open_dfs)
        self.ui.btn_view_df_3.clicked.connect(self.show_df)
        self.ui.dfs_listbox.itemSelectionChanged.connect(self.update_gui)

        # FILTER
        self.filter = Filter(self.ui.ent_filter_query_4,
                             self.ui.filter_listbox_3,
                             self.sel_df)
        self.filtered_dfs = None
        self.filtered_df = None
        self.ui.btn_add_filter_4.clicked.connect(self.filter.add_filter)
        self.ui.ent_filter_query_4.returnPressed.connect(self.filter.add_filter)
        self.ui.btn_remove_filters_4.clicked.connect(self.filter.remove_filter)
        self.ui.btn_apply_filters_4.clicked.connect(self.apply_filters)
        self.ui.cbb_palette.addItems(PALETTE)
        # GRAPH
        self.plots = {}  # Dictionary to store plots
        self.plot_number = 0  # Initialize plot number

        self.ui.btn_add_graph.clicked.connect(self.add_graph)
        self.ui.btn_upd_graph.clicked.connect(self.upd_graph)
        self.ui.btn_copy_graph.clicked.connect(self.copy_fig_to_clb)

        # Track selected sub-window
        self.ui.mdiArea.subWindowActivated.connect(self.on_selected_graph)

    def add_graph(self):
        """Plot new graph"""
        # Increment plot number
        self.plot_number += 1

        # Get the selected dataframe
        if self.filtered_df is None:
            df = self.sel_df
        else:
            df = self.filtered_df

        x = self.ui.cbb_x_2.currentText()
        y = self.ui.cbb_y_2.currentText()
        z = self.ui.cbb_z_2.currentText()
        title = self.ui.lbl_plot_title.text()
        plot_style = self.get_plot_style()

        # Create an instance of the Graph class
        graph = Graph(plot_number=self.plot_number)

        graph.plot_style = plot_style
        graph.plot_title = title
        graph.df = df
        graph.x = x
        graph.y = y
        graph.z = z if z != "None" else None
        graph.xlabel = x
        graph.ylabel = y
        graph.zlabel = z

        # Plot the graph
        graph.plot()

        # Store the plot in the dictionary with plot_number as key
        self.plots[self.plot_number] = graph

        # Create a QDialog to hold the Graph instance
        graph_dialog = QDialog(self)
        graph_dialog.setWindowTitle(
            f"Graph_{self.plot_number}_{y}_vs._{y}")
        layout = QVBoxLayout()
        layout.addWidget(graph)
        graph_dialog.setLayout(layout)

        # Add the QDialog to the mdiArea
        sub_window = QMdiSubWindow()
        sub_window.setWidget(graph_dialog)
        self.ui.mdiArea.addSubWindow(sub_window)
        sub_window.show()

    def upd_graph(self):
        """ Update the existing graph with new properties"""
        sel_graph = self.get_sel_graph()
        if sel_graph:
            df = self.sel_df if self.filtered_df is None else \
                self.filtered_df
            plot_style = self.get_plot_style()
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
            zmax = self.ui.xmax_2.text()
            palette = self.ui.cbb_palette.currentText()
            x_rot = float(self.ui.x_rot.text())


            # Apply values for "graph" object
            sel_graph.df = df
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
            sel_graph.plot()
            sel_graph.setWindowTitle(
                f"Graph_{sel_graph.plot_number}_{x}_vs._{y}")

    def on_selected_graph(self, sub_window):
        """Reflect all properties of selected graph object to GUI"""
        sel_graph = self.get_sel_graph()

        if sel_graph:
            # Update plot style radio buttons
            if sel_graph.plot_style == "point":
                self.ui.rdbtn_point.setChecked(True)
            elif sel_graph.plot_style == "scatter":
                self.ui.rdbtn_scatter.setChecked(True)
            elif sel_graph.plot_style == "box":
                self.ui.rdbtn_box.setChecked(True)
            elif sel_graph.plot_style == "heatmap":
                self.ui.rdbtn_heatmap.setChecked(True)
            elif sel_graph.plot_style == "line":
                self.ui.rdbtn_line.setChecked(True)
            elif sel_graph.plot_style == "bar":
                self.ui.rdbtn_bar.setChecked(True)
            elif sel_graph.plot_style == "wafer":
                self.ui.rdbtn_wafer.setChecked(True)
            elif sel_graph.plot_style == "histogram":
                self.ui.rdbtn_histogram.setChecked(True)

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

            # Update QLineEdit:
            self.ui.lbl_plot_title.setText(sel_graph.plot_title)
            self.ui.lbl_xlabel.setText(sel_graph.xlabel)
            self.ui.lbl_ylabel.setText(sel_graph.ylabel)
            self.ui.lbl_zlabel.setText(sel_graph.zlabel)

    def get_plot_style(self):
        if self.ui.rdbtn_point.isChecked():
            plot_style = "point"
        elif self.ui.rdbtn_scatter.isChecked():
            plot_style = "scatter"
        elif self.ui.rdbtn_box.isChecked():
            plot_style = "box"
        elif self.ui.rdbtn_heatmap.isChecked():
            plot_style = "heatmap"
        elif self.ui.rdbtn_line.isChecked():
            plot_style = "line"
        elif self.ui.rdbtn_bar.isChecked():
            plot_style = "bar"
        elif self.ui.rdbtn_wafer.isChecked():
            plot_style = "wafer"
        elif self.ui.rdbtn_histogram.isChecked():
            plot_style = "histogram"
        return plot_style

    def open_dfs(self, dfs=None, fnames=None):
        if self.dfs is None:
            self.dfs = {}
        if dfs:
            self.dfs = dfs
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
                            self.dfs[df_name] = pd.read_excel(
                                excel_file, sheet_name=sheet_name)
                    else:
                        pass
        self.upd_dfs_list()

    def upd_dfs_list(self):
        """ To update the dataframe listbox"""
        current_row = self.ui.dfs_listbox.currentRow()
        self.ui.dfs_listbox.clear()
        df_names = list(self.dfs.keys())
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

    def get_sel_graph(self):
        """Get the canvas of the selected sub window"""
        sub_window = self.ui.mdiArea.activeSubWindow()
        if sub_window:
            graph_dialog = sub_window.widget()
            if graph_dialog:
                graph = graph_dialog.layout().itemAt(0).widget()
                if graph:
                    sel_graph = graph
        return sel_graph

    def get_sel_df(self):
        """Get selected dataframe"""
        sel_item = self.ui.dfs_listbox.currentItem()
        sel_df_name = sel_item.text()
        self.sel_df = self.dfs[sel_df_name]
        return self.sel_df

    def show_df(self):
        """To view selected dataframe"""
        if self.filtered_df is None:
            df = self.sel_df
        else:
            df = self.filtered_df

        if df is not None:
            view_df(self.ui.tabWidget, df)
        else:
            show_alert("No fit dataframe to display")

    def show_df_in_gui(self):
        self.sel_df = self.get_sel_df()
        if self.filtered_df is None:
            df = self.sel_df
        else:
            df = self.filtered_df
        common_utils = CommonUtilities()
        common_utils.display_df_in_table(self.ui.tableWidget, df)

    def apply_filters(self):
        """Apply all checked filters to the current dataframe"""
        self.filter.set_dataframe(self.sel_df)
        self.filtered_df = self.filter.apply_filters()
        common_utils = CommonUtilities()
        common_utils.display_df_in_table(self.ui.tableWidget, self.filtered_df)
