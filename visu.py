import os
import numpy as np
import pandas as pd
import json
from copy import deepcopy
from pathlib import Path
import dill

from common import view_df, show_alert, CommonUtilities, Graph

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtWidgets import QApplication, QFileDialog, QDialog, QVBoxLayout, \
    QLineEdit, QListWidgetItem, QMdiSubWindow
from PySide6.QtGui import QResizeEvent
from PySide6.QtCore import Qt, QFileInfo, QTimer, QObject, Signal


class Visu(QDialog):
    """Class to GUI and callbacks"""

    def __init__(self, settings, ui):
        super().__init__()
        self.ui = ui
        self.settings = settings
        self.setWindowTitle("Graph Plot")

        self.dfs = {}
        self.filters = None
        self.filtered_dfs = None

        # DATAFRAME
        self.ui.btn_open_dfs.clicked.connect(self.open_dfs)
        self.ui.btn_view_df_3.clicked.connect(self.show_df)
        self.ui.dfs_listbox.itemSelectionChanged.connect(self.update_gui)

        self.plot_number = 0  # Initialize plot number
        # GRAPH
        self.ui.btn_add_graph.clicked.connect(self.add_graph)

        # Track selected sub-window
        self.ui.mdiArea.subWindowActivated.connect(
            self.update_selected_plot_label)

    def add_graph(self):
        # Increment plot number
        self.plot_number += 1

        # Get the selected dataframe
        sel_df = self.get_sel_df()
        x_value = self.ui.cbb_x_2.currentText()
        y_value = self.ui.cbb_y_2.currentText()
        z_value = self.ui.cbb_z_2.currentText()

        plot_style = self.get_plot_style()

        # Create an instance of the Graph class
        graph = Graph()
        graph.plot_style = plot_style
        # Set x, y, z values and labels
        graph.x = sel_df[x_value]
        graph.y = sel_df[y_value]
        if z_value == "None":
            graph.z = None
        else:
            graph.z = sel_df[z_value]
        graph.xlabel = x_value
        graph.ylabel = y_value
        graph.zlabel = z_value

        # Plot the graph
        graph.plot()

        # Create a QDialog to hold the Graph instance
        graph_dialog = QDialog(self)
        graph_dialog.setWindowTitle(
            f"Graph_{self.plot_number}_{x_value}_vs._{y_value}")
        layout = QVBoxLayout()
        layout.addWidget(graph)
        graph_dialog.setLayout(layout)

        # Add the QDialog to the mdiArea
        sub_window = QMdiSubWindow()
        sub_window.setWidget(graph_dialog)
        self.ui.mdiArea.addSubWindow(sub_window)
        sub_window.show()

    def update_selected_plot_label(self, sub_window):
        if sub_window:
            plot_title = sub_window.windowTitle()
            self.ui.lbl_selected_ploted.setText(plot_title)

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
        self.show_df_in_gui()
        self.update_cbb()

    def update_cbb(self):
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

    def get_sel_df(self):
        """Get selected dataframe"""
        sel_item = self.ui.dfs_listbox.currentItem()
        sel_df_name = sel_item.text()
        sel_df = self.dfs[sel_df_name]
        return sel_df

    def show_df(self):
        """To view selected dataframe"""
        sel_df = self.get_sel_df()
        if sel_df is not None:
            view_df(self.ui.tabWidget, sel_df)
        else:
            show_alert("No fit dataframe to display")

    def show_df_in_gui(self):
        sel_df = self.get_sel_df()
        common_utils = CommonUtilities()
        common_utils.display_df_in_table(self.ui.tableWidget, sel_df)
