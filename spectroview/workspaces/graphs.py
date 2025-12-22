import os
import pandas as pd
import json
import gzip
from pathlib import Path
from io import StringIO

from spectroview import PLOT_STYLES, LEGEND_LOCATION, ICON_DIR

from spectroview.modules.utils import view_df, show_alert, copy_fig_to_clb
from spectroview.modules.df_filter import DataframeFilter
from spectroview.modules.graph import Graph
from spectroview.modules.utils import CustomizedPalette

from PySide6.QtWidgets import QWidget, QFileDialog, QDialog, QVBoxLayout, QListWidgetItem, QMdiSubWindow, QCheckBox, QMessageBox,QLabel
from PySide6.QtCore import Qt, QTimer, Signal, QSize
from PySide6.QtGui import  QIcon, Qt


class Graphs(QDialog):
    """This class provides a GUI for plotting graphs/figures."""
    def __init__(self, settings, ui):
        super().__init__()
        self.ui = ui
        self.settings = settings
        
        self.setWindowTitle("Graph Plot")

        # DATAFRAME
        self.original_dfs = {}
        self.selected_df = None
        self.ui.btn_view_df_3.clicked.connect(self.show_df)
        self.ui.dfs_listbox.itemSelectionChanged.connect(self.update_gui)
        
        
        self.ui.btn_remove_df_2.clicked.connect(self.remove_df)
        self.ui.btn_save_df_2.clicked.connect(self.save_df_to_excel)

        # FILTER
        self.filter = DataframeFilter(self.selected_df)
        self.ui.filter_widget_layout.addWidget(self.filter.gb_filter_widget)
        self.filtered_df = None

        # GRAPH
        self.plots = {}
        self.graph_id = 0  # Initialize graph number
        # Add a graph
        self.ui.btn_add_graph.clicked.connect(self.plotting)
        self.ui.btn_add_multi_wafer_plots.clicked.connect(self.plotting_multi_wafer_plots)

        self.ui.btn_get_limits.clicked.connect(self.set_current_limits)
        self.ui.btn_clear_limits.clicked.connect(self.clear_limits)
        # Update an existing graph
        self.ui.btn_upd_graph.clicked.connect(lambda: self.plotting(update_graph=True))

        # GRAPH: add 2nd and 3rd lines for the current ax
        self.ui.btn_add_y12.clicked.connect(self.add_y12)
        self.ui.btn_add_y13.clicked.connect(self.add_y13)
        # GRAPH: add twin axis (second and third y axis)
        self.ui.btn_add_y2.clicked.connect(self.add_y2)
        self.ui.btn_remove_y2.clicked.connect(self.remove_y2)
        self.ui.btn_add_y3.clicked.connect(self.add_y3)
        self.ui.btn_remove_y3.clicked.connect(self.remove_y3)

        self.ui.btn_copy_graph.clicked.connect(self.copy_fig_to_clb)

        self.cbb_palette = CustomizedPalette()
        self.cbb_palette.currentIndexChanged.connect(lambda: self.plotting(update_graph=True))
        self.ui.horizontalLayout_115.addWidget(self.cbb_palette)

        # Plot_style comboboxes
        self.ui.cbb_plotstyle.setIconSize(QSize(40, 40))
        for style in PLOT_STYLES:
            icon_path = os.path.join(ICON_DIR, f"{style}.png")
            if os.path.exists(icon_path):
                icon = QIcon(icon_path)
            else:
                icon = QIcon()  # Fallback in case the icon is missing
            self.ui.cbb_plotstyle.addItem(icon, style)
        self.ui.cbb_plotstyle.setToolTip("Select Plot Style")
        self.ui.cbb_plotstyle.currentIndexChanged.connect(self.auto_select_XY_for_wafer_plot)

        self.ui.cbb_legend_loc.addItems(LEGEND_LOCATION)
        
        # Track selected sub-window
        self.ui.mdiArea.subWindowActivated.connect(self.on_selected_graph)
        self.ui.cbb_graph_list.currentIndexChanged.connect(
            self.select_sub_window_from_combo_box)

        self.ui.btn_minimize_all.clicked.connect(self.minimize_all_graph)
                 
    def update_gui(self):
        """Update the GUI elements based on the selected dataframe"""
        self._update_cbb()
        self.show_wafer_slot_selector()
        self.auto_select_XY_for_wafer_plot()
        self.selected_df = self.get_sel_df()
        
    
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
                    fname = file_path.stem  # get fname without extension
                    extension = file_path.suffix.lower()
                    if extension == '.xlsx':
                        # Open and read all sheets into memory, then close the file
                        with pd.ExcelFile(file_path) as excel_file:
                            sheet_names = excel_file.sheet_names
                            for sheet_name in sheet_names:
                                sheet_name_cleaned = sheet_name.replace(" ", "")
                                df_name = f"{fname}_{sheet_name_cleaned}"
                                # Read each sheet and store in self.original_dfs
                                self.original_dfs[df_name] = pd.read_excel(
                                    excel_file, sheet_name=sheet_name)
                    else:
                        show_alert(f"Unsupported file format: {extension}")
        self.update_dfs_list()

    def update_dfs_list(self):
        """Update the listbox showing available dataframes."""
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
            graph_id = min(available_ids) if available_ids else len(self.plots) + 1
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

        graph.xlogscale = self.ui.xaxis_log_scale.isChecked()
        graph.ylogscale = self.ui.yaxis_log_scale.isChecked()

        # Check if z has changed and reset legend_properties if needed
        self._is_z_changed(graph)

        graph.x = x
        if len(graph.y) == 0:
            graph.y.append(y)
        else:
            graph.y[0] = y
        graph.z = z if z != "None" else None

        graph.color_palette = self.cbb_palette.currentText()
        graph.wafer_size = float(self.ui.lbl_wafersize.text())

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
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(graph)
            graph_dialog.setLayout(layout)
            graph_dialog.setContentsMargins(2, 2, 2, 0)

            # Add the QDialog to a QMdiSubWindow
            sub_window = MdiSubWindow(graph_id, self.ui.lbl_figsize, mdi_area=self.ui.mdiArea)
            sub_window.setWidget(graph_dialog)
            # When creating a new graph dialog
            sub_window.closed.connect(lambda graph_id=graph.graph_id: self.delete_graph(graph_id))

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

            graph.x_rot = float(self.ui.x_rot.text())
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

        text = f"{graph.graph_id}-{graph.plot_style}_plot: [{x}] - [{y}] - [{z}]"
        graph_dialog.setWindowTitle(text)

        # Plot action
        QTimer.singleShot(100, self._plot_action)
        QTimer.singleShot(200, self.customize_legend)
        
    def plotting_multi_wafer_plots(self):
        """ Plot multiple wafer plots based on selected slots (via checkboxes)."""
        sel_df = self.selected_df
        if sel_df is None:
            show_alert("No dataframe is selected. Please select a dataframe before plotting.")
            return

        # Only relevant for wafer plots
        self.ui.cbb_plotstyle.setCurrentIndex(6)

        if self.ui.cbb_z_2.currentText().strip().lower() == "none":
            show_alert("No Z value selected. Please choose a data column for the wafer map.")
            return

    
        # Collect checked slots from the slot selector
        checked_slots = [cb.text() for cb in getattr(self, 'slot_checkboxes', []) if cb.isChecked()]
        if not checked_slots:
            show_alert("No wafer slots are selected. Please check at least one slot before plotting.")
            return

        # Base filters from GUI
        base_filters = self.filter.get_current_filters() or []

        # Helper to produce a canonical expression string for a slot number
        def slot_expression(n):
            return f"Slot == {n}"

        # Helper: merge base_filters with (Slot == n) ensuring only this slot is active
        def merged_filters_with_slot(slot_num):
            expr_to_activate = slot_expression(slot_num)
            merged = []
            slot_found = False
            for f in base_filters:
                f_copy = dict(f)
                if "Slot" in f_copy.get("expression", ""):
                    # deactivate all Slot filters first
                    f_copy["state"] = False
                    if f_copy["expression"] == expr_to_activate:
                        f_copy["state"] = True
                        slot_found = True
                merged.append(f_copy)
            if not slot_found:
                # append the active Slot filter if not present
                merged.append({"expression": expr_to_activate, "state": True})
            return merged

        # For each checked slot, create and plot a new Graph
        for slot_text in checked_slots:
            try:
                slot_num = int(slot_text)
            except Exception:
                slot_num = slot_text

            # Build filters for this slot plot
            new_filters = merged_filters_with_slot(slot_num)

            # Determine a new graph_id same way plotting() does
            available_ids = [i for i in range(1, len(self.plots) + 2) if i not in self.plots]
            graph_id = min(available_ids) if available_ids else len(self.plots) + 1

            # Create and register the Graph object
            graph = Graph(graph_id=graph_id)
            self.plots[graph.graph_id] = graph

            # Collect GUI properties
            graph.plot_style = self.ui.cbb_plotstyle.currentText()
            title = self.ui.lbl_plot_title.text()
            graph.plot_title = title if title != "None" else None
            current_df_name = self.ui.dfs_listbox.currentItem().text() if self.ui.dfs_listbox.currentItem() else None
            graph.df_name = current_df_name
            graph.filters = new_filters

            x = self.ui.cbb_x_2.currentText()
            y = self.ui.cbb_y_2.currentText()
            z = self.ui.cbb_z_2.currentText()

            if new_filters != graph.filters:
                graph.legend_properties = []

            graph.x = x
            if len(graph.y) == 0:
                graph.y.append(y)
            else:
                graph.y[0] = y
            graph.z = z if z != "None" else None

            graph.color_palette = self.cbb_palette.currentText()
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

            # Create the plotting widget and subwindow
            graph.create_plot_widget(graph.dpi)
            graph_dialog = QDialog(self)
            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(graph)
            graph_dialog.setLayout(layout)
            graph_dialog.setContentsMargins(2, 2, 2, 0)

            sub_window = MdiSubWindow(graph_id, self.ui.lbl_figsize, mdi_area=self.ui.mdiArea)
            sub_window.setWidget(graph_dialog)
            sub_window.closed.connect(lambda graph_id=graph.graph_id: self.delete_graph(graph_id))
            sub_window.resize(graph.plot_width, graph.plot_height)
            self.ui.mdiArea.addSubWindow(sub_window)
            sub_window.show()

            # Set window title and add to combobox
            text = f"{graph.graph_id}-{graph.plot_style}_plot: [{x}] - [{y}] - [{z}]"
            graph_dialog.setWindowTitle(text)
            self.add_graph_list_to_combobox()

            # Apply filters and plot
            filtered_df = self.apply_filters(self.selected_df, graph.filters)
            try:
                if graph.plot_style == 'wafer':
                    graph.create_plot_widget(graph.dpi, graph.graph_layout)
                    graph.plot(filtered_df)
                else:
                    graph.plot(filtered_df)
            except Exception as e:
                show_alert(f"Failed plotting slot {slot_num}: {e}")

        QTimer.singleShot(200, self.customize_legend)

    def _plot_action(self):
        """Perform the actual plotting of the graph."""
        graph, graph_dialog, sub_window = self.get_sel_graph()
        self.filtered_df = self.apply_filters(self.selected_df, graph.filters)
      
        if graph:
            if graph.plot_style == 'wafer':
                graph.create_plot_widget(graph.dpi, graph.graph_layout)
                graph.plot(self.filtered_df)
            else:
                graph.plot(self.filtered_df)

    def _is_z_changed(self, graph):
        """Check if z-axis value has changed from the current graph settings"""
        current_z = self.ui.cbb_z_2.currentText()
        if current_z != graph.z:
            graph.legend_properties = []
            return True
        return False

    def customize_legend(self):
        """ Show all legend's properties in GUI for customization"""
        graph, graph_dialog, sub_window = self.get_sel_graph()
        main_layout = self.ui.main_layout
        graph.customize_legend_widget(main_layout)

    def set_current_limits(self):
        """Get and set current scales for selected plot"""
        graph, graph_dialog, sub_window = self.get_sel_graph()
        graph.xmin, graph.xmax = graph.ax.get_xlim()
        graph.ymin, graph.ymax = graph.ax.get_ylim()
        def format_value(value):
            if isinstance(value, (int, float)):  # Check if the value is a number
                return str(round(value, 3))
            elif value is None:  # Handle None values
                return ""
            else:  # If the value is already a string or another type
                return str(value)
        # Update the QLineEdit widgets
        self.ui.xmin_2.setText(format_value(graph.xmin))
        self.ui.xmax_2.setText(format_value(graph.xmax))
        self.ui.ymin_2.setText(format_value(graph.ymin))
        self.ui.ymax_2.setText(format_value(graph.ymax))
        self.plotting(update_graph=True)

    def clear_limits(self):
        """Clear all entryboxes of x and y limits"""
        self.ui.xmin_2.clear()
        self.ui.xmax_2.clear()
        self.ui.ymin_2.clear()
        self.ui.ymax_2.clear()
    
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
            self._reflect_filters_to_gui(graph)

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

            self.ui.xaxis_log_scale.setChecked(graph.xlogscale)
            self.ui.yaxis_log_scale.setChecked(graph.ylogscale)


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

            # Reflect limits values:
            def format_value(value):
                if isinstance(value, (int, float)):  # Check if the value is a number
                    return str(round(value, 3))
                elif value is None:  # Handle None values
                    return ""
                else:  # If the value is already a string or another type
                    return str(value)

            # Update the QLineEdit widgets
            self.ui.xmin_2.setText(format_value(graph.xmin))
            self.ui.xmax_2.setText(format_value(graph.xmax))
            self.ui.ymin_2.setText(format_value(graph.ymin))
            self.ui.ymax_2.setText(format_value(graph.ymax))
            self.ui.y2min_2.setText(format_value(graph.y2min))
            self.ui.y2max_2.setText(format_value(graph.y2max))
            self.ui.y3min_2.setText(format_value(graph.y3min))
            self.ui.y3max_2.setText(format_value(graph.y3max))
            self.ui.zmax_2.setText(format_value(graph.zmax))
            self.ui.zmin_2.setText(format_value(graph.zmin))

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
            combo_items = [self.cbb_palette.itemText(i) for i in
                           range(self.cbb_palette.count())]
            if color_palette in combo_items:
                self.cbb_palette.setCurrentText(color_palette)

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

    def _reflect_filters_to_gui(self, sel_graph):
        """Reflect the state of filters associated with a graph to the GUI"""
        # Clear the existing items and uncheck them
        for index in range(self.filter.filter_listbox.count()):
            item = self.filter.filter_listbox.item(index)
            if isinstance(item, QListWidgetItem):
                widget = self.filter.filter_listbox.itemWidget(item)
                if isinstance(widget, QCheckBox):
                    widget.setChecked(False)

        for filter_info in sel_graph.filters:
            filter_expression = filter_info["expression"]
            filter_state = filter_info["state"]
            # Check if the filter expression already exists in the listbox
            existing_item = None
            for index in range(self.filter.filter_listbox.count()):
                item = self.filter.filter_listbox.item(index)
                if isinstance(item, QListWidgetItem):
                    widget = self.filter.filter_listbox.itemWidget(item)
                    if isinstance(widget,
                                  QCheckBox) and widget.text() == \
                            filter_expression:
                        existing_item = item
                        break
            # Update the state if the filter expression already exists,
            # otherwise add a new item
            if existing_item:
                checkbox = self.filter.filter_listbox.itemWidget(existing_item)
                checkbox.setChecked(filter_state)
            else:
                item = QListWidgetItem()
                checkbox = QCheckBox(filter_expression)
                checkbox.setChecked(filter_state)
                item.setSizeHint(checkbox.sizeHint())
                self.filter.filter_listbox.addItem(item)
                self.filter.filter_listbox.setItemWidget(item, checkbox)

    def _update_cbb(self):
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
        copy_fig_to_clb(sel_graph.canvas)

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
                self.selected_df = self.original_dfs[sel_df_name]
            else:
                self.selected_df = None 
        else:
            self.selected_df = None 
        return self.selected_df

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
        current_df = self.apply_filters(self.selected_df, current_filters)
        if current_df is not None:
            view_df(self.ui.tabWidget, current_df)
        else:
            show_alert("No fit dataframe to display")

    def apply_filters(self, df=None, filters=None):
        """Apply filters to the selected dataframe and return the filtered dataframe."""
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

    def auto_select_XY_for_wafer_plot(self):
        """Auto select X and Y columns if 'wafer' plot style is selected."""
        if self.ui.cbb_plotstyle.currentText().lower() == 'wafer':
            # Auto select X and Y columns if 'X' and 'Y' exist in dataframe
            sel_df = self.get_sel_df()
            if sel_df is not None:
                columns = sel_df.columns.tolist()
                if 'X' in columns and 'Y' in columns:
                    self.ui.cbb_x_2.setCurrentText('X')
                    self.ui.cbb_y_2.setCurrentText('Y')
    
    def show_wafer_slot_selector(self):
        """Build UI for slot selector checkboxes if 'Slot' column exists in the selected dataframe."""
        layout = self.ui.layout_slotselector 
        # Clear existing widgets
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        sel_df = self.selected_df
        if sel_df is not None and 'Slot' in sel_df.columns:
            unique_slots = sorted(sel_df['Slot'].dropna().unique())
            if not unique_slots:
                return
            # --- Add QLabel before the "Select All" checkbox ---
            label = QLabel("Select wafer slot:")
            label.setStyleSheet("font-weight: bold;")  
            layout.addWidget(label, 0, 0, 1, 3)

            # "Select All" checkbox (placed on first row)
            self.select_all_checkbox = QCheckBox("Select All")
            self.select_all_checkbox.setChecked(True)
            layout.addWidget(self.select_all_checkbox, 0, 3, 1, 6)

            # Create individual slot checkboxes
            self.slot_checkboxes = []
            row, col = 1, 0
            for slot in unique_slots:
                cb = QCheckBox(str(slot))
                cb.setChecked(True)
                layout.addWidget(cb, row, col)
                self.slot_checkboxes.append(cb)

                col += 1
                if col >= 9:  # wrap after 9 columns
                    col = 0
                    row += 1

            # Connect signals
            self.select_all_checkbox.stateChanged.connect(self.toggle_all_slots)
            for cb in self.slot_checkboxes:
                cb.stateChanged.connect(self.update_select_all_status)

    def toggle_all_slots(self, checked: bool):
        """Toggle all slot checkboxes when Select All is changed."""
        for cb in getattr(self, 'slot_checkboxes', []):
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        if hasattr(self, 'select_all_checkbox'):
            self.select_all_checkbox.setChecked(checked)

    def update_select_all_status(self):
        """Update Select All checkbox if any individual slot checkbox is unchecked."""
        # compute new desired state
        all_checked = all(cb.isChecked() for cb in getattr(self, 'slot_checkboxes', []))
        if hasattr(self, 'select_all_checkbox'):
            self.select_all_checkbox.blockSignals(True)
            self.select_all_checkbox.setChecked(all_checked)
            self.select_all_checkbox.blockSignals(False)
            
    def minimize_all_graph(self):
        for sub_window in self.ui.mdiArea.subWindowList():
            sub_window.showMinimized()

    def clear_env(self):
        # Clear original dataframes
        self.original_dfs = {}
        self.selected_df = None
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
        self.filter.filter_listbox.clear()
        self.ui.cbb_graph_list.clear()
        self.clear_limits()
        print("'Graphs' Tab environment has been cleared.")

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
                
                compressed_dfs = {}
                for k, v in self.original_dfs.items():
                    # Convert DataFrame to a CSV string and compress it
                    compressed_df = v.to_csv(index=False).encode('utf-8')
                    compressed_dfs[k] = gzip.compress(compressed_df)

                # Prepare data to save
                data_to_save = {
                    'plots': plots_data,
                    'original_dfs': {k: v.hex() for k, v in compressed_dfs.items()},
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

                self.original_dfs = {}
                for k, v in load.get('original_dfs', {}).items():
                        compressed_data = bytes.fromhex(v)
                        csv_data = gzip.decompress(compressed_data).decode('utf-8')
                        self.original_dfs[k] = pd.read_csv(StringIO(csv_data)) 
                
                self.update_dfs_list()
                
                # Load plots
                plots_data = load.get('plots', {})
                for graph_id, graph_data in plots_data.items():
                    # Recreate graph instance
                    graph = Graph(graph_id=graph_id)
                    graph.filters = graph_data.get('filters', [])
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
                    layout.setContentsMargins(0, 0, 0, 0)
                    layout.addWidget(graph)
                    graph_dialog.setLayout(layout)
                    graph_dialog.setContentsMargins(2, 2, 2, 0)

                    # Add the QDialog to the mdiArea
                    sub_window = MdiSubWindow(graph_id, self.ui.lbl_figsize, mdi_area=self.ui.mdiArea)
                    sub_window.setWidget(graph_dialog)
                    
                    # Connect the closed signal to delete_graph with a lambda to pass graph_id
                    sub_window.closed.connect(lambda _, graph_id=graph.graph_id: self.delete_graph(graph_id))

                    self.ui.mdiArea.addSubWindow(sub_window)
                    sub_window.resize(graph.plot_width, graph.plot_height)
                    sub_window.show()

                    self._plot_action()
                    
                self.filter.upd_filter_listbox()
                self.add_graph_list_to_combobox()

        except Exception as e:
            show_alert(f"Error loading saved work (Graphs Tab): {e}")
            print(f"Error loading work: {e}")

    def delete_graph(self, graph_id):
        """Delete the specified graph from the plots dictionary by graph_id"""
        graph = self.plots.get(graph_id)
        if graph is None:
            return

        sub_window = None
        # Find the subwindow related to the graph
        for window in self.ui.mdiArea.subWindowList():
            if isinstance(window, MdiSubWindow) and window.graph_id == graph_id:
                sub_window = window
                break

        # Remove the graph and close the subwindow
        if graph_id in self.plots:
            self.plots.pop(graph_id, None)
            if sub_window:
                self.ui.mdiArea.removeSubWindow(sub_window)
                sub_window.close()
            self.add_graph_list_to_combobox()
            print(f"Plot {graph_id} deleted")

            
class MdiSubWindow(QMdiSubWindow):
    """
    Custom class of QMdiSubWindow to prevent automatic selection of other windows
    when one subwindow is closed.
    """
    closed = Signal(int)

    def __init__(self, graph_id, figsize_label, mdi_area, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_id = graph_id
        self.figsize_label = figsize_label
        self.mdi_area = mdi_area  # Reference to the parent QMdiArea

    def closeEvent(self, event):
        """Override closeEvent to prevent automatic selection of another subwindow"""
        # Clear focus to prevent any subwindow from being automatically selected
        self.mdi_area.clearFocus()
        self.mdi_area.setActiveSubWindow(None)

        # Emit the signal when the window is closing
        self.closed.emit(self.graph_id)
        
        # Call the parent close event to actually close the window
        super().closeEvent(event)

    def resizeEvent(self, event):
        """Override resizeEvent to handle window resizing"""
        new_size = self.size()
        width, height = new_size.width(), new_size.height()
        # Update QLabel with the new size
        self.figsize_label.setText(f"({width}x{height})")
        super().resizeEvent(event)

    def focusInEvent(self, event):
        """Override focusInEvent to prevent automatic selection"""
        # Prevent the window from being focused (optional)
        if not self.mdi_area.activeSubWindow():
            self.mdi_area.setActiveSubWindow(None)
        super().focusInEvent(event)