# callbacks_plotting.py module
import os
import traceback
import numpy as np
from functools import partial
from io import BytesIO
#import win32clipboard
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from wafer_plot import WaferPlot

from PySide6.QtWidgets import (
    QFileDialog, QVBoxLayout, QMessageBox, QFrame, QPushButton,
    QHBoxLayout, QApplication, QSpacerItem, QSizePolicy
)

from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt, QSize, QCoreApplication, QSettings, QFileInfo

DIRNAME = os.path.dirname(__file__)
PLOT_POLICY = os.path.join(DIRNAME, "resources", "plotpolicy.mplstyle")
ICON_DIR = os.path.join(DIRNAME, "ui", "iconpack")


class CallbacksPlot:
    def __init__(self, ui, callbacks_df):
        self.ui = ui
        self.callbacks_df = callbacks_df  # connect with callbacks_df

        # List to store elements of created plot widgets (plot + buttons),
        self.plot_widgets = []
        self.plot_counter = 0
        self.active_plot_ids = []  # list to keep track of active plot IDs

        self.figure_list = {}  # Dict of all figures for copy to clipboard

        # Specs of plot
        self.plot_specs = {}

        self.selected_plot_style = "scatter plot"  # Set the default plot style
        self.plot_styles = ["scatter plot",
                            "point plot",
                            "point plot: 2 Yaxis (pos(Si) & stress)",
                            "box plot",
                            "bar plot",
                            "------",
                            "2D map_LS6",
                            "2D map_FITSPY",
                            "wafer map (200mm)",
                            "wafer map (300mm)"
                            ]

        self.selected_palette_colors = 'jet'  # Set the default plot style
        self.palette_colors = ['jet', 'viridis', 'plasma', 'inferno', 'magma',
                               'cividis', 'cool', 'hot', 'YlGnBu', 'YlOrRd']

        self.save_dpi = None
        self.display_dpi = None

    def create_plot_widget(self, spec):
        """ To create a Qframe that hold figure and associated buttons
        inside"""
        self.plots_per_line = self.ui.spinBox_plot_per_row.value()

        # determine row to place the plot_widget:
        row = len(self.plot_widgets) // self.plots_per_line
        if spec is not None:
            plot_width = int(spec["plot_width"])
            plot_height = int(spec["plot_height"])
        else:
            plot_width = float(self.ui.ent_plotwidth.text())
            plot_height = float(self.ui.ent_plotheight.text())

        # Create a QFrame to hold the plot and buttons
        plot_widget_frame = QFrame()
        plot_widget_frame.setFixedSize(plot_width, plot_height)
        self.ui.gridLayout.addWidget(plot_widget_frame, row,
                                     self.plot_counter % self.plots_per_line)
        widget_layout = QVBoxLayout()
        plot_widget_frame.setLayout(widget_layout)

        self.plot_counter += 1

        return widget_layout, plot_widget_frame

    def add_a_plot(self, btn_update_plot=None, btn_remove_plot=None,
                   btn_copy_plot=None):
        """ Plotting figures in plot_widget and ensure the arrangement of
        plot_widget in a gridlayout of GUI"""
        # Create a plot widget frame to contain plot and buttons
        widget_layout, plot_widget_frame = self.create_plot_widget(spec=None)

        # Collect spec for the plot
        plot_id = self.plot_counter

        self.plot_specs[plot_id] = self.collect_plot_spec()

        # PLOTTING and create associated button
        self.plot_action(widget_layout, plot_widget_frame, plot_id)

        # Ensure that the scroll area updates its contents
        self.ui.scrollAreaWidgetContents.adjustSize()
        self.ui.scrollAreaWidgetContents.updateGeometry()

        # Store the plot widget in a "plot_widgets" list
        self.plot_widgets.append((plot_widget_frame, btn_update_plot,
                                  btn_remove_plot, btn_copy_plot))
        # tracking remaining plots (whenever a plot is removed)
        self.active_plot_ids.append(plot_id)  #

        self.update_plot_arrangement()

    def add_wafer_plots(self, btn_update_plot=None, btn_remove_plot=None,
                        btn_copy_plot=None):
        """ Similar to "add_a_plot" function, this function is used to plot
        multiples figures for each wafer"""

        df = self.callbacks_df.selected_df
        wafer_names = df['Wafer'].unique()

        # Retrieve dataframe of each wafer and store in a dict{}
        wafer_dfs = {}
        for wafer_name in wafer_names:
            wafer_df = df[df['Wafer'] == wafer_name]
            wafer_dfs[wafer_name] = wafer_df

        # Create plot for each wafer
        for wafer_name, wafer_df in wafer_dfs.items():
            widget_layout, plot_widget_frame = self.create_plot_widget(
                spec=None)
            plot_id = self.plot_counter

            # Collect spec for the plot
            self.plot_specs[plot_id] = self.collect_plot_spec()
            spec = self.plot_specs[plot_id]
            spec["associated_df"] = wafer_df
            spec["wafer_name"] = wafer_name
            # PLOTTING and create associated button
            self.plot_action(widget_layout, plot_widget_frame, plot_id)

            self.ui.scrollAreaWidgetContents.adjustSize()
            self.ui.scrollAreaWidgetContents.updateGeometry()
            self.plot_widgets.append((plot_widget_frame, btn_update_plot,
                                      btn_remove_plot, btn_copy_plot))
            self.active_plot_ids.append(plot_id)
            self.update_plot_arrangement()

    def plot_action(self, widget_layout, plot_widget_frame, plot_id):
        """Use to plot figur with selected spec or reload the workspace"""
        plt.style.use(PLOT_POLICY)

        # Retrieve specs of a plot (plot_id)
        spec = self.plot_specs[plot_id]
        selected_df = spec["associated_df"]
        self.display_dpi = float(spec["display_dpi"])
        try:
            fig, ax = self.create_plot(selected_df, spec)
            self.style_plot(ax, spec)
        except Exception as e:
            self.handle_plot_error(plot_id, str(e))
        self.display_plot(widget_layout, plot_widget_frame, plot_id, fig)

    def plot_with_recipe(self, widget_layout, plot_widget_frame, plot_id):
        """Use to plot figures using a recipe (saved workspace file). """

        plt.style.use(PLOT_POLICY)

        # Retrive specs of a plot (plot_id)
        spec = self.plot_specs[plot_id]
        self.display_dpi = float(spec["display_dpi"])
        # Apply filters
        self.callbacks_df.df_filters = spec["df_filters"]
        self.callbacks_df.apply_selected_filters(self.callbacks_df.df_filters)
        selected_df_name = self.callbacks_df.selected_df_name

        selected_df = self.callbacks_df.selected_df
        spec["associated_df"] = selected_df
        spec["selected_df_name"] = selected_df_name

        try:
            fig, ax = self.create_plot(selected_df, spec)
            self.style_plot(ax, spec)
        except Exception as e:
            self.handle_plot_error(plot_id, str(e))
        self.display_plot(widget_layout, plot_widget_frame, plot_id, fig)

    def reload_plot_recipe(self, btn_update_plot=None, btn_remove_plot=None,
                           btn_copy_plot=None):

        for plot_id, spec in self.plot_specs.items():
            # Create a widget frame to contain Plot and buttons
            widget_layout, plot_widget_frame = self.create_plot_widget(spec)

            self.plot_with_recipe(widget_layout, plot_widget_frame, plot_id)

            # Ensure that the scroll area updates its contents
            self.ui.scrollAreaWidgetContents.adjustSize()
            self.ui.scrollAreaWidgetContents.updateGeometry()

            # Store the plot widget reference in a "plot_widgets" list
            self.plot_widgets.append((plot_widget_frame, btn_update_plot,
                                      btn_remove_plot, btn_copy_plot))
            self.update_plot_arrangement()
            self.active_plot_ids.append(plot_id)

    def create_plot(self, selected_df, spec):
        """Create the plot based on spec and selected_df"""
        x = (int(spec["plot_width"])) / 100
        y = (int(spec["plot_height"]) - 38) / 100  # subtract the buttons size
        fig = plt.figure(figsize=(x, y), dpi=float(spec["display_dpi"]))

        ax = fig.add_subplot(111)
        if spec["selected_plot_style"] in ["wafer map (200mm)",
                                           "wafer map (300mm)", "2D map_LS6",
                                           "2D map_FITSPY"]:
            self.plot_color_map(ax, selected_df, spec)
        else:
            self.plot_other_styles(ax, selected_df, spec)
        return fig, ax

    def plot_other_styles(self, ax, selected_df, spec):
        """Plot for other styles (scatter plot, point plot, box plot,
        bar plot, etc.)"""
        data = selected_df
        x = spec["selected_x_column"]
        y = spec["selected_y_column"]
        hue_column = spec["selected_hue_column"]

        if hue_column == "Select hue values":
            hue = None
        else:
            hue = hue_column if hue_column != "" else None

        alpha = float(spec["alpha"])  # transparency

        # Set X Y limits
        x_min, x_max = spec["x_min"], spec["x_max"]
        y_min, y_max = spec["y_min"], spec["y_max"]
        if x_min and x_max:
            ax.set_xlim(float(x_min), float(x_max))
        if y_min and y_max:
            ax.set_ylim(float(y_min), float(y_max))

        # Plotting
        if spec["selected_plot_style"] == "scatter plot":
            sns.scatterplot(data=data, x=x, y=y, hue=hue, alpha=alpha,
                            s=100, ax=ax)
        elif spec["selected_plot_style"] == "point plot":
            sns.pointplot(data=data, x=x, y=y, hue=hue, linestyle='none',
                          dodge=True, capsize=0.00, ax=ax)
        elif spec["selected_plot_style"] == "box plot":
            sns.boxplot(data=data, x=x, y=y, hue=hue, dodge=True, ax=ax)

        elif spec["selected_plot_style"] == "bar plot":
            sns.barplot(data=data, x=x, y=y, hue=hue, errorbar=None,
                        ax=ax)
        elif spec[
            "selected_plot_style"] == "point plot: 2 Yaxis (pos(Si) & stress)":
            sns.pointplot(data=data, x=x, y=y, hue=hue, linestyle='none',
                          dodge=True, capsize=0.00, ax=ax)
            ax2 = plt.gca().twinx()
            # Convert y_min and y_max to float, handling empty strings
            y_values = selected_df[spec["selected_y_column"]]
            y_min = float(y_min) if y_min else min(y_values)
            y_max = float(y_max) if y_max else max(y_values)

            ymin_ax2 = (521 - y_min) * 0.2175
            ymax_ax2 = (521 - y_max) * 0.2175
            ax2.set_ylim(ymin_ax2, ymax_ax2)
            ax2.set_ylabel("Tensile strain (GPa)")

        # Styling
        x_label = spec["selected_x_column"] if not spec["Xaxis_title"] else \
            spec["Xaxis_title"]
        y_label = spec["selected_y_column"] if not spec["Yaxis_title"] else \
            spec["Yaxis_title"]

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(spec["plot_title"])

    def plot_color_map(self, ax, selected_df, spec):
        """Plot for 2D color map style"""
        if spec["selected_plot_style"] == "2D map_LS6":
            data_array = selected_df.values.astype(float)
            x_range = np.array(selected_df.columns[1:]).astype(float)
            y_range = data_array[:, 0]
            data_array[data_array == 0] = np.nan

            x_min, x_max = x_range[0], x_range[-1]
            y_min, y_max = np.nanmin(y_range), np.nanmax(y_range)

            colorscale_min, colorscale_max = spec["colorscale_min"], spec[
                "colorscale_max"]
            vmin = float(colorscale_min) if colorscale_min else np.nanmin(
                data_array[:, 1:])
            vmax = float(colorscale_max) if colorscale_min else np.nanmax(
                data_array[:, 1:])

            im = ax.imshow(data_array[:, 1:],
                           cmap=spec["palette_colors"],
                           extent=[x_min, x_max, y_min, y_max], vmin=vmin,
                           vmax=vmax)

            ax.set_xlabel("X (µm)")
            ax.set_ylabel("Y (µm)")
            ax.set_title(spec["plot_title"])
            ax.figure.colorbar(im, ax=ax)

        elif spec["selected_plot_style"] == "2D map_FITSPY":
            # Extract X, Y, and Z values from the dataframe
            x_column = spec["selected_x_column"]
            y_column = spec["selected_y_column"]

            # Aggregate duplicate entries
            aggregated_df = selected_df.groupby([y_column, x_column])[
                spec["selected_hue_column"]].mean().reset_index()

            x_values = aggregated_df[x_column].unique()
            y_values = aggregated_df[y_column].unique()

            # Replace 0 values with NaN
            aggregated_df.replace(0, np.nan, inplace=True)

            # Extract Z values from the DataFrame
            z_values = aggregated_df.pivot(index=y_column, columns=x_column,
                                           values=spec[
                                               "selected_hue_column"]).values

            # Set color scale
            colorscale_min, colorscale_max = spec.get(
                "colorscale_min"), spec.get("colorscale_max")
            vmin = float(colorscale_min) if colorscale_min else np.nanmin(
                z_values)
            vmax = float(colorscale_max) if colorscale_max else np.nanmax(
                z_values)

            # Create a 2D meshgrid
            X, Y = np.meshgrid(x_values, y_values)
            im = ax.imshow(z_values, cmap=spec["palette_colors"],
                           origin='lower',
                           extent=[x_values.min(), x_values.max(),
                                   y_values.min(), y_values.max()],
                           vmin=vmin,
                           vmax=vmax
                           )
            ax.set_xlabel("X (µm)")
            ax.set_ylabel("Y (µm)")
            ax.set_title(spec["plot_title"])
            ax.figure.colorbar(im, ax=ax)

        elif spec["selected_plot_style"] == "wafer map (300mm)":
            wdf = WaferPlot(wafer_df=selected_df, wafer_size=300, margin=1,
                            hue=spec["selected_hue_column"])
            wdf.plot(ax, spec, stats=spec["include_stats"])

        elif spec["selected_plot_style"] == "wafer map (200mm)":
            wdf = WaferPlot(wafer_df=selected_df, wafer_size=200, margin=1,
                            hue=spec["selected_hue_column"])
            wdf.plot(ax, spec, stats=spec["include_stats"])

    def style_plot(self, ax, spec):
        """Style the plot based on spec"""
        # Rotation x label
        xlabel_rot = float(spec["xlabel_rot"])
        plt.setp(ax.get_xticklabels(), rotation=xlabel_rot, ha="right",
                 rotation_mode="anchor")

        plt.grid(spec["grid"])  # Add grid or not
        legend_inside = spec.get("legend_inside", False)
        if legend_inside:
            ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        else:
            None
        ax.figure.tight_layout()

    def display_plot(self, widget_layout, plot_widget_frame, plot_id, fig):
        """Display the plot on the UI"""
        canvas = FigureCanvas(fig)  # Create a canvas for fig
        widget_layout.addWidget(canvas)
        # Create associate button
        self.create_plot_buttons(widget_layout, plot_widget_frame, plot_id)
        # Store all FigureCanvas associated with plot_id
        self.figure_list[plot_id] = canvas

    def reload_workspace(self, btn_update_plot=None, btn_remove_plot=None,
                         btn_copy_plot=None):

        for plot_id, spec in self.plot_specs.items():
            # Create a widget frame to contain Plot and buttons
            widget_layout, plot_widget_frame = self.create_plot_widget(spec)
            self.plot_action(widget_layout, plot_widget_frame, plot_id)

            # Ensure that the scroll area updates its contents
            self.ui.scrollAreaWidgetContents.adjustSize()
            self.ui.scrollAreaWidgetContents.updateGeometry()

            # Store the plot widget reference in a "plot_widgets" list
            self.plot_widgets.append((plot_widget_frame, btn_update_plot,
                                      btn_remove_plot, btn_copy_plot))
            self.update_plot_arrangement()
            self.active_plot_ids.append(plot_id)

    def remove_plot_widget(self, plot_widget_frame, plot_id):
        """ To remove entire plot_widget from GUI"""
        # Remove the plot widgets and their references from the list
        for plot_widgets in self.plot_widgets:
            if plot_widgets[0] == plot_widget_frame:
                self.plot_widgets.remove(plot_widgets)
                break
        # Remove the plot specs from spec dictionary
        if plot_id in self.plot_specs:
            del self.plot_specs[plot_id]

        # Remove the plot id from the active_plot_ids list
        if plot_id in self.active_plot_ids:
            self.active_plot_ids.remove(plot_id)

        # Remove the figure from the figure dictionary
        if plot_id in self.figure_list:
            del self.figure_list[plot_id]

        # Delete plot_widget_frame
        plot_widget_frame.deleteLater()

        # update GUI based on remaining plots
        self.update_plot_arrangement()

    def update_plot_handler(self, widget_layout, button_layout,
                            plot_widget_frame, plot_id):
        """Switch between 2 update function by CTRL key"""
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.update2_plot(widget_layout, button_layout, plot_widget_frame,
                              plot_id)
        else:
            self.update_plot(widget_layout, button_layout, plot_widget_frame,
                             plot_id)

    def update_plot(self, widget_layout, button_layout, plot_widget_frame,
                    plot_id):
        """ Update all specs (except x and y values) for an existing plot """
        # Retrieve the current specifications of plot(plot_id)
        existing_spec = self.plot_specs.get(plot_id)
        if existing_spec is None:
            return  # No existing specifications found, return without updating

        # Retrieve the updated plot specifications for the given plot_id
        updated_spec = self.collect_plot_spec()

        # Keep the existing X and Y values unchanged
        updated_spec["associated_df"] = existing_spec["associated_df"]
        updated_spec["selected_x_column"] = existing_spec["selected_x_column"]
        updated_spec["selected_y_column"] = existing_spec["selected_y_column"]
        updated_spec["selected_hue_column"] = existing_spec[
            "selected_hue_column"]
        updated_spec["selected_plot_style"] = existing_spec[
            "selected_plot_style"]
        updated_spec["display_dpi"] = existing_spec["display_dpi"]
        updated_spec["plot_width"] = existing_spec["plot_width"]
        updated_spec["plot_height"] = existing_spec["plot_height"]

        # Check if "wafer_name" key exists in existing_spec before updating
        if "wafer_name" in existing_spec:
            updated_spec["wafer_name"] = existing_spec["wafer_name"]
        else:
            pass

        # Apply the updated specifications to the plot
        self.plot_specs[plot_id] = updated_spec

        # Clear the existing widgets inside the plot_frame_layout
        for i in reversed(range(widget_layout.count())):
            widget = widget_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        for i in reversed(range(button_layout.count())):
            widget = button_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        self.plot_action(widget_layout, plot_widget_frame, plot_id)

    def update2_plot(self, widget_layout, button_layout, plot_widget_frame,
                     plot_id):
        """ To update all specs for an existing plot """
        # Collect the current plot specs
        updated_spec = self.collect_plot_spec()
        # Update current specs to plot of a given plot_id
        self.plot_specs[plot_id] = updated_spec

        # Clear figure canvas inside the plot_frame_layout
        for i in reversed(range(widget_layout.count())):
            widget = widget_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        # Clear buttons inside the button_layout
        for i in reversed(range(button_layout.count())):
            widget = button_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        self.plot_action(widget_layout, plot_widget_frame, plot_id)

    def update_plot_arrangement(self):
        num_plots = len(self.plot_widgets)
        rows = (num_plots + self.plots_per_line - 1) // self.plots_per_line

        for i, (plot_widget_frame, _, _, _) in enumerate(self.plot_widgets):
            row = i // self.plots_per_line
            col = i % self.plots_per_line
            self.ui.gridLayout.addWidget(plot_widget_frame, row, col)

            # Remove empty rows
            for i in range(rows, self.ui.gridLayout.rowCount()):
                self.ui.gridLayout.setRowStretch(i, 0)

            # Configure rows to expand to fill available space
            for i in range(rows):
                self.ui.gridLayout.setRowStretch(i, 1)

        # update plot_widgets
        self.plot_widgets = [plot_widgets for plot_widgets in
                             self.plot_widgets
                             if plot_widgets[0] is not None]

    def copy_plot_label(self, plot_id):
        # Retrieve the current specifications of plot(plot_id)
        existing_spec = self.plot_specs.get(plot_id)
        if existing_spec is None:
            return  # No existing specifications found, return without updating
        xlabel = existing_spec["Xaxis_title"]
        ylabel = existing_spec["Yaxis_title"]
        title = existing_spec["plot_title"]
        self.ui.ent_xaxis_title.setText(xlabel)
        self.ui.ent_yaxis_title.setText(ylabel)
        self.ui.ent_plot_title.setText(title),

    def creat_btn(self, icon_file, tooltip):
        """ To create buttons associated with each plot"""
        btn_name = QPushButton("")
        icon = QIcon()
        icon.addFile(os.path.join(ICON_DIR, icon_file))
        btn_name.setIcon(icon)
        btn_name.setIconSize(QSize(25, 25))
        btn_name.setToolTip(
            QCoreApplication.translate("mainWindow", tooltip, None))
        return btn_name

    def create_plot_buttons(self, widget_layout, plot_widget_frame, plot_id):
        """ To create function buttons associated with each plot"""
        btn_copy_labels = self.creat_btn("copy_label.png",
                                         u"Copy axis labels to another plot")
        btn_remove_plot = self.creat_btn("remove.png", u"Remove plot")
        btn_update_plot = self.creat_btn("update.png",
                                         u"Update plot styles (title, "
                                         u"axis limits, ...)\n(Hold Ctrl "
                                         u"key if u want to update axis "
                                         u"values)")
        btn_copy_plot = self.creat_btn("copy.png", u"Copy figure to clipboard")

        btn_remove_plot.clicked.connect(
            partial(self.remove_plot_widget, plot_widget_frame, plot_id))
        btn_update_plot.clicked.connect(
            lambda: self.update_plot_handler(widget_layout, button_layout,
                                             plot_widget_frame, plot_id))
        btn_copy_labels.clicked.connect(
            lambda: self.copy_plot_label(plot_id))
        btn_copy_plot.clicked.connect(partial(self.copy_plot, plot_id))

        # Create a QHBoxLayout to hold the buttons
        button_layout = QHBoxLayout()
        spacer = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        button_layout.addItem(spacer)

        button_layout.addWidget(btn_remove_plot)
        button_layout.addWidget(btn_update_plot)
        button_layout.addWidget(btn_copy_labels)
        button_layout.addWidget(btn_copy_plot)

        widget_layout.addLayout(button_layout)

    def set_selected_plot_style(self, index):
        self.selected_plot_style = self.plot_styles[index]

    def set_selected_palette_colors(self, index):
        self.selected_palette_colors = self.palette_colors[index]

    def save_all_figs(self, plot_id):
        self.save_dpi = float(self.ui.ent_plot_save_dpi.text())
        # Initialize the last used directory from QSettings
        last_dir = self.callbacks_df.settings.value("last_directory", "/")
        save_dir = QFileDialog.getExistingDirectory(
            self.ui.tabWidget, "Select Folder to Save all figures", last_dir)
        if save_dir:

            for plot_id in self.active_plot_ids:
                canvas_widget = self.figure_list.get(plot_id)
                if canvas_widget:
                    canvas = canvas_widget.figure
                    filename = f"{save_dir}/figure_{plot_id}.png"
                    canvas.savefig(filename, dpi=self.save_dpi)
                    plt.close(canvas)

    def copy_plot(self, plot_id):
        self.save_dpi = float(self.ui.ent_plot_save_dpi.text())
        canvas_widget = self.figure_list.get(plot_id)
        if canvas_widget:
            figure = canvas_widget.figure
            with BytesIO() as buf:
                figure.savefig(buf, format='png', dpi=self.save_dpi)
                data = buf.getvalue()
            format_id = win32clipboard.RegisterClipboardFormat('PNG')
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(format_id, data)
            win32clipboard.CloseClipboard()
        else:
            QMessageBox.critical("Error", "No plot to copy.")

    def handle_plot_error(self, plot_id, error_message):
        """Handle the error occurred during plot rendering"""
        QMessageBox.critical(self.ui.tabWidget, "Plotting Error",
                             f"An error occurred while plotting:\n"
                             f"{error_message}\n\nPlot {plot_id} will be "
                             f"removed.",
                             QMessageBox.Ok)
        del self.plot_specs[plot_id]  # Remove the spec of the failed plot

    def clear_workspace(self):
        """ To cleanout entire plot_widgets from GUI"""
        for plot_widget_frame, _, _, _ in self.plot_widgets:
            # Remove the plot_widget_frame from the layout
            self.ui.gridLayout.removeWidget(plot_widget_frame)
            plot_widget_frame.deleteLater()

        self.plot_widgets.clear()
        self.plot_specs.clear()
        self.active_plot_ids.clear()
        self.figure_list.clear()
        self.callbacks_df.df_filters.clear()
        # self.callbacks_df.filter_cb_state.clear()
        # Clear the plot_specs dictionary

        self.callbacks_df.original_dfs = {}
        self.callbacks_df.working_dfs = {}
        self.callbacks_df.selected_df = pd.DataFrame()
        self.callbacks_df.selected_x_column = ""
        self.callbacks_df.selected_y_column = ""
        self.callbacks_df.selected_hue_column = ""
        self.plot_counter = 0
        self.callbacks_df.update_listbox_dfs()
        self.ui.filter_list.clear()
        self.clear_entries()

    def clear_entries(self):
        line_edits_to_clear = [
            self.ui.ent_plot_title,
            self.ui.ent_xaxis_title,
            self.ui.ent_yaxis_title,
            self.ui.ent_hue_title,
            self.ui.ent_xmin,
            self.ui.ent_xmax,
            self.ui.ent_ymin,
            self.ui.ent_ymax,
            self.ui.ent_colorscale_min,
            self.ui.ent_colorscale_max
        ]
        for line_edit in line_edits_to_clear:
            line_edit.clear()
        self.ui.ent_plotwidth.setText("470")
        self.ui.ent_plotheight.setText("400")

    def collect_plot_spec(self):
        # Collect and return the plot specifications as a dictionary
        spec = {
            # dataframe
            "selected_df_name": self.callbacks_df.selected_df_name,
            "associated_df": self.callbacks_df.selected_df,
            "df_filters": self.callbacks_df.df_filters,

            "selected_plot_style": self.selected_plot_style,

            "selected_x_column": self.callbacks_df.selected_x_column,
            "selected_y_column": self.callbacks_df.selected_y_column,
            "selected_hue_column": self.callbacks_df.selected_hue_column,

            "plot_width": self.ui.ent_plotwidth.text(),
            "plot_height": self.ui.ent_plotheight.text(),
            "display_dpi": self.ui.ent_plot_display_dpi.text(),

            "x_min": self.ui.ent_xmin.text(),
            "x_max": self.ui.ent_xmax.text(),
            "y_min": self.ui.ent_ymin.text(),
            "y_max": self.ui.ent_ymax.text(),
            "colorscale_min": self.ui.ent_colorscale_min.text(),
            "colorscale_max": self.ui.ent_colorscale_max.text(),

            "plot_title": self.ui.ent_plot_title.text(),
            "Xaxis_title": self.ui.ent_xaxis_title.text(),
            "Yaxis_title": self.ui.ent_yaxis_title.text(),
            "hueaxis_title": self.ui.ent_hue_title.text(),

            "palette_colors": self.selected_palette_colors,

            "xlabel_rot": self.ui.ent_xlabel_rot.text(),
            "alpha": self.ui.lineEdit_2.text(),

            # include state of wafer in plot
            "include_stats": self.ui.checkBox.isChecked(),
            "grid": self.ui.checkBox_3.isChecked(),
            "legend_inside": self.ui.checkBox_2.isChecked(),

        }
        return spec
