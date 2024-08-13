"""
Module contains all utilities common methods of the application
"""
import markdown
import os
import json
import base64
import zlib
from copy import deepcopy
from io import BytesIO
import pandas as pd
import numpy as np
try:
    import win32clipboard
except:
    pass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns

from scipy.interpolate import griddata
from PySide6.QtWidgets import QMessageBox, QDialog, QTableWidget, \
    QTableWidgetItem,  QHBoxLayout, QTextBrowser, QLabel, \
     QWidget, QPushButton, QComboBox, QCheckBox, QListWidgetItem, \
    QApplication,  QWidget, QStyledItemDelegate, QComboBox, QLabel, QVBoxLayout, QLineEdit
from PySide6.QtCore import Signal, QThread, Qt, QSize
from PySide6.QtGui import QPalette,  QTextCursor, QIcon, QResizeEvent, \
    QAction, Qt, QColor


DIRNAME = os.path.dirname(__file__)
RELPATH = os.path.join(DIRNAME, "resources")
ICON_DIR = os.path.join(DIRNAME, "ui", "iconpack")

PLOT_POLICY = os.path.join(DIRNAME, "resources", "plotpolicy.mplstyle")

PEAK_MODELS = ["Lorentzian", "Gaussian", "PseudoVoigt", "GaussianAsym",
               "LorentzianAsym"]
WAFER_SIZE = ['100', '150', '200', '300']

FIT_PARAMS = {'method': 'leastsq', 'fit_negative': False, 'fit_outliers': False,
              'max_ite': 200, 'coef_noise': 1, 'xtol': 1.e-4, 'ncpus': 'auto'}
FIT_METHODS = {'Leastsq': 'leastsq', 'Least_squares': 'least_squares',
               'Nelder-Mead': 'nelder', 'SLSQP': 'slsqp'}
NCPUS = ['auto', '1', '2', '3', '4', '6', '8', '10', '12', '14', '16', '20',
         '24', '28', '32']
PALETTE = ['jet', 'viridis', 'plasma', 'inferno', 'magma',
           'cividis', 'cool', 'hot', 'YlGnBu', 'YlOrRd']
PLOT_STYLES = ['point', 'scatter', 'box', 'bar', 'line', 'trendline', 'wafer']
DEFAULT_COLORS = ['#0000ff', '#ff0000', '#008200', '#ffa000', '#8d0085', '#00ffff', '#ff00ff',
                  '#00ff00', '#ffbdcb', '#b41722', '#ffd500', '#008281', '#000086', '#c0c0c0',
                  '#808000', '#8d0000', '#6fd0ef']
MARKERS = ['o', 's', 'D', '^', '*', 'x', '+', 'v', '<', '>']
DEFAULT_MARKERS = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
LEGEND_LOCATION = ['upper right', 'upper left', 'lower left', 'lower right',
                   'center left', 'center right', 'lower center',
                   'upper center', 'center']
X_AXIS = ['Raman shift (cm-1)', 'Wavelength (nm)', 'Emission energy (eV)']


def rgb_to_hex(rgb):
    """Convert an RGB tuple to a hex color string."""
    # Check if the values are floats and convert them to integers in the range 0-255
    rgb = tuple(int(c * 255) if isinstance(c, float) else int(c) for c in rgb)
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def compress(array):
    """Compress and encode a numpy array to a base64 string."""
    compressed = zlib.compress(array.tobytes())
    encoded = base64.b64encode(compressed).decode('utf-8')
    return encoded

def decompress(data, dtype):
    """Decode and decompress a base64 string to a numpy array."""
    decoded = base64.b64decode(data.encode('utf-8'))
    decompressed = zlib.decompress(decoded)
    return np.frombuffer(decompressed, dtype=dtype)

def spectrum_to_dict(spectrum):
    """Custom "save" method to save Spectrum object in a dictionary"""
    excluded_keys = ['outliers_limit', 'peak_models', 'peak_index', 'bkg_model', 'result_fit', 'baseline', 'attractors']
    model_dict = {}
    for key, val in vars(spectrum).items():
        if key not in excluded_keys:
            if isinstance(val, np.ndarray):  # 'x0', 'y0', 'x', 'y'
                model_dict[key] = compress(val)
            else:
                model_dict[key] = val

    model_dict['baseline'] = dict(vars(spectrum.baseline).items())
    bkg_model = spectrum.bkg_model
    if bkg_model is not None:
        model_dict['bkg_model'] = {bkg_model.name2: bkg_model.param_hints}

    peak_models = {}
    for i, peak_model in enumerate(spectrum.peak_models):
        model_name = spectrum.get_model_name(peak_model)
        peak_models[i] = {}
        peak_models[i][model_name] = peak_model.param_hints
    model_dict['peak_models'] = peak_models

    if hasattr(spectrum.result_fit, 'success'):
        model_dict['result_fit_success'] = spectrum.result_fit.success

    return model_dict
def set_attributes(spectrum, model_dict):
    """To set attributes of 'Spectrum' object from JSON dict"""
    spectrum.set_attributes(model_dict)
    if 'x0' in model_dict:
        spectrum.x0 = decompress(model_dict['x0'], dtype=np.float64)
    if 'y0' in model_dict:
        spectrum.y0 = decompress(model_dict['y0'], dtype=np.float64)
    if 'x' in model_dict:
        spectrum.x = decompress(model_dict['x'], dtype=np.float64)
    if 'y' in model_dict:
        spectrum.y = decompress(model_dict['y'], dtype=np.float64)

def show_alert(message):
    """Show alert"""
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setWindowTitle("Alert")
    msg_box.setText(message)
    msg_box.exec_()

def clear_layout(layout):
    """To clear a given layout"""
    if layout is not None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()


def view_df(tabWidget, df):
    """View selected dataframe"""
    df_viewer = QDialog(tabWidget.parent())
    df_viewer.setWindowTitle("DataFrame Viewer")
    df_viewer.setWindowFlags(df_viewer.windowFlags() & ~Qt.WindowStaysOnTopHint)
    # Create a QTableWidget and populate it with data from the DataFrame
    table_widget = QTableWidget(df_viewer)
    table_widget.setColumnCount(df.shape[1])
    table_widget.setRowCount(df.shape[0])
    table_widget.setHorizontalHeaderLabels(df.columns)
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            item = QTableWidgetItem(str(df.iat[row, col]))
            table_widget.setItem(row, col, item)
    table_widget.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
    layout = QVBoxLayout(df_viewer)
    layout.addWidget(table_widget)
    df_viewer.show()

class DataframeTable(QWidget):
    """Class to display a given dataframe in GUI via QTableWidget with copy/paste functionality"""

    def __init__(self, df, layout):
        super().__init__()
        self.df = df
        self.external_layout = layout
        self.initUI()

    def initUI(self):
        """Initializes the user interface by creating and configuring the
        QTableWidget"""
        # Clear existing widgets from external layout
        while self.external_layout.count():
            item = self.external_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Set the internal layout
        layout = QVBoxLayout(self)
        self.setLayout(layout)

        # Create QTableWidget
        self.table_widget = QTableWidget()
        layout.addWidget(self.table_widget)

        # Display the DataFrame in the QTableWidget
        self.display_df_in_table()

        # Enable copy action via context menu
        self.table_widget.setContextMenuPolicy(Qt.ActionsContextMenu)
        copy_action = QAction("Copy", self)
        copy_action.triggered.connect(self.copy_data)
        self.table_widget.addAction(copy_action)

        # Add this widget to the external layout
        self.external_layout.addWidget(self)

    def display_df_in_table(self):
        """Populates the QTableWidget with data from the DataFrame"""
        self.table_widget.setRowCount(self.df.shape[0])
        self.table_widget.setColumnCount(self.df.shape[1])
        self.table_widget.setHorizontalHeaderLabels(self.df.columns)
        for row in range(self.df.shape[0]):
            for col in range(self.df.shape[1]):
                item = QTableWidgetItem(str(self.df.iat[row, col]))
                self.table_widget.setItem(row, col, item)
        self.table_widget.resizeColumnsToContents()

    def copy_data(self):
        """Copies selected data from the QTableWidget to the clipboard"""
        selected_indexes = self.table_widget.selectedIndexes()
        if not selected_indexes:
            return

        # Collect unique rows and columns
        rows = set(index.row() for index in selected_indexes)
        cols = set(index.column() for index in selected_indexes)

        data = []
        for row in sorted(rows):
            row_data = []
            for col in sorted(cols):
                item = self.table_widget.item(row, col)
                if item is not None:
                    row_data.append(item.text())
                else:
                    row_data.append('')
            data.append('\t'.join(row_data))

        # Join all rows with newline character and copy to clipboard
        clipboard = QApplication.clipboard()
        clipboard.setText('\n'.join(data))
        print(
            f"Copied data to clipboard:\n{clipboard.text()}")
    def keyPressEvent(self, event):
        """Handles key press events to enable copying with Ctrl+C"""
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_C:
            self.copy_data()
        else:
            super().keyPressEvent(event)


class Graph(QWidget):
    """Class to create and handle plot/graph objects."""

    def __init__(self, graph_id=None):
        super().__init__()
        self.df_name = None
        self.filters = {}
        self.graph_id = graph_id
        self.plot_width = 600
        self.plot_height = 500
        self.plot_style = "point"
        self.x = None
        self.y = []
        self.z = None
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.zmin = None
        self.zmax = None

        self.plot_title = None
        self.xlabel = None
        self.ylabel = None
        self.zlabel = None

        self.y2 = None
        self.y3 = None
        self.y2min = None
        self.y2max = None
        self.y3min = None
        self.y3max = None
        self.y2label = None
        self.y3label = None

        self.x_rot = 0
        self.grid = False
        self.legend_visible = True
        self.legend_location = 'upper right'
        self.legend_outside = False
        self.legend_properties = []

        self.color_palette = "jet"
        self.dpi = 100
        self.wafer_size = 300
        self.wafer_stats = True
        self.trendline_order = 1
        self.show_trendline_eq = True
        self.show_bar_plot_error_bar = True
        self.join_for_point_plot = False

        self.figure = None
        self.ax = None
        self.ax2 = None
        self.ax3 = None
        self.canvas = None
        self.graph_layout = QVBoxLayout()
        self.setLayout(self.graph_layout)
    def clear_layout(self, layout):
        """Clears all widgets and layouts from the specified layout."""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())
    def create_plot_widget(self, dpi, layout=None):
        """Creates a new plot canvas with the specified DPI and adds it to the layout."""
        if dpi:
            self.dpi = dpi
        else:
            self.dpi = 100

        self.clear_layout(self.graph_layout)
        plt.close('all')

        self.figure = plt.figure(dpi=self.dpi)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        for action in self.toolbar.actions():
            if action.text() in ['Save', 'Subplots']:
                action.setVisible(False)

        if layout:
            layout.addWidget(self.canvas)
            layout.addWidget(self.toolbar)
        else:
            self.graph_layout.addWidget(self.canvas)
            self.graph_layout.addWidget(self.toolbar)

        self.canvas.figure.tight_layout()
        self.canvas.draw()

        # Set default labels if not already set
        self.xlabel = self.xlabel if self.xlabel is not None else self.x
        self.ylabel = self.ylabel if self.ylabel is not None else self.y[0] if self.y else None
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.plot_title)

    def plot(self, df):
        """Updates the plot based on the provided DataFrame and plot settings."""
        # Clear the axes
        self.ax.clear()
        if self.ax2:
            self.ax2.clear()
        if self.ax3:
            self.ax3.clear()

        # Perform the plotting
        if self.df_name is not None and self.x is not None and self.y is not None:
            self._plot_primary_axis(df)
            self._plot_secondary_axis(df)
            self._plot_tertiary_axis(df)
        else:
            self.ax.plot([], [])

        self._set_limits()
        self._set_labels()
        self._set_grid()
        self._set_rotation()
        self._set_legend()

        self.get_legend_properties()
        self.ax.get_figure().tight_layout()
        self.canvas.draw()

        self.canvas.mpl_connect('draw_event', self.update_graph_properties)
    def update_graph_properties(self, event=None):
        """Updates the graph attributes based on the current plot state."""
        if self.ax:
            self.xmin, self.xmax = self.ax.get_xlim()
            self.ymin, self.ymax = self.ax.get_ylim()
            self.xlabel = self.ax.get_xlabel()
            self.ylabel = self.ax.get_ylabel()
            self.plot_title = self.ax.get_title()

        if self.ax2:
            self.y2min, self.y2max = self.ax2.get_ylim()
            self.y2label = self.ax2.get_ylabel()

        if self.ax3:
            self.y3min, self.y3max = self.ax3.get_ylim()
            self.y3label = self.ax3.get_ylabel()

    def _set_limits(self):
        """Set the limits of the axis."""
        if self.xmin and self.xmax:
            self.ax.set_xlim(float(self.xmin), float(self.xmax))
        if self.ymin and self.ymax:
            self.ax.set_ylim(float(self.ymin), float(self.ymax))
        if self.ax2 and self.y2min and self.y2max:
            self.ax2.set_ylim(float(self.y2min), float(self.y2max))
        if self.ax3 and self.y3min and self.y3max:
            self.ax3.set_ylim(float(self.y3min), float(self.y3max))

    def get_legend_properties(self):
        """Retrieves properties of each existing legend item."""
        legend_properties = []
        if self.ax:
            legend = self.ax.get_legend()
            if legend:
                legend_texts = legend.get_texts()
                legend_handles = legend.legendHandles
                for idx, text in enumerate(legend_texts):

                    label = text.get_text()
                    handle = legend_handles[idx]
                    if self.plot_style in ['point', 'scatter', 'line']:
                        color = handle.get_markerfacecolor()
                        marker = handle.get_marker()
                    # Box & bar plots do not use markers → set defautl values
                    elif self.plot_style in ['box', 'bar']:
                        color = handle.get_facecolor()
                        marker = 'o'
                    else:
                        color = 'blue'
                        marker = 'o'
                    legend_properties.append(
                        {'label': label, 'marker': marker, 'color': color})
        self.legend_properties = legend_properties
        return self.legend_properties



    def customize_legend_via_gui(self, main_layout):
        """Displays legend properties in the GUI for user modifications."""
        self.clear_layout(main_layout)
        headers = ['Label', 'Marker', 'Color']
        # Create vertical layouts for each property type
        label_layout = QVBoxLayout()
        marker_layout = QVBoxLayout()
        color_layout = QVBoxLayout()
        for header in headers:
            label = QLabel(header)
            label.setAlignment(Qt.AlignCenter)
            if header == "Label":
                label_layout.addWidget(label)
            elif header == "Marker":
                if self.plot_style == 'point':
                    marker_layout.addWidget(label)
                else:
                    pass
            elif header == "Color":
                color_layout.addWidget(label)

        for idx, prop in enumerate(self.legend_properties):
            # LABEL
            label = QLineEdit(prop['label'])
            label.setFixedWidth(200)
            label.textChanged.connect(
                lambda text, idx=idx: self.udp_legend(idx, 'label', text))
            label_layout.addWidget(label)

            if self.plot_style == 'point':
                # MARKER
                marker = QComboBox()
                marker.addItems(MARKERS)  # Add more markers as needed
                marker.setCurrentText(prop['marker'])
                marker.currentTextChanged.connect(
                    lambda text, idx=idx: self.udp_legend(idx, 'marker', text))
                marker_layout.addWidget(marker)
            else:
                pass

            # COLOR
            color = QComboBox()
            delegate = ColorDelegate(color)
            color.setItemDelegate(delegate)
            for color_code in DEFAULT_COLORS:
                item = color.addItem(color_code)
                item = color.model().item(color.count() - 1)
                item.setBackground(QColor(color_code))

            # Convert RGB list to hex if necessary
            if isinstance(prop['color'], list):
                prop_color = rgb_to_hex(prop['color'])
            else:
                prop_color = prop['color']
            color.setCurrentText(prop_color)
            color.currentIndexChanged.connect(lambda idx, color=color: self.update_combobox_color(color))
            color.currentTextChanged.connect(
                lambda text, idx=idx: self.udp_legend(idx, 'color', text))
            color_layout.addWidget(color)

            # Ensure the color is updated on load
            self.update_combobox_color(color)

        # Add vertical layouts to main layout
        main_layout.addLayout(label_layout)
        main_layout.addLayout(marker_layout)
        main_layout.addLayout(color_layout)

    def update_combobox_color(self, combobox):
        """Update combobox background color based on the selected color."""
        selected_color = combobox.currentText()
        color = QColor(selected_color)
        palette = combobox.palette()
        palette.setColor(QPalette.Button, color)
        palette.setColor(QPalette.ButtonText, Qt.white)
        combobox.setAutoFillBackground(True)
        combobox.setPalette(palette)
        combobox.update()
    def udp_legend(self, idx, property_type, text):
        """Updates legend properties based on user modifications via GUI."""
        self.legend_properties[idx][property_type] = text
        self._set_legend()

    def _plot_primary_axis(self, df):
        """Plots data on the primary axis based on the current plot style."""
        if not self.legend_properties:
            markers = DEFAULT_MARKERS
            colors = DEFAULT_COLORS
        else:
            markers = [str(prop['marker']) for prop in self.legend_properties]
            colors = [str(prop['color']) for prop in self.legend_properties]
        for y in self.y:
            if self.plot_style == 'point':
                sns.pointplot(data=df, x=self.x, y=y, hue=self.z,
                              ax=self.ax,
                              linestyles='-' if self.join_for_point_plot
                              else 'none', marker=markers, palette=colors,
                              markeredgecolor='black', markeredgewidth=1,
                              dodge=True,
                              err_kws={'linewidth': 1, 'color': 'black'},
                              capsize=0.02)
            elif self.plot_style == 'scatter':
                sns.scatterplot(data=df, x=self.x, y=y, hue=self.z, ax=self.ax,
                                s=100, edgecolor='black', palette=colors)
            elif self.plot_style == 'box':
                sns.boxplot(data=df, x=self.x, y=y, hue=self.z, dodge=True,
                            ax=self.ax, palette=colors)
            elif self.plot_style == 'line':
                sns.lineplot(data=df, x=self.x, y=y, hue=self.z, ax=self.ax,
                             palette=colors)
            elif self.plot_style == 'bar':
                sns.barplot(data=df, x=self.x, y=y, hue=self.z,
                            errorbar='sd' if self.show_bar_plot_error_bar
                            else None, err_kws={'linewidth': 1},
                            ax=self.ax, palette=colors)

            elif self.plot_style == 'trendline':
                sns.regplot(data=df, x=self.x, y=y, ax=self.ax, scatter=True,
                            order=self.trendline_order)
                if self.show_trendline_eq:
                    self._annotate_trendline_eq(df)
            elif self.plot_style == 'wafer':
                self._plot_wafer(df)
            else:
                show_alert("Unsupported plot style")

    def _set_legend(self):
        """Sets up and displays the legend for the plot."""
        handles, labels = self.ax.get_legend_handles_labels()
        if self.ax2:
            handles2, labels2 = self.ax2.get_legend_handles_labels()
            handles += handles2
            labels += labels2
            self.ax2.legend().remove()  # Turn off legend for ax2
        if self.ax3:
            handles3, labels3 = self.ax3.get_legend_handles_labels()
            handles += handles3
            labels += labels3
            self.ax3.legend().remove()  # Turn off legend for ax3
        if handles:
            legend_labels = []
            if self.legend_properties:
                try:
                    for idx, prop in enumerate(self.legend_properties):
                        legend_labels.append(prop['label'])
                        handles[idx].set_label(
                            prop['label'])  # Set legend label
                        handles[idx].set_color(prop['color'])  # Set color
                        if self.plot_style in ['point', 'scatter']:
                            handles[idx].set_marker(
                                prop['marker'])  # Set marker
                        else:
                            pass
                except Exception as e:
                    self.legend_properties = []
                    legend_labels = labels
                    self.legend_properties = self.get_legend_properties()

            else:
                legend_labels = labels
                self.legend_properties = self.get_legend_properties()

            if self.legend_visible:
                self.ax.legend(handles, legend_labels, loc=self.legend_location)
            else:
                self.ax.legend().remove()
            if self.legend_outside:
                self.ax.legend(handles, legend_labels, loc='center left',
                               bbox_to_anchor=(1, 0.5))

    def _set_grid(self):
        """Add grid for the plot"""
        if self.grid:
            self.ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
        else:
            self.ax.grid(False)

    def _set_rotation(self):
        """Set rotation of the ticklabels of the x axis"""
        if self.x_rot != 0:
            plt.setp(self.ax.get_xticklabels(), rotation=self.x_rot, ha="right",
                     rotation_mode="anchor")

    def _annotate_trendline_eq(self, df):
        """Add the trendline equation in the plot"""
        x_data = df[self.x]
        y_data = df[self.y[0]]
        coefficients = np.polyfit(x_data, y_data, self.trendline_order)
        equation = 'y = '
        for i, coeff in enumerate(coefficients[::-1]):
            equation += (f'{coeff:.2f}x^{self.trendline_order - i} + '
                         if i < self.trendline_order else f'{coeff:.2f}')
        self.ax.annotate(equation, xy=(0.02, 0.95), xycoords='axes fraction',
                         fontsize=10, color='blue')

    def _plot_wafer(self, df):
        """PLot wafer plot by creating an object of WaferPlot Class"""
        vmin = self.zmin if self.zmin else None
        vmax = self.zmax if self.zmax else None
        wdf = WaferPlot()
        wdf.plot(self.ax, x=df[self.x], y=df[self.y[0]], z=df[self.z],
                 cmap=self.color_palette,
                 vmin=vmin, vmax=vmax, stats=self.wafer_stats,
                 r=(self.wafer_size / 2))


    def _set_labels(self):
        """Set titles and labels for axis and plot"""
        if self.plot_style == 'wafer':
            if self.plot_title:
                self.ax.set_title(self.plot_title)
            else:
                self.ax.set_title(self.z)
        else:
            self.ax.set_title(self.plot_title)
            if self.xlabel:
                self.ax.set_xlabel(self.xlabel)
            else:
                self.xlabel=self.x
                self.ax.set_xlabel(self.xlabel)
            if self.ylabel:
                self.ax.set_ylabel(self.ylabel)
            else:
                self.ylabel = self.y
                self.ax.set_ylabel(self.ylabel)

    def _plot_secondary_axis(self, df):
        if self.ax2:
            self.ax2.remove()
            self.ax2 = None
        if hasattr(self, 'y2') and self.y2:
            self.ax2 = self.ax.twinx()
            if self.plot_style == 'line':
                sns.lineplot(data=df, x=self.x, y=self.y2, hue=self.z,
                             ax=self.ax2, color='red')
            elif self.plot_style == 'point':
                sns.pointplot(
                    data=df, x=self.x, y=self.y2, hue=self.z, ax=self.ax2,
                    linestyles='-' if self.join_for_point_plot else 'none',
                    marker='s', color='red', markeredgecolor='black',
                    markeredgewidth=1,
                    dodge=True, err_kws={'linewidth': 1, 'color': 'black'},
                    capsize=0.02
                )
            elif self.plot_style == 'scatter':
                sns.scatterplot(data=df, x=self.x, y=self.y2, hue=self.z,
                                ax=self.ax2,
                                s=100, edgecolor='black', color='red')
            else:
                self.ax2.remove()
                self.ax2 = None

            self.ax2.set_ylabel(self.y2label, color='red')
            self.ax2.tick_params(axis='y', colors='red')

    def _plot_tertiary_axis(self, df):
        if self.ax3:
            self.ax3.remove()
            self.ax3 = None
        if hasattr(self, 'y3') and self.y3:
            self.ax3 = self.ax.twinx()
            self.ax3.spines["right"].set_position(("outward", 100))
            if self.plot_style == 'line':
                sns.lineplot(data=df, x=self.x, y=self.y3, hue=self.z,
                             ax=self.ax3, color='green')
            elif self.plot_style == 'point':
                sns.pointplot(
                    data=df, x=self.x, y=self.y3, hue=self.z, ax=self.ax3,
                    linestyles='-' if self.join_for_point_plot else 'none',
                    marker='s', color='red', markeredgecolor='black',
                    markeredgewidth=1,
                    dodge=True, err_kws={'linewidth': 1, 'color': 'black'},
                    capsize=0.02
                )
            elif self.plot_style == 'scatter':
                sns.scatterplot(data=df, x=self.x, y=self.y3, hue=self.z,
                                ax=self.ax3,
                                s=100, edgecolor='black', color='green')
            else:
                self.ax3.remove()
                self.ax3 = None
            self.ax3.set_ylabel(self.y3label, color='green')
            self.ax3.tick_params(axis='y', colors='green')

    def save(self, fname=None):
        """ Save Graph object to serialization. Save it if a fname is given """
        # List of keys to exclude from serialization
        excluded_keys = ['figure', 'canvas', 'setLayout', 'graph_layout', 'some_signal_instance']

        dict_graph = {}
        for key, val in vars(self).items():
            if key not in excluded_keys and not callable(val):
                try:
                    json.dumps(val)
                    dict_graph[key] = val
                except TypeError:
                   continue

        if fname is not None:
            with open(fname, 'w') as f:
                json.dump(dict_graph, f, indent=4)

        return dict_graph

    def set_attributes(self, attributes_dict):
        """Set attributes of the Graph object from a given dictionary."""
        for key, value in attributes_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
class ShowParameters:
    """Class dedicated to show fit paramters of Spectrum objects in the GUI"""

    def __init__(self, main_layout, sel_spectrum, cb_limits, cb_expr, update):
        self.main_layout = main_layout
        self.sel_spectrum = sel_spectrum
        self.cb_limits = cb_limits
        self.cb_expr = cb_expr
        self.update = update

    def clear_layout(self, layout):
        """To clear a given layout"""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())

    def show_peak_table(self, main_layout, sel_spectrum, cb_limits, cb_expr):
        """ To show all fitted parameters in GUI"""
        self.clear_layout(main_layout)

        header_labels = ["  ", "Label", "Model"]
        param_hint_order = ['x0', 'fwhm', 'ampli', 'alpha', 'fwhm_l', 'fwhm_r']

        # Create and add headers to list
        for param_hint_key in param_hint_order:
            if any(param_hint_key in peak_model.param_hints for peak_model in
                   sel_spectrum.peak_models):
                header_labels.append(param_hint_key.title())
                header_labels.append(f"fix {param_hint_key.title()}")
                if cb_limits.isChecked():
                    header_labels.append(f"min {param_hint_key.title()}")
                    header_labels.append(f"max {param_hint_key.title()}")
                if cb_expr.isChecked():
                    header_labels.append(f"expression {param_hint_key.title()}")

        # Create vertical layouts for each column type
        delete_layout = QVBoxLayout()
        label_layout = QVBoxLayout()
        model_layout = QVBoxLayout()
        param_hint_layouts = {param_hint: {var: QVBoxLayout() for var in
                                           ['value', 'min', 'max', 'expr',
                                            'vary']} for
                              param_hint in param_hint_order}

        # Add header labels to each layout
        for header_label in header_labels:
            label = QLabel(header_label)
            label.setAlignment(Qt.AlignCenter)
            if header_label == "  ":
                delete_layout.addWidget(label)
            elif header_label == "Label":
                label_layout.addWidget(label)
            elif header_label == "Model":
                model_layout.addWidget(label)
            elif header_label.startswith("fix"):
                param_hint_key = header_label.split()[1].lower()
                param_hint_layouts[param_hint_key]['vary'].addWidget(label)
            elif "min" in header_label:
                param_hint_key = header_label.split()[1].lower()
                param_hint_layouts[param_hint_key]['min'].addWidget(label)
            elif "max" in header_label:
                param_hint_key = header_label.split()[1].lower()
                param_hint_layouts[param_hint_key]['max'].addWidget(label)
            elif "expression" in header_label:
                param_hint_key = header_label.split()[1].lower()
                param_hint_layouts[param_hint_key]['expr'].addWidget(label)
            else:
                param_hint_key = header_label.lower()
                param_hint_layouts[param_hint_key]['value'].addWidget(label)

        for i, peak_model in enumerate(sel_spectrum.peak_models):
            # Button to delete peak_model
            delete = QPushButton(peak_model.prefix)
            icon = QIcon()
            icon.addFile(os.path.join(ICON_DIR, "close.png"))
            delete.setIcon(icon)
            delete.setFixedWidth(50)
            delete.clicked.connect(
                lambda idx=i, spectrum=sel_spectrum: self.delete_peak_model(
                    spectrum, idx))
            delete_layout.addWidget(delete)

            # Peak_label
            label = QLineEdit(sel_spectrum.peak_labels[i])
            label.setFixedWidth(80)
            label.textChanged.connect(
                lambda text, idx=i,
                       spectrum=sel_spectrum: self.update_peak_label(spectrum,
                                                                     idx, text))
            label_layout.addWidget(label)

            # Peak model : Lorentizan, Gaussien, etc...
            model = QComboBox()
            model.addItems(PEAK_MODELS)
            current_model_index = PEAK_MODELS.index(
                peak_model.name2) if peak_model.name2 in PEAK_MODELS else 0
            model.setCurrentIndex(current_model_index)
            model.setFixedWidth(120)
            model.currentIndexChanged.connect(
                lambda index, spectrum=sel_spectrum, idx=i,
                       combo=model: self.update_model_name(spectrum, index, idx,
                                                           combo.currentText()))
            model_layout.addWidget(model)

            # variables of peak_model
            param_hints = peak_model.param_hints
            for param_hint_key in param_hint_order:
                if param_hint_key in param_hints:
                    param_hint_value = param_hints[param_hint_key]

                    # 4.1 VALUE
                    value_val = round(param_hint_value.get('value', 0.0), 2)
                    value = QLineEdit(str(value_val))
                    value.setFixedWidth(70)
                    value.setFixedHeight(24)
                    value.setAlignment(Qt.AlignRight)
                    value.textChanged.connect(
                        lambda text, pm=peak_model,
                               key=param_hint_key: self.update_param_hint_value(
                            pm, key, text))
                    param_hint_layouts[param_hint_key]['value'].addWidget(value)

                    # 4.2 FIXED or NOT
                    vary = QCheckBox()
                    vary.setChecked(not param_hint_value.get('vary', False))
                    vary.setFixedHeight(24)
                    vary.stateChanged.connect(
                        lambda state, pm=peak_model,
                               key=param_hint_key: self.update_param_hint_vary(
                            pm, key,
                            not state))
                    param_hint_layouts[param_hint_key]['vary'].addWidget(vary)

                    # 4.3 MIN MAX
                    if cb_limits.isChecked():
                        min_val = round(param_hint_value.get('min', 0.0), 2)
                        min_lineedit = QLineEdit(str(min_val))
                        min_lineedit.setFixedWidth(70)
                        min_lineedit.setFixedHeight(24)
                        min_lineedit.setAlignment(Qt.AlignRight)
                        min_lineedit.textChanged.connect(
                            lambda text, pm=peak_model,
                                   key=param_hint_key:
                            self.update_param_hint_min(
                                pm, key, text))
                        param_hint_layouts[param_hint_key]['min'].addWidget(
                            min_lineedit)

                        max_val = round(param_hint_value.get('max', 0.0), 2)
                        max_lineedit = QLineEdit(str(max_val))
                        max_lineedit.setFixedWidth(70)
                        max_lineedit.setFixedHeight(24)
                        max_lineedit.setAlignment(Qt.AlignRight)
                        max_lineedit.textChanged.connect(
                            lambda text, pm=peak_model,
                                   key=param_hint_key:
                            self.update_param_hint_max(
                                pm, key, text))
                        param_hint_layouts[param_hint_key]['max'].addWidget(
                            max_lineedit)

                    # 4.4 EXPRESSION
                    if cb_expr.isChecked():
                        expr_val = str(param_hint_value.get('expr', ''))
                        expr = QLineEdit(expr_val)
                        expr.setFixedWidth(150)
                        expr.setFixedHeight(
                            24)  # Set a fixed height for QLineEdit
                        expr.setAlignment(Qt.AlignRight)
                        expr.textChanged.connect(
                            lambda text, pm=peak_model,
                                   key=param_hint_key:
                            self.update_param_hint_expr(
                                pm, key, text))
                        param_hint_layouts[param_hint_key]['expr'].addWidget(
                            expr)
                else:
                    # Add empty labels for alignment
                    empty_label = QLabel()
                    empty_label.setFixedHeight(24)
                    param_hint_layouts[param_hint_key]['value'].addWidget(
                        empty_label)
                    param_hint_layouts[param_hint_key]['vary'].addWidget(
                        empty_label)
                    if cb_limits.isChecked():
                        param_hint_layouts[param_hint_key]['min'].addWidget(
                            empty_label)
                        param_hint_layouts[param_hint_key]['max'].addWidget(
                            empty_label)
                    if cb_expr.isChecked():
                        param_hint_layouts[param_hint_key]['expr'].addWidget(
                            empty_label)

        # Add vertical layouts to main layout
        main_layout.addLayout(delete_layout)
        main_layout.addLayout(label_layout)
        main_layout.addLayout(model_layout)

        for param_hint_key, param_hint_layout in param_hint_layouts.items():
            for var_layout in param_hint_layout.values():
                main_layout.addLayout(var_layout)

    def update_model_name(self, spectrum, index, idx, new_model):
        """ Update the model function (Lorentizan, Gaussian...) related to
        the ith-model """
        old_model_name = spectrum.peak_models[idx].name2
        new_model_name = new_model
        if new_model_name != old_model_name:
            ampli = spectrum.peak_models[idx].param_hints['ampli']['value']
            x0 = spectrum.peak_models[idx].param_hints['x0']['value']
            peak_model = spectrum.create_peak_model(idx + 1, new_model_name,
                                                    x0=x0, ampli=ampli)
            spectrum.peak_models[idx] = peak_model
            spectrum.result_fit = lambda: None
            self.update()

    def delete_peak_model(self, spectrum, idx):
        """"To delete a peak model"""
        del spectrum.peak_models[idx]
        del spectrum.peak_labels[idx]
        self.update()

    def update_peak_label(self, spectrum, idx, text):
        spectrum.peak_labels[idx] = text

    def update_param_hint_value(self, pm, key, text):
        pm.param_hints[key]['value'] = float(text)

    def update_param_hint_min(self, pm, key, text):
        pm.param_hints[key]['min'] = float(text)

    def update_param_hint_max(self, pm, key, text):
        pm.param_hints[key]['max'] = float(text)

    def update_param_hint_vary(self, pm, key, state):
        pm.param_hints[key]['vary'] = state

    def update_param_hint_expr(self, pm, key, text):
        pm.param_hints[key]['expr'] = text


class Filter:
    """
    Class for Handling "Filter Features" in Querying Pandas DataFrames

    Attributes:
    line_edit (QLineEdit): Input field for filter expressions.
    listbox (QListWidget): List widget to display filter expressions as
    checkboxes.
    df (pandas.DataFrame): DataFrame to be filtered.
    filters (list): List to store filter expressions and their states.
    """

    def __init__(self, line_edit, listbox, df):
        self.line_edit = line_edit
        self.listbox = listbox
        self.df = df
        self.filters = []

    def set_dataframe(self, df):
        """Set the dataframe to be filtered"""
        self.df = df

    def add_filter(self):
        """Add a filter expression to the filters list and update the UI"""
        filter_expression = self.line_edit.text().strip()
        if filter_expression:
            filter = {"expression": filter_expression, "state": False}
            self.filters.append(filter)
        # Add the filter expression to QListWidget as a checkbox item
        item = QListWidgetItem()
        checkbox = QCheckBox(filter_expression)
        item.setSizeHint(checkbox.sizeHint())
        self.listbox.addItem(item)
        self.listbox.setItemWidget(item, checkbox)

    def remove_filter(self):
        """Remove selected filter(s) from the filters list and UI"""
        selected_items = [item for item in
                          self.listbox.selectedItems()]
        for item in selected_items:
            checkbox = self.listbox.itemWidget(item)
            filter_expression = checkbox.text()
            for filter in self.filters[:]:
                if filter.get("expression") == filter_expression:
                    self.filters.remove(filter)
            self.listbox.takeItem(self.listbox.row(item))

    def get_current_filters(self):
        """
        Retrieve the current state of filters as displayed in the UI.

        Returns:
        list: List of dictionaries representing filter expressions and their
        states.
        Each dictionary has keys 'expression' and 'state'.
        """
        checked_filters = []
        for i in range(self.listbox.count()):
            item = self.listbox.item(i)
            checkbox = self.listbox.itemWidget(item)
            expression = checkbox.text()
            state = checkbox.isChecked()
            checked_filters.append({"expression": expression, "state": state})
        return checked_filters

    def apply_filters(self, filters=None):
        """
        Apply filters to the DataFrame (self.df) based on the current or
        provided filters.
        Args:
        filters (list, optional): List of dictionaries representing filter
        expressions and their states.Defaults to None, meaning current UI filters are used.
        Returns:
        pandas.DataFrame or None: Filtered DataFrame based on applied filters
        or None if self.df is None.
        """
        if filters:
            self.filters = filters
        else:
            checked_filters = self.get_current_filters()
            self.filters = checked_filters
        # Apply all filters at once
        # Check if self.df is not None
        self.filtered_df = self.df.copy() if self.df is not None else None

        if self.filtered_df is not None:  # Check if filtered_df is not None
            for filter_data in self.filters:
                filter_expr = filter_data["expression"]
                is_checked = filter_data["state"]
                if is_checked:
                    try:
                        filter_expr = str(filter_expr)
                        print(f"Applying filter expression: {filter_expr}")
                        # Apply the filter
                        self.filtered_df = self.filtered_df.query(filter_expr)
                    except Exception as e:
                        # show_alert(f"Filter error: {str(e)}")
                        print(f"Error applying filter: {str(e)}")

        return self.filtered_df

    def upd_filter_listbox(self):
        """Update the listbox UI to reflect changes in filters"""
        self.listbox.clear()
        for filter_data in self.filters:
            filter_expression = filter_data["expression"]
            item = QListWidgetItem()
            checkbox = QCheckBox(filter_expression)
            item.setSizeHint(checkbox.sizeHint())
            self.listbox.addItem(item)
            self.listbox.setItemWidget(item, checkbox)
            checkbox.setChecked(filter_data["state"])


class FitModelManager:
    """Class to manage the folter containing fit models created by user"""

    def __init__(self, settings):
        self.settings = settings
        self.default_model_folder = self.settings.value("default_model_folder",
                                                        "")
        self.available_models = []
        if self.default_model_folder:
            self.scan_models()

    def set_default_model_folder(self, folder_path):
        """Set the default folder storing fit models"""
        self.default_model_folder = folder_path
        self.settings.setValue("default_model_folder", folder_path)
        self.scan_models()

    def scan_models(self):
        """Scan the default folder and populate available models to list"""
        self.available_models = []
        if self.default_model_folder:
            for file_name in os.listdir(self.default_model_folder):
                if file_name.endswith('.json'):
                    self.available_models.append(file_name)

    def get_available_models(self):
        """Retrieve the list of available fit model filenames"""
        return self.available_models


class CommonUtilities():
    """ Class contain all common methods or utility codes used other modules"""

    def plot_graph(self, ax, dfr, x, y, z, style, xmin, xmax, ymin, ymax, title,
                   x_text,
                   y_text, xlabel_rot):
        """Plot graph """

        ax.clear()
        if style == "box plot":
            sns.boxplot(data=dfr, x=x, y=y, hue=z, dodge=True, ax=ax)
        elif style == "point plot":
            sns.pointplot(data=dfr, x=x, y=y, hue=z, linestyle='none',
                          dodge=True, capsize=0.00, ax=ax)
        elif style == "scatter plot":
            sns.scatterplot(data=dfr, x=x, y=y, hue=z, s=100, ax=ax)
        elif style == "bar plot":
            sns.barplot(data=dfr, x=x, y=y, hue=z, errorbar='sd', ax=ax)

        if xmin and xmax:
            ax.set_xlim(float(xmin), float(xmax))
        if ymin and ymax:
            ax.set_ylim(float(ymin), float(ymax))

        xlabel = x if not x_text else x_text
        ylabel = y if not y_text else y_text
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.legend(loc='upper right')

        if xlabel_rot != 0:
            plt.setp(ax.get_xticklabels(), rotation=xlabel_rot, ha="right",
                     rotation_mode="anchor")
        return ax

    def reinit_spectrum(self, fnames, spectrums):
        """Reinitilize a FITSPY spectrum object"""
        for fname in fnames:
            spectrum, _ = spectrums.get_objects(fname)
            spectrum.range_min = None
            spectrum.range_max = None
            spectrum.x = spectrum.x0.copy()
            spectrum.y = spectrum.y0.copy()
            spectrum.norm_mode = None
            spectrum.result_fit = lambda: None
            spectrum.remove_models()
            spectrum.baseline.points = [[], []]
            spectrum.baseline.is_subtracted = False

    def clear_layout(self, layout):
        """Clear everything in a given Qlayout"""
        if layout is not None:
            for i in reversed(range(layout.count())):
                item = layout.itemAt(i)
                if isinstance(item.widget(),
                              (FigureCanvas, NavigationToolbar2QT)):
                    widget = item.widget()
                    layout.removeWidget(widget)
                    widget.close()

    def translate_param(self, fit_model, param):
        """Translate parameter names to column headers:e.g. 'x0' to
        'Position', 'ampli' to  'Intensity'..."""

        peak_labels = fit_model["peak_labels"]
        param_unit_mapping = {"ampli": "Intensity", "fwhm": "FWHM",
                              "fwhm_l": "FWHM_left", "fwhm_r": "FWHM_right",
                              "alpha": "L/G ratio",
                              "x0": "Position"}
        if "_" in param:
            prefix, param = param.split("_", 1)
            if param in param_unit_mapping:
                if param == "alpha":
                    unit = ""  # Set unit to empty string for "alpha"
                else:
                    unit = "(a.u)" if param == "ampli" else "(cm⁻¹)"
                label = param_unit_mapping[param]
                # Convert prefix to peak_label
                peak_index = int(prefix[1:]) - 1
                if 0 <= peak_index < len(peak_labels):
                    peak_label = peak_labels[peak_index]
                    return f"{label} of peak {peak_label} {unit}"
        return param

    def quadrant(self, row):
        """Define 4 quadrant of a wafer"""
        if row['X'] < 0 and row['Y'] < 0:
            return 'Q1'
        elif row['X'] < 0 and row['Y'] > 0:
            return 'Q2'
        elif row['X'] > 0 and row['Y'] > 0:
            return 'Q3'
        elif row['X'] > 0 and row['Y'] < 0:
            return 'Q4'
        else:
            return np.nan

    def zone(self, row, diameter):
        """Define 3 zones (Center, Mid-Rayon, Edge) based on X and Y
        coordinates."""
        rad = diameter / 2
        x = row['X']
        y = row['Y']
        distance_to_center = np.sqrt(x ** 2 + y ** 2)
        if distance_to_center <= rad * 0.35:
            return 'Center'
        elif distance_to_center > rad * 0.35 and distance_to_center < rad * 0.8:
            return 'Mid-Radius'
        elif distance_to_center >= 0.8 * rad:
            return 'Edge'
        else:
            return np.nan

    def copy_fig_to_clb(self, canvas):
        """Function to copy canvas figure to clipboard"""
        if canvas:
            figure = canvas.figure
            with BytesIO() as buf:
                figure.savefig(buf, format='png', dpi=400)
                data = buf.getvalue()
            format_id = win32clipboard.RegisterClipboardFormat('PNG')
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(format_id, data)
            win32clipboard.CloseClipboard()
        else:
            QMessageBox.critical(None, "Error", "No plot to copy.")

    def display_df_in_table(self, table_widget, df_results):
        """Display pandas DataFrame in QTableWidget in GUI"""
        table_widget.setRowCount(df_results.shape[0])
        table_widget.setColumnCount(df_results.shape[1])
        table_widget.setHorizontalHeaderLabels(df_results.columns)
        for row in range(df_results.shape[0]):
            for col in range(df_results.shape[1]):
                item = QTableWidgetItem(str(df_results.iat[row, col]))
                table_widget.setItem(row, col, item)
        table_widget.resizeColumnsToContents()

    def view_text(self, ui, title, text):
        """ Create a QTextBrowser to display a text content"""
        report_viewer = QDialog(ui)
        report_viewer.setWindowTitle(title)
        report_viewer.setGeometry(100, 100, 800, 600)
        text_browser = QTextBrowser(report_viewer)
        text_browser.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        text_browser.setOpenExternalLinks(True)
        text_browser.setPlainText(text)
        text_browser.moveCursor(QTextCursor.Start)
        layout = QVBoxLayout(report_viewer)
        layout.addWidget(text_browser)
        report_viewer.show()

    def view_markdown(self, ui, title, fname, x, y):
        """To convert MD file to html format and display them in GUI"""
        with open(fname, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        html_content = markdown.markdown(markdown_content)
        DIRNAME = os.path.dirname(__file__)
        html_content = html_content.replace('src="',
                                            f'src="'
                                            f'{os.path.join(DIRNAME, "resources/")}')
        about_dialog = QDialog(ui)
        about_dialog.setWindowTitle(title)
        about_dialog.resize(x, y)
        text_browser = QTextBrowser(about_dialog)
        text_browser.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        text_browser.setOpenExternalLinks(True)
        text_browser.setHtml(html_content)
        layout = QVBoxLayout(about_dialog)
        layout.addWidget(text_browser)
        about_dialog.setLayout(layout)
        about_dialog.show()

    def dark_palette(self):
        """Palette color for dark mode of the appli's GUI"""
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(70, 70, 70))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base,
                              QColor(65, 65, 65))  # QlineEdit Listbox bg
        dark_palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(64, 64, 64))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.white)
        dark_palette.setColor(QPalette.PlaceholderText, QColor(140, 140, 140))
        return dark_palette

    def light_palette(self):
        """Palette color for light mode of the appli's GUI"""
        light_palette = QPalette()
        light_palette.setColor(QPalette.Window, QColor(225, 225, 225))
        light_palette.setColor(QPalette.WindowText, Qt.black)
        light_palette.setColor(QPalette.Base, QColor(215, 215, 215))
        light_palette.setColor(QPalette.AlternateBase, QColor(230, 230, 230))
        light_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
        light_palette.setColor(QPalette.ToolTipText, Qt.black)
        light_palette.setColor(QPalette.Text, Qt.black)
        light_palette.setColor(QPalette.Button, QColor(230, 230, 230))
        light_palette.setColor(QPalette.ButtonText, Qt.black)
        light_palette.setColor(QPalette.BrightText, Qt.red)
        light_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        light_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        light_palette.setColor(QPalette.HighlightedText, Qt.black)
        light_palette.setColor(QPalette.PlaceholderText, QColor(150, 150, 150))
        return light_palette


class FitThread(QThread):
    """ Class to perform fitting in a separate Thread """
    progress_changed = Signal(int)

    def __init__(self, spectrums, fit_model, fnames, ncpus=1):
        super().__init__()
        self.spectrums = spectrums
        self.fit_model = fit_model
        self.fnames = fnames
        self.ncpus = ncpus

    def run(self):
        fit_model = deepcopy(self.fit_model)
        self.spectrums.apply_model(fit_model, fnames=self.fnames,
                                   ncpus=self.ncpus, show_progressbar=False)

        self.progress_changed.emit(100)
class ColorDelegate(QStyledItemDelegate):
    """Show color in background of color selector comboboxes."""
    def paint(self, painter, option, index):
        painter.save()
        color = index.data(Qt.BackgroundRole)
        if color:
            painter.fillRect(option.rect, color)
        painter.drawText(option.rect, Qt.AlignCenter, index.data(Qt.DisplayRole))
        painter.restore()

    def sizeHint(self, option, index):
        return QSize(70, 20)

class WaferPlot:
    """Class to plot wafer map"""

    def __init__(self, inter_method='linear'):
        self.inter_method = inter_method  # Interpolation method

    def plot(self, ax, x, y, z, cmap="jet", r=100, vmax=None, vmin=None,
             stats=True):
        """
        Plot a wafer map on the provided axes.

        Args:
        ax (matplotlib.axes.Axes): Axes object to plot the wafer map.
        x (array-like): X-coordinates of measurement points.
        y (array-like): Y-coordinates of measurement points.
        z (array-like): measurement data corresponding to (x,y) points.
        cmap (str, optional): Colormap for the plot. Defaults to "jet".
        r (float, optional): Radius of the wafer in millimeters.
        vmax (float, optional): Maximum value for the color scale.
        vmin (float, optional): Minimum value for the color scale.
        stats (bool, optional): Display statistical values on the plot.
        """
        # Generate a meshgrid for the wafer and Interpolate z onto the meshgrid
        xi, yi = np.meshgrid(np.linspace(-r, r, 300), np.linspace(-r, r, 300))
        zi = self.interpolate_data(x, y, z, xi, yi)

        # Plot the wafer map
        im = ax.imshow(zi, extent=[-r - 1, r + 1, -r - 0.5, r + 0.5],
                       origin='lower', cmap=cmap, interpolation='nearest')

        # Add open circles corresponding to measurement points
        ax.scatter(x, y, facecolors='none', edgecolors='black', s=20)

        # Add a circle as a decoration
        wafer_circle = patches.Circle((0, 0), radius=r, fill=False,
                                      color='black', linewidth=1)
        ax.add_patch(wafer_circle)

        ax.set_ylabel("Wafer size (mm)")

        # Remove unnecessary axes
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x', which='both', bottom=False, top=False)
        ax.tick_params(axis='y', which='both', right=False, left=True)
        ax.set_xticklabels([])

        # Set color scale limits if provided
        if vmax is not None and vmin is not None:
            im.set_clim(vmin, vmax)
        plt.colorbar(im, ax=ax)
        if stats:
            self.stats(z, ax)

    def stats(self, z, ax):
        """
        Calculate and display statistical values in the wafer plot.
        """
        # Calculate statistical values
        mean_value = z.mean()
        max_value = z.max()
        min_value = z.min()
        sigma_value = z.std()
        three_sigma_value = 3 * sigma_value

        # Annotate the plot with statistical values
        ax.text(0.05, -0.1, f"3\u03C3: {three_sigma_value:.2f}",
                transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom')
        ax.text(0.99, -0.1, f"Max: {max_value:.2f}",
                transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom',
                horizontalalignment='right')
        ax.text(0.99, -0.05, f"Min: {min_value:.2f}",
                transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom',
                horizontalalignment='right')
        ax.text(0.05, -0.05, f"Mean: {mean_value:.2f}",
                transform=ax.transAxes, fontsize=10, verticalalignment='bottom')

    def interpolate_data(self, x, y, z, xi, yi):
        """
        Interpolate data onto a regular grid using the specified
        interpolation method.
        """
        zi = griddata((x, y), z, (xi, yi), method=self.inter_method)
        return zi
