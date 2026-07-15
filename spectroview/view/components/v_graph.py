"""Matplotlib graph visualization widget for MVVM pattern."""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from PySide6.QtWidgets import QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QDialog
from PySide6.QtCore import QObject, QEvent, QSize, Signal, QTimer
from PySide6.QtGui import QIcon

from spectroview import DEFAULT_COLORS, DEFAULT_MARKERS, ICON_DIR, PLOT_POLICY_LIGHT
from spectroview.view.components.customize_graph_dialog import (
    EditLineDialog, EditTextDialog
)
from spectroview.model.m_settings import MSettings
from spectroview.viewmodel.utils import rgba_to_default_color, show_alert, copy_fig_to_clb, get_tinted_icon
from spectroview.view.components.v_plot_renderer import PlotRenderer


class VGraph(QWidget):
    """Graph widget rendering plots based on MGraph model properties."""
    # Signal emitted when graph properties are directly changed
    properties_changed = Signal(int, dict)
    # Signal emitted when annotation position changes (graph_id, ann_id, new_x, new_y)
    annotation_position_changed = Signal(int, str, float, float)
    # Signal emitted when replicate is requested
    replicate_requested = Signal(int)
    # Signal emitted when customize dialog is requested (graph_id)
    customize_requested = Signal(int)
    
    def __init__(self, graph_id=None):
        super().__init__()
        self.graph_id = graph_id
        
        # Data source
        self.df_name = None
        self.filters = {}
        # Store DataFrame for replotting
        self.df = None
        
        # Plot dimensions
        self.plot_width = 480  
        self.plot_height = 400
        self.dpi = 100
        
        # Plot type and axes
        self.plot_style = "point"
        self.x = None
        self.y = []
        self.z = None
        
        # Axis limits
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.zmin = None
        self.zmax = None
        
        # Labels
        self.plot_title = None
        self.xlabel = None
        self.ylabel = None
        self.zlabel = None
        
        # Axis scales
        self.xlogscale = False
        self.ylogscale = False
        self.y2logscale = False
        self.y3logscale = False
        
        # Secondary/tertiary axes
        self.y2 = None
        self.y3 = None
        self.y2min = None
        self.y2max = None
        self.y3min = None
        self.y3max = None
        self.y2label = None
        self.y3label = None
        
        # Secondary X axis
        self.x2 = None
        self.x2label = None
        self.x2min = None
        self.x2max = None
        self.x2logscale = False
        
        # Visual properties
        self.x_rot = 0
        self.grid = False
        self.legend_visible = True
        self.legend_outside = False
        self.legend_properties = []
        self.legend_bbox = None  # (x, y) in axes coords for dragged position
        
        # Plot-specific settings
        self.color_palette = "jet"
        self.wafer_size = 300
        self.wafer_stats = True
        self.trendline_order = 1
        self.show_trendline_eq = True
        self.trendline_anchor_enabled = False
        self.trendline_anchor_origin = True   # True = through (0,0), False = custom point
        self.trendline_anchor_x = 0.0
        self.trendline_anchor_y = 0.0
        self.trendline_equations = []  # List of dicts: {label, equation, r2} per hue group
        self.show_bar_plot_error_bar = False
        self.join_for_point_plot = False
        self.dodge_point_plot = True
        self.dodge_scatter_plot = False
        self.scatter_size = 70  # Marker size for scatter plots
        self.scatter_edgecolor = 'black'  # Edge color for scatter plot markers
        self.x_as_numeric = None  # None=Auto, True=Numerical, False=Category
        self.y_as_numeric = None  # None=Auto, True=Numerical, False=Category
        # Histogram-specific
        self.hist_bins = 20
        self.hist_kde = False
        self.hist_step = False
        
        # Data sorting
        self.sort_data_enabled = True   # Enable intelligent sorting
        self.sort_data_by = "Z"          # Sort by: "Z" (hue), "X", or "Y"
        
        # Annotations
        self.annotations = []
        
        # Axis breaks storage
        self.axis_breaks = {'x': None, 'y': None}
        self._break_markers = []  # Store artist references for cleanup
        
        # Matplotlib objects
        self.figure = None
        self.ax = None
        self.ax2 = None
        self.ax3 = None
        self.ax_x2 = None
        self.canvas = None
        
        # Layout setup
        self.graph_layout = QVBoxLayout()
        self.setLayout(self.graph_layout)
        self.graph_layout.setContentsMargins(0, 0, 0, 0)
        self.graph_layout.setSpacing(0)

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
        """Creates matplotlib figure canvas and adds it to layout."""
        if dpi:
            self.dpi = dpi
        else:
            self.dpi = 100
        
        self.clear_layout(self.graph_layout)
        plt.close('all')
        
        with plt.style.context(PLOT_POLICY_LIGHT):
            self.figure = plt.figure(layout="compressed", dpi=self.dpi)
            self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        original_set_style = self.toolbar.setStyleSheet
        def custom_set_style(css):
            original_set_style("QToolBar { background: transparent; border: none; }")
        self.toolbar.setStyleSheet = custom_set_style
        self.toolbar.setStyleSheet("")

        class ToolbarEventFilter(QObject):
            def __init__(self, toolbar):
                super().__init__()
                self.toolbar = toolbar
                self._timer = QTimer(self)
                self._timer.setSingleShot(True)
                self._timer.setInterval(100)
                self._timer.timeout.connect(self._update_icons)

            def eventFilter(self, obj, event):
                if event.type() == QEvent.PaletteChange:
                    self._timer.start()
                return False

            def _update_icons(self):
                action_dict = {action.text(): action for action in self.toolbar.actions() if action.text()}
                for text, tooltip_text, image_file, name_of_method in self.toolbar.toolitems:
                    if text in action_dict and image_file is not None:
                        try:
                            icon = self.toolbar._icon(image_file + '.png')
                            action_dict[text].setIcon(icon)
                        except Exception:
                            pass
                
        self.toolbar_filter = ToolbarEventFilter(self.toolbar)
        self.toolbar.installEventFilter(self.toolbar_filter)

        self.toolbar.setIconSize(QSize(30, 30))  # Set larger icon size
        for action in self.toolbar.actions():
            if action.text() in ['Save', 'Back', 'Forward']:
                action.setVisible(False)
        
        # Create Replicate button
        self.btn_replicate = QPushButton()
        self.btn_replicate.setIcon(QIcon(f"{ICON_DIR}/replicate.png"))
        self.btn_replicate.setIconSize(QSize(26, 26))
        self.btn_replicate.setFixedSize(30, 30)
        self.btn_replicate.setToolTip("Replicate graph")
        self.btn_replicate.clicked.connect(lambda *args: self.replicate_requested.emit(self.graph_id))

        # Create Customize button
        self.btn_customize = QPushButton()
        self.btn_customize.setIcon(QIcon(f"{ICON_DIR}/customize.png"))
        self.btn_customize.setIconSize(QSize(26, 26))
        self.btn_customize.setFixedSize(30, 30)
        self.btn_customize.setToolTip("Customize graph")
        self.btn_customize.clicked.connect(lambda: self.customize_requested.emit(self.graph_id))
        
        # Create Copy button
        self.btn_copy_figure = QPushButton()
        self.btn_copy_figure.setIcon(QIcon(f"{ICON_DIR}/copy.png"))
        self.btn_copy_figure.setIconSize(QSize(26, 26))
        self.btn_copy_figure.setFixedSize(30, 30)
        self.btn_copy_figure.setToolTip("Copy figure to clipboard")
        self.btn_copy_figure.clicked.connect(self.copy_to_clipboard)
        
        # Initialize icon colors using actual app settings
        theme = MSettings().get_theme()
        self.update_icon_colors(theme)
        
        # Create toolbar layout with customize and copy buttons
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(4)
        toolbar_layout.addWidget(self.toolbar)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.btn_replicate)
        toolbar_layout.addWidget(self.btn_customize)
        toolbar_layout.addWidget(self.btn_copy_figure)
        
        # Create container widget for toolbar layout
        toolbar_container = QWidget()
        toolbar_container.setLayout(toolbar_layout)
        toolbar_container.setFixedHeight(35)
        
        if layout:
            layout.addWidget(self.canvas)
            layout.addWidget(toolbar_container)
        else:
            self.graph_layout.addWidget(self.canvas)
            self.graph_layout.addWidget(toolbar_container)
        
        # Connect pick event for legend customization
        self.canvas.mpl_connect('pick_event', self._on_legend_pick)
        
        # Connect annotation drag events
        self.canvas.mpl_connect('motion_notify_event', self._on_annotation_drag)
        self.canvas.mpl_connect('button_release_event', self._on_annotation_release)
        self.canvas.mpl_connect('button_press_event', self._on_annotation_click)
        
        self.canvas.draw_idle()
    
    def copy_to_clipboard(self):
        """Copy the current figure to clipboard."""
        copy_fig_to_clb(self.canvas)
    
    def update_icon_colors(self, theme: str):
        """Update toolbar icons color based on current application theme."""
        icon_color = "#404040" if theme != "dark" else "#F0F0F0"
        if hasattr(self, 'btn_replicate'):
            self.btn_replicate.setIcon(get_tinted_icon(f"{ICON_DIR}/replicate.png", icon_color))
        if hasattr(self, 'btn_customize'):
            self.btn_customize.setIcon(get_tinted_icon(f"{ICON_DIR}/customize.png", icon_color))
        if hasattr(self, 'btn_copy_figure'):
            self.btn_copy_figure.setIcon(get_tinted_icon(f"{ICON_DIR}/copy.png", icon_color))

    def plot(self, df=None):
        """Wrapper to render plot using local style context."""
        with plt.style.context(PLOT_POLICY_LIGHT):
            self._plot_internal(df)

    def _plot_internal(self, df):
        """Renders plot based on DataFrame and current properties."""
        self.df = df
        
        # Ensure scatter_edgecolor is always a valid string color, defaulting to 'black'
        edge_c = getattr(self, 'scatter_edgecolor', 'black')
        if not edge_c or not isinstance(edge_c, str) or edge_c.strip() in ("", "None", "none", "null"):
            self.scatter_edgecolor = 'black'
        
        self.ax.clear()
        if self.ax2:
            self.ax2.clear()
        if self.ax3:
            self.ax3.clear()
        if self.ax_x2:
            self.ax_x2.clear()
        
        if df is not None and self.df_name is not None and self.x is not None and self.y is not None:
            self._plot_primary_axis(df)
            self._plot_secondary_axis(df)
            self._plot_tertiary_axis(df)
            self._plot_secondary_x_axis(df)
        else:
            self.ax.plot([], [])
        
        self._set_limits()
        self._set_axis_scale(df)
        self._set_labels()
        self._set_grid()
        self._set_rotation()
        self._set_legend()
        self._apply_axis_breaks()  # Apply breaks before annotations
        self._render_annotations()  # Render annotations after all plot elements
        
        self.get_legend_properties()
        self.canvas.draw_idle()
    
    def get_legend_properties(self):
        """Retrieves properties of each legend within legend box."""
        if hasattr(self, 'legend_properties') and self.legend_properties:
            return self.legend_properties
        
        import matplotlib.colors as mcolors
        legend_properties = []
        if self.ax:
            legend = self.ax.get_legend()
            if legend:
                legend_texts = legend.get_texts()
                # legend_handles holds the actual drawn artists — read their true colors
                legend_handles = getattr(legend, 'legend_handles', [])
                for idx, text in enumerate(legend_texts):
                    label = text.get_text()
                    color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
                    marker = DEFAULT_MARKERS[idx % len(DEFAULT_MARKERS)] if DEFAULT_MARKERS else 'o'

                    # Read true color from the matplotlib handle
                    if idx < len(legend_handles):
                        handle = legend_handles[idx]
                        try:
                            c = None
                            if hasattr(handle, 'get_color'):
                                c = handle.get_color()
                            elif hasattr(handle, 'get_facecolor'):
                                fc = handle.get_facecolor()
                                if hasattr(fc, '__len__') and len(fc) > 0:
                                    c = fc[0]
                                    
                            if c is not None:
                                hex_c = None
                                if isinstance(c, str) and c not in ('none', '', 'None'):
                                    if c.startswith('#'):
                                        hex_c = mcolors.to_hex(c)
                                    else:
                                        color = c
                                elif not isinstance(c, str):
                                    hex_c = mcolors.to_hex(c)
                                    
                                if hex_c is not None:
                                    matched = False
                                    for dc in DEFAULT_COLORS:
                                        if mcolors.to_hex(dc).lower() == hex_c.lower():
                                            color = dc
                                            matched = True
                                            break
                                    if not matched:
                                        color = hex_c
                        except Exception:
                            pass
                        try:
                            if hasattr(handle, 'get_marker'):
                                m = handle.get_marker()
                                if m and str(m) not in ('None', 'none', ''):
                                    marker = str(m)
                        except Exception:
                            pass

                    rgba_color = list(mcolors.to_rgba(color))
                    legend_properties.append({
                        'label': label,
                        'marker': marker,
                        'color': color,
                        'rgba': rgba_color
                    })
            elif self.plot_style not in ['2Dmap', 'wafer']:
                color = DEFAULT_COLORS[0] if DEFAULT_COLORS else 'steelblue'
                rgba_color = list(mcolors.to_rgba(color))
                legend_properties.append({
                    'label': 'All data',
                    'marker': 'o',
                    'color': color,
                    'rgba': rgba_color
                })
        
        self.legend_properties = legend_properties
        return self.legend_properties
    
    
    def _on_legend_pick(self, event):
        """Handle pick event — record annotation candidate for drag, or legend double-click."""
        artist = event.artist
        if hasattr(artist, '_annotation_data'):
            # Record as drag candidate (actual drag starts on mouse move)
            self._drag_candidate = artist
            self._drag_start_x = event.mouseevent.xdata
            self._drag_start_y = event.mouseevent.ydata
            return
        
        # Check if legend was clicked
        if artist.get_label() == '_legend_':
            return
        
        # Check if it's a double-click on the legend
        legend = self.ax.get_legend()
        if legend and artist == legend and event.mouseevent.dblclick:
            # Emit signal to let workspace handle the dialog
            self.customize_requested.emit(self.graph_id)
    
    def _plot_primary_axis(self, df):
        """Plots data on the primary axis based on the current plot style."""
        self.renderer = PlotRenderer(self)
        # Determine number of hue categories
        n_categories = df[self.z].nunique() if self.z and self.z in df.columns else 0
        
        # Reset legend_properties if number of categories changed
        if self.legend_properties and n_categories > 0 and len(self.legend_properties) != n_categories:
            self.legend_properties = []
        
        if not self.legend_properties:
            markers = DEFAULT_MARKERS.copy()
            colors = DEFAULT_COLORS.copy()
        else:
            markers = [str(prop['marker']) for prop in self.legend_properties]
            colors = [str(prop['color']) for prop in self.legend_properties]
        
        # Extend or trim colors/markers to match actual number of hue categories
        if n_categories > 0:
            # Extend by cycling through DEFAULT_COLORS/DEFAULT_MARKERS if needed
            while len(colors) < n_categories:
                idx = len(colors)
                colors.append(DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])
                markers.append(DEFAULT_MARKERS[idx % len(DEFAULT_MARKERS)])
            # Trim if too many
            colors = colors[:n_categories]
            markers = markers[:n_categories]
        
        for y in self.y:
            c = colors[0] if colors else 'steelblue'
            
            if self.plot_style == 'point':
                self.renderer._plot_point(df, y, colors, markers, c)
            elif self.plot_style == 'scatter':
                self.renderer._plot_scatter(df, y, colors, c)
            elif self.plot_style == 'box':
                self.renderer._plot_box(df, y, colors, c)
            elif self.plot_style == 'line':
                self.renderer._plot_line(df, y, colors, c)
            elif self.plot_style == 'bar':
                self.renderer._plot_bar(df, y, colors, c)
            elif self.plot_style == 'trendline':
                self.renderer._plot_trendline(df, y, colors, c)
            elif self.plot_style == 'histogram':
                self.renderer._plot_histogram(df, colors)
            elif self.plot_style == 'wafer':
                self.renderer._plot_wafer(df)
            elif self.plot_style == '2Dmap':
                self.renderer._plot_2dmap(df, y)
            else:
                show_alert("Unsupported plot style")

    
    def _set_legend(self):
        """Sets up and displays the legend for the plot."""
        handles, labels = self.ax.get_legend_handles_labels()
        
        if self.ax2:
            handles2, labels2 = self.ax2.get_legend_handles_labels()
            handles += handles2
            labels += labels2
            if self.ax2.get_legend():
                self.ax2.get_legend().remove()
        
        if self.ax3:
            handles3, labels3 = self.ax3.get_legend_handles_labels()
            handles += handles3
            labels += labels3
            if self.ax3.get_legend():
                self.ax3.get_legend().remove()
        
        if self.ax_x2:
            handles_x2, labels_x2 = self.ax_x2.get_legend_handles_labels()
            handles += handles_x2
            labels += labels_x2
            if self.ax_x2.get_legend():
                self.ax_x2.get_legend().remove()
        
        if handles:
                
            if self.legend_visible:
                unique_labels = []
                unique_handles = []
                for h, l in zip(handles, labels):
                    if l not in unique_labels:
                        unique_labels.append(l)
                        unique_handles.append(h)
                        
                # Apply custom labels from legend_properties
                if getattr(self, 'legend_properties', []):
                    custom_labels = []
                    for i, l in enumerate(unique_labels):
                        if i < len(self.legend_properties):
                            custom_labels.append(self.legend_properties[i].get('label', l))
                        else:
                            custom_labels.append(l)
                    unique_labels = custom_labels
                        
                if getattr(self, 'legend_outside', False):
                    legend = self.ax.legend(unique_handles, unique_labels, loc='center left', bbox_to_anchor=(1.02, 0.5), framealpha=0.7)
                else:
                    legend = self.ax.legend(unique_handles, unique_labels, loc='best', framealpha=0.7)
                    if getattr(self, 'legend_bbox', None) is not None:
                        legend._loc = tuple(self.legend_bbox)
                        
                legend.set_picker(True)
                legend.set_draggable(True)
            else:
                if self.ax.get_legend():
                    self.ax.get_legend().remove()
    
    def _save_legend_position(self):
        """Save current legend position in axes coordinates."""
        legend = self.ax.get_legend()
        if legend is not None:
            loc = legend._loc
            # After dragging, _loc becomes a tuple (x, y) in axes coords
            if isinstance(loc, tuple):
                self.legend_bbox = [float(loc[0]), float(loc[1])]
    
    def _set_grid(self):
        """Add grid for the plot (supports linear & log scale automatically)."""
        is_wafer = (self.plot_style == 'wafer')
        self.ax.set_axisbelow(True)
        
        self.ax.tick_params(
            which='major',
            bottom=False if is_wafer else True,
            left=True,
            top=False if is_wafer else getattr(self, 'minor_ticks_top', False),
            right=getattr(self, 'minor_ticks_right', False)
        )
        
        if any([getattr(self, 'minor_ticks_bottom', True),
                getattr(self, 'minor_ticks_left', True),
                getattr(self, 'minor_ticks_top', False),
                getattr(self, 'minor_ticks_right', False)]):
            self.ax.minorticks_on()
            self.ax.tick_params(
                which='minor',
                bottom=False if is_wafer else getattr(self, 'minor_ticks_bottom', True),
                top=False if is_wafer else getattr(self, 'minor_ticks_top', False),
                left=getattr(self, 'minor_ticks_left', True),
                right=getattr(self, 'minor_ticks_right', False)
            )
        else:
            self.ax.minorticks_off()

        if not self.grid:
            self.ax.grid(False)
            return
        
        self.ax.grid(True, which='major', linestyle='--')
        
        if self.xlogscale or self.ylogscale:
            self.ax.grid(True, which='minor', alpha=0.15, linestyle='--')
    
    def _set_rotation(self):
        """Set rotation of the ticklabels of the x axis."""
        if self.x_rot != 0:
            plt.setp(
                self.ax.get_xticklabels(),
                rotation=self.x_rot,
                ha="right",
                rotation_mode="anchor"
            )
        else:
            # Reset to default when rotation is 0
            plt.setp(
                self.ax.get_xticklabels(),
                rotation=0,
                ha="center",
                rotation_mode=None
            )
    
    
    @staticmethod
    def _apply_limit_pair(setter, vmin, vmax, axis_name: str) -> None:
        """Apply a (min, max) limit pair via *setter* (e.g. ax.set_xlim),
        skipping degenerate equal bounds instead of handing matplotlib a
        zero-width range (which triggers a "singular transformation"
        UserWarning and silently auto-expands anyway). Uses explicit
        `is not None` checks (not truthy checks) so a limit of exactly 0.0
        -- a common, legitimate axis bound -- is not silently ignored.
        """
        if vmin is None or vmax is None:
            return
        vmin, vmax = float(vmin), float(vmax)
        if vmin == vmax:
            print(f"[INFO] Skipping {axis_name} limits: min == max ({vmin}).")
            return
        setter(vmin, vmax)

    def _set_limits(self):
        """Set the limits of axis."""
        self._apply_limit_pair(self.ax.set_xlim, self.xmin, self.xmax, "x-axis")
        self._apply_limit_pair(self.ax.set_ylim, self.ymin, self.ymax, "y-axis")
        if self.ax2:
            self._apply_limit_pair(self.ax2.set_ylim, self.y2min, self.y2max, "y2-axis")
        if self.ax3:
            self._apply_limit_pair(self.ax3.set_ylim, self.y3min, self.y3max, "y3-axis")
        if self.ax_x2:
            self._apply_limit_pair(self.ax_x2.set_xlim, self.x2min, self.x2max, "x2-axis")
    
    def _set_axis_scale(self, df):
        """Apply log scale only if the corresponding axis column is numeric."""
        if self.xlogscale:
            x_data = df[self.x]
            if np.issubdtype(x_data.dtype, np.number):
                self.ax.set_xscale('log')
            else:
                print(f"[INFO] Skipping x-logscale because '{self.x}' is categorical.")
        
        if self.ylogscale and len(self.y) > 0:
            y_data = df[self.y[0]]
            if np.issubdtype(y_data.dtype, np.number):
                self.ax.set_yscale('log')
            else:
                print(f"[INFO] Skipping y-logscale because '{self.y[0]}' is categorical.")
        
        if self.ax2 and self.y2 and self.y2logscale:
            y2_data = df[self.y2]
            if np.issubdtype(y2_data.dtype, np.number):
                self.ax2.set_yscale('log')
        
        if self.ax3 and self.y3 and self.y3logscale:
            y3_data = df[self.y3]
            if np.issubdtype(y3_data.dtype, np.number):
                self.ax3.set_yscale('log')
        
        if self.ax_x2 and self.x2 and self.x2logscale:
            x2_data = df[self.x2]
            if np.issubdtype(x2_data.dtype, np.number):
                self.ax_x2.set_xscale('log')
    
    def _format_axis_label(self, col_name) -> str:
        """Format axis label based on specific parameter rules.

        E.g. "x0_Si" -> "Si peak position (cm$^{-1}$)". Column names of the
        form "<param>_<peaklabel>" (fit-result columns) are turned into a
        friendly, unit-annotated label instead of shown raw.
        """
        if not col_name or not isinstance(col_name, str):
            return str(col_name) if col_name is not None else ""

        parts = col_name.split('_', 1)
        if len(parts) == 2:
            param, peaklabel = parts
            param = param.lower()
            if param == "x0":
                return f"{peaklabel} peak position (cm$^{{-1}}$)"
            elif param == "fwhm":
                return f"{peaklabel} peak width (cm$^{{-1}}$)"
            elif param == "ampli":
                return f"{peaklabel} peak intensity (a.u.)"
            elif param == "area":
                return f"{peaklabel} peak area (a.u.)"

        return col_name

    def _get_y_label_default(self, y_col):
        if isinstance(y_col, list) and len(y_col) > 0:
            return self._format_axis_label(y_col[0])
        return self._format_axis_label(y_col)

    def _set_labels(self):
        """Set titles and labels for axis and plot."""
        if self.plot_style == 'wafer':
            if self.plot_title:
                self.ax.set_title(self.plot_title)
            else:
                self.ax.set_title(self._format_axis_label(self.z) if self.z else "")
            self.ax.tick_params(axis='x', labelbottom=False)
        elif self.plot_style == '2Dmap':
            if self.plot_title:
                self.ax.set_title(self.plot_title)
            else:
                self.ax.set_title(self._format_axis_label(self.z) if self.z else "")
            if self.xlabel:
                self.ax.set_xlabel(self.xlabel)
            else:
                self.ax.set_xlabel(self._format_axis_label(self.x))
            if self.ylabel:
                self.ax.set_ylabel(self.ylabel)
            else:
                self.ax.set_ylabel(self._get_y_label_default(self.y))
        else:
            self.ax.set_title(self.plot_title)
            if self.xlabel:
                self.ax.set_xlabel(self.xlabel)
            else:
                self.ax.set_xlabel(self._format_axis_label(self.x))
            if self.ylabel:
                self.ax.set_ylabel(self.ylabel)
            else:
                if self.plot_style == 'histogram':
                    self.ax.set_ylabel("Frequency")
                else:
                    self.ax.set_ylabel(self._get_y_label_default(self.y))
    
    def _draw_twin_series(self, twin_ax, x_vals, y_vals, *, color, marker, series_label):
        """Draw one series onto a twin axis, dispatching by the primary
        plot_style (point/line/scatter share this look across y2/y3/x2).

        Returns True if drawn, False if plot_style isn't supported on a twin
        axis — the caller is responsible for removing the (now-empty) axis
        in that case, matching each twin axis's prior standalone behavior.
        """
        if self.plot_style == 'line':
            twin_ax.plot(x_vals, y_vals, color=color, label=series_label)
        elif self.plot_style == 'point':
            twin_ax.errorbar(
                x_vals, y_vals,
                fmt=marker, color=color, markeredgecolor='black',
                markeredgewidth=1, capsize=3,
                linestyle='-' if self.join_for_point_plot else 'none',
                label=series_label
            )
        elif self.plot_style == 'scatter':
            twin_ax.scatter(
                x_vals, y_vals,
                s=self.scatter_size, edgecolors=self.scatter_edgecolor,
                color=color, label=series_label
            )
        else:
            return False
        return True

    def _plot_secondary_axis(self, df):
        """Plot data on secondary y-axis (y2, twinx, red)."""
        if self.ax2:
            self.ax2.remove()
            self.ax2 = None

        if not self.y2:
            return

        self.ax2 = self.ax.twinx()

        # For simplicity with secondary axes, just use numeric spacing if x is numeric,
        # otherwise just plot directly (matplotlib will handle it if X is strings)
        x_vals = df[self.x]
        if getattr(self, 'x_as_numeric', False):
            x_vals = pd.to_numeric(x_vals, errors='coerce')

        if self._draw_twin_series(self.ax2, x_vals, df[self.y2], color='red', marker='s', series_label=self.y2):
            self.ax2.set_ylabel(self.y2label or self._format_axis_label(self.y2), color='red')
            self.ax2.tick_params(axis='y', colors='red')
        else:
            self.ax2.remove()
            self.ax2 = None

    def _plot_tertiary_axis(self, df):
        """Plot data on tertiary y-axis (y3, twinx offset outward, green)."""
        if self.ax3:
            self.ax3.remove()
            self.ax3 = None

        if not self.y3:
            return

        self.ax3 = self.ax.twinx()
        self.ax3.spines["right"].set_position(("outward", 100))

        x_vals = df[self.x]
        if getattr(self, 'x_as_numeric', False):
            x_vals = pd.to_numeric(x_vals, errors='coerce')

        if self._draw_twin_series(self.ax3, x_vals, df[self.y3], color='green', marker='s', series_label=self.y3):
            self.ax3.set_ylabel(self.y3label or self._format_axis_label(self.y3), color='green')
            self.ax3.tick_params(axis='y', colors='green')
        else:
            self.ax3.remove()
            self.ax3 = None

    def _plot_secondary_x_axis(self, df):
        """Plot data on secondary x-axis (x2, twiny, purple)."""
        if self.ax_x2:
            self.ax_x2.remove()
            self.ax_x2 = None

        if not (self.x2 and self.x2 in df.columns):
            return

        self.ax_x2 = self.ax.twiny()

        x_vals = df[self.x2]
        if getattr(self, 'x2logscale', False):
            x_vals = pd.to_numeric(x_vals, errors='coerce')

        if self._draw_twin_series(self.ax_x2, x_vals, df[self.y[0]], color='purple', marker='D', series_label=self.y[0]):
            self.ax_x2.set_xlabel(
                self.x2label or self._format_axis_label(self.x2), color='purple'
            )
            self.ax_x2.tick_params(axis='x', colors='purple')
        else:
            self.ax_x2.remove()
            self.ax_x2 = None
    
    # ═══════════════════════════════════════════════════════════════════
    # Annotation Rendering
    # ═══════════════════════════════════════════════════════════════════
    
    def _render_annotations(self):
        """Render all annotations (lines and text) on the plot."""
        if not self.annotations:
            return
        
        for ann in self.annotations:
            ann_type = ann.get('type')
            try:
                if ann_type == 'vline':
                    self._render_vline(ann)
                elif ann_type == 'hline':
                    self._render_hline(ann)
                elif ann_type == 'text':
                    self._render_text(ann)
            except Exception as e:
                print(f"[WARNING] Failed to render annotation {ann.get('id')}: {e}")
    
    def _render_vline(self, ann: dict):
        """Render vertical line annotation."""
        x_pos = ann.get('x', 0)
        color = ann.get('color', 'red')
        linestyle = ann.get('linestyle', '--')
        linewidth = ann.get('linewidth', 1.5)
        label = ann.get('label', None)
        
        line = self.ax.axvline(
            x=x_pos,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            zorder=100,  # Render on top of other elements
            picker=5  # Enable picking with 5pt tolerance
        )
        
        # Attach metadata for drag handling
        line._annotation_data = ann
        line._is_dragging = False
        return line
    
    def _render_hline(self, ann: dict):
        """Render horizontal line annotation."""
        y_pos = ann.get('y', 0)
        color = ann.get('color', 'blue')
        linestyle = ann.get('linestyle', '--')
        linewidth = ann.get('linewidth', 1.5)
        label = ann.get('label', None)
        
        line = self.ax.axhline(
            y=y_pos,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            zorder=100,  # Render on top of other elements
            picker=5  # Enable picking with 5pt tolerance
        )
        
        # Attach metadata for drag handling
        line._annotation_data = ann
        line._is_dragging = False
        return line
    
    def _render_text(self, ann: dict):
        """Render text annotation."""
        x_pos = ann.get('x', 0)
        y_pos = ann.get('y', 0)
        text = ann.get('text', '')
        fontsize = ann.get('fontsize', 12)
        color = ann.get('color', 'black')
        ha = ann.get('ha', 'center')
        va = ann.get('va', 'center')
        
        # Get bbox from annotation, use default if not specified
        bbox_props = ann.get('bbox')
        
        text_obj = self.ax.text(
            x_pos,
            y_pos,
            text,
            fontsize=fontsize,
            color=color,
            ha=ha,
            va=va,
            bbox=bbox_props,
            zorder=101,  # Render on top of lines
            picker=True  # Enable picking for drag functionality
        )
        
        # Attach metadata for drag handling
        text_obj._annotation_data = ann
        text_obj._is_dragging = False
        return text_obj
    
    def _on_annotation_click(self, event):
        """Handle click on annotation — only double-click opens edit dialog."""
        if not event.dblclick or event.inaxes != self.ax:
            return
        
        # Cancel any pending drag on double-click
        if hasattr(self, '_drag_candidate'):
            del self._drag_candidate
        if hasattr(self, '_dragged_annotation'):
            self._dragged_annotation._is_dragging = False
            del self._dragged_annotation
        
        # Check if double-click is on an annotation
        for ann in self.ax.findobj():
            if hasattr(ann, '_annotation_data'):
                contains, _ = ann.contains(event)
                if contains:
                    self._edit_annotation_direct(ann._annotation_data)
                    return
                    
    def _edit_annotation_direct(self, annotation):
        """Open edit dialog for annotation (called from double-click)."""
        
        # Open appropriate edit dialog based on type
        if annotation['type'] in ['vline', 'hline']:
            dialog = EditLineDialog(annotation, None)
            if dialog.exec() == QDialog.Accepted:
                # Update annotation properties
                props = dialog.get_properties()
                annotation.update(props)
                
                # Update label
                if annotation['type'] == 'vline':
                    annotation['label'] = f"V-Line at x={annotation['x']:.2f}"
                else:
                    annotation['label'] = f"H-Line at y={annotation['y']:.2f}"
                
                # Refresh plot
                self.ax.clear()
                if self.df is not None:
                    self.plot(self.df)
        
        elif annotation['type'] == 'text':
            dialog = EditTextDialog(annotation, None)
            if dialog.exec() == QDialog.Accepted:
                # Update annotation properties
                props = dialog.get_properties()
                annotation.update(props)
                
                # Refresh plot
                self.ax.clear()
                if self.df is not None:
                    self.plot(self.df)
    
    def _on_annotation_drag(self, event):
        """Handle annotation drag (mouse move while dragging)."""
        if event.xdata is None or event.ydata is None:
            return
        
        # Promote drag candidate to actual drag once mouse moves
        if hasattr(self, '_drag_candidate') and not hasattr(self, '_dragged_annotation'):
            dx = abs(event.xdata - (self._drag_start_x or 0))
            dy = abs(event.ydata - (self._drag_start_y or 0))
            # Use a small threshold to distinguish click from drag
            x_range = abs(self.ax.get_xlim()[1] - self.ax.get_xlim()[0])
            y_range = abs(self.ax.get_ylim()[1] - self.ax.get_ylim()[0])
            if dx > x_range * 0.005 or dy > y_range * 0.005:
                self._dragged_annotation = self._drag_candidate
                self._dragged_annotation._is_dragging = True
                del self._drag_candidate
        
        if not hasattr(self, '_dragged_annotation'):
            return
        
        ann = self._dragged_annotation
        if not getattr(ann, '_is_dragging', False):
            return
        
        ann_data = ann._annotation_data
        
        # Update visual position based on annotation type
        if ann_data['type'] == 'vline':
            ann.set_xdata([event.xdata, event.xdata])
            ann_data['x'] = event.xdata
        elif ann_data['type'] == 'hline':
            ann.set_ydata([event.ydata, event.ydata])
            ann_data['y'] = event.ydata
        elif ann_data['type'] == 'text':
            ann.set_position((event.xdata, event.ydata))
            ann_data['x'] = event.xdata
            ann_data['y'] = event.ydata
        
        self.canvas.draw_idle()
    
    def _on_annotation_release(self, event):
        """Handle mouse release (finish dragging)."""
        # Clean up drag candidate if no drag occurred
        if hasattr(self, '_drag_candidate'):
            del self._drag_candidate
        
        if not hasattr(self, '_dragged_annotation'):
            return
        
        ann = self._dragged_annotation
        ann._is_dragging = False
        
        # Emit signal to update model
        ann_data = ann._annotation_data
        if ann_data['type'] == 'vline':
            self.annotation_position_changed.emit(
                self.graph_id, ann_data['id'], ann_data['x'], 0
            )
        elif ann_data['type'] == 'hline':
            self.annotation_position_changed.emit(
                self.graph_id, ann_data['id'], 0, ann_data['y']
            )
        elif ann_data['type'] == 'text':
            self.annotation_position_changed.emit(
                self.graph_id, ann_data['id'], ann_data['x'], ann_data['y']
            )
        
        del self._dragged_annotation
        self.canvas.draw_idle()
    
    def _apply_axis_breaks(self):
        """Apply axis breaks by adjusting limits and hiding ticks in break range."""
        if not hasattr(self, 'axis_breaks') or not self.axis_breaks:
            return
        
        # Clear old break markers to prevent accumulation
        if hasattr(self, '_break_markers'):
            for artist in self._break_markers:
                try:
                    artist.remove()
                except:
                    pass  # Artist may already be removed
            self._break_markers = []
        else:
            self._break_markers = []
        
        # Apply X-axis break  
        if self.axis_breaks.get('x'):
            self._apply_single_axis_break('x', self.axis_breaks['x']['start'], self.axis_breaks['x']['end'])
        
        # Apply Y-axis break
        if self.axis_breaks.get('y'):
            self._apply_single_axis_break('y', self.axis_breaks['y']['start'], self.axis_breaks['y']['end'])

    def _apply_single_axis_break(self, axis, break_start, break_end):
        """Apply a single axis break (shared logic for X and Y)."""
        is_x = (axis == 'x')
        
        # Get current limits
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        
        min_val, max_val = (x_min, x_max) if is_x else (y_min, y_max)
        
        # Don't apply if break is outside data range
        if break_start < min_val or break_end > max_val:
            return
            
        break_range = break_end - break_start
        
        # Use fixed pixel gap for consistent appearance
        gap_pixels = 3
        # Convert pixels to data coordinates
        bbox = self.ax.get_window_extent().transformed(self.figure.dpi_scale_trans.inverted())
        size_idx = 0 if is_x else 1
        bbox_size = bbox.width if is_x else bbox.height
        gap_size = gap_pixels / bbox_size * (max_val - min_val) / self.figure.get_size_inches()[size_idx]
        
        for line in self.ax.get_lines():
            xdata = np.asarray(line.get_xdata())
            ydata = np.asarray(line.get_ydata())
            
            data = xdata if is_x else ydata
            other_data = ydata if is_x else xdata
            
            data_new = data.copy()
            mask = data >= break_end
            data_new[mask] = data[mask] - break_range + gap_size
            
            keep_mask = (data <= break_start) | (data >= break_end)
            
            if is_x:
                line.set_data(data_new[keep_mask], other_data[keep_mask])
            else:
                line.set_data(other_data[keep_mask], data_new[keep_mask])
                
        for collection in self.ax.collections:
            offsets = collection.get_offsets()
            if offsets is not None and len(offsets) > 0:
                xdata = offsets[:, 0]
                ydata = offsets[:, 1]
                
                data = xdata if is_x else ydata
                other_data = ydata if is_x else xdata
                
                data_new = data.copy()
                mask = data >= break_end
                data_new[mask] = data[mask] - break_range + gap_size
                
                keep_mask = (data <= break_start) | (data >= break_end)
                
                if is_x:
                    collection.set_offsets(np.column_stack([data_new[keep_mask], other_data[keep_mask]]))
                else:
                    collection.set_offsets(np.column_stack([other_data[keep_mask], data_new[keep_mask]]))
                
                for get_func, set_func in [
                    (collection.get_facecolors, collection.set_facecolors),
                    (collection.get_edgecolors, collection.set_edgecolors),
                    (collection.get_sizes, collection.set_sizes)
                ]:
                    try:
                        vals = get_func()
                        if vals is not None and len(vals) == len(data):
                            set_func(vals[keep_mask])
                    except Exception:
                        pass
        
        # Adjust axis limits and add markers
        new_range = max_val - break_range + gap_size - min_val
        ax_break = (break_start + gap_size / 2 - min_val) / new_range
        
        d = 0.015  # how big to make the diagonal lines in axes coordinates
        dx = 0.01  # half distance between the two parallel lines
        kwargs = dict(transform=self.ax.transAxes, color='gray', clip_on=False, linewidth=1)
        
        if is_x:
            self.ax.set_xlim(min_val, max_val - break_range + gap_size)
            p1, = self.ax.plot([ax_break - dx - d, ax_break - dx + d], [-d, d], **kwargs)
            p2, = self.ax.plot([ax_break + dx - d, ax_break + dx + d], [-d, d], **kwargs)
            p3, = self.ax.plot([ax_break - dx - d, ax_break - dx + d], [1 - d, 1 + d], **kwargs)
            p4, = self.ax.plot([ax_break + dx - d, ax_break + dx + d], [1 - d, 1 + d], **kwargs)
            
            def break_formatter(val, pos):
                if val > break_start + gap_size / 2:
                    return f"{val + break_range - gap_size:g}"
                return f"{val:g}"
            self.ax.xaxis.set_major_formatter(FuncFormatter(break_formatter))
        else:
            self.ax.set_ylim(min_val, max_val - break_range + gap_size)
            p1, = self.ax.plot([-d, d], [ax_break - dx - d, ax_break - dx + d], **kwargs)
            p2, = self.ax.plot([-d, d], [ax_break + dx - d, ax_break + dx + d], **kwargs)
            p3, = self.ax.plot([1 - d, 1 + d], [ax_break - dx - d, ax_break - dx + d], **kwargs)
            p4, = self.ax.plot([1 - d, 1 + d], [ax_break + dx - d, ax_break + dx + d], **kwargs)
            
            def break_formatter(val, pos):
                if val > break_start + gap_size / 2:
                    return f"{val + break_range - gap_size:g}"
                return f"{val:g}"
            self.ax.yaxis.set_major_formatter(FuncFormatter(break_formatter))
            
        self._break_markers.extend([p1, p2, p3, p4])
        




