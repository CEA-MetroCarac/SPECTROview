# view/v_graph.py
"""Matplotlib graph visualization widget for MVVM pattern."""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from scipy.interpolate import griddata
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from PySide6.QtWidgets import QVBoxLayout, QLabel, QLineEdit, QWidget, QComboBox, QStyledItemDelegate, QPushButton, QHBoxLayout
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QPalette, QColor, QIcon

from spectroview import ICON_DIR
from spectroview.view.components.customize_graph_dialog import CustomizeGraphDialog
from spectroview import DEFAULT_COLORS, DEFAULT_MARKERS, MARKERS
from spectroview.viewmodel.utils import rgba_to_default_color, show_alert, copy_fig_to_clb


class VGraph(QWidget):
    """Graph widget rendering plots based on MGraph model properties."""
    
    # Signal emitted when annotation position changes (graph_id, ann_id, new_x, new_y)
    annotation_position_changed = Signal(int, str, float, float)
    
    def __init__(self, graph_id=None):
        super().__init__()
        self.graph_id = graph_id
        
        # Data source
        self.df_name = None
        self.filters = {}
        
        # Plot dimensions
        self.plot_width = 480  
        self.plot_height = 400
        self.dpi = 90
        
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
        self.show_bar_plot_error_bar = True
        self.join_for_point_plot = False
        
        # Annotations
        self.annotations = []
        
        # Axis breaks storage
        self.axis_breaks = {'x': None, 'y': None}
        
        # Matplotlib objects
        self.figure = None
        self.ax = None
        self.ax2 = None
        self.ax3 = None
        self.canvas = None
        
        # Store DataFrame for replotting
        self.df = None
        
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
        
        self.figure = plt.figure(layout="compressed", dpi=self.dpi)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setIconSize(QSize(30, 30))  # Set larger icon size
        for action in self.toolbar.actions():
            if action.text() in ['Save', 'Back', 'Forward']:
                action.setVisible(False)
        
        # Create Customize button
        self.btn_customize = QPushButton()
        self.btn_customize.setIcon(QIcon(f"{ICON_DIR}/customize.png"))
        self.btn_customize.setIconSize(QSize(26, 26))
        self.btn_customize.setFixedSize(30, 30)
        self.btn_customize.setToolTip("Customize graph and annotations")
        self.btn_customize.clicked.connect(self._show_customize_dialog)
        
        # Create Copy button
        self.btn_copy_figure = QPushButton()
        self.btn_copy_figure.setIcon(QIcon(f"{ICON_DIR}/copy.png"))
        self.btn_copy_figure.setIconSize(QSize(26, 26))
        self.btn_copy_figure.setFixedSize(30, 30)
        self.btn_copy_figure.setToolTip("Copy figure to clipboard")
        self.btn_copy_figure.clicked.connect(self.copy_to_clipboard)
        
        # Create toolbar layout with customize and copy buttons
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(4)
        toolbar_layout.addWidget(self.toolbar)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.btn_customize)
        toolbar_layout.addWidget(self.btn_copy_figure)
        
        # Create container widget for toolbar layout
        from PySide6.QtWidgets import QWidget as QWidgetContainer
        toolbar_container = QWidgetContainer()
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
        try:
            
            copy_fig_to_clb(self.canvas)
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Copy Error", f"Error copying figure: {str(e)}")
    
    def plot(self, df):
        """Renders plot based on DataFrame and current properties."""
        self.df = df
        
        self.ax.clear()
        if self.ax2:
            self.ax2.clear()
        if self.ax3:
            self.ax3.clear()
        
        if self.df_name is not None and self.x is not None and self.y is not None:
            self._plot_primary_axis(df)
            self._plot_secondary_axis(df)
            self._plot_tertiary_axis(df)
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
        legend_properties = []
        if self.ax:
            legend = self.ax.get_legend()
            if legend:
                legend_texts = legend.get_texts()
                legend_handles = legend.legend_handles
                for idx, text in enumerate(legend_texts):
                    label = text.get_text()
                    handle = legend_handles[idx]
                    
                    # Extract RGBA color from seaborn
                    if self.plot_style in ['point', 'scatter', 'line']:
                        rgba_color = handle.get_markerfacecolor()
                        marker = handle.get_marker()
                    elif self.plot_style in ['box', 'bar']:
                        rgba_color = handle.get_facecolor()
                        marker = 'o'
                    else:
                        rgba_color = 'blue'
                        marker = 'o'
                    
                    # Map to closest DEFAULT_COLOR
                    color = rgba_to_default_color(rgba_color)
                    
                    legend_properties.append({
                        'label': label,
                        'marker': marker,
                        'color': color,
                        'rgba': rgba_color  # Keep original RGBA for mapping
                    })
        
        self.legend_properties = legend_properties
        
        # Fix box/bar patch colors to match DEFAULT_COLORS
        if self.plot_style in ['box', 'bar'] and self.legend_properties:
            self._fix_box_bar_colors()
        
        return self.legend_properties
    
    def customize_legend_widget(self, main_layout):
        """Displays legend properties in the GUI for user modifications."""
        self.clear_layout(main_layout)
        headers = ['Label', 'Marker', 'Color']
        
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
            elif header == "Color":
                color_layout.addWidget(label)
        
        for idx, prop in enumerate(self.legend_properties):
            # Label
            label = QLineEdit(prop['label'])
            label.setFixedWidth(200)
            label.textChanged.connect(lambda text, idx=idx: self.udp_legend(idx, 'label', text))
            label_layout.addWidget(label)
            
            # Marker
            if self.plot_style == 'point':
                marker = QComboBox()
                marker.addItems(MARKERS)
                marker.setCurrentText(prop['marker'])
                marker.currentTextChanged.connect(lambda text, idx=idx: self.udp_legend(idx, 'marker', text))
                marker_layout.addWidget(marker)
            
            # Color - only show original DEFAULT_COLORS (limit to 12 unique colors)
            color = QComboBox()
            delegate = ColorDelegate(color)
            color.setItemDelegate(delegate)
            
            # Use only the first 12 unique colors from DEFAULT_COLORS
            unique_colors = list(dict.fromkeys(DEFAULT_COLORS))[:12]
            for color_code in unique_colors:
                color.addItem(color_code)
                item = color.model().item(color.count() - 1)
                item.setBackground(QColor(color_code))
            
            color.setCurrentText(prop['color'])
            color.currentIndexChanged.connect(lambda idx, color=color: self.update_combobox_color(color))
            color.currentTextChanged.connect(lambda text, idx=idx: self.udp_legend(idx, 'color', text))
            color_layout.addWidget(color)
            
            self.update_combobox_color(color)
        
        # Add vertical stretch to absorb remaining space
        label_layout.addStretch()
        if self.plot_style == 'point':
            marker_layout.addStretch()
        color_layout.addStretch()
        
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
        self.canvas.draw_idle()
    
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
            # Open CustomizeGraphDialog with Legend tab
            if not hasattr(self, '_customize_dialog') or self._customize_dialog is None:
                self._customize_dialog = CustomizeGraphDialog(self, self.graph_id, parent=self)
            
            # Open the Legend tab
            self._customize_dialog.open_legend_tab()
    
    def _plot_primary_axis(self, df):
        """Plots data on the primary axis based on the current plot style."""
        # Determine number of hue categories
        n_categories = df[self.z].nunique() if self.z and self.z in df.columns else 0
        
        # Reset legend_properties if number of categories changed
        if self.legend_properties and n_categories > 0 and len(self.legend_properties) != n_categories:
            self.legend_properties = []
        
        if not self.legend_properties:
            markers = DEFAULT_MARKERS
            colors = DEFAULT_COLORS
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
            if self.plot_style == 'point':
                sns.pointplot(
                    data=df, x=self.x, y=y, hue=self.z, ax=self.ax,
                    linestyles='-' if self.join_for_point_plot else 'none',
                    marker=markers, palette=colors,
                    markeredgecolor='black', markeredgewidth=1,
                    err_kws={'linewidth': 1, 'color': 'black'},
                    capsize=0.02
                )
            elif self.plot_style == 'scatter':
                if self.z:
                    sns.scatterplot(
                        data=df, x=self.x, y=y, hue=self.z, ax=self.ax,
                        s=70, edgecolor='black', palette=colors
                    )
                else:
                    sns.scatterplot(
                        data=df, x=self.x, y=y, ax=self.ax,
                        s=70, edgecolor='black'
                    )
            elif self.plot_style == 'box':
                sns.boxplot(
                    data=df, x=self.x, y=y, hue=self.z,
                    ax=self.ax, palette=colors, width=0.4
                )
            elif self.plot_style == 'line':
                # Only pass palette if hue is provided
                line_kwargs = {'data': df, 'x': self.x, 'y': y, 'ax': self.ax}
                if self.z:
                    line_kwargs['hue'] = self.z
                    line_kwargs['palette'] = colors
                sns.lineplot(**line_kwargs)
            elif self.plot_style == 'bar':
                sns.barplot(
                    data=df, x=self.x, y=y, hue=self.z,
                    errorbar='sd' if self.show_bar_plot_error_bar else None,
                    err_kws={'linewidth': 1},
                    ax=self.ax, palette=colors
                )
            elif self.plot_style == 'trendline':
                sns.regplot(
                    data=df, x=self.x, y=y, ax=self.ax,
                    scatter=True, order=self.trendline_order
                )
                if self.show_trendline_eq:
                    self._annotate_trendline_eq(df)
            elif self.plot_style == 'wafer':
                self._plot_wafer(df)
            elif self.plot_style == '2Dmap':
                self._plot_2dmap(df, y)
            else:
                show_alert("Unsupported plot style")
    
    def _fix_box_bar_colors(self):
        """Fix box/bar patch colors to use exact DEFAULT_COLORS by mapping seaborn's RGBA."""
        patches = [p for p in self.ax.patches if hasattr(p, 'get_facecolor')]
        
        if not patches or not self.legend_properties:
            return
        
        # Build RGBA -> DEFAULT_COLOR mapping from legend
        rgba_to_color_map = {}
        for prop in self.legend_properties:
            if 'rgba' in prop:
                # Convert RGBA tuple to string for dictionary key
                rgba_key = tuple(prop['rgba']) if hasattr(prop['rgba'], '__iter__') else prop['rgba']
                rgba_to_color_map[rgba_key] = prop['color']
        
        # Update each patch's facecolor to exact DEFAULT_COLOR
        for patch in patches:
            current_rgba = patch.get_facecolor()
            rgba_key = tuple(current_rgba)
            
            # Find matching DEFAULT_COLOR from legend mapping
            if rgba_key in rgba_to_color_map:
                default_color = rgba_to_color_map[rgba_key]
            else:
                # Fallback: find closest RGBA in mapping
                default_color = rgba_to_default_color(current_rgba)
            
            patch.set_facecolor(default_color)
            patch.set_edgecolor('black')
            patch.set_linewidth(0.8)
    
    def _plot_2dmap(self, df, y):
        """Plot 2D heatmap."""
        x_col = self.x
        y_col = y if isinstance(self.y, list) else self.y
        z_col = self.z
        
        xmin = df[x_col].min()
        xmax = df[x_col].max()
        ymin = df[y_col].min()
        ymax = df[y_col].max()
        
        heatmap_data = df.pivot(index=y_col, columns=x_col, values=z_col)
        vmin = self.zmin if self.zmin else heatmap_data.min().min()
        vmax = self.zmax if self.zmax else heatmap_data.max().max()
        
        heatmap = self.ax.imshow(
            heatmap_data,
            aspect='equal',
            extent=[xmin, xmax, ymin, ymax],
            cmap=self.color_palette,
            origin='lower',
            vmin=vmin,
            vmax=vmax
        )
        
        # Remove existing colorbar if present to prevent accumulation
        if hasattr(self.ax, '_2dmap_colorbar') and self.ax._2dmap_colorbar is not None:
            try:
                self.ax._2dmap_colorbar.remove()
            except:
                pass
        
        # Add new colorbar and store reference
        self.ax._2dmap_colorbar = plt.colorbar(heatmap, orientation='vertical')
    
    def _set_legend(self):
        """Sets up and displays the legend for the plot."""
        handles, labels = self.ax.get_legend_handles_labels()
        
        if self.ax2:
            handles2, labels2 = self.ax2.get_legend_handles_labels()
            handles += handles2
            labels += labels2
            self.ax2.legend().remove()
        
        if self.ax3:
            handles3, labels3 = self.ax3.get_legend_handles_labels()
            handles += handles3
            labels += labels3
            self.ax3.legend().remove()
        
        if handles:
            legend_labels = []
            if self.legend_properties:
                try:
                    for idx, prop in enumerate(self.legend_properties):
                        legend_labels.append(prop['label'])
                        handles[idx].set_label(prop['label'])
                        # For box/bar plots, use set_facecolor to preserve edge color
                        if self.plot_style in ['box', 'bar']:
                            handles[idx].set_facecolor(prop['color'])
                            handles[idx].set_edgecolor('black')
                            handles[idx].set_linewidth(0.8)
                        else:
                            handles[idx].set_color(prop['color'])
                        if self.plot_style in ['point', 'scatter']:
                            handles[idx].set_marker(prop['marker'])
                except Exception:
                    self.legend_properties = []
                    legend_labels = labels
                    self.legend_properties = self.get_legend_properties()
            else:
                legend_labels = labels
                self.legend_properties = self.get_legend_properties()
            
            if self.legend_visible:
                if self.legend_outside:
                    legend = self.ax.legend(
                        handles, legend_labels,
                        loc='center left',
                        bbox_to_anchor=(1, 0.5)
                    )
                else:
                    legend = self.ax.legend(
                        handles, legend_labels, loc='best'
                    )
                legend.set_picker(True)  # Make legend clickable
                legend.set_draggable(True)  # Make legend draggable
                
                # Restore dragged position if saved
                if self.legend_bbox is not None and not self.legend_outside:
                    legend._loc = tuple(self.legend_bbox)
            else:
                self.ax.legend().remove()
    
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
        if not self.grid:
            self.ax.grid(False)
            return
        
        self.ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=1)
        
        if self.xlogscale or self.ylogscale:
            self.ax.minorticks_on()
            self.ax.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=1)
    
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
    
    def _annotate_trendline_eq(self, df):
        """Add the trendline equation in the plot."""
        x_data = df[self.x]
        y_data = df[self.y[0]]
        coefficients = np.polyfit(x_data, y_data, self.trendline_order)
        
        equation = 'y = '
        for i, coeff in enumerate(coefficients[::-1]):
            equation += (
                f'{coeff:.4f}x^{self.trendline_order - i} + '
                if i < self.trendline_order
                else f'{coeff:.4f}'
            )
        
        self.ax.annotate(
            equation,
            xy=(0.02, 0.95),
            xycoords='axes fraction',
            fontsize=10,
            color='blue'
        )
    
    def _plot_wafer(self, df):
        """Plot wafer plot by creating an object of WaferPlot Class."""
        vmin = self.zmin if self.zmin else None
        vmax = self.zmax if self.zmax else None
        
        wdf = WaferPlot()
        wdf.plot(
            self.ax,
            x=df[self.x],
            y=df[self.y[0]],
            z=df[self.z],
            cmap=self.color_palette,
            vmin=vmin,
            vmax=vmax,
            stats=self.wafer_stats,
            r=(self.wafer_size / 2)
        )
        
        # Annotate slot number if active filter
        if hasattr(self, "filters") and isinstance(self.filters, (list, dict)):
            filters_list = self.filters if isinstance(self.filters, list) else self.filters.get("filters", [])
            for f in filters_list:
                expr = f.get("expression", "")
                state = f.get("state", False)
                if state and "Slot ==" in expr:
                    try:
                        slot_num = expr.split("==")[1].strip()
                        self.ax.text(
                            0.02, 0.98, f"Slot {slot_num}",
                            transform=self.ax.transAxes,
                            fontsize=12, color='black',
                            fontweight='bold',
                            verticalalignment='top',
                            horizontalalignment='left'
                        )
                    except Exception:
                        pass
                    break
    
    def _set_limits(self):
        """Set the limits of axis."""
        if self.xmin and self.xmax:
            self.ax.set_xlim(float(self.xmin), float(self.xmax))
        if self.ymin and self.ymax:
            self.ax.set_ylim(float(self.ymin), float(self.ymax))
        if self.ax2 and self.y2min and self.y2max:
            self.ax2.set_ylim(float(self.y2min), float(self.y2max))
        if self.ax3 and self.y3min and self.y3max:
            self.ax3.set_ylim(float(self.y3min), float(self.y3max))
    
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
        
        if self.ax2 and self.y2 and self.ylogscale:
            y2_data = df[self.y2]
            if np.issubdtype(y2_data.dtype, np.number):
                self.ax2.set_yscale('log')
        
        if self.ax3 and self.y3 and self.ylogscale:
            y3_data = df[self.y3]
            if np.issubdtype(y3_data.dtype, np.number):
                self.ax3.set_yscale('log')
    
    def _set_labels(self):
        """Set titles and labels for axis and plot."""
        if self.plot_style == 'wafer':
            if self.plot_title:
                self.ax.set_title(self.plot_title)
            else:
                self.ax.set_title(self.z)
            self.ax.tick_params(axis='x', labelbottom=False)
        elif self.plot_style == '2Dmap':
            if self.plot_title:
                self.ax.set_title(self.plot_title)
            else:
                self.ax.set_title(self.z)
            if self.xlabel:
                self.ax.set_xlabel(self.xlabel)
            else:
                self.ax.set_xlabel(self.x)
            if self.ylabel:
                self.ax.set_ylabel(self.ylabel)
            else:
                self.ax.set_ylabel(self.y[0])
        else:
            self.ax.set_title(self.plot_title)
            if self.xlabel:
                self.ax.set_xlabel(self.xlabel)
            else:
                self.ax.set_xlabel(self.x)
            if self.ylabel:
                self.ax.set_ylabel(self.ylabel)
            else:
                self.ax.set_ylabel(self.y[0])
    
    def _plot_secondary_axis(self, df):
        """Plot data on secondary y-axis."""
        if self.ax2:
            self.ax2.remove()
            self.ax2 = None
        
        if hasattr(self, 'y2') and self.y2:
            self.ax2 = self.ax.twinx()
            
            if self.plot_style == 'line':
                sns.lineplot(data=df, x=self.x, y=self.y2, hue=self.z, ax=self.ax2, color='red')
            elif self.plot_style == 'point':
                sns.pointplot(
                    data=df, x=self.x, y=self.y2, hue=self.z, ax=self.ax2,
                    linestyles='-' if self.join_for_point_plot else 'none',
                    marker='s', color='red', markeredgecolor='black',
                    markeredgewidth=1, dodge=True,
                    err_kws={'linewidth': 1, 'color': 'black'},
                    capsize=0.02
                )
            elif self.plot_style == 'scatter':
                sns.scatterplot(
                    data=df, x=self.x, y=self.y2, hue=self.z, ax=self.ax2,
                    s=100, edgecolor='black', color='red'
                )
            else:
                self.ax2.remove()
                self.ax2 = None
            
            if self.ax2:
                self.ax2.set_ylabel(self.y2label, color='red')
                self.ax2.tick_params(axis='y', colors='red')
    
    def _plot_tertiary_axis(self, df):
        """Plot data on tertiary y-axis."""
        if self.ax3:
            self.ax3.remove()
            self.ax3 = None
        
        if hasattr(self, 'y3') and self.y3:
            self.ax3 = self.ax.twinx()
            self.ax3.spines["right"].set_position(("outward", 100))
            
            if self.plot_style == 'line':
                sns.lineplot(data=df, x=self.x, y=self.y3, hue=self.z, ax=self.ax3, color='green')
            elif self.plot_style == 'point':
                sns.pointplot(
                    data=df, x=self.x, y=self.y3, hue=self.z, ax=self.ax3,
                    linestyles='-' if self.join_for_point_plot else 'none',
                    marker='s', color='red', markeredgecolor='black',
                    markeredgewidth=1, dodge=True,
                    err_kws={'linewidth': 1, 'color': 'black'},
                    capsize=0.02
                )
            elif self.plot_style == 'scatter':
                sns.scatterplot(
                    data=df, x=self.x, y=self.y3, hue=self.z, ax=self.ax3,
                    s=100, edgecolor='black', color='green'
                )
            else:
                self.ax3.remove()
                self.ax3 = None
            
            if self.ax3:
                self.ax3.set_ylabel(self.y3label, color='green')
                self.ax3.tick_params(axis='y', colors='green')
    
    # ═══════════════════════════════════════════════════════════════════
    # Annotation Rendering
    # ═══════════════════════════════════════════════════════════════════
    
    def _render_annotations(self):
        """Render all annotations (lines and text) on the plot."""
        if not hasattr(self, 'annotations') or not self.annotations:
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
        bbox = ann.get('bbox')
        if bbox is None:
            # Default: no background/frame
            bbox_props = None
        else:
            # Use bbox from annotation
            bbox_props = bbox
        
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
        from .customize_graph_dialog import AnnotationLineEditDialog, AnnotationTextEditDialog
        from PySide6.QtWidgets import QDialog
        
        # Open appropriate edit dialog based on type
        if annotation['type'] in ['vline', 'hline']:
            dialog = AnnotationLineEditDialog(annotation, None)
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
            dialog = AnnotationTextEditDialog(annotation, None)
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
        
        # Apply X-axis break  
        if self.axis_breaks.get('x'):
            break_start = self.axis_breaks['x']['start']
            break_end = self.axis_breaks['x']['end']
            
            # Get current limits
            x_min, x_max = self.ax.get_xlim()
            y_min, y_max = self.ax.get_ylim()
            
            # Calculate new compressed range (remove break range from display)
            total_range = x_max - x_min
            break_range = break_end - break_start
            new_range = total_range - break_range
            
            # Don't apply if break is outside data range
            if break_start < x_min or break_end > x_max:
                return
            
            # Adjust x-axis to skip the break range
            # Map data points: values before break stay same, values after break shift left
            # BUT leave a small visual gap for the break markers
            # Use fixed pixel gap for consistent appearance
            gap_pixels = 3  # 5 pixels
            # Convert pixels to data coordinates
            bbox = self.ax.get_window_extent().transformed(self.figure.dpi_scale_trans.inverted())
            gap_size = gap_pixels / bbox.width * (x_max - x_min) / self.figure.get_size_inches()[0]
            
            for line in self.ax.get_lines():
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                
                # Convert to numpy arrays if needed
                import numpy as np
                xdata = np.asarray(xdata)
                ydata = np.asarray(ydata)
                
                # Shift x values after break (compress break range, keep small gap)
                xdata_new = xdata.copy()
                mask = xdata >= break_end
                xdata_new[mask] = xdata[mask] - break_range + gap_size
                
                # Remove points in break range
                keep_mask = (xdata < break_start) | (xdata >= break_end)
                line.set_data(xdata_new[keep_mask], ydata[keep_mask])
            
            # Adjust axis limits (compress but keep gap)
            self.ax.set_xlim(x_min, x_max - break_range + gap_size)
            
            # Add zigzag break markers in the gap
            break_x = break_start + gap_size / 2
            gap_height = (y_max - y_min) * 0.05
            
            # Left zigzag
            self.ax.plot([break_start, break_start + gap_size*0.3], 
                        [y_min, y_min + gap_height], 
                        'k-', linewidth=2, clip_on=False, zorder=100)
            self.ax.plot([break_start + gap_size*0.3, break_start + gap_size*0.5], 
                        [y_min + gap_height, y_min], 
                        'k-', linewidth=2, clip_on=False, zorder=100)
            
            # Right zigzag  
            self.ax.plot([break_start + gap_size*0.5, break_start + gap_size*0.7], 
                        [y_max, y_max - gap_height], 
                        'k-', linewidth=2, clip_on=False, zorder=100)
            self.ax.plot([break_start + gap_size*0.7, break_start + gap_size], 
                        [y_max - gap_height, y_max], 
                        'k-', linewidth=2, clip_on=False, zorder=100)
        
        # Apply Y-axis break
        if self.axis_breaks.get('y'):
            break_start = self.axis_breaks['y']['start']
            break_end = self.axis_breaks['y']['end']
            
            # Get current limits
            x_min, x_max = self.ax.get_xlim()
            y_min, y_max = self.ax.get_ylim()
            
            # Calculate new compressed range
            total_range = y_max - y_min
            break_range = break_end - break_start  
            new_range = total_range - break_range
            
            # Don't apply if break is outside data range
            if break_start < y_min or break_end > y_max:
                return
            
            # Adjust y-axis to skip the break range
            # Leave a small visual gap for the break markers
            # Use fixed pixel gap for consistent appearance
            gap_pixels = 3  # 5 pixels
            # Convert pixels to data coordinates
            bbox = self.ax.get_window_extent().transformed(self.figure.dpi_scale_trans.inverted())
            gap_size = gap_pixels / bbox.height * (y_max - y_min) / self.figure.get_size_inches()[1]
            
            for line in self.ax.get_lines():
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                
                # Convert to numpy arrays if needed
                import numpy as np
                xdata = np.asarray(xdata)
                ydata = np.asarray(ydata)
                
                # Shift y values after break (compress break range, keep small gap)
                ydata_new = ydata.copy()
                mask = ydata >= break_end
                ydata_new[mask] = ydata[mask] - break_range + gap_size
                
                # Remove points in break range
                keep_mask = (ydata < break_start) | (ydata >= break_end)
                line.set_data(xdata[keep_mask], ydata_new[keep_mask])
            
            # Adjust axis limits (compress but keep gap)
            self.ax.set_ylim(y_min, y_max - break_range + gap_size)
            
            # Add zigzag break markers in the gap
            break_y = break_start + gap_size / 2
            gap_width = (x_max - x_min) * 0.05
            
            # Bottom zigzag
            self.ax.plot([x_min, x_min + gap_width], 
                        [break_start, break_start + gap_size*0.3], 
                        'k-', linewidth=2, clip_on=False, zorder=100)
            self.ax.plot([x_min + gap_width, x_min], 
                        [break_start + gap_size*0.3, break_start + gap_size*0.5], 
                        'k-', linewidth=2, clip_on=False, zorder=100)
            
            # Top zigzag
            self.ax.plot([x_max, x_max - gap_width], 
                        [break_start + gap_size*0.5, break_start + gap_size*0.7], 
                        'k-', linewidth=2, clip_on=False, zorder=100)
            self.ax.plot([x_max - gap_width, x_max], 
                        [break_start + gap_size*0.7, break_start + gap_size], 
                        'k-', linewidth=2, clip_on=False, zorder=100)
    
    def _show_customize_dialog(self):
        """Show customize dialog for this graph."""
        # Check if dialog already exists
        if not hasattr(self, '_customize_dialog') or self._customize_dialog is None:
            self._customize_dialog = CustomizeGraphDialog(self, self.graph_id, parent=self)
        
        # Show non-modal dialog
        self._customize_dialog.show()
        self._customize_dialog.raise_()
        self._customize_dialog.activateWindow()


class WaferPlot:
    """Class to plot wafer map."""
    
    def __init__(self, inter_method='linear'):
        self.inter_method = inter_method
    
    def plot(self, ax, x, y, z, cmap="jet", r=100, vmax=None, vmin=None, stats=True):
        """Plot wafer map with interpolated data."""
        xi, yi = np.meshgrid(np.linspace(-r, r, 600), np.linspace(-r, r, 600))
        zi = griddata((x, y), z, (xi, yi), method=self.inter_method)
        
        im = ax.imshow(
            zi,
            extent=[-r - 1, r + 1, -r - 0.5, r + 0.5],
            origin='lower',
            cmap=cmap,
            interpolation='nearest'
        )
        
        ax.scatter(x, y, facecolors='none', edgecolors='black', s=20)
        
        wafer_circle = patches.Circle((0, 0), radius=r, fill=False, color='black', linewidth=1)
        ax.add_patch(wafer_circle)
        
        ax.set_ylabel("Wafer size (mm)")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x', which='both', bottom=False, top=False)
        ax.tick_params(axis='y', which='both', right=False, left=True)
        ax.set_xticklabels([])
        
        if vmax is not None and vmin is not None:
            im.set_clim(vmin, vmax)
        
        # Remove existing colorbar if present to prevent accumulation
        if hasattr(ax, '_wafer_colorbar') and ax._wafer_colorbar is not None:
            try:
                ax._wafer_colorbar.remove()
            except:
                pass
        
        # Add new colorbar and store reference
        ax._wafer_colorbar = plt.colorbar(im, ax=ax)
        
        if stats:
            self.stats(z, ax)
    
    def stats(self, z, ax):
        """Calculate and display statistical values in the wafer plot."""
        mean_value = z.mean()
        max_value = z.max()
        min_value = z.min()
        sigma_value = z.std()
        three_sigma_value = 3 * sigma_value
        
        ax.text(0.05, -0.1, f"3\u03C3: {three_sigma_value:.2f}",
                transform=ax.transAxes, fontsize=10, verticalalignment='bottom')
        ax.text(0.99, -0.1, f"Max: {max_value:.2f}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right')
        ax.text(0.99, -0.05, f"Min: {min_value:.2f}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right')
        ax.text(0.05, -0.05, f"Mean: {mean_value:.2f}",
                transform=ax.transAxes, fontsize=10, verticalalignment='bottom')


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
