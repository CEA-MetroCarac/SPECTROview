"""Matplotlib graph visualization widget for MVVM pattern."""

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from PySide6.QtWidgets import QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QDialog
from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QIcon

from spectroview import DEFAULT_COLORS, DEFAULT_MARKERS, ICON_DIR, PLOT_POLICY_LIGHT
from spectroview.view.components.customize_graph_dialog import (
    EditLineDialog, EditTextDialog
)
from spectroview.viewmodel.utils import rgba_to_default_color, show_alert, copy_fig_to_clb


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
        self.show_bar_plot_error_bar = True
        self.join_for_point_plot = False
        self.dodge_point_plot = True
        self.scatter_size = 70  # Marker size for scatter plots
        self.scatter_edgecolor = 'black'  # Edge color for scatter plot markers
        # Histogram-specific
        self.hist_bins = 20
        self.hist_kde = False
        self.hist_step = False
        
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
    
    def plot(self, df):
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
                    if self.plot_style in ['point', 'scatter', 'line', 'trendline']:
                        if hasattr(handle, 'get_markerfacecolor'):
                            rgba_color = handle.get_markerfacecolor()
                            marker = handle.get_marker() if hasattr(handle, 'get_marker') else 'o'
                        elif hasattr(handle, 'get_facecolor'):
                            facecolors = handle.get_facecolor()
                            rgba_color = facecolors[0] if len(facecolors) > 0 else 'blue'
                            marker = 'o'
                        else:
                            rgba_color = 'blue'
                            marker = 'o'
                    elif self.plot_style in ['box', 'bar', 'histogram']:
                        rgba_color = handle.get_facecolor()
                        marker = 'o'
                    else:
                        rgba_color = 'blue'
                        marker = 'o'
                    
                    # Convert numpy arrays to lists for JSON serialization
                    if hasattr(rgba_color, 'tolist'):
                        rgba_color = rgba_color.tolist()
                    elif isinstance(rgba_color, tuple):
                        rgba_color = list(rgba_color)
                    
                    # Map to closest DEFAULT_COLOR
                    color = rgba_to_default_color(rgba_color)
                    
                    legend_properties.append({
                        'label': label,
                        'marker': marker,
                        'color': color,
                        'rgba': rgba_color  # Keep original RGBA for mapping
                    })
            elif self.plot_style not in ['2Dmap', 'wafer']:
                # No legend on plot, but we can still customize the main color
                color = DEFAULT_COLORS[0] if DEFAULT_COLORS else 'steelblue'
                import matplotlib.colors as mcolors
                rgba_color = mcolors.to_rgba(color)
                
                if hasattr(self, 'legend_properties') and self.legend_properties:
                    color = self.legend_properties[0].get('color', color)
                    rgba_color = self.legend_properties[0].get('rgba', rgba_color)
                
                legend_properties.append({
                    'label': 'All data',
                    'marker': 'o',
                    'color': color,
                    'rgba': rgba_color
                })
        
        self.legend_properties = legend_properties
        
        # Fix box/bar patch colors to match DEFAULT_COLORS
        if self.plot_style in ['box', 'bar', 'histogram'] and self.legend_properties:
            self._fix_box_bar_colors()
        
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
                self._plot_point(df, y, colors, markers, c)
            elif self.plot_style == 'scatter':
                self._plot_scatter(df, y, colors, c)
            elif self.plot_style == 'box':
                self._plot_box(df, y, colors, c)
            elif self.plot_style == 'line':
                self._plot_line(df, y, colors, c)
            elif self.plot_style == 'bar':
                self._plot_bar(df, y, colors, c)
            elif self.plot_style == 'trendline':
                self._plot_trendline(df, y, colors, c)
            elif self.plot_style == 'histogram':
                self._plot_histogram(df, colors)
            elif self.plot_style == 'wafer':
                self._plot_wafer(df)
            elif self.plot_style == '2Dmap':
                self._plot_2dmap(df, y)
            else:
                show_alert("Unsupported plot style")

    def _plot_point(self, df, y, colors, markers, c):
        point_kwargs = {
            'data': df, 'x': self.x, 'y': y, 'ax': self.ax,
            'linestyles': '-' if self.join_for_point_plot else 'none',
            'markeredgecolor': getattr(self, 'scatter_edgecolor', 'black'),
            'markeredgewidth': 1,
            'err_kws': {'linewidth': 1, 'zorder': 1},
            'capsize': 0.05,
            'dodge': True if getattr(self, 'dodge_point_plot', False) else False,
            'markersize': np.sqrt(self.scatter_size) if hasattr(self, 'scatter_size') else 7
        }
        if self.z:
            point_kwargs['hue'] = self.z
            point_kwargs['palette'] = colors
            point_kwargs['marker'] = markers
        else:
            point_kwargs['color'] = c
        sns.pointplot(**point_kwargs)

    def _plot_scatter(self, df, y, colors, c):
        if self.z:
            sns.scatterplot(
                data=df, x=self.x, y=y, hue=self.z, ax=self.ax,
                s=self.scatter_size, edgecolor=self.scatter_edgecolor,
                palette=colors
            )
        else:
            sns.scatterplot(
                data=df, x=self.x, y=y, ax=self.ax,
                s=self.scatter_size, edgecolor=self.scatter_edgecolor,
                color=c
            )

    def _plot_box(self, df, y, colors, c):
        box_kwargs = {'data': df, 'x': self.x, 'y': y, 'ax': self.ax, 'width': 0.4}
        if self.z:
            box_kwargs['hue'] = self.z
            box_kwargs['palette'] = colors
        else:
            box_kwargs['color'] = c
        sns.boxplot(**box_kwargs)

    def _plot_line(self, df, y, colors, c):
        line_kwargs = {'data': df, 'x': self.x, 'y': y, 'ax': self.ax}
        if self.z:
            line_kwargs['hue'] = self.z
            line_kwargs['palette'] = colors
        else:
            line_kwargs['color'] = c
        sns.lineplot(**line_kwargs)

    def _plot_bar(self, df, y, colors, c):
        bar_kwargs = {
            'data': df, 'x': self.x, 'y': y, 'ax': self.ax,
            'errorbar': 'sd' if self.show_bar_plot_error_bar else None,
            'err_kws': {'linewidth': 1},
            'capsize': 0.05
        }
        if self.z:
            bar_kwargs['hue'] = self.z
            bar_kwargs['palette'] = colors
        else:
            bar_kwargs['color'] = c
        sns.barplot(**bar_kwargs)

    def _plot_trendline(self, df, y, colors, c):
        self.trendline_equations = []  # reset before recomputing
        anchor = getattr(self, 'trendline_anchor_enabled', False)
        
        if self.z and self.z in df.columns:
            # Hue support: one regplot per category
            categories = df[self.z].unique()
            while len(colors) < len(categories):
                colors.append(DEFAULT_COLORS[len(colors) % len(DEFAULT_COLORS)])
            for idx, cat in enumerate(categories):
                subset = df[df[self.z] == cat]
                color = colors[idx]
                x_fit, y_fit, coeffs = self._fit_trendline(subset)
                
                # Plot scatter manually to control legend and color exactly
                self.ax.scatter(
                    subset[self.x], subset[y],
                    color=color, s=self.scatter_size,
                    edgecolors=self.scatter_edgecolor, linewidths=0.5,
                    label=str(cat), zorder=3
                )
                
                if anchor:
                    # Plot custom anchored line
                    self.ax.plot(x_fit, y_fit, color=color, linewidth=2)
                else:
                    # Let seaborn plot standard line + confidence intervals
                    sns.regplot(
                        data=subset, x=self.x, y=y, ax=self.ax,
                        scatter=False, order=self.trendline_order,
                        color=color
                    )
                    
                eq_str, r2 = self._build_equation_str(coeffs, subset)
                self.trendline_equations.append({
                    'label': str(cat), 'equation': eq_str, 'r2': f"{r2:.4f}"
                })
        else:
            # No hue — single fit
            x_fit, y_fit, coeffs = self._fit_trendline(df)
            
            # Plot scatter manually for consistency
            self.ax.scatter(
                df[self.x], df[y],
                color=c, s=self.scatter_size,
                edgecolors=self.scatter_edgecolor, linewidths=0.5,
                label='All data', zorder=3
            )
            
            if anchor:
                # Plot custom anchored line
                self.ax.plot(x_fit, y_fit, color=c, linewidth=2)
            else:
                # Let seaborn plot standard line + confidence intervals
                sns.regplot(
                    data=df, x=self.x, y=y, ax=self.ax,
                    scatter=False, order=self.trendline_order,
                    color=c
                )
                
            eq_str, r2 = self._build_equation_str(coeffs, df)
            self.trendline_equations.append({
                'label': 'All data', 'equation': eq_str, 'r2': f"{r2:.4f}"
            })

    def _plot_histogram(self, df, colors):
        hist_kwargs = {
            'data': df,
            'x': self.x,
            'ax': self.ax,
            'bins': self.hist_bins,
            'kde': self.hist_kde,
            'element': 'step' if self.hist_step else 'bars',
            'fill': not self.hist_step,
            'stat': 'count',
        }
        if self.z and self.z in df.columns:
            hist_kwargs['hue'] = self.z
            hist_kwargs['palette'] = colors
        else:
            hist_kwargs['color'] = colors[0] if colors else 'steelblue'
        sns.histplot(**hist_kwargs)

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
                        elif self.plot_style in ['scatter', 'trendline']:
                            edge_c = getattr(self, 'scatter_edgecolor', 'black')
                            if not edge_c or not isinstance(edge_c, str) or edge_c.strip() in ("", "None", "none", "null"):
                                edge_c = 'black'
                            if hasattr(handles[idx], 'set_facecolor'):
                                handles[idx].set_facecolor(prop['color'])
                            if hasattr(handles[idx], 'set_edgecolor'):
                                handles[idx].set_edgecolor(edge_c)
                            if hasattr(handles[idx], 'set_linewidth'):
                                handles[idx].set_linewidth(0.5)
                        else:
                            handles[idx].set_color(prop['color'])
                            if self.plot_style == 'point' and hasattr(handles[idx], 'set_markeredgecolor'):
                                edge_c = getattr(self, 'scatter_edgecolor', 'black')
                                if not edge_c or not isinstance(edge_c, str) or edge_c.strip() in ("", "None", "none", "null"):
                                    edge_c = 'black'
                                handles[idx].set_markeredgecolor(edge_c)
                        
                        if self.plot_style in ['point', 'scatter', 'trendline']:
                            if hasattr(handles[idx], 'set_marker'):
                                handles[idx].set_marker(prop['marker'])
                except Exception:
                    self.legend_properties = []
                    legend_labels = labels
                    self.legend_properties = self.get_legend_properties()
            else:
                legend_labels = labels
                self.legend_properties = self.get_legend_properties()
            
            if self.legend_visible:
                legend = self.ax.legend(handles, legend_labels, loc='best')
                legend.set_picker(True)  # Make legend clickable
                legend.set_draggable(True)  # Make legend draggable
                
                # Restore dragged position if saved
                if self.legend_bbox is not None:
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
        is_wafer = (self.plot_style == 'wafer')
        
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
    
    def _fit_trendline(self, df):
        """Fit polynomial trendline with optional anchor constraint.
        
        Returns (x_fit, y_fit, coefficients).
        """
        try:
            x_data = df[self.x].dropna().values.astype(float)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Could not convert values in X column '{self.x}' to numeric values. "
                "Trendline fitting requires numeric columns."
            ) from e
        
        try:
            y_data = df[self.y[0]].dropna().values.astype(float)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Could not convert values in Y column '{self.y[0]}' to numeric values. "
                "Trendline fitting requires numeric columns."
            ) from e
        
        # Align lengths after dropna
        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_data = x_data[mask]
        y_data = y_data[mask]
        
        if self.trendline_anchor_enabled:
            # Determine anchor coordinates
            if self.trendline_anchor_origin:
                ax_val, ay_val = 0.0, 0.0
            else:
                ax_val = float(self.trendline_anchor_x)
                ay_val = float(self.trendline_anchor_y)
            
            # Shift data so anchor becomes origin, then fit without intercept
            x_shifted = x_data - ax_val
            y_shifted = y_data - ay_val
            
            if self.trendline_order == 1:
                # Force through shifted origin: y = m*x
                m = np.dot(x_shifted, y_shifted) / np.dot(x_shifted, x_shifted)
                coeffs = np.array([m, 0.0])  # slope, zero intercept (shifted)
                # Build coefficients back in original space: y = m*(x-ax)+ay
                # Represent as standard polyfit form shifted back:
                x_fit = np.linspace(x_data.min(), x_data.max(), 200)
                y_fit = m * (x_fit - ax_val) + ay_val
            else:
                # Higher order: fit shifted data with zero constant term (no intercept)
                # Use least squares with Vandermonde matrix excluding constant column
                A = np.column_stack([x_shifted**i for i in range(self.trendline_order, 0, -1)])
                result = np.linalg.lstsq(A, y_shifted, rcond=None)
                poly_coeffs = result[0]
                x_fit = np.linspace(x_data.min(), x_data.max(), 200)
                x_fit_shifted = x_fit - ax_val
                y_fit = sum(poly_coeffs[i] * x_fit_shifted**(self.trendline_order - i)
                            for i in range(self.trendline_order)) + ay_val
                coeffs = np.append(poly_coeffs, 0.0)  # zero constant (shifted origin)
        else:
            # Standard unconstrained polynomial fit
            coeffs = np.polyfit(x_data, y_data, self.trendline_order)
            x_fit = np.linspace(x_data.min(), x_data.max(), 200)
            y_fit = np.polyval(coeffs, x_fit)
        
        return x_fit, y_fit, coeffs

    def _build_equation_str(self, coeffs, df):
        """Build human-readable equation string and compute R²."""
        x_data = df[self.x].dropna().values.astype(float)
        y_data = df[self.y[0]].dropna().values.astype(float)
        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_data = x_data[mask]
        y_data = y_data[mask]
        
        order = self.trendline_order
        
        # Build equation string from coefficients (highest power first)
        _sup = {2: '\u00b2', 3: '\u00b3', 4: '\u2074', 5: '\u2075', 6: '\u2076', 7: '\u2077', 8: '\u2078', 9: '\u2079'}
        terms = []
        for i, c in enumerate(coeffs):
            power = order - i
            if power == 0:
                terms.append(f"{c:+.4f}")
            elif power == 1:
                terms.append(f"{c:+.4f}x")
            else:
                sup = _sup.get(power, f"^{power}")
                terms.append(f"{c:+.4f}x{sup}")
        eq_str = "y = " + " ".join(terms).lstrip("+")
        
        # Compute R²
        y_pred = np.polyval(coeffs, x_data)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        
        return eq_str, r2

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
        if self.ax_x2 and self.x2min and self.x2max:
            self.ax_x2.set_xlim(float(self.x2min), float(self.x2max))
    
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
        
        if self.y2:
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
                    s=self.scatter_size, edgecolor=self.scatter_edgecolor,
                    color='red'
                )
            else:
                self.ax2.remove()
                self.ax2 = None
            
            if self.ax2:
                self.ax2.set_ylabel(self.y2label or self.y2, color='red')
                self.ax2.tick_params(axis='y', colors='red')
    
    def _plot_tertiary_axis(self, df):
        """Plot data on tertiary y-axis."""
        if self.ax3:
            self.ax3.remove()
            self.ax3 = None
        
        if self.y3:
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
                    s=self.scatter_size, edgecolor=self.scatter_edgecolor,
                    color='green'
                )
            else:
                self.ax3.remove()
                self.ax3 = None
            
            if self.ax3:
                self.ax3.set_ylabel(self.y3label or self.y3, color='green')
                self.ax3.tick_params(axis='y', colors='green')
    
    def _plot_secondary_x_axis(self, df):
        """Plot data on secondary x-axis (top)."""
        if self.ax_x2:
            self.ax_x2.remove()
            self.ax_x2 = None
        
        if self.x2 and self.x2 in df.columns:
            self.ax_x2 = self.ax.twiny()
            
            if self.plot_style == 'line':
                sns.lineplot(
                    data=df, x=self.x2, y=self.y[0], ax=self.ax_x2,
                    color='purple'
                )
            elif self.plot_style == 'point':
                sns.pointplot(
                    data=df, x=self.x2, y=self.y[0], ax=self.ax_x2,
                    linestyles='-' if self.join_for_point_plot else 'none',
                    marker='D', color='purple', markeredgecolor='black',
                    markeredgewidth=1,
                    err_kws={'linewidth': 1, 'color': 'black'},
                    capsize=0.02
                )
            elif self.plot_style == 'scatter':
                sns.scatterplot(
                    data=df, x=self.x2, y=self.y[0], ax=self.ax_x2,
                    s=self.scatter_size, edgecolor=self.scatter_edgecolor,
                    color='purple'
                )
            else:
                self.ax_x2.remove()
                self.ax_x2 = None
            
            if self.ax_x2:
                self.ax_x2.set_xlabel(
                    self.x2label or self.x2, color='purple'
                )
                self.ax_x2.tick_params(axis='x', colors='purple')
    
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
        


class WaferPlot:
    """Class to plot wafer map."""
    
    def __init__(self, inter_method='linear'):
        self.inter_method = inter_method
    
    def plot(self, ax, x, y, z, cmap="jet", r=100, vmax=None, vmin=None, stats=True):
        """Plot wafer map with interpolated data."""
        xi, yi = np.meshgrid(np.linspace(-r, r, 600), np.linspace(-r, r, 600))
        from scipy.interpolate import griddata
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


