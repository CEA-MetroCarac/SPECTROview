"""Matplotlib graph visualization widget for MVVM pattern."""

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        self.show_bar_plot_error_bar = False
        self.join_for_point_plot = False
        self.dodge_point_plot = True
        self.dodge_scatter_plot = False
        self.scatter_size = 70  # Marker size for scatter plots
        self.scatter_edgecolor = 'black'  # Edge color for scatter plot markers
        self.x_as_numeric = False  # Treat X-axis as numerical (for point/box/bar)
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
        
        original_set_style = self.toolbar.setStyleSheet
        def custom_set_style(css):
            original_set_style("background: transparent; border: none;")
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
            
        legend_properties = []
        if self.ax:
            legend = self.ax.get_legend()
            if legend:
                legend_texts = legend.get_texts()
                for idx, text in enumerate(legend_texts):
                    label = text.get_text()
                    color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
                    marker = DEFAULT_MARKERS[idx % len(DEFAULT_MARKERS)]
                    
                    import matplotlib.colors as mcolors
                    rgba_color = mcolors.to_rgba(color)
                    
                    legend_properties.append({
                        'label': label,
                        'marker': marker,
                        'color': color,
                        'rgba': list(rgba_color)
                    })
            elif self.plot_style not in ['2Dmap', 'wafer']:
                color = DEFAULT_COLORS[0] if DEFAULT_COLORS else 'steelblue'
                import matplotlib.colors as mcolors
                rgba_color = mcolors.to_rgba(color)
                
                legend_properties.append({
                    'label': 'All data',
                    'marker': 'o',
                    'color': color,
                    'rgba': list(rgba_color)
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

    def _prepare_plot_data(self, df, y):
        """Prepare dataframe and X positions for plotting."""
        cols = []
        if self.x not in cols: cols.append(self.x)
        if y not in cols: cols.append(y)
        if self.z and self.z in df.columns and self.z not in cols:
            cols.append(self.z)
            
        plot_df = df[cols].copy()
        
        treat_as_numeric = getattr(self, 'x_as_numeric', False)
        # Auto-detect numeric if plot style expects it by default
        if not treat_as_numeric and self.plot_style in ['scatter', 'line', 'trendline', 'histogram']:
            num_vals = pd.to_numeric(plot_df[self.x], errors='coerce')
            if num_vals.notna().sum() > 0.5 * len(num_vals):
                treat_as_numeric = True
        
        if treat_as_numeric:
            plot_df[self.x] = pd.to_numeric(plot_df[self.x], errors='coerce')
            plot_df = plot_df.dropna(subset=[self.x, y])
            x_unique = sorted(plot_df[self.x].unique())
            x_positions = {v: v for v in x_unique}
        else:
            plot_df = plot_df.dropna(subset=[self.x, y])
            # Preserve original order for categorical
            x_unique = list(plot_df[self.x].unique())
            x_positions = {v: i for i, v in enumerate(x_unique)}
            
        return plot_df, x_unique, x_positions, treat_as_numeric

    def _plot_point(self, df, y, colors, markers, c):
        plot_df, x_unique, x_positions, is_numeric = self._prepare_plot_data(df, y)
        edge_c = getattr(self, 'scatter_edgecolor', 'black')
        ms = np.sqrt(self.scatter_size) if hasattr(self, 'scatter_size') else 7
        join = getattr(self, 'join_for_point_plot', False)
        
        if self.z and self.z in plot_df.columns:
            categories = plot_df[self.z].unique()
            n_hue = len(categories)
            dodge = getattr(self, 'dodge_point_plot', True) and not is_numeric
            if dodge and n_hue > 1:
                offsets = np.linspace(-0.2, 0.2, n_hue)
            else:
                offsets = np.zeros(n_hue)

            for idx, cat in enumerate(categories):
                subset = plot_df[plot_df[self.z] == cat]
                if subset.empty: continue
                grouped = subset.groupby(self.x)[y]
                
                x_cat_vals = list(grouped.groups.keys())
                x_vals = np.array([x_positions[val] + offsets[idx] for val in x_cat_vals], dtype=float)
                
                means = grouped.mean().values
                cis = grouped.sem().values * 1.96  # 95% CI
                cis = np.nan_to_num(cis)
                
                color = colors[idx % len(colors)]
                marker = markers[idx % len(markers)] if markers else 'o'
                
                self.ax.errorbar(
                    x_vals, means, yerr=cis,
                    fmt=marker, color=color, markersize=ms,
                    markeredgecolor=edge_c, markeredgewidth=1,
                    capsize=3, elinewidth=1,
                    linestyle='-' if join else 'none'
                )
                self.ax.plot([], [], marker=marker, color=color, markersize=ms,
                             markeredgecolor=edge_c, markeredgewidth=1,
                             linestyle='-' if join else 'none', label=str(cat))
        else:
            grouped = plot_df.groupby(self.x)[y]
            x_cat_vals = list(grouped.groups.keys())
            x_vals = np.array([x_positions[val] for val in x_cat_vals], dtype=float)
            means = grouped.mean().values
            cis = grouped.sem().values * 1.96
            cis = np.nan_to_num(cis)
            
            self.ax.errorbar(
                x_vals, means, yerr=cis,
                fmt='o', color=c, markersize=ms,
                markeredgecolor=edge_c, markeredgewidth=1,
                capsize=3, elinewidth=1,
                linestyle='-' if join else 'none'
            )
            
        if not is_numeric:
            self.ax.set_xticks(list(x_positions.values()))
            self.ax.set_xticklabels([str(v) for v in x_unique])

    def _plot_scatter(self, df, y, colors, c):
        plot_df, x_unique, x_positions, is_numeric = self._prepare_plot_data(df, y)
        edge_c = getattr(self, 'scatter_edgecolor', 'black')
        dodge = getattr(self, 'dodge_scatter_plot', False) and not is_numeric
        
        if self.z and self.z in plot_df.columns:
            categories = plot_df[self.z].unique()
            n_hue = len(categories)
            if dodge and n_hue > 1:
                offsets = np.linspace(-0.3, 0.3, n_hue)
            else:
                offsets = np.zeros(n_hue)

            for idx, cat in enumerate(categories):
                subset = plot_df[plot_df[self.z] == cat]
                if subset.empty: continue
                color = colors[idx % len(colors)]
                
                x_vals = np.array([x_positions[val] + offsets[idx] for val in subset[self.x]], dtype=float)
                
                self.ax.scatter(
                    x_vals, subset[y].values,
                    color=color, s=self.scatter_size,
                    edgecolors=edge_c, linewidths=0.5, label=str(cat)
                )
        else:
            x_vals = np.array([x_positions[val] for val in plot_df[self.x]], dtype=float)
            self.ax.scatter(
                x_vals, plot_df[y].values,
                color=c, s=self.scatter_size,
                edgecolors=edge_c, linewidths=0.5
            )
            
        if not is_numeric:
            self.ax.set_xticks(list(x_positions.values()))
            self.ax.set_xticklabels([str(v) for v in x_unique])

    def _plot_box(self, df, y, colors, c):
        plot_df, x_unique, x_positions, is_numeric = self._prepare_plot_data(df, y)
        
        if len(x_unique) < 2:
            box_width = 0.4
        else:
            if is_numeric:
                min_gap = min(x_positions[x_unique[i+1]] - x_positions[x_unique[i]] for i in range(len(x_unique) - 1))
            else:
                min_gap = 1.0
            box_width = min_gap * 0.6

        if self.z and self.z in plot_df.columns:
            hue_cats = plot_df[self.z].unique()
            n_hue = len(hue_cats)
            sub_width = box_width / n_hue
            offsets = np.linspace(-(box_width - sub_width) / 2,
                                  (box_width - sub_width) / 2, n_hue)
            legend_handles = []
            for h_idx, cat in enumerate(hue_cats):
                subset = plot_df[plot_df[self.z] == cat]
                color = colors[h_idx % len(colors)]
                
                data_groups = []
                positions = []
                for xv in x_unique:
                    vals = subset[subset[self.x] == xv][y].values
                    if len(vals) > 0:
                        data_groups.append(vals)
                        positions.append(x_positions[xv] + offsets[h_idx])
                
                if data_groups:
                    bp = self.ax.boxplot(
                        data_groups, positions=positions, widths=sub_width * 0.9,
                        patch_artist=True, manage_ticks=False
                    )
                    for patch in bp['boxes']:
                        patch.set_facecolor(color)
                        patch.set_edgecolor('black')
                        patch.set_linewidth(0.8)
                    for element in ['whiskers', 'caps', 'medians', 'fliers']:
                        for line in bp.get(element, []):
                            line.set_color('black')
                
                p = patches.Rectangle((0,0), 0, 0, facecolor=color, edgecolor='black', label=str(cat))
                self.ax.add_patch(p)
        else:
            data_groups = []
            positions = []
            for xv in x_unique:
                vals = plot_df[plot_df[self.x] == xv][y].values
                if len(vals) > 0:
                    data_groups.append(vals)
                    positions.append(x_positions[xv])
            
            if data_groups:
                bp = self.ax.boxplot(
                    data_groups, positions=positions, widths=box_width,
                    patch_artist=True, manage_ticks=False
                )
                for patch in bp['boxes']:
                    patch.set_facecolor(c)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(0.8)
                for element in ['whiskers', 'caps', 'medians', 'fliers']:
                    for line in bp.get(element, []):
                        line.set_color('black')

        if is_numeric:
            self.ax.set_xticks([x_positions[xv] for xv in x_unique])
            self.ax.set_xticklabels([str(xv) for xv in x_unique])
        else:
            self.ax.set_xticks(list(x_positions.values()))
            self.ax.set_xticklabels([str(v) for v in x_unique])

    def _plot_line(self, df, y, colors, c):
        plot_df, x_unique, x_positions, is_numeric = self._prepare_plot_data(df, y)
        
        if self.z and self.z in plot_df.columns:
            categories = plot_df[self.z].unique()
            for idx, cat in enumerate(categories):
                subset = plot_df[plot_df[self.z] == cat]
                if subset.empty: continue
                grouped = subset.groupby(self.x)[y]
                
                x_cat_vals = list(grouped.groups.keys())
                x_vals = np.array([x_positions[val] for val in x_cat_vals], dtype=float)
                
                means = grouped.mean().values
                cis = grouped.sem().values * 1.96  # 95% CI
                cis = np.nan_to_num(cis)
                
                color = colors[idx % len(colors)]
                
                self.ax.plot(x_vals, means, color=color, label=str(cat))
                self.ax.fill_between(x_vals, means - cis, means + cis, color=color, alpha=0.2)
        else:
            grouped = plot_df.groupby(self.x)[y]
            x_cat_vals = list(grouped.groups.keys())
            x_vals = np.array([x_positions[val] for val in x_cat_vals], dtype=float)
            means = grouped.mean().values
            cis = grouped.sem().values * 1.96
            cis = np.nan_to_num(cis)
            
            self.ax.plot(x_vals, means, color=c)
            self.ax.fill_between(x_vals, means - cis, means + cis, color=c, alpha=0.2)
            
        if not is_numeric:
            self.ax.set_xticks(list(x_positions.values()))
            self.ax.set_xticklabels([str(v) for v in x_unique])

    def _plot_bar(self, df, y, colors, c):
        plot_df, x_unique, x_positions, is_numeric = self._prepare_plot_data(df, y)
        
        if len(x_unique) < 2:
            bar_width = 0.4
        else:
            if is_numeric:
                min_gap = min(x_positions[x_unique[i+1]] - x_positions[x_unique[i]] for i in range(len(x_unique) - 1))
            else:
                min_gap = 1.0
            bar_width = min_gap * 0.6

        if self.z and self.z in plot_df.columns:
            hue_cats = plot_df[self.z].unique()
            n_hue = len(hue_cats)
            sub_width = bar_width / n_hue
            offsets = np.linspace(-(bar_width - sub_width) / 2,
                                  (bar_width - sub_width) / 2, n_hue)
            for h_idx, cat in enumerate(hue_cats):
                subset = plot_df[plot_df[self.z] == cat]
                grouped = subset.groupby(self.x)[y]
                means = grouped.mean()
                stds = grouped.std() if self.show_bar_plot_error_bar else None
                color = colors[h_idx % len(colors)]
                
                positions = [x_positions[xv] + offsets[h_idx] for xv in x_unique]
                heights = [means.get(xv, 0) for xv in x_unique]
                yerr = [stds.get(xv, 0) for xv in x_unique] if stds is not None else None
                
                self.ax.bar(
                    positions, heights, width=sub_width * 0.9,
                    color=color, edgecolor='black', linewidth=0.8,
                    yerr=yerr, capsize=3, ecolor='black',
                    label=str(cat)
                )
        else:
            grouped = plot_df.groupby(self.x)[y]
            means = grouped.mean()
            stds = grouped.std() if self.show_bar_plot_error_bar else None
            
            positions = [x_positions[xv] for xv in x_unique]
            heights = [means.get(xv, 0) for xv in x_unique]
            yerr = [stds.get(xv, 0) for xv in x_unique] if stds is not None else None
            
            self.ax.bar(
                positions, heights, width=bar_width,
                color=c, edgecolor='black', linewidth=0.8,
                yerr=yerr, capsize=3, ecolor='black'
            )
            
        if is_numeric:
            self.ax.set_xticks([x_positions[xv] for xv in x_unique])
            self.ax.set_xticklabels([str(xv) for xv in x_unique])
        else:
            self.ax.set_xticks(list(x_positions.values()))
            self.ax.set_xticklabels([str(v) for v in x_unique])

    def _plot_trendline(self, df, y, colors, c):
        self.trendline_equations = []  # reset before recomputing
        anchor = getattr(self, 'trendline_anchor_enabled', False)
        
        if self.z and self.z in df.columns:
            categories = df[self.z].unique()
            for idx, cat in enumerate(categories):
                subset = df[df[self.z] == cat]
                color = colors[idx % len(colors)]
                
                try:
                    x_fit, y_fit, coeffs = self._fit_trendline(subset)
                except Exception:
                    continue
                
                self.ax.scatter(
                    subset[self.x], subset[y],
                    color=color, s=self.scatter_size,
                    edgecolors=self.scatter_edgecolor, linewidths=0.5,
                    label=str(cat), zorder=3
                )
                
                if anchor:
                    self.ax.plot(x_fit, y_fit, color=color, linewidth=2)
                else:
                    self.ax.plot(x_fit, y_fit, color=color, linewidth=2)
                    x_data = subset[self.x].dropna().values.astype(float)
                    y_data = subset[y].dropna().values.astype(float)
                    if len(x_data) > 2:
                        p = np.poly1d(coeffs)
                        y_model = p(x_data)
                        t_targ = 1.96
                        se = np.sqrt(np.sum((y_data - y_model)**2) / (len(y_data) - 2))
                        ci = t_targ * se * np.sqrt(1/len(x_data) + (x_fit - x_data.mean())**2 / np.sum((x_data - x_data.mean())**2))
                        self.ax.fill_between(x_fit, y_fit - ci, y_fit + ci, color=color, alpha=0.2)
                    
                eq_str, r2 = self._build_equation_str(coeffs, subset)
                self.trendline_equations.append({
                    'label': str(cat), 'equation': eq_str, 'r2': f"{r2:.4f}"
                })
        else:
            x_fit, y_fit, coeffs = self._fit_trendline(df)
            
            self.ax.scatter(
                df[self.x], df[y],
                color=c, s=self.scatter_size,
                edgecolors=self.scatter_edgecolor, linewidths=0.5,
                label='All data', zorder=3
            )
            
            if anchor:
                self.ax.plot(x_fit, y_fit, color=c, linewidth=2)
            else:
                self.ax.plot(x_fit, y_fit, color=c, linewidth=2)
                x_data = df[self.x].dropna().values.astype(float)
                y_data = df[y].dropna().values.astype(float)
                if len(x_data) > 2:
                    p = np.poly1d(coeffs)
                    y_model = p(x_data)
                    t_targ = 1.96
                    se = np.sqrt(np.sum((y_data - y_model)**2) / (len(y_data) - 2))
                    ci = t_targ * se * np.sqrt(1/len(x_data) + (x_fit - x_data.mean())**2 / np.sum((x_data - x_data.mean())**2))
                    self.ax.fill_between(x_fit, y_fit - ci, y_fit + ci, color=c, alpha=0.2)
                
            eq_str, r2 = self._build_equation_str(coeffs, df)
            self.trendline_equations.append({
                'label': 'All data', 'equation': eq_str, 'r2': f"{r2:.4f}"
            })

    def _plot_histogram(self, df, colors):
        from scipy import stats
        plot_df = df.dropna(subset=[self.x])
        bins = self.hist_bins
        hist_step = getattr(self, 'hist_step', False)
        histtype = 'step' if hist_step else 'bar'
        alpha = 1.0 if hist_step else 0.7
        kde = getattr(self, 'hist_kde', False)
        
        hist_kwargs = {'bins': bins, 'histtype': histtype, 'alpha': alpha}
        if histtype == 'bar':
            hist_kwargs['edgecolor'] = 'black'
            hist_kwargs['linewidth'] = 0.8
            
        if self.z and self.z in plot_df.columns:
            categories = plot_df[self.z].unique()
            data_list = []
            labels = []
            c_list = []
            
            for idx, cat in enumerate(categories):
                subset = plot_df[plot_df[self.z] == cat][self.x]
                if not subset.empty:
                    data_list.append(subset.values)
                    labels.append(str(cat))
                    c_list.append(colors[idx % len(colors)])
            
            if data_list:
                self.ax.hist(data_list, color=c_list, label=labels, stacked=False, **hist_kwargs)
                
                if kde:
                    x_min, x_max = self.ax.get_xlim()
                    x_grid = np.linspace(x_min, x_max, 200)
                    for i, data in enumerate(data_list):
                        if len(data) > 1:
                            try:
                                density = stats.gaussian_kde(data)
                                bin_width = (x_max - x_min) / bins
                                y_grid = density(x_grid) * len(data) * bin_width
                                self.ax.plot(x_grid, y_grid, color=c_list[i], linewidth=2)
                            except Exception:
                                pass
        else:
            data = plot_df[self.x].values
            if len(data) > 0:
                color = colors[0] if colors else 'steelblue'
                self.ax.hist(data, color=color, label='All data', **hist_kwargs)
                
                if kde and len(data) > 1:
                    try:
                        x_min, x_max = self.ax.get_xlim()
                        x_grid = np.linspace(x_min, x_max, 200)
                        density = stats.gaussian_kde(data)
                        bin_width = (x_max - x_min) / bins
                        y_grid = density(x_grid) * len(data) * bin_width
                        self.ax.plot(x_grid, y_grid, color=color, linewidth=2)
                    except Exception:
                        pass
    
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
                        
                legend = self.ax.legend(unique_handles, unique_labels, loc='best', framealpha=0.7)
                legend.set_picker(True)
                legend.set_draggable(True)
                
                if getattr(self, 'legend_bbox', None) is not None:
                    legend._loc = tuple(self.legend_bbox)
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
    
    def _format_axis_label(self, col_name) -> str:
        """Format axis label based on specific parameter rules."""
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
    
    def _plot_secondary_axis(self, df):
        """Plot data on secondary y-axis."""
        if self.ax2:
            self.ax2.remove()
            self.ax2 = None
        
        if self.y2:
            self.ax2 = self.ax.twinx()
            
            # For simplicity with secondary axes, just use numeric spacing if x is numeric, 
            # otherwise just plot directly (matplotlib will handle it if X is strings)
            x_vals = df[self.x]
            if getattr(self, 'x_as_numeric', False):
                x_vals = pd.to_numeric(x_vals, errors='coerce')
                
            if self.plot_style == 'line':
                self.ax2.plot(x_vals, df[self.y2], color='red', label=self.y2)
            elif self.plot_style == 'point':
                self.ax2.errorbar(
                    x_vals, df[self.y2],
                    fmt='s', color='red', markeredgecolor='black',
                    markeredgewidth=1, capsize=3,
                    linestyle='-' if self.join_for_point_plot else 'none',
                    label=self.y2
                )
            elif self.plot_style == 'scatter':
                self.ax2.scatter(
                    x_vals, df[self.y2],
                    s=self.scatter_size, edgecolors=self.scatter_edgecolor,
                    color='red', label=self.y2
                )
            else:
                self.ax2.remove()
                self.ax2 = None
            
            if self.ax2:
                self.ax2.set_ylabel(self.y2label or self._format_axis_label(self.y2), color='red')
                self.ax2.tick_params(axis='y', colors='red')

    def _plot_tertiary_axis(self, df):
        """Plot data on tertiary y-axis."""
        if self.ax3:
            self.ax3.remove()
            self.ax3 = None
        
        if self.y3:
            self.ax3 = self.ax.twinx()
            self.ax3.spines["right"].set_position(("outward", 100))
            
            x_vals = df[self.x]
            if getattr(self, 'x_as_numeric', False):
                x_vals = pd.to_numeric(x_vals, errors='coerce')
            
            if self.plot_style == 'line':
                self.ax3.plot(x_vals, df[self.y3], color='green', label=self.y3)
            elif self.plot_style == 'point':
                self.ax3.errorbar(
                    x_vals, df[self.y3],
                    fmt='s', color='green', markeredgecolor='black',
                    markeredgewidth=1, capsize=3,
                    linestyle='-' if self.join_for_point_plot else 'none',
                    label=self.y3
                )
            elif self.plot_style == 'scatter':
                self.ax3.scatter(
                    x_vals, df[self.y3],
                    s=self.scatter_size, edgecolors=self.scatter_edgecolor,
                    color='green', label=self.y3
                )
            else:
                self.ax3.remove()
                self.ax3 = None
            
            if self.ax3:
                self.ax3.set_ylabel(self.y3label or self._format_axis_label(self.y3), color='green')
                self.ax3.tick_params(axis='y', colors='green')

    def _plot_secondary_x_axis(self, df):
        """Plot data on secondary x-axis (top)."""
        if self.ax_x2:
            self.ax_x2.remove()
            self.ax_x2 = None
        
        if self.x2 and self.x2 in df.columns:
            self.ax_x2 = self.ax.twiny()
            
            x_vals = df[self.x2]
            if getattr(self, 'x2logscale', False):
                x_vals = pd.to_numeric(x_vals, errors='coerce')
                
            if self.plot_style == 'line':
                self.ax_x2.plot(x_vals, df[self.y[0]], color='purple', label=self.y[0])
            elif self.plot_style == 'point':
                self.ax_x2.errorbar(
                    x_vals, df[self.y[0]],
                    fmt='D', color='purple', markeredgecolor='black',
                    markeredgewidth=1, capsize=3,
                    linestyle='-' if self.join_for_point_plot else 'none',
                    label=self.y[0]
                )
            elif self.plot_style == 'scatter':
                self.ax_x2.scatter(
                    x_vals, df[self.y[0]],
                    s=self.scatter_size, edgecolors=self.scatter_edgecolor,
                    color='purple', label=self.y[0]
                )
            else:
                self.ax_x2.remove()
                self.ax_x2 = None
            
            if self.ax_x2:
                self.ax_x2.set_xlabel(
                    self.x2label or self._format_axis_label(self.x2), color='purple'
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


