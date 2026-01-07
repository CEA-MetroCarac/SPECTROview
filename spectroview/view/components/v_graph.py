# view/v_graph.py
"""Graph plotting widget - handles matplotlib visualization following MVVM pattern."""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.interpolate import griddata
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from PySide6.QtWidgets import QVBoxLayout, QLabel, QLineEdit, QWidget, QComboBox, QStyledItemDelegate
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPalette, QColor

from spectroview import DEFAULT_COLORS, DEFAULT_MARKERS, MARKERS
from spectroview.modules.utils import rgba_to_default_color, show_alert


class VGraph(QWidget):
    """Graph plotting widget - renders plots based on MGraph model properties."""
    
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
        self.legend_location = 'best'
        self.legend_outside = False
        self.legend_properties = []
        
        # Plot-specific settings
        self.color_palette = "jet"
        self.wafer_size = 300
        self.wafer_stats = True
        self.trendline_order = 1
        self.show_trendline_eq = True
        self.show_bar_plot_error_bar = True
        self.join_for_point_plot = False
        
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
        for action in self.toolbar.actions():
            if action.text() in ['Save', 'Subplots']:
                action.setVisible(False)
        
        if layout:
            layout.addWidget(self.canvas)
            layout.addWidget(self.toolbar)
        else:
            self.graph_layout.addWidget(self.canvas)
            self.graph_layout.addWidget(self.toolbar)
        
        # Connect pick event for legend customization
        self.canvas.mpl_connect('pick_event', self._on_legend_pick)
        
        self.canvas.draw_idle()
    
    def plot(self, df):
        """Renders plot based on DataFrame and current properties."""
        # Store DataFrame for replotting when legend is customized
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
        
        self.get_legend_properties()
        self.canvas.draw_idle()
    
    def get_legend_properties(self):
        """Retrieves properties of each legend within legend box."""
        legend_properties = []
        if self.ax:
            legend = self.ax.get_legend()
            if legend:
                legend_texts = legend.get_texts()
                legend_handles = legend.legendHandles
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
    
    def _on_legend_pick(self, event):
        """Handle legend click event to show customization dialog."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox, QHBoxLayout
        import copy
        
        # Check if legend was clicked
        if event.artist.get_label() == '_legend_':
            return
        
        legend = self.ax.get_legend()
        if legend and event.artist == legend:
            # Save original legend properties before allowing edits
            original_legend_properties = copy.deepcopy(self.legend_properties)
            
            # Create customization dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Customize Legend")
            dialog.resize(250, 400)
            
            # Main layout
            main_layout = QVBoxLayout(dialog)
            
            # Legend customization widget layout
            legend_layout = QHBoxLayout()
            self.customize_legend_widget(legend_layout)
            main_layout.addLayout(legend_layout)
            
            # Dialog buttons
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            main_layout.addWidget(button_box)
            
            # Show dialog
            result = dialog.exec()
            if result == QDialog.Accepted:
                # Replot with updated colors and labels
                if self.df is not None:
                    self.plot(self.df)
                else:
                    self.canvas.draw_idle()
            else:
                # Restore original legend properties if cancelled
                self.legend_properties = original_legend_properties
                self._set_legend()
                self.canvas.draw_idle()
    
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
                sns.lineplot(
                    data=df, x=self.x, y=y, hue=self.z, ax=self.ax,
                    palette=colors
                )
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
        plt.colorbar(heatmap, orientation='vertical')
    
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
                legend = self.ax.legend(handles, legend_labels, loc=self.legend_location)
                legend.set_picker(True)  # Make legend clickable
                #legend.set_draggable(True)
            else:
                self.ax.legend().remove()
            
            if self.legend_outside:
                legend = self.ax.legend(
                    handles, legend_labels,
                    loc='center left',
                    bbox_to_anchor=(1, 0.5)
                )
                legend.set_picker(True)  # Make legend clickable
                #legend.set_draggable(True)
    
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
        
        plt.colorbar(im, ax=ax)
        
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
