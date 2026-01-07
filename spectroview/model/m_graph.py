# model/m_graph.py
"""Model for Graph - represents a single graph/plot with all its properties."""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any


class MGraph:
    """Data model for a graph/plot.
    
    Stores all properties needed to render a graph including:
    - Data source (DataFrame name)
    - Plot style and axes configuration
    - Visual properties (colors, markers, legend)
    - Filters applied to data
    """
    
    def __init__(self, graph_id: Optional[int] = None):
        """Initialize a new graph with default properties.
        
        Args:
            graph_id: Unique identifier for this graph
        """
        self.graph_id = graph_id
        
        # Data source
        self.df_name: Optional[str] = None
        self.filters: List[str] = []
        
        # Plot configuration
        self.plot_style: str = "point"  # point, line, bar, wafer, 2Dmap, etc.
        self.plot_width: int = 480
        self.plot_height: int = 420
        self.dpi: int = 100
        
        # Axes
        self.x: Optional[str] = None
        self.y: List[str] = []  # Primary y-axis (can have multiple series)
        self.z: Optional[str] = None  # For wafer/2D plots
        
        # Secondary axes
        self.y2: Optional[str] = None  # Secondary y-axis
        self.y3: Optional[str] = None  # Tertiary y-axis
        
        # Axis limits
        self.xmin: Optional[float] = None
        self.xmax: Optional[float] = None
        self.ymin: Optional[float] = None
        self.ymax: Optional[float] = None
        self.zmin: Optional[float] = None
        self.zmax: Optional[float] = None
        self.y2min: Optional[float] = None
        self.y2max: Optional[float] = None
        self.y3min: Optional[float] = None
        self.y3max: Optional[float] = None
        
        # Axis scales
        self.xlogscale: bool = False
        self.ylogscale: bool = False
        self.y2logscale: bool = False
        self.y3logscale: bool = False
        
        # Labels
        self.plot_title: Optional[str] = None
        self.xlabel: Optional[str] = None
        self.ylabel: Optional[str] = None
        self.zlabel: Optional[str] = None
        self.y2label: Optional[str] = None
        self.y3label: Optional[str] = None
        
        # Visual properties
        self.x_rot: int = 0  # X-axis label rotation
        self.grid: bool = False
        
        # Legend
        self.legend_visible: bool = True
        self.legend_location: str = 'upper right'
        self.legend_outside: bool = False
        self.legend_properties: List[Dict[str, Any]] = []
        
        # Plot-specific properties
        self.color_palette: str = "jet"  # For wafer/2D maps
        self.wafer_size: float = 300.0  # Wafer diameter in mm
        self.wafer_stats: bool = True
        self.trendline_order: int = 1
        self.show_trendline_eq: bool = True
        self.show_bar_plot_error_bar: bool = True
        self.join_for_point_plot: bool = False
    
    def save(self) -> Dict[str, Any]:
        """Serialize graph to dictionary for saving.
        
        Returns:
            Dictionary containing all graph properties
        """
        return {
            'graph_id': self.graph_id,
            'df_name': self.df_name,
            'filters': self.filters,
            'plot_style': self.plot_style,
            'plot_width': self.plot_width,
            'plot_height': self.plot_height,
            'dpi': self.dpi,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'y2': self.y2,
            'y3': self.y3,
            'xmin': self.xmin,
            'xmax': self.xmax,
            'ymin': self.ymin,
            'ymax': self.ymax,
            'zmin': self.zmin,
            'zmax': self.zmax,
            'y2min': self.y2min,
            'y2max': self.y2max,
            'y3min': self.y3min,
            'y3max': self.y3max,
            'xlogscale': self.xlogscale,
            'ylogscale': self.ylogscale,
            'y2logscale': self.y2logscale,
            'y3logscale': self.y3logscale,
            'plot_title': self.plot_title,
            'xlabel': self.xlabel,
            'ylabel': self.ylabel,
            'zlabel': self.zlabel,
            'y2label': self.y2label,
            'y3label': self.y3label,
            'x_rot': self.x_rot,
            'grid': self.grid,
            'legend_visible': self.legend_visible,
            'legend_location': self.legend_location,
            'legend_outside': self.legend_outside,
            'legend_properties': self.legend_properties,
            'color_palette': self.color_palette,
            'wafer_size': self.wafer_size,
            'wafer_stats': self.wafer_stats,
            'trendline_order': self.trendline_order,
            'show_trendline_eq': self.show_trendline_eq,
            'show_bar_plot_error_bar': self.show_bar_plot_error_bar,
            'join_for_point_plot': self.join_for_point_plot,
        }
    
    def load(self, data: Dict[str, Any]):
        """Load graph properties from dictionary.
        
        Args:
            data: Dictionary containing graph properties
        """
        # List of float-type limit properties
        float_limit_keys = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax', 
                           'y2min', 'y2max', 'y3min', 'y3max']
        
        for key, value in data.items():
            if hasattr(self, key):
                # Convert string limit values to float or None
                if key in float_limit_keys and value is not None:
                    try:
                        value = float(value) if value != '' else None
                    except (ValueError, TypeError):
                        value = None
                
                setattr(self, key, value)
    
    def get_display_name(self) -> str:
        """Get a display name for this graph.
        
        Returns:
            String representation like "1-line_plot: [X] - [Y] - [Z]"
        """
        x_str = self.x if self.x else "None"
        y_str = self.y[0] if self.y else "None"
        z_str = self.z if self.z else "None"
        return f"{self.graph_id}-{self.plot_style}_plot: [{x_str}] - [{y_str}] - [{z_str}]"
