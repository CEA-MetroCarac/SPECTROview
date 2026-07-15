# model/m_graph.py
"""Graph data model with all plot properties."""

from typing import Optional, List, Dict, Any


class MGraph:
    """Graph/plot data model storing configuration and visual properties."""
    
    def __init__(self, graph_id: Optional[int] = None):
        """Initialize graph with default properties."""
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
        self.x2: Optional[str] = None  # Secondary x-axis
        
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
        self.x2min: Optional[float] = None
        self.x2max: Optional[float] = None
        
        # Axis scales
        self.xlogscale: bool = False
        self.ylogscale: bool = False
        self.y2logscale: bool = False
        self.y3logscale: bool = False
        self.x2logscale: bool = False
        
        # Labels
        self.plot_title: Optional[str] = None
        self.xlabel: Optional[str] = None
        self.ylabel: Optional[str] = None
        self.zlabel: Optional[str] = None
        self.y2label: Optional[str] = None
        self.y3label: Optional[str] = None
        self.x2label: Optional[str] = None
        
        # Visual properties
        self.x_rot: int = 0  # X-axis label rotation
        self.grid: bool = False
        
        # Legend
        self.legend_visible: bool = True
        self.legend_outside: bool = False
        self.legend_properties: List[Dict[str, Any]] = []
        self.legend_bbox = None  # (x, y) in axes coords for dragged position
        
        # Plot-specific properties
        self.color_palette: str = "jet"  # For wafer/2D maps
        self.wafer_size: float = 300.0  # Wafer diameter in mm
        self.wafer_stats: bool = True
        self.trendline_order: int = 1
        self.show_trendline_eq: bool = True
        self.trendline_anchor_enabled: bool = False
        self.trendline_anchor_origin: bool = True   # True=through (0,0), False=custom
        self.trendline_anchor_x: float = 0.0
        self.trendline_anchor_y: float = 0.0
        self.show_bar_plot_error_bar: bool = False
        self.join_for_point_plot: bool = False
        self.dodge_point_plot: bool = True
        self.dodge_scatter_plot: bool = False
        self.scatter_size: int = 70  # Marker size for scatter plots
        self.scatter_edgecolor: str = "black"  # Edge color for scatter plot markers
        self.x_as_numeric: Optional[bool] = None  # None=Auto, True=Numerical, False=Category
        self.y_as_numeric: Optional[bool] = None  # None=Auto, True=Numerical, False=Category
        
        self.minor_ticks_bottom: bool = True
        self.minor_ticks_left: bool = True
        self.minor_ticks_top: bool = False
        self.minor_ticks_right: bool = False
        
        # Histogram-specific
        self.hist_bins: int = 20
        self.hist_kde: bool = False
        self.hist_step: bool = False
        
        # Data sorting
        self.sort_data_enabled: bool = True   # Enable intelligent sorting
        self.sort_data_by: str = "Z"          # Sort by: "Z" (hue), "X", or "Y"
        
        # Annotations (lines and text)
        self.annotations: List[Dict[str, Any]] = []

        # Axis breaks: {'x': {'start','end'} | None, 'y': {'start','end'} | None}
        self.axis_breaks: Dict[str, Optional[Dict[str, float]]] = {'x': None, 'y': None}
    
    def save(self) -> Dict[str, Any]:
        """Serialize graph to dictionary.

        Built from `vars(self)` (in the same order `__init__` assigns them,
        starting with `graph_id`) """
        return dict(vars(self))
    
    def load(self, data: Dict[str, Any]):
        """Load graph properties from dictionary."""
        # List of float-type limit properties
        float_limit_keys = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax', 
                           'y2min', 'y2max', 'y3min', 'y3max',
                           'x2min', 'x2max']
        
        for key, value in data.items():
            if hasattr(self, key):
                # Convert string limit values to float or None
                if key in float_limit_keys and value is not None:
                    try:
                        value = float(value) if value != '' else None
                    except (ValueError, TypeError):
                        value = None
                # Backward compatibility for x_as_numeric: False -> None
                if key == 'x_as_numeric' and value is False:
                    value = None
                
                setattr(self, key, value)
        
        # Backward compatibility: ensure annotations exists for old .graphs files
        if not hasattr(self, 'annotations') or self.annotations is None:
            self.annotations = []
    
    def get_display_name(self) -> str:
        """Get display name for graph list."""
        x_str = self.x if self.x else "None"
        y_str = self.y[0] if self.y else "None"
        z_str = self.z if self.z else "None"
        return f"{self.graph_id}-{self.plot_style}_plot: [{x_str}] - [{y_str}] - [{z_str}]"
