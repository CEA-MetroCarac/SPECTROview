# model/m_graph.py
"""Graph data model with all plot properties."""

import copy
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class MGraph:
    """Graph/plot data model storing configuration and visual properties.

    A plain (non-slotted) dataclass: every field name/type/default lives in
    exactly one place here, but the class still supports free `setattr`,
    `vars()`, and `hasattr` the same way a hand-written class would -- callers
    across the codebase (the scripting API, the AI agent, save/load, the View
    layer's model->widget sync) rely on that flat, freely-mutable shape.
    """

    graph_id: Optional[int] = None

    # Data source
    df_name: Optional[str] = None
    filters: List[Dict[str, Any]] = field(default_factory=list)

    # Plot configuration
    plot_style: str = "point"  # point, line, bar, wafer, 2Dmap, etc.
    plot_width: int = 480
    plot_height: int = 420
    dpi: int = 100

    # Axes
    x: Optional[str] = None
    y: List[str] = field(default_factory=list)  # Primary y-axis (can have multiple series)
    z: Optional[str] = None  # For wafer/2D plots

    # Secondary axes
    y2: Optional[str] = None  # Secondary y-axis
    y3: Optional[str] = None  # Tertiary y-axis
    x2: Optional[str] = None  # Secondary x-axis

    # Axis limits
    xmin: Optional[float] = None
    xmax: Optional[float] = None
    ymin: Optional[float] = None
    ymax: Optional[float] = None
    zmin: Optional[float] = None
    zmax: Optional[float] = None
    y2min: Optional[float] = None
    y2max: Optional[float] = None
    y3min: Optional[float] = None
    y3max: Optional[float] = None
    x2min: Optional[float] = None
    x2max: Optional[float] = None

    # Axis scales
    xlogscale: bool = False
    ylogscale: bool = False
    y2logscale: bool = False
    y3logscale: bool = False
    x2logscale: bool = False

    # Labels
    plot_title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    zlabel: Optional[str] = None
    y2label: Optional[str] = None
    y3label: Optional[str] = None
    x2label: Optional[str] = None

    # Visual properties
    x_rot: int = 0  # X-axis label rotation
    grid: bool = False

    # Legend
    legend_visible: bool = True
    legend_outside: bool = False
    legend_properties: List[Dict[str, Any]] = field(default_factory=list)
    legend_bbox: Optional[List[float]] = None  # (x, y) in axes coords for dragged position

    # Plot-specific properties
    color_palette: str = "jet"  # For wafer/2D maps
    wafer_size: float = 300.0  # Wafer diameter in mm
    wafer_stats: bool = True
    trendline_order: int = 1
    show_trendline_eq: bool = True
    trendline_anchor_enabled: bool = False
    trendline_anchor_origin: bool = True   # True=through (0,0), False=custom
    trendline_anchor_x: float = 0.0
    trendline_anchor_y: float = 0.0
    show_bar_plot_error_bar: bool = False
    join_for_point_plot: bool = False
    dodge_point_plot: bool = True
    dodge_scatter_plot: bool = False
    scatter_size: int = 70  # Marker size for scatter plots
    scatter_edgecolor: str = "black"  # Edge color for scatter plot markers
    x_as_numeric: Optional[bool] = None  # None=Auto, True=Numerical, False=Category
    y_as_numeric: Optional[bool] = None  # None=Auto, True=Numerical, False=Category

    minor_ticks_bottom: bool = True
    minor_ticks_left: bool = True
    minor_ticks_top: bool = False
    minor_ticks_right: bool = False

    # Histogram-specific
    hist_bins: int = 20
    hist_kde: bool = False
    hist_step: bool = False

    # Data sorting
    sort_data_enabled: bool = True   # Enable intelligent sorting
    sort_data_by: str = "Z"          # Sort by: "Z" (hue), "X", or "Y"

    # Annotations (lines and text)
    annotations: List[Dict[str, Any]] = field(default_factory=list)

    # Axis breaks: {'x': {'start','end'} | None, 'y': {'start','end'} | None}
    axis_breaks: Dict[str, Optional[Dict[str, float]]] = field(
        default_factory=lambda: {'x': None, 'y': None}
    )

    def save(self) -> Dict[str, Any]:
        """Serialize graph to dictionary.

        Built from `vars(self)` (in the same order the fields are declared
        above, starting with `graph_id`). Deep-copied so a caller holding
        onto the returned dict (e.g. a saved template) never aliases this
        model's own mutable fields (`y`, `filters`, `annotations`,
        `legend_properties`, `axis_breaks`) and is unaffected by later edits
        to this graph.
        """
        return copy.deepcopy(vars(self))

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
