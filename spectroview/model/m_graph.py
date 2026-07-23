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
    
    plot_width: int = 408
    plot_height: int = 327
    dpi: int = 100

    # Axes
    x: Optional[str] = None
    y: List[str] = field(default_factory=list)  # Primary y-axis (can have multiple series)
    z: Optional[str] = None  # For wafer/2D plots

    # Secondary axes
    y2: Optional[str] = None  # Secondary y-axis
    y3: Optional[str] = None  # Tertiary y-axis
    x2: Optional[str] = None  # Secondary x-axis
    y2color: str = "red"      # matches the pre-existing hardcoded color in _plot_secondary_axis
    y2marker: str = "s"
    y3color: str = "green"    # matches the pre-existing hardcoded color in _plot_tertiary_axis
    y3marker: str = "s"
    x2color: str = "purple"   # matches the pre-existing hardcoded color in _plot_secondary_x_axis
    x2marker: str = "D"

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
    xscale_mode: str = "log"  # log/symlog -- which scale xlogscale switches to when True
    yscale_mode: str = "log"  # log/symlog -- which scale ylogscale switches to when True

    # Labels
    plot_title: Optional[str] = None
    plot_subtitle: Optional[str] = None
    subtitle_fontsize: int = 10  # matches the pre-existing hardcoded fallback; only rendered once plot_subtitle is set
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    zlabel: Optional[str] = None
    y2label: Optional[str] = None
    y3label: Optional[str] = None
    x2label: Optional[str] = None

    # Visual properties
    x_rot: int = 0  # X-axis label rotation
    grid: bool = False
    tick_direction: Optional[str] = None   # in/out/inout; None = matplotlib's own default
    tick_label_format: Optional[str] = None  # e.g. "%.2f"; None = default ScalarFormatter
    x_inverted: bool = False
    y_inverted: bool = False
    title_fontsize: int = 12       # matches mplstyle's axes.titlesize
    axis_label_fontsize: int = 12  # matches mplstyle's axes.labelsize
    tick_label_fontsize: int = 9   # matches mplstyle's x/ytick.labelsize
    figure_facecolor: Optional[str] = None     # None = mplstyle's figure/axes facecolor
    figure_margins: List[float] = field(default_factory=lambda: [0.05, 0.05])  # [x_margin, y_margin]; matches matplotlib's own default
    spines_visible: Dict[str, bool] = field(
        default_factory=lambda: {'top': True, 'right': True, 'bottom': True, 'left': True}
    )
    figure_theme: str = "light"  # light/dark/soft_dark -- selects the mplstyle used to render this graph
    export_width_mm: Optional[float] = None   # None = use the current on-screen figure size at export time
    export_height_mm: Optional[float] = None

    # Legend
    legend_visible: bool = True
    legend_outside: bool = False
    legend_properties: List[Dict[str, Any]] = field(default_factory=list)
    legend_bbox: Optional[List[float]] = None  # (x, y) in axes coords for dragged position
    legend_ncol: int = 1
    legend_frame: bool = True
    legend_title: Optional[str] = None
    legend_fontsize: int = 9
    legend_alpha: float = 0.7  # matches the pre-existing hardcoded framealpha=0.7
    legend_loc: str = "best"   # inside-legend position; ignored when legend_outside is True

    # Plot-specific properties
    color_palette: str = "jet"  # For wafer/2D maps
    colormap_norm: str = "linear"  # linear/log/centered -- color-scale normalization for wafer/2Dmap
    colormap_center: float = 0.0   # vcenter for "centered" norm (e.g. 0 for diverging stress/strain data)
    wafer_size: float = 300.0  # Wafer diameter in mm
    wafer_stats: bool = True
    trendline_order: int = 1
    show_trendline_eq: bool = True
    trendline_anchor_enabled: bool = False
    trendline_anchor_origin: bool = True   # True=through (0,0), False=custom
    trendline_anchor_x: float = 0.0
    trendline_anchor_y: float = 0.0
    show_bar_plot_error_bar: bool = False
    error_bar_type: str = "ci95"       # none/sd/sem/ci95 -- point & line, shown unconditionally
    bar_error_bar_type: str = "sd"     # none/sd/sem/ci95 -- bar, only used when show_bar_plot_error_bar
    error_bar_capsize: float = 3.0
    join_for_point_plot: bool = False
    dodge_point_plot: bool = True
    dodge_scatter_plot: bool = False
    scatter_size: int = 70  # Marker size for scatter plots
    scatter_edgecolor: str = "black"  # Edge color for scatter plot markers
    unify_marker_style: bool = True  # True: every series uses scatter_size/scatter_edgecolor; False: per-series legend_properties overrides apply
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

    # Inset (zoom) axes: one optional inset per graph, showing the same
    # series as the main plot at its own x/y limits.
    inset_enabled: bool = False
    inset_bounds: List[float] = field(default_factory=lambda: [0.55, 0.55, 0.35, 0.35])  # [x0, y0, width, height] in axes-fraction
    inset_xmin: Optional[float] = None
    inset_xmax: Optional[float] = None
    inset_ymin: Optional[float] = None
    inset_ymax: Optional[float] = None
    inset_show_zoom_indicator: bool = True

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
