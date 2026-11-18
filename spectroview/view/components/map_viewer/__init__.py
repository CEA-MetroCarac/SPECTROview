"""Map viewer package: 2D/wafer heatmap canvas, selection helpers, and floating dialog."""

from spectroview.view.components.map_viewer.map_selection import (
    MapCanvasSelector,
    Point2D,
    compute_2d_spatial_geometry,
    get_drag_box_edgecolor,
    is_ctrl_event,
    resolve_1d_profile_selection,
    resolve_xy_selection,
)
from spectroview.view.components.map_viewer.v_map_data import MapDataProcessingMixin
from spectroview.view.components.map_viewer.v_map_interactions import MapInteractionMixin
from spectroview.view.components.map_viewer.v_map_plot import MapPlotMixin
from spectroview.view.components.map_viewer.v_map_viewer import VMapViewer
from spectroview.view.components.map_viewer.v_map_viewer_dialog import VMapViewerDialog

__all__ = [
    "VMapViewer",
    "VMapViewerDialog",
    "MapDataProcessingMixin",
    "MapPlotMixin",
    "MapInteractionMixin",
    "MapCanvasSelector",
    "Point2D",
    "compute_2d_spatial_geometry",
    "get_drag_box_edgecolor",
    "is_ctrl_event",
    "resolve_xy_selection",
    "resolve_1d_profile_selection",
]
