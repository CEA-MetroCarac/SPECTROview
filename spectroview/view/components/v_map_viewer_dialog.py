"""
Floating map viewer dialog window.
"""
from PySide6.QtWidgets import QDialog, QVBoxLayout
from PySide6.QtCore import Qt, Signal
from spectroview.view.components.v_map_viewer import VMapViewer


class VMapViewerDialog(QDialog):
    """Floating dialog window containing a map viewer.
    
    This dialog can be moved anywhere on screen, allowing users to compare
    multiple map parameters side-by-side across different monitors.
    """
    
    # Forward signals from the contained viewer
    spectra_selected = Signal(list)
    extract_profile_requested = Signal(str)
    
    def __init__(self, parent=None, viewer_number=1):
        super().__init__(parent)
        
        self.viewer_number = viewer_number
        
        # Dialog setup
        self.setWindowTitle(f"Map Viewer {viewer_number}")
        self.setWindowFlags(Qt.Window)  # Independent window
        self.resize(450, 700)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # Create map viewer instance
        self.map_viewer = VMapViewer(self)
        layout.addWidget(self.map_viewer)
        
        # Forward signals from viewer to parent
        self.map_viewer.spectra_selected.connect(self.spectra_selected.emit)
        self.map_viewer.extract_profile_requested.connect(self.extract_profile_requested.emit)
        
        # Don't allow nested viewers from dialogs
        self.map_viewer.btn_add_viewer.setEnabled(False)
        self.map_viewer.btn_add_viewer.setToolTip("Only available in main viewer")
    
    # Public API - delegate to internal viewer
    
    def set_map_data(self, map_df, map_name, df_fit_results=None):
        """Set map data in the viewer."""
        self.map_viewer.set_map_data(map_df, map_name, df_fit_results)
    
    def set_selected_points(self, points: list):
        """Set selected points in the viewer."""
        self.map_viewer.set_selected_points(points)
    
    def clear_cache_for_map(self, map_name: str):
        """Clear cache for specific map."""
        self.map_viewer.clear_cache_for_map(map_name)
    
    def blockSignals(self, block: bool) -> bool:
        """Block/unblock signals from the viewer."""
        return self.map_viewer.blockSignals(block)
