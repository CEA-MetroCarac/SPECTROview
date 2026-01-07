"""View for Maps Workspace - extends Spectra Workspace with map-specific features."""
import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QCheckBox, QComboBox, QFrame,
    QGroupBox, QLineEdit, QDoubleSpinBox, QSpacerItem, QSizePolicy,
    QScrollArea
)

from spectroview import ICON_DIR
from spectroview.model.m_settings import MSettings
from spectroview.view.v_workspace_spectra import VWorkspaceSpectra
from spectroview.view.components.v_map_list import VMapsList
from spectroview.view.components.v_map_viewer import VMapViewer
from spectroview.viewmodel.vm_workspace_maps import VMWorkspaceMaps


class VWorkspaceMaps(VWorkspaceSpectra):
    """Maps Workspace View - inherits from Spectra Workspace and adds map visualization."""
    
    # ───── Additional signals for Maps ─────
    map_type_changed = Signal(str)
    send_to_spectra_requested = Signal()
    
    def __init__(self, parent=None):
        # Call parent constructor - this will create the base Spectra layout
        super().__init__(parent)
        
        # Override parent's ViewModel with Maps-specific ViewModel
        settings = MSettings()
        self.vm = VMWorkspaceMaps(settings)
        
        self._add_maps_panel()
        self._connect_signals()
    
    def _add_maps_panel(self):
        """Replace the simple spectra list sidebar with Maps-specific controls."""
        main_layout = self.layout()
        main_splitter = main_layout.itemAt(0).widget()  # QSplitter
        
        # Remove the old right widget and create new one
        old_right = main_splitter.widget(1)
        if old_right:
            old_right.setParent(None)
            old_right.deleteLater()
        
        # Create new right panel with Maps features
        right_widget = self._create_maps_right_panel()
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([900, 400])
    
    def _create_maps_right_panel(self):
        """Create the right panel with map viewer and controls in a scroll area."""
        # Main panel wrapper
        panel = QFrame()
        panel.setMaximumWidth(450)
        panel.setFrameShape(QFrame.StyledPanel)
        main_layout = QVBoxLayout(panel)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll area for all controls
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QFrame.NoFrame)
        
        # Scrollable content widget
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        
        # ══════════════════════════════════════════════════════════════
        # MAP VIEWER WIDGET (with matplotlib canvas and controls)
        # ══════════════════════════════════════════════════════════════
        self.v_map_viewer = VMapViewer()
        layout.addWidget(self.v_map_viewer)
        
        # ══════════════════════════════════════════════════════════════
        # MAPS AND SPECTRA LIST (VMapsList widget with action buttons)
        # ══════════════════════════════════════════════════════════════
        self.v_maps_list = VMapsList()
        layout.addWidget(self.v_maps_list, stretch=1)
        
        # ── Progress info (from parent) ──
        self.lbl_count.setText("0 points")
        layout.addWidget(self.lbl_count)
        
        # Set scroll content and add to main panel
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        
        # Progress bar outside scroll area
        main_layout.addWidget(self.progress_bar)
        
        return panel
    
    def _connect_signals(self):
        """Connect Maps-specific signals between View and ViewModel."""
        # ── VMapsList → ViewModel connections ──
        self.v_maps_list.files_dropped.connect(self.vm.load_map_files)
        self.v_maps_list.map_selection_changed.connect(self._on_map_selected)
        self.v_maps_list.spectra_selection_changed.connect(self.vm.set_selected_indices)
        
        # ── ViewModel → VMapsList connections ──
        self.vm.maps_list_changed.connect(self.v_maps_list.set_maps_names)
        
        # The spectra list is updated via inherited VMWorkspaceSpectra signals
        # when vm.select_map() calls _extract_spectra_from_map()
        self.vm.spectra_list_changed.connect(
            lambda names: self.v_maps_list.set_spectra_names(names)
        )
        
        # Connect spectra selection to viewer (inherited from parent)
        self.vm.spectra_selection_changed.connect(self.v_spectra_viewer.set_plot_data)
    
    def _on_map_selected(self, index: int):
        """Handle map selection - convert index to map name and call ViewModel."""
        if index >= 0 and index < len(self.vm.maps):
            map_names = list(self.vm.maps.keys())
            map_name = map_names[index]
            self.vm.select_map(map_name)
