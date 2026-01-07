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
from spectroview.view.v_workspace_spectra import VWorkspaceSpectra
from spectroview.view.components.v_map_list import VMapsList
from spectroview.view.components.v_map_viewer import VMapViewer


class VWorkspaceMaps(VWorkspaceSpectra):
    """Maps Workspace View - inherits from Spectra Workspace and adds map visualization."""
    
    # ───── Additional signals for Maps ─────
    map_type_changed = Signal(str)
    send_to_spectra_requested = Signal()
    
    def __init__(self, parent=None):
        # Call parent constructor - this will create the base Spectra layout
        super().__init__(parent)
        self._add_maps_panel()
    
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
        # ADD MORE MAP VIEWER BUTTONS
        # ══════════════════════════════════════════════════════════════
        layout.addWidget(self._create_multi_viewer_group())
        
        # ══════════════════════════════════════════════════════════════
        # MAPS AND SPECTRA LIST (VMapsList widget with action buttons)
        # ══════════════════════════════════════════════════════════════
        self.v_maps_list = VMapsList()
        layout.addWidget(self.v_maps_list, stretch=1)
        
        # ── Progress info (from parent) ──
        self.lbl_count.setText("0 points")
        layout.addWidget(self.lbl_count)
        layout.addWidget(self.progress_bar)
        
        # Set scroll content and add to main panel
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        
        return panel
    
    def _create_multi_viewer_group(self):
        """Create 'Add more map viewer' controls."""
        gb = QGroupBox("Add more map viewer:")
        layout = QHBoxLayout(gb)
        layout.setContentsMargins(4, 6, 4, 4)
        
        self.btn_viewer_2 = QPushButton("2")
        self.btn_viewer_3 = QPushButton("3")
        self.btn_viewer_4 = QPushButton("4")
        
        for btn in [self.btn_viewer_2, self.btn_viewer_3, self.btn_viewer_4]:
            btn.setCheckable(True)
            btn.setFixedSize(40, 26)
            layout.addWidget(btn)
        
        layout.addStretch()
        
        return gb
