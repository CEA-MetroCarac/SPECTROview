"""View for Maps Workspace - extends Spectra Workspace with map-specific features."""
import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QCheckBox, QComboBox, QFrame,
    QGroupBox, QLineEdit, QDoubleSpinBox, QSpacerItem, QSizePolicy
)

from spectroview import ICON_DIR
from spectroview.view.v_workspace_spectra import VWorkspaceSpectra
from spectroview.modules.map_viewer import MapViewer


class VWorkspaceMaps(VWorkspaceSpectra):
    """Maps Workspace View - inherits from Spectra Workspace and adds map visualization."""
    
    # ───── Additional signals for Maps ─────
    map_type_changed = Signal(str)
    quick_selection_requested = Signal(str)  # V, H, Q1-Q4
    send_to_spectra_requested = Signal()
    
    def __init__(self, parent=None):
        # Call parent constructor - this will create the base Spectra layout
        super().__init__(parent)
        
        # Override right panel to add Maps-specific controls
        self._add_maps_panel()
    
    def _add_maps_panel(self):
        """Replace the simple spectra list sidebar with Maps-specific controls."""
        # Get the right widget that was created in parent's init_ui
        # We need to replace its content
        
        # The parent creates a main_splitter with left and right widgets
        # We need to find and modify the right widget
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
        """Create the right panel with map viewer and controls."""
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        
        # ── Maps section header ──
        maps_header = QLabel("Maps")
        maps_header.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(maps_header)
        
        # ── Map Viewer (placeholder - will be replaced with actual MapViewer) ──
        self.map_display_frame = QFrame()
        self.map_display_frame.setFrameShape(QFrame.Box)
        self.map_display_frame.setMinimumHeight(200)
        self.map_display_frame.setMaximumHeight(300)
        
        map_layout = QVBoxLayout(self.map_display_frame)
        map_placeholder = QLabel("Map visualization")
        map_placeholder.setAlignment(Qt.AlignCenter)
        map_layout.addWidget(map_placeholder)
        
        layout.addWidget(self.map_display_frame)
        
        # ── Top buttons row (icons) ──
        buttons_layout = QHBoxLayout()
        
        self.btn_view_map = QPushButton()
        self.btn_view_map.setIcon(QIcon(os.path.join(ICON_DIR, "map.png")))
        self.btn_view_map.setToolTip("View map")
        
        self.btn_export_map = QPushButton()
        self.btn_export_map.setIcon(QIcon(os.path.join(ICON_DIR, "save.png")))
        self.btn_export_map.setToolTip("Export map")
        
        buttons_layout.addWidget(self.btn_view_map)
        buttons_layout.addWidget(self.btn_export_map)
        buttons_layout.addStretch()
        
        layout.addLayout(buttons_layout)
        
        # ── Quick selection for Wafer map ──
        layout.addWidget(self._create_quick_selection_group())
        
        # ── Map Type selection ──
        layout.addWidget(self._create_map_type_group())
        
        # ── X-range controls ──
        layout.addWidget(self._create_xrange_group())
        
        # ── Mask controls ──
        layout.addWidget(self._create_mask_group())
        
        # ── Add more map viewer ──
        layout.addWidget(self._create_multi_viewer_group())
        
        # ── Spectra list from parent ──
        spectra_list_label = QLabel("Spectrum(s):")
        spectra_list_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(spectra_list_label)
        
        # Checkbox for Check all
        self.cb_check_all_maps = QCheckBox("Check all")
        layout.addWidget(self.cb_check_all_maps)
        
        # Add the spectra list widget from parent
        layout.addWidget(self.v_spectra_list, stretch=1)
        
        # ── Reinitialize and Stats buttons ──
        bottom_buttons = QHBoxLayout()
        self.btn_reinit_maps = QPushButton("Reinitialize")
        self.btn_stats_maps = QPushButton("Stats")
        bottom_buttons.addWidget(self.btn_reinit_maps)
        bottom_buttons.addWidget(self.btn_stats_maps)
        layout.addLayout(bottom_buttons)
        
        # ── Send to Spectra Tab ──
        self.btn_send_to_spectra = QPushButton("Send selected spectra to 'Spectra' Tab")
        self.btn_send_to_spectra.clicked.connect(self.send_to_spectra_requested.emit)
        layout.addWidget(self.btn_send_to_spectra)
        
        # ── Progress info (from parent) ──
        layout.addWidget(self.lbl_count)
        layout.addWidget(self.progress_bar)
        
        return panel
    
    def _create_quick_selection_group(self):
        """Create quick selection group for wafer quadrants."""
        gb = QGroupBox("Quick selection (for Wafer map) :")
        layout = QVBoxLayout(gb)
        layout.setContentsMargins(4, 8, 4, 4)
        
        # Row 1: V, H, Q1, Q2
        row1 = QHBoxLayout()
        self.btn_quick_v = QPushButton("V")
        self.btn_quick_h = QPushButton("H")
        self.btn_quick_q1 = QPushButton("Q1")
        self.btn_quick_q2 = QPushButton("Q2")
        
        for btn in [self.btn_quick_v, self.btn_quick_h, self.btn_quick_q1, self.btn_quick_q2]:
            btn.setFixedSize(50, 28)
            row1.addWidget(btn)
        
        # Row 2: Q3, Q4
        row2 = QHBoxLayout()
        self.btn_quick_q3 = QPushButton("Q3")
        self.btn_quick_q4 = QPushButton("Q4")
        
        for btn in [self.btn_quick_q3, self.btn_quick_q4]:
            btn.setFixedSize(50, 28)
            row2.addWidget(btn)
        row2.addStretch()
        
        # Connect signals
        self.btn_quick_v.clicked.connect(lambda: self.quick_selection_requested.emit("V"))
        self.btn_quick_h.clicked.connect(lambda: self.quick_selection_requested.emit("H"))
        self.btn_quick_q1.clicked.connect(lambda: self.quick_selection_requested.emit("Q1"))
        self.btn_quick_q2.clicked.connect(lambda: self.quick_selection_requested.emit("Q2"))
        self.btn_quick_q3.clicked.connect(lambda: self.quick_selection_requested.emit("Q3"))
        self.btn_quick_q4.clicked.connect(lambda: self.quick_selection_requested.emit("Q4"))
        
        layout.addLayout(row1)
        layout.addLayout(row2)
        
        return gb
    
    def _create_map_type_group(self):
        """Create map type selection group."""
        gb = QGroupBox("Map Type")
        layout = QHBoxLayout(gb)
        layout.setContentsMargins(4, 8, 4, 4)
        
        self.cbb_map_type = QComboBox()
        self.cbb_map_type.addItems(['2Dmap', 'Wafer_300mm', 'Wafer_200mm', 'Wafer_100mm'])
        self.cbb_map_type.currentTextChanged.connect(self.map_type_changed.emit)
        
        self.cb_remove_outlier = QCheckBox("Remove Outlier")
        self.cb_remove_outlier.setChecked(True)
        
        layout.addWidget(self.cbb_map_type)
        layout.addWidget(self.cb_remove_outlier)
        layout.addStretch()
        
        return gb
    
    def _create_xrange_group(self):
        """Create X-range controls."""
        gb = QGroupBox("X-range")
        layout = QHBoxLayout(gb)
        layout.setContentsMargins(4, 8, 4, 4)
        
        self.cb_fix_xrange = QCheckBox("Fix")
        
        self.spin_xmin_map = QDoubleSpinBox()
        self.spin_xmin_map.setRange(-1e9, 1e9)
        self.spin_xmin_map.setDecimals(2)
        self.spin_xmin_map.setValue(0)
        self.spin_xmin_map.setFixedWidth(70)
        
        # Slider placeholder (can be replaced with actual slider later)
        slider_placeholder = QFrame()
        slider_placeholder.setFrameShape(QFrame.Box)
        slider_placeholder.setFixedHeight(20)
        
        self.spin_xmax_map = QDoubleSpinBox()
        self.spin_xmax_map.setRange(-1e9, 1e9)
        self.spin_xmax_map.setDecimals(2)
        self.spin_xmax_map.setValue(100)
        self.spin_xmax_map.setFixedWidth(70)
        
        layout.addWidget(self.cb_fix_xrange)
        layout.addWidget(self.spin_xmin_map)
        layout.addWidget(slider_placeholder, stretch=1)
        layout.addWidget(self.spin_xmax_map)
        
        return gb
    
    def _create_mask_group(self):
        """Create masking controls."""
        gb = QGroupBox("Enable mask")
        layout = QHBoxLayout(gb)
        layout.setContentsMargins(4, 8, 4, 4)
        
        self.cbb_mask_param = QComboBox()
        self.cbb_mask_param.setFixedWidth(100)
        
        self.cbb_mask_operator = QComboBox()
        self.cbb_mask_operator.addItems([">", "<", ">=", "<=", "=="])
        self.cbb_mask_operator.setFixedWidth(50)
        
        self.spin_mask_threshold = QDoubleSpinBox()
        self.spin_mask_threshold.setRange(0, 1e12)
        self.spin_mask_threshold.setDecimals(2)
        self.spin_mask_threshold.setValue(0.00)
        self.spin_mask_threshold.setFixedWidth(80)
        
        layout.addWidget(self.cbb_mask_param)
        layout.addWidget(self.cbb_mask_operator)
        layout.addWidget(self.spin_mask_threshold)
        layout.addStretch()
        
        return gb
    
    def _create_multi_viewer_group(self):
        """Create 'Add more map viewer' controls."""
        gb = QGroupBox("Add more map viewer")
        layout = QHBoxLayout(gb)
        layout.setContentsMargins(4, 8, 4, 4)
        
        self.btn_viewer_2 = QPushButton("2")
        self.btn_viewer_3 = QPushButton("3")
        self.btn_viewer_4 = QPushButton("4")
        
        for btn in [self.btn_viewer_2, self.btn_viewer_3, self.btn_viewer_4]:
            btn.setCheckable(True)
            btn.setFixedSize(40, 28)
            layout.addWidget(btn)
        
        layout.addStretch()
        
        return gb
