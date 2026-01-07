# spectroview/view/components/v_map_viewer.py
"""View component for Map visualization with matplotlib canvas and controls."""
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from superqt import QLabeledDoubleRangeSlider

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QCheckBox, QDoubleSpinBox,
    QFrame, QLineEdit, QToolButton, QMenu, QWidgetAction,
    QSpacerItem, QSizePolicy, QGroupBox
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon, QAction

from spectroview import ICON_DIR
from spectroview.modules.utils import CustomizedPalette


class VMapViewer(QWidget):
    """Map viewer widget with matplotlib canvas and control widgets."""
    
    # ───── View → ViewModel signals ─────
    map_type_changed = Signal(str)
    palette_changed = Signal(int)
    remove_outlier_changed = Signal(bool)
    xrange_changed = Signal(tuple)  # (xmin, xmax)
    zrange_changed = Signal(tuple)  # (zmin, zmax)
    zparameter_changed = Signal(str)  # 'Intensity', 'Area', or fit parameter
    mask_enabled_changed = Signal(bool)
    mask_settings_changed = Signal(dict)  # {param, operator, threshold}
    plot_option_changed = Signal(dict)  # {smoothing, grid, show_stats}
    extract_profile_requested = Signal(str)  # profile name
    multi_viewer_requested = Signal(int)  # number of viewers (2, 3, or 4)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the UI components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)
        
        # Add all UI sections
        main_layout.addWidget(self._create_canvas())
        main_layout.addLayout(self._create_map_type_controls())
        main_layout.addLayout(self._create_z_range_slider())
        main_layout.addLayout(self._create_x_range_slider())
        main_layout.addLayout(self._create_mask_controls())
        main_layout.addWidget(self._create_multi_viewer_controls())
    
    def _create_canvas(self):
        """Create matplotlib canvas and toolbar."""
        canvas_frame = QFrame()
        canvas_layout = QVBoxLayout(canvas_frame)
        canvas_layout.setContentsMargins(2, 2, 2, 2)
        
        # Create matplotlib figure and canvas
        self.figure = plt.figure(dpi=70, figsize=(5, 4))
        self.ax = self.figure.add_subplot(111)
        self.ax.tick_params(axis='both', which='both')
        
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(200)
        self.canvas.setMaximumHeight(350)
        
        # Matplotlib toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        # Hide some toolbar actions
        for action in self.toolbar.actions():
            if action.text() in ['Customize','Zoom','Save', 'Pan', 'Back', 'Forward', 'Subplots']:
                action.setVisible(False)
        
        # Custom toolbar with copy button
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(2, 2, 2, 2)
        toolbar_layout.addWidget(self.toolbar)
        toolbar_layout.addStretch()
        
        self.btn_copy = QPushButton()
        self.btn_copy.setIcon(QIcon(os.path.join(ICON_DIR, "copy.png")))
        self.btn_copy.setIconSize(QSize(20, 20))
        self.btn_copy.setToolTip("Copy figure to clipboard")
        self.btn_copy.setFixedSize(28, 28)
        toolbar_layout.addWidget(self.btn_copy)
        
        canvas_layout.addWidget(self.canvas, stretch=1)
        canvas_layout.addLayout(toolbar_layout)
        
        return canvas_frame
    
    def _create_map_type_controls(self):
        """Create map type and palette selector controls."""
        type_layout = QHBoxLayout()
        type_layout.setContentsMargins(4, 4, 4, 4)
        
        type_layout.addWidget(QLabel("Map Type:"))
        
        self.cbb_map_type = QComboBox()
        self.cbb_map_type.addItems(['2Dmap', 'Wafer_300mm', 'Wafer_200mm', 'Wafer_100mm'])
        self.cbb_map_type.setFixedWidth(120)
        self.cbb_map_type.currentTextChanged.connect(self.map_type_changed.emit)
        type_layout.addWidget(self.cbb_map_type)
        
        self.cbb_palette = CustomizedPalette()
        self.cbb_palette.currentIndexChanged.connect(self.palette_changed.emit)
        type_layout.addWidget(self.cbb_palette)
        #type_layout.addStretch()
        
        return type_layout
    
    def _create_z_range_slider(self):
        """Create Z-range slider with parameter selector."""
        z_slider_layout = QHBoxLayout()
        z_slider_layout.setContentsMargins(4, 2, 4, 2)
        
        self.cbb_zparameter = QComboBox()
        self.cbb_zparameter.addItems(['Intensity', 'Area'])
        self.cbb_zparameter.setFixedWidth(50)
        self.cbb_zparameter.setToolTip("Select parameter to plot in heatmap")
        self.cbb_zparameter.currentTextChanged.connect(self.zparameter_changed.emit)
        z_slider_layout.addWidget(self.cbb_zparameter)
        
        self.cb_fix_z = QCheckBox("Fix")
        self.cb_fix_z.setToolTip("Fix Z-range when changing maps")
        z_slider_layout.addWidget(self.cb_fix_z)
        
        self.spin_zmin = QDoubleSpinBox()
        self.spin_zmin.setRange(-1e9, 1e9)
        self.spin_zmin.setDecimals(2)
        self.spin_zmin.setValue(0)
        self.spin_zmin.setFixedWidth(70)
        z_slider_layout.addWidget(self.spin_zmin)
        
        self.z_range_slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.z_range_slider.setEdgeLabelMode(QLabeledDoubleRangeSlider.EdgeLabelMode.NoLabel)
        self.z_range_slider.setHandleLabelPosition(QLabeledDoubleRangeSlider.LabelPosition.NoLabel)
        self.z_range_slider.setSingleStep(0.01)
        self.z_range_slider.setRange(0, 100)
        self.z_range_slider.setValue((0, 100))
        self.z_range_slider.valueChanged.connect(self._on_z_slider_changed)
        z_slider_layout.addWidget(self.z_range_slider, stretch=1)
        
        self.spin_zmax = QDoubleSpinBox()
        self.spin_zmax.setRange(-1e9, 1e9)
        self.spin_zmax.setDecimals(2)
        self.spin_zmax.setValue(100)
        self.spin_zmax.setFixedWidth(70)
        z_slider_layout.addWidget(self.spin_zmax)
        
        # Connect spinboxes to slider
        self.spin_zmin.valueChanged.connect(self._update_z_slider_from_spins)
        self.spin_zmax.valueChanged.connect(self._update_z_slider_from_spins)
        
        return z_slider_layout
    
    def _create_x_range_slider(self):
        """Create X-range slider controls."""
        x_slider_layout = QHBoxLayout()
        x_slider_layout.setContentsMargins(4, 2, 4, 2)
        
        lbl_x_slider = QLabel("X-range:")
        lbl_x_slider.setFixedWidth(50)
        x_slider_layout.addWidget(lbl_x_slider)
        
        self.cb_fix_x = QCheckBox("Fix")
        self.cb_fix_x.setToolTip("Fix X-range when refreshing")
        x_slider_layout.addWidget(self.cb_fix_x)
        
        self.spin_xmin = QDoubleSpinBox()
        self.spin_xmin.setRange(-1e9, 1e9)
        self.spin_xmin.setDecimals(2)
        self.spin_xmin.setValue(0)
        self.spin_xmin.setFixedWidth(70)
        x_slider_layout.addWidget(self.spin_xmin)
        
        self.x_range_slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.x_range_slider.setEdgeLabelMode(QLabeledDoubleRangeSlider.EdgeLabelMode.NoLabel)
        self.x_range_slider.setHandleLabelPosition(QLabeledDoubleRangeSlider.LabelPosition.NoLabel)
        self.x_range_slider.setSingleStep(0.01)
        self.x_range_slider.setRange(0, 100)
        self.x_range_slider.setValue((0, 100))
        self.x_range_slider.valueChanged.connect(self._on_x_slider_changed)
        x_slider_layout.addWidget(self.x_range_slider, stretch=1)
        
        self.spin_xmax = QDoubleSpinBox()
        self.spin_xmax.setRange(-1e9, 1e9)
        self.spin_xmax.setDecimals(2)
        self.spin_xmax.setValue(100)
        self.spin_xmax.setFixedWidth(70)
        x_slider_layout.addWidget(self.spin_xmax)
        
        # Connect spinboxes to slider
        self.spin_xmin.valueChanged.connect(self._update_x_slider_from_spins)
        self.spin_xmax.valueChanged.connect(self._update_x_slider_from_spins)
        
        return x_slider_layout
    
    def _create_mask_controls(self):
        """Create mask controls and options menu."""
        mask_layout = QHBoxLayout()
        mask_layout.setContentsMargins(4, 2, 4, 2)
        
        self.cb_enable_mask = QCheckBox("Enable mask:")
        self.cb_enable_mask.stateChanged.connect(
            lambda state: self.mask_enabled_changed.emit(state == Qt.Checked)
        )
        mask_layout.addWidget(self.cb_enable_mask)
        
        self.cbb_mask_param = QComboBox()
        self.cbb_mask_param.setFixedWidth(100)
        self.cbb_mask_param.currentTextChanged.connect(self._emit_mask_settings)
        mask_layout.addWidget(self.cbb_mask_param)
        
        self.cbb_mask_operator = QComboBox()
        self.cbb_mask_operator.addItems([">", "<", ">=", "<=", "=="])
        self.cbb_mask_operator.setFixedWidth(50)
        self.cbb_mask_operator.currentTextChanged.connect(self._emit_mask_settings)
        mask_layout.addWidget(self.cbb_mask_operator)
        
        self.spin_mask_threshold = QDoubleSpinBox()
        self.spin_mask_threshold.setRange(0, 1e12)
        self.spin_mask_threshold.setDecimals(2)
        self.spin_mask_threshold.setValue(0.00)
        self.spin_mask_threshold.setFixedWidth(80)
        self.spin_mask_threshold.valueChanged.connect(self._emit_mask_settings)
        mask_layout.addWidget(self.spin_mask_threshold)
        
        mask_layout.addStretch()
        
        # Options menu button
        self.btn_options = QToolButton()
        self.btn_options.setText("...")
        self.btn_options.setPopupMode(QToolButton.InstantPopup)
        self.btn_options.setIcon(QIcon(os.path.join(ICON_DIR, "options.png")))
        self.btn_options.setToolTip("Additional options")
        
        # Create options menu
        self.options_menu = QMenu(self)
        self._create_options_menu()
        self.btn_options.setMenu(self.options_menu)
        
        mask_layout.addWidget(self.btn_options)
        
        return mask_layout
    
    def _create_multi_viewer_controls(self):
        """Create multi-viewer buttons."""
        multi_viewer_gb = QGroupBox()
        multi_viewer_layout = QHBoxLayout(multi_viewer_gb)
        multi_viewer_layout.setContentsMargins(4, 6, 4, 4)
        
        multi_viewer_layout.addWidget(QLabel("Add more map viewer:"))
        
        self.btn_viewer_2 = QPushButton("2")
        self.btn_viewer_3 = QPushButton("3")
        self.btn_viewer_4 = QPushButton("4")
        
        for btn in [self.btn_viewer_2, self.btn_viewer_3, self.btn_viewer_4]:
            btn.setCheckable(True)
            btn.setFixedSize(40, 26)
            multi_viewer_layout.addWidget(btn)
        
        # Connect buttons to signal with viewer count
        self.btn_viewer_2.clicked.connect(lambda: self.multi_viewer_requested.emit(2))
        self.btn_viewer_3.clicked.connect(lambda: self.multi_viewer_requested.emit(3))
        self.btn_viewer_4.clicked.connect(lambda: self.multi_viewer_requested.emit(4))
        
        multi_viewer_layout.addStretch()
        
        return multi_viewer_gb
    
    def _create_options_menu(self):
        """Create options menu with smoothing, grid, stats, and profile extraction."""
        # Profile extraction widget
        profile_widget = QWidget()
        profile_layout = QHBoxLayout(profile_widget)
        profile_layout.setContentsMargins(5, 5, 5, 5)
        
        self.txt_profile_name = QLineEdit()
        self.txt_profile_name.setText("Profile_1")
        self.txt_profile_name.setPlaceholderText("Profile name...")
        self.txt_profile_name.setFixedWidth(150)
        
        self.btn_extract_profile = QPushButton("Extract profile")
        self.btn_extract_profile.setFixedWidth(110)
        self.btn_extract_profile.clicked.connect(
            lambda: self.extract_profile_requested.emit(self.txt_profile_name.text())
        )
        
        profile_layout.addWidget(self.txt_profile_name)
        profile_layout.addWidget(self.btn_extract_profile)
        
        profile_action = QWidgetAction(self)
        profile_action.setDefaultWidget(profile_widget)
        self.options_menu.addAction(profile_action)
        
        self.options_menu.addSeparator()
        
        # Checkable options
        self.action_remove_outlier = QAction("Remove Outlier", self)
        self.action_remove_outlier.setCheckable(True)
        self.action_remove_outlier.setChecked(True)
        self.action_remove_outlier.triggered.connect(
            lambda checked: self.remove_outlier_changed.emit(checked)
        )
        self.options_menu.addAction(self.action_remove_outlier)
        
        self.action_smoothing = QAction("Smoothing", self)
        self.action_smoothing.setCheckable(True)
        self.action_smoothing.setChecked(False)
        self.action_smoothing.triggered.connect(self._emit_plot_options)
        self.options_menu.addAction(self.action_smoothing)
        
        self.action_grid = QAction("Grid", self)
        self.action_grid.setCheckable(True)
        self.action_grid.setChecked(False)
        self.action_grid.triggered.connect(self._emit_plot_options)
        self.options_menu.addAction(self.action_grid)
        
        self.action_show_stats = QAction("Show stats", self)
        self.action_show_stats.setCheckable(True)
        self.action_show_stats.setChecked(True)
        self.action_show_stats.triggered.connect(self._emit_plot_options)
        self.options_menu.addAction(self.action_show_stats)
    
    # ───── Public API (ViewModel → View) ─────────────────────────
    def set_map_type(self, map_type: str):
        """Set map type selection."""
        self.cbb_map_type.setCurrentText(map_type)
    
    def set_xrange(self, xmin: float, xmax: float):
        """Set X-range slider and spinboxes."""
        self.x_range_slider.setRange(xmin, xmax)
        self.x_range_slider.setValue((xmin, xmax))
        self.spin_xmin.setValue(xmin)
        self.spin_xmax.setValue(xmax)
    
    def set_zrange(self, zmin: float, zmax: float):
        """Set Z-range slider and spinboxes."""
        self.z_range_slider.setRange(zmin, zmax)
        self.z_range_slider.setValue((zmin, zmax))
        self.spin_zmin.setValue(zmin)
        self.spin_zmax.setValue(zmax)
    
    def add_zparameters(self, parameters: list):
        """Add fit parameters to Z-value combobox."""
        current = self.cbb_zparameter.currentText()
        self.cbb_zparameter.clear()
        self.cbb_zparameter.addItems(['Intensity', 'Area'] + parameters)
        if current in ['Intensity', 'Area'] + parameters:
            self.cbb_zparameter.setCurrentText(current)
    
    def set_mask_parameters(self, parameters: list):
        """Set available mask parameters."""
        self.cbb_mask_param.clear()
        self.cbb_mask_param.addItems(parameters)
    
    # ───── Internal signal handlers ──────────────────────────────
    def _on_x_slider_changed(self, values):
        """X-range slider changed - update spinboxes and emit signal."""
        self.spin_xmin.blockSignals(True)
        self.spin_xmax.blockSignals(True)
        self.spin_xmin.setValue(values[0])
        self.spin_xmax.setValue(values[1])
        self.spin_xmin.blockSignals(False)
        self.spin_xmax.blockSignals(False)
        self.xrange_changed.emit(values)
    
    def _on_z_slider_changed(self, values):
        """Z-range slider changed - update spinboxes and emit signal."""
        self.spin_zmin.blockSignals(True)
        self.spin_zmax.blockSignals(True)
        self.spin_zmin.setValue(values[0])
        self.spin_zmax.setValue(values[1])
        self.spin_zmin.blockSignals(False)
        self.spin_zmax.blockSignals(False)
        self.zrange_changed.emit(values)
    
    def _update_x_slider_from_spins(self):
        """Update X slider when spinboxes change."""
        xmin = self.spin_xmin.value()
        xmax = self.spin_xmax.value()
        if xmin <= xmax:
            self.x_range_slider.setValue((xmin, xmax))
    
    def _update_z_slider_from_spins(self):
        """Update Z slider when spinboxes change."""
        zmin = self.spin_zmin.value()
        zmax = self.spin_zmax.value()
        if zmin <= zmax:
            self.z_range_slider.setValue((zmin, zmax))
    
    def _emit_mask_settings(self):
        """Emit mask settings signal."""
        settings = {
            'parameter': self.cbb_mask_param.currentText(),
            'operator': self.cbb_mask_operator.currentText(),
            'threshold': self.spin_mask_threshold.value()
        }
        self.mask_settings_changed.emit(settings)
    
    def _emit_plot_options(self):
        """Emit plot options signal."""
        options = {
            'smoothing': self.action_smoothing.isChecked(),
            'grid': self.action_grid.isChecked(),
            'show_stats': self.action_show_stats.isChecked()
        }
        self.plot_option_changed.emit(options)
