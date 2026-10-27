# spectroview/view/components/v_map_viewer.py
"""View component for Map visualization with matplotlib canvas and controls."""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from superqt import QLabeledDoubleRangeSlider

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QCheckBox, QDoubleSpinBox,
    QFrame, QLineEdit, QToolButton, QMenu, QWidgetAction, QApplication, QMessageBox,
    QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QSize, QTimer
from PySide6.QtGui import QAction

from spectroview import PLOT_POLICY_LIGHT, PLOT_POLICY_DARK
from spectroview.view.theme import Icons, current_tokens
from spectroview.viewmodel.utils import copy_fig_to_clb
from spectroview.view.components.customized_widgets import CustomizedPalette


class VMapViewer(QWidget):
    """Map viewer widget with matplotlib canvas and control widgets."""
    
    # ───── View → ViewModel signals ─────
    spectra_selected = Signal(list)  # List of (x, y) tuples for selected spectra
    extract_profile_requested = Signal(str)  # profile name
    multi_viewer_requested = Signal(int) # Add more map_viewer
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Data state
        self.map_df = None
        self.map_df_name = None
        self.df_fit_results = None
        self.selected_points = []
        self.number_of_points = 0
        self.current_style = 'Light Mode'
        
        # Rectangle selection state
        self.rect_start = None
        self.rect_patch = None
        
        # Plot artifacts
        self.img = None
        self.cbar = None
        self._last_final_z_col = None
        self._last_stats_text_artist = None
        self._profile_line = None  # Profile line connecting 2 points
        self._profile_height_line = None  # Profile height visualization
        
        # Cache for expensive griddata computation (wafer maps)
        self._griddata_cache = {}  # {cache_key: (heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col)}
        
        # Debounce timer for plot updates (matches legacy 100ms delay)
        self.plot_timer = QTimer(self)
        self.plot_timer.setSingleShot(True)
        self.plot_timer.setInterval(100)
        self.plot_timer.timeout.connect(self._do_plot_heatmap)
        
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
    
    def _create_canvas(self):
        """Create matplotlib canvas and toolbar."""
        canvas_frame = QFrame()
        canvas_frame.setStyleSheet("QFrame { border: none; background: transparent; }")
        canvas_layout = QVBoxLayout(canvas_frame)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create matplotlib figure and canvas
        with plt.style.context(PLOT_POLICY_LIGHT):
            self.figure = plt.figure(dpi=70, figsize=(5, 4))
            self.ax = self.figure.add_subplot(111)
            self.ax.tick_params(axis='both', which='both')
        
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Connect mouse events for interactive selection
        self.canvas.mpl_connect('button_press_event', self._on_mouse_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('figure_leave_event', self._on_figure_leave)
        
        # Matplotlib toolbar kept hidden as a background interaction helper for pan/zoom/rescale
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.hide()

        # Dedicated styled navigation buttons (Home/Rescale, Pan/Drag, Zoom) matching SPECTROview controls
        self.btn_home = QPushButton()
        self.btn_home.setIcon(Icons.rescale())
        self.btn_home.setIconSize(QSize(20, 20))
        self.btn_home.setFixedSize(28, 28)
        self.btn_home.setToolTip("Rescale / Reset original view (Ctrl+R)")
        self.btn_home.clicked.connect(self.toolbar.home)
        self.btn_rescale = self.btn_home  # Alias matching v_spectra_viewer

        self.btn_pan = QToolButton()
        self.btn_pan.setCheckable(True)
        self.btn_pan.setIcon(Icons.pan())
        self.btn_pan.setIconSize(QSize(20, 20))
        self.btn_pan.setFixedSize(28, 28)
        self.btn_pan.setToolTip("Pan / Drag axes with left mouse, zoom with right")
        self.btn_pan.clicked.connect(self.toolbar.pan)

        self.btn_zoom = QToolButton()
        self.btn_zoom.setCheckable(True)
        self.btn_zoom.setIcon(Icons.zoom())
        self.btn_zoom.setIconSize(QSize(20, 20))
        self.btn_zoom.setFixedSize(28, 28)
        self.btn_zoom.setToolTip("Zoom to rectangle")
        self.btn_zoom.clicked.connect(self.toolbar.zoom)

        # Bidirectional sync with matplotlib toolbar action checked states
        action_dict = {a.text(): a for a in self.toolbar.actions() if a.text()}
        if 'Pan' in action_dict:
            action_dict['Pan'].toggled.connect(self.btn_pan.setChecked)
        if 'Zoom' in action_dict:
            action_dict['Zoom'].toggled.connect(self.btn_zoom.setChecked)

        # Forward any matplotlib toolbar message events to our built-in coordinate label
        def _on_toolbar_message(s):
            clean_s = " ".join(s.split()) if s else ""
            if hasattr(self, "lbl_coordinates"):
                self.lbl_coordinates.setText(clean_s)
            if hasattr(self.toolbar, "locLabel"):
                self.toolbar.locLabel.setText(s)

        self.toolbar.set_message = _on_toolbar_message

        # Custom toolbar with navigation, built-in coordinates, add viewer, and copy buttons
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(2, 2, 2, 2)
        toolbar_layout.setSpacing(6)
        toolbar_layout.addWidget(self.btn_home)
        toolbar_layout.addWidget(self.btn_pan)
        toolbar_layout.addWidget(self.btn_zoom)
        toolbar_layout.addStretch()

        # Built-in live coordinates display
        self.lbl_coordinates = QLabel("")
        self.lbl_coordinates.setObjectName("lblCoordinates")
        self.lbl_coordinates.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_coordinates.setMinimumWidth(180)
        self.lbl_coordinates.setStyleSheet("font-size: 11px; padding: 0px 4px;")
        toolbar_layout.addWidget(self.lbl_coordinates)

        # Vertical separator between coordinates and action buttons
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        toolbar_layout.addWidget(separator)
        
        # Add viewer button
        self.btn_add_viewer = QPushButton()
        self.btn_add_viewer.setIcon(Icons.plus())
        self.btn_add_viewer.setIconSize(QSize(20, 20))
        self.btn_add_viewer.setToolTip("Open a new map viewer window")
        self.btn_add_viewer.setFixedSize(28, 28)
        self.btn_add_viewer.clicked.connect(lambda: self.multi_viewer_requested.emit(1))
        toolbar_layout.addWidget(self.btn_add_viewer)
        
        # Copy button
        self.btn_copy = QPushButton()
        self.btn_copy.setIcon(Icons.copy())
        self.btn_copy.setIconSize(QSize(20, 20))
        self.btn_copy.setToolTip("Copy figure to clipboard")
        self.btn_copy.setFixedSize(28, 28)
        self.btn_copy.clicked.connect(self._copy_figure_to_clipboard)
        toolbar_layout.addWidget(self.btn_copy)
        
        self.canvas_card = QFrame(canvas_frame)
        self.canvas_card.setObjectName("canvasCard")
        self.canvas_card.setFrameShape(QFrame.NoFrame)
        card_layout = QVBoxLayout(self.canvas_card)
        card_layout.setContentsMargins(5, 5, 5, 5)
        card_layout.setSpacing(0)
        card_layout.addWidget(self.canvas)
        canvas_layout.addWidget(self.canvas_card, stretch=1)
        canvas_layout.addLayout(toolbar_layout)
        
        return canvas_frame
    
    def apply_plot_style(self, style_name=None):
        """Update figure and axes colors using local style context."""
        if not hasattr(self, 'figure') or not hasattr(self, 'ax'):
            return
            
        if style_name is not None:
            self.current_style = style_name
            
        # Manually replot heatmap to update colorbars and contents with new style
        if self.map_df is None or self.map_df.empty:
            self._do_plot_heatmap()
        else:
            self.plot_heatmap()

    def apply_global_theme(self, global_theme: str):
        """Update toolbar icons based on the global application theme."""
        tok = current_tokens()
        icon_color = tok.text_primary
        if hasattr(self, 'btn_home'):
            self.btn_home.setIcon(Icons.rescale(color=icon_color))
        if hasattr(self, 'btn_pan'):
            self.btn_pan.setIcon(Icons.pan(color=icon_color))
        if hasattr(self, 'btn_zoom'):
            self.btn_zoom.setIcon(Icons.zoom(color=icon_color))
        if hasattr(self, 'btn_copy'):
            self.btn_copy.setIcon(Icons.copy(color=icon_color))
        if hasattr(self, 'btn_add_viewer'):
            self.btn_add_viewer.setIcon(Icons.plus(color=icon_color))
        if hasattr(self, 'btn_options'):
            self.btn_options.setIcon(Icons.settings(color=icon_color))
        if hasattr(self, 'lbl_coordinates'):
            self.lbl_coordinates.setStyleSheet(f"font-size: 11px; padding: 0px 4px; color: {tok.text_secondary};")

    def _create_map_type_controls(self):
        """Create map type and palette selector controls."""
        type_layout = QHBoxLayout()
        type_layout.setContentsMargins(2, 2, 2, 2)
        
        lbl = QLabel("Select map type:")
        type_layout.addWidget(lbl)
        self.cbb_map_type = QComboBox()
        self.cbb_map_type.addItems(['2Dmap', 'Wafer_300mm', 'Wafer_200mm', 'Wafer_100mm'])
        self.cbb_map_type.setFixedWidth(120)
        self.cbb_map_type.currentTextChanged.connect(lambda: self.plot_heatmap())
        type_layout.addWidget(self.cbb_map_type)
        
        self.cbb_palette = CustomizedPalette()
        self.cbb_palette.currentIndexChanged.connect(lambda: self.plot_heatmap())
        type_layout.addWidget(self.cbb_palette)
        
        return type_layout
    
    def _create_z_range_slider(self):
        """Create Z-range slider with parameter selector."""
        z_slider_layout = QHBoxLayout()
        z_slider_layout.setContentsMargins(2, 2, 2, 2)
        
        self.cbb_zparameter = QComboBox()
        self.cbb_zparameter.addItems(['Intensity', 'Area'])
        self.cbb_zparameter.setFixedWidth(68)
        self.cbb_zparameter.setToolTip("Select parameter to plot in heatmap")
        self.cbb_zparameter.currentTextChanged.connect(lambda: self._on_parameter_changed())
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
        x_slider_layout.setContentsMargins(2, 2, 2, 2)
        
        lbl_x_slider = QLabel("X-range:")
        lbl_x_slider.setFixedWidth(68)
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
        mask_layout.setContentsMargins(2, 2, 2, 2)
        
        self.cb_enable_mask = QCheckBox("Enable mask:")
        self.cb_enable_mask.stateChanged.connect(lambda: self.plot_heatmap())
        mask_layout.addWidget(self.cb_enable_mask)
        
        self.cbb_mask_param = QComboBox()
        self.cbb_mask_param.setFixedWidth(100)
        self.cbb_mask_param.currentTextChanged.connect(lambda: self.plot_heatmap())
        mask_layout.addWidget(self.cbb_mask_param)
        
        self.cbb_mask_operator = QComboBox()
        self.cbb_mask_operator.addItems([">", "<", ">=", "<=", "=="])
        self.cbb_mask_operator.setFixedWidth(50)
        self.cbb_mask_operator.currentTextChanged.connect(lambda: self.plot_heatmap())
        mask_layout.addWidget(self.cbb_mask_operator)
        
        self.spin_mask_threshold = QDoubleSpinBox()
        self.spin_mask_threshold.setRange(0, 1e12)
        self.spin_mask_threshold.setDecimals(2)
        self.spin_mask_threshold.setValue(0.00)
        self.spin_mask_threshold.setFixedWidth(80)
        self.spin_mask_threshold.valueChanged.connect(lambda: self.plot_heatmap())
        mask_layout.addWidget(self.spin_mask_threshold)
        
        mask_layout.addStretch()
        
        # Options menu button
        self.btn_options = QToolButton()
        self.btn_options.setText("...")
        self.btn_options.setPopupMode(QToolButton.InstantPopup)
        self.btn_options.setIcon(Icons.settings())
        self.btn_options.setToolTip("Additional options")
        
        # Create options menu
        self.options_menu = QMenu(self)
        self._create_options_menu()
        self.btn_options.setMenu(self.options_menu)
        
        mask_layout.addWidget(self.btn_options)
        
        return mask_layout
    
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
        self.action_remove_outlier.triggered.connect(lambda: self._on_plot_option_changed())
        self.options_menu.addAction(self.action_remove_outlier)
        
        self.action_smoothing = QAction("Smoothing", self)
        self.action_smoothing.setCheckable(True)
        self.action_smoothing.setChecked(False)
        self.action_smoothing.triggered.connect(lambda: self.plot_heatmap())
        self.options_menu.addAction(self.action_smoothing)
        
        self.action_grid = QAction("Grid", self)
        self.action_grid.setCheckable(True)
        self.action_grid.setChecked(False)
        self.action_grid.triggered.connect(lambda: self.plot_heatmap())
        self.options_menu.addAction(self.action_grid)
        
        self.action_show_stats = QAction("Show stats", self)
        self.action_show_stats.setCheckable(True)
        self.action_show_stats.setChecked(False)
        self.action_show_stats.triggered.connect(lambda: self.plot_heatmap())
        self.options_menu.addAction(self.action_show_stats)

        self.options_menu.addSeparator()

        # Color-scale selector: linear vs log color normalization for the heatmap.
        scale_widget = QWidget()
        scale_layout = QHBoxLayout(scale_widget)
        scale_layout.setContentsMargins(5, 5, 5, 5)
        scale_layout.addWidget(QLabel("Color scale:"))
        self.cbb_color_scale = QComboBox()
        self.cbb_color_scale.addItems(["Linear", "Log"])
        self.cbb_color_scale.setToolTip(
            "Color-scale normalization for the heatmap (log needs positive values)"
        )
        self.cbb_color_scale.currentTextChanged.connect(lambda: self.plot_heatmap())
        scale_layout.addWidget(self.cbb_color_scale)
        scale_layout.addStretch()

        scale_action = QWidgetAction(self)
        scale_action.setDefaultWidget(scale_widget)
        self.options_menu.addAction(scale_action)
    
    # ═══ Public API (ViewModel → View) ═══
    
    def set_map_data(self, map_df, map_name, df_fit_results=None, selected_points=None):
        """Set map data and schedule plot update with debounce."""
        is_map_change = (map_name != getattr(self, "map_df_name", None))

        self.map_df = map_df
        self.map_df_name = map_name
        self.df_fit_results = df_fit_results if df_fit_results is not None else pd.DataFrame()
        
        # Cache coordinate arrays for fast reuse across plots and mouse interactions
        if self.map_df is not None and not self.map_df.empty:
            col_x = 'X' if 'X' in self.map_df.columns else self.map_df.columns[0]
            col_y = 'Y' if 'Y' in self.map_df.columns else self.map_df.columns[1]
            self._sites_x = self.map_df[col_x].values.astype(float)
            self._sites_y = self.map_df[col_y].values.astype(float)
            self.number_of_points = len(self._sites_x)
        else:
            self._sites_x = np.array([], dtype=float)
            self._sites_y = np.array([], dtype=float)
            self.number_of_points = 0

        # Auto-select the first coordinate on map open or switch
        if selected_points is not None:
            self.selected_points = list(dict.fromkeys(selected_points))
        elif is_map_change or not self.selected_points:
            if self.number_of_points > 0:
                first_x = float(self._sites_x[0])
                first_y = float(self._sites_y[0])
                self.selected_points = [(first_x, first_y)]
            else:
                self.selected_points = []
        else:
            # Same map: retain valid selected points
            if self.number_of_points > 0:
                valid_coords = set(zip(np.round(self._sites_x, 4), np.round(self._sites_y, 4)))
                self.selected_points = [
                    pt for pt in self.selected_points
                    if (round(float(pt[0]), 4), round(float(pt[1]), 4)) in valid_coords
                ]
                if not self.selected_points:
                    first_x = float(self._sites_x[0])
                    first_y = float(self._sites_y[0])
                    self.selected_points = [(first_x, first_y)]
            else:
                self.selected_points = []
        
        # Note: griddata cache is NOT cleared here - it persists across map switches
        # for faster map switching. Cache is only cleared when data changes (after fitting).
        
        # Update available parameters
        self._update_parameter_lists()
        
        # Update sliders based on new data
        self._update_range_sliders()
        
        # Schedule plot with debounce (matches legacy performance)
        self.plot_timer.start()
    
    def set_selected_points(self, points: list):
        """Set externally selected points and refresh plot."""
        # Ensure uniqueness while preserving order
        self.selected_points = list(dict.fromkeys(points))
        self._update_selection_overlay()
    
    def clear_cache_for_map(self, map_name: str):
        """Clear cached griddata for specific map (e.g., after fitting changes data).
        
        Args:
            map_name: Name of the map whose cache entries should be cleared
        """
        keys_to_remove = [k for k in self._griddata_cache if k[0] == map_name]
        for key in keys_to_remove:
            del self._griddata_cache[key]
    
    def _update_parameter_lists(self):
        """Update z-parameter and mask parameter lists from fit results."""
        self.cbb_zparameter.blockSignals(True)
        self.cbb_mask_param.blockSignals(True)
        
        current_z = self.cbb_zparameter.currentText()
        current_mask = self.cbb_mask_param.currentText()
        
        # Update Z parameters
        self.cbb_zparameter.clear()
        self.cbb_zparameter.addItems(['Intensity', 'Area'])
        self.cbb_mask_param.clear()
        
        if not self.df_fit_results.empty:
            fit_columns = [col for col in self.df_fit_results.columns 
                          if col not in ['Filename', 'X', 'Y']]
            self.cbb_zparameter.addItems(fit_columns)
            self.cbb_mask_param.addItems(fit_columns)
        
        # Restore selection if possible
        if current_z and self.cbb_zparameter.findText(current_z) >= 0:
            self.cbb_zparameter.setCurrentText(current_z)
        else:
            self.cbb_zparameter.setCurrentIndex(0)
            
        if current_mask and self.cbb_mask_param.findText(current_mask) >= 0:
            self.cbb_mask_param.setCurrentText(current_mask)
        elif self.cbb_mask_param.count() > 0:
            self.cbb_mask_param.setCurrentIndex(0)
        
        self.cbb_zparameter.blockSignals(False)
        self.cbb_mask_param.blockSignals(False)
    
    def _update_range_sliders(self):
        """Update X and Z range sliders based on current data."""
        if self.map_df is None or self.map_df.empty:
            return
        
        # X-range: based on wavenumber columns
        column_labels = self.map_df.columns[2:]
        try:
            numeric_cols = column_labels.astype(float)
            xmin, xmax = float(numeric_cols.min()), float(numeric_cols.max())
            
            # Update X-range slider with blocked signals to avoid cascade
            if not self.cb_fix_x.isChecked():
                self.x_range_slider.blockSignals(True)
                self.spin_xmin.blockSignals(True)
                self.spin_xmax.blockSignals(True)
                try:
                    self.x_range_slider.setRange(xmin, xmax)
                    self.x_range_slider.setValue((xmin, xmax))
                    self.spin_xmin.setValue(xmin)
                    self.spin_xmax.setValue(xmax)
                finally:
                    self.x_range_slider.blockSignals(False)
                    self.spin_xmin.blockSignals(False)
                    self.spin_xmax.blockSignals(False)
        except:
            pass
        
        # Z-range: computed from current parameter
        self._update_z_range()
    
    def _update_z_range(self):
        """Update Z-range slider based on current parameter selection."""
        if self.map_df is None:
            return
        
        try:
            _, _, vmin, vmax, _, _ = self._get_data_for_heatmap()
            
            # Update Z-range slider with blocked signals to avoid cascade
            self.z_range_slider.blockSignals(True)
            self.spin_zmin.blockSignals(True)
            self.spin_zmax.blockSignals(True)
            try:
                if not self.cb_fix_z.isChecked():
                    self.z_range_slider.setRange(vmin, vmax)
                    self.z_range_slider.setValue((vmin, vmax))
                    self.spin_zmin.setValue(vmin)
                    self.spin_zmax.setValue(vmax)
                else:
                    # Just update the range, keep current values
                    self.z_range_slider.setRange(vmin, vmax)
            finally:
                self.z_range_slider.blockSignals(False)
                self.spin_zmin.blockSignals(False)
                self.spin_zmax.blockSignals(False)
        except:
            pass
    
    # ═══ Internal signal handlers ═══
    
    def _on_parameter_changed(self):
        """Z parameter changed - update range and refresh plot."""
        self._update_z_range()
        if self.map_df is None or self.map_df.empty:
            self._do_plot_heatmap()
        else:
            self.plot_heatmap()
    
    def _on_plot_option_changed(self):
        """Plot option changed - update range if needed and refresh."""
        self._update_z_range()
        if self.map_df is None or self.map_df.empty:
            self._do_plot_heatmap()
        else:
            self.plot_heatmap()
    
    def _on_x_slider_changed(self, values):
        """X-range slider changed - update spinboxes and schedule debounced plot."""
        self.spin_xmin.blockSignals(True)
        self.spin_xmax.blockSignals(True)
        self.spin_xmin.setValue(values[0])
        self.spin_xmax.setValue(values[1])
        self.spin_xmin.blockSignals(False)
        self.spin_xmax.blockSignals(False)
        self._need_z_range_update = True
        if self.map_df is None or self.map_df.empty:
            self._do_plot_heatmap()
        else:
            self.plot_heatmap()
    
    def _on_z_slider_changed(self, values):
        """Z-range slider changed - update spinboxes and trigger plot."""
        self.spin_zmin.blockSignals(True)
        self.spin_zmax.blockSignals(True)
        self.spin_zmin.setValue(values[0])
        self.spin_zmax.setValue(values[1])
        self.spin_zmin.blockSignals(False)
        self.spin_zmax.blockSignals(False)
        if self.map_df is None or self.map_df.empty:
            self._do_plot_heatmap()
        else:
            self.plot_heatmap()
    
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
    
    # ═══ Core plotting methods ═══
    
    def _update_selection_overlay(self):
        """Update selection overlay without full plot redraw (fast).
        
        Uses PatchCollection for O(1) rendering instead of individual add_patch() calls.
        """
        # Remove old selection collection
        if hasattr(self, '_selection_collection') and self._selection_collection is not None:
            try:
                self._selection_collection.remove()
            except:
                pass
            self._selection_collection = None
        
        # Remove old profile lines
        if self._profile_line is not None:
            try:
                self._profile_line.remove()
            except:
                pass
            self._profile_line = None
        
        if self._profile_height_line is not None:
            try:
                self._profile_height_line.remove()
            except:
                pass
            self._profile_height_line = None
        
        # Add new selection overlay
        if self.selected_points:
            try:
                # Use stored dx/dy if available, otherwise estimate or default
                if hasattr(self, '_current_dx') and hasattr(self, '_current_dy'):
                    dx, dy = self._current_dx, self._current_dy
                else:
                    # Fallback if plot hasn't updated yet
                    dx, dy = 1.0, 1.0
                
                # Build all rectangles then add as single PatchCollection (10-50x faster)
                rects = [
                    patches.Rectangle((x - dx/2, y - dy/2), dx, dy)
                    for x, y in self.selected_points
                ]
                collection = PatchCollection(
                    rects, linewidth=1.2, edgecolor='red',
                    facecolor='none', zorder=10
                )
                self.ax.add_collection(collection)
                self._selection_collection = collection
                
                # Plot profile if exactly 2 points selected and in 2D map mode
                map_type = self.cbb_map_type.currentText()
                if len(self.selected_points) == 2 and map_type == '2Dmap':
                    self._plot_height_profile_on_map()
                    
            except Exception as e:
                print(f"Error drawing selection overlay: {e}")
        
        self.canvas.draw_idle()
    
    def plot_heatmap(self):
        """Schedule a debounced heatmap plot update."""
        self.plot_timer.start()
    
    def _do_plot_heatmap(self):
        """Wrapper to plot using local style context, isolating from global rcParams."""
        if self.current_style == 'Dark Mode':
            style_path = PLOT_POLICY_DARK
        else:
            style_path = PLOT_POLICY_LIGHT
        with plt.style.context(style_path):
            self._do_plot_heatmap_internal()

    def _do_plot_heatmap_internal(self):
        """Plot 2D heatmap or wafer map based on current data and settings."""
        if self.map_df is None or self.map_df.empty:
            self.figure.clf()
            fig_bg = plt.rcParams.get('figure.facecolor', 'white')
            self.figure.patch.set_facecolor(fig_bg)
            if hasattr(self, "canvas_card"):
                tok = current_tokens()
                fig_hex = fig_bg if str(fig_bg).startswith("#") or str(fig_bg).isalpha() else f"#{fig_bg}"
                self.canvas_card.setStyleSheet(
                    f"QFrame#canvasCard {{ background-color: {fig_hex}; border: 1px solid {tok.border}; border-radius: 8px; }}"
                )
            self.ax = self.figure.add_subplot(111)
            self.cbar = None
            self.canvas.draw_idle()
            return
        
        map_type = self.cbb_map_type.currentText()
        show_stats = self.action_show_stats.isChecked() if map_type != '2Dmap' else False
        show_grid = self.action_grid.isChecked()

        if getattr(self, '_need_z_range_update', False):
            self._need_z_range_update = False
            self._update_z_range()

        cache_key = self._build_cache_key()
        last_rendered_key = getattr(self, '_last_rendered_cache_key', None)

        # Fast path: update existing Artists in-place without rebuilding Figure / tight_layout
        can_fast_update = (
            getattr(self, 'ax', None) is not None
            and getattr(self, 'img', None) is not None
            and getattr(self, 'cbar', None) is not None
            and getattr(self, '_last_map_type', None) == map_type
            and getattr(self, '_last_show_stats', None) == show_stats
            and getattr(self, '_last_grid', None) == show_grid
            and getattr(self, '_last_style', None) == self.current_style
            and not show_stats  # full redraw if stats box is enabled
        )

        if can_fast_update:
            try:
                vmin_plot, vmax_plot = self.z_range_slider.value()

                # ULTRA-FAST PATH: if heatmap data parameters did not change,
                # only update clim/norm and colorbar without touching any data or collections
                if last_rendered_key == cache_key:
                    if self.cbb_color_scale.currentText() == "Log" and vmin_plot is not None and vmin_plot > 0:
                        self.img.set_norm(LogNorm(vmin=vmin_plot, vmax=vmax_plot))
                    else:
                        self.img.set_norm(None)
                        self.img.set_clim(vmin_plot, vmax_plot)
                    self.cbar.update_normal(self.img)
                    self.canvas.draw_idle()
                    return

                heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col = self._get_data_for_heatmap()
                self._last_final_z_col = final_z_col
                self._last_rendered_cache_key = cache_key

                cmap = plt.get_cmap(self.cbb_palette.currentText()).copy()
                cmap.set_bad(color='none')
                interpolation = 'bilinear' if self.action_smoothing.isChecked() else 'none'

                if map_type != '2Dmap' and grid_z is not None:
                    img_data = grid_z
                    width = extent[1] - extent[0]
                    height = extent[3] - extent[2]
                    self._current_dx = width / 80.0
                    self._current_dy = height / 80.0
                    if hasattr(self, '_scatter_sites') and self._scatter_sites is not None:
                        all_x, all_y = self._get_measurement_sites()
                        if len(all_x) > 0:
                            self._scatter_sites.set_offsets(np.column_stack([all_x, all_y]))
                elif not heatmap_pivot.empty:
                    img_data = heatmap_pivot.values if hasattr(heatmap_pivot, 'values') else np.asarray(heatmap_pivot)
                    rows, cols = heatmap_pivot.shape
                    if cols > 0 and rows > 0:
                        self._current_dx = (extent[1] - extent[0]) / cols
                        self._current_dy = (extent[3] - extent[2]) / rows
                    else:
                        self._current_dx = 1.0
                        self._current_dy = 1.0
                else:
                    img_data = None

                if img_data is not None:
                    self.img.set_cmap(cmap)
                    self.img.set_interpolation(interpolation)
                    if self.cbb_color_scale.currentText() == "Log" and vmin_plot is not None and vmin_plot > 0:
                        self.img.set_norm(LogNorm(vmin=vmin_plot, vmax=vmax_plot))
                    else:
                        self.img.set_norm(None)
                        self.img.set_clim(vmin_plot, vmax_plot)
                    self.img.set_data(img_data)
                    self.img.set_extent(extent)
                    self.ax.set_xlim(extent[0], extent[1])
                    self.ax.set_ylim(extent[2], extent[3])
                    self.cbar.update_normal(self.img)
                    title = self.cbb_zparameter.currentText()
                    self.ax.set_title(title, fontsize=13)
                    self._update_selection_overlay()
                    self.canvas.draw_idle()
                    return
            except Exception:
                # If anything fails during fast path, seamlessly fall through to full redraw
                pass
        
        # Clear figure and recreate axes to ensure no leftover colorbars and reset sizes
        self.figure.clf()
        fig_bg = plt.rcParams.get('figure.facecolor', 'white')
        self.figure.patch.set_facecolor(fig_bg)
        if hasattr(self, "canvas_card"):
            tok = current_tokens()
            fig_hex = fig_bg if str(fig_bg).startswith("#") or str(fig_bg).isalpha() else f"#{fig_bg}"
            self.canvas_card.setStyleSheet(
                f"QFrame#canvasCard {{ background-color: {fig_hex}; border: 1px solid {tok.border}; border-radius: 8px; }}"
            )
        self.ax = self.figure.add_subplot(111)
        self.cbar = None
        self._last_map_type = map_type
        self._last_show_stats = show_stats
        self._last_grid = show_grid
        self._last_style = self.current_style
        self._last_rendered_cache_key = cache_key
        self._wafer_circle = None
        self._scatter_sites = None

        # Plot wafer circle for wafer maps
        if map_type != '2Dmap':
            r = self._get_wafer_radius(map_type)
            if r:
                edge_color = plt.rcParams.get('axes.edgecolor', 'black')
                self._wafer_circle = patches.Circle((0, 0), radius=r, fill=False, 
                                             color=edge_color, linewidth=1)
                self.ax.add_patch(self._wafer_circle)
                
                # Show all measurement sites
                all_x, all_y = self._get_measurement_sites()
                self._scatter_sites = self.ax.scatter(all_x, all_y, marker='x', color='gray', s=15)
                
                # Remove X labels for wafer, keep Y
                self.ax.tick_params(axis='x', which='both', bottom=False, 
                                   top=False, labelbottom=False)
                self.ax.tick_params(axis='y', which='both', left=True, 
                                   right=False, labelleft=True)
        else:
            # 2D map: show both axes
            self.ax.tick_params(axis='x', which='both', bottom=True, 
                               top=False, labelbottom=True)
            self.ax.tick_params(axis='y', which='both', left=True, 
                               right=False, labelleft=True)
        
        # Get heatmap data and plot (consolidated error handling)
        try:
            # Get heatmap data
            heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col = self._get_data_for_heatmap()
            self._last_final_z_col = final_z_col
        
            
            # Plot settings
            cmap = plt.get_cmap(self.cbb_palette.currentText()).copy()
            # Set NaN values to transparent so they don't hide the background
            cmap.set_bad(color='none')
            
            interpolation = 'bilinear' if self.action_smoothing.isChecked() else 'none'
            vmin_plot, vmax_plot = self.z_range_slider.value()
            color_kwargs = self._build_color_kwargs(vmin_plot, vmax_plot)

            # Plot heatmap - different approaches for wafer vs 2D maps
            if map_type != '2Dmap' and grid_z is not None:
                # Wafer maps: Use griddata result with user-selected interpolation (default 'none' / nearest-neighbor)
                self.img = self.ax.imshow(grid_z, extent=extent,
                                         origin='lower', aspect='equal', cmap=cmap,
                                         interpolation=interpolation,
                                         **color_kwargs)
                                         
                # Calculate pixel size for wafer map (grid_x is 80x80)
                # extent = [-r-1, r+1, -r-0.5, r+0.5] roughly
                width = extent[1] - extent[0]
                height = extent[3] - extent[2]
                self._current_dx = width / 80.0
                self._current_dy = height / 80.0
                
            elif not heatmap_pivot.empty:
                # 2D maps: Use pivot table with optional smoothing
                self.img = self.ax.imshow(heatmap_pivot, extent=extent,
                                         origin='lower', aspect='equal', cmap=cmap,
                                         interpolation=interpolation, **color_kwargs)
                                         
                # Calculate pixel size from pivot table shape and extent
                rows, cols = heatmap_pivot.shape
                if cols > 0 and rows > 0:
                    self._current_dx = (extent[1] - extent[0]) / cols
                    self._current_dy = (extent[3] - extent[2]) / rows
                else:
                    self._current_dx = 1.0
                    self._current_dy = 1.0
                                         
        except (ValueError, TypeError) as e:
            # Handle non-numeric data errors (from both scipy griddata and matplotlib imshow)
            error_msg = str(e)
            if "could not convert string to float" in error_msg or "dtype object cannot be converted to float" in error_msg:

                QMessageBox.critical(
                    None,
                    "Non-Numeric Data",
                    f"The selected parameter contains non-numerical values and cannot be visualized on the map.\n\n"
                    f"Please select a different parameter that contains only numeric data.\n\n"
                    f"Error: {error_msg}"
                )
            else:

                QMessageBox.critical(
                    None,
                    "Plot Error",
                    f"An error occurred while plotting the heatmap:\n\n{str(e)}"
                )
            self.ax.clear()
            self.canvas.draw_idle()
            return
        except Exception as e:
            # Handle any other errors
            QMessageBox.critical(
                None,
                "Plot Error",
                f"An error occurred while plotting the heatmap:\n\n{str(e)}"
            )
            self.ax.clear()
            self.canvas.draw_idle()
            return
        
        # Colorbar
        if self.img:
            if hasattr(self, 'cbar') and self.cbar is not None:
                try:
                    self.cbar.remove()
                except:
                    pass
            self.cbar = self.ax.figure.colorbar(self.img, ax=self.ax)
        
        # Grid
        if self.action_grid.isChecked():
            self.ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
        
        # Show stats (wafer only)
        if map_type != '2Dmap' and self.action_show_stats.isChecked():
            self._draw_stats_box()
        
        # Title
        title = self.cbb_zparameter.currentText()
        self.ax.set_title(title, fontsize=13)
        
        self.ax.get_figure().tight_layout()
        self.canvas.draw_idle()
        
        # Update selection overlay after plot is complete
        # (separate method handles adding/removing selection highlights)
        self._update_selection_overlay()

    def _build_color_kwargs(self, vmin, vmax):
        """imshow color kwargs for the selected color scale.

        Returns a LogNorm when 'Log' is chosen and the range is strictly
        positive; otherwise plain vmin/vmax (linear). Log requires positive
        values -- intensity/area are clipped >=0 and fit parameters can be
        <=0 -- so fall back to linear rather than raising when vmin <= 0.
        """
        if self.cbb_color_scale.currentText() == "Log" and vmin is not None and vmin > 0:
            return dict(norm=LogNorm(vmin=vmin, vmax=vmax))
        return dict(vmin=vmin, vmax=vmax)

    def _build_cache_key(self):
        """Construct cache key for the current heatmap configuration."""
        xmin, xmax = self.x_range_slider.value()
        parameter = self.cbb_zparameter.currentText()
        map_type = self.cbb_map_type.currentText()
        mask_enabled = self.cb_enable_mask.isChecked() if map_type == '2Dmap' else False
        remove_outliers = self.action_remove_outlier.isChecked()
        
        if mask_enabled:
            mask_config = (
                self.cbb_mask_param.currentText(),
                self.cbb_mask_operator.currentText(),
                self.spin_mask_threshold.value()
            )
        else:
            mask_config = None
        
        return (
            self.map_df_name,
            parameter,
            (xmin, xmax),
            map_type,
            mask_config,
            remove_outliers
        )

    def _get_data_for_heatmap(self):
        """Compute heatmap data from map DataFrame (with caching for wafer griddata).
        
        Returns:
            tuple: (heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col)
        """
        cache_key = self._build_cache_key()
        
        # Check cache first (avoid expensive griddata computation!)
        if cache_key in self._griddata_cache:
            return self._griddata_cache[cache_key]
        
        # Default returns
        heatmap_pivot = pd.DataFrame()
        extent = [0, 0, 0, 0]
        vmin, vmax = 0, 100
        grid_z = None
        final_z_col = None
        
        if self.map_df is None or self.map_df.empty:
            return heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col
        
        # Get X-range filter
        xmin, xmax = self.x_range_slider.value()
        column_labels = self.map_df.columns[2:]
        
        try:
            numeric_cols = column_labels.astype(float)
            col_mask = (numeric_cols >= xmin) & (numeric_cols <= xmax)
            filtered_columns = column_labels[col_mask]
        except Exception:
            filtered_columns = column_labels
        
        if len(filtered_columns) == 0:
            return heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col
        
        # Coordinates (cached in set_map_data)
        x_col, y_col = self._get_measurement_sites()
        
        # Compute Z values based on selected parameter
        parameter = self.cbb_zparameter.currentText()
        
        if parameter == 'Area':
            data_matrix = self.map_df[filtered_columns].to_numpy(dtype=float, copy=False)
            data_clean = np.nan_to_num(data_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(data_clean, 0, None, out=data_clean)
            z_col = data_clean.sum(axis=1)
        elif parameter == 'Intensity':
            data_matrix = self.map_df[filtered_columns].to_numpy(dtype=float, copy=False)
            data_clean = np.nan_to_num(data_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(data_clean, 0, None, out=data_clean)
            z_col = data_clean.max(axis=1)
        else:
            # Fit parameter - merge with map coordinates to handle filtered (active only) fit results
            if not self.df_fit_results.empty and parameter in self.df_fit_results.columns:
                filtered_results = self.df_fit_results[self.df_fit_results['Filename'] == self.map_df_name]
                if not filtered_results.empty and 'X' in filtered_results.columns and 'Y' in filtered_results.columns:
                    map_coords_temp = pd.DataFrame({
                        'X_round': np.round(x_col, 6),
                        'Y_round': np.round(y_col, 6)
                    })
                    filtered_results_temp = pd.DataFrame({
                        'X_round': np.round(filtered_results['X'].values.astype(float), 6),
                        'Y_round': np.round(filtered_results['Y'].values.astype(float), 6),
                        parameter: filtered_results[parameter].values
                    })
                    
                    merged = map_coords_temp.merge(
                        filtered_results_temp,
                        on=['X_round', 'Y_round'], how='left'
                    )
                    # Fill missing values (unchecked spectra) with 0 ONLY for amplitude, area, height, or intensity related parameters.
                    if any(x in parameter.lower() for x in ['ampli', 'area', 'intensity', 'height', 'fraction', 'A', 'B']):
                        z_col = merged[parameter].fillna(0).values
                    else:
                        z_col = merged[parameter].values
                else:
                    z_col = np.full(len(x_col), np.nan)
            else:
                z_col = np.full(len(x_col), np.nan)
        
        if z_col is None or len(z_col) == 0:
            return heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col
        
        # Apply mask (2D map only)
        map_type = self.cbb_map_type.currentText()
        if map_type == '2Dmap' and self.cb_enable_mask.isChecked():
            z_col = self._apply_mask(z_col, self.map_df)
        
        # Remove outliers if requested
        if self.action_remove_outlier.isChecked():
            z_col = self._remove_outliers(z_col)
        
        final_z_col = z_col
        
        # Compute vmin/vmax
        try:
            valid_z = final_z_col[~np.isnan(final_z_col)]
            if len(valid_z) > 0:
                vmin = float(valid_z.min())
                vmax = float(valid_z.max())
            else:
                vmin, vmax = 0.0, 100.0
            if np.isnan(vmin) or np.isnan(vmax):
                vmin, vmax = 0.0, 100.0
        except Exception:
            vmin, vmax = 0.0, 100.0
        
        # Different approach for wafer vs 2D maps
        if map_type != '2Dmap' and self.number_of_points >= 4:
            # Wafer maps: Use griddata for scattered point interpolation
            r = self._get_wafer_radius(map_type)
            if r:
                grid_x, grid_y = np.meshgrid(
                    np.linspace(-r, r, 80),  
                    np.linspace(-r, r, 80)
                )
                
                # Filter out NaN values before interpolation (masked points)
                # This prevents interpolation from filling in masked regions
                valid_mask = ~pd.isna(final_z_col)
                if valid_mask.any():
                    x_valid = x_col[valid_mask]
                    y_valid = y_col[valid_mask]
                    z_valid = final_z_col[valid_mask]
                    
                    # Linear interpolation for smooth result (only on valid points)
                    from scipy.interpolate import griddata
                    grid_z = griddata((x_valid, y_valid), z_valid, 
                                     (grid_x, grid_y), method='linear')
                else:
                    grid_z = np.full_like(grid_x, np.nan)
                
                extent = [-r-1, r+1, -r-0.5, r+0.5]
                heatmap_pivot = pd.DataFrame()  # Not used for wafer
        else:
            # 2D maps: Use fast pivot table (regular grid, no interpolation needed)
            # NaN values in final_z_col will be preserved in the pivot table
            heatmap_data = pd.DataFrame({'X': x_col, 'Y': y_col, 'Z': final_z_col})
            heatmap_pivot = heatmap_data.pivot(index='Y', columns='X', values='Z')
            
            # Calculate extent to center pixels on coordinates 
            # Use sorted unique coordinates
            u_x = np.sort(np.unique(x_col))
            u_y = np.sort(np.unique(y_col))
            
            xmin, xmax = 0.0, 1.0
            ymin, ymax = 0.0, 1.0
            
            if len(u_x) > 1:
                dx = (u_x[1] - u_x[0])
                xmin = u_x[0] - 0.5 * dx
                xmax = u_x[-1] + 0.5 * (u_x[-1] - u_x[-2])
            elif len(u_x) == 1:
                xmin = u_x[0] - 0.5
                xmax = u_x[0] + 0.5
                
            if len(u_y) > 1:
                dy = (u_y[1] - u_y[0])
                ymin = u_y[0] - 0.5 * dy
                ymax = u_y[-1] + 0.5 * (u_y[-1] - u_y[-2])
            elif len(u_y) == 1:
                ymin = u_y[0] - 0.5
                ymax = u_y[0] + 0.5
            
            extent = [xmin, xmax, ymin, ymax]
            grid_z = None  # Not used for 2D maps
        
        result = (heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col)
        
        # Cache the result to avoid recomputing expensive griddata
        self._griddata_cache[cache_key] = result
        
        return result
    
    def _apply_mask(self, z_col, filtered_df):
        """Apply mask to z_col based on mask settings."""
        mask_param = self.cbb_mask_param.currentText()
        mask_operator = self.cbb_mask_operator.currentText()
        threshold = self.spin_mask_threshold.value()
        
        mask_values = None
        
        # Try to get mask values from map DataFrame
        if mask_param in filtered_df.columns:
            mask_values = filtered_df[mask_param].astype(float).values
        # Or from fit results
        elif not self.df_fit_results.empty and mask_param in self.df_fit_results.columns:
            filtered_results = self.df_fit_results[self.df_fit_results['Filename'] == self.map_df_name]
            if not filtered_results.empty:
                mask_values = filtered_results[mask_param].astype(float).values
        
        if mask_values is None:
            return z_col
        
        # Apply operator
        if mask_operator == ">":
            valid = mask_values > threshold
        elif mask_operator == "<":
            valid = mask_values < threshold
        elif mask_operator == ">=":
            valid = mask_values >= threshold
        elif mask_operator == "<=":
            valid = mask_values <= threshold
        elif mask_operator == "==":
            valid = mask_values == threshold
        else:
            valid = np.ones_like(mask_values, dtype=bool)
        
        # Set invalid points to NaN
        return np.where(valid, z_col, np.nan)
    
    def _remove_outliers(self, z_col):
        """Remove outliers using IQR method with interpolation."""
        try:
            s = pd.Series(z_col)
            Q1 = s.quantile(0.05)
            Q3 = s.quantile(0.95)
            IQR = Q3 - Q1
            
            outlier_mask = (s < (Q1 - 1.5 * IQR)) | (s > (Q3 + 1.5 * IQR))
            
            s_interpolated = s.copy()
            s_interpolated[outlier_mask] = np.nan
            s_interpolated = s_interpolated.interpolate(
                method='linear', limit_direction='both'
            )
            return s_interpolated.to_numpy()
        except Exception:
            return z_col
    
    def _get_wafer_radius(self, map_type_text):
        """Extract wafer radius from map type string."""
        match = re.search(r'Wafer_(\d+)mm', map_type_text)
        if match:
            diameter = int(match.group(1))
            return diameter / 2
        return None
    
    def _get_measurement_sites(self):
        """Get all measurement site coordinates."""
        if hasattr(self, '_sites_x') and self._sites_x is not None and len(self._sites_x) > 0:
            return self._sites_x, self._sites_y
        if self.map_df is None or self.map_df.empty:
            return np.array([]), np.array([])
        return self.map_df['X'].values, self.map_df['Y'].values
    
    def _draw_stats_box(self):
        """Draw statistics text box on the plot (wafer mode)."""
        # Remove old stats
        if self._last_stats_text_artist is not None:
            try:
                self._last_stats_text_artist.remove()
            except Exception:
                pass
            self._last_stats_text_artist = None
        
        if self._last_final_z_col is None:
            return
        
        arr = np.asarray(self._last_final_z_col, dtype=float)
        arr_clean = arr[~np.isnan(arr)]
        if len(arr_clean) == 0:
            return
        
        mean = float(np.mean(arr_clean))
        mn = float(np.min(arr_clean))
        mx = float(np.max(arr_clean))
        std = float(np.std(arr_clean, ddof=1)) if len(arr_clean) > 1 else 0.0
        sigma3 = 3 * std
        
        txt = (f"mean: {mean:.2f}\n"
               f"min: {mn:.2f}\n"
               f"max: {mx:.2f}\n"
               f"3σ: {sigma3:.2f}")
        
        # Place text box on axes
        self._last_stats_text_artist = self.ax.text(
            0.02, 0.98, txt,
            transform=self.ax.transAxes,
            fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="black")
        )
    
    # ═══ Mouse interaction handlers ═══
    
    def _on_mouse_click(self, event):
        """Start selection on mouse click."""
        # Disable selection if toolbar mode (Zoom/Pan) is active
        if self.toolbar.mode:
            return
            
        if event.inaxes != self.ax or event.button != 1:
            return
        
        self.rect_start = (event.xdata, event.ydata)
        
        # Remove previous rectangle
        if self.rect_patch is not None:
            try:
                self.rect_patch.remove()
            except:
                pass
            self.rect_patch = None
    
    def _update_coordinates(self, event):
        """Update built-in live coordinate display label under the map canvas."""
        if hasattr(self, "lbl_coordinates"):
            if event.inaxes and event.xdata is not None and event.ydata is not None:
                try:
                    coord_str = event.inaxes.format_coord(event.xdata, event.ydata)
                    clean_str = " ".join(coord_str.split())
                    self.lbl_coordinates.setText(clean_str)
                    if hasattr(self.toolbar, "locLabel"):
                        self.toolbar.locLabel.setText(clean_str)
                except Exception:
                    self.lbl_coordinates.setText("")
            else:
                self.lbl_coordinates.setText("")

    def _on_figure_leave(self, event):
        """Clear live coordinate display when cursor leaves the map canvas."""
        if hasattr(self, "lbl_coordinates"):
            self.lbl_coordinates.setText("")
        if hasattr(self.toolbar, "locLabel"):
            self.toolbar.locLabel.setText("")

    def _on_mouse_move(self, event):
        """Update rectangle during drag."""
        self._update_coordinates(event)
        if self.toolbar.mode:
            return
            
        if self.rect_start is None or event.inaxes != self.ax:
            return
        
        x0, y0 = self.rect_start
        x1, y1 = event.xdata, event.ydata
        
        if self.rect_patch is None:
            self.rect_patch = patches.Rectangle(
                (min(x0, x1), min(y0, y1)),
                abs(x1 - x0), abs(y1 - y0),
                linewidth=1.2, edgecolor='black', 
                facecolor='none', linestyle='--'
            )
            self.ax.add_patch(self.rect_patch)
        else:
            self.rect_patch.set_x(min(x0, x1))
            self.rect_patch.set_y(min(y0, y1))
            self.rect_patch.set_width(abs(x1 - x0))
            self.rect_patch.set_height(abs(y1 - y0))
        
        self.canvas.draw_idle()
    
    def _on_mouse_release(self, event):
        """Finish selection on mouse release."""
        if self.toolbar.mode:
            return
            
        if self.rect_start is None or event.inaxes != self.ax:
            return
        
        x0, y0 = self.rect_start
        x1, y1 = event.xdata, event.ydata
        
        all_x, all_y = self._get_measurement_sites()
        if len(all_x) == 0:
            self.rect_start = None
            return
        
        modifiers = QApplication.keyboardModifiers()
        
        if self.rect_patch is not None:
            # Rectangle selection
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
            inside = ((all_x >= xmin) & (all_x <= xmax) & 
                     (all_y >= ymin) & (all_y <= ymax))
            
            if modifiers != Qt.ControlModifier:
                self.selected_points = []
            
            # Add selected points (ensure uniqueness)
            new_points = [(float(x), float(y)) for x, y in zip(all_x[inside], all_y[inside])]
            for pt in new_points:
                if pt not in self.selected_points:
                    self.selected_points.append(pt)
            
            try:
                self.rect_patch.remove()
            except Exception:
                pass
            self.rect_patch = None
        else:
            # Single click - nearest point
            if all_x.size > 0:
                distances_sq = (all_x - x1) ** 2 + (all_y - y1) ** 2
                idx = np.argmin(distances_sq)
                pt = (float(all_x[idx]), float(all_y[idx]))
                
                if modifiers == Qt.ControlModifier:
                    if pt in self.selected_points:
                        self.selected_points.remove(pt)
                    else:
                        self.selected_points.append(pt)
                else:
                    self.selected_points = [pt]
        
        self.rect_start = None
        
        # Emit signal with selected points
        self.spectra_selected.emit(self.selected_points)
        
        # Update selection overlay without full redraw
        self._update_selection_overlay()
    
    def _copy_figure_to_clipboard(self):
        """Copy the matplotlib figure to clipboard."""
        target_theme = "Light Mode"
        
        if hasattr(self, 'v_spectra_viewer_ref'):
            sv = self.v_spectra_viewer_ref
            if hasattr(sv, 'cbb_copy_theme'):
                target_theme = sv.cbb_copy_theme.currentText()
                
        original_theme = getattr(self, 'current_style', "Light Mode")
        
        if target_theme != original_theme:
            try:
                self.current_style = target_theme
                self._do_plot_heatmap()
                self.canvas.draw()
                copy_fig_to_clb(self.canvas)
            finally:
                self.current_style = original_theme
                self._do_plot_heatmap()
        else:
            copy_fig_to_clb(self.canvas)
    
    def _plot_height_profile_on_map(self):
        """Plot height profile directly on heatmap when exactly 2 points selected."""
        if len(self.selected_points) != 2:
            return
        
        x, y = zip(*self.selected_points)
        
        # Draw dotted line connecting the two points
        self._profile_line, = self.ax.plot(
            x, y, color='black', linestyle='dotted', linewidth=2, zorder=11
        )
        
        # Extract profile values and plot height visualization
        profile_df = self._extract_profile()
        if profile_df is not None and not profile_df.empty:
            extent = self.ax.get_xlim() + self.ax.get_ylim()
            diagonal_length = np.sqrt((extent[1] - extent[0])**2 + (extent[3] - extent[2])**2)
            max_normalized_value = 0.3 * diagonal_length
            
            # Calculate height values (normalized)
            values = profile_df['values'].values
            val_max = values.max()
            if val_max > 0:
                height = (values / val_max) * max_normalized_value
                height -= height.min()
                
                dx = x[1] - x[0]
                dy = y[1] - y[0]
                length = np.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    perp_dx = -dy / length
                    perp_dy = dx / length
                    x_vals = profile_df['X'].values + perp_dx * height
                    y_vals = profile_df['Y'].values + perp_dy * height
                    self._profile_height_line, = self.ax.plot(
                        x_vals, y_vals, color='black', linestyle='-', linewidth=2, zorder=11
                    )
    
    def _extract_profile(self) -> pd.DataFrame | None:
        """Extract a profile from 2D map plot via interpolation.
        
        Returns:
            DataFrame with columns: X, Y, distance, values
            or None if extraction fails
        """
        if len(self.selected_points) != 2:
            return None
        
        (x1, y1), (x2, y2) = self.selected_points
        heatmap_pivot, _, _, _, _, _ = self._get_data_for_heatmap()
        if heatmap_pivot.empty:
            return None
        
        x_values = heatmap_pivot.columns.values.astype(float)
        y_values = heatmap_pivot.index.values.astype(float)
        z_values = heatmap_pivot.values
        
        from scipy.interpolate import RegularGridInterpolator
        try:
            interpolator = RegularGridInterpolator((y_values, x_values), z_values)
            num_samples = 100
            x_samples = np.linspace(x1, x2, num_samples)
            y_samples = np.linspace(y1, y2, num_samples)
            sample_points = np.column_stack((y_samples, x_samples))
            z_samples = interpolator(sample_points)
            dists_from_start = np.sqrt((x_samples - x1)**2 + (y_samples - y1)**2)
            return pd.DataFrame({
                'X': x_samples,
                'Y': y_samples,
                'distance': dists_from_start,
                'values': z_samples
            })
        except Exception as e:
            print(f"Profile interpolation failed: {e}")
            return None
