# spectroview/view/components/v_map_viewer.py
"""View component for Map visualization with matplotlib canvas and controls."""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from scipy.interpolate import griddata
from superqt import QLabeledDoubleRangeSlider

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QCheckBox, QDoubleSpinBox,
    QFrame, QLineEdit, QToolButton, QMenu, QWidgetAction,
    QSpacerItem, QSizePolicy, QGroupBox, QApplication
)
from PySide6.QtCore import Qt, Signal, QSize, QTimer
from PySide6.QtGui import QIcon, QAction

from spectroview import ICON_DIR
from spectroview.viewmodel.utils import CustomizedPalette, copy_fig_to_clb


class VMapViewer(QWidget):
    """Map viewer widget with matplotlib canvas and control widgets."""
    
    # ───── View → ViewModel signals ─────
    spectra_selected = Signal(list)  # List of (x, y) tuples for selected spectra
    extract_profile_requested = Signal(str)  # profile name
    multi_viewer_requested = Signal(int)  # number of viewers (2, 3, or 4)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Data state
        self.map_df = None
        self.map_df_name = None
        self.df_fit_results = None
        self.selected_points = []
        self.number_of_points = 0
        
        # Rectangle selection state
        self.rect_start = None
        self.rect_patch = None
        
        # Plot artifacts
        self.img = None
        self.cbar = None
        self._last_final_z_col = None
        self._last_stats_text_artist = None
        self._selection_scatter = None  # Cache for selection overlay
        
        # Cache for expensive griddata computation (wafer maps)
        self._griddata_cache = {}  # {cache_key: (heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col)}
        
        # Debounce timer for plot updates (matches legacy 100ms delay)
        self.plot_timer = QTimer()
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
        canvas_layout = QVBoxLayout(canvas_frame)
        canvas_layout.setContentsMargins(2, 2, 2, 2)
        
        # Create matplotlib figure and canvas
        self.figure = plt.figure(dpi=70, figsize=(5, 4))
        self.ax = self.figure.add_subplot(111)
        self.ax.tick_params(axis='both', which='both')
        
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(200)
        self.canvas.setMaximumHeight(350)
        
        # Connect mouse events for interactive selection
        self.canvas.mpl_connect('button_press_event', self._on_mouse_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        
        # Matplotlib toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        # Hide some toolbar actions
        for action in self.toolbar.actions():
            if action.text() in ['Customize','Zoom','Save', 'Pan', 'Back', 'Forward', 'Subplots']:
                action.setVisible(False)
        
        # Custom toolbar with add viewer and copy buttons
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(2, 2, 2, 2)
        toolbar_layout.addWidget(self.toolbar)
        toolbar_layout.addStretch()
        
        # Add viewer button
        self.btn_add_viewer = QPushButton()
        self.btn_add_viewer.setIcon(QIcon(os.path.join(ICON_DIR, "add.png")))
        self.btn_add_viewer.setIconSize(QSize(20, 20))
        self.btn_add_viewer.setToolTip("Open a new map viewer window")
        self.btn_add_viewer.setFixedSize(28, 28)
        self.btn_add_viewer.clicked.connect(lambda: self.multi_viewer_requested.emit(1))
        toolbar_layout.addWidget(self.btn_add_viewer)
        
        # Copy button
        self.btn_copy = QPushButton()
        self.btn_copy.setIcon(QIcon(os.path.join(ICON_DIR, "copy.png")))
        self.btn_copy.setIconSize(QSize(20, 20))
        self.btn_copy.setToolTip("Copy figure to clipboard")
        self.btn_copy.setFixedSize(28, 28)
        self.btn_copy.clicked.connect(self._copy_figure_to_clipboard)
        toolbar_layout.addWidget(self.btn_copy)
        
        canvas_layout.addWidget(self.canvas, stretch=1)
        canvas_layout.addLayout(toolbar_layout)
        
        return canvas_frame
    
    def _create_map_type_controls(self):
        """Create map type and palette selector controls."""
        type_layout = QHBoxLayout()
        type_layout.setContentsMargins(4, 4, 4, 4)
        
        lbl = QLabel("Select Map type:")
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
        z_slider_layout.setContentsMargins(4, 2, 4, 2)
        
        self.cbb_zparameter = QComboBox()
        self.cbb_zparameter.addItems(['Intensity', 'Area'])
        self.cbb_zparameter.setFixedWidth(50)
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
        self.btn_options.setIcon(QIcon(os.path.join(ICON_DIR, "options.png")))
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
    
    # ═══ Public API (ViewModel → View) ═══
    
    def set_map_data(self, map_df, map_name, df_fit_results=None):
        """Set map data and schedule plot update with debounce."""
        self.map_df = map_df
        self.map_df_name = map_name
        self.df_fit_results = df_fit_results if df_fit_results is not None else pd.DataFrame()
        
        # Clear selection points when switching maps (prevents out-of-range highlights)
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
        
        # Update Z parameters
        self.cbb_zparameter.clear()
        self.cbb_zparameter.addItems(['Intensity', 'Area'])
        
        if not self.df_fit_results.empty:
            fit_columns = [col for col in self.df_fit_results.columns 
                          if col not in ['Filename', 'X', 'Y']]
            self.cbb_zparameter.addItems(fit_columns)
            self.cbb_mask_param.clear()
            self.cbb_mask_param.addItems(fit_columns)
        
        # Restore selection if possible
        if current_z and self.cbb_zparameter.findText(current_z) >= 0:
            self.cbb_zparameter.setCurrentText(current_z)
        
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
            
            # Update X-range slider
            if not self.cb_fix_x.isChecked():
                self.x_range_slider.setRange(xmin, xmax)
                self.x_range_slider.setValue((xmin, xmax))
                self.spin_xmin.setValue(xmin)
                self.spin_xmax.setValue(xmax)
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
            
            # Update Z-range slider
            if not self.cb_fix_z.isChecked():
                self.z_range_slider.setRange(vmin, vmax)
                self.z_range_slider.setValue((vmin, vmax))
                self.spin_zmin.setValue(vmin)
                self.spin_zmax.setValue(vmax)
            else:
                # Just update the range, keep current values
                self.z_range_slider.setRange(vmin, vmax)
        except:
            pass
    
    # ═══ Internal signal handlers ═══
    
    def _on_parameter_changed(self):
        """Z parameter changed - update range and refresh plot."""
        self._update_z_range()
        self.plot_heatmap()
    
    def _on_plot_option_changed(self):
        """Plot option changed - update range if needed and refresh."""
        self._update_z_range()
        self.plot_heatmap()
    
    def _on_x_slider_changed(self, values):
        """X-range slider changed - update spinboxes and trigger plot."""
        self.spin_xmin.blockSignals(True)
        self.spin_xmax.blockSignals(True)
        self.spin_xmin.setValue(values[0])
        self.spin_xmax.setValue(values[1])
        self.spin_xmin.blockSignals(False)
        self.spin_xmax.blockSignals(False)
        self._update_z_range()  # Recalculate Z values for new X range
        self.plot_heatmap()
    
    def _on_z_slider_changed(self, values):
        """Z-range slider changed - update spinboxes and trigger plot."""
        self.spin_zmin.blockSignals(True)
        self.spin_zmax.blockSignals(True)
        self.spin_zmin.setValue(values[0])
        self.spin_zmax.setValue(values[1])
        self.spin_zmin.blockSignals(False)
        self.spin_zmax.blockSignals(False)
        self.plot_heatmap()
    
    def _update_x_slider_from_spins(self):
        """Update X slider when spinboxes change."""
        xmin = self.spin_xmin.value()
        xmax = self.spin_xmax.value()
        if xmin <= xmax:
            self.x_range_slider.setValue((xmin, xmax))
            self._update_z_range()
            self.plot_heatmap()
    
    def _update_z_slider_from_spins(self):
        """Update Z slider when spinboxes change."""
        zmin = self.spin_zmin.value()
        zmax = self.spin_zmax.value()
        if zmin <= zmax:
            self.z_range_slider.setValue((zmin, zmax))
            self.plot_heatmap()
    
    # ═══ Core plotting methods ═══
    
    def _update_selection_overlay(self):
        """Update selection overlay without full plot redraw (fast)."""
        # Remove old selection scatter
        if self._selection_scatter is not None:
            try:
                self._selection_scatter.remove()
            except:
                pass
            self._selection_scatter = None
        
        # Add new selection overlay
        if self.selected_points:
            x, y = zip(*self.selected_points)
            self._selection_scatter = self.ax.scatter(
                x, y, facecolors='none', edgecolors='red', 
                marker='s', s=60, linewidths=1, zorder=10
            )
        
        self.canvas.draw_idle()
    
    def plot_heatmap(self):
        """Schedule a debounced heatmap plot update."""
        self.plot_timer.start()
    
    def _do_plot_heatmap(self):
        """Plot 2D heatmap or wafer map based on current data and settings."""
        if self.map_df is None or self.map_df.empty:
            self.ax.clear()
            self.canvas.draw_idle()
            return
        
        map_type = self.cbb_map_type.currentText()
        
        # Clear axes and cached selection
        self.ax.clear()
        self._selection_scatter = None
        
        # Plot wafer circle for wafer maps
        if map_type != '2Dmap':
            r = self._get_wafer_radius(map_type)
            if r:
                wafer_circle = patches.Circle((0, 0), radius=r, fill=False, 
                                             color='black', linewidth=1)
                self.ax.add_patch(wafer_circle)
                
                # Show all measurement sites
                all_x, all_y = self._get_measurement_sites()
                self.ax.scatter(all_x, all_y, marker='x', color='gray', s=15)
                
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
        
        # Get heatmap data
        heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col = self._get_data_for_heatmap()
        self._last_final_z_col = final_z_col
        
        # Plot settings
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(self.cbb_palette.currentText()).copy()
        cmap.set_bad(color='white')  # Set NaN values to white (masked regions)
        
        interpolation = 'bilinear' if self.action_smoothing.isChecked() else 'none'
        vmin_plot, vmax_plot = self.z_range_slider.value()
        
        # Plot heatmap - different approaches for wafer vs 2D maps
        if map_type != '2Dmap' and grid_z is not None:
            # Wafer maps: Use griddata result (already interpolated, smooth)
            self.img = self.ax.imshow(grid_z, extent=extent,
                                     vmin=vmin_plot, vmax=vmax_plot,
                                     origin='lower', aspect='equal', cmap=cmap,
                                     interpolation='bilinear')  # Smooth the 100x100 grid
        elif not heatmap_pivot.empty:
            # 2D maps: Use pivot table with optional smoothing
            self.img = self.ax.imshow(heatmap_pivot, extent=extent, 
                                     vmin=vmin_plot, vmax=vmax_plot,
                                     origin='lower', aspect='equal', cmap=cmap, 
                                     interpolation=interpolation)
        
        # Colorbar
        if self.img:
            if hasattr(self, 'cbar') and self.cbar is not None:
                try:
                    self.cbar.update_normal(self.img)
                except:
                    self.cbar = self.ax.figure.colorbar(self.img, ax=self.ax)
            else:
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
    
    def _get_data_for_heatmap(self):
        """Compute heatmap data from map DataFrame (with caching for wafer griddata).
        
        Returns:
            tuple: (heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col)
        """
        # Build cache key from all parameters that affect the result
        xmin, xmax = self.x_range_slider.value()
        parameter = self.cbb_zparameter.currentText()
        map_type = self.cbb_map_type.currentText()
        mask_enabled = self.cb_enable_mask.isChecked() if map_type == '2Dmap' else False
        remove_outliers = self.action_remove_outlier.isChecked()
        
        # Build mask configuration tuple for cache key
        if mask_enabled:
            mask_config = (
                self.cbb_mask_param.currentText(),
                self.cbb_mask_operator.currentText(),
                self.spin_mask_threshold.value()
            )
        else:
            mask_config = None
        
        cache_key = (
            self.map_df_name,
            parameter,
            (xmin, xmax),
            map_type,
            mask_config,  # Full mask configuration (not just enabled flag)
            remove_outliers
        )
        
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
            filtered_columns = column_labels[
                (column_labels.astype(float) >= xmin) &
                (column_labels.astype(float) <= xmax)
            ]
        except:
            filtered_columns = column_labels
        
        if len(filtered_columns) == 0:
            return heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col
        
        # Get filtered data
        filtered_df = self.map_df[['X', 'Y'] + list(filtered_columns)]
        x_col = filtered_df['X'].values
        y_col = filtered_df['Y'].values
        
        # Compute Z values based on selected parameter
        parameter = self.cbb_zparameter.currentText()
        
        if parameter == 'Area':
            z_col = (filtered_df[filtered_columns]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0).clip(lower=0).sum(axis=1))
        elif parameter == 'Intensity':
            z_col = (filtered_df[filtered_columns]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0).clip(lower=0).max(axis=1))
        else:
            # Fit parameter - merge with map coordinates to handle filtered (active only) fit results
            if not self.df_fit_results.empty and parameter in self.df_fit_results.columns:
                filtered_results = self.df_fit_results.query("Filename == @self.map_df_name")
                if not filtered_results.empty and 'X' in filtered_results.columns and 'Y' in filtered_results.columns:
                    # Create DataFrame with map coordinates
                    map_coords = pd.DataFrame({'X': x_col, 'Y': y_col})
                    # Merge with fit results on X, Y coordinates (left join to keep all map points)
                    merged = map_coords.merge(filtered_results[['X', 'Y', parameter]], 
                                             on=['X', 'Y'], how='left')
                    # Fill missing values (unchecked spectra) with 0
                    z_col = merged[parameter].fillna(0)
                else:
                    z_col = pd.Series([0] * len(x_col))
            else:
                z_col = pd.Series([0] * len(x_col))
        
        if z_col is None or len(z_col) == 0:
            return heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col
        
        # Apply mask (2D map only)
        map_type = self.cbb_map_type.currentText()
        if map_type == '2Dmap' and self.cb_enable_mask.isChecked():
            z_col = self._apply_mask(z_col, filtered_df)
        
        # Remove outliers if requested
        if self.action_remove_outlier.isChecked():
            z_col = self._remove_outliers(z_col)
        
        final_z_col = z_col
        
        # Compute vmin/vmax
        try:
            vmin = float(final_z_col.min())
            vmax = float(final_z_col.max())
        except:
            vmin, vmax = 0, 100
        
        # Count unique sites
        self.number_of_points = len(set(zip(x_col, y_col)))
        
        # Different approach for wafer vs 2D maps
        if map_type != '2Dmap' and self.number_of_points >= 4:
            # Wafer maps: Use griddata for scattered point interpolation
            r = self._get_wafer_radius(map_type)
            if r:
                grid_x, grid_y = np.meshgrid(
                    np.linspace(-r, r, 80),  
                    np.linspace(-r, r, 80)
                )
                from scipy.interpolate import griddata
                
                # Filter out NaN values before interpolation (masked points)
                # This prevents interpolation from filling in masked regions
                valid_mask = ~pd.isna(final_z_col)
                if valid_mask.any():
                    x_valid = x_col[valid_mask]
                    y_valid = y_col[valid_mask]
                    z_valid = final_z_col[valid_mask]
                    
                    # Linear interpolation for smooth result (only on valid points)
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
            extent = [x_col.min(), x_col.max(), y_col.min(), y_col.max()]
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
            filtered_results = self.df_fit_results.query("Filename == @self.map_df_name")
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
            Q1 = z_col.quantile(0.05)
            Q3 = z_col.quantile(0.95)
            IQR = Q3 - Q1
            
            outlier_mask = (z_col < (Q1 - 1.5 * IQR)) | (z_col > (Q3 + 1.5 * IQR))
            
            z_col_interpolated = z_col.copy()
            z_col_interpolated[outlier_mask] = np.nan
            z_col_interpolated = z_col_interpolated.interpolate(
                method='linear', limit_direction='both'
            )
            return z_col_interpolated
        except:
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
        if self.map_df is None:
            return np.array([]), np.array([])
        return self.map_df['X'].values, self.map_df['Y'].values
    
    def _draw_stats_box(self):
        """Draw statistics text box on the plot (wafer mode)."""
        # Remove old stats
        if self._last_stats_text_artist is not None:
            try:
                self._last_stats_text_artist.remove()
            except:
                pass
            self._last_stats_text_artist = None
        
        if self._last_final_z_col is None:
            return
        
        # Calculate statistics
        arr_clean = pd.Series(self._last_final_z_col).dropna()
        if arr_clean.empty:
            return
        
        mean = float(arr_clean.mean())
        mn = float(arr_clean.min())
        mx = float(arr_clean.max())
        std = float(arr_clean.std())
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
    
    def _on_mouse_move(self, event):
        """Update rectangle during drag."""
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
        if self.rect_start is None or event.inaxes != self.ax:
            return
        
        x0, y0 = self.rect_start
        x1, y1 = event.xdata, event.ydata
        
        all_x, all_y = self._get_measurement_sites()
        all_x, all_y = np.array(all_x), np.array(all_y)
        
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
            except:
                pass
            self.rect_patch = None
        else:
            # Single click - nearest point
            if all_x.size > 0:
                distances = np.sqrt((all_x - x1) ** 2 + (all_y - y1) ** 2)
                idx = np.argmin(distances)
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
        copy_fig_to_clb(self.canvas)
