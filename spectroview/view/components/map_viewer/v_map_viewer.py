"""Main MapViewer widget: canvas layout, navigation controls, sliders, and options menu."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
import numpy as np
import pandas as pd
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)
from superqt import QLabeledDoubleRangeSlider

from spectroview import PLOT_POLICY_LIGHT
from spectroview.view.components.common.customized_widgets import CustomizedPalette
from spectroview.i18n import (
    bind_combo_items,
    bind_placeholder,
    bind_text,
    bind_tooltip,
    combo_data_or_text,
    on_language_changed,
    set_combo_by_data_or_text,
    tr,
)
from spectroview.view.components.map_viewer.map_selection import MapCanvasSelector
from spectroview.view.components.map_viewer.v_map_data import MapDataProcessingMixin
from spectroview.view.components.map_viewer.v_map_interactions import (
    MapInteractionMixin,
)
from spectroview.view.components.map_viewer.v_map_plot import MapPlotMixin
from spectroview.view.theme import Icons, current_tokens


class VMapViewer(
    QWidget,
    MapPlotMixin,
    MapDataProcessingMixin,
    MapInteractionMixin,
):
    """Map viewer widget with matplotlib canvas and control widgets."""

    # ───── View → ViewModel signals ─────
    spectra_selected = Signal(list)  # List of (x, y) tuples for selected spectra
    extract_profile_requested = Signal(str)  # profile name
    multi_viewer_requested = Signal(int)  # Add more map_viewer

    def __init__(self, parent=None):
        super().__init__(parent)
        # Data state
        self.map_df = None
        self.map_df_name = None
        self.df_fit_results = None
        self.selected_points = []
        self.number_of_points = 0
        self.current_style = 'Light Mode'

        # Rectangle / canvas selection helper (initialized in _create_canvas)
        self._selector = None

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
        on_language_changed(self, self._retranslate_map_viewer)

    def _retranslate_map_viewer(self, _lang: str = ""):
        """Update dynamic Z-parameter labels and re-render map plot."""
        self.cbb_zparameter.blockSignals(True)
        try:
            for idx in range(self.cbb_zparameter.count()):
                key = self.cbb_zparameter.itemData(idx)
                if key == "Intensity":
                    self.cbb_zparameter.setItemText(idx, tr("Intensity"))
                elif key == "Area":
                    self.cbb_zparameter.setItemText(idx, tr("Area"))
        finally:
            self.cbb_zparameter.blockSignals(False)
        if self.map_df is not None and not self.map_df.empty:
            self.plot_heatmap()

    @property
    def rect_start(self):
        return self._selector.rect_start if self._selector is not None else None

    @rect_start.setter
    def rect_start(self, value):
        if self._selector is not None:
            self._selector.rect_start = value

    @property
    def rect_patch(self):
        return self._selector.rect_patch if self._selector is not None else None

    @rect_patch.setter
    def rect_patch(self, value):
        if self._selector is not None:
            self._selector.rect_patch = value

    @property
    def _selection_collection(self):
        return self._selector.selection_collection if self._selector is not None else None

    @_selection_collection.setter
    def _selection_collection(self, value):
        if self._selector is not None:
            self._selector.selection_collection = value

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
        self._selector = MapCanvasSelector(self.ax, self.canvas)

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
        bind_tooltip(self.btn_home, "Rescale / Reset original view (Ctrl+R)")
        self.btn_home.clicked.connect(self.toolbar.home)
        self.btn_rescale = self.btn_home  # Alias matching v_spectra_viewer

        self.btn_pan = QToolButton()
        self.btn_pan.setCheckable(True)
        self.btn_pan.setIcon(Icons.pan())
        self.btn_pan.setIconSize(QSize(20, 20))
        self.btn_pan.setFixedSize(28, 28)
        bind_tooltip(self.btn_pan, "Pan / Drag axes with left mouse, zoom with right")
        self.btn_pan.clicked.connect(self.toolbar.pan)

        self.btn_zoom = QToolButton()
        self.btn_zoom.setCheckable(True)
        self.btn_zoom.setIcon(Icons.zoom())
        self.btn_zoom.setIconSize(QSize(20, 20))
        self.btn_zoom.setFixedSize(28, 28)
        bind_tooltip(self.btn_zoom, "Zoom to rectangle")
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
        bind_tooltip(self.btn_add_viewer, "Open a new map viewer window")
        self.btn_add_viewer.setFixedSize(28, 28)
        self.btn_add_viewer.clicked.connect(lambda: self.multi_viewer_requested.emit(1))
        toolbar_layout.addWidget(self.btn_add_viewer)

        # Copy button
        self.btn_copy = QPushButton()
        self.btn_copy.setIcon(Icons.copy())
        self.btn_copy.setIconSize(QSize(20, 20))
        bind_tooltip(self.btn_copy, "Copy figure to clipboard")
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
        if hasattr(self, "canvas_card") and getattr(self, "figure", None) is not None:
            from matplotlib.colors import to_hex
            fig_hex = to_hex(self.figure.patch.get_facecolor())
            self.canvas_card.setStyleSheet(
                f"QFrame#canvasCard {{ background-color: {fig_hex}; border: 1px solid {tok.border}; border-radius: 8px; }}"
            )

    def _create_map_type_controls(self):
        """Create map type and palette selector controls."""
        type_layout = QHBoxLayout()
        type_layout.setContentsMargins(2, 2, 2, 2)

        lbl = QLabel()
        bind_text(lbl, "Select map type:")
        type_layout.addWidget(lbl)
        self.cbb_map_type = QComboBox()
        bind_combo_items(
            self.cbb_map_type,
            [
                ("2Dmap", "2Dmap"),
                ("Wafer_300mm", "Wafer_300mm"),
                ("Wafer_200mm", "Wafer_200mm"),
                ("Wafer_100mm", "Wafer_100mm"),
            ],
        )
        self.cbb_map_type.setFixedWidth(120)
        self.cbb_map_type.currentIndexChanged.connect(lambda: self.plot_heatmap())
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
        self.cbb_zparameter.addItem(tr('Intensity'), 'Intensity')
        self.cbb_zparameter.addItem(tr('Area'), 'Area')
        self.cbb_zparameter.setMinimumWidth(68)
        bind_tooltip(self.cbb_zparameter, "Select parameter to plot in heatmap")
        self.cbb_zparameter.currentIndexChanged.connect(lambda: self._on_parameter_changed())
        z_slider_layout.addWidget(self.cbb_zparameter)

        self.cb_fix_z = QCheckBox()
        bind_text(self.cb_fix_z, "Fix")
        bind_tooltip(self.cb_fix_z, "Fix Z-range when changing maps")
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

        lbl_x_slider = QLabel()
        bind_text(lbl_x_slider, "X-range:")
        lbl_x_slider.setFixedWidth(68)
        x_slider_layout.addWidget(lbl_x_slider)

        self.cb_fix_x = QCheckBox()
        bind_text(self.cb_fix_x, "Fix")
        bind_tooltip(self.cb_fix_x, "Fix X-range when refreshing")
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

        self.cb_enable_mask = QCheckBox()
        bind_text(self.cb_enable_mask, "Enable mask:")
        self.cb_enable_mask.stateChanged.connect(lambda: self.plot_heatmap())
        mask_layout.addWidget(self.cb_enable_mask)

        self.cbb_mask_param = QComboBox()
        self.cbb_mask_param.setFixedWidth(100)
        self.cbb_mask_param.currentIndexChanged.connect(lambda: self.plot_heatmap())
        mask_layout.addWidget(self.cbb_mask_param)

        self.cbb_mask_operator = QComboBox()
        self.cbb_mask_operator.addItems([">", "<", ">=", "<=", "=="])
        self.cbb_mask_operator.setFixedWidth(50)
        self.cbb_mask_operator.currentIndexChanged.connect(lambda: self.plot_heatmap())
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
        bind_tooltip(self.btn_options, "Additional options")

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
        bind_placeholder(self.txt_profile_name, "Profile name...")
        self.txt_profile_name.setFixedWidth(150)

        self.btn_extract_profile = QPushButton()
        bind_text(self.btn_extract_profile, "Extract profile")
        self.btn_extract_profile.setMinimumWidth(110)
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
        self.action_remove_outlier = QAction(self)
        bind_text(self.action_remove_outlier, "Remove Outlier")
        self.action_remove_outlier.setCheckable(True)
        self.action_remove_outlier.setChecked(True)
        self.action_remove_outlier.triggered.connect(lambda: self._on_plot_option_changed())
        self.options_menu.addAction(self.action_remove_outlier)

        self.action_smoothing = QAction(self)
        bind_text(self.action_smoothing, "Smoothing")
        self.action_smoothing.setCheckable(True)
        self.action_smoothing.setChecked(False)
        self.action_smoothing.triggered.connect(lambda: self.plot_heatmap())
        self.options_menu.addAction(self.action_smoothing)

        self.action_grid = QAction(self)
        bind_text(self.action_grid, "Grid")
        self.action_grid.setCheckable(True)
        self.action_grid.setChecked(False)
        self.action_grid.triggered.connect(lambda: self.plot_heatmap())
        self.options_menu.addAction(self.action_grid)

        self.action_show_stats = QAction(self)
        bind_text(self.action_show_stats, "Show stats")
        self.action_show_stats.setCheckable(True)
        self.action_show_stats.setChecked(False)
        self.action_show_stats.triggered.connect(lambda: self.plot_heatmap())
        self.options_menu.addAction(self.action_show_stats)

        self.options_menu.addSeparator()

        # Color-scale selector: linear vs log color normalization for the heatmap.
        scale_widget = QWidget()
        scale_layout = QHBoxLayout(scale_widget)
        scale_layout.setContentsMargins(5, 5, 5, 5)
        lbl_scale = QLabel()
        bind_text(lbl_scale, "Color scale:")
        scale_layout.addWidget(lbl_scale)
        self.cbb_color_scale = QComboBox()
        bind_combo_items(
            self.cbb_color_scale,
            [("Linear", "Linear"), ("Log", "Log")],
        )
        bind_tooltip(
            self.cbb_color_scale,
            "Color-scale normalization for the heatmap (log needs positive values)",
        )
        self.cbb_color_scale.currentIndexChanged.connect(lambda: self.plot_heatmap())
        scale_layout.addWidget(self.cbb_color_scale)
        scale_layout.addStretch()

        scale_action = QWidgetAction(self)
        scale_action.setDefaultWidget(scale_widget)
        self.options_menu.addAction(scale_action)

    # ═══ Public API (ViewModel → View) ═══

    def set_map_data(self, map_df, map_name, df_fit_results=None, selected_points=None):
        """Set map data and schedule plot update with debounce."""
        is_map_change = (map_name != getattr(self, "map_df_name", None))
        old_sites_x = getattr(self, "_sites_x", np.array([], dtype=float))
        old_sites_y = getattr(self, "_sites_y", np.array([], dtype=float))

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

        coords_changed = (
            is_map_change
            or len(self._sites_x) != len(old_sites_x)
            or not (np.array_equal(self._sites_x, old_sites_x) and np.array_equal(self._sites_y, old_sites_y))
        )

        # When map dataset or its coordinate geometry changes, reset rendered artists and cached 2D geometry
        if coords_changed:
            self._cached_2d_geometry = None
            self._last_rendered_map_name = None
            self._last_rendered_cache_key = None
            self.img = None
            self.cbar = None
            if hasattr(self, '_current_dx'):
                del self._current_dx
            if hasattr(self, '_current_dy'):
                del self._current_dy
            if not is_map_change and map_name:
                self.clear_cache_for_map(map_name)

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

    def _update_parameter_lists(self):
        """Update z-parameter and mask parameter lists from fit results."""
        self.cbb_zparameter.blockSignals(True)
        self.cbb_mask_param.blockSignals(True)

        current_z = combo_data_or_text(self.cbb_zparameter)
        current_mask = combo_data_or_text(self.cbb_mask_param)

        # Update Z parameters
        self.cbb_zparameter.clear()
        self.cbb_zparameter.addItem(tr('Intensity'), 'Intensity')
        self.cbb_zparameter.addItem(tr('Area'), 'Area')
        self.cbb_mask_param.clear()

        if not self.df_fit_results.empty:
            fit_columns = [col for col in self.df_fit_results.columns
                          if col not in ['Filename', 'X', 'Y']]
            for col in fit_columns:
                self.cbb_zparameter.addItem(col, col)
                self.cbb_mask_param.addItem(col, col)

        # Restore selection if possible
        self.cbb_zparameter.setCurrentIndex(0)
        if current_z:
            set_combo_by_data_or_text(self.cbb_zparameter, current_z)

        if self.cbb_mask_param.count() > 0:
            self.cbb_mask_param.setCurrentIndex(0)
            if current_mask:
                set_combo_by_data_or_text(self.cbb_mask_param, current_mask)

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
        except Exception:
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
        except Exception:
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
