import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from superqt import QLabeledDoubleRangeSlider 

from spectroview import  ICON_DIR
from spectroview.modules.utils import copy_fig_to_clb
from spectroview.modules.utils import CustomizedPalette

from PySide6.QtWidgets import  QVBoxLayout, QHBoxLayout,  QLabel, QToolButton, QWidgetAction, \
    QLineEdit, QWidget, QPushButton, QComboBox, QCheckBox, QDoubleSpinBox,\
    QApplication,  QWidget, QMenu, QSizePolicy,QFrame, QSpacerItem
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import  QIcon, QAction, Qt

class MapViewer(QWidget):
    """Class to manage the 2Dmap viewer widget"""
    def __init__(self, parent, app_settings):
        super().__init__()
        self.parent = parent
        self.app_settings = app_settings  

        self.map_df_name = None
        self.map_df =pd.DataFrame() 
        self.df_fit_results =pd.DataFrame() 
        self.map_type = '2Dmap'

        self.dpi = 70
        self.figure = None
        self.ax = None
        self.canvas = None
        self.toolbar = None
        self.spectra_listbox = None
        self.menu_actions = {}
        
        # Selection state
        self.selected_points = []
        self.rect_start = None
        self.rect_patch = None
        
        # last computed arrays (for stats & profile)
        self._last_final_z_col = None
        self._last_grid_z = None
        self._last_extent = [0,0,0,0]
        self._last_stats_text_artist = None
        
        self.initUI()

    def initUI(self):
        """Initialize the UI components."""
        self.widget = QWidget()
        self.widget.setFixedWidth(400)
        
        self.map_widget_layout = QVBoxLayout(self.widget)
        self.map_widget_layout.setContentsMargins(0, 0, 0, 0) 
        self._create_ui()
    
    def _create_ui(self):
        """Create 2Dmap plot widgets"""
        # Create a frame to hold the canvas with a fixed siz
        self.canvas_frame = QFrame(self.widget)
        frame_layout = QVBoxLayout(self.canvas_frame)
        frame_layout.setContentsMargins(5, 0, 5, 0)

        self.figure = plt.figure(dpi=70)
        self.ax = self.figure.add_subplot(111)
        
        self.ax.tick_params(axis='x', which='both')
        self.ax.tick_params(axis='y', which='both')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar =NavigationToolbar2QT(self.canvas)
        for action in self.toolbar.actions():
            if action.text() in ['Customize','Zoom','Save', 'Pan', 'Back', 'Forward', 'Subplots']:
                action.setVisible(False)

        frame_layout.addWidget(self.canvas)
        toolbar_layout=QHBoxLayout()
        toolbar_layout.setContentsMargins(5, 0, 5, 0)
        
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.btn_copy = QPushButton("", self)
        icon = QIcon()
        icon.addFile(os.path.join(ICON_DIR, "copy.png"))
        self.btn_copy.setIcon(icon)
        self.btn_copy.setIconSize(QSize(24, 24))
        self.btn_copy.clicked.connect(self.copy_fig)
        
        toolbar_layout.addWidget(self.toolbar)
        toolbar_layout.addItem(spacer)
        
        toolbar_layout.addWidget(self.btn_copy)

        frame_layout.addLayout(toolbar_layout)

        # Add the map frame to the main layout
        self.map_widget_layout.addWidget(self.canvas_frame)

        # Variables to keep track of highlighted points and Ctrl key status
        self.selected_points = []
        
        # Connect the mouse and key events to the handler functions
        self.figure.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.figure.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        
        self.canvas.draw_idle()

        #### MAP_TYPE ComboBox (wafer or 2Dmap)
        combobox_layout = QHBoxLayout()
        self.map_type_label = QLabel("Map Type:")
        self.cbb_map_type = QComboBox(self)
        self.cbb_map_type.setFixedWidth(93)
        self.cbb_map_type.addItems(['2Dmap', 'Wafer_300mm', 'Wafer_200mm', 'Wafer_100mm'])
        
        
        self.cbb_map_type.currentIndexChanged.connect(self.refresh_plot) 
        self.cbb_map_type.currentIndexChanged.connect(self.update_settings) # syc to app_settings
        saved_map_type = self.app_settings.map_type
        
        # Set the saved values in the combo boxes
        if saved_map_type in ['2Dmap', 'Wafer_300mm', 'Wafer_200mm', 'Wafer_100mm']:
            self.cbb_map_type.setCurrentText(saved_map_type)
        
        # Palette selector with preview 
        self.cbb_palette = CustomizedPalette()
        self.cbb_palette.currentIndexChanged.connect(self.refresh_plot)

        self.cb_auto_scale = QCheckBox("Remove Outlier")
        self.cb_auto_scale.setChecked(True)
        self.cb_auto_scale.stateChanged.connect(self.update_z_range_slider)
        self.cb_auto_scale.setToolTip("Automatically adjust the scale by removing outliter data points.")
        
        # Add to the layout
        spacer1 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        combobox_layout.addWidget(self.map_type_label)
        combobox_layout.addWidget(self.cbb_map_type)
        combobox_layout.addWidget(self.cbb_palette)
        combobox_layout.addWidget(self.cb_auto_scale)
        combobox_layout.setContentsMargins(5, 5, 5, 5)

        # Add the combobox layout below the profile layout
        self.map_widget_layout.addLayout(combobox_layout)
        
        #### CREATE range sliders
        self.create_range_sliders(0,100)

        # Create a new layout
        option_menu_layout = QHBoxLayout()
        
        # --- MAP MASKING ELEMENTS ---
        self.mask_cb = QCheckBox("Enable mask:")
        self.mask_cb.setToolTip("Apply mask on heatmap based on another parameter.")
        self.mask_cb.stateChanged.connect(self.refresh_plot)

        self.mask_param_cb = QComboBox()
        self.mask_param_cb.setToolTip("Select the parameter used as mask.")

        self.mask_operator_cb = QComboBox()
        self.mask_operator_cb.addItems([">", "<", ">=", "<=", "=="])
        self.mask_operator_cb.setToolTip("Condition to apply.")

        self.mask_threshold_edit = QDoubleSpinBox()
        self.mask_threshold_edit.setRange(0, 1e12)
        self.mask_threshold_edit.setDecimals(2)
        self.mask_threshold_edit.setValue(0)
        self.mask_threshold_edit.setFixedWidth(100)
        self.mask_threshold_edit.valueChanged.connect(self.refresh_plot)

        # Add to option_menu_layout
        option_menu_layout.addWidget(self.mask_cb)
        option_menu_layout.addWidget(self.mask_param_cb)
        option_menu_layout.addWidget(self.mask_operator_cb)
        option_menu_layout.addWidget(self.mask_threshold_edit)
        
        # Create Options Menu
        self.create_options_menu()
        
        self.tool_btn_options = QToolButton(self)
        self.tool_btn_options.setText("... ")
        self.tool_btn_options.setPopupMode(QToolButton.InstantPopup) 
        self.tool_btn_options.setIcon(QIcon(os.path.join(ICON_DIR, "options.png")))
        self.tool_btn_options.setMenu(self.options_menu) 
        spacer2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        option_menu_layout.addItem(spacer2)
        option_menu_layout.addWidget(self.tool_btn_options)

        self.map_widget_layout.addLayout(option_menu_layout)
        option_menu_layout.setContentsMargins(5, 5, 5, 5)

        vspacer = QSpacerItem(40, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.map_widget_layout.addItem(vspacer)
        
       
    def update_settings(self):
        """Save selected wafer size to settings"""
        map_type = self.cbb_map_type.currentText()
        self.app_settings.map_type = map_type
        self.app_settings.save()

    def create_options_menu(self):
        """Create more view options for 2Dmap plot"""
        
        self.options_menu = QMenu(self)
        # --- ALL MENU OPTIONs ---  
        options = [
            ("Smoothing", "Smoothing", False),
            ("Grid", "Grid", False),
            ("ShowStats", "Show stats", True),
        ]
        
        for option_name, option_label, *checked in options:
            action = QAction(option_label, self)
            action.setCheckable(True)
            action.setChecked(checked[0] if checked else False)
            action.triggered.connect(self.refresh_plot)
            self.menu_actions[option_name] = action
            self.options_menu.addAction(action)  
            
        # --- PROFILE EXTRACTION WIDGET ---  
        profile_widget = QWidget(self)
        profile_widget_layout = QHBoxLayout(profile_widget)
        profile_widget_layout.setContentsMargins(5, 5, 5, 5)
        
        self.profile_name = QLineEdit(self)
        self.profile_name.setText("Profile_1")
        self.profile_name.setPlaceholderText("Profile_name...")
        self.profile_name.setFixedWidth(150)

        self.btn_extract_profile = QPushButton("Extract profil", self)
        self.btn_extract_profile.setToolTip("Extract profile data and plot it in Visu tab")
        self.btn_extract_profile.setFixedWidth(100)
        

        profile_widget_layout.addWidget(self.profile_name)
        profile_widget_layout.addWidget(self.btn_extract_profile)

        # Convert the widget into a menu action
        profile_action = QWidgetAction(self)
        profile_action.setDefaultWidget(profile_widget)

        # Add it at the top of the options menu
        self.options_menu.insertAction(self.options_menu.actions()[0], profile_action)
        self.options_menu.insertSeparator(self.options_menu.actions()[1])
        
        
    def on_histogram_range_changed(self, vmin, vmax):
        self.z_range_slider.setValue((vmin, vmax))
        self.refresh_plot()    
    
    def create_range_sliders(self, xmin, xmax):
        """Create xrange and intensity-range sliders"""
        # ---------------------------------------------------------
        # X-AXIS SLIDER SETUP
        # ---------------------------------------------------------
        self.x_range_slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.x_range_slider.setEdgeLabelMode(QLabeledDoubleRangeSlider.EdgeLabelMode.NoLabel)
        self.x_range_slider.setHandleLabelPosition(QLabeledDoubleRangeSlider.LabelPosition.NoLabel)
        self.x_range_slider.setSingleStep(0.01)
        self.x_range_slider.setRange(xmin, xmax)  
        self.x_range_slider.setValue((xmin, xmax)) 
        self.x_range_slider.setTracking(True)
        self.x_range_slider.valueChanged.connect(self.update_z_range_slider)

        self.x_range_slider_label = QLabel('X-range :')
        self.x_range_slider_label.setFixedWidth(50)  
        self.x_range_slider_label.setToolTip("Define the spectral range (X-range) for 'Area' and 'Maximum Intensity' calculation.")

        # --- Fix checkbox ---
        self.fix_x_checkbox = QCheckBox("Fix")
        self.fix_x_checkbox.setToolTip("If checked, the X-range will not reset when refreshing the plot.")
        self.fix_x_checkbox.stateChanged.connect(self.refresh_plot)

        # Entry boxes for X-range
        self.x_min_edit = QLineEdit(str(xmin))
        self.x_max_edit = QLineEdit(str(xmax))
        self.x_min_edit.setFixedWidth(60)
        self.x_max_edit.setFixedWidth(60)
        
          # Connect entry boxes → slider
        self.x_min_edit.editingFinished.connect(lambda: self._update_slider_from_edit(self.x_range_slider, self.x_min_edit, 0))
        self.x_max_edit.editingFinished.connect(lambda: self._update_slider_from_edit(self.x_range_slider, self.x_max_edit, 1))

        # Connect slider → entry boxes
        self.x_range_slider.valueChanged.connect(lambda v: self._update_edit_from_slider(v, self.x_min_edit, self.x_max_edit))

        self.x_slider_layout = QHBoxLayout()
        self.x_slider_layout.addWidget(self.x_range_slider_label)
        self.x_slider_layout.addWidget(self.fix_x_checkbox)
        self.x_slider_layout.addWidget(self.x_min_edit)
        self.x_slider_layout.addWidget(self.x_range_slider)
        self.x_slider_layout.addWidget(self.x_max_edit)
        self.x_slider_layout.setContentsMargins(5, 0, 5, 0)

        
        # # HISTOGRAM
        # self.histogram_widget = HistogramWidget()
        # self.histogram_widget.setFixedHeight(80)  
        # self.histogram_widget.rangeChanged.connect(self.on_histogram_range_changed)

        # # Add after all layouts and spacers
        # self.map_widget_layout.addWidget(self.histogram_widget)

        # ---------------------------------------------------------
        # Z-AXIS SLIDER SETUP
        # ---------------------------------------------------------
        self.z_range_slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)  
        self.z_range_slider.setEdgeLabelMode(QLabeledDoubleRangeSlider.EdgeLabelMode.NoLabel)
        self.z_range_slider.setHandleLabelPosition(QLabeledDoubleRangeSlider.LabelPosition.NoLabel)
        self.z_range_slider.setSingleStep(0.01)
        self.z_range_slider.setRange(0, 100) 
        self.z_range_slider.setValue((0, 100)) 
        self.z_range_slider.setTracking(True)    

        self.z_values_cbb = QComboBox()
        self.z_values_cbb.addItems(['Intensity', 'Area']) 
        self.z_values_cbb.setFixedWidth(50) 
        self.z_values_cbb.setToolTip("Select parameter to plot 2Dmap")
        self.z_values_cbb.currentIndexChanged.connect(self.update_z_range_slider)
        self.z_range_slider.valueChanged.connect(self.refresh_plot)

        self.fix_z_checkbox = QCheckBox("Fix")
        self.fix_z_checkbox.setToolTip("If checked, the Z-range (Intensity/Area) will not reset when changing maps.")
        # Connect to refresh_plot so the user sees the effect immediately if they toggle it
        self.fix_z_checkbox.stateChanged.connect(self.refresh_plot)
        
        # Entry boxes for Z-range
        self.z_min_edit = QLineEdit("0")
        self.z_max_edit = QLineEdit("100")
        self.z_min_edit.setFixedWidth(60)
        self.z_max_edit.setFixedWidth(60)

        # Connect entry boxes → slider
        self.z_min_edit.editingFinished.connect(lambda: self._update_slider_from_edit(self.z_range_slider, self.z_min_edit, 0))
        self.z_max_edit.editingFinished.connect(lambda: self._update_slider_from_edit(self.z_range_slider, self.z_max_edit, 1))

        # Connect slider → entry boxes
        self.z_range_slider.valueChanged.connect(lambda v: self._update_edit_from_slider(v, self.z_min_edit, self.z_max_edit))

        self.z_slider_layout = QHBoxLayout()
        self.z_slider_layout.addWidget(self.z_values_cbb)
        self.z_slider_layout.addWidget(self.fix_z_checkbox)
        self.z_slider_layout.addWidget(self.z_min_edit)
        self.z_slider_layout.addWidget(self.z_range_slider)
        self.z_slider_layout.addWidget(self.z_max_edit)
        self.z_slider_layout.setContentsMargins(5, 0, 5, 0)
            
        self.map_widget_layout.addLayout(self.z_slider_layout)
        # self.map_widget_layout.addWidget(self.histogram_widget)
        self.map_widget_layout.addLayout(self.x_slider_layout)
    
    def _update_slider_from_edit(self, slider, edit, index):
        """Update slider value when user edits text boxes"""
        try:
            value = max(slider.minimum(), min(slider.maximum(), float(edit.text())))
            current = list(slider.value())
            current[index] = value
            # Ensure min <= max
            if current[0] <= current[1]:
                slider.setValue(tuple(current))
        except ValueError:
            pass  # ignore invalid input

    def _update_edit_from_slider(self, values, min_edit, max_edit):
        """Update entry boxes when slider moves"""
        min_edit.setText(f"{values[0]:.2f}")
        max_edit.setText(f"{values[1]:.2f}")
    
    def populate_z_values_cbb(self):
        self.z_values_cbb.clear() 
        self.z_values_cbb.addItems(['Intensity', 'Area'])
        if not self.df_fit_results.empty:
            fit_columns = [col for col in self.df_fit_results.columns if col not in ['Filename', 'X', 'Y']]
            self.z_values_cbb.addItems(fit_columns)
            
        self.populate_mask_parameters()
            
    def populate_mask_parameters(self):
        """Fill the mask parameter combobox with dataframe column names."""
        self.mask_param_cb.clear() 
        if self.df_fit_results is not None:
            cols = [c for c in self.df_fit_results.columns if c not in ("Filename", "X", "Y")]
            self.mask_param_cb.clear()
            self.mask_param_cb.addItems(cols)
  
    def refresh_plot(self):
        """Call the refresh_gui method of the main application."""
        if hasattr(self.parent, 'refresh_gui'):
            self.parent.refresh_gui()
        else:
            return
        
    def update_xrange_slider(self, xmin, xmax,current_min, current_max):
        """Update the range of the slider based on new min and max values."""        
        self.x_range_slider.setRange(xmin, xmax)
        if self.fix_x_checkbox.isChecked():
            self.x_range_slider.setValue((current_min, current_max))
        else: 
            self.x_range_slider.setValue((xmin, xmax))
    
    def update_z_range_slider(self):
        if self.z_values_cbb.count() > 0 and self.z_values_cbb.currentIndex() >= 0:
            _,_, vmin, vmax, _ , _ =self.get_data_for_heatmap()
            current_handle_min, current_handle_max = self.z_range_slider.value()
            self.z_range_slider.setRange(vmin, vmax)
            
            if self.fix_z_checkbox.isChecked():
                self.z_range_slider.setValue((current_handle_min, current_handle_max))
            else: 
                self.z_range_slider.setValue((vmin, vmax))
        else:
            return
        
        # # Later, after loading the map or computing Z-values:
        # if self._last_final_z_col is not None:
        #     self.histogram_widget.set_data(self._last_final_z_col, vmin, vmax)
        
    
    def get_data_for_heatmap(self, map_type='2Dmap'):
        """
        Returns: heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col
        final_z_col is the 1D array of z-values corresponding to each site (used for stats).
        """

        # Default return values
        heatmap_pivot = pd.DataFrame()
        extent = [0, 0, 0, 0]
        vmin = 0
        vmax = 0
        grid_z = None
        final_z_col = None

        if self.map_df is not None:

            # ------------------------------
            # SLIDER RANGE AND COLUMN FILTER
            # ------------------------------
            xmin, xmax = self.x_range_slider.value()
            column_labels = self.map_df.columns[2:-1]  # keep as strings

            filtered_columns = column_labels[
                (column_labels.astype(float) >= xmin) &
                (column_labels.astype(float) <= xmax)
            ]

            if len(filtered_columns) == 0:
                return heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col

            # Slice X, Y and selected columns
            filtered_map_df = self.map_df[['X', 'Y'] + list(filtered_columns)]
            x_col = filtered_map_df['X'].values
            y_col = filtered_map_df['Y'].values

            # ------------------------------
            # COMPUTE z_col (parameter)
            # ------------------------------
            z_col = None
            parameter = self.z_values_cbb.currentText()

            if parameter == 'Area':
                z_col = (
                    filtered_map_df[filtered_columns]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0)
                    .clip(lower=0)
                    .sum(axis=1)
                )

            elif parameter == 'Intensity':
                z_col = (
                    filtered_map_df[filtered_columns]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0)
                    .clip(lower=0)
                    .max(axis=1)
                )

            else:
                # Fit parameter from results DataFrame
                if not self.df_fit_results.empty:
                    map_name = self.map_df_name
                    filtered_df = self.df_fit_results.query("Filename == @map_name")

                    if not filtered_df.empty and parameter in filtered_df.columns:
                        z_col = filtered_df[parameter]
                    else:
                        z_col = None

            # Safety return
            if z_col is None or len(z_col) == 0:
                return heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col

            # ------------------------------------------------------
            # APPLY MASK (ONLY FOR 2D MAP)
            # ------------------------------------------------------
            if map_type == '2Dmap' and self.mask_cb.isChecked():

                mask_param = self.mask_param_cb.currentText()
                mask_operator = self.mask_operator_cb.currentText()
                threshold = self.mask_threshold_edit.value()

                if mask_param in filtered_map_df.columns:   # raw intensities case
                    mask_values = filtered_map_df[mask_param].astype(float).values

                elif not self.df_fit_results.empty and mask_param in self.df_fit_results.columns:
                    map_name = self.map_df_name
                    df_mask = self.df_fit_results.query("Filename == @map_name")
                    if not df_mask.empty:
                        mask_values = df_mask[mask_param].astype(float).values
                    else:
                        mask_values = None
                else:
                    mask_values = None

                # If mask available → apply it
                if mask_values is not None:

                    # Build boolean mask condition
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

                    # Apply mask: failing points become NaN → blank on heatmap
                    z_col = np.where(valid, z_col, np.nan)

            # ------------------------------------------------------
            # OUTLIER REMOVAL / AUTO SCALE
            # ------------------------------------------------------
            if self.cb_auto_scale.isChecked():
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

                    final_z_col = z_col_interpolated

                except Exception as e:
                    print(f"Auto scale interpolation error: {e}")
                    final_z_col = z_col

            else:
                final_z_col = z_col

            # Determine vmin / vmax
            try:
                vmin = round(final_z_col.min(), 0)
                vmax = round(final_z_col.max(), 0)
            except Exception:
                self.z_values_cbb.setCurrentIndex(0)
                vmin, vmax = 0, 0

            # Count unique sites
            self.number_of_points = len(set(zip(x_col, y_col)))

            # ------------------------------------------------------
            # WAFFER MAP → GRID INTERPOLATION
            # ------------------------------------------------------
            if map_type != '2Dmap' and self.number_of_points >= 4:
                r = self.get_wafer_radius(map_type)
                grid_x, grid_y = np.meshgrid(
                    np.linspace(-r, r, 300),
                    np.linspace(-r, r, 300)
                )
                grid_z = griddata((x_col, y_col), final_z_col, (grid_x, grid_y), method='linear')
                extent = [-r - 1, r + 1, -r - 0.5, r + 0.5]

            else:
                # ------------------------------------------------------
                # 2D MAP PIVOT TABLE
                # ------------------------------------------------------
                heatmap_data = pd.DataFrame({'X': x_col, 'Y': y_col, 'Z': final_z_col})
                heatmap_pivot = heatmap_data.pivot(index='Y', columns='X', values='Z')

                xmin, xmax = x_col.min(), x_col.max()
                ymin, ymax = y_col.min(), y_col.max()
                extent = [xmin, xmax, ymin, ymax]

        return heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col

    
    def get_wafer_radius(self, map_type_text):
        match = re.search(r'Wafer_(\d+)mm', map_type_text)
        if match:
            diameter = int(match.group(1))
            return diameter / 2
        return None
    
    def plot(self, selected_pts):
        """Plot 2D maps of measurement points"""
        if selected_pts is None:
            selected_pts = []
       
        map_type = self.cbb_map_type.currentText()
        r = self.get_wafer_radius(map_type)
        self.ax.clear()
        
        # Plot wafer circle for wafer maps
        if map_type != '2Dmap':
            wafer_circle = patches.Circle((0, 0), radius=r, fill=False, color='black', linewidth=1)
            self.ax.add_patch(wafer_circle)

            all_x, all_y = self.get_mes_sites_coord()
            self.ax.scatter(all_x, all_y, marker='x', color='gray', s=15)
            
            # X labels removed for wafer map, Y kept
            self.ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            self.ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        
        else:
            # 2D map: show both axes labels
            self.ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
            self.ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
            
        # Data preparation    
        heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col  = self.get_data_for_heatmap(map_type)
        self._last_final_z_col = final_z_col
        self._last_grid_z = grid_z
        self._last_extent = extent
        
        color = self.cbb_palette.currentText()
        interpolation_option = 'bilinear' if self.menu_actions['Smoothing'].isChecked() else 'none'
        vmin, vmax = self.z_range_slider.value()
    

        if map_type != '2Dmap' and self.number_of_points >= 4:
            self.img = self.ax.imshow(grid_z, extent=[-r - 0.5, r + 0.5, -r - 0.5, r + 0.5],
                            origin='lower', aspect='equal', cmap=color, interpolation='nearest')
            
        else: 
            self.img = self.ax.imshow(heatmap_pivot, extent=extent, vmin=vmin, vmax=vmax,
                            origin='lower', aspect='equal', cmap=color, interpolation=interpolation_option)
        
        # COLORBAR
        if hasattr(self, 'cbar') and self.cbar is not None:
            self.cbar.update_normal(self.img)
        else:
            self.cbar = self.ax.figure.colorbar(self.img, ax=self.ax)
       
        # Grid
        if self.menu_actions.get('Grid') and self.menu_actions['Grid'].isChecked():
            self.ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
        else:
            self.ax.grid(False)

        # Show stats when requested (wafer only)
        # Stats computed from final_z_col (the one used for values)
        if map_type != '2Dmap' and self.menu_actions.get('ShowStats') and self.menu_actions['ShowStats'].isChecked():
            self._draw_stats_box()
            
        # Highlight selected points
        if selected_pts:
            x, y = zip(*selected_pts)
            self.ax.scatter(x, y,facecolors='none', edgecolors='red', marker='s', s=60, linewidths=1, zorder=10)

            #if map_type == '2Dmap':   
            self.plot_height_profile_on_map(selected_pts)

        title = self.z_values_cbb.currentText()
        self.ax.set_title(title, fontsize=13)
        self.ax.get_figure().tight_layout()
        self.canvas.draw_idle()
    
    def _draw_stats_box(self):
        """Draw mean/min/max/3sigma box on the axes (wafer mode)."""
        # remove old text
        if hasattr(self, '_last_stats_text_artist') and self._last_stats_text_artist is not None:
            try:
                self._last_stats_text_artist.remove()
            except Exception:
                pass
            self._last_stats_text_artist = None

        arr = self._last_final_z_col
        if arr is None:
            return
        # drop NA
        arr_clean = pd.Series(arr).dropna()
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

        # place on top-left inside axes
        self._last_stats_text_artist = self.ax.text(0.02, 0.98, txt,
                                                    transform=self.ax.transAxes,
                                                    fontsize=9, va='top', ha='left',
                                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="black"))


    def plot_height_profile_on_map(self, selected_pts):
        """Plot height profile directly on heatmap"""
        if len(selected_pts) == 2:
            x, y = zip(*selected_pts)
            self.ax.plot(x, y, color='black', linestyle='dotted', linewidth=2)

            # Extract profile values from the heatmap
            profile_df = self.extract_profile()
            if profile_df is not None:
                # Calculate the diagonal length of the heatmap
                extent = self.get_data_for_heatmap()[1]
                diagonal_length = np.sqrt((extent[1] - extent[0])**2 + (extent[3] - extent[2])**2)
                max_normalized_value = 0.3 * diagonal_length
                
                # Calculate height values
                profile_df['height'] = (profile_df['values'] / profile_df['values'].max()) * max_normalized_value
                profile_df['height'] -= profile_df['height'].min()

                x_vals = []
                y_vals = []

                # Calculate perpendicular points
                for i, row in profile_df.iterrows():
                    x_val = row['X']
                    y_val = row['Y']
                    normalized_distance = row['height']

                    # Calculate the direction vector of the line connecting the two points
                    dx = x[1] - x[0]
                    dy = y[1] - y[0]
                    length = np.sqrt(dx**2 + dy**2)

                    # Normalize the direction vector
                    if length > 0:
                        dx /= length
                        dy /= length

                    # Calculate the perpendicular direction
                    perp_dx = -dy
                    perp_dy = dx

                    # Calculate the new x and y values
                    x_vals.append(x_val + perp_dx * normalized_distance)
                    y_vals.append(y_val + perp_dy * normalized_distance)

                # Plot the height profile 
                self.ax.plot(x_vals, y_vals, color='black', linestyle='-', lw=2)

        
    # ----------------------------
    # Mouse interactions (point / rectangle selection)
    # ----------------------------
    def on_mouse_click(self, event):
        """Start selection (left button)."""
        if event.inaxes != self.ax or event.button != 1:
            return
        self.rect_start = (event.xdata, event.ydata)
        # reset potential previous rectangle
        if self.rect_patch is not None:
            try:
                self.rect_patch.remove()
            except Exception:
                pass
            self.rect_patch = None

    def on_mouse_move(self, event):
        """Update rectangle during drag."""
        if self.rect_start is None or event.inaxes != self.ax:
            return
        x0, y0 = self.rect_start
        x1, y1 = event.xdata, event.ydata
        if self.rect_patch is None:
            self.rect_patch = patches.Rectangle(
                (min(x0, x1), min(y0, y1)),
                abs(x1 - x0), abs(y1 - y0),
                linewidth=1.2, edgecolor='black', facecolor='none', linestyle='--'
            )
            self.ax.add_patch(self.rect_patch)
        else:
            self.rect_patch.set_x(min(x0, x1))
            self.rect_patch.set_y(min(y0, y1))
            self.rect_patch.set_width(abs(x1 - x0))
            self.rect_patch.set_height(abs(y1 - y0))
        self.canvas.draw_idle()

    def on_mouse_release(self, event):
        """Finish selection (point or rectangle)."""
        if self.rect_start is None or event.inaxes != self.ax:
            return

        x0, y0 = self.rect_start
        x1, y1 = event.xdata, event.ydata
        all_x, all_y = map(np.array, self.get_mes_sites_coord())
        modifiers = QApplication.keyboardModifiers()

        if self.rect_patch is not None:
            # rectangle selection
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
            inside = (all_x >= xmin) & (all_x <= xmax) & (all_y >= ymin) & (all_y <= ymax)

            if modifiers != Qt.ControlModifier:
                self.selected_points = []

            for x, y in zip(all_x[inside], all_y[inside]):
                self.selected_points.append((float(x), float(y)))

            try:
                self.rect_patch.remove()
            except Exception:
                pass
            self.rect_patch = None
        else:
            # single click -> nearest point
            distances = np.sqrt((all_x - x1) ** 2 + (all_y - y1) ** 2)
            if distances.size == 0:
                self.rect_start = None
                return
            idx = np.argmin(distances)
            pt = (float(all_x[idx]), float(all_y[idx]))
            if modifiers == Qt.ControlModifier:
                self.selected_points.append(pt)
            else:
                self.selected_points = [pt]

        self.rect_start = None
        self.update_spectra_selection()
        self.canvas.draw_idle()
        
    def update_spectra_selection(self):
        """Update external spectra_listbox selection according to selected_points."""
        if self.spectra_listbox is None:
            return
        self.spectra_listbox.clearSelection()
        for index in range(self.spectra_listbox.count()):
            item = self.spectra_listbox.item(index)
            try:
                x, y = map(float, item.text().strip('()').split(','))
            except Exception:
                continue
            if (x, y) in self.selected_points:
                item.setSelected(True)
                self.spectra_listbox.setCurrentRow(index)
            else:
                item.setSelected(False)
            
    def extract_profile(self):
        """Extract a profile from 2D map plot via interpolation."""
        # Ensure exactly two points have been selected
        if len(self.selected_points) != 2:
            print("Select 2 points on map plot to define a profile")
            return None
        (x1, y1), (x2, y2) = self.selected_points
        heatmap_pivot, _, _, _, _, _ = self.get_data_for_heatmap()
        # Extract X and Y coordinates of the heatmap grid
        x_values = heatmap_pivot.columns.values
        y_values = heatmap_pivot.index.values
        z_values = heatmap_pivot.values
        # Interpolate Z values at the sampled points along the profile
        interpolator = RegularGridInterpolator((y_values, x_values), z_values)
        num_samples = 100 
        x_samples = np.linspace(x1, x2, num_samples)
        y_samples = np.linspace(y1, y2, num_samples)
        sample_points = np.vstack((y_samples, x_samples)).T 
        z_samples = interpolator(sample_points)
        # Calculate the distance from (x1, y1) to each sample point
        dists_from_start = np.sqrt((x_samples - x1)**2 + (y_samples - y1)**2)

        profile_df = pd.DataFrame({'X': x_samples, 'Y': y_samples, 'distance': dists_from_start,'values': z_samples})

        self.canvas.draw_idle()
        return profile_df
            
    def get_mes_sites_coord(self):
        """
        Get all coordinates of measurement sites of the selected map.
        """
        df = self.map_df
        all_x = df['X']
        all_y = df['Y']
        return all_x, all_y 

    def copy_fig(self):
        """Copy figure canvas to clipboard"""
        copy_fig_to_clb(self.canvas)