
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

from PySide6.QtWidgets import  QVBoxLayout, QHBoxLayout,  QLabel, QToolButton, \
    QLineEdit, QWidget, QPushButton, QComboBox, QCheckBox, \
    QApplication,  QWidget, QMenu, QSizePolicy,QFrame, QSpacerItem
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import  QIcon, QAction, Qt

class MapViewer(QWidget):
    """Class to manage the 2Dmap view widget"""
    def __init__(self, main_app, app_settings):
        super().__init__()
        self.main_app = main_app
        self.app_settings = app_settings  

        self.map_df_name = None
        self.map_df =pd.DataFrame() 
        self.df_fit_results =pd.DataFrame() 
        self.map_type = '2Dmap'
        # self.map_type = getattr(self.app_settings, "map_type", None)

        self.dpi = 70
        self.figure = None
        self.ax = None
        self.canvas = None
        self.toolbar = None
        self.spectra_listbox = None
        self.menu_actions = {}
        self.initUI()

    def initUI(self):
        """Initialize the UI components."""
        self.widget = QWidget()
        self.widget.setFixedWidth(400)
        self.map_widget_layout = QVBoxLayout(self.widget)
        self.map_widget_layout.setContentsMargins(0, 0, 0, 0) 
        self.create_widget()
    

    def create_widget(self):
        """Create 2Dmap plot widgets"""
        # Create a frame to hold the canvas with a fixed size
        self.canvas_frame = QFrame(self.widget)
        self.canvas_frame.setFixedSize(400, 350)
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
        self.ctrl_pressed = False
        
        # Connect the mouse and key events to the handler functions
        self.figure.canvas.mpl_connect('button_press_event', self.on_left_click_2Dmap)
        self.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.figure.canvas.mpl_connect('key_release_event', self.on_key_release)
        
        self.canvas.draw_idle()

        #### MAP_TYPE ComboBox (wafer or 2Dmap)
        combobox_layout = QHBoxLayout()
        self.map_type_label = QLabel("Map Type:")
        self.cbb_map_type = QComboBox(self)
        self.cbb_map_type.setFixedWidth(93)
        self.cbb_map_type.addItems(['2Dmap', 'Wafer_300mm', 'Wafer_200mm', 'Wafer_100mm'])
        
        
        self.cbb_map_type.currentIndexChanged.connect(self.refresh_plot)

        # Load last saved settings
        self.cbb_map_type.currentIndexChanged.connect(self.update_settings)
        saved_map_type = self.app_settings.map_type
        
        # Set the saved values in the combo boxes
        if saved_map_type in ['2Dmap', 'Wafer_300mm', 'Wafer_200mm', 'Wafer_100mm']:
            self.cbb_map_type.setCurrentText(saved_map_type)
        
        # Palette selector with preview 
        self.cbb_palette = CustomizedPalette()
        self.cbb_palette.currentIndexChanged.connect(self.refresh_plot)

        self.cb_auto_scale = QCheckBox("Auto Scale")
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

        #### EXTRACT profil from 2Dmap
        profile_layout = QHBoxLayout()
        self.profile_name = QLineEdit(self)
        self.profile_name.setText("Profile_1")
        self.profile_name.setPlaceholderText("Profile_name...")
        self.profile_name.setFixedWidth(150)

        self.btn_extract_profile = QPushButton("Extract profil", self)
        self.btn_extract_profile.setToolTip("Extract profile data and plot it in Visu tab")
        self.btn_extract_profile.setFixedWidth(100)
        
        profile_layout.addWidget(self.profile_name)
        profile_layout.addWidget(self.btn_extract_profile)

        # Create Options Menu
        self.create_options_menu()
        
        self.tool_btn_options = QToolButton(self)
        self.tool_btn_options.setText("... ")
        self.tool_btn_options.setPopupMode(QToolButton.InstantPopup) 
        self.tool_btn_options.setMenu(self.options_menu) 
        spacer2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        profile_layout.addItem(spacer2)
        profile_layout.addWidget(self.tool_btn_options)

        #### ADD PROFIL
        self.map_widget_layout.addLayout(profile_layout)
        profile_layout.setContentsMargins(5, 5, 5, 5)

    def update_settings(self):
        """Save selected wafer size to settings"""
        map_type = self.cbb_map_type.currentText()
        self.app_settings.map_type = map_type
        self.app_settings.save()

        #self.settings.setValue("map_type", map_type)
    
    def create_options_menu(self):
        """Create more view options for 2Dmap plot"""
        
        self.options_menu = QMenu(self)
        # Smoothing option
        options = [ ("Smoothing", "Smoothing", False),]
        for option_name, option_label, *checked in options:
            action = QAction(option_label, self)
            action.setCheckable(True)
            action.setChecked(checked[0] if checked else False)
            action.triggered.connect(self.refresh_plot)
            self.menu_actions[option_name] = action
            self.options_menu.addAction(action)  
        
    def create_range_sliders(self, xmin, xmax):
        """Create xrange and intensity-range sliders"""
        # Create x-axis range slider
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

        # Create z-axis range slider
        self.z_range_slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)  
        self.z_range_slider.setEdgeLabelMode(QLabeledDoubleRangeSlider.EdgeLabelMode.NoLabel)
        self.z_range_slider.setHandleLabelPosition(QLabeledDoubleRangeSlider.LabelPosition.NoLabel)
        self.z_range_slider.setSingleStep(0.01)
        self.z_range_slider.setRange(0, 100) 
        self.z_range_slider.setValue((0, 100)) 
        self.z_range_slider.setTracking(True)    

        self.z_values_cbb = QComboBox()
        self.z_values_cbb.addItems(['Max Intensity', 'Area']) 
        self.z_values_cbb.setFixedWidth(97) 
        self.z_values_cbb.setToolTip("Select parameter to plot 2Dmap")
        self.z_values_cbb.currentIndexChanged.connect(self.update_z_range_slider)
        self.z_range_slider.valueChanged.connect(self.refresh_plot)

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
        self.z_slider_layout.addWidget(self.z_min_edit)
        self.z_slider_layout.addWidget(self.z_range_slider)
        self.z_slider_layout.addWidget(self.z_max_edit)
        self.z_slider_layout.setContentsMargins(5, 0, 5, 0)
            
        self.map_widget_layout.addLayout(self.z_slider_layout)
        self.map_widget_layout.addLayout(self.x_slider_layout)

        vspacer = QSpacerItem(40, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.map_widget_layout.addItem(vspacer)
    
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
        self.z_values_cbb.addItems(['Max Intensity', 'Area'])
        if not self.df_fit_results.empty:
            fit_columns = [col for col in self.df_fit_results.columns if col not in ['Filename', 'X', 'Y']]
            self.z_values_cbb.addItems(fit_columns)
  
    def refresh_plot(self):
        """Call the refresh_gui method of the main application."""
        if hasattr(self.main_app, 'refresh_gui'):
            self.main_app.refresh_gui()
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
            _,_, vmin, vmax, _ =self.get_data_for_heatmap()
            self.z_range_slider.setRange(vmin, vmax)
            self.z_range_slider.setValue((vmin, vmax))
        else:
            return
    
    def get_data_for_heatmap(self, map_type='2Dmap'):
        """Prepare data for heatmap based on range sliders values"""

        # Default return values in case of no valid map_df or filtered columns
        heatmap_pivot = pd.DataFrame()  # Empty DataFrame for heatmap
        extent = [0, 0, 0, 0]  # Default extent values
        vmin = 0
        vmax = 0
        grid_z = None # for waferplot
        
        if self.map_df is not None:
            xmin, xmax = self.x_range_slider.value()
            column_labels = self.map_df.columns[2:-1]  # Keep labels as strings

            # Convert slider range values to strings for comparison
            filtered_columns = column_labels[(column_labels.astype(float) >= xmin) &
                                            (column_labels.astype(float) <= xmax)]
            
            if len(filtered_columns) > 0:
                # Create a filtered DataFrame including X, Y, and the selected range of columns
                filtered_map_df = self.map_df[['X', 'Y'] + list(filtered_columns)]
                x_col = filtered_map_df['X'].values
                y_col = filtered_map_df['Y'].values
                final_z_col = []

                parameter = self.z_values_cbb.currentText()
                if parameter == 'Area':
                    # Intensity sums of of each spectrum over the selected range
                    z_col = filtered_map_df[filtered_columns].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0).sum(axis=1)
                    
                elif parameter == 'Max Intensity':
                    # Max intensity value of each spectrum over the selected range
                    z_col = filtered_map_df[filtered_columns].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0).max(axis=1)
                else:
                    if not self.df_fit_results.empty:
                        map_name = self.map_df_name
                        # Plot only selected wafer/2Dmap
                        filtered_df = self.df_fit_results.query('Filename == @map_name')
                        if not filtered_df.empty and parameter in filtered_df.columns:
                            z_col = filtered_df[parameter]
                        else:
                            z_col = None
                
                # Auto scale 
                if self.cb_auto_scale.isChecked():
                    # Remove outliers using IQR method and replace them with interpolated values
                    Q1 = z_col.quantile(0.05)
                    Q3 = z_col.quantile(0.95)
                    IQR = Q3 - Q1
                    # Identify the outliers
                    outlier_mask = (z_col < (Q1 - 1.5 * IQR)) | (z_col > (Q3 + 1.5 * IQR))
                    # Interpolate values for the outliers using linear interpolation
                    z_col_interpolated = z_col.copy()
                    z_col_interpolated[outlier_mask] = np.nan  # Mark outliers as NaN for interpolation
                    z_col_interpolated = z_col_interpolated.interpolate(method='linear', limit_direction='both')
                    final_z_col=z_col_interpolated
                else:
                    final_z_col=z_col  

                try:
                    vmin = round(final_z_col.min(), 0)
                    vmax = round(final_z_col.max(), 0)
                except Exception as e:
                    #When the selected 'parameters' does not exist for selected wafer.
                    self.z_values_cbb.setCurrentIndex(0)

                    vmin = round(final_z_col.min(), 0)
                    vmax = round(final_z_col.max(), 0)

                self.number_of_points = len(set(zip(x_col, y_col)))
                
                if map_type != '2Dmap' and self.number_of_points >= 4:
                    # Create meshgrid for WaferPlot
                    r = self.get_wafer_radius(map_type)
                    grid_x, grid_y = np.meshgrid(np.linspace(-r, r, 300), np.linspace(-r, r, 300))
                    grid_z = griddata((x_col, y_col), final_z_col, (grid_x, grid_y), method='linear')
                    extent = [-r - 1, r + 1, -r - 0.5, r + 0.5]

                else:
                    # Regular 2D map
                    heatmap_data = pd.DataFrame({'X': x_col, 'Y': y_col, 'Z': z_col})
                    heatmap_pivot = heatmap_data.pivot(index='Y', columns='X', values='Z')
                    xmin, xmax = x_col.min(), x_col.max()
                    ymin, ymax = y_col.min(), y_col.max()
                    extent = [xmin, xmax, ymin, ymax]
                
        return heatmap_pivot, extent, vmin, vmax, grid_z
    
    def get_wafer_radius(self, map_type_text):
        match = re.search(r'Wafer_(\d+)mm', map_type_text)
        if match:
            diameter = int(match.group(1))
            return diameter / 2
        return None
    
    def plot(self, coords):
        """Plot 2D maps of measurement points"""
        map_type = self.cbb_map_type.currentText()
        r = self.get_wafer_radius(map_type)
        self.ax.clear()
        
        # Plot wafer map
        if map_type != '2Dmap':
            wafer_circle = patches.Circle((0, 0), radius=r, fill=False,
                                        color='black', linewidth=1)
            self.ax.add_patch(wafer_circle)
            self.ax.set_yticklabels([])

            all_x, all_y = self.get_mes_sites_coord()
            self.ax.scatter(all_x, all_y, marker='x', color='gray', s=15)
            self.ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
            
            
        heatmap_pivot, extent, vmin, vmax, grid_z  = self.get_data_for_heatmap(map_type)
        color = self.cbb_palette.currentText()
        interpolation_option = 'bilinear' if self.menu_actions['Smoothing'].isChecked() else 'none'
        vmin, vmax = self.z_range_slider.value()
    

        if map_type != '2Dmap' and self.number_of_points >= 4:
            self.img = self.ax.imshow(grid_z, extent=[-r - 0.5, r + 0.5, -r - 0.5, r + 0.5],
                            origin='lower', aspect='equal', cmap=color, interpolation='nearest')
            
        else: 
            self.img = self.ax.imshow(heatmap_pivot, extent=extent, vmin=vmin, vmax=vmax,
                            origin='lower', aspect='equal', cmap=color, interpolation=interpolation_option)
            #print(f'2Dmap: {heatmap_pivot}')
        
        # COLORBAR
        if hasattr(self, 'cbar') and self.cbar is not None:
            self.cbar.update_normal(self.img)
        else:
            self.cbar = self.ax.figure.colorbar(self.img, ax=self.ax)

        # MEASUREMENT SITES
        if coords:
            x, y = zip(*coords)
            self.ax.scatter(x, y, marker='o', color='red', s=20)

            #if map_type == '2Dmap':   
            self.plot_height_profile_on_map(coords)

        title = self.z_values_cbb.currentText()
        self.ax.set_title(title, fontsize=13)
        self.ax.get_figure().tight_layout()
        self.canvas.draw_idle()

    def plot_height_profile_on_map(self, coords):
        """Plot height profile directly on heatmap"""
        if len(coords) == 2:
            x, y = zip(*coords)
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

        
    def on_left_click_2Dmap(self, event):
        """select the measurement points via 2Dmap plot"""
        all_x, all_y = self.get_mes_sites_coord()
        self.spectra_listbox.clearSelection()
        if event.inaxes == self.ax:
            x_clicked, y_clicked = event.xdata, event.ydata
            if event.button == 1:  # Left mouse button
                all_x = np.array(all_x)
                all_y = np.array(all_y)
                distances = np.sqrt(
                    (all_x - x_clicked) ** 2 + (all_y - y_clicked) ** 2)
                nearest_index = np.argmin(distances)
                nearest_x, nearest_y = all_x[nearest_index], all_y[
                    nearest_index]

                # Check if Ctrl key is pressed
                modifiers = QApplication.keyboardModifiers()
                if modifiers == Qt.ControlModifier:
                    self.selected_points.append((nearest_x, nearest_y))
                else:
                    # Clear the selected points list and add the current one
                    self.selected_points = [(nearest_x, nearest_y)]

        # Set the current selection in the spectra_listbox
        for index in range(self.spectra_listbox.count()):
            item = self.spectra_listbox.item(index)
            item_text = item.text()
            x, y = map(float, item_text.strip('()').split(','))
            if (x, y) in self.selected_points:
                item.setSelected(True)
                self.current_row= index
                self.spectra_listbox.setCurrentRow(self.current_row)
            else:
                item.setSelected(False) 
            
    def extract_profile(self):
        """Extract a profile from 2D map plot via interpolation."""
        # Ensure exactly two points have been selected
        if len(self.selected_points) != 2:
            print("Select 2 points on map plot to define a profile")
            return None
        (x1, y1), (x2, y2) = self.selected_points
        heatmap_pivot, _, _, _, _ = self.get_data_for_heatmap()
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
            
    def on_key_press(self, event):
        """Handler function for key press event"""
        if event.key == 'ctrl':
            self.ctrl_pressed = True

    def on_key_release(self, event):
        """Handler function for key release event"""
        if event.key == 'ctrl':
            self.ctrl_pressed = False

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