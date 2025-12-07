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

from spectroview import ICON_DIR
from spectroview.modules.utils import copy_fig_to_clb
from spectroview.modules.utils import CustomizedPalette

from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QToolButton, \
    QLineEdit, QWidget, QPushButton, QComboBox, QCheckBox, QApplication, QMenu, QSizePolicy, QFrame, QSpacerItem
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QAction

class MapViewer(QWidget):
    """
    Class to manage a single 2D map / wafer viewer widget.
    Each instance has its own Figure/Canvas so multiple viewers can be created independently.
    """
    def __init__(self, parent, app_settings, fixed_width=400, fixed_height=350):
        super().__init__()
        self.parent = parent
        self.app_settings = app_settings

        # Data holders
        self.map_df_name = None
        self.map_df = pd.DataFrame()
        self.df_fit_results = pd.DataFrame()

        # UI / plotting
        self.map_type = '2Dmap'
        self.dpi = 70
        self.figure = None
        self.ax = None
        self.canvas = None
        self.toolbar = None
        self.spectra_listbox = None  # assigned externally
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

        # Build UI
        self._fixed_width = fixed_width
        self._fixed_height = fixed_height
        self.initUI()

    # ----------------------------
    # Plotting / data helpers
    # ----------------------------
    def plot(self, selected_pts=None):
        """Plot 2D maps of measurement points. selected_pts = list of (x,y) tuples"""
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
        heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col = self.get_data_for_heatmap(map_type)
        self._last_final_z_col = final_z_col
        self._last_grid_z = grid_z
        self._last_extent = extent

        color = self.cbb_palette.currentText()
        interpolation_option = 'bilinear' if self.menu_actions.get('Smoothing', None) and self.menu_actions['Smoothing'].isChecked() else 'none'
        vmin, vmax = self.z_range_slider.value()

        if map_type != '2Dmap' and grid_z is not None and self.number_of_points >= 4:
            self.img = self.ax.imshow(grid_z, extent=[-r - 0.5, r + 0.5, -r - 0.5, r + 0.5],
                                      origin='lower', aspect='equal', cmap=color, interpolation='nearest',
                                      vmin=vmin, vmax=vmax)
        else:
            # pivot can be empty if no data -> handle gracefully
            if heatmap_pivot is None or heatmap_pivot.empty:
                # blank image
                self.ax.text(0.5, 0.5, "No data", transform=self.ax.transAxes, ha='center', va='center')
                self.img = None
            else:
                self.img = self.ax.imshow(heatmap_pivot, extent=extent, vmin=vmin, vmax=vmax,
                                          origin='lower', aspect='equal', cmap=color, interpolation=interpolation_option)

        # colorbar management
        if hasattr(self, 'cbar') and self.cbar is not None and self.img is not None:
            self.cbar.update_normal(self.img)
        elif self.img is not None:
            self.cbar = self.ax.figure.colorbar(self.img, ax=self.ax)

        # optional grid
        if self.menu_actions.get('Grid') and self.menu_actions['Grid'].isChecked():
            self.ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
        else:
            self.ax.grid(False)

        # show stats when requested (wafer only)
        # Stats computed from final_z_col (the one used for values)
        if map_type != '2Dmap' and self.menu_actions.get('ShowStats') and self.menu_actions['ShowStats'].isChecked():
            self._draw_stats_box()

        # Highlight selected points
        if selected_pts:
            x, y = zip(*selected_pts)
            self.ax.scatter(x, y, facecolors='none', edgecolors='red', marker='s', s=60, linewidths=1, zorder=10)
            if len(selected_pts) == 2:
                self.plot_height_profile_on_map(selected_pts)

        # Title
        title = self.z_values_cbb.currentText()
        self.ax.set_title(title, fontsize=13)

        # finalize & redraw
        self.figure.tight_layout()
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

        txt = (f"mean: {mean:.3f}\n"
               f"min: {mn:.3f}\n"
               f"max: {mx:.3f}\n"
               f"3σ: {sigma3:.3f}")

        # place on top-left inside axes
        self._last_stats_text_artist = self.ax.text(0.02, 0.98, txt,
                                                    transform=self.ax.transAxes,
                                                    fontsize=9, va='top', ha='left',
                                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="black"))

    def plot_height_profile_on_map(self, selected_pts):
        """Plot height profile directly on heatmap (visual only)."""
        if len(selected_pts) != 2:
            return
        (x0, y0), (x1, y1) = selected_pts
        # simple line
        self.ax.plot([x0, x1], [y0, y1], color='black', linestyle='dotted', linewidth=2)

        profile_df = self.extract_profile()
        if profile_df is None:
            return

        extent = self._last_extent
        # normalize heights to a fraction of diagonal length
        diagonal_length = np.sqrt((extent[1] - extent[0])**2 + (extent[3] - extent[2])**2)
        if profile_df['values'].max() == 0:
            return
        max_normalized_value = 0.3 * diagonal_length
        profile_df['height'] = (profile_df['values'] / profile_df['values'].max()) * max_normalized_value
        profile_df['height'] -= profile_df['height'].min()

        x_vals = []
        y_vals = []
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            return
        dx /= length; dy /= length
        perp_dx = -dy; perp_dy = dx

        for _, row in profile_df.iterrows():
            xv = row['X'] + perp_dx * row['height']
            yv = row['Y'] + perp_dy * row['height']
            x_vals.append(xv); y_vals.append(yv)

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

    # ----------------------------
    # Data & interpolation
    # ----------------------------
    def extract_profile(self):
        """Interpolate z values on profile connecting two selected points (returns DataFrame)."""
        if len(self.selected_points) != 2:
            return None
        (x1, y1), (x2, y2) = self.selected_points
        heatmap_pivot, _, _, _, _, _ = self.get_data_for_heatmap()
        if heatmap_pivot is None or heatmap_pivot.empty:
            return None
        x_values = heatmap_pivot.columns.values
        y_values = heatmap_pivot.index.values
        z_values = heatmap_pivot.values
        interpolator = RegularGridInterpolator((y_values, x_values), z_values, bounds_error=False, fill_value=np.nan)
        num_samples = 100
        x_samples = np.linspace(x1, x2, num_samples)
        y_samples = np.linspace(y1, y2, num_samples)
        sample_points = np.vstack((y_samples, x_samples)).T
        z_samples = interpolator(sample_points)
        dists_from_start = np.sqrt((x_samples - x1)**2 + (y_samples - y1)**2)
        profile_df = pd.DataFrame({'X': x_samples, 'Y': y_samples, 'distance': dists_from_start, 'values': z_samples})
        return profile_df

    def get_mes_sites_coord(self):
        """Return X,Y arrays from self.map_df"""
        if self.map_df is None or self.map_df.empty:
            return np.array([]), np.array([])
        all_x = self.map_df['X'].values
        all_y = self.map_df['Y'].values
        return all_x, all_y

    # ----------------------------
    # Options / settings
    # ----------------------------
    def copy_fig(self):
        copy_fig_to_clb(self.canvas)

    def update_settings(self):
        map_type = self.cbb_map_type.currentText()
        self.app_settings.map_type = map_type
        self.app_settings.save()

    def create_options_menu(self):
        """Create more view options for 2Dmap plot"""
        self.options_menu = QMenu(self)
        options = [
            ("RemoveOutliers", "RemoveOutliers", True),
            ("Smoothing", "Smoothing", False),
            ("Grid", "Grid", False),
            ("ShowStats", "Show stats", True),
        ]
        for option_name, option_label, *checked in options:
            action = QAction(option_label, self)
            action.setCheckable(True)
            action.setChecked(checked[0] if checked else False)
            if option_name == "RemoveOutliers":
                action.triggered.connect(self.update_z_range_slider)
                
            else:
                action.triggered.connect(self.refresh_plot)
            self.menu_actions[option_name] = action
            self.options_menu.addAction(action)

    # ----------------------------
    # Heatmap data preparation
    # ----------------------------
    def _update_slider_from_edit(self, slider, edit, index):
        try:
            value = max(slider.minimum(), min(slider.maximum(), float(edit.text())))
            current = list(slider.value())
            current[index] = value
            if current[0] <= current[1]:
                slider.setValue(tuple(current))
        except ValueError:
            pass

    def _update_edit_from_slider(self, values, min_edit, max_edit):
        min_edit.setText(f"{values[0]:.2f}")
        max_edit.setText(f"{values[1]:.2f}")

    def populate_z_values_cbb(self):
        self.z_values_cbb.clear()
        self.z_values_cbb.addItems(['Max Intensity', 'Area'])
        if not self.df_fit_results.empty:
            fit_columns = [col for col in self.df_fit_results.columns if col not in ['Filename', 'X', 'Y']]
            self.z_values_cbb.addItems(fit_columns)

    def update_xrange_slider(self, xmin, xmax, current_min, current_max):
        self.x_range_slider.setRange(xmin, xmax)
        if self.fix_x_checkbox.isChecked():
            self.x_range_slider.setValue((current_min, current_max))
        else:
            self.x_range_slider.setValue((xmin, xmax))

    def update_z_range_slider(self):
        if self.z_values_cbb.count() > 0 and self.z_values_cbb.currentIndex() >= 0:
            _,_, vmin, vmax, _, _ = self.get_data_for_heatmap()
            self.z_range_slider.setRange(vmin, vmax)
            self.z_range_slider.setValue((vmin, vmax))
        else:
            return

    def get_data_for_heatmap(self, map_type='2Dmap'):
        """
        Returns: heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col
        final_z_col is the 1D array of z-values corresponding to each site (used for stats).
        """
        heatmap_pivot = pd.DataFrame()
        extent = [0,0,0,0]
        vmin = 0; vmax = 0
        grid_z = None
        final_z_col = None
        if self.map_df is None or self.map_df.empty:
            return heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col

        xmin, xmax = self.x_range_slider.value()
        column_labels = self.map_df.columns[2:-1]  # keep labels as strings
        filtered_columns = column_labels[(column_labels.astype(float) >= xmin) & (column_labels.astype(float) <= xmax)]

        if len(filtered_columns) == 0:
            return heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col

        filtered_map_df = self.map_df[['X', 'Y'] + list(filtered_columns)]
        x_col = filtered_map_df['X'].values
        y_col = filtered_map_df['Y'].values

        parameter = self.z_values_cbb.currentText()
        if parameter == 'Area':
            z_col = filtered_map_df[filtered_columns].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0).sum(axis=1)
        elif parameter == 'Max Intensity':
            z_col = filtered_map_df[filtered_columns].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0).max(axis=1)
        else:
            # fit results based parameter
            if not self.df_fit_results.empty:
                map_name = self.map_df_name
                filtered_df = self.df_fit_results.query('Filename == @map_name')
                if not filtered_df.empty and parameter in filtered_df.columns:
                    z_col = filtered_df[parameter].reindex(range(len(filtered_map_df))).values
                else:
                    z_col = pd.Series(np.nan, index=filtered_map_df.index)
            else:
                z_col = pd.Series(np.nan, index=filtered_map_df.index)

        # optional outlier handling
        if self.menu_actions.get('RemoveOutliers') and self.menu_actions['RemoveOutliers'].isChecked(): 
            try:
                Q1 = z_col.quantile(0.05)
                Q3 = z_col.quantile(0.95)
                IQR = Q3 - Q1
                outlier_mask = (z_col < (Q1 - 1.5 * IQR)) | (z_col > (Q3 + 1.5 * IQR))
                z_col_interpolated = z_col.copy()
                z_col_interpolated[outlier_mask] = np.nan
                z_col_interpolated = z_col_interpolated.interpolate(method='linear', limit_direction='both')
                final_z_col = z_col_interpolated
            except Exception:
                final_z_col = z_col
        else:
            final_z_col = z_col

        try:
            vmin = round(final_z_col.min(), 0)
            vmax = round(final_z_col.max(), 0)
        except Exception:
            vmin, vmax = 0, 0

        # store number of unique sites
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

        return heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col

    def get_wafer_radius(self, map_type_text):
        match = re.search(r'Wafer_(\d+)mm', map_type_text)
        if match:
            diameter = int(match.group(1))
            return diameter / 2
        return None

    # ----------------------------
    # UI construction
    # ----------------------------
    def initUI(self):
        self.widget = QWidget()
        self.widget.setFixedWidth(self._fixed_width)
        self.map_widget_layout = QVBoxLayout(self.widget)
        self.map_widget_layout.setContentsMargins(0, 0, 0, 0)
        self._setup_ui()

    def _setup_ui(self):
        self.canvas_frame = QFrame(self.widget)
        self.canvas_frame.setFixedSize(self._fixed_width, self._fixed_height)
        frame_layout = QVBoxLayout(self.canvas_frame)
        frame_layout.setContentsMargins(5, 0, 5, 0)

        self.figure = plt.figure(dpi=self.dpi)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas)
        for action in self.toolbar.actions():
            if action.text() in ['Customize','Zoom','Save', 'Pan', 'Back', 'Forward', 'Subplots']:
                action.setVisible(False)

        frame_layout.addWidget(self.canvas)
        toolbar_layout = QHBoxLayout()
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

        self.map_widget_layout.addWidget(self.canvas_frame)

        # selection state
        self.selected_points = []
        self.ctrl_pressed = False

        # connect events
        self.figure.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.figure.canvas.mpl_connect('button_release_event', self.on_mouse_release)

        self.canvas.draw_idle()

        # controls (map type, palette, autoscale, etc.)
        combobox_layout = QHBoxLayout()
        self.map_type_label = QLabel("Map Type:")
        self.cbb_map_type = QComboBox(self)
        self.cbb_map_type.setFixedWidth(93)
        self.cbb_map_type.addItems(['2Dmap', 'Wafer_300mm', 'Wafer_200mm', 'Wafer_100mm'])
        self.cbb_map_type.currentIndexChanged.connect(self.refresh_plot)
        self.cbb_map_type.currentIndexChanged.connect(self.update_settings)
        saved_map_type = getattr(self.app_settings, 'map_type', '2Dmap')
        if saved_map_type in ['2Dmap', 'Wafer_300mm', 'Wafer_200mm', 'Wafer_100mm']:
            self.cbb_map_type.setCurrentText(saved_map_type)

        self.cbb_palette = CustomizedPalette()
        self.cbb_palette.currentIndexChanged.connect(self.refresh_plot)

        # self.cb_auto_scale = QCheckBox("Auto Scale")
        # self.cb_auto_scale.setChecked(True)
        # self.cb_auto_scale.stateChanged.connect(self.update_z_range_slider)
        # self.cb_auto_scale.setToolTip("Automatically adjust the scale by removing outlier data points.")

        combobox_layout.addWidget(self.map_type_label)
        combobox_layout.addWidget(self.cbb_map_type)
        combobox_layout.addWidget(self.cbb_palette)
        #combobox_layout.addWidget(self.cb_auto_scale)
        combobox_layout.setContentsMargins(5, 5, 5, 5)
        self.map_widget_layout.addLayout(combobox_layout)

        # sliders and extract profile button
        self.create_range_sliders(0,100)

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

        self.create_options_menu()
        self.tool_btn_options = QToolButton(self)
        self.tool_btn_options.setText("... ")
        self.tool_btn_options.setPopupMode(QToolButton.InstantPopup)
        self.tool_btn_options.setIcon(QIcon(os.path.join(ICON_DIR, "options.png")))
        self.tool_btn_options.setMenu(self.options_menu)

        spacer2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        profile_layout.addItem(spacer2)
        profile_layout.addWidget(self.tool_btn_options)

        self.map_widget_layout.addLayout(profile_layout)
        profile_layout.setContentsMargins(5, 5, 5, 5)

        vspacer = QSpacerItem(40, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.map_widget_layout.addItem(vspacer)

    def create_range_sliders(self, xmin, xmax):
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

        self.fix_x_checkbox = QCheckBox("Fix")
        self.fix_x_checkbox.setToolTip("If checked, the X-range will not reset when refreshing the plot.")
        self.fix_x_checkbox.stateChanged.connect(self.refresh_plot)

        self.x_min_edit = QLineEdit(str(xmin)); self.x_max_edit = QLineEdit(str(xmax))
        self.x_min_edit.setFixedWidth(60); self.x_max_edit.setFixedWidth(60)
        self.x_min_edit.editingFinished.connect(lambda: self._update_slider_from_edit(self.x_range_slider, self.x_min_edit, 0))
        self.x_max_edit.editingFinished.connect(lambda: self._update_slider_from_edit(self.x_range_slider, self.x_max_edit, 1))
        self.x_range_slider.valueChanged.connect(lambda v: self._update_edit_from_slider(v, self.x_min_edit, self.x_max_edit))

        self.x_slider_layout = QHBoxLayout()
        self.x_slider_layout.addWidget(self.x_range_slider_label)
        self.x_slider_layout.addWidget(self.fix_x_checkbox)
        self.x_slider_layout.addWidget(self.x_min_edit)
        self.x_slider_layout.addWidget(self.x_range_slider)
        self.x_slider_layout.addWidget(self.x_max_edit)
        self.x_slider_layout.setContentsMargins(5, 0, 5, 0)

        # z slider
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

        self.z_min_edit = QLineEdit("0"); self.z_max_edit = QLineEdit("100")
        self.z_min_edit.setFixedWidth(60); self.z_max_edit.setFixedWidth(60)
        self.z_min_edit.editingFinished.connect(lambda: self._update_slider_from_edit(self.z_range_slider, self.z_min_edit, 0))
        self.z_max_edit.editingFinished.connect(lambda: self._update_slider_from_edit(self.z_range_slider, self.z_max_edit, 1))
        self.z_range_slider.valueChanged.connect(lambda v: self._update_edit_from_slider(v, self.z_min_edit, self.z_max_edit))

        self.z_slider_layout = QHBoxLayout()
        self.z_slider_layout.addWidget(self.z_values_cbb)
        self.z_slider_layout.addWidget(self.z_min_edit)
        self.z_slider_layout.addWidget(self.z_range_slider)
        self.z_slider_layout.addWidget(self.z_max_edit)
        self.z_slider_layout.setContentsMargins(5, 0, 5, 0)

        self.map_widget_layout.addLayout(self.z_slider_layout)
        self.map_widget_layout.addLayout(self.x_slider_layout)

    # ----------------------------
    # Integration with parent
    # ----------------------------
    def refresh_plot(self):
        """Ask parent to refresh full GUI/plot (keeps central logic in Maps class)"""
        if hasattr(self.parent, 'refresh_gui'):
            self.parent.refresh_gui()

    def clear(self):
        """Clear figure"""
        if self.ax:
            self.ax.clear()
            self.canvas.draw_idle()

# ----------------------------
# Manager for multiple viewers
# ----------------------------
class MapViewerManager:
    """
    Lightweight manager to create and manage multiple MapViewer instances.
    Usage:
        mgr = MapViewerManager()
        v0 = mgr.create_viewer(parent, app_settings)
        v1 = mgr.create_viewer(parent, app_settings)
        mgr.register(v0)  # optional
        mgr.update_all_maps(map_name, map_df)
    """
    def __init__(self):
        self.viewers = []

    def create_viewer(self, parent, app_settings, fixed_width=400, fixed_height=350):
        v = MapViewer(parent, app_settings, fixed_width=fixed_width, fixed_height=fixed_height)
        self.viewers.append(v)
        return v

    def register(self, viewer):
        if viewer not in self.viewers:
            self.viewers.append(viewer)

    def unregister(self, viewer):
        if viewer in self.viewers:
            self.viewers.remove(viewer)

    def update_all_maps(self, map_name, map_df):
        """Set map_df/map_name on all managed viewers and update sliders."""
        for v in self.viewers:
            v.map_df_name = map_name
            v.map_df = map_df
            # update x-range slider according to df if available:
            try:
                column_labels = map_df.columns[2:-1].astype(float)
                current_min, current_max = v.x_range_slider.value()
                v.update_xrange_slider(float(column_labels.min()), float(column_labels.max()), current_min, current_max)
            except Exception:
                pass

    def broadcast_fit_results(self, df_fit_results):
        for v in self.viewers:
            v.df_fit_results = df_fit_results
            v.populate_z_values_cbb()

    def refresh_all(self):
        for v in self.viewers:
            v.refresh_plot()
