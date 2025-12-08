import os

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from copy import deepcopy

from spectroview import X_AXIS_UNIT, ICON_DIR, PLOT_POLICY
from spectroview.modules.utils import plot_baseline_dynamically, copy_fig_to_clb, show_alert

from PySide6.QtWidgets import  QWidgetAction, QHBoxLayout, QLabel, QToolButton, QDoubleSpinBox,\
    QLineEdit, QWidget, QPushButton, QComboBox, QApplication,  QWidget, QMenu, QColorDialog, QInputDialog
from PySide6.QtCore import Qt, QSize, QPoint
from PySide6.QtGui import  QIcon, QAction, Qt, QCursor
from PySide6.QtCore import QSettings

class SpectraViewer(QWidget):
    """Class to manage the spectra view widget."""
    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.settings = QSettings("CEA-Leti", "SPECTROview") 
         
        self.peak_model = 'Lorentzian'
        self.dpi = 80
        self.figure = None
        self.ax = None
        self.canvas = None
        self.toolbar = None
        self.zoom_pan_active = False
        self.menu_actions = {}

        self.dragging_peak = None
        self.drag_event_connection = None
        self.release_event_connection = None

        self.initUI()
        QApplication.instance().focusChanged.connect(self.on_focus_changed) # To hide tooltip when focus is lost

    def initUI(self):
        """Initialize the UI components."""
        self.create_plot_widget()

    def create_plot_widget(self):
        """Create or update canvas and toolbar for plotting in the GUI."""
        plt.style.use(PLOT_POLICY)
        if not self.figure:
            self.create_figure_canvas_and_toolbar()
            self.create_tool_buttons()

            self.create_options_menu()
            self.create_normalization_widgets()
            self.create_copy_and_legend_buttons()
            self.create_control_layout()

        self.update_plot_styles()

    def create_figure_canvas_and_toolbar(self):
        self.figure = plt.figure(dpi=self.dpi)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)

        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.zoom()  # Default zoom active

        for action in self.toolbar.actions():
            if action.text() in ['Home', 'Save', 'Pan', 'Back', 'Forward', 'Subplots', 'Zoom']:
                action.setVisible(False)
    
    def create_tool_buttons(self):
        self.btn_rescale = QPushButton("", self)
        self.btn_rescale.setToolTip("Rescale")
        self.btn_rescale.setIcon(QIcon(os.path.join(ICON_DIR, "rescale.png")))
        self.btn_rescale.setIconSize(QSize(24, 24))
        self.btn_rescale.clicked.connect(self.rescale)

        self.btn_zoom = QToolButton(self)
        self.btn_zoom.setCheckable(True)
        self.btn_zoom.setAutoExclusive(False)
        self.btn_zoom.setToolTip("Zoom")
        self.btn_zoom.setIcon(QIcon(os.path.join(ICON_DIR, "zoom.png")))
        self.btn_zoom.setIconSize(QSize(24, 24))
        self.btn_zoom.setChecked(True)
        self.btn_zoom.toggled.connect(self.toggle_zoom_pan)

        self.btn_baseline = QToolButton(self)
        self.btn_baseline.setCheckable(True)
        self.btn_baseline.setAutoExclusive(False)
        self.btn_baseline.setToolTip("Baseline")
        self.btn_baseline.setIcon(QIcon(os.path.join(ICON_DIR, "baseline.png")))
        self.btn_baseline.setIconSize(QSize(24, 24))

        self.btn_peak = QToolButton(self)
        self.btn_peak.setCheckable(True)
        self.btn_peak.setAutoExclusive(False)
        self.btn_peak.setToolTip("Peak")
        self.btn_peak.setIcon(QIcon(os.path.join(ICON_DIR, "peak.png")))
        self.btn_peak.setIconSize(QSize(24, 24))
        
        # Connect toggled signals
        self.btn_zoom.toggled.connect(lambda checked: self._toggle_tool_buttons(self.btn_zoom, checked))
        self.btn_baseline.toggled.connect(lambda checked: self._toggle_tool_buttons(self.btn_baseline, checked))
        self.btn_peak.toggled.connect(lambda checked: self._toggle_tool_buttons(self.btn_peak, checked))
        
    def _toggle_tool_buttons(self, btn, checked):
        if checked:
            # Uncheck all other buttons
            for other in [self.btn_zoom, self.btn_baseline, self.btn_peak]:
                if other != btn:
                    other.setChecked(False)

    def create_normalization_widgets(self):
        self.btn_norm = QToolButton(self)
        self.btn_norm.setCheckable(True)
        self.btn_norm.setAutoExclusive(False)
        self.btn_norm.setToolTip("Normalization")
        self.btn_norm.setIcon(QIcon(os.path.join(ICON_DIR, "norm.png")))
        self.btn_norm.setIconSize(QSize(24, 24))
        self.btn_norm.clicked.connect(self.refresh_plot)
        self.btn_norm.clicked.connect(self.rescale)

        self.norm_x_min = QLineEdit(self)
        self.norm_x_min.setFixedWidth(40)
        self.norm_x_min.setPlaceholderText("Xmin")
        self.norm_x_min.setToolTip("Type Xmin for normalization")

        self.norm_x_max = QLineEdit(self)
        self.norm_x_max.setFixedWidth(40)
        self.norm_x_max.setPlaceholderText("Xmax")
        self.norm_x_max.setToolTip("Type Xmax for normalization")

    def create_copy_and_legend_buttons(self):
        self.btn_legend = QToolButton(self)
        self.btn_legend.setCheckable(True)
        self.btn_legend.setAutoExclusive(False)
        self.btn_legend.setToolTip("Show legend")
        self.btn_legend.setIcon(QIcon(os.path.join(ICON_DIR, "legend.png")))
        self.btn_legend.setIconSize(QSize(24, 24))
        self.btn_legend.clicked.connect(self.refresh_plot)

        self.btn_copy = QPushButton("", self)
        text = "Copy figure to clipboard.\nHold Ctrl & click to copy RAW & fitted curves to clipboard"
        self.btn_copy.setToolTip(text)
        self.btn_copy.setIcon(QIcon(os.path.join(ICON_DIR, "copy.png")))
        self.btn_copy.setIconSize(QSize(24, 24))
        self.btn_copy.clicked.connect(self.copy_fnc)

        self.R2 = QLabel("R2=0", self)

        self.tool_btn_options = QToolButton(self)
        self.tool_btn_options.setText("More options")
        self.tool_btn_options.setPopupMode(QToolButton.InstantPopup)
        self.tool_btn_options.setIcon(QIcon(os.path.join(ICON_DIR, "options.png")))
        self.tool_btn_options.setToolTip("More view options")
        self.tool_btn_options.setIconSize(QSize(24, 24))
        self.tool_btn_options.setMenu(self.options_menu)

    def create_control_layout(self):
        self.control_widget = QWidget(self)
        self.control_layout = QHBoxLayout(self.control_widget)
        self.control_layout.setContentsMargins(0, 0, 0, 0)

        self.control_layout.addWidget(self.btn_rescale)
        self.control_layout.addSpacing(10)
        self.control_layout.addWidget(self.btn_zoom)
        self.control_layout.addWidget(self.btn_baseline)
        self.control_layout.addWidget(self.btn_peak)
        self.control_layout.addSpacing(20)

        self.control_layout.addWidget(self.btn_norm)
        self.control_layout.addWidget(self.norm_x_min)
        self.control_layout.addWidget(self.norm_x_max)
        self.control_layout.addSpacing(20)

        self.control_layout.addWidget(self.btn_legend)
        
        
        self.control_layout.addWidget(self.btn_copy)
        self.control_layout.addSpacing(20)
        self.control_layout.addWidget(self.tool_btn_options)
        self.control_layout.addWidget(self.toolbar)
        self.control_layout.addWidget(self.R2)

        self.control_widget.setLayout(self.control_layout)


    def toggle_zoom_pan(self, checked):
        """Toggle zoom and pan functionality for spectra plot based on tool button selection."""
        if self.btn_zoom.isChecked():
            self.zoom_pan_active = True
            self.toolbar.zoom()  # Activate the zoom feature
        else:
            self.zoom_pan_active = False
            self.toolbar.zoom()  # Deactivate the zoom feature
            

    def create_options_menu(self):
        """Create widget containing all view options."""
        self.options_menu = QMenu(self)
        
        # X axis unit combobox
        xaxis_unit_widget = QWidget(self.options_menu)
        xaxis_unit_layout = QHBoxLayout(xaxis_unit_widget)
        xaxis_unit_label = QLabel("X-axis unit:", xaxis_unit_widget)
        xaxis_unit_layout.addWidget(xaxis_unit_label)

        self.cbb_xaxis_unit = QComboBox(xaxis_unit_widget)
        self.cbb_xaxis_unit.addItems(X_AXIS_UNIT)
        self.cbb_xaxis_unit.currentIndexChanged.connect(self.refresh_plot)
        xaxis_unit_layout.addWidget(self.cbb_xaxis_unit)
        xaxis_unit_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create a QWidgetAction to hold the combined QLabel and QComboBox
        combo_action = QWidgetAction(self)
        combo_action.setDefaultWidget(xaxis_unit_widget)
        self.options_menu.addAction(combo_action)
        
        # Y axis scale
        yaxis_scale_widget = QWidget(self.options_menu)
        yaxis_scale_layout = QHBoxLayout(yaxis_scale_widget)
        yaxis_scale_label = QLabel("Y-axis scale:", yaxis_scale_widget)
        yaxis_scale_layout.addWidget(yaxis_scale_label)

        self.cbb_yaxis_scale = QComboBox(yaxis_scale_widget)
        self.cbb_yaxis_scale.addItems(['Linear scale', 'Log scale'])
        self.cbb_yaxis_scale.currentIndexChanged.connect(self.refresh_plot)
        yaxis_scale_layout.addWidget(self.cbb_yaxis_scale)
        yaxis_scale_layout.setContentsMargins(5, 5, 5, 5)
        
         # Create a QWidgetAction to hold the combined QLabel and QComboBox
        combo_action2 = QWidgetAction(self)
        combo_action2.setDefaultWidget(yaxis_scale_widget)
        self.options_menu.addAction(combo_action2)
        

        # Add a separator to distinguish the combobox from checkable actions
        self.options_menu.addSeparator()

        # Define view options with checkable actions
        options = [
            ("Colors", "Colors", True),
            ("Peaks", "Show Peaks"),
            ("Bestfit", "Best Fit", True),
            ("Raw", "Raw data"),
            ("Residual", "Residual"),
            ("Grid", "Grid", False),
        ]

        # Add actions to the menu
        for option_name, option_label, *checked in options:
            action = QAction(option_label, self)
            action.setCheckable(True)
            action.setChecked(checked[0] if checked else False)
            action.triggered.connect(self.refresh_plot)
            self.menu_actions[option_name] = action
            self.options_menu.addAction(action)

        # Entry boxes for figure ratio
        ratio_widget = QWidget(self.options_menu)
        ratio_layout = QHBoxLayout(ratio_widget)

        fig_size_label = QLabel("Copied figure size:", ratio_widget)
        self.width_entry = QLineEdit(ratio_widget)
        self.width_entry.setFixedWidth(30)
        self.width_entry.setText("5.5")

        self.height_entry = QLineEdit(ratio_widget)
        self.height_entry.setFixedWidth(30)
        self.height_entry.setText("4")

        ratio_layout.addWidget(fig_size_label)
        ratio_layout.addWidget(self.width_entry)
        ratio_layout.addWidget(self.height_entry)
        ratio_layout.setContentsMargins(5, 5, 5, 5)

        # Create a QWidgetAction to hold the ratio input fields
        ratio_action = QWidgetAction(self)
        ratio_action.setDefaultWidget(ratio_widget)
        self.options_menu.addAction(ratio_action)
        
        # --- Line width adjustment ---
        lw_widget = QWidget(self.options_menu)
        lw_layout = QHBoxLayout(lw_widget)
        lw_label = QLabel("Line width:", lw_widget)
        lw_layout.addWidget(lw_label)
        
        self.spin_lw = QDoubleSpinBox(lw_widget)
        self.spin_lw.setRange(0.1, 5.0) 
        self.spin_lw.setSingleStep(0.5)
        self.spin_lw.setValue(1.5)  # default
        self.spin_lw.valueChanged.connect(self.refresh_plot)  
        lw_layout.addWidget(self.spin_lw)
        lw_layout.setContentsMargins(5, 5, 5, 5)

        lw_action = QWidgetAction(self)
        lw_action.setDefaultWidget(lw_widget)
        self.options_menu.addAction(lw_action)


    def update_plot_styles(self):
        """Apply styles and settings to the plot."""
        xlable = self.cbb_xaxis_unit.currentText()
        self.ax.set_xlabel(xlable)
        self.ax.set_ylabel("Intensity (a.u)")
        #self.ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
        
        # Apply grid state based on the checkbox in the options menu
       
        if "Grid" in self.menu_actions:
            if self.menu_actions["Grid"].isChecked():
                self.ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
            else:
                self.ax.grid(False)
    
    def rescale(self):
        """Rescale the spectra plot to fit within the axes."""
        self.ax.autoscale()
        self.canvas.draw_idle()

    
    def set_peak_model(self, model):
        """Set the peak model to be used when clicking on the plot."""
        self.peak_model = model

    def refresh_gui(self):
        """Call the refresh_gui method of the main application."""
        if hasattr(self.main_app, 'refresh_gui'):
            self.main_app.refresh_gui()
        else:
            return


    def plot(self, sel_spectrums):
        """Plot spectra or fit results in the figure canvas."""
        if not sel_spectrums:
            self.clear_plot()
            return
        self.sel_spectrums = sel_spectrums

        self.prepare_plot()

        for spectrum in self.sel_spectrums:
            self.plot_spectrum(spectrum)

        self.finalize_plot()

    def prepare_plot(self):
        """Prepare the plot area before plotting spectra."""
        # Save current xlim and ylim to maintain zoom/pan state
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.clear() 

        # Restore xlim and ylim if they were changed
        if not xlim == ylim == (0.0, 1.0):
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

    def plot_spectrum(self, spectrum):
        """Plot a single spectrum on the canvas."""
        x_values = spectrum.x
        y_values = self.get_y_values(spectrum)

        lw = self.spin_lw.value()
        line, = self.ax.plot(x_values, y_values, label=spectrum.label or spectrum.fname, ms=3, lw=lw, color=spectrum.color if spectrum.color else None)
        
        # Attach spectrum reference to line for later use in legend editing
        line.spectrum_fname = spectrum.fname
        line._spectrum_ref = spectrum

        plot_baseline_dynamically(ax=self.ax, spectrum=spectrum)

        if self.menu_actions['Raw'].isChecked():
            self.plot_raw_data(spectrum)

        if self.menu_actions['Bestfit'].isChecked():
            self.plot_peaks_and_bestfit(spectrum)

        if self.menu_actions['Residual'].isChecked() and hasattr(spectrum.result_fit, 'residual'):
            try:
                self.plot_residual(spectrum)
            except:
                print("plot residual is not succesful")
            
        
        if hasattr(spectrum.result_fit, 'rsquared'):
            self.show_R2(spectrum)
        else:
            self.show_R2(None)

        # Reset color cycle if Colors option is not checked
        if not self.menu_actions['Colors'].isChecked():
            self.ax.set_prop_cycle(None)

    def on_legend_pick(self, event):
        """Handle clicks on legend items (text or marker)."""
        artist = event.artist

        # --- If user clicked a legend text (label) ---
        if isinstance(artist, plt.Text):
            current_label = artist.get_text()
            new_label, ok = QInputDialog.getText(
                self, "Edit Legend Label", "Enter new label:", text=current_label
            )
            if ok and new_label.strip():
                artist.set_text(new_label)

                # Find corresponding plotted line and update its label + spectrum.label
                for line in self.ax.get_lines():
                    if line.get_label() == current_label:
                        line.set_label(new_label)
                        if hasattr(line, "_spectrum_ref"):
                            line._spectrum_ref.label = new_label
                        break

                self.canvas.draw_idle()

        # --- If user clicked a legend line/marker (color) ---
        elif isinstance(artist, plt.Line2D):
            new_color = QColorDialog.getColor()
            if new_color.isValid():
                color_hex = new_color.name()
                artist.set_color(color_hex)

                # Update corresponding plotted line(s) + spectrum.color
                for line in self.ax.get_lines():
                    if line.get_label() == artist.get_label():
                        line.set_color(color_hex)
                        if hasattr(line, "_spectrum_ref"):
                            line._spectrum_ref.color = color_hex
                        break

                self.canvas.draw_idle()
                
    def get_y_values(self, spectrum):
        """Get y-values for a spectrum, applying normalization if needed."""
        x_values = spectrum.x
        y_values = spectrum.y

        if self.btn_norm.isChecked():
            norm_x_min = self.norm_x_min.text().strip()
            norm_x_max = self.norm_x_max.text().strip() 

            if norm_x_min and norm_x_max:  # If user provided both X min and X max values
                try:
                    norm_x_min = float(norm_x_min)
                    norm_x_max = float(norm_x_max)
                    # Ensure min is less than max
                    if norm_x_min > norm_x_max:
                        norm_x_min, norm_x_max = norm_x_max, norm_x_min 
                    
                    # Find the closest indices in x_values
                    min_index = (np.abs(x_values - norm_x_min)).argmin()
                    max_index = (np.abs(x_values - norm_x_max)).argmin()

                    # Get max Y value within the range
                    norm_y_value = max(y_values[min_index:max_index + 1])
                except ValueError:
                    print("Invalid X value. Normalizing to max intensity instead.")
                    norm_y_value = max(y_values) 
            else:
                norm_y_value = max(y_values)  

            if norm_y_value != 0:
                y_values = y_values / norm_y_value 
        return y_values


    def plot_raw_data(self, spectrum):
        """Plot raw data points if the option is checked."""
        x0_values = spectrum.x0
        y0_values = spectrum.y0
        lw = self.spin_lw.value()
        self.ax.plot(x0_values, y0_values, 'ko-', label='raw', ms=3, lw=lw)

    def plot_peak(self, y_peak, x_values, peak_label, peak_model):
        """Plot individual peak, optionally filled, and return line and peak info."""
        lw = self.spin_lw.value()
        line, = self.ax.plot(x_values, y_peak, '-', label=peak_label, lw=lw)
        
        # Annotate if enabled
        if self.menu_actions['Peaks'].isChecked():
            self.annotate_peak(peak_model, peak_label)

        # Extract peak info for hover and interaction
        peak_info = {
            "peak_label": peak_label,
            "peak_model": peak_model, # For peak dragging features.
        }

        # Extract parameter values (x0, fwhm, amplitude, etc.)
        if hasattr(peak_model, 'param_names') and hasattr(peak_model, 'param_hints'):
            for param_name in peak_model.param_names:
                key = param_name.split('_', 1)[1]  # e.g., x0, amplitude
                if key in peak_model.param_hints and 'value' in peak_model.param_hints[key]:
                    val = peak_model.param_hints[key]['value']
                    peak_info[key] = val
        return line, peak_info

    def plot_peaks_and_bestfit(self, spectrum):
        x_values = spectrum.x
        y_peaks = np.zeros_like(x_values)
        y_bkg = self.get_background_y_values(spectrum)

        peak_labels = spectrum.peak_labels
        self.fitted_lines = []  # Store line and info for hover

        for i, peak_model in enumerate(spectrum.peak_models):
            y_peak = self.evaluate_peak_model(peak_model, x_values)
            y_peaks += y_peak
            result = self.plot_peak(y_peak, x_values, peak_labels[i], peak_model)

            if result is not None:
                line, peak_info = result
                if line is not None:
                    self.fitted_lines.append((line, peak_info))
                else:
                    pass
            else:
                pass

        if hasattr(spectrum.result_fit, 'success'):
            y_fit = y_bkg + y_peaks
            lw = self.spin_lw.value()
            self.ax.plot(x_values, y_fit, label="bestfit", lw=lw)

        self.enable_hover_highlight()  # connect hover after drawing lines
    
    def enable_hover_highlight(self):
        if not hasattr(self, 'hover_connection'):
            self.hover_connection = self.canvas.mpl_connect('motion_notify_event', self.on_hover)
            
    def on_scroll(self, event):
        ax = event.inaxes
        if ax is None:
            return  # Ignore scrolls outside the plot area

        y_min, y_max = ax.get_ylim()
        dy = (y_max - y_min) * 0.1  # 10% zoom step

        if event.step < 0:
            # Scroll up: increase max Y
            y_max = y_max + dy
        elif event.step > 0:
            # Scroll down: decrease max Y
            y_max = max(y_min + 1e-6, y_max - dy)  # prevent collapse

        ax.set_ylim(y_min, y_max)
        self.refresh_plot()   
        
    def on_hover(self, event):
        if event.inaxes != self.ax or not self.canvas.isActiveWindow():
            self.hide_tooltip()
            return
        if not hasattr(self, 'fitted_lines') or not self.fitted_lines:
            self.hide_tooltip()
            return
        
        for line, info in self.fitted_lines:
            if line.contains(event)[0]:
                # Define the keys we want to show, in order
                fields = [
                    ('label', info.get('peak_label')),
                    ('center', info.get('x0')),
                    ('intensity', info.get('ampli')),
                    ('fwhm', info.get('fwhm')),
                    ('fwhm_l', info.get('fwhm_l')),
                    ('fwhm_r', info.get('fwhm_r')),
                    ('alpha', info.get('alpha')),
                ]

                # Build the tooltip string dynamically
                lines = []
                for label, val in fields:
                    if val is not None:
                        try:
                            val_str = f"{val:.3f}" if isinstance(val, (float, int)) else str(val)
                        except Exception:
                            val_str = str(val)
                        lines.append(f"{label}: {val_str}")

                text = "\n".join(lines)
                self.show_tooltip(event, text)
                self._highlight_line(line)

                # Connect mouse press for dragging
                self.canvas.mpl_disconnect(getattr(self, 'click_connection', None))
                self.click_connection = self.canvas.mpl_connect('button_press_event', self.on_mouse_click)
                return

        self.hide_tooltip()
        self._reset_highlight()
    
    def on_mouse_click(self, event):
        """interaction with peak model and background via left-right mouse click"""
        if event.inaxes != self.ax or not self.sel_spectrums:
            return

        # Skip clicks inside cached legend bbox
        if getattr(self, 'legend_bbox', None) is not None:
            if self.legend_bbox.contains(event.x, event.y):
                return
    
        if self.zoom_pan_active:
            return

        x_click = event.xdata
        y_click = event.ydata
        sel_spectrum = self.sel_spectrums[0]

        if self.btn_peak.isChecked():
            if event.button == 1:
                # Try to drag peak if hovered over a line
                for line, info in self.fitted_lines:
                    if line.contains(event)[0]:
                        self.dragging_peak = (line, info)
                        self.drag_event_connection = self.canvas.mpl_connect('motion_notify_event', self.on_drag_peak)
                        self.release_event_connection = self.canvas.mpl_connect('button_release_event', self.on_release_drag)
                        return  # do not add a new peak if we start dragging

                # Else, normal left-click to add peak
                maxshift= self.settings.value("fit_settings/maxshift", 20, type=float)
                maxfwhm= self.settings.value("fit_settings/maxfwhm", 200, type=float)
                            
                sel_spectrum.add_peak_model(self.peak_model, x_click, dx0=(maxshift,maxshift), dfwhm=maxfwhm)
                self.refresh_gui()

            elif event.button == 3:
                # Right-click: remove closest peak
                if hasattr(sel_spectrum, "peak_models") and sel_spectrum.peak_models:
                    closest_idx = min(
                        range(len(sel_spectrum.peak_models)),
                        key=lambda i: abs(sel_spectrum.peak_models[i].param_hints['x0']['value'] - x_click)
                    )
                    del sel_spectrum.peak_models[closest_idx]
                    del sel_spectrum.peak_labels[closest_idx]
                    self.refresh_gui()

        elif self.btn_baseline.isChecked():
            if event.button == 1:
                if sel_spectrum.baseline.is_subtracted:
                    show_alert("Baseline is already subtracted. Reinitialize spectrum to perform new baseline")
                else:
                    sel_spectrum.baseline.add_point(x_click, y_click)
                self.refresh_gui()

            elif event.button == 3:
                if (hasattr(sel_spectrum.baseline, "points") and
                    isinstance(sel_spectrum.baseline.points, list) and
                    len(sel_spectrum.baseline.points[0]) > 0):
                    x_points = sel_spectrum.baseline.points[0]
                    y_points = sel_spectrum.baseline.points[1]
                    closest_idx = min(range(len(x_points)), key=lambda i: abs(x_points[i] - x_click))
                    x_points.pop(closest_idx)
                    y_points.pop(closest_idx)
                    self.refresh_gui()                
    
    def show_tooltip(self, event, text):
        """Show tooltip near the cursor with peak info."""
        if not hasattr(self, 'tooltip'):
            from PySide6.QtWidgets import QLabel
            self.tooltip = QLabel(self.canvas)

            self.tooltip.setStyleSheet("""
                background-color: rgba(255, 255, 255, 0.5);
                color: black;
                border: 0.1px gray;
                padding: 2px;
            """)
            self.tooltip.setWindowFlags(Qt.ToolTip)

        self.tooltip.setText(text)

        cursor_pos = QCursor.pos()
        offset = QPoint(5, -75)
        self.tooltip.move(cursor_pos + offset)
        self.tooltip.show()

    def on_focus_changed(self, old, new):
        if not self.canvas.isActiveWindow():
            self.hide_tooltip()
        
    def hide_tooltip(self):
        if hasattr(self, 'tooltip'):
            self.tooltip.hide()

    def _highlight_line(self, line_to_highlight):
        """Highlight the peak upon hover mouse cursor"""
        # If already highlighted this line, do nothing
        if getattr(self, 'highlighted_line', None) == line_to_highlight:
            return
        self._reset_highlight()

        # Save current linewidth to restore later
        line_to_highlight._orig_lw = line_to_highlight.get_linewidth()

        # Increase linewidth
        line_to_highlight.set_linewidth(3)
        self.highlighted_line = line_to_highlight

        # Redraw canvas to reflect changes
        self.canvas.draw_idle()

    def _reset_highlight(self):
        """Un-Highlight the peak upon hover mouse cursor"""
        if hasattr(self, 'highlighted_line') and self.highlighted_line is not None:
            # Restore original linewidth
            orig_lw = getattr(self.highlighted_line, '_orig_lw', 1.5)
            self.highlighted_line.set_linewidth(orig_lw)

            self.highlighted_line = None
            self.canvas.draw_idle()

    def on_drag_peak(self, event):
        """Dragging peak to adjust x0 of peak_model in real-time"""
        if self.dragging_peak is None or event.xdata is None:
            return

        line, info = self.dragging_peak
        peak_model = info.get('peak_model')
        if not peak_model:
            return

        # Update x0, ampli in the model
        peak_model.param_hints['x0']['value'] = event.xdata
        peak_model.param_hints['ampli']['value'] = event.ydata

        # Re-plot this spectrum
        self.plot(self.sel_spectrums)
        

    def on_release_drag(self, event):
        self.dragging_peak = None
        if hasattr(self, 'drag_event_connection'):
            self.canvas.mpl_disconnect(self.drag_event_connection)
            self.drag_event_connection = None
        if hasattr(self, 'release_event_connection'):
            self.canvas.mpl_disconnect(self.release_event_connection)
            self.release_event_connection = None
        self.refresh_gui() 


    def get_background_y_values(self, spectrum):
        """Get y-values for the background model."""
        x_values = spectrum.x
        if spectrum.bkg_model is not None:
            return spectrum.bkg_model.eval(spectrum.bkg_model.make_params(), x=x_values)
        return np.zeros_like(x_values)

    def evaluate_peak_model(self, peak_model, x_values):
        """Evaluate the peak model to get y-values."""
        param_hints_orig = deepcopy(peak_model.param_hints)
        for key in peak_model.param_hints.keys():
            peak_model.param_hints[key]['expr'] = ''
        
        params = peak_model.make_params()
        peak_model.param_hints = param_hints_orig
        
        return peak_model.eval(params, x=x_values) 

    def annotate_peak(self, peak_model, peak_label):
        """Annotate peaks on the plot with labels."""
        position = peak_model.param_hints['x0']['value']
        intensity = peak_model.param_hints['ampli']['value']
        position = round(position, 2)
        text = f"{peak_label}\n({position})"
        self.ax.text(position, intensity, text, ha='center', va='bottom', color='black', fontsize=12)

    def compute_residual(self, spectrum):
        """Compute residual = raw data - (background + sum of peaks)."""
        x_values = spectrum.x
        y_values = spectrum.y
        y_bkg = self.get_background_y_values(spectrum)

        # Sum all peak models
        y_peaks = np.zeros_like(x_values)
        for peak_model in spectrum.peak_models:
            y_peak = self.evaluate_peak_model(peak_model, x_values)
            y_peaks += y_peak

        y_fit = y_bkg + y_peaks
        residual = y_values - y_fit
        return x_values, residual

    def plot_residual(self, spectrum):
        """Plot the residuals if available."""
        x_values, residual = self.compute_residual(spectrum)
        # x_values = spectrum.x
        # residual = spectrum.result_fit.residual  # Bug of fitspy 2025.6 version
        self.ax.plot(x_values, residual, 'ro-', ms=3, label='residual')

    def show_R2(self, spectrum):
        """Display R² value in the GUI."""
        if spectrum is not None and hasattr(spectrum.result_fit, 'rsquared'):
            rsquared = round(spectrum.result_fit.rsquared, 4)
            self.R2.setText(f"R²={rsquared}")
        else:
            self.R2.setText("R²=0")

    def finalize_plot(self):
        """Finalize plot settings and draw the canvas."""
        # Use the selected x-axis label from the combobox
        xlabel = self.cbb_xaxis_unit.currentText() if self.cbb_xaxis_unit else "Wavenumber (cm-1)"
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel("Intensity (a.u)")
        
        y_scale = self.cbb_yaxis_scale.currentText()
        self.ax.set_yscale('log' if y_scale == 'Log scale' else 'linear')

        # Handle legend
        if self.btn_legend.isChecked():
            legend = self.ax.legend(loc='upper right')
            
            # Make legend items pickable
            for text in legend.get_texts():
                text.set_picker(True)
            for handle in legend.legendHandles:
                handle.set_picker(True)

            # Cache legend bbox to skip mouse clicks inside it
            self.legend_bbox = legend.get_window_extent(self.canvas.renderer)

            # Connect pick-event once
            if not hasattr(self, "_legend_pick_connected"):
                self.canvas.mpl_connect("pick_event", self.on_legend_pick)
                self._legend_pick_connected = True
        else:
            self.legend_bbox = None

        # Make sure grid follows the checkbox after full redraw
        if self.menu_actions["Grid"].isChecked():
            self.ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
        else:
            self.ax.grid(False)

        self.figure.tight_layout()
        self.canvas.draw_idle()
        
    def clear_plot(self):
        """Explicitly clear the spectra plot."""
        if self.ax:
            self.ax.clear()
            self.ax.set_xlabel("X-axis")
            self.ax.set_ylabel("Y-axis")
            self.ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
            self.canvas.draw_idle()  
        
        # Clear hover-related data to prevent warnings
        self.fitted_lines = []
        self.highlighted_line = None
        self.hide_tooltip()
            
    def refresh_plot(self):
        """Refresh the plot based on user view options."""
        if not self.sel_spectrums:
            self.clear_plot() 
        else:
            self.plot(self.sel_spectrums)

    def copy_fig(self):
        """Copy figure canvas to clipboard"""
        width_text = self.width_entry.text().strip()
        height_text = self.height_entry.text().strip()

        # Set default values if the entry boxes are empty
        width = float(width_text) if width_text else 5.5  # Default width
        height = float(height_text) if height_text else 4.0
        copy_fig_to_clb(self.canvas, size_ratio=(width, height))
        
    def copy_spectra_data(self):
        """Copy X, Y, and peak model data of the first selected spectrum to clipboard as a DataFrame."""
        import pandas as pd

        if not self.sel_spectrums or len(self.sel_spectrums) == 0:
            print("No spectrum selected.")
            return

        spectrum = self.sel_spectrums[0]
        x_values = spectrum.x
        y_values = spectrum.y

        # Create a dictionary for the DataFrame
        data = {
            "X values": x_values,
            "Y values": y_values
        }

        # Add each peak model’s evaluated Y values as a new column
        for i, peak_model in enumerate(spectrum.peak_models):
            y_peak = self.evaluate_peak_model(peak_model, x_values)

            if hasattr(spectrum, 'peak_labels') and i < len(spectrum.peak_labels):
                label = spectrum.peak_labels[i]
            else:
                label = f"Peak {i + 1}"

            data[label] = y_peak

        df = pd.DataFrame(data)
        df.to_clipboard(index=False)
        print("Spectrum data copied to clipboard.")
        
    def copy_fnc(self):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.copy_spectra_data()
        else:
            self.copy_fig()