from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QFrame, QRadioButton, QLabel,
                             QPushButton, QSpinBox, QScrollArea, QToolButton, QGroupBox, QGridLayout,
                             QCheckBox, QSizePolicy, QSpacerItem)
from PyQt6.QtCore import QSize, QRect
from PyQt6.QtGui import QIcon
from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt6agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

class SpectraViewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Main layout for the widget
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setSpacing(0)
        
        # Upper Zone Layout
        self.upper_zone_layout = QVBoxLayout()
        self.upper_zone_layout.setContentsMargins(0, -1, 10, -1)

        # Spectra View Frame
        self.spectra_view_frame = QFrame(self)
        self.spectra_view_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.spectra_view_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.spectra_view_frame_layout = QVBoxLayout(self.spectra_view_frame)
        self.spectra_view_frame_layout.setContentsMargins(0, 0, 0, 0)

        # Inner Layout for Spectra View Frame
        self.inner_layout = QVBoxLayout()
        self.inner_layout.setSpacing(6)
        self.spectra_view_frame_layout.addLayout(self.inner_layout)

        # Add Spectra View Frame to Upper Zone Layout
        self.upper_zone_layout.addWidget(self.spectra_view_frame)

        # Bottom Frame Layout
        self.bottom_frame_layout = QHBoxLayout()
        self.bottom_frame_layout.setSpacing(10)
        self.bottom_frame_layout.setSizeConstraint(QLayout.SizeConstraint.SetMaximumSize)
        self.bottom_frame_layout.setContentsMargins(2, 2, 2, 2)

        # Toolbar Frame
        self.toolbar_frame = QHBoxLayout()
        self.bottom_frame_layout.addLayout(self.toolbar_frame)

        # Toolbar
        self.figure = plt.figure()  # Create a matplotlib figure
        self.canvas = FigureCanvas(self.figure)  # Create the canvas to display the figure
        self.toolbar = NavigationToolbar(self.canvas, self)  # Create the navigation toolbar
        self.toolbar_frame.addWidget(self.toolbar)  # Add toolbar to the layout
        self.inner_layout.addWidget(self.canvas)  # Add canvas to the layout

        # Radio Buttons
        self.rdbtn_baseline = QRadioButton("Baseline", self)
        self.rdbtn_baseline.setChecked(True)
        self.bottom_frame_layout.addWidget(self.rdbtn_baseline)

        self.rdbtn_peak = QRadioButton("Peak", self)
        self.rdbtn_peak.setChecked(False)
        self.bottom_frame_layout.addWidget(self.rdbtn_peak)

        # Additional Widgets
        self.rsquared_label = QLabel("R²", self)
        self.bottom_frame_layout.addWidget(self.rsquared_label)

        self.copy_button = QPushButton(self)
        self.copy_button.setIcon(QIcon(":/icon/iconpack/copy.png"))
        self.copy_button.setIconSize(QSize(24, 24))
        self.bottom_frame_layout.addWidget(self.copy_button)

        self.label_dpi = QLabel("DPI", self)
        self.bottom_frame_layout.addWidget(self.label_dpi)

        self.dpi_spinbox = QSpinBox(self)
        self.dpi_spinbox.setMinimum(100)
        self.dpi_spinbox.setMaximum(200)
        self.dpi_spinbox.setSingleStep(10)
        self.bottom_frame_layout.addWidget(self.dpi_spinbox)

        # Spacers
        self.bottom_frame_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        self.bottom_frame_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        self.bottom_frame_layout.setStretch(0, 50)
        self.bottom_frame_layout.setStretch(1, 25)

        # Add Bottom Frame to Upper Zone Layout
        self.upper_zone_layout.addLayout(self.bottom_frame_layout)

        # Widget for Scroll Area
        self.scroll_widget = QWidget(self)
        self.scroll_widget.setMinimumSize(QSize(300, 0))
        self.scroll_widget_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_widget_layout.setContentsMargins(2, 0, 2, 0)

        # Scroll Area
        self.scroll_area = QScrollArea(self.scroll_widget)
        self.scroll_area.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum))
        self.scroll_area.setMinimumSize(QSize(250, 450))
        self.scroll_area.setMaximumSize(QSize(350, 16777215))
        self.scroll_area.setWidgetResizable(True)
        
        self.scroll_area_contents = QWidget()
        self.scroll_area_layout = QVBoxLayout(self.scroll_area_contents)
        self.scroll_area_layout.setContentsMargins(0, 0, 0, 0)

        # Tool Button
        self.tool_button = QToolButton(self.scroll_area_contents)
        self.scroll_area_layout.addWidget(self.tool_button)

        # Spacer
        self.scroll_area_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # View Options Box
        self.view_options_box = QGroupBox("View Options", self.scroll_area_contents)
        self.view_options_box.setMaximumSize(QSize(320, 16777215))
        self.view_options_grid = QGridLayout(self.view_options_box)

        # Checkboxes
        self.cb_residual = QCheckBox("Residual", self.view_options_box)
        self.cb_residual.setChecked(False)
        self.view_options_grid.addWidget(self.cb_residual, 1, 1)

        self.cb_filled = QCheckBox("Filled", self.view_options_box)
        self.cb_filled.setChecked(True)
        self.view_options_grid.addWidget(self.cb_filled, 0, 2)

        self.cb_bestfit = QCheckBox("Best Fit", self.view_options_box)
        self.cb_bestfit.setChecked(True)
        self.view_options_grid.addWidget(self.cb_bestfit, 0, 1)

        self.cb_legend = QCheckBox("Legend", self.view_options_box)
        self.cb_legend.setChecked(False)
        self.view_options_grid.addWidget(self.cb_legend, 0, 0)

        self.cb_raw = QCheckBox("Raw", self.view_options_box)
        self.cb_raw.setChecked(False)
        self.view_options_grid.addWidget(self.cb_raw, 1, 0)

        self.cb_colors = QCheckBox("Colors", self.view_options_box)
        self.cb_colors.setChecked(True)
        self.view_options_grid.addWidget(self.cb_colors, 1, 2)

        self.cb_peaks = QCheckBox("Peaks", self.view_options_box)
        self.cb_peaks.setChecked(False)
        self.view_options_grid.addWidget(self.cb_peaks, 0, 3)

        self.cb_normalize = QCheckBox("Normalize", self.view_options_box)
        self.view_options_grid.addWidget(self.cb_normalize, 1, 3)

        self.scroll_area_layout.addWidget(self.view_options_box)
        self.scroll_area.setWidget(self.scroll_area_contents)
        self.scroll_widget_layout.addWidget(self.scroll_area)

        # Add Scroll Widget to Upper Zone Layout
        self.upper_zone_layout.addWidget(self.scroll_widget)

        # Add Upper Zone Layout to Main Layout
        self.main_layout.addLayout(self.upper_zone_layout)
        self.main_layout.setStretch(0, 75)

    def plot(self, x, y):
        """Plot data on the figure canvas."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x, y)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('Spectra Plot')
        self.canvas.draw()

    def get_dpi(self):
        """Return the current DPI setting from the spinbox."""
        return self.dpi_spinbox.value()

    def set_dpi(self, dpi):
        """Set the DPI value in the spinbox."""
        self.dpi_spinbox.setValue(dpi)
