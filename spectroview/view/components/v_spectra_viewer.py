# view/components/spectra_viewer.py
import warnings
from copy import deepcopy
import numpy as np
from PySide6.QtCore import QSettings
from spectroview.fit_engine.noise import detect_noise_level
from spectroview.fit_engine.evaluator import eval_peak_initial
# Suppress harmless Matplotlib constrained_layout warning on 0-size UI initialization
warnings.filterwarnings("ignore", message=".*constrained_layout not applied because axes sizes collapsed to zero.*")

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QToolButton, QLabel,
    QComboBox, QMenu, QWidgetAction,
    QLineEdit, QDoubleSpinBox, QColorDialog, QInputDialog,
    QSlider, QGroupBox
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.collections as mcoll
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas
)
import matplotlib as mpl
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QCursor
from PySide6.QtCore import QPoint

import matplotlib.lines as mlines

from spectroview import ICON_DIR, X_AXIS_UNIT, Y_AXIS_UNIT, PLOT_POLICY_LIGHT, PLOT_POLICY_DARK, DEFAULT_COLORS
from spectroview.viewmodel.utils import copy_fig_to_clb
from spectroview.view.components.customized_widgets import NoDoubleClickZoomToolbar


class _MockPeakModelObj:
    """Lightweight stand-in for tensor-mode plotting to carry a shape name."""
    __slots__ = ('name2',)
    def __init__(self, name):
        self.name2 = name


class VSpectraViewer(QWidget):
    # ───── View → ViewModel signals ─────
    mouseClicked = Signal(float, float, int)
    zoomToggled = Signal(bool)
    rescaleRequested = Signal()
    viewOptionsChanged = Signal(dict)
    copy_data_requested = Signal()  # Request ViewModel to copy spectrum data
    toolModeChanged = Signal(str)  # zoom / baseline / peak
    normalizationChanged = Signal(bool, float, float)
    plotStyleChanged = Signal()
    allOptionsSyncChanged = Signal(dict)

    peak_add_requested = Signal(float)
    peak_remove_requested = Signal(float)
    baseline_add_requested = Signal(float, float)
    baseline_remove_requested = Signal(float)
    peak_drag_started = Signal(object)   # optional (advanced)
    peak_dragged = Signal(float, float)
    peak_drag_finished = Signal()
    spectrumCustomized = Signal()

    _MAX_HEAVY_OVERLAYS = 10  # Maximum number of best-fit spectra to display in Tensor Mode (all spectra are still plotted)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._current_spectra = []
        self._tensor_data = None
        
        self._fitted_lines = []     # [(line, peak_info)]
        self._highlighted_line = None
        self._dragging_peak = None  # Stores (line, info) when dragging

        self.zoom_pan_active = True
        QApplication.instance().focusChanged.connect(self._hide_tooltip)   
        
    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ─── Figure ───
        with plt.style.context(PLOT_POLICY_LIGHT):
            self.figure = Figure(layout="compressed", dpi=80)
            self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        
        self.canvas.mpl_connect("button_press_event", self._on_mouse_click)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)


        self.toolbar = NoDoubleClickZoomToolbar(self.canvas, self)
        self.toolbar.zoom() # Start with zoom enabled
        for action in self.toolbar.actions():
            if action.text() in ['Home', 'Save', 'Pan', 'Back', 'Forward', 'Subplots', 'Zoom']:
                action.setVisible(False)

        # ─── Shift sliders panel (right of canvas) ───
        self.shift_panel = self._create_shift_panel()

        # ─── Canvas + Sliders row ───
        canvas_row = QHBoxLayout()
        canvas_row.setContentsMargins(0, 0, 0, 0)
        canvas_row.addWidget(self.canvas, stretch=1)
        canvas_row.addWidget(self.shift_panel)

        # ─── Control bar ───
        self.control_bar = self._create_control_bar()
        
        main_layout.addLayout(canvas_row)
        main_layout.addWidget(self.control_bar)

    # ─── Shift sliders ───
    def _create_shift_panel(self):
        """Create a vertical panel with Y-shift and X-shift sliders."""
        panel = QGroupBox()
        panel.setFixedWidth(25)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # ── Y Shift slider ──
        lbl_y = QLabel("Y")
        lbl_y.setAlignment(Qt.AlignCenter)
        lbl_y.setToolTip("Shift spectra vertically (waterfall offset)")

        self.slider_y_shift = QSlider(Qt.Vertical)
        self.slider_y_shift.setRange(0, 100)
        self.slider_y_shift.setValue(0)
        self.slider_y_shift.setToolTip("Y Shift: offset each spectrum vertically")
        self.slider_y_shift.valueChanged.connect(self._plot)

        layout.addWidget(lbl_y)
        layout.addWidget(self.slider_y_shift, stretch=1)

        # ── X Shift slider ──
        lbl_x = QLabel("X")
        lbl_x.setAlignment(Qt.AlignCenter)
        lbl_x.setToolTip("Shift spectra horizontally")

        self.slider_x_shift = QSlider(Qt.Vertical)
        self.slider_x_shift.setRange(0, 100)
        self.slider_x_shift.setValue(0)
        self.slider_x_shift.setToolTip("X Shift: offset each spectrum horizontally")
        self.slider_x_shift.valueChanged.connect(self._plot)

        layout.addWidget(lbl_x)
        layout.addWidget(self.slider_x_shift, stretch=1)

        return panel

    # Control bar
    def _create_control_bar(self):
        bar = QWidget(self)
        bar.setFixedHeight(50)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(5, 5, 5, 5)

        # Rescale
        self.btn_rescale = QPushButton()
        self.btn_rescale.setIcon(QIcon(f"{ICON_DIR}/rescale.png"))
        self.btn_rescale.setIconSize(QSize(22, 22))
        self.btn_rescale.setToolTip("Rescale (Ctrl+R)")
        self.btn_rescale.clicked.connect(self.rescaleRequested)
        self.btn_rescale.clicked.connect(self._rescale)
        
        layout.addWidget(self.btn_rescale)

        # Tool buttons
        self.btn_zoom = self._tool_btn("zoom.png", "Zoom", True)
        self.btn_zoom.setCheckable(True)
        self.btn_zoom.setChecked(True)
        self.btn_zoom.toggled.connect(self._toggle_zoom_pan)
        self.btn_baseline = self._tool_btn("baseline.png", "Baseline")
        self.btn_peak = self._tool_btn("peak.png", "Peak")

        for btn, mode in [
            (self.btn_zoom, "zoom"),
            (self.btn_baseline, "baseline"),
            (self.btn_peak, "peak"),
        ]:
            btn.toggled.connect(lambda c, m=mode: c and self.toolModeChanged.emit(m))
            layout.addWidget(btn)

        layout.addSpacing(15)

        # Normalization
        self.btn_norm = QToolButton()
        self.btn_norm.setCheckable(True)
        self.btn_norm.setIcon(QIcon(f"{ICON_DIR}/norm.png"))
        self.btn_norm.setToolTip("Intensity normalization to maximum of spectrum or selected region")
        self.btn_norm.setIconSize(QSize(22, 22))
        self.btn_norm.toggled.connect(self._emit_norm)
        self.btn_norm.toggled.connect(self._plot)
        self.btn_norm.clicked.connect(self._rescale)

        layout.addWidget(self.btn_norm)

        self.norm_xmin = QLineEdit()
        self.norm_xmin.setFixedWidth(45)
        self.norm_xmin.setPlaceholderText("Xmin")
        self.norm_xmin.setToolTip("Intensity normalization to maximum of spectrum or selected region")
        layout.addWidget(self.norm_xmin)

        self.norm_xmax = QLineEdit()
        self.norm_xmax.setFixedWidth(45)
        self.norm_xmax.setPlaceholderText("Xmax")
        self.norm_xmax.setToolTip("Intensity normalization to maximum of spectrum or selected region")
        layout.addWidget(self.norm_xmax)

        self.norm_xmin.editingFinished.connect(self._plot)
        self.norm_xmax.editingFinished.connect(self._plot)
        self.norm_xmin.editingFinished.connect(self._rescale)
        self.norm_xmax.editingFinished.connect(self._rescale)

        # Bestfit
        self.btn_bestfit = QToolButton()
        self.btn_bestfit.setCheckable(True)
        self.btn_bestfit.setChecked(True)  # Checked by default
        self.btn_bestfit.setIcon(QIcon(f"{ICON_DIR}/bestfit.png"))
        self.btn_bestfit.setToolTip("Show or hide best-fit lines")
        self.btn_bestfit.setIconSize(QSize(22, 22))
        self.btn_bestfit.toggled.connect(self._emit_view_options)
        layout.addWidget(self.btn_bestfit)


        # Legend
        self.btn_legend = QToolButton()
        self.btn_legend.setCheckable(True)
        self.btn_legend.setIcon(QIcon(f"{ICON_DIR}/legend.png"))
        self.btn_legend.setToolTip("Show or hide legend box")
        self.btn_legend.setIconSize(QSize(22, 22))
        self.btn_legend.toggled.connect(self._emit_view_options)
        layout.addWidget(self.btn_legend)

        # Copy
        self.btn_copy = QPushButton()
        self.btn_copy.setIcon(QIcon(f"{ICON_DIR}/copy.png"))
        self.btn_copy.setToolTip("Copy figure to clipboard. Hold Ctrl + Click to copy data to clipboard")
        self.btn_copy.setIconSize(QSize(22, 22))
        self.btn_copy.clicked.connect(self._emit_copy)
        layout.addWidget(self.btn_copy)

        # Options
        self.options_menu = self._create_options_menu()
        self.btn_options = QToolButton()
        self.btn_options.setIcon(QIcon(f"{ICON_DIR}/options.png"))
        self.btn_options.setToolTip("More view options")
        self.btn_options.setIconSize(QSize(22, 22))
        self.btn_options.setPopupMode(QToolButton.InstantPopup)
        self.btn_options.setMenu(self.options_menu)
        layout.addWidget(self.btn_options)

        # layout.addStretch()
        layout.addWidget(self.toolbar)
        # R²
        self.lbl_r2 = QLabel("R²=0")
        layout.addWidget(self.lbl_r2)
        # Noise level
        self.lbl_noise = QLabel("Noise=0")
        layout.addWidget(self.lbl_noise)

        return bar

    def _tool_btn(self, icon, tooltip, checked=False):
        btn = QToolButton()
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setAutoExclusive(True)
        btn.setIcon(QIcon(f"{ICON_DIR}/{icon}"))
        btn.setIconSize(QSize(22, 22))
        btn.setToolTip(tooltip)
        return btn

    # Options menu
    def _create_options_menu(self):
        menu = QMenu(self)

        # Plot Theme
        self.cbb_theme = QComboBox()
        self.cbb_theme.addItems(["Light Mode", "Dark Mode"])
        self.cbb_theme.currentIndexChanged.connect(self._apply_plot_style)
        menu.addAction(self._wrap("Plot Theme:", self.cbb_theme))

        # X-axis
        self.cbb_xaxis = QComboBox()
        self.cbb_xaxis.addItems(X_AXIS_UNIT)
        self.cbb_xaxis.currentIndexChanged.connect(self._emit_view_options)
        menu.addAction(self._wrap("X-axis:", self.cbb_xaxis))

        # Y-axis unit
        self.cbb_yaxis = QComboBox()
        self.cbb_yaxis.addItems(Y_AXIS_UNIT)
        self.cbb_yaxis.currentIndexChanged.connect(self._emit_view_options)
        menu.addAction(self._wrap("Y-axis:", self.cbb_yaxis))

        # Y-scale
        self.cbb_yscale = QComboBox()
        self.cbb_yscale.addItems(["Linear", "Log"])
        self.cbb_yscale.currentIndexChanged.connect(self._emit_view_options)
        menu.addAction(self._wrap("Y-scale:", self.cbb_yscale))   

        # Main spectrum plot style
        self.cbb_plotstyle = QComboBox()
        self.cbb_plotstyle.addItems(["line", "dot"])
        self.cbb_plotstyle.currentIndexChanged.connect(self._emit_view_options)
        menu.addAction(self._wrap("Spectrum plot style:", self.cbb_plotstyle))

        # Line width
        self.spin_lw = QDoubleSpinBox()
        self.spin_lw.setRange(0.1, 5)
        self.spin_lw.setSingleStep(0.5)
        self.spin_lw.setValue(1.5)
        self.spin_lw.valueChanged.connect(self._emit_view_options)
        menu.addAction(self._wrap("Line width:", self.spin_lw))
        # Dot size
        self.spin_dotsize = QDoubleSpinBox()
        self.spin_dotsize.setRange(0.5, 10)
        self.spin_dotsize.setSingleStep(0.5)
        self.spin_dotsize.setValue(3)
        self.spin_dotsize.valueChanged.connect(self._emit_view_options)
        menu.addAction(self._wrap("Dot size:", self.spin_dotsize))

        menu.addSeparator()

        # Toggles
        self._add_checkbox(menu, "Raw")
        # Bestfit is now controlled by btn_bestfit button in toolbar
        
        # Bestfit colorful
        self.act_bestfit_colorful = menu.addAction("Bestfit curve colorful")
        self.act_bestfit_colorful.setCheckable(True)
        self.act_bestfit_colorful.setChecked(True)
        self.act_bestfit_colorful.toggled.connect(self._emit_view_options)

        # Show peak label
        self.act_show_peak_label = menu.addAction("Show peak label")
        self.act_show_peak_label.setCheckable(True)
        self.act_show_peak_label.setChecked(False)
        self.act_show_peak_label.toggled.connect(self._emit_view_options)

        self._add_checkbox(menu, "Residual")
        self._add_checkbox(menu, "Grid")

        # Show noise level
        self.act_noise_level = menu.addAction("Show noise level")
        self.act_noise_level.setCheckable(True)
        self.act_noise_level.setChecked(False)
        self.act_noise_level.toggled.connect(self._emit_view_options)

        menu.addSeparator()

        
        
        # ─── Copied figure size (NEW) ───
        ratio_widget = QWidget()
        ratio_layout = QHBoxLayout(ratio_widget)
        ratio_layout.setContentsMargins(5, 5, 5, 5)

        ratio_layout.addWidget(QLabel("Copied figure size:"))

        self.width_entry = QLineEdit()
        self.width_entry.setFixedWidth(40)
        self.width_entry.setText("5.5")
        self.width_entry.editingFinished.connect(self._emit_view_options)
        ratio_layout.addWidget(self.width_entry)

        self.height_entry = QLineEdit()
        self.height_entry.setFixedWidth(40)
        self.height_entry.setText("4.0")
        self.height_entry.editingFinished.connect(self._emit_view_options)
        ratio_layout.addWidget(self.height_entry)

        ratio_action = QWidgetAction(self)
        ratio_action.setDefaultWidget(ratio_widget)
        menu.addAction(ratio_action)

        # Copied figure theme
        self.cbb_copy_theme = QComboBox()
        self.cbb_copy_theme.addItems(["Light Mode", "Dark Mode"])
        self.cbb_copy_theme.currentIndexChanged.connect(self._emit_view_options)
        menu.addAction(self._wrap("Copied figure theme:", self.cbb_copy_theme))

        return menu

    def _add_checkbox(self, menu, name, checked=False):
        """Helper to add a checkable action to the options menu."""
        act = menu.addAction(name)
        act.setCheckable(True)
        act.setChecked(checked)
        act.toggled.connect(self._emit_view_options)
        setattr(self, f"act_{name.lower()}", act)

    def _wrap(self, label, widget):
        """Helper to wrap a widget with a label into a QWidgetAction."""
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(2, 2, 2, 2)
        l.addWidget(QLabel(label))
        l.addWidget(widget)
        act = QWidgetAction(self)
        act.setDefaultWidget(w)
        return act

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────
    def set_plot_data(self, data):
        if isinstance(data, dict) and data.get("type") in ["tensor", "tensor_list"]:
            self._tensor_data = data
            self._current_spectra = data.get("proxies", [])
        else:
            self._tensor_data = None
            self._current_spectra = []
        self._plot()

    def _compute_shift_steps(self):
        """Compute per-spectrum X and Y shift steps based on slider values and data range."""
        y_slider = self.slider_y_shift.value()  # 0–100
        x_slider = self.slider_x_shift.value()  # 0–100

        if y_slider == 0 and x_slider == 0:
            return 0.0, 0.0

        if not self._tensor_data or len(self._tensor_data["y"]) == 0:
            return 0.0, 0.0

        x = self._tensor_data["x"]
        Y = self._tensor_data["y"]
        Y_norm = self._get_normalized_y_tensor(x, Y)

        if isinstance(Y_norm, list):
            y_max = max(np.max(y_arr) for y_arr in Y_norm)
            y_min = min(np.min(y_arr) for y_arr in Y_norm)
            y_range = float(y_max - y_min)
        else:
            y_range = float(np.max(Y_norm) - np.min(Y_norm))

        if isinstance(x, list):
            x_max = max(np.max(x_arr) for x_arr in x)
            x_min = min(np.min(x_arr) for x_arr in x)
            x_range = float(x_max - x_min)
        else:
            x_range = float(np.max(x) - np.min(x))

        # Map slider percentage to a fraction of the data range
        y_shift_step = (y_slider / 100.0) * y_range if y_range > 0 else 0.0
        x_shift_step = (x_slider / 100.0) * x_range if x_range > 0 else 0.0

        return x_shift_step, y_shift_step

    def _get_normalized_y_tensor(self, x, Y):
        """Apply normalization if enabled for a batch of spectra (VIEW-ONLY)."""
        if not self.btn_norm.isChecked():
            return Y

        try:
            xmin = float(self.norm_xmin.text()) if self.norm_xmin.text() else None
            xmax = float(self.norm_xmax.text()) if self.norm_xmax.text() else None
        except ValueError:
            xmin = xmax = None

        if isinstance(Y, list):
            Y_norm = []
            for i in range(len(Y)):
                xi = x[i] if isinstance(x, list) else x
                Y_norm.append(self._get_normalized_y(xi, Y[i]))
            return Y_norm
        else:
            if xmin is not None and xmax is not None:
                if xmin > xmax:
                    xmin, xmax = xmax, xmin

                imin = np.abs(x - xmin).argmin()
                imax = np.abs(x - xmax).argmin()
                
                # Extract slice handling empty slices safely
                sub_y = Y[:, min(imin, imax):max(imin, imax) + 1]
                if sub_y.shape[1] > 0:
                    norm_vals = np.max(sub_y, axis=1, keepdims=True)
                else:
                    norm_vals = np.max(Y, axis=1, keepdims=True)
            else:
                norm_vals = np.max(Y, axis=1, keepdims=True)

            # Avoid div by zero
            norm_vals[norm_vals == 0] = 1.0
            return Y / norm_vals

    def _plot(self):
        style_name = self.cbb_theme.currentText()
        style_path = PLOT_POLICY_LIGHT if style_name != "Dark Mode" else PLOT_POLICY_DARK
        with plt.style.context(style_path):
            self._plot_internal()

    def _plot_internal(self):
        if not self._tensor_data:
            self.ax.clear()
            self.lbl_r2.setText("R²=0")
            self.canvas.draw_idle()
            return

        # Save current zoom/pan limits so they survive a full ax.clear()
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        is_default = (xlim == (0.0, 1.0) and ylim == (0.0, 1.0))

        fg_color = plt.rcParams.get('axes.labelcolor', 'black')

        self.ax.clear()
        self._fitted_lines.clear()

        self._update_r2_display()

        x_shift_step, y_shift_step = self._compute_shift_steps()
        plot_style = self.cbb_plotstyle.currentText()
        lw = self.spin_lw.value()
        dot_size = self.spin_dotsize.value()
        colors_cycle = self._get_colors_cycle()

        # ── Step 1: Build bulk segment data (main spectra + raw overlay) ──
        segments = self._build_tensor_segments(
            x_shift_step, y_shift_step, plot_style, lw, fg_color,
            colors_cycle)

        # ── Step 2: Draw heavy per-spectrum overlays ──
        self._draw_tensor_overlays(x_shift_step, y_shift_step, lw,
                                   fg_color)

        # ── Step 3: Commit everything to the canvas ──
        self._finalize_plot(segments, plot_style, lw, dot_size, fg_color,
                            xlim, ylim, is_default)

    # ─────────────────────────────────────────────
    # Plot helpers: segment builders
    # ─────────────────────────────────────────────

    def _get_colors_cycle(self):
        """Return the current matplotlib color cycle or a sensible default."""
        prop_cycle = plt.rcParams.get('axes.prop_cycle')
        return (prop_cycle.by_key()['color'] if prop_cycle
                else DEFAULT_COLORS)

    def _build_tensor_segments(self, x_shift_step, y_shift_step,
                                plot_style, lw, fg_color, colors_cycle):
        """Build bulk LineCollection / scatter data for tensor-mode rendering."""
        # These lists are filled below and passed to _finalize_plot, which
        # converts them into a single LineCollection for efficient rendering.
        main_segments, main_colors = [], []  
        raw_segments, raw_colors = [], []   
        all_x_dots, all_y_dots, all_c_dots = [], [], [] 

        N = len(self._tensor_data["y"])
        x_offsets = np.arange(N) * x_shift_step      
        y_offsets = np.arange(N) * y_shift_step     

        x = self._tensor_data["x"]                   
        Y = self._tensor_data["y"]                   
        Y_norm = self._get_normalized_y_tensor(x, Y) 

        # Resolve each spectrum's color: use stored color if available, else cycle
        t_colors = self._tensor_data.get("colors", [None] * N)
        main_colors = [c if c else colors_cycle[i % len(colors_cycle)]
                       for i, c in enumerate(t_colors)]

        # tensor_list stores ragged arrays as a Python list; tensor stores a uniform 2-D ndarray
        is_list = isinstance(Y, list)

        if plot_style == "line":
            # Build one (M×2) coordinate array per spectrum for LineCollection
            if is_list:  # Ragged case: each spectrum may have a different length
                main_segments = [
                    np.column_stack([
                        (x[i] if isinstance(x, list) else x) + x_offsets[i],
                        Y_norm[i] + y_offsets[i],
                    ]) for i in range(N)
                ]
            else:  # Uniform case: broadcast offset addition on the whole 2-D array at once
                Y_plot = Y_norm + y_offsets[:, None]
                X_plot = x + x_offsets[:, None]
                main_segments = [np.column_stack([X_plot[i], Y_plot[i]])
                                 for i in range(N)]
        else:  # "dot" — collect all points into flat arrays for a single scatter call
            if is_list:
                for i in range(N):
                    xi = x[i] if isinstance(x, list) else x
                    all_x_dots.append(xi + x_offsets[i])
                    all_y_dots.append(Y_norm[i] + y_offsets[i])
                    all_c_dots.extend([main_colors[i]] * len(xi))  
            else:
                Y_plot = Y_norm + y_offsets[:, None]
                X_plot = x + x_offsets[:, None]
                all_x_dots.append(X_plot.flatten())
                all_y_dots.append(Y_plot.flatten())
                for i in range(N):
                    all_c_dots.extend([main_colors[i]] * len(x))

        # ── Raw (unprocessed) overlay segments ──
        # x0/y0 hold the original data before baseline subtraction / normalisation.
        if self.act_raw.isChecked():
            x0 = self._tensor_data.get("x0")
            Y0 = self._tensor_data.get("y0")
            if x0 is not None and Y0 is not None:
                if isinstance(Y0, list):
                    raw_segments = [
                        np.column_stack([
                            (x0[i] if isinstance(x0, list) else x0) + x_offsets[i],
                            Y0[i] + y_offsets[i],
                        ]) for i in range(N)
                    ]
                else:
                    Y0_plot = Y0 + y_offsets[:, None]
                    X0_plot = x0 + x_offsets[:, None]
                    raw_segments = [np.column_stack([X0_plot[i], Y0_plot[i]])
                                    for i in range(N)]
                raw_colors = [fg_color] * N  # Raw data drawn in a neutral foreground color

        # ── Proxy legend lines ──
        # Capped by _MAX_HEAVY_OVERLAYS to avoid an unmanageable legend for large maps.
        t_labels = self._tensor_data.get("labels", [])
        t_fnames = self._tensor_data.get("fnames", [])
        proxies = self._tensor_data.get("proxies", [])  # Spectrum objects for interactive tooltip
        for spec_idx in range(min(N, self._MAX_HEAVY_OVERLAYS)):
            spec_color = main_colors[spec_idx]
            # Priority: custom label > filename > generic fallback
            label_str = (
                t_labels[spec_idx]
                if spec_idx < len(t_labels) and t_labels[spec_idx]
                else (t_fnames[spec_idx] if spec_idx < len(t_fnames)
                      else f"Spectrum {spec_idx+1}")
            )
            proxy_line = mlines.Line2D([], [], color=spec_color,
                                        label=label_str, lw=lw)  # Zero-length: invisible on plot
            if spec_idx < len(proxies):
                proxy_line._spectrum_ref = proxies[spec_idx]  # Used by hover tooltip handler
            self.ax.add_line(proxy_line)

        # Return all collected data; _finalize_plot will render it
        return {
            "main_segments": main_segments, "main_colors": main_colors,
            "raw_segments": raw_segments, "raw_colors": raw_colors,
            "all_x_dots": all_x_dots, "all_y_dots": all_y_dots,
            "all_c_dots": all_c_dots,
        }


    # ─────────────────────────────────────────────
    # Plot helpers: overlay renderers
    # ─────────────────────────────────────────────

    def _get_tensor_item(self, key, spec_idx, is_tensor_list):
        """Safely retrieve a per-spectrum item from ``_tensor_data``."""
        items = self._tensor_data.get(key, [])
        if is_tensor_list:
            return items[spec_idx] if items and spec_idx < len(items) else None
        return items[spec_idx] if items is not None and len(items) > spec_idx else None

    def _draw_tensor_overlays(self, x_shift_step, y_shift_step, lw,
                               fg_color):
        """Draw baseline, bestfit peaks, residual, and noise overlays (tensor mode)."""
        n_specs = len(self._tensor_data.get("fnames", []))
        is_tensor_list = self._tensor_data.get("type") == "tensor_list"
        proxies = self._tensor_data.get("proxies", [])

        for spec_idx in range(min(n_specs, self._MAX_HEAVY_OVERLAYS)):
            x_val = self._tensor_data.get("x")
            x = x_val[spec_idx] if isinstance(x_val, list) else x_val

            y_val = self._tensor_data.get("y")
            y_raw = (y_val[spec_idx] if isinstance(y_val, list)
                     else y_val[spec_idx])

            x_offset = spec_idx * x_shift_step
            y_offset = spec_idx * y_shift_step

            # Check if baseline is subtracted
            is_sub = False
            if spec_idx < len(proxies):
                is_sub = getattr(proxies[spec_idx].baseline,
                                 "is_subtracted", False)
                if isinstance(is_sub, np.ndarray):
                    is_sub = bool(is_sub.any())

            # ── Baseline curve ──
            y_base = self._get_tensor_item("y_baseline", spec_idx,
                                            is_tensor_list)
            if y_base is not None and not is_sub:
                self.ax.plot(x + x_offset, y_base + y_offset,
                             'g--', lw=1.5, label="baseline")

            # ── Baseline anchor points ──
            bl_config = self._get_tensor_item("baseline_config", spec_idx,
                                               is_tensor_list)
            if bl_config and bl_config.get("points") and not is_sub:
                self._draw_tensor_baseline_points(
                    bl_config, x, y_raw, x_offset, y_offset, fg_color)

            # ── Peaks and Bestfit ──
            if self.btn_bestfit.isChecked():
                self._draw_tensor_bestfit(
                    spec_idx, is_tensor_list, x, y_raw, y_base,
                    x_offset, y_offset, lw, fg_color)

            # ── Residual ──
            if self.act_residual.isChecked():
                y_bestfit = self._get_tensor_item("y_bestfit", spec_idx,
                                                   is_tensor_list)
                if y_bestfit is not None:
                    residual = y_raw - y_bestfit
                    self.ax.plot(x + x_offset, residual + y_offset,
                                 "r-", lw=1.0, label="residual")

            # ── Noise level ──
            if self.act_noise_level.isChecked():
                self._draw_noise_level(y_raw, spec_idx, y_offset)

    def _draw_tensor_baseline_points(self, bl_config, x, y_raw,
                                      x_offset, y_offset, fg_color):
        """Draw anchor-point markers for a tensor-mode baseline."""
        mode = bl_config.get("mode", "")
        if mode not in ("Linear", "Polynomial"):
            return
        pts = bl_config["points"]
        if len(pts) != 2 or len(pts[0]) == 0:
            return
        xs_arr = np.asarray(pts[0], dtype=float)
        if bl_config.get("attached", False):
            inds = [np.argmin(np.abs(x - xp)) for xp in xs_arr]
            sigma = bl_config.get("sigma", 4)
            if sigma > 0:
                from scipy.ndimage import gaussian_filter1d
                y_curve = gaussian_filter1d(y_raw, sigma=sigma)
            else:
                y_curve = y_raw
            ys_arr = np.asarray([y_curve[ind] for ind in inds], dtype=float)
            xs_arr = np.asarray([x[ind] for ind in inds], dtype=float)
        else:
            ys_arr = np.asarray(pts[1], dtype=float)
        self.ax.plot(xs_arr + x_offset, ys_arr + y_offset,
                     marker="o", color=fg_color, mfc="none", ms=5, ls="none")

    def _draw_tensor_bestfit(self, spec_idx, is_tensor_list, x, y_raw,
                              y_base, x_offset, y_offset, lw, fg_color):
        """Draw bestfit peak curves and composite for one spectrum (tensor mode)."""
        fit_model = self._get_tensor_item("fit_models", spec_idx,
                                           is_tensor_list)

        # Retrieve individual peak curves
        y_peaks_list = self._tensor_data.get("y_peaks", [])
        y_peaks = None
        if y_peaks_list:
            if is_tensor_list:
                y_peaks = (y_peaks_list[spec_idx]
                           if spec_idx < len(y_peaks_list) else None)
            else:
                y_peaks = [p[spec_idx] for p in y_peaks_list]

        color_kwargs = {}
        if (hasattr(self, "act_bestfit_colorful")
                and not self.act_bestfit_colorful.isChecked()):
            color_kwargs["color"] = fg_color

        if fit_model and fit_model.get("peak_models") and len(x) > 1:
            # ── High-resolution smooth curves via eval_peak_initial ──
            x_fine = np.linspace(x.min(), x.max(), 1000)
            sorted_keys = sorted(fit_model["peak_models"].keys(),
                                 key=lambda k: int(k))

            y_fit_fine = np.zeros_like(x_fine)
            for i, k in enumerate(sorted_keys):
                p_model = fit_model["peak_models"][k]
                y_peak_fine = eval_peak_initial(x_fine, p_model)
                y_fit_fine += y_peak_fine

                peak_line, = self.ax.plot(
                    x_fine + x_offset, y_peak_fine + y_offset,
                    lw=(lw * 0.6), **color_kwargs)

                peak_label = f"Peak{i+1}"
                if ('peak_labels' in fit_model
                        and i < len(fit_model['peak_labels'])):
                    peak_label = fit_model['peak_labels'][i]

                peak_info = self._build_tensor_peak_info(
                    p_model, peak_label, int(k), spec_idx)
                self._fitted_lines.append((peak_line, peak_info))

                self._annotate_peak_label(
                    x_fine, y_peak_fine, peak_label, peak_line,
                    x_offset, y_offset, fg_color)

            # Add interpolated baseline for smooth composite curve
            if y_base is not None:
                y_base_fine = np.interp(x_fine, x, y_base)
                y_fit_fine += y_base_fine

            # Plot smooth composite bestfit
            self.ax.plot(x_fine + x_offset, y_fit_fine + y_offset,
                         lw=(lw * 0.6), color=fg_color)

        elif y_peaks is not None:
            # ── Fallback: discrete peak curves ──
            for i, y_peak in enumerate(y_peaks):
                peak_line, = self.ax.plot(
                    x + x_offset, y_peak + y_offset,
                    lw=(lw * 0.6), **color_kwargs)

                peak_label = f"Peak{i+1}"
                if (fit_model and 'peak_labels' in fit_model
                        and i < len(fit_model['peak_labels'])):
                    peak_label = fit_model['peak_labels'][i]

                peak_info = {
                    "peak_label": peak_label, "peak_model": None,
                    "index": i, "spec_idx": spec_idx,
                }
                if fit_model and "peak_models" in fit_model:
                    k = str(i)
                    if k in fit_model["peak_models"]:
                        p_model = fit_model["peak_models"][k]
                        shape_name = (list(p_model.keys())[0]
                                      if p_model else "")
                        peak_info["peak_model"] = (
                            _MockPeakModelObj(shape_name) if shape_name
                            else None)
                        peak_info["shape"] = shape_name
                        if shape_name:
                            params = p_model[shape_name]
                            for param_name, param_dict in params.items():
                                if (isinstance(param_dict, dict)
                                        and "value" in param_dict):
                                    peak_info[param_name] = (
                                        param_dict["value"])
                                    if param_name == "amplitude":
                                        peak_info["ampli"] = (
                                            param_dict["value"])

                self._fitted_lines.append((peak_line, peak_info))

                self._annotate_peak_label(
                    x, y_peak, peak_label, peak_line,
                    x_offset, y_offset, fg_color)

        # Composite bestfit curve (fallback from y_bestfit data)
        y_bestfit = self._get_tensor_item("y_bestfit", spec_idx,
                                           is_tensor_list)
        if y_bestfit is not None:
            if not (fit_model and fit_model.get("peak_models")):
                self.ax.plot(x + x_offset, y_bestfit + y_offset,
                             lw=(lw * 0.6), color=fg_color)

    def _build_tensor_peak_info(self, p_model, peak_label, index, spec_idx):
        """Build a peak-info dict from a tensor-mode peak model dictionary."""
        shape_name = list(p_model.keys())[0] if p_model else ""
        peak_info = {
            "peak_label": peak_label,
            "peak_model": (
                _MockPeakModelObj(shape_name) if shape_name else None),
            "index": index,
            "shape": shape_name,
            "spec_idx": spec_idx,
        }
        if shape_name:
            params = p_model[shape_name]
            for param_name, param_dict in params.items():
                if isinstance(param_dict, dict) and "value" in param_dict:
                    peak_info[param_name] = param_dict["value"]
                    if param_name == "amplitude":
                        peak_info["ampli"] = param_dict["value"]
        return peak_info

    def _annotate_peak_label(self, x_arr, y_arr, peak_label, peak_line,
                              x_offset, y_offset, fg_color):
        """Draw a text label at the peak maximum if the option is enabled."""
        if not (hasattr(self, "act_show_peak_label")
                and self.act_show_peak_label.isChecked()):
            return
        idx = np.argmax(np.abs(y_arr))
        px, py = x_arr[idx], y_arr[idx]
        txt_color = (fg_color if not self.act_bestfit_colorful.isChecked()
                     else peak_line.get_color())
        self.ax.text(px + x_offset, py + y_offset, peak_label,
                     color=txt_color, fontsize=9, ha="center",
                     va="bottom" if py >= 0 else "top")


    def _draw_noise_level(self, y_data, spec_idx, y_offset):
        """Draw a horizontal noise-level line."""
        try:
            coef_noise = QSettings("CEA-Leti", "SPECTROview").value(
                "fit_settings/coef_noise", 1.0, float)
            ampli_noise = detect_noise_level(y_data)
            noise_level = coef_noise * ampli_noise
            noise_label = (f"noise ({coef_noise}×{ampli_noise:.1f})"
                           if spec_idx == 0 else None)
            self.ax.axhline(y=noise_level + y_offset, color="orange",
                            ls="--", lw=1.0, alpha=0.8, label=noise_label)
        except Exception:
            pass

    # ─────────────────────────────────────────────
    # Plot helpers: finalize
    # ─────────────────────────────────────────────

    def _finalize_plot(self, segments, plot_style, lw, dot_size, fg_color,
                        xlim, ylim, is_default):
        """Render bulk collections, then apply legend, grid, axes labels, and limits."""
        main_segments = segments["main_segments"]
        main_colors = segments["main_colors"]
        raw_segments = segments["raw_segments"]
        raw_colors = segments["raw_colors"]
        all_x_dots = segments["all_x_dots"]
        all_y_dots = segments["all_y_dots"]
        all_c_dots = segments["all_c_dots"]

        # ── Render Bulk Collections ──
        if plot_style == "dot" and all_x_dots:
            self.ax.scatter(
                np.concatenate(all_x_dots), np.concatenate(all_y_dots),
                s=dot_size, c=all_c_dots, marker='o', edgecolors='none')
            all_points = np.column_stack(
                [np.concatenate(all_x_dots), np.concatenate(all_y_dots)])
            self.ax.update_datalim(all_points)
        elif main_segments:
            lc = mcoll.LineCollection(main_segments, colors=main_colors,
                                      linewidths=lw)
            self.ax.add_collection(lc)
            all_points = np.vstack(main_segments)
            self.ax.update_datalim(all_points)

        if raw_segments:
            lc_raw = mcoll.LineCollection(raw_segments, colors=raw_colors,
                                           linewidths=0.8, alpha=0.8)
            self.ax.add_collection(lc_raw)
            raw_points = np.vstack(raw_segments)
            self.ax.update_datalim(raw_points)

        self.ax.autoscale_view()

        # ── Legend / axes / grid ──
        if self.btn_legend.isChecked():
            legend = self.ax.legend(loc="best")
            self._make_legend_pickable(legend)

        if self.act_grid.isChecked():
            self.ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

        self.ax.set_xlabel(self.cbb_xaxis.currentText())
        self.ax.set_ylabel(self.cbb_yaxis.currentText())
        self.ax.set_yscale(
            "log" if self.cbb_yscale.currentText() == "Log" else "linear")

        if not is_default:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

        self.canvas.draw_idle()

    def _get_normalized_y(self, x, y):
        """Apply normalization if enabled (VIEW-ONLY)."""
        if not self.btn_norm.isChecked():
            return y

        try:
            xmin = float(self.norm_xmin.text()) if self.norm_xmin.text() else None
            xmax = float(self.norm_xmax.text()) if self.norm_xmax.text() else None
        except ValueError:
            xmin = xmax = None

        if xmin is not None and xmax is not None:
            if xmin > xmax:
                xmin, xmax = xmax, xmin

            imin = np.abs(x - xmin).argmin()
            imax = np.abs(x - xmax).argmin()
            norm_val = np.max(y[min(imin, imax):max(imin, imax) + 1])
        else:
            norm_val = np.max(y)

        if norm_val != 0:
            return y / norm_val

        return y

    
    def _make_legend_pickable(self, legend):
        """Make legend texts and handles pickable for interaction (double-click)."""
        for text in legend.get_texts():
            text.set_picker(True)

        for handle in legend.legend_handles:
            handle.set_picker(True)

        # Cache legend and its artists for double-click hit-testing
        self._legend_obj = legend
        self._legend_bbox = legend.get_window_extent(self.canvas.renderer)

        # Connect double-click handler once (replaces pick_event)
        if not hasattr(self, "_legend_dblclick_connected"):
            self.canvas.mpl_connect("button_press_event", self._on_legend_double_click)
            self._legend_dblclick_connected = True


    def set_r2(self, value):
        self.lbl_r2.setText(f"R²={value:.4f}")
    
    def _update_r2_display(self):
        """Display R² value and noise level from the first selected spectrum."""
        if not self._tensor_data:
            self.lbl_r2.setText("R²=0")
            self.lbl_noise.setText("Noise=0")
            return

        r2_vals = self._tensor_data.get("fit_r2")
        y_vals = self._tensor_data.get("y")
        if r2_vals is not None and len(r2_vals) > 0 and r2_vals[0] is not None:
            r2 = r2_vals[0]
            self.lbl_r2.setText(f"R²={r2:.3f}" if r2 > 0 else "R²=0")
        else:
            self.lbl_r2.setText("R²=0")

        if y_vals is not None and len(y_vals) > 0:
            y0 = y_vals[0] if not isinstance(y_vals, list) else y_vals[0]
            try:
                coef_noise = QSettings("CEA-Leti", "SPECTROview").value(
                    "fit_settings/coef_noise", 1.0, float
                )
                noise = detect_noise_level(y0)
                noise_val = coef_noise * noise
                self.lbl_noise.setText(f"Noise={noise_val:.1f}")
            except Exception:
                self.lbl_noise.setText("Noise=0")

    def _on_legend_double_click(self, event):
        """Handle double-click on legend text or marker to edit label/color."""
        if not event.dblclick:
            return
        if event.inaxes != self.ax:
            return

        legend = getattr(self, "_legend_obj", None)
        if legend is None:
            return

        # Hit-test legend texts
        for text in legend.get_texts():
            contains, _ = text.contains(event)
            if contains:
                self._edit_legend_label(text)
                return

        # Hit-test legend handles
        for handle_idx, handle in enumerate(legend.legend_handles):
            contains, _ = handle.contains(event)
            if contains:
                if handle_idx < len(legend.get_texts()):
                    text_artist = legend.get_texts()[handle_idx]
                    self._edit_legend_color_with_label(handle, text_artist.get_text())
                else:
                    self._edit_legend_color(handle)
                return

    def _edit_legend_label(self, artist):
        """Open dialog to rename the spectrum label."""
        old_label = artist.get_text()

        new_label, ok = QInputDialog.getText(
            self,
            "Edit legend label",
            "New label:",
            text=old_label
        )

        if not ok or not new_label.strip():
            return

        artist.set_text(new_label)

        for line in self.ax.get_lines():
            if line.get_label() == old_label:
                line.set_label(new_label)

                if hasattr(line, "_spectrum_ref"):
                    line._spectrum_ref.label = new_label
                break

        self.spectrumCustomized.emit()
        self.canvas.draw_idle()

    def _edit_legend_color(self, artist):
        """Open color picker to change the spectrum color."""
        color = QColorDialog.getColor()

        if not color.isValid():
            return

        hex_color = color.name()
        artist.set_color(hex_color)

        for line in self.ax.get_lines():
            if line.get_label() == artist.get_label():
                line.set_color(hex_color)

                if hasattr(line, "_spectrum_ref"):
                    line._spectrum_ref.color = hex_color
                break

        self.spectrumCustomized.emit()
        self.canvas.draw_idle()

    def _edit_legend_color_with_label(self, artist, label):
        """Open color picker to change the spectrum color using text label as key."""
        color = QColorDialog.getColor()

        if not color.isValid():
            return

        hex_color = color.name()
        artist.set_color(hex_color)

        for line in self.ax.get_lines():
            if line.get_label() == label:
                line.set_color(hex_color)

                if hasattr(line, "_spectrum_ref"):
                    line._spectrum_ref.color = hex_color
                break

        self.spectrumCustomized.emit()
        self.canvas.draw_idle()


    # ─────────────────────────────────────────
    # Signal emitters & Synchronization between spectra and maps workspaces
    # ─────────────────────────────────────────
    def get_options_state(self):
        """Get the complete state of all view options."""
        return {
            "theme": self.cbb_theme.currentText() if hasattr(self, "cbb_theme") else "Light Mode",
            "xaxis": self.cbb_xaxis.currentText() if hasattr(self, "cbb_xaxis") else "",
            "yaxis": self.cbb_yaxis.currentText() if hasattr(self, "cbb_yaxis") else "",
            "yscale": self.cbb_yscale.currentText() if hasattr(self, "cbb_yscale") else "Linear",
            "plotstyle": self.cbb_plotstyle.currentText() if hasattr(self, "cbb_plotstyle") else "line",
            "lw": self.spin_lw.value() if hasattr(self, "spin_lw") else 1.5,
            "dotsize": self.spin_dotsize.value() if hasattr(self, "spin_dotsize") else 3.0,
            "raw": self.act_raw.isChecked() if hasattr(self, "act_raw") else False,
            "bestfit_colorful": self.act_bestfit_colorful.isChecked() if hasattr(self, "act_bestfit_colorful") else True,
            "show_peak_label": self.act_show_peak_label.isChecked() if hasattr(self, "act_show_peak_label") else False,
            "residual": self.act_residual.isChecked() if hasattr(self, "act_residual") else False,
            "grid": self.act_grid.isChecked() if hasattr(self, "act_grid") else False,
            "noise_level": self.act_noise_level.isChecked() if hasattr(self, "act_noise_level") else False,
            "width": self.width_entry.text() if hasattr(self, "width_entry") else "5.5",
            "height": self.height_entry.text() if hasattr(self, "height_entry") else "4.0",
            "legend": self.btn_legend.isChecked() if hasattr(self, "btn_legend") else False,
            "bestfit": self.btn_bestfit.isChecked() if hasattr(self, "btn_bestfit") else False,
            "copy_fig_theme": self.cbb_copy_theme.currentText() if hasattr(self, "cbb_copy_theme") else "Light Mode",
        }

    def set_options_state(self, state):
        """Set the complete state of all view options without causing infinite loops."""
        if not hasattr(self, "cbb_theme"):
            return
            
        current_state = self.get_options_state()
        if current_state == state:
            return
            
        def _update(widget, setter, value):
            if value is None: return
            widget.blockSignals(True)
            setter(value)
            widget.blockSignals(False)

        _update(self.cbb_theme, self.cbb_theme.setCurrentText, state.get("theme"))
        _update(self.cbb_xaxis, self.cbb_xaxis.setCurrentText, state.get("xaxis"))
        _update(self.cbb_yaxis, self.cbb_yaxis.setCurrentText, state.get("yaxis"))
        _update(self.cbb_yscale, self.cbb_yscale.setCurrentText, state.get("yscale"))
        _update(self.cbb_plotstyle, self.cbb_plotstyle.setCurrentText, state.get("plotstyle"))
        _update(self.spin_lw, self.spin_lw.setValue, state.get("lw"))
        _update(self.spin_dotsize, self.spin_dotsize.setValue, state.get("dotsize"))
        _update(self.act_raw, self.act_raw.setChecked, state.get("raw"))
        _update(self.act_bestfit_colorful, self.act_bestfit_colorful.setChecked, state.get("bestfit_colorful"))
        _update(self.act_show_peak_label, self.act_show_peak_label.setChecked, state.get("show_peak_label"))
        _update(self.act_residual, self.act_residual.setChecked, state.get("residual"))
        _update(self.act_grid, self.act_grid.setChecked, state.get("grid"))
        if hasattr(self, "act_noise_level"):
            _update(self.act_noise_level, self.act_noise_level.setChecked, state.get("noise_level"))
        _update(self.width_entry, self.width_entry.setText, state.get("width"))
        _update(self.height_entry, self.height_entry.setText, state.get("height"))
        _update(self.btn_legend, self.btn_legend.setChecked, state.get("legend"))
        _update(self.btn_bestfit, self.btn_bestfit.setChecked, state.get("bestfit"))
        if hasattr(self, "cbb_copy_theme"):
            _update(self.cbb_copy_theme, self.cbb_copy_theme.setCurrentText, state.get("copy_fig_theme", "Light Mode"))

        self._apply_plot_style()
        
    def _apply_plot_style(self):
        style_name = self.cbb_theme.currentText()
        style_path = PLOT_POLICY_LIGHT if style_name != "Dark Mode" else PLOT_POLICY_DARK
        
        # Parse the style file without modifying global rcParams
        style_dict = mpl.rc_params_from_file(style_path)
            
        self.figure.patch.set_facecolor(style_dict.get('figure.facecolor', 'white'))
        self.figure.patch.set_edgecolor(style_dict.get('figure.edgecolor', 'white'))
        self.ax.set_facecolor(style_dict.get('axes.facecolor', 'white'))
        
        edge_color = style_dict.get('axes.edgecolor', 'black')
        for spine in self.ax.spines.values():
            spine.set_color(edge_color)
            
        tick_color = style_dict.get('xtick.color', 'black')
        self.ax.tick_params(colors=tick_color, which='both')
        
        label_color = style_dict.get('axes.labelcolor', 'black')
        self.ax.xaxis.label.set_color(label_color)
        self.ax.yaxis.label.set_color(label_color)
        self.ax.title.set_color(label_color)
        
        if not getattr(self, '_is_copying', False):
            self.plotStyleChanged.emit()
            self._emit_view_options()

    def _emit_view_options(self):
        def _to_float(text, default):
            try:
                return float(text)
            except Exception:
                return default

        self.viewOptionsChanged.emit({
            "legend": self.btn_legend.isChecked(),
            "grid": self.act_grid.isChecked(),
            #"colors": self.act_colors.isChecked(),
            "raw": self.act_raw.isChecked(),
            "bestfit": self.btn_bestfit.isChecked(),  # Now using toolbar button
            "plot_theme": self.cbb_theme.currentText() if hasattr(self, "cbb_theme") else "Light Mode",
            "bestfit_colorful": self.act_bestfit_colorful.isChecked(),
            "show_peak_label": self.act_show_peak_label.isChecked(),
            "residual": self.act_residual.isChecked(),
            "noise_level": self.act_noise_level.isChecked(),
            "x_unit": self.cbb_xaxis.currentText(),
            "y_scale": self.cbb_yscale.currentText(),
            "linewidth": self.spin_lw.value(),
            "copy_width": _to_float(self.width_entry.text(), 5.5),
            "copy_height": _to_float(self.height_entry.text(), 4.0),
        })
        # Sync options with other viewers
        self.allOptionsSyncChanged.emit(self.get_options_state())
        
        # Update plot immediately whenever an view option changes
        self._plot() 


    def _emit_copy(self):
        modifiers = QApplication.keyboardModifiers() & Qt.ControlModifier
        
        if modifiers:
            # Request ViewModel to copy spectrum data
            self.copy_data_requested.emit()
        else:
            # Copy canvas directly using viewmodel utility
            width = float(self.width_entry.text()) if self.width_entry.text() else 5.5
            height = float(self.height_entry.text()) if self.height_entry.text() else 4.0
            
            # Get target copied theme from combobox
            target_theme = self.cbb_copy_theme.currentText() if hasattr(self, "cbb_copy_theme") else "Light Mode"
            original_theme = self.cbb_theme.currentText()
            
            if target_theme != original_theme:
                self._is_copying = True
                try:
                    self.cbb_theme.blockSignals(True)
                    self.cbb_theme.setCurrentText(target_theme)
                    self._apply_plot_style()
                    self._plot()
                    self.canvas.draw()
                    
                    copy_fig_to_clb(self.canvas, size_ratio=(width, height))
                finally:
                    self.cbb_theme.setCurrentText(original_theme)
                    self.cbb_theme.blockSignals(False)
                    self._apply_plot_style()
                    self._plot()
                    self._is_copying = False
            else:
                copy_fig_to_clb(self.canvas, size_ratio=(width, height))

    def _emit_norm(self):
        try:
            xmin = float(self.norm_xmin.text()) if self.norm_xmin.text() else None
            xmax = float(self.norm_xmax.text()) if self.norm_xmax.text() else None
        except ValueError:
            xmin = xmax = None
        self.normalizationChanged.emit(self.btn_norm.isChecked(), xmin, xmax)

    def _on_mouse_click(self, event):
        if event.inaxes != self.ax:
            return

        if self.zoom_pan_active:
            return

        # Skip legend clicks
        if getattr(self, "_legend_bbox", None) is not None:
            if self._legend_bbox.contains(event.x, event.y):
                return

        x = event.xdata
        y = event.ydata

        # ─── Peak tool ─────────────────────────
        if self.btn_peak.isChecked():
            if event.button == 1:      # left click
                # Check if clicking on an existing peak to drag it
                clicked_peak = self._find_peak_at_position(x, y)
                if clicked_peak:
                    # Start dragging
                    self._dragging_peak = clicked_peak
                    self.peak_drag_started.emit(clicked_peak)
                    # Connect drag events
                    self._drag_cid = self.canvas.mpl_connect('motion_notify_event', self._on_drag_peak)
                    self._release_cid = self.canvas.mpl_connect('button_release_event', self._on_release_drag)
                else:
                    # Add new peak
                    self.peak_add_requested.emit(x)
            elif event.button == 3:    # right click
                self.peak_remove_requested.emit(x)

        # ─── Baseline tool ─────────────────────
        elif self.btn_baseline.isChecked():
            if event.button == 1:
                self.baseline_add_requested.emit(x, y)
            elif event.button == 3:
                self.baseline_remove_requested.emit(x)



    def _rescale(self):
        """Rescale the spectra plot to fit within the axes."""
        self.ax.autoscale()
        self.canvas.draw_idle()
        
    def _toggle_zoom_pan(self):
        """Zoom feature for button Zoom"""
        if self.btn_zoom.isChecked():
            self.zoom_pan_active = True
            self.toolbar.zoom() 
        else:
            self.zoom_pan_active = False
            self.toolbar.zoom() 

    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return

        y_min, y_max = self.ax.get_ylim()
        dy = (y_max - y_min) * 0.1

        if event.step < 0:      # scroll up
            y_max += dy
        else:                   # scroll down
            y_max = max(y_min + 1e-9, y_max - dy)

        self.ax.set_ylim(y_min, y_max)
        self.canvas.draw_idle()

    def _on_hover(self, event):
        # Same early exits as legacy
        if event.inaxes != self.ax:
            self._hide_tooltip()
            self._reset_highlight()
            return

        if not self._fitted_lines:
            self._hide_tooltip()
            self._reset_highlight()
            return

        for line, info in self._fitted_lines:
            # Check if line still has a valid figure before calling contains()
            if line.figure is None:
                continue
            
            hit, _ = line.contains(event)
            if not hit:
                continue

            peak_model_obj = info.get("peak_model")
            model_name = getattr(peak_model_obj, "name2", "") if peak_model_obj else ""
            
            raw_intensity = info.get("amplitude") or info.get("ampli")
            display_intensity = raw_intensity
            
            # Fano model correction: show actual peak height for 'ampli'
            if model_name == "Fano" and display_intensity is not None:
                q = info.get("q", 50.0)
                display_intensity = display_intensity * (q**2 + 1)

            # Define fields exactly like legacy
            fields = [
                ("label", info.get("peak_label")),
                ("center", info.get("x0")),
                ("intensity", display_intensity),
                ("fwhm", info.get("fwhm")),
                ("fwhm_l", info.get("fwhm_l")),
                ("fwhm_r", info.get("fwhm_r")),
                ("alpha", info.get("alpha")),
            ]

            lines = []
            for label, val in fields:
                if val is None:
                    continue
                try:
                    val_str = f"{val:.3f}" if isinstance(val, (float, int)) else str(val)
                except Exception:
                    val_str = str(val)
                lines.append(f"{label}: {val_str}")

            # 🔴 CRITICAL: do NOT show tooltip if nothing to show
            if not lines:
                self._hide_tooltip()
                self._reset_highlight()
                return

            text = "\n".join(lines)
            self._show_tooltip(event, text)
            self._highlight_line(line)
            return

        # Nothing hit
        self._hide_tooltip()
        self._reset_highlight()

    def _show_tooltip(self, event, text):

        if not hasattr(self, "_tooltip"):
            self._tooltip = QLabel(self.canvas)
            self._tooltip.setStyleSheet("""
                background-color: rgba(255, 255, 255, 0.5);
                color: black;
                border: 0.1px solid gray;
                padding: 2px;
            """)
            self._tooltip.setWindowFlags(Qt.ToolTip)

        self._tooltip.setText(text)
        self._tooltip.move(QCursor.pos() + QPoint(10, -40))
        self._tooltip.show()


    def _hide_tooltip(self):
        if hasattr(self, "_tooltip"):
            self._tooltip.hide()

    def _highlight_line(self, line):
        if self._highlighted_line == line:
            return

        self._reset_highlight()
        line._orig_lw = line.get_linewidth()
        line.set_linewidth(3)
        self._highlighted_line = line
        self.canvas.draw_idle()


    def _reset_highlight(self):
        if self._highlighted_line is not None:
            lw = getattr(self._highlighted_line, "_orig_lw", 1.5)
            self._highlighted_line.set_linewidth(lw)
            self._highlighted_line = None
            self.canvas.draw_idle()

    def _find_peak_at_position(self, x, y):
        """Find if a peak line is near the clicked position."""
        if not self._fitted_lines:
            return None

        # Use pixel-based tolerance for better UX (consistent "visual" click area)
        # 10 pixels is a standard "slop" for mouse clicks
        TOLERANCE_PX = 10 
        
        # Convert click x-coordinate to pixels
        # We use [(x, 0)] because we only care about the x-projection
        try:
            click_pix = self.ax.transData.transform([(x, 0)])[0][0]
        except Exception:
             return None

        closest_peak = None
        min_distance = float('inf')

        for line, info in self._fitted_lines:
            # Check distance from peak center (x0)
            if "x0" in info:
                peak_x = info["x0"]
                
                try:
                    # Convert peak x to pixels
                    peak_pix = self.ax.transData.transform([(peak_x, 0)])[0][0]
                    distance = abs(click_pix - peak_pix)
                    
                    if distance < TOLERANCE_PX and distance < min_distance:
                        min_distance = distance
                        closest_peak = (line, info)
                except Exception:
                    continue

        return closest_peak

    def _on_drag_peak(self, event):
        """Handle peak dragging - update peak position in real-time."""
        if not hasattr(self, '_dragging_peak') or self._dragging_peak is None:
            return

        if event.xdata is None or event.ydata is None:
            return

        line, info = self._dragging_peak

        # Update peak model parameters
        new_x = event.xdata
        new_y = event.ydata

        # Emit signal to ViewModel to update model
        self.peak_dragged.emit(new_x, new_y)

        # Update visual feedback immediately
        spec_idx = info.get("spec_idx", 0)
        peak_idx_str = str(info.get("index", 0))

        if self._tensor_data:
            fit_models = self._tensor_data.get("fit_models", [])
            if fit_models and spec_idx < len(fit_models):
                fit_model = fit_models[spec_idx]
                if fit_model and "peak_models" in fit_model and peak_idx_str in fit_model["peak_models"]:
                    pdict = fit_model["peak_models"][peak_idx_str]
                    shape = list(pdict.keys())[0] if pdict else ""
                    if shape:
                        if "x0" in pdict[shape]:
                            pdict[shape]["x0"]["value"] = new_x
                        if "ampli" in pdict[shape]:
                            pdict[shape]["ampli"]["value"] = new_y
                        elif "amplitude" in pdict[shape]:
                            pdict[shape]["amplitude"]["value"] = new_y

        # Redraw
        self._plot()

    def _on_release_drag(self, event):
        """Handle peak drag release - finalize the change."""
        if hasattr(self, '_dragging_peak') and self._dragging_peak is not None:
            # Disconnect drag events
            if hasattr(self, '_drag_cid'):
                self.canvas.mpl_disconnect(self._drag_cid)
                delattr(self, '_drag_cid')
            
            if hasattr(self, '_release_cid'):
                self.canvas.mpl_disconnect(self._release_cid)
                delattr(self, '_release_cid')

            # Emit final signal
            self.peak_drag_finished.emit()
            
            # Clear dragging state
            self._dragging_peak = None

