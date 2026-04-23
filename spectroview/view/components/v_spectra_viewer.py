# view/components/spectra_viewer.py
from copy import deepcopy
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QToolButton, QLabel,
    QComboBox, QMenu, QWidgetAction,
    QLineEdit, QDoubleSpinBox, QColorDialog, QInputDialog,
    QSlider, QGroupBox
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon, QShortcut, QKeySequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT
)

from PySide6.QtWidgets import QLabel, QApplication
from PySide6.QtGui import QCursor
from PySide6.QtCore import QPoint

import matplotlib.lines as mlines
import matplotlib.text as mtext

from spectroview import ICON_DIR, X_AXIS_UNIT, Y_AXIS_UNIT, PLOT_POLICY, PLOT_POLICY_LIGHT, PLOT_POLICY_DARK
from spectroview.viewmodel.utils import copy_fig_to_clb
from spectroview.view.components.customized_widgets import NoDoubleClickZoomToolbar


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

    peak_add_requested = Signal(float)
    peak_remove_requested = Signal(float)
    baseline_add_requested = Signal(float, float)
    baseline_remove_requested = Signal(float)
    peak_drag_started = Signal(object)   # optional (advanced)
    peak_dragged = Signal(float, float)
    peak_drag_finished = Signal()


    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._current_spectra = []
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
    def set_plot_data(self, selected_spectra):
        self._current_spectra = selected_spectra or []
        self._plot()

    def _compute_shift_steps(self):
        """Compute per-spectrum X and Y shift steps based on slider values and data range."""
        if not self._current_spectra:
            return 0.0, 0.0

        y_slider = self.slider_y_shift.value()  # 0–100
        x_slider = self.slider_x_shift.value()  # 0–100

        if y_slider == 0 and x_slider == 0:
            return 0.0, 0.0

        # Determine global data range across all selected spectra
        all_y = []
        all_x = []
        for s in self._current_spectra:
            y = self._get_normalized_y(s.x, s.y)
            all_y.append(y)
            all_x.append(s.x)

        y_range = max(np.max(a) for a in all_y) - min(np.min(a) for a in all_y)
        x_range = max(np.max(a) for a in all_x) - min(np.min(a) for a in all_x)

        # Map slider percentage to a fraction of the data range
        y_shift_step = (y_slider / 100.0) * y_range if y_range > 0 else 0.0
        x_shift_step = (x_slider / 100.0) * x_range if x_range > 0 else 0.0

        return x_shift_step, y_shift_step

    def _plot(self):
        style_name = self.cbb_theme.currentText()
        style_path = PLOT_POLICY_LIGHT if style_name != "Dark Mode" else PLOT_POLICY_DARK
        with plt.style.context(style_path):
            self._plot_internal()

    def _plot_internal(self):
        if not self._current_spectra:
            self.ax.clear()
            self.lbl_r2.setText("R²=0")
            self.canvas.draw_idle()
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        is_default = (xlim == (0.0, 1.0) and ylim == (0.0, 1.0))
        
        fg_color = plt.rcParams.get('axes.labelcolor', 'black')

        self.ax.clear()
        self._fitted_lines.clear()  # always reset
        
        # Display R² value from first spectrum (if fitted)
        self._update_r2_display()

        # Compute per-spectrum shift offsets from sliders
        x_shift_step, y_shift_step = self._compute_shift_steps()

        for spec_idx, spectrum in enumerate(self._current_spectra):
            x = spectrum.x
            y_raw = spectrum.y
            y = self._get_normalized_y(x, y_raw)
            lw = self.spin_lw.value()

            # Per-spectrum cascading offset (waterfall)
            x_offset = spec_idx * x_shift_step
            y_offset = spec_idx * y_shift_step

            # ── RAW data (original x0 / y0)
            if self.act_raw.isChecked() and hasattr(spectrum, "x0") and hasattr(spectrum, "y0"):
                try:
                    self.ax.plot(
                        spectrum.x0 + x_offset,
                        spectrum.y0 + y_offset,
                        "o-",
                        ms=3,
                        lw=0.8,
                        alpha=0.8,
                        color=fg_color,
                        label="raw", 
                    )
                except Exception:
                    pass

            # ── Main spectrum (always shown)
            plot_style = self.cbb_plotstyle.currentText()
            if plot_style == "dot":
                dot_size = self.spin_dotsize.value()
                line, = self.ax.plot(
                    x + x_offset, y + y_offset,
                    'o',
                    ms=dot_size,
                    label=spectrum.label or spectrum.fname,
                    color=spectrum.color
                )
            else:  # "line"
                line, = self.ax.plot(
                    x + x_offset, y + y_offset,
                    lw=lw,
                    label=spectrum.label or spectrum.fname,
                    color=spectrum.color
                )
            line._spectrum_ref = spectrum

            # ── Baseline (independent of bestfit toggle)
            y_base = self._plot_baseline(spectrum, x_offset, y_offset)


            # ── Peaks + Bestfit 
            if (
                self.btn_bestfit.isChecked()  # Now using toolbar button
                and getattr(spectrum, "peak_models", None)
                and spectrum.peak_models
            ):
                # Create dense interpolation grid for SMOOTH curves
                x_fine = np.linspace(x.min(), x.max(), 1000)
                y_peaks_fine = np.zeros_like(x_fine)
                
                # Also keep track on original grid for residual calculation
                y_peaks_orig = np.zeros_like(x)

                for i, peak_model in enumerate(spectrum.peak_models):
                    # Evaluate on FINE grid for smooth plotting
                    y_peak_fine = self._eval_peak_model_safe(peak_model, x_fine)
                    y_peaks_fine += y_peak_fine
                    
                    # Also evaluate on original grid (needed for residuals)
                    y_peak_orig = self._eval_peak_model_safe(peak_model, x)
                    y_peaks_orig += y_peak_orig

                    # ── Individual peak curve (SMOOTH) — with offset
                    color_kwargs = {}
                    if hasattr(self, "act_bestfit_colorful") and not self.act_bestfit_colorful.isChecked():
                        color_kwargs["color"] = fg_color

                    peak_line, = self.ax.plot(
                        x_fine + x_offset, y_peak_fine + y_offset,
                        lw=(lw*0.6),  **color_kwargs
                    )

                    peak_info = {
                        "peak_label": (
                            spectrum.peak_labels[i]
                            if i < len(spectrum.peak_labels)
                            else f"Peak {i+1}"
                        ),
                        "peak_model": peak_model,
                    }

                    for pname in getattr(peak_model, "param_names", []):
                        key = pname.split("_", 1)[1]
                        if key in peak_model.param_hints:
                            peak_info[key] = peak_model.param_hints[key].get("value")

                    self._fitted_lines.append((peak_line, peak_info))

                    # ── Peak label ──
                    if hasattr(self, "act_show_peak_label") and self.act_show_peak_label.isChecked():
                        idx = np.argmax(np.abs(y_peak_fine))
                        px = x_fine[idx]
                        py = y_peak_fine[idx]
                        txt_color = fg_color if not self.act_bestfit_colorful.isChecked() else peak_line.get_color()
                        self.ax.text(
                            px + x_offset, py + y_offset,
                            peak_info["peak_label"],
                            color=txt_color,
                            fontsize=9,
                            ha="center", va="bottom" if py >= 0 else "top"
                        )

                # ── Best-fit curve (SMOOTH) — with offset
                if (
                    hasattr(spectrum, "result_fit")
                    and getattr(spectrum.result_fit, "success", False)
                ):
                    # Evaluate baseline on fine grid for smooth bestfit
                    if y_base is not None:
                        baseline = spectrum.baseline
                        try:
                            y_base_fine = baseline.eval(x_fine, None, attached=False)
                        except Exception:
                            y_base_fine = np.zeros_like(x_fine)
                        y_fit_fine = y_peaks_fine + y_base_fine
                    else:
                        y_fit_fine = y_peaks_fine
                    
                    self.ax.plot(
                        x_fine + x_offset, y_fit_fine + y_offset,
                        lw=(lw*0.6), color=fg_color
                    )


            # ── Residual — with offset
            if self.act_residual.isChecked():
                try:
                    xr, residual = self._compute_residual(spectrum)
                    self.ax.plot(
                        xr + x_offset, residual + y_offset,
                        "r-", lw=1.0, label="residual"
                    )
                except Exception:
                    pass

        # ── Legend / axes / grid
        if self.btn_legend.isChecked():
            legend = self.ax.legend(loc="best")
            self._make_legend_pickable(legend)

        if self.act_grid.isChecked():
            self.ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

        self.ax.set_xlabel(self.cbb_xaxis.currentText())
        self.ax.set_ylabel(self.cbb_yaxis.currentText())
        self.ax.set_yscale(
            "log" if self.cbb_yscale.currentText() == "Log" else "linear"
        )

        if not is_default:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

        self.canvas.draw_idle()


    def _eval_peak_model_safe(self, peak_model, x):
        """
        Evaluate a peak model for plotting purposes.
        Expressions are temporarily disabled to avoid lmfit NameError.
        """
        # Backup param_hints
        param_hints_orig = deepcopy(peak_model.param_hints)

        # Disable expressions
        for key in peak_model.param_hints:
            peak_model.param_hints[key]["expr"] = ""

        try:
            params = peak_model.make_params()
            y = peak_model.eval(params, x=x)
        finally:
            # Restore expressions
            peak_model.param_hints = param_hints_orig

        return y
    
    def _get_baseline_y(self, spectrum, x):
        baseline = spectrum.baseline

        if baseline is None or baseline.is_subtracted:
            return np.zeros_like(x)

        if not baseline.points or not baseline.points[0]:
            return np.zeros_like(x)

        try:
            y_attached = spectrum.y if baseline.attached else None
            return baseline.eval(x, y_attached, attached=baseline.attached)
        except Exception:
            return np.zeros_like(x)

    def _get_peaks_y(self, spectrum, x):
        if not getattr(spectrum, "peak_models", None):
            return np.zeros_like(x)

        y_peaks = np.zeros_like(x)
        for peak_model in spectrum.peak_models:
            try:
                params = peak_model.make_params()
                y_peaks += peak_model.eval(params, x=x)
            except Exception:
                pass

        return y_peaks
    
    def _compute_residual(self, spectrum):
        x = spectrum.x
        y_raw = spectrum.y

        y_base = self._get_baseline_y(spectrum, x)
        y_peaks = self._get_peaks_y(spectrum, x)

        y_fit = y_base + y_peaks
        residual = y_raw - y_fit

        return x, residual

    _MANUAL_BASELINE_MODES = {None, "Linear", "Polynomial"}

    def _plot_baseline(self, spectrum, x_offset=0.0, y_offset=0.0):
        baseline = spectrum.baseline

        if baseline.is_subtracted:
            return None
        if baseline.mode is None:
            return None

        is_auto = baseline.mode not in self._MANUAL_BASELINE_MODES

        # For manual modes, require at least one anchor point
        if not is_auto and (not baseline.points or not baseline.points[0]):
            return None

        x = spectrum.x
        y = spectrum.y

        try:
            y_base = baseline.eval(x, y, attached=baseline.attached)
        except Exception:
            return None

        if y_base is None:
            return None

        # Baseline curve — with offset
        self.ax.plot(
            x + x_offset, y_base + y_offset,
            "--", color="red", lw=1.4, label="Baseline"
        )

        # Show anchor point markers only for manual modes — with offset
        if not is_auto and baseline.points and baseline.points[0]:
            y_att = y if baseline.attached else None
            if baseline.attached and y_att is not None:
                xs, ys = baseline.attached_points(x, y)
            else:
                xs, ys = baseline.points
            xs_arr = np.asarray(xs, dtype=float)
            ys_arr = np.asarray(ys, dtype=float)
            self.ax.plot(xs_arr + x_offset, ys_arr + y_offset, "ko", mfc="none", ms=5)

        return y_base

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
        """Display R² value from the first selected spectrum."""
        if not self._current_spectra:
            self.lbl_r2.setText("R²=0")
            return
        
        # Get first spectrum
        spectrum = self._current_spectra[0]
        
        # Check if it has fit results with R²
        if (hasattr(spectrum, 'result_fit') and 
            spectrum.result_fit is not None and 
            hasattr(spectrum.result_fit, 'rsquared')):
            rsquared = round(spectrum.result_fit.rsquared, 4)
            self.lbl_r2.setText(f"R²={rsquared}")
        else:
            self.lbl_r2.setText("R²=0")

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
        for handle in legend.legend_handles:
            contains, _ = handle.contains(event)
            if contains:
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

        self.canvas.draw_idle()


    # ─────────────────────────────────────────
    # Signal emitters
    # ─────────────────────────────────────────
    def _apply_plot_style(self):
        style_name = self.cbb_theme.currentText()
        style_path = PLOT_POLICY_LIGHT if style_name != "Dark Mode" else PLOT_POLICY_DARK
        
        # Parse the style file without modifying global rcParams
        import matplotlib as mpl
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
            "x_unit": self.cbb_xaxis.currentText(),
            "y_scale": self.cbb_yscale.currentText(),
            "linewidth": self.spin_lw.value(),
            "copy_width": _to_float(self.width_entry.text(), 5.5),
            "copy_height": _to_float(self.height_entry.text(), 4.0),
        })
        # Update plot immediately whenever an view option changes
        self._plot() 


    def _emit_copy(self):
        from PySide6.QtWidgets import QApplication
        ctrl = QApplication.keyboardModifiers() & Qt.ControlModifier
        
        if ctrl:
            # Request ViewModel to copy spectrum data
            self.copy_data_requested.emit()
        else:
            # Copy canvas directly using viewmodel utility
            width = float(self.width_entry.text()) if self.width_entry.text() else 5.5
            height = float(self.height_entry.text()) if self.height_entry.text() else 4.0
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
        peak_model = info.get('peak_model')
        
        if not peak_model:
            return

        # Update peak model parameters
        new_x = event.xdata
        new_y = event.ydata

        # Emit signal to ViewModel to update model
        self.peak_dragged.emit(new_x, new_y)

        # Update visual feedback immediately
        peak_model.param_hints['x0']['value'] = new_x
        if 'ampli' in peak_model.param_hints:
            peak_model.param_hints['ampli']['value'] = new_y
        elif 'amplitude' in peak_model.param_hints:
            peak_model.param_hints['amplitude']['value'] = new_y

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

