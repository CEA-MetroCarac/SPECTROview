# view/components/spectra_viewer.py
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QToolButton, QLabel,
    QComboBox, QMenu, QWidgetAction,
    QLineEdit, QDoubleSpinBox, QColorDialog, QInputDialog
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT
)

import matplotlib.lines as mlines
import matplotlib.text as mtext

from spectroview import ICON_DIR, X_AXIS_UNIT, PLOT_POLICY


class VSpectraViewer(QWidget):
    mouseClicked = Signal(float, float, int)
    zoomToggled = Signal(bool)
    rescaleRequested = Signal()
    viewOptionsChanged = Signal(dict)
    copyRequested = Signal(bool)   # ctrl_pressed
    toolModeChanged = Signal(str)  # zoom / baseline / peak
    normalizationChanged = Signal(bool, float, float)

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
        self._current_lines = []

        self.zoom_pan_active = True

        
    def _init_ui(self):
        plt.style.use(PLOT_POLICY)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ─── Figure ───
        self.figure = Figure(layout="compressed", dpi=80)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        
        self.canvas.mpl_connect("button_press_event", self._on_mouse_click)

        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.zoom() # Start with zoom enabled
        for action in self.toolbar.actions():
            if action.text() in ['Home', 'Save', 'Pan', 'Back', 'Forward', 'Subplots', 'Zoom']:
                action.setVisible(False)

        # ─── Control bar ───
        self.control_bar = self._create_control_bar()
        
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(self.control_bar)

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
        self.btn_norm.setIconSize(QSize(22, 22))
        self.btn_norm.toggled.connect(self._emit_norm)
        self.btn_norm.toggled.connect(self._redraw)
        self.btn_norm.clicked.connect(self._rescale)

        layout.addWidget(self.btn_norm)

        self.norm_xmin = QLineEdit()
        self.norm_xmin.setFixedWidth(45)
        self.norm_xmin.setPlaceholderText("Xmin")
        layout.addWidget(self.norm_xmin)

        self.norm_xmax = QLineEdit()
        self.norm_xmax.setFixedWidth(45)
        self.norm_xmax.setPlaceholderText("Xmax")
        layout.addWidget(self.norm_xmax)

        self.norm_xmin.editingFinished.connect(self._redraw)
        self.norm_xmax.editingFinished.connect(self._redraw)
        self.norm_xmin.editingFinished.connect(self._rescale)
        self.norm_xmax.editingFinished.connect(self._rescale)

        # Legend
        self.btn_legend = QToolButton()
        self.btn_legend.setCheckable(True)
        self.btn_legend.setIcon(QIcon(f"{ICON_DIR}/legend.png"))
        self.btn_legend.setIconSize(QSize(22, 22))
        self.btn_legend.toggled.connect(self._emit_view_options)
        layout.addWidget(self.btn_legend)

        # Copy
        self.btn_copy = QPushButton()
        self.btn_copy.setIcon(QIcon(f"{ICON_DIR}/copy.png"))
        self.btn_copy.setIconSize(QSize(22, 22))
        self.btn_copy.clicked.connect(self._emit_copy)
        layout.addWidget(self.btn_copy)

        # Options
        self.options_menu = self._create_options_menu()
        self.btn_options = QToolButton()
        self.btn_options.setIcon(QIcon(f"{ICON_DIR}/options.png"))
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

        # X-axis
        self.cbb_xaxis = QComboBox()
        self.cbb_xaxis.addItems(X_AXIS_UNIT)
        self.cbb_xaxis.currentIndexChanged.connect(self._emit_view_options)
        menu.addAction(self._wrap("X-axis:", self.cbb_xaxis))

        # Y-scale
        self.cbb_yscale = QComboBox()
        self.cbb_yscale.addItems(["Linear", "Log"])
        self.cbb_yscale.currentIndexChanged.connect(self._emit_view_options)
        menu.addAction(self._wrap("Y-scale:", self.cbb_yscale))

        menu.addSeparator()

        # Toggles
        self._add_checkbox(menu, "Colors")
        self._add_checkbox(menu, "Peaks")
        self._add_checkbox(menu, "Raw")
        self._add_checkbox(menu, "Bestfit", True)
        self._add_checkbox(menu, "Residual")
        self._add_checkbox(menu, "Grid")

        menu.addSeparator()

        # Line width
        self.spin_lw = QDoubleSpinBox()
        self.spin_lw.setRange(0.1, 5)
        self.spin_lw.setSingleStep(0.5)
        self.spin_lw.setValue(1.5)
        self.spin_lw.valueChanged.connect(self._emit_view_options)
        menu.addAction(self._wrap("Line width:", self.spin_lw))
        
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
        l.setContentsMargins(5, 5, 5, 5)
        l.addWidget(QLabel(label))
        l.addWidget(widget)
        act = QWidgetAction(self)
        act.setDefaultWidget(w)
        return act

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────
    def set_plot_data(self, lines):
        self._current_lines = lines or []
        self._redraw()

    def _redraw(self):
        """Redraw the spectra plot based on current lines and view options."""
        # Preserve zoom / pan state
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        is_default_limits = (xlim == (0.0, 1.0) and ylim == (0.0, 1.0))

        self.ax.clear()

        for line_data in self._current_lines:
            spectrum = line_data.get("_spectrum_ref")
            if spectrum is None:
                continue

            x = line_data["x"]
            y_raw = line_data["y"]
            y = self._get_normalized_y(x, y_raw)
            lw = self.spin_lw.value()

            kwargs = {
                k: v for k, v in line_data.items()
                if k not in ("x", "y", "_spectrum_ref")
            }

            # ── Main spectrum
            line, = self.ax.plot(x, y, lw=lw, **kwargs)
            line._spectrum_ref = spectrum

            # =====================================================
            # BASELINE POINTS
            # =====================================================
            if hasattr(spectrum.baseline, "points"):
                xs, ys = spectrum.baseline.points
                if xs and ys:
                    self.ax.plot(xs, ys, "ro", ms=4, zorder=5)

            # =====================================================
            # BASELINE CURVE (SAFE)
            # =====================================================
            y_base = None
            if spectrum.baseline.mode is not None:
                try:
                    params = spectrum.baseline.mode.make_params()
                    y_base = spectrum.baseline.mode.eval(params, x=x)
                    self.ax.plot(
                        x, y_base, "--", color="gray", lw=1.2, label="baseline"
                    )
                except Exception:
                    y_base = None

            # =====================================================
            # PEAKS + BEST FIT (SAFE)
            # =====================================================
            y_peaks = None
            if hasattr(spectrum, "peak_models") and spectrum.peak_models:
                y_peaks = 0.0
                for peak_model in spectrum.peak_models:
                    try:
                        params = peak_model.make_params()
                        y_peak = peak_model.eval(params, x=x)
                        y_peaks += y_peak
                        self.ax.plot(x, y_peak, lw=lw, alpha=0.8)
                    except Exception:
                        pass

            # ── Best-fit = baseline + peaks
            if y_peaks is not None:
                if y_base is not None:
                    y_fit = y_base + y_peaks
                else:
                    y_fit = y_peaks

                self.ax.plot(
                    x, y_fit, lw=lw, color="black", label="bestfit"
                )

        # ── Legend / grid / axes
        if self.btn_legend.isChecked():
            legend = self.ax.legend(loc="best")
            self._make_legend_pickable(legend)

        if self.act_grid.isChecked():
            self.ax.grid(True, linestyle="--", alpha=0.4)

        self.ax.set_xlabel(self.cbb_xaxis.currentText())
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_yscale(
            "log" if self.cbb_yscale.currentText() == "Log" else "linear"
        )

        # Restore zoom
        if not is_default_limits:
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
        """Make legend texts and handles pickable for interaction."""
        for text in legend.get_texts():
            text.set_picker(True)

        for handle in legend.legendHandles:
            handle.set_picker(True)

        # Cache bbox to avoid conflicts with plot clicks
        self._legend_bbox = legend.get_window_extent(self.canvas.renderer)

        # Connect pick event once
        if not hasattr(self, "_legend_pick_connected"):
            self.canvas.mpl_connect("pick_event", self._on_legend_pick)
            self._legend_pick_connected = True


    def set_r2(self, value):
        self.lbl_r2.setText(f"R²={value:.4f}")

    def _on_legend_pick(self, event):
        artist = event.artist

        # Legend TEXT → rename spectrum
        if isinstance(artist, mtext.Text):
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

        # Legend LINE → change color
        elif isinstance(artist, mlines.Line2D):
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
    def _emit_view_options(self):
        def _to_float(text, default):
            try:
                return float(text)
            except Exception:
                return default

        self.viewOptionsChanged.emit({
            "legend": self.btn_legend.isChecked(),
            "grid": self.act_grid.isChecked(),
            "colors": self.act_colors.isChecked(),
            "raw": self.act_raw.isChecked(),
            "bestfit": self.act_bestfit.isChecked(),
            "residual": self.act_residual.isChecked(),
            "x_unit": self.cbb_xaxis.currentText(),
            "y_scale": self.cbb_yscale.currentText(),
            "linewidth": self.spin_lw.value(),
            "copy_width": _to_float(self.width_entry.text(), 5.5),
            "copy_height": _to_float(self.height_entry.text(), 4.0),
        })
        # Update plot immediately whenever an view option changes
        self._redraw() 


    def _emit_copy(self):
        from PySide6.QtWidgets import QApplication
        ctrl = QApplication.keyboardModifiers() & Qt.ControlModifier
        self.copyRequested.emit(bool(ctrl))

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
        if getattr(self, "legend_bbox", None) is not None:
            if self.legend_bbox.contains(event.x, event.y):
                return

        x = event.xdata
        y = event.ydata

        # ─── Peak tool ─────────────────────────
        if self.btn_peak.isChecked():
            if event.button == 1:      # left click
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