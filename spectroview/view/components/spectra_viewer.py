# view/components/spectra_viewer.py

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QToolButton, QLabel,
    QComboBox, QMenu, QWidgetAction,
    QLineEdit, QDoubleSpinBox
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT
)

from spectroview import ICON_DIR, X_AXIS_UNIT, PLOT_POLICY


class SpectraViewer(QWidget):
    mouseClicked = Signal(float, float, int)
    zoomToggled = Signal(bool)
    rescaleRequested = Signal()
    viewOptionsChanged = Signal(dict)
    copyRequested = Signal(bool)   # ctrl_pressed
    toolModeChanged = Signal(str)  # zoom / baseline / peak
    normalizationChanged = Signal(bool, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        
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
        for action in self.toolbar.actions():
            if action.text() in ['Home', 'Save', 'Pan', 'Back', 'Forward', 'Subplots', 'Zoom']:
                action.setVisible(False)

        # ─── Control bar ───
        self.control_bar = self._create_control_bar()

        
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(self.control_bar)

    # ─────────────────────────────────────────
    # Control bar
    # ─────────────────────────────────────────
    def _create_control_bar(self):
        bar = QWidget(self)
        bar.setFixedHeight(50)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(5, 5, 5, 5)

        # Rescale
        self.btn_rescale = QPushButton()
        self.btn_rescale.setIcon(QIcon(f"{ICON_DIR}/rescale.png"))
        self.btn_rescale.setIconSize(QSize(24, 24))
        self.btn_rescale.clicked.connect(self.rescaleRequested)
        layout.addWidget(self.btn_rescale)

        # Tool buttons
        self.btn_zoom = self._tool_btn("zoom.png", "Zoom", True)
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
        self.btn_norm.setIconSize(QSize(24, 24))
        self.btn_norm.toggled.connect(self._emit_norm)
        layout.addWidget(self.btn_norm)

        self.norm_xmin = QLineEdit()
        self.norm_xmin.setFixedWidth(45)
        self.norm_xmin.setPlaceholderText("Xmin")
        layout.addWidget(self.norm_xmin)

        self.norm_xmax = QLineEdit()
        self.norm_xmax.setFixedWidth(45)
        self.norm_xmax.setPlaceholderText("Xmax")
        layout.addWidget(self.norm_xmax)

        # Legend
        self.btn_legend = QToolButton()
        self.btn_legend.setCheckable(True)
        self.btn_legend.setIcon(QIcon(f"{ICON_DIR}/legend.png"))
        self.btn_legend.setIconSize(QSize(24, 24))
        self.btn_legend.toggled.connect(self._emit_view_options)
        layout.addWidget(self.btn_legend)

        # Copy
        self.btn_copy = QPushButton()
        self.btn_copy.setIcon(QIcon(f"{ICON_DIR}/copy.png"))
        self.btn_copy.setIconSize(QSize(24, 24))
        self.btn_copy.clicked.connect(self._emit_copy)
        layout.addWidget(self.btn_copy)

        # Options
        self.options_menu = self._create_options_menu()
        self.btn_options = QToolButton()
        self.btn_options.setIcon(QIcon(f"{ICON_DIR}/options.png"))
        self.btn_options.setIconSize(QSize(24, 24))
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
        btn.setIconSize(QSize(24, 24))
        btn.setToolTip(tooltip)
        return btn

    # ─────────────────────────────────────────
    # Options menu
    # ─────────────────────────────────────────
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
        self._add_check(menu, "Colors")
        self._add_check(menu, "Peaks")
        self._add_check(menu, "Raw")
        self._add_check(menu, "Bestfit", True)
        self._add_check(menu, "Residual")
        self._add_check(menu, "Grid")

        menu.addSeparator()

        # Line width
        self.spin_lw = QDoubleSpinBox()
        self.spin_lw.setRange(0.1, 5)
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

    def _add_check(self, menu, name, checked=False):
        act = menu.addAction(name)
        act.setCheckable(True)
        act.setChecked(checked)
        act.toggled.connect(self._emit_view_options)
        setattr(self, f"act_{name.lower()}", act)

    def _wrap(self, label, widget):
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
        self.ax.clear()
        for line in lines:
            self.ax.plot(**line)

        if self.btn_legend.isChecked():
            self.ax.legend()

        if self.act_grid.isChecked():
            self.ax.grid(True, linestyle="--", alpha=0.4)

        self.ax.set_xlabel(self.cbb_xaxis.currentText())
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_yscale("log" if self.cbb_yscale.currentText() == "Log" else "linear")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def set_r2(self, value):
        self.lbl_r2.setText(f"R²={value:.4f}")

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
        if event.inaxes == self.ax:
            self.mouseClicked.emit(event.xdata, event.ydata, event.button)
