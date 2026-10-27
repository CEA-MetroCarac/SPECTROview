# view/components/spectra_viewer.py
import re
import warnings
from typing import Optional, Tuple
import numpy as np
from PySide6.QtCore import QSettings
from spectroview.fit_engine.noise import detect_noise_level
from spectroview.fit_engine.evaluator import eval_peak_initial
# Suppress harmless Matplotlib constrained_layout warning on 0-size UI initialization
warnings.filterwarnings("ignore", message=".*constrained_layout not applied because axes sizes collapsed to zero.*")

from PySide6.QtWidgets import (
    QFrame, QGroupBox,
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QToolButton, QLabel,
    QComboBox, QDialog, QMenu, QWidgetAction,
    QLineEdit, QDoubleSpinBox, QSpinBox,
    QSlider, QMessageBox, QApplication, QButtonGroup
)
from PySide6.QtCore import QObject, QEvent, Qt, Signal, QSize, QTimer, QPoint
from PySide6.QtGui import QCursor

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.collections as mcoll
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas
)
import matplotlib as mpl

from functools import lru_cache
import matplotlib.lines as mlines

from spectroview import X_AXIS_UNIT, Y_AXIS_UNIT, PLOT_POLICY_LIGHT, PLOT_POLICY_DARK, DEFAULT_COLORS
from spectroview.view.theme import Icons, current_tokens
from spectroview.viewmodel.utils import copy_fig_to_clb, fano_display_amplitude

@lru_cache(maxsize=8)
def get_cached_style_dict(style_path: str) -> dict:
    """Cache parsed mplstyle files to prevent disk re-reading on every plot/theme switch."""
    return dict(mpl.rc_params_from_file(style_path))

# Precompiled regex for numeric tokens in filenames
_RE_NUMBERS = re.compile(r"\d+\.?\d*")
from spectroview.view.components.customized_widgets import (
    CustomizedPalette,
    NoDoubleClickZoomToolbar,
)
from spectroview.view.components.v_spectra_legend_editor import (
    SpectraLegendEditorDialog,
)
from spectroview.view.components.v_range_selector import (
    VRangeSelectorPanel,
    RangeSelectorManager,
    SELECTOR_RED,
    SELECTOR_BLUE,
    SELECTOR_GREEN,
)


SPECTRA_DISCRETE_PALETTES = (
    "DEFAULT_COLORS",
    "tab20", "tab20b", "tab20c", "Dark2", "Paired", "Accent",
)
SPECTRA_GRADIENT_PALETTES = (
    "viridis", "plasma", "jet", "cividis", "magma",
)
SPECTRA_COLOR_PALETTES = (
    SPECTRA_DISCRETE_PALETTES + SPECTRA_GRADIENT_PALETTES
)
SPECTRA_CUSTOM_PALETTES = {
    "DEFAULT_COLORS": DEFAULT_COLORS,
}

# A selection can contain thousands of spectra.  The plot itself is batched,
# but constructing thousands of per-row Qt editors would be costly in either
# the Spectra or Maps workspace.
LEGEND_EDITOR_MAX_ROWS = 100


class _MockPeakModelObj:
    """Lightweight stand-in for tensor-mode plotting to carry a shape name."""
    __slots__ = ('name2',)
    def __init__(self, name):
        self.name2 = name


class VSpectraViewer(QWidget):
    # ───── View → ViewModel signals ─────
    copy_data_requested = Signal()  # Request ViewModel to copy spectrum data
    plotStyleChanged = Signal()
    allOptionsSyncChanged = Signal(dict)

    peak_add_requested = Signal(float)
    peak_remove_requested = Signal(float)
    baseline_add_requested = Signal(float, float)
    baseline_remove_requested = Signal(float)
    peak_dragged = Signal(float, float)
    peak_drag_finished = Signal()
    spectrumCustomized = Signal()

    # ── Cosmic-ray eraser signals ──
    cosmicRayErased = Signal(int, object)  # (spec_idx, modified_y_array) — live preview
    cosmicRayValidated = Signal()          # User accepted all erasures → persist to model

    # ── Spectral Range Selector signals ──
    range_toggled = Signal(str, bool)          # (color_id, is_active)
    range_changed = Signal(str, float, float)  # (color_id, x_min, x_max)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._current_spectra = []
        self._tensor_data = None
        
        self._fitted_lines = []     # [(line, peak_info)]
        self._highlighted_line = None
        self._dragging_peak = None  # Stores (line, info) when dragging

        self.zoom_pan_active = True
        app_inst = QApplication.instance()
        if app_inst is not None:
            app_inst.focusChanged.connect(lambda old, now: self._hide_tooltip())

        # ── Heatmap mode state ──
        self._heatmap_mode = False
        self._heatmap_colorbar = None      # Reference to colorbar for removal

        # ── Cosmic-ray eraser state ──
        self._erase_mode = False           # Whether eraser mode is active
        self._erase_drag_start_x = None    # x-coordinate where drag began
        self._erase_highlight = None       # Matplotlib axvspan for visual feedback
        self._erase_history = []           # Stack of (spec_idx, y_snapshot) for undo
        self._erase_cids = []              # Matplotlib event connection IDs

        # ── Spectral Range Selector state & manager ──
        self.range_manager = RangeSelectorManager(self)
        self.range_manager.range_toggled.connect(self.range_toggled.emit)
        self.range_manager.range_changed.connect(self.range_changed.emit)
        self._range_drag_cids = []
        self._range_hover_active = False
        self._ranges_initialized = False
        
    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 2, 2)

        # ─── Figure ───
        with plt.style.context(PLOT_POLICY_LIGHT):
            self.figure = Figure(layout="compressed", dpi=80)
            self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        
        self.canvas.mpl_connect("button_press_event", self._on_mouse_click)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)


        self.toolbar = NoDoubleClickZoomToolbar(self.canvas, self)
        
        original_set_style = self.toolbar.setStyleSheet
        def custom_set_style(css):
            original_set_style("background: transparent; border: none;")
        self.toolbar.setStyleSheet = custom_set_style
        self.toolbar.setStyleSheet("")

        class ToolbarEventFilter(QObject):
            def __init__(self, toolbar):
                # Parented to `toolbar` so Qt cancels the pending single-shot
                # timer on teardown -- unparented, it could fire 100ms later
                # against an already-deleted toolbar.
                super().__init__(toolbar)
                self.toolbar = toolbar
                self._timer = QTimer(self)
                self._timer.setSingleShot(True)
                self._timer.setInterval(100)
                self._timer.timeout.connect(self._update_icons)

            def eventFilter(self, obj, event):
                if event.type() == QEvent.PaletteChange:
                    self._timer.start()
                return False

            def _update_icons(self):
                try:
                    action_dict = {action.text(): action for action in self.toolbar.actions() if action.text()}
                except RuntimeError:
                    return  # toolbar's C++ object already deleted
                for text, tooltip_text, image_file, name_of_method in self.toolbar.toolitems:
                    if text in action_dict and image_file is not None:
                        try:
                            icon = self.toolbar._icon(image_file + '.png')
                            action_dict[text].setIcon(icon)
                        except Exception:
                            pass
                
        self.toolbar_filter = ToolbarEventFilter(self.toolbar)
        self.toolbar.installEventFilter(self.toolbar_filter)

        self.toolbar.zoom() # Start with zoom enabled
        for action in self.toolbar.actions():
            if action.text() in ['Home', 'Save', 'Pan', 'Back', 'Forward', 'Subplots', 'Zoom']:
                action.setVisible(False)

        # ─── Vertical right panel (eraser buttons + shift sliders) ───
        self.shift_panel = self._create_vertical_right_panel()

        # ─── Canvas + Sliders row ───
        canvas_row = QHBoxLayout()
        canvas_row.setContentsMargins(0, 0, 0, 0)
        self.canvas_card = QFrame(self)
        self.canvas_card.setObjectName("canvasCard")
        self.canvas_card.setFrameShape(QFrame.NoFrame)
        canvas_card_layout = QVBoxLayout(self.canvas_card)
        canvas_card_layout.setContentsMargins(5, 5, 5, 5)
        canvas_card_layout.setSpacing(0)
        canvas_card_layout.addWidget(self.canvas)
        canvas_row.addWidget(self.canvas_card, stretch=1)
        canvas_row.addWidget(self.shift_panel)

        # ─── Control bar ───
        self.control_bar = self._create_control_bar()
        
        main_layout.addLayout(canvas_row)
        main_layout.addWidget(self.control_bar)

    # ─── Vertical right panel (eraser + shift sliders) ───
    def _create_vertical_right_panel(self):
        """Create a vertical panel with eraser tool buttons and Y/X-shift sliders."""
        panel = QGroupBox()

        panel.setFixedWidth(32)
        panel.setObjectName("shiftPanel")
       
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # ── Eraser button ──
        self.btn_eraser = QToolButton()
        self.btn_eraser.setCheckable(True)
        self.btn_eraser.setIcon(Icons.eraser())
        self.btn_eraser.setIconSize(QSize(18, 18))
        self.btn_eraser.setFixedSize(22, 22)
        self.btn_eraser.setToolTip("Erase Cosmic Ray - click and drag over a spike to remove it")
        self.btn_eraser.toggled.connect(self._toggle_erase_mode)
        layout.addWidget(self.btn_eraser, 0, Qt.AlignHCenter)

        # ── Undo button (hidden until eraser is active) ──
        self.btn_erase_undo = QToolButton()
        self.btn_erase_undo.setIcon(Icons.undo())
        self.btn_erase_undo.setIconSize(QSize(18, 18))
        self.btn_erase_undo.setFixedSize(22, 22)
        self.btn_erase_undo.setToolTip("Undo last erasure")
        self.btn_erase_undo.clicked.connect(self._erase_undo)
        self.btn_erase_undo.setVisible(False)
        layout.addWidget(self.btn_erase_undo, 0, Qt.AlignHCenter)

        # ── Validate button (hidden until eraser is active) ──
        self.btn_erase_validate = QToolButton()
        self.btn_erase_validate.setIcon(Icons.check())
        self.btn_erase_validate.setIconSize(QSize(18, 18))
        self.btn_erase_validate.setFixedSize(22, 22)
        self.btn_erase_validate.setToolTip("Validate erasures and save corrections")
        self.btn_erase_validate.clicked.connect(self._erase_validate)
        self.btn_erase_validate.setVisible(False)
        layout.addWidget(self.btn_erase_validate, 0, Qt.AlignHCenter)

        # ── Spectral Range Selector Panel (Red, Blue, Green) ──
        self.range_panel = VRangeSelectorPanel(panel)
        self.range_panel.selector_toggled.connect(self._toggle_range_selector)
        self.btn_range_red = self.range_panel.buttons[SELECTOR_RED]
        self.btn_range_blue = self.range_panel.buttons[SELECTOR_BLUE]
        self.btn_range_green = self.range_panel.buttons[SELECTOR_GREEN]
        self.range_panel.setVisible(False)  # Hidden by default; active only in Maps workspace
        layout.addWidget(self.range_panel, 0, Qt.AlignHCenter)

        layout.addStretch(1)
        
        # ── Y Shift slider ──
        lbl_y = QLabel("Y")
        lbl_y.setAlignment(Qt.AlignCenter)
        lbl_y.setToolTip("Shift spectra vertically (waterfall offset)")

        self.slider_y_shift = QSlider(Qt.Vertical)
        self.slider_y_shift.setFixedHeight(120)
        self.slider_y_shift.setRange(0, 100)
        self.slider_y_shift.setValue(0)
        self.slider_y_shift.setToolTip("Y Shift: offset each spectrum vertically")
        self.slider_y_shift.valueChanged.connect(self._plot)

        layout.addWidget(lbl_y, 0, Qt.AlignHCenter)
        layout.addWidget(self.slider_y_shift, 0, Qt.AlignHCenter)
        
        layout.addSpacing(15)

        # ── X Shift slider ──
        lbl_x = QLabel("X")
        lbl_x.setAlignment(Qt.AlignCenter)
        lbl_x.setToolTip("Shift spectra horizontally")

        self.slider_x_shift = QSlider(Qt.Vertical)
        self.slider_x_shift.setFixedHeight(120)
        self.slider_x_shift.setRange(0, 100)
        self.slider_x_shift.setValue(0)
        self.slider_x_shift.setToolTip("X Shift: offset each spectrum horizontally")
        self.slider_x_shift.valueChanged.connect(self._plot)

        layout.addWidget(lbl_x, 0, Qt.AlignHCenter)
        layout.addWidget(self.slider_x_shift, 0, Qt.AlignHCenter)
        
        # Removed bottom stretch to stick sliders to the bottom
        return panel

    # Control bar
    def _create_control_bar(self):
        bar = QGroupBox(self)

        bar.setObjectName("controlBarPanel")
        bar.setFixedHeight(38)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(2, 2, 2, 2)

        # Rescale
        self.btn_rescale = QPushButton()
        self.btn_rescale.setIcon(Icons.rescale())
        self.btn_rescale.setIconSize(QSize(22, 22))
        self.btn_rescale.setFixedSize(30, 30)
        self.btn_rescale.setToolTip("Rescale (Ctrl+R)")
        self.btn_rescale.clicked.connect(self._rescale)
        
        layout.addWidget(self.btn_rescale)

        # Tool buttons
        self.btn_zoom = self._tool_btn("zoom.png", "Zoom", True)
        self.btn_baseline = self._tool_btn("baseline.png", "Baseline")
        self.btn_peak = self._tool_btn("peak.png", "Peak")

        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(False)
        for btn in (self.btn_zoom, self.btn_baseline, self.btn_peak):
            self._tool_group.addButton(btn)
            layout.addWidget(btn)
        self._tool_group.buttonClicked.connect(self._on_tool_btn_clicked)
        self.btn_zoom.toggled.connect(self._toggle_zoom_pan)

        layout.addSpacing(15)

        # Normalization
        self.btn_norm = QToolButton()
        self.btn_norm.setCheckable(True)
        self.btn_norm.setObjectName("btn_norm")
        self._update_norm_button_style()
        self.btn_norm.setIcon(Icons.norm())
        self.btn_norm.setToolTip("Intensity normalization to maximum of spectrum or selected region.\nNote: This normalization is just for visual change, the actual data is not changed.")
        self.btn_norm.setIconSize(QSize(22, 22))
        self.btn_norm.setFixedSize(30, 30)
        self.btn_norm.toggled.connect(self._plot)
        self.btn_norm.clicked.connect(self._rescale)

        layout.addWidget(self.btn_norm)

        self.norm_xmin = QLineEdit()
        self.norm_xmin.setFixedSize(50, 30)
        self.norm_xmin.setPlaceholderText("Xmin")
        self.norm_xmin.setToolTip("Intensity normalization to maximum of spectrum or selected region")
        layout.addWidget(self.norm_xmin)

        self.norm_xmax = QLineEdit()
        self.norm_xmax.setFixedSize(50, 30)
        self.norm_xmax.setPlaceholderText("Xmax")
        self.norm_xmax.setToolTip("Intensity normalization to maximum of spectrum or selected region")
        layout.addWidget(self.norm_xmax)

        self.norm_xmin.editingFinished.connect(self._on_norm_range_edited)
        self.norm_xmax.editingFinished.connect(self._on_norm_range_edited)

        # Bestfit
        self.btn_bestfit = QToolButton()
        self.btn_bestfit.setCheckable(True)
        self.btn_bestfit.setChecked(True)  # Checked by default
        self.btn_bestfit.setIcon(Icons.bestfit())
        self.btn_bestfit.setToolTip("Show or hide best-fit lines")
        self.btn_bestfit.setIconSize(QSize(22, 22))
        self.btn_bestfit.setFixedSize(30, 30)
        self.btn_bestfit.toggled.connect(self._emit_view_options)
        layout.addWidget(self.btn_bestfit)


        # Legend
        self.btn_legend = QToolButton()
        self.btn_legend.setCheckable(True)
        self.btn_legend.setIcon(Icons.legend())
        self.btn_legend.setToolTip(
            "Show or hide the legend box. Double-click the legend to edit "
            "all spectrum labels and colors.")
        self.btn_legend.setIconSize(QSize(22, 22))
        self.btn_legend.setFixedSize(30, 30)
        self.btn_legend.toggled.connect(self._emit_view_options)
        layout.addWidget(self.btn_legend)

        # Heatmap toggle
        self.btn_heatmap = QToolButton()
        self.btn_heatmap.setCheckable(True)
        self.btn_heatmap.setIcon(Icons.map_2d())
        self.btn_heatmap.setToolTip(
            "Plot Heatmap — display multiple spectra as a 2D intensity map.\n"
            "X-axis: spectral axis, Y-axis: varying parameter (see options), "
            "Color: intensity.")
        self.btn_heatmap.setIconSize(QSize(22, 22))
        self.btn_heatmap.setFixedSize(30, 30)
        self.btn_heatmap.toggled.connect(self._toggle_heatmap_mode)
        layout.addWidget(self.btn_heatmap)

        # Heatmap controls (placed on toolbar next to btn_heatmap, hidden until heatmap mode is active)
        self.cbb_heatmap_yaxis = QComboBox()
        self.cbb_heatmap_yaxis.setFixedHeight(30)
        self.cbb_heatmap_yaxis.addItem("Index")
        self.cbb_heatmap_yaxis.setToolTip(
            "Heatmap Y-axis: choose parameter for vertical axis.\n"
            "Index = spectrum order, or select a token parsed from filenames.")
        self.cbb_heatmap_yaxis.currentIndexChanged.connect(self._emit_view_options)
        self.cbb_heatmap_yaxis.setVisible(False)
        layout.addWidget(self.cbb_heatmap_yaxis)

        self.le_heatmap_ylabel = QLineEdit()
        self.le_heatmap_ylabel.setFixedHeight(30)
        self.le_heatmap_ylabel.setFixedWidth(80)
        self.le_heatmap_ylabel.setPlaceholderText("Y label")
        self.le_heatmap_ylabel.setToolTip(
            "Custom label for the heatmap Y-axis.\n"
            "Leave empty for the automatic token name.\n"
            "Example: Temperature (°C)")
        self.le_heatmap_ylabel.textChanged.connect(self._on_heatmap_ylabel_changed)
        self.le_heatmap_ylabel.setVisible(False)
        layout.addWidget(self.le_heatmap_ylabel)

        self.spin_heatmap_interp = QSpinBox()
        self.spin_heatmap_interp.setFixedHeight(30)
        self.spin_heatmap_interp.setFixedWidth(85)
        self.spin_heatmap_interp.setRange(0, 2000)
        self.spin_heatmap_interp.setValue(300)
        self.spin_heatmap_interp.setPrefix("Pts: ")
        self.spin_heatmap_interp.setToolTip(
            "Number of interpolation points along the Y-axis for smooth\n"
            "rendering. Set to 0 to disable interpolation (raw rows only).")
        self.spin_heatmap_interp.valueChanged.connect(self._emit_view_options)
        self.spin_heatmap_interp.setVisible(False)
        layout.addWidget(self.spin_heatmap_interp)

        layout.addStretch()

        # Matplotlib toolbar & stats
        layout.addWidget(self.toolbar)
        # R²
        self.lbl_r2 = QLabel("R²=0")
        layout.addWidget(self.lbl_r2)
        # Noise level
        self.lbl_noise = QLabel("Noise=0")
        layout.addWidget(self.lbl_noise)

        # Vertical separator between navigation/coordinates and options/copy actions
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # Very right side: More view options and Copy to clipboard
        self.options_menu = self._create_options_menu()
        self.btn_options = QToolButton()
        self.btn_options.setIcon(Icons.settings())
        self.btn_options.setToolTip("More view options")
        self.btn_options.setIconSize(QSize(22, 22))
        self.btn_options.setFixedSize(30, 30)
        self.btn_options.setPopupMode(QToolButton.InstantPopup)
        self.btn_options.setMenu(self.options_menu)
        layout.addWidget(self.btn_options)

        self.btn_copy = QPushButton()
        self.btn_copy.setIcon(Icons.copy())
        self.btn_copy.setToolTip("Copy figure to clipboard. Hold Ctrl + Click to copy data to clipboard")
        self.btn_copy.setIconSize(QSize(22, 22))
        self.btn_copy.setFixedSize(30, 30)
        self.btn_copy.clicked.connect(self._emit_copy)
        layout.addWidget(self.btn_copy)

        return bar

    def _tool_btn(self, icon, tooltip, checked=False):
        btn = QToolButton()
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setAutoExclusive(False)
        btn.setIcon(Icons.get(icon))
        btn.setIconSize(QSize(22, 22))
        btn.setFixedSize(30, 30)
        btn.setToolTip(tooltip)
        return btn

    def _on_tool_btn_clicked(self, clicked_btn):
        """Handle mutual exclusivity across toolbar tools while allowing toggling off."""
        if clicked_btn.isChecked():
            for b in (self.btn_zoom, self.btn_baseline, self.btn_peak):
                if b is not clicked_btn and b.isChecked():
                    b.blockSignals(True)
                    b.setChecked(False)
                    b.blockSignals(False)
        if clicked_btn is self.btn_zoom:
            self._set_zoom_tool_active(clicked_btn.isChecked())
        else:
            if self.btn_zoom.isChecked():
                self.btn_zoom.blockSignals(True)
                self.btn_zoom.setChecked(False)
                self.btn_zoom.blockSignals(False)
            self._set_zoom_tool_active(False)

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

        # Spectrum color palette. Qualitative palettes cycle through distinct
        # colors; sequential/rainbow palettes are sampled across all selected
        # spectra so ordered series (time, temperature, etc.) form a gradient.
        self.cbb_color_palette = CustomizedPalette(
            palette_list=SPECTRA_COLOR_PALETTES,
            custom_palettes=SPECTRA_CUSTOM_PALETTES,
        )
        self.cbb_color_palette.setCurrentText("DEFAULT_COLORS")
        self.cbb_color_palette.setToolTip(
            "Choose colors for the selected spectrum series. Custom colors "
            "set from the legend continue to take priority."
        )
        self.cbb_color_palette.currentIndexChanged.connect(
            self._emit_view_options)
        menu.addAction(self._wrap("Color palette:", self.cbb_color_palette))

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

        # Max legend items / heavy overlays
        self.spin_max_legend_items = QSpinBox()
        self.spin_max_legend_items.setRange(1, 1000)
        self.spin_max_legend_items.setValue(15)
        self.spin_max_legend_items.valueChanged.connect(self._emit_view_options)
        menu.addAction(self._wrap(
            "Max legend items:", self.spin_max_legend_items))

        # Move legend box outside
        self.act_legend_outside = menu.addAction("Move legend box outside")
        self.act_legend_outside.setCheckable(True)
        self.act_legend_outside.setChecked(False)
        self.act_legend_outside.toggled.connect(self._emit_view_options)

        # Backwards-compatible alias for internal/external code that used the
        # old name when this control was first introduced.
        self.spin_max_overlays = self.spin_max_legend_items

        menu.addSeparator()
        
        # ─── Copied figure size (NEW) ───
        ratio_widget = QWidget()
        ratio_layout = QHBoxLayout(ratio_widget)
        ratio_layout.setContentsMargins(2, 2, 2, 2)

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
            x_arr = data.get("x")
            if x_arr is not None:
                first_x = x_arr[0] if isinstance(x_arr, list) and len(x_arr) > 0 else x_arr
                if hasattr(first_x, "__len__") and len(first_x) > 1:
                    if not getattr(self, "_ranges_initialized", False):
                        self.range_manager.initialize_ranges_from_axis(first_x)
                        self._ranges_initialized = True
                    else:
                        self.range_manager.refresh_or_clamp_ranges(float(np.min(first_x)), float(np.max(first_x)))
        else:
            self._tensor_data = None
            self._current_spectra = []
            
        # Keep heatmap Y-axis choices up-to-date with the current selection
        if getattr(self, "_heatmap_mode", False):
            self._refresh_heatmap_yaxis_choices()
            
        self._plot()

    def _on_norm_range_edited(self):
        """Unified handler when norm bounds change: plot and rescale once."""
        self._plot()
        self._rescale()

    def _compute_shift_steps(self, Y_norm=None):
        """Compute per-spectrum X and Y shift steps based on slider values and data range."""
        y_slider = self.slider_y_shift.value()  # 0–100
        x_slider = self.slider_x_shift.value()  # 0–100

        if y_slider == 0 and x_slider == 0:
            return 0.0, 0.0

        if not self._tensor_data or len(self._tensor_data["y"]) == 0:
            return 0.0, 0.0

        x = self._tensor_data["x"]
        if Y_norm is None:
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
                Y_norm.append(self._get_normalized_y(xi, Y[i], xmin=xmin, xmax=xmax))
            return Y_norm
        else:
            if xmin is not None and xmax is not None:
                if xmin > xmax:
                    xmin, xmax = xmax, xmin

                imin = np.abs(x - xmin).argmin()
                imax = np.abs(x - xmax).argmin()
                
                idx_start = min(imin, imax)
                idx_end = max(imin, imax) + 1
                if idx_end > idx_start:
                    sub_y = Y[:, idx_start:idx_end]
                    norm_vals = np.max(sub_y, axis=1, keepdims=True)
                else:
                    norm_vals = np.max(Y, axis=1, keepdims=True)
            else:
                norm_vals = np.max(Y, axis=1, keepdims=True)

            # Avoid div by zero
            norm_vals = np.where(norm_vals == 0, 1.0, norm_vals)
            return Y / norm_vals

    def _plot(self):
        style_name = self.cbb_theme.currentText()
        if style_name == "Dark Mode":
            style_path = PLOT_POLICY_DARK
        else:
            style_path = PLOT_POLICY_LIGHT
        with plt.style.context(style_path):
            self._plot_internal()

    def _plot_internal(self):
        if not self._tensor_data:
            self.ax.clear()
            self._remove_heatmap_colorbar()
            self.lbl_r2.setText("R²=0")
            self.canvas.draw_idle()
            return

        # ── Heatmap branch ──
        if self._heatmap_mode:
            self._plot_heatmap()
            return

        # Save current zoom/pan limits so they survive a full ax.clear()
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        is_default = (xlim == (0.0, 1.0) and ylim == (0.0, 1.0))

        fg_color = plt.rcParams.get('axes.labelcolor', 'black')

        self.ax.clear()
        self._fitted_lines.clear()

        self._update_r2_display()

        x = self._tensor_data["x"]
        Y = self._tensor_data["y"]
        Y_norm = self._get_normalized_y_tensor(x, Y)
        x_shift_step, y_shift_step = self._compute_shift_steps(Y_norm=Y_norm)
        plot_style = self.cbb_plotstyle.currentText()
        lw = self.spin_lw.value()
        dot_size = self.spin_dotsize.value()
        colors_cycle = self._get_colors_cycle(len(Y))

        # ── Step 1: Build bulk segment data (main spectra + raw overlay) ──
        segments = self._build_tensor_segments(
            x_shift_step, y_shift_step, plot_style, lw, fg_color,
            colors_cycle, Y_norm=Y_norm)

        # ── Step 2: Draw heavy per-spectrum overlays ──
        self._draw_tensor_overlays(x_shift_step, y_shift_step, lw,
                                   fg_color)

        # ── Step 3: Commit everything to the canvas ──
        self._finalize_plot(segments, plot_style, lw, dot_size, fg_color,
                            xlim, ylim, is_default)

    # ─────────────────────────────────────────────
    # Plot helpers: segment builders
    # ─────────────────────────────────────────────

    def _get_colors_for_palette(self, palette_name, count=None):
        """Return ``count`` colors sampled from a named spectrum palette.

        Discrete palettes repeat only after all of their distinct colors have
        been used. Gradient palettes are sampled evenly from end to end, which
        makes the spectrum order visible for time/temperature series.
        """
        requested_count = (
            len(DEFAULT_COLORS) if count is None else max(0, int(count))
        )

        if palette_name in SPECTRA_CUSTOM_PALETTES:
            base_colors = list(SPECTRA_CUSTOM_PALETTES[palette_name])
        else:
            try:
                cmap = mpl.colormaps[palette_name]
            except (KeyError, AttributeError):
                try:
                    cmap = mpl.cm.get_cmap(palette_name)
                except (ValueError, KeyError):
                    base_colors = list(DEFAULT_COLORS)
                    cmap = None

            if palette_name in SPECTRA_DISCRETE_PALETTES and cmap is not None:
                cmap_colors = getattr(cmap, "colors", None)
                if cmap_colors is None:
                    cmap_colors = cmap(np.linspace(0.0, 1.0, cmap.N))
                base_colors = [mpl.colors.to_hex(color)
                               for color in cmap_colors]
            elif cmap is not None:
                if requested_count == 0:
                    return []
                positions = (np.array([0.5]) if requested_count == 1
                             else np.linspace(0.0, 1.0, requested_count))
                return [mpl.colors.to_hex(cmap(position))
                        for position in positions]

        if not base_colors:
            base_colors = list(DEFAULT_COLORS) or ["#1f77b4"]
        return [base_colors[i % len(base_colors)]
                for i in range(requested_count)]

    def _get_colors_cycle(self, count=None):
        """Return colors sampled from the currently selected palette."""
        palette_name = (
            self.cbb_color_palette.currentText()
            if hasattr(self, "cbb_color_palette") else "DEFAULT_COLORS"
        )
        return self._get_colors_for_palette(palette_name, count)

    def _get_palette_color_choices(self, palette_name, spectrum_count=0):
        """Return the selectable swatches for a palette in the legend editor."""
        if palette_name in SPECTRA_CUSTOM_PALETTES:
            return list(SPECTRA_CUSTOM_PALETTES[palette_name])

        try:
            cmap = mpl.colormaps[palette_name]
        except (KeyError, AttributeError):
            return list(DEFAULT_COLORS)

        if palette_name in SPECTRA_DISCRETE_PALETTES:
            cmap_colors = getattr(cmap, "colors", None)
            if cmap_colors is None:
                cmap_colors = cmap(np.linspace(0.0, 1.0, cmap.N))
            return [mpl.colors.to_hex(color) for color in cmap_colors]

        # A continuous gradient has infinitely many colors. Offer a useful,
        # bounded set of evenly spaced swatches without making every table
        # combobox excessively large for maps containing hundreds of spectra.
        swatch_count = max(20, min(32, max(0, int(spectrum_count))))
        return [
            mpl.colors.to_hex(cmap(position))
            for position in np.linspace(0.0, 1.0, swatch_count)
        ]

    def _build_tensor_segments(self, x_shift_step, y_shift_step,
                                plot_style, lw, fg_color, colors_cycle,
                                Y_norm=None):
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
        if Y_norm is None:
            Y_norm = self._get_normalized_y_tensor(x, Y) 

        # Resolve each spectrum's color: use stored color if available, else cycle
        t_colors = self._tensor_data.get("colors")
        if t_colors is None:
            t_colors = []
        main_colors = [
            t_colors[i]
            if i < len(t_colors) and t_colors[i]
            else colors_cycle[i]
            for i in range(N)
        ]

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
        # Capped by spin_max_overlays to avoid an unmanageable legend for large maps.
        t_labels = self._tensor_data.get("labels", [])
        t_fnames = self._tensor_data.get("fnames", [])
        proxies = self._tensor_data.get("proxies", [])  # Spectrum objects for interactive tooltip
        for spec_idx in range(min(N, self.spin_max_legend_items.value())):
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

        for spec_idx in range(
                min(n_specs, self.spin_max_legend_items.value())):
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
                                 "blue", lw=1.0, label="residual")

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
            # Ensure we don't downsample high-res spectra by using at least 3x the original points
            n_fine_pts = max(3000, len(x) * 5)
            x_fine = np.linspace(x.min(), x.max(), n_fine_pts)
            sorted_keys = sorted(fit_model["peak_models"].keys(),
                                 key=lambda k: int(k))

            y_fit_fine = np.zeros_like(x_fine)
            for i, k in enumerate(sorted_keys):
                p_model = fit_model["peak_models"][k]
                y_peak_fine = eval_peak_initial(x_fine, p_model)
                y_fit_fine += y_peak_fine

                curr_kwargs = dict(color_kwargs)
                if "color" not in curr_kwargs:
                    curr_kwargs["color"] = f"C{(i+1)%10}"

                peak_line, = self.ax.plot(
                    x_fine + x_offset, y_peak_fine + y_offset,
                    lw=(lw * 0.6), **curr_kwargs)

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
                curr_kwargs = dict(color_kwargs)
                if "color" not in curr_kwargs:
                    curr_kwargs["color"] = f"C{(i+1)%10}"

                peak_line, = self.ax.plot(
                    x + x_offset, y_peak + y_offset,
                    lw=(lw * 0.6), **curr_kwargs)

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
        self._legend_obj = None
        self._legend_bbox = None
        if self.btn_legend.isChecked():
            handles, labels = self.ax.get_legend_handles_labels()
            max_items = self.spin_max_legend_items.value()
            if getattr(self, "act_legend_outside", None) and self.act_legend_outside.isChecked():
                legend = self.ax.legend(
                    handles[:max_items],
                    labels[:max_items],
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                )
            else:
                legend = self.ax.legend(
                    handles[:max_items], labels[:max_items], loc="upper right")
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

        # ── Render active spectral range selectors ──
        if hasattr(self, "range_manager") and self.ax is not None:
            self.range_manager.render(self.ax)

        self.canvas.draw_idle()

    # ─────────────────────────────────────────────
    # Heatmap mode
    # ─────────────────────────────────────────────

    def _remove_heatmap_colorbar(self):
        """Remove the colorbar created by a previous heatmap draw, if any."""
        if self._heatmap_colorbar is not None:
            try:
                self._heatmap_colorbar.remove()
            except Exception:
                pass
            self._heatmap_colorbar = None

    # ── Filename-based numeric token extraction for heatmap Y-axis ──

    @staticmethod
    def _extract_filename_tokens(fname):
        """Split *fname* on underscores and return tokens that contain digits.

        Returns a list of ``(position, token_text, numeric_value)`` tuples.
        *position* is the 0-based index inside the underscore-split list so
        the same slot can be looked up across different filenames.
        """
        stem = fname.rsplit(".", 1)[0] if "." in fname else fname
        tokens = stem.split("_")
        result = []
        for pos, tok in enumerate(tokens):
            numbers = _RE_NUMBERS.findall(tok)
            if numbers:
                result.append((pos, tok, float(numbers[0])))
        return result

    def _collect_heatmap_yaxis_tokens(self):
        """Return filename tokens suitable as heatmap Y-axis choices.

        Parse the *first* selected spectrum's filename, find every
        underscore-delimited token that contains a numeric value, and
        return them as ``(position, token_text)`` pairs.
        """
        fnames = (self._tensor_data.get("fnames", [])
                  if self._tensor_data else [])
        if not fnames:
            return []
        return [(pos, tok)
                for pos, tok, _val in self._extract_filename_tokens(fnames[0])]

    def _refresh_heatmap_yaxis_choices(self):
        """Rebuild the Y-axis combobox items from the selected filenames."""
        prev = self.cbb_heatmap_yaxis.currentText()
        self.cbb_heatmap_yaxis.blockSignals(True)
        self.cbb_heatmap_yaxis.clear()
        self.cbb_heatmap_yaxis.addItem("Index")
        for _pos, token_text in self._collect_heatmap_yaxis_tokens():
            self.cbb_heatmap_yaxis.addItem(token_text)
        # Restore previous selection if still available
        idx = self.cbb_heatmap_yaxis.findText(prev)
        if idx >= 0:
            self.cbb_heatmap_yaxis.setCurrentIndex(idx)
        self.cbb_heatmap_yaxis.blockSignals(False)

    def _get_heatmap_y_values(self):
        """Return ``(y_values, y_label)`` for the heatmap Y-axis.

        * **Index** → simple 0 … N-1.
        * Otherwise the selected token text is matched to a token position
          from the *first* filename, and the numeric part of that same
          position is extracted from every filename.
        """
        fnames = self._tensor_data.get("fnames", []) if self._tensor_data else []
        selected = self.cbb_heatmap_yaxis.currentText()
        N = len(self._tensor_data.get("y", []))

        if selected == "Index" or not fnames:
            label = self.le_heatmap_ylabel.text().strip() or "Spectrum index"
            return np.arange(N, dtype=float), label

        # ── Filename token ──
        # Identify the token position from the first filename
        ref_tokens = self._extract_filename_tokens(fnames[0])
        token_pos = None
        for pos, tok, _val in ref_tokens:
            if tok == selected:
                token_pos = pos
                break

        if token_pos is None:
            label = self.le_heatmap_ylabel.text().strip() or selected
            return np.arange(N, dtype=float), label

        # Extract the numeric value at *token_pos* from every filename
        values = []
        for fname in fnames:
            stem = fname.rsplit(".", 1)[0] if "." in fname else fname
            parts = stem.split("_")
            if token_pos < len(parts):
                nums = _RE_NUMBERS.findall(parts[token_pos])
                values.append(float(nums[0]) if nums else float("nan"))
            else:
                values.append(float("nan"))

        auto_label = selected
        label = self.le_heatmap_ylabel.text().strip() or auto_label
        return np.array(values, dtype=float), label

    def _plot_heatmap(self):
        """Render the selected spectra as a 2D pcolormesh heatmap."""
        from scipy.interpolate import interp1d
        style_name = self.cbb_theme.currentText()
        if style_name == "Dark Mode":
            style_path = PLOT_POLICY_DARK
        else:
            style_path = PLOT_POLICY_LIGHT

        fg_color = plt.rcParams.get('axes.labelcolor', 'black')

        # Save current zoom/pan limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        is_default = (xlim == (0.0, 1.0) and ylim == (0.0, 1.0))

        self.ax.clear()
        self._remove_heatmap_colorbar()
        self._fitted_lines.clear()

        Y_data = self._tensor_data.get("y", [])
        x_data = self._tensor_data.get("x")
        N = len(Y_data)

        if N < 2 or x_data is None:
            self.ax.set_title("Heatmap requires ≥ 2 spectra", color=fg_color)
            self.canvas.draw_idle()
            return

        # ── Build intensity matrix ──
        # For tensor_list (ragged), interpolate all spectra onto a common x grid
        is_list = isinstance(Y_data, list)
        if is_list:
            # Determine common x range
            x_arrays = [np.asarray(x_data[i] if isinstance(x_data, list) else x_data)
                        for i in range(N)]
            x_min = max(np.nanmin(xa) for xa in x_arrays)
            x_max = min(np.nanmax(xa) for xa in x_arrays)
            n_pts = max(len(xa) for xa in x_arrays)
            raman_shift = np.linspace(x_min, x_max, n_pts)

            intensity_rows = []
            for i in range(N):
                yi = np.asarray(Y_data[i])
                xi = x_arrays[i]
                f = interp1d(xi, yi, kind="linear", bounds_error=False,
                             fill_value=np.nan)
                intensity_rows.append(f(raman_shift))
            intensity_matrix = np.array(intensity_rows)
        else:
            raman_shift = np.asarray(x_data)
            intensity_matrix = np.asarray(Y_data)

        # ── Apply normalization if enabled ──
        intensity_matrix = np.asarray(
            self._get_normalized_y_tensor(raman_shift, intensity_matrix))

        # ── Y-axis values ──
        y_values, y_label = self._get_heatmap_y_values()

        valid_y = ~np.isnan(y_values)
        if not np.all(valid_y):
            y_values = y_values[valid_y]
            intensity_matrix = intensity_matrix[valid_y]
            N = len(y_values)
            if N == 0:
                return


        # Sort by Y values for a coherent image
        sort_idx = np.argsort(y_values)
        y_values = y_values[sort_idx]
        intensity_matrix = intensity_matrix[sort_idx]

        # ── Handle duplicate Y-values (average them) ──
        unique_y, inverse_idx = np.unique(y_values, return_inverse=True)
        if len(unique_y) < len(y_values):
            import warnings
            new_intensity = np.zeros((len(unique_y), intensity_matrix.shape[1]))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                for i in range(len(unique_y)):
                    new_intensity[i] = np.nanmean(intensity_matrix[inverse_idx == i], axis=0)
            y_values = unique_y
            intensity_matrix = new_intensity
            N = len(y_values)

        # ── Optional interpolation along Y ──
        n_interp = self.spin_heatmap_interp.value()
        if n_interp > 0 and N >= 3 and len(np.unique(y_values)) >= 3:
            y_fine = np.linspace(y_values.min(), y_values.max(), n_interp)
            kind = "cubic" if len(y_values) >= 4 else "linear"

            # Fast vectorized path: when all entries are valid (typical case),
            # interpolate the whole matrix along axis=0 at once!
            if not np.isnan(intensity_matrix).any():
                f = interp1d(y_values, intensity_matrix, kind=kind, axis=0,
                             bounds_error=False, fill_value=np.nan)
                intensity_plot = f(y_fine)
            else:
                interp_cols = []
                for col_idx in range(intensity_matrix.shape[1]):
                    col = intensity_matrix[:, col_idx]
                    valid = ~np.isnan(col)
                    if valid.sum() >= 2:
                        col_kind = "cubic" if valid.sum() >= 4 else "linear"
                        f = interp1d(y_values[valid], col[valid], kind=col_kind,
                                     bounds_error=False, fill_value=np.nan)
                        interp_cols.append(f(y_fine))
                    else:
                        interp_cols.append(np.full_like(y_fine, np.nan))
                intensity_plot = np.array(interp_cols).T
            y_plot = y_fine
        else:
            intensity_plot = intensity_matrix
            y_plot = y_values

        # ── Colormap from current palette ──
        palette_name = self.cbb_color_palette.currentText()
        if palette_name in SPECTRA_DISCRETE_PALETTES:
            cmap_name = "jet"  # Discrete palettes aren't suitable for heatmaps
        else:
            cmap_name = palette_name
        try:
            cmap = mpl.colormaps[cmap_name]
        except (KeyError, AttributeError):
            cmap = mpl.colormaps["jet"]

        # ── Draw pcolormesh ──
        with plt.style.context(style_path):
            mesh = self.ax.pcolormesh(
                raman_shift, y_plot, intensity_plot,
                shading="auto", cmap=cmap)

            cbar = self.figure.colorbar(mesh, ax=self.ax)
            cbar.set_label(self.cbb_yaxis.currentText(), color=fg_color)
            cbar.ax.tick_params(colors=fg_color)
            self._heatmap_colorbar = cbar

        self.ax.set_xlabel(self.cbb_xaxis.currentText())
        self.ax.set_ylabel(y_label)

        if self.act_grid.isChecked():
            self.ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

        if not is_default:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

        self.lbl_r2.setText("")
        self.lbl_noise.setText("")
        self.canvas.draw_idle()

    def _toggle_heatmap_mode(self, checked: bool):
        """Activate or deactivate heatmap mode, disabling incompatible tools."""
        self._heatmap_mode = checked

        if checked:
            # Refresh Y-axis choices from current data
            self._refresh_heatmap_yaxis_choices()

            # Exit eraser mode if active
            if getattr(self, "_erase_mode", False):
                self.btn_eraser.setChecked(False)

            # Force zoom mode on and hide incompatible tool buttons
            self.btn_zoom.blockSignals(True)
            self.btn_zoom.setChecked(True)
            self.btn_zoom.blockSignals(False)
            self._set_zoom_tool_active(True)

            self.btn_peak.setVisible(False)
            self.btn_baseline.setVisible(False)
            self.btn_eraser.setVisible(False)
            self.btn_norm.setVisible(False)
            self.norm_xmin.setVisible(False)
            self.norm_xmax.setVisible(False)
            self.btn_bestfit.setVisible(False)

            # Hide shift sliders
            self.shift_panel.setVisible(False)

            # Show heatmap-specific options
            self.cbb_heatmap_yaxis.setVisible(True)
            self.le_heatmap_ylabel.setVisible(True)
            self.spin_heatmap_interp.setVisible(True)
        else:
            # Re-show tools
            self.btn_peak.setVisible(True)
            self.btn_baseline.setVisible(True)
            self.btn_eraser.setVisible(True)
            self.btn_norm.setVisible(True)
            self.norm_xmin.setVisible(True)
            self.norm_xmax.setVisible(True)
            self.btn_bestfit.setVisible(True)

            # Restore shift sliders
            self.shift_panel.setVisible(True)

            # Remove the colorbar left over from heatmap
            self._remove_heatmap_colorbar()

            # Hide heatmap-specific options
            self.cbb_heatmap_yaxis.setVisible(False)
            self.le_heatmap_ylabel.setVisible(False)
            self.spin_heatmap_interp.setVisible(False)

        self._emit_view_options()


    def _get_normalized_y(self, x, y, xmin=None, xmax=None):
        """Apply normalization if enabled (VIEW-ONLY)."""
        if not self.btn_norm.isChecked():
            return y

        if xmin is None and xmax is None:
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
            idx_start = min(imin, imax)
            idx_end = max(imin, imax) + 1
            norm_val = np.max(y[idx_start:idx_end]) if idx_end > idx_start else np.max(y)
        else:
            norm_val = np.max(y)

        if norm_val != 0:
            return y / norm_val

        return y

    
    def _make_legend_pickable(self, legend):
        """Cache the legend bounds for whole-box double-click interaction."""
        self._legend_obj = legend
        # ``renderer`` is only installed as an attribute after the first draw;
        # get_renderer() also works when data arrives before the widget has
        # been painted (common during workspace restoration and in tests).
        self._legend_bbox = legend.get_window_extent(
            self.canvas.get_renderer())

        # Connect double-click handler once (replaces pick_event)
        if not hasattr(self, "_legend_dblclick_connected"):
            self.canvas.mpl_connect("button_press_event", self._on_legend_double_click)
            self._legend_dblclick_connected = True


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
            y0 = y_vals[0]
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
        """Open the batch editor when any part of the legend is double-clicked."""
        if not getattr(event, "dblclick", False):
            return

        legend_obj = getattr(self, "_legend_obj", None)
        if legend_obj is not None:
            try:
                self._legend_bbox = legend_obj.get_window_extent(
                    self.canvas.get_renderer()
                )
            except Exception:
                pass

        legend_bbox = getattr(self, "_legend_bbox", None)
        if legend_bbox is None or event.x is None or event.y is None:
            return

        if legend_bbox.contains(event.x, event.y):
            self._open_legend_editor()

    def _legend_editor_entries(self):
        """Describe editable plotted spectra, capped for responsive editing."""
        if not self._tensor_data:
            return []

        spectra = self._tensor_data.get("y")
        spectrum_count = len(spectra) if spectra is not None else 0
        editor_count = min(spectrum_count, LEGEND_EDITOR_MAX_ROWS)
        raw_labels = self._tensor_data.get("labels")
        raw_fnames = self._tensor_data.get("fnames")
        raw_colors = self._tensor_data.get("colors")
        labels = [] if raw_labels is None else list(raw_labels)
        fnames = [] if raw_fnames is None else list(raw_fnames)
        stored_colors = [] if raw_colors is None else list(raw_colors)

        entries = []
        for index in range(editor_count):
            name = (
                str(fnames[index])
                if index < len(fnames) and fnames[index]
                else f"Spectrum {index + 1}"
            )
            display_label = (
                str(labels[index])
                if index < len(labels) and labels[index]
                else name
            )
            stored_color = (
                stored_colors[index]
                if index < len(stored_colors) and stored_colors[index]
                else None
            )
            entries.append({
                "index": index,
                "name": name,
                "display_label": display_label,
                "stored_color": stored_color,
            })
        return entries

    def _create_legend_editor_dialog(self):
        spectrum_count = len(self._tensor_data.get("y", []))
        entries = self._legend_editor_entries()
        truncation_message = ""
        if len(entries) < spectrum_count:
            truncation_message = (
                f"Performance safeguard: showing the first "
                f"{len(entries)} of {spectrum_count} plotted spectra. "
                "Narrow the selection to edit spectra outside this batch."
            )
        return SpectraLegendEditorDialog(
            entries=entries,
            total_entry_count=spectrum_count,
            truncation_message=truncation_message,
            palette_names=SPECTRA_COLOR_PALETTES,
            custom_palettes=SPECTRA_CUSTOM_PALETTES,
            current_palette=self.cbb_color_palette.currentText(),
            max_legend_items=self.spin_max_legend_items.value(),
            series_colors_provider=self._get_colors_for_palette,
            palette_colors_provider=self._get_palette_color_choices,
            parent=self,
        )

    def _open_legend_editor(self):
        """Open the transactional editor and apply its values only on OK."""
        if not self._legend_editor_entries():
            return

        dialog = self._create_legend_editor_dialog()
        self._legend_editor_dialog = dialog
        if dialog.exec() == QDialog.Accepted:
            self._apply_legend_editor_values(dialog.values())

    @staticmethod
    def _normalized_optional_color(color):
        if not color:
            return None
        try:
            return mpl.colors.to_hex(color)
        except (TypeError, ValueError):
            return None

    def _apply_legend_editor_values(self, values):
        """Commit accepted dialog values to view state and spectrum proxies."""
        if not self._tensor_data:
            return

        palette = values.get("palette", self.cbb_color_palette.currentText())
        max_items = int(values.get(
            "max_legend_items", self.spin_max_legend_items.value()))
        options_changed = (
            palette != self.cbb_color_palette.currentText()
            or max_items != self.spin_max_legend_items.value()
        )

        self.cbb_color_palette.blockSignals(True)
        self.spin_max_legend_items.blockSignals(True)
        self.cbb_color_palette.setCurrentText(palette)
        self.spin_max_legend_items.setValue(max_items)
        self.cbb_color_palette.blockSignals(False)
        self.spin_max_legend_items.blockSignals(False)

        spectrum_count = len(self._tensor_data.get("y", []))
        raw_labels = self._tensor_data.get("labels")
        raw_colors = self._tensor_data.get("colors")
        raw_proxies = self._tensor_data.get("proxies")
        labels = [] if raw_labels is None else list(raw_labels)
        colors = [] if raw_colors is None else list(raw_colors)
        proxies = [] if raw_proxies is None else list(raw_proxies)
        labels.extend([None] * (spectrum_count - len(labels)))
        colors.extend([None] * (spectrum_count - len(colors)))

        spectra_changed = False
        for edited in values.get("spectra", []):
            index = int(edited.get("index", -1))
            if index < 0 or index >= spectrum_count:
                continue

            new_label = str(edited.get("label", "")).strip()
            if not new_label:
                new_label = (
                    self._tensor_data.get("fnames", [])[index]
                    if index < len(self._tensor_data.get("fnames", []))
                    else f"Spectrum {index + 1}"
                )
            if new_label != edited.get("original_display_label"):
                labels[index] = new_label
                if index < len(proxies):
                    proxies[index].label = new_label
                spectra_changed = True

            old_color = self._normalized_optional_color(
                edited.get("original_color"))
            new_color = self._normalized_optional_color(edited.get("color"))
            if new_color != old_color:
                colors[index] = new_color
                if index < len(proxies):
                    proxies[index].color = new_color
                spectra_changed = True

        self._tensor_data["labels"] = labels
        self._tensor_data["colors"] = colors

        if spectra_changed:
            self.spectrumCustomized.emit()
        if options_changed:
            self._emit_view_options()
        elif spectra_changed:
            self._plot()


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
            "color_palette": self.cbb_color_palette.currentText() if hasattr(self, "cbb_color_palette") else "DEFAULT_COLORS",
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
            "legend_outside": self.act_legend_outside.isChecked() if hasattr(self, "act_legend_outside") else False,
            "bestfit": self.btn_bestfit.isChecked() if hasattr(self, "btn_bestfit") else True,
            "max_legend_items": self.spin_max_legend_items.value() if hasattr(self, "spin_max_legend_items") else 15,
            "copy_fig_theme": self.cbb_copy_theme.currentText() if hasattr(self, "cbb_copy_theme") else "Light Mode",
            "heatmap": self.btn_heatmap.isChecked() if hasattr(self, "btn_heatmap") else False,
            "heatmap_yaxis": self.cbb_heatmap_yaxis.currentText() if hasattr(self, "cbb_heatmap_yaxis") else "Index",
            "heatmap_ylabel": self.le_heatmap_ylabel.text() if hasattr(self, "le_heatmap_ylabel") else "",
            "heatmap_interp": self.spin_heatmap_interp.value() if hasattr(self, "spin_heatmap_interp") else 300,
        }

    def set_options_state(self, state):
        """Set the complete state of all view options without causing infinite loops."""
        if not hasattr(self, "cbb_theme"):
            return
            
        current_state = self.get_options_state()
        if current_state == state:
            return
            
        self._is_setting_options = True
        def _update(widget, setter, value):
            if value is None: return
            widget.blockSignals(True)
            setter(value)
            widget.blockSignals(False)

        theme_val = state.get("theme")
        if theme_val == "Soft Dark Mode":
            theme_val = "Dark Mode"
        _update(self.cbb_theme, self.cbb_theme.setCurrentText, theme_val)
        _update(self.cbb_xaxis, self.cbb_xaxis.setCurrentText, state.get("xaxis"))
        _update(self.cbb_yaxis, self.cbb_yaxis.setCurrentText, state.get("yaxis"))
        _update(self.cbb_yscale, self.cbb_yscale.setCurrentText, state.get("yscale"))
        _update(self.cbb_plotstyle, self.cbb_plotstyle.setCurrentText, state.get("plotstyle"))
        if hasattr(self, "cbb_color_palette"):
            _update(self.cbb_color_palette,
                    self.cbb_color_palette.setCurrentText,
                    state.get("color_palette", "DEFAULT_COLORS"))
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
        if hasattr(self, "act_legend_outside"):
            _update(self.act_legend_outside, self.act_legend_outside.setChecked, state.get("legend_outside", False))
        _update(self.btn_bestfit, self.btn_bestfit.setChecked, state.get("bestfit"))
        if hasattr(self, "spin_max_legend_items"):
            _update(self.spin_max_legend_items,
                    self.spin_max_legend_items.setValue,
                    state.get("max_legend_items", 15))
        if hasattr(self, "cbb_copy_theme"):
            _update(self.cbb_copy_theme, self.cbb_copy_theme.setCurrentText, state.get("copy_fig_theme", "Light Mode"))
        if hasattr(self, "btn_heatmap"):
            _update(self.btn_heatmap, self.btn_heatmap.setChecked, state.get("heatmap", False))
        if hasattr(self, "cbb_heatmap_yaxis"):
            _update(self.cbb_heatmap_yaxis, self.cbb_heatmap_yaxis.setCurrentText, state.get("heatmap_yaxis", "Index"))
        if hasattr(self, "le_heatmap_ylabel"):
            _update(self.le_heatmap_ylabel, self.le_heatmap_ylabel.setText, state.get("heatmap_ylabel", ""))
        if hasattr(self, "spin_heatmap_interp"):
            _update(self.spin_heatmap_interp, self.spin_heatmap_interp.setValue, state.get("heatmap_interp", 300))

        self._apply_plot_style()
        self._plot()
        self._is_setting_options = False
        
    def _apply_plot_style(self):
        style_name = self.cbb_theme.currentText()
        if style_name == "Dark Mode":
            style_path = PLOT_POLICY_DARK
        else:
            style_path = PLOT_POLICY_LIGHT
            
        # Parse the style file from cached dict without modifying global rcParams
        style_dict = get_cached_style_dict(style_path)
            
        fig_bg = style_dict.get('figure.facecolor', 'white')
        self.figure.patch.set_facecolor(fig_bg)
        if hasattr(self, "canvas_card"):
            tok = current_tokens()
            fig_hex = fig_bg if str(fig_bg).startswith("#") or str(fig_bg).isalpha() else f"#{fig_bg}"
            self.canvas_card.setStyleSheet(
                f"QFrame#canvasCard {{ background-color: {fig_hex}; border: 1px solid {tok.border}; border-radius: 8px; }}"
            )
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
            if not getattr(self, '_is_setting_options', False):
                self.allOptionsSyncChanged.emit(self.get_options_state())
            self._plot()

    def _on_heatmap_ylabel_changed(self):
        if not hasattr(self, '_ylabel_timer'):
            from PySide6.QtCore import QTimer
            self._ylabel_timer = QTimer(self)
            self._ylabel_timer.setSingleShot(True)
            self._ylabel_timer.setInterval(400)
            self._ylabel_timer.timeout.connect(self._emit_view_options)
        self._ylabel_timer.start()

    def _emit_view_options(self):
        if getattr(self, '_is_setting_options', False):
            return
            
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

    def _on_mouse_click(self, event):
        if event.inaxes != self.ax:
            return

        # ── Check if clicking on active range selector marker or span ──
        if hasattr(self, "range_manager") and any(s.is_active for s in self.range_manager.selectors.values()):
            hit = self.range_manager.hit_test(self.ax, event, zoom_active=self.zoom_pan_active)
            if hit and event.button == 1:
                # Suppress concurrent Matplotlib zoom interaction if active
                if hasattr(self.toolbar, "_zoom_info") and self.toolbar._zoom_info is not None:
                    try:
                        if hasattr(self.toolbar._zoom_info, "cid"):
                            self.canvas.mpl_disconnect(self.toolbar._zoom_info.cid)
                        self.toolbar.remove_rubberband()
                        self.toolbar._zoom_info = None
                    except Exception:
                        pass

                color_id, target = hit
                self.range_manager.start_drag(color_id, target, event.xdata)
                self.canvas.setCursor(Qt.ClosedHandCursor if target == "body" else Qt.SizeHorCursor)
                cid_m = self.canvas.mpl_connect("motion_notify_event", self._on_range_drag_motion)
                cid_r = self.canvas.mpl_connect("button_release_event", self._on_range_drag_release)
                self._range_drag_cids = [cid_m, cid_r]
                return

        if self.zoom_pan_active:
            return

        # Heatmap mode: no peak/baseline clicking
        if self._heatmap_mode:
            return

        # The cosmic-ray eraser has its own press/motion/release handlers
        # (see _erase_on_press etc.) — without this guard, peak/baseline
        # picking could also fire for the same click if btn_peak/btn_baseline
        # happened to still be checked from before erase mode was entered.
        if self._erase_mode:
            return

        # Skip legend clicks
        legend_obj = getattr(self, "_legend_obj", None)
        if legend_obj is not None:
            try:
                self._legend_bbox = legend_obj.get_window_extent(
                    self.canvas.get_renderer()
                )
            except Exception:
                pass

        legend_bbox = getattr(self, "_legend_bbox", None)
        if legend_bbox is not None and event.x is not None and event.y is not None:
            if legend_bbox.contains(event.x, event.y):
                return

        x = event.xdata
        y = event.ydata

        # ─── Peak tool ─────────────────────────
        if self.btn_peak.isChecked():
            if event.button == 1:      # left click
                if self.btn_norm.isChecked():
                    msg = "The 'Intensity normalization' tool is active.\nThis normalization is just for visualization, not affected to data.\n\nPlease deactivate the Normalization (unpush the button) before adding a peak to ensure it is placed in the real intensity scale."
                    QMessageBox.information(self, "Normalization Active", msg)
                    return
                # Check if clicking on an existing peak to drag it
                clicked_peak = self._find_peak_at_position(x, y)
                if clicked_peak:
                    # Start dragging
                    self._dragging_peak = clicked_peak
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
                if self.btn_norm.isChecked():
                    msg = "The 'Intensity normalization' tool is active.\nThis normalization is just for visualization, not affected to data.\n\nPlease deactivate the Normalization (unpush the button) before adding a baseline point to ensure it is placed in the real intensity scale."
                    QMessageBox.information(self, "Normalization Active", msg)
                    return
                self.baseline_add_requested.emit(x, y)
            elif event.button == 3:
                self.baseline_remove_requested.emit(x)



    def _rescale(self):
        """Rescale the spectra plot to fit within the axes."""
        self.ax.autoscale()
        self.canvas.draw_idle()
        
    def _mpl_zoom_active(self) -> bool:
        """Whether matplotlib's own internal zoom-rectangle mode is currently on."""
        return bool(getattr(self.toolbar.mode, "value", self.toolbar.mode))

    def _set_zoom_tool_active(self, active: bool):
        """Synchronize zoom_pan_active with matplotlib's internal zoom mode.

        `toolbar.zoom()` is a raw toggle, not a setter — calling it
        unconditionally (as every call site used to) silently flips
        matplotlib's zoom-rectangle tool out of sync with our own
        zoom_pan_active flag whenever it's called from more than one place
        (e.g. the eraser tool disabling/restoring zoom around its own
        activation). Routing every on/off request through here makes it
        idempotent: the toolbar is only actually toggled when its current
        state disagrees with what we're asking for.
        """
        if self._mpl_zoom_active() != active:
            self.toolbar.zoom()
        self.zoom_pan_active = active

    def _toggle_zoom_pan(self):
        """Zoom feature for button Zoom"""
        self._set_zoom_tool_active(self.btn_zoom.isChecked())

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

        # ── Range selector hover feedback ──
        if hasattr(self, "range_manager") and any(s.is_active for s in self.range_manager.selectors.values()):
            if not self.range_manager.is_dragging():
                hit = self.range_manager.hit_test(self.ax, event, zoom_active=self.zoom_pan_active)
                if hit:
                    if hit[1] in ("min", "max"):
                        self.canvas.setCursor(Qt.SizeHorCursor)
                        self._range_hover_active = True
                        return
                    elif hit[1] == "body":
                        self.canvas.setCursor(Qt.OpenHandCursor)
                        self._range_hover_active = True
                        return
                elif getattr(self, "_range_hover_active", False):
                    self.canvas.unsetCursor()
                    self._range_hover_active = False

        if not self._fitted_lines:
            self._hide_tooltip()
            self._reset_highlight()
            return

        for line, info in self._fitted_lines:
            # Check if line still has a valid figure before calling contains()
            if line.figure is None:
                continue
            
            # Fast bounding check along x before calling expensive line.contains()
            if "x0" in info and event.xdata is not None:
                fwhm = info.get("fwhm", 10.0) or 10.0
                if abs(event.xdata - info["x0"]) > max(3.0 * float(fwhm), 20.0):
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
                display_intensity = fano_display_amplitude(display_intensity, q)

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


    def _hide_tooltip(self, *args):
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

    # ═════════════════════════════════════════════════════════════════════
    # Spectral Range Selector Tool
    # ═════════════════════════════════════════════════════════════════════

    def set_range_selectors_visible(self, visible: bool):
        """Show or hide the spectral range selector panel on the right toolbar.

        Active only in the Maps workspace. When hidden, any active selectors
        are cleanly deactivated.
        """
        self.range_panel.setVisible(visible)
        if not visible:
            self.deactivate_all_range_selectors()

    def deactivate_all_range_selectors(self):
        """Deactivate all range selectors and clear their visual markers."""
        for color_id, btn in self.range_panel.buttons.items():
            if btn.isChecked():
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
                self.range_manager.set_active(color_id, False)
        if self.ax is not None:
            self.range_manager.render(self.ax)
            self.canvas.draw_idle()

    def _get_current_data_xrange(self) -> Tuple[Optional[float], Optional[float]]:
        """Return (min_x, max_x) of currently loaded spectra data."""
        if self._tensor_data:
            x_arr = self._tensor_data.get("x")
            if x_arr is not None:
                first_x = x_arr[0] if isinstance(x_arr, list) and len(x_arr) > 0 else x_arr
                if hasattr(first_x, "__len__") and len(first_x) > 1:
                    return float(np.min(first_x)), float(np.max(first_x))
        if self._current_spectra:
            x_vals = []
            for s in self._current_spectra:
                if hasattr(s, "x") and s.x is not None and len(s.x) > 0:
                    x_vals.extend([float(np.min(s.x)), float(np.max(s.x))])
            if x_vals:
                return min(x_vals), max(x_vals)
        if self.ax is not None:
            xlim = self.ax.get_xlim()
            if xlim[0] < xlim[1]:
                return float(xlim[0]), float(xlim[1])
        return None, None

    def _toggle_range_selector(self, color_id: str, checked: bool):
        """Activate or deactivate one of the spectral range selectors."""
        if checked and self.btn_zoom.isChecked():
            self.btn_zoom.blockSignals(True)
            self.btn_zoom.setChecked(False)
            self.btn_zoom.blockSignals(False)
            self._set_zoom_tool_active(False)

        if checked:
            curr_xmin, curr_xmax = self._get_current_data_xrange()
            if curr_xmin is not None and curr_xmax is not None:
                self.range_manager.refresh_or_clamp_ranges(curr_xmin, curr_xmax, target_color_id=color_id)

        self.range_manager.set_active(color_id, checked)
        if self.ax is not None:
            self.range_manager.render(self.ax)
            self.canvas.draw_idle()

    def _on_range_drag_motion(self, event):
        """Update range selector boundaries while user drags."""
        if not hasattr(self, "range_manager") or not self.range_manager.is_dragging():
            return
        if event.inaxes != self.ax or event.xdata is None:
            return
        color_id = self.range_manager.update_drag(event.xdata)
        if color_id and self.ax is not None:
            self.range_manager.render(self.ax)
            self.canvas.draw_idle()

    def _on_range_drag_release(self, event):
        """Finalize range selector dragging on mouse button release."""
        if hasattr(self, "_range_drag_cids"):
            for cid in self._range_drag_cids:
                try:
                    self.canvas.mpl_disconnect(cid)
                except Exception:
                    pass
            self._range_drag_cids.clear()

        # Suppress any lingering Matplotlib zoom interaction
        if hasattr(self.toolbar, "_zoom_info") and self.toolbar._zoom_info is not None:
            try:
                if hasattr(self.toolbar._zoom_info, "cid"):
                    self.canvas.mpl_disconnect(self.toolbar._zoom_info.cid)
                self.toolbar.remove_rubberband()
                self.toolbar._zoom_info = None
            except Exception:
                pass

        if hasattr(self, "range_manager") and self.range_manager.is_dragging():
            self.range_manager.finish_drag()
            if self.ax is not None:
                self.range_manager.render(self.ax)
                self.canvas.draw_idle()
            self.canvas.unsetCursor()
            self._range_hover_active = False

    # ═════════════════════════════════════════════════════════════════════
    # Cosmic-Ray Eraser Tool
    # ═════════════════════════════════════════════════════════════════════

    def _toggle_erase_mode(self, checked: bool):
        """Activate or deactivate the cosmic-ray erase mode."""
        self._erase_mode = checked

        # Show/hide companion buttons
        self.btn_erase_undo.setVisible(checked)
        self.btn_erase_validate.setVisible(checked)

        if checked:
            # Remember whichever of zoom/baseline/peak was active so it can
            # be restored exactly (not just defaulted back to zoom) once
            # erase mode ends, and disable zoom so it doesn't fight with
            # the erase drag gesture.
            self._pre_erase_tool_btn = next(
                (b for b in (self.btn_zoom, self.btn_baseline, self.btn_peak) if b.isChecked()),
                None,
            )
            if self.btn_zoom.isChecked():
                self.btn_zoom.blockSignals(True)
                self.btn_zoom.setChecked(False)
                self.btn_zoom.blockSignals(False)
            self._set_zoom_tool_active(False)

            # Set an eraser-style cursor on the canvas
            self.canvas.setCursor(Qt.CursorShape.SizeHorCursor)

            # Connect matplotlib mouse events
            cid_press   = self.canvas.mpl_connect("button_press_event",   self._erase_on_press)
            cid_motion  = self.canvas.mpl_connect("motion_notify_event",  self._erase_on_motion)
            cid_release = self.canvas.mpl_connect("button_release_event", self._erase_on_release)
            self._erase_cids = [cid_press, cid_motion, cid_release]

            # Reset session state
            self._erase_drag_start_x = None
            self._erase_highlight = None
            self._erase_history.clear()
        else:
            # Disconnect matplotlib events
            for cid in self._erase_cids:
                self.canvas.mpl_disconnect(cid)
            self._erase_cids.clear()

            # Remove any leftover highlight
            self._erase_clear_highlight()

            # Restore default cursor
            self.canvas.unsetCursor()

            # Restore whichever zoom/baseline/peak tool was active before
            # erase mode started (falls back to zoom if none was, e.g. at
            # startup — matches the previous hardcoded behaviour in that case).
            restore_btn = getattr(self, "_pre_erase_tool_btn", None) or self.btn_zoom
            self._pre_erase_tool_btn = None
            restore_btn.blockSignals(True)
            restore_btn.setChecked(True)
            restore_btn.blockSignals(False)
            self._set_zoom_tool_active(restore_btn is self.btn_zoom)

    def _erase_on_press(self, event):
        """Record the start position of an erase drag."""
        if not self._erase_mode:
            return
        if event.inaxes != self.ax or event.button != 1:
            return
        if event.xdata is None:
            return

        self._erase_drag_start_x = event.xdata
        # Clear any previous highlight; the span will be drawn on first motion
        self._erase_clear_highlight()

    def _erase_on_motion(self, event):
        """Update the highlight span while the user drags."""
        if not self._erase_mode:
            return
        if self._erase_drag_start_x is None:
            return
        if event.inaxes != self.ax or event.xdata is None:
            return

        xmin = min(self._erase_drag_start_x, event.xdata)
        xmax = max(self._erase_drag_start_x, event.xdata)

        # Remove old span and draw a new one — avoids mutating get_xy() which
        # returns an immutable tuple in newer Matplotlib versions.
        if self._erase_highlight is not None:
            try:
                self._erase_highlight.remove()
            except Exception:
                pass
        self._erase_highlight = self.ax.axvspan(
            xmin, xmax, color="red", alpha=0.25, zorder=5
        )
        self.canvas.draw_idle()

    def _erase_on_release(self, event):
        """Perform the erasure when the user releases the mouse button."""
        if not self._erase_mode:
            return
        if self._erase_drag_start_x is None:
            return
        if event.button != 1 or event.xdata is None:
            self._erase_drag_start_x = None
            self._erase_clear_highlight()
            return

        x0 = min(self._erase_drag_start_x, event.xdata)
        x1 = max(self._erase_drag_start_x, event.xdata)
        self._erase_drag_start_x = None
        self._erase_clear_highlight()

        if not self._tensor_data:
            return

        # Slot 0 in the viewer batch corresponds to the first selected spectrum.
        # For the Maps workspace, _tensor_data["indices"] holds the actual md.Y row
        # indices; we pass that through the signal so the ViewModel can write to the
        # correct row.  For the Spectra workspace there are no "indices" and spec_idx
        # stays at 0 (each spectrum is its own MapData with a single row).
        viewer_slot = 0  # we always edit the first displayed spectrum
        tensor_indices = self._tensor_data.get("indices")
        md_row = int(tensor_indices[viewer_slot]) if tensor_indices is not None else viewer_slot

        x_data = self._tensor_data.get("x")
        Y_data = self._tensor_data.get("y")

        if x_data is None or Y_data is None:
            return

        # Handle tensor_list (ragged, Spectra WS) vs ndarray (uniform, Maps WS)
        if isinstance(x_data, list):
            if viewer_slot >= len(x_data):
                return
            x_arr = np.asarray(x_data[viewer_slot], dtype=float)
            y_arr = np.asarray(Y_data[viewer_slot], dtype=float).copy()
        else:
            x_arr = np.asarray(x_data, dtype=float)
            y_arr = np.asarray(Y_data[viewer_slot], dtype=float).copy()

        # Find indices inside the erased region
        mask = (x_arr >= x0) & (x_arr <= x1)
        if not mask.any():
            return

        # Boundary indices for interpolation (first valid points outside the region)
        erase_indices = np.where(mask)[0]
        i_left  = max(erase_indices[0] - 1, 0)
        i_right = min(erase_indices[-1] + 1, len(x_arr) - 1)

        # Save snapshot for undo BEFORE modifying (use viewer_slot as the undo key)
        self._erase_history.append((viewer_slot, y_arr.copy()))

        # ── Estimate noise sigma ────────
        # Use first-difference MAD (Median Absolute Deviation / 0.6745 / sqrt(2)),
        # which is robust, insensitive to peaks/edges/cosmic rays, and 100x faster than polyfit loops
        _n = len(y_arr)
        if _n >= 10:
            diffs = np.diff(y_arr)
            med_diff = np.median(diffs)
            mad = np.median(np.abs(diffs - med_diff))
            noise_sigma = float(mad / (0.6745 * np.sqrt(2)))
        else:
            noise_sigma = 0.0

        # ── Linear interpolation across the erased region ──────────────────────
        x_left,  y_left  = x_arr[i_left],  y_arr[i_left]
        x_right, y_right = x_arr[i_right], y_arr[i_right]

        if x_right != x_left:
            slope = (y_right - y_left) / (x_right - x_left)
            y_interp = y_left + slope * (x_arr[mask] - x_left)
        else:
            y_interp = np.full(mask.sum(), (y_left + y_right) / 2.0)

        # Add Gaussian noise matching the local noise level
        if noise_sigma > 0:
            rng = np.random.default_rng()
            y_interp = y_interp + rng.normal(0.0, noise_sigma, size=y_interp.shape)

        y_arr[mask] = y_interp

        # Write modified array back into the viewer's tensor cache for immediate redraw
        if isinstance(Y_data, list):
            self._tensor_data["y"][viewer_slot] = y_arr
        else:
            Y_data[viewer_slot] = y_arr

        # Emit preview signal — md_row is the actual row in md.Y (important for Maps WS)
        self.cosmicRayErased.emit(md_row, y_arr)

        self._plot()


    def _erase_clear_highlight(self):
        """Remove the selection highlight span from the axes."""
        if self._erase_highlight is not None:
            try:
                self._erase_highlight.remove()
            except Exception:
                pass
            self._erase_highlight = None
            self.canvas.draw_idle()

    def _erase_undo(self):
        """Undo the last erase operation."""
        if not self._erase_history:
            return

        spec_idx, y_snapshot = self._erase_history.pop()

        if not self._tensor_data:
            return

        Y_data = self._tensor_data.get("y")
        if Y_data is None:
            return

        if isinstance(Y_data, list):
            if spec_idx < len(Y_data):
                self._tensor_data["y"][spec_idx] = y_snapshot
        else:
            Y_data[spec_idx] = y_snapshot

        self._plot()

    def _erase_validate(self):
        """Accept all erasures and persist them to the model."""
        # Clear undo history — changes are now final
        self._erase_history.clear()

        # Emit validation signal so the ViewModel can persist Y and Y0
        self.cosmicRayValidated.emit()

        # Exit eraser mode
        self.btn_eraser.blockSignals(True)
        self.btn_eraser.setChecked(False)
        self.btn_eraser.blockSignals(False)
        self._toggle_erase_mode(False)

    def _update_norm_button_style(self):
        """Update norm toggle button styling based on current theme tokens."""
        tok = current_tokens()
        self.btn_norm.setStyleSheet(
            f"QToolButton#btn_norm:checked {{ background-color: {tok.warning_tint}; border: 1px solid {tok.warning}; }}"
        )

    def apply_global_theme(self, global_theme: str):
        """Update SpectraViewer's icons based on the global application theme, independently of the plot theme."""
        tok = current_tokens()
        icon_color = tok.text_primary
        self._update_norm_button_style()
        if hasattr(self, 'btn_rescale'):
            self.btn_rescale.setIcon(Icons.rescale(color=icon_color))
            self.btn_zoom.setIcon(Icons.zoom(color=icon_color))
            self.btn_baseline.setIcon(Icons.baseline(color=icon_color))
            self.btn_peak.setIcon(Icons.peak(color=icon_color))
            self.btn_norm.setIcon(Icons.norm(color=icon_color))
            self.btn_bestfit.setIcon(Icons.bestfit(color=icon_color))
            self.btn_legend.setIcon(Icons.legend(color=icon_color))
            self.btn_copy.setIcon(Icons.copy(color=icon_color))
            self.btn_options.setIcon(Icons.settings(color=icon_color))
            self.btn_heatmap.setIcon(Icons.map_2d(color=icon_color))
        
        if hasattr(self, 'btn_eraser'):
            self.btn_eraser.setIcon(Icons.eraser(color=icon_color))
            self.btn_erase_undo.setIcon(Icons.undo(color=icon_color))
            self.btn_erase_validate.setIcon(Icons.check(color=icon_color))
