"""Main SpectraViewer widget: canvas layout, toolbar, options menu, and state sync."""

from __future__ import annotations

import warnings

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from spectroview import (
    PLOT_POLICY_DARK,
    PLOT_POLICY_LIGHT,
    X_AXIS_UNIT,
    Y_AXIS_UNIT,
)
from spectroview.view.components.customized_widgets import (
    CustomizedPalette,
    NoDoubleClickZoomToolbar,
)
from spectroview.view.components.rgb_live_maps.v_range_selector import (
    SpectraRangeBracketMixin,
)
from spectroview.view.components.spectra_viewer.common import (
    SPECTRA_COLOR_PALETTES,
    SPECTRA_CUSTOM_PALETTES,
    get_cached_style_dict,
)
from spectroview.view.components.spectra_viewer.v_interactions import (
    SpectraInteractionMixin,
)
from spectroview.view.components.spectra_viewer.v_plot_1d import SpectraPlot1DMixin
from spectroview.view.components.spectra_viewer.v_plot_3d import SpectraPlot3DMixin
from spectroview.view.components.spectra_viewer.v_plot_heatmap import (
    SpectraHeatmapMixin,
)
from spectroview.view.components.spectra_viewer.v_spectra_legend_editor import (
    SpectraLegendMixin,
)
from spectroview.view.theme import Icons, current_tokens
from spectroview.viewmodel.utils import copy_fig_to_clb

# Suppress harmless Matplotlib constrained_layout and log-limit transition warnings
warnings.filterwarnings(
    "ignore",
    message=".*constrained_layout not applied because axes sizes collapsed to zero.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*Attempt to set non-positive .*lim on a log-scaled axis.*",
    category=UserWarning,
)


class VSpectraViewer(
    QWidget,
    SpectraPlot1DMixin,
    SpectraHeatmapMixin,
    SpectraPlot3DMixin,
    SpectraLegendMixin,
    SpectraInteractionMixin,
    SpectraRangeBracketMixin,
):
    """Interactive spectral canvas supporting 1D overlays, 2D heatmaps, and 3D views."""

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
    range_drag_finished = Signal(str)          # (color_id) emitted on mouse release

    def __init__(self, parent=None):
        super().__init__(parent)
        self._colorbar_param_selection = {
            "text": "Index",
            "pos": None,
            "pattern": "Index",
        }
        self._heatmap_yaxis_selection = {
            "text": "Index",
            "pos": None,
            "pattern": "Index",
        }
        self._init_ui()
        self._current_spectra = []
        self._tensor_data = None

        self._fitted_lines = []     # [(line, peak_info)]
        self._highlighted_line = None
        self._dragging_peak = None  # Stores (line, info) when dragging

        self.zoom_pan_active = True
        app_inst = QApplication.instance()
        if app_inst is not None:
            app_inst.focusChanged.connect(self._hide_tooltip)

        # ── Heatmap mode state ──
        self._heatmap_mode = False
        self._heatmap_colorbar = None  # Reference to colorbar for removal

        # ── 3D view mode state ──
        self._3d_mode = False
        self._reset_3d_view = False
        self._default_3d_elev = 25.0
        self._default_3d_azim = -60.0

        # ── Cosmic-ray eraser state ──
        self._erase_mode = False
        self._erase_drag_start_x = None
        self._erase_highlight = None
        self._erase_history = []
        self._erase_cids = []

        # ── Spectral Range Selector state & manager ──
        self._init_range_brackets()

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
        self.canvas.mpl_connect("figure_leave_event", self._on_figure_leave)

        # Matplotlib toolbar kept hidden as a background helper for zoom and rubberband
        self.toolbar = NoDoubleClickZoomToolbar(self.canvas, self)
        self.toolbar.hide()
        self.toolbar.zoom()  # Start with zoom enabled

        # Forward any matplotlib toolbar message events to our built-in coordinate label
        def _on_toolbar_message(s):
            clean_s = " ".join(s.split()) if s else ""
            if hasattr(self, "lbl_coordinates"):
                self.lbl_coordinates.setText(clean_s)
            if hasattr(self.toolbar, "locLabel"):
                self.toolbar.locLabel.setText(s)

        self.toolbar.set_message = _on_toolbar_message

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
        self.btn_eraser.setToolTip(
            "Erase Cosmic Ray - click and drag over a spike to remove it"
        )
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
        layout.addWidget(self._create_range_selector_panel(panel), 0, Qt.AlignHCenter)

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
        self.btn_norm.setToolTip(
            "Intensity normalization to maximum of spectrum or selected region.\n"
            "Note: This normalization is just for visual change, the actual data is not changed."
        )
        self.btn_norm.setIconSize(QSize(22, 22))
        self.btn_norm.setFixedSize(30, 30)
        self.btn_norm.toggled.connect(self._plot)
        self.btn_norm.clicked.connect(self._rescale)
        layout.addWidget(self.btn_norm)

        self.norm_xmin = QLineEdit()
        self.norm_xmin.setFixedSize(50, 30)
        self.norm_xmin.setPlaceholderText("Xmin")
        self.norm_xmin.setToolTip(
            "Intensity normalization to maximum of spectrum or selected region"
        )
        layout.addWidget(self.norm_xmin)

        self.norm_xmax = QLineEdit()
        self.norm_xmax.setFixedSize(50, 30)
        self.norm_xmax.setPlaceholderText("Xmax")
        self.norm_xmax.setToolTip(
            "Intensity normalization to maximum of spectrum or selected region"
        )
        layout.addWidget(self.norm_xmax)

        self.norm_xmin.editingFinished.connect(self._on_norm_range_edited)
        self.norm_xmax.editingFinished.connect(self._on_norm_range_edited)

        # Bestfit
        self.btn_bestfit = QToolButton()
        self.btn_bestfit.setCheckable(True)
        self.btn_bestfit.setChecked(True)
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
            "all spectrum labels and colors."
        )
        self.btn_legend.setIconSize(QSize(22, 22))
        self.btn_legend.setFixedSize(30, 30)
        self.btn_legend.toggled.connect(self._emit_view_options)
        layout.addWidget(self.btn_legend)

        # Heatmap / 3D Y-axis controls (hidden until heatmap or 3D mode is active)
        self.cbb_heatmap_yaxis = QComboBox()
        self.cbb_heatmap_yaxis.setFixedHeight(30)
        self.cbb_heatmap_yaxis.addItem("Index", None)
        self.cbb_heatmap_yaxis.setToolTip(
            "Heatmap Y-axis: choose parameter for vertical axis.\n"
            "Index = spectrum order, or select a token parsed from filenames."
        )
        self.cbb_heatmap_yaxis.currentIndexChanged.connect(
            self._on_heatmap_yaxis_changed
        )
        self.cbb_heatmap_yaxis.setVisible(False)
        layout.addWidget(self.cbb_heatmap_yaxis)

        self.le_heatmap_ylabel = QLineEdit()
        self.le_heatmap_ylabel.setFixedHeight(30)
        self.le_heatmap_ylabel.setFixedWidth(80)
        self.le_heatmap_ylabel.setPlaceholderText("Y label")
        self.le_heatmap_ylabel.setToolTip(
            "Custom label for the heatmap Y-axis.\n"
            "Leave empty for the automatic token name.\n"
            "Example: Temperature (°C)"
        )
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
            "rendering. Set to 0 to disable interpolation (raw rows only)."
        )
        self.spin_heatmap_interp.valueChanged.connect(self._emit_view_options)
        self.spin_heatmap_interp.setVisible(False)
        layout.addWidget(self.spin_heatmap_interp)

        layout.addStretch()

        # Built-in live coordinates display
        self.lbl_coordinates = QLabel("")
        self.lbl_coordinates.setObjectName("lblCoordinates")
        self.lbl_coordinates.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_coordinates.setMinimumWidth(180)
        self.lbl_coordinates.setStyleSheet("font-size: 11px; padding: 0px 4px;")
        layout.addWidget(self.lbl_coordinates)
        # R²
        self.lbl_r2 = QLabel("R²=0")
        layout.addWidget(self.lbl_r2)
        # Noise level
        self.lbl_noise = QLabel("Noise=0")
        layout.addWidget(self.lbl_noise)

        # Vertical separator between navigation/coordinates and right-side mode/options/copy actions
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # Right-side action group: Plot Heatmap, 3D View, More view options (Gear), Copy
        self.btn_heatmap = QToolButton()
        self.btn_heatmap.setCheckable(True)
        self.btn_heatmap.setIcon(Icons.map_2d())
        self.btn_heatmap.setToolTip(
            "Plot Heatmap — display multiple spectra as a 2D intensity map.\n"
            "X-axis: spectral axis, Y-axis: varying parameter (see options), "
            "Color: intensity."
        )
        self.btn_heatmap.setIconSize(QSize(22, 22))
        self.btn_heatmap.setFixedSize(30, 30)
        self.btn_heatmap.toggled.connect(self._toggle_heatmap_mode)
        layout.addWidget(self.btn_heatmap)

        self.btn_3d = QToolButton()
        self.btn_3d.setCheckable(True)
        self.btn_3d.setIcon(Icons.view_3d())
        self.btn_3d.setToolTip(
            "3D View — display selected spectra in 3D.\n"
            "Drag with the mouse to rotate and explore viewing angles."
        )
        self.btn_3d.setIconSize(QSize(22, 22))
        self.btn_3d.setFixedSize(30, 30)
        self.btn_3d.toggled.connect(self._toggle_3d_mode)
        layout.addWidget(self.btn_3d)

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
        self.btn_copy.setToolTip(
            "Copy figure to clipboard. Hold Ctrl + Click to copy data to clipboard"
        )
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

    # Options menu
    def _wrap_menu_row(
        self, parent_widget: QWidget, label: str, widget: QWidget
    ) -> QWidget:
        """Build a compact label + control row for the 2-column options menu."""
        row_widget = QWidget(parent_widget)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(2, 1, 2, 1)
        row_layout.setSpacing(6)
        row_layout.addWidget(QLabel(label, row_widget))
        row_layout.addWidget(widget, stretch=1)
        return row_widget

    @staticmethod
    def _menu_h_separator(parent_widget: QWidget) -> QFrame:
        """Return a hairline horizontal separator for the 2-column options menu."""
        line = QFrame(parent_widget)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setFixedHeight(1)
        return line

    def _bind_menu_checkbox(
        self,
        parent_widget: QWidget,
        menu: QMenu,
        text: str,
        checked: bool,
        on_toggled,
    ):
        """Create a compact QCheckBox synchronized with a programmatic QAction."""
        act = QAction(text, menu)
        act.setCheckable(True)
        act.setChecked(bool(checked))

        chk = QCheckBox(text, parent_widget)
        chk.setChecked(bool(checked))

        def _on_act_toggled(val: bool):
            val = bool(val)
            if chk.isChecked() != val:
                chk.blockSignals(True)
                chk.setChecked(val)
                chk.blockSignals(False)
            on_toggled(val)

        def _on_chk_toggled(val: bool):
            val = bool(val)
            if act.isChecked() != val:
                act.setChecked(val)

        act.toggled.connect(_on_act_toggled)
        chk.toggled.connect(_on_chk_toggled)
        return act, chk

    def _create_options_menu(self):
        """Build a compact 2-column 'More view options' menu matching VLiveMaps spacing."""
        menu = QMenu(self)
        panel = QWidget(menu)
        self._options_panel = panel

        outer_layout = QVBoxLayout(panel)
        outer_layout.setContentsMargins(6, 4, 6, 4)
        outer_layout.setSpacing(2)

        cols_layout = QHBoxLayout()
        cols_layout.setContentsMargins(0, 0, 0, 0)
        cols_layout.setSpacing(8)

        left_col = QWidget(panel)
        left_layout = QVBoxLayout(left_col)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)

        right_col = QWidget(panel)
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        # ── Left Column: Theme, Axes, Plot Style, Palette, Line/Dot Sizes & Copy Options ──
        self.cbb_theme = QComboBox(left_col)
        self.cbb_theme.addItems(["Light Mode", "Dark Mode"])
        self.cbb_theme.currentIndexChanged.connect(self._apply_plot_style)
        left_layout.addWidget(
            self._wrap_menu_row(left_col, "Plot Theme:", self.cbb_theme)
        )

        self.cbb_xaxis = QComboBox(left_col)
        self.cbb_xaxis.addItems(X_AXIS_UNIT)
        self.cbb_xaxis.currentIndexChanged.connect(self._emit_view_options)
        left_layout.addWidget(
            self._wrap_menu_row(left_col, "X-axis:", self.cbb_xaxis)
        )

        self.cbb_yaxis = QComboBox(left_col)
        self.cbb_yaxis.addItems(Y_AXIS_UNIT)
        self.cbb_yaxis.currentIndexChanged.connect(self._emit_view_options)
        left_layout.addWidget(
            self._wrap_menu_row(left_col, "Y-axis:", self.cbb_yaxis)
        )

        self.cbb_yscale = QComboBox(left_col)
        self.cbb_yscale.addItems(["Linear", "Log"])
        self.cbb_yscale.currentIndexChanged.connect(self._emit_view_options)
        left_layout.addWidget(
            self._wrap_menu_row(left_col, "Y-scale:", self.cbb_yscale)
        )

        left_layout.addWidget(self._menu_h_separator(left_col))

        self.cbb_plotstyle = QComboBox(left_col)
        self.cbb_plotstyle.addItems(["line", "dot"])
        self.cbb_plotstyle.currentIndexChanged.connect(self._emit_view_options)
        left_layout.addWidget(
            self._wrap_menu_row(left_col, "Spectrum plot style:", self.cbb_plotstyle)
        )

        self.cbb_color_palette = CustomizedPalette(
            palette_list=SPECTRA_COLOR_PALETTES,
            custom_palettes=SPECTRA_CUSTOM_PALETTES,
            parent=left_col,
        )
        self.cbb_color_palette.setCurrentText("DEFAULT_COLORS")
        self.cbb_color_palette.setToolTip(
            "Choose colors for the selected spectrum series. Custom colors "
            "set from the legend continue to take priority."
        )
        self.cbb_color_palette.currentIndexChanged.connect(self._emit_view_options)
        left_layout.addWidget(
            self._wrap_menu_row(left_col, "Color palette:", self.cbb_color_palette)
        )

        self.spin_lw = QDoubleSpinBox(left_col)
        self.spin_lw.setRange(0.1, 5)
        self.spin_lw.setSingleStep(0.5)
        self.spin_lw.setValue(1.5)
        self.spin_lw.valueChanged.connect(self._emit_view_options)
        left_layout.addWidget(
            self._wrap_menu_row(left_col, "Line width:", self.spin_lw)
        )

        self.spin_dotsize = QDoubleSpinBox(left_col)
        self.spin_dotsize.setRange(0.5, 10)
        self.spin_dotsize.setSingleStep(0.5)
        self.spin_dotsize.setValue(3)
        self.spin_dotsize.valueChanged.connect(self._emit_view_options)
        left_layout.addWidget(
            self._wrap_menu_row(left_col, "Dot size:", self.spin_dotsize)
        )

        left_layout.addWidget(self._menu_h_separator(left_col))

        ratio_widget = QWidget(left_col)
        ratio_layout = QHBoxLayout(ratio_widget)
        ratio_layout.setContentsMargins(2, 1, 2, 1)
        ratio_layout.setSpacing(4)
        ratio_layout.addWidget(QLabel("Copied figure size:", ratio_widget))
        ratio_layout.addStretch(1)

        self.width_entry = QLineEdit(ratio_widget)
        self.width_entry.setFixedWidth(40)
        self.width_entry.setText("5.5")
        self.width_entry.editingFinished.connect(self._emit_view_options)
        ratio_layout.addWidget(self.width_entry)

        self.height_entry = QLineEdit(ratio_widget)
        self.height_entry.setFixedWidth(40)
        self.height_entry.setText("4.0")
        self.height_entry.editingFinished.connect(self._emit_view_options)
        ratio_layout.addWidget(self.height_entry)
        left_layout.addWidget(ratio_widget)

        self.cbb_copy_theme = QComboBox(left_col)
        self.cbb_copy_theme.addItems(["Light Mode", "Dark Mode"])
        self.cbb_copy_theme.currentIndexChanged.connect(self._emit_view_options)
        left_layout.addWidget(
            self._wrap_menu_row(left_col, "Copied figure theme:", self.cbb_copy_theme)
        )
        left_layout.addStretch(1)

        # ── Right Column: Display Toggles & Legend / Colorbar Settings ──
        self.act_raw, self.chk_raw = self._bind_menu_checkbox(
            right_col, menu, "Raw", False, lambda _v: self._emit_view_options()
        )
        right_layout.addWidget(self.chk_raw)

        self.act_bestfit_colorful, self.chk_bestfit_colorful = self._bind_menu_checkbox(
            right_col,
            menu,
            "Bestfit curve colorful",
            True,
            lambda _v: self._emit_view_options(),
        )
        right_layout.addWidget(self.chk_bestfit_colorful)

        self.act_show_peak_label, self.chk_show_peak_label = self._bind_menu_checkbox(
            right_col,
            menu,
            "Show peak label",
            False,
            lambda _v: self._emit_view_options(),
        )
        right_layout.addWidget(self.chk_show_peak_label)

        self.act_residual, self.chk_residual = self._bind_menu_checkbox(
            right_col, menu, "Residual", False, lambda _v: self._emit_view_options()
        )
        right_layout.addWidget(self.chk_residual)

        self.act_grid, self.chk_grid = self._bind_menu_checkbox(
            right_col, menu, "Grid", False, lambda _v: self._emit_view_options()
        )
        right_layout.addWidget(self.chk_grid)

        self.act_noise_level, self.chk_noise_level = self._bind_menu_checkbox(
            right_col,
            menu,
            "Show noise level",
            False,
            lambda _v: self._emit_view_options(),
        )
        right_layout.addWidget(self.chk_noise_level)

        right_layout.addWidget(self._menu_h_separator(right_col))

        self.spin_max_legend_items = QSpinBox(right_col)
        self.spin_max_legend_items.setRange(1, 1000)
        self.spin_max_legend_items.setValue(15)
        self.spin_max_legend_items.valueChanged.connect(self._emit_view_options)
        right_layout.addWidget(
            self._wrap_menu_row(
                right_col, "Max legend items:", self.spin_max_legend_items
            )
        )
        self.spin_max_overlays = self.spin_max_legend_items

        self.act_legend_outside, self.chk_legend_outside = self._bind_menu_checkbox(
            right_col,
            menu,
            "Move legend box outside",
            False,
            lambda _v: self._emit_view_options(),
        )
        right_layout.addWidget(self.chk_legend_outside)

        self.act_legend_colorbar, self.chk_legend_colorbar = self._bind_menu_checkbox(
            right_col,
            menu,
            "Show colorbar instead of legend",
            False,
            self._on_legend_colorbar_toggled,
        )
        right_layout.addWidget(self.chk_legend_colorbar)

        # Colorbar sub-options (shown only when "Show colorbar instead of legend" is checked)
        self.cbb_colorbar_param = QComboBox(right_col)
        self.cbb_colorbar_param.addItem("Index", None)
        self.cbb_colorbar_param.setToolTip(
            "Colorbar parameter: choose parameter for the colorbar.\n"
            "Index = spectrum order, or select a token parsed from filenames."
        )
        self.cbb_colorbar_param.currentIndexChanged.connect(
            self._on_colorbar_param_changed
        )
        self.row_colorbar_param = self._wrap_menu_row(
            right_col, "Colorbar parameter:", self.cbb_colorbar_param
        )
        right_layout.addWidget(self.row_colorbar_param)

        self.le_colorbar_label = QLineEdit(right_col)
        self.le_colorbar_label.setPlaceholderText("Colorbar label")
        self.le_colorbar_label.setToolTip(
            "Custom label for the spectrum colorbar.\n"
            "Leave empty for the automatic token name.\n"
            "Example: Temperature (°C)"
        )
        self.le_colorbar_label.textChanged.connect(self._on_heatmap_ylabel_changed)
        self.row_colorbar_label = self._wrap_menu_row(
            right_col, "Colorbar label:", self.le_colorbar_label
        )
        right_layout.addWidget(self.row_colorbar_label)
        right_layout.addStretch(1)

        v_sep = QFrame(panel)
        v_sep.setFrameShape(QFrame.VLine)
        v_sep.setFrameShadow(QFrame.Sunken)

        cols_layout.addWidget(left_col, stretch=1)
        cols_layout.addWidget(v_sep)
        cols_layout.addWidget(right_col, stretch=1)
        outer_layout.addLayout(cols_layout)

        container_action = QWidgetAction(menu)
        container_action.setDefaultWidget(panel)
        menu.addAction(container_action)

        # Hide colorbar parameter & label rows until colorbar mode is checked
        self._update_colorbar_options_visibility(False)

        return menu

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────
    def set_plot_data(self, data):
        if isinstance(data, dict) and data.get("type") in ["tensor", "tensor_list"]:
            self._tensor_data = data
            self._current_spectra = data.get("proxies", [])
            self._sync_ranges_with_plot_data(data.get("x"))
        else:
            self._tensor_data = None
            self._current_spectra = []

        # Keep parameter token choices up-to-date with the current selection
        self._refresh_colorbar_param_choices()
        if getattr(self, "_heatmap_mode", False) or getattr(self, "_3d_mode", False):
            self._refresh_heatmap_yaxis_choices()

        self._plot()

    def _on_norm_range_edited(self):
        """Unified handler when norm bounds change: plot and rescale once."""
        self._plot()
        self._rescale()

    def _plot(self):
        style_name = self.cbb_theme.currentText()
        if style_name == "Dark Mode":
            style_path = PLOT_POLICY_DARK
        else:
            style_path = PLOT_POLICY_LIGHT
        with plt.style.context(style_path):
            self._plot_internal()

    def _plot_internal(self):
        if self._range_drag_bg is not None or self._suspended_range_layout is not None:
            self._end_range_drag_blit()
        if not self._tensor_data:
            self._ensure_axes_projection(bool(getattr(self, "_3d_mode", False)))
            self.ax.clear()
            self._remove_heatmap_colorbar()
            if getattr(self, "_3d_mode", False):
                self._apply_3d_axes_style()
                self.ax.set_navigate_mode(None)
            self.lbl_r2.setText("R²=0")
            self.canvas.draw_idle()
            return

        # ── Heatmap branch ──
        if self._heatmap_mode:
            self._ensure_axes_projection(False)
            self._plot_heatmap()
            return

        # ── 3D view branch ──
        if getattr(self, "_3d_mode", False):
            self._ensure_axes_projection(True)
            self._plot_3d()
            return

        self._ensure_axes_projection(False)

        # Save current zoom/pan limits so they survive a full ax.clear()
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        is_default = xlim == (0.0, 1.0) and ylim == (0.0, 1.0)

        fg_color = plt.rcParams.get("axes.labelcolor", "black")

        self.ax.clear()
        self._remove_heatmap_colorbar()
        self._fitted_lines.clear()

        self._update_r2_display()

        x = self._tensor_data["x"]
        Y = self._tensor_data["y"]
        Y_norm = self._get_normalized_y_tensor(x, Y)
        x_shift_step, y_shift_step = self._compute_shift_steps(Y_norm=Y_norm)
        plot_style = self.cbb_plotstyle.currentText()
        lw = self.spin_lw.value()
        dot_size = self.spin_dotsize.value()
        if self._is_legend_colorbar_active():
            param_values, _cbar_label = self._get_colorbar_values()
            _cmap, _norm, colors_cycle = self._resolve_colorbar_cmap_and_norm(
                len(Y), param_values
            )
        else:
            colors_cycle = self._get_colors_cycle(len(Y))

        # ── Step 1: Build bulk segment data (main spectra + raw overlay) ──
        segments = self._build_tensor_segments(
            x_shift_step,
            y_shift_step,
            plot_style,
            lw,
            fg_color,
            colors_cycle,
            Y_norm=Y_norm,
        )

        # ── Step 2: Draw heavy per-spectrum overlays ──
        self._draw_tensor_overlays(x_shift_step, y_shift_step, lw, fg_color)

        # ── Step 3: Commit everything to the canvas ──
        self._finalize_plot(
            segments, plot_style, lw, dot_size, fg_color, xlim, ylim, is_default
        )

    def _sync_mode_toolbar_state(self):
        """Synchronize toolbar button visibility, eraser state, and zoom mode across 2D/Heatmap/3D."""
        is_heatmap = bool(getattr(self, "_heatmap_mode", False))
        is_3d = bool(getattr(self, "_3d_mode", False))
        is_standard = not is_heatmap and not is_3d

        if is_heatmap or is_3d:
            self._refresh_heatmap_yaxis_choices()
            if getattr(self, "_erase_mode", False):
                self.btn_eraser.setChecked(False)

        if is_3d:
            self.btn_zoom.blockSignals(True)
            self.btn_zoom.setChecked(False)
            self.btn_zoom.blockSignals(False)
            self._set_zoom_tool_active(False)
            for b in (self.btn_peak, self.btn_baseline):
                if b.isChecked():
                    b.blockSignals(True)
                    b.setChecked(False)
                    b.blockSignals(False)
        else:
            self.btn_zoom.blockSignals(True)
            self.btn_zoom.setChecked(True)
            self.btn_zoom.blockSignals(False)
            self._set_zoom_tool_active(True)
            for b in (self.btn_peak, self.btn_baseline):
                if b.isChecked():
                    b.blockSignals(True)
                    b.setChecked(False)
                    b.blockSignals(False)

        self.btn_zoom.setVisible(not is_3d)
        self.btn_peak.setVisible(is_standard)
        self.btn_baseline.setVisible(is_standard)
        self.btn_eraser.setVisible(is_standard)
        self.shift_panel.setVisible(is_standard)

        self.btn_norm.setVisible(not is_heatmap)
        self.norm_xmin.setVisible(not is_heatmap)
        self.norm_xmax.setVisible(not is_heatmap)
        self.btn_bestfit.setVisible(not is_heatmap)

        self.cbb_heatmap_yaxis.setVisible(is_heatmap or is_3d)
        self.le_heatmap_ylabel.setVisible(is_heatmap or is_3d)
        self.spin_heatmap_interp.setVisible(is_heatmap)

        if not is_heatmap:
            self._remove_heatmap_colorbar()

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
            "legend_colorbar": self.act_legend_colorbar.isChecked() if hasattr(self, "act_legend_colorbar") else False,
            "colorbar_param": self.cbb_colorbar_param.currentText() if hasattr(self, "cbb_colorbar_param") else "Index",
            "colorbar_label": self.le_colorbar_label.text() if hasattr(self, "le_colorbar_label") else "",
            "bestfit": self.btn_bestfit.isChecked() if hasattr(self, "btn_bestfit") else True,
            "max_legend_items": self.spin_max_legend_items.value() if hasattr(self, "spin_max_legend_items") else 15,
            "copy_fig_theme": self.cbb_copy_theme.currentText() if hasattr(self, "cbb_copy_theme") else "Light Mode",
            "heatmap": self.btn_heatmap.isChecked() if hasattr(self, "btn_heatmap") else False,
            "view_3d": self.btn_3d.isChecked() if hasattr(self, "btn_3d") else False,
            "heatmap_yaxis": self.cbb_heatmap_yaxis.currentText() if hasattr(self, "cbb_heatmap_yaxis") else "Index",
            "heatmap_ylabel": self.le_heatmap_ylabel.text() if hasattr(self, "le_heatmap_ylabel") else "",
            "heatmap_interp": self.spin_heatmap_interp.value() if hasattr(self, "spin_heatmap_interp") else 300,
        }

    def set_options_state(self, state):
        """Set the complete state of all view options without causing infinite loops."""
        if not hasattr(self, "cbb_theme"):
            return

        current_state = self.get_options_state()
        if current_state == state and getattr(self, "_has_applied_plot_style", False):
            return

        self._is_setting_options = True

        def _update(widget, setter, value):
            if value is None:
                return
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
            _update(
                self.cbb_color_palette,
                self.cbb_color_palette.setCurrentText,
                state.get("color_palette", "DEFAULT_COLORS"),
            )
        _update(self.spin_lw, self.spin_lw.setValue, state.get("lw"))
        _update(self.spin_dotsize, self.spin_dotsize.setValue, state.get("dotsize"))
        _update(self.act_raw, self.act_raw.setChecked, state.get("raw"))
        _update(
            self.act_bestfit_colorful,
            self.act_bestfit_colorful.setChecked,
            state.get("bestfit_colorful"),
        )
        _update(
            self.act_show_peak_label,
            self.act_show_peak_label.setChecked,
            state.get("show_peak_label"),
        )
        _update(self.act_residual, self.act_residual.setChecked, state.get("residual"))
        _update(self.act_grid, self.act_grid.setChecked, state.get("grid"))
        if hasattr(self, "act_noise_level"):
            _update(
                self.act_noise_level,
                self.act_noise_level.setChecked,
                state.get("noise_level"),
            )
        _update(self.width_entry, self.width_entry.setText, state.get("width"))
        _update(self.height_entry, self.height_entry.setText, state.get("height"))
        _update(self.btn_legend, self.btn_legend.setChecked, state.get("legend"))
        if hasattr(self, "act_legend_outside"):
            _update(
                self.act_legend_outside,
                self.act_legend_outside.setChecked,
                state.get("legend_outside", False),
            )
        if hasattr(self, "act_legend_colorbar"):
            _update(
                self.act_legend_colorbar,
                self.act_legend_colorbar.setChecked,
                state.get("legend_colorbar", False),
            )
        if hasattr(self, "cbb_colorbar_param"):
            cb_param = state.get("colorbar_param", "Index")
            if cb_param:
                if self.cbb_colorbar_param.findText(cb_param) < 0:
                    self.cbb_colorbar_param.blockSignals(True)
                    self.cbb_colorbar_param.addItem(cb_param, None)
                    self.cbb_colorbar_param.blockSignals(False)
                _update(
                    self.cbb_colorbar_param,
                    self.cbb_colorbar_param.setCurrentText,
                    cb_param,
                )
            self._record_token_selection(
                self.cbb_colorbar_param, self._colorbar_param_selection
            )
            if self._tensor_data:
                self._refresh_colorbar_param_choices()
        if hasattr(self, "le_colorbar_label"):
            _update(
                self.le_colorbar_label,
                self.le_colorbar_label.setText,
                state.get("colorbar_label", ""),
            )
        _update(self.btn_bestfit, self.btn_bestfit.setChecked, state.get("bestfit"))
        if hasattr(self, "spin_max_legend_items"):
            _update(
                self.spin_max_legend_items,
                self.spin_max_legend_items.setValue,
                state.get("max_legend_items", 15),
            )
        if hasattr(self, "cbb_copy_theme"):
            _update(
                self.cbb_copy_theme,
                self.cbb_copy_theme.setCurrentText,
                state.get("copy_fig_theme", "Light Mode"),
            )
        want_heatmap = bool(state.get("heatmap", False))
        want_3d = bool(state.get("view_3d", False)) and not want_heatmap
        if hasattr(self, "btn_heatmap"):
            _update(self.btn_heatmap, self.btn_heatmap.setChecked, want_heatmap)
        if hasattr(self, "btn_3d"):
            _update(self.btn_3d, self.btn_3d.setChecked, want_3d)
        mode_changed = (
            getattr(self, "_heatmap_mode", False) != want_heatmap
            or getattr(self, "_3d_mode", False) != want_3d
        )
        self._heatmap_mode = want_heatmap
        self._3d_mode = want_3d
        if want_3d and mode_changed:
            self._reset_3d_view = True
        if mode_changed:
            self._sync_mode_toolbar_state()

        if hasattr(self, "cbb_heatmap_yaxis"):
            hm_yaxis = state.get("heatmap_yaxis", "Index")
            if hm_yaxis:
                if self.cbb_heatmap_yaxis.findText(hm_yaxis) < 0:
                    self.cbb_heatmap_yaxis.blockSignals(True)
                    self.cbb_heatmap_yaxis.addItem(hm_yaxis, None)
                    self.cbb_heatmap_yaxis.blockSignals(False)
                _update(
                    self.cbb_heatmap_yaxis,
                    self.cbb_heatmap_yaxis.setCurrentText,
                    hm_yaxis,
                )
            self._record_token_selection(
                self.cbb_heatmap_yaxis, self._heatmap_yaxis_selection
            )
            if self._tensor_data and (
                getattr(self, "_heatmap_mode", False)
                or getattr(self, "_3d_mode", False)
            ):
                self._refresh_heatmap_yaxis_choices()
        if hasattr(self, "le_heatmap_ylabel"):
            _update(
                self.le_heatmap_ylabel,
                self.le_heatmap_ylabel.setText,
                state.get("heatmap_ylabel", ""),
            )
        if hasattr(self, "spin_heatmap_interp"):
            _update(
                self.spin_heatmap_interp,
                self.spin_heatmap_interp.setValue,
                state.get("heatmap_interp", 300),
            )

        self._sync_menu_checkboxes()
        self._apply_plot_style()
        self._plot()
        self._is_setting_options = False

    def _sync_menu_checkboxes(self):
        """Synchronize 2-column menu QCheckBox widgets and conditional rows with QAction states."""
        pairs = (
            ("act_raw", "chk_raw"),
            ("act_bestfit_colorful", "chk_bestfit_colorful"),
            ("act_show_peak_label", "chk_show_peak_label"),
            ("act_residual", "chk_residual"),
            ("act_grid", "chk_grid"),
            ("act_noise_level", "chk_noise_level"),
            ("act_legend_outside", "chk_legend_outside"),
            ("act_legend_colorbar", "chk_legend_colorbar"),
        )
        for act_attr, chk_attr in pairs:
            act = getattr(self, act_attr, None)
            chk = getattr(self, chk_attr, None)
            if (
                act is not None
                and chk is not None
                and chk.isChecked() != act.isChecked()
            ):
                chk.blockSignals(True)
                chk.setChecked(act.isChecked())
                chk.blockSignals(False)
        if hasattr(self, "act_legend_colorbar"):
            self._update_colorbar_options_visibility(
                self.act_legend_colorbar.isChecked()
            )

    def _apply_plot_style(self):
        self._has_applied_plot_style = True
        style_name = self.cbb_theme.currentText()
        if style_name == "Dark Mode":
            style_path = PLOT_POLICY_DARK
        else:
            style_path = PLOT_POLICY_LIGHT

        # Parse the style file from cached dict without modifying global rcParams
        style_dict = get_cached_style_dict(style_path)

        fig_bg = style_dict.get("figure.facecolor", "white")
        self.figure.patch.set_facecolor(fig_bg)
        if hasattr(self, "canvas_card"):
            tok = current_tokens()
            fig_hex = (
                fig_bg
                if str(fig_bg).startswith("#") or str(fig_bg).isalpha()
                else f"#{fig_bg}"
            )
            self.canvas_card.setStyleSheet(
                f"QFrame#canvasCard {{ background-color: {fig_hex}; border: 1px solid {tok.border}; border-radius: 8px; }}"
            )
        self.figure.patch.set_edgecolor(style_dict.get("figure.edgecolor", "white"))
        if getattr(self.ax, "name", None) == "3d":
            self._apply_3d_axes_style()
        else:
            self._apply_2d_axes_style(style_dict)

        if not getattr(self, "_is_copying", False):
            self.plotStyleChanged.emit()
            if not getattr(self, "_is_setting_options", False):
                self.allOptionsSyncChanged.emit(self.get_options_state())
            self._plot()

    def _emit_view_options(self):
        if getattr(self, "_is_setting_options", False):
            return

        # Sync options with other viewers
        self.allOptionsSyncChanged.emit(self.get_options_state())

        # Update plot immediately whenever a view option changes
        self._plot()

    def _emit_copy(self):
        modifiers = QApplication.keyboardModifiers() & Qt.ControlModifier

        if modifiers:
            # Request ViewModel to copy spectrum data
            self.copy_data_requested.emit()
        else:
            width = float(self.width_entry.text()) if self.width_entry.text() else 5.5
            height = float(self.height_entry.text()) if self.height_entry.text() else 4.0

            target_theme = (
                self.cbb_copy_theme.currentText()
                if hasattr(self, "cbb_copy_theme")
                else "Light Mode"
            )
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

        self._flash_copy_feedback()

    def _flash_copy_feedback(self) -> None:
        """Provide 1-second visual checkmark feedback on copy button."""
        self.btn_copy.setIcon(Icons.check())
        QTimer.singleShot(1000, self, self._reset_copy_feedback)

    def _reset_copy_feedback(self) -> None:
        try:
            from spectroview.view.theme import ThemeManager

            icon_color = (
                ThemeManager.instance().icon_color if ThemeManager.instance() else None
            )
            self.btn_copy.setIcon(Icons.copy(color=icon_color))
        except (RuntimeError, AttributeError, TypeError):
            pass

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
        if hasattr(self, "btn_rescale"):
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
            self.btn_3d.setIcon(Icons.view_3d(color=icon_color))

        if hasattr(self, "lbl_coordinates"):
            self.lbl_coordinates.setStyleSheet(
                f"font-size: 11px; padding: 0px 4px; color: {tok.text_secondary};"
            )

        if hasattr(self, "btn_eraser"):
            self.btn_eraser.setIcon(Icons.eraser(color=icon_color))
            self.btn_erase_undo.setIcon(Icons.undo(color=icon_color))
            self.btn_erase_validate.setIcon(Icons.check(color=icon_color))
