"""Main SpectraViewer widget: canvas layout, toolbar, options menu, and state sync."""

from __future__ import annotations

import warnings

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from spectroview import (
    PLOT_POLICY_DARK,
    PLOT_POLICY_LIGHT,
)
from spectroview.view.components.common.customized_widgets import (
    NoDoubleClickZoomToolbar,
)
from spectroview.view.components.rgb_live_maps.v_range_bracket_mixin import (
    SpectraRangeBracketMixin,
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
from spectroview.view.components.spectra_viewer.v_spectra_options import (
    SpectraOptionsMenuMixin,
)
from spectroview.view.theme import Icons, current_tokens
from spectroview.viewmodel.common.export_utils import copy_fig_to_clb

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
    SpectraOptionsMenuMixin,
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

        if hasattr(self, "canvas_card") and getattr(self, "figure", None) is not None:
            from matplotlib.colors import to_hex
            fig_hex = to_hex(self.figure.patch.get_facecolor())
            self.canvas_card.setStyleSheet(
                f"QFrame#canvasCard {{ background-color: {fig_hex}; border: 1px solid {tok.border}; border-radius: 8px; }}"
            )
