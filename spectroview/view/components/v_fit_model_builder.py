# spectroview/view/components/v_fit_model_builder.py
from spectroview import ICON_DIR, PEAK_MODELS
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QGroupBox, QLabel, QPushButton, QComboBox,
    QDoubleSpinBox, QSpinBox, QRadioButton,
    QScrollArea, QCheckBox, QApplication, QSlider
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon

from fitspy.core.baseline_methods import (
    _INTERNAL_METHODS, _PYBASELINES_WHITELIST, get_baseline_method_meta
)

from spectroview import ICON_DIR, PEAK_MODELS
from spectroview.view.components.v_peak_table import VPeakTable


class VFitModelBuilder(QWidget):
    """View: Fit Model Builder panel."""
    # ───── View → ViewModel signals ─────
    peak_shape_changed = Signal(str)
    spectral_range_apply_requested = Signal(float, float, bool)

    baseline_settings_changed = Signal(dict)   # sent when mode/params change (persistent)
    baseline_preview_requested = Signal(dict)   # sent on slider/spin change (live preview)
    baseline_copy_requested = Signal()
    baseline_paste_requested = Signal(bool)     # apply_all
    baseline_subtract_requested = Signal(bool)  # apply_all
    baseline_delete_requested = Signal(bool)

    peaks_copy_requested = Signal()
    peaks_paste_requested = Signal(bool)    # apply_all
    peaks_delete_requested = Signal(bool)  # apply_all

    fitmodel_copy_requested = Signal()
    fitmodel_paste_requested = Signal(bool)
    fitmodel_save_requested = Signal()

    fit_requested = Signal(bool)  # apply_all

    load_fit_models_requested = Signal()
    refresh_fit_models_requested = Signal()
    apply_loaded_fit_model_requested = Signal(bool)  # apply_all

    #PeakTable signals: 
    peak_label_changed = Signal(int, str)
    peak_model_changed = Signal(int, str)
    peak_param_changed = Signal(int, str, str, float)
    peak_deleted = Signal(int)


    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal, self)
        main_layout.addWidget(splitter)

        # ==================================================
        # LEFT SIDE – Fit settings (scrollable)
        # ==================================================
        left_scroll = QScrollArea()
        left_scroll.setMaximumWidth(400) 
        left_scroll.setMinimumWidth(300) 
        left_scroll.setWidgetResizable(True)

        left_container = QWidget() # Set maximum width
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(8)

        left_layout.addWidget(self._x_correction_group())
        left_layout.addWidget(self._spectral_range_group())
        left_layout.addWidget(self._baseline_group())
        left_layout.addWidget(self._peaks_group())
        left_layout.addStretch()

        left_scroll.setWidget(left_container)
        splitter.addWidget(left_scroll)

        # ==================================================
        # RIGHT SIDE – Peak table + Fit controls
        # ==================================================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # ── Peak table in scroll area
        peak_scroll = QScrollArea()
        peak_scroll.setWidgetResizable(True)
        peak_scroll.setContentsMargins(0, 0, 0, 0)

        self.peak_table = VPeakTable()
        self.peak_table.peak_label_changed.connect(self.peak_label_changed.emit)
        self.peak_table.peak_model_changed.connect(self.peak_model_changed.emit)
        self.peak_table.peak_param_changed.connect(self.peak_param_changed.emit)
        self.peak_table.peak_deleted.connect(self.peak_deleted.emit)

        peak_scroll.setWidget(self.peak_table)

        right_layout.addWidget(peak_scroll, stretch=1)

        # ── Fit control panel (fixed)
        right_layout.addWidget(self._fit_control_panel(), stretch=0)

        splitter.addWidget(right_widget)

        # Initial sizes
        splitter.setSizes([390, 650])

    # ======================================================
    # GROUPS
    # ======================================================
    def _x_correction_group(self):
        gb = QGroupBox("X-axis correction")
        l = QHBoxLayout(gb)
        l.setContentsMargins(2, 2, 2, 2)

        # Reference selection
        self.cbb_xcorr = QComboBox()
        self.cbb_xcorr.addItems(["None", "Si-Ref"])
        self.cbb_xcorr.setCurrentText("Si-Ref")

        # User correction value
        self.spin_xcorr = QDoubleSpinBox()
        self.spin_xcorr.setRange(-100000, 100000)
        self.spin_xcorr.setFixedWidth(80)
        self.spin_xcorr.setDecimals(3)  
        self.spin_xcorr.setValue(0)
        self.spin_xcorr.setToolTip("Type measured reference peak position")

        # Label showing corrected value
        self.lbl_xcorr_value = QLabel("(0)")
        
        # Buttons
        self.btn_xcorrect = QPushButton("Correct")
        self.btn_xcorrect.setIcon(QIcon(f"{ICON_DIR}/done.png"))
        

        self.btn_undo_corr = QPushButton()
        self.btn_undo_corr.setIcon(QIcon(f"{ICON_DIR}/undo.png"))
        self.btn_undo_corr.setFixedSize(30, 24)
        self.btn_undo_corr.setToolTip("Undo X correction")

        # --- Layout order ---
        l.addWidget(self.cbb_xcorr)
        l.addWidget(self.spin_xcorr)
        l.addWidget(self.lbl_xcorr_value)
        l.addStretch()
        l.addWidget(self.btn_xcorrect)
        l.addWidget(self.btn_undo_corr)

        return gb

    def set_xcorrection_value(self, value: float):
        """Update label from Model state (xcorrection_value)"""
        if abs(value) < 1e-9:
            self.lbl_xcorr_value.setText("(0)")
        else:
            self.lbl_xcorr_value.setText(f"({value:+.2f})")

    def set_spectral_range(self, xmin, xmax):
        self.spin_xmin.blockSignals(True)
        self.spin_xmax.blockSignals(True)

        self.spin_xmin.setValue(xmin)
        self.spin_xmax.setValue(xmax)

        self.spin_xmin.blockSignals(False)
        self.spin_xmax.blockSignals(False) 

    def _spectral_range_group(self):
        gb = QGroupBox("Spectral range (X axis)")
        l = QHBoxLayout(gb)
        l.setContentsMargins(2, 2, 2, 2)

        self.spin_xmin = QDoubleSpinBox()
        self.spin_xmin.setFixedWidth(80)
        self.spin_xmax = QDoubleSpinBox()
        self.spin_xmax.setFixedWidth(80)
        self.spin_xmin.setDecimals(3)
        self.spin_xmax.setDecimals(3)

        self.spin_xmin.setRange(-1e9, 1e9)
        self.spin_xmax.setRange(-1e9, 1e9)

        self.btn_extract = QPushButton("Crop")
        self.btn_extract.setIcon(QIcon(f"{ICON_DIR}/cut.png"))
        self.btn_extract.setToolTip("Extract selected spectra range. Hold Ctrl to paste to all spectra.")

        self.btn_extract.clicked.connect(self._on_extract_clicked)

        l.addWidget(QLabel("Min/Max:"))
        l.addWidget(self.spin_xmin)
        l.addWidget(QLabel("/"))
        l.addWidget(self.spin_xmax)
        l.addStretch()
        l.addWidget(self.btn_extract)
        
        return gb
    
    def _on_extract_clicked(self):
        modifiers = QApplication.keyboardModifiers()
        apply_all = modifiers & Qt.ControlModifier

        self.spectral_range_apply_requested.emit(
            self.spin_xmin.value(),
            self.spin_xmax.value(),
            bool(apply_all)
    )


    _BASELINE_MODE_KEYS = []  # populated at runtime

    def _baseline_group(self):
        gb = QGroupBox("Baseline")
        v = QVBoxLayout(gb)
        v.setContentsMargins(2, 4, 2, 4)
        v.setSpacing(4)

        # ── Row 0: mode combobox ────────────────────────────────────
        row0 = QHBoxLayout()

        self.cbb_baseline_mode = QComboBox()
        self._populate_baseline_mode_combobox()
        self.cbb_baseline_mode.setToolTip(
            "Baseline correction mode.\n"
            "Manual modes (Linear/Polynomial): place anchor points on the plot.\n"
            "Auto modes (arPLS, airPLS, …): no anchor points needed; "
            "adjust the smoothness slider for a real-time preview."
        )

        row0.addWidget(QLabel("Mode:"))
        row0.addWidget(self.cbb_baseline_mode)
        row0.addStretch()

        # ── Row 1: manual controls ───────────────────────────────────
        # Shown for Linear and Polynomial modes. Contents vary based on mode:
        self._row1_widget = QWidget()
        row1 = QHBoxLayout(self._row1_widget)
        row1.setContentsMargins(0, 0, 0, 0)

        self.lbl_poly_order = QLabel("Order:")
        self.spin_poly = QSpinBox()
        self.spin_poly.setRange(2, 20)
        self.spin_poly.setFixedWidth(44)
        self.spin_poly.setToolTip("Polynomial order")

        self.chk_attached = QCheckBox("Attached")
        self.chk_attached.setChecked(True)
        self.chk_attached.setToolTip(
            "Snap baseline anchor points vertically to the spectrum profile"
        )

        self.lbl_sigma = QLabel("Corr. noise:")
        self.spin_noise = QSpinBox()
        self.spin_noise.setRange(0, 20)
        self.spin_noise.setValue(4)
        self.spin_noise.setFixedWidth(40)
        self.spin_noise.setToolTip(
            "Gaussian smoothing σ applied before attachment (noise correction)"
        )

        row1.addWidget(self.lbl_poly_order)
        row1.addWidget(self.spin_poly)
        row1.addStretch()
        row1.addWidget(self.chk_attached)
        row1.addWidget(self.lbl_sigma)
        row1.addWidget(self.spin_noise)

        # ── Row 2: auto controls ────────────────────────────────────
        self._row2_widget = QWidget()
        row2 = QHBoxLayout(self._row2_widget)
        row2.setContentsMargins(0, 0, 0, 0)

        # Coef slider: integer 10–100 maps to coef 1.0–10.0 (λ = 10^coef)
        self.sld_coef = QSlider(Qt.Horizontal)
        self.sld_coef.setFixedWidth(200)
        self.sld_coef.setRange(10, 100)   # ×10 for one decimal precision
        self.sld_coef.setValue(50)         # default coef = 5 → λ = 1e5
        self.sld_coef.setToolTip(
            "Smoothness parameter (λ = 10^coef).\n"
            "Move right for smoother / higher background insensitivity."
        )

        self.lbl_coef = QLabel("λ=1e5")
        self.lbl_coef.setFixedWidth(54)

        # Secondary parameter (order / sigma depending on method)
        self.lbl_coef2 = QLabel("Order:")
        self.spin_coef2 = QSpinBox()
        self.spin_coef2.setRange(1, 20)
        self.spin_coef2.setValue(1)
        self.spin_coef2.setFixedWidth(44)
        self.spin_coef2.setVisible(False)
        self.lbl_coef2.setVisible(False)

        row2.addWidget(QLabel("Coef:"))
        row2.addWidget(self.sld_coef)
        row2.addWidget(self.lbl_coef)
        row2.addWidget(self.lbl_coef2)
        row2.addWidget(self.spin_coef2)
        row2.addStretch()

        # ── Row 3: shared action buttons ──────────────────────────────
        row3 = QHBoxLayout()

        self.btn_base_delete = QPushButton()
        self.btn_base_delete.setIcon(QIcon(f"{ICON_DIR}/trash3.png"))
        self.btn_base_delete.setFixedSize(30, 24)
        self.btn_base_delete.setToolTip(
            "Undo baseline subtraction (reinitialise + re-crop spectrum).\n"
            "Hold Ctrl to apply to all spectra."
        )
        self.btn_base_delete.clicked.connect(
            lambda: self._emit_with_ctrl(self.baseline_delete_requested)
        )

        self.btn_base_copy = QPushButton()
        self.btn_base_copy.setIcon(QIcon(f"{ICON_DIR}/copy3.png"))
        self.btn_base_copy.setFixedSize(30, 24)
        self.btn_base_copy.setToolTip("Copy baseline")
        self.btn_base_copy.clicked.connect(lambda: self.baseline_copy_requested.emit())

        self.btn_base_paste = QPushButton()
        self.btn_base_paste.setIcon(QIcon(f"{ICON_DIR}/paste.png"))
        self.btn_base_paste.setFixedSize(30, 24)
        self.btn_base_paste.setToolTip("Paste baseline. Hold Ctrl to paste to all spectra.")
        self.btn_base_paste.clicked.connect(
            lambda: self._emit_with_ctrl(self.baseline_paste_requested)
        )

        self.btn_base_subtract = QPushButton("Subtract")
        self.btn_base_subtract.setIcon(QIcon(f"{ICON_DIR}/done.png"))
        self.btn_base_subtract.setFixedSize(80, 24)
        self.btn_base_subtract.setToolTip(
            "Subtract baseline. Hold Ctrl to subtract from all spectra."
        )
        self.btn_base_subtract.clicked.connect(
            lambda: self._emit_with_ctrl(self.baseline_subtract_requested)
        )

        for b in (self.btn_base_delete, self.btn_base_copy, self.btn_base_paste):
            row3.addWidget(b)
        row3.addStretch()
        row3.addWidget(self.btn_base_subtract)

        # ── Assemble ─────────────────────────────────────────────────
        v.addLayout(row0)
        v.addWidget(self._row1_widget)
        v.addWidget(self._row2_widget)
        v.addLayout(row3)

        # ── Connect signals ───────────────────────────────────────────
        self.cbb_baseline_mode.currentIndexChanged.connect(self._on_baseline_mode_changed)
        self.spin_poly.valueChanged.connect(self._emit_baseline_preview)
        self.chk_attached.toggled.connect(self._emit_baseline_preview)
        self.spin_noise.valueChanged.connect(self._emit_baseline_preview)
        self.sld_coef.valueChanged.connect(self._on_coef_slider_changed)
        self.spin_coef2.valueChanged.connect(self._on_coef2_changed)

        # Initialise UI state for the default mode ("Linear" = index 1)
        self._on_baseline_mode_changed(self.cbb_baseline_mode.currentIndex())

        return gb

    # ── Mode combobox helpers ─────────────────────────────────────────

    def _populate_baseline_mode_combobox(self):
        """Build the mode combobox with manual and the three supported auto methods."""
        cbb = self.cbb_baseline_mode
        cbb.clear()
        VFitModelBuilder._BASELINE_MODE_KEYS = []

        # ── Manual methods
        for key in ("Linear", "Polynomial"):
            cbb.addItem(key)
            VFitModelBuilder._BASELINE_MODE_KEYS.append(key)

        # separator
        cbb.insertSeparator(len(VFitModelBuilder._BASELINE_MODE_KEYS))
        VFitModelBuilder._BASELINE_MODE_KEYS.append("__sep__")

        # ── Auto methods: only these three methods
        _AUTO_WHITELIST = {
            "arpls":  "arPLS",
            "airpls": "airPLS",
            "asls":   "asLS",
        }

        for key, label in _AUTO_WHITELIST.items():
            cbb.addItem(label)
            VFitModelBuilder._BASELINE_MODE_KEYS.append(key)

        # Set default to "Linear" 
        cbb.setCurrentIndex(0)

    def _current_mode_key(self):
        """Return the fitspy mode key for the current combobox selection."""
        idx = self.cbb_baseline_mode.currentIndex()
        keys = VFitModelBuilder._BASELINE_MODE_KEYS
        if 0 <= idx < len(keys):
            return keys[idx]
        return None

    def _apply_mode_visibility(self, mode):
        """Update which widgets are visible for the given mode key.

        Pure side-effect on widget visibility – no signals emitted.
        Safe to call from both _on_baseline_mode_changed and update_baseline_ui.
        """
        is_manual = mode in ("Linear", "Polynomial")
        is_auto   = not is_manual and mode is not None

        self._row1_widget.setVisible(is_manual)
        self._row2_widget.setVisible(is_auto)

        if is_manual:
            show_order = (mode == "Polynomial")
            self.lbl_poly_order.setVisible(show_order)
            self.spin_poly.setVisible(show_order)

        if is_auto:
            meta = get_baseline_method_meta(mode)
            if meta.get("order_kwarg"):
                self.lbl_coef2.setText("Order:")
                self.spin_coef2.setRange(1, 20)
                self.lbl_coef2.setVisible(True)
                self.spin_coef2.setVisible(True)
            elif meta.get("sigma_kwarg"):
                self.lbl_coef2.setText("σ:")
                self.spin_coef2.setRange(0, 50)
                self.lbl_coef2.setVisible(True)
                self.spin_coef2.setVisible(True)
            else:
                self.lbl_coef2.setVisible(False)
                self.spin_coef2.setVisible(False)

    def _on_baseline_mode_changed(self, _index):
        """Handle user combobox change: update visibility and emit settings."""
        mode = self._current_mode_key()
        if mode == "__sep__":           # separator selected – jump to next valid
            self.cbb_baseline_mode.setCurrentIndex(_index + 1)
            return

        self._apply_mode_visibility(mode)

        # Refresh coef label for auto modes (label only, no preview emission)
        if mode not in ("Linear", "Polynomial", None, "__sep__"):
            coef = self.sld_coef.value() / 10.0
            lam  = 10 ** coef
            if lam >= 1e6:
                self.lbl_coef.setText(f"λ=1e{coef:.0f}")
            else:
                self.lbl_coef.setText(f"λ={lam:.0f}")

        # Emit so ViewModel stays in sync
        self._emit_baseline_settings()


    def _on_coef_slider_changed(self, int_val):
        """Slider integer (10‒100) → coef float (1.0‒10.0) → update label & preview."""
        coef = int_val / 10.0
        lam  = 10 ** coef
        if lam >= 1e6:
            self.lbl_coef.setText(f"λ=1e{coef:.0f}")
        else:
            self.lbl_coef.setText(f"λ={lam:.0f}")
        self._emit_baseline_preview()

    def _on_coef2_changed(self, _val):
        self._emit_baseline_preview()

    # ── Signal emitters ───────────────────────────────────────────────

    def _build_baseline_dict(self) -> dict:
        """Build the common settings dict from current widget state."""
        mode = self._current_mode_key()
        coef = self.sld_coef.value() / 10.0
        meta = get_baseline_method_meta(mode) if mode not in (None, "__sep__") else {}

        d = {
            "mode":      mode,
            "order_max": self.spin_poly.value()  if mode == "Polynomial" else self.spin_coef2.value(),
            "sigma":     self.spin_noise.value() if mode in ("Linear", "Polynomial") else (
                             self.spin_coef2.value() if meta.get("sigma_kwarg") else 0
                         ),
            "attached":  self.chk_attached.isChecked(),
            "coef":      coef,
        }
        return d

    def _emit_baseline_settings(self):
        """Persistent settings – emitted on combobox / checkbox / spinbox change."""
        self.baseline_settings_changed.emit(self._build_baseline_dict())

    def _emit_baseline_preview(self):
        """Live preview – emitted on slider move."""
        self.baseline_preview_requested.emit(self._build_baseline_dict())

    # ── Public: sync UI from selected spectrum ─────────────────────────────

    def update_baseline_ui(self, spectra: list):
        """Reflect the first selected spectrum's baseline settings in the GUI.

        Called whenever the spectrum selection changes so the panel always shows
        the parameters currently stored on the selected spectrum's baseline object.
        Signals are blocked during the update to avoid feedback loops.
        """
        if not spectra:
            return
        bl = spectra[0].baseline

        keys = VFitModelBuilder._BASELINE_MODE_KEYS
        mode = bl.mode  # e.g. "Linear", "arpls", None …

        # Find the combobox index for this mode key
        try:
            idx = keys.index(mode)
        except ValueError:
            idx = 0   # fall back to first manual mode

        # Block all signals to prevent cascading updates
        self.cbb_baseline_mode.blockSignals(True)
        self.sld_coef.blockSignals(True)
        self.spin_coef2.blockSignals(True)
        self.spin_poly.blockSignals(True)
        self.spin_noise.blockSignals(True)
        self.chk_attached.blockSignals(True)

        try:
            self.cbb_baseline_mode.setCurrentIndex(idx)
            # Update attached / sigma for manual modes
            self.chk_attached.setChecked(bool(bl.attached))
            self.spin_noise.setValue(int(bl.sigma) if bl.sigma else 0)
            self.spin_poly.setValue(int(bl.order_max) if bl.order_max else 2)
            # Update coef slider for auto modes
            coef_int = max(10, min(100, round(bl.coef * 10)))
            self.sld_coef.setValue(coef_int)
            lam = 10 ** bl.coef
            if lam >= 1e6:
                self.lbl_coef.setText(f"λ=1e{bl.coef:.0f}")
            else:
                self.lbl_coef.setText(f"λ={lam:.0f}")
        finally:
            self.cbb_baseline_mode.blockSignals(False)
            self.sld_coef.blockSignals(False)
            self.spin_coef2.blockSignals(False)
            self.spin_poly.blockSignals(False)
            self.spin_noise.blockSignals(False)
            self.chk_attached.blockSignals(False)

        # Apply widget visibility without triggering any signals or emissions
        self._apply_mode_visibility(mode)


    def _peaks_group(self):
        gb = QGroupBox("Peaks")
        v = QVBoxLayout(gb)
        v.setContentsMargins(2, 2, 2, 2)

        row1 = QHBoxLayout()
        self.cbb_peak_shape = QComboBox()
        self.cbb_peak_shape.addItems(PEAK_MODELS)
        self.cbb_peak_shape.currentTextChanged.connect(self.peak_shape_changed.emit)

        row1.addWidget(QLabel("Peak shape:"))
        row1.addWidget(self.cbb_peak_shape)

        row2 = QHBoxLayout()
        self.btn_peak_delete = QPushButton()
        self.btn_peak_delete.setIcon(QIcon(f"{ICON_DIR}/trash3.png"))
        self.btn_peak_delete.setFixedSize(30, 24)
        self.btn_peak_delete.setToolTip("Delete peaks. Hold Ctrl to delete from all spectra.")
        self.btn_peak_delete.clicked.connect(
            lambda: self._emit_with_ctrl(self.peaks_delete_requested)
        )

        self.btn_peak_copy = QPushButton()
        self.btn_peak_copy.setIcon(QIcon(f"{ICON_DIR}/copy3.png"))
        self.btn_peak_copy.setFixedSize(30, 24)
        self.btn_peak_copy.setToolTip("Copy peaks.")
        self.btn_peak_copy.clicked.connect(self.peaks_copy_requested.emit)

        self.btn_peak_paste = QPushButton()
        self.btn_peak_paste.setIcon(QIcon(f"{ICON_DIR}/paste.png"))
        self.btn_peak_paste.setFixedSize(30, 24)
        self.btn_peak_paste.setToolTip("Paste peaks. Hold Ctrl to paste to all spectra.")
        self.btn_peak_paste.clicked.connect(
            lambda: self._emit_with_ctrl(self.peaks_paste_requested)
        )

        for b in (self.btn_peak_delete, self.btn_peak_copy, self.btn_peak_paste):
            row2.addWidget(b)
        row2.addStretch()

        v.addLayout(row1)
        v.addLayout(row2)
        return gb
    
    def _emit_with_ctrl(self, signal):
        """Helper to emit signal with apply_all based on Ctrl key."""
        apply_all = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        signal.emit(apply_all)


    def _fit_control_panel(self):
        gb = QGroupBox("")
        v = QVBoxLayout(gb)
        v.setContentsMargins(4, 4, 4, 4)

        # ── Row 1: Fit actions
        row1 = QHBoxLayout()
        self.btn_fit = QPushButton("Fit")
        self.btn_fit.setToolTip("Fit the spectrum. Hold Ctrl to fit all spectra.")
        self.btn_fit.clicked.connect(
            lambda: self._emit_with_ctrl(self.fit_requested)
        )

        self.btn_copy = QPushButton("Copy")
        self.btn_copy.setIcon(QIcon(f"{ICON_DIR}/copy3.png"))
        #self.btn_copy.setFixedSize(80, 24)
        self.btn_copy.setToolTip("Copy fit model of selected spectrum.")
        self.btn_copy.clicked.connect(self.fitmodel_copy_requested.emit)


        self.btn_paste = QPushButton("Paste")
        self.btn_paste.setIcon(QIcon(f"{ICON_DIR}/paste.png"))
        #self.btn_paste.setFixedSize(80, 24)
        self.btn_paste.setToolTip("Paste fit model of selected spectrum. Hold Ctrl to paste to all spectra.")
        self.btn_paste.clicked.connect(
            lambda: self._emit_with_ctrl(self.fitmodel_paste_requested)
        )

        self.btn_save = QPushButton("Save")
        self.btn_save.setIcon(QIcon(f"{ICON_DIR}/save.png"))
        #self.btn_save.setFixedSize(80, 24)
        self.btn_save.setToolTip("Save the current fit model.")
        self.btn_save.clicked.connect(self.fitmodel_save_requested.emit)

        self.chk_limits = QCheckBox("Limits")
        self.chk_expr = QCheckBox("Expression")

        self.chk_limits.toggled.connect(self.peak_table.set_show_limits)
        self.chk_expr.toggled.connect(self.peak_table.set_show_expr)

        for b in (self.btn_fit, self.btn_copy, self.btn_paste, self.btn_save, self.chk_limits, self.chk_expr):
            row1.addWidget(b)
        row1.addStretch()

        # ── Row 2: Model selection
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Fit model:"))

        self.cbb_model = QComboBox()
        self.cbb_model.setFixedWidth(300)

        self.cbb_model.setToolTip("Select fit model.")

        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setIcon(QIcon(f"{ICON_DIR}/done.png"))
        self.btn_apply.setToolTip("Apply selected fit model to current selected spectrum(s).")
        self.btn_apply.clicked.connect(
            lambda: self._emit_with_ctrl(self.apply_loaded_fit_model_requested)
        )

        self.btn_load = QPushButton("Load")
        self.btn_load.setIcon(QIcon(f"{ICON_DIR}/load.png"))
        self.btn_load.setToolTip("Load fit model from a JSON file.")
        self.btn_load.clicked.connect(self.load_fit_models_requested.emit)  

        self.btn_refresh = QPushButton()
        self.btn_refresh.setIcon(QIcon(f"{ICON_DIR}/refresh.png"))
        self.btn_refresh.setToolTip("Refresh fit model list.")
        self.btn_refresh.clicked.connect(self.refresh_fit_models_requested.emit)    

        row2.addWidget(self.cbb_model)
        row2.addWidget(self.btn_apply)
        row2.addWidget(self.btn_load)
        row2.addWidget(self.btn_refresh)
        row2.addStretch()

        v.addLayout(row1)
        v.addLayout(row2)

        return gb

    def set_fit_buttons_enabled(self, enabled: bool):
        """Enable/disable fit-related buttons to prevent concurrent operations."""
        self.btn_fit.setEnabled(enabled)
        self.btn_apply.setEnabled(enabled)
        self.btn_paste.setEnabled(enabled)
        if not enabled:
            self.btn_fit.setText("Fitting")
            self.btn_apply.setText("Applying")
        else:
            self.btn_fit.setText("Fit")
            self.btn_apply.setText("Apply")

