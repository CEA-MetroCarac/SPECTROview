# spectroview/view/components/v_fit_model_builder.py
from spectroview import ICON_DIR, PEAK_MODELS
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QGroupBox, QLabel, QPushButton, QComboBox,
    QDoubleSpinBox, QSpinBox, QRadioButton,
    QScrollArea, QTableView, QCheckBox
)
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt


class VFitModelBuilder(QWidget):
    """View: Fit Model Builder panel"""

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
        left_scroll.setWidgetResizable(True)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
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
        right_splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(right_splitter)

        # --- Peak table
        self.peak_table = QTableView()
        right_splitter.addWidget(self.peak_table)

        # --- Fit controls
        right_splitter.addWidget(self._fit_control_panel())

        splitter.setSizes([370, 650])
        right_splitter.setSizes([500, 100])

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
        self.spin_xcorr = QSpinBox()
        self.spin_xcorr.setRange(-100000, 100000)
        self.spin_xcorr.setValue(0)
        self.spin_xcorr.setToolTip("Type measured reference peak position")

        # Label showing corrected value
        self.lbl_xcorr_value = QLabel("(0)")
        
        # Buttons
        self.btn_correct = QPushButton("Correct")
        self.btn_correct.setIcon(QIcon(f"{ICON_DIR}/done.png"))
        

        self.btn_undo_corr = QPushButton()
        self.btn_undo_corr.setIcon(QIcon(f"{ICON_DIR}/undo.png"))
        self.btn_undo_corr.setFixedSize(30, 24)
        self.btn_undo_corr.setToolTip("Undo X correction")

        # --- Layout order ---
        l.addWidget(self.cbb_xcorr)
        l.addWidget(self.spin_xcorr)
        l.addWidget(self.lbl_xcorr_value)
        l.addStretch()
        l.addWidget(self.btn_correct)
        l.addWidget(self.btn_undo_corr)

        return gb

    def set_xcorrection_value(self, value: float):
        """Update label from Model state (xcorrection_value)"""
        if abs(value) < 1e-9:
            self.lbl_xcorr_value.setText("Δx = 0")
        else:
            self.lbl_xcorr_value.setText(f"Δx = {value:+.2f}")

    def _spectral_range_group(self):
        gb = QGroupBox("Spectral range")
        l = QHBoxLayout(gb)
        l.setContentsMargins(2, 2, 2, 2)

        self.spin_xmin = QDoubleSpinBox()
        self.spin_xmax = QDoubleSpinBox()
        self.spin_xmin.setDecimals(3)
        self.spin_xmax.setDecimals(3)

        self.btn_extract = QPushButton("Extract")
        self.btn_extract.setIcon(QIcon(f"{ICON_DIR}/cut.png"))
        self.btn_extract.setToolTip("Extract selected spectra range. Hold Ctrl to paste to all spectra.")

        l.addWidget(QLabel("X min/max:"))
        l.addWidget(self.spin_xmin)
        l.addWidget(QLabel("/"))
        l.addWidget(self.spin_xmax)
        l.addStretch()
        l.addWidget(self.btn_extract)
        

        return gb

    def _baseline_group(self):
        gb = QGroupBox("Baseline")
        v = QVBoxLayout(gb)
        v.setContentsMargins(2, 2, 2, 2)

        # ── Row 1: baseline type
        row1 = QHBoxLayout()

        self.rb_linear = QRadioButton("Linear")
        self.rb_poly = QRadioButton("Poly")
        self.rb_linear.setChecked(True)  

        self.spin_poly = QSpinBox()
        self.spin_poly.setRange(1, 20)

        # ── Row 3: actions
        row2 = QHBoxLayout()

        self.btn_base_delete = QPushButton()
        self.btn_base_delete.setIcon(QIcon(f"{ICON_DIR}/trash3.png"))
        self.btn_base_delete.setFixedSize(30, 24)
        self.btn_base_delete.setToolTip("Delete baseline")

        self.btn_base_copy = QPushButton()
        self.btn_base_copy.setIcon(QIcon(f"{ICON_DIR}/copy3.png"))
        self.btn_base_copy.setFixedSize(30, 24)
        self.btn_base_copy.setToolTip("Copy baseline")

        self.btn_base_paste = QPushButton()
        self.btn_base_paste.setIcon(QIcon(f"{ICON_DIR}/paste.png"))
        self.btn_base_paste.setFixedSize(30, 24)
        self.btn_base_paste.setToolTip("Paste baseline. Hold Ctrl to paste to all spectra.")


        self.btn_base_subtract = QPushButton("Subtract")
        self.btn_base_subtract.setIcon(QIcon(f"{ICON_DIR}/done.png"))
        self.btn_base_subtract.setFixedSize(80, 24)
        self.btn_base_subtract.setToolTip("Subtract baseline. Hold Ctrl to subtract from all spectra.")

        self.chk_attached = QCheckBox("Attached")
        self.spin_noise = QSpinBox()
        self.spin_noise.setRange(0, 20)
        self.spin_noise.setValue(4)  
        self.spin_noise.setToolTip("Correct noise")

        row1.addWidget(self.rb_linear)
        row1.addWidget(self.rb_poly)
        row1.addWidget(self.spin_poly)
        row1.addStretch()

        row1.addWidget(self.chk_attached)
        
        for b in (
            self.btn_base_delete,
            self.btn_base_copy,
            self.btn_base_paste, 
        ):
            row2.addWidget(b)


        row2.addWidget(QLabel("Noise cor:"))
        row2.addWidget(self.spin_noise)
        row2.addStretch()
        row2.addWidget(self.btn_base_subtract)

        v.addLayout(row1)
        v.addLayout(row2)

        return gb

    def _peaks_group(self):
        gb = QGroupBox("Peaks")
        v = QVBoxLayout(gb)
        v.setContentsMargins(2, 2, 2, 2)

        row1 = QHBoxLayout()
        self.cbb_peak_shape = QComboBox()
        self.cbb_peak_shape.addItems(PEAK_MODELS)

        row1.addWidget(QLabel("Peak shape:"))
        row1.addWidget(self.cbb_peak_shape)

        row2 = QHBoxLayout()
        self.btn_peak_delete = QPushButton()
        self.btn_peak_delete.setIcon(QIcon(f"{ICON_DIR}/trash3.png"))
        self.btn_peak_delete.setFixedSize(30, 24)
        self.btn_peak_delete.setToolTip("Delete peaks.")

        self.btn_peak_copy = QPushButton()
        self.btn_peak_copy.setIcon(QIcon(f"{ICON_DIR}/copy3.png"))
        self.btn_peak_copy.setFixedSize(30, 24)
        self.btn_peak_copy.setToolTip("Copy peaks.")

        self.btn_peak_paste = QPushButton()
        self.btn_peak_paste.setIcon(QIcon(f"{ICON_DIR}/paste.png"))
        self.btn_peak_paste.setFixedSize(30, 24)
        self.btn_peak_paste.setToolTip("Paste peaks. Hold Ctrl to paste to all spectra.")

        for b in (self.btn_peak_delete, self.btn_peak_copy, self.btn_peak_paste):
            row2.addWidget(b)
        row2.addStretch()

        v.addLayout(row1)
        v.addLayout(row2)
        return gb

    def _fit_control_panel(self):
        gb = QGroupBox("")
        v = QVBoxLayout(gb)

        # ── Row 1: Fit actions
        row1 = QHBoxLayout()
        self.btn_fit = QPushButton("Fit")
        #self.btn_fit.setFixedSize(80, 24)
        self.btn_fit.setToolTip("Fit the spectrum. Hold Ctrl to fit all spectra.")


        self.btn_copy = QPushButton("Copy")
        self.btn_copy.setIcon(QIcon(f"{ICON_DIR}/copy3.png"))
        #self.btn_copy.setFixedSize(80, 24)
        self.btn_copy.setToolTip("Copy fit model of selected spectrum.")


        self.btn_paste = QPushButton("Paste")
        self.btn_paste.setIcon(QIcon(f"{ICON_DIR}/paste.png"))
        #self.btn_paste.setFixedSize(80, 24)
        self.btn_paste.setToolTip("Paste fit model of selected spectrum. Hold Ctrl to paste to all spectra.")

        self.btn_save = QPushButton("Save")
        self.btn_save.setIcon(QIcon(f"{ICON_DIR}/save.png"))
        #self.btn_save.setFixedSize(80, 24)
        self.btn_save.setToolTip("Save the current fit model.")

        self.chk_limits = QCheckBox("Limits")
        self.chk_expr = QCheckBox("Expression")


        for b in (self.btn_fit, self.btn_copy, self.btn_paste, self.btn_save, self.chk_limits, self.chk_expr):
            row1.addWidget(b)
        row1.addStretch()

        # ── Row 2: Model selection
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Fit model:"))

        self.cbb_model = QComboBox()
        self.cbb_model.setMinimumWidth(100)

        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setIcon(QIcon(f"{ICON_DIR}/done.png"))
        self.btn_load = QPushButton("Load")
        self.btn_load.setIcon(QIcon(f"{ICON_DIR}/load.png"))
        self.btn_refresh = QPushButton()
        self.btn_refresh.setIcon(QIcon(f"{ICON_DIR}/refresh.png"))

        row2.addWidget(self.cbb_model)
        row2.addWidget(self.btn_apply)
        row2.addWidget(self.btn_load)
        row2.addWidget(self.btn_refresh)
        row2.addStretch()

        
        v.addLayout(row1)
        v.addLayout(row2)

        return gb

