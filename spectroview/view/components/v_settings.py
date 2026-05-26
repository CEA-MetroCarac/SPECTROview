# spectroview/view/components/v_settings.py
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit, QDialogButtonBox,
    QSpacerItem, QSizePolicy
)
from PySide6.QtGui import QFont

from spectroview.viewmodel.vm_settings import VMSettings


class VSettingsDialog(QDialog):
    
    def __init__(self, vm: VMSettings, parent=None):
        super().__init__(parent)
        self.vm = vm

        self.setWindowTitle("Settings")
        self.resize(400, 400)

        self._init_ui()
        self._connect_vm()

        # Load settings from VM
        self.vm.load()

    # ──────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────
    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        bold = QFont()
        bold.setBold(True)

        # ───── Optimization Settings ─────
        lbl_opt = QLabel("Optimization Settings:")
        lbl_opt.setFont(bold)
        main_layout.addWidget(lbl_opt)

        # Max iterations
        row = QHBoxLayout()
        row.addWidget(QLabel("Maximum iterations:"))
        self.spin_max_iter = QSpinBox()
        self.spin_max_iter.setRange(1, 10000)
        self.spin_max_iter.setSingleStep(20)
        row.addWidget(self.spin_max_iter)
        main_layout.addLayout(row)

        # x-tolerance
        row = QHBoxLayout()
        row.addWidget(QLabel("x-tolerance (xtol):"))
        self.spin_x_tol = QDoubleSpinBox()
        self.spin_x_tol.setDecimals(6)
        self.spin_x_tol.setRange(1e-6, 1e-1)
        self.spin_x_tol.setSingleStep(1e-5)
        row.addWidget(self.spin_x_tol)
        main_layout.addLayout(row)

        # f-tolerance
        row = QHBoxLayout()
        row.addWidget(QLabel("f-tolerance (ftol):"))
        self.spin_f_tol = QDoubleSpinBox()
        self.spin_f_tol.setDecimals(6)
        self.spin_f_tol.setRange(1e-6, 1e-1)
        self.spin_f_tol.setSingleStep(1e-5)
        row.addWidget(self.spin_f_tol)
        main_layout.addLayout(row)

        main_layout.addSpacing(10)

        # ───── Data Treatment ─────
        lbl_data = QLabel("Data Treatment:")
        lbl_data.setFont(bold)
        main_layout.addWidget(lbl_data)

        self.chk_fit_negative = QCheckBox("Fit negative values")
        main_layout.addWidget(self.chk_fit_negative)

        self.chk_fit_outliers = QCheckBox("Include outliers in fit")
        main_layout.addWidget(self.chk_fit_outliers)

        # Noise coeff
        row = QHBoxLayout()
        row.addWidget(QLabel("Noise threshold coeff:"))
        self.spin_coef_noise = QDoubleSpinBox()
        self.spin_coef_noise.setDecimals(2)
        self.spin_coef_noise.setRange(0, 100)
        self.spin_coef_noise.setSingleStep(0.5)
        self.spin_coef_noise.setValue(1.0)
        row.addWidget(self.spin_coef_noise)
        main_layout.addLayout(row)

        main_layout.addSpacing(10)

        # ───── Peak Bounds ─────
        lbl_bounds = QLabel("Peak Assignment Bounds:")
        lbl_bounds.setFont(bold)
        main_layout.addWidget(lbl_bounds)

        # Max peak shift
        row = QHBoxLayout()
        row.addWidget(QLabel("Max peak shift:"))
        self.spin_maxshift = QDoubleSpinBox()
        self.spin_maxshift.setRange(0, 100)
        self.spin_maxshift.setSingleStep(5)
        self.spin_maxshift.setDecimals(2)
        row.addWidget(self.spin_maxshift)
        main_layout.addLayout(row)

        # Min peak fwhm
        row = QHBoxLayout()
        row.addWidget(QLabel("Min peak FWHM:"))
        self.spin_minfwhm = QDoubleSpinBox()
        self.spin_minfwhm.setRange(0.01, 500)
        self.spin_minfwhm.setSingleStep(1)
        self.spin_minfwhm.setDecimals(2)
        row.addWidget(self.spin_minfwhm)
        main_layout.addLayout(row)

        # Max peak fwhm
        row = QHBoxLayout()
        row.addWidget(QLabel("Max peak FWHM:"))
        self.spin_maxfwhm = QDoubleSpinBox()
        self.spin_maxfwhm.setRange(1, 1000)
        self.spin_maxfwhm.setSingleStep(20)
        self.spin_maxfwhm.setDecimals(2)
        row.addWidget(self.spin_maxfwhm)
        main_layout.addLayout(row)

        # Max intensity
        row = QHBoxLayout()
        row.addWidget(QLabel("Max peak Intensity:"))
        self.spin_maxintensity = QDoubleSpinBox()
        self.spin_maxintensity.setRange(1, 1e9)
        self.spin_maxintensity.setSingleStep(1000)
        self.spin_maxintensity.setDecimals(0)
        row.addWidget(self.spin_maxintensity)
        main_layout.addLayout(row)

        # Spacer
        main_layout.addSpacerItem(
            QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        # ───── Fit model management ─────
        lbl_model = QLabel("Fit model management:")
        lbl_model.setFont(bold)
        main_layout.addWidget(lbl_model)

        folder_row = QHBoxLayout()
        self.btn_model_folder = QPushButton("Path:")
        self.btn_model_folder.setMaximumWidth(40)
        self.le_model_folder = QLineEdit()

        folder_row.addWidget(self.btn_model_folder)
        folder_row.addWidget(self.le_model_folder)
        main_layout.addLayout(folder_row)

        main_layout.addSpacerItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        # ───── OK / Cancel ─────
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok_btn = buttons.button(QDialogButtonBox.Ok)
        cancel_btn = buttons.button(QDialogButtonBox.Cancel)

        ok_btn.setStyleSheet("background-color: green; color: white; font-weight: bold;")
        cancel_btn.setStyleSheet("background-color: red; color: white; font-weight: bold;")

        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        main_layout.addWidget(buttons)

        # Folder picker
        self.btn_model_folder.clicked.connect(self.vm.pick_model_folder)

    # ──────────────────────────────────────────────
    # VM Connections
    # ──────────────────────────────────────────────
    def _connect_vm(self):
        self.vm.settings_loaded.connect(self._apply_settings)
        self.vm.model_folder_changed.connect(self.le_model_folder.setText)
        self.vm.settings_saved.connect(self.accept)

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────
    def _apply_settings(self, data: dict):
        self.chk_fit_negative.setChecked(data.get("fit_negative", False))
        self.chk_fit_outliers.setChecked(data.get("fit_outliers", True))
        
        self.spin_max_iter.setValue(data.get("max_ite", 200))
        self.spin_x_tol.setValue(data.get("xtol", 1e-4))
        self.spin_f_tol.setValue(data.get("ftol", 1e-4))
        self.spin_coef_noise.setValue(data.get("coef_noise", 1.0))
        
        self.spin_maxshift.setValue(data.get("maxshift", 20.0))
        self.spin_minfwhm.setValue(data.get("minfwhm", 0.1))
        self.spin_maxfwhm.setValue(data.get("maxfwhm", 200.0))
        self.spin_maxintensity.setValue(data.get("maxintensity", 100000.0))
        
        self.le_model_folder.setText(data.get("model_folder", ""))

    def _on_accept(self):
        self.vm.save({
            "fit_negative": self.chk_fit_negative.isChecked(),
            "fit_outliers": self.chk_fit_outliers.isChecked(),
            "max_ite": self.spin_max_iter.value(),
            "xtol": self.spin_x_tol.value(),
            "ftol": self.spin_f_tol.value(),
            "coef_noise": self.spin_coef_noise.value(),
            "maxshift": self.spin_maxshift.value(),
            "minfwhm": self.spin_minfwhm.value(),
            "maxfwhm": self.spin_maxfwhm.value(),
            "maxintensity": self.spin_maxintensity.value(),
            "model_folder": self.le_model_folder.text(),
        })
