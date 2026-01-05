# spectroview/view/components/v_settings.py
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QCheckBox, QSpinBox, QComboBox,
    QDoubleSpinBox, QLineEdit, QDialogButtonBox,
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

        # ───── Fit Settings ─────
        lbl_fit = QLabel("Fit Settings:")
        lbl_fit.setFont(bold)
        main_layout.addWidget(lbl_fit)

        self.chk_fit_negative = QCheckBox("Fit negative value")
        main_layout.addWidget(self.chk_fit_negative)

        # Fit method
        row = QHBoxLayout()
        row.addWidget(QLabel("Fit Method:"))
        self.cbb_fit_method = QComboBox()
        self.cbb_fit_method.addItems(
            ["Leastsq", "Least_squares", "Nelder-Mead", "SLSQP"]
        )
        row.addWidget(self.cbb_fit_method)
        main_layout.addLayout(row)

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
        row.addWidget(QLabel("x-tolerance:"))
        self.spin_x_tol = QDoubleSpinBox()
        self.spin_x_tol.setDecimals(6)
        self.spin_x_tol.setRange(1e-5, 1e-3)
        self.spin_x_tol.setSingleStep(1e-5)
        self.spin_x_tol.setValue(1e-4)
        row.addWidget(self.spin_x_tol)
        main_layout.addLayout(row)

        # CPU cores
        row = QHBoxLayout()
        row.addWidget(QLabel("Number of CPU cores:"))
        self.spin_cpu = QSpinBox()
        self.spin_cpu.setRange(1, 64)
        row.addWidget(self.spin_cpu)
        main_layout.addLayout(row)

        # Max peak shift
        row = QHBoxLayout()
        row.addWidget(QLabel("Max peak shift:"))
        self.spin_maxshift = QDoubleSpinBox()
        self.spin_maxshift.setRange(0, 100)
        self.spin_maxshift.setSingleStep(5)
        self.spin_maxshift.setDecimals(2)
        row.addWidget(self.spin_maxshift)
        main_layout.addLayout(row)

        # Max peak fwhm
        row = QHBoxLayout()
        row.addWidget(QLabel("Max peak fwhm:"))
        self.spin_maxfwhm = QDoubleSpinBox()
        self.spin_maxfwhm.setRange(0, 500)
        self.spin_maxfwhm.setSingleStep(20)
        self.spin_maxfwhm.setDecimals(2)
        row.addWidget(self.spin_maxfwhm)
        main_layout.addLayout(row)

        # Spacer (same as legacy)
        main_layout.addSpacerItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
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
        self.chk_fit_negative.setChecked(data["fit_negative"])
        self.cbb_fit_method.setCurrentText(data["method"])
        self.spin_max_iter.setValue(data["max_ite"])
        self.spin_x_tol.setValue(data["xtol"])
        self.spin_cpu.setValue(data["ncpu"])
        self.spin_maxshift.setValue(data["maxshift"])
        self.spin_maxfwhm.setValue(data["maxfwhm"])
        self.le_model_folder.setText(data.get("model_folder", ""))

    def _on_accept(self):
        self.vm.save({
            "fit_negative": self.chk_fit_negative.isChecked(),
            "method": self.cbb_fit_method.currentText(),
            "max_ite": self.spin_max_iter.value(),
            "xtol": self.spin_x_tol.value(),
            "ncpu": self.spin_cpu.value(),
            "maxshift": self.spin_maxshift.value(),
            "maxfwhm": self.spin_maxfwhm.value(),
            "model_folder": self.le_model_folder.text(),
        })
