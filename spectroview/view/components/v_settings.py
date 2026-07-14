# spectroview/view/components/v_settings.py
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit, QDialogButtonBox,
    QSpacerItem, QSizePolicy, QFrame, QGroupBox, QTabWidget, QWidget
)
from PySide6.QtGui import QFont

from spectroview.viewmodel.vm_settings import VMSettings


class VSettingsDialog(QDialog):
    
    def __init__(self, vm: VMSettings, parent=None):
        super().__init__(parent)
        self.vm = vm

        self.setWindowTitle("Settings")
        self.resize(480, 550)

        self._init_ui()
        self._connect_vm()

        # Load settings from VM
        self.vm.load()

    # ──────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────
    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(8)

        bold = QFont()
        bold.setBold(True)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # ==========================================
        # TAB 1: Fitting
        # ==========================================
        tab_fitting = QWidget()
        fitting_tab_layout = QVBoxLayout(tab_fitting)
        fitting_tab_layout.setContentsMargins(4, 4, 4, 4)
        fitting_tab_layout.setSpacing(8)

        # ───── Fitting & Data Settings ─────
        grp_fitting = QGroupBox("Fit Parameters")
        fitting_layout = QVBoxLayout(grp_fitting)
        fitting_layout.setContentsMargins(4, 4, 4, 4)
        fitting_layout.setSpacing(8)

        # Max iterations
        row = QHBoxLayout()
        row.addWidget(QLabel("Maximum iterations:"))
        self.spin_max_iter = QSpinBox()
        self.spin_max_iter.setRange(1, 10000)
        self.spin_max_iter.setSingleStep(20)
        row.addWidget(self.spin_max_iter)
        fitting_layout.addLayout(row)

        # x-tolerance
        row = QHBoxLayout()
        row.addWidget(QLabel("x-tolerance (xtol):"))
        self.spin_x_tol = QDoubleSpinBox()
        self.spin_x_tol.setDecimals(6)
        self.spin_x_tol.setRange(1e-6, 1e-1)
        self.spin_x_tol.setSingleStep(1e-5)
        row.addWidget(self.spin_x_tol)
        fitting_layout.addLayout(row)

        # f-tolerance
        row = QHBoxLayout()
        row.addWidget(QLabel("f-tolerance (ftol):"))
        self.spin_f_tol = QDoubleSpinBox()
        self.spin_f_tol.setDecimals(6)
        self.spin_f_tol.setRange(1e-6, 1e-1)
        self.spin_f_tol.setSingleStep(1e-5)
        row.addWidget(self.spin_f_tol)
        fitting_layout.addLayout(row)

        # Noise coeff
        row = QHBoxLayout()
        row.addWidget(QLabel("Noise threshold coeff:"))
        self.spin_coef_noise = QDoubleSpinBox()
        self.spin_coef_noise.setDecimals(2)
        self.spin_coef_noise.setRange(0, 100)
        self.spin_coef_noise.setSingleStep(0.5)
        self.spin_coef_noise.setValue(1.0)
        row.addWidget(self.spin_coef_noise)
        fitting_layout.addLayout(row)

        self.chk_fit_negative = QCheckBox("Fit negative values")
        fitting_layout.addWidget(self.chk_fit_negative)

        fitting_tab_layout.addWidget(grp_fitting)

        fitting_tab_layout.addSpacing(10)

        # ───── Peak Bounds ─────
        grp_bounds = QGroupBox("Global Peak Limits")
        bounds_layout = QVBoxLayout(grp_bounds)
        bounds_layout.setContentsMargins(4, 4, 4, 4)
        bounds_layout.setSpacing(8)

        # Max peak shift
        row = QHBoxLayout()
        row.addWidget(QLabel("Max peak shift:"))
        self.spin_maxshift = QDoubleSpinBox()
        self.spin_maxshift.setRange(0, 100)
        self.spin_maxshift.setSingleStep(5)
        self.spin_maxshift.setDecimals(2)
        row.addWidget(self.spin_maxshift)
        bounds_layout.addLayout(row)

        # Min peak fwhm
        row = QHBoxLayout()
        row.addWidget(QLabel("Min peak FWHM:"))
        self.spin_minfwhm = QDoubleSpinBox()
        self.spin_minfwhm.setRange(0.01, 500)
        self.spin_minfwhm.setSingleStep(1)
        self.spin_minfwhm.setDecimals(2)
        row.addWidget(self.spin_minfwhm)
        bounds_layout.addLayout(row)

        # Max peak fwhm
        row = QHBoxLayout()
        row.addWidget(QLabel("Max peak FWHM:"))
        self.spin_maxfwhm = QDoubleSpinBox()
        self.spin_maxfwhm.setRange(1, 1000)
        self.spin_maxfwhm.setSingleStep(20)
        self.spin_maxfwhm.setDecimals(2)
        row.addWidget(self.spin_maxfwhm)
        bounds_layout.addLayout(row)

        # Max intensity
        row = QHBoxLayout()
        row.addWidget(QLabel("Max peak Intensity:"))
        self.spin_maxintensity = QDoubleSpinBox()
        self.spin_maxintensity.setRange(1, 1e9)
        self.spin_maxintensity.setSingleStep(1000)
        self.spin_maxintensity.setDecimals(0)
        row.addWidget(self.spin_maxintensity)
        bounds_layout.addLayout(row)

        fitting_tab_layout.addWidget(grp_bounds)

        # Spacer
        fitting_tab_layout.addSpacerItem(
            QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        # Separation line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        fitting_tab_layout.addWidget(line)
        fitting_tab_layout.addSpacing(10)

        # ───── Fit model management ─────
        lbl_model = QLabel("Fit Model Path:")
        lbl_model.setFont(bold)
        fitting_tab_layout.addWidget(lbl_model)

        folder_row = QHBoxLayout()
        self.btn_model_folder = QPushButton("Browse")
        self.btn_model_folder.setMaximumWidth(60)
        self.le_model_folder = QLineEdit()

        folder_row.addWidget(self.btn_model_folder)
        folder_row.addWidget(self.le_model_folder)
        fitting_tab_layout.addLayout(folder_row)

        self.tabs.addTab(tab_fitting, "Fitting")


        # ==========================================
        # TAB 2: AI Settings
        # ==========================================
        tab_ai = QWidget()
        ai_tab_layout = QVBoxLayout(tab_ai)
        ai_tab_layout.setContentsMargins(4, 4, 4, 4)
        ai_tab_layout.setSpacing(8)

        # API Keys
        grp_api = QGroupBox("API Keys")
        api_layout = QVBoxLayout(grp_api)
        api_layout.setContentsMargins(4, 4, 4, 4)
        api_layout.setSpacing(8)

        self.edit_openai = QLineEdit()
        self.edit_openai.setEchoMode(QLineEdit.Password)
        self.edit_anthropic = QLineEdit()
        self.edit_anthropic.setEchoMode(QLineEdit.Password)
        self.edit_gemini = QLineEdit()
        self.edit_gemini.setEchoMode(QLineEdit.Password)
        self.edit_deepseek = QLineEdit()
        self.edit_deepseek.setEchoMode(QLineEdit.Password)
        self.edit_custom = QLineEdit()
        self.edit_custom.setEchoMode(QLineEdit.Password)
        self.edit_custom_url = QLineEdit()

        def add_api_row(label, widget):
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFixedWidth(120)
            row.addWidget(lbl)
            row.addWidget(widget)
            api_layout.addLayout(row)
            
        add_api_row("OpenAI API Key:", self.edit_openai)
        add_api_row("Anthropic API Key:", self.edit_anthropic)
        add_api_row("Gemini API Key:", self.edit_gemini)
        add_api_row("DeepSeek API Key:", self.edit_deepseek)
        add_api_row("Custom API Key:", self.edit_custom)
        add_api_row("Custom Base URL:", self.edit_custom_url)

        ai_tab_layout.addWidget(grp_api)

        # Chat History Folder
        grp_history = QGroupBox("Chat History")
        history_layout = QVBoxLayout(grp_history)
        history_layout.setContentsMargins(4, 4, 4, 4)
        history_layout.setSpacing(8)
        
        lbl_history = QLabel("History Folder:")
        history_layout.addWidget(lbl_history)
        
        history_row = QHBoxLayout()
        self.btn_history_folder = QPushButton("Browse")
        self.btn_history_folder.setMaximumWidth(60)
        self.le_history_folder = QLineEdit()
        history_row.addWidget(self.btn_history_folder)
        history_row.addWidget(self.le_history_folder)
        history_layout.addLayout(history_row)

        ai_tab_layout.addWidget(grp_history)

        # Plot Template Folder
        grp_templates = QGroupBox("Plot Templates")
        templates_layout = QVBoxLayout(grp_templates)
        templates_layout.setContentsMargins(4, 4, 4, 4)
        templates_layout.setSpacing(8)

        lbl_templates = QLabel("Template Folder:")
        templates_layout.addWidget(lbl_templates)

        templates_row = QHBoxLayout()
        self.btn_template_folder = QPushButton("Browse")
        self.btn_template_folder.setMaximumWidth(60)
        self.le_template_folder = QLineEdit()
        templates_row.addWidget(self.btn_template_folder)
        templates_row.addWidget(self.le_template_folder)
        templates_layout.addLayout(templates_row)

        ai_tab_layout.addWidget(grp_templates)
        ai_tab_layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.tabs.addTab(tab_ai, "AI")

        # ───── OK / Cancel ─────
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok_btn = buttons.button(QDialogButtonBox.Ok)
        cancel_btn = buttons.button(QDialogButtonBox.Cancel)

        ok_btn.setStyleSheet("background-color: rgba(76, 175, 80, 180); color: white; font-weight: bold; border-radius: 6px;")
        cancel_btn.setStyleSheet("background-color: rgba(220, 60, 60, 180); color: white; font-weight: bold; border-radius: 6px;")

        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        main_layout.addWidget(buttons)

        # Folder picker
        self.btn_model_folder.clicked.connect(self.vm.pick_model_folder)
        self.btn_history_folder.clicked.connect(self.vm.pick_history_folder)
        self.btn_template_folder.clicked.connect(self.vm.pick_template_folder)

    # ──────────────────────────────────────────────
    # VM Connections
    # ──────────────────────────────────────────────
    def _connect_vm(self):
        self.vm.settings_loaded.connect(self._apply_settings)
        self.vm.model_folder_changed.connect(self.le_model_folder.setText)
        self.vm.history_folder_changed.connect(self.le_history_folder.setText)
        self.vm.template_folder_changed.connect(self.le_template_folder.setText)
        self.vm.settings_saved.connect(self.accept)

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────
    def _apply_settings(self, data: dict):
        self.chk_fit_negative.setChecked(data.get("fit_negative", False))
        
        self.spin_max_iter.setValue(data.get("max_ite", 200))
        self.spin_x_tol.setValue(data.get("xtol", 1e-4))
        self.spin_f_tol.setValue(data.get("ftol", 1e-4))
        self.spin_coef_noise.setValue(data.get("coef_noise", 1.0))
        
        self.spin_maxshift.setValue(data.get("maxshift", 20.0))
        self.spin_minfwhm.setValue(data.get("minfwhm", 0.1))
        self.spin_maxfwhm.setValue(data.get("maxfwhm", 200.0))
        self.spin_maxintensity.setValue(data.get("maxintensity", 100000.0))
        
        self.le_model_folder.setText(data.get("model_folder", ""))
        
        # AI settings
        self.edit_openai.setText(data.get("api_key_OpenAI", ""))
        self.edit_anthropic.setText(data.get("api_key_Anthropic", ""))
        self.edit_gemini.setText(data.get("api_key_Gemini", ""))
        self.edit_deepseek.setText(data.get("api_key_DeepSeek", ""))
        self.edit_custom.setText(data.get("api_key_Custom", ""))
        self.edit_custom_url.setText(data.get("custom_base_url", ""))
        self.le_history_folder.setText(data.get("history_folder", ""))
        self.le_template_folder.setText(data.get("template_folder", ""))

    def _on_accept(self):
        self.vm.save({
            "fit_negative": self.chk_fit_negative.isChecked(),
            "max_ite": self.spin_max_iter.value(),
            "xtol": self.spin_x_tol.value(),
            "ftol": self.spin_f_tol.value(),
            "coef_noise": self.spin_coef_noise.value(),
            "maxshift": self.spin_maxshift.value(),
            "minfwhm": self.spin_minfwhm.value(),
            "maxfwhm": self.spin_maxfwhm.value(),
            "maxintensity": self.spin_maxintensity.value(),
            "model_folder": self.le_model_folder.text(),
            
            "api_key_OpenAI": self.edit_openai.text(),
            "api_key_Anthropic": self.edit_anthropic.text(),
            "api_key_Gemini": self.edit_gemini.text(),
            "api_key_DeepSeek": self.edit_deepseek.text(),
            "api_key_Custom": self.edit_custom.text(),
            "custom_base_url": self.edit_custom_url.text(),
            "history_folder": self.le_history_folder.text(),
            "template_folder": self.le_template_folder.text(),
        })

