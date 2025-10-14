import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,QDialogButtonBox,
     QFileDialog, QLabel, QLineEdit, QCheckBox, QSpinBox, QComboBox,QSpacerItem, QSizePolicy,QDoubleSpinBox
)
from PySide6.QtGui import QFont

class SettingsPanel(QDialog):
    """Open dialog to set general settings of the application."""
    
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        
        self.setWindowTitle("Settings")
        self.resize(400, 400)

        self._create_ui()
                
        # --- Connect model folder button ---
        self.btn_model_folder.clicked.connect(self.specify_fit_model_folder)                
        
    def specify_fit_model_folder(self):
        """Open folder dialog and save path."""
        folder = QFileDialog.getExistingDirectory(self, "Select Fit Model Folder", "")
        if folder:
            self.le_model_folder.setText(folder)
            self.settings.setValue("model_folder", folder)
            self.settings.sync()
            print(f"✅ Default model folder set to: {folder}")    

    def get_fit_settings(self):
        """Return current fit settings from setting panel as a dictionary."""
        return {
            "fit_negative": self.chk_fit_negative.isChecked(),
            "method": self.cbb_fit_method.currentText(),
            "max_ite": self.spin_max_iter.value(),
            "xtol": self.spin_x_tol.value(),
            "ncpu": self.spin_cpu.value(),
            "maxshift": self.spin_maxshift.value(),
            "maxfwhm": self.spin_maxfwhm.value(),
            
        }

    def accept(self):
        """Save all settings to QSetting when dialog is accepted."""
        fit_settings = self.get_fit_settings()

        # Save each setting properly
        for key, value in fit_settings.items():
            self.settings.setValue(f"fit_settings/{key}", value)

        self.settings.sync()  
        print("All Settings are saved and applied")
        super().accept()

    def reject(self):
        """Handle Cancel button — discard changes."""
        print("Settings dialog cancelled")
        super().reject()
        
    
    def load_settings(self):
        """Load saved settings into the dialog widgets."""
        # Retrieve each setting with a default value
        fit_negative = self.settings.value("fit_settings/fit_negative", False, type=bool)
        method = self.settings.value("fit_settings/method", "Leastsq")
        max_ite = self.settings.value("fit_settings/max_ite", 200, type=int)
        xtol = self.settings.value("fit_settings/xtol", 1e-4, type=float)
        ncpu = self.settings.value("fit_settings/ncpu", 1, type=int)
        maxshift = self.settings.value("fit_settings/maxshift", 20, type=float)
        maxfwhm = self.settings.value("fit_settings/maxfwhm", 200, type=float)
    
        # Apply loaded values to widgets
        self.chk_fit_negative.setChecked(fit_negative)
        self.cbb_fit_method.setCurrentText(method)
        self.spin_max_iter.setValue(max_ite)
        self.spin_x_tol.setValue(xtol)
        self.spin_cpu.setValue(ncpu)    
        self.spin_maxshift.setValue(maxshift)
        self.spin_maxfwhm.setValue(maxfwhm) 

        # Remember model folder path if saved
        model_folder = self.settings.value("model_folder", "")
        self.le_model_folder.setText(model_folder)
        
    def _create_ui(self):
        """Builds the UI layout for the settings window."""
        main_layout = QVBoxLayout(self)

        # --- Fit Settings ---
        fit_label = QLabel("Fit Settings:")
        font_bold = QFont()
        font_bold.setBold(True)
        fit_label.setFont(font_bold)
        main_layout.addWidget(fit_label)

        # Fit negative checkbox
        self.chk_fit_negative = QCheckBox("Fit negative value")
        main_layout.addWidget(self.chk_fit_negative)

         # Fit Method combobox
        method_layout = QHBoxLayout()
        method_label = QLabel("Fit Method:")
        self.cbb_fit_method = QComboBox()
        self.cbb_fit_method.addItems(['Leastsq', 'Least_squares', 'Nelder-Mead', 'SLSQP'])
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.cbb_fit_method)
        main_layout.addLayout(method_layout)
        
        # Maximum iterations
        iter_layout = QHBoxLayout()
        iter_label = QLabel("Maximum iterations:")
        self.spin_max_iter = QSpinBox()
        self.spin_max_iter.setRange(1, 10000)
        self.spin_max_iter.setSingleStep(20)
        self.spin_max_iter.setValue(200)
        iter_layout.addWidget(iter_label)
        iter_layout.addWidget(self.spin_max_iter)
        main_layout.addLayout(iter_layout)

        # X-tolerance line edit
        tol_layout = QHBoxLayout()
        tol_label = QLabel("x-tolerance:")
        self.spin_x_tol = QDoubleSpinBox()
        self.spin_x_tol.setRange(1e-5, 1e-3)
        self.spin_x_tol.setSingleStep(1e-5)
        self.spin_x_tol.setDecimals(6)  # Number of decimal digits
        self.spin_x_tol.setValue(1e-4)  # Default value
        tol_layout.addWidget(tol_label)
        tol_layout.addWidget(self.spin_x_tol)
        main_layout.addLayout(tol_layout)
        
        # Number of CPU cores
        cpu_layout = QHBoxLayout()
        cpu_label = QLabel("Number of CPU cores:")
        self.spin_cpu = QSpinBox()   
        self.spin_cpu.setRange(1, os.cpu_count() or 64)  # Default to 8 if os.cpu_count() is None
        self.spin_cpu.setValue(1)  # Default value
        cpu_layout.addWidget(cpu_label) 
        cpu_layout.addWidget(self.spin_cpu) 
        main_layout.addLayout(cpu_layout)

        # Max peak shift
        maxshift_layout = QHBoxLayout()
        maxshift_lb = QLabel("Max peak shift:")
        self.spin_maxshift = QDoubleSpinBox()
        self.spin_maxshift.setRange(0, 100)
        self.spin_maxshift.setSingleStep(5)
        self.spin_maxshift.setDecimals(2) 
        self.spin_maxshift.setValue(20)  # Default value
        maxshift_layout.addWidget(maxshift_lb)
        maxshift_layout.addWidget(self.spin_maxshift)
        main_layout.addLayout(maxshift_layout)
        
        # Max peak fwhm
        maxfwhm_layout = QHBoxLayout()
        maxfwhm_lb = QLabel("Max peak fwhm:")
        self.spin_maxfwhm = QDoubleSpinBox()
        self.spin_maxfwhm.setRange(0, 500)
        self.spin_maxfwhm.setSingleStep(20)
        self.spin_maxfwhm.setDecimals(2) 
        self.spin_maxfwhm.setValue(200)  # Default value
        maxfwhm_layout.addWidget(maxfwhm_lb)
        maxfwhm_layout.addWidget(self.spin_maxfwhm)
        main_layout.addLayout(maxfwhm_layout)

        ################################################################################
        main_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # --- Fit model management label and Model folder line ---
        model_layout = QHBoxLayout()
        model_label = QLabel("Fit model management:")
        model_label.setFont(font_bold)
        model_layout.addWidget(model_label)
        main_layout.addLayout(model_layout)

        folder_layout = QHBoxLayout()
        self.btn_model_folder = QPushButton("Path:")
        self.btn_model_folder.setMaximumWidth(40)
        self.le_model_folder = QLineEdit()
        # self.btn_refresh = QPushButton("Refresh")
        # self.btn_refresh.setMaximumWidth(60)
        folder_layout.addWidget(self.btn_model_folder)
        folder_layout.addWidget(self.le_model_folder)
        # folder_layout.addWidget(self.btn_refresh)
        main_layout.addLayout(folder_layout)

        main_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
                
        # --- OK / Cancel buttons ---
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok_button = button_box.button(QDialogButtonBox.Ok)
        cancel_button = button_box.button(QDialogButtonBox.Cancel)
        # Set background colors
        ok_button.setStyleSheet("background-color: green; color: white; font-weight: bold;")
        cancel_button.setStyleSheet("background-color: red; color: white; font-weight: bold;")

        button_box.accepted.connect(self.accept)  
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

    def launch(self):
        """Show the settings dialog with preloaded settings."""
        self.load_settings()  # Load previously saved values first
        self.exec()  # Show the dialog modally