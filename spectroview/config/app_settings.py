import os
from dataclasses import dataclass, field
from typing import Any
from PySide6.QtCore import QSettings
from typing import Literal
import logging

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,QDialogButtonBox,
     QFileDialog, QLabel, QLineEdit, QCheckBox, QSpinBox, QComboBox,QSpacerItem, QSizePolicy,QDoubleSpinBox
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Signal, Qt

logger = logging.getLogger(__name__)


class Settings_Dialog(QDialog):
    """Open dialog to set application settings."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.settings = QSettings("CEA-Leti", "SPECTROview")
        
        self.setWindowTitle("Settings")
        self.resize(400, 400)

        self._create_ui()
                
        # --- Connect model folder button ---
        self.btn_model_folder.clicked.connect(self.specify_fit_model_folder)

        self.btn_spike_removal.clicked.connect(self.spike_removal)
                
    def spike_removal(self):
        # Placeholder for spike removal functionality
        print("Spike removal clicked")
        
    def specify_fit_model_folder(self):
        """Open folder dialog and save path."""
        folder = QFileDialog.getExistingDirectory(self, "Select Fit Model Folder", "")
        if folder:
            self.le_model_folder.setText(folder)
            self.settings.setValue("model_folder", folder)
            self.settings.sync()
            print(f"✅ Default model folder set to: {folder}")    

    def get_fit_settings(self):
        """Return current fit settings as a dictionary."""
        return {
            "fit_negative": self.chk_fit_negative.isChecked(),
            "method": self.cbb_fit_method.currentText(),
            "max_ite": self.spin_max_iter.value(),
            "xtol": self.spin_x_tol.value(),
            "ncpu": self.spin_cpu.value(),
        }

    def accept(self):
        """Save all settings when dialog is accepted."""
        fit_settings = self.get_fit_settings()

        # Save each setting properly
        for key, value in fit_settings.items():
            self.settings.setValue(f"fit_settings/{key}", value)

        self.settings.sync()  # Ensure settings are written to disk
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

        # Apply loaded values to widgets
        self.chk_fit_negative.setChecked(fit_negative)
        self.cbb_fit_method.setCurrentText(method)
        self.spin_max_iter.setValue(max_ite)
        self.spin_x_tol.setValue(xtol)
        self.spin_cpu.setValue(ncpu)

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
        method_label = QLabel("Fit Method")
        self.cbb_fit_method = QComboBox()
        self.cbb_fit_method.addItems(['Leastsq', 'Least_squares', 'Nelder-Mead', 'SLSQP'])
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.cbb_fit_method)
        main_layout.addLayout(method_layout)
        
        # Maximum iterations
        iter_layout = QHBoxLayout()
        iter_label = QLabel("Maximum iterations")
        self.spin_max_iter = QSpinBox()
        self.spin_max_iter.setRange(1, 10000)
        self.spin_max_iter.setSingleStep(20)
        self.spin_max_iter.setValue(200)
        iter_layout.addWidget(iter_label)
        iter_layout.addWidget(self.spin_max_iter)
        main_layout.addLayout(iter_layout)

       
        # X-tolerance line edit
        tol_layout = QHBoxLayout()
        tol_label = QLabel("x-tolerance")
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
        cpu_label = QLabel("Number of CPU cores")
        self.spin_cpu = QSpinBox()   
        self.spin_cpu.setRange(1, os.cpu_count() or 64)  # Default to 8 if os.cpu_count() is None
        self.spin_cpu.setValue(1)  # Default value
        cpu_layout.addWidget(cpu_label) 
        cpu_layout.addWidget(self.spin_cpu) 
        main_layout.addLayout(cpu_layout)

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
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setMaximumWidth(60)
        folder_layout.addWidget(self.btn_model_folder)
        folder_layout.addWidget(self.le_model_folder)
        folder_layout.addWidget(self.btn_refresh)
        main_layout.addLayout(folder_layout)

        main_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        # --- Spike removal button at the end ---
        self.btn_spike_removal = QPushButton("Spike removal")
        main_layout.addWidget(self.btn_spike_removal)
        
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
        self.exec()

        

def _getattr_by_path(obj: Any, path: str):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set_by_path(obj: Any, path: str, value):
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


@dataclass
class FitSettings:
    #Fitting
    ncpu: int = 1
    fit_negative: bool = False
    max_iteration: int = 500
    method: str = "leastsq"
    xtol: float = 1e-4
    
    #Baseline settings
    baseline_attached: bool = True
    baseline_type: Literal["linear", "polynomial"] = "linear"
    baseline_degre: int = 1
    baseline_noise: int = 4



@dataclass
class GraphSettings:
    grid: bool = False


@dataclass
class AppSettings:
    qsettings: QSettings = field(init=False)

    # last_directory: str = ""
    # default_model_folder: str = ""
    mode: str = "dark"  # "light" or "dark"

    map_type: str = "2Dmap"

    maps: FitSettings = field(default_factory=FitSettings)
    spectra: FitSettings = field(default_factory=FitSettings)
    graph: GraphSettings = field(default_factory=GraphSettings)

    SETTINGS_KEY_MAPPING = {
        #MAPS TAB
        "ncpu": ("maps.ncpu", int),
        "fit_negative": ("maps.fit_negative", bool),
        "max_ite": ("maps.max_iteration", int),
        "method": ("maps.method", str),
        "xtol": ("maps.xtol", float),

        "baseline_attached": ("maps.baseline_attached", bool),
        "baseline_type": ("maps.baseline_type", str),
        "baseline_degre": ("maps.baseline_degre", int),
        "baseline_noise": ("maps.baseline_noise", int),
         "map_type": ("map_type", str),
        
        #SPECTRA TAB
        # "ncpu_2": ("spectra.ncpu", int),
        # "fit_negative2": ("spectra.fit_negative", bool),
        # "max_ite2": ("spectra.max_iteration", int),
        # "method2": ("spectra.method", str),
        # "xtol2": ("spectra.xtol", float),

        "baseline_attached2": ("spectra.baseline_attached", bool),
        "baseline_type2": ("spectra.baseline_type", str),
        "baseline_degre2": ("spectra.baseline_degre", int),
        "baseline_noise2": ("spectra.baseline_noise", int),

        #GRAHP TAB
        "grid": ("graph.grid", bool),

        #MAIN GUI
        "mode": ("mode", str),
        #"last_directory": ("last_directory", str),
        #"default_model_folder": ("default_model_folder", str),

    }

    def __post_init__(self):
        QSettings.setDefaultFormat(QSettings.IniFormat)
        self.qsettings = QSettings("CEA-Leti", "SPECTROview")

    def update_from_ui(self, ui: Any):
        """
        Pull current values from UI widgets into this structured settings object.
        """
        try:
            #MAPS TAB
            # self.maps.ncpu = ui.ncpu.value()
            # self.maps.fit_negative = ui.cb_fit_negative.isChecked()
            # self.maps.max_iteration = ui.max_iteration.value()
            # self.maps.method = ui.cbb_fit_methods.currentText()
            # try:
            #     self.maps.xtol = float(ui.xtol.text())
            # except (ValueError, TypeError):
            #     pass

            self.maps.baseline_attached = ui.cb_attached.isChecked()
            self.maps.baseline_type = "polynomial" if ui.rbtn_polynomial.isChecked() else "linear"
            self.maps.baseline_degre = ui.degre.value()
            self.maps.baseline_noise= ui.noise.value()

            #SPECTRA TAB
            # self.spectra.ncpu = ui.ncpu_2.value()
            # self.spectra.fit_negative = ui.cb_fit_negative_2.isChecked()
            # self.spectra.max_iteration = ui.max_iteration_2.value()
            # self.spectra.method = ui.cbb_fit_methods_2.currentText()
            # try:
            #     self.spectra.xtol = float(ui.xtol_2.text())
            # except (ValueError, TypeError):
            #     pass

            self.spectra.baseline_attached = ui.cb_attached_2.isChecked()
            self.spectra.baseline_type = "polynomial" if ui.rbtn_polynomial_2.isChecked() else "linear"
            self.spectra.baseline_degre = ui.degre_2.value()
            self.spectra.baseline_noise= ui.noise_2.value()
            
            #GRAHP TAB
            self.graph.grid = ui.cb_grid.isChecked()
            logger.debug("Updating from UI: maps.baseline_type=%s, spectra.baseline_type=%s", 
             self.maps.baseline_type, self.spectra.baseline_type)


        except Exception:  
            logger.exception("Failed to update AppSettings from UI")

    def apply_to_ui(self, ui: Any):
        """Push stored settings into UI widgets."""
        try:
            #MAPS TAB
            # ui.ncpu.setValue(self.maps.ncpu)
            # ui.cb_fit_negative.setChecked(self.maps.fit_negative)
            # ui.max_iteration.setValue(self.maps.max_iteration)
            # ui.cbb_fit_methods.setCurrentText(self.maps.method)
            # ui.xtol.setText(str(self.maps.xtol))

            ui.cb_attached.setChecked(self.maps.baseline_attached)
            ui.rbtn_linear.setChecked(self.maps.baseline_type == "linear")
            ui.rbtn_polynomial.setChecked(self.maps.baseline_type == "polynomial")
            ui.degre.setValue(self.maps.baseline_degre)
            ui.noise.setValue(self.maps.baseline_noise)
            
            #SPECTRA TAB
            # ui.ncpu_2.setValue(self.spectra.ncpu)
            # ui.cb_fit_negative_2.setChecked(self.spectra.fit_negative)
            # ui.max_iteration_2.setValue(self.spectra.max_iteration)
            # ui.cbb_fit_methods_2.setCurrentText(self.spectra.method)
            # ui.xtol_2.setText(str(self.spectra.xtol))

            ui.cb_attached_2.setChecked(self.spectra.baseline_attached) 
            ui.rbtn_linear_2.setChecked(self.spectra.baseline_type == "linear")
            ui.rbtn_polynomial_2.setChecked(self.spectra.baseline_type == "polynomial")
            ui.degre_2.setValue(self.spectra.baseline_degre)
            ui.noise_2.setValue(self.spectra.baseline_noise)

            #GRAHP TAB
            ui.cb_grid.setChecked(self.graph.grid)
            logger.debug("Applying to UI: maps.baseline_type=%s, spectra.baseline_type=%s", 
             self.maps.baseline_type, self.spectra.baseline_type)


        except Exception: 
            logger.exception("Failed to apply AppSettings to UI")


    def sync_app_settings(self, ui: Any, *args, **kwargs):
        """Update settings from the UI and immediately persist them."""
        try:
            self.update_from_ui(ui)
            self.save()
        except Exception:  
            logger.exception("Failed to sync settings from UI and persist")

    def load(self):
        """Load all mapped settings from persistent storage into this object."""
        # self.last_directory = self.qsettings.value("last_directory", "", str)
        # self.default_model_folder = self.qsettings.value("default_model_folder", "", str)

        for key, (attr_path, expected_type) in self.SETTINGS_KEY_MAPPING.items():
            try:
                default_val = _getattr_by_path(self, attr_path)
                raw = self.qsettings.value(key, defaultValue=default_val, type=expected_type)
                if expected_type is bool:
                    val = bool(raw)
                elif expected_type is int:
                    val = int(raw)
                elif expected_type is float:
                    val = float(raw)
                else:
                    val = raw
                _set_by_path(self, attr_path, val)
            except Exception: 
                logger.exception("Failed to load setting %s", key)

    def save(self):
        """Persist all mapped settings from this object into storage."""
        # self.qsettings.setValue("last_directory", self.last_directory)
        # self.qsettings.setValue("default_model_folder", self.default_model_folder)

        for key, (attr_path, _) in self.SETTINGS_KEY_MAPPING.items():
            try:
                value = _getattr_by_path(self, attr_path)
                self.qsettings.setValue(key, value)
            except Exception:  
                logger.exception("Failed to save setting %s", key)
                
                
