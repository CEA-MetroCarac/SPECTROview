# app/app_settings.py
from dataclasses import dataclass, field
from typing import Any
from PySide6.QtCore import QSettings
from typing import Literal
import logging

logger = logging.getLogger(__name__)


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
    maps: FitSettings = field(default_factory=FitSettings)
    spectra: FitSettings = field(default_factory=FitSettings)
    graph: GraphSettings = field(default_factory=GraphSettings)
    mode: str = "dark"  # "light" or "dark"

    qsettings: QSettings = field(init=False)

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


        #SPECTRA TAB
        "ncpu_2": ("spectra.ncpu", int),
        "fit_negative2": ("spectra.fit_negative", bool),
        "max_ite2": ("spectra.max_iteration", int),
        "method2": ("spectra.method", str),
        "xtol2": ("spectra.xtol", float),

        "baseline_attached2": ("spectra.baseline_attached", bool),
        "baseline_type2": ("spectra.baseline_type", str),
        "baseline_degre2": ("spectra.baseline_degre", int),
        "baseline_noise2": ("spectra.baseline_noise", int),

        #GRAHP TAB
        "grid": ("graph.grid", bool),

        #MAIN GUI
        "mode": ("mode", str),
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
            self.maps.ncpu = ui.ncpu.value()
            self.maps.fit_negative = ui.cb_fit_negative.isChecked()
            self.maps.max_iteration = ui.max_iteration.value()
            self.maps.method = ui.cbb_fit_methods.currentText()
            try:
                self.maps.xtol = float(ui.xtol.text())
            except (ValueError, TypeError):
                pass

            self.maps.baseline_attached = ui.cb_attached.isChecked()
            self.maps.baseline_type = "polynomial" if ui.rbtn_polynomial.isChecked() else "linear"
            self.maps.baseline_degre = ui.degre.value()
            self.maps.baseline_noise= ui.noise.value()

            #SPECTRA TAB
            self.spectra.ncpu = ui.ncpu_2.value()
            self.spectra.fit_negative = ui.cb_fit_negative_2.isChecked()
            self.spectra.max_iteration = ui.max_iteration_2.value()
            self.spectra.method = ui.cbb_fit_methods_2.currentText()
            try:
                self.spectra.xtol = float(ui.xtol_2.text())
            except (ValueError, TypeError):
                pass

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
            ui.ncpu.setValue(self.maps.ncpu)
            ui.cb_fit_negative.setChecked(self.maps.fit_negative)
            ui.max_iteration.setValue(self.maps.max_iteration)
            ui.cbb_fit_methods.setCurrentText(self.maps.method)
            ui.xtol.setText(str(self.maps.xtol))

            ui.cb_attached.setChecked(self.maps.baseline_attached)
            ui.rbtn_linear.setChecked(self.maps.baseline_type == "linear")
            ui.rbtn_polynomial.setChecked(self.maps.baseline_type == "polynomial")
            ui.degre.setValue(self.maps.baseline_degre)
            ui.noise.setValue(self.maps.baseline_noise)
            
            #SPECTRA TAB
            ui.ncpu_2.setValue(self.spectra.ncpu)
            ui.cb_fit_negative_2.setChecked(self.spectra.fit_negative)
            ui.max_iteration_2.setValue(self.spectra.max_iteration)
            ui.cbb_fit_methods_2.setCurrentText(self.spectra.method)
            ui.xtol_2.setText(str(self.spectra.xtol))

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
        for key, (attr_path, _) in self.SETTINGS_KEY_MAPPING.items():
            try:
                value = _getattr_by_path(self, attr_path)
                self.qsettings.setValue(key, value)
            except Exception:  
                logger.exception("Failed to save setting %s", key)