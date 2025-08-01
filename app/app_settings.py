# app/app_settings.py
from dataclasses import dataclass, field
from typing import Any
from PySide6.QtCore import QSettings
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
    fit_negative: bool = False
    max_iteration: int = 500
    method: str = "leastsq"
    xtol: float = 1e-4
    attached: bool = True


@dataclass
class VisualizationSettings:
    grid: bool = False


@dataclass
class AppSettings:
    ncpu: int = 1
    ncpu_2: int = 1
    maps: FitSettings = field(default_factory=FitSettings)
    spectra: FitSettings = field(default_factory=FitSettings)
    visualization: VisualizationSettings = field(default_factory=VisualizationSettings)
    mode: str = "dark"  # "light" or "dark"

    qsettings: QSettings = field(init=False)

    # mapping between QSettings keys and attribute paths + expected types
    SETTINGS_KEY_MAPPING = {
        "ncpu": ("ncpu", int),
        "ncpu_2": ("ncpu_2", int),
        "fit_negative": ("maps.fit_negative", bool),
        "max_ite": ("maps.max_iteration", int),
        "method": ("maps.method", str),
        "xtol": ("maps.xtol", float),
        "attached": ("maps.attached", bool),

        "fit_negative2": ("spectra.fit_negative", bool),
        "max_ite2": ("spectra.max_iteration", int),
        "method2": ("spectra.method", str),
        "xtol2": ("spectra.xtol", float),
        "attached2": ("spectra.attached", bool),

        "grid": ("visualization.grid", bool),
        "mode": ("mode", str),
    }

    def __post_init__(self):
        QSettings.setDefaultFormat(QSettings.IniFormat)
        self.qsettings = QSettings("CEA-Leti", "SPECTROview")

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
            except Exception:  # pragma: no cover
                logger.exception("Failed to load setting %s", key)

    def save(self):
        """Persist all mapped settings from this object into storage."""
        for key, (attr_path, _) in self.SETTINGS_KEY_MAPPING.items():
            try:
                value = _getattr_by_path(self, attr_path)
                self.qsettings.setValue(key, value)
            except Exception:  # pragma: no cover
                logger.exception("Failed to save setting %s", key)

    def update_from_ui(self, ui: Any):
        """
        Pull current values from UI widgets into this structured settings object.
        """
        try:
            self.ncpu = ui.ncpus.value()
            self.ncpu_2 = ui.ncpus_2.value()
            self.maps.fit_negative = ui.cb_fit_negative.isChecked()
            self.maps.max_iteration = ui.max_iteration.value()
            self.maps.method = ui.cbb_fit_methods.currentText()
            try:
                self.maps.xtol = float(ui.xtol.text())
            except (ValueError, TypeError):
                pass
            self.maps.attached = ui.cb_attached.isChecked()

            self.spectra.fit_negative = ui.cb_fit_negative_2.isChecked()
            self.spectra.max_iteration = ui.max_iteration_2.value()
            self.spectra.method = ui.cbb_fit_methods_2.currentText()
            try:
                self.spectra.xtol = float(ui.xtol_2.text())
            except (ValueError, TypeError):
                pass
            self.spectra.attached = ui.cb_attached_2.isChecked()

            self.visualization.grid = ui.cb_grid.isChecked()
        except Exception:  # pragma: no cover
            logger.exception("Failed to update AppSettings from UI")

    def apply_to_ui(self, ui: Any):
        """
        Push stored settings into UI widgets.
        """
        try:
            ui.ncpus.setValue(self.ncpu)
            ui.ncpus_2.setValue(self.ncpu_2)
            ui.cb_fit_negative.setChecked(self.maps.fit_negative)
            ui.max_iteration.setValue(self.maps.max_iteration)
            ui.cbb_fit_methods.setCurrentText(self.maps.method)
            ui.xtol.setText(str(self.maps.xtol))
            # ui.cb_attached.setChecked(self.maps.attached)  # kept commented as in original

            ui.cb_fit_negative_2.setChecked(self.spectra.fit_negative)
            ui.max_iteration_2.setValue(self.spectra.max_iteration)
            ui.cbb_fit_methods_2.setCurrentText(self.spectra.method)
            ui.xtol_2.setText(str(self.spectra.xtol))
            # ui.cb_attached_2.setChecked(self.spectra.attached)  # kept commented

            ui.cb_grid.setChecked(self.visualization.grid)
        except Exception:  # pragma: no cover
            logger.exception("Failed to apply AppSettings to UI")


    def sync_app_settings(self, ui: Any, *args, **kwargs):
        """
        Update settings from the UI and immediately persist them.
        Extra args are tolerated so it can be hooked directly to Qt signals.
        """
        try:
            self.update_from_ui(ui)
            self.save()
        except Exception:  # pragma: no cover
            logger.exception("Failed to sync settings from UI and persist")
