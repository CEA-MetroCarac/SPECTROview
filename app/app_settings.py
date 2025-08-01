# app/app_settings.py
from dataclasses import dataclass, field
from PySide6.QtCore import QSettings
import logging

logger = logging.getLogger(__name__)


def _getattr_by_path(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set_by_path(obj, path: str, value):
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
    maps: FitSettings = field(default_factory=FitSettings)
    spectra: FitSettings = field(default_factory=FitSettings)
    visualization: VisualizationSettings = field(default_factory=VisualizationSettings)

    # mapping between QSettings keys and attribute paths + expected types
    SETTINGS_KEY_MAPPING = {
        "ncpu": ("ncpu", int),
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
    }

    def load_from_qsettings(self, qsettings: QSettings):
        for key, (attr_path, expected_type) in self.SETTINGS_KEY_MAPPING.items():
            try:
                default_val = _getattr_by_path(self, attr_path)
                raw = qsettings.value(key, defaultValue=default_val, type=expected_type)
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

    def save_to_qsettings(self, qsettings: QSettings):
        for key, (attr_path, _) in self.SETTINGS_KEY_MAPPING.items():
            try:
                value = _getattr_by_path(self, attr_path)
                qsettings.setValue(key, value)
            except Exception:
                logger.exception("Failed to save setting %s", key)
