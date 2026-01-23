import os
import sys
from pathlib import Path

VERSION = "26.4.1"


TEXT_EXPIRE = (
    "The current SPECTROview version has expired. Checkout the SPECTROview's "
    "Github page (cf. About) to update newest version."
)

PEAK_MODELS = [
    "Lorentzian", "Gaussian", "PseudoVoigt",
    "GaussianAsym", "LorentzianAsym"
]

FIT_PARAMS = {
    'method': 'leastsq',
    'fit_negative': False,
    'fit_outliers': False,
    'max_ite': 200,
    'coef_noise': 1,
    'xtol': 1.e-4,
    'ncpus': 'auto'
}

FIT_METHODS = {
    'Leastsq': 'leastsq',
    'Least_squares': 'least_squares',
    'Nelder-Mead': 'nelder',
    'SLSQP': 'slsqp'
}

PALETTE = [
    'jet', 'viridis', 'plasma', 'magma',
    'cividis', 'cool', 'hot', 'YlGnBu', 'YlOrRd'
]

PLOT_STYLES = [
    'point', 'scatter', 'box', 'bar',
    'line', 'trendline', 'wafer', '2Dmap'
]

DEFAULT_COLORS = [
    '#1a6fdf', '#f14040', '#37ad6b', '#515151', '#b177de',
    '#cc9900', '#00cbcc', '#7d4e4e', '#8e8e00', '#fb6501',
    '#6699cc', '#6fb802',
]

MARKERS = [
    'o', 's', 'D', '^', '*', 'x', '+', 'v', '<', '>',
    'p', 'h', 'H', '|', '_', 'P', 'X',
    '1', '2', '3', '4', '5', '6', '7', '8'
]

DEFAULT_MARKERS = ['o'] * len(MARKERS)

LEGEND_LOCATION = [
    'best', 'upper right', 'upper left', 'lower left', 'lower right',
    'center left', 'center right', 'lower center',
    'upper center', 'center'
]

X_AXIS_UNIT = [
    'Wavenumber (cm⁻¹)',
    'Wavelength (nm)',
    'Emission energy (eV)',
    'Binding energy (eV)',
    'Frequency (GHz)',
    r'$2\theta$ (°)',
]


# ---------------------------------------------------------------------
# 🔑 PyInstaller-safe resource path helper
# ---------------------------------------------------------------------
def resource_path(relative_path: str) -> str:
    """
    Return absolute path to resource.
    Works in development and in PyInstaller onefile/onedir.
    """
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller extracts files here
        base_path = Path(sys._MEIPASS)
    else:
        # Project root (one level ABOVE spectroview/)
        base_path = Path(__file__).resolve().parent.parent

    return str(base_path / relative_path)


# ---------------------------------------------------------------------
# 📁 Resource paths 
# ---------------------------------------------------------------------
RESOURCES_DIR = resource_path("spectroview/resources")

ICON_DIR = resource_path("spectroview/resources/icons")
PLOT_POLICY = resource_path("spectroview/resources/plotpolicy.mplstyle")

UI_FILE = resource_path("spectroview/config/gui/gui.ui")
LOGO_APPLI = resource_path("spectroview/resources/icons/logo_spectroview.png")

USER_MANUAL_PDF = resource_path(
    "spectroview/resources/SPECTROview_UserManual.pdf"
)
