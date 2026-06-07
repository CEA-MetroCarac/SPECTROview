
import sys
from pathlib import Path
VERSION = "26.25.1"

TEXT_EXPIRE = (
    "The current SPECTROview version has expired. Checkout the SPECTROview's "
    "Github page (cf. About) to update newest version."
)


# Predefined axis labels for autocomplete
AXIS_LABELS = [
    "Si peak position (cm$^{-1}$)",
    "Si peak FWHM (cm$^{-1}$)",
    "Si peak intensity (a.u.)", 
    r"Stress $\sigma_{yy}$ (MPa)", 
    r"Strain $\epsilon_{xx}$ (%)",
]

PEAK_MODELS = [
    "Lorentzian", "Gaussian", "PseudoVoigt",
    "GaussianAsym", "LorentzianAsym", "Fano",
    "DecaySingleExp", "DecayBiExp"
]

PALETTE = [
    'jet', 'viridis', 'plasma', 'magma',
    'cividis', 'cool', 'hot', 'YlGnBu', 'YlOrRd', 'seismic', 'bwr', 'Spectral',
]

PLOT_STYLES = [
    'point', 'scatter', 'box', 'bar',
    'line', 'trendline', 'histogram', 'wafer', '2Dmap'
]

DEFAULT_COLORS = [
    '#E31A1C', '#33A02C', '#FF7F00', '#1F78B4', '#6A3D9A',
    '#FB9A99', '#B2DF8A', '#FDBF6F', '#A6CEE3', '#CAB2D6',
]

MARKERS = [
    'o', 's', 'D', '^', '*', 'x', '+', 'v', '<', '>',
    'p', 'h', 'H', '|', '_', 'P', 'X',
    '1', '2', '3', '4', '5', '6', '7', '8'
]

DEFAULT_MARKERS = ['o'] * len(MARKERS)

X_AXIS_UNIT = [
    'Wavenumber (cm$^{-1}$)',
    'Wavelength (nm)',
    'Emission energy (eV)',
    'Binding energy (eV)',
    'Frequency (GHz)',
    r'$2\theta$ (°)',
    'Time (ns)',
]

Y_AXIS_UNIT = [
    'Intensity (a.u.)',
    'Intensity (a.u./s)',
]
 
# 🔑 PyInstaller-safe resource path helper
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


# 📁 Resource paths 
RESOURCES_DIR = resource_path("spectroview/resources")

ICON_DIR = resource_path("spectroview/resources/icons")
PLOT_POLICY = resource_path("spectroview/resources/plotpolicy.mplstyle")
PLOT_POLICY_LIGHT = resource_path("spectroview/resources/plotpolicy_light.mplstyle")
PLOT_POLICY_DARK = resource_path("spectroview/resources/plotpolicy_dark.mplstyle")

UI_FILE = resource_path("spectroview/config/gui/gui.ui")
LOGO_APPLI = resource_path("spectroview/resources/icons/logo_spectroview.png")

USER_MANUAL_DIR = resource_path(
    "spectroview/resources/user_manual"
)


