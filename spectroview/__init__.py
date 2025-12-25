import os

VERSION = "0.9.8"


TEXT_EXPIRE = (
    "The current SPECTROview version has expired. Checkout the SPECTROview's "
    "Github page (cf. About) to update newest version."
)

PEAK_MODELS = ["Lorentzian", "Gaussian", "PseudoVoigt", "GaussianAsym",
               "LorentzianAsym"]

FIT_PARAMS = {'method': 'leastsq', 'fit_negative': False, 'fit_outliers': False,
              'max_ite': 200, 'coef_noise': 1, 'xtol': 1.e-4, 'ncpus': 'auto'}


FIT_METHODS = {'Leastsq': 'leastsq', 'Least_squares': 'least_squares',
               'Nelder-Mead': 'nelder', 'SLSQP': 'slsqp'}
PALETTE = ['jet', 'viridis', 'plasma',  'magma',
           'cividis', 'cool', 'hot', 'YlGnBu', 'YlOrRd']
PLOT_STYLES = ['point', 'scatter', 'box', 'bar', 'line', 'trendline', 'wafer', '2Dmap']


DEFAULT_COLORS = [
     '#1a6fdf', '#f14040', '#37ad6b','#515151', '#b177de', 
    '#cc9900', '#00cbcc', '#7d4e4e', '#8e8e00', "#fb6501", 
    '#6699cc', '#6fb802', '#515151', '#f14040', '#1a6fdf',
]

MARKERS = ['o', 's', 'D', '^', '*', 'x', '+', 'v', '<', '>', 'p', 'h', 'H', '|', '_', 'P', 'X', '1', '2', '3', '4','5','6','7','8']
DEFAULT_MARKERS = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o','o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
LEGEND_LOCATION = ['upper right', 'upper left', 'lower left', 'lower right',
                   'center left', 'center right', 'lower center',
                   'upper center', 'center']

X_AXIS_UNIT = ['Wavenumber (cm-1)', 'Wavelength (nm)', 'Emission energy (eV)']


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.dirname(__file__)
RESOURCES_DIR = os.path.join(DIR, "resources")

ICON_DIR = os.path.join(RESOURCES_DIR, "icons")
PLOT_POLICY = os.path.join(RESOURCES_DIR, "plotpolicy.mplstyle")

UI_FILE = os.path.join(DIR, "config", "gui", "gui.ui")
LOGO_APPLI = os.path.join(RESOURCES_DIR, "icons", "logo_spectroview.png")

#ABOUT = os.path.join(RESOURCES_DIR, "doc", "about.md")
USER_MANUAL_MD = "https://github.com/CEA-MetroCarac/SPECTROview/blob/main/doc/user_manual.md"
USER_MANUAL_PDF = os.path.join(RESOURCES_DIR, "SPECTROview_UserManual.pdf")
