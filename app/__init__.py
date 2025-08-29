import os

VERSION = "0.4.9"

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
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', "#17b8ca", 
    '#ffd500', '#008281', '#000086', '#c0c0c0', '#808000', 
    '#8d0000', '#6fd0ef', '#ff1493', '#00ff7f', '#ff4500', 
    '#191970', '#ffdab9', '#228b22', '#dda0dd', '#ff6347', 
]
MARKERS = ['o', 's', 'D', '^', '*', 'x', '+', 'v', '<', '>', 'p', 'h', 'H', '|', '_', 'P', 'X', '1', '2', '3', '4','5','6','7','8']
DEFAULT_MARKERS = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o','o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
LEGEND_LOCATION = ['upper right', 'upper left', 'lower left', 'lower right',
                   'center left', 'center right', 'lower center',
                   'upper center', 'center']

X_AXIS_UNIT = ['Wavenumber (cm-1)', 'Wavelength (nm)', 'Emission energy (eV)']


DIR = os.path.dirname(__file__)
RESOURCES_DIR = os.path.join(DIR, "resources")

ICON_DIR = os.path.join(DIR, "resources", "icons")
PLOT_POLICY = os.path.join(DIR, "resources", "plotpolicy.mplstyle")

UI_FILE = os.path.join(DIR, "resources", "ui", "gui.ui")
LOGO_APPLI = os.path.join(DIR, "resources", "icons", "logo_spectroview.png")

ABOUT = os.path.join(DIR, "resources", "about.md")
USER_MANUAL = os.path.join(DIR, "doc", "user_manual.md")