"""Lower tabs package: Fit Model Builder, Peak Table, Fit Results, MVA, and More."""

from spectroview.view.components.lower_tabs.v_fit_model_builder import VFitModelBuilder
from spectroview.view.components.lower_tabs.v_fit_results import VFitResults
from spectroview.view.components.lower_tabs.v_moretab import VMoreTab
from spectroview.view.components.lower_tabs.v_mva import VMVA
from spectroview.view.components.lower_tabs.v_mva_plots import MVAPlotMixin, _PlotTab
from spectroview.view.components.lower_tabs.v_peak_table import ROW_HEIGHT, VPeakTable

__all__ = [
    "ROW_HEIGHT",
    "MVAPlotMixin",
    "VFitModelBuilder",
    "VFitResults",
    "VMVA",
    "VMoreTab",
    "VPeakTable",
    "_PlotTab",
]
