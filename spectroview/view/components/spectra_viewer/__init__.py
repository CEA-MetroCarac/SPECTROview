"""Spectra viewer package: 1D/2D/3D canvas, interactions, and legend editor."""

from spectroview.view.components.spectra_viewer.common import (
    LEGEND_EDITOR_MAX_ROWS,
    SPECTRA_COLOR_PALETTES,
    SPECTRA_CUSTOM_PALETTES,
    SPECTRA_DISCRETE_PALETTES,
    SPECTRA_GRADIENT_PALETTES,
    get_cached_style_dict,
)
from spectroview.view.components.spectra_viewer.v_interactions import (
    SpectraInteractionMixin,
)
from spectroview.view.components.spectra_viewer.v_plot_1d import SpectraPlot1DMixin
from spectroview.view.components.spectra_viewer.v_plot_3d import (
    FastLine3DCollection,
    SpectraPlot3DMixin,
)
from spectroview.view.components.spectra_viewer.v_plot_heatmap import (
    SpectraHeatmapMixin,
)
from spectroview.view.components.spectra_viewer.v_spectra_legend_editor import (
    SpectraLegendEditorDialog,
    SpectraLegendMixin,
    SpectrumColorComboBox,
)
from spectroview.view.components.spectra_viewer.v_spectra_viewer import VSpectraViewer

__all__ = [
    "VSpectraViewer",
    "FastLine3DCollection",
    "SpectraLegendEditorDialog",
    "SpectrumColorComboBox",
    "SPECTRA_DISCRETE_PALETTES",
    "SPECTRA_GRADIENT_PALETTES",
    "SPECTRA_COLOR_PALETTES",
    "SPECTRA_CUSTOM_PALETTES",
    "LEGEND_EDITOR_MAX_ROWS",
    "get_cached_style_dict",
    "SpectraPlot1DMixin",
    "SpectraHeatmapMixin",
    "SpectraPlot3DMixin",
    "SpectraLegendMixin",
    "SpectraInteractionMixin",
]
