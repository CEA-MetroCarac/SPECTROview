# spectroview/view/components/customize_graph/__init__.py
"""Customize Graph package: two-panel Settings-like dialog and modular tabs."""

from spectroview.view.components.customize_graph.customize_annotation_dialogs import (
    ColorDelegate,
    EditArrowDialog,
    EditBoxDialog,
    EditCalloutDialog,
    EditLineDialog,
    EditSpanDialog,
    EditTextDialog,
)
from spectroview.view.components.customize_graph.customize_annotations import CustomizeAnnotations
from spectroview.view.components.customize_graph.customize_axis import CustomizeAxis
from spectroview.view.components.customize_graph.customize_graph_dialog import (
    CustomizeGraphDialog,
)
from spectroview.view.components.customize_graph.customize_legend import CustomizeLegend
from spectroview.view.components.customize_graph.customize_more_options import CustomizeMoreOptions
from spectroview.view.components.customize_graph.tab_annotations import TabAnnotations
from spectroview.view.components.customize_graph.tab_axes_limits import TabAxesLimits
from spectroview.view.components.customize_graph.tab_general import TabGeneral
from spectroview.view.components.customize_graph.tab_legend_series import TabLegendSeries
from spectroview.view.components.customize_graph.tab_plot_styles import TabPlotStyles
from spectroview.view.components.customize_graph.tab_secondary_inset import TabSecondaryInset
from spectroview.view.components.customize_graph.tab_ticks_spines import TabTicksSpines

__all__ = [
    "CustomizeGraphDialog",
    "TabGeneral",
    "TabAxesLimits",
    "TabTicksSpines",
    "TabSecondaryInset",
    "TabLegendSeries",
    "TabPlotStyles",
    "TabAnnotations",
    # Backward compatibility exports
    "CustomizeAxis",
    "CustomizeLegend",
    "CustomizeMoreOptions",
    "CustomizeAnnotations",
    "EditLineDialog",
    "EditTextDialog",
    "EditArrowDialog",
    "EditSpanDialog",
    "EditBoxDialog",
    "EditCalloutDialog",
    "ColorDelegate",
]
