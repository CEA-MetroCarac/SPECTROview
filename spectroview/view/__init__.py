"""SPECTROview View layer package with PEP 562 lazy resolution.

Subpackages:
    - ``workspaces``: Top-level workspace views (Spectra, Maps, Graphs).
    - ``components``: Reusable UI widgets, viewers, dialogs, and lists.
    - ``theme``: Design tokens, QSS templates, palettes, fonts, and SVG icons.
"""

from __future__ import annotations

import importlib

_SUBPACKAGES = frozenset({
    "components",
    "theme",
    "workspaces",
})

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Workspaces
    "VWorkspaceSpectra": ("spectroview.view.workspaces.v_workspace_spectra", "VWorkspaceSpectra"),
    "VWorkspaceMaps": ("spectroview.view.workspaces.v_workspace_maps", "VWorkspaceMaps"),
    "VWorkspaceGraphs": ("spectroview.view.workspaces.graphs.v_workspace_graphs", "VWorkspaceGraphs"),
    "MdiSubWindow": ("spectroview.view.workspaces.graphs.v_mdi_subwindow", "MdiSubWindow"),
    # Core components
    "VSpectraViewer": ("spectroview.view.components.spectra_viewer", "VSpectraViewer"),
    "VMapViewer": ("spectroview.view.components.map_viewer", "VMapViewer"),
    "VMapViewerDialog": ("spectroview.view.components.map_viewer", "VMapViewerDialog"),
    "VLiveMaps": ("spectroview.view.components.rgb_live_maps", "VLiveMaps"),
    "VGraph": ("spectroview.view.components.graphs", "VGraph"),
    "PlotRenderer": ("spectroview.view.components.graphs", "PlotRenderer"),
    "CustomizeGraphDialog": ("spectroview.view.components.customize_graph", "CustomizeGraphDialog"),
    "VMenuBar": ("spectroview.view.components.common", "VMenuBar"),
    "VUpdateBanner": ("spectroview.view.components.common", "VUpdateBanner"),
    "NotificationManager": ("spectroview.view.components.common", "NotificationManager"),
    "MQuickCalc": ("spectroview.view.components.common", "MQuickCalc"),
    # Theme
    "ThemeManager": ("spectroview.view.theme", "ThemeManager"),
    "Icons": ("spectroview.view.theme", "Icons"),
}

__all__ = sorted(_SUBPACKAGES | _LAZY_EXPORTS.keys())


def __getattr__(name: str):
    if name in _SUBPACKAGES:
        return importlib.import_module(f"{__name__}.{name}")
    if name in _LAZY_EXPORTS:
        module_path, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
