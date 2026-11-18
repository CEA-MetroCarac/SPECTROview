"""Reusable UI component packages with PEP 562 lazy subpackage resolution.

Subpackages:
    - ``common``: Menu bar, notifications, update banner, about, quick calculators, shared widgets.
    - ``customize_graph``: Two-panel Customize Graph dialog and tab widgets.
    - ``graphs``: ``VGraph``, ``PlotRenderer``, ``VDataFilter``, export/multipanel/recipe/style dialogs.
    - ``lists``: ``VSpectraList``, ``VMapsList``, ``VDataframeTable``, metadata tooltips.
    - ``lower_tabs``: Fit Model Builder, Peak Table, Fit Results, MVA, and More tabs.
    - ``map_viewer``: ``VMapViewer``, ``VMapViewerDialog``, and spatial selection helpers.
    - ``rgb_live_maps``: ``VLiveMaps``, ``LiveMapSubpanel``, and RGB range selectors.
    - ``spectra_viewer``: ``VSpectraViewer`` (1D, Heatmap, 3D, interactions, legend editor).
"""

from __future__ import annotations

import importlib

_SUBPACKAGES = frozenset({
    "common",
    "customize_graph",
    "graphs",
    "lists",
    "lower_tabs",
    "map_viewer",
    "rgb_live_maps",
    "spectra_viewer",
})

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "VSettingsDialog": ("spectroview.view.components.v_settings", "VSettingsDialog"),
    "SettingsDialog": ("spectroview.view.components.v_settings", "SettingsDialog"),
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
