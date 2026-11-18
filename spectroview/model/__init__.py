"""Model layer package facade with lazy subpackage and symbol resolution.

Subpackages:
    - ``io``: File detection, format readers (WDF, SPC, L6S, NGS), and ``WorkspaceIO``.
    - ``spectra``: ``SpectraStore``, ``MapData``, spectral parameters, heatmaps, and peak models.
    - ``graphs``: ``MGraph`` schema, patch validation, style partitioning, recipes, and templates.
    - ``analysis``: Multivariate analysis (PCA/NMF) and spectroscopic quick calculators.
    - ``system``: Installation detection, TLS trust store, auto-updater, and OS patches.
"""

from __future__ import annotations

import importlib

_SUBPACKAGES = frozenset({"analysis", "graphs", "io", "spectra", "system"})

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "MSettings": ("spectroview.model.m_settings", "MSettings"),
    "WorkspaceIO": ("spectroview.model.io.workspace_io", "WorkspaceIO"),
    "SpectraStore": ("spectroview.model.spectra.spectra_store", "SpectraStore"),
    "MapData": ("spectroview.model.spectra.spectra_store", "MapData"),
    "SpectrumProxy": ("spectroview.model.spectra.spectra_store", "SpectrumProxy"),
    "BaselineProxy": ("spectroview.model.spectra.spectra_store", "BaselineProxy"),
    "MFitModelManager": ("spectroview.model.spectra.m_fit_model_manager", "MFitModelManager"),
    "build_heatmap_grid": ("spectroview.model.spectra.heatmap", "build_heatmap_grid"),
    "initialize_peak_params": ("spectroview.model.spectra.peak_model", "initialize_peak_params"),
    "MGraph": ("spectroview.model.graphs.m_graph", "MGraph"),
    "GraphValidationError": ("spectroview.model.graphs.graph_control", "GraphValidationError"),
    "MPlotRecipe": ("spectroview.model.graphs.m_plot_recipe", "MPlotRecipe"),
    "MPlotRecipeStore": ("spectroview.model.graphs.m_plot_recipe_store", "MPlotRecipeStore"),
    "MStyleTemplateStore": ("spectroview.model.graphs.m_style_template_store", "MStyleTemplateStore"),
    "MMVA": ("spectroview.model.analysis.m_mva", "MMVA"),
    "PCAResult": ("spectroview.model.analysis.m_mva", "PCAResult"),
    "NMFResult": ("spectroview.model.analysis.m_mva", "NMFResult"),
    "InstallationManager": ("spectroview.model.system.m_installation", "InstallationManager"),
    "InstallType": ("spectroview.model.system.m_installation", "InstallType"),
    "UpdateManager": ("spectroview.model.system.m_update_manager", "UpdateManager"),
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
