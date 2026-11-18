"""ViewModel layer package facade with lazy subpackage and symbol resolution.

Subpackages:
    - ``spectra``: ``VMWorkspaceSpectra``, preprocessing/fitting/results/persistence mixins, and ``VMFitModelBuilder``.
    - ``maps``: ``VMWorkspaceMaps``, maps fitting/persistence mixins, ``VMLiveMaps``, and ``_LiveMapWorker``.
    - ``graphs``: ``VMWorkspaceGraphs`` and ``GraphUndoMixin``.
    - ``analysis``: ``VMMVA`` (PCA/NMF orchestration).
    - ``common``: Shared spectral math, export/clipboard, and UI notification helpers.
"""

from __future__ import annotations

import importlib

_SUBPACKAGES = frozenset({"analysis", "common", "graphs", "maps", "spectra"})

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "VMSettings": ("spectroview.viewmodel.vm_settings", "VMSettings"),
    "VMWorkspaceSpectra": (
        "spectroview.viewmodel.spectra.vm_workspace_spectra",
        "VMWorkspaceSpectra",
    ),
    "VMFitModelBuilder": (
        "spectroview.viewmodel.spectra.vm_fit_model_builder",
        "VMFitModelBuilder",
    ),
    "VMWorkspaceMaps": (
        "spectroview.viewmodel.maps.vm_workspace_maps",
        "VMWorkspaceMaps",
    ),
    "VMLiveMaps": ("spectroview.viewmodel.maps.vm_live_maps", "VMLiveMaps"),
    "VMWorkspaceGraphs": (
        "spectroview.viewmodel.graphs.vm_workspace_graphs",
        "VMWorkspaceGraphs",
    ),
    "VMMVA": ("spectroview.viewmodel.analysis.vm_mva", "VMMVA"),
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
