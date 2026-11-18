"""Workspace view packages with PEP 562 lazy resolution.

Workspaces:
    - ``VWorkspaceSpectra``: Discrete 1D spectra workspace view.
    - ``VWorkspaceMaps``: Hyperspectral 2D/wafer maps workspace view.
    - ``VWorkspaceGraphs``: Publication plotting workspace view (``graphs`` subpackage).
    - ``MdiSubWindow``: Draggable/resizable MDI plot subwindow for Graphs workspace.
"""

from __future__ import annotations

import importlib

_SUBPACKAGES = frozenset({"graphs"})

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "VWorkspaceSpectra": ("spectroview.view.workspaces.v_workspace_spectra", "VWorkspaceSpectra"),
    "VWorkspaceMaps": ("spectroview.view.workspaces.v_workspace_maps", "VWorkspaceMaps"),
    "VWorkspaceGraphs": ("spectroview.view.workspaces.graphs.v_workspace_graphs", "VWorkspaceGraphs"),
    "MdiSubWindow": ("spectroview.view.workspaces.graphs.v_mdi_subwindow", "MdiSubWindow"),
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
