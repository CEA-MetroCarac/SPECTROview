"""Structural and lazy-facade tests for the reorganized View package."""

from __future__ import annotations

import ast
from pathlib import Path
import pytest


_VIEW_ROOT = Path(__file__).resolve().parents[3] / "spectroview" / "view"
_COMPONENTS_ROOT = _VIEW_ROOT / "components"
_WORKSPACES_ROOT = _VIEW_ROOT / "workspaces"


class TestViewPackageOrganization:
    """Verify the View package hierarchy, PEP 562 lazy facades, and import boundaries."""

    def test_top_level_view_contains_only_expected_subpackages(self):
        py_files = sorted(p.name for p in _VIEW_ROOT.glob("*.py"))
        assert py_files == ["__init__.py"]

        subpackages = sorted(
            p.name for p in _VIEW_ROOT.iterdir() if p.is_dir() and (p / "__init__.py").is_file()
        )
        assert subpackages == ["components", "theme", "workspaces"]

    def test_components_package_contains_only_subpackages_and_settings_shim(self):
        py_files = sorted(p.name for p in _COMPONENTS_ROOT.glob("*.py"))
        assert py_files == ["__init__.py", "v_settings.py"]

        subpackages = sorted(
            p.name
            for p in _COMPONENTS_ROOT.iterdir()
            if p.is_dir() and (p / "__init__.py").is_file()
        )
        assert subpackages == [
            "common",
            "customize_graph",
            "graphs",
            "lists",
            "lower_tabs",
            "map_viewer",
            "rgb_live_maps",
            "spectra_viewer",
        ]

    def test_workspaces_package_structure(self):
        py_files = sorted(p.name for p in _WORKSPACES_ROOT.glob("*.py"))
        assert py_files == [
            "__init__.py",
            "v_workspace_maps.py",
            "v_workspace_spectra.py",
        ]
        graphs_dir = _WORKSPACES_ROOT / "graphs"
        assert (graphs_dir / "__init__.py").is_file()

    def test_view_lazy_facade_resolves_subpackages_and_exports(self):
        import spectroview.view as view

        assert hasattr(view, "components")
        assert hasattr(view, "theme")
        assert hasattr(view, "workspaces")
        assert "VWorkspaceSpectra" in view.__all__
        assert "VWorkspaceMaps" in view.__all__
        assert "VWorkspaceGraphs" in view.__all__
        assert "MdiSubWindow" in view.__all__
        assert view.VWorkspaceSpectra.__name__ == "VWorkspaceSpectra"
        assert view.VWorkspaceMaps.__name__ == "VWorkspaceMaps"
        assert view.VWorkspaceGraphs.__name__ == "VWorkspaceGraphs"
        assert view.MdiSubWindow.__name__ == "MdiSubWindow"
        assert view.ThemeManager.__name__ == "ThemeManager"

        with pytest.raises(AttributeError):
            _ = view.NonExistentViewSymbol

    def test_components_lazy_facade_resolves_subpackages_and_settings_shim(self):
        import spectroview.view.components as components

        for subpkg in (
            "common",
            "customize_graph",
            "graphs",
            "lists",
            "lower_tabs",
            "map_viewer",
            "rgb_live_maps",
            "spectra_viewer",
        ):
            assert hasattr(components, subpkg)

        assert components.VSettingsDialog is components.SettingsDialog

        with pytest.raises(AttributeError):
            _ = components.NonExistentComponent

    def test_workspaces_lazy_facade_resolves_all_exports(self):
        import spectroview.view.workspaces as workspaces

        assert hasattr(workspaces, "graphs")
        assert workspaces.VWorkspaceSpectra.__name__ == "VWorkspaceSpectra"
        assert workspaces.VWorkspaceMaps.__name__ == "VWorkspaceMaps"
        assert workspaces.VWorkspaceGraphs.__name__ == "VWorkspaceGraphs"
        assert workspaces.MdiSubWindow.__name__ == "MdiSubWindow"

        with pytest.raises(AttributeError):
            _ = workspaces.NonExistentWorkspace

    def test_no_top_level_scipy_imports_in_view(self):
        offenders: list[str] = []
        for path in sorted(_VIEW_ROOT.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "scipy" or alias.name.startswith("scipy."):
                            offenders.append(f"{path.relative_to(_VIEW_ROOT)}:{node.lineno}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if node.module == "scipy" or node.module.startswith("scipy."):
                        offenders.append(f"{path.relative_to(_VIEW_ROOT)}:{node.lineno}")
        assert offenders == [], f"Unexpected top-level scipy imports in view/: {offenders}"
