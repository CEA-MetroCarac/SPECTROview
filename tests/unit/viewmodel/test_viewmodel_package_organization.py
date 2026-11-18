"""Tests for ViewModel package organization, lazy facade exports, and MVVM boundaries."""

import ast
from pathlib import Path

import pytest


class TestViewModelPackageOrganization:
    """Verify subpackage layout, PEP 562 lazy facade, and backward compatibility."""

    def test_lazy_facade_exports_classes_and_subpackages(self):
        import spectroview.viewmodel as vm_pkg

        for subpkg in ("analysis", "common", "graphs", "maps", "spectra"):
            assert hasattr(vm_pkg, subpkg)

        for cls_name in (
            "VMSettings",
            "VMWorkspaceSpectra",
            "VMFitModelBuilder",
            "VMWorkspaceMaps",
            "VMLiveMaps",
            "VMWorkspaceGraphs",
            "VMMVA",
        ):
            assert getattr(vm_pkg, cls_name) is not None

        with pytest.raises(AttributeError):
            _ = vm_pkg.NonExistentSymbol

    def test_utils_backward_compatible_facade(self):
        from spectroview.view.components.v_metadata_tooltip import (
            MetadataTooltipManager,
            MetadataTooltipPopup,
        )
        from spectroview.viewmodel import utils

        assert utils.MetadataTooltipManager is MetadataTooltipManager
        assert utils.MetadataTooltipPopup is MetadataTooltipPopup
        assert callable(utils.closest_index)
        assert callable(utils.parse_coords_from_fname)
        assert callable(utils.build_clean_fit_model)
        assert callable(utils.save_df_to_excel)

    def test_viewmodel_has_no_top_level_view_imports_or_widget_classes(self):
        vm_root = Path(__file__).resolve().parents[3] / "spectroview" / "viewmodel"
        for py_file in vm_root.rglob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert alias.name != "spectroview.view" and not alias.name.startswith("spectroview.view."), (
                            f"{py_file} has top-level import of {alias.name}"
                        )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    assert node.module != "spectroview.view" and not node.module.startswith("spectroview.view."), (
                        f"{py_file} has top-level import from {node.module}"
                    )
                elif isinstance(node, ast.ClassDef):
                    base_names = {
                        b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
                        for b in node.bases
                    }
                    assert not base_names.intersection({"QWidget", "QFrame", "QDialog"}), (
                        f"{py_file} defines UI widget class {node.name}"
                    )
