# tests/unit/model/test_build_info.py
"""Unit tests for spectroview.settings.build_info and dual-mode runtime resolution."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys

import pytest

from spectroview import BUILD_NUMBER
from spectroview.settings.build_info import (
    TRACKED_PACKAGES,
    generate_build_snapshot,
    get_build_info_path,
    get_build_number,
    get_environment_summary,
    get_runtime_package_versions,
    is_frozen_app,
    load_build_snapshot,
    query_package_version,
)
from spectroview.settings.ui.tab_about import AboutTabPage


class TestBuildInfoSnapshot:
    def test_generate_and_load_build_snapshot(self, tmp_path: Path):
        target = tmp_path / "custom_build_info.json"
        data = generate_build_snapshot(target)

        assert target.is_file()
        assert data["build_mode"] == "standalone_exe"
        assert data["python_version"] == platform.python_version()
        assert data["python_architecture"] == platform.architecture()[0]
        assert data["build_os"] == f"{platform.system()} {platform.release()}".strip()
        assert isinstance(data["packages"], dict)

        for pkg in TRACKED_PACKAGES:
            assert pkg in data["packages"]

        loaded = load_build_snapshot(target)
        assert loaded is not None
        assert loaded["build_mode"] == "standalone_exe"
        assert loaded["python_version"] == data["python_version"]
        assert loaded["packages"]["numpy"] == data["packages"]["numpy"]

    def test_load_build_snapshot_missing_or_corrupt(self, tmp_path: Path):
        missing = tmp_path / "nonexistent.json"
        assert load_build_snapshot(missing) is None

        corrupted = tmp_path / "corrupt.json"
        corrupted.write_text("{bad json syntax", encoding="utf-8")
        assert load_build_snapshot(corrupted) is None

    def test_query_package_version_known_and_unknown(self):
        # numpy should be installed in the active test environment
        np_ver = query_package_version("numpy")
        assert np_ver != "Not installed"

        # fictional package
        assert query_package_version("totally_fake_package_xyz_99") == "Not installed"


class TestExecutionModes:
    def test_is_frozen_app(self, monkeypatch):
        monkeypatch.delattr(sys, "frozen", raising=False)
        assert is_frozen_app() is False

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        assert is_frozen_app() is True

    def test_get_build_number_unfrozen_vs_frozen(self, monkeypatch, tmp_path: Path):
        # 1. Unfrozen mode returns default BUILD_NUMBER ("Dev")
        monkeypatch.delattr(sys, "frozen", raising=False)
        assert get_build_number() == BUILD_NUMBER

        # 2. Frozen mode returns snapshot timestamp if present
        mock_snapshot = {
            "build_number": "2026-10-29 12-00-00",
            "packages": {},
        }
        snap_file = tmp_path / "build_info.json"
        snap_file.write_text(json.dumps(mock_snapshot), encoding="utf-8")

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(
            "spectroview.settings.build_info.get_build_info_path",
            lambda: snap_file,
        )
        assert get_build_number() == "2026-10-29 12-00-00"

    def test_runtime_package_versions_frozen_mode_reads_snapshot(self, monkeypatch, tmp_path: Path):
        mock_snapshot = {
            "build_mode": "standalone_exe",
            "build_os": "Windows 11",
            "packages": {
                "numpy": "2.99.0",
                "scipy": "1.99.0",
                "PySide6": "6.99.0",
                "pandas": "3.99.0",
            },
        }
        snap_file = tmp_path / "build_info.json"
        snap_file.write_text(json.dumps(mock_snapshot), encoding="utf-8")

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(
            "spectroview.settings.build_info.get_build_info_path",
            lambda: snap_file,
        )

        get_runtime_package_versions.cache_clear()
        try:
            versions = get_runtime_package_versions()
            assert versions["numpy"] == "2.99.0"
            assert versions["scipy"] == "1.99.0"
            assert "6.99.0" in versions["PySide6"]
            assert versions["pandas"] == "3.99.0"
            # Packages not in mock snapshot fallback to "Not installed"
            assert versions["renishawWiRE"] == "Not installed"
        finally:
            get_runtime_package_versions.cache_clear()

    def test_environment_summary_structure(self, monkeypatch):
        monkeypatch.delattr(sys, "frozen", raising=False)
        get_runtime_package_versions.cache_clear()
        summary = get_environment_summary()

        assert summary["is_frozen"] is False
        assert "Python Package" in summary["mode"]
        assert summary["python_version"] == platform.python_version()
        assert "packages" in summary
        assert "numpy" in summary["packages"]


class TestAboutTabPageDualModeUI:
    def test_about_tab_unfrozen_ui(self, qapp, monkeypatch):
        monkeypatch.delattr(sys, "frozen", raising=False)
        get_runtime_package_versions.cache_clear()

        page = AboutTabPage()
        from PySide6.QtWidgets import QLabel
        labels = [lbl.text() for lbl in page.findChildren(QLabel)]
        text = " ".join(labels)

        assert "Python Package (.whl / local)" in text
        assert f"Python:</b> {platform.python_version()}" in text
        assert page.grp_env.toolTip() == "Runtime dependencies detected in your local Python environment."

    def test_about_tab_frozen_ui(self, qapp, monkeypatch, tmp_path: Path):
        mock_snapshot = {
            "build_mode": "standalone_exe",
            "build_os": "Windows 11 CustomBuild",
            "python_version": "3.14.99",
            "python_architecture": "64bit",
            "build_number": "2026-10-29 10-00-00",
            "packages": {
                "numpy": "2.5.4",
                "scipy": "1.17.1",
            },
        }
        snap_file = tmp_path / "build_info.json"
        snap_file.write_text(json.dumps(mock_snapshot), encoding="utf-8")

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(
            "spectroview.settings.build_info.get_build_info_path",
            lambda: snap_file,
        )
        get_runtime_package_versions.cache_clear()

        try:
            page = AboutTabPage()
            from PySide6.QtWidgets import QLabel
            labels = [lbl.text() for lbl in page.findChildren(QLabel)]
            text = " ".join(labels)

            assert "Standalone Executable (.exe)" in text
            assert "Built on:</b> Windows 11 CustomBuild" in text
            assert "Python:</b> 3.14.99" in text
            assert "Build: 2026-10-29 10-00-00" in text
            assert page.grp_env.toolTip() == "Dependencies and build environment bundled into this standalone executable."
        finally:
            get_runtime_package_versions.cache_clear()

