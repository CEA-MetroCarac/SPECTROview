"""Unit tests for centralized multi-platform InstallationManager classification."""

from __future__ import annotations

import os
from pathlib import Path

from spectroview.model.m_installation import InstallType, InstallationManager


class TestUnfrozenDetection:
    def test_detects_dev_source_checkout_without_false_positive_from_env(self, tmp_path):
        # Even if APPIMAGE leaks into an unfrozen developer shell, unfrozen checkouts
        # must remain DEV_SOURCE rather than claiming a production AppImage.
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        pkg = repo / "spectroview"
        pkg.mkdir()

        info = InstallationManager.detect(
            frozen=False,
            executable=sys_exe(tmp_path),
            platform_name="linux",
            package_root=pkg,
            env={"APPIMAGE": "/opt/fake/SPECTROview.AppImage", "APPDIR": "/tmp/.mount_fake"},
        )

        assert info.install_type == InstallType.DEV_SOURCE
        assert info.is_dev is True
        assert info.is_frozen is False
        assert info.is_portable is False
        assert info.update_supported is False
        assert info.auto_update_supported is False

    def test_detects_pip_wheel_in_site_packages(self, tmp_path):
        site_pkg = tmp_path / "venv" / "Lib" / "site-packages" / "spectroview"
        site_pkg.mkdir(parents=True)

        info = InstallationManager.detect(
            frozen=False,
            executable=tmp_path / "venv" / "Scripts" / "python.exe",
            platform_name="win32",
            package_root=site_pkg,
            check_dist_metadata=False,
        )

        assert info.install_type == InstallType.PIP_WHEEL
        assert info.is_dev is False
        assert info.is_frozen is False
        assert info.is_portable is False
        assert info.update_supported is True
        assert info.auto_update_supported is True
        assert info.update_backend == "pip_auto"


class TestWindowsFrozenDetection:
    def test_detects_windows_velopack_layout(self, tmp_path):
        root = tmp_path / "AppData" / "Local" / "SPECTROview"
        current = root / "current"
        current.mkdir(parents=True)
        (root / "Update.exe").write_text("", encoding="utf-8")
        (current / "sq.version").write_text("<package/>", encoding="utf-8")
        exe = current / "SPECTROview.exe"
        exe.write_text("", encoding="utf-8")

        info = InstallationManager.detect(
            frozen=True,
            executable=exe,
            platform_name="win32",
            machine="AMD64",
        )

        assert info.install_type == InstallType.WINDOWS_VELOPACK
        assert info.is_frozen is True
        assert info.is_portable is False
        assert info.update_supported is True
        assert info.auto_update_supported is True
        assert info.architecture == "x64"
        assert InstallationManager.has_velopack(
            frozen=True, executable=exe, platform_name="win32"
        ) is True
        assert InstallationManager.should_run_velopack_hook(
            argv=[], frozen=True, executable=exe, platform_name="win32"
        ) is True

    def test_detects_windows_portable_zip_layout(self, tmp_path):
        portable_dir = tmp_path / "Downloads" / "SPECTROview Portable (1) α"
        portable_dir.mkdir(parents=True)
        exe = portable_dir / "SPECTROview.exe"
        exe.write_text("", encoding="utf-8")

        info = InstallationManager.detect(
            frozen=True,
            executable=exe,
            platform_name="win32",
            machine="AMD64",
        )

        assert info.install_type == InstallType.WINDOWS_PORTABLE
        assert info.is_frozen is True
        assert info.is_portable is True
        assert info.update_supported is True
        assert info.auto_update_supported is False
        assert InstallationManager.should_run_velopack_hook(
            argv=[], frozen=True, executable=exe, platform_name="win32"
        ) is False

    def test_veloapp_hook_arg_triggers_hook_even_before_full_install(self, tmp_path):
        exe = tmp_path / "SPECTROview.exe"
        assert (
            InstallationManager.should_run_velopack_hook(
                argv=["--veloapp-install", "26.46.1"],
                frozen=True,
                executable=exe,
                platform_name="win32",
            )
            is True
        )


class TestMacOSAndLinuxFrozenDetection:
    def test_detects_macos_app_bundle(self):
        info = InstallationManager.detect(
            frozen=True,
            executable="/Applications/SPECTROview.app/Contents/MacOS/SPECTROview",
            platform_name="darwin",
            machine="arm64",
        )

        assert info.install_type == InstallType.MACOS_BUNDLE
        assert info.app_root == "/Applications/SPECTROview.app"
        assert info.architecture == "arm64"
        assert info.is_portable is False
        assert info.update_supported is True
        assert info.auto_update_supported is False

    def test_detects_linux_appimage_writable_vs_readonly(self, tmp_path, monkeypatch):
        appimage_file = tmp_path / "SPECTROview_26.46.1_Linux_x86_64.AppImage"
        appimage_file.write_bytes(b"\x7fELF")

        monkeypatch.setattr(os, "access", lambda p, m: True)
        info_rw = InstallationManager.detect(
            frozen=True,
            executable="/tmp/.mount_SV1234/usr/bin/SPECTROview",
            platform_name="linux",
            machine="x86_64",
            env={"APPIMAGE": str(appimage_file), "APPDIR": "/tmp/.mount_SV1234"},
        )
        assert info_rw.install_type == InstallType.LINUX_APPIMAGE
        assert info_rw.is_portable is True
        assert info_rw.update_supported is True
        assert info_rw.auto_update_supported is True

        monkeypatch.setattr(os, "access", lambda p, m: False)
        info_ro = InstallationManager.detect(
            frozen=True,
            executable="/tmp/.mount_SV1234/usr/bin/SPECTROview",
            platform_name="linux",
            machine="x86_64",
            env={"APPIMAGE": str(appimage_file), "APPDIR": "/tmp/.mount_SV1234"},
        )
        assert info_ro.install_type == InstallType.LINUX_APPIMAGE
        assert info_ro.auto_update_supported is False

    def test_detects_linux_deb_prefix_and_marker(self, tmp_path):
        info_prefix = InstallationManager.detect(
            frozen=True,
            executable="/opt/spectroview/SPECTROview",
            platform_name="linux",
            machine="x86_64",
            env={},
        )
        assert info_prefix.install_type == InstallType.LINUX_DEB
        assert info_prefix.is_portable is False
        assert info_prefix.update_supported is True
        assert info_prefix.auto_update_supported is False

        custom_deb_dir = tmp_path / "deb_root"
        custom_deb_dir.mkdir()
        (custom_deb_dir / ".spectroview-deb").write_text("deb\n", encoding="utf-8")
        exe = custom_deb_dir / "SPECTROview"
        info_marker = InstallationManager.detect(
            frozen=True,
            executable=exe,
            platform_name="linux",
            machine="x86_64",
            env={},
        )
        assert info_marker.install_type == InstallType.LINUX_DEB

    def test_unrecognized_frozen_binary_is_standalone_unknown(self, tmp_path):
        info = InstallationManager.detect(
            frozen=True,
            executable=tmp_path / "custom_dir" / "SPECTROview",
            platform_name="linux",
            env={},
        )
        assert info.install_type == InstallType.STANDALONE_UNKNOWN
        assert info.update_supported is False


def sys_exe(base: Path) -> Path:
    return base / "python"

