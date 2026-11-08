"""Unit tests for Linux AppImage and .deb packaging helpers (packaging/linux/build_linux.py)."""

from __future__ import annotations

import configparser
import importlib.util
from pathlib import Path

import pytest


def _load_build_linux():
    script_path = Path(__file__).resolve().parents[2] / "packaging" / "linux" / "build_linux.py"
    spec = importlib.util.spec_from_file_location("spectroview_build_linux", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bl = _load_build_linux()


def test_linux_desktop_entry_valid_ini():
    desktop_path = Path(__file__).resolve().parents[2] / "packaging" / "linux" / "spectroview.desktop"
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(desktop_path, encoding="utf-8")
    assert "Desktop Entry" in parser
    entry = parser["Desktop Entry"]
    assert entry["Type"] == "Application"
    assert entry["Name"] == "SPECTROview"
    assert entry["Icon"] == "spectroview"
    assert "Science" in entry["Categories"]


def test_filenames_and_debian_version_normalization():
    assert bl.appimage_filename("26.47.0") == "SPECTROview_26.47.0_Linux_x86_64.AppImage"
    assert bl.deb_filename("26.47.0") == "SPECTROview_26.47.0_Linux_amd64.deb"
    assert bl.debian_package_version("26.47.0") == "26.47.0"
    assert bl.debian_package_version("26.47.0-rc.1") == "26.47.0~rc.1"


def test_verify_linux_bundle_and_stage_appdir_and_deb(tmp_path):
    dist_dir = tmp_path / "dist" / "SPECTROview"
    (dist_dir / "_internal" / "spectroview" / "resources").mkdir(parents=True)
    (dist_dir / "_internal" / "spectroview" / "ai_agent" / "config").mkdir(parents=True)
    (dist_dir / "_internal" / "spectroview" / "ai_agent" / "prompts").mkdir(parents=True)
    (dist_dir / "_internal" / "certifi").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="missing required assets"):
        bl.verify_linux_bundle(dist_dir)

    (dist_dir / "SPECTROview").write_bytes(b"\x7fELF")
    (dist_dir / "_internal" / "spectroview" / "resources" / "build_info.json").write_text("{}", encoding="utf-8")
    (dist_dir / "_internal" / "spectroview" / "ai_agent" / "config" / "model.yaml").write_text("m: 1", encoding="utf-8")
    (dist_dir / "_internal" / "spectroview" / "ai_agent" / "prompts" / "system.md").write_text("# hi", encoding="utf-8")
    (dist_dir / "_internal" / "certifi" / "cacert.pem").write_text("CERT", encoding="utf-8")

    bl.verify_linux_bundle(dist_dir)

    appdir = tmp_path / "AppDir"
    bl.stage_appdir(dist_dir, appdir)
    assert (appdir / "AppRun").is_file()
    assert (appdir / "spectroview.desktop").is_file()
    assert (appdir / "spectroview.png").is_file()
    assert (appdir / ".DirIcon").is_file()
    assert (appdir / "usr" / "bin" / "SPECTROview").is_file()

    deb_root = tmp_path / "deb_root"
    bl.stage_deb_root(dist_dir, deb_root, version="26.47.0")
    assert (deb_root / "opt" / "spectroview" / "SPECTROview").is_file()
    assert (deb_root / "opt" / "spectroview" / ".spectroview-deb").is_file()
    assert (deb_root / "usr" / "share" / "applications" / "spectroview.desktop").is_file()
    assert (deb_root / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps" / "spectroview.png").is_file()

    control = (deb_root / "DEBIAN" / "control").read_text(encoding="utf-8")
    assert "Package: spectroview" in control
    assert "Version: 26.47.0" in control
    assert "Architecture: amd64" in control
    assert "libxcb-cursor0" in control
    assert "libegl1" in control

    postrm = (deb_root / "DEBIAN" / "postrm").read_text(encoding="utf-8")
    assert ".config" in postrm and "rm -rf" not in postrm
