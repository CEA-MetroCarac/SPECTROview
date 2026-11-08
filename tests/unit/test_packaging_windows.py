"""Unit tests for Windows packaging helpers (packaging/windows/build_windows.py)."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest


def _load_build_windows():
    script_path = Path(__file__).resolve().parents[2] / "packaging" / "windows" / "build_windows.py"
    spec = importlib.util.spec_from_file_location("spectroview_build_windows", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bw = _load_build_windows()


def test_read_version_matches_package():
    from spectroview import VERSION

    assert bw.read_version() == VERSION


def test_write_win_version_info_contains_version_and_company(tmp_path):
    target = tmp_path / "win_version_info.txt"
    bw.write_win_version_info("26.47.0", target)
    text = target.read_text(encoding="utf-8")
    assert "(26, 47, 0, 0)" in text
    assert "CEA-MetroCarac" in text
    assert "SPECTROview.exe" in text


def test_verify_onedir_bundle_and_create_portable_zip(tmp_path):
    dist_dir = tmp_path / "dist" / "SPECTROview"
    (dist_dir / "_internal" / "spectroview" / "resources").mkdir(parents=True)
    (dist_dir / "_internal" / "spectroview" / "ai_agent" / "config").mkdir(parents=True)
    (dist_dir / "_internal" / "spectroview" / "ai_agent" / "prompts").mkdir(parents=True)
    (dist_dir / "_internal" / "certifi").mkdir(parents=True)

    # Missing exe fails cleanly
    with pytest.raises(RuntimeError, match="missing required assets"):
        bw.verify_onedir_bundle(dist_dir)

    (dist_dir / "SPECTROview.exe").write_bytes(b"MZ")
    (dist_dir / "_internal" / "spectroview" / "resources" / "build_info.json").write_text("{}", encoding="utf-8")
    (dist_dir / "_internal" / "spectroview" / "ai_agent" / "config" / "model.yaml").write_text("k: v", encoding="utf-8")
    (dist_dir / "_internal" / "spectroview" / "ai_agent" / "prompts" / "system.md").write_text("# hi", encoding="utf-8")
    (dist_dir / "_internal" / "certifi" / "cacert.pem").write_text("CERT", encoding="utf-8")

    bw.verify_onedir_bundle(dist_dir)

    zip_out = tmp_path / "out" / "SPECTROview_26.47.0_Windows_x64_Portable.zip"
    bw.create_portable_zip(dist_dir, zip_out)
    assert zip_out.is_file()
    with zipfile.ZipFile(zip_out, "r") as zf:
        names = set(zf.namelist())
    assert "SPECTROview/SPECTROview.exe" in names
    assert "SPECTROview/_internal/certifi/cacert.pem" in names
    assert not any(n.endswith("Update.exe") for n in names)
