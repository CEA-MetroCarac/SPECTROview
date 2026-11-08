"""Unit tests for macOS .app / .dmg packaging helpers (packaging/macos/build_dmg.py)."""

from __future__ import annotations

import importlib.util
import plistlib
from pathlib import Path

import pytest


def _load_build_dmg():
    script_path = Path(__file__).resolve().parents[2] / "packaging" / "macos" / "build_dmg.py"
    spec = importlib.util.spec_from_file_location("spectroview_build_dmg", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bm = _load_build_dmg()


def test_entitlements_plist_is_valid_xml_with_hardened_runtime_keys():
    plist_path = Path(__file__).resolve().parents[2] / "packaging" / "macos" / "entitlements.plist"
    with plist_path.open("rb") as fh:
        data = plistlib.load(fh)
    assert data["com.apple.security.cs.allow-jit"] is True
    assert data["com.apple.security.cs.allow-unsigned-executable-memory"] is True
    assert data["com.apple.security.cs.disable-library-validation"] is True


def test_arch_normalization_and_dmg_filename():
    assert bm.normalize_arch("arm64") == "arm64"
    assert bm.normalize_arch("aarch64") == "arm64"
    assert bm.normalize_arch("apple_silicon") == "arm64"

    # Intel macOS architectures are rejected with clear explanation
    for intel_arch in ("x86_64", "x64", "intel", "amd64"):
        with pytest.raises(ValueError, match="Intel macOS builds are no longer supported"):
            bm.normalize_arch(intel_arch)

    with pytest.raises(ValueError, match="Unsupported macOS architecture"):
        bm.normalize_arch("sparc64")

    assert bm.dmg_filename("26.47.0", "arm64") == "SPECTROview_26.47.0_macOS_arm64.dmg"
    with pytest.raises(ValueError, match="Intel macOS builds are no longer supported"):
        bm.dmg_filename("26.47.0", "x86_64")


def test_verify_app_bundle_and_stage_dmg_contents(tmp_path):
    app_dir = tmp_path / "SPECTROview.app"
    contents = app_dir / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    (contents / "Frameworks" / "spectroview" / "resources").mkdir(parents=True)
    (contents / "Frameworks" / "spectroview" / "ai_agent" / "config").mkdir(parents=True)
    (contents / "Frameworks" / "spectroview" / "ai_agent" / "prompts").mkdir(parents=True)
    (contents / "Frameworks" / "certifi").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="missing required files"):
        bm.verify_app_bundle(app_dir)

    (contents / "Info.plist").write_text("<plist/>", encoding="utf-8")
    (contents / "MacOS" / "SPECTROview").write_bytes(b"mach-o")
    (contents / "Frameworks" / "spectroview" / "resources" / "build_info.json").write_text("{}", encoding="utf-8")
    (contents / "Frameworks" / "spectroview" / "ai_agent" / "config" / "model.yaml").write_text("m: 1", encoding="utf-8")
    (contents / "Frameworks" / "spectroview" / "ai_agent" / "prompts" / "system.md").write_text("# hi", encoding="utf-8")
    (contents / "Frameworks" / "certifi" / "cacert.pem").write_text("CERT", encoding="utf-8")

    bm.verify_app_bundle(app_dir)

    staging = tmp_path / "staging"
    bm.stage_dmg_contents(app_dir, staging, version="26.47.0", arch="arm64", signed=False)
    assert (staging / "SPECTROview.app" / "Contents" / "Info.plist").is_file()
    note = staging / "First Launch (Unsigned Build).txt"
    assert note.is_file()
    assert "Control-click" in note.read_text(encoding="utf-8")


def test_build_codesign_commands_unsigned_vs_developer_id(tmp_path):
    app_dir = tmp_path / "SPECTROview.app"
    adhoc_cmds = bm.build_codesign_commands(app_dir, signing_identity=None)
    assert len(adhoc_cmds) == 1
    assert "-" in adhoc_cmds[0]

    signed_cmds = bm.build_codesign_commands(app_dir, signing_identity="Developer ID Application: CEA")
    assert len(signed_cmds) == 2
    assert "runtime" in signed_cmds[0]
    assert "Developer ID Application: CEA" in signed_cmds[0]
    assert "--verify" in signed_cmds[1]
