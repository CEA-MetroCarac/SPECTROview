"""Unit tests for macOS matplotlib font cache discovery patch and smoke tests."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from spectroview.model.m_macos_font_patch import (
    _is_system_profiler_font_query,
    apply_macos_font_cache_fix,
    is_macos_font_cache_fix_applied,
)


def test_is_system_profiler_font_query():
    """Verify font query detector recognizes various argument formats and rejects others."""
    # Matched list formats
    assert _is_system_profiler_font_query(["system_profiler", "-xml", "SPFontsDataType"])
    assert _is_system_profiler_font_query(["/usr/sbin/system_profiler", "-xml", "SPFontsDataType"])
    assert _is_system_profiler_font_query(["system_profiler", "spfontsdatatype"])

    # Matched string formats
    assert _is_system_profiler_font_query("system_profiler -xml SPFontsDataType")
    assert _is_system_profiler_font_query("/usr/sbin/system_profiler SPFontsDataType")

    # Unrelated queries must NOT be intercepted
    assert not _is_system_profiler_font_query(["system_profiler", "SPSoftwareDataType"])
    assert not _is_system_profiler_font_query(["system_profiler", "-xml", "SPHardwareDataType"])
    assert not _is_system_profiler_font_query(["git", "status"])
    assert not _is_system_profiler_font_query("echo SPFontsDataType")
    assert not _is_system_profiler_font_query([])
    assert not _is_system_profiler_font_query("")


def test_apply_macos_font_cache_fix_is_idempotent():
    """Applying the patch multiple times must succeed without error."""
    res1 = apply_macos_font_cache_fix()
    res2 = apply_macos_font_cache_fix()
    if sys.platform == "darwin":
        assert res1 is True
        assert res2 is True
        assert is_macos_font_cache_fix_applied() is True


def test_apply_macos_font_cache_fix_non_darwin(monkeypatch):
    """On non-macOS platforms, the patch should be a clean no-op returning False."""
    monkeypatch.setattr(sys, "platform", "linux")
    # Fresh check without pre-existing patch flag
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: b"linux",
    )
    assert apply_macos_font_cache_fix() is False


def test_subprocess_check_output_transparent_for_normal_commands():
    """Subprocess check_output must forward normal system calls transparently."""
    apply_macos_font_cache_fix(force=True)
    out = subprocess.check_output([sys.executable, "-c", "print('spectroview_ok')"])
    assert b"spectroview_ok" in out


def test_subprocess_check_output_intercepts_system_profiler_fonts():
    """Calling system_profiler SPFontsDataType must raise CalledProcessError."""
    apply_macos_font_cache_fix(force=True)
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_output(["system_profiler", "-xml", "SPFontsDataType"])


def test_matplotlib_rebuilds_font_cache_without_keyerror(tmp_path):
    """Rebuilding font cache under clean MPLCONFIGDIR must never raise KeyError: '_items'."""
    apply_macos_font_cache_fix()
    code = (
        "import os; "
        "import sys; "
        "from spectroview.model.m_macos_font_patch import apply_macos_font_cache_fix; "
        "apply_macos_font_cache_fix(); "
        "import matplotlib.font_manager as fm; "
        "cache = fm._load_fontmanager(try_read_cache=False); "
        "assert len(cache.ttflist) > 0, 'No fonts found'; "
        "print('FONTS_LOADED_OK')"
    )
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "clean_mpl")
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    assert "FONTS_LOADED_OK" in res.stdout


def test_main_version_cli_flag():
    """Invoking spectroview.main with --version must print version and exit 0."""
    res = subprocess.run(
        [sys.executable, "-m", "spectroview.main", "--version"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "SPECTROview" in res.stdout


def test_main_smoke_test_cli_flag():
    """Invoking spectroview.main with --smoke-test offscreen must exit cleanly with code 0."""
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    res = subprocess.run(
        [sys.executable, "-m", "spectroview.main", "--smoke-test"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
