"""Unit tests for Main's centralized UpdateManager banner, dialog, and bootstrap flows."""

from __future__ import annotations

import os
import subprocess
import sys
import types
from typing import TYPE_CHECKING
from PySide6.QtWidgets import QMessageBox, QVBoxLayout, QWidget

from spectroview.model.m_installation import InstallationManager
from spectroview.model.m_update_manager import (
    InstallResult,
    ReleaseAsset,
    ReleaseInfo,
    UpdateCheckResult,
    UpdateStatus,
)
from spectroview.settings.m_settings import MSettings

if TYPE_CHECKING:
    from spectroview.main import Main


def _make_stub_main() -> Main:
    from spectroview.main import Main
    from PySide6.QtWidgets import QMainWindow

    win = Main.__new__(Main)
    QMainWindow.__init__(win)
    win.settings = MSettings()
    win._update_banner = None
    win._update_download_worker = None
    win._update_manager = None
    win._pending_release = None
    win._manual_check_worker = None
    win._startup_check_worker = None
    win._manual_check_found_update = False
    win._manual_check_error = ""
    return win


def _sample_result(tag: str = "v26.47.0", available: bool = True) -> UpdateCheckResult:
    ver = tag.lstrip("v")
    asset = ReleaseAsset(
        name=f"SPECTROview_{ver}_Windows_x64_Setup.exe",
        url=f"https://example.test/SPECTROview_{ver}_Windows_x64_Setup.exe",
        sha256="a" * 64,
    )
    release = ReleaseInfo(
        tag=tag,
        version=ver,
        name=tag,
        html_url=f"https://example.test/releases/{tag}",
        notes="",
        prerelease=False,
        published_at="",
        assets=(asset,),
    )
    return UpdateCheckResult(
        status=UpdateStatus.UPDATE_AVAILABLE if available else UpdateStatus.UP_TO_DATE,
        current_version="26.46.1",
        latest_version=ver,
        update_available=available,
        release=release,
        selected_asset=asset,
        can_auto_install=False,
        action_label="Download Update",
        action_tooltip="Download verified update.",
    )


def test_startup_check_shows_banner_unless_skipped(qapp, monkeypatch):
    win = _make_stub_main()
    central = QWidget()
    QVBoxLayout(central)
    win.centralWidget = lambda: central

    shown = []
    monkeypatch.setattr(win, "_show_update_banner_from_result", lambda r: shown.append(r.release.tag))

    # Up-to-date -> nothing shown
    win._on_startup_check_succeeded(_sample_result(available=False))
    assert shown == []

    # Skipped version -> nothing shown
    win.settings.set_skipped_version("v26.47.0")
    win._on_startup_check_succeeded(_sample_result("v26.47.0", available=True))
    assert shown == []

    # Different version -> banner displayed
    win._on_startup_check_succeeded(_sample_result("v26.48.0", available=True))
    assert shown == ["v26.48.0"]


def test_manual_check_done_reports_error_or_up_to_date(qapp, monkeypatch):
    win = _make_stub_main()
    warnings = []
    infos = []
    monkeypatch.setattr(QMessageBox, "warning", lambda p, title, msg: warnings.append((title, msg)))
    monkeypatch.setattr(QMessageBox, "information", lambda p, title, msg: infos.append((title, msg)))

    win._manual_check_error = "GitHub API rate limit reached"
    win._on_manual_check_done()
    assert len(warnings) == 1
    assert "rate limit" in warnings[0][1]

    win._manual_check_error = ""
    win._manual_check_found_update = False
    win._on_manual_check_done()
    assert len(infos) == 1
    assert "latest version" in infos[0][1]

    win._manual_check_found_update = True
    win._on_manual_check_done()
    assert len(infos) == 1  # No extra modal when banner was already presented


def test_on_update_download_finished_handles_restart_and_guided_paths(qapp, monkeypatch, tmp_path):
    win = _make_stub_main()
    guided_calls = []
    restarts = []
    closed = []

    class FakeManager:
        def __init__(self, result):
            self._result = result

        def install_update(self, path, release=None):
            return self._result

        def restart_application(self):
            restarts.append(True)
            return True

    # 1) Auto restart flow (Velopack / pip wheel)
    auto_mgr = FakeManager(
        InstallResult(status=UpdateStatus.RESTART_REQUIRED, success=True, restart_required=True)
    )
    monkeypatch.setattr(win, "_get_update_manager", lambda: auto_mgr)
    monkeypatch.setattr(win, "close", lambda: closed.append(True))
    win._on_update_download_finished(str(tmp_path / "update.whl"))
    assert restarts == [True]
    assert closed == [True]

    # 2) Guided flow (Portable ZIP / macOS DMG / Linux .deb)
    win2 = _make_stub_main()
    guided_res = InstallResult(
        status=UpdateStatus.GUIDED_ACTION_REQUIRED,
        success=True,
        requires_manual_action=True,
        artifact_path=str(tmp_path / "app.dmg"),
        instructions="Drag to Applications",
    )
    monkeypatch.setattr(win2, "_get_update_manager", lambda: FakeManager(guided_res))
    monkeypatch.setattr(win2, "_show_guided_update_dialog", lambda r: guided_calls.append(r))
    win2._on_update_download_finished(str(tmp_path / "app.dmg"))
    assert guided_calls == [guided_res]


def test_run_velopack_bootstrap_only_imports_when_needed(monkeypatch):
    from spectroview.main import _run_velopack_bootstrap

    called = []
    fake_vpk = types.SimpleNamespace(
        App=lambda: types.SimpleNamespace(run=lambda: called.append("ran"))
    )
    monkeypatch.setitem(sys.modules, "velopack", fake_vpk)

    monkeypatch.setattr(InstallationManager, "should_run_velopack_hook", classmethod(lambda cls: False))
    _run_velopack_bootstrap()
    assert called == []

    monkeypatch.setattr(InstallationManager, "should_run_velopack_hook", classmethod(lambda cls: True))
    _run_velopack_bootstrap()
    assert called == ["ran"]


def test_import_main_after_headless_pyplot_does_not_fail():
    """Importing spectroview.main after pyplot under headless Linux (no DISPLAY) must not raise ImportError."""
    code = (
        "import matplotlib, matplotlib._c_internal_utils as _c; "
        "_c.display_is_valid = lambda: False; "
        "matplotlib.use('Agg'); "
        "import matplotlib.pyplot; "
        "import spectroview.main"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"stderr:\n{res.stderr}"


def test_import_main_before_headless_pyplot_does_not_fail():
    """Importing spectroview.main before pyplot under headless Linux must not leave qtagg forced without QApplication."""
    code = (
        "import matplotlib._c_internal_utils as _c; "
        "_c.display_is_valid = lambda: False; "
        "import spectroview.main; "
        "import matplotlib.pyplot as plt; "
        "fig, ax = plt.subplots(); "
        "plt.close(fig)"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"stderr:\n{res.stderr}"


def test_import_main_selects_qtagg_when_qapp_exists_under_headless():
    """Once QApplication exists under offscreen Qt, spectroview.main can configure qtagg safely."""
    code = (
        "import os; os.environ['QT_QPA_PLATFORM'] = 'offscreen'; "
        "import matplotlib._c_internal_utils as _c; "
        "_c.display_is_valid = lambda: False; "
        "from PySide6.QtWidgets import QApplication; app = QApplication([]); "
        "import spectroview.main; "
        "import matplotlib.pyplot as plt; "
        "assert plt.get_backend().lower() == 'qtagg', plt.get_backend()"
    )
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"stderr:\n{res.stderr}"


def test_is_worker_running_handles_deleted_cpp_object(qapp):
    """Main._is_worker_running must return False when a QThread has been deleted by Qt."""
    from PySide6.QtCore import QThread
    from spectroview.main import Main

    assert Main._is_worker_running(None) is False

    thread = QThread()
    assert Main._is_worker_running(thread) is False

    thread.deleteLater()
    qapp.processEvents()

    # Once deleted by Qt, calling isRunning directly would raise RuntimeError.
    # _is_worker_running must safely catch it and return False.
    assert Main._is_worker_running(thread) is False


def test_repeated_manual_update_check_does_not_crash(qapp, monkeypatch):
    """Calling _manual_update_check multiple times after worker deletion must not raise RuntimeError."""
    from spectroview.model.m_update_manager import UpdateManager

    win = _make_stub_main()
    monkeypatch.setattr(win, "_get_update_manager", lambda: UpdateManager("26.46.1"))
    monkeypatch.setattr(win, "_on_manual_check_done", lambda: None)
    monkeypatch.setattr(win, "_on_manual_check_succeeded", lambda r: None)
    monkeypatch.setattr(win, "_on_manual_check_failed", lambda e: None)

    # First check
    win._manual_update_check()
    assert win._manual_check_worker is not None
    first_worker = win._manual_check_worker
    first_worker.wait(2000)
    qapp.processEvents()

    # Second check must safely proceed without RuntimeError: Internal C++ object already deleted
    win._manual_update_check()
    assert win._manual_check_worker is not None
    win._manual_check_worker.wait(2000)
    qapp.processEvents()


def test_manual_update_check_with_deleted_cpp_worker_does_not_crash(qapp, monkeypatch):
    """If _manual_check_worker points to a destroyed C++ object, _manual_update_check cleans it up and succeeds."""
    from PySide6.QtCore import QThread
    from spectroview.model.m_update_manager import UpdateManager

    win = _make_stub_main()
    monkeypatch.setattr(win, "_get_update_manager", lambda: UpdateManager("26.46.1"))
    monkeypatch.setattr(win, "_on_manual_check_done", lambda: None)
    monkeypatch.setattr(win, "_on_manual_check_succeeded", lambda r: None)
    monkeypatch.setattr(win, "_on_manual_check_failed", lambda e: None)

    # Artificially set a worker that has been destroyed via deleteLater
    dead_worker = QThread()
    dead_worker.deleteLater()
    qapp.processEvents()
    win._manual_check_worker = dead_worker

    # Calling _manual_update_check must not raise RuntimeError: Internal C++ object already deleted
    win._manual_update_check()
    assert win._manual_check_worker is not None
    assert win._manual_check_worker is not dead_worker
    win._manual_check_worker.wait(2000)
    qapp.processEvents()


