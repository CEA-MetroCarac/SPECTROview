"""Integration tests for end-to-end multi-platform update detection, verification, and installation."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

import spectroview.model.m_update_manager as um
from spectroview.model.m_installation import InstallType, InstallationInfo
from spectroview.model.m_update_manager import (
    UnifiedUpdateCheckWorker,
    UnifiedUpdateDownloadWorker,
    UpdateManager,
    UpdateStatus,
    WindowsPortableUpdater,
)
from spectroview.settings.m_settings import MSettings


def _build_fake_wheel_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("spectroview/__init__.py", "VERSION = '26.47.0'\n")
    return buf.getvalue()


def _build_fake_portable_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("SPECTROview/SPECTROview.exe", b"MZ-new-exe")
        zf.writestr("SPECTROview/_internal/certifi/cacert.pem", b"PATCHED-CA")
    return buf.getvalue()


def _info(install_type: InstallType, **overrides) -> InstallationInfo:
    base = dict(
        install_type=install_type,
        platform_name="win32",
        architecture="x64",
        executable="C:/SPECTROview/current/SPECTROview.exe",
        app_root="C:/SPECTROview",
        is_frozen=install_type not in (InstallType.DEV_SOURCE, InstallType.PIP_WHEEL),
        is_portable=install_type in (InstallType.WINDOWS_PORTABLE, InstallType.LINUX_APPIMAGE),
        is_dev=(install_type == InstallType.DEV_SOURCE),
        update_supported=install_type not in (InstallType.DEV_SOURCE, InstallType.STANDALONE_UNKNOWN),
        auto_update_supported=install_type in (InstallType.PIP_WHEEL, InstallType.WINDOWS_VELOPACK),
        update_backend=install_type.value,
        label=install_type.value,
        appimage_path="",
        details="integration test",
    )
    base.update(overrides)
    return InstallationInfo(**base)


@pytest.fixture()
def simulated_release_transport(monkeypatch):
    """Serve complete GitHub release payloads and SHA256SUMS.txt via m_update_manager.urlopen."""
    payloads = {
        "spectroview-26.47.0-py3-none-any.whl": _build_fake_wheel_bytes(),
        "releases.win.json": b'{"Assets": []}',
        "SPECTROview-26.47.0-delta.nupkg": b"vpk-delta-bytes",
        "SPECTROview_26.47.0_Windows_x64_Setup.exe": b"MZ-setup",
        "SPECTROview_26.47.0_Windows_x64_Portable.zip": _build_fake_portable_zip_bytes(),
        "SPECTROview_26.47.0_macOS_arm64.dmg": b"koly-dmg-arm64",
        "SPECTROview_26.47.0_Linux_x86_64.AppImage": b"\x7fELF-appimage-v26.47.0",
        "SPECTROview_26.47.0_Linux_amd64.deb": b"!<arch>\ndebian-binary",
    }
    sums_lines = [
        f"{hashlib.sha256(data).hexdigest()}  {name}"
        for name, data in payloads.items()
    ]
    sums_bytes = ("\n".join(sums_lines) + "\n").encode("utf-8")

    api_json = json.dumps(
        {
            "tag_name": "v26.47.0",
            "name": "SPECTROview 26.47.0",
            "html_url": "https://github.com/CEA-MetroCarac/SPECTROview/releases/tag/v26.47.0",
            "body": "Multi-platform release",
            "prerelease": False,
            "draft": False,
            "published_at": "2026-11-08T12:00:00Z",
            "assets": [
                {
                    "name": "SHA256SUMS.txt",
                    "browser_download_url": "https://downloads.test/SHA256SUMS.txt",
                    "size": len(sums_bytes),
                },
                *[
                    {
                        "name": name,
                        "browser_download_url": f"https://downloads.test/{name}",
                        "size": len(data),
                    }
                    for name, data in payloads.items()
                ],
            ],
        }
    ).encode("utf-8")

    tamper_targets: set[str] = set()

    class FakeResp:
        def __init__(self, body: bytes):
            self._buf = io.BytesIO(body)
            self.headers = {"Content-Length": str(len(body)), "ETag": 'W/"v26.47.0"'}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, n=-1):
            return self._buf.read(n)

    def fake_urlopen(req, timeout=0, context=None):
        url = req.full_url
        if "releases/latest" in url:
            return FakeResp(api_json)
        if url.endswith("/SHA256SUMS.txt"):
            return FakeResp(sums_bytes)
        name = url.rsplit("/", 1)[-1]
        data = payloads[name]
        if name in tamper_targets:
            data = data + b"CORRUPTED"
        return FakeResp(data)

    monkeypatch.setattr(um, "urlopen", fake_urlopen)
    monkeypatch.setattr(um, "build_ssl_context", lambda strict_certifi=True: None)
    return tamper_targets


@pytest.mark.parametrize(
    "install_type,extra_info,expected_status",
    [
        (InstallType.PIP_WHEEL, {}, UpdateStatus.RESTART_REQUIRED),
        (
            InstallType.WINDOWS_PORTABLE,
            {},
            UpdateStatus.GUIDED_ACTION_REQUIRED,
        ),
        (
            InstallType.MACOS_BUNDLE,
            {"platform_name": "darwin", "architecture": "arm64"},
            UpdateStatus.GUIDED_ACTION_REQUIRED,
        ),
        (
            InstallType.LINUX_DEB,
            {"platform_name": "linux", "architecture": "x64"},
            UpdateStatus.GUIDED_ACTION_REQUIRED,
        ),
    ],
)
def test_e2e_update_check_download_verify_and_settings_preservation(
    tmp_path,
    monkeypatch,
    simulated_release_transport,
    install_type,
    extra_info,
    expected_status,
):
    monkeypatch.setattr(um, "resolve_downloads_dir", lambda: tmp_path / "downloads")
    monkeypatch.setattr(um, "install_update_and_restart", lambda p: None)

    settings = MSettings()
    work_dir = tmp_path / "Working Folder NonASCII café"
    work_dir.mkdir()
    settings.set_working_folder(str(work_dir))
    settings.set_theme("classic_dark")

    mgr = UpdateManager(
        current_version="26.46.1",
        settings=settings,
        install_info=_info(install_type, **extra_info),
    )

    check_res = mgr.check_for_updates(force=True)
    assert check_res.update_available is True
    assert check_res.latest_version == "26.47.0"
    assert check_res.selected_asset is not None
    assert len(check_res.selected_asset.sha256) == 64

    dl_path = mgr.download_update(release=check_res.release)
    assert dl_path.exists()

    install_res = mgr.install_update(dl_path, release=check_res.release)
    assert install_res.status == expected_status
    assert install_res.success is True

    if install_type == InstallType.WINDOWS_PORTABLE:
        extracted = WindowsPortableUpdater.extract_portable_archive(
            dl_path,
            tmp_path / "Updated Portable Folder",
        )
        assert (extracted / "SPECTROview" / "SPECTROview.exe").read_bytes() == b"MZ-new-exe"

    # Settings & Working Folder survive upgrade independently of the binary location
    reloaded = MSettings()
    assert Path(reloaded.get_working_folder()) == work_dir
    assert reloaded.get_theme() == "classic_dark"


def test_e2e_corrupted_download_aborts_before_install(tmp_path, monkeypatch, simulated_release_transport):
    monkeypatch.setattr(um, "resolve_downloads_dir", lambda: tmp_path / "downloads")
    simulated_release_transport.add("SPECTROview_26.47.0_Linux_amd64.deb")

    mgr = UpdateManager(
        current_version="26.46.1",
        install_info=_info(InstallType.LINUX_DEB, platform_name="linux"),
    )
    res = mgr.check_for_updates(force=True)
    assert res.update_available is True

    with pytest.raises(ValueError, match="Integrity verification failed"):
        mgr.download_update(release=res.release)
    assert mgr.get_update_status() == UpdateStatus.FAILED
    assert not (tmp_path / "downloads" / "SPECTROview_26.47.0_Linux_amd64.deb").exists()


def test_e2e_workers_drive_qt_signals_end_to_end(qapp, qtbot, tmp_path, monkeypatch, simulated_release_transport):
    monkeypatch.setattr(um, "resolve_downloads_dir", lambda: tmp_path / "downloads")

    # Exercise Velopack worker path end-to-end cleanly
    class FakeAsset:
        def __init__(self, filename):
            self.FileName = filename
            self.SHA256 = ""

    class FakeUpdateInfo:
        TargetFullRelease = FakeAsset("SPECTROview-26.47.0-full.nupkg")
        DeltasToTarget = [FakeAsset("SPECTROview-26.47.0-delta.nupkg")]

    class FakeVpkMgr:
        def __init__(self, source):
            assert not source.startswith("http")

        def get_current_version(self):
            return "26.46.1"

        def check_for_updates(self):
            return FakeUpdateInfo()

        def download_updates(self, info, cb=None):
            if cb:
                cb(100)

        def wait_exit_then_apply_updates(self, info, silent=False, restart=True):
            return None

    monkeypatch.setitem(sys.modules, "velopack", types.SimpleNamespace(UpdateManager=FakeVpkMgr))

    mgr = UpdateManager(
        current_version="26.46.1",
        install_info=_info(InstallType.WINDOWS_VELOPACK),
    )

    check_worker = UnifiedUpdateCheckWorker(mgr, force=True)
    with qtbot.waitSignal(check_worker.check_succeeded, timeout=5000) as check_blocker:
        check_worker.start()
    check_worker.wait(5000)

    check_res = check_blocker.args[0]
    assert check_res.can_auto_install is True

    dl_worker = UnifiedUpdateDownloadWorker(mgr, check_res.release)
    with qtbot.waitSignal(dl_worker.download_finished, timeout=5000) as dl_blocker:
        dl_worker.start()
    dl_worker.wait(5000)

    staged = Path(dl_blocker.args[0])
    assert staged.is_dir()
    install_res = mgr.install_update(staged, release=check_res.release)
    assert install_res.restart_required is True

