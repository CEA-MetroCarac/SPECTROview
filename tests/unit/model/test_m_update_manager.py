"""Unit tests for centralized UpdateManager, semver, caching, and platform backends."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import types
import zipfile
from pathlib import Path
from urllib.error import HTTPError

import pytest

import spectroview.model.m_update_manager as um
from spectroview.model.m_installation import InstallType, InstallationInfo
from spectroview.model.m_update_manager import (
    GitHubReleaseClient,
    LinuxAppImageUpdater,
    LinuxDebUpdater,
    MacOSUpdater,
    ReleaseAsset,
    ReleaseInfo,
    UpdateManager,
    UpdateStatus,
    WindowsPortableUpdater,
    is_newer_version,
    parse_semver,
    parse_sha256sums_text,
)
from spectroview.settings.m_settings import MSettings


def _info(install_type: InstallType, **overrides) -> InstallationInfo:
    base = dict(
        install_type=install_type,
        platform_name="win32",
        architecture="x64",
        executable="C:/App/SPECTROview.exe",
        app_root="C:/App",
        is_frozen=install_type not in (InstallType.DEV_SOURCE, InstallType.PIP_WHEEL),
        is_portable=install_type in (InstallType.WINDOWS_PORTABLE, InstallType.LINUX_APPIMAGE),
        is_dev=(install_type == InstallType.DEV_SOURCE),
        update_supported=install_type not in (InstallType.DEV_SOURCE, InstallType.STANDALONE_UNKNOWN),
        auto_update_supported=install_type in (InstallType.PIP_WHEEL, InstallType.WINDOWS_VELOPACK),
        update_backend=install_type.value,
        label=install_type.value,
        appimage_path="",
        details="test",
    )
    base.update(overrides)
    return InstallationInfo(**base)


def _sample_release(tag: str = "v26.47.0", prerelease: bool = False) -> ReleaseInfo:
    ver = tag.lstrip("v")
    return ReleaseInfo(
        tag=tag,
        version=ver,
        name=tag,
        html_url=f"https://github.com/CEA-MetroCarac/SPECTROview/releases/tag/{tag}",
        notes="Release notes",
        prerelease=prerelease,
        published_at="2026-11-08T00:00:00Z",
        assets=(
            ReleaseAsset(
                name=f"spectroview-{ver}-py3-none-any.whl",
                url=f"https://example.test/spectroview-{ver}-py3-none-any.whl",
                sha256="",
            ),
            ReleaseAsset(
                name="releases.win.json",
                url="https://example.test/releases.win.json",
                sha256="",
            ),
            ReleaseAsset(
                name=f"SPECTROview-{ver}-delta.nupkg",
                url=f"https://example.test/SPECTROview-{ver}-delta.nupkg",
                sha256="",
            ),
            ReleaseAsset(
                name=f"SPECTROview-{ver}-full.nupkg",
                url=f"https://example.test/SPECTROview-{ver}-full.nupkg",
                sha256="",
            ),
            ReleaseAsset(
                name=f"SPECTROview_{ver}_Windows_x64_Setup.exe",
                url=f"https://example.test/SPECTROview_{ver}_Windows_x64_Setup.exe",
            ),
            ReleaseAsset(
                name=f"SPECTROview_{ver}_Windows_x64_Portable.zip",
                url=f"https://example.test/SPECTROview_{ver}_Windows_x64_Portable.zip",
            ),
            ReleaseAsset(
                name=f"SPECTROview_{ver}_macOS_arm64.dmg",
                url=f"https://example.test/SPECTROview_{ver}_macOS_arm64.dmg",
            ),
            ReleaseAsset(
                name=f"SPECTROview_{ver}_Linux_x86_64.AppImage",
                url=f"https://example.test/SPECTROview_{ver}_Linux_x86_64.AppImage",
            ),
            ReleaseAsset(
                name=f"SPECTROview_{ver}_Linux_amd64.deb",
                url=f"https://example.test/SPECTROview_{ver}_Linux_amd64.deb",
            ),
        ),
    )


class TestSemverAndChecksums:
    def test_semver_ordering_handles_prereleases(self):
        assert parse_semver("26.47.0-rc.1") < parse_semver("26.47.0")
        assert parse_semver("26.47.0-beta.2") < parse_semver("26.47.0-rc.1")
        assert parse_semver("26.47.0-rc.1") < parse_semver("26.47.0-rc.2")
        assert is_newer_version("v26.47.0", "26.46.1") is True
        assert is_newer_version("v26.47.0-rc.1", "26.47.0") is False
        assert is_newer_version("26.47.0", "26.47.0-rc.1") is True
        assert is_newer_version("26.46.1", "26.46.1") is False

    def test_parse_sha256sums_text(self):
        h1 = "a" * 64
        h2 = "b" * 64
        text = f"{h1}  SPECTROview_26.47.0_Windows_x64_Setup.exe\n{h2} *dist/spectroview-26.47.0-py3-none-any.whl\n"
        parsed = parse_sha256sums_text(text)
        assert parsed["SPECTROview_26.47.0_Windows_x64_Setup.exe"] == h1
        assert parsed["spectroview-26.47.0-py3-none-any.whl"] == h2


class TestGitHubReleaseClient:
    def test_etag_and_ttl_cache_and_sha256sums(self, monkeypatch):
        settings = MSettings()
        client = GitHubReleaseClient(settings=settings)
        h_wheel = "c" * 64
        api_payload = {
            "tag_name": "v26.47.0",
            "name": "SPECTROview 26.47.0",
            "html_url": "https://github.com/CEA-MetroCarac/SPECTROview/releases/tag/v26.47.0",
            "body": "Notes",
            "prerelease": False,
            "draft": False,
            "published_at": "2026-11-08T00:00:00Z",
            "assets": [
                {
                    "name": "spectroview-26.47.0-py3-none-any.whl",
                    "browser_download_url": "https://example.test/spectroview-26.47.0-py3-none-any.whl",
                    "size": 123,
                },
                {
                    "name": "SHA256SUMS.txt",
                    "browser_download_url": "https://example.test/SHA256SUMS.txt",
                    "size": 100,
                },
            ],
        }

        calls = []

        class FakeResponse:
            def __init__(self, body_bytes: bytes, etag: str = ""):
                self._bytes = body_bytes
                self.headers = {"ETag": etag}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return self._bytes

        def fake_urlopen(req, timeout=0, context=None):
            calls.append((req.full_url, req.headers.get("If-none-match", "")))
            if "SHA256SUMS.txt" in req.full_url:
                return FakeResponse(f"{h_wheel}  spectroview-26.47.0-py3-none-any.whl\n".encode())
            return FakeResponse(json.dumps(api_payload).encode(), etag='W/"abc123"')

        monkeypatch.setattr(um, "urlopen", fake_urlopen)
        monkeypatch.setattr(um, "build_ssl_context", lambda strict_certifi=True: None)

        rel, from_cache = client.fetch_latest_release(force=False)
        assert from_cache is False
        assert rel is not None
        assert rel.asset_by_name("spectroview-26.47.0-py3-none-any.whl").sha256 == h_wheel

        # Subsequent non-forced call hits local TTL cache without HTTP
        calls.clear()
        rel_cached, from_cache_2 = client.fetch_latest_release(force=False)
        assert from_cache_2 is True
        assert rel_cached is not None
        assert calls == []

        # Simulate expired TTL -> sends If-None-Match & falls back cleanly on 304 Not Modified
        cache = settings.get_update_cache()
        cache["checked_at"] = 0.0
        settings.set_update_cache(cache)

        def fake_urlopen_304(req, timeout=0, context=None):
            calls.append(req.headers.get("If-none-match", ""))
            raise HTTPError(req.full_url, 304, "Not Modified", {}, None)

        monkeypatch.setattr(um, "urlopen", fake_urlopen_304)
        rel_304, from_cache_3 = client.fetch_latest_release(force=False)
        assert from_cache_3 is True
        assert rel_304.tag == "v26.47.0"
        assert calls == ['W/"abc123"']

    def test_rate_limit_403_raises_friendly_error(self, monkeypatch):
        client = GitHubReleaseClient()
        monkeypatch.setattr(um, "build_ssl_context", lambda strict_certifi=True: None)

        def fake_urlopen(req, timeout=0, context=None):
            raise HTTPError(req.full_url, 403, "Forbidden", {"X-RateLimit-Reset": "1800000000"}, None)

        monkeypatch.setattr(um, "urlopen", fake_urlopen)
        with pytest.raises(RuntimeError, match="GitHub API rate limit reached"):
            client.fetch_latest_release(force=True)

    def test_download_file_verifies_sha256_and_cleans_part_on_mismatch(self, tmp_path, monkeypatch):
        client = GitHubReleaseClient()
        payload = b"hello verified update"
        good_sha = hashlib.sha256(payload).hexdigest()
        bad_sha = "0" * 64

        class StreamResp:
            headers = {"Content-Length": str(len(payload))}

            def __init__(self):
                self._buf = io.BytesIO(payload)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, n):
                return self._buf.read(n)

        monkeypatch.setattr(um, "urlopen", lambda *a, **kw: StreamResp())
        monkeypatch.setattr(um, "build_ssl_context", lambda strict_certifi=True: None)

        dest_bad = tmp_path / "bad.bin"
        with pytest.raises(ValueError, match="Integrity verification failed"):
            client.download_file("https://example.test/f.bin", dest_bad, expected_sha256=bad_sha)
        assert not dest_bad.exists()
        assert list(tmp_path.glob("*.part*")) == []

        progress = []
        dest_ok = tmp_path / "good.bin"
        client.download_file(
            "https://example.test/f.bin",
            dest_ok,
            expected_sha256=good_sha,
            progress_cb=progress.append,
        )
        assert dest_ok.read_bytes() == payload
        assert progress[-1] == 100


class TestPlatformBackends:
    def test_asset_selection_across_all_platforms(self):
        rel = _sample_release("v26.47.0")
        client = GitHubReleaseClient()

        # Windows Velopack selects releases.win.json
        vpk = UpdateManager("26.46.1", install_info=_info(InstallType.WINDOWS_VELOPACK), client=client)
        assert vpk.get_backend().select_asset(rel).name == "releases.win.json"

        # Windows Portable selects Portable.zip
        port = UpdateManager("26.46.1", install_info=_info(InstallType.WINDOWS_PORTABLE), client=client)
        assert port.get_backend().select_asset(rel).name.endswith("_Windows_x64_Portable.zip")

        # macOS Apple Silicon selects _macOS_arm64.dmg
        mac_arm = UpdateManager(
            "26.46.1",
            install_info=_info(InstallType.MACOS_BUNDLE, platform_name="darwin", architecture="arm64"),
            client=client,
        )
        assert mac_arm.get_backend().select_asset(rel).name.endswith("_macOS_arm64.dmg")

        # macOS Intel does not select ARM64 binary from newer releases
        mac_intel = UpdateManager(
            "26.46.1",
            install_info=_info(InstallType.MACOS_BUNDLE, platform_name="darwin", architecture="x64"),
            client=client,
        )
        assert mac_intel.get_backend().select_asset(rel) is None

        # But if a legacy release contains an x64 or universal DMG, it still selects it
        legacy_rel = ReleaseInfo(
            tag="v26.43.1",
            version="26.43.1",
            name="v26.43.1",
            html_url="https://example.test",
            notes="",
            prerelease=False,
            published_at="",
            assets=(
                ReleaseAsset(
                    name="SPECTROview_26.43.1_macOS_x64.dmg",
                    url="https://example.test/SPECTROview_26.43.1_macOS_x64.dmg",
                ),
            ),
        )
        assert mac_intel.get_backend().select_asset(legacy_rel).name.endswith("_macOS_x64.dmg")

        # Linux AppImage vs .deb
        appimg = UpdateManager(
            "26.46.1",
            install_info=_info(InstallType.LINUX_APPIMAGE, platform_name="linux"),
            client=client,
        )
        assert appimg.get_backend().select_asset(rel).name.endswith(".AppImage")

        deb = UpdateManager(
            "26.46.1",
            install_info=_info(InstallType.LINUX_DEB, platform_name="linux"),
            client=client,
        )
        assert deb.get_backend().select_asset(rel).name.endswith("_Linux_amd64.deb")

    def test_windows_velopack_uses_verified_local_feed_dir_never_rust_http(self, tmp_path, monkeypatch):
        delta_bytes = b"delta-nupkg-content"
        delta_sha = hashlib.sha256(delta_bytes).hexdigest()
        rel = ReleaseInfo(
            tag="v26.47.0",
            version="26.47.0",
            name="26.47.0",
            html_url="https://example.test",
            notes="",
            prerelease=False,
            published_at="",
            assets=(
                ReleaseAsset(name="releases.win.json", url="https://example.test/releases.win.json"),
                ReleaseAsset(
                    name="SPECTROview-26.47.0-delta.nupkg",
                    url="https://example.test/SPECTROview-26.47.0-delta.nupkg",
                    sha256=delta_sha,
                ),
            ),
        )

        # Fake velopack SDK asserting it receives a local directory path, not an HTTPS URL
        constructed_sources = []
        waited_updates = []

        class FakeAsset:
            def __init__(self, filename, sha256=""):
                self.FileName = filename
                self.SHA256 = sha256

        class FakeUpdateInfo:
            def __init__(self):
                self.TargetFullRelease = FakeAsset("SPECTROview-26.47.0-full.nupkg")
                self.DeltasToTarget = [FakeAsset("SPECTROview-26.47.0-delta.nupkg", delta_sha)]

        class FakeUpdateManager:
            def __init__(self, source):
                constructed_sources.append(source)
                self._info = FakeUpdateInfo()

            def get_current_version(self):
                return "26.46.1"

            def check_for_updates(self):
                return self._info

            def download_updates(self, info, cb=None):
                # Must already be downloaded locally by Python m_tls client!
                assert Path(constructed_sources[-1], info.DeltasToTarget[0].FileName).read_bytes() == delta_bytes
                if cb:
                    cb(100)

            def wait_exit_then_apply_updates(self, info, silent=False, restart=True):
                waited_updates.append((info, silent, restart))

        fake_velopack = types.SimpleNamespace(UpdateManager=FakeUpdateManager)
        monkeypatch.setitem(sys.modules, "velopack", fake_velopack)

        class FakeClient(GitHubReleaseClient):
            def download_file(self, url, dest_path, **kwargs):
                if url.endswith("releases.win.json"):
                    dest_path.write_text('{"Assets": []}', encoding="utf-8")
                else:
                    assert kwargs.get("expected_sha256") == delta_sha
                    dest_path.write_bytes(delta_bytes)
                return dest_path

        mgr = UpdateManager("26.46.1", install_info=_info(InstallType.WINDOWS_VELOPACK), client=FakeClient())
        staged_dir = mgr.download_update(release=rel)
        assert staged_dir.is_dir()
        assert len(constructed_sources) == 1
        assert not constructed_sources[0].startswith("http")

        install_res = mgr.install_update(staged_dir, release=rel)
        assert install_res.restart_required is True
        assert mgr.restart_application() is True
        assert len(waited_updates) == 1
        assert waited_updates[0][2] is True

    def test_windows_portable_extraction_blocks_zip_slip(self, tmp_path):
        good_zip = tmp_path / "good.zip"
        with zipfile.ZipFile(good_zip, "w") as zf:
            zf.writestr("SPECTROview/SPECTROview.exe", b"MZ")

        dest = tmp_path / "extracted"
        WindowsPortableUpdater.extract_portable_archive(good_zip, dest)
        assert (dest / "SPECTROview" / "SPECTROview.exe").read_bytes() == b"MZ"

        evil_zip = tmp_path / "evil.zip"
        with zipfile.ZipFile(evil_zip, "w") as zf:
            zf.writestr("../../outside.txt", b"evil")

        with pytest.raises(ValueError, match="Blocked unsafe archive path"):
            WindowsPortableUpdater.extract_portable_archive(evil_zip, tmp_path / "safe_dest")

    def test_linux_appimage_checks_elf_header_and_builds_swap_script(self, tmp_path):
        appimage = tmp_path / "SPECTROview.AppImage"
        appimage.write_bytes(b"\x7fELFold")
        new_appimage = tmp_path / "staging" / "new.AppImage"
        new_appimage.parent.mkdir()
        new_appimage.write_bytes(b"\x7fELFnew")

        info = _info(
            InstallType.LINUX_APPIMAGE,
            platform_name="linux",
            auto_update_supported=True,
            appimage_path=str(appimage),
        )
        backend = LinuxAppImageUpdater(info, GitHubReleaseClient())
        res = backend.install_update(new_appimage)
        assert res.restart_required is True

        script = LinuxAppImageUpdater._swap_script_content(new_appimage, appimage, 12345)
        assert "PID=12345" in script
        assert appimage.as_posix() in script
        assert "chmod +x" in script

    def test_linux_deb_install_returns_apt_command(self, tmp_path):
        deb_file = tmp_path / "SPECTROview_26.47.0_Linux_amd64.deb"
        deb_file.write_bytes(b"!<arch>\n")
        backend = LinuxDebUpdater(_info(InstallType.LINUX_DEB, platform_name="linux"), GitHubReleaseClient())
        res = backend.install_update(deb_file)
        assert res.requires_manual_action is True
        assert "sudo apt install" in res.command
        assert str(deb_file) in res.command

    def test_dev_checkout_refuses_self_install(self, tmp_path):
        mgr = UpdateManager("26.46.1", install_info=_info(InstallType.DEV_SOURCE), client=GitHubReleaseClient())
        res = mgr.install_update(tmp_path / "dummy.whl")
        assert res.status == UpdateStatus.UNSUPPORTED
        assert res.success is False

    def test_python_updater_downloads_validates_and_delegates_restart(self, tmp_path, monkeypatch):
        wheel_buf = io.BytesIO()
        with zipfile.ZipFile(wheel_buf, "w") as zf:
            zf.writestr("spectroview/__init__.py", "VERSION = '26.47.0'\n")
        wheel_bytes = wheel_buf.getvalue()
        wheel_sha = hashlib.sha256(wheel_bytes).hexdigest()

        rel = ReleaseInfo(
            tag="v26.47.0",
            version="26.47.0",
            name="v26.47.0",
            html_url="https://example.test",
            notes="",
            prerelease=False,
            published_at="",
            assets=(
                ReleaseAsset(
                    name="spectroview-26.47.0-py3-none-any.whl",
                    url="https://example.test/spectroview-26.47.0-py3-none-any.whl",
                    sha256=wheel_sha,
                ),
            ),
        )

        class FakeClient(GitHubReleaseClient):
            def download_file(self, url, dest_path, **kwargs):
                assert kwargs.get("expected_sha256") == wheel_sha
                dest_path.write_bytes(wheel_bytes)
                return dest_path

        restarted = []
        monkeypatch.setattr(um, "install_update_and_restart", lambda p: restarted.append(p))

        mgr = UpdateManager("26.46.1", install_info=_info(InstallType.PIP_WHEEL), client=FakeClient())
        dl_path = mgr.download_update(release=rel)
        assert dl_path.name == "spectroview-26.47.0-py3-none-any.whl"
        assert mgr.get_update_status() == UpdateStatus.READY_TO_INSTALL

        res = mgr.install_update(dl_path, release=rel)
        assert res.status == UpdateStatus.RESTART_REQUIRED
        assert res.restart_required is True
        assert restarted == [dl_path]

    def test_macos_updater_guided_instructions_mention_gatekeeper(self, tmp_path):
        dmg = tmp_path / "SPECTROview_26.47.0_macOS_arm64.dmg"
        dmg.write_bytes(b"DMG")
        backend = MacOSUpdater(
            _info(InstallType.MACOS_BUNDLE, platform_name="darwin", architecture="arm64"),
            GitHubReleaseClient(),
        )
        res = backend.install_update(dmg)
        assert res.status == UpdateStatus.GUIDED_ACTION_REQUIRED
        assert res.requires_manual_action is True
        assert "Privacy & Security" in res.instructions
        assert "Control-click" in res.instructions

    def test_windows_velopack_fallback_to_setup_when_sdk_missing(self, tmp_path, monkeypatch):
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def block_velopack(name, *args, **kwargs):
            if name == "velopack":
                raise ImportError("no velopack")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", block_velopack)

        rel = _sample_release("v26.47.0")
        setup_bytes = b"MZ-setup-installer"

        class FakeClient(GitHubReleaseClient):
            def download_file(self, url, dest_path, **kwargs):
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_bytes(setup_bytes)
                return dest_path

        mgr = UpdateManager("26.46.1", install_info=_info(InstallType.WINDOWS_VELOPACK), client=FakeClient())
        backend = mgr.get_backend()
        assert backend.can_auto_install(rel) is False
        res_path = mgr.download_update(release=rel)
        assert res_path.name.endswith("_Windows_x64_Setup.exe")
        install_res = mgr.install_update(res_path, release=rel)
        assert install_res.status == UpdateStatus.GUIDED_ACTION_REQUIRED
        assert install_res.requires_manual_action is True

    def test_update_manager_check_respects_prerelease_setting_and_error_state(self, monkeypatch):
        settings = MSettings()
        settings.set_update_include_prereleases(True)
        rel_rc = _sample_release("v26.47.0-rc.1", prerelease=True)

        class FakeClient(GitHubReleaseClient):
            def __init__(self, settings=None):
                super().__init__(settings=settings)
                self.seen_flags = []

            def fetch_latest_release(self, *, include_prereleases=False, force=False, cache_ttl_seconds=0):
                self.seen_flags.append((include_prereleases, force))
                return rel_rc, False

        client = FakeClient(settings=settings)
        mgr = UpdateManager("26.46.1", settings=settings, install_info=_info(InstallType.PIP_WHEEL), client=client)
        res = mgr.check_for_updates(force=True)
        assert client.seen_flags == [(True, True)]
        assert res.update_available is True
        assert res.latest_version == "26.47.0-rc.1"
        assert mgr.get_latest_version() == "26.47.0-rc.1"
        assert mgr.get_update_status() == UpdateStatus.UPDATE_AVAILABLE

        # Error state transitions UpdateStatus -> FAILED cleanly
        class FailingClient(GitHubReleaseClient):
            def fetch_latest_release(self, **kwargs):
                raise RuntimeError("offline")

        failing_mgr = UpdateManager("26.46.1", install_info=_info(InstallType.PIP_WHEEL), client=FailingClient())
        with pytest.raises(RuntimeError, match="offline"):
            failing_mgr.check_for_updates(force=True)
        assert failing_mgr.get_update_status() == UpdateStatus.FAILED
        assert failing_mgr.get_last_error() == "offline"
