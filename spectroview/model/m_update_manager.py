"""Unified multi-platform version checking, release resolution, and update engine.

Provides a single :class:`UpdateManager` facade backed by distribution-specific
updaters selected via :class:`~spectroview.model.m_installation.InstallationManager`:

- ``PIP_WHEEL``: hardens in-place wheel upgrade and relaunch via ``pip``.
- ``WINDOWS_VELOPACK``: fetches release metadata and ``.nupkg`` full/delta
  packages in Python using hardened ``m_tls`` (avoiding stale bundled CAs in
  native Rust HTTP crates), then hands the verified local feed directory to
  ``velopack.UpdateManager``.
- ``WINDOWS_PORTABLE``: downloads and SHA-256 verifies the Portable ``.zip`` with
  safe zip-slip-checked extraction or Explorer reveal.
- ``MACOS_BUNDLE``: downloads and verifies architecture-matched ``.dmg`` and
  surfaces clear Gatekeeper instructions for unsigned bundles.
- ``LINUX_APPIMAGE``: downloads and verifies ``.AppImage`` (ELF + SHA-256) and
  performs atomic in-place replacement + relaunch when writable.
- ``LINUX_DEB``: downloads and verifies ``.deb`` (``!<arch>`` + SHA-256) and
  provides installation instructions preserving user configuration.
- ``DEV_SOURCE`` / ``STANDALONE_UNKNOWN``: checks versions for awareness while
  never overwriting source trees or unrecognised binaries.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from PySide6.QtCore import QThread, Signal

from spectroview import VERSION as _CURRENT_VERSION
from spectroview.model.m_installation import (
    InstallType,
    InstallationInfo,
    InstallationManager,
)
from spectroview.model.m_tls import InsecureTrustStoreError, build_ssl_context
from spectroview.model.m_update_checker import (
    _normalise_sha256,
    _wheel_filename_from_url,
    install_update_and_restart,
)

log = logging.getLogger(__name__)

GITHUB_REPO = "CEA-MetroCarac/SPECTROview"
GITHUB_LATEST_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
GITHUB_RELEASES_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases?per_page=10"
GITHUB_RELEASES_HTML_URL = f"https://github.com/{GITHUB_REPO}/releases"

REQUEST_TIMEOUT = 8
DOWNLOAD_TIMEOUT = 120
DOWNLOAD_CHUNK_SIZE = 64 * 1024
DEFAULT_CACHE_TTL_SECONDS = 6 * 3600


class UpdateStatus(str, Enum):
    """Lifecycle states exposed by :class:`UpdateManager`."""

    IDLE = "idle"
    CHECKING = "checking"
    UP_TO_DATE = "up_to_date"
    UPDATE_AVAILABLE = "update_available"
    DOWNLOADING = "downloading"
    READY_TO_INSTALL = "ready_to_install"
    INSTALLING = "installing"
    RESTART_REQUIRED = "restart_required"
    GUIDED_ACTION_REQUIRED = "guided_action_required"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


@dataclass(frozen=True)
class ReleaseAsset:
    """One downloadable file attached to a GitHub release."""

    name: str
    url: str
    size: int = 0
    sha256: str = ""
    content_type: str = ""


@dataclass(frozen=True)
class ReleaseInfo:
    """Normalised metadata for a published GitHub release."""

    tag: str
    version: str
    name: str
    html_url: str
    notes: str
    prerelease: bool
    published_at: str
    assets: tuple[ReleaseAsset, ...] = field(default_factory=tuple)
    sha256_sums: dict[str, str] = field(default_factory=dict)

    def asset_by_name(self, name: str) -> Optional[ReleaseAsset]:
        target = name.lower()
        for asset in self.assets:
            if asset.name.lower() == target:
                return asset
        return None


@dataclass(frozen=True)
class UpdateCheckResult:
    """Outcome of checking for updates against the active backend."""

    status: UpdateStatus
    current_version: str
    latest_version: str
    update_available: bool
    release: Optional[ReleaseInfo] = None
    selected_asset: Optional[ReleaseAsset] = None
    can_auto_install: bool = False
    action_label: str = "Update"
    action_tooltip: str = ""
    message: str = ""
    from_cache: bool = False


@dataclass(frozen=True)
class InstallResult:
    """Outcome of preparing or applying a downloaded update artifact."""

    status: UpdateStatus
    success: bool
    restart_required: bool = False
    requires_manual_action: bool = False
    message: str = ""
    artifact_path: str = ""
    command: str = ""
    instructions: str = ""
    can_extract_zip: bool = False


_SEMVER_RE = re.compile(
    r"^v?(?P<major>\d+)(?:\.(?P<minor>\d+))?(?:\.(?P<patch>\d+))?"
    r"(?:[-.]?(?P<pre>(?:alpha|beta|rc|a|b|dev|pre)\w*(?:[.\-]\d+)*))?$",
    re.IGNORECASE,
)


def parse_semver(version_str: str) -> tuple[tuple[int, int, int], int, tuple[tuple[int, int | str], ...]]:
    """Return a comparison key ``((major, minor, patch), stability_flag, pre_parts)``.

    ``stability_flag`` is ``1`` for stable releases and ``0`` for pre-releases so
    that ``26.47.0-rc.1 < 26.47.0`` and ``26.47.0-rc.1 < 26.47.0-rc.2``.
    """
    cleaned = str(version_str or "").strip().split("+", 1)[0]
    match = _SEMVER_RE.match(cleaned)
    if not match:
        nums = [int(x) for x in re.findall(r"\d+", cleaned)[:3]]
        while len(nums) < 3:
            nums.append(0)
        return (nums[0], nums[1], nums[2]), 1, ()

    core = (
        int(match.group("major") or 0),
        int(match.group("minor") or 0),
        int(match.group("patch") or 0),
    )
    pre_raw = match.group("pre")
    if not pre_raw:
        return core, 1, ()

    pre_norm = pre_raw.lower()
    if pre_norm.startswith("a") and not pre_norm.startswith("alpha"):
        pre_norm = "alpha" + pre_norm[1:]
    elif pre_norm.startswith("b") and not pre_norm.startswith("beta"):
        pre_norm = "beta" + pre_norm[1:]

    tokens: list[tuple[int, int | str]] = []
    for part in re.split(r"[.\-]|(?<=[a-z])(?=\d)", pre_norm):
        if not part:
            continue
        if part.isdigit():
            tokens.append((0, int(part)))
        else:
            tokens.append((1, part))
    return core, 0, tuple(tokens)


def is_newer_version(candidate: str, current: str) -> bool:
    """Return ``True`` strictly when *candidate* is newer than *current*."""
    cand_text = str(candidate or "").strip()
    curr_text = str(current or "").strip()
    if not cand_text:
        return False
    return parse_semver(cand_text) > parse_semver(curr_text)


def parse_sha256sums_text(text: str) -> dict[str, str]:
    """Parse GNU ``sha256sum`` formatted text into ``{filename: sha256_hex}``."""
    sums: dict[str, str] = {}
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"^([0-9a-fA-F]{64})\s+[\*\s]?(?:.*/)?([^\s/\\]+)$", stripped)
        if match:
            digest = match.group(1).lower()
            filename = match.group(2).strip()
            sums[filename] = digest
            sums[filename.lower()] = digest
    return sums


def resolve_downloads_dir() -> Path:
    """Return a user-friendly directory for downloaded standalone installers."""
    downloads = Path.home() / "Downloads"
    try:
        if downloads.is_dir():
            probe = downloads / f".spectroview_write_test_{os.getpid()}"
            probe.write_text("", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return downloads
    except OSError:
        pass
    return Path(tempfile.mkdtemp(prefix="spectroview-update-"))


def _asset_filename_from_url(url: str) -> str:
    return Path(unquote(urlparse(url).path)).name


def _release_from_api_dict(data: dict[str, Any], sha256_sums: Optional[dict[str, str]] = None) -> ReleaseInfo:
    sums = dict(sha256_sums or {})
    assets: list[ReleaseAsset] = []
    for raw_asset in data.get("assets", []) or []:
        if not isinstance(raw_asset, dict):
            continue
        name = str(raw_asset.get("name", "")).strip()
        url = str(raw_asset.get("browser_download_url", "")).strip()
        if not name or not url:
            continue
        digest = _normalise_sha256(str(raw_asset.get("digest", "") or ""))
        if not digest:
            digest = sums.get(name) or sums.get(name.lower()) or ""
        assets.append(
            ReleaseAsset(
                name=name,
                url=url,
                size=int(raw_asset.get("size") or 0),
                sha256=digest,
                content_type=str(raw_asset.get("content_type", "") or ""),
            )
        )

    tag = str(data.get("tag_name", "") or "").strip()
    return ReleaseInfo(
        tag=tag,
        version=tag.lstrip("v"),
        name=str(data.get("name") or tag or "").strip(),
        html_url=str(data.get("html_url") or GITHUB_RELEASES_HTML_URL).strip(),
        notes=str(data.get("body") or "").strip(),
        prerelease=bool(data.get("prerelease", False)),
        published_at=str(data.get("published_at") or ""),
        assets=tuple(assets),
        sha256_sums=sums,
    )


def _format_rate_limit_message(headers: Any) -> str:
    reset_ts = ""
    if headers is not None:
        raw_reset = headers.get("X-RateLimit-Reset")
        if raw_reset and str(raw_reset).isdigit():
            try:
                reset_str = time.strftime("%H:%M", time.localtime(int(raw_reset)))
                reset_ts = f" after {reset_str}"
            except Exception:
                pass
    return (
        f"GitHub API rate limit reached{reset_ts}. "
        f"Please try again later or open {GITHUB_RELEASES_HTML_URL}."
    )


class GitHubReleaseClient:
    """HTTP client for querying GitHub Releases and streaming verified downloads."""

    def __init__(self, settings: Any = None) -> None:
        self._settings = settings

    def _headers(self, etag: str = "") -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": f"SPECTROview-updater/{_CURRENT_VERSION}",
        }
        if etag:
            headers["If-None-Match"] = etag
        return headers

    def _load_cache(self, include_prereleases: bool) -> dict[str, Any]:
        if self._settings is None or not hasattr(self._settings, "get_update_cache"):
            return {}
        try:
            cache = self._settings.get_update_cache()
        except Exception:
            return {}
        if not isinstance(cache, dict):
            return {}
        if bool(cache.get("include_prereleases", False)) != bool(include_prereleases):
            return {}
        return cache

    def _save_cache(self, payload: dict[str, Any], etag: str, include_prereleases: bool) -> None:
        if self._settings is None or not hasattr(self._settings, "set_update_cache"):
            return
        try:
            self._settings.set_update_cache(
                {
                    "etag": etag,
                    "checked_at": time.time(),
                    "include_prereleases": bool(include_prereleases),
                    "payload": payload,
                }
            )
        except Exception:
            pass

    def _fetch_sha256_sums(self, release_dict: dict[str, Any], context: Any) -> dict[str, str]:
        for raw_asset in release_dict.get("assets", []) or []:
            name = str(raw_asset.get("name", "")).strip()
            url = str(raw_asset.get("browser_download_url", "")).strip()
            if name.upper() in ("SHA256SUMS.TXT", "SHA256SUMS") and url:
                try:
                    req = Request(url, headers={"User-Agent": f"SPECTROview-updater/{_CURRENT_VERSION}"})
                    with urlopen(req, timeout=REQUEST_TIMEOUT, context=context) as resp:
                        text = resp.read().decode("utf-8", errors="replace")
                    return parse_sha256sums_text(text)
                except Exception as exc:
                    log.debug("Failed to fetch SHA256SUMS asset %s: %s", name, exc)
                    return {}
        return {}

    def fetch_latest_release(
        self,
        *,
        include_prereleases: bool = False,
        force: bool = False,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    ) -> tuple[Optional[ReleaseInfo], bool]:
        """Return ``(ReleaseInfo, from_cache)`` for the newest matching GitHub release."""
        cache = self._load_cache(include_prereleases)
        cached_payload = cache.get("payload") if isinstance(cache.get("payload"), dict) else None
        cached_at = float(cache.get("checked_at") or 0.0)
        cached_etag = str(cache.get("etag") or "")

        if not force and cached_payload and (time.time() - cached_at) < cache_ttl_seconds:
            sums = cached_payload.get("_sha256_sums") or {}
            return _release_from_api_dict(cached_payload, sha256_sums=sums), True

        context = build_ssl_context(strict_certifi=True)
        url = GITHUB_RELEASES_API_URL if include_prereleases else GITHUB_LATEST_API_URL
        req = Request(url, headers=self._headers(etag="" if force else cached_etag))

        try:
            with urlopen(req, timeout=REQUEST_TIMEOUT, context=context) as response:
                etag = str(response.headers.get("ETag", "") or "")
                raw_obj = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 304 and cached_payload:
                self._save_cache(cached_payload, cached_etag, include_prereleases)
                sums = cached_payload.get("_sha256_sums") or {}
                return _release_from_api_dict(cached_payload, sha256_sums=sums), True
            if exc.code in (403, 429):
                raise RuntimeError(_format_rate_limit_message(exc.headers)) from exc
            raise RuntimeError(f"GitHub Releases request returned HTTP {exc.code}.") from exc
        except InsecureTrustStoreError:
            raise
        except (URLError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not contact GitHub Releases: {exc}") from exc

        release_dict: Optional[dict[str, Any]] = None
        if isinstance(raw_obj, list):
            candidates = [
                r
                for r in raw_obj
                if isinstance(r, dict)
                and not r.get("draft", False)
                and (include_prereleases or not r.get("prerelease", False))
            ]
            candidates.sort(key=lambda r: parse_semver(str(r.get("tag_name", ""))), reverse=True)
            if candidates:
                release_dict = candidates[0]
        elif isinstance(raw_obj, dict):
            if not raw_obj.get("draft", False):
                release_dict = raw_obj

        if not release_dict or not str(release_dict.get("tag_name", "")).strip():
            return None, False

        sums = self._fetch_sha256_sums(release_dict, context)
        cached_copy = dict(release_dict)
        cached_copy["_sha256_sums"] = sums
        self._save_cache(cached_copy, etag, include_prereleases)
        return _release_from_api_dict(release_dict, sha256_sums=sums), False

    def download_file(
        self,
        url: str,
        dest_path: Path,
        *,
        expected_sha256: str = "",
        progress_cb: Optional[Callable[[int], None]] = None,
        timeout: int = DOWNLOAD_TIMEOUT,
    ) -> Path:
        """Stream *url* over hardened TLS into *dest_path* cleanly with SHA-256 verification."""
        context = build_ssl_context(strict_certifi=True)
        expected = _normalise_sha256(expected_sha256)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = dest_path.with_name(f"{dest_path.name}.part.{os.getpid()}")

        try:
            req = Request(url, headers={"User-Agent": f"SPECTROview-updater/{_CURRENT_VERSION}"})
            with urlopen(req, timeout=timeout, context=context) as response:
                total_bytes = int(response.headers.get("Content-Length") or 0)
                hasher = hashlib.sha256()
                received = 0
                with part_path.open("wb") as out:
                    while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                        out.write(chunk)
                        hasher.update(chunk)
                        received += len(chunk)
                        if progress_cb is not None and total_bytes > 0:
                            progress_cb(min(99, int(received * 100 // total_bytes)))

            actual = hasher.hexdigest().lower()
            if expected and actual != expected:
                raise ValueError(
                    f"Integrity verification failed for {dest_path.name}: "
                    f"expected SHA-256 {expected[:12]}..., got {actual[:12]}..."
                )

            os.replace(part_path, dest_path)
            if progress_cb is not None:
                progress_cb(100)
            return dest_path
        except Exception:
            part_path.unlink(missing_ok=True)
            raise


class BaseUpdater:
    """Abstract base interface for installation-specific update backends."""

    action_label: str = "Update"
    action_tooltip: str = "Download and apply update."

    def __init__(self, info: InstallationInfo, client: GitHubReleaseClient) -> None:
        self.info = info
        self.client = client

    def select_asset(self, release: ReleaseInfo) -> Optional[ReleaseAsset]:
        return None

    def can_auto_install(self, release: Optional[ReleaseInfo] = None) -> bool:
        return False

    def download_update(
        self,
        release: ReleaseInfo,
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> Path:
        asset = self.select_asset(release)
        if asset is None:
            raise ValueError(
                f"Release {release.tag} does not attach an asset matching "
                f"{self.info.label}."
            )
        target_dir = resolve_downloads_dir()
        filename = _asset_filename_from_url(asset.url) or asset.name
        dest = target_dir / filename
        return self.client.download_file(
            asset.url,
            dest,
            expected_sha256=asset.sha256,
            progress_cb=progress_cb,
        )

    def install_update(self, downloaded_path: Path, release: Optional[ReleaseInfo] = None) -> InstallResult:
        raise NotImplementedError

    def restart_application(self) -> bool:
        return False


class PythonUpdater(BaseUpdater):
    """Automatic in-place pip upgrade backend preserving existing wheel updates."""

    action_label = "Update and Restart"
    action_tooltip = "Download verified wheel, upgrade via pip, and relaunch SPECTROview."

    def select_asset(self, release: ReleaseInfo) -> Optional[ReleaseAsset]:
        for asset in release.assets:
            if asset.name.startswith("spectroview-") and asset.name.endswith(".whl"):
                return asset
        return None

    def can_auto_install(self, release: Optional[ReleaseInfo] = None) -> bool:
        return True if release is None else self.select_asset(release) is not None

    def download_update(
        self,
        release: ReleaseInfo,
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> Path:
        asset = self.select_asset(release)
        if asset is None:
            raise ValueError(f"Release {release.tag} does not include an installable Python wheel.")
        wheel_name = _wheel_filename_from_url(asset.url)
        dest_dir = Path(tempfile.mkdtemp(prefix="spectroview-update-"))
        wheel_path = dest_dir / wheel_name
        try:
            self.client.download_file(
                asset.url,
                wheel_path,
                expected_sha256=asset.sha256,
                progress_cb=progress_cb,
            )
            if not zipfile.is_zipfile(wheel_path):
                raise ValueError("Downloaded file is not a valid Python wheel.")
            return wheel_path
        except Exception:
            wheel_path.unlink(missing_ok=True)
            try:
                dest_dir.rmdir()
            except OSError:
                pass
            raise

    def install_update(self, downloaded_path: Path, release: Optional[ReleaseInfo] = None) -> InstallResult:
        install_update_and_restart(downloaded_path)
        return InstallResult(
            status=UpdateStatus.RESTART_REQUIRED,
            success=True,
            restart_required=True,
            artifact_path=str(downloaded_path),
            message="Wheel downloaded and verified. SPECTROview will close to complete the upgrade.",
        )


class WindowsVelopackUpdater(BaseUpdater):
    """Windows Velopack backend using Python hardened-TLS feed + local-dir UpdateManager."""

    action_label = "Update and Restart"
    action_tooltip = "Download delta or full Velopack packages via verified HTTPS and apply on restart."

    def __init__(self, info: InstallationInfo, client: GitHubReleaseClient) -> None:
        super().__init__(info, client)
        self._pending_manager: Any = None
        self._pending_update_info: Any = None
        self._fallback_setup: bool = False

    def _find_feed_asset(self, release: ReleaseInfo) -> Optional[ReleaseAsset]:
        for name in ("releases.win.json", "releases.win-x64.json"):
            asset = release.asset_by_name(name)
            if asset is not None:
                return asset
        for asset in release.assets:
            low = asset.name.lower()
            if low.startswith("releases.win") and low.endswith(".json"):
                return asset
        return None

    def _find_setup_asset(self, release: ReleaseInfo) -> Optional[ReleaseAsset]:
        for asset in release.assets:
            low = asset.name.lower()
            if "windows" in low and low.endswith("setup.exe"):
                return asset
        for asset in release.assets:
            if asset.name.lower().endswith("-setup.exe"):
                return asset
        return None

    def select_asset(self, release: ReleaseInfo) -> Optional[ReleaseAsset]:
        return self._find_feed_asset(release) or self._find_setup_asset(release)

    def can_auto_install(self, release: Optional[ReleaseInfo] = None) -> bool:
        try:
            import velopack  # noqa: F401
        except ImportError:
            return False
        if release is None:
            return True
        return self._find_feed_asset(release) is not None

    def download_update(
        self,
        release: ReleaseInfo,
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> Path:
        self._pending_manager = None
        self._pending_update_info = None
        self._fallback_setup = False

        feed_asset = self._find_feed_asset(release)
        try:
            import velopack
        except ImportError:
            velopack = None  # type: ignore[assignment]

        if feed_asset is None or velopack is None:
            setup_asset = self._find_setup_asset(release)
            if setup_asset is None:
                raise ValueError(f"Release {release.tag} has no Windows installer asset.")
            self._fallback_setup = True
            dest = resolve_downloads_dir() / (_asset_filename_from_url(setup_asset.url) or setup_asset.name)
            return self.client.download_file(
                setup_asset.url, dest, expected_sha256=setup_asset.sha256, progress_cb=progress_cb
            )

        feed_dir = Path(tempfile.mkdtemp(prefix="spectroview-vpk-"))
        feed_file = feed_dir / feed_asset.name
        try:
            self.client.download_file(feed_asset.url, feed_file, expected_sha256=feed_asset.sha256)

            # Mirror between releases.win.json and releases.win-x64.json so
            # Velopack's local-directory feed lookup succeeds regardless of channel.
            for alias in ("releases.win.json", "releases.win-x64.json"):
                alias_path = feed_dir / alias
                if not alias_path.exists() and feed_file.exists():
                    shutil.copy2(feed_file, alias_path)

            manager = velopack.UpdateManager(str(feed_dir))
            update_info = manager.check_for_updates()
            if update_info is None:
                raise ValueError(
                    f"Velopack update feed in {release.tag} does not list an update newer than "
                    f"{manager.get_current_version()}."
                )

            # Decide delta sequence vs full package strictly from verified GitHub assets
            deltas = list(getattr(update_info, "DeltasToTarget", []) or [])
            packages_to_fetch = deltas
            if not packages_to_fetch or any(release.asset_by_name(d.FileName) is None for d in packages_to_fetch):
                target_full = update_info.TargetFullRelease
                packages_to_fetch = [target_full]
                # Clear stale deltas so Velopack uses the downloaded full package cleanly
                try:
                    update_info.DeltasToTarget = []
                except Exception:
                    pass

            total_count = len(packages_to_fetch)
            for idx, vpk_asset in enumerate(packages_to_fetch):
                gh_asset = release.asset_by_name(vpk_asset.FileName)
                if gh_asset is None:
                    raise ValueError(
                        f"Velopack package '{vpk_asset.FileName}' referenced in feed was not found in release {release.tag}."
                    )
                expected_sha = _normalise_sha256(gh_asset.sha256 or getattr(vpk_asset, "SHA256", ""))

                def _scaled_progress(pct: int, index: int = idx, count: int = total_count) -> None:
                    if progress_cb is not None and count > 0:
                        overall = int(((index + (pct / 100.0)) / count) * 90)
                        progress_cb(min(90, overall))

                self.client.download_file(
                    gh_asset.url,
                    feed_dir / vpk_asset.FileName,
                    expected_sha256=expected_sha,
                    progress_cb=_scaled_progress,
                )

            def _stage_progress(pct: int) -> None:
                if progress_cb is not None:
                    progress_cb(min(100, 90 + int(pct * 0.1)))

            manager.download_updates(update_info, _stage_progress)
            if progress_cb is not None:
                progress_cb(100)

            self._pending_manager = manager
            self._pending_update_info = update_info
            return feed_dir
        except Exception:
            shutil.rmtree(feed_dir, ignore_errors=True)
            raise

    def install_update(self, downloaded_path: Path, release: Optional[ReleaseInfo] = None) -> InstallResult:
        if self._fallback_setup or self._pending_manager is None or self._pending_update_info is None:
            return InstallResult(
                status=UpdateStatus.GUIDED_ACTION_REQUIRED,
                success=True,
                requires_manual_action=True,
                artifact_path=str(downloaded_path),
                instructions=(
                    f"The verified Windows installer was downloaded to:\n{downloaded_path}\n\n"
                    "Run Setup.exe to update SPECTROview. Your settings and workspace folders will be preserved."
                ),
            )
        return InstallResult(
            status=UpdateStatus.RESTART_REQUIRED,
            success=True,
            restart_required=True,
            artifact_path=str(downloaded_path),
            message="Update staged cleanly via Velopack. SPECTROview will restart to finish applying it.",
        )

    def restart_application(self) -> bool:
        if self._pending_manager is not None and self._pending_update_info is not None:
            manager = self._pending_manager
            update_info = self._pending_update_info
            self._pending_manager = None
            self._pending_update_info = None
            manager.wait_exit_then_apply_updates(update_info, silent=False, restart=True)
            return True
        return False


class WindowsPortableUpdater(BaseUpdater):
    """Windows Portable ZIP backend supporting verified download and safe extraction."""

    action_label = "Download Update"
    action_tooltip = "Download and verify the latest Windows Portable ZIP archive."

    def select_asset(self, release: ReleaseInfo) -> Optional[ReleaseAsset]:
        for asset in release.assets:
            low = asset.name.lower()
            if "windows" in low and "portable" in low and low.endswith(".zip"):
                return asset
        for asset in release.assets:
            if asset.name.lower().endswith("portable.zip"):
                return asset
        return None

    def download_update(
        self,
        release: ReleaseInfo,
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> Path:
        zip_path = super().download_update(release, progress_cb=progress_cb)
        if not zipfile.is_zipfile(zip_path):
            zip_path.unlink(missing_ok=True)
            raise ValueError("Downloaded Windows Portable archive is not a valid ZIP file.")
        return zip_path

    @staticmethod
    def extract_portable_archive(zip_path: Path, destination_dir: Path) -> Path:
        """Safely extract *zip_path* into *destination_dir*, guarding against zip-slip."""
        dest_root = destination_dir.resolve()
        dest_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                target = (dest_root / member.filename).resolve()
                if not target.is_relative_to(dest_root):
                    raise ValueError(f"Blocked unsafe archive path outside destination: {member.filename}")
            zf.extractall(dest_root)
        return dest_root

    def install_update(self, downloaded_path: Path, release: Optional[ReleaseInfo] = None) -> InstallResult:
        return InstallResult(
            status=UpdateStatus.GUIDED_ACTION_REQUIRED,
            success=True,
            requires_manual_action=True,
            artifact_path=str(downloaded_path),
            can_extract_zip=True,
            instructions=(
                f"Verified Portable ZIP saved to:\n{downloaded_path}\n\n"
                "Portable builds do not overwrite a running directory automatically.\n"
                "You can extract it into a fresh folder right now or open it in Explorer.\n"
                "Your preferences and Working Folder settings remain preserved."
            ),
        )


class MacOSUpdater(BaseUpdater):
    """macOS ``.app`` bundle updater downloading verified Apple Silicon DMGs."""

    action_label = "Download Update"
    action_tooltip = "Download and verify the latest macOS (Apple Silicon arm64) .dmg disk image."

    def select_asset(self, release: ReleaseInfo) -> Optional[ReleaseAsset]:
        arch = self.info.architecture
        dmg_assets = [a for a in release.assets if a.name.lower().endswith(".dmg")]
        if arch != "arm64":
            # Intel macOS builds are discontinued; avoid serving an incompatible arm64 DMG
            # unless a release explicitly carries a matching universal or x64 disk image.
            for asset in dmg_assets:
                if "universal" in asset.name.lower():
                    return asset
            for asset in dmg_assets:
                low = asset.name.lower()
                if any(tok in low for tok in ("x64", "x86_64", "intel")):
                    return asset
            return None

        for asset in dmg_assets:
            low = asset.name.lower()
            if any(tok in low for tok in ("arm64", "apple_silicon")):
                return asset
        for asset in dmg_assets:
            if "universal" in asset.name.lower():
                return asset
        # Fallback only for untagged DMGs, never Intel-only DMGs
        for asset in dmg_assets:
            low = asset.name.lower()
            if not any(tok in low for tok in ("x64", "x86_64", "intel")):
                return asset
        return None

    def install_update(self, downloaded_path: Path, release: Optional[ReleaseInfo] = None) -> InstallResult:
        return InstallResult(
            status=UpdateStatus.GUIDED_ACTION_REQUIRED,
            success=True,
            requires_manual_action=True,
            artifact_path=str(downloaded_path),
            instructions=(
                f"Verified macOS disk image saved to:\n{downloaded_path}\n\n"
                "To update SPECTROview:\n"
                "1. Open the downloaded .dmg and drag SPECTROview.app into Applications (replace existing).\n"
                "2. Unsigned build note: on first launch, Control-click SPECTROview.app -> Open "
                "(or allow it in System Settings -> Privacy & Security).\n\n"
                "Your preferences and Working Folder are preserved automatically."
            ),
        )


class LinuxAppImageUpdater(BaseUpdater):
    """Linux AppImage updater supporting atomic replacement when ``$APPIMAGE`` is writable."""

    action_label = "Update and Restart"
    action_tooltip = "Download verified AppImage and swap/relaunch cleanly."

    def __init__(self, info: InstallationInfo, client: GitHubReleaseClient) -> None:
        super().__init__(info, client)
        self._pending_swap: Optional[tuple[Path, Path]] = None
        if not info.auto_update_supported:
            self.action_label = "Download Update"
            self.action_tooltip = "Download the latest verified AppImage to your Downloads folder."

    def select_asset(self, release: ReleaseInfo) -> Optional[ReleaseAsset]:
        for asset in release.assets:
            low = asset.name.lower()
            if low.endswith(".appimage") and ("x86_64" in low or "x64" in low or "amd64" in low):
                return asset
        for asset in release.assets:
            if asset.name.lower().endswith(".appimage"):
                return asset
        return None

    def can_auto_install(self, release: Optional[ReleaseInfo] = None) -> bool:
        if not self.info.auto_update_supported or not self.info.appimage_path:
            return False
        return True if release is None else self.select_asset(release) is not None

    def download_update(
        self,
        release: ReleaseInfo,
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> Path:
        asset = self.select_asset(release)
        if asset is None:
            raise ValueError(f"Release {release.tag} does not attach a Linux AppImage asset.")
        filename = _asset_filename_from_url(asset.url) or asset.name
        if self.can_auto_install(release):
            target_dir = Path(tempfile.mkdtemp(prefix="spectroview-appimage-"))
        else:
            target_dir = resolve_downloads_dir()
        dest = target_dir / filename
        self.client.download_file(asset.url, dest, expected_sha256=asset.sha256, progress_cb=progress_cb)

        header = dest.read_bytes()[:4]
        if header != b"\x7fELF":
            dest.unlink(missing_ok=True)
            raise ValueError("Downloaded file is not a valid Linux ELF/AppImage binary.")

        current_mode = dest.stat().st_mode
        os.chmod(dest, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return dest

    @staticmethod
    def _swap_script_content(new_appimage: Path, target_appimage: Path, wait_pid: int) -> str:
        new_q = shlex.quote(new_appimage.as_posix())
        target_q = shlex.quote(target_appimage.as_posix())
        staging_q = shlex.quote(new_appimage.parent.as_posix())
        return (
            "#!/bin/sh\n"
            f"PID={int(wait_pid)}\n"
            "while kill -0 \"$PID\" 2>/dev/null; do sleep 0.2; done\n"
            f"mv -f {new_q} {target_q}\n"
            f"chmod +x {target_q}\n"
            f"rm -rf {staging_q}\n"
            f"nohup {target_q} >/dev/null 2>&1 &\n"
        )

    def install_update(self, downloaded_path: Path, release: Optional[ReleaseInfo] = None) -> InstallResult:
        target = Path(self.info.appimage_path) if self.info.appimage_path else None
        if target is not None and self.info.auto_update_supported and target.exists():
            self._pending_swap = (downloaded_path, target)
            return InstallResult(
                status=UpdateStatus.RESTART_REQUIRED,
                success=True,
                restart_required=True,
                artifact_path=str(downloaded_path),
                message="Verified AppImage ready. SPECTROview will replace the AppImage on exit and relaunch.",
            )

        return InstallResult(
            status=UpdateStatus.GUIDED_ACTION_REQUIRED,
            success=True,
            requires_manual_action=True,
            artifact_path=str(downloaded_path),
            command=f'chmod +x "{downloaded_path}" && "{downloaded_path}"',
            instructions=(
                f"Verified AppImage saved and marked executable at:\n{downloaded_path}\n\n"
                "The current AppImage location is read-only, so replace or launch the new AppImage directly."
            ),
        )

    def restart_application(self) -> bool:
        if self._pending_swap is None:
            return False
        new_appimage, target_appimage = self._pending_swap
        self._pending_swap = None
        script_path = new_appimage.parent / "apply_appimage_update.sh"
        script_path.write_text(
            self._swap_script_content(new_appimage, target_appimage, os.getpid()),
            encoding="utf-8",
        )
        os.chmod(script_path, 0o700)
        subprocess.Popen(["/bin/sh", str(script_path)], start_new_session=True)
        return True


class LinuxDebUpdater(BaseUpdater):
    """Linux Debian package backend downloading verified ``.deb`` installers."""

    action_label = "Download Update"
    action_tooltip = "Download and verify the latest Debian .deb package."

    def select_asset(self, release: ReleaseInfo) -> Optional[ReleaseAsset]:
        for asset in release.assets:
            low = asset.name.lower()
            if low.endswith(".deb") and ("amd64" in low or "x86_64" in low or "x64" in low):
                return asset
        for asset in release.assets:
            if asset.name.lower().endswith(".deb"):
                return asset
        return None

    def download_update(
        self,
        release: ReleaseInfo,
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> Path:
        deb_path = super().download_update(release, progress_cb=progress_cb)
        if deb_path.read_bytes()[:8] != b"!<arch>\n":
            deb_path.unlink(missing_ok=True)
            raise ValueError("Downloaded file is not a valid Debian (.deb) archive.")
        return deb_path

    def install_update(self, downloaded_path: Path, release: Optional[ReleaseInfo] = None) -> InstallResult:
        cmd = f'sudo apt install "{downloaded_path}"'
        return InstallResult(
            status=UpdateStatus.GUIDED_ACTION_REQUIRED,
            success=True,
            requires_manual_action=True,
            artifact_path=str(downloaded_path),
            command=cmd,
            instructions=(
                f"Verified Debian package saved to:\n{downloaded_path}\n\n"
                "Installing system-wide .deb packages requires administrator privileges.\n"
                f"Run in a terminal:\n{cmd}\n\n"
                "Your preferences (~/.config/CEA-Leti/) and Working Folder are preserved."
            ),
        )


class UnsupportedUpdater(BaseUpdater):
    """Read-only version checker used in dev checkouts or unrecognised frozen layouts."""

    action_label = "View Release"
    action_tooltip = "Self-update is disabled for this environment. Open GitHub Releases."

    def install_update(self, downloaded_path: Path, release: Optional[ReleaseInfo] = None) -> InstallResult:
        return InstallResult(
            status=UpdateStatus.UNSUPPORTED,
            success=False,
            requires_manual_action=True,
            instructions=(
                f"SPECTROview is running as {self.info.label}. "
                "Automatic in-app installation is disabled to avoid modifying your development environment."
            ),
        )


class UpdateManager:
    """Centralized multi-platform update facade consumed by UI and ViewModels."""

    def __init__(
        self,
        current_version: str = _CURRENT_VERSION,
        *,
        settings: Any = None,
        install_info: Optional[InstallationInfo] = None,
        client: Optional[GitHubReleaseClient] = None,
    ) -> None:
        self._current_version = str(current_version).lstrip("v").strip()
        self._settings = settings
        self._info = install_info or InstallationManager.detect()
        self._client = client or GitHubReleaseClient(settings=settings)
        self._backend = self._build_backend(self._info, self._client)
        self._status: UpdateStatus = UpdateStatus.IDLE
        self._latest_version: str = ""
        self._last_result: Optional[UpdateCheckResult] = None
        self._last_error: str = ""
        self._downloaded_path: Optional[Path] = None

    @staticmethod
    def _build_backend(info: InstallationInfo, client: GitHubReleaseClient) -> BaseUpdater:
        mapping = {
            InstallType.PIP_WHEEL: PythonUpdater,
            InstallType.WINDOWS_VELOPACK: WindowsVelopackUpdater,
            InstallType.WINDOWS_PORTABLE: WindowsPortableUpdater,
            InstallType.MACOS_BUNDLE: MacOSUpdater,
            InstallType.LINUX_APPIMAGE: LinuxAppImageUpdater,
            InstallType.LINUX_DEB: LinuxDebUpdater,
        }
        backend_cls = mapping.get(info.install_type, UnsupportedUpdater)
        return backend_cls(info, client)

    def get_installation_info(self) -> InstallationInfo:
        return self._info

    def get_backend(self) -> BaseUpdater:
        return self._backend

    def get_latest_version(self) -> str:
        return self._latest_version

    def get_update_status(self) -> UpdateStatus:
        return self._status

    def get_last_error(self) -> str:
        return self._last_error

    def check_for_updates(
        self,
        *,
        force: bool = False,
        include_prereleases: Optional[bool] = None,
    ) -> UpdateCheckResult:
        self._status = UpdateStatus.CHECKING
        self._last_error = ""
        if include_prereleases is None and self._settings is not None:
            getter = getattr(self._settings, "get_update_include_prereleases", None)
            include_prereleases = bool(getter()) if callable(getter) else False

        try:
            release, from_cache = self._client.fetch_latest_release(
                include_prereleases=bool(include_prereleases),
                force=force,
            )
        except Exception as exc:
            self._status = UpdateStatus.FAILED
            self._last_error = str(exc)
            raise

        if release is None:
            self._status = UpdateStatus.UP_TO_DATE
            self._latest_version = self._current_version
            result = UpdateCheckResult(
                status=UpdateStatus.UP_TO_DATE,
                current_version=self._current_version,
                latest_version=self._current_version,
                update_available=False,
                from_cache=from_cache,
            )
            self._last_result = result
            return result

        self._latest_version = release.version
        newer = is_newer_version(release.tag, self._current_version)
        asset = self._backend.select_asset(release)
        can_auto = self._backend.can_auto_install(release)

        action_label = self._backend.action_label
        action_tooltip = self._backend.action_tooltip
        message = "" if asset is not None else "No matching binary asset attached; open GitHub Releases."
        if isinstance(self._backend, WindowsVelopackUpdater) and not can_auto and asset is not None:
            action_label = "Download Installer"
            action_tooltip = "Download verified Windows Setup.exe installer."
        elif isinstance(self._backend, MacOSUpdater) and asset is None and self._info.architecture != "arm64":
            message = "Intel macOS builds are no longer supported (Apple Silicon arm64 only); open GitHub Releases."
            action_tooltip = message

        status = UpdateStatus.UPDATE_AVAILABLE if newer else UpdateStatus.UP_TO_DATE
        self._status = status
        result = UpdateCheckResult(
            status=status,
            current_version=self._current_version,
            latest_version=release.version,
            update_available=newer,
            release=release,
            selected_asset=asset,
            can_auto_install=can_auto,
            action_label=action_label,
            action_tooltip=action_tooltip,
            message=message,
            from_cache=from_cache,
        )
        self._last_result = result
        return result

    def download_update(
        self,
        release: Optional[ReleaseInfo] = None,
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> Path:
        target_release = release or (self._last_result.release if self._last_result else None)
        if target_release is None:
            raise ValueError("No release selected to download.")
        self._status = UpdateStatus.DOWNLOADING
        self._last_error = ""
        try:
            path = self._backend.download_update(target_release, progress_cb=progress_cb)
        except Exception as exc:
            self._status = UpdateStatus.FAILED
            self._last_error = str(exc)
            raise
        self._downloaded_path = path
        self._status = UpdateStatus.READY_TO_INSTALL
        return path

    def install_update(
        self,
        downloaded_path: Optional[Path | str] = None,
        release: Optional[ReleaseInfo] = None,
    ) -> InstallResult:
        target_path = Path(downloaded_path) if downloaded_path is not None else self._downloaded_path
        if target_path is None:
            raise ValueError("No downloaded update artifact is ready to install.")
        target_release = release or (self._last_result.release if self._last_result else None)
        self._status = UpdateStatus.INSTALLING
        self._last_error = ""
        try:
            res = self._backend.install_update(target_path, release=target_release)
        except Exception as exc:
            self._status = UpdateStatus.FAILED
            self._last_error = str(exc)
            raise
        self._status = res.status
        return res

    def restart_application(self) -> bool:
        return self._backend.restart_application()


class UnifiedUpdateCheckWorker(QThread):
    """Background worker executing :meth:`UpdateManager.check_for_updates` cleanly."""

    check_succeeded = Signal(object)
    check_failed = Signal(str)
    check_finished = Signal()

    def __init__(
        self,
        update_manager: UpdateManager,
        *,
        force: bool = False,
        include_prereleases: Optional[bool] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._manager = update_manager
        self._force = force
        self._include_prereleases = include_prereleases

    def run(self) -> None:
        try:
            result = self._manager.check_for_updates(
                force=self._force,
                include_prereleases=self._include_prereleases,
            )
            self.check_succeeded.emit(result)
        except Exception as exc:
            log.debug("Unified update check failed: %s", exc)
            self.check_failed.emit(str(exc))
        finally:
            self.check_finished.emit()


class UnifiedUpdateDownloadWorker(QThread):
    """Background worker streaming & verifying platform update downloads."""

    progress_changed = Signal(int)
    download_finished = Signal(str)
    download_failed = Signal(str)

    def __init__(
        self,
        update_manager: UpdateManager,
        release: ReleaseInfo,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._manager = update_manager
        self._release = release

    def run(self) -> None:
        try:
            path = self._manager.download_update(
                release=self._release,
                progress_cb=lambda pct: self.progress_changed.emit(int(pct)),
            )
            self.download_finished.emit(str(path))
        except Exception as exc:
            self.download_failed.emit(str(exc))
