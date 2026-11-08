"""Centralized installation & packaging environment detection for SPECTROview.

Reliably classifies runtime environments across source checkouts, pip wheels,
Windows Velopack installers, Windows portable archives, macOS application
bundles, Linux AppImages, and Linux Debian packages without relying solely on
``sys.frozen``.

Pure Model layer: holds zero Qt dependencies and zero heavy scientific imports.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Mapping, Optional


class InstallType(str, Enum):
    """Supported execution / packaging classifications for SPECTROview."""

    DEV_SOURCE = "dev_source"
    PIP_WHEEL = "pip_wheel"
    WINDOWS_VELOPACK = "windows_velopack"
    WINDOWS_PORTABLE = "windows_portable"
    MACOS_BUNDLE = "macos_bundle"
    LINUX_APPIMAGE = "linux_appimage"
    LINUX_DEB = "linux_deb"
    STANDALONE_UNKNOWN = "standalone_unknown"


_LABELS: dict[InstallType, str] = {
    InstallType.DEV_SOURCE: "Development Checkout (Local Source)",
    InstallType.PIP_WHEEL: "Python Package (.whl / pip)",
    InstallType.WINDOWS_VELOPACK: "Windows Installer (Velopack)",
    InstallType.WINDOWS_PORTABLE: "Windows Portable (.zip)",
    InstallType.MACOS_BUNDLE: "macOS Application (.app)",
    InstallType.LINUX_APPIMAGE: "Linux AppImage",
    InstallType.LINUX_DEB: "Linux Debian Package (.deb)",
    InstallType.STANDALONE_UNKNOWN: "Standalone Executable",
}

_UPDATE_BACKENDS: dict[InstallType, str] = {
    InstallType.DEV_SOURCE: "unsupported",
    InstallType.PIP_WHEEL: "pip_auto",
    InstallType.WINDOWS_VELOPACK: "velopack_auto",
    InstallType.WINDOWS_PORTABLE: "windows_portable_guided",
    InstallType.MACOS_BUNDLE: "macos_dmg_guided",
    InstallType.LINUX_APPIMAGE: "linux_appimage",
    InstallType.LINUX_DEB: "linux_deb_guided",
    InstallType.STANDALONE_UNKNOWN: "unsupported",
}


@dataclass(frozen=True)
class InstallationInfo:
    """Immutable snapshot describing how SPECTROview is currently installed."""

    install_type: InstallType
    platform_name: str
    architecture: str
    executable: str
    app_root: str
    is_frozen: bool
    is_portable: bool
    is_dev: bool
    update_supported: bool
    auto_update_supported: bool
    update_backend: str
    label: str
    appimage_path: str = ""
    details: str = ""


def _normalise_arch(machine: str) -> str:
    m = str(machine or "").strip().lower()
    if m in ("amd64", "x86_64", "x64"):
        return "x64"
    if m in ("arm64", "aarch64"):
        return "arm64"
    if m in ("i386", "i686", "x86"):
        return "x86"
    return m or "unknown"


def _normalise_platform(plat: str) -> str:
    p = str(plat or "").strip().lower()
    if p.startswith("win"):
        return "win32"
    if p == "darwin" or "mac" in p:
        return "darwin"
    if p.startswith("linux"):
        return "linux"
    return p


def _path_exists_file(path_obj: Path) -> bool:
    try:
        return path_obj.is_file()
    except OSError:
        return False


def _is_editable_or_checkout(pkg_root: Path, check_dist_metadata: bool = True) -> tuple[bool, str]:
    """Return ``(is_dev, reason)`` for an unfrozen Python runtime."""
    repo_root = pkg_root.parent
    try:
        if (repo_root / ".git").exists():
            return True, "git repository root"
        if (repo_root / "pyproject.toml").is_file() and (repo_root / "setup.py").is_file():
            return True, "source checkout (pyproject.toml)"
    except OSError:
        pass

    if check_dist_metadata:
        try:
            dist = importlib.metadata.distribution("spectroview")
            direct_url_raw = dist.read_text("direct_url.json")
            if direct_url_raw:
                direct_url = json.loads(direct_url_raw)
                if isinstance(direct_url, dict):
                    dir_info = direct_url.get("dir_info")
                    if isinstance(dir_info, dict) and dir_info.get("editable") is True:
                        return True, "editable pip install"
        except Exception:
            pass

    return False, "installed Python package"


def _detect_windows_frozen(exe_str: str) -> tuple[InstallType, str, str]:
    """Classify a frozen Windows executable as Velopack-managed or Portable."""
    exe_pure = PureWindowsPath(exe_str)
    exe_fs = Path(exe_str)
    parent = exe_pure.parent
    root_candidate = parent.parent

    # Canonical Velopack Windows layout: <AppRoot>\current\SPECTROview.exe with
    # <AppRoot>\Update.exe and/or <AppRoot>\current\sq.version
    if parent.name.lower() == "current":
        update_exe = Path(str(root_candidate / "Update.exe"))
        sq_version = Path(str(Path(exe_str).parent / "sq.version"))
        if _path_exists_file(update_exe) or _path_exists_file(sq_version):
            return (
                InstallType.WINDOWS_VELOPACK,
                str(root_candidate),
                "Velopack managed directory (Update.exe / sq.version)",
            )

    # Secondary check for sibling sq.version / Update.exe directly beside exe
    sibling_update = Path(str(exe_fs.parent / "Update.exe"))
    sibling_sq = Path(str(exe_fs.parent / "sq.version"))
    if _path_exists_file(sibling_sq) and _path_exists_file(sibling_update):
        return (
            InstallType.WINDOWS_VELOPACK,
            str(parent),
            "Velopack installation markers present",
        )

    return (
        InstallType.WINDOWS_PORTABLE,
        str(parent),
        "Standalone Windows portable directory",
    )


def _detect_macos_frozen(exe_str: str) -> tuple[InstallType, str, str]:
    """Classify a frozen macOS executable inside an ``.app`` bundle."""
    exe_pure = PurePosixPath(exe_str.replace("\\", "/"))
    parts_lower = [p.lower() for p in exe_pure.parts]
    for idx, part in enumerate(parts_lower):
        if part.endswith(".app"):
            app_root = str(PurePosixPath(*exe_pure.parts[: idx + 1]))
            # Verify bundle shape: <Name>.app/Contents/MacOS/<exe>
            if idx + 2 < len(parts_lower) and parts_lower[idx + 1] == "contents" and parts_lower[idx + 2] == "macos":
                return (
                    InstallType.MACOS_BUNDLE,
                    app_root,
                    "macOS .app application bundle",
                )
            plist = Path(app_root) / "Contents" / "Info.plist"
            if _path_exists_file(plist):
                return (
                    InstallType.MACOS_BUNDLE,
                    app_root,
                    "macOS .app bundle (Info.plist verified)",
                )
    return (
        InstallType.STANDALONE_UNKNOWN,
        str(exe_pure.parent),
        "Frozen macOS binary outside .app bundle",
    )


def _detect_linux_frozen(
    exe_str: str,
    environ: Mapping[str, str],
) -> tuple[InstallType, str, str, str]:
    """Classify a frozen Linux executable as AppImage, .deb, or unknown standalone."""
    norm_exe = exe_str.replace("\\", "/")
    exe_pure = PurePosixPath(norm_exe)

    appimage_file = str(environ.get("APPIMAGE", "") or "").strip()
    appdir = str(environ.get("APPDIR", "") or "").strip().replace("\\", "/")

    if appimage_file:
        # Ensure we don't inherit an unrelated APPIMAGE variable if APPDIR
        # disagrees cleanly with the frozen executable path.
        if not appdir or norm_exe.startswith(appdir.rstrip("/") + "/") or norm_exe == appdir:
            return (
                InstallType.LINUX_APPIMAGE,
                str(PurePosixPath(appimage_file.replace("\\", "/")).parent),
                "AppImage runtime (APPIMAGE environment detected)",
                appimage_file,
            )

    # Debian package check: marker files or standard .deb installation prefix
    exe_fs = Path(exe_str)
    if _path_exists_file(exe_fs.parent / ".spectroview-deb"):
        return (
            InstallType.LINUX_DEB,
            str(exe_pure.parent),
            "Debian package (.spectroview-deb marker)",
            "",
        )

    deb_prefixes = ("/opt/spectroview/", "/usr/lib/spectroview/", "/usr/local/lib/spectroview/")
    if any(norm_exe.startswith(prefix) for prefix in deb_prefixes):
        return (
            InstallType.LINUX_DEB,
            str(exe_pure.parent),
            f"System Debian installation prefix ({exe_pure.parent})",
            "",
        )

    if appimage_file:
        return (
            InstallType.LINUX_APPIMAGE,
            str(PurePosixPath(appimage_file.replace("\\", "/")).parent),
            "AppImage runtime",
            appimage_file,
        )

    return (
        InstallType.STANDALONE_UNKNOWN,
        str(exe_pure.parent),
        "Standalone Linux executable",
        "",
    )


def _is_path_writable(target: str) -> bool:
    if not target:
        return False
    p = Path(target)
    try:
        if p.exists():
            return os.access(p, os.W_OK)
        return os.access(p.parent, os.W_OK)
    except OSError:
        return False


class InstallationManager:
    """Centralized installation detection API for SPECTROview."""

    @classmethod
    def detect(
        cls,
        *,
        frozen: Optional[bool] = None,
        executable: Optional[str | os.PathLike[str]] = None,
        platform_name: Optional[str] = None,
        machine: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        package_root: Optional[str | os.PathLike[str]] = None,
        check_dist_metadata: Optional[bool] = None,
    ) -> InstallationInfo:
        """Inspect runtime environment markers and return an :class:`InstallationInfo`."""
        is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else bool(frozen)
        plat = _normalise_platform(platform_name or sys.platform)
        arch = _normalise_arch(machine or platform.machine())
        exe_str = str(executable if executable is not None else sys.executable)
        environ = os.environ if env is None else env
        pkg_root = (
            Path(package_root)
            if package_root is not None
            else Path(__file__).resolve().parent.parent
        )
        # Skip live importlib.metadata inspection when caller supplied synthetic package_root
        probe_dist = (package_root is None) if check_dist_metadata is None else bool(check_dist_metadata)

        if not is_frozen:
            is_dev, dev_reason = _is_editable_or_checkout(pkg_root, check_dist_metadata=probe_dist)
            itype = InstallType.DEV_SOURCE if is_dev else InstallType.PIP_WHEEL
            app_root = str(pkg_root.parent)
            backend = _UPDATE_BACKENDS[itype]
            return InstallationInfo(
                install_type=itype,
                platform_name=plat,
                architecture=arch,
                executable=exe_str,
                app_root=app_root,
                is_frozen=False,
                is_portable=False,
                is_dev=is_dev,
                update_supported=(itype == InstallType.PIP_WHEEL),
                auto_update_supported=(itype == InstallType.PIP_WHEEL),
                update_backend=backend,
                label=_LABELS[itype],
                details=dev_reason,
            )

        # Frozen standalone distribution
        appimage_path = ""
        if plat == "win32":
            itype, app_root, details = _detect_windows_frozen(exe_str)
        elif plat == "darwin":
            itype, app_root, details = _detect_macos_frozen(exe_str)
        elif plat == "linux":
            itype, app_root, details, appimage_path = _detect_linux_frozen(exe_str, environ)
        else:
            itype = InstallType.STANDALONE_UNKNOWN
            app_root = str(Path(exe_str).parent)
            details = f"Frozen standalone executable on {plat}"

        is_portable = itype in (InstallType.WINDOWS_PORTABLE, InstallType.LINUX_APPIMAGE)
        if itype == InstallType.WINDOWS_VELOPACK:
            auto_update = True
        elif itype == InstallType.LINUX_APPIMAGE:
            auto_update = _is_path_writable(appimage_path)
        else:
            auto_update = False

        update_supported = itype in (
            InstallType.WINDOWS_VELOPACK,
            InstallType.WINDOWS_PORTABLE,
            InstallType.MACOS_BUNDLE,
            InstallType.LINUX_APPIMAGE,
            InstallType.LINUX_DEB,
        )

        return InstallationInfo(
            install_type=itype,
            platform_name=plat,
            architecture=arch,
            executable=exe_str,
            app_root=app_root,
            is_frozen=True,
            is_portable=is_portable,
            is_dev=False,
            update_supported=update_supported,
            auto_update_supported=auto_update,
            update_backend=_UPDATE_BACKENDS[itype],
            label=_LABELS[itype],
            appimage_path=appimage_path,
            details=details,
        )

    @classmethod
    def get_installation_type(cls, **kwargs) -> InstallType:
        """Return the classified :class:`InstallType` for the current environment."""
        return cls.detect(**kwargs).install_type

    @classmethod
    def get_installation_label(cls, **kwargs) -> str:
        """Return a human-readable installation type label."""
        return cls.detect(**kwargs).label

    @classmethod
    def is_portable(cls, **kwargs) -> bool:
        """Return ``True`` when running as a portable Windows ZIP or Linux AppImage."""
        return cls.detect(**kwargs).is_portable

    @classmethod
    def is_frozen(cls, **kwargs) -> bool:
        """Return ``True`` for frozen standalone builds."""
        return cls.detect(**kwargs).is_frozen

    @classmethod
    def is_dev(cls, **kwargs) -> bool:
        """Return ``True`` when running from a source checkout or editable install."""
        return cls.detect(**kwargs).is_dev

    @classmethod
    def is_update_supported(cls, **kwargs) -> bool:
        """Return ``True`` when this installation supports integrated update checking/actions."""
        return cls.detect(**kwargs).update_supported

    @classmethod
    def is_auto_update_supported(cls, **kwargs) -> bool:
        """Return ``True`` when unattended in-place replacement + restart is supported."""
        return cls.detect(**kwargs).auto_update_supported

    @classmethod
    def has_velopack(cls, **kwargs) -> bool:
        """Return ``True`` when running under a Velopack-managed installation."""
        return cls.detect(**kwargs).install_type == InstallType.WINDOWS_VELOPACK

    @classmethod
    def should_run_velopack_hook(cls, argv: Optional[list[str]] = None, **kwargs) -> bool:
        """Return ``True`` if Velopack's startup hook should execute before QApplication.

        Runs when Velopack installer flags (``--veloapp-*``) are passed to a
        frozen binary, or when running as an installed Velopack application.
        Always returns ``False`` for unfrozen dev / pip environments so startup
        time is unaffected.
        """
        info = cls.detect(**kwargs)
        if not info.is_frozen:
            return False
        args = sys.argv[1:] if argv is None else argv
        if any(str(arg).startswith("--veloapp-") for arg in args):
            return True
        return info.install_type == InstallType.WINDOWS_VELOPACK

