# spectroview/settings/build_info.py
"""Build environment capture and runtime dependency resolution.

Provides utilities to:
1. Snapshot dependencies and build-environment metadata at PyInstaller packaging time
   into ``spectroview/resources/build_info.json``.
2. Serve build-time package versions when executing from a standalone frozen executable (.exe).
3. Query the active host environment dynamically when running as an installed Python package
   (.whl / pip) or source development workspace.
"""

from __future__ import annotations

import datetime
from functools import lru_cache
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Optional

from spectroview import BUILD_NUMBER, resource_path

TRACKED_PACKAGES: tuple[str, ...] = (
    "PySide6",
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "pybaselines",
    "openpyxl",
    "superqt",
    "pyarrow",
    "pyqt-toast-notification",
    "openai",
    "anthropic",
    "ollama",
    "mcp",
    "pyyaml",
    "markdown",
    "renishawWiRE",
    "pywin32",
)

BUILD_INFO_FILENAME = "build_info.json"


def get_build_info_path() -> Path:
    """Return the filesystem path to the bundled or target build_info.json."""
    return Path(resource_path(f"spectroview/resources/{BUILD_INFO_FILENAME}"))


def is_frozen_app() -> bool:
    """Return True if running inside a frozen standalone bundle (PyInstaller)."""
    return bool(getattr(sys, "frozen", False))


def get_build_timestamp() -> str:
    """Return build timestamp in standard format: YYYY-MM-DD HH:MM:SS."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def query_package_version(pkg: str) -> str:
    """Query a single package version via metadata with fallback to __version__."""
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        pass
    except Exception:
        pass

    # Fallback to sys.modules if already imported
    mod = sys.modules.get(pkg)
    if mod is not None and hasattr(mod, "__version__"):
        try:
            return str(mod.__version__)
        except Exception:
            pass

    return "Not installed"


def generate_build_snapshot(target_path: Path | str | None = None) -> dict[str, Any]:
    """Capture current environment dependencies and metadata to a JSON file.

    Executed during PyInstaller packaging (main.spec) to freeze build machine
    dependency versions into the application resources.
    """
    timestamp = get_build_timestamp()
    arch = platform.architecture()[0]
    os_info = f"{platform.system()} {platform.release()}".strip()
    py_ver = platform.python_version()

    packages: dict[str, str] = {}
    for pkg in TRACKED_PACKAGES:
        packages[pkg] = query_package_version(pkg)

    data: dict[str, Any] = {
        "build_mode": "standalone_exe",
        "build_number": timestamp,
        "build_timestamp": timestamp,
        "python_version": py_ver,
        "python_architecture": arch,
        "build_os": os_info,
        "packages": packages,
    }

    out_file = Path(target_path) if target_path else get_build_info_path()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def load_build_snapshot(path: Path | str | None = None) -> Optional[dict[str, Any]]:
    """Load the build metadata snapshot from disk if present."""
    target = Path(path) if path else get_build_info_path()
    if not target.is_file():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_build_number() -> str:
    """Return the build identifier (timestamp or Dev)."""
    if is_frozen_app():
        snapshot = load_build_snapshot()
        if snapshot and snapshot.get("build_number"):
            return str(snapshot["build_number"])
    return BUILD_NUMBER


@lru_cache(maxsize=1)
def get_runtime_package_versions() -> dict[str, str]:
    """Retrieve package versions appropriate for the current execution mode.

    - When running as a standalone executable (PyInstaller frozen):
      Loads the build snapshot recorded on the packaging PC.
    - When running as an installed package (.whl / pip) or dev repo:
      Queries the host environment dynamically via importlib.metadata.
    """
    qt_ver_str = ""
    try:
        from PySide6.QtCore import qVersion
        qt_ver_str = f" (Qt {qVersion()})"
    except Exception:
        pass

    versions: dict[str, str] = {}

    # Mode 1: Frozen Executable (.exe)
    if is_frozen_app():
        snapshot = load_build_snapshot()
        if snapshot and isinstance(snapshot.get("packages"), dict):
            pkg_snapshot: dict[str, str] = snapshot["packages"]
            for pkg in TRACKED_PACKAGES:
                ver = pkg_snapshot.get(pkg, "Not installed")
                if pkg == "PySide6" and ver != "Not installed" and qt_ver_str and "Qt" not in ver:
                    ver = f"{ver}{qt_ver_str}"
                versions[pkg] = ver
            return versions

    # Mode 2: Dynamic Host Lookup (.whl or Dev)
    for pkg in TRACKED_PACKAGES:
        ver = query_package_version(pkg)
        if pkg == "PySide6" and ver != "Not installed" and qt_ver_str and "Qt" not in ver:
            ver = f"{ver}{qt_ver_str}"
        versions[pkg] = ver

    return versions


def get_environment_summary() -> dict[str, Any]:
    """Return a comprehensive summary of execution mode, OS, Python, and dependencies."""
    frozen = is_frozen_app()
    arch = platform.architecture()[0]
    host_os = f"{platform.system()} {platform.release()}".strip()
    py_ver = platform.python_version()

    build_os: Optional[str] = None
    build_py: Optional[str] = None
    build_arch: Optional[str] = None
    build_num: str = BUILD_NUMBER

    if frozen:
        snapshot = load_build_snapshot()
        if snapshot:
            build_os = snapshot.get("build_os")
            build_py = snapshot.get("python_version", py_ver)
            build_arch = snapshot.get("python_architecture", arch)
            build_num = snapshot.get("build_number", BUILD_NUMBER)
        mode = "Standalone Executable (.exe)"
    else:
        mode = "Python Package (.whl / local)"

    return {
        "is_frozen": frozen,
        "mode": mode,
        "python_version": py_ver,
        "python_architecture": arch,
        "host_os": host_os,
        "build_os": build_os,
        "build_python": build_py,
        "build_architecture": build_arch,
        "build_number": build_num,
        "packages": get_runtime_package_versions(),
    }

