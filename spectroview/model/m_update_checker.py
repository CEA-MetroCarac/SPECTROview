"""GitHub release checking and safe in-place wheel updates.

Network operations run in :class:`QThread` workers so update checks and wheel
downloads never block the SPECTROview interface. Once a wheel is available,
the running application starts a short-lived helper script, closes, and lets
that helper install the wheel before relaunching the installed application.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from PySide6.QtCore import QThread, Signal


GITHUB_API_URL = (
    "https://api.github.com/repos/CEA-MetroCarac/SPECTROview/releases/latest"
)
REQUEST_TIMEOUT = 5
DOWNLOAD_TIMEOUT = 30
DOWNLOAD_CHUNK_SIZE = 64 * 1024

log = logging.getLogger(__name__)


class UpdateInstallationError(RuntimeError):
    """Raised when this SPECTROview installation cannot apply a wheel update."""


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Return a comparable tuple for a version such as ``v26.28.2``."""
    cleaned = version_str.lstrip("v").strip()
    parts = []
    for part in cleaned.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def _ssl_context():
    """Use certifi when available, while keeping the stdlib-only fallback."""
    try:
        import certifi
        import ssl

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # pragma: no cover - exercised on minimal installations
        return None


def _normalise_sha256(digest: str) -> str:
    """Return a GitHub asset digest as a bare SHA-256 hexadecimal string."""
    digest = digest.removeprefix("sha256:").lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        return ""
    return digest


def _find_wheel_asset(release: dict) -> tuple[str, str]:
    """Return the SPECTROview wheel URL and optional SHA-256 for *release*."""
    for asset in release.get("assets", []):
        name = str(asset.get("name", ""))
        url = str(asset.get("browser_download_url", ""))
        if name.startswith("spectroview-") and name.endswith(".whl") and url:
            return url, _normalise_sha256(str(asset.get("digest", "")))
    return "", ""


def _wheel_filename_from_url(wheel_url: str) -> str:
    """Extract and validate the published wheel filename from its release URL.

    ``pip install`` validates a wheel's filename, so the downloaded file must
    retain the PEP 427-style asset name rather than use a random temporary name.
    """
    filename = Path(unquote(urlparse(wheel_url).path)).name
    parts = filename.removesuffix(".whl").split("-")
    if (
        filename != unquote(urlparse(wheel_url).path).rsplit("/", 1)[-1]
        or not filename.endswith(".whl")
        or parts[0] != "spectroview"
        or len(parts) not in (5, 6)
        or any(not part for part in parts)
    ):
        raise ValueError("The release asset does not have a valid SPECTROview wheel filename.")
    return filename


class UpdateCheckerWorker(QThread):
    """Check GitHub Releases for a newer version and its wheel release asset."""

    update_available = Signal(str, str, str, str, str)
    check_finished = Signal()

    def __init__(self, current_version: str, parent=None):
        super().__init__(parent)
        self._current_version = current_version

    def run(self) -> None:
        try:
            request = Request(
                GITHUB_API_URL,
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "SPECTROview-update-checker",
                },
            )
            with urlopen(request, timeout=REQUEST_TIMEOUT, context=_ssl_context()) as response:
                release = json.loads(response.read().decode("utf-8"))

            tag = str(release.get("tag_name", "")).strip()
            if not tag or _parse_version(tag) <= _parse_version(self._current_version):
                return

            wheel_url, wheel_sha256 = _find_wheel_asset(release)
            notes_preview = str(release.get("body", ""))[:600].strip()
            self.update_available.emit(
                tag,
                notes_preview,
                str(release.get("html_url", "")),
                wheel_url,
                wheel_sha256,
            )
        except (URLError, OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            # An update check is strictly best-effort; users can continue working offline.
            log.debug("Update check failed: %s", error)
        finally:
            self.check_finished.emit()


class UpdateDownloadWorker(QThread):
    """Download one release wheel to a temporary file without blocking the UI."""

    download_finished = Signal(str)
    download_failed = Signal(str)
    progress_changed = Signal(int)

    def __init__(self, wheel_url: str, expected_sha256: str = "", parent=None):
        super().__init__(parent)
        self._wheel_url = wheel_url
        self._expected_sha256 = _normalise_sha256(expected_sha256)

    def run(self) -> None:
        wheel_path: Path | None = None
        try:
            wheel_filename = _wheel_filename_from_url(self._wheel_url)
            update_directory = Path(tempfile.mkdtemp(prefix="spectroview-update-"))
            wheel_path = update_directory / wheel_filename
            request = Request(
                self._wheel_url,
                headers={"User-Agent": "SPECTROview-updater"},
            )
            with urlopen(request, timeout=DOWNLOAD_TIMEOUT, context=_ssl_context()) as response:
                total_bytes = int(response.headers.get("Content-Length", 0))
                hasher = hashlib.sha256()
                with wheel_path.open("xb") as output:
                    received_bytes = 0
                    while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                        output.write(chunk)
                        hasher.update(chunk)
                        received_bytes += len(chunk)
                        if total_bytes:
                            self.progress_changed.emit(min(100, received_bytes * 100 // total_bytes))

            if self._expected_sha256 and hasher.hexdigest() != self._expected_sha256:
                raise ValueError("The downloaded update did not match GitHub's SHA-256 checksum.")
            if wheel_path is None or not zipfile.is_zipfile(wheel_path):
                raise ValueError("GitHub did not return a valid Python wheel.")

            self.progress_changed.emit(100)
            self.download_finished.emit(str(wheel_path))
        except Exception as error:  # Network and file errors are shown in the update banner.
            if wheel_path is not None:
                wheel_path.unlink(missing_ok=True)
                try:
                    wheel_path.parent.rmdir()
                except OSError:
                    pass
            self.download_failed.emit(str(error))


def get_update_python_executable() -> Path:
    """Return the interpreter that installed SPECTROview, or explain why it cannot update."""
    if getattr(sys, "frozen", False):
        raise UpdateInstallationError(
            "Automatic updates require a pip-installed SPECTROview. "
            "The standalone executable cannot install a Python wheel."
        )

    interpreter = Path(sys.executable)
    if os.name == "nt" and interpreter.name.lower() == "pythonw.exe":
        console_interpreter = interpreter.with_name("python.exe")
        if console_interpreter.exists():
            interpreter = console_interpreter
    if not interpreter.exists():
        raise UpdateInstallationError("The Python interpreter used by SPECTROview is unavailable.")
    return interpreter


def _restart_interpreter(python_executable: Path) -> Path:
    """Prefer ``pythonw.exe`` on Windows so the relaunched UI has no console."""
    if os.name == "nt":
        pythonw = python_executable.with_name("pythonw.exe")
        if pythonw.exists():
            return pythonw
    return python_executable


def _update_script_content(
    wheel_path: Path,
    python_executable: Path,
    restart_executable: Path,
) -> str:
    """Build the standalone helper script used after the application has closed."""
    return f'''# Generated by SPECTROview; deleted after a successful update.
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

wheel_path = Path({str(wheel_path)!r})
python_executable = Path({str(python_executable)!r})
restart_executable = Path({str(restart_executable)!r})

# Give the old Qt process enough time to release imported package files.
time.sleep(3)
result = subprocess.run([
    str(python_executable), "-m", "pip", "install", "--upgrade", str(wheel_path)
])
if result.returncode:
    print("SPECTROview update failed. The downloaded wheel was kept at:")
    print(wheel_path)
    if os.name == "nt":
        input("Press Enter to close this window.")
    sys.exit(result.returncode)

wheel_path.unlink(missing_ok=True)
try:
    wheel_path.parent.rmdir()
except OSError:
    pass
restart_environment = os.environ.copy()
restart_environment.pop("PYTHONPATH", None)
restart_environment.pop("PYTHONHOME", None)
subprocess.Popen(
    [str(restart_executable), "-m", "spectroview.main"],
    cwd=tempfile.gettempdir(),
    env=restart_environment,
)
Path(__file__).unlink(missing_ok=True)
'''


def install_update_and_restart(wheel_path: Path) -> None:
    """Start a detached helper that installs *wheel_path* and relaunches SPECTROview.

    The caller must close the Qt application immediately after this function
    returns. The helper waits three seconds before running pip so no installed
    package file remains held by the current process.
    """
    python_executable = get_update_python_executable()
    restart_executable = _restart_interpreter(python_executable)
    script_path = Path(tempfile.gettempdir()) / "spectroview_apply_update.py"
    script_path.write_text(
        _update_script_content(wheel_path, python_executable, restart_executable),
        encoding="utf-8",
    )

    popen_kwargs = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    else:
        popen_kwargs["start_new_session"] = True
    subprocess.Popen([str(python_executable), str(script_path)], **popen_kwargs)
