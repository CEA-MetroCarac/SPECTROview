# spectroview/model/m_update_checker.py
"""
Background update checker that queries the GitHub Releases API.
Runs in a QThread so the UI is never blocked.
Fails silently when offline or rate-limited.
"""
import json
from urllib.request import urlopen, Request
from urllib.error import URLError

from PySide6.QtCore import QThread, Signal


GITHUB_API_URL = (
    "https://api.github.com/repos/CEA-MetroCarac/SPECTROview/releases/latest"
)
# Timeout for the HTTP request (seconds)
REQUEST_TIMEOUT = 5


def _parse_version(version_str: str) -> tuple:
    """
    Convert a version string like '26.28.2' or 'v26.28.2' into a comparable tuple.
    Non-numeric parts are treated as 0 so comparisons never crash.
    """
    cleaned = version_str.lstrip("v").strip()
    parts = []
    for part in cleaned.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    return tuple(parts)


class UpdateCheckerWorker(QThread):
    """
    Worker thread that checks for a newer release on GitHub.

    Signals
    -------
    update_available(tag: str, release_notes: str, html_url: str)
        Emitted when a newer version is found on GitHub.
    check_finished()
        Emitted when the check is done (regardless of outcome).
    """

    update_available = Signal(str, str, str)   # (tag, release_notes, html_url)
    check_finished = Signal()

    def __init__(self, current_version: str, parent=None):
        super().__init__(parent)
        self._current_version = current_version

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------
    def run(self):
        try:
            req = Request(
                GITHUB_API_URL,
                headers={"Accept": "application/vnd.github+json",
                         "User-Agent": "SPECTROview-update-checker"},
            )
            
            context = None
            try:
                import ssl
                import certifi
                context = ssl.create_default_context(cafile=certifi.where())
            except Exception:
                pass
                
            with urlopen(req, timeout=REQUEST_TIMEOUT, context=context) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            tag = data.get("tag_name", "").strip()
            html_url = data.get("html_url", "")
            body = data.get("body", "")   # release notes (Markdown)

            if not tag:
                return

            remote_ver = _parse_version(tag)
            local_ver  = _parse_version(self._current_version)

            if remote_ver > local_ver:
                # Trim release notes to a reasonable length for the banner
                notes_preview = body[:600].strip() if body else ""
                self.update_available.emit(tag, notes_preview, html_url)

        except (URLError, OSError, json.JSONDecodeError, KeyError):
            # Network unavailable, timeout, malformed JSON – just skip silently
            pass
        finally:
            self.check_finished.emit()
