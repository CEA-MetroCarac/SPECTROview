"""
Standalone demo script to preview the VUpdateBanner widget.

Run this directly (no need to launch the full SPECTROview application):

    python tests/test_update_banner_preview.py

It opens a small window showing the banner in BOTH dark and light themes
so you can check the appearance without touching the GitHub API.
"""
import sys
from pathlib import Path

# Make sure the project root is on the path so imports work from the repo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QPushButton, QHBoxLayout, QFrame
)
from PySide6.QtCore import Qt, QTimer

from spectroview.view.components.v_update_banner import VUpdateBanner


# ── Demo window ───────────────────────────────────────────────────────────────
class BannerPreviewWindow(QMainWindow):
    """
    Shows one banner in dark theme and one in light theme,
    plus buttons to simulate the GitHub version-check signal.
    """

    FAKE_TAG = "v99.99.99"
    FAKE_URL = "https://github.com/CEA-MetroCarac/SPECTROview/releases"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VUpdateBanner — Preview / Test")
        self.setMinimumWidth(900)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(20)
        root.setContentsMargins(20, 20, 20, 20)

        # ── Title ──────────────────────────────────────────────────────────
        title = QLabel("Update Banner Preview")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        root.addWidget(title)

        subtitle = QLabel(
            "The banners below mimic exactly what users see when a new release "
            "is detected. The 'Download / Changelog' button opens the real "
            "GitHub releases page."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: gray;")
        root.addWidget(subtitle)

        self._add_separator(root)

        # ── Dark theme banner ──────────────────────────────────────────────
        root.addWidget(QLabel("🌙  Dark theme:"))
        self._dark_banner = self._make_banner("dark")
        root.addWidget(self._dark_banner)

        self._add_separator(root)

        # ── Light theme banner ─────────────────────────────────────────────
        root.addWidget(QLabel("☀️  Light theme:"))
        self._light_banner = self._make_banner("light")
        root.addWidget(self._light_banner)

        self._add_separator(root)

        # ── Status label ───────────────────────────────────────────────────
        self._status = QLabel("Click a banner button above to test its action.")
        self._status.setStyleSheet("color: #1976d2; font-style: italic;")
        root.addWidget(self._status)

        # ── Re-show buttons ────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_dark = QPushButton("Reset dark banner")
        btn_dark.clicked.connect(lambda: self._reset_banner("dark"))
        btn_light = QPushButton("Reset light banner")
        btn_light.clicked.connect(lambda: self._reset_banner("light"))
        btn_row.addWidget(btn_dark)
        btn_row.addWidget(btn_light)
        btn_row.addStretch()
        root.addLayout(btn_row)

        root.addStretch()

    # ------------------------------------------------------------------
    def _make_banner(self, theme: str) -> VUpdateBanner:
        banner = VUpdateBanner(
            tag=self.FAKE_TAG,
            html_url=self.FAKE_URL,
            on_skip=lambda tag: self._log(f"[{theme}] Skipped version: {tag}"),
            on_dismiss=lambda: self._log(f"[{theme}] Banner dismissed"),
        )
        banner.apply_theme(theme)
        return banner

    def _reset_banner(self, theme: str):
        if theme == "dark":
            self._dark_banner.show()
            self._log("Dark banner restored.")
        else:
            self._light_banner.show()
            self._log("Light banner restored.")

    def _log(self, msg: str):
        self._status.setText(msg)

    @staticmethod
    def _add_separator(layout: QVBoxLayout):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)


# ── Also test the real UpdateCheckerWorker ────────────────────────────────────
def _run_real_check():
    """
    Fires an actual GitHub API request and prints the result to stdout.
    Useful to verify the version comparison logic with your real tags.
    """
    from spectroview import VERSION
    from spectroview.model.m_update_checker import UpdateCheckerWorker

    print(f"\n[Real check] Current app version : {VERSION}")
    print(f"[Real check] Querying GitHub API  : https://api.github.com/repos/CEA-MetroCarac/SPECTROview/releases/latest\n")

    app = QApplication.instance() or QApplication(sys.argv)

    worker = UpdateCheckerWorker(current_version=VERSION)

    def on_update(tag, notes, url):
        print(f"  ✅ UPDATE AVAILABLE  →  {tag}")
        print(f"     URL  : {url}")
        preview = notes[:200].replace("\n", " ") if notes else "(no release notes)"
        print(f"     Notes: {preview}")

    def on_done():
        print("  [check_finished] — check complete.")

    worker.update_available.connect(on_update)
    worker.check_finished.connect(on_done)
    worker.start()
    worker.wait(10_000)   # block up to 10 s for the result


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VUpdateBanner preview / test")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run only the real GitHub API check (no GUI) and exit.",
    )
    args = parser.parse_args()

    if args.check_only:
        _run_real_check()
        sys.exit(0)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = BannerPreviewWindow()
    win.show()
    sys.exit(app.exec())
