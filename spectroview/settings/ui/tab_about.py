# spectroview/settings/tab_about.py
"""About tab: application information, versioning, Python environment, runtime dependencies, and citation."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys

from PySide6.QtCore import Qt, Signal, qVersion
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from spectroview import BUILD_NUMBER, ICON_DIR, VERSION
from spectroview.view.theme import Icons


def get_runtime_package_versions() -> dict[str, str]:
    """Retrieve runtime versions of key SPECTROview dependencies via importlib.metadata."""
    tracked_packages = [
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
    ]

    versions: dict[str, str] = {}
    for pkg in tracked_packages:
        try:
            ver = importlib.metadata.version(pkg)
            if pkg == "PySide6":
                ver = f"{ver} (Qt {qVersion()})"
            versions[pkg] = ver
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "Not installed"
        except Exception:
            versions[pkg] = "Unknown"

    return versions


class AboutTabPage(QWidget):
    """The About tab displaying logo, version, Python & package dependencies, and citation."""

    check_updates_requested = Signal()

    KEYWORDS = [
        "about", "version", "build", "python", "environment", "dependency",
        "dependencies", "packages", "pyside6", "qt", "numpy", "scipy",
        "matplotlib", "pandas", "pybaselines", "citation", "documentation",
        "features", "author", "license"
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(14, 12, 14, 16)
        layout.setSpacing(14)

        # ───── Header & Logo ─────
        logo_label = QLabel()
        logo_path = os.path.join(ICON_DIR, "logo_spectroview.png")
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path).scaled(96, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        # Title & Subtitle
        title_label = QLabel("SPECTROview")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title_label)

        subtitle_label = QLabel("Tool for Spectroscopic Data Processing and Visualization")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("font-size: 12px; color: gray; font-style: italic;")
        layout.addWidget(subtitle_label)

        # Version & Build row
        version_row = QHBoxLayout()
        version_row.setAlignment(Qt.AlignCenter)
        version_row.setSpacing(12)

        ver_lbl = QLabel(f"Version {VERSION}")
        ver_lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
        version_row.addWidget(ver_lbl)

        build_lbl = QLabel(f"Build: {BUILD_NUMBER}")
        build_lbl.setStyleSheet("color: #888888; font-size: 12px;")
        version_row.addWidget(build_lbl)

        self.btn_check_updates = QPushButton("Check for Updates")
        self.btn_check_updates.setIcon(Icons.check_update())
        self.btn_check_updates.setCursor(Qt.PointingHandCursor)
        self.btn_check_updates.clicked.connect(self.check_updates_requested.emit)
        version_row.addWidget(self.btn_check_updates)

        layout.addLayout(version_row)

        # ───── Python Environment & Dependencies ─────
        grp_env = QGroupBox("Environment & Dependencies")
        env_layout = QVBoxLayout(grp_env)
        env_layout.setContentsMargins(12, 10, 12, 10)
        env_layout.setSpacing(8)

        # Python and OS info
        py_ver = platform.python_version()
        arch = platform.architecture()[0]
        os_info = f"{platform.system()} {platform.release()}"
        env_info_lbl = QLabel(f"<b>Python:</b> {py_ver} ({arch}) &nbsp;|&nbsp; <b>OS:</b> {os_info}")
        env_info_lbl.setStyleSheet("font-size: 11px;")
        env_layout.addWidget(env_info_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: rgba(128, 128, 128, 0.3);")
        env_layout.addWidget(sep)

        # Two-column dependency grid
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(4)

        versions = get_runtime_package_versions()
        row = 0
        col = 0
        for pkg, ver in versions.items():
            lbl_name = QLabel(f"<b>{pkg}:</b>")
            lbl_name.setStyleSheet("font-size: 11px;")
            lbl_val = QLabel(ver)
            if ver == "Not installed":
                lbl_val.setStyleSheet("font-size: 11px; color: #888888; font-style: italic;")
            else:
                lbl_val.setStyleSheet("font-size: 11px; color: #2E7D32;" if not ver.startswith("Unk") else "font-size: 11px;")

            grid.addWidget(lbl_name, row, col * 2)
            grid.addWidget(lbl_val, row, col * 2 + 1)

            col += 1
            if col >= 2:
                col = 0
                row += 1

        env_layout.addLayout(grid)
        layout.addWidget(grp_env)

        # ───── Information & Citation Browser ─────
        info_browser = QTextBrowser()
        info_browser.setOpenExternalLinks(True)
        info_browser.setFrameStyle(QTextBrowser.NoFrame)
        info_browser.setStyleSheet("background: transparent; font-size: 11px;")

        info_html = """
        <div style="line-height: 1.4;">
            <h4 style="margin-bottom: 4px;">Key Capabilities</h4>
            <ul style="margin-top: 2px; padding-left: 20px;">
                <li>Processing of 1D discrete spectra and 2D/wafer hyperspectral maps.</li>
                <li>Simultaneous multi-spectra fitting using the Levenberg-Marquardt optimizer (Vectorized Batch Fit engine).</li>
                <li>Automated plot recipes, custom stylesheet templates, and publication figure generation.</li>
                <li>Built-in AI Data Assistant with natural language queries and tool execution.</li>
            </ul>

            <p><b>Documentation:</b> <a href="https://CEA-MetroCarac.github.io/SPECTROview">https://CEA-MetroCarac.github.io/SPECTROview</a></p>

            <h4 style="margin-bottom: 4px;">Citation</h4>
            <p style="margin-top: 2px;">
                Le, Van Hoan. "SPECTROview: An Open-Source Application for Interactive Spectroscopic Data Processing, Fitting, and Visualization." <i>SoftwareX</i> 35 (September 2026): 102862.
                <br>
                <a href="https://doi.org/10.1016/j.softx.2026.102862">https://doi.org/10.1016/j.softx.2026.102862</a>
            </p>
        </div>
        """
        info_browser.setHtml(info_html)
        layout.addWidget(info_browser)

        layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))

        scroll.setWidget(container)
        outer_layout.addWidget(scroll)

