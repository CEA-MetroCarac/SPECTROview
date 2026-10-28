# spectroview/settings/ui/tab_about.py
"""About tab: application information, versioning, Python environment, runtime dependencies, and citation."""

from __future__ import annotations

import importlib.metadata
import os
import platform
from typing import Optional

from PySide6.QtCore import QTimer, Qt, Signal, qVersion
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
    QVBoxLayout,
    QWidget,
)

from spectroview import BUILD_NUMBER, ICON_DIR, VERSION
from spectroview.view.theme import Icons


def _highlight_group_style() -> str:
    return (
        "QGroupBox { border: 1.5px solid #3B82F6; border-radius: 6px; margin-top: 10px; font-weight: bold; } "
        "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 4px; color: #3B82F6; }"
    )


def _highlight_label_style() -> str:
    return "background-color: rgba(59, 130, 246, 0.2); border: 1px solid #3B82F6; border-radius: 4px; padding: 1px 3px; color: #3B82F6; font-weight: bold;"


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
        "matplotlib", "pandas", "pybaselines", "citation",
        "author", "license", "doi", "paper", "softwarex"
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pkg_labels: list[tuple[str, QLabel]] = []
        self._init_ui()

    def _init_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)

        # ───── Header & Logo ─────
        header_row = QHBoxLayout()
        header_row.setSpacing(12)
        header_row.setAlignment(Qt.AlignCenter)

        logo_label = QLabel()
        logo_path = os.path.join(ICON_DIR, "logo_spectroview.png")
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path).scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
        header_row.addWidget(logo_label)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title_label = QLabel("SPECTROview")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        title_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        title_col.addWidget(title_label)

        subtitle_label = QLabel("Tool for Spectroscopic Data Processing and Visualization")
        subtitle_label.setStyleSheet("font-size: 11px; color: gray; font-style: italic;")
        subtitle_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        title_col.addWidget(subtitle_label)

        header_row.addLayout(title_col)
        layout.addLayout(header_row)

        # Version & Check for Updates row
        version_row = QHBoxLayout()
        version_row.setAlignment(Qt.AlignCenter)
        version_row.setSpacing(12)

        ver_lbl = QLabel(f"Version {VERSION}")
        ver_lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
        ver_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        version_row.addWidget(ver_lbl)

        self.btn_check_updates = QPushButton("Check for Updates")
        self.btn_check_updates.setIcon(Icons.check_update())
        self.btn_check_updates.setCursor(Qt.PointingHandCursor)
        self.btn_check_updates.setToolTip("Check online for newer releases of SPECTROview")
        self.btn_check_updates.clicked.connect(self._on_check_updates_clicked)
        version_row.addWidget(self.btn_check_updates)

        layout.addLayout(version_row)

        # Separate line for Build number (under Version and Check for Updates)
        build_lbl = QLabel(f"Build: {BUILD_NUMBER}")
        build_lbl.setAlignment(Qt.AlignCenter)
        build_lbl.setStyleSheet("color: #888888; font-size: 11px;")
        build_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(build_lbl)

        # ───── Python Environment & Dependencies ─────
        self.grp_env = QGroupBox("Environment & Dependencies")
        env_layout = QVBoxLayout(self.grp_env)
        env_layout.setContentsMargins(8, 6, 8, 6)
        env_layout.setSpacing(4)

        # Python and OS info
        py_ver = platform.python_version()
        arch = platform.architecture()[0]
        os_info = f"{platform.system()} {platform.release()}"
        env_info_lbl = QLabel(f"<b>Python:</b> {py_ver} ({arch}) &nbsp;|&nbsp; <b>OS:</b> {os_info}")
        env_info_lbl.setStyleSheet("font-size: 11px;")
        env_info_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        env_layout.addWidget(env_info_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: rgba(128, 128, 128, 0.25);")
        env_layout.addWidget(sep)

        # Two-column dependency grid
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(2)

        versions = get_runtime_package_versions()
        row = 0
        col = 0
        for pkg, ver in versions.items():
            lbl_name = QLabel(f"<b>{pkg}:</b>")
            lbl_name.setStyleSheet("font-size: 11px;")
            lbl_name.setTextInteractionFlags(Qt.TextSelectableByMouse)

            lbl_val = QLabel(ver)
            lbl_val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            if ver == "Not installed":
                lbl_val.setStyleSheet("font-size: 11px; color: #888888; font-style: italic;")
            else:
                lbl_val.setStyleSheet("font-size: 11px; color: #2E7D32;" if not ver.startswith("Unk") else "font-size: 11px;")

            grid.addWidget(lbl_name, row, col * 2)
            grid.addWidget(lbl_val, row, col * 2 + 1)
            self._pkg_labels.append((pkg.lower(), lbl_name))

            col += 1
            if col >= 2:
                col = 0
                row += 1

        env_layout.addLayout(grid)
        layout.addWidget(self.grp_env)

        # ───── Citation ─────
        self.grp_citation = QGroupBox("Citation")
        citation_layout = QVBoxLayout(self.grp_citation)
        citation_layout.setContentsMargins(8, 6, 8, 6)
        citation_layout.setSpacing(4)

        lbl_prompt = QLabel(
            "If you use SPECTROview in your research or publications, please consider citing:"
        )
        lbl_prompt.setStyleSheet("color: gray; font-style: italic; font-size: 11px;")
        lbl_prompt.setWordWrap(True)
        lbl_prompt.setTextInteractionFlags(Qt.TextSelectableByMouse)
        citation_layout.addWidget(lbl_prompt)

        lbl_cite = QLabel(
            'Le, Van Hoan. "SPECTROview: An Open-Source Application for Interactive Spectroscopic Data Processing, Fitting, and Visualization." '
            '<i>SoftwareX</i> 35 (September 2026): 102862.<br>'
            '<a href="https://doi.org/10.1016/j.softx.2026.102862" style="color: #3B82F6;">https://doi.org/10.1016/j.softx.2026.102862</a>'
        )
        lbl_cite.setStyleSheet("font-size: 11px;")
        lbl_cite.setWordWrap(True)
        lbl_cite.setOpenExternalLinks(True)
        lbl_cite.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
        citation_layout.addWidget(lbl_cite)

        layout.addWidget(self.grp_citation)

        layout.addSpacerItem(QSpacerItem(20, 6, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self._scroll.setWidget(container)
        outer_layout.addWidget(self._scroll)

    def _on_check_updates_clicked(self) -> None:
        """Trigger update check and provide immediate visual feedback on button."""
        self.btn_check_updates.setEnabled(False)
        self.btn_check_updates.setText("Checking for updates...")
        self.check_updates_requested.emit()
        QTimer.singleShot(2500, self, self._reset_check_updates_btn)

    def _reset_check_updates_btn(self) -> None:
        self.btn_check_updates.setEnabled(True)
        self.btn_check_updates.setText("Check for Updates")

    # ───── Search Highlighting & Scrolling ─────
    def highlight_search(self, query: str) -> None:
        """Highlight matching dependencies, update button, or citation and scroll into view."""
        q = query.strip().lower()

        # Clear existing styles
        self.grp_env.setStyleSheet("")
        self.grp_citation.setStyleSheet("")
        self.btn_check_updates.setStyleSheet("")
        for _pkg, lbl in self._pkg_labels:
            lbl.setStyleSheet("font-size: 11px;")

        if not q:
            return

        first_match: Optional[QWidget] = None

        if "update" in q or "check" in q or "version" in q:
            self.btn_check_updates.setStyleSheet("border: 1.5px solid #3B82F6; font-weight: bold;")
            first_match = self.btn_check_updates

        cite_terms = ["citation", "cite", "doi", "paper", "softwarex", "author", "le"]
        if any(term in q or q in term for term in cite_terms):
            self.grp_citation.setStyleSheet(_highlight_group_style())
            if first_match is None:
                first_match = self.grp_citation

        for pkg, lbl in self._pkg_labels:
            if q in pkg or pkg in q:
                lbl.setStyleSheet(_highlight_label_style())
                if first_match is None:
                    first_match = lbl

        if first_match:
            self._scroll.ensureWidgetVisible(first_match, 0, 30)
