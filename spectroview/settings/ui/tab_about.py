# spectroview/settings/ui/tab_about.py
"""About tab: application information, versioning, Python environment, runtime dependencies, and citation."""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
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

from spectroview import ICON_DIR, VERSION
from spectroview.i18n import bind_text, bind_title, bind_tooltip, on_language_changed, tr
from spectroview.settings.build_info import (
    get_build_number,
    get_environment_summary,
    get_runtime_package_versions as get_runtime_package_versions,
)
from spectroview.view.theme import Icons, current_tokens


def _highlight_group_style() -> str:
    return (
        "QGroupBox { border: 1.5px solid #3B82F6; border-radius: 6px; margin-top: 10px; font-weight: bold; } "
        "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 4px; color: #3B82F6; }"
    )


def _highlight_label_style() -> str:
    return "background-color: rgba(59, 130, 246, 0.2); border: 1px solid #3B82F6; border-radius: 4px; padding: 1px 3px; color: #3B82F6; font-weight: bold;"


class AboutTabPage(QWidget):
    """The About tab displaying logo, version, Python & package dependencies, and citation."""

    check_updates_requested = Signal()

    KEYWORDS = [
        "about", "à propos", "version", "build", "python", "environment", "environnement",
        "dependency", "dependencies", "dépendances", "packages", "paquets", "pyside6", "qt",
        "numpy", "scipy", "matplotlib", "pandas", "pybaselines", "citation",
        "author", "auteur", "license", "licence", "doi", "paper", "article", "softwarex",
        "exe", "whl", "standalone", "mode", "copy", "copier", "update", "mise à jour",
    ]

    CITATION_TEXT = (
        'Le, Van Hoan. "SPECTROview: An Open-Source Application for Interactive '
        'Spectroscopic Data Processing, Fitting, and Visualization." '
        'SoftwareX 35 (September 2026): 102862. '
        'https://doi.org/10.1016/j.softx.2026.102862'
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pkg_labels: list[tuple[str, QLabel]] = []
        self._pkg_val_labels: list[tuple[str, QLabel]] = []
        self._env_summary: dict = {}
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

        tok = current_tokens()
        self._subtitle_label = QLabel()
        bind_text(self._subtitle_label, "Tool for Spectroscopic Data Processing and Visualization")
        self._subtitle_label.setStyleSheet(f"font-size: 11px; color: {tok.text_muted}; font-style: italic;")
        self._subtitle_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        title_col.addWidget(self._subtitle_label)

        header_row.addLayout(title_col)
        layout.addLayout(header_row)

        # Version & Check for Updates row
        version_row = QHBoxLayout()
        version_row.setAlignment(Qt.AlignCenter)
        version_row.setSpacing(12)

        ver_lbl = QLabel()
        bind_text(ver_lbl, "Version {version}", version=VERSION)
        ver_lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
        ver_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        version_row.addWidget(ver_lbl)

        self.btn_check_updates = QPushButton()
        bind_text(self.btn_check_updates, "Check for Updates")
        self.btn_check_updates.setIcon(Icons.check_update())
        self.btn_check_updates.setCursor(Qt.PointingHandCursor)
        bind_tooltip(self.btn_check_updates, "Check online for newer releases of SPECTROview")
        self.btn_check_updates.clicked.connect(self._on_check_updates_clicked)
        version_row.addWidget(self.btn_check_updates)

        layout.addLayout(version_row)

        # Separate line for Build number (under Version and Check for Updates)
        self._build_lbl = QLabel()
        bind_text(self._build_lbl, "Build: {build}", build=get_build_number())
        self._build_lbl.setAlignment(Qt.AlignCenter)
        self._build_lbl.setStyleSheet(f"color: {tok.text_muted}; font-size: 11px;")
        self._build_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._build_lbl)

        # ───── Python Environment & Dependencies ─────
        self.grp_env = QGroupBox()
        bind_title(self.grp_env, "Environment & Dependencies")
        env_layout = QVBoxLayout(self.grp_env)
        env_layout.setContentsMargins(8, 6, 8, 6)
        env_layout.setSpacing(4)

        self._env_summary = get_environment_summary()
        self._env_info_lbl = QLabel()
        self._env_info_lbl.setStyleSheet("font-size: 11px; line-height: 1.3;")
        self._env_info_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._refresh_env_info()
        on_language_changed(self, self._refresh_env_info, slot_key="env_info")
        env_layout.addWidget(self._env_info_lbl)

        self._env_sep = QFrame()
        self._env_sep.setFrameShape(QFrame.HLine)
        self._env_sep.setStyleSheet(f"color: {tok.border_subtle};")
        env_layout.addWidget(self._env_sep)

        # Two-column dependency grid
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(2)

        versions = self._env_summary["packages"]
        row = 0
        col = 0
        for pkg, ver in versions.items():
            lbl_name = QLabel(f"<b>{pkg}:</b>")
            lbl_name.setStyleSheet("font-size: 11px;")
            lbl_name.setTextInteractionFlags(Qt.TextSelectableByMouse)

            lbl_val = QLabel()
            if ver == "Not installed":
                bind_text(lbl_val, "Not installed")
                lbl_val.setStyleSheet(f"font-size: 11px; color: {tok.text_muted}; font-style: italic;")
            else:
                lbl_val.setText(ver)
                lbl_val.setStyleSheet(f"font-size: 11px; color: {tok.success};" if not ver.startswith("Unk") else "font-size: 11px;")
            lbl_val.setTextInteractionFlags(Qt.TextSelectableByMouse)

            grid.addWidget(lbl_name, row, col * 2)
            grid.addWidget(lbl_val, row, col * 2 + 1)
            self._pkg_labels.append((pkg.lower(), lbl_name))
            self._pkg_val_labels.append((ver, lbl_val))

            col += 1
            if col >= 2:
                col = 0
                row += 1

        env_layout.addLayout(grid)
        layout.addWidget(self.grp_env)

        # ───── Citation ─────
        self.grp_citation = QGroupBox()
        bind_title(self.grp_citation, "Citation")
        citation_layout = QVBoxLayout(self.grp_citation)
        citation_layout.setContentsMargins(8, 6, 8, 6)
        citation_layout.setSpacing(4)

        prompt_row = QHBoxLayout()
        prompt_row.setSpacing(6)

        self._lbl_prompt = QLabel()
        bind_text(
            self._lbl_prompt,
            "If you use SPECTROview in your research or publications, please consider citing:",
        )
        self._lbl_prompt.setStyleSheet(f"color: {tok.text_muted}; font-style: italic; font-size: 11px;")
        self._lbl_prompt.setWordWrap(True)
        self._lbl_prompt.setTextInteractionFlags(Qt.TextSelectableByMouse)
        prompt_row.addWidget(self._lbl_prompt, 1)

        self.btn_copy_citation = QPushButton()
        self.btn_copy_citation.setIcon(Icons.copy())
        self.btn_copy_citation.setFixedSize(24, 24)
        self.btn_copy_citation.setCursor(Qt.PointingHandCursor)
        bind_tooltip(self.btn_copy_citation, "Copy citation to clipboard")
        self.btn_copy_citation.clicked.connect(self._on_copy_citation_clicked)
        prompt_row.addWidget(self.btn_copy_citation)

        citation_layout.addLayout(prompt_row)

        self._lbl_cite = QLabel(self._citation_html(tok.accent))
        self._lbl_cite.setStyleSheet("font-size: 11px;")
        self._lbl_cite.setWordWrap(True)
        self._lbl_cite.setOpenExternalLinks(True)
        self._lbl_cite.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
        citation_layout.addWidget(self._lbl_cite)

        layout.addWidget(self.grp_citation)

        layout.addSpacerItem(QSpacerItem(20, 6, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self._scroll.setWidget(container)
        outer_layout.addWidget(self._scroll)

    def _refresh_env_info(self) -> None:
        env_summary = self._env_summary
        is_frozen = env_summary["is_frozen"]
        py_ver = env_summary["python_version"]
        arch = env_summary["python_architecture"]
        host_os = env_summary["host_os"]

        if is_frozen:
            raw_mode = env_summary.get("mode") or "Standalone Executable (.exe)"
            mode_str = tr(raw_mode)
            build_os = env_summary.get("build_os") or tr("Unknown")
            build_py = env_summary.get("build_python") or py_ver
            build_arch = env_summary.get("build_architecture") or arch

            self.grp_env.setToolTip(
                tr("Dependencies and build environment bundled into this standalone executable.")
            )
            env_info_text = (
                f"<b>{tr('Mode:')}</b> {mode_str} &nbsp;|&nbsp; <b>{tr('Host OS:')}</b> {host_os}<br>"
                f"<b>Python:</b> {build_py} ({build_arch}) [{tr('Bundled')}] &nbsp;|&nbsp; <b>{tr('Built on:')}</b> {build_os}"
            )
        else:
            install_label = env_summary.get("installation_label") or "Python Package (.whl / local)"
            raw_mode = "Python Package (.whl / local)"
            if install_label and "Development" in install_label:
                raw_mode = "Python Package (.whl / local) — Dev"
            mode_str = tr(raw_mode)
            self.grp_env.setToolTip(
                tr("Runtime dependencies detected in your local Python environment.")
            )
            env_info_text = (
                f"<b>{tr('Mode:')}</b> {mode_str} &nbsp;|&nbsp; <b>{tr('OS:')}</b> {host_os} &nbsp;|&nbsp; "
                f"<b>Python:</b> {py_ver} ({arch})"
            )

        self._env_info_lbl.setText(env_info_text)

    def _on_copy_citation_clicked(self) -> None:
        """Copy formatted citation to system clipboard with temporary visual feedback."""
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self.CITATION_TEXT)
        self.btn_copy_citation.setIcon(Icons.check())
        bind_tooltip(self.btn_copy_citation, "Copied!")
        QTimer.singleShot(1500, self, self._reset_copy_citation_btn)

    def _reset_copy_citation_btn(self) -> None:
        self.btn_copy_citation.setIcon(Icons.copy())
        bind_tooltip(self.btn_copy_citation, "Copy citation to clipboard")

    def _on_check_updates_clicked(self) -> None:
        """Trigger update check and provide immediate visual feedback on button."""
        self.btn_check_updates.setEnabled(False)
        bind_text(self.btn_check_updates, "Checking for updates...")
        self.check_updates_requested.emit()
        QTimer.singleShot(2500, self, self._reset_check_updates_btn)

    def _reset_check_updates_btn(self) -> None:
        self.btn_check_updates.setEnabled(True)
        bind_text(self.btn_check_updates, "Check for Updates")

    # ───── Search Highlighting & Scrolling ─────
    def highlight_search(self, query: str) -> None:
        """Highlight matching dependencies, update button, or citation and scroll into view."""
        q = query.strip().lower()

        # Clear existing styles
        self.grp_env.setStyleSheet("")
        self.grp_citation.setStyleSheet("")
        self.btn_check_updates.setStyleSheet("")
        self.btn_copy_citation.setStyleSheet("")
        for _pkg, lbl in self._pkg_labels:
            lbl.setStyleSheet("font-size: 11px;")

        if not q:
            return

        first_match: Optional[QWidget] = None

        if any(k in q for k in ("update", "check", "version", "mise", "vérifier")):
            self.btn_check_updates.setStyleSheet("border: 1.5px solid #3B82F6; font-weight: bold;")
            first_match = self.btn_check_updates

        cite_terms = ["citation", "cite", "citer", "doi", "paper", "article", "softwarex", "author", "auteur", "le", "copy", "copier"]
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

    @staticmethod
    def _citation_html(link_color: str) -> str:
        return (
            'Le, Van Hoan. "SPECTROview: An Open-Source Application for Interactive Spectroscopic Data Processing, Fitting, and Visualization." '
            '<i>SoftwareX</i> 35 (September 2026): 102862.<br>'
            f'<a href="https://doi.org/10.1016/j.softx.2026.102862" style="color: {link_color};">https://doi.org/10.1016/j.softx.2026.102862</a>'
        )

    def refresh_theme_styles(self) -> None:
        """Re-apply token-based label styling after a live theme change."""
        tok = current_tokens()
        self._subtitle_label.setStyleSheet(f"font-size: 11px; color: {tok.text_muted}; font-style: italic;")
        self._build_lbl.setStyleSheet(f"color: {tok.text_muted}; font-size: 11px;")
        self._env_sep.setStyleSheet(f"color: {tok.border_subtle};")
        for ver, lbl_val in self._pkg_val_labels:
            if ver == "Not installed":
                lbl_val.setStyleSheet(f"font-size: 11px; color: {tok.text_muted}; font-style: italic;")
            else:
                lbl_val.setStyleSheet(f"font-size: 11px; color: {tok.success};" if not ver.startswith("Unk") else "font-size: 11px;")
        self._lbl_prompt.setStyleSheet(f"color: {tok.text_muted}; font-style: italic; font-size: 11px;")
        self._lbl_cite.setText(self._citation_html(tok.accent))
