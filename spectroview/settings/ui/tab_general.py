# spectroview/settings/ui/tab_general.py
"""General settings tab: universal working folder, config backup/restore, appearance/theme, and update options."""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from spectroview.view.theme import Icons


def _highlight_group_style() -> str:
    return (
        "QGroupBox { border: 1.5px solid #3B82F6; border-radius: 6px; margin-top: 10px; font-weight: bold; } "
        "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 4px; color: #3B82F6; }"
    )


class GeneralTabPage(QWidget):
    """The General tab: working folder, theme selection, backup & restore, and update preferences."""

    theme_changed = Signal(str)

    KEYWORDS = [
        "general", "working", "folder", "directory", "path", "backup",
        "restore", "config", "json", "export", "import", "update",
        "updates", "check updates", "startup", "auto-update",
        "theme", "appearance", "dark", "light", "modern", "classic",
        "modern dark", "modern light", "classic dark", "classic light",
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ───── Universal Working Folder ─────
        self.grp_working = QGroupBox("SPECTROview Working Folder")
        working_layout = QVBoxLayout(self.grp_working)
        working_layout.setContentsMargins(8, 8, 8, 8)
        working_layout.setSpacing(6)

        folder_row = QHBoxLayout()
        folder_row.setSpacing(8)

        self.btn_working_folder = QPushButton("Browse")
        self.btn_working_folder.setIcon(Icons.folder())
        self.btn_working_folder.setMaximumWidth(90)
        self.btn_working_folder.setToolTip("Select or change the SPECTROview working folder")

        self.le_working_folder = QLineEdit()
        self.le_working_folder.setPlaceholderText("Select a folder to store models, recipes, styles, and chat history...")

        self.btn_open_working_folder = QPushButton("Open")
        self.btn_open_working_folder.setIcon(Icons.open())
        self.btn_open_working_folder.setMaximumWidth(80)
        self.btn_open_working_folder.setToolTip("Open working folder in system file manager")
        self.btn_open_working_folder.clicked.connect(self._on_open_working_folder)

        folder_row.addWidget(self.btn_working_folder)
        folder_row.addWidget(self.le_working_folder, 1)
        folder_row.addWidget(self.btn_open_working_folder)
        working_layout.addLayout(folder_row)

        self.lbl_working_hint = QLabel(
            "Universal working folder. Automatically manages:\n"
            "• fit_model/ (saved peak fitting models)\n"
            "• plot_recipe/ & plot_style/ (graph recipes and custom styles)\n"
            "• ai_chat_history/ (AI assistant conversation logs)\n"
            "• template/ (general plot templates)\n"
            "• config.json (portable settings backup for seamless PC migration)"
        )
        self.lbl_working_hint.setStyleSheet("color: gray; font-style: italic; font-size: 11px;")
        self.lbl_working_hint.setWordWrap(True)
        self.lbl_working_hint.setTextInteractionFlags(Qt.TextSelectableByMouse)
        working_layout.addWidget(self.lbl_working_hint)

        btn_config_row = QHBoxLayout()
        btn_config_row.setSpacing(8)

        self.btn_export_config = QPushButton("Export Config...")
        self.btn_export_config.setIcon(Icons.export())
        self.btn_export_config.setToolTip("Export application settings to a custom JSON backup file")

        self.btn_import_config = QPushButton("Restore Config...")
        self.btn_import_config.setIcon(Icons.refresh())
        self.btn_import_config.setToolTip("Restore application settings from a JSON backup file")

        btn_config_row.addWidget(self.btn_export_config)
        btn_config_row.addWidget(self.btn_import_config)
        btn_config_row.addStretch(1)
        working_layout.addLayout(btn_config_row)

        layout.addWidget(self.grp_working)

        # ───── Appearance & Theme ─────
        self.grp_theme = QGroupBox("Interface Theme")
        theme_layout = QVBoxLayout(self.grp_theme)
        theme_layout.setContentsMargins(8, 8, 8, 8)
        theme_layout.setSpacing(6)

        theme_row = QHBoxLayout()
        theme_row.setSpacing(8)

        self.lbl_theme = QLabel("Theme:")
        self.lbl_theme.setFixedWidth(80)
        self.lbl_theme.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.cbb_theme = QComboBox()
        self.cbb_theme.addItem("Modern Dark", "dark")
        self.cbb_theme.addItem("Modern Light", "light")
        self.cbb_theme.addItem("Classic Dark", "classic_dark")
        self.cbb_theme.addItem("Classic Light", "classic_light")
        self.cbb_theme.setToolTip("Select application theme (Modern Dark, Modern Light, Classic Dark, Classic Light)")
        self.cbb_theme.currentIndexChanged.connect(self._on_theme_combobox_changed)

        theme_row.addWidget(self.lbl_theme)
        theme_row.addWidget(self.cbb_theme, 1)
        theme_layout.addLayout(theme_row)

        layout.addWidget(self.grp_theme)

        # ───── Application Updates ─────
        self.grp_updates = QGroupBox("Application Updates")
        updates_layout = QVBoxLayout(self.grp_updates)
        updates_layout.setContentsMargins(8, 8, 8, 8)
        updates_layout.setSpacing(6)

        self.chk_check_updates_startup = QCheckBox("Check for updates automatically on application startup")
        self.chk_check_updates_startup.setChecked(True)
        updates_layout.addWidget(self.chk_check_updates_startup)

        layout.addWidget(self.grp_updates)

        layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self._scroll.setWidget(container)
        outer_layout.addWidget(self._scroll)

    def _on_theme_combobox_changed(self, index: int) -> None:
        key = self.cbb_theme.itemData(index)
        if key:
            self.theme_changed.emit(key)

    def _on_open_working_folder(self) -> None:
        folder = self.le_working_folder.text().strip()
        if folder:
            if not os.path.exists(folder):
                try:
                    os.makedirs(folder, exist_ok=True)
                except OSError:
                    pass
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def apply_settings(self, data: dict) -> None:
        self.le_working_folder.setText(data.get("working_folder", ""))
        self.chk_check_updates_startup.setChecked(data.get("check_updates_startup", True))

        theme_key = data.get("theme", "")
        if not theme_key:
            from spectroview.view.theme import ThemeManager
            mgr = ThemeManager.instance()
            theme_key = mgr.current_theme if mgr else "dark"
        if theme_key == "soft_dark":
            theme_key = "dark"
        elif theme_key in ("modern_warm", "warm"):
            theme_key = "light"

        idx = self.cbb_theme.findData(theme_key)
        if idx >= 0:
            self.cbb_theme.blockSignals(True)
            self.cbb_theme.setCurrentIndex(idx)
            self.cbb_theme.blockSignals(False)

    def get_settings(self) -> dict:
        return {
            "working_folder": self.le_working_folder.text().strip(),
            "check_updates_startup": self.chk_check_updates_startup.isChecked(),
            "theme": self.cbb_theme.currentData() or "dark",
        }

    # ───── Search Highlighting & Scrolling ─────
    def highlight_search(self, query: str) -> None:
        """Highlight matching options/groupboxes and scroll first match into view."""
        q = query.strip().lower()

        # Clear existing styles
        self.grp_working.setStyleSheet("")
        self.grp_theme.setStyleSheet("")
        self.grp_updates.setStyleSheet("")

        if not q:
            return

        working_terms = ["working", "folder", "directory", "path", "fit_model", "plot_recipe",
                         "plot_style", "ai_chat", "template", "config", "backup", "restore", "export", "import"]
        theme_terms = ["theme", "appearance", "dark", "light", "modern", "classic"]
        update_terms = ["update", "updates", "startup", "check", "version"]

        first_match: Optional[QWidget] = None

        if any(term in q or q in term for term in working_terms):
            self.grp_working.setStyleSheet(_highlight_group_style())
            if first_match is None:
                first_match = self.grp_working

        if any(term in q or q in term for term in theme_terms):
            self.grp_theme.setStyleSheet(_highlight_group_style())
            if first_match is None:
                first_match = self.grp_theme

        if any(term in q or q in term for term in update_terms):
            self.grp_updates.setStyleSheet(_highlight_group_style())
            if first_match is None:
                first_match = self.grp_updates

        if first_match:
            self._scroll.ensureWidgetVisible(first_match, 0, 30)
