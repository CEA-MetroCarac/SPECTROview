# spectroview/settings/tab_general.py
"""General settings tab: universal working folder, config backup/restore, and update options."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
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


class GeneralTabPage(QWidget):
    """The General tab: working folder, backup & restore, and update preferences."""

    KEYWORDS = [
        "general", "working", "folder", "directory", "path", "backup",
        "restore", "config", "json", "export", "import", "update",
        "updates", "check updates", "startup", "auto-update"
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
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        # ───── Universal Working Folder ─────
        self.grp_working = QGroupBox("SPECTROview Working Folder")
        working_layout = QVBoxLayout(self.grp_working)
        working_layout.setContentsMargins(12, 12, 12, 12)
        working_layout.setSpacing(10)

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
        working_layout.addWidget(self.lbl_working_hint)

        btn_config_row = QHBoxLayout()
        btn_config_row.setSpacing(10)

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

        # ───── Application Updates ─────
        self.grp_updates = QGroupBox("Application Updates")
        updates_layout = QVBoxLayout(self.grp_updates)
        updates_layout.setContentsMargins(12, 12, 12, 12)
        updates_layout.setSpacing(8)

        self.chk_check_updates_startup = QCheckBox("Check for updates automatically on application startup")
        self.chk_check_updates_startup.setChecked(True)
        updates_layout.addWidget(self.chk_check_updates_startup)

        layout.addWidget(self.grp_updates)

        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        scroll.setWidget(container)
        outer_layout.addWidget(scroll)

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

    def get_settings(self) -> dict:
        return {
            "working_folder": self.le_working_folder.text().strip(),
            "check_updates_startup": self.chk_check_updates_startup.isChecked(),
        }

