# spectroview/settings/ui/tab_general.py
"""General settings tab: working folder, config backup/restore, theme, language, and updates."""

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

from spectroview.i18n import (
    SUPPORTED_LANGUAGES,
    bind_combo_items,
    bind_placeholder,
    bind_text,
    bind_title,
    bind_tooltip,
    normalize_language,
)
from spectroview.view.theme import Icons, current_tokens, normalize_theme_key


def _highlight_group_style() -> str:
    return (
        "QGroupBox { border: 1.5px solid #3B82F6; border-radius: 6px; margin-top: 10px; font-weight: bold; } "
        "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 4px; color: #3B82F6; }"
    )


class GeneralTabPage(QWidget):
    """The General tab: working folder, theme, language selection, backup & restore, and updates."""

    theme_changed = Signal(str)
    language_changed = Signal(str)

    KEYWORDS = [
        "general", "général", "working", "folder", "directory", "path", "dossier", "travail",
        "backup", "restore", "sauvegarde", "restaurer", "config", "json", "export", "import",
        "update", "updates", "mise à jour", "mises à jour", "check updates", "startup",
        "démarrage", "auto-update", "prerelease", "préversion", "beta", "rc",
        "theme", "thème", "appearance", "apparence", "dark", "sombre", "gray", "grey", "gris",
        "light", "clair", "modern", "moderne", "classic", "classique",
        "modern dark", "modern gray", "modern grey", "modern light", "classic dark", "classic light",
        "language", "langue", "sprache", "язык", "语言",
        "english", "anglais", "englisch", "английский", "英语",
        "french", "français", "francais", "französisch", "французский", "法语",
        "chinese", "简体中文", "中文", "chinois", "chinesisch", "китайский",
        "german", "deutsch", "allemand", "немецкий", "德语",
        "russian", "русский", "russe", "russisch", "俄语", "locale",
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
        self.grp_working = QGroupBox()
        bind_title(self.grp_working, "SPECTROview Working Folder")
        working_layout = QVBoxLayout(self.grp_working)
        working_layout.setContentsMargins(8, 8, 8, 8)
        working_layout.setSpacing(6)

        folder_row = QHBoxLayout()
        folder_row.setSpacing(8)

        self.btn_working_folder = QPushButton()
        bind_text(self.btn_working_folder, "Browse")
        self.btn_working_folder.setIcon(Icons.folder())
        self.btn_working_folder.setMinimumWidth(95)
        bind_tooltip(self.btn_working_folder, "Select or change the SPECTROview working folder")

        self.le_working_folder = QLineEdit()
        bind_placeholder(
            self.le_working_folder,
            "Select a folder to store models, recipes, styles, and chat history...",
        )

        self.btn_open_working_folder = QPushButton()
        bind_text(self.btn_open_working_folder, "Open")
        self.btn_open_working_folder.setIcon(Icons.open())
        self.btn_open_working_folder.setMinimumWidth(85)
        bind_tooltip(self.btn_open_working_folder, "Open working folder in system file manager")
        self.btn_open_working_folder.clicked.connect(self._on_open_working_folder)

        folder_row.addWidget(self.btn_working_folder)
        folder_row.addWidget(self.le_working_folder, 1)
        folder_row.addWidget(self.btn_open_working_folder)
        working_layout.addLayout(folder_row)

        self.lbl_working_hint = QLabel()
        bind_text(
            self.lbl_working_hint,
            "Universal working folder. Automatically manages:\n"
            "• fit_model/ (saved peak fitting models)\n"
            "• peak_profile/ (saved global peak limit profiles)\n"
            "• plot_recipe/ & plot_style/ (graph recipes and custom styles)\n"
            "• ai_chat_history/ (AI assistant conversation logs)\n"
            "• template/ (general plot templates)\n"
            "• config.json (portable settings backup for seamless PC migration)",
        )
        self.lbl_working_hint.setStyleSheet(
            f"color: {current_tokens().text_muted}; font-style: italic; font-size: 11px;"
        )
        self.lbl_working_hint.setWordWrap(True)
        self.lbl_working_hint.setTextInteractionFlags(Qt.TextSelectableByMouse)
        working_layout.addWidget(self.lbl_working_hint)

        btn_config_row = QHBoxLayout()
        btn_config_row.setSpacing(8)

        self.btn_export_config = QPushButton()
        bind_text(self.btn_export_config, "Export Config...")
        self.btn_export_config.setIcon(Icons.export())
        bind_tooltip(self.btn_export_config, "Export application settings to a custom JSON backup file")

        self.btn_import_config = QPushButton()
        bind_text(self.btn_import_config, "Restore Config...")
        self.btn_import_config.setIcon(Icons.refresh())
        bind_tooltip(self.btn_import_config, "Restore application settings from a JSON backup file")

        btn_config_row.addWidget(self.btn_export_config)
        btn_config_row.addWidget(self.btn_import_config)
        btn_config_row.addStretch(1)
        working_layout.addLayout(btn_config_row)

        layout.addWidget(self.grp_working)

        # ───── Appearance & Theme ─────
        self.grp_theme = QGroupBox()
        bind_title(self.grp_theme, "Interface Theme")
        theme_layout = QVBoxLayout(self.grp_theme)
        theme_layout.setContentsMargins(8, 8, 8, 8)
        theme_layout.setSpacing(6)

        theme_row = QHBoxLayout()
        theme_row.setSpacing(8)

        self.lbl_theme = QLabel()
        bind_text(self.lbl_theme, "Theme:")
        self.lbl_theme.setFixedWidth(80)
        self.lbl_theme.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.cbb_theme = QComboBox()
        bind_combo_items(
            self.cbb_theme,
            [
                ("Modern Dark", "dark"),
                ("Modern Gray", "gray"),
                ("Modern Light", "light"),
                ("Classic Dark", "classic_dark"),
                ("Classic Light", "classic_light"),
            ],
        )
        bind_tooltip(
            self.cbb_theme,
            "Select application theme (Modern Dark, Modern Gray, Modern Light, Classic Dark, Classic Light)",
        )
        self.cbb_theme.currentIndexChanged.connect(self._on_theme_combobox_changed)

        theme_row.addWidget(self.lbl_theme)
        theme_row.addWidget(self.cbb_theme, 1)
        theme_layout.addLayout(theme_row)

        layout.addWidget(self.grp_theme)

        # ───── Language ─────
        self.grp_language = QGroupBox()
        bind_title(self.grp_language, "Interface Language")
        lang_layout = QVBoxLayout(self.grp_language)
        lang_layout.setContentsMargins(8, 8, 8, 8)
        lang_layout.setSpacing(6)

        lang_row = QHBoxLayout()
        lang_row.setSpacing(8)

        self.lbl_language = QLabel()
        bind_text(self.lbl_language, "Language:")
        self.lbl_language.setFixedWidth(80)
        self.lbl_language.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.cbb_language = QComboBox()
        for code, label in SUPPORTED_LANGUAGES:
            self.cbb_language.addItem(label, code)
        bind_tooltip(
            self.cbb_language,
            "Select the graphical interface language (changes apply immediately)",
        )
        self.cbb_language.currentIndexChanged.connect(self._on_language_combobox_changed)

        lang_row.addWidget(self.lbl_language)
        lang_row.addWidget(self.cbb_language, 1)
        lang_layout.addLayout(lang_row)

        layout.addWidget(self.grp_language)

        # ───── Application Updates ─────
        self.grp_updates = QGroupBox()
        bind_title(self.grp_updates, "Application Updates")
        updates_layout = QVBoxLayout(self.grp_updates)
        updates_layout.setContentsMargins(8, 8, 8, 8)
        updates_layout.setSpacing(6)

        self.chk_check_updates_startup = QCheckBox()
        bind_text(self.chk_check_updates_startup, "Check for updates automatically on application startup")
        self.chk_check_updates_startup.setChecked(True)
        updates_layout.addWidget(self.chk_check_updates_startup)

        self.chk_include_prereleases = QCheckBox()
        bind_text(self.chk_include_prereleases, "Include pre-release / release candidate builds")
        self.chk_include_prereleases.setChecked(False)
        bind_tooltip(
            self.chk_include_prereleases,
            "When enabled, SPECTROview also offers beta / release-candidate versions during update checks.",
        )
        updates_layout.addWidget(self.chk_include_prereleases)

        layout.addWidget(self.grp_updates)

        layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self._scroll.setWidget(container)
        outer_layout.addWidget(self._scroll)

    def _on_theme_combobox_changed(self, index: int) -> None:
        key = self.cbb_theme.itemData(index)
        if key:
            self.theme_changed.emit(key)

    def _on_language_combobox_changed(self, index: int) -> None:
        code = self.cbb_language.itemData(index)
        if code:
            self.language_changed.emit(code)

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
        self.chk_include_prereleases.setChecked(data.get("include_prereleases", False))

        theme_key = data.get("theme", "")
        if not theme_key:
            from spectroview.view.theme import ThemeManager
            mgr = ThemeManager.instance()
            theme_key = mgr.current_theme if mgr else "gray"
        theme_key = normalize_theme_key(theme_key)

        idx = self.cbb_theme.findData(theme_key)
        if idx >= 0:
            self.cbb_theme.blockSignals(True)
            self.cbb_theme.setCurrentIndex(idx)
            self.cbb_theme.blockSignals(False)

        lang_code = normalize_language(data.get("language", "en"))
        lang_idx = self.cbb_language.findData(lang_code)
        if lang_idx >= 0:
            self.cbb_language.blockSignals(True)
            self.cbb_language.setCurrentIndex(lang_idx)
            self.cbb_language.blockSignals(False)

    def get_settings(self) -> dict:
        return {
            "working_folder": self.le_working_folder.text().strip(),
            "check_updates_startup": self.chk_check_updates_startup.isChecked(),
            "include_prereleases": self.chk_include_prereleases.isChecked(),
            "theme": self.cbb_theme.currentData() or "gray",
            "language": self.cbb_language.currentData() or "en",
        }

    # ───── Search Highlighting & Scrolling ─────
    def highlight_search(self, query: str) -> None:
        """Highlight matching options/groupboxes and scroll first match into view."""
        q = query.strip().lower()

        # Clear existing styles
        self.grp_working.setStyleSheet("")
        self.grp_theme.setStyleSheet("")
        self.grp_language.setStyleSheet("")
        self.grp_updates.setStyleSheet("")

        if not q:
            return

        working_terms = [
            "working", "folder", "directory", "path", "fit_model", "plot_recipe",
            "plot_style", "ai_chat", "template", "config", "backup", "restore", "export", "import",
            "dossier", "travail", "sauvegarde", "restaurer", "exporter",
        ]
        theme_terms = [
            "theme", "thème", "appearance", "apparence", "dark", "sombre",
            "gray", "grey", "gris", "light", "clair", "modern", "moderne", "classic", "classique",
        ]
        language_terms = [
            "language", "langue", "sprache", "язык", "语言",
        "english", "anglais", "englisch", "английский", "英语",
        "french", "français", "francais", "französisch", "французский", "法语",
        "chinese", "简体中文", "中文", "chinois", "chinesisch", "китайский",
        "german", "deutsch", "allemand", "немецкий", "德语",
        "russian", "русский", "russe", "russisch", "俄语", "locale",
        ]
        update_terms = [
            "update", "updates", "mise à jour", "mises à jour", "startup", "démarrage",
            "check", "version", "prerelease", "préversion", "beta", "rc",
        ]

        first_match: Optional[QWidget] = None

        if any(term in q or q in term for term in working_terms):
            self.grp_working.setStyleSheet(_highlight_group_style())
            if first_match is None:
                first_match = self.grp_working

        if any(term in q or q in term for term in theme_terms):
            self.grp_theme.setStyleSheet(_highlight_group_style())
            if first_match is None:
                first_match = self.grp_theme

        if any(term in q or q in term for term in language_terms):
            self.grp_language.setStyleSheet(_highlight_group_style())
            if first_match is None:
                first_match = self.grp_language

        if any(term in q or q in term for term in update_terms):
            self.grp_updates.setStyleSheet(_highlight_group_style())
            if first_match is None:
                first_match = self.grp_updates

        if first_match:
            self._scroll.ensureWidgetVisible(first_match, 0, 30)

    def refresh_theme_styles(self) -> None:
        """Re-apply token-based label styling after a live theme change."""
        self.lbl_working_hint.setStyleSheet(
            f"color: {current_tokens().text_muted}; font-style: italic; font-size: 11px;"
        )
