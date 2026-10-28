#view/toolbar.py
from PySide6.QtWidgets import QToolBar, QWidget, QSizePolicy, QLabel, QMenu
from PySide6.QtCore import QSize, Signal, Qt

from spectroview import VERSION
from spectroview.view.theme import Icons

# Available themes: internal key → display label
THEME_OPTIONS = [
    ("dark", "Modern Dark"),
    ("light", "Modern Light"),
    ("---", ""),  # separator
    ("classic_dark", "Classic Dark"),
    ("classic_light", "Classic Light"),
]


class VMenuBar(QToolBar):
    # ───── View → ViewModel signals ─────
    open_requested = Signal()
    save_requested = Signal()
    clear_requested = Signal()
    convert_requested = Signal()
    calc_requested = Signal()
    ai_chat_requested = Signal()       # opens the SPECTROview AI Agent panel
    settings_requested = Signal()
    github_requested = Signal()
    theme_selected = Signal(str)       # emits theme key, e.g. "dark" / "light" / "modern_warm" / "classic_dark" / "classic_light"
    manual_requested = Signal()
    about_requested = Signal()
    version_requested = Signal()
    check_update_requested = Signal()
    
    def __init__(self):
        super().__init__()
        self._current_theme = "dark"
        self._default_icons = {}
        self.init_ui()
        
    def init_ui(self):
        self.setIconSize(QSize(22, 22))
        
        self.actionOpen = self.addAction(Icons.open(size=22), "Open")
        self.actionOpen.triggered.connect(self.open_requested.emit)

        self.actionSave = self.addAction(Icons.save(size=22), "Save")
        self.actionSave.triggered.connect(self.save_requested.emit)

        self.actionClearWS= self.addAction(Icons.clear(size=22), "Clear current workspace")
        self.actionClearWS.triggered.connect(self.clear_requested.emit)

        self.addSeparator()
        self.actionConvert= self.addAction(Icons.convert(size=22), "Convert 2Dmap format")
        self.actionConvert.triggered.connect(self.convert_requested.emit)

        self.actionCalc = self.addAction(Icons.calc(size=22), "Quick Calculators")
        self.actionCalc.setShortcut("Ctrl+Shift+C")
        self.actionCalc.triggered.connect(self.calc_requested.emit)

        self.addSeparator()
        self.actionSettings= self.addAction(Icons.settings(size=22), "Settings")
        self.actionSettings.setShortcut("Ctrl+Shift+S")
        self.actionSettings.triggered.connect(self.settings_requested.emit)
        
        # ----- Expanding spacer -----
        spacer = QWidget(self)
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.addWidget(spacer)

        # ----- SPECTROview AI Agent -----
        self.actionAIChat = self.addAction(Icons.sparkles(size=22), "SPECTROview AI Agent")
        self.actionAIChat.setShortcut("Ctrl+Shift+A")
        self.actionAIChat.setToolTip(
            "Toggle SPECTROview AI Agent (Ctrl+Shift+A)\n"
            "Query your fit results using natural language and get instant answers.\n"
        )
        self.actionAIChat.triggered.connect(self.ai_chat_requested.emit)

        # ----- Check for updates -----
        self.actionCheckUpdate = self.addAction(Icons.check_update(size=22), "Check for updates")
        self.actionCheckUpdate.triggered.connect(self.check_update_requested.emit)
        
        # ----- Theme dropdown button -----
        self.actionTheme = self.addAction(Icons.theme(self._current_theme, size=22), "Select Theme")
        self.actionTheme.setToolTip("Select Theme (Dark / Light)")
        self.actionTheme.triggered.connect(self._show_theme_menu)

        self.actionManual= self.addAction(Icons.manual(size=22), "Open User Manual")
        self.actionManual.setShortcut("Ctrl+Shift+M")
        self.actionManual.setToolTip("Open User Manual PDF\n(Ctrl + Click to open online web manual)")
        self.actionManual.triggered.connect(self.manual_requested.emit)

        self.actionGithub= self.addAction(Icons.github(size=22), "Open Github repository")
        self.actionGithub.triggered.connect(self.github_requested.emit)

        self.actionAbout= self.addAction(Icons.about(size=22), "About SPECTROview")
        self.actionAbout.triggered.connect(self.about_requested.emit)
        self.addSeparator()

        version_label = QLabel(f" v{VERSION} ")
        version_label.setToolTip("Click to view lastest release notes")
        version_label.setStyleSheet("color: gray; font-size: 11px; padding-right: 5px;")
        version_label.setCursor(Qt.CursorShape.PointingHandCursor)
        version_label.mousePressEvent = lambda event: self.version_requested.emit()
        self.addWidget(version_label)

        # Store baseline icons for dynamic theme switching
        self._default_icons = {
            self.actionOpen: Icons.open(size=22),
            self.actionSave: Icons.save(size=22),
            self.actionClearWS: Icons.clear(size=22),
            self.actionConvert: Icons.convert(size=22),
            self.actionCalc: Icons.calc(size=22),
            self.actionAIChat: Icons.sparkles(size=22),
            self.actionSettings: Icons.settings(size=22),
            self.actionCheckUpdate: Icons.check_update(size=22),
            self.actionTheme: Icons.theme(self._current_theme, size=22),
            self.actionManual: Icons.manual(size=22),
            self.actionGithub: Icons.github(size=22),
            self.actionAbout: Icons.about(size=22),
        }
        self._update_theme_icon()

    # ── Theme dropdown ────────────────────────────────────────
    def _update_theme_icon(self) -> None:
        """Update theme action icon based on active theme (moon for dark, sun for light)."""
        is_dark = self._current_theme in ("dark", "classic_dark", "soft_dark") or "dark" in str(self._current_theme).lower()
        icon = Icons.moon(size=22) if is_dark else Icons.sun(size=22)
        self.actionTheme.setIcon(icon)
        self._default_icons[self.actionTheme] = icon

    def set_current_theme(self, theme_key: str):
        """Update internal state so the checkmark appears on the right item."""
        if theme_key == "soft_dark":
            theme_key = "dark"
        elif theme_key in ("modern_warm", "warm"):
            theme_key = "light"
        self._current_theme = theme_key
        self._update_theme_icon()

    def apply_theme(self, theme_key: str) -> None:
        """Update top menubar icons across theme switches."""
        self.set_current_theme(theme_key)
        for action, icon in self._default_icons.items():
            if action is not self.actionTheme:
                action.setIcon(icon)

    def _show_theme_menu(self, checked=False):
        """Build and display a popup menu below the theme toolbar button."""
        menu = QMenu(self)
        for key, label in THEME_OPTIONS:
            if key == "---":
                menu.addSeparator()
                continue
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(key == self._current_theme)
            # Capture `key` in the lambda default argument
            action.triggered.connect(lambda checked=False, k=key: self._on_theme_chosen(k))
        
        # Position the menu just below the toolbar button
        btn = self.widgetForAction(self.actionTheme)
        if btn:
            pos = btn.mapToGlobal(btn.rect().bottomLeft())
        else:
            pos = self.mapToGlobal(self.rect().bottomLeft())
        menu.exec(pos)

    def _on_theme_chosen(self, theme_key: str):
        self.set_current_theme(theme_key)
        self.theme_selected.emit(theme_key)

    