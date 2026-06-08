#view/toolbar.py
import os

from PySide6.QtWidgets import QToolBar, QWidget, QSizePolicy, QLabel, QMenu
from PySide6.QtCore import QSize, Signal, Qt
from PySide6.QtGui import  QIcon

from spectroview import ICON_DIR, VERSION

# Available themes: internal key → display label
THEME_OPTIONS = [
    ("dark", "Dark"),
    ("soft_dark", "Soft Dark"),
    ("light", "Light"),
]


class VMenuBar(QToolBar):
    # ───── View → ViewModel signals ─────
    open_requested = Signal()
    save_requested = Signal()
    clear_requested = Signal()
    convert_requested = Signal()
    calc_requested = Signal()
    settings_requested = Signal()
    github_requested = Signal()
    theme_selected = Signal(str)       # emits theme key, e.g. "dark" / "soft_dark" / "light"
    manual_requested = Signal()
    about_requested = Signal()
    version_requested = Signal()
    
    def __init__(self):
        super().__init__()
        self._current_theme = "dark"
        self.init_ui()
        
    def init_ui(self):
        self.setIconSize(QSize(30, 30))
        
        self.actionOpen = self.addAction(QIcon(os.path.join(ICON_DIR, "open.png")), "Open")
        self.actionOpen.triggered.connect(self.open_requested.emit)

        self.actionSave = self.addAction(QIcon(os.path.join(ICON_DIR, "save.png")), "Save")
        self.actionSave.triggered.connect(self.save_requested.emit)

        self.actionClearWS= self.addAction(QIcon(os.path.join(ICON_DIR, "clear.png")), "Clear current workspace")
        self.actionClearWS.triggered.connect(self.clear_requested.emit)

        self.addSeparator()
        self.actionConvert= self.addAction(QIcon(os.path.join(ICON_DIR, "FileConvert.png")), "Convert 2Dmap format")
        self.actionConvert.triggered.connect(self.convert_requested.emit)

        self.actionCalc = self.addAction(QIcon(os.path.join(ICON_DIR, "calc.png")), "Quick Calculators")
        self.actionCalc.triggered.connect(self.calc_requested.emit)

        self.addSeparator()
        self.actionSettings= self.addAction(QIcon(os.path.join(ICON_DIR, "settings.png")), "Settings")
        self.actionSettings.triggered.connect(self.settings_requested.emit)
        
        # ----- Expanding spacer -----
        spacer = QWidget(self)
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.addWidget(spacer)
        
        # ----- Theme dropdown button -----
        self.actionTheme = self.addAction(QIcon(os.path.join(ICON_DIR, "dark-light.png")), "Select Theme")
        self.actionTheme.triggered.connect(self._show_theme_menu)

        self.actionManual= self.addAction(QIcon(os.path.join(ICON_DIR, "manual.png")), "Open User Manual")
        self.actionManual.setToolTip("Open User Manual PDF\n(Ctrl + Click to open online web manual)")
        self.actionManual.triggered.connect(self.manual_requested.emit)

        self.actionGithub= self.addAction(QIcon(os.path.join(ICON_DIR, "github1.png")), "Open Github repository")
        self.actionGithub.triggered.connect(self.github_requested.emit)

        self.actionAbout= self.addAction(QIcon(os.path.join(ICON_DIR, "about.png")), "About SPECTROview")
        self.actionAbout.triggered.connect(self.about_requested.emit)
        self.addSeparator()

        version_label = QLabel(f" v{VERSION} ")
        version_label.setToolTip("Click to view lastest release notes")
        version_label.setStyleSheet("color: gray; font-size: 11px; padding-right: 5px;")
        version_label.setCursor(Qt.CursorShape.PointingHandCursor)
        version_label.mousePressEvent = lambda event: self.version_requested.emit()
        self.addWidget(version_label)

    # ── Theme dropdown ────────────────────────────────────────
    def set_current_theme(self, theme_key: str):
        """Update internal state so the checkmark appears on the right item."""
        self._current_theme = theme_key

    def _show_theme_menu(self):
        """Build and display a popup menu below the theme toolbar button."""
        menu = QMenu(self)
        for key, label in THEME_OPTIONS:
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(key == self._current_theme)
            # Capture `key` in the lambda default argument
            action.triggered.connect(lambda checked, k=key: self._on_theme_chosen(k))
        
        # Position the menu just below the toolbar button
        btn = self.widgetForAction(self.actionTheme)
        if btn:
            pos = btn.mapToGlobal(btn.rect().bottomLeft())
        else:
            pos = self.mapToGlobal(self.rect().bottomLeft())
        menu.exec(pos)

    def _on_theme_chosen(self, theme_key: str):
        self._current_theme = theme_key
        self.theme_selected.emit(theme_key)

    