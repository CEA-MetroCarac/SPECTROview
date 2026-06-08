from spectroview import ICON_DIR
ICON_DIR_QT = ICON_DIR.replace("\\", "/")
# spectroview/view/style.py
"""
Glass-style QSS stylesheets for SPECTROview.

  - Semi-transparent layered surfaces for depth hierarchy
  - Subtle translucent borders to define edges without harsh lines
  - Rounded corners on panels, tabs, and controls
  - Modern thin scrollbars
  - Neutral gray accent tones
"""


def dark_glass_stylesheet() -> str:
    """Return the dark-theme glass-style QSS."""
    return """

    /* ═══════════════════════════════════════════════
       GLOBAL DEFAULTS
       ═══════════════════════════════════════════════ */

    QWidget {
        font-family: "Helvetica Neue", Helvetica, Arial;
        font-size: 13px;
    }
    QToolBar QToolButton {
        background: transparent;
        border: none;
    }
    QToolBar QToolButton:hover {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }


    /* ═══════════════════════════════════════════════
       MAIN TAB WIDGET — workspace tabs (Spectra/Maps/Graphs)
       ═══════════════════════════════════════════════ */

    QTabWidget::pane {
        margin-top: 2px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 8px;
        background: rgba(38, 38, 38, 230);
        top: -1px;
    }

    QTabBar::tab {
        margin-top: 4px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-bottom: none;
        border-top-left-radius: 7px;
        border-top-right-radius: 7px;
        padding: 3px;
        margin-right: 2px;
        color: rgba(255, 255, 255, 0.70);
        min-width: 70px;
    }

    QTabBar::tab:selected {
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-bottom: 1px solid rgba(38, 38, 38, 230);
        color: white;
    }

    QTabBar::tab:hover:!selected {
        background: rgba(255, 255, 255, 0.07);
        color: rgba(255, 255, 255, 0.90);
    }

    /* ═══════════════════════════════════════════════
       TOOLBAR — top menu bar
       ═══════════════════════════════════════════════ */

    QToolBar {
        background: rgba(40, 40, 40, 240);
        border-bottom: 1px solid rgba(255, 255, 255, 0.15);
        spacing: 3px;
        padding: 3px;
    }

    QToolBar::separator {
        width: 1px;
        background: rgba(255, 255, 255, 0.20);
        margin: 4px 6px;
    }

    /* ═══════════════════════════════════════════════
       GROUP BOXES — glass cards
       ═══════════════════════════════════════════════ */

    QGroupBox {
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        margin-top: 10px;
        padding: 8px 3px 3px 3px;
        font-weight: bold;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center;
        padding: 2px 3px;
        color: rgba(255, 255, 255, 0.80);
    }

    /* ═══════════════════════════════════════════════
       BUTTONS — frosted pill / rounded style
       ═══════════════════════════════════════════════ */

    QPushButton, QToolButton {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.20);
        border-radius: 6px;
        padding: 3px;
        color: rgba(255, 255, 255, 0.90);
    }

    QPushButton:hover, QToolButton:hover {
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.30);
    }

    QPushButton:pressed, QToolButton:pressed {
        background: rgba(64, 156, 255, 0.35);
        border: 1px solid rgba(64, 156, 255, 0.60);
    }

    QPushButton:disabled, QToolButton:disabled {
        color: rgba(255, 255, 255, 0.30);
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.04);
    }

    QPushButton:checked, QToolButton:checked {
        background: rgba(64, 156, 255, 0.40);
        border: 1px solid rgba(64, 156, 255, 0.70);
        color: white;
    }

    /* ═══════════════════════════════════════════════
       INPUT FIELDS — subtle inset glass
       ═══════════════════════════════════════════════ */

    QLineEdit {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.20);
        border-radius: 5px;
        padding: 3px;
        color: white;
        selection-background-color: rgba(255, 255, 255, 0.20);
    }

    QLineEdit:focus {
        border: 1px solid rgba(255, 255, 255, 0.30);
    }


    /* ═══════════════════════════════════════════════
       COMBOBOX & SPINBOX
       ═══════════════════════════════════════════════ */
    QComboBox, QSpinBox, QDoubleSpinBox {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.20);
        border-radius: 5px;
        padding: 3px 5px;
        color: white;
    }
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
        border: 1px solid rgba(255, 255, 255, 0.30);
    }
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        background: transparent;
        border-left: 1px solid rgba(255, 255, 255, 0.20);
        width: 16px;
    }
    QSpinBox::up-button, QDoubleSpinBox::up-button {
        subcontrol-origin: border;
        subcontrol-position: top right;
        background: transparent;
        border-left: 1px solid rgba(255, 255, 255, 0.20);
        width: 16px;
    }
    QSpinBox::down-button, QDoubleSpinBox::down-button {
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        background: transparent;
        border-left: 1px solid rgba(255, 255, 255, 0.20);
        width: 16px;
    }
    QComboBox::drop-down:pressed, QSpinBox::up-button:pressed, QSpinBox::down-button:pressed, QDoubleSpinBox::up-button:pressed, QDoubleSpinBox::down-button:pressed {
        background: rgba(255, 255, 255, 0.15);
    }

    QComboBox::down-arrow {
        image: url({ICON_DIR_QT}/arrow-down.svg);
        width: 10px; height: 10px;
    }
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
        image: url({ICON_DIR_QT}/arrow-up.svg);
        width: 9px; height: 9px;
    }
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
        image: url({ICON_DIR_QT}/arrow-down.svg);
        width: 9px; height: 9px;
    }


    /* ═══════════════════════════════════════════════
       CHECK BOX & LIST INDICATORS
       ═══════════════════════════════════════════════ */
    QCheckBox::indicator, QListView::indicator, QTreeView::indicator, QGroupBox::indicator {
        width: 14px;
        height: 14px;
        border-radius: 3px;
        background: rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.25);
    }
    QCheckBox::indicator:checked, QListView::indicator:checked, QTreeView::indicator:checked, QGroupBox::indicator:checked {
        background: rgba(64, 156, 255, 0.90);
        border: 1px solid rgba(42, 130, 218, 0.90);
    }
    QCheckBox::indicator:hover, QListView::indicator:hover, QTreeView::indicator:hover, QGroupBox::indicator:hover {
        border: 1px solid rgba(255, 255, 255, 0.35);
    }


    
    /* ═══════════════════════════════════════════════
       CHECK BOX
       ═══════════════════════════════════════════════ */

    QCheckBox {
        spacing: 6px;
        color: rgba(255, 255, 255, 0.85);
    }
    QCheckBox:disabled {
        color: rgba(255, 255, 255, 0.30);
    }
    QCheckBox::indicator:disabled, QListView::indicator:disabled, QTreeView::indicator:disabled, QGroupBox::indicator:disabled {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    QCheckBox::indicator:checked:disabled, QListView::indicator:checked:disabled, QTreeView::indicator:checked:disabled, QGroupBox::indicator:checked:disabled {
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    
    
    
    /* ═══════════════════════════════════════════════
       LIST WIDGETS — glass list with subtle items
       ═══════════════════════════════════════════════ */

    
    QScrollArea > QWidget > QWidget {
        background: transparent;
    }

    QListWidget, QTreeWidget, QScrollArea, #workspaceRightPanel, #bottomToolbarPanel {
        background: rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 6px;
        color: white;
        outline: none;
    }

    QListWidget::item {
        padding: 3px;
        border-radius: 4px;
        margin: 1px 2px;
    }

    QListWidget::item:selected {
        background: rgba(255, 255, 255, 0.15);
        color: white;
    }

    QListWidget::item:hover:!selected {
        background: rgba(255, 255, 255, 0.06);
    }

    QTreeWidget::item {
        padding: 3px;
        border-radius: 3px;
    }

    QTreeWidget::item:selected {
        background: rgba(255, 255, 255, 0.15);
        color: white;
    }

    /* ═══════════════════════════════════════════════
       TABLE VIEW / HEADER VIEW
       ═══════════════════════════════════════════════ */

    QTableWidget, QTableView {
        background: rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 6px;
        gridline-color: rgba(255, 255, 255, 0.15);
        color: white;
    }

    QHeaderView::section {
        background: rgba(255, 255, 255, 0.05);
        border: none;
        border-right: 1px solid rgba(255, 255, 255, 0.15);
        border-bottom: 1px solid rgba(255, 255, 255, 0.20);
        padding: 3px;
        color: rgba(255, 255, 255, 0.80);
        font-weight: bold;
    }

    /* ═══════════════════════════════════════════════
       TOOLTIPS — floating glass pill
       ═══════════════════════════════════════════════ */

    QToolTip {
        background: rgba(50, 50, 50, 245);
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 6px;
        color: white;
        padding: 3px;
    }

    /* ═══════════════════════════════════════════════
       SPLITTER — minimal handle
       ═══════════════════════════════════════════════ */

    QSplitter::handle {
        background: transparent;
    }

    QSplitter::handle:horizontal {
        width: 4px;
    }

    QSplitter::handle:vertical {
        height: 4px;
    }

    QSplitter::handle:hover {
        background: rgba(255, 255, 255, 0.18);
    }

    /* ═══════════════════════════════════════════════
       SCROLLBARS — thin, modern, floating
       ═══════════════════════════════════════════════ */

    QScrollBar:vertical {
        background: transparent;
        width: 4px;
        margin: 0;
        border: none;
    }

    QScrollBar::handle:vertical {
        background: rgba(255, 255, 255, 0.14);
        border-radius: 4px;
        min-height: 30px;
    }

    QScrollBar::handle:vertical:hover {
        background: rgba(255, 255, 255, 0.25);
    }

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }

    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
        background: none;
    }

    QScrollBar:horizontal {
        background: transparent;
        height: 4px;
        margin: 0;
        border: none;
    }

    QScrollBar::handle:horizontal {
        background: rgba(255, 255, 255, 0.14);
        border-radius: 4px;
        min-width: 30px;
    }

    QScrollBar::handle:horizontal:hover {
        background: rgba(255, 255, 255, 0.25);
    }

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
    }

    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
        background: none;
    }

    /* ═══════════════════════════════════════════════
       PROGRESS BAR — glowing accent
       ═══════════════════════════════════════════════ */

    QProgressBar {
        background: rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 4px;
        text-align: center;
        color: rgba(255, 255, 255, 0.80);
    }

    QProgressBar::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 rgba(25, 100, 200, 0.70),
                                    stop:1 rgba(64, 156, 255, 0.70));
        border-radius: 3px;
    }

    

    /* ═══════════════════════════════════════════════
       DIALOG — glass card
       ═══════════════════════════════════════════════ */

    QDialog {
        background: rgb(42, 42, 42);
    }

    /* ═══════════════════════════════════════════════
       FRAME — subtle border
       ═══════════════════════════════════════════════ */

    QFrame[frameShape="4"],    /* HLine */
    QFrame[frameShape="5"] {   /* VLine */
        color: rgba(255, 255, 255, 0.20);
    }

    /* ═══════════════════════════════════════════════
       MENU — floating glass dropdown
       ═══════════════════════════════════════════════ */

    QMenu {
        background: rgba(44, 44, 44, 250);
        border: 1px solid rgba(255, 255, 255, 0.20);
        border-radius: 8px;
        padding: 3px;
        color: white;
    }

    QMenu::item {
        padding: 3px;
        border-radius: 5px;
        margin: 1px 2px;
    }

    QMenu::item:selected {
        background: rgba(255, 255, 255, 0.12);
    }

    QMenu::separator {
        height: 1px;
        background: rgba(255, 255, 255, 0.20);
        margin: 4px 8px;
    }

    /* ═══════════════════════════════════════════════
       TEXT BROWSER — for user manual & about
       ═══════════════════════════════════════════════ */

    QTextBrowser {
        background: rgba(0, 0, 0, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 6px;
        color: white;
    }

    /* ═══════════════════════════════════════════════
       LABEL — default transparency
       ═══════════════════════════════════════════════ */

    QLabel {
        background: transparent;
        color: rgba(255, 255, 255, 0.90);
    }

    """.replace("{ICON_DIR_QT}", ICON_DIR_QT)


def soft_dark_glass_stylesheet() -> str:
    """Return the soft-dark-theme glass-style QSS (charcoal/slate tones)."""
    return """

    /* ═══════════════════════════════════════════════
       GLOBAL DEFAULTS
       ═══════════════════════════════════════════════ */

    QWidget {
        font-family: "Helvetica Neue", Helvetica, Arial;
        font-size: 13px;
    }
    QToolBar QToolButton {
        background: transparent;
        border: none;
    }
    QToolBar QToolButton:hover {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 4px;
    }


    /* ═══════════════════════════════════════════════
       MAIN TAB WIDGET — workspace tabs (Spectra/Maps/Graphs)
       ═══════════════════════════════════════════════ */

    QTabWidget::pane {
        margin-top: 2px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        background: rgba(60, 60, 60, 230);
        top: -1px;
    }

    QTabBar::tab {
        margin-top: 4px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-bottom: none;
        border-top-left-radius: 7px;
        border-top-right-radius: 7px;
        padding: 3px;
        margin-right: 2px;
        color: rgba(255, 255, 255, 0.65);
        min-width: 70px;
    }

    QTabBar::tab:selected {
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.13);
        border-bottom: 1px solid rgba(60, 60, 60, 230);
        color: white;
    }

    QTabBar::tab:hover:!selected {
        background: rgba(255, 255, 255, 0.07);
        color: rgba(255, 255, 255, 0.85);
    }

    /* ═══════════════════════════════════════════════
       TOOLBAR — top menu bar
       ═══════════════════════════════════════════════ */

    QToolBar {
        background: rgba(58, 58, 58, 240);
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        spacing: 3px;
        padding: 3px;
    }

    QToolBar::separator {
        width: 1px;
        background: rgba(255, 255, 255, 0.09);
        margin: 4px 6px;
    }

    /* ═══════════════════════════════════════════════
       GROUP BOXES — glass cards
       ═══════════════════════════════════════════════ */

    QGroupBox {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 8px;
        margin-top: 10px;
        padding: 8px 3px 3px 3px;
        font-weight: bold;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center;
        padding: 2px 3px;
        color: rgba(255, 255, 255, 0.78);
    }

    /* ═══════════════════════════════════════════════
       BUTTONS — frosted pill / rounded style
       ═══════════════════════════════════════════════ */

    QPushButton, QToolButton {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.11);
        border-radius: 6px;
        padding: 3px;
        color: rgba(255, 255, 255, 0.88);
    }

    QPushButton:hover, QToolButton:hover {
        background: rgba(255, 255, 255, 0.11);
        border: 1px solid rgba(255, 255, 255, 0.16);
    }

    QPushButton:pressed, QToolButton:pressed {
        background: rgba(64, 156, 255, 0.35);
        border: 1px solid rgba(64, 156, 255, 0.60);
    }

    QPushButton:disabled, QToolButton:disabled {
        color: rgba(255, 255, 255, 0.28);
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    QPushButton:checked, QToolButton:checked {
        background: rgba(64, 156, 255, 0.40);
        border: 1px solid rgba(64, 156, 255, 0.70);
        color: white;
    }

    /* ═══════════════════════════════════════════════
       INPUT FIELDS — subtle inset glass
       ═══════════════════════════════════════════════ */

    QLineEdit {
        background: rgba(0, 0, 0, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 5px;
        padding: 3px;
        color: white;
        selection-background-color: rgba(255, 255, 255, 0.20);
    }

    QLineEdit:focus {
        border: 1px solid rgba(255, 255, 255, 0.28);
    }


    /* ═══════════════════════════════════════════════
       COMBOBOX & SPINBOX
       ═══════════════════════════════════════════════ */
    QComboBox, QSpinBox, QDoubleSpinBox {
        background: rgba(0, 0, 0, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 5px;
        padding: 3px 5px;
        color: white;
    }
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
        border: 1px solid rgba(255, 255, 255, 0.28);
    }
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        background: transparent;
        border-left: 1px solid rgba(255, 255, 255, 0.09);
        width: 16px;
    }
    QSpinBox::up-button, QDoubleSpinBox::up-button {
        subcontrol-origin: border;
        subcontrol-position: top right;
        background: transparent;
        border-left: 1px solid rgba(255, 255, 255, 0.09);
        width: 16px;
    }
    QSpinBox::down-button, QDoubleSpinBox::down-button {
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        background: transparent;
        border-left: 1px solid rgba(255, 255, 255, 0.09);
        width: 16px;
    }
    QComboBox::drop-down:pressed, QSpinBox::up-button:pressed, QSpinBox::down-button:pressed, QDoubleSpinBox::up-button:pressed, QDoubleSpinBox::down-button:pressed {
        background: rgba(255, 255, 255, 0.12);
    }

    QComboBox::down-arrow {
        image: url({ICON_DIR_QT}/arrow-down.svg);
        width: 10px; height: 10px;
    }
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
        image: url({ICON_DIR_QT}/arrow-up.svg);
        width: 9px; height: 9px;
    }
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
        image: url({ICON_DIR_QT}/arrow-down.svg);
        width: 9px; height: 9px;
    }


    /* ═══════════════════════════════════════════════
       CHECK BOX & LIST INDICATORS
       ═══════════════════════════════════════════════ */
    QCheckBox::indicator, QListView::indicator, QTreeView::indicator, QGroupBox::indicator {
        width: 14px;
        height: 14px;
        border-radius: 3px;
        background: rgba(0, 0, 0, 0.20);
        border: 1px solid rgba(255, 255, 255, 0.16);
    }
    QCheckBox::indicator:checked, QListView::indicator:checked, QTreeView::indicator:checked, QGroupBox::indicator:checked {
        background: rgba(64, 156, 255, 0.90);
        border: 1px solid rgba(42, 130, 218, 0.90);
    }
    QCheckBox::indicator:hover, QListView::indicator:hover, QTreeView::indicator:hover, QGroupBox::indicator:hover {
        border: 1px solid rgba(255, 255, 255, 0.32);
    }


    
    /* ═══════════════════════════════════════════════
       CHECK BOX
       ═══════════════════════════════════════════════ */

    QCheckBox {
        spacing: 6px;
        color: rgba(255, 255, 255, 0.82);
    }
    QCheckBox:disabled {
        color: rgba(255, 255, 255, 0.28);
    }
    QCheckBox::indicator:disabled, QListView::indicator:disabled, QTreeView::indicator:disabled, QGroupBox::indicator:disabled {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    QCheckBox::indicator:checked:disabled, QListView::indicator:checked:disabled, QTreeView::indicator:checked:disabled, QGroupBox::indicator:checked:disabled {
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    
    
    
    /* ═══════════════════════════════════════════════
       LIST WIDGETS — glass list with subtle items
       ═══════════════════════════════════════════════ */

    
    QScrollArea > QWidget > QWidget {
        background: transparent;
    }

    QListWidget, QTreeWidget, QScrollArea, #workspaceRightPanel, #bottomToolbarPanel {
        background: rgba(0, 0, 0, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 6px;
        color: white;
        outline: none;
    }

    QListWidget::item {
        padding: 3px;
        border-radius: 4px;
        margin: 1px 2px;
    }

    QListWidget::item:selected {
        background: rgba(255, 255, 255, 0.13);
        color: white;
    }

    QListWidget::item:hover:!selected {
        background: rgba(255, 255, 255, 0.06);
    }

    QTreeWidget::item {
        padding: 3px;
        border-radius: 3px;
    }

    QTreeWidget::item:selected {
        background: rgba(255, 255, 255, 0.13);
        color: white;
    }

    /* ═══════════════════════════════════════════════
       TABLE VIEW / HEADER VIEW
       ═══════════════════════════════════════════════ */

    QTableWidget, QTableView {
        background: rgba(0, 0, 0, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 6px;
        gridline-color: rgba(255, 255, 255, 0.07);
        color: white;
    }

    QHeaderView::section {
        background: rgba(255, 255, 255, 0.06);
        border: none;
        border-right: 1px solid rgba(255, 255, 255, 0.07);
        border-bottom: 1px solid rgba(255, 255, 255, 0.09);
        padding: 3px;
        color: rgba(255, 255, 255, 0.78);
        font-weight: bold;
    }

    /* ═══════════════════════════════════════════════
       TOOLTIPS — floating glass pill
       ═══════════════════════════════════════════════ */

    QToolTip {
        background: rgba(68, 68, 68, 245);
        border: 1px solid rgba(255, 255, 255, 0.13);
        border-radius: 6px;
        color: white;
        padding: 3px;
    }

    /* ═══════════════════════════════════════════════
       SPLITTER — minimal handle
       ═══════════════════════════════════════════════ */

    QSplitter::handle {
        background: transparent;
    }

    QSplitter::handle:horizontal {
        width: 4px;
    }

    QSplitter::handle:vertical {
        height: 4px;
    }

    QSplitter::handle:hover {
        background: rgba(255, 255, 255, 0.15);
    }

    /* ═══════════════════════════════════════════════
       SCROLLBARS — thin, modern, floating
       ═══════════════════════════════════════════════ */

    QScrollBar:vertical {
        background: transparent;
        width: 4px;
        margin: 0;
        border: none;
    }

    QScrollBar::handle:vertical {
        background: rgba(255, 255, 255, 0.14);
        border-radius: 4px;
        min-height: 30px;
    }

    QScrollBar::handle:vertical:hover {
        background: rgba(255, 255, 255, 0.24);
    }

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }

    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
        background: none;
    }

    QScrollBar:horizontal {
        background: transparent;
        height: 4px;
        margin: 0;
        border: none;
    }

    QScrollBar::handle:horizontal {
        background: rgba(255, 255, 255, 0.14);
        border-radius: 4px;
        min-width: 30px;
    }

    QScrollBar::handle:horizontal:hover {
        background: rgba(255, 255, 255, 0.24);
    }

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
    }

    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
        background: none;
    }

    /* ═══════════════════════════════════════════════
       PROGRESS BAR — glowing accent
       ═══════════════════════════════════════════════ */

    QProgressBar {
        background: rgba(0, 0, 0, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 4px;
        text-align: center;
        color: rgba(255, 255, 255, 0.78);
    }

    QProgressBar::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 rgba(25, 100, 200, 0.70),
                                    stop:1 rgba(64, 156, 255, 0.70));
        border-radius: 3px;
    }

    

    /* ═══════════════════════════════════════════════
       DIALOG — glass card
       ═══════════════════════════════════════════════ */

    QDialog {
        background: rgb(58, 58, 58);
    }

    /* ═══════════════════════════════════════════════
       FRAME — subtle border
       ═══════════════════════════════════════════════ */

    QFrame[frameShape="4"],    /* HLine */
    QFrame[frameShape="5"] {   /* VLine */
        color: rgba(255, 255, 255, 0.09);
    }

    /* ═══════════════════════════════════════════════
       MENU — floating glass dropdown
       ═══════════════════════════════════════════════ */

    QMenu {
        background: rgba(62, 62, 62, 250);
        border: 1px solid rgba(255, 255, 255, 0.11);
        border-radius: 8px;
        padding: 3px;
        color: white;
    }

    QMenu::item {
        padding: 3px;
        border-radius: 5px;
        margin: 1px 2px;
    }

    QMenu::item:selected {
        background: rgba(255, 255, 255, 0.11);
    }

    QMenu::separator {
        height: 1px;
        background: rgba(255, 255, 255, 0.09);
        margin: 4px 8px;
    }

    /* ═══════════════════════════════════════════════
       TEXT BROWSER — for user manual & about
       ═══════════════════════════════════════════════ */

    QTextBrowser {
        background: rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 6px;
        color: white;
    }

    /* ═══════════════════════════════════════════════
       LABEL — default transparency
       ═══════════════════════════════════════════════ */

    QLabel {
        background: transparent;
        color: rgba(255, 255, 255, 0.88);
    }

    """.replace("{ICON_DIR_QT}", ICON_DIR_QT)


def light_glass_stylesheet() -> str:
    """Return the light-theme glass-style QSS."""
    return """

    /* ═══════════════════════════════════════════════
       GLOBAL DEFAULTS
       ═══════════════════════════════════════════════ */

    QWidget {
        font-family: "Helvetica Neue", Helvetica, Arial;
        font-size: 13px;
    }
    QMainWindow, QDialog {
        background-color: #E8EAED;
    }
    QToolBar QToolButton {
        background: transparent;
        border: none;
    }
    QToolBar QToolButton:hover {
        background: rgba(0, 0, 0, 0.05);
        border-radius: 4px;
    }


    /* ═══════════════════════════════════════════════
       MAIN TAB WIDGET
       ═══════════════════════════════════════════════ */

    QTabWidget::pane {
        margin-top: 2px;
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 8px;
        background: #EDF0F2;
        top: -1px;
    }

    QTabBar::tab {
        margin-top: 4px;
        background: rgba(0, 0, 0, 0.03);
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-bottom: none;
        border-top-left-radius: 7px;
        border-top-right-radius: 7px;
        padding: 3px;
        margin-right: 2px;
        color: rgba(0, 0, 0, 0.55);
        min-width: 70px;
    }

    QTabBar::tab:selected {
        background: rgba(255, 255, 255, 240);
        border: 1px solid rgba(0, 0, 0, 0.10);
        border-bottom: 1px solid rgba(255, 255, 255, 240);
        color: rgba(0, 0, 0, 0.90);
    }

    QTabBar::tab:hover:!selected {
        background: rgba(0, 0, 0, 0.05);
        color: rgba(0, 0, 0, 0.75);
    }

    /* ═══════════════════════════════════════════════
       TOOLBAR
       ═══════════════════════════════════════════════ */

    QToolBar {
        background: rgba(245, 245, 245, 245);
        border-bottom: 1px solid rgba(0, 0, 0, 0.06);
        spacing: 3px;
        padding: 3px;
    }

    QToolBar::separator {
        width: 1px;
        background: rgba(0, 0, 0, 0.08);
        margin: 4px 6px;
    }

    /* ═══════════════════════════════════════════════
       GROUP BOXES
       ═══════════════════════════════════════════════ */

    QGroupBox {
        background: rgba(0, 0, 0, 0.03);
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 8px;
        margin-top: 10px;
        padding: 8px 3px 3px 3px;
        font-weight: bold;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center;
        padding: 2px 3px;
        color: rgba(0, 0, 0, 0.70);
    }

    /* ═══════════════════════════════════════════════
       BUTTONS
       ═══════════════════════════════════════════════ */

    QPushButton, QToolButton {
        background: rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(0, 0, 0, 0.10);
        border-radius: 6px;
        padding: 3px;
        color: rgba(0, 0, 0, 0.80);
    }

    QPushButton:hover, QToolButton:hover {
        background: rgba(0, 0, 0, 0.07);
        border: 1px solid rgba(0, 0, 0, 0.15);
    }

    QPushButton:pressed, QToolButton:pressed {
        background: rgba(64, 156, 255, 0.25);
        border: 1px solid rgba(64, 156, 255, 0.50);
    }

    QPushButton:disabled, QToolButton:disabled {
        color: rgba(0, 0, 0, 0.30);
        background: rgba(0, 0, 0, 0.02);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }

    QPushButton:checked, QToolButton:checked {
        background: rgba(64, 156, 255, 0.35);
        border: 1px solid rgba(64, 156, 255, 0.60);
        color: rgba(0, 0, 0, 0.90);
    }

    /* ═══════════════════════════════════════════════
       INPUT FIELDS
       ═══════════════════════════════════════════════ */

    QLineEdit {
        background: rgba(255, 255, 255, 220);
        border: 1px solid rgba(0, 0, 0, 0.10);
        border-radius: 5px;
        padding: 3px;
        color: rgba(0, 0, 0, 0.85);
        selection-background-color: rgba(0, 0, 0, 0.12);
    }

    QLineEdit:focus {
        border: 1px solid rgba(0, 0, 0, 0.25);
    }


    /* ═══════════════════════════════════════════════
       COMBOBOX & SPINBOX
       ═══════════════════════════════════════════════ */
    QComboBox, QSpinBox, QDoubleSpinBox {
        background: rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 5px;
        padding: 3px 5px;
        color: rgba(0, 0, 0, 0.8);
    }
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
        border: 1px solid rgba(0, 0, 0, 0.30);
    }
    QComboBox::drop-down, QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
        subcontrol-origin: padding;
        background: transparent;
        border-left: 1px solid rgba(0, 0, 0, 0.08);
        width: 15px;
    }
    QComboBox::drop-down:pressed, QSpinBox::up-button:pressed, QSpinBox::down-button:pressed, QDoubleSpinBox::up-button:pressed, QDoubleSpinBox::down-button:pressed {
        background: rgba(0, 0, 0, 0.05);
    }

    QComboBox::down-arrow {
        image: url({ICON_DIR_QT}/arrow-down-dark.svg);
        width: 10px; height: 10px;
    }
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
        image: url({ICON_DIR_QT}/arrow-up-dark.svg);
        width: 9px; height: 9px;
    }
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
        image: url({ICON_DIR_QT}/arrow-down-dark.svg);
        width: 9px; height: 9px;
    }


    /* ═══════════════════════════════════════════════
       CHECK BOX & LIST INDICATORS
       ═══════════════════════════════════════════════ */
    QCheckBox::indicator, QListView::indicator, QTreeView::indicator, QGroupBox::indicator {
        width: 14px;
        height: 14px;
        border-radius: 3px;
        background: rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(0, 0, 0, 0.15);
    }
    QCheckBox::indicator:checked, QListView::indicator:checked, QTreeView::indicator:checked, QGroupBox::indicator:checked {
        background: rgba(42, 130, 218, 0.90);
        border: 1px solid rgba(25, 100, 200, 0.90);
    }
    QCheckBox::indicator:hover, QListView::indicator:hover, QTreeView::indicator:hover, QGroupBox::indicator:hover {
        border: 1px solid rgba(0, 0, 0, 0.35);
    }


    
    /* ═══════════════════════════════════════════════
       CHECK BOX
       ═══════════════════════════════════════════════ */

    QCheckBox {
        spacing: 6px;
        color: rgba(0, 0, 0, 0.85);
    }
    QCheckBox:disabled {
        color: rgba(0, 0, 0, 0.30);
    }
    QCheckBox::indicator:disabled, QListView::indicator:disabled, QTreeView::indicator:disabled, QGroupBox::indicator:disabled {
        background: rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    QCheckBox::indicator:checked:disabled, QListView::indicator:checked:disabled, QTreeView::indicator:checked:disabled, QGroupBox::indicator:checked:disabled {
        background: rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(0, 0, 0, 0.15);
    }
    
    
    
    /* ═══════════════════════════════════════════════
       LIST WIDGETS
       ═══════════════════════════════════════════════ */

    
    QScrollArea > QWidget > QWidget {
        background: transparent;
    }

    QListWidget, QTreeWidget, QScrollArea, #workspaceRightPanel, #bottomToolbarPanel {
        background: #EDF0F2;
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 6px;
        color: rgba(0, 0, 0, 0.85);
        outline: none;
    }

    QListWidget::item {
        padding: 3px;
        border-radius: 4px;
        margin: 1px 2px;
    }

    QListWidget::item:selected {
        background: rgba(0, 0, 0, 0.08);
        color: rgba(0, 0, 0, 0.90);
    }

    QListWidget::item:hover:!selected {
        background: rgba(0, 0, 0, 0.04);
    }

    QTreeWidget::item {
        padding: 3px;
        border-radius: 3px;
    }

    QTreeWidget::item:selected {
        background: rgba(0, 0, 0, 0.08);
        color: rgba(0, 0, 0, 0.90);
    }

    /* ═══════════════════════════════════════════════
       TABLE VIEW / HEADER VIEW
       ═══════════════════════════════════════════════ */

    QTableWidget, QTableView {
        background: rgba(255, 255, 255, 200);
        border: 1px solid rgba(0, 0, 0, 0.07);
        border-radius: 6px;
        gridline-color: rgba(0, 0, 0, 0.06);
        color: rgba(0, 0, 0, 0.85);
    }

    QHeaderView::section {
        background: rgba(0, 0, 0, 0.03);
        border: none;
        border-right: 1px solid rgba(0, 0, 0, 0.06);
        border-bottom: 1px solid rgba(0, 0, 0, 0.08);
        padding: 3px;
        color: rgba(0, 0, 0, 0.70);
        font-weight: bold;
    }

    /* ═══════════════════════════════════════════════
       TOOLTIPS
       ═══════════════════════════════════════════════ */

    QToolTip {
        background: rgba(255, 255, 255, 245);
        border: 1px solid rgba(0, 0, 0, 0.10);
        border-radius: 6px;
        color: rgba(0, 0, 0, 0.85);
        padding: 3px;
    }

    /* ═══════════════════════════════════════════════
       SPLITTER
       ═══════════════════════════════════════════════ */

    QSplitter::handle {
        background: rgba(0, 0, 0, 0.04);
    }

    QSplitter::handle:horizontal {
        width: 4px;
    }

    QSplitter::handle:vertical {
        height: 4px;
    }

    QSplitter::handle:hover {
        background: rgba(0, 0, 0, 0.12);
    }

    /* ═══════════════════════════════════════════════
       SCROLLBARS
       ═══════════════════════════════════════════════ */

    QScrollBar:vertical {
        background: transparent;
        width: 4px;
        margin: 0;
        border: none;
    }

    QScrollBar::handle:vertical {
        background: rgba(0, 0, 0, 0.12);
        border-radius: 4px;
        min-height: 30px;
    }

    QScrollBar::handle:vertical:hover {
        background: rgba(0, 0, 0, 0.22);
    }

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }

    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
        background: none;
    }

    QScrollBar:horizontal {
        background: transparent;
        height: 4px;
        margin: 0;
        border: none;
    }

    QScrollBar::handle:horizontal {
        background: rgba(0, 0, 0, 0.12);
        border-radius: 4px;
        min-width: 30px;
    }

    QScrollBar::handle:horizontal:hover {
        background: rgba(0, 0, 0, 0.22);
    }

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
    }

    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
        background: none;
    }

    /* ═══════════════════════════════════════════════
       PROGRESS BAR
       ═══════════════════════════════════════════════ */

    QProgressBar {
        background: rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 4px;
        text-align: center;
        color: rgba(0, 0, 0, 0.70);
    }

    QProgressBar::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 rgba(42, 130, 218, 0.60),
                                    stop:1 rgba(64, 156, 255, 0.60));
        border-radius: 3px;
    }

    
    /* ═══════════════════════════════════════════════
       DIALOG
       ═══════════════════════════════════════════════ */

    QDialog {
        background: rgb(245, 245, 245);
    }

    /* ═══════════════════════════════════════════════
       FRAME
       ═══════════════════════════════════════════════ */

    QFrame[frameShape="4"],
    QFrame[frameShape="5"] {
        color: rgba(0, 0, 0, 0.08);
    }

    /* ═══════════════════════════════════════════════
       MENU
       ═══════════════════════════════════════════════ */

    QMenu {
        background: rgba(255, 255, 255, 250);
        border: 1px solid rgba(0, 0, 0, 0.10);
        border-radius: 8px;
        padding: 3px;
        color: rgba(0, 0, 0, 0.85);
    }

    QMenu::item {
        padding: 3px;
        border-radius: 5px;
        margin: 1px 2px;
    }

    QMenu::item:selected {
        background: rgba(0, 0, 0, 0.06);
    }

    QMenu::separator {
        height: 1px;
        background: rgba(0, 0, 0, 0.08);
        margin: 4px 8px;
    }

    /* ═══════════════════════════════════════════════
       TEXT BROWSER
       ═══════════════════════════════════════════════ */

    QTextBrowser {
        background: rgba(255, 255, 255, 220);
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 6px;
        color: rgba(0, 0, 0, 0.85);
    }

    /* ═══════════════════════════════════════════════
       LABEL
       ═══════════════════════════════════════════════ */

    QLabel {
        background: transparent;
        color: rgba(0, 0, 0, 0.85);
    }

    """.replace("{ICON_DIR_QT}", ICON_DIR_QT)
