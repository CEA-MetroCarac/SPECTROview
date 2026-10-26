# spectroview/view/theme/qss_template.py
"""
Single parameterized QSS template for SPECTROview.

Adopts the PLUME design system tokens and component hierarchy, providing
a crisp, cohesive theme stylesheet with integer-alpha rgba colors, vector SVG subcontrols,
and reusable component style builders.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tokens import ThemeTokens

from spectroview import ICON_DIR

ICON_DIR_QT = ICON_DIR.replace("\\", "/")


def _icon_url(filename: str) -> str:
    """Return a forward-slash normalized path for Qt QSS image URLs."""
    return os.path.join(ICON_DIR, filename).replace("\\", "/")


def _resolve_tokens(tokens: "ThemeTokens | None" = None) -> "ThemeTokens":
    """Resolve explicit tokens or fall back to current active application tokens."""
    if tokens is not None:
        return tokens
    from .manager import current_tokens
    return current_tokens()


# ─────────────────────────────────────────────────────────────
#  PLUME-inspired QSS Template with @placeholder tokens
# ─────────────────────────────────────────────────────────────

QSS_TEMPLATE = r"""
    /* ═══════════════════════════════════════════════
       GLOBAL ROOT & CONTAINERS
       ═══════════════════════════════════════════════ */

    QWidget {
        color: @text_primary;
    }

    QMainWindow, QDialog, QMessageBox {
        background-color: @background;
    }

    QLabel, QCheckBox, QRadioButton, QGroupBox {
        background: transparent;
    }

    QScrollArea, QStackedWidget, QSplitter {
        background: transparent;
        border: none;
    }

    QWidget#qt_scrollarea_viewport {
        background: transparent;
    }

    /* ═══════════════════════════════════════════════
       TOP TOOLBAR & MENUBAR
       ═══════════════════════════════════════════════ */

    QToolBar {
        background-color: @sidebar_bg;
        border: none;
        border-bottom: 1px solid @border;
        spacing: 3px;
        padding: 2px 6px;
    }

    QToolBar::separator {
        width: 1px;
        background-color: @border;
        margin: 4px 6px;
    }

    QToolBar QToolButton {
        background: transparent;
        border: none;
        border-radius: 4px;
        padding: 2px;
    }

    QToolBar QToolButton:hover {
        background-color: @surface_hover;
    }

    QToolBar QToolButton:checked {
        background-color: @selection;
        border: 1px solid @accent;
    }

    QToolBar QToolButton:pressed {
        background-color: @surface_pressed;
        border: 1px solid @accent;
    }

    QMenuBar {
        background-color: @sidebar_bg;
        color: @text_primary;
        border-bottom: 1px solid @border;
        padding: 2px;
    }

    QMenuBar::item {
        padding: 4px 8px;
        border-radius: 5px;
        background: transparent;
    }

    QMenuBar::item:selected {
        background-color: @surface_hover;
    }

    QMenu {
        background-color: @surface_elevated;
        color: @text_primary;
        border: 1px solid @border_strong;
        border-radius: 8px;
        padding: 4px;
    }

    QMenu::item {
        padding: 6px 24px 6px 10px;
        border-radius: 5px;
        background-color: transparent;
    }

    QMenu::item:selected {
        background-color: @surface_hover;
        color: @text_primary;
    }

    QMenu::item:disabled {
        color: @text_muted;
    }

    QMenu::separator {
        height: 1px;
        background: @border;
        margin: 4px 6px;
    }

    /* ═══════════════════════════════════════════════
       TOOLTIPS & METADATA POPUP
       ═══════════════════════════════════════════════ */

    QToolTip {
        background-color: @surface_elevated;
        color: @text_primary;
        border: 1px solid @border_strong;
        border-radius: 4px;
        padding: 1px 2px;
    }

    QFrame#MetadataTooltipPopup {
        background-color: @surface_elevated;
        border: 1px solid @border_strong;
        border-radius: 8px;
    }

    QFrame#MetadataTooltipPopup QLabel {
        color: @text_primary;
        background: transparent;
        border: none;
        font-size: 11px;
        line-height: 1.4;
    }

    /* ═══════════════════════════════════════════════
       SCROLLBARS — PLUME slim pill floating style
       ═══════════════════════════════════════════════ */

    QScrollBar:vertical {
        background: transparent;
        width: 10px;
        margin: 2px;
        border: none;
    }

    QScrollBar::handle:vertical {
        background: @scrollbar_thumb;
        border-radius: 3px;
        min-height: 24px;
    }

    QScrollBar::handle:vertical:hover {
        background: @scrollbar_thumb_hover;
    }

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
        width: 0px;
        background: transparent;
    }

    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
        background: transparent;
    }

    QScrollBar:horizontal {
        background: transparent;
        height: 10px;
        margin: 2px;
        border: none;
    }

    QScrollBar::handle:horizontal {
        background: @scrollbar_thumb;
        border-radius: 3px;
        min-width: 24px;
    }

    QScrollBar::handle:horizontal:hover {
        background: @scrollbar_thumb_hover;
    }

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        height: 0px;
        width: 0px;
        background: transparent;
    }

    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
        background: transparent;
    }

    /* ═══════════════════════════════════════════════
       SPLITTERS — PLUME 5px handle with SVG grip
       ═══════════════════════════════════════════════ */

    QSplitter::handle:horizontal {
        width: 5px;
        background-color: @border_subtle;
        border: none;
        border-radius: 2px;
        image: url("@splitter_grip_v_url");
    }

    QSplitter::handle:vertical {
        height: 5px;
        background-color: @border_subtle;
        border: none;
        border-radius: 2px;
        image: url("@splitter_grip_h_url");
    }

    QSplitter::handle:hover, QSplitter::handle:pressed {
        background-color: @accent_tint;
        border: none;
    }

    /* ═══════════════════════════════════════════════
       STATUS BAR & PROGRESS BAR
       ═══════════════════════════════════════════════ */

    QStatusBar {
        background-color: @sidebar_bg;
        color: @text_secondary;
        border-top: 1px solid @border;
    }

    QProgressBar {
        background-color: @surface_sunken;
        border: 1px solid @border;
        border-radius: 6px;
        text-align: center;
        color: @progress_text;
        font-weight: 600;
        min-height: 16px;
        max-height: 18px;
    }

    QProgressBar::chunk {
        background-color: @accent;
        border-radius: 5px;
    }

    /* ═══════════════════════════════════════════════
       3-TIER WORKSPACE SURFACES & MDI SUBWINDOWS
       ═══════════════════════════════════════════════ */

    QWidget#workspaceRightPanel {
        background-color: transparent;
        border: none;
        border-radius: 0px;
    }

    QFrame#canvasCard {
        background-color: @surface;
        border: 1px solid @border;
        border-radius: 8px;
    }

    QMdiArea {
        background-color: @background;
        border: none;
    }

    /* ═══════════════════════════════════════════════
       MAIN TAB WIDGET — workspace tabs
       ═══════════════════════════════════════════════ */

    QTabWidget::pane {
        margin-top: 0px;
        border: none;
        border-top: 1px solid @border;
        border-radius: 0px;
        background-color: transparent;
        top: -1px;
    }

    QTabBar::tab {
        margin-top: 4px;
        background-color: @background;
        border: 1px solid @border_subtle;
        border-bottom: none;
        border-top-left-radius: 7px;
        border-top-right-radius: 7px;
        padding: 4px 12px;
        margin-right: 2px;
        color: @text_secondary;
        min-width: 70px;
        font-weight: 500;
    }

    QTabBar::tab:selected {
        background-color: @surface_elevated;
        border: 1px solid @border_strong;
        border-bottom: 2px solid @accent;
        color: @text_primary;
        font-weight: 600;
    }

    QTabBar::tab:hover:!selected {
        background-color: @surface_hover;
        color: @text_primary;
    }

    /* ═══════════════════════════════════════════════
       GROUP BOXES — cards
       ═══════════════════════════════════════════════ */

    QGroupBox {
        background-color: @surface;
        border: 1px solid @border;
        border-radius: 8px;
        margin-top: 10px;
        padding: 8px 4px 4px 4px;
        font-weight: bold;
    }

    QGroupBox#controlBarPanel, QGroupBox#shiftPanel, QGroupBox#bottomToolbarPanel {
        margin-top: 0px;
        padding: 0px;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center;
        padding: 2px 5px;
        color: @text_secondary;
        font-size: 11px;
        font-weight: 600;
    }

    /* ═══════════════════════════════════════════════
       BUTTONS & SEMANTIC PROPERTY VARIANTS
       ═══════════════════════════════════════════════ */

    QPushButton, QDialogButtonBox QPushButton {
        background-color: @surface_elevated;
        border: 1px solid @border;
        border-radius: 6px;
        padding: 2px 4px;
        color: @text_primary;
        font-weight: 500;
    }

    QToolButton {
        background-color: @surface_elevated;
        border: 1px solid @border;
        border-radius: 6px;
        padding: 2px;
        color: @text_primary;
        font-weight: 500;
    }

    QToolButton::menu-indicator {
        image: url("@chevron_down_url");
        width: 7px;
        height: 7px;
        subcontrol-origin: padding;
        subcontrol-position: bottom right;
    }

    QPushButton:hover, QToolButton:hover, QDialogButtonBox QPushButton:hover {
        background-color: @surface_hover;
        border-color: @border_strong;
    }

    QPushButton:pressed, QToolButton:pressed, QDialogButtonBox QPushButton:pressed {
        background-color: @surface_pressed;
        border-color: @accent;
    }

    QPushButton:disabled, QToolButton:disabled, QDialogButtonBox QPushButton:disabled {
        color: @text_muted;
        background-color: transparent;
        border: 1px solid @border;
    }

    QPushButton:checked, QToolButton:checked {
        background-color: @selection;
        border: 1px solid @accent;
        color: @text_primary;
    }

    QPushButton[variant="solid"] {
        background-color: @solid;
        color: @on_solid;
        border: none;
        font-weight: 600;
    }
    QPushButton[variant="solid"]:hover {
        background-color: @solid_hover;
    }
    QPushButton[variant="solid"]:pressed {
        background-color: @solid_pressed;
    }

    QPushButton[variant="primary"] {
        background-color: @accent;
        color: #FFFFFF;
        border: none;
        font-weight: 600;
    }
    QPushButton[variant="primary"]:hover {
        background-color: @accent_hover;
    }
    QPushButton[variant="primary"]:pressed {
        background-color: @accent_pressed;
    }

    QPushButton[variant="ghost"] {
        background-color: transparent;
        color: @text_secondary;
        border: none;
    }
    QPushButton[variant="ghost"]:hover {
        background-color: @surface_hover;
        color: @text_primary;
    }
    QPushButton[variant="ghost"]:pressed {
        background-color: @surface_pressed;
    }

    QPushButton[variant="danger"] {
        background-color: transparent;
        color: @error;
        border: 1px solid @error;
    }
    QPushButton[variant="danger"]:hover {
        background-color: @error;
        color: #FFFFFF;
    }
    QPushButton[variant="danger"]:pressed {
        background-color: @error_tint;
    }

    QPushButton[variant="pill"] {
        border-radius: 12px;
        padding: 2px 10px;
    }

    /* ═══════════════════════════════════════════════
       INPUT FIELDS
       ═══════════════════════════════════════════════ */

    QLineEdit {
        background-color: @input_bg;
        border: 1px solid @input_border;
        border-radius: 6px;
        padding: 2px 4px;
        color: @text_primary;
        selection-background-color: @selection;
        selection-color: @text_primary;
    }

    QLineEdit:focus {
        border: 1px solid @accent;
    }

    /* ═══════════════════════════════════════════════
       COMBOBOX & SPINBOX
       ═══════════════════════════════════════════════ */

    QComboBox {
        background-color: @input_bg;
        border: 1px solid @input_border;
        border-radius: 6px;
        padding: 1px 3px;
        color: @text_primary;
    }

    QComboBox:hover {
        border-color: @border_strong;
    }

    QComboBox:focus {
        border: 1px solid @accent;
    }

    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 15px;
        border: none;
        border-left: 1px solid @border_subtle;
        background: transparent;
    }

    QComboBox::down-arrow {
        image: url("@chevron_down_url");
        width: 10px;
        height: 10px;
    }

    QComboBox QAbstractItemView {
        background-color: @surface;
        color: @text_primary;
        border: 1px solid @border;
        border-radius: 8px;
        padding: 4px;
        selection-background-color: @surface_hover;
        selection-color: @text_primary;
    }

    QSpinBox, QDoubleSpinBox {
        background-color: @input_bg;
        border: 1px solid @input_border;
        border-radius: 6px;
        padding: 1px 2px 1px 4px;
        color: @text_primary;
        selection-background-color: @selection;
        selection-color: @text_primary;
    }

    QSpinBox:hover, QDoubleSpinBox:hover {
        border-color: @border_strong;
    }

    QSpinBox:focus, QDoubleSpinBox:focus {
        border: 1px solid @accent;
    }

    QSpinBox::up-button, QDoubleSpinBox::up-button {
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 14px;
        border: none;
        border-left: 1px solid @border;
        border-bottom: 1px solid @border_subtle;
        border-top-right-radius: 5px;
        background-color: @surface_elevated;
        margin: 1px 1px 0px 0px;
    }

    QSpinBox::down-button, QDoubleSpinBox::down-button {
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 14px;
        border: none;
        border-left: 1px solid @border;
        border-bottom-right-radius: 5px;
        background-color: @surface_elevated;
        margin: 0px 1px 1px 0px;
    }

    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
        background-color: @surface_hover;
    }

    QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
    QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
        background-color: @surface_pressed;
    }

    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
        image: url("@chevron_up_url");
        width: 8px;
        height: 8px;
    }

    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
        image: url("@chevron_down_url");
        width: 8px;
        height: 8px;
    }

    /* ═══════════════════════════════════════════════
       CHECK BOX & RADIO BUTTONS
       ═══════════════════════════════════════════════ */

    QCheckBox {
        spacing: 6px;
        color: @text_primary;
    }

    QCheckBox::indicator {
        width: 15px;
        height: 15px;
        border-radius: 4px;
        border: 1px solid @border_strong;
        background-color: @input_bg;
    }

    QCheckBox::indicator:hover {
        border-color: @accent;
    }

    QCheckBox::indicator:checked {
        background-color: @accent;
        border-color: @accent;
        image: url("@checkbox_checked_url");
    }

    QCheckBox::indicator:disabled {
        border-color: @border;
        background-color: @surface_sunken;
    }

    QRadioButton::indicator {
        width: 14px;
        height: 14px;
        border-radius: 7px;
        border: 1px solid @border_strong;
        background-color: @input_bg;
    }

    QRadioButton::indicator:hover {
        border-color: @accent;
    }

    QRadioButton::indicator:checked {
        background-color: @accent;
        border: 3px solid @surface_elevated;
    }

    /* ═══════════════════════════════════════════════
       TABLE VIEW & HEADER VIEW
       ═══════════════════════════════════════════════ */

    QHeaderView {
        background-color: transparent;
        border: none;
    }

    QHeaderView::section {
        background-color: @surface_elevated;
        color: @text_muted;
        border: none;
        border-bottom: 1px solid @border;
        border-right: 1px solid @border;
        padding: 4px 8px;
        font-size: 11px;
        font-weight: 500;
    }

    QTableCornerButton::section {
        background-color: @surface_elevated;
        border: none;
        border-bottom: 1px solid @border;
        border-right: 1px solid @border;
    }

    QTableView, QTableWidget {
        background-color: @surface;
        alternate-background-color: @surface_elevated;
        border: 1px solid @border;
        border-radius: 8px;
        gridline-color: @border_subtle;
        color: @text_primary;
        selection-background-color: @selection;
        selection-color: @text_primary;
        outline: none;
    }

    QGroupBox QTableView, QGroupBox QTableWidget {
        border: none;
        background-color: transparent;
    }

    QTableView::item, QTableWidget::item {
        padding: 3px 6px;
    }

    QTableView::item:selected, QTableWidget::item:selected {
        background-color: @selection;
        color: @text_primary;
    }

    QListView, QTreeView, QListWidget, QTreeWidget {
        background-color: @surface;
        color: @text_primary;
        border: 1px solid @border;
        border-radius: 8px;
        outline: none;
    }

    QListView::item, QTreeView::item,
    QListWidget::item, QTreeWidget::item {
        padding: 4px 6px;
        border-radius: 4px;
    }

    QListView::item:hover, QTreeView::item:hover,
    QListWidget::item:hover, QTreeWidget::item:hover {
        background-color: @surface_hover;
    }

    QListView::item:selected, QTreeView::item:selected,
    QListWidget::item:selected, QTreeWidget::item:selected {
        background-color: @selection;
        color: @text_primary;
    }

    /* ═══════════════════════════════════════════════
       TEXT EDIT & TEXT BROWSER
       ═══════════════════════════════════════════════ */

    QTextEdit, QPlainTextEdit, QTextBrowser {
        background-color: @surface_sunken;
        color: @text_primary;
        border: 1px solid @border;
        border-radius: 8px;
        padding: 4px 8px;
    }

    QTextEdit:focus, QPlainTextEdit:focus, QTextBrowser:focus {
        border: 1px solid @accent;
    }

    /* ═══════════════════════════════════════════════
       SLIDER
       ═══════════════════════════════════════════════ */

    QSlider::groove:horizontal {
        height: 4px;
        background: @input_border;
        border-radius: 2px;
    }
    QSlider::sub-page:horizontal {
        background: @accent;
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: @solid;
        border: 1px solid @border_strong;
        width: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }
    QSlider::handle:horizontal:hover {
        border: 1px solid @border_strong;
    }
    QSlider::handle:horizontal:pressed {
        background: @accent;
        border: 1px solid @accent;
    }

    QSlider::groove:vertical {
        width: 4px;
        background: @input_border;
        border-radius: 2px;
    }
    QSlider::add-page:vertical {
        background: @accent;
        border-radius: 2px;
    }
    QSlider::handle:vertical {
        background: @solid;
        border: 1px solid @border_strong;
        height: 14px;
        margin: 0 -5px;
        border-radius: 7px;
    }
    QSlider::handle:vertical:hover {
        border: 1px solid @border_strong;
    }
    QSlider::handle:vertical:pressed {
        background: @accent;
        border: 1px solid @accent;
    }

    QRangeSlider, QLabeledDoubleRangeSlider {
        qproperty-barColor: @accent;
    }

    /* ═══════════════════════════════════════════════
       FRAME
       ═══════════════════════════════════════════════ */

    QFrame[frameShape="4"],    /* HLine */
    QFrame[frameShape="5"] {   /* VLine */
        color: @border;
    }

    /* ═══════════════════════════════════════════════
       AI CHAT PANEL
       ═══════════════════════════════════════════════ */

    #chatHeader {
        background: @sidebar_bg;
        border-bottom: 1px solid @border;
    }

    #chatTitleEdit {
        background: transparent;
        border: none;
        color: @text_primary;
        font-weight: bold;
    }
    #chatTitleEdit:hover, #chatTitleEdit:focus {
        background: @input_bg;
        border-radius: 4px;
    }

    #replyPreview {
        background: @accent_tint;
        border: 1px solid @border_subtle;
        border-left: 3px solid @accent;
        border-radius: 6px;
    }

    #chatComposerContainer {
        background: transparent;
    }

    #chatComposerFrame {
        background: @surface;
        border: 1px solid @border_strong;
        border-radius: 8px;
    }

    #chatComposerFrame:hover {
        border: 1px solid @accent;
    }

    #chatComposerInput {
        background: transparent;
        border: none;
        color: @text_primary;
        font-size: 12px;
        padding: 2px;
    }

    QComboBox#cbb_provider, QComboBox#cbb_model {
        background: @input_bg;
        border: 1px solid @input_border;
        border-radius: 5px;
        padding: 1px 6px;
        font-size: 11px;
        color: @text_primary;
        min-height: 21px;
        max-height: 23px;
    }
    QComboBox#cbb_provider:hover, QComboBox#cbb_model:hover {
        background: @surface_hover;
        border: 1px solid @border_strong;
    }
    QComboBox#cbb_model QLineEdit {
        background: transparent;
        border: none;
        color: @text_primary;
        font-size: 11px;
        padding: 0px;
    }

    #btn_mic {
        background: transparent;
        border: none;
        border-radius: 5px;
        padding: 0px;
        font-size: 13px;
        color: @text_primary;
    }
    #btn_mic:hover {
        background: @surface_hover;
    }
    #btn_mic:pressed {
        background: @surface_pressed;
    }
    #btn_mic:checked {
        background: @error;
        color: white;
    }

    #card_user {
        background: @card_user_bg;
        border: 1px solid @card_user_border;
        border-radius: 8px;
    }
    #card_assistant {
        background: @card_assistant_bg;
        border: 1px solid @card_assistant_border;
        border-radius: 8px;
    }
    #card_error {
        background: @error_tint;
        border: 1px solid @error;
        border-radius: 8px;
    }

    #btn_refresh_models, #btn_history, #btn_new_chat, #btn_clear, #btn_templates, #btn_reasoning_toggle {
        background: transparent;
        border: none;
        border-radius: 4px;
    }
    #btn_refresh_models:hover, #btn_history:hover, #btn_new_chat:hover, #btn_clear:hover,
    #btn_templates:hover, #btn_reasoning_toggle:hover {
        background: @surface_hover;
    }
    #btn_reasoning_toggle:checked {
        background: @selection;
        border: 1px solid @accent;
    }

    #chatThinkingDots {
        color: @text_secondary;
    }

    #reasoningView {
        background: @surface_sunken;
        border: none;
        border-left: 2px solid @border_strong;
        border-radius: 0px;
        color: @text_secondary;
        font-style: italic;
    }

    #btnRowPrimary {
        background: @accent;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: bold;
    }
    #btnRowPrimary:hover {
        background: @accent_hover;
    }
    #btnRowLink {
        background: transparent;
        color: @accent_hover;
        text-decoration: underline;
        border: none;
    }
    #btnRowLink:hover {
        color: @accent;
    }
    #btnRowDanger {
        background: transparent;
        color: @error;
        border: none;
    }
    #btnRowDanger:hover {
        color: @error;
    }
    #btnDeleteAllOutline {
        background: transparent;
        color: @error;
        border: 1px solid @error;
        border-radius: 4px;
        padding: 4px 12px;
    }
    #btnDeleteAllOutline:hover {
        background: @error;
        color: white;
    }

    #historyDateHeader {
        color: @text_secondary;
        font-weight: bold;
        font-size: 11px;
        padding: 4px 2px;
    }

    #scrollToBottomBtn {
        background: @accent;
        color: white;
        border: none;
        border-radius: 16px;
    }
    #scrollToBottomBtn:hover {
        background: @accent_hover;
    }
"""

_QSS_TEMPLATE = QSS_TEMPLATE


# ─────────────────────────────────────────────────────────────
#  Token dictionary builder
# ─────────────────────────────────────────────────────────────

def _build_token_dict(tokens: "ThemeTokens") -> dict[str, str]:
    """Map ThemeTokens attributes and vector icon asset URLs to template placeholder names."""
    is_dark = tokens.is_dark
    d: dict[str, str] = {
        # Core PLUME design tokens
        "background": tokens.background,
        "rail_bg": tokens.rail_bg,
        "sidebar_bg": tokens.sidebar_bg,
        "surface": tokens.surface,
        "surface_sunken": tokens.surface_sunken,
        "surface_elevated": tokens.surface_elevated,
        "surface_hover": tokens.surface_hover,
        "surface_pressed": tokens.surface_pressed,
        "input_bg": tokens.input_bg,
        "input_border": tokens.input_border,
        "border": tokens.border,
        "border_subtle": tokens.border_subtle,
        "border_strong": tokens.border_strong,
        "card_user_bg": tokens.card_user_bg,
        "card_user_border": tokens.card_user_border,
        "card_assistant_bg": tokens.card_assistant_bg,
        "card_assistant_border": tokens.card_assistant_border,
        "text_primary": tokens.text_primary,
        "text_secondary": tokens.text_secondary,
        "text_muted": tokens.text_muted,
        "solid": tokens.solid,
        "solid_hover": tokens.solid_hover,
        "solid_pressed": tokens.solid_pressed,
        "on_solid": tokens.on_solid,
        "accent": tokens.accent,
        "accent_hover": tokens.accent_hover,
        "accent_pressed": tokens.accent_pressed,
        "accent_tint": tokens.accent_tint,
        "selection": tokens.selection,
        "success": tokens.success,
        "success_tint": tokens.success_tint,
        "warning": tokens.warning,
        "warning_tint": tokens.warning_tint,
        "error": tokens.error,
        "error_tint": tokens.error_tint,
        "scrollbar_thumb": tokens.scrollbar_thumb,
        "scrollbar_thumb_hover": tokens.scrollbar_thumb_hover,
        "progress_text": tokens.progress_text,
        "checkbox_checked_image": tokens.checkbox_checked_image,
        "icon_color": tokens.icon_color,
        "is_dark": str(tokens.is_dark),

        # SVG subcontrol asset URLs
        "checkbox_checked_url": _icon_url("check_white.svg"),
        "chevron_down_url": _icon_url("chevron_down_dark.svg" if is_dark else "chevron_down_light.svg"),
        "chevron_up_url": _icon_url("chevron_up_dark.svg" if is_dark else "chevron_up_light.svg"),
        "splitter_grip_v_url": _icon_url("splitter_grip_v_dark.svg" if is_dark else "splitter_grip_v_light.svg"),
        "splitter_grip_h_url": _icon_url("splitter_grip_h_dark.svg" if is_dark else "splitter_grip_h_light.svg"),
        "icon_dir": ICON_DIR_QT,

        # Backward compatibility mappings
        "surface_bg": tokens.background,
        "surface_card": tokens.surface,
        "surface_card_border": tokens.border,
        "surface_input": tokens.input_bg,
        "surface_overlay": tokens.surface_elevated,
        "surface_list": tokens.surface,
        "surface_table": tokens.surface,
        "surface_table_alt": tokens.surface_elevated,
        "surface_toolbar": tokens.sidebar_bg,
        "text_disabled": tokens.text_muted,
        "text_placeholder": tokens.text_muted,
        "border_medium": tokens.border_strong,
        "border_separator": tokens.border,
        "accent_light": tokens.accent_hover,
        "accent_bg": tokens.accent_tint,
        "accent_bg_strong": tokens.selection,
        "accent_border": tokens.accent,
        "hover_bg": tokens.surface_hover,
        "hover_bg_strong": tokens.surface_pressed,
        "pressed_bg": tokens.surface_pressed,
        "selected_bg": tokens.surface_elevated,
        "selected_border": tokens.border_subtle,
        "slider_groove": tokens.input_border,
        "slider_filled": tokens.accent,
        "slider_handle": tokens.solid,
        "slider_handle_border": tokens.border_strong,
        "progress_bg": tokens.surface_sunken,
        "progress_chunk_start": tokens.accent,
        "progress_chunk_mid": tokens.accent,
        "progress_chunk_end": tokens.accent,
        "checkbox_bg": tokens.input_bg,
        "checkbox_border": tokens.border_strong,
        "checkbox_checked_bg": tokens.accent,
        "checkbox_checked_border": tokens.accent,
        "checkbox_disabled_bg": tokens.surface_sunken,
        "checkbox_disabled_border": tokens.border,
        "checkbox_checked_disabled_bg": tokens.surface_elevated,
        "checkbox_checked_disabled_border": tokens.border,
        "tab_bg": tokens.background,
        "tab_border": tokens.border_subtle,
        "tab_text": tokens.text_secondary,
        "tab_selected_bg": tokens.surface_elevated,
        "tab_selected_border": tokens.border_subtle,
        "tab_selected_bottom": tokens.surface_elevated,
        "tab_hover_bg": tokens.surface_hover,
        "tab_hover_text": tokens.text_primary,
        "tab_pane_bg": tokens.surface,
        "tab_pane_border": tokens.border,
        "tooltip_bg": tokens.surface_elevated,
        "tooltip_border": tokens.border_strong,
        "tooltip_text": tokens.text_primary,
        "dialog_bg": tokens.background,
        "menu_bg": tokens.surface_elevated,
        "menu_border": tokens.border_strong,
        "menu_item_hover": tokens.surface_hover,
        "menu_separator": tokens.border,
        "gridline": tokens.border_subtle,
        "header_bg": tokens.surface_elevated,
        "header_border_right": tokens.border,
        "header_border_bottom": tokens.border,
        "header_text": tokens.text_muted,
        "frame_line_color": tokens.border,
        "arrow_suffix": "" if is_dark else "-dark",
    }
    return d


@lru_cache(maxsize=8)
def build_stylesheet(tokens: "ThemeTokens") -> str:
    """Render the QSS template with the given *tokens*.

    Returns a complete stylesheet string ready for ``QApplication.setStyleSheet()``.
    """
    token_dict = _build_token_dict(tokens)
    qss = QSS_TEMPLATE

    # Replace @placeholder tokens (sorted by length descending to prevent partial match)
    for key, val in sorted(token_dict.items(), key=lambda kv: len(kv[0]), reverse=True):
        qss = qss.replace(f"@{key}", str(val))
        qss = qss.replace(f"{{{key}}}", str(val))

    return qss


# ─────────────────────────────────────────────────────────────
#  PLUME-inspired Reusable Component Style Builders
# ─────────────────────────────────────────────────────────────

def solid_button_style(radius: int = 6, tokens: "ThemeTokens | None" = None) -> str:
    """Monochrome solid primary CTA button (PLUME visual language)."""
    tok = _resolve_tokens(tokens)
    return f"""
        QPushButton {{
            background-color: {tok.solid};
            color: {tok.on_solid};
            border: 1px solid {tok.solid};
            border-radius: {radius}px;
            padding: 4px 14px;
            font-weight: 600;
        }}
        QPushButton:hover {{
            background-color: {tok.solid_hover};
            border-color: {tok.solid_hover};
        }}
        QPushButton:pressed {{
            background-color: {tok.solid_pressed};
            border-color: {tok.solid_pressed};
        }}
        QPushButton:disabled {{
            background-color: {tok.surface_elevated};
            color: {tok.text_muted};
            border-color: {tok.border};
        }}
    """


def primary_button_style(radius: int = 6, tokens: "ThemeTokens | None" = None) -> str:
    """Accent-colored CTA button."""
    tok = _resolve_tokens(tokens)
    return f"""
        QPushButton {{
            background-color: {tok.accent};
            color: #FFFFFF;
            border: 1px solid {tok.accent};
            border-radius: {radius}px;
            padding: 4px 14px;
            font-weight: 600;
        }}
        QPushButton:hover {{
            background-color: {tok.accent_hover};
            border-color: {tok.accent_hover};
        }}
        QPushButton:pressed {{
            background-color: {tok.accent_pressed};
            border-color: {tok.accent_pressed};
        }}
        QPushButton:disabled {{
            background-color: {tok.surface_elevated};
            color: {tok.text_muted};
            border-color: {tok.border};
        }}
    """


def ghost_button_style(radius: int = 6, tokens: "ThemeTokens | None" = None) -> str:
    """Transparent background secondary/icon action button."""
    tok = _resolve_tokens(tokens)
    return f"""
        QPushButton {{
            background-color: transparent;
            color: {tok.text_secondary};
            border: 1px solid transparent;
            border-radius: {radius}px;
            padding: 4px 10px;
        }}
        QPushButton:hover {{
            background-color: {tok.surface_hover};
            color: {tok.text_primary};
            border-color: {tok.border_subtle};
        }}
        QPushButton:pressed {{
            background-color: {tok.surface_pressed};
            color: {tok.text_primary};
        }}
        QPushButton:disabled {{
            color: {tok.text_muted};
            background-color: transparent;
            border-color: transparent;
        }}
    """


def danger_button_style(radius: int = 6, tokens: "ThemeTokens | None" = None) -> str:
    """Destructive action button with red error accent."""
    tok = _resolve_tokens(tokens)
    return f"""
        QPushButton {{
            background-color: transparent;
            color: {tok.error};
            border: 1px solid {tok.error};
            border-radius: {radius}px;
            padding: 4px 12px;
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: {tok.error};
            color: #FFFFFF;
        }}
        QPushButton:pressed {{
            background-color: {tok.error_tint};
            color: {tok.error};
        }}
        QPushButton:disabled {{
            color: {tok.text_muted};
            border-color: {tok.border};
            background-color: transparent;
        }}
    """


def card_frame_style(
    object_name: str,
    radius: int = 8,
    elevated: bool = False,
    tokens: "ThemeTokens | None" = None,
) -> str:
    """Themed card container QSS rule for a specific QFrame objectName."""
    tok = _resolve_tokens(tokens)
    bg = tok.surface_elevated if elevated else tok.surface
    return f"""
        QFrame#{object_name} {{
            background-color: {bg};
            border: 1px solid {tok.border};
            border-radius: {radius}px;
        }}
    """


def badge_style(kind: str = "accent", tokens: "ThemeTokens | None" = None) -> str:
    """Status or category pill badge styling."""
    tok = _resolve_tokens(tokens)
    if kind == "success":
        bg, fg, border = tok.success_tint, tok.success, tok.success
    elif kind == "warning":
        bg, fg, border = tok.warning_tint, tok.warning, tok.warning
    elif kind == "error":
        bg, fg, border = tok.error_tint, tok.error, tok.error
    elif kind == "muted":
        bg, fg, border = tok.surface_elevated, tok.text_muted, tok.border_subtle
    else:  # "accent" or default
        bg, fg, border = tok.accent_tint, tok.accent, tok.accent

    return f"""
        background-color: {bg};
        color: {fg};
        border: 1px solid {border};
        border-radius: 9px;
        padding: 2px 7px;
        font-size: 11px;
        font-weight: 500;
    """
