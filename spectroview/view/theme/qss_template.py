# spectroview/view/theme/qss_template.py
"""
Single parameterized QSS template for SPECTROview.

Replaces the three duplicated stylesheet functions (dark, soft_dark, light)
with one template that accepts a ``ThemeTokens`` instance.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tokens import ThemeTokens

from spectroview import ICON_DIR

ICON_DIR_QT = ICON_DIR.replace("\\", "/")

# ─────────────────────────────────────────────────────────────
#  The template uses Python str.format_map() placeholders.
#  Every {name} corresponds to a ThemeTokens field.
# ─────────────────────────────────────────────────────────────

_QSS_TEMPLATE = r"""

    /* ═══════════════════════════════════════════════
       GLOBAL DEFAULTS
       ═══════════════════════════════════════════════ */

    QWidget {{
        font-family: "Helvetica Neue", Helvetica, Arial;
        font-size: 13px;
    }}

    QMainWindow, QDialog {{
        background-color: {surface_bg};
    }}

    QToolBar QToolButton {{
        background: transparent;
        border: none;
    }}
    QToolBar QToolButton:hover {{
        background: {hover_bg};
        border-radius: 4px;
    }}
    QToolBar QToolButton:checked {{
        background: {accent_bg_strong};
        border: 1px solid {accent_border};
        border-radius: 4px;
    }}
    QToolBar QToolButton:pressed {{
        background: {pressed_bg};
        border: 1px solid {accent_border};
        border-radius: 4px;
    }}


    /* ═══════════════════════════════════════════════
       MAIN TAB WIDGET — workspace tabs
       ═══════════════════════════════════════════════ */

    QTabWidget::pane {{
        margin-top: 2px;
        border: 1px solid {tab_pane_border};
        border-radius: 8px;
        background: {tab_pane_bg};
        top: -1px;
    }}

    QTabBar::tab {{
        margin-top: 4px;
        background: {tab_bg};
        border: 1px solid {tab_border};
        border-bottom: none;
        border-top-left-radius: 7px;
        border-top-right-radius: 7px;
        padding: 3px;
        margin-right: 2px;
        color: {tab_text};
        min-width: 70px;
    }}

    QTabBar::tab:selected {{
        background: {tab_selected_bg};
        border: 1px solid {tab_selected_border};
        border-bottom: 1px solid {tab_selected_bottom};
        color: {text_primary};
        font-weight: bold;
    }}

    QTabBar::tab:hover:!selected {{
        background: {tab_hover_bg};
        color: {tab_hover_text};
    }}

    /* ═══════════════════════════════════════════════
       TOOLBAR — top menu bar
       ═══════════════════════════════════════════════ */

    QToolBar {{
        background: {surface_toolbar};
        border-bottom: 1px solid {border_subtle};
        spacing: 3px;
        padding: 3px;
    }}

    QToolBar::separator {{
        width: 1px;
        background: {border_separator};
        margin: 4px 6px;
    }}


    /* ═══════════════════════════════════════════════
       GROUP BOXES — glass cards
       ═══════════════════════════════════════════════ */

    QGroupBox {{
        background: {surface_card};
        border: 1px solid {surface_card_border};
        border-radius: 8px;
        margin-top: 10px;
        padding: 8px 3px 3px 3px;
        font-weight: bold;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top center;
        padding: 2px 3px;
        color: {text_secondary};
    }}

    #controlBarPanel, #shiftPanel, #bottomToolbarPanel {{
        background: {surface_card};
        border: 1px solid {surface_card_border};
        border-radius: 6px;
        margin: 0px;
        padding: 0px;
    }}

    /* ═══════════════════════════════════════════════
       BUTTONS — frosted pill / rounded style
       ═══════════════════════════════════════════════ */

    QPushButton, QToolButton {{
        background: {surface_card};
        border: 1px solid {border_subtle};
        border-radius: 6px;
        padding: 3px;
        color: {text_primary};
    }}

    QPushButton:hover, QToolButton:hover {{
        background: {hover_bg_strong};
        border: 1px solid {border_medium};
    }}

    QPushButton:pressed, QToolButton:pressed {{
        background: {pressed_bg};
        border: 1px solid {accent_border};
    }}

    QPushButton:disabled, QToolButton:disabled {{
        color: {text_disabled};
        background: transparent;
        border: 1px solid {checkbox_disabled_border};
    }}

    QPushButton:checked, QToolButton:checked {{
        background: {accent_bg_strong};
        border: 1px solid {accent_border};
        color: {text_primary};
    }}

    /* ═══════════════════════════════════════════════
       INPUT FIELDS — subtle inset glass
       ═══════════════════════════════════════════════ */

    QLineEdit {{
        background: {surface_input};
        border: 1px solid {border_subtle};
        border-radius: 5px;
        padding: 3px 4px;
        color: {text_primary};
        selection-background-color: {accent_bg};
    }}

    QLineEdit:focus {{
        border: 1px solid {border_medium};
    }}


    /* ═══════════════════════════════════════════════
       COMBOBOX & SPINBOX
       ═══════════════════════════════════════════════ */
    QComboBox, QSpinBox, QDoubleSpinBox {{
        background: {surface_input};
        border: 1px solid {border_subtle};
        border-radius: 5px;
        padding: 3px 5px;
        color: {text_primary};
    }}
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 1px solid {border_medium};
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        background: transparent;
        border-left: 1px solid {border_subtle};
        width: 16px;
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        subcontrol-origin: border;
        subcontrol-position: top right;
        background: transparent;
        border-left: 1px solid {border_subtle};
        width: 16px;
    }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        background: transparent;
        border-left: 1px solid {border_subtle};
        width: 16px;
    }}
    QComboBox::drop-down:pressed,
    QSpinBox::up-button:pressed, QSpinBox::down-button:pressed,
    QDoubleSpinBox::up-button:pressed, QDoubleSpinBox::down-button:pressed {{
        background: {hover_bg_strong};
    }}

    QComboBox::down-arrow {{
        image: url({icon_dir}/arrow-down{arrow_suffix}.svg);
        width: 10px; height: 10px;
    }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        image: url({icon_dir}/arrow-up{arrow_suffix}.svg);
        width: 9px; height: 9px;
    }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        image: url({icon_dir}/arrow-down{arrow_suffix}.svg);
        width: 9px; height: 9px;
    }}


    /* ═══════════════════════════════════════════════
       CHECK BOX & LIST INDICATORS
       ═══════════════════════════════════════════════ */
    QCheckBox::indicator, QListView::indicator,
    QTreeView::indicator, QGroupBox::indicator {{
        width: 14px;
        height: 14px;
        border-radius: 3px;
        background: {checkbox_bg};
        border: 1px solid {checkbox_border};
    }}
    QCheckBox::indicator:checked, QListView::indicator:checked,
    QTreeView::indicator:checked, QGroupBox::indicator:checked {{
        background: {checkbox_checked_bg};
        border: 1px solid {checkbox_checked_border};
    }}
    QCheckBox::indicator:hover, QListView::indicator:hover,
    QTreeView::indicator:hover, QGroupBox::indicator:hover {{
        border: 1px solid {border_strong};
    }}


    /* ═══════════════════════════════════════════════
       CHECK BOX
       ═══════════════════════════════════════════════ */

    QCheckBox {{
        spacing: 6px;
        color: {text_primary};
    }}
    QCheckBox:disabled {{
        color: {text_disabled};
    }}
    QCheckBox::indicator:disabled, QListView::indicator:disabled,
    QTreeView::indicator:disabled, QGroupBox::indicator:disabled {{
        background: {checkbox_disabled_bg};
        border: 1px solid {checkbox_disabled_border};
    }}
    QCheckBox::indicator:checked:disabled, QListView::indicator:checked:disabled,
    QTreeView::indicator:checked:disabled, QGroupBox::indicator:checked:disabled {{
        background: {checkbox_checked_disabled_bg};
        border: 1px solid {checkbox_checked_disabled_border};
    }}



    /* ═══════════════════════════════════════════════
       LIST WIDGETS — glass list with subtle items
       ═══════════════════════════════════════════════ */


    QScrollArea > QWidget > QWidget {{
        background: transparent;
    }}

    QListWidget, QTreeWidget, QScrollArea,
    #workspaceRightPanel {{
        background: {surface_list};
        border: 1px solid {border_subtle};
        border-radius: 6px;
        color: {text_primary};
        outline: none;
    }}

    QListWidget::item {{
        padding: 3px;
        border-radius: 4px;
        margin: 1px 2px;
        border: 1px solid transparent;
    }}

    QListWidget::item:selected {{
        background: {selected_bg};
        border: 1px solid {selected_border};
        color: {text_primary};
    }}

    QListWidget::item:hover:!selected {{
        background: {hover_bg};
    }}

    QTreeWidget::item {{
        padding: 3px;
        border-radius: 3px;
        border: 1px solid transparent;
    }}

    QTreeWidget::item:selected {{
        background: {selected_bg};
        border: 1px solid {selected_border};
        color: {text_primary};
    }}

    /* ═══════════════════════════════════════════════
       TABLE VIEW / HEADER VIEW
       ═══════════════════════════════════════════════ */

    QTableWidget, QTableView {{
        background: {surface_table};
        alternate-background-color: {surface_table_alt};
        border: 1px solid {border_subtle};
        border-radius: 6px;
        gridline-color: {gridline};
        color: {text_primary};
    }}

    QHeaderView::section {{
        background: {header_bg};
        border: none;
        border-right: 1px solid {header_border_right};
        border-bottom: 1px solid {header_border_bottom};
        padding: 3px;
        color: {header_text};
        font-weight: bold;
    }}

    /* ═══════════════════════════════════════════════
       TOOLTIPS — floating glass pill
       ═══════════════════════════════════════════════ */

    QToolTip {{
        background: {tooltip_bg};
        border: 1px solid {tooltip_border};
        border-radius: 6px;
        color: {tooltip_text};
        padding: 4px 6px;
    }}

    /* ═══════════════════════════════════════════════
       SPLITTER — minimal handle
       ═══════════════════════════════════════════════ */

    QSplitter::handle {{
        background: transparent;
    }}

    QSplitter::handle:horizontal {{
        width: 4px;
    }}

    QSplitter::handle:vertical {{
        height: 4px;
    }}

    QSplitter::handle:hover {{
        background: {border_medium};
    }}

    /* ═══════════════════════════════════════════════
       SCROLLBARS — thin, modern, floating
       ═══════════════════════════════════════════════ */

    QScrollBar:vertical {{
        background: transparent;
        width: 5px;
        margin: 0;
        border: none;
    }}

    QScrollBar::handle:vertical {{
        background: {scrollbar_thumb};
        border-radius: 2px;
        min-height: 30px;
    }}

    QScrollBar::handle:vertical:hover {{
        background: {scrollbar_thumb_hover};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: none;
    }}

    QScrollBar:horizontal {{
        background: transparent;
        height: 5px;
        margin: 0;
        border: none;
    }}

    QScrollBar::handle:horizontal {{
        background: {scrollbar_thumb};
        border-radius: 2px;
        min-width: 30px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background: {scrollbar_thumb_hover};
    }}

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}

    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
        background: none;
    }}

    /* ═══════════════════════════════════════════════
       PROGRESS BAR — glowing accent
       ═══════════════════════════════════════════════ */

    QProgressBar {{
        background: {progress_bg};
        border: 1px solid {border_subtle};
        border-radius: 4px;
        text-align: center;
        color: {text_secondary};
    }}

    QProgressBar::chunk {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0   {progress_chunk_start},
                                    stop:0.5 {progress_chunk_mid},
                                    stop:1   {progress_chunk_end});
        border-radius: 3px;
    }}


    /* ═══════════════════════════════════════════════
       DIALOG — glass card
       ═══════════════════════════════════════════════ */

    QDialog {{
        background: {dialog_bg};
    }}

    /* ═══════════════════════════════════════════════
       FRAME — subtle border
       ═══════════════════════════════════════════════ */

    QFrame[frameShape="4"],    /* HLine */
    QFrame[frameShape="5"] {{  /* VLine */
        color: {frame_line_color};
    }}

    /* ═══════════════════════════════════════════════
       MENU — floating glass dropdown
       ═══════════════════════════════════════════════ */

    QMenu {{
        background: {menu_bg};
        border: 1px solid {menu_border};
        border-radius: 8px;
        padding: 4px;
        color: {text_primary};
    }}

    QMenu::item {{
        padding: 4px 12px;
        border-radius: 5px;
        margin: 1px 2px;
    }}

    QMenu::item:selected {{
        background: {menu_item_hover};
    }}

    QMenu::separator {{
        height: 1px;
        background: {menu_separator};
        margin: 4px 8px;
    }}

    /* ═══════════════════════════════════════════════
       TEXT BROWSER — for user manual & about
       ═══════════════════════════════════════════════ */

    QTextBrowser {{
        background: {surface_list};
        border: 1px solid {border_subtle};
        border-radius: 6px;
        color: {text_primary};
    }}

    /* ═══════════════════════════════════════════════
       LABEL — default transparency
       ═══════════════════════════════════════════════ */

    QLabel {{
        background: transparent;
        color: {text_primary};
    }}

    /* ═══════════════════════════════════════════════
       SLIDER
       ═══════════════════════════════════════════════ */

    QSlider::groove:horizontal {{
        height: 4px;
        background: {slider_groove};
        border-radius: 2px;
    }}
    QSlider::sub-page:horizontal {{
        background: {slider_filled};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {slider_handle};
        border: 1px solid {slider_handle_border};
        width: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }}
    QSlider::handle:horizontal:hover {{
        border: 1px solid {border_strong};
    }}
    QSlider::handle:horizontal:pressed {{
        background: {accent};
        border: 1px solid {accent};
    }}

    QSlider::groove:vertical {{
        width: 4px;
        background: {slider_groove};
        border-radius: 2px;
    }}
    QSlider::add-page:vertical {{
        background: {slider_filled};
        border-radius: 2px;
    }}
    QSlider::handle:vertical {{
        background: {slider_handle};
        border: 1px solid {slider_handle_border};
        height: 14px;
        margin: 0 -5px;
        border-radius: 7px;
    }}
    QSlider::handle:vertical:hover {{
        border: 1px solid {border_strong};
    }}
    QSlider::handle:vertical:pressed {{
        background: {accent};
        border: 1px solid {accent};
    }}

    QRangeSlider, QLabeledDoubleRangeSlider {{
        qproperty-barColor: {slider_filled};
    }}

"""


def build_stylesheet(tokens: "ThemeTokens") -> str:
    """Render the QSS template with the given *tokens*.

    Returns a complete stylesheet string ready for
    ``QApplication.setStyleSheet()``.
    """
    values = asdict(tokens)
    values["icon_dir"] = ICON_DIR_QT
    return _QSS_TEMPLATE.format_map(values)
