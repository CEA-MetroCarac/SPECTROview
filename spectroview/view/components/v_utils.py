# spectroview/view/v_utils.py
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt


def dark_palette():
    """Dark palette tuned for SPECTROview UI"""

    p = QPalette()

    # ---------- Base surfaces ----------
    p.setColor(QPalette.Window, QColor(53, 53, 53))          # main background
    p.setColor(QPalette.Base, QColor(42, 42, 42))            # lists, tables, editors
    p.setColor(QPalette.AlternateBase, QColor(48, 48, 48))   # alternating rows

    # ---------- Text ----------
    p.setColor(QPalette.WindowText, Qt.white)
    p.setColor(QPalette.Text, Qt.white)
    p.setColor(QPalette.ButtonText, Qt.white)
    p.setColor(QPalette.PlaceholderText, QColor(140, 140, 140))

    # ---------- Buttons / controls ----------
    p.setColor(QPalette.Button, QColor(64, 64, 64))
    p.setColor(QPalette.Light, QColor(90, 90, 90))
    p.setColor(QPalette.Mid, QColor(72, 72, 72))
    p.setColor(QPalette.Dark, QColor(40, 40, 40))
    p.setColor(QPalette.Shadow, QColor(20, 20, 20))

    # ---------- Tooltips ----------
    p.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    p.setColor(QPalette.ToolTipText, Qt.black)

    # ---------- Highlights / accent ----------
    accent = QColor(42, 130, 218)  # Qt blue (matches screenshot)
    p.setColor(QPalette.Highlight, accent)
    p.setColor(QPalette.HighlightedText, Qt.white)
    p.setColor(QPalette.Link, accent)

    # ---------- Disabled ----------
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(130, 130, 130))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(130, 130, 130))
    p.setColor(QPalette.Disabled, QPalette.WindowText, QColor(130, 130, 130))

    return p

def light_palette():
    """Light palette with soft blue accent"""

    p = QPalette()

    # ---- Base colors ----
    p.setColor(QPalette.Window, QColor(245, 246, 248))        # main background
    p.setColor(QPalette.Base, QColor(255, 255, 255))          # inputs, tables
    p.setColor(QPalette.AlternateBase, QColor(238, 240, 243)) # alternate rows

    # ---- Text ----
    p.setColor(QPalette.WindowText, QColor(30, 30, 30))
    p.setColor(QPalette.Text, QColor(30, 30, 30))
    p.setColor(QPalette.ButtonText, QColor(30, 30, 30))
    p.setColor(QPalette.PlaceholderText, QColor(150, 150, 150))

    # ---- Buttons ----
    p.setColor(QPalette.Button, QColor(235, 236, 239))
    p.setColor(QPalette.Light, QColor(255, 255, 255))
    p.setColor(QPalette.Midlight, QColor(220, 220, 220))
    p.setColor(QPalette.Mid, QColor(200, 200, 200))
    p.setColor(QPalette.Dark, QColor(160, 160, 160))

    # ---- Blue accent ----
    accent = QColor(64, 156, 255)  # soft modern blue
    accent_hover = QColor(90, 170, 255)

    p.setColor(QPalette.Highlight, accent)
    p.setColor(QPalette.HighlightedText, Qt.white)
    p.setColor(QPalette.Link, accent)

    # ---- Tooltips ----
    p.setColor(QPalette.ToolTipBase, QColor(255, 255, 240))
    p.setColor(QPalette.ToolTipText, QColor(20, 20, 20))

    # ---- Disabled state ----
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(160, 160, 160))
    p.setColor(QPalette.Disabled, QPalette.WindowText, QColor(160, 160, 160))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(160, 160, 160))

    return p