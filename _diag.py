"""Diagnostic: check DPI, Qt version, and visually test MDI title bar blur."""
import sys, os
import PySide6
from PySide6.QtWidgets import QApplication, QMdiArea, QMdiSubWindow, QTextEdit
from PySide6.QtGui import QPalette, QColor, QPainter, QPaintEvent
from PySide6.QtCore import Qt

class InstrumentedMdiSubWindow(QMdiSubWindow):
    """MdiSubWindow that logs every paintEvent call."""
    paint_count = 0
    def paintEvent(self, event):
        self.paint_count += 1
        if self.paint_count <= 5:
            print(f"  paintEvent #{self.paint_count}  rect={event.rect()}  active={self.isActiveWindow()}")
        super().paintEvent(event)

app = QApplication(sys.argv)
app.setStyle("Fusion")

screen = app.primaryScreen()
print("=== ENVIRONMENT ===")
print(f"PySide6 version : {PySide6.__version__}")
print(f"Qt version      : {PySide6.QtCore.__version__}")
print(f"Logical DPI     : {screen.logicalDotsPerInch()}")
print(f"Physical DPI    : {screen.physicalDotsPerInch()}")
print(f"Device pixel rat: {screen.devicePixelRatio()}")

# Dark palette (same as SPECTROview dark theme)
p = QPalette()
p.setColor(QPalette.Window,        QColor(33, 33, 33))
p.setColor(QPalette.Base,          QColor(60, 60, 60))
p.setColor(QPalette.AlternateBase, QColor(68, 68, 68))
p.setColor(QPalette.WindowText,    QColor(240, 240, 240))
p.setColor(QPalette.Text,          QColor(240, 240, 240))
p.setColor(QPalette.ButtonText,    QColor(235, 235, 235))
p.setColor(QPalette.Button,        QColor(58, 58, 58))
p.setColor(QPalette.Light,         QColor(85, 85, 85))
p.setColor(QPalette.Mid,           QColor(68, 68, 68))
p.setColor(QPalette.Dark,          QColor(38, 38, 38))
p.setColor(QPalette.Shadow,        QColor(18, 18, 18))
p.setColor(QPalette.Highlight,     QColor(212, 160, 74))
p.setColor(QPalette.HighlightedText, Qt.white)
app.setPalette(p)

mdi = QMdiArea()
mdi.setWindowTitle("MDI Title Bar Blur Test")
mdi.resize(800, 600)

print("\n=== CREATING SUB-WINDOWS ===")
sub1 = InstrumentedMdiSubWindow()
sub1.setWindowTitle("Active Window - Check if this title is blurry")
sub1.setWidget(QTextEdit("This window should be ACTIVE (selected)"))
mdi.addSubWindow(sub1)
sub1.resize(380, 280)

sub2 = InstrumentedMdiSubWindow()
sub2.setWindowTitle("Inactive Window - This should be crisp")
sub2.setWidget(QTextEdit("This window should be INACTIVE"))
mdi.addSubWindow(sub2)
sub2.resize(380, 280)
sub2.move(50, 50)

sub1.show()
sub2.show()
mdi.setActiveSubWindow(sub1)
mdi.show()

print("\n=== COMPARE THE TWO TITLE BARS ===")
print("Is the active window title blurry? If yes, it is a Qt/Fusion issue.")
print("Close the window when done observing.")

app.exec()
