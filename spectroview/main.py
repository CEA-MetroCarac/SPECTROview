# main.py
import sys
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtGui import QIcon

from spectroview import LOGO_APPLI
from spectroview.view.main_view import MainView


def launcher():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(LOGO_APPLI))
    app.setStyle("Fusion")

    window = MainView()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launcher()
