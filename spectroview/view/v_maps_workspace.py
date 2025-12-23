from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt


class VMapsWorkspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        label = QLabel("Maps workspace")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
