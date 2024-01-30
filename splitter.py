from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, \
    QSplitter, QLabel
from PySide6.QtCore import Qt


class YourMainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Create your layouts
        upper_layout = QHBoxLayout()
        bottom_layout = QHBoxLayout()

        # Add widgets to your layouts (replace these with your actual widgets)
        upper_layout.addWidget(QLabel("Upper Zone Content"))
        bottom_layout.addWidget(QLabel("Bottom Zone Content"))

        # Create widgets for the splitter
        upper_widget = QWidget()
        upper_widget.setLayout(upper_layout)

        bottom_widget = QWidget()
        bottom_widget.setLayout(bottom_layout)

        # Create splitter and add widgets to it
        splitter = QSplitter(
            Qt.Horizontal)  # Use Qt.Vertical for a vertical splitter
        splitter.addWidget(upper_widget)
        splitter.addWidget(bottom_widget)

        # Create a main layout for the central widget
        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)

        # Set the main layout for the central widget
        self.setLayout(main_layout)

        # Set the main window properties
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Your Main Window')


if __name__ == '__main__':
    app = QApplication([])
    window = YourMainWindow()
    window.show()
    app.exec_()
