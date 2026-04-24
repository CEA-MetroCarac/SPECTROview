from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt

class VMVA(QWidget):
    """View for Multivariate Analysis (MVA) feature."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Placeholder label
        label = QLabel("Multivariate Analyses (MVA) feature are underdevelopement")
        label.setAlignment(Qt.AlignCenter)
        
        # Optional: styling
        font = label.font()
        font.setPointSize(12)
        font.setItalic(True)
        label.setFont(font)
        
        layout.addWidget(label)
