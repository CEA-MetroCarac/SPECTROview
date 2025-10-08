import os
from PySide6.QtWidgets import (
    QWidget, QLabel, QComboBox, QPushButton, QHBoxLayout, QFileDialog, QMessageBox
)
from PySide6.QtCore import QSettings
from PySide6.QtGui import  QIcon, Qt 

from spectroview import ICON_DIR

def show_alert(msg):
    """Simple alert dialog"""
    QMessageBox.warning(None, "Warning", msg)


class FitModelManager(QWidget):
    """Self-contained Fit Model Manager GUI widget"""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.settings = QSettings("CEA-Leti", "SPECTROview")
        self.default_model_folder = self.settings.value("model_folder", "")
        self.available_models = []

        # --- GUI Elements ---
        self.label = QLabel("Fit model:")
        self.label.setToolTip("Select a model for fitting")
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)    
        self.label.setFixedWidth(70)

        self.combo_models = QComboBox()
        self.combo_models.setToolTip("Select a model for fitting")

        self.btn_apply = QPushButton("Apply")
        icon = QIcon()
        icon.addFile(os.path.join(ICON_DIR, "done.png"))
        self.btn_apply.setIcon(icon)
        self.btn_apply.setToolTip("Apply the selected fit model to the spectra")
        self.btn_apply.setFixedWidth(80)
        
        self.btn_load = QPushButton("Load")
        icon = QIcon()
        icon.addFile(os.path.join(ICON_DIR, "load.png"))
        self.btn_load.setIcon(icon) 
        self.btn_load.setToolTip("Load a new fit model from file")
        self.btn_load.setFixedWidth(80)
        
        # Layout
        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.combo_models)
        layout.addWidget(self.btn_apply)
        layout.addWidget(self.btn_load)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Populate models if folder exists
        if self.default_model_folder:
            self.scan_models()

    # -----------------------------
    # Folder & Model Management
    # -----------------------------
    def set_default_model_folder(self, folder_path=None):
        """Set the default folder path for models (optional folder dialog)"""
        if folder_path is None:
            folder_path = QFileDialog.getExistingDirectory(
                self, "Select Default Folder", options=QFileDialog.ShowDirsOnly
            )

        if not folder_path:
            return  # User canceled

        self.default_model_folder = folder_path
        self.settings.setValue("model_folder", folder_path)
        self.settings.sync()
        self.scan_models()

    def scan_models(self):
        """Scan the folder and populate the combobox"""
        self.available_models = []
        self.combo_models.clear()

        if not self.default_model_folder or not os.path.exists(self.default_model_folder):
            show_alert(f"Folder '{self.default_model_folder}' not found. Please select another folder.")
            return

        try:
            for fname in os.listdir(self.default_model_folder):
                if fname.endswith(".json"):
                    self.available_models.append(fname)
        except Exception as e:
            print(f"Error scanning folder '{self.default_model_folder}': {e}")
            return

        if self.available_models:
            self.combo_models.addItems(self.available_models)

    def get_selected_model_path(self):
        """Return full path of selected model"""
        model_name = self.combo_models.currentText()
        if not model_name:
            return None
        return os.path.join(self.default_model_folder, model_name)

    # -----------------------------
    # Connect callbacks
    # -----------------------------
    def connect_apply(self, callback):
        """Connect the Apply button to a custom function"""
        self.btn_apply.clicked.connect(callback)

    def connect_load(self, callback):
        """Connect the Load button to a custom function"""
        self.btn_load.clicked.connect(callback)
