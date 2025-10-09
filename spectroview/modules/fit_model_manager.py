import os
from PySide6.QtWidgets import (
    QWidget, QLabel, QComboBox, QPushButton, QHBoxLayout, QFileDialog
)
from PySide6.QtCore import QSettings, QFileInfo
from PySide6.QtGui import  QIcon, Qt 

from spectroview import ICON_DIR

class FitModelManager(QWidget):
    """Self-contained Fit Model Manager GUI widget"""

    def __init__(self):
        super().__init__()
        self.settings = QSettings("CEA-Leti", "SPECTROview")

        self.default_model_folder = self.settings.value("model_folder", "")
        self.available_models = []
        self.loaded_fit_model = None # Full path of the currently loaded model

        self._create_ui()

        if self.default_model_folder:  # Populate models if folder exists
            self.scan_and_udp_model_cbb()

    def scan_and_udp_model_cbb(self):
        """Scan the folder and populate all fit model into the combobox"""
        self.available_models = []
        self.combo_models.clear()

        if not self.default_model_folder or not os.path.exists(self.default_model_folder):
            print(f"Folder '{self.default_model_folder}' not found. Please select another folder.")
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
    
    def refresh_cbb(self):
        """Manually refresh the model list (used by Refresh button)."""
        self.default_model_folder = self.settings.value("model_folder", "")
        self.scan_and_udp_model_cbb()
    
    def load_fit_model(self):
        """Select a JSON model file and add it to the combo box."""
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        selected_file, _ = QFileDialog.getOpenFileName(
            self, "Select JSON Model File", "",
            "JSON Files (*.json);;All Files (*)", options=options)
        if not selected_file:
            return None
        display_name = QFileInfo(selected_file).fileName()
        if display_name not in [self.combo_models.itemText(i) for i in range(self.combo_models.count())]:
            self.combo_models.addItem(display_name)
        self.combo_models.setCurrentText(display_name)
        self.loaded_fit_model = selected_file
        return self.loaded_fit_model

    def connect_apply(self, callback):
        """Connect the Apply button to a custom function"""
        self.btn_apply.clicked.connect(callback)

    def _create_ui(self):
        """Create the GUI elements and layout"""
        self.label = QLabel("Fit model:")
        self.label.setToolTip("Select a model for fitting")
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label.setFixedWidth(60)

        self.combo_models = QComboBox()
        self.combo_models.setToolTip("Select a model for fitting")

        # --- Apply Button ---
        self.btn_apply = QPushButton("Apply")
        icon = QIcon(os.path.join(ICON_DIR, "done.png"))
        self.btn_apply.setIcon(icon)
        self.btn_apply.setToolTip("Apply the selected fit model to the spectra")
        self.btn_apply.setFixedWidth(70)

        # --- Load Button ---
        self.btn_load = QPushButton("Load")
        icon = QIcon(os.path.join(ICON_DIR, "load.png"))
        self.btn_load.setIcon(icon)
        self.btn_load.setToolTip("Load a new fit model from file")
        self.btn_load.setFixedWidth(70)
        self.btn_load.clicked.connect(self.load_fit_model)

        # --- Refresh Button ---
        self.btn_refresh = QPushButton("")
        icon = QIcon(os.path.join(ICON_DIR, "refresh.png"))
        self.btn_refresh.setIcon(icon)
        self.btn_refresh.setToolTip("Reload available fit models from the default folder")
        self.btn_refresh.setFixedWidth(25)
        self.btn_refresh.clicked.connect(self.refresh_cbb)

        # --- Layout ---
        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.combo_models)
        layout.addWidget(self.btn_apply)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_refresh)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.setLayout(layout)