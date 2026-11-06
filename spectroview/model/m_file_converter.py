import os

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QMessageBox, QFileDialog, QLabel
)
from PySide6.QtCore import QFileInfo

from spectroview.model.m_io import convert_action


class MFileConverter(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.selected_file_paths = []

        self.setWindowTitle("Hyperspectral Data Converter")
        self.setMinimumWidth(450)
        self.resize(500, 350)

        # Create GUI
        self._create_ui()

        # Connect signals
        self.btn_browse.clicked.connect(lambda: self.browse_files())
        self.btn_convert.clicked.connect(lambda: self.convert())

    def _create_ui(self):
        """Builds the UI layout for the converter window."""
        main_layout = QVBoxLayout(self)

        # Instruction text
        instruction_label = QLabel(
            "This tool converts hyperspectral data obtained with a Renishaw InVia Raman spectrometer into a format compatible with SPECTROview or LabSpec6 (HORIBA).\n\n"
            "(If you have *.wdf files, convert them to *.txt format before using this tool (use the 'Save As' function in WIRE software)."
        )
        instruction_label.setWordWrap(True)
        main_layout.addWidget(instruction_label)

        # Buttons
        button_layout = QHBoxLayout()
        self.btn_browse = QPushButton("Browse")
        self.btn_convert = QPushButton("Convert")
        button_layout.addWidget(self.btn_browse)
        button_layout.addWidget(self.btn_convert)

        # File list
        self.listbox = QListWidget()

        # Assemble layout
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.listbox)
      
    def browse_files(self, file_paths=None):
        # Ensure file_paths is None or a list
        if file_paths is not None and not isinstance(file_paths, list):
            print(f"Warning: Invalid file_paths type: {type(file_paths)}. Expected list or None.")
            return

        self.selected_file_paths.clear()
        self.listbox.clear()

        # Show dialog if no file_paths provided
        if file_paths is None:
            last_dir = self.settings.get_last_directory()
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, "Open 2D Map File(s) to Convert", last_dir, "Text Files (*.txt)", options=options
            )

        # If files were selected
        if file_paths:
            last_dir = QFileInfo(file_paths[0]).absolutePath()
            self.settings.set_last_directory(last_dir)

            for file_path in file_paths:
                if os.path.isfile(file_path):
                    abs_path = os.path.abspath(file_path)
                    self.selected_file_paths.append(abs_path)
                    self.listbox.addItem(os.path.basename(abs_path))

    def convert(self, output_dir=None):
        if not self.selected_file_paths:
            QMessageBox.warning(self, "No Files", "Please select files first.")
            return

        for input_path in self.selected_file_paths:
            selected_file_name = os.path.basename(input_path)
            if selected_file_name.endswith('_converted.txt'):
                continue

            name, _ = os.path.splitext(selected_file_name)
            target_dir = output_dir if output_dir is not None else os.path.dirname(input_path)
            output_path = os.path.join(target_dir, f"{name}_converted.txt")

            try:
                self.convert_action(input_path, output_path)
                self.listbox.addItem(os.path.basename(output_path))
            except Exception as e:
                QMessageBox.critical(
                    self, "Conversion Error", f"Failed to convert {selected_file_name}\n{str(e)}"
                )

    def convert_action(self, input_path, output_path):
        """Delegates to the module-level convert_action (Qt-free)."""
        convert_action(input_path, output_path)

    def launch(self):
        """Show the converter window."""
        self.exec()