import sys
import os
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from app.main import Main

def ex_reading_all_supported_data(run_app=True):
    """Example of reading different supported .CSV and .txt data formats"""
    
    DATA_DIR = Path(__file__).parent / "spectroscopic_data"
    file_paths = [str(file) for file in DATA_DIR.glob("*.txt")]
    
    app = QApplication.instance() or QApplication([])  
    window = Main()
    app.setStyle("Fusion")
    
    if file_paths is None or len(file_paths) == 0:
        raise ValueError("No valid file paths provided.")
    try:
        window.open(file_paths=file_paths)
    except Exception as e:
        raise RuntimeError(f"Failed to open files: {e}")
    
    window.ui.show()
    
    if run_app:
        sys.exit(app.exec())

if __name__ == "__main__":    
    ex_reading_all_supported_data()