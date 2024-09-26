import sys
import os
from pathlib import Path
import pytest

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from app.main import Main

def ex_loading_saved_files(run_app=True):
    """Example of reading different supported .CSV and .txt data formats"""
    
    DATA_DIR = Path(__file__).parent
    file_paths = [str(file) for ext in ['*.maps', '*.spectra']  for file in DATA_DIR.glob(ext)]

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
    ex_loading_saved_files()