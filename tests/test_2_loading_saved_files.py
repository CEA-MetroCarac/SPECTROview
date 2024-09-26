# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# import pytest
# # from examples.ex_loading_saved_files import ex_loading_saved_files

# def test_loading_saved_files(qtbot):
#     print("test")
#     # ex_loading_saved_files(run_app=False) 

import pytest
from PySide6.QtWidgets import QApplication

@pytest.fixture(scope="module", autouse=True)
def app(qtbot):
    app = QApplication([])
    yield app
    app.quit()

def test_some_functionality(qtbot):
    print("test")
    assert True  # Replace with your actual test
